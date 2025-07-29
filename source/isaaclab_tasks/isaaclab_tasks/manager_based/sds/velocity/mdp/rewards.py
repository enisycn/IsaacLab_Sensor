# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def sds_custom_reward(env) -> torch.Tensor:
    """Improved reward for stable, human-like walking."""
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    commands = env.command_manager.get_command("base_velocity")  # [vx, vy, wz]

    # === 1. Velocity tracking (body frame) ===
    vx = robot.data.root_lin_vel_b[:, 0]
    vx_cmd = commands[:, 0]
    err_v = (vx - vx_cmd).abs()
    tol_v = torch.clamp(vx_cmd.abs() * 0.3 + 0.3, min=0.3)
    r_vel = torch.exp(-err_v / tol_v)  # in [0,1]

    # === 2. Height maintenance ===
    z = robot.data.root_pos_w[:, 2]
    err_z = (z - 0.90).abs()
    r_height = torch.exp(-err_z / 0.03)  # ±3cm tolerance

    # === 3. Torso lean penalty (continuous) ===
    up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=env.device)
    up_vec = up_vec.unsqueeze(0).expand(env.num_envs, 3)
    body_up = quat_apply_inverse(robot.data.root_quat_w, up_vec)
    lean = torch.atan2(body_up[:, 0].abs(), torch.clamp(body_up[:, 2], min=1e-6))
    r_lean = torch.exp(-lean / 0.1)  # moderate decay

    # === 4. Foot clearance (Gaussian around 2.5cm) ===
    foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=env.device)
    foot_pos = robot.data.body_pos_w[:, foot_ids, 2]  # [num_envs,2]
    forces = contact_sensor.data.net_forces_w[:, foot_ids, :]
    in_contact = (forces.norm(dim=-1) > 50.0).float()
    swing_h = (foot_pos * (1.0 - in_contact)).sum(dim=1) / torch.clamp((1.0 - in_contact).sum(dim=1), min=1e-6)
    err_clr = swing_h - 0.025
    r_clr = torch.exp(- (err_clr ** 2) / (2 * (0.01 ** 2)))  # σ=1cm

    # === 5. Foot-sliding penalty ===
    vel_b = robot.data.body_lin_vel_w[:, foot_ids, :2]
    slide = (vel_b.norm(dim=-1) * in_contact).mean(dim=1)
    r_slide = torch.exp(-slide / 0.05)  # penalize sliding >5cm/s

    # === 6. Continuous arm-leg anti-phase coordination ===
    # get shoulder & hip pitch velocities
    ls_idx, _ = robot.find_joints(["left_shoulder_pitch_joint"])
    rs_idx, _ = robot.find_joints(["right_shoulder_pitch_joint"])
    lh_idx, _ = robot.find_joints(["left_hip_pitch_joint"])
    rh_idx, _ = robot.find_joints(["right_hip_pitch_joint"])
    ls = torch.tensor(ls_idx, dtype=torch.long, device=env.device)
    rs = torch.tensor(rs_idx, dtype=torch.long, device=env.device)
    lh = torch.tensor(lh_idx, dtype=torch.long, device=env.device)
    rh = torch.tensor(rh_idx, dtype=torch.long, device=env.device)
    vel_j = robot.data.joint_vel
    sv_l = vel_j[:, ls].squeeze()
    sv_r = vel_j[:, rs].squeeze()
    hv_l = vel_j[:, lh].squeeze()
    hv_r = vel_j[:, rh].squeeze()
    corr_l = 1.0 - torch.abs(torch.cos(sv_l.sign() * hv_l.sign() * 1.0))
    corr_r = 1.0 - torch.abs(torch.cos(sv_r.sign() * hv_r.sign() * 1.0))
    r_coord = (corr_l + corr_r) * 0.5  # in [0,1]

    # === 7. Joint-limit safety penalty ===
    limits = robot.data.soft_joint_pos_limits  # [env,j,2]
    center = (limits[..., 0] + limits[..., 1]) * 0.5
    half_range = (limits[..., 1] - limits[..., 0]) * 0.5
    norm = (robot.data.joint_pos - center) / torch.clamp(half_range, min=1e-6)
    over = norm.abs() - 0.8
    pen = torch.where(over > 0.0, torch.exp(over * 3.0) - 1.0, torch.zeros_like(over))
    r_jlim = 1.0 / (1.0 + pen.sum(dim=1))  # in (0,1]

    # === Combine additively with weights and baseline ===
    reward = (
        r_vel * 1.5 +
        r_height * 1.0 +
        r_lean * 0.5 +
        r_clr * 0.5 +
        r_slide * 0.3 +
        r_coord * 0.3 +
        r_jlim * 1.0 +
        0.2
    )

    return reward.clamp(min=0.1, max=10.0)



