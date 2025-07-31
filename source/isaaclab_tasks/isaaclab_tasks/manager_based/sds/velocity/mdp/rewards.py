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
    """
    ENVIRONMENTAL ANALYSIS DECISION:
    Based on environment analysis: Total Gaps Detected: 0, Total Obstacles Detected: 0, Average Terrain Roughness: 0.2cm (height variation 0.068m–0.074m)
    - Gaps detected: 0
    - Obstacles detected: 0
    - Terrain roughness: 0.2cm average
    - Safety assessment: LOW RISK - Suitable for basic navigation

    ENVIRONMENTAL SENSING DECISION: NOT_NEEDED
    JUSTIFICATION: No gaps or obstacles and minimal terrain roughness → foundation locomotion + movement‐quality terms only
    """
    import torch
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    # aliases
    device = env.device
    robot = env.scene["robot"]
    contact = env.scene.sensors["contact_forces"]
    commands = env.command_manager.get_command("base_velocity")           # [N,3]
    cmd_xy = commands[:, :2]
    cmd_mag = torch.norm(cmd_xy, dim=1)

    # 1) VELOCITY TRACKING (yaw‐aligned frame)
    vel_body = robot.data.root_lin_vel_w[:, :3]
    vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), vel_body)
    lin_err = torch.sum((cmd_xy - vel_yaw[:, :2])**2, dim=1)
    temp_v = 0.8
    vel_reward = torch.exp(-torch.clamp(lin_err, max=4.0) / (temp_v**2))
    vel_reward *= (cmd_mag > 0.1).float()
    vel_reward = torch.clamp(vel_reward, min=0.0, max=2.5)

    # 2) BIPEDAL GAIT (single‐stance air‐time)
    foot_ids, _ = contact.find_bodies(".*_ankle_roll_link")
    foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=device)
    air = torch.clamp(contact.data.current_air_time[:, foot_ids], min=0.0)
    touch = torch.clamp(contact.data.current_contact_time[:, foot_ids], min=0.0)
    in_contact = touch > 0.0
    single = (in_contact.int().sum(dim=1) == 1)
    phase_time = torch.where(in_contact, touch, air)
    gait_raw = torch.min(torch.where(single.unsqueeze(1), phase_time, torch.zeros_like(phase_time)), dim=1)[0]
    gait_reward = torch.clamp(gait_raw, max=0.5) * (cmd_mag > 0.1).float()
    gait_reward = torch.clamp(gait_reward, min=0.0, max=3.0)

    # 3) HEIGHT MAINTENANCE (terrain‐relative)
    hs = env.scene.sensors["height_scanner"]
    terrain_z = hs.data.ray_hits_w[..., 2].mean(dim=1)
    rel_h = robot.data.root_pos_w[:, 2] - terrain_z
    err_h = torch.abs(rel_h - 0.74)
    temp_h = 0.3
    height_reward = torch.exp(-torch.clamp(err_h, max=1.0) / temp_h)
    height_reward = torch.clamp(height_reward, min=0.0, max=1.5)

    # 4) ORIENTATION STABILITY (lean control)
    # project gravity into body frame
    N = commands.shape[0]
    grav_w = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0).expand(N, -1)
    grav_b = quat_apply_inverse(robot.data.root_quat_w, grav_w)
    lean = torch.norm(grav_b[:, :2], dim=1)
    lean_reward = torch.exp(-2.0 * torch.clamp(lean, max=10.0))
    lean_reward = torch.clamp(lean_reward, min=0.0, max=1.0)

    # 5) ARM SWING COORDINATION (reciprocal)
    # shoulder pitch joints indices
    sh_ids, _ = robot.find_joints(["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])
    sh_ids = torch.tensor(sh_ids, dtype=torch.long, device=device)
    sh_angles = robot.data.joint_pos[:, sh_ids]
    left_sh, right_sh = sh_angles[:, 0], sh_angles[:, 1]
    arm_rec = -torch.abs(left_sh + right_sh)                       # negative when they swing together
    arm_reward = torch.exp(torch.clamp(arm_rec, min=-1.0, max=0.0))
    arm_reward = torch.clamp(arm_reward, min=0.0, max=0.5)

    # 6) FOOT CLEARANCE (swing‐phase toe lift)
    body_pos = robot.data.body_pos_w[:, foot_ids, 2]               # [N,2]
    swing = (~in_contact).float()                                  # swing mask
    clearance = (body_pos * swing).max(dim=1)[0]                   # max foot height when swinging
    temp_f = 0.05
    foot_clear = torch.exp(-torch.clamp((0.02 - clearance).abs(), max=0.1) / temp_f)
    foot_clear_reward = torch.clamp(foot_clear, min=0.0, max=1.0)

    # 7) FOOT SLIDING PENALTY (contact‐aware)
    forces = contact.data.net_forces_w                               # [N,2,3]
    contact_mag = forces.norm(dim=-1) > 1.0
    vel_feet = robot.data.body_lin_vel_w[:, foot_ids, :2]
    slide = (vel_feet.norm(dim=-1) * contact_mag.float()).sum(dim=1)
    slide_penalty = -0.2 * torch.clamp(slide, max=0.5)

    # === COMBINE ALL COMPONENTS ===
    total = (
        vel_reward * 2.0 +
        gait_reward * 3.0 +
        height_reward * 1.5 +
        lean_reward * 1.0 +
        arm_reward * 0.5 +
        foot_clear_reward * 0.5 +
        slide_penalty +
        0.2   # baseline bonus
    )
    return torch.clamp(total, min=0.1, max=10.0)


