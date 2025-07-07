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
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, matrix_from_quat

# SDS reward functions use inline contact analysis
# No external imports needed - all analysis done within sds_custom_reward()

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
    # Use body frame velocity directly (already in robot frame)
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in body frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # Use body frame angular velocity (consistent with Isaac Lab standards)
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


# SDS Custom Reward Integration
# The following function will be dynamically updated by the SDS system
# GPT will generate the complete reward logic - no hardcoded components

def sds_custom_reward(env) -> torch.Tensor:
    """Gentle bilateral squat-jump reward with height, velocity, symmetry, and stability."""
    import torch
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    # initialize reward
    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    # --- Height and vertical velocity rewards (gentle jump) ---
    # Height above nominal standing
    current_height = robot.data.root_pos_w[:, 2]
    nominal_height = 0.74
    height_gain = current_height - nominal_height
    gentle_height = torch.where(
    (height_gain > 0.05) & (height_gain < 0.25),
    torch.exp(-((height_gain - 0.15) / max(0.05, 1e-6)).abs()),
    torch.zeros_like(height_gain)
    )
    # Vertical velocity in world frame
    vz = robot.data.root_lin_vel_w[:, 2]
    gentle_velocity = torch.where(
    (vz > 0.5) & (vz < 1.5),
    torch.exp(-((vz - 1.0) / max(0.3, 1e-6)).abs()),
    torch.zeros_like(vz)
    )
    # --- Contact and gait phase (jump pattern) ---
    forces = contact_sensor.data.net_forces_w  # [envs, bodies, 3]
    foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    foot_forces = forces[:, foot_ids, :]       # [envs, 2, 3]
    force_mags = foot_forces.norm(dim=-1)      # [envs, 2]
    foot_contacts = (force_mags > 50.0).float()
    left_contact = foot_contacts[:, 0]
    right_contact = foot_contacts[:, 1]
    total_contacts = left_contact + right_contact
    jump_pattern = ((total_contacts == 0) | (total_contacts == 2)).float()
    # --- Bilateral symmetry (joint + air time) ---
    jp = robot.data.joint_pos
    left_idx  = [0, 2, 4, 6, 9, 11]   # G1 left leg indices (interleaved PhysX pattern)
    right_idx = [1, 3, 5, 7, 10, 12]  # G1 right leg indices (interleaved PhysX pattern)
    jl = jp[:, left_idx]
    jr = jp[:, right_idx]
    joint_diff = (jl - jr).abs().mean(dim=1)
    joint_symmetry = torch.exp(-joint_diff / max(0.1, 1e-6))
    airtime = contact_sensor.data.current_air_time[:, foot_ids]  # [envs,2]
    air_diff = (airtime[:, 0] - airtime[:, 1]).abs()
    air_symmetry = torch.exp(-air_diff / max(0.1, 1e-6))
    # --- Torso orientation stability ---
    proj_g = robot.data.projected_gravity_b[:, :2]  # [envs,2]
    orient_pen = torch.sum(proj_g * proj_g, dim=1)
    orientation_reward = torch.exp(-5.0 * orient_pen)
    # --- Combine components with weights and mask by jump phases ---
    total = (
    gentle_height * 3.0 +
    gentle_velocity * 2.0 +
    joint_symmetry * 1.5 +
    air_symmetry * 1.0 +
    orientation_reward * 1.0
    )
    reward = jump_pattern * total
    return reward.clamp(min=0.0, max=10.0)


# SDS_FUNCTION_END_MARKER







