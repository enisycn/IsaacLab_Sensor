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
    """Reward squat-jump gait: gentle height/velocity, symmetry, phase pattern, stability, smoothness, impact control."""
    import torch

    # Access environment data
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    # Initialize reward
    reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # --- HEIGHT & VELOCITY GENTLENESS ---
    current_height = robot.data.root_pos_w[:, 2]
    # Height gain above nominal standing height (0.74m)
    height_gain = current_height - 0.74
    gentle_height = torch.where(
    (height_gain > 0.05) & (height_gain < 0.25),
    torch.exp(-((height_gain - 0.15) / max(0.05, 1e-6)).abs()),
    torch.zeros_like(height_gain)
    )
    # Vertical velocity gentleness (world frame)
    z_vel = robot.data.root_lin_vel_w[:, 2]
    gentle_vel = torch.where(
    (z_vel > 0.5) & (z_vel < 1.5),
    torch.exp(-((z_vel - 0.8) / max(0.3, 1e-6)).abs()),
    torch.zeros_like(z_vel)
    )

    # --- CONTACT PHASE PATTERN (Synchronized Jumping) ---
    contact_forces = contact_sensor.data.net_forces_w  # [envs, bodies, 3]
    foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    foot_forces = contact_forces[:, foot_ids, :]       # [envs, 2, 3]
    force_magnitudes = foot_forces.norm(dim=-1)        # [envs, 2]
    foot_contacts = (force_magnitudes > 50.0).float()  # binary contacts
    contact_count = foot_contacts.sum(dim=-1)          # 0,1,2
    # Reward only flight (0) or double support (2), penalize single support
    phase_pattern = ((contact_count == 0) | (contact_count == 2)).float()

    # --- BILATERAL JOINT SYMMETRY ---
    joint_pos = robot.data.joint_pos  # [envs, 37]
    left_idx = [0, 1, 2, 3, 9, 10]     # hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll (left)
    right_idx = [4, 5, 6, 7, 11, 12]   # same for right
    diff = (joint_pos[:, left_idx] - joint_pos[:, right_idx]).abs().mean(dim=1)
    joint_symmetry = torch.exp(- diff / max(0.1, 1e-6))

    # --- TORQUE SMOOTHNESS & ENERGY EFFICIENCY ---
    joint_vel = robot.data.joint_vel  # [envs, 37]
    smoothness_reward = torch.exp(-0.001 * torch.sum(joint_vel * joint_vel, dim=1))

    # --- STABILITY: MINIMIZE ROLL/PITCH ANGULAR VELOCITY ---
    ang_vel = robot.data.root_ang_vel_b[:, :2]  # roll, pitch
    stability_reward = torch.exp(-2.0 * torch.sum(ang_vel * ang_vel, dim=1))

    # --- IMPACT GENTLENESS ON LANDING ---
    # Penalize excessive contact forces (>200N) across both feet
    excess_force = (force_magnitudes - 200.0).clamp(min=0.0)
    impact_penalty = torch.sum(excess_force, dim=1)
    impact_reward = torch.exp(-0.01 * impact_penalty)

    # --- COMBINE COMPONENTS ---
    # Weights: height (2.0), vel (1.5), symmetry (1.0), phase (1.0), stability (0.5), smoothness (0.5), impact (0.5)
    reward = (
    2.0 * gentle_height
    + 1.5 * gentle_vel
    + 1.0 * joint_symmetry
    + 1.0 * phase_pattern
    + 0.5 * stability_reward
    + 0.5 * smoothness_reward
    + 0.5 * impact_reward
    )

    return reward.clamp(min=0.0, max=10.0)


# SDS_FUNCTION_END_MARKER











