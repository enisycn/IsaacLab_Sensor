# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


# === FORWARD MOTION SPECIFIC REWARDS ===

def forward_velocity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward forward velocity (X-axis) in robot base frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0]


def lateral_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize lateral velocity (Y-axis) in robot base frame."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def straight_motion_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for moving purely in X direction (negative absolute lateral velocity)."""
    asset = env.scene[asset_cfg.name]
    return -torch.abs(asset.data.root_lin_vel_b[:, 1])


def angular_velocity_z_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize angular velocity around Z-axis (yaw rotation)."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def maintain_heading_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for maintaining forward heading (yaw close to 0)."""
    asset = env.scene[asset_cfg.name]
    return -torch.square(asset.data.heading_w)


# === HEADING ERROR REWARDS FOR FORWARD MOTION ===

def forward_heading_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize deviation from forward heading (0 radians)."""
    asset = env.scene[asset_cfg.name]
    # Get yaw angle from quaternion (heading_w gives us yaw)
    yaw_angle = asset.data.heading_w
    # Wrap to [-pi, pi] and penalize deviation from 0
    yaw_error = torch.remainder(yaw_angle + torch.pi, 2 * torch.pi) - torch.pi
    return torch.square(yaw_error)


def forward_direction_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for robot's forward vector aligning with world X-axis."""
    asset = env.scene[asset_cfg.name]
    # Get robot's forward direction in world frame
    forward_vec_world = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    # World X-axis (forward direction)
    world_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    # Compute dot product (cosine of angle between vectors)
    alignment = torch.sum(forward_vec_world * world_x, dim=1)
    # Return alignment score (1.0 = perfect alignment, -1.0 = opposite direction)
    return alignment


def velocity_command_tracking_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Enhanced velocity tracking reward with exponential kernel specifically for forward motion."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Extract commanded and actual velocities
    cmd_vel_x = command[:, 0]  # Forward velocity command
    cmd_vel_y = command[:, 1]  # Lateral velocity command (should be 0)
    actual_vel_x = asset.data.root_lin_vel_b[:, 0]
    actual_vel_y = asset.data.root_lin_vel_b[:, 1]
    
    # Compute errors
    forward_error = torch.square(cmd_vel_x - actual_vel_x)
    lateral_error = torch.square(cmd_vel_y - actual_vel_y)
    
    # Combined error with higher weight on forward tracking
    total_error = forward_error + 2.0 * lateral_error  # Penalize lateral error more
    
    return torch.exp(-total_error / (std**2))
