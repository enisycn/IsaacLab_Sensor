# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
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


# === CONTACT-BASED OBSTACLE AVOIDANCE ===

def robot_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get robot contact forces for observation (flattened and normalized)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get net contact forces for all robot links
    net_forces = contact_sensor.data.net_forces_w_history[:, -1, :, :]  # Latest timestep: (num_envs, num_bodies, 3)
    
    # Compute force magnitudes for each body
    force_magnitudes = torch.norm(net_forces, dim=2)  # (num_envs, num_bodies)
    
    # Flatten to get a single observation vector per environment
    obs = force_magnitudes.flatten(start_dim=1)  # (num_envs, num_bodies)
    
    # Normalize to prevent explosion
    obs = torch.clamp(obs / 10.0, 0.0, 1.0)  # Scale forces and clip
    
    return obs


def robot_obstacle_collision_penalty(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0
) -> torch.Tensor:
    """Penalize robot collisions with obstacles using contact sensor."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get net contact forces
    net_forces = contact_sensor.data.net_forces_w_history[:, -1, :, :]  # (num_envs, num_bodies, 3)
    
    # Compute force magnitudes for each body
    force_magnitudes = torch.norm(net_forces, dim=2)  # (num_envs, num_bodies)
    
    # Only consider forces above threshold AND not on feet (to exclude ground contact)
    # Heuristic: foot contacts typically have large Z forces (vertical), obstacle contacts have more XY components
    force_xy = torch.norm(net_forces[:, :, :2], dim=2)  # XY components (horizontal forces)
    force_z = torch.abs(net_forces[:, :, 2])  # Z component (vertical forces)
    
    # Obstacle collision: high total force AND significant horizontal component
    is_obstacle_contact = (force_magnitudes > threshold) & (force_xy > (threshold * 0.3))
    
    # Sum obstacle contacts across all robot bodies (excluding likely ground contacts)
    total_obstacle_contacts = torch.sum(is_obstacle_contact.float(), dim=1)  # (num_envs,)
    
    return total_obstacle_contacts


def robot_obstacle_collision_termination(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    threshold: float = 10.0
) -> torch.Tensor:
    """Terminate episode if robot has strong collision with obstacles."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get net contact forces
    net_forces = contact_sensor.data.net_forces_w_history[:, -1, :, :]  # (num_envs, num_bodies, 3)
    
    # Check for strong contacts (indicating collision)
    force_magnitudes = torch.norm(net_forces, dim=2)  # (num_envs, num_bodies)
    strong_contact = force_magnitudes > threshold  # (num_envs, num_bodies)
    
    # Terminate if any robot body has a strong contact
    should_terminate = torch.any(strong_contact, dim=1)  # (num_envs,)
    
    return should_terminate


def obstacle_distance_reward_position_based(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_distance: float = 1.5,
    std: float = 0.5
) -> torch.Tensor:
    """Reward for maintaining safe distance from obstacles using robot and obstacle positions."""
    # Get robot position
    robot = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w  # (num_envs, 3)
    
    # Define obstacle position (updated to match current obstacle location)
    obstacle_position = torch.tensor([
        [2.0, 0.3, 0.5],   # Red obstacle (moved to side for navigation learning)
    ], device=env.device, dtype=torch.float32)
    
    # Calculate distance from robot to the obstacle
    robot_pos_expanded = robot_pos.unsqueeze(1)  # (num_envs, 1, 3)
    obstacle_pos_expanded = obstacle_position.unsqueeze(0)  # (1, 1, 3)
    
    distances = torch.norm(robot_pos_expanded - obstacle_pos_expanded, dim=2)  # (num_envs, 1)
    
    # Get distance to the single obstacle
    distances_to_obstacle = distances.squeeze(1)  # (num_envs,)
    
    # Reward based on distance - higher reward for maintaining safe distance
    distance_error = torch.clamp(min_distance - distances_to_obstacle, min=0.0)
    reward = torch.exp(-distance_error / std)
    
    return reward


def progressive_lateral_penalty_position_based(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    safe_distance: float = 3.0,
    max_penalty_distance: float = 5.0
) -> torch.Tensor:
    """Progressive penalty for lateral movement - less penalty when obstacles are near (position-based)."""
    robot = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w
    
    # Define obstacle position (updated to match current obstacle location)
    obstacle_position = torch.tensor([
        [2.0, 0.3, 0.5],   # Red obstacle (moved to side for navigation learning)
    ], device=env.device, dtype=torch.float32)
    
    # Calculate distance to the obstacle
    robot_pos_expanded = robot_pos.unsqueeze(1)
    obstacle_pos_expanded = obstacle_position.unsqueeze(0)
    distances = torch.norm(robot_pos_expanded - obstacle_pos_expanded, dim=2)
    distance_to_obstacle = distances.squeeze(1)  # (num_envs,)
    
    # Get lateral velocity squared
    lateral_vel_sq = torch.square(robot.data.root_lin_vel_b[:, 1])
    
    # Progressive penalty factor: 0 when very close, 1 when far
    penalty_factor = torch.clamp(
        (distance_to_obstacle - safe_distance) / (max_penalty_distance - safe_distance),
        min=0.0, max=1.0
    )
    
    return penalty_factor * lateral_vel_sq
