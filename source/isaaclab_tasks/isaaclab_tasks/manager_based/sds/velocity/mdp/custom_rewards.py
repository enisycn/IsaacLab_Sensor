# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for G1 robot based on optimization interface examples."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def jump_and_keep_left_hand_high(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward function: (left_hand_height > 1) * (head_height > 1.8)
    Encourages jumping while keeping left hand high.
    
    This reward promotes behaviors where the robot:
    1. Raises its left hand above 1 meter
    2. Gets its head height above 1.8 meters (jumping/standing tall)
    """
    # Get robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get body positions
    body_pos_w = robot.data.body_pos_w
    body_names = robot.data.body_names
    
    # Find left hand and head links
    # Note: Adjust these names based on your G1 robot URDF
    try:
        left_hand_idx = body_names.index("left_hand_link")  # Adjust name as needed
        left_hand_height = body_pos_w[:, left_hand_idx, 2]  # Z coordinate (height)
    except ValueError:
        # Fallback: estimate from elbow position + offset
        try:
            left_elbow_idx = body_names.index("left_elbow_pitch_link")
            left_hand_height = body_pos_w[:, left_elbow_idx, 2] + 0.3  # Rough estimate
        except ValueError:
            left_hand_height = robot.data.root_pos_w[:, 2] + 0.5  # Fallback estimate
    
    try:
        head_idx = body_names.index("head_link")  # Adjust name as needed
        head_height = body_pos_w[:, head_idx, 2]
    except ValueError:
        # Fallback: base height + head offset
        head_height = robot.data.root_pos_w[:, 2] + 0.7
    
    # Apply reward conditions
    left_hand_condition = (left_hand_height > 1.0).float()
    head_condition = (head_height > 1.8).float()
    
    reward = left_hand_condition * head_condition
    return reward


def spin_with_closed_arms(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward function: (body_angular_velocity_yaw > 6) * (left_hand_lateral_distance < 0.3) * (right_hand_lateral_distance < 0.3)
    Encourages spinning while keeping arms close to body.
    """
    # Get robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get angular velocity (yaw component)
    angular_vel = robot.data.root_ang_vel_w[:, 2]  # Z-axis rotation (yaw)
    
    # Get body positions for hand distance calculation
    body_pos_w = robot.data.body_pos_w
    body_names = robot.data.body_names
    root_pos = robot.data.root_pos_w
    
    # Calculate hand distances from body center
    try:
        left_hand_idx = body_names.index("left_hand_link")
        left_hand_pos = body_pos_w[:, left_hand_idx, :]
        left_hand_distance = torch.norm(left_hand_pos[:, :2] - root_pos[:, :2], dim=1)  # XY distance only
    except ValueError:
        # Fallback: estimate from shoulder/elbow
        left_hand_distance = torch.ones(env.num_envs, device=env.device) * 0.4
    
    try:
        right_hand_idx = body_names.index("right_hand_link")
        right_hand_pos = body_pos_w[:, right_hand_idx, :]
        right_hand_distance = torch.norm(right_hand_pos[:, :2] - root_pos[:, :2], dim=1)
    except ValueError:
        right_hand_distance = torch.ones(env.num_envs, device=env.device) * 0.4
    
    # Apply reward conditions
    spin_condition = (angular_vel > 6.0).float()
    left_arm_condition = (left_hand_distance < 0.3).float()
    right_arm_condition = (right_hand_distance < 0.3).float()
    
    reward = spin_condition * left_arm_condition * right_arm_condition
    return reward


def move_backward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward function: (body_speed_forward < -2)
    Encourages moving backward at high speed.
    """
    # Get robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get forward velocity (X component in robot's frame)
    lin_vel_w = robot.data.root_lin_vel_w
    
    # Convert to robot's local frame to get forward/backward velocity
    # Assuming robot's forward direction is +X in world frame initially
    root_quat = robot.data.root_quat_w
    forward_dir = math_utils.quat_apply(root_quat, torch.tensor([1.0, 0.0, 0.0], device=env.device))
    forward_velocity = torch.sum(lin_vel_w * forward_dir, dim=1)
    
    # Apply reward condition (negative means backward)
    backward_condition = (forward_velocity < -2.0).float()
    return backward_condition


def reach_forward_with_both_hands(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Custom reward: Reach forward with both hands while standing.
    """
    # Get robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    body_pos_w = robot.data.body_pos_w
    body_names = robot.data.body_names
    root_pos = robot.data.root_pos_w
    
    # Calculate hand forward distances
    try:
        left_hand_idx = body_names.index("left_hand_link")
        left_hand_pos = body_pos_w[:, left_hand_idx, :]
        left_hand_forward = left_hand_pos[:, 0] - root_pos[:, 0]  # X distance from body center
    except ValueError:
        left_hand_forward = torch.zeros(env.num_envs, device=env.device)
    
    try:
        right_hand_idx = body_names.index("right_hand_link")
        right_hand_pos = body_pos_w[:, right_hand_idx, :]
        right_hand_forward = right_hand_pos[:, 0] - root_pos[:, 0]
    except ValueError:
        right_hand_forward = torch.zeros(env.num_envs, device=env.device)
    
    # Body height (standing condition)
    body_height = root_pos[:, 2]
    
    # Apply reward conditions
    left_reach_condition = (left_hand_forward > 0.5).float()
    right_reach_condition = (right_hand_forward > 0.5).float()
    standing_condition = (body_height > 0.6).float()
    
    reward = left_reach_condition * right_reach_condition * standing_condition
    return reward


def dance_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Custom reward: Dance-like pose with arms up and balanced stance.
    """
    # Get robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    body_pos_w = robot.data.body_pos_w
    body_names = robot.data.body_names
    root_pos = robot.data.root_pos_w
    
    # Get hand heights
    try:
        left_hand_idx = body_names.index("left_hand_link")
        left_hand_height = body_pos_w[:, left_hand_idx, 2]
    except ValueError:
        left_hand_height = root_pos[:, 2] + 0.5
    
    try:
        right_hand_idx = body_names.index("right_hand_link")
        right_hand_height = body_pos_w[:, right_hand_idx, 2]
    except ValueError:
        right_hand_height = root_pos[:, 2] + 0.5
    
    # Calculate stability (low angular velocity indicates balance)
    angular_vel = robot.data.root_ang_vel_w
    angular_vel_magnitude = torch.norm(angular_vel, dim=1)
    stability = torch.exp(-angular_vel_magnitude)  # Higher stability = lower angular velocity
    
    # Apply reward conditions
    left_arm_up = (left_hand_height > 1.2).float()
    right_arm_up = (right_hand_height > 1.2).float()
    stable_condition = (stability > 0.8).float()
    
    reward = left_arm_up * right_arm_up * stable_condition
    return reward


def walk_forward_with_arm_swing(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Custom reward: Natural walking with coordinated arm swing.
    """
    # Get robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Forward velocity
    lin_vel_w = robot.data.root_lin_vel_w
    root_quat = robot.data.root_quat_w
    forward_dir = math_utils.quat_apply(root_quat, torch.tensor([1.0, 0.0, 0.0], device=env.device))
    forward_velocity = torch.sum(lin_vel_w * forward_dir, dim=1)
    
    # Arm swing coordination (simplified)
    joint_vel = robot.data.joint_vel
    joint_names = robot.data.joint_names
    
    # Find shoulder joint velocities for arm swing
    left_shoulder_swing = torch.zeros(env.num_envs, device=env.device)
    right_shoulder_swing = torch.zeros(env.num_envs, device=env.device)
    
    for i, name in enumerate(joint_names):
        if "left_shoulder_pitch" in name:
            left_shoulder_swing = torch.abs(joint_vel[:, i])
        elif "right_shoulder_pitch" in name:
            right_shoulder_swing = torch.abs(joint_vel[:, i])
    
    # Reward conditions
    forward_condition = (forward_velocity > 0.5).float() * (forward_velocity < 2.0).float()  # Reasonable walking speed
    arm_swing_condition = ((left_shoulder_swing > 0.1) & (right_shoulder_swing > 0.1)).float()
    
    reward = forward_condition * arm_swing_condition
    return reward


def maintain_balance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward for maintaining upright balance.
    """
    # Get robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get orientation
    root_quat = robot.data.root_quat_w
    
    # Calculate upright orientation (dot product with up vector)
    up_vector = torch.tensor([0.0, 0.0, 1.0], device=env.device)
    robot_up = math_utils.quat_apply(root_quat, up_vector)
    
    # Reward for being upright (close to world up direction)
    upright_reward = torch.clamp(robot_up[:, 2], min=0.0)  # Z component of robot's up vector
    
    return upright_reward


def avoid_joint_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalty for approaching joint limits.
    """
    # Get robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get joint positions and limits
    joint_pos = robot.data.joint_pos
    joint_limits = robot.data.soft_joint_pos_limits
    
    # Normalize joint positions to [-1, 1] range
    joint_pos_normalized = 2.0 * (joint_pos - joint_limits[:, :, 0]) / (joint_limits[:, :, 1] - joint_limits[:, :, 0]) - 1.0
    
    # Penalty for being close to limits (|normalized_pos| > 0.8)
    limit_penalty = torch.sum(torch.clamp(torch.abs(joint_pos_normalized) - 0.8, min=0.0), dim=1)
    
    # Return reward (negative penalty)
    return -limit_penalty 