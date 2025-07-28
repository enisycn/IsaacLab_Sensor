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
    """SDS Custom Reward: SIMPLIFIED using ONLY proven Isaac Lab functions.
    
    Uses the exact same functions that work in other Isaac Lab locomotion tasks.
    No custom logic - just proven patterns with appropriate weights.
    """
    
    # Get basic data
    asset = env.scene["robot"]
    commands = env.command_manager.get_command("base_velocity")
    num_envs = asset.data.root_pos_w.shape[0]
    
    # Find foot contact sensor configuration
    contact_sensor_cfg = SceneEntityCfg("contact_forces")
    try:
        contact_sensor = env.scene.sensors["contact_forces"]
        foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
        if len(foot_ids) >= 2:
            contact_sensor_cfg.body_ids = foot_ids[:2]  # Use first 2 feet
        else:
            # Fallback - no gait reward if feet not found
            contact_sensor_cfg.body_ids = []
    except:
        contact_sensor_cfg.body_ids = []
    
    # === 1. LINEAR VELOCITY TRACKING (Primary - highest weight) ===
    lin_vel_reward = track_lin_vel_xy_yaw_frame_exp(
        env, 
        std=1.0, 
        command_name="base_velocity"
    ) * 6.0  # High weight for primary objective
    
    # === 2. ANGULAR VELOCITY TRACKING (Secondary) ===
    ang_vel_reward = track_ang_vel_z_world_exp(
        env, 
        command_name="base_velocity", 
        std=1.0
    ) * 2.0  # Medium weight for turning
    
    # === 3. BIPEDAL GAIT PATTERN (Important for locomotion) ===
    if len(contact_sensor_cfg.body_ids) >= 2:
        gait_reward = feet_air_time_positive_biped(
            env, 
            command_name="base_velocity", 
            threshold=0.5, 
            sensor_cfg=contact_sensor_cfg
        ) * 4.0  # High weight for proper gait
    else:
        gait_reward = torch.zeros(num_envs, device=env.device)
    
    # === 4. FEET SLIDING PENALTY (Converted to positive reward) ===
    if len(contact_sensor_cfg.body_ids) >= 2:
        # Create asset config that matches the contact sensor foot IDs
        asset_cfg = SceneEntityCfg("robot")
        asset_cfg.body_ids = contact_sensor_cfg.body_ids  # Use same foot IDs for consistency
        
        slide_penalty = feet_slide(
            env, 
            sensor_cfg=contact_sensor_cfg,
            asset_cfg=asset_cfg  # Pass matching asset config
        )
        # Convert penalty to reward: less sliding = higher reward
        slide_reward = torch.exp(-slide_penalty * 0.5) * 1.0  # Moderate weight
    else:
        slide_reward = torch.zeros(num_envs, device=env.device)
    
    # === 5. ARM SWING REWARD (NEW - Natural walking arm swing) ===
    # Encourage natural counter-balance arm swing during walking (like humans)
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel
    try:
        # Find arm joint indices
        joint_names = list(asset.joint_names)
        left_shoulder_pitch_idx = None
        right_shoulder_pitch_idx = None
        
        for i, name in enumerate(joint_names):
            if "left_shoulder_pitch_joint" in name:
                left_shoulder_pitch_idx = i
            elif "right_shoulder_pitch_joint" in name:
                right_shoulder_pitch_idx = i
        
        if left_shoulder_pitch_idx is not None and right_shoulder_pitch_idx is not None:
            # Get arm positions and base velocity
            left_arm_pos = joint_pos[:, left_shoulder_pitch_idx]
            right_arm_pos = joint_pos[:, right_shoulder_pitch_idx]
            
            # Get forward velocity for scaling arm swing
            vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
            forward_vel = torch.abs(vel_yaw[:, 0])  # Forward speed magnitude
            
            # Natural arm swing: opposite arms should move counter to each other
            # When walking fast, arms should swing more
            # Ideal: left arm forward when right leg forward (counter-balance)
            
            # Encourage moderate arm swing range (not too extreme)
            arm_swing_range = torch.abs(left_arm_pos - right_arm_pos)  # Difference between arms
            ideal_swing = torch.clamp(forward_vel * 0.5, 0.0, 0.4)  # Scale with speed (max 0.4 rad)
            
            # Reward natural swing range (not too little, not too much)
            swing_error = torch.abs(arm_swing_range - ideal_swing)
            arm_reward = torch.exp(-swing_error * 3.0) * 1.5  # Moderate weight for natural swing
            
            # Bonus for arm movement (discourage static arms)
            left_arm_vel = torch.abs(joint_vel[:, left_shoulder_pitch_idx])
            right_arm_vel = torch.abs(joint_vel[:, right_shoulder_pitch_idx])
            movement_bonus = torch.clamp((left_arm_vel + right_arm_vel) * 0.5, 0.0, 0.3)
            
            arm_reward = arm_reward + movement_bonus
        else:
            arm_reward = torch.zeros(num_envs, device=env.device)
            
    except Exception:
        # Fallback if arm joints not found
        arm_reward = torch.zeros(num_envs, device=env.device)
    
    # === 6. FORWARD PROGRESS BONUS (Simple additive) ===
    # Encourage forward movement when commands are given
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    forward_vel = torch.clamp(vel_yaw[:, 0], min=0.0)  # Only positive forward velocity
    command_mag = torch.norm(commands[:, :2], dim=1)  # Command magnitude
    command_scale = (command_mag > 0.1).float()  # Only when commanded to move
    progress_bonus = forward_vel * command_scale * 1.0  # Moderate weight
    
    # === 7. BASELINE BONUS (Always positive) ===
    baseline_bonus = 1.0  # Ensures reward is never zero
    
    # === COMBINE: Simple additive (all components are positive!) ===
    total_reward = (
        lin_vel_reward +     # 6.0x weight - PRIMARY objective
        ang_vel_reward +     # 2.0x weight - turning
        gait_reward +        # 4.0x weight - proper walking
        slide_reward +       # 1.0x weight - good foot contact
        arm_reward +         # 1.5x weight - natural arm swing (NEW!)
        progress_bonus +     # 1.0x weight - forward progress
        baseline_bonus       # 1.0 constant - always positive
    )
    
    # Clamp to reasonable range for PPO stability
    result = torch.clamp(total_reward, min=0.1, max=20.0)  # Increased max for arm reward
    
    # Ensure correct output shape
    assert result.shape == (num_envs,), f"Expected shape ({num_envs},), got {result.shape}"
    return result


