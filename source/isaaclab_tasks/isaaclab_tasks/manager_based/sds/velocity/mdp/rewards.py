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

def sds_custom_reward(env) -> torch.Tensor:
        """
        Custom SDS-generated reward function for locomotion.
        Args:
            env: The Isaac Lab environment instance
        Returns:
            torch.Tensor: Reward values for each environment (shape: [num_envs])
        """
        # Access robot data
        robot = env.scene["robot"]
        contact_sensor = env.scene.sensors["contact_forces"]
        commands = env.command_manager.get_command("base_velocity")
        
        # Initialize reward tensor
        reward = torch.zeros(env.num_envs, device=env.device)
        
        # Velocity tracking reward
        velocity_error = torch.norm(robot.data.root_lin_vel_b[:, 0] - commands[:, 0])
        velocity_reward = torch.exp(-velocity_error / 0.5)
        reward += velocity_reward
    
        # Torso height reward
        height_error = torch.abs(robot.data.root_pos_w[:, 2] - 0.34)
        height_reward = torch.exp(-height_error / 0.02)
        reward += height_reward
    
        # Orientation reward (projected gravity error)
        up_vector = matrix_from_quat(robot.data.root_quat_w)[:, :3, 2]
        gravity_vector = torch.tensor([0, 0, -1], device=env.device)
        orientation_error = torch.norm(up_vector - gravity_vector, dim=-1)
        orientation_reward = torch.exp(-orientation_error)
        reward += orientation_reward
        
        # Foot contact pattern reward (Walk-like pattern)
        foot_ids, foot_names = contact_sensor.find_bodies(".*_foot")
        contact_forces = contact_sensor.data.net_forces_w[:, foot_ids, :]
        contact_magnitudes = torch.norm(contact_forces, dim=-1)
        in_contact = contact_magnitudes > 5.0
        
        fl, fr, rl, rr = in_contact[:, 0], in_contact[:, 1], in_contact[:, 2], in_contact[:, 3]
        walk_pattern = (fl & ~fr & rl & ~rr) | (~fl & fr & ~rl & rr)
        walk_reward = walk_pattern.float() * 2.0
        reward += walk_reward
        
        # Penalty for high joint velocities to ensure smooth movements
        joint_velocity_penalty = torch.norm(robot.data.joint_vel, dim=-1)
        reward -= 0.01 * joint_velocity_penalty
    
        return reward















