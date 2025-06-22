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

# Import SDS contact analysis function
import sys
import os
# Add the SDS environment path to sys.path for importing
sds_env_path = os.path.join(os.path.dirname(__file__), "../../../../../../../SDS_ANONYM/SDS/envs")
if sds_env_path not in sys.path:
    sys.path.append(sds_env_path)

from isaac_lab_sds_env import get_foot_contact_analysis, extract_foot_contacts

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
        robot = env.scene["robot"]
        contact_sensor = env.scene.sensors["contact_forces"]
    
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        root_lin_vel_b = robot.data.root_lin_vel_b
        joint_vel = robot.data.joint_vel
    
        num_envs = env.num_envs
        device = env.device
    
        # Contact analysis
        contact_forces = contact_sensor.data.net_forces_w
        foot_ids, foot_names = contact_sensor.find_bodies(".*_foot")
        foot_forces = contact_forces[:, foot_ids, :]      # [num_envs, 4, 3]
        force_magnitudes = foot_forces.norm(dim=-1)       # [num_envs, 4]
        foot_contacts = force_magnitudes > 2.0            # [num_envs, 4]
    
        fl = foot_contacts[:, 0]
        fr = foot_contacts[:, 1]
        rl = foot_contacts[:, 2]
        rr = foot_contacts[:, 3]
    
        # Pace: FL+RL or FR+RR and not the other
        pace_left = fl & rl & (~fr) & (~rr)
        pace_right = fr & rr & (~fl) & (~rl)
        pace_pattern = pace_left | pace_right
    
        num_contacts = foot_contacts.sum(dim=-1)  # [num_envs]
    
        reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    
        # 1. Forward velocity tracking (L1 around 2.5 m/s, body-x)
        target_vx = 2.5
        lin_vel_bx = root_lin_vel_b[:, 0]
        vel_error = torch.abs(lin_vel_bx - target_vx)
        vel_reward = 1.0 - 0.33 * vel_error  # Range: [very negative, 1.0], rapidly decreases when off target
        vel_reward = torch.clamp(vel_reward, min=-1.0, max=1.0)
        reward += 1.0 * vel_reward
    
        # 2. Trunk height control (soft region at 0.34, hard penalty <0.28)
        target_z = 0.34
        height = root_pos_w[:, 2]
        height_error = torch.abs(height - target_z)
        height_soft = torch.clamp(1.0 - 7.0 * height_error, 0.0, 1.0)
        reward += 0.8 * height_soft
        # Hard penalty if falling too low
        reward -= 2.0 * (height < 0.28).float()
    
        # 3. Uprightness (no large roll/pitch: use world-z in body frame)
        up_vector = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).expand(num_envs, 3)
        projected_up = quat_apply_inverse(root_quat_w, up_vector)
        uprightness = projected_up[:, 2]  # z-up component in body frame
        upright_reward = torch.clamp(uprightness, min=0.0, max=1.0)
        reward += 0.7 * upright_reward
    
        # 4. Pace pattern soft reward
        reward += 0.35 * pace_pattern.float()
    
        # 5. #contacts pattern (prefer 2 or 4)
        contacts_preferred = ((num_contacts == 2) | (num_contacts == 4)).float()
        reward += 0.15 * contacts_preferred
    
        # 6. Joint velocity penalty (L1 norm)
        reward -= 0.015 * torch.sum(torch.abs(joint_vel), dim=-1)
    
        # 7. Foot force penalty for excessive impacts
        force_slamming = torch.clamp(force_magnitudes - 35.0, min=0.0)
        reward -= 0.004 * torch.sum(force_slamming, dim=-1)
    
        # 8. Penalty for not being in pace pattern (merge with 4)
        reward -= 0.25 * (~pace_pattern).float()
    
        # Clip negative rewards to stabilize
        reward = torch.clamp(reward, min=-2.0)
        return reward



























