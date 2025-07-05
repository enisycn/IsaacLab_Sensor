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
        """Gentle, efficient walking gait reward for Unitree G1 humanoid."""
        import torch
        robot = env.scene["robot"]
        contact_sensor = env.scene.sensors["contact_forces"]
        commands = env.command_manager.get_command("base_velocity")
        num_envs = env.num_envs
        device = env.device
    
        # Initialize reward
        reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
    
        # 1. Velocity tracking (forward biased)
        vel_b = robot.data.root_lin_vel_b  # [num_envs,3]
        forward_err = (vel_b[:, 0] - commands[:, 0]).abs()
        forward_r = torch.exp(-forward_err / 0.25) * 3.0
        lateral_err = (vel_b[:, 1] - commands[:, 1]).abs()
        lateral_r = torch.exp(-lateral_err / 0.5) * 1.0
        yaw_err = (robot.data.root_ang_vel_b[:, 2] - commands[:, 2]).abs()
        yaw_r = torch.exp(-yaw_err / 1.0) * 0.5
    
        # 2. Height maintenance around nominal walking height
        height = robot.data.root_pos_w[:, 2]
        height_err = (height - 0.74).abs()
        height_r = torch.exp(-height_err / 0.1)
    
        # 3. Torso orientation stability (penalize roll/pitch)
        quat = robot.data.root_quat_w  # [w,x,y,z]
        orient_pen = quat[:, 1]**2 + quat[:, 2]**2
        orient_r = torch.exp(-10.0 * orient_pen)
    
        # 4. Angular velocity stability (roll/pitch damping)
        ang_vel = robot.data.root_ang_vel_b[:, :2]
        ang_pen = torch.sum(ang_vel**2, dim=1)
        stability_r = torch.exp(-2.0 * ang_pen)
    
        # 5. Contact pattern for walking (1 or 2 feet)
        net_forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
        foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
        foot_forces = net_forces[:, foot_ids, :]  # [num_envs,2,3]
        force_mag = foot_forces.norm(dim=-1)  # [num_envs,2]
        contacts = (force_mag > 50.0).float()
        num_contacts = contacts.sum(dim=-1)
        contact_r = ((num_contacts == 1) | (num_contacts == 2)).float()
    
        # 6. Step timing consistency (low variance in last air times)
        last_air = contact_sensor.data.last_air_time[:, foot_ids]  # [num_envs,2]
        std_air = last_air.std(dim=1)
        mean_air = last_air.mean(dim=1)
        step_consistency = 1.0 - std_air / torch.clamp(mean_air, min=1e-6)
        step_consistency = step_consistency.clamp(min=0.0, max=1.0)
    
        # 7. Energy efficiency (joint velocity penalty)
        joint_vel = robot.data.joint_vel  # [num_envs,37]
        vel_pen = torch.sum(joint_vel**2, dim=1)
        efficiency_r = torch.exp(-0.001 * vel_pen)
    
        # Combine reward components with weights
        reward = forward_r + lateral_r + yaw_r
        reward = reward + height_r * 1.0 + orient_r * 1.5 + stability_r * 1.0
        reward = reward + contact_r * 0.5 + step_consistency * 0.5 + efficiency_r * 0.3
    
        # Final bounds
        return reward.clamp(min=0.0, max=10.0)




