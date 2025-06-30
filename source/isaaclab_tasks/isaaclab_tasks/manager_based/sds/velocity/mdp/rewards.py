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
        """Combined reward: velocity & yaw tracking, height & orientation stability, contact stability."""
        import torch
        from omni.isaac.core.utils.quaternion import quat_apply_inverse
        # Access data
        robot = env.scene["robot"]
        contact_sensor = env.scene.sensors["contact_forces"]
        commands = env.command_manager.get_command("base_velocity")
        # Initialize
        reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        # Velocity tracking (forward)
        vx = robot.data.root_lin_vel_b[:, 0]
        vx_cmd = commands[:, 0]
        vel_err = (vx - vx_cmd).abs()
        vel_reward = torch.exp(-2.0 * vel_err)
        # Yaw rate tracking
        wz = robot.data.root_ang_vel_b[:, 2]
        wz_cmd = commands[:, 2]
        yaw_err = (wz - wz_cmd).abs()
        yaw_reward = torch.exp(-1.0 * yaw_err)
        # Height stability
        height = robot.data.root_pos_w[:, 2]
        height_err = (height - 0.34).abs()
        height_reward = torch.exp(-5.0 * height_err)
        # Uprightness
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=env.device).expand(env.num_envs, 3)
        inv_up = quat_apply_inverse(robot.data.root_quat_w, up)
        upright = torch.sum(inv_up * up, dim=-1).clamp(min=0.0, max=1.0)
        # Contact stability
        cf = contact_sensor.data.net_forces_w
        foot_ids, _ = contact_sensor.find_bodies(".*_foot")
        fforces = cf[:, foot_ids, :]
        mags = fforces.norm(dim=-1)
        contacts = (mags > 2.0).float()
        num_c = contacts.sum(dim=1)
        stable = ((num_c >= 2) & (num_c <= 3)).float()
        # Aggregate with weights
        reward = 2.0 * vel_reward + 1.5 * yaw_reward + 1.0 * height_reward + 1.0 * upright + 1.0 * stable
        return reward.clamp(min=0.0, max=10.0)



























































