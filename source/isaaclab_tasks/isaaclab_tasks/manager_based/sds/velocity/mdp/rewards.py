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
        """Phase‐based squat‐jump reward: prep squat, controlled flight, gentle landing, symmetry, stability."""
        robot = env.scene["robot"]
        contact = env.scene.sensors["contact_forces"]
        device = env.device
    
        # contact forces & foot contacts
        forces = contact.data.net_forces_w  # [envs, bodies, 3]
        foot_ids, _ = contact.find_bodies(".*_ankle_roll_link")
        foot_forces = forces[:, foot_ids, :]  # [envs,2,3]
        fm = foot_forces.norm(dim=-1)  # [envs,2]
        contacts = fm > torch.tensor(50.0, dtype=torch.float32, device=device)
        left_c, right_c = contacts[:, 0], contacts[:, 1]
        double_support = left_c & right_c
        flight = ~left_c & ~right_c
    
        # height & squat preparation
        h = robot.data.root_pos_w[:, 2]
        base_h = torch.tensor(0.74, dtype=torch.float32, device=device)
        dh = (h - base_h).clamp(min=0.0)
        squat_reward = torch.where(
            double_support & (h > 0.32) & (h < 0.45),
            torch.exp(-((h - 0.38) / 0.04).abs()),
            torch.zeros_like(h)
        )
    
        # flight height reward (5-25cm)
        height_reward = torch.where(
            flight & (dh > 0.05) & (dh < 0.25),
            torch.exp(-((dh - 0.15) / 0.05).abs()),
            torch.zeros_like(dh)
        )
    
        # vertical velocity reward (0.3-1.5 m/s, peak 0.8)
        vz = robot.data.root_lin_vel_b[:, 2]
        vel_reward = torch.where(
            flight & (vz > 0.3) & (vz < 1.5),
            torch.exp(-((vz - 0.8) / 0.3).abs()),
            torch.zeros_like(vz)
        )
    
        # bilateral joint symmetry
        jpos = robot.data.joint_pos
        left_idx = [0, 1, 2, 3, 9, 10]
        right_idx = [4, 5, 6, 7, 11, 12]
        lj = jpos[:, left_idx]
        rj = jpos[:, right_idx]
        joint_sym = torch.exp(-((lj - rj).abs().mean(dim=1)) / 0.1)
    
        # airtime symmetry during flight
        airtime = contact.data.current_air_time[:, foot_ids]
        air_sym = flight.float() * torch.exp(-((airtime[:, 0] - airtime[:, 1]).abs()) / 0.1)
    
        # gentle landing: penalize high impact
        first_ct = contact.compute_first_contact(env.step_dt)[:, foot_ids]
        land_evt = (first_ct[:, 0] > 0) & (first_ct[:, 1] > 0)
        impact_mag = fm.max(dim=1)[0]
        impact_pen = torch.clamp(
            (impact_mag - torch.tensor(200.0, dtype=torch.float32, device=device))
            / max(200.0, 1e-6),
            min=0.0, max=1.0
        )
        land_reward = land_evt.float() * (1.0 - impact_pen)
    
        # orientation stability
        proj_g = robot.data.projected_gravity_b[:, :2]
        orient_pen = torch.sum(proj_g * proj_g, dim=1)
        orient_reward = torch.exp(-10.0 * orient_pen)
    
        # smoothness penalty
        jv = robot.data.joint_vel
        smooth_reward = torch.exp(-0.005 * torch.sum(jv * jv, dim=1))
    
        # combine with tuned weights
        reward = (
            squat_reward * 1.0 +
            height_reward * 2.0 +
            vel_reward * 1.5 +
            joint_sym * 1.0 +
            air_sym * 0.5 +
            land_reward * 1.0 +
            orient_reward * 2.0 +
            smooth_reward * 0.5
        )
    
        return reward.clamp(min=0.0, max=10.0)



