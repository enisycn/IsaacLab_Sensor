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
    """
    Image analysis: The sequence shows a human performing squat jump: a deep knee bend,
    rapid explosive takeoff, aerial flight phase, and controlled landing.

    ENVIRONMENTAL ANALYSIS DECISION:
    Based on environment: Flat terrain, no gaps or obstacles.
    PRIMARY SCENARIO: FLAT
    ENVIRONMENTAL SENSING DECISION: NOT_NEEDED
    REWARD STRATEGY: JUMP: Combine forward velocity tracking, vertical effort,
    bilateral flight time coordination, and height maintenance for squat-jump behavior.
    Components:
      - velocity_reward: yaw-aligned forward velocity tracking
      - vertical_effort: upward velocity toward jump initiation
      - flight_time_reward: synchronized bilateral air‐time
      - height_reward: maintain body height close to target post‐takeoff
      - baseline: constant small bonus for exploration
    """
    # Isaac Lab proven patterns and available utilities
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]

    # 1. FORWARD VELOCITY TRACKING (yaw-aligned)
    commands = env.command_manager.get_command("base_velocity")
    cmd_xy = commands[:, :2]                                # [vx, vy]
    cmd_norm = torch.norm(cmd_xy, dim=1)
    # transform actual velocity into yaw frame
    vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w),
                                 robot.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum((cmd_xy - vel_yaw[:, :2])**2, dim=1)
    vel_reward = torch.exp(-lin_vel_error / (1.0**2))
    # only reward when command magnitude significant
    vel_reward = vel_reward * (cmd_norm > 0.1).float()

    # 2. VERTICAL EFFORT (encourage upward takeoff)
    vert_vel = robot.data.root_lin_vel_w[:, 2]
    # target vertical velocity ~2.0 m/s for squat jump
    up_target = 2.0
    vertical_effort = torch.clamp(vert_vel / up_target, 0.0, 1.0)

    # 3. BILATERAL FLIGHT TIME (both feet off ground)
    foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=env.device)
    air_time = contact_sensor.data.current_air_time[:, foot_ids]  # [N, n_feet]
    # require both feet to leave ground: use minimum air time
    bilateral_air = torch.min(air_time, dim=1)[0]
    # target ~0.3s air time for good jump clearance
    flight_time_reward = torch.clamp(bilateral_air / 0.3, 0.0, 1.0)

    # 4. HEIGHT MAINTENANCE (encourage achieving jump height and controlled posture)
    # target post‐takeoff height above world origin ~0.75 m
    height_err = torch.abs(robot.data.root_pos_w[:, 2] - 0.75)
    height_reward = torch.exp(-height_err / 0.3)

    # 5. BASELINE BONUS (prevent zero rewards, encourage exploration)
    baseline = 0.05

    # COMBINED DENSE REWARD
    reward = (
        vel_reward * 1.0 +
        vertical_effort * 1.0 +
        flight_time_reward * 0.5 +
        height_reward * 0.5 +
        baseline
    )

    return reward.clamp(min=0.0)


