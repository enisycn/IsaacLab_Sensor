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
    🔍 COMPREHENSIVE ENVIRONMENT ANALYSIS:

    📊 NUMERICAL ANALYSIS RESULTS:
    - Gaps Detected: 30 gaps (2 steppable ≤0.30m, 15 jumpable 0.30–0.60m, 13 impossible >0.60m)
    - Obstacles Detected: 5 obstacles (0.273–0.545m width)
    - Terrain Roughness: 2.6cm (moderate complexity)
    - Safety Score: 88.3% traversable terrain

    📸 VISUAL ANALYSIS INSIGHTS:
    - Primary terrain type: Controlled indoor studio floor (visually flat)
    - Visual environment features: No visible holes or blocks in camera view
    - Movement challenges observed: Steady forward walking, upright posture, precise foot placement, minimal lateral sway
    - Navigation requirements: Maintain forward velocity, avoid sensor-detected gaps/obstacles
    - Visual-sensor correlation: Visually flat but sensors report moderate gap/obstacle complexity

    🎯 REWARD STRATEGY DECISION:
    - PRIMARY SCENARIO: GAP_NAVIGATION (dominant: 30 gaps)
    - Environmental sensing: NEEDED (gaps & obstacles present)
    - Component priorities:
        1. Gap crossing/adaptation (jumping primary, stepping secondary)
        2. Obstacle avoidance (safe distance maintenance)
        3. Foundation locomotion (velocity tracking, gait, posture)

    📋 IMPLEMENTATION COMPONENTS:
    - Foundation: vel_tracking, ang_tracking, gait_pattern, height_maint, lean_stability
    - Environmental: gap_navigation (classify & reward), obstacle_avoidance (LiDAR), terrain_adaptation
    - Weights chosen to emphasize gap mechanics (highest), then safety, then foundation
    """

    # --- SCENE AND SENSORS ---
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    height_sensor = env.scene.sensors["height_scanner"]
    lidar_sensor = env.scene.sensors["lidar"]
    commands = env.command_manager.get_command("base_velocity")
    cmd_mag = torch.norm(commands[:, :2], dim=1) > 0.1

    # --- FOUNDATION LOCOMOTION ---
    # 1. Yaw-aligned linear velocity tracking
    vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w),
                                 robot.data.root_lin_vel_w[:, :3])
    lin_err = torch.sum((commands[:, :2] - vel_yaw[:, :2])**2, dim=1)
    vel_reward = torch.exp(-lin_err / 1.0**2) * cmd_mag.float()

    # 2. Angular velocity tracking (yaw)
    ang_err = (commands[:, 2] - robot.data.root_ang_vel_w[:, 2])**2
    ang_reward = torch.exp(-ang_err / 1.0**2)

    # 3. Bipedal gait single-stance reward
    foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=env.device)
    air_time = contact_sensor.data.current_air_time[:, foot_ids]
    contact_time = contact_sensor.data.current_contact_time[:, foot_ids]
    in_contact = contact_time > 0.0
    mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    gait_raw = torch.min(torch.where(single_stance.unsqueeze(-1),
                                     mode_time, torch.zeros_like(mode_time)), dim=1)[0]
    gait_reward = torch.clamp(gait_raw, max=0.5) * cmd_mag.float()

    # 4. Height maintenance (absolute)
    height_err = torch.abs(robot.data.root_pos_w[:, 2] - 0.74)
    height_reward = torch.exp(-height_err / 0.3)

    # 5. Lean stability (projected gravity)
    gravity_proj = robot.data.projected_gravity_b[:, :2]
    lean_reward = torch.exp(-2.0 * torch.norm(gravity_proj, dim=1))

    # --- ENVIRONMENTAL SENSING ---
    # Raw height scanner (meters)
    height_meas = (height_sensor.data.pos_w[:, 2].unsqueeze(1)
                   - height_sensor.data.ray_hits_w[..., 2] - 0.5)
    gap_depth = -height_meas  # positive = depth below sensor

    # Classify gaps
    steppable = gap_depth <= 0.30
    jumpable = (gap_depth > 0.30) & (gap_depth <= 0.60)
    impossible = gap_depth > 0.60
    steppable_any = steppable.any(dim=1).float()
    jumpable_any = jumpable.any(dim=1).float()
    impossible_any = impossible.any(dim=1).float()

    # Gap navigation rewards
    # 1. Stepping for small gaps
    stride_sep = torch.norm(
        robot.data.body_pos_w[:, foot_ids[0], :2]
        - robot.data.body_pos_w[:, foot_ids[1], :2],
        dim=1
    )
    stride_norm = torch.clamp((stride_sep - 0.2) / 0.3, 0.0, 1.0)
    stepping_reward = stride_norm * steppable_any * 0.2

    # 2. Jumping for medium gaps (use air time & vertical velocity)
    vertical_vel = robot.data.root_lin_vel_w[:, 2].clamp(min=0.0, max=2.0)
    up_norm = torch.clamp(vertical_vel / 1.0, 0.0, 1.0)
    flight_time = air_time.min(dim=1)[0]
    flight_norm = torch.clamp((flight_time - 0.1) / 0.4, 0.0, 1.0)
    jumping_reward = (up_norm * 0.3 + flight_norm * 0.7) * jumpable_any * 0.5

    # 3. Avoid impossible gaps
    avoidance_reward = -0.3 * impossible_any

    # LiDAR-based obstacle avoidance
    lidar_dist = torch.norm(
        lidar_sensor.data.ray_hits_w
        - lidar_sensor.data.pos_w.unsqueeze(1), dim=-1
    )
    min_dist = torch.min(lidar_dist, dim=1)[0]
    safety_bonus = torch.clamp((min_dist - 0.2) / 1.0, 0.0, 1.0) * 0.5

    # Terrain adaptation (favor low roughness)
    terrain_clear = height_meas.mean(dim=1).abs()
    terrain_reward = torch.exp(-terrain_clear / 0.15) * 0.1

    # --- COMBINE REWARDS ---
    foundation = (vel_reward * 2.0 +
                  ang_reward * 1.0 +
                  gait_reward * 2.0 +
                  height_reward * 1.5 +
                  lean_reward * 1.0)

    env_components = (stepping_reward * 1.5 +
                      jumping_reward * 3.0 +
                      avoidance_reward * 1.0 +
                      safety_bonus * 1.5 +
                      terrain_reward * 0.5)

    total_reward = foundation + env_components
    return total_reward.clamp(min=0.1, max=10.0)


