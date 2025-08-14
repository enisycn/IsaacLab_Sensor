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
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_apply

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
    # Resolve consistent foot indices ONCE using the same pattern for both contacts and velocities.
    # Do not rely on sensor_cfg/asset_cfg.body_ids here as those may be unresolved for ad-hoc cfgs.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot_asset = env.scene[asset_cfg.name]

    # Prefer the explicitly provided body_names pattern; default to ankle roll links for G1 humanoid.
    pattern = getattr(sensor_cfg, "body_names", None) or getattr(asset_cfg, "body_names", None) or ".*_ankle_roll_link"

    foot_ids_list, _ = contact_sensor.find_bodies(pattern)
    foot_ids = torch.tensor(foot_ids_list, dtype=torch.long, device=env.device)

    # Compute contact mask using force history for robust contact detection
    forces_hist = contact_sensor.data.net_forces_w_history[:, :, foot_ids, :]  # [N, H, F, 3]
    contact_forces = forces_hist.norm(dim=-1).max(dim=1)[0]  # [N, F]
    contacts = contact_forces > 50.0  # Use 50N threshold suitable for G1 humanoid

    # Use the SAME indices for body velocities
    body_vel = robot_asset.data.body_lin_vel_w[:, foot_ids, :2]  # [N, F, 2]

    # Shape guard to catch any mismatch early
    assert body_vel.shape[:2] == contacts.shape, (
        f"feet_slide shape mismatch: vel {tuple(body_vel.shape)} vs contacts {tuple(contacts.shape)}"
    )

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
    🔍 COMPREHENSIVE ENVIRONMENT ANALYSIS (FOUNDATION-ONLY MODE):

    📊 NUMERICAL ANALYSIS RESULTS:
    - Gaps Detected: N/A gaps
    - Obstacles Detected: N/A large obstacles
    - Terrain Roughness: N/A cm
    - Safety Score: N/A% traversable terrain

    📸 VISUAL ANALYSIS INSIGHTS (foundation-only summary):
    - Primary terrain type: Flat, featureless studio floor (demonstration of steady heel-to-toe walking)
    - Visual environment features: Smooth cadence, minimal vertical excursion, clear left–right alternation
    - Movement challenges observed: Maintain stance>swing (~60:40), brief double support (~20%), consistent step length/cadence
    - Navigation requirements: None (terrain-agnostic locomotion mechanics only)

    🎯 REWARD STRATEGY DECISION:
    - PRIMARY SCENARIO: FLAT / SIMPLE (TERRAIN_CLASS: 0) — environmental sensing NOT_NEEDED
    - Environmental sensing: NOT_NEEDED (no height scanner/LiDAR; foundation-only gait/posture control)
    - Component priorities:
        1) Velocity tracking (xy) in yaw-aligned frame (primary)
        2) Biped gait quality with single-stance dominance (feet air/contact timing)
        3) Uprightness and base height stability
        4) Yaw-rate tracking for heading control
        5) Contact quality (minimize foot sliding), smooth joint motion and natural joint poses
    - Expected robot behavior:
        • Track commanded planar velocity smoothly without lateral drift
        • Alternate feet with ≈180° phase offset; stance fraction > swing with brief double support
        • Maintain upright torso (small roll/pitch), slight forward pitch allowed
        • Heel-to-toe like progression emergent via contact timing and sliding minimization
        • Smooth, non-jerky joint motions with conservative joint excursions

    📋 IMPLEMENTATION COMPONENTS (FOUNDATION-ONLY):
    - Foundation (Isaac Lab proven):
        • track_lin_vel_xy_yaw_frame_exp(std=0.6)
        • track_ang_vel_z_world_exp(std=0.6)
        • feet_air_time_positive_biped(threshold=0.50s, contact bodies=".*_ankle_roll_link")
        • feet_slide(sensor_cfg/asset_cfg body_names identical: ".*_ankle_roll_link")
    - Additional posture/quality (terrain-agnostic, proprioceptive only):
        • Base height maintenance around 0.74 m (absolute, flat-terrain nominal)
        • Uprightness via projected gravity (minimize roll/pitch)
        • Joint deviation L1 (small penalty toward neutral pose)
        • Smoothness via joint velocity L2 (small penalty)
    - Weights (additive, stable):
        • Velocity: 3.0, Yaw: 1.0, Gait: 2.0, Height: 1.5, Upright: 1.5
        • Slide penalty: -0.3, Joint deviation: -0.05, Smoothness: -0.05
        • Baseline: +0.2, Final clamp: [0.0, 5.0]

    Notes:
    - Foundation-only mode: no use of height_scanner/lidar/terrain-specific rewards.
    - Contact quality uses the contact sensor with G1 ankle roll links only (validated pattern).
    - All components combined additively with moderate scales and final clamp for PPO stability.
    """
    # Short aliases
    robot = env.scene["robot"]
    device = env.device

    # === Commands ===
    commands = env.command_manager.get_command("base_velocity")
    cmd_xy_mag = torch.norm(commands[:, :2], dim=1)

    # === 1) Proven velocity tracking (yaw-aligned xy) ===
    # Moderate std keeps gradients smooth without saturation
    vel_reward = track_lin_vel_xy_yaw_frame_exp(env, std=0.6, command_name="base_velocity")
    # gate velocity reward when command is effectively zero to avoid stationary exploitation
    vel_reward = vel_reward * (cmd_xy_mag > 0.1).float()

    # === 2) Proven yaw-rate tracking (world frame z) ===
    yaw_reward = track_ang_vel_z_world_exp(env, command_name="base_velocity", std=0.6)

    # === 3) Gait quality: single-stance dominance & clear swing (proven biped pattern) ===
    gait_reward = feet_air_time_positive_biped(
        env,
        command_name="base_velocity",
        threshold=0.50,  # cap air/contact time at 0.5s to bias walk (not hop/run)
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    )

    # === 4) Postural stability: base height around flat-terrain nominal ===
    # Absolute target is appropriate for SIMPLE terrain; shaped linearly and clamped
    height_err = torch.abs(robot.data.root_pos_w[:, 2] - 0.74)
    height_reward = torch.clamp(1.0 - (height_err / 0.25), 0.0, 1.0)

    # === 5) Uprightness: minimize roll/pitch using projected gravity in body frame ===
    # projected_gravity_b[:, :2] ≈ 0 when upright; reward larger when closer to 0
    gravity_proj_xy = robot.data.projected_gravity_b[:, :2]  # [N,2]
    lean_mag = torch.norm(gravity_proj_xy, dim=1)            # 0 is best (upright)
    upright_reward = torch.clamp(1.0 - lean_mag, 0.0, 1.0)

    # === 6) Contact quality: minimize stance-phase foot sliding (force-gated) ===
    slide_penalty = feet_slide(
        env,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        asset_cfg=SceneEntityCfg("robot",           body_names=".*_ankle_roll_link"),
    )
    # Bound penalty scale
    slide_penalty = torch.clamp(slide_penalty, 0.0, 2.0)

    # === 7) Natural joint posture (small L1 penalty to neutral) ===
    # Keep small to avoid fighting tracking; terrain-agnostic proprioceptive prior
    joint_pos = robot.data.joint_pos  # [N, n_joints]
    joint_dev = torch.mean(torch.abs(joint_pos), dim=1)  # L1 mean per env
    joint_dev_pen = torch.clamp(joint_dev, 0.0, 2.0)

    # === 8) Smoothness (use joint velocities as a proxy for action rate) ===
    joint_vel = robot.data.joint_vel  # [N, n_joints]
    smooth_pen = torch.mean(joint_vel * joint_vel, dim=1)  # L2 per env
    smooth_pen = torch.clamp(smooth_pen, 0.0, 5.0)

    # === Weighted additive combination ===
    reward = (
        vel_reward * 3.0 +
        yaw_reward * 1.0 +
        gait_reward * 2.0 +
        height_reward * 1.5 +
        upright_reward * 1.5 +
        (-slide_penalty) * 0.3 +
        (-joint_dev_pen) * 0.05 +
        (-smooth_pen) * 0.05 +
        0.2  # small baseline to keep gradients alive early
    )

    # === Final clamp for PPO stability ===
    reward = torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))
    return reward.clamp(min=0.0, max=5.0)



