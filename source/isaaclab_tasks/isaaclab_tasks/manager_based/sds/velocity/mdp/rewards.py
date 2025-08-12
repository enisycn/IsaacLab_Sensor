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
    🔍 FEEDBACK ANALYSIS (WALKING SPECIALIST • FOUNDATION-ONLY MODE):
    - TERRAIN_CLASS: 0 (Flat). Rationale: Height scanner summary shows 28350/28350 rays normal, 0 gaps, 0 obstacles; visual context is an open, uniform floor.
    - Prior policy trends indicate adequate forward walking but weak heading control and occasional instability:
      • velocity_error_yaw Mean=1.99 (Max=3.23) is large → robot drifts/oscillates in yaw.
      • velocity_error_xy Mean=0.46 is moderate but improvable.
      • termination_base_contact Mean=10.41% (Max=84.88%) → falls and non‑foot contacts present.
      • reward_sds_custom Mean=2.65 (Max=4.59) under a 0–5 clamp → headroom remains.
      • episode length Mean=918.25 (Max=1497.57) shows progress but instability spikes remain.
    - Exploration is healthy (entropy_loss Mean=41.15; action_noise_std Mean=1.50), training stable (value_function_loss Mean=0.09; surrogate_loss ≈0).

    📊 PERFORMANCE METRICS (observed during training):
    - reward: Max=135.82, Mean=79.30, Min=1.12
    - episode length: Max=1497.57, Mean=918.25, Min=14.31
    - reward_sds_custom: Max=4.59, Mean=2.65, Min=0.03
    - velocity_error_xy: Max=0.72, Mean=0.46, Min=0.01
    - velocity_error_yaw: Max=3.23, Mean=1.99, Min=0.02
    - termination_base_contact: Max=84.88, Mean=10.41, Min=0.00

    🌍 ENVIRONMENTAL CONTEXT (from SUS – used only for feedback, not for sensors in reward):
    - Total rays: 28350 (from 50 robots)
    - Height readings: 0.189–0.278 m (avg: 0.242 m)
    - Gaps Detected: 0 gaps; Obstacles Detected: 0; Normal terrain: 100.0%
    - Terrain Roughness: N/Acm; Safety Score: 100.0% traversable terrain
    - Decision: Foundation-only shaping (no height_scanner/lidar in reward).

    🎯 SOLUTION STRATEGY:
    - Strengthen heading regulation and lateral-path discipline on flat ground.
    - Shape cadence to discourage rapid micro-steps (short swing/stance) and reduce flight phases (walking only).
    - Increase stability via gentle height/posture tracking and stance slip reduction.
    - Keep rewards dense, additive, and bounded to maintain PPO stability.

    📋 IMPLEMENTATION COMPONENTS (weights chosen from feedback trends):
    - Proven components (Isaac Lab):
      1) track_lin_vel_xy_yaw_frame_exp(std=0.6) → strong XY tracking (weight 3.2)
      2) track_ang_vel_z_world_exp(std=0.6) → sharper yaw control vs previous (weight 1.5; lower std for sensitivity)
      3) feet_air_time_positive_biped(threshold=0.6 s) → stable single‑stance rhythm (weight 2.2)
      4) feet_slide() → stance-slip penalty (weight −0.6)
    - Additional flat-ground shapers:
      • Absolute height maintenance around 0.74 m (weight 1.6)
      • Roll/pitch angular‑rate damping (weight 1.2)
      • Lateral drift penalty when command vy≈0 (weight −0.6)
      • Anti‑micro‑step penalty (hinge on very short swing/stance) (weight −0.4)
      • Flight-phase penalty (both feet airborne) for walk (weight −0.3)
      • Smoothness: small joint-velocity L2 (weight −0.02)
    - Baseline bonus: +0.2
    - Final clamp: [0.0, 5.0]

    Expected outcome:
    - Lower velocity_error_yaw via tighter yaw tracking and lateral drift control.
    - Reduced termination_base_contact via shorter-flight discouragement, height/posture shaping, and slip penalty.
    - Higher reward_sds_custom mean with denser, cadence-aware feedback while staying PPO-stable.

    NOTE: FOUNDATION-ONLY MODE — no usage of height_scanner/lidar; contact sensor is used for gait/slide shaping only.
    """
    import torch

    robot = env.scene["robot"]
    commands = env.command_manager.get_command("base_velocity")
    cmd_xy = commands[:, :2]
    cmd_mag = torch.norm(cmd_xy, dim=1)
    is_moving = (cmd_mag > 0.1).float()

    # === 1) Proven velocity tracking (XY in yaw-aligned frame) and yaw tracking ===
    # Stronger XY tracking; keep std moderate for stability
    vel_reward = track_lin_vel_xy_yaw_frame_exp(env, std=0.6, command_name="base_velocity")
    # Sharper yaw control (smaller std than before to reduce yaw error ~2.0)
    ang_reward = track_ang_vel_z_world_exp(env, command_name="base_velocity", std=0.6)

    # === 2) Proven gait shaping: single-stance rhythm for walking ===
    feet_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    gait_reward = feet_air_time_positive_biped(
        env,
        command_name="base_velocity",
        threshold=0.6,  # promote ~0.35–0.55 s swing; cap to avoid high-knee tapping
        sensor_cfg=feet_cfg,
    )

    # === 3) Proven stance slip penalty ===
    slide_penalty = feet_slide(
        env,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        asset_cfg=SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    )

    # === 4) Flat-ground posture: gentle absolute height tracking around nominal 0.74 m ===
    height_err = (robot.data.root_pos_w[:, 2] - 0.74).abs()
    height_reward = torch.clamp(1.0 - height_err / 0.20, 0.0, 1.0)  # slightly tighter than before

    # === 5) Upright stability: roll/pitch angular-rate damping (world frame) ===
    if hasattr(robot.data, "root_ang_vel_w") and robot.data.root_ang_vel_w is not None:
        pr_rate = torch.norm(robot.data.root_ang_vel_w[:, :2], dim=1)
    else:
        pr_rate = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    stability_reward = torch.clamp(1.0 - pr_rate / 3.0, 0.0, 1.0)

    # === 6) Lateral drift penalty when lateral command is ~0 (keep straight path) ===
    # Recompute yaw-aligned velocity to read lateral component safely
    vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    lateral_mask = (torch.abs(commands[:, 1]) < 0.05).float()
    lateral_drift_penalty = torch.clamp(torch.abs(vel_yaw[:, 1]), 0.0, 1.5) * lateral_mask * is_moving

    # === 7) Cadence hygiene: discourage ultra-short swing/stance and flight (walking only) ===
    contact_sensor = env.scene.sensors["contact_forces"]
    foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=env.device)
    # Guard for unexpected matches
    if foot_ids.numel() > 2:
        foot_ids = foot_ids[:2]
    elif foot_ids.numel() == 1:
        foot_ids = torch.stack([foot_ids[0], foot_ids[0]])

    # Current phase times (clamped to non-negative)
    air_t = contact_sensor.data.current_air_time[:, foot_ids]
    con_t = contact_sensor.data.current_contact_time[:, foot_ids]
    air_t = torch.clamp(air_t, min=0.0)
    con_t = torch.clamp(con_t, min=0.0)

    # Penalize very short swing (<0.15 s) and very short stance (<0.20 s)
    short_swing = torch.clamp(0.15 - air_t, min=0.0) / 0.15
    short_stance = torch.clamp(0.20 - con_t, min=0.0) / 0.20
    short_phase_penalty = (short_swing + short_stance).mean(dim=1) * is_moving
    short_phase_penalty = torch.clamp(short_phase_penalty, 0.0, 2.0)

    # Discourage flight (both feet off ground) for walking tasks
    in_contact = con_t > 0.0
    feet_in_contact = torch.sum(in_contact.int(), dim=1)
    flight_phase = (feet_in_contact == 0).float() * is_moving
    flight_penalty = flight_phase  # already 0/1; scaled later

    # === 8) Smoothness: small joint-velocity L2 penalty ===
    if hasattr(robot.data, "joint_vel") and robot.data.joint_vel is not None:
        joint_vel_l2 = torch.mean(torch.square(robot.data.joint_vel), dim=1)
    else:
        joint_vel_l2 = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # === 9) Assemble additive reward (foundation-dominant) ===
    baseline = 0.2
    total = (
        vel_reward * 3.2
        + ang_reward * 1.5
        + gait_reward * 2.2
        + height_reward * 1.6
        + stability_reward * 1.2
        - slide_penalty * 0.6
        - lateral_drift_penalty * 0.6
        - short_phase_penalty * 0.4
        - flight_penalty * 0.3
        - joint_vel_l2 * 0.02
        + baseline
    )

    # PPO-safe clamp
    return torch.clamp(total, min=0.0, max=5.0)



