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
    🔍 COMPREHENSIVE ENVIRONMENT ANALYSIS (ENVIRONMENT-AWARE MODE):

    🏞️ TERRAIN CLASSIFICATION (from SUS prompt):
    - TERRAIN_CLASS: 3 (STAIRS)
    - Classification confidence: HIGH
    - Visual justification: Repeating, evenly spaced tread–riser edges and L-shaped landings dominate the scene; humanoid steps on discrete treads.
    - Sensor validation: Height scanner shows Obstacles: Count: 13161 rays (46.4%); LiDAR obstacles: 3199 rays (44.4%); Closest obstacle: 1.043m.
      Baseline shift observed: measured baseline 0.111m vs Isaac G1 baseline 0.209m → risers classified as obstacles (expected on stairs).

    📊 NUMERICAL ANALYSIS RESULTS (exact data preserved):
    - Total rays (height scanner): 28350 (from 50 robots)
    - Total LiDAR rays: 7200 (from 50 robots)
    - Gaps Detected: 66 gaps with varying characteristics (0.2% of height rays)
    - Obstacles Detected: 13161 large obstacles (46.4% of height rays)
    - Terrain Roughness: 2.1cm (MODERATE, below threshold 7.0cm)
    - Safety Score: 53.3% traversable terrain
    - LiDAR obstacle distances: closest 1.043m, farthest 4.980m, average 2.845m; near <2.0m: 4.6%, moderate 2–4m: 36.2%, far >4.0m: 3.6%
    - Environment Verdict: DANGEROUS (Reason: obstacles 46.4% > 30% limit; gaps minimal 0.2%)

    📸 VISUAL ANALYSIS INSIGHTS:
    - Primary terrain type: Stair fields with periodic riser/tread patterns and L-shaped landings.
    - Movement challenges: Toe clearance over each riser, precise mid-tread foot placement, CoM control during ascent/descent and turns.
    - Navigation requirements: Use height scanner to estimate riser height 2–3 steps ahead; use LiDAR to maintain standoff to vertical faces; prioritize step-up planning over gap handling.

    🎯 REWARD STRATEGY DECISION (Single-skill: Stair Walking Specialist):
    - PRIMARY SCENARIO: STAIRS
    - Environmental sensing: NEEDED (dense vertical faces; DANGEROUS verdict)
    - Component priorities (weights reflected below):
      1) CoM/torso stability and safe, non-slipping stance (major)
      2) Healthy alternating steps without flight (walking, not jumping)
      3) Sensor-adaptive swing-foot clearance above riser height (height scanner)
      4) LiDAR-based safe standoff from vertical faces (1–3m band)
      5) Cautious progress and heading control through turns
    - Expected behavior:
      - Alternating single-support with brief double-support on landings (0.10–0.20 cycle).
      - Swing foot clears sensed riser height with safety margin.
      - Maintain upright torso, suppress vertical/heave and roll/pitch.
      - Maintain safe distance to vertical faces; plan smooth heading changes.

    📋 IMPLEMENTATION COMPONENTS:
    - Foundation (Isaac Lab proven):
      - track_lin_vel_xy_yaw_frame_exp (std=0.45) for cautious forward progress
      - track_ang_vel_z_world_exp (std=0.6) for smooth heading regulation
      - feet_air_time_positive_biped (threshold=0.45s) for alternating steps without flight
      - feet_slide penalty (G1 feet, 50N+ contact-aware) to limit tangential slip
      - Upright posture via projected gravity (lean control)
      - Suppress destabilizing lin_vel_z and ang_vel_xy
    - Environmental (stairs-focused):
      - Height scanner (567 rays, G1 baseline): dynamic-baseline riser estimation (near zone of scan)
      - Swing-foot clearance shaping: target ≥ riser_height + 3cm safety
      - Double-support landing bonus (brief and bounded)
      - LiDAR standoff bonus: maintain ~≥1.2m from vertical faces ahead
      - Optional upper-body collision penalty (>300N) for walls/risers
    - Weights (normalized intent): Stability ~0.35, Footstep/gait ~0.30, Clearance ~0.15, Progress/heading ~0.10, Efficiency/penalties ~0.10
    """
    import torch
    from isaaclab.managers import SceneEntityCfg

    device = env.device
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]

    # --------------------------------------------------------------------------------------
    # Proven Isaac Lab FOUNDATION components (walking on stairs; cautious tracking)
    # --------------------------------------------------------------------------------------
    vel_reward = track_lin_vel_xy_yaw_frame_exp(env, std=0.45, command_name="base_velocity")  # [N]
    yaw_reward = track_ang_vel_z_world_exp(env, command_name="base_velocity", std=0.6)        # [N]

    gait_reward = feet_air_time_positive_biped(
        env,
        command_name="base_velocity",
        threshold=0.45,  # cap air/contact mode time to promote walking cadence (no flight)
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    )  # [N]

    slide_pen = feet_slide(
        env,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        asset_cfg=SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    )  # [N] (penalty; contact-aware)

    # Upright posture: projected gravity in body (lean small -> better)
    grav_xy = robot.data.projected_gravity_b[:, :2]  # [N,2]
    lean_reward = torch.clamp(1.0 - torch.norm(grav_xy, dim=1), 0.0, 1.0)  # [0,1]

    # Stabilizers: suppress vertical heave and roll/pitch rates (dense, small penalties)
    lin_vel_z = torch.abs(robot.data.root_lin_vel_w[:, 2])  # [N]
    ang_vel_xy = torch.norm(robot.data.root_ang_vel_w[:, :2], dim=1)  # [N]
    lin_vel_z_pen = torch.clamp(lin_vel_z / 0.6, 0.0, 1.0)            # normalize by 0.6 m/s
    ang_vel_xy_pen = torch.clamp(ang_vel_xy / 2.0, 0.0, 1.0)          # normalize by 2 rad/s

    # --------------------------------------------------------------------------------------
    # ENVIRONMENTAL components (STAIRS): Height-scanner riser estimation + LiDAR standoff
    # --------------------------------------------------------------------------------------
    # Height scanner: G1-standard relative measurements with sanitization
    height_sensor = env.scene.sensors["height_scanner"]
    hm = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2] - 0.5  # [N,567]
    hm = torch.where(torch.isfinite(hm), hm, torch.zeros_like(hm))

    # Dynamic baseline per env (robust on stairs with baseline shift 0.111m vs 0.209m):
    # masked mean over finite values (vectorized)
    valid_mask = torch.isfinite(hm)
    valid_count = torch.clamp(valid_mask.sum(dim=1).float(), min=1.0)
    baseline_dyn = (torch.where(valid_mask, hm, torch.zeros_like(hm)).sum(dim=1) / valid_count)  # [N]
    # G1 thresholds (±7 cm) around dynamic baseline
    obst_thr = baseline_dyn - 0.07  # terrain higher than expected (risers)
    gap_thr = baseline_dyn + 0.07   # terrain lower than expected (gaps; rare here)

    # Reshape to 27x21 to analyze forward zones (2.0m × 1.5m, 7.5cm resolution)
    try:
        height_grid = hm.view(-1, 27, 21)  # [N,27,21] forward×lateral
    except Exception:
        # Fallback: if shape differs, use central slice heuristics on flat vector
        height_grid = hm  # keep as [N,567]

    # Near forward zone (first 9 forward rows ≈ 0–0.675m): estimate upcoming riser height
    if height_grid.dim() == 3:
        near_zone = height_grid[:, :9, :]  # [N,9,21]
        # Obstacles in near zone: readings lower than obst_thr (risers)
        # Approx riser reading = min value in near zone (closest/highest vertical face)
        near_min = torch.amin(near_zone.view(near_zone.shape[0], -1), dim=1)  # [N]
    else:
        # Use central 150 rays as "near" proxy
        near_min = torch.amin(hm[:, :150], dim=1)

    # Riser height estimate relative to dynamic baseline (non-negative)
    riser_delta = torch.clamp(baseline_dyn - near_min, min=0.0, max=0.30)  # cap to 30cm
    # Safety margin: +3cm (toe clearance target above riser)
    clearance_need = torch.clamp(riser_delta + 0.03, 0.03, 0.20)  # 3–20cm target band

    # Foot states for clearance and landing shaping
    foot_ids_list, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    foot_ids = torch.tensor(foot_ids_list[:2], dtype=torch.long, device=device)
    # Body z positions for those feet
    foot_z = robot.data.body_pos_w[:, foot_ids, 2]  # [N,2]
    # Contact timing and state
    contact_time = contact_sensor.data.current_contact_time[:, foot_ids]  # [N,2]
    air_time = contact_sensor.data.current_air_time[:, foot_ids]          # [N,2]
    in_contact = contact_time > 0.0                                      # [N,2]

    # Determine stance and swing foot z (prefer exactly-one-in-contact; else neutral)
    left_in = in_contact[:, 0]
    right_in = in_contact[:, 1]
    # stance_z: if left in contact → left_z, elif right → right_z, else → min of both
    stance_z = torch.where(
        left_in, foot_z[:, 0],
        torch.where(right_in, foot_z[:, 1], torch.min(foot_z, dim=1)[0]),
    )
    # swing_z: other foot if one-in-contact; else → max of both
    swing_z = torch.where(
        left_in, foot_z[:, 1],
        torch.where(right_in, foot_z[:, 0], torch.max(foot_z, dim=1)[0]),
    )
    one_in_contact = (left_in ^ right_in)

    # Clearance reward: encourage swing_z - stance_z ≥ clearance_need
    clearance_raw = torch.clamp((swing_z - stance_z - clearance_need) / 0.10, 0.0, 1.0)  # normalize by 10cm
    clearance_reward = torch.where(one_in_contact, clearance_raw, torch.zeros_like(clearance_raw))

    # Landing double-support bonus: brief, bounded (target 0.10–0.20s)
    both_in_contact = (in_contact.int().sum(dim=1) == 2)
    ds_time = torch.min(contact_time, dim=1)[0]  # min contact time among feet
    ds_bonus = torch.clamp(ds_time / 0.20, 0.0, 1.0) * both_in_contact.float()

    # LiDAR standoff in the forward sector (152 rays, use center 32)
    lidar = env.scene.sensors["lidar"]
    lidar_dist = torch.norm(lidar.data.ray_hits_w - lidar.data.pos_w.unsqueeze(1), dim=-1)  # [N,152]
    lidar_dist = torch.where(torch.isfinite(lidar_dist), lidar_dist, torch.ones_like(lidar_dist) * 5.0)
    center = lidar_dist.shape[1] // 2
    front_slice = slice(max(center - 16, 0), min(center + 16, lidar_dist.shape[1]))
    min_front = torch.amin(lidar_dist[:, front_slice], dim=1)  # [N]
    # Target safe standoff ~1.2m+ (bounded), gentle shaping
    standoff_bonus = torch.clamp((min_front - 0.8) / 0.4, 0.0, 1.0)  # 0.8→1.2m maps to 0→1

    # Optional upper-body collision penalty (walls/risers)
    collision_sensor = env.scene.sensors.get("collision_sensor") or env.scene.sensors.get("torso_contact")
    if collision_sensor is not None and getattr(collision_sensor.data, "net_forces_w_history", None) is not None:
        peak_forces = collision_sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1)[0]  # [N,B]
        upper_body_parts = [
            "pelvis", "torso_link", "pelvis_contour_link",
            "left_shoulder_pitch_link", "right_shoulder_pitch_link",
            "left_shoulder_roll_link", "right_shoulder_roll_link",
            "left_shoulder_yaw_link", "right_shoulder_yaw_link",
            "left_elbow_pitch_link", "right_elbow_pitch_link",
            "left_elbow_roll_link", "right_elbow_roll_link",
            "left_palm_link", "right_palm_link",
        ]
        collision_body_ids = []
        if hasattr(robot, "body_names") and robot.body_names is not None:
            for name in upper_body_parts:
                if name in robot.body_names:
                    idx = robot.body_names.index(name)
                    if idx < peak_forces.shape[1]:
                        collision_body_ids.append(idx)
        if len(collision_body_ids) > 0:
            col_forces = peak_forces[:, collision_body_ids]
            col_mask = col_forces > 300.0
            col_count = torch.sum(col_mask, dim=1).float()
            collision_pen = col_count * 0.2  # to be subtracted later
        else:
            collision_pen = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
    else:
        collision_pen = torch.zeros(env.num_envs, device=device, dtype=torch.float32)

    # --------------------------------------------------------------------------------------
    # ENERGY/SMOOTHNESS (simple, dense)
    # --------------------------------------------------------------------------------------
    # Penalize joint velocity norm (proxy for effort/jerk; small weight)
    joint_vel = robot.data.joint_vel
    if joint_vel is not None:
        dof_vel_l2 = torch.clamp((joint_vel**2).mean(dim=1), 0.0, 4.0)
    else:
        dof_vel_l2 = torch.zeros(env.num_envs, device=device, dtype=torch.float32)

    # --------------------------------------------------------------------------------------
    # COMBINATION (stable, additive; foundation-dominant with environmental support)
    # --------------------------------------------------------------------------------------
    # Foundation weights (stability and natural walking on stairs)
    w_vel = 1.6
    w_yaw = 1.0
    w_gait = 1.8
    w_lean = 1.0
    w_slide = 0.8
    w_linZ_pen = 0.3
    w_angXY_pen = 0.3

    foundation = (
        w_vel * vel_reward +
        w_yaw * yaw_reward +
        w_gait * gait_reward +
        w_lean * lean_reward -
        w_slide * slide_pen -
        w_linZ_pen * lin_vel_z_pen -
        w_angXY_pen * ang_vel_xy_pen +
        0.2  # baseline to maintain gradients
    )

    # Environmental (stairs) weights
    w_clear = 1.2
    w_ds = 0.6
    w_standoff = 0.6
    w_collision = 0.3
    w_effort = 0.2

    environmental = (
        w_clear * clearance_reward +
        w_ds * ds_bonus +
        w_standoff * standoff_bonus -
        w_collision * collision_pen -
        w_effort * dof_vel_l2 * 0.1  # small cost on excessive joint velocity
    )

    # Composition: foundation dominates (60/40 split prevents in-place turning on obstacles)
    total = foundation * 0.6 + environmental * 0.4

    # Final safety: finite + bounds
    total = torch.where(torch.isfinite(total), total, torch.ones_like(total) * 0.2)
    return total.clamp(min=0.0, max=6.0)


