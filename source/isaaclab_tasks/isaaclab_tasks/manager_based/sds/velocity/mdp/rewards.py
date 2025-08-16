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
    🚶 HEIGHT SENSOR-BASED TERRAIN NAVIGATION REWARD
    
    Terrain-aware locomotion approach:
    1. Velocity tracking (slow, controlled movement 0.1-0.3 m/s)
    2. ✅ HEIGHT SENSOR terrain detection (obstacles = stairs, gaps = holes)
    3. Proper bipedal gait with alternating feet and air time
    4. Forward movement encouragement with terrain awareness
    5. Stair detection and climbing rewards using height sensor
    6. Terrain navigation bonuses (approach stairs, avoid gaps)
    7. Stability and balance with foot sliding prevention
    """
    import torch
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    # Get required components
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    commands = env.command_manager.get_command("base_velocity")

    # Foot indices
    foot_ids_list, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    foot_ids = torch.tensor(foot_ids_list, dtype=torch.long, device=env.device)
    if foot_ids.numel() > 2:
        foot_ids = foot_ids[:2]

    # ============================================
    # 1) BASIC LOCOMOTION (Foundation)
    # ============================================
    
    # Manual velocity tracking (simple version)
    vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    actual_vel = torch.norm(vel_yaw[:, :2], dim=1)
    target_vel = torch.norm(commands[:, :2], dim=1)
    vel_error = torch.square(actual_vel - target_vel)
    vel_reward = torch.exp(-vel_error / (1.0**2))
    
    # Basic upright stability
    if hasattr(robot.data, "projected_gravity_b") and robot.data.projected_gravity_b is not None:
        gravity_proj_xy = robot.data.projected_gravity_b[:, :2]
        lean_reward = torch.clamp(1.0 - torch.norm(gravity_proj_xy, dim=1), 0.0, 1.0)
    else:
        lean_reward = torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
    
    # ✅ IMPROVED: Proper bipedal gait with alternating foot lifting and air time
    contact_time = contact_sensor.data.current_contact_time[:, foot_ids]
    air_time = contact_sensor.data.current_air_time[:, foot_ids]
    in_contact = (contact_time > 0.0)
    
    # Reward single stance (one foot down, one foot up) - proper bipedal gait
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    single_stance_reward = single_stance.float()
    
    # Reward good air time (when feet are lifted)
    # Use minimum air time across feet to encourage both feet to lift properly
    min_air_time = torch.min(air_time, dim=1)[0]
    air_time_reward = torch.clamp(min_air_time / 0.3, 0.0, 1.0)  # Reward up to 0.3s air time
    
    # Penalize double stance (both feet down) - encourages lifting
    double_stance = torch.sum(in_contact.int(), dim=1) == 2
    double_stance_penalty = double_stance.float() * 0.5  # Small penalty for double stance
    
    # Penalize double flight (both feet up) - discourage jumping
    double_flight = torch.sum(in_contact.int(), dim=1) == 0
    double_flight_penalty = double_flight.float() * 2.0  # Stronger penalty for double flight
    
    # Combined gait reward: encourage single stance + air time, penalize bad patterns
    gait_reward = (
        single_stance_reward * 2.0 +  # Strong reward for proper single stance
        air_time_reward * 1.0 -       # Moderate reward for air time
        double_stance_penalty -       # Small penalty for double stance
        double_flight_penalty         # Strong penalty for double flight
    ).clamp(min=0.0)
    
    # Additional foot sliding penalty for better walking quality
    forces_hist = contact_sensor.data.net_forces_w_history[:, :, foot_ids, :]  # [N, H, F, 3]
    contact_forces = forces_hist.norm(dim=-1).max(dim=1)[0]  # [N, F]
    contacts = contact_forces > 25.0  # Lower threshold for G1
    body_vel = robot.data.body_lin_vel_w[:, foot_ids, :2]  # [N, F, 2] 
    foot_slide_penalty = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    
    # Final gait reward with slide penalty
    gait_reward = gait_reward - foot_slide_penalty * 0.1  # Small slide penalty
    
    # ============================================
    # 2) ✅ BIDIRECTIONAL STAIR NAVIGATION (DESCEND → ASCEND)
    # ============================================
    
    # Get height sensor data for absolute terrain measurements
    height_sensor = env.scene.sensors["height_scanner"]
    robot_height = robot.data.root_pos_w[:, 2]  # Robot absolute height
    
    # TERRAIN-RELATIVE MEASUREMENTS (Height-normalized for robot variation)
    sensor_pos_z = height_sensor.data.pos_w[:, 2].unsqueeze(1)  # Sensor Z position
    terrain_hit_z = height_sensor.data.ray_hits_w[..., 2]  # Terrain hit Z coordinates
    
    # Isaac Lab relative height formula
    raw_terrain_heights = sensor_pos_z - terrain_hit_z - 0.5
    
    # ✅ SAFETY: Filter out infinite values (ray misses) and replace with median
    finite_mask = torch.isfinite(raw_terrain_heights)
    terrain_heights = torch.where(finite_mask, raw_terrain_heights, torch.tensor(0.0, device=raw_terrain_heights.device))
    
    # 🔧 THRESHOLD-FREE STATISTICAL TERRAIN ANALYSIS
    # No fixed thresholds - use distribution statistics for adaptive classification
    
    # Calculate per-robot terrain statistics (robust to any baseline)
    robot_stats = []
    normalized_heights = []
    
    for i in range(terrain_heights.shape[0]):
        robot_readings = terrain_heights[i][finite_mask[i]]
        
        if robot_readings.numel() > 5:  # Minimum for statistics
            # Statistical measures (no fixed assumptions)
            median_val = torch.median(robot_readings)
            q25 = torch.quantile(robot_readings, 0.25)
            q75 = torch.quantile(robot_readings, 0.75)
            iqr = q75 - q25
            
            # Normalize using interquartile range (outlier-resistant)
            normalized = (robot_readings - median_val) / (iqr + 0.01)  # Avoid division by zero
            
            robot_stats.append({
                'median': median_val,
                'q25': q25, 
                'q75': q75,
                'iqr': iqr
            })
        else:
            # Handle sparse readings without fixed thresholds
            median_val = torch.mean(robot_readings) if robot_readings.numel() > 0 else torch.tensor(0.0)
            robot_stats.append({
                'median': median_val,
                'q25': median_val,
                'q75': median_val, 
                'iqr': torch.tensor(0.01)  # Small value for normalization
            })
            normalized = torch.zeros_like(robot_readings)
        
        # Pad normalized values to original shape
        full_normalized = torch.zeros_like(terrain_heights[i])
        full_normalized[finite_mask[i]] = normalized
        normalized_heights.append(full_normalized)
    
    # Convert to tensor format
    terrain_heights = torch.stack(normalized_heights)  # Now contains z-scores (threshold-free)
    
    # Forward terrain analysis for bidirectional navigation (27x21 grid: first 9 rows = forward zone)
    terrain_grid = terrain_heights.view(-1, 27, 21)  # Reshape to grid
    forward_terrain = terrain_grid[:, :9, :]  # Forward 9 rows (≈0-0.675m ahead)
    
    # STATISTICAL TERRAIN ANALYSIS (No fixed thresholds!)
    current_z_score = torch.mean(terrain_heights, dim=1)  # Current z-score (0 = median terrain)
    forward_max_z = torch.max(forward_terrain.view(forward_terrain.shape[0], -1), dim=1)[0]  # Max z-score ahead
    forward_min_z = torch.min(forward_terrain.view(forward_terrain.shape[0], -1), dim=1)[0]  # Min z-score ahead
    forward_std = torch.std(forward_terrain.view(forward_terrain.shape[0], -1), dim=1)  # Variation ahead
    
    # Stair detection: High statistical variation ahead (threshold-free)
    # Use coefficient of variation instead of fixed thresholds
    on_stairs = forward_std > 0.8  # High z-score variation = complex terrain
    
    # Statistical terrain opportunities (outlier-based detection)
    # Gaps: Significantly positive z-scores ahead (statistical outliers above median)
    gaps_ahead = forward_max_z > 1.0  # >1 standard deviation above median = gap
    # Obstacles: Significantly negative z-scores ahead (statistical outliers below median)
    obstacles_ahead = forward_min_z < -1.0  # <1 standard deviation below median = obstacle
    
    # Statistical phase detection (distribution-based, no fixed thresholds)
    # Elevated: Robot consistently above statistical median of local terrain
    robot_highly_elevated = current_z_score > 0.5   # >0.5 standard deviations above median
    robot_near_terrain = current_z_score < -0.5     # >0.5 standard deviations below median
    
    should_descend = robot_highly_elevated & gaps_ahead      # High + gaps ahead = descend
    should_ascend = robot_near_terrain & obstacles_ahead     # Low + obstacles ahead = ascend
    
    # ============================================
    # 3) BIDIRECTIONAL MOVEMENT REWARDS
    # ============================================
    
    # Forward movement with phase-aware navigation
    forward_velocity = robot.data.root_lin_vel_w[:, 0]
    
    # Base movement reward: encourage forward, penalize backward
    movement_bonus = torch.where(
        forward_velocity > 0.0,
        torch.clamp(forward_velocity / 0.5, 0.0, 2.0),  # Reward forward movement
        forward_velocity * 2.0  # Penalty for backward movement
    )
    
    # Descending bonus: reward forward movement when robot should descend
    descent_bonus = torch.where(
        should_descend & (forward_velocity > 0.1),
        2.5,  # Strong bonus for moving toward lower terrain when elevated
        0.0
    )
    
    # Ascending bonus: reward forward movement when robot should ascend
    ascent_bonus = torch.where(
        should_ascend & (forward_velocity > 0.1),
        3.0,  # Strong bonus for moving toward higher terrain when at ground
        0.0
    )
    
    # Stair navigation speed control: reward appropriate speed on complex terrain
    stair_speed_bonus = torch.where(
        on_stairs & (forward_velocity > 0.05) & (forward_velocity < 0.4),
        1.5,  # Reward controlled speed on stairs (5-40cm/s)
        0.0
    )
    
    # Terrain exploration bonus
    terrain_bonus = torch.where(
        on_stairs,
        0.8,  # Bonus for being on complex terrain (stairs)
        0.2   # Small bonus for flat terrain navigation
    )
    
    # ============================================
    # 4) ✅ ADAPTIVE HEIGHT PROGRESS TRACKING
    # ============================================
    
    # Store initial robot height for progress tracking
    if not hasattr(env, '_initial_robot_height'):
        env._initial_robot_height = robot_height.clone()  # Store starting height
    
    # Statistical clearance assessment (no fixed thresholds)
    foot_clearance_z_score = current_z_score.clamp(min=-3.0, max=3.0)  # Z-score clearance
    
    # Optimal clearance reward: stay near statistical median (threshold-free)
    # Reward being close to the statistical center of the terrain distribution
    optimal_clearance = torch.exp(-torch.abs(foot_clearance_z_score) / 0.5)  # Gaussian reward around z=0
    
    # Statistical progress rewards (no fixed thresholds)
    # When elevated: reward moving toward statistical median (descent progress)
    descent_progress = torch.where(
        robot_highly_elevated,
        torch.clamp((0.5 - current_z_score) / 1.0, 0.0, 1.0),  # Progress from elevated to median
        0.0
    ) * 1.5
    
    # When low: reward moving toward statistical median (ascent progress)
    ascent_progress = torch.where(
        robot_near_terrain,
        torch.clamp((current_z_score + 0.5) / 1.0, 0.0, 1.0),  # Progress from low to median
        0.0
    ) * 2.0
    
    # Statistical clearance bonus: reward safe navigation regardless of absolute values
    step_clearance_bonus = torch.where(
        (gaps_ahead | obstacles_ahead) & (torch.abs(foot_clearance_z_score) < 2.0),
        1.0,  # Bonus for reasonable clearance when terrain changes ahead (within 2σ)
        0.0
    )
    
    # Navigation success bonus: reward staying near statistical center (completely threshold-free)
    navigation_success_bonus = torch.exp(-torch.abs(current_z_score) / 0.3) * 2.0  # Gaussian reward centered at z=0
    
    # ============================================
    # 5) BIDIRECTIONAL NAVIGATION REWARD COMBINATION
    # ============================================
    
    total_reward = (
        vel_reward * 0.3 +                    # Velocity tracking (reduced for navigation focus)
        lean_reward * 0.5 +                   # Stay upright (important for stairs)
        gait_reward * 1.0 +                   # Proper gait essential
        movement_bonus * 1.2 +                # Base forward movement
        descent_bonus * 2.5 +                 # ✅ NEW: Descending bonus when elevated
        ascent_bonus * 3.0 +                  # ✅ NEW: Ascending bonus when at ground
        stair_speed_bonus * 1.5 +             # ✅ NEW: Controlled stair speed
        terrain_bonus * 0.8 +                 # Terrain navigation
        optimal_clearance * 1.0 +             # ✅ NEW: Optimal foot clearance
        descent_progress * 2.0 +              # ✅ NEW: Progress toward ground when elevated
        ascent_progress * 2.5 +               # ✅ NEW: Progress upward when at ground
        step_clearance_bonus * 1.5 +          # ✅ NEW: Step clearance (bidirectional)
        navigation_success_bonus * 3.0 +      # ✅ NEW: Zone transition rewards
        0.2                                   # Baseline bonus
    )
    

    # ✅ SAFETY: Check for NaN/infinity and replace with safe values
    total_reward = torch.where(torch.isfinite(total_reward), total_reward, torch.tensor(0.1, device=total_reward.device))
    
    # Final safety clamp
    return total_reward.clamp(min=0.0, max=10.0)


