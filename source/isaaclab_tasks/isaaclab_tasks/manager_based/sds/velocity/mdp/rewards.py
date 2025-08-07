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
    🔬 MATHEMATICAL HEIGHT-SENSOR REWARD DECOMPOSITION
    
    🎯 PHYSICS-BASED GAP JUMPING WITH HEIGHT SENSOR ONLY
    r_t = Σ w_i * r_i
    
    📋 REWARD COMPONENTS:
    1. Standard Locomotion: velocity tracking, forward progress
    2. Balance & Posture: uprightness, height regulation
    3. Contact & Smoothness: foot slip, control cost
    4. Sensor-Driven Shaping: gap detection, approach speed
    5. Jump-Timing & Clearance: takeoff position, foot clearance
    6. Success & Failure: crossing bonus, collision penalty
    7. Enhanced Forward Bias: prevent backward jumping
    """
    import torch
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    robot = env.scene["robot"]
    height_sensor = env.scene.sensors["height_scanner"]
    contact_sensor = env.scene.sensors["contact_forces"]

    batch_size = robot.data.root_pos_w.shape[0]
    device = robot.data.root_pos_w.device
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 CORE ROBOT STATE EXTRACTION (MOVED UP TO FIX VARIABLE ORDER)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Robot state
    root_pos = robot.data.root_pos_w              # [N, 3]
    root_vel = robot.data.root_lin_vel_w          # [N, 3]
    root_quat = robot.data.root_quat_w            # [N, 4]
    applied_torque = robot.data.applied_torque    # [N, joints]
    
    # Extract key variables
    x_base = root_pos[:, 0]                       # [N] - current x position
    h_pelvis = root_pos[:, 2]                     # [N] - pelvis height
    v_x = root_vel[:, 0]                          # [N] - forward velocity
    v_z = root_vel[:, 2]                          # [N] - vertical velocity
    
    # Body orientation
    gravity_proj = robot.data.projected_gravity_b[:, :2]  # [N, 2]
    u_torso_y = 1.0 - torch.norm(gravity_proj, dim=1)    # [N] - uprightness (0=fallen, 1=upright)
    
    # Robot height target (G1 humanoid)
    h_star = 0.74  # G1 pelvis height target
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 SENSOR DATA EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Height sensor data (Isaac Lab formula)
    sensor_pos = height_sensor.data.pos_w[:, 2].unsqueeze(1)  # [N, 1]
    ray_hits = height_sensor.data.ray_hits_w[..., 2]          # [N, rays]
    terrain_heights = sensor_pos - ray_hits - 0.5            # [N, rays] - subtract 0.5m offset
    
    # Handle invalid readings and filter out readings from unstable robots
    # ✅ STABILITY FILTER: Exclude readings from tilted/fallen robots
    robot_upright = u_torso_y > 0.7  # Robot is reasonably upright (gravity projection)
    robot_height_ok = h_pelvis > (h_star - 0.2)  # Robot hasn't fallen too much
    robot_stable = robot_upright & robot_height_ok  # Combined stability check
    
    terrain_heights = torch.where(
        torch.isfinite(terrain_heights) & (torch.abs(terrain_heights) < 10.0),
        terrain_heights,
        torch.zeros_like(terrain_heights)
    )
    
    # ✅ SIMPLIFIED: Use fixed baseline consistently for training stability
    baseline_height_standard = 0.209  # Standard G1 baseline from Isaac Lab guide
    baseline_height = torch.full((batch_size,), baseline_height_standard, device=device)
    
    # Apply stability filter - zero out readings from unstable robots
    terrain_heights = torch.where(
        robot_stable.unsqueeze(1).expand_as(terrain_heights),
        terrain_heights,
        torch.full_like(terrain_heights, baseline_height_standard)  # Use baseline for unstable robots
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🕳️ GAP DETECTION USING HEIGHT SENSOR
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Configuration (following comprehensive guide)
    D_look = 1.5              # Look-ahead distance (m)
    
    # ✅ SYMMETRIC BASELINE THRESHOLDS: ±7cm classification zone
    baseline_threshold = 0.07         # ±7cm threshold for balanced detection
    gap_threshold = baseline_threshold      # 7cm above baseline = gaps  
    obstacle_threshold = baseline_threshold # 7cm below baseline = obstacles
    
    # Extract rays for gap detection (use ALL rays for unbiased detection)
    forward_heights = terrain_heights  # [N, all_rays] - Use all sensor data for fair detection
    
    # ✅ CORRECT: Isaac Lab terrain classification (following guide exactly)
    # OBSTACLES: Lower readings (< baseline - threshold) = Terrain HIGHER than expected
    obstacle_detected_rays = forward_heights < (baseline_height.unsqueeze(-1) - obstacle_threshold)  # [N, all_rays]
    
    # GAPS: Higher readings (> baseline + threshold) = Terrain LOWER than expected
    gap_detected_rays = forward_heights > (baseline_height.unsqueeze(-1) + gap_threshold)  # [N, all_rays]
    
    # NORMAL TERRAIN: Between thresholds = Expected terrain level
    normal_terrain_rays = ~obstacle_detected_rays & ~gap_detected_rays & torch.isfinite(forward_heights)  # [N, all_rays]
    
    # EXTREME GAPS: Infinite readings (no ground detected within range)
    extreme_gap_rays = ~torch.isfinite(forward_heights)  # [N, all_rays]
    
    # Summary detection flags
    any_gap_detected = torch.any(gap_detected_rays | extreme_gap_rays, dim=1)  # [N] - any gaps ahead
    any_obstacle_detected = torch.any(obstacle_detected_rays, dim=1)  # [N] - any obstacles ahead
    mostly_normal_terrain = torch.sum(normal_terrain_rays, dim=1) > (forward_heights.shape[1] * 0.6)  # [N] - 60%+ normal
    
    # ✅ SIMPLIFIED: Use consistent gap distance for training stability
    d_gap = torch.where(any_gap_detected, 
                       torch.full((batch_size,), 0.5, device=device),  # Assume gaps ~50cm away
                       torch.full((batch_size,), D_look + 1.0, device=device))  # No gap detected
    
    gap_in_range = any_gap_detected & (d_gap < D_look)  # [N] - gaps detected and within range
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 👣 CONTACT STATE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Get foot contact forces
    foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    if len(foot_ids) >= 2:
        foot_ids = torch.tensor(foot_ids[:2], dtype=torch.long, device=device)  # Take first 2 feet
        contact_forces = contact_sensor.data.net_forces_w[:, foot_ids, 2]       # [N, 2] - vertical forces
        
        CONTACT_THRESHOLD = 15.0  # Newtons
        foot_contacts = contact_forces > CONTACT_THRESHOLD                      # [N, 2]
        both_feet_contact = torch.all(foot_contacts, dim=1)                     # [N]
        any_foot_contact = torch.any(foot_contacts, dim=1)                      # [N]
        
        # Foot positions for clearance calculation
        foot_pos = robot.data.body_pos_w[:, foot_ids, :]                        # [N, 2, 3]
        foot_heights = foot_pos[:, :, 2]                                        # [N, 2] - foot z positions
        
        contact_sensor_working = True
        
    else:
        # ❌ CONTACT SENSORS NOT WORKING: Use fallback detection
        print(f"⚠️ WARNING: Contact sensors not working! Found {len(foot_ids)} bodies, need 2")
        print(f"   Available bodies in contact sensor: {contact_sensor.body_names}")
        
        # Fallback: Use height-based contact detection
        both_feet_contact = h_pelvis < (h_star + 0.1)  # Assume grounded if pelvis is low
        any_foot_contact = h_pelvis < (h_star + 0.1)   # Assume grounded if pelvis is low
        foot_heights = h_pelvis.unsqueeze(1).repeat(1, 2) - 0.7  # Estimate foot height
        
        contact_sensor_working = False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 REWARD COMPUTATION - 7 CATEGORIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    reward = torch.zeros(batch_size, device=device)
    
    # ─── 1. STANDARD LOCOMOTION ───────────────────────────────────────────────
    
    # Forward-velocity tracking
    commands = env.command_manager.get_command("base_velocity")
    if commands is not None:
        v_cmd = commands[:, 0]  # [N] - commanded forward velocity
        alpha_v = 2.0
        r_vel = torch.exp(-alpha_v * (v_x - v_cmd) ** 2)
        w_vel = 1.0
        reward += w_vel * r_vel
    
    # ✅ LATERAL VELOCITY PENALTY: Discourage sideways drift (from Zhuang et al.)
    # r_lateral = -α₂·|v_y|² - penalize lateral movement
    v_y = root_vel[:, 1]  # [N] - lateral velocity
    alpha_lateral = 1.0
    r_lateral = -alpha_lateral * (v_y ** 2)
    w_lateral_vel = 0.5
    reward += w_lateral_vel * r_lateral
    
    # ✅ YAW STABILITY BONUS: Reward staying straight (from Zhuang et al.)  
    # r_yaw = α₃·e^(-|ω_yaw|) - bonus for minimal yaw rotation
    omega_yaw = root_vel[:, 2]  # [N] - yaw angular velocity (assuming this is stored in root_vel)
    # Note: If angular velocity is in robot.data.root_ang_vel_w, use that instead
    if hasattr(robot.data, 'root_ang_vel_w'):
        omega_yaw = robot.data.root_ang_vel_w[:, 2]  # [N] - yaw angular velocity
    
    alpha_yaw = 2.0
    r_yaw_stability = alpha_yaw * torch.exp(-torch.abs(omega_yaw))
    w_yaw = 0.3
    reward += w_yaw * r_yaw_stability
    
    # Progress along x (only positive progress)
    # Store previous x position (simplified: assume constant dt)
    if not hasattr(env, '_prev_x_base'):
        env._prev_x_base = x_base.clone()
    
    x_progress = torch.clamp(x_base - env._prev_x_base, 0.0, 1.0)  # [N] - positive progress only
    w_prog = 0.5
    reward += w_prog * x_progress
    env._prev_x_base = x_base.clone()  # Update for next step
    
    # ✅ WAYPOINT ATTRACTION REWARDS (for Fixed Gap Challenge - Terrain Type 6)
    # Target waypoint is at (0, 3) in world coordinates
    target_x = 3.0  # Target X position (3m forward from spawn)
    target_y = 0.0  # Target Y position (stay on centerline)
    
    # Calculate distance to waypoint
    current_x = root_pos[:, 0]  # Robot's current X position
    current_y = root_pos[:, 1]  # Robot's current Y position
    
    distance_to_waypoint = torch.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)  # [N]
    
    # 🎯 DISTANCE-BASED ATTRACTION: Stronger reward as robot gets closer to waypoint
    max_distance = 5.0  # Maximum expected distance (robot spawns ~3m away)
    normalized_distance = torch.clamp(distance_to_waypoint / max_distance, 0.0, 1.0)  # [N] - [0,1]
    proximity_reward = (1.0 - normalized_distance) * 2.0  # Higher reward when closer (max +2.0)
    w_proximity = 0.8
    reward += w_proximity * proximity_reward
    
    # 🏆 WAYPOINT REACHED BONUS: Major reward for reaching the target
    waypoint_threshold = 0.5  # 50cm radius around target (0, 3)
    waypoint_reached = distance_to_waypoint < waypoint_threshold  # [N] - boolean
    waypoint_bonus = waypoint_reached.float() * 10.0  # +10.0 bonus for reaching waypoint
    w_waypoint = 1.0
    reward += w_waypoint * waypoint_bonus
    
    # 🎯 DIRECTIONAL GUIDANCE: Reward moving in the correct direction toward waypoint
    # Calculate direction vector to waypoint
    direction_to_waypoint_x = target_x - current_x  # [N] - X direction to target
    direction_to_waypoint_y = target_y - current_y  # [N] - Y direction to target
    
    # Normalize direction vector
    direction_magnitude = torch.sqrt(direction_to_waypoint_x**2 + direction_to_waypoint_y**2)  # [N]
    direction_magnitude = torch.clamp(direction_magnitude, min=0.1)  # Avoid division by zero
    direction_to_waypoint_x_norm = direction_to_waypoint_x / direction_magnitude  # [N]
    direction_to_waypoint_y_norm = direction_to_waypoint_y / direction_magnitude  # [N]
    
    # Reward velocity aligned with waypoint direction
    velocity_alignment = (root_vel[:, 0] * direction_to_waypoint_x_norm + 
                         root_vel[:, 1] * direction_to_waypoint_y_norm)  # [N] - dot product
    alignment_reward = torch.clamp(velocity_alignment, 0.0, 2.0)  # Only positive alignment
    w_alignment = 0.6
    reward += w_alignment * alignment_reward
    
    # 🚧 LATERAL DEVIATION PENALTY: Penalize moving away from centerline (Y=0)
    lateral_deviation = torch.abs(current_y)  # [N] - distance from centerline
    lateral_penalty = -torch.clamp(lateral_deviation, 0.0, 1.0) * 1.0  # Penalty for going off-track
    w_lateral = 0.3
    reward += w_lateral * lateral_penalty
    
    # ─── 2. BALANCE & POSTURE ─────────────────────────────────────────────────
    
    # Torso uprightness
    w_orient = 0.2
    r_orient = torch.clamp(u_torso_y, 0.0, 1.0)
    reward += w_orient * r_orient
    
    # Pelvis-height regularizer
    alpha_h = 1.0
    r_height = torch.exp(-alpha_h * (h_pelvis - h_star) ** 2)
    w_height = 0.2
    reward += w_height * r_height
    
    # ─── 3. CONTACT & SMOOTHNESS ───────────────────────────────────────────────
    
    # ✅ ALIVE BONUS: Constant survival reward (from parkour research)
    # Encourages robot to stay upright and avoid falling into gaps
    alive_bonus = 2.0  # Constant +2.0 per timestep for staying alive
    w_alive = 1.0
    reward += w_alive * alive_bonus
    
    # ✅ IMPROVED ENERGY PENALTY: Joint torque × velocity (from parkour research)
    # More realistic energy cost than just torque squared
    joint_velocities = robot.data.joint_vel  # [N, joints]
    energy_cost = torch.sum(torch.abs(applied_torque * joint_velocities), dim=1)  # [N]
    lambda_energy = 5e-4  # Reduced from torque-only penalty
    r_energy = -lambda_energy * energy_cost
    w_energy = 1.0
    reward += w_energy * r_energy
    
    # Control cost (keep original for comparison)
    lambda_a = 1e-3
    r_act = -lambda_a * torch.sum(applied_torque ** 2, dim=1)
    w_act = 0.5  # Reduced weight since we have energy penalty
    reward += w_act * r_act
    
    # Foot-slip penalty (simplified: penalize high foot velocities during contact)
    if len(foot_ids) >= 2:
        # Get foot velocities (simplified approximation)
        body_lin_vel = robot.data.body_lin_vel_w[:, foot_ids, :]  # [N, 2, 3]
        foot_vel_horizontal = torch.norm(body_lin_vel[:, :, :2], dim=2)  # [N, 2]
        
        # Penalize slip only when foot is in contact
        slip_penalty = torch.sum(foot_vel_horizontal * foot_contacts.float(), dim=1)  # [N]
        beta = 0.1
        r_slip = -beta * slip_penalty
        w_slip = 0.1
        reward += w_slip * r_slip
    
    # ─── 4. SENSOR-DRIVEN SHAPING ─────────────────────────────────────────────
    
    # Gap detection reward
    C_det = 1.0
    r_detect = gap_in_range.float() * C_det
    w_detect = 0.3
    reward += w_detect * r_detect
    
    # Approach-speed shaping (when gap detected, target optimal takeoff speed)
    v_star = 1.2  # Optimal approach speed for jumping
    alpha_app = 0.5
    r_app = torch.exp(-alpha_app * (v_x - v_star) ** 2) * gap_in_range.float()
    w_app = 0.5
    reward += w_app * r_app
    
    # ─── 5. JUMP-TIMING & CLEARANCE ───────────────────────────────────────────
    
    # Take-off position reward (when airborne and gap was detected)
    is_airborne = ~any_foot_contact & (h_pelvis > h_star + 0.05)  # Airborne = no contact + 5cm above target
    
    if torch.any(is_airborne & gap_in_range):
        # Optimal takeoff distance before gap
        L_star = 0.3  # Takeoff 30cm before gap edge
        x_gap = x_base - d_gap  # Estimate gap start position
        takeoff_error = torch.abs(x_base - x_gap - L_star)
        
        alpha_to = 2.0
        r_to = torch.exp(-alpha_to * takeoff_error ** 2) * is_airborne.float() * gap_in_range.float()
        w_to = 1.0
        reward += w_to * r_to
    
    # ✅ ENHANCED: Stronger jumping encouragement when gaps ahead
    # Penalize staying grounded when gaps are detected ahead
    grounded_near_gap = any_foot_contact & gap_in_range & (d_gap < 1.0)  # Within 1.0m of gap
    grounded_penalty = grounded_near_gap.float() * -2.0  # Stronger penalty for not jumping
    reward += grounded_penalty
    
    # Bonus for attempting to jump when gap ahead (even if not perfectly airborne)
    attempting_jump = (v_z > 0.1) & gap_in_range  # More sensitive upward velocity detection
    jump_attempt_bonus = attempting_jump.float() * 5.0  # Higher reward for jump attempts
    reward += jump_attempt_bonus
    
    # Additional bonus for higher jump attempts
    strong_jump_attempt = (v_z > 0.3) & gap_in_range  # Strong upward velocity
    strong_jump_bonus = strong_jump_attempt.float() * 3.0  # Extra bonus for strong jumps
    reward += strong_jump_bonus
    
    # Foot clearance (when airborne, reward high foot positions)
    if torch.any(is_airborne):
        z_lip = baseline_height.unsqueeze(1)  # Expand [N] to [N, 1] to match foot_heights [N, 2]
        foot_clearance = torch.clamp(foot_heights - z_lip, 0.0, 1.0)  # [N, 2]
        r_clear = torch.sum(foot_clearance, dim=1) * is_airborne.float()  # [N]
        w_clear = 0.5
        reward += w_clear * r_clear
    
    # ✅ FOOT EDGE CLEARANCE PENALTY: Prevent stepping too close to gap edges (from parkour research)
    # Critical for safe gap jumping - penalize foot contacts near gap edges
    if len(foot_ids) >= 2:
        # Define gap edge zone based on gap detection
        gap_edge_threshold = 0.3  # 30cm safety zone around detected gaps
        
        # Get foot positions relative to robot
        foot_pos_relative = foot_pos - root_pos.unsqueeze(1)  # [N, 2, 3] - relative to robot center
        foot_forward_distance = foot_pos_relative[:, :, 0]     # [N, 2] - how far forward each foot is
        
        # Check if robot is near a gap (within sensor range)
        near_gap = gap_in_range  # [N] - boolean, robot detecting gap ahead
        
        # For robots near gaps, penalize foot contacts too close to gap edge
        # Gap is at ~2m forward, so danger zone is 1.7-2.3m forward from robot
        gap_approach_zone = (foot_forward_distance > 1.4) & (foot_forward_distance < 2.6)  # [N, 2]
        
        # Edge contact penalty: foot in contact AND in danger zone AND robot near gap
        edge_contacts = foot_contacts & gap_approach_zone & near_gap.unsqueeze(1)  # [N, 2]
        edge_contact_count = torch.sum(edge_contacts.float(), dim=1)  # [N] - number of risky contacts
        
        # Strong penalty for edge contacts (prevents edge clipping)
        alpha_clear = 3.0  # Strong penalty coefficient
        r_edge_clearance = -alpha_clear * edge_contact_count  # [N]
        w_edge_clearance = 1.0
        reward += w_edge_clearance * r_edge_clearance
        
        # ✅ ENHANCED: Reward proper takeoff positioning
        # Encourage taking off BEFORE reaching the gap edge (at ~1.5m forward)
        optimal_takeoff_zone = (foot_forward_distance > 1.2) & (foot_forward_distance < 1.8)  # [N, 2]
        takeoff_contacts = foot_contacts & optimal_takeoff_zone & near_gap.unsqueeze(1)  # [N, 2]
        
        # Bonus for last contact in optimal takeoff zone
        takeoff_bonus = torch.sum(takeoff_contacts.float(), dim=1) * 0.5  # [N]
        reward += takeoff_bonus
    
    # ─── 6. SUCCESS & FAILURE ─────────────────────────────────────────────────
    
    # Successful cross bonus (when past the gap)
    gap_width = 0.9  # Approximate gap width from terrain config (80-100cm)
    x_gap_end = x_base - d_gap + gap_width  # Estimate gap end position
    successfully_crossed = (x_base > x_gap_end) & gap_in_range
    
    C_succ = 5.0
    r_succ = successfully_crossed.float() * C_succ
    w_succ = 1.0
    reward += w_succ * r_succ
    
    # Collision penalty (simplified: if pelvis drops too low)
    collision_detected = h_pelvis < 0.5  # Below safe height
    C_col = 2.0
    r_col = -collision_detected.float() * C_col
    w_col = 1.0
    reward += w_col * r_col
    
    # ✅ PENETRATION PENALTY: Discourage "walking through" gaps (from Zhuang et al.)
    # r_penetrate = -Σ(α₅·1[p] + α₆·d(p))·vₓ - penalize being inside gap zones
    if torch.any(gap_in_range):
        # Check if robot is in "penetration zone" - too close to gap center
        gap_center_x = 2.0  # Gap is at ~2m forward from spawn
        gap_center_y = 0.0  # Gap is on centerline
        
        # Distance to gap center
        dist_to_gap_center = torch.sqrt((x_base - gap_center_x)**2 + (root_pos[:, 1] - gap_center_y)**2)
        
        # Penetration zone: within 0.5m of gap center (robot shouldn't be here during approach)
        penetration_threshold = 0.5  # 50cm radius around gap center
        in_penetration_zone = (dist_to_gap_center < penetration_threshold) & gap_in_range  # [N]
        
        # Penetration depth: how deep into the dangerous zone
        penetration_depth = torch.clamp(penetration_threshold - dist_to_gap_center, 0.0, penetration_threshold)  # [N]
        
        # Penetration penalty: binary flag + depth penalty, scaled by forward velocity
        alpha_penetrate_flag = 2.0   # Binary penetration penalty
        alpha_penetrate_depth = 5.0  # Depth-based penalty
        
        penetration_penalty = -(alpha_penetrate_flag * in_penetration_zone.float() + 
                              alpha_penetrate_depth * penetration_depth) * torch.clamp(v_x, 0.0, 2.0)
        
        w_penetrate = 1.0
        reward += w_penetrate * penetration_penalty
        
        # ✅ SAFE APPROACH BONUS: Reward staying at safe distance while approaching
        # Encourage approaching gap but not getting too close before jumping
        safe_approach_distance = torch.clamp(1.5 - dist_to_gap_center, 0.0, 1.0)  # Closer to 1.5m = better
        approach_bonus = safe_approach_distance * gap_in_range.float() * 0.5
        reward += approach_bonus
    
    # ─── 7. ENHANCED FORWARD BIAS (Anti-Backward Jumping) ─────────────────────
    
    # Strong forward movement bonus
    forward_bonus = torch.clamp(v_x, 0.0, 2.0) * 2.0  # +4.0 max for forward velocity
    reward += forward_bonus
    
    # Backward movement penalty
    backward_penalty = torch.clamp(-v_x, 0.0, 2.0) * -3.0  # -6.0 max for backward velocity
    reward += backward_penalty
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🐛 ENHANCED DEBUG OUTPUT (Show terrain stats every time)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Show key info every 200 steps
    if hasattr(env, '_debug_counter'):
        env._debug_counter += 1
    else:
        env._debug_counter = 0
        
    if env._debug_counter % 200 == 0:  # Less frequent debug output
        gap_count = torch.sum(gap_in_range).item()
        obstacle_count = torch.sum(any_obstacle_detected).item()
        airborne_count = torch.sum(is_airborne).item()
        normal_count = torch.sum(mostly_normal_terrain).item()
        
        # ✅ ALWAYS show terrain classification stats
        total_rays = forward_heights.shape[1] * batch_size
        total_obstacle_rays = torch.sum(obstacle_detected_rays).item()
        total_gap_rays = torch.sum(gap_detected_rays).item()
        total_normal_rays = torch.sum(normal_terrain_rays).item()
        
        print(f"🔍 TERRAIN CLASSIFICATION STATS:")
        print(f"   Total environments: {batch_size}")
        print(f"   Total rays: {total_rays} ({forward_heights.shape[1]} per robot)")
        print(f"   Baseline: {baseline_height_standard:.3f}m (fixed)")
        print(f"   Thresholds: obstacles <{baseline_height_standard - obstacle_threshold:.3f}m, gaps >{baseline_height_standard + gap_threshold:.3f}m")
        
        print(f"   📊 RAY CLASSIFICATION:")
        print(f"     🔺 Obstacle rays: {total_obstacle_rays}/{total_rays} ({100*total_obstacle_rays/total_rays:.1f}%)")
        print(f"     🕳️ Gap rays: {total_gap_rays}/{total_rays} ({100*total_gap_rays/total_rays:.1f}%)")
        print(f"     🏞️ Normal rays: {total_normal_rays}/{total_rays} ({100*total_normal_rays/total_rays:.1f}%)")
        
        print(f"   🤖 ENVIRONMENT DETECTION:")
        print(f"     Envs with obstacles: {obstacle_count}/{batch_size} ({100*obstacle_count/batch_size:.1f}%)")
        print(f"     Envs with gaps: {gap_count}/{batch_size} ({100*gap_count/batch_size:.1f}%)")
        print(f"     Envs with normal terrain: {normal_count}/{batch_size} ({100*normal_count/batch_size:.1f}%)")
        
        # Show height reading statistics for debugging
        valid_heights = forward_heights[torch.isfinite(forward_heights)]
        if len(valid_heights) > 0:
            min_height = valid_heights.min().item()
            max_height = valid_heights.max().item()
            mean_height = valid_heights.mean().item()
            print(f"   📏 HEIGHT READINGS:")
            print(f"     Range: {min_height:.3f}m to {max_height:.3f}m")
            print(f"     Average: {mean_height:.3f}m")
            print(f"     Expected baseline: {baseline_height_standard:.3f}m")
            print(f"     Difference from baseline: {abs(mean_height - baseline_height_standard):.3f}m")
        
        if gap_count > 0:
            print(f"🕳️ GAPS: {gap_count}/{batch_size} ({100*gap_count/batch_size:.1f}%)")
            jump_attempts = torch.sum(attempting_jump).item()
            if jump_attempts > 0:
                print(f"   🚀 Jump attempts: {jump_attempts}/{batch_size}")
            if torch.any(successfully_crossed):
                print(f"   ✅ Crossings: {torch.sum(successfully_crossed).item()}/{batch_size}")
        
        if obstacle_count > 0:
            print(f"🔺 OBSTACLES: {obstacle_count}/{batch_size} ({100*obstacle_count/batch_size:.1f}%)")
        
        if not contact_sensor_working:
            print(f"⚠️ Contact sensors: FALLBACK MODE")
        
        # ✅ WAYPOINT PROGRESS TRACKING
        waypoint_reached_count = torch.sum(waypoint_reached).item()
        avg_distance_to_waypoint = distance_to_waypoint.mean().item()
        closest_distance = distance_to_waypoint.min().item()
        
        print(f"🎯 WAYPOINT PROGRESS:")
        print(f"   Target: (3.0, 0.0) - 50cm acceptance radius")
        print(f"   Robots reached waypoint: {waypoint_reached_count}/{batch_size} ({100*waypoint_reached_count/batch_size:.1f}%)")
        print(f"   Average distance to target: {avg_distance_to_waypoint:.2f}m")
        print(f"   Closest robot distance: {closest_distance:.2f}m")
        if waypoint_reached_count > 0:
            print(f"   🏆 SUCCESS: {waypoint_reached_count} robots reached the target!")
        
        print()  # Add spacing between debug outputs
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 FINAL PROCESSING
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Clamp to training-friendly range
    final_reward = reward.clamp(min=-10.0, max=25.0)
    
    return final_reward 