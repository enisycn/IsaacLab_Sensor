# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Environmental Sensing and Robot Stability Metrics for SDS Feedback System.

This file contains stability metrics that are logged during training to provide
enhanced feedback to GPT for reward function improvement. These metrics are 
SEPARATE from the GPT-generated reward function and will NOT be overwritten.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def height_tracking_accuracy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Compute height tracking accuracy using Isaac Lab ground truth data.
    Measures how well robot maintains appropriate height for G1 humanoid (1.30m total height).
    Simple ground truth measurement without complex terrain/task adaptation.
    
    Returns:
        Height tracking error [N] - absolute error from target height (1.25m)
    """
    robot = env.scene["robot"]
    
    # ✅ ISAAC LAB GROUND TRUTH: Use center of mass position
    current_height = robot.data.root_com_state_w[:, 2]  # [N] - ground truth COM height
    
    # ✅ FIXED TARGET: G1 humanoid normal standing height
    target_height = 1.25  # meters - appropriate for G1 humanoid locomotion
    
    # Height tracking error
    height_error = torch.abs(current_height - target_height)  # [N]
    
    return height_error


def robot_height_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Compute robot height stability using Isaac Lab ground truth center of mass data.
    Measures robot's median height above terrain using ground truth COM position.
    
    Returns:
        Robot height above terrain baseline [N] - ground truth measurement
    """
    robot = env.scene["robot"]
    
    # ✅ ISAAC LAB GROUND TRUTH: Center of mass height
    robot_com_height = robot.data.root_com_state_w[:, 2]  # [N]
    
    # Get height sensor data for terrain baseline
    height_sensor = env.scene.sensors["height_scanner"]
    # ✅ CORRECT ISAAC LAB HEIGHT SENSOR FORMULA
    terrain_heights = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2] - 0.5
    finite_mask = terrain_heights != float('inf')
    
    # Calculate median terrain height for each environment
    robot_heights_above_terrain = torch.zeros(terrain_heights.shape[0], device=terrain_heights.device)
    
    for env_idx in range(terrain_heights.shape[0]):
        env_terrain = terrain_heights[env_idx]
        env_finite = env_terrain[finite_mask[env_idx]]
        
        if env_finite.numel() > 0:
            # Ground truth: Robot COM height - median terrain height
            terrain_median = torch.median(env_finite)
            robot_heights_above_terrain[env_idx] = robot_com_height[env_idx] - terrain_median
        else:
            # ❌ NO FALLBACK: Fail if no valid terrain readings
            raise RuntimeError(f"No valid terrain readings for environment {env_idx}")
            
    return robot_heights_above_terrain


def body_orientation_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Compute body orientation stability using Isaac Lab ground truth projected gravity.
    Measures robot's roll/pitch deviation from upright using ground truth gravity vector.
    
    Returns:
        Body orientation deviation [N] - deviation from upright in degrees (0-180°)
    """
    robot = env.scene["robot"]
    
    # ✅ ISAAC LAB GROUND TRUTH: Projected gravity in body frame
    # This is already computed ground truth data from articulation physics
    projected_gravity_b = robot.data.projected_gravity_b  # [N, 3]
    
    # Calculate roll and pitch from ground truth projected gravity
    # Roll: rotation around x-axis (forward/backward tilt)
    # Pitch: rotation around y-axis (left/right tilt)
    
    # For upright robot: projected_gravity_b should be [0, 0, -9.81]
    # Roll: arctan2(gy, gz), Pitch: arctan2(-gx, sqrt(gy^2 + gz^2))
    
    roll = torch.atan2(projected_gravity_b[:, 1], projected_gravity_b[:, 2])  # [N] radians
    pitch = torch.atan2(-projected_gravity_b[:, 0], 
                       torch.sqrt(projected_gravity_b[:, 1]**2 + projected_gravity_b[:, 2]**2))  # [N] radians
    
    # Total orientation deviation (combined roll and pitch magnitude)
    orientation_deviation_rad = torch.sqrt(roll**2 + pitch**2)  # [N] radians
    
    # ✅ Convert to degrees for better interpretability (0-180° range)
    orientation_deviation_deg = orientation_deviation_rad * 180.0 / torch.pi  # [N] degrees
    
    return orientation_deviation_deg


def terrain_height_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Compute terrain height variance using Isaac Lab ground truth height sensor data.
    Measures terrain roughness under robot using correct Isaac Lab height sensor formula.
    
    Returns:
        Terrain height variance [N] - variance of height readings around robot
    """
    height_sensor = env.scene.sensors["height_scanner"]
    
    # ✅ CORRECT ISAAC LAB HEIGHT SENSOR FORMULA from height sensor guide
    height_readings = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2] - 0.5
    
    # Filter out infinite readings (no ground detected)
    finite_mask = height_readings != float('inf')
    
    # Calculate variance for each environment
    variances = []
    for env_idx in range(height_readings.shape[0]):
        env_readings = height_readings[env_idx]
        env_finite = env_readings[finite_mask[env_idx]]
        
        if env_finite.numel() > 1:
            # Ground truth terrain variance from valid height sensor readings
            variance = torch.var(env_finite)
        else:
            # ❌ NO FALLBACK: Fail if insufficient readings for variance
            raise RuntimeError(f"Insufficient valid height readings for variance calculation in environment {env_idx} (need >1, got {env_finite.numel()})")
        
        variances.append(variance)
    
    return torch.stack(variances)


def terrain_complexity_analysis(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Compute terrain complexity using Isaac Lab ground truth height sensor data.
    Analyzes obstacles, gaps, and extreme terrain features using G1 robot specifications.
    
    Returns:
        Terrain complexity score [N] - weighted complexity based on obstacle/gap detection
    """
    height_sensor = env.scene.sensors["height_scanner"]
    
    # ✅ CORRECT ISAAC LAB HEIGHT SENSOR FORMULA
    height_readings = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2] - 0.5
    
    # ✅ G1 HUMANOID SPECIFICATIONS from height sensor guide
    baseline = 0.209  # G1 robot baseline height (meters)
    threshold = 0.07   # ±7cm threshold for obstacle/gap detection
    
    # Ground truth terrain feature classification
    obstacles = height_readings < (baseline - threshold)      # Below baseline - threshold
    gaps = height_readings > (baseline + threshold)           # Above baseline + threshold  
    extreme_gaps = height_readings == float('inf')           # Infinite readings (cliffs/deep gaps)
    
    total_rays = height_readings.shape[-1]
    
    # Calculate feature ratios for each environment
    obstacle_ratio = obstacles.sum(dim=-1).float() / total_rays    # [N]
    gap_ratio = gaps.sum(dim=-1).float() / total_rays              # [N]
    extreme_ratio = extreme_gaps.sum(dim=-1).float() / total_rays  # [N]
    
    # Weighted complexity score (higher = more dangerous/complex terrain)
    # Obstacles: 2x weight (need navigation)
    # Gaps: 1.5x weight (moderate challenge)  
    # Extreme gaps: 3x weight (high danger)
    complexity = obstacle_ratio * 2.0 + gap_ratio * 1.5 + extreme_ratio * 3.0
    
    return complexity 