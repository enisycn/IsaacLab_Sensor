# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SDS G1 Flat Environment with Box-Shaped Height Variations Configuration.

This configuration creates terrain with BOX-SHAPED HEIGHT PATTERNS:
- Rectangular height platforms and depressions (box-like terrain features)
- Complete environmental sensing suite (height scanner, lidar, IMU)
- Perfect for learning sensor-aware locomotion with simple geometric obstacles
- Box-shaped terrain features for testing navigation and adaptation
"""

import os
import torch
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.envs import ViewerCfg  # Add ViewerCfg import for camera configuration
import isaaclab.envs.mdp as base_mdp
from isaaclab_tasks.manager_based.sds.velocity import mdp

from .rough_env_cfg import SDSG1RoughEnvCfg, SIMPLE_LEARNING_TERRAIN_CFG  # Import the same terrain!


@configclass
class SDSG1FlatWithBoxEnvCfg(SDSG1RoughEnvCfg):
    """SDS Unitree G1 environment with SAME simple terrain as rough_env_cfg + environmental sensors.

    UNIFIED CONFIGURATION:
    - Uses EXACT same terrain as rough_env_cfg (70% flat + 30% gentle bumps)
    - Uses EXACT same command settings (0.3-0.6 m/s forward, 2% standing)
    - Uses EXACT same physics and robot configuration
    - ADDS height scanner + lidar for environmental sensing capabilities
    - Perfect for learning basic locomotion + environmental awareness
    """

    def __post_init__(self):
        # Call parent post_init to get all the same settings
        super().__post_init__()

        # 🎯 CRITICAL: Use EXACT same terrain as rough_env_cfg (no changes!)
        self.scene.terrain.terrain_generator = SIMPLE_LEARNING_TERRAIN_CFG  # Same 70% flat + 30% bumps
        
        # ✅ FIX WEIRD ARM POSITIONS: Natural walking pose for enhanced environment
        # Override the default arm positions to be more natural for walking
        # FIXED: Use EXACT same naming convention as original G1 config
        self.scene.robot.init_state.joint_pos.update({
            # NATURAL ARM POSITIONS for walking (match original naming convention)
            "left_shoulder_pitch_joint": 0.0,    # ✅ Specific name (was 0.35 forward)
            "right_shoulder_pitch_joint": 0.0,   # ✅ Specific name (was 0.35 forward)
            "left_shoulder_roll_joint": 0.0,     # ✅ Specific name (was 0.16 out)
            "right_shoulder_roll_joint": 0.0,    # ✅ Specific name (was -0.16 out)  
            ".*_elbow_pitch_joint": 0.2,         # ✅ Pattern (was 0.87 too bent)
        })
        
        # Multiple environments for parallel training
        self.scene.num_envs = 512  # 512 robots for good parallelization
        self.scene.env_spacing = 2.5
        
        # === ENHANCED SENSING FOR ENVIRONMENTAL ANALYSIS (VISUALIZATION ENABLED) ===
        
        # 1. HEIGHT SCANNER - For terrain height detection
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link", 
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.6)),  # 60cm above torso for coverage
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.15,  # ⚡ IMPROVED: 15cm resolution (was 20cm) - 33% better gap detection
                size=[2.0, 1.5],  # 2m x 1.5m scanning area  
            ),
            debug_vis=True,  # ✅ ENABLE: Red dots showing height scanner rays
            mesh_prim_paths=["/World/ground"],
            max_distance=3.0,  # 3m range 
            update_period=0.02,  # 50Hz updates
        )
        
        # 2. LIDAR SENSOR - For 360° environmental awareness
        self.scene.lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.4)),  # 40cm above torso
            attach_yaw_only=True,
            pattern_cfg=patterns.LidarPatternCfg(
                channels=8,  # 8 vertical channels
                vertical_fov_range=(-15.0, 15.0),  # 30° vertical field of view
                horizontal_fov_range=(-90.0, 90.0),  # 180° front coverage
                horizontal_res=10.0,  # 10° horizontal resolution
            ),
            debug_vis=True,  # ✅ ENABLE: Red dots showing LiDAR rays
            mesh_prim_paths=["/World/ground"],
            max_distance=5.0,  # 5m range
            update_period=0.02,  # 50Hz updates
        )

        # === ENVIRONMENTAL OBSERVATIONS (PROPERLY NORMALIZED) ===

        # Height scan observation - NORMALIZED to reasonable range
        self.observations.policy.height_scan = ObsTerm(
            func=base_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.01, n_max=0.01),  # 1cm noise
            clip=(-0.5, 3.0),  # Range for 3m max distance  
            scale=0.286,       # ✅ NORMALIZE: 1.0 / 3.5 = 0.286 -> maps [-0.5,3.0] to [-0.143, 0.857] ≈ [0,1] range
        )

        # Lidar range observation - NORMALIZED to [0,1] range  
        self.observations.policy.lidar_range = ObsTerm(
            func=mdp.lidar_range,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 5cm noise
            clip=(0.1, 15.0),  # 15m sensor range
            scale=0.067,       # ✅ NORMALIZE: 1.0 / 14.9 = 0.067 -> maps [0.1,15.0] to [0.007, 1.005] ≈ [0,1]
        )

        # 📝 NOTE: Everything else (commands, physics, robot, rewards) is IDENTICAL to rough_env_cfg!
        # This config = rough_env_cfg + height_scanner + lidar + observations


@configclass
class SDSG1FlatWithBoxEnvCfg_PLAY(SDSG1FlatWithBoxEnvCfg):
    
    # High-angle overhead camera for spectacular multi-robot video footage
    viewer = ViewerCfg(
        origin_type="world",           # Fixed world position for stable overhead view
        eye=(-35.0, 0.0, 60.0),      # Higher overhead (35m) and positioned to the left (-15m) and back (-5m)
        lookat=(0.0, 0.0, 0.0),       # Look down at ground center
    )
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # ✅ ENSURE: Sensor visualization is enabled for play mode
        self.scene.height_scanner.debug_vis = True
        self.scene.lidar.debug_vis = True
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False 