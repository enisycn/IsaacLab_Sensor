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

# 🎯 TERRAIN COMPLEXITY TOGGLE: Change this ONE line to switch terrain types
USE_COMPLEX_TERRAIN = False  # Set to False for simple terrain, True for complex height variations

# 🏔️ COMPLEX TERRAIN: Box-shaped height variations for environmental sensing testing
COMPLEX_BOX_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,     
    num_cols=8,     
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # Enable curriculum for progressive difficulty
    sub_terrains={
        # 20% FLAT for baseline locomotion
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,  # 20% flat terrain for basic walking
        ),
        
        # 25% STEP TERRAIN: Box-like stepping stones (up to 15cm heights)
        "box_steps": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.25,
            stone_height_max=0.15,           # 15cm max step height
            stone_width_range=(0.4, 0.8),   # 40-80cm wide stones (box-like)
            stone_distance_range=(0.3, 0.6), # 30-60cm between stones
            platform_width=1.0,             # 1m platform size
        ),
        
        # 25% PLATFORM TERRAIN: Elevated box platforms (10-20cm)
        "box_platforms": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.10, 0.20),  # 10-20cm platform heights
            step_width=0.8,                  # 80cm wide platforms (box-shaped)
            platform_width=1.2,             # 1.2m platform size
        ),
        
        # 15% DISCRETE OBSTACLES: Box-shaped height bumps (5-12cm)
        "box_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_height_range=(0.05, 0.12),  # 5-12cm obstacles
            obstacle_width_range=(0.3, 0.6),     # 30-60cm wide (box-like)
            num_obstacles=8,                     # 8 box obstacles per terrain
            platform_width=1.0,                 # 1m platform
        ),
        
        # 15% GENTLE ROUGH: Small height variations (2-8cm) for fine adaptation
        "fine_variations": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, 
            noise_range=(0.02, 0.08),   # 2-8cm variations
            noise_step=0.01, 
            border_width=0.25
        ),
    },
)

@configclass
class SDSG1FlatWithBoxEnvCfg(SDSG1RoughEnvCfg):
    """SDS Unitree G1 environment with CONFIGURABLE terrain complexity + environmental sensors.

    TERRAIN TOGGLE SYSTEM:
    - Set USE_COMPLEX_TERRAIN = True: Box-shaped height variations (steps, platforms, obstacles)
    - Set USE_COMPLEX_TERRAIN = False: Simple terrain (70% flat + 30% gentle bumps)
    - Uses EXACT same physics, robot, and command settings
    - ADDS height scanner + lidar for environmental sensing capabilities
    - Perfect for testing environmental sensing vs. basic locomotion
    """

    def __post_init__(self):
        # Call parent post_init to get all the same settings
        super().__post_init__()

        # 🎯 DYNAMIC GRAVITY CONTROL: Enable/disable based on analysis vs training mode
        # When SDS_ANALYSIS_MODE=true (environment analysis): disable gravity to prevent robot falling
        # When SDS_ANALYSIS_MODE not set (training): enable gravity for realistic physics
        if os.environ.get('SDS_ANALYSIS_MODE', 'false').lower() == 'true':
            # ANALYSIS MODE: Disable gravity so robot stays stable during environmental sensing
            self.sim.gravity = (0.0, 0.0, 0.0)
            print("🔧 SDS ANALYSIS MODE: Gravity DISABLED for stable environmental analysis")
        else:
            # TRAINING MODE: Enable gravity for realistic physics simulation
            self.sim.gravity = (0.0, 0.0, -9.81)
            print("🚀 SDS TRAINING MODE: Gravity ENABLED for realistic physics")

        # 🎯 TERRAIN SELECTION: One-line toggle between simple and complex terrain
        if USE_COMPLEX_TERRAIN:
            # COMPLEX: Box-shaped height variations for environmental sensing testing
            self.scene.terrain.terrain_generator = COMPLEX_BOX_TERRAIN_CFG
            print("🏔️ TERRAIN MODE: COMPLEX box-shaped height variations (steps, platforms, obstacles)")
        else:
            # SIMPLE: Flat terrain with gentle bumps (same as rough_env_cfg)
            self.scene.terrain.terrain_generator = SIMPLE_LEARNING_TERRAIN_CFG
            print("🏞️ TERRAIN MODE: SIMPLE flat terrain with gentle bumps")
        
        # Simple arm positions for natural walking
        self.scene.robot.init_state.joint_pos.update({
            # Arms - simple positions
            "left_shoulder_pitch_joint": 0.0,    
            "right_shoulder_pitch_joint": 0.0,   
            "left_shoulder_roll_joint": 0.0,     
            "right_shoulder_roll_joint": 0.0,    
            ".*_elbow_pitch_joint": 0.2,         
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

        # 3. CONTACT FORCES VISUALIZATION - For foot contact analysis
        # ✅ ENABLE: Contact visualization on existing contact_forces sensor (inherited from base config)
        # This shows REAL contact forces as visual markers where robot feet actually touch the ground
        if hasattr(self.scene, 'contact_forces') and self.scene.contact_forces is not None:
            self.scene.contact_forces.debug_vis = True  # ✅ ENABLE: Visual markers at contact points
            print("🎯 CONTACT VISUALIZATION ENABLED: Real contact forces will be shown as visual markers")
        else:
            print("⚠️ WARNING: contact_forces sensor not found in base configuration")

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
        # This config = rough_env_cfg + height_scanner + lidar + observations + terrain_toggle


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
        
        # ✅ ENABLE: Contact forces visualization for play mode
        if hasattr(self.scene, 'contact_forces') and self.scene.contact_forces is not None:
            self.scene.contact_forces.debug_vis = True
            print("🎯 PLAY MODE: Contact visualization enabled - see real foot contact forces!")
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False 