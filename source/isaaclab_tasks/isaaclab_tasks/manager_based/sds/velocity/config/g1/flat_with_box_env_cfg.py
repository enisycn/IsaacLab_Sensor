# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SDS G1 Flat Environment with Obstacle Avoidance Visualization Configuration.

This configuration creates terrain specifically designed for VISUALIZING OBSTACLE AVOIDANCE BEHAVIOR:
- Discrete obstacles of varying heights (5-20cm) that force clear avoidance strategies
- Strategic obstacle placement for demonstrating learned navigation behaviors
- Flat corridors for baseline comparison vs obstacle avoidance
- Multiple obstacle types: step-over, stepping stones, obstacle fields, mixed challenges
- Complete environmental sensing suite (height scanner, lidar, IMU)
- Perfect for thesis videos showing humanoid adaptive locomotion capabilities
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
USE_COMPLEX_TERRAIN = True  # Set to False for simple terrain, True for complex height variations

# 🏔️ COMPLEX TERRAIN: Obstacle avoidance visualization terrain
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
        # 20% FLAT CORRIDORS: Clear paths to demonstrate normal walking vs obstacle avoidance
        "flat_corridors": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,  # 20% completely flat - shows baseline walking behavior
        ),
        
        # 30% SMALL DISCRETE OBSTACLES: Low obstacles requiring step-over behavior (5-12cm)
        "small_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.3,
            obstacle_height_range=(0.05, 0.12),  # 5-12cm - forces step-over behavior
            obstacle_width_range=(0.2, 0.4),     # 20-40cm wide - clearly visible obstacles
            num_obstacles=6,                     # 6 obstacles per terrain - spaced for clear avoidance
            platform_width=1.5,                 # 1.5m platform - enough space for maneuvering
        ),
        
        # 25% MEDIUM STEPPING OBSTACLES: Requires precise foot placement with realistic gaps (8-18cm high, 60cm deep gaps)
        "stepping_obstacles": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.25,
            stone_height_max=0.18,              # 18cm max - requires high steps
            stone_width_range=(0.3, 0.5),      # 30-50cm stones - good foot placement targets
            stone_distance_range=(0.4, 0.7),   # 40-70cm gaps - forces strategic stepping
            holes_depth=-0.6,                  # 60cm deep gaps (was -10.0m) - realistic and detectable
            platform_width=1.2,                # 1.2m platform
        ),
        
        # 15% DENSE OBSTACLE FIELD: Multiple small obstacles requiring path planning (3-10cm)
        "obstacle_field": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.3,                     # 30cm grid cells - creates obstacle maze
            grid_height_range=(0.03, 0.10),     # 3-10cm heights - low but visible obstacles
            platform_width=1.0,                # 1m platform
        ),
        
        # 10% MIXED HEIGHT CHALLENGE: Combination for complex avoidance strategies (5-20cm)
        "mixed_heights": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.1,
            obstacle_height_range=(0.05, 0.20),  # 5-20cm - wide range requiring different strategies
            obstacle_width_range=(0.15, 0.6),    # 15-60cm wide - various obstacle sizes
            num_obstacles=8,                     # 8 obstacles - dense field for complex navigation
            platform_width=1.0,                 # 1m platform - challenging navigation
        ),
    },
)

@configclass
class SDSG1FlatWithBoxEnvCfg(SDSG1RoughEnvCfg):
    """SDS Unitree G1 environment with OBSTACLE AVOIDANCE VISUALIZATION terrain + environmental sensors.

    TERRAIN TOGGLE SYSTEM:
    - Set USE_COMPLEX_TERRAIN = True: Discrete obstacles for visualizing avoidance behavior
      * Small obstacles (5-12cm): Step-over behavior demonstration
      * Stepping stones (8-18cm): Precise foot placement visualization  
      * Obstacle fields (3-10cm): Path planning and navigation strategies
      * Mixed challenges (5-20cm): Complex avoidance behavior combinations
    - Set USE_COMPLEX_TERRAIN = False: Simple terrain (70% flat + 30% gentle bumps)
    - Uses EXACT same physics, robot, and command settings
    - ADDS height scanner + lidar for environmental sensing capabilities
    - Perfect for thesis videos demonstrating learned obstacle avoidance capabilities
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
            # COMPLEX: Discrete obstacles for visualizing avoidance behavior
            self.scene.terrain.terrain_generator = COMPLEX_BOX_TERRAIN_CFG
            print("🎯 TERRAIN MODE: COMPLEX obstacle avoidance visualization (discrete obstacles, stepping stones, obstacle fields)")
        else:
            # SIMPLE: Flat terrain with gentle bumps (same as rough_env_cfg)
            self.scene.terrain.terrain_generator = SIMPLE_LEARNING_TERRAIN_CFG
            print("🏞️ TERRAIN MODE: SIMPLE flat terrain with gentle bumps")
        
        # Simple arm positions for natural walking
        self.scene.robot.init_state.joint_pos.update({
            # Arms - using asset defaults for consistency
            "left_shoulder_pitch_joint": 0.35,    # Asset default (arms slightly forward)
            "right_shoulder_pitch_joint": 0.35,   # Asset default (arms slightly forward)
            "left_shoulder_roll_joint": 0.16,     # Asset default (slight outward angle)
            "right_shoulder_roll_joint": -0.16,   # Asset default (slight outward angle)
            ".*_elbow_pitch_joint": 0.87,         # Asset default (natural elbow bend)         
        })
        
        # Enable self-collisions for realistic arm movement
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

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
    
    # Robot-focused tracking camera for optimal environment analysis and gameplay footage
    viewer = ViewerCfg(
        origin_type="asset_root",      # Follow the robot for dynamic tracking
        asset_name="robot",            # Track the robot asset
        env_index=0,                   # Focus on first environment
        eye=(-6.0, 2.0, 4.0),         # 6m back, 2m right, 4m up - robot-focused view
        lookat=(0.0, 0.0, 0.8),       # Look at robot torso height
    )
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 100
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