# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SDS G1 Flat Environment with Terrain Type Selection Configuration.

This configuration creates different terrain types for various training scenarios:
- 0: Simple terrain (70% flat + 30% gentle bumps) - for basic locomotion learning
- 1: Gaps terrain - mixed flat + gaps: 30% flat areas, 35% easy gaps (15-25cm), 35% medium gaps (25-35cm), reduced depth 30-40cm
- 2: Obstacles terrain - for discrete obstacle avoidance  
- 3: Stairs terrain - for stair climbing and step navigation
- Complete environmental sensing suite (height scanner, lidar, IMU)
- Perfect for progressive training and thesis videos demonstrating different capabilities
"""

import os
import torch
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
# Removed complex/jump/fixed terrains: mesh configs not needed
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.envs import ViewerCfg  # Add ViewerCfg import for camera configuration
import isaaclab_tasks.manager_based.sds.velocity.mdp as base_mdp
from isaaclab_tasks.manager_based.sds.velocity import mdp

from .rough_env_cfg import SDSG1RoughEnvCfg, SIMPLE_LEARNING_TERRAIN_CFG  # Import the same terrain!

# 🚀 REMOVED: Custom gap terrain functions - now using standard Isaac Lab height field terrain like Environment 4
# Environment 1 now uses terrain_gen.HfSteppingStonesTerrainCfg (same as Environment 4's approach)
# This creates terrain with gaps cut INTO the surface rather than separate floating platforms

# 🎯 TERRAIN TYPE SELECTION: Change this value to switch terrain types
# 0: Simple terrain (70% flat + 30% gentle bumps)
# 1: Gaps terrain (mixed flat + gaps: 30% flat areas, 35% easy gaps 15-25cm, 35% medium gaps 25-35cm, depth 30-40cm)
# 2: Obstacles terrain (discrete obstacle avoidance)
# 3: Stairs terrain (stair climbing and steps)
TERRAIN_TYPE = 3 # Default to SIMPLE for baseline training

# 🔧 SENSORS CONTROL: Toggle environmental sensing capabilities
# SENSORS_ENABLED = True:  Full environmental sensing (height scanner + lidar + sensor observations)
# SENSORS_ENABLED = False: Proprioceptive-only (no environmental sensors, only robot internal state)
# Perfect for ablation studies comparing sensor-based vs. non-sensor learning approaches
SENSORS_ENABLED = True

# 🕳️ RANDOM MIXED GAP TERRAIN: Small + Medium + Large gaps in single terrain (Type 1)
# 🚀 DESIGN: Using single HfDiscreteObstaclesTerrainCfg with wide range for random gap sizes
# DESIGN FOR RANDOM MIXED GAP TERRAIN:
# - Central flat platform for robot spawning
# - Random gap sizes (0.2m-2.0m long, 20-30cm deep) all mixed together
# - Single terrain generator creates natural random distribution
# - Uniform shallow depth (20-30cm) for consistent training
# - No infinite LiDAR readings - all rays hit gap bottoms within 5m range
# - Complete gap detection training with varied navigation strategies
GAPS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,  # 2m border for boundaries
    num_rows=10,       # 10 rows - uniform density
    num_cols=10,       # 10 cols - uniform density
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # Progressive difficulty for variable gap depths
    sub_terrains={
        # 🕳️ RANDOM MIXED GAPS - All sizes in single terrain (100% proportion)
        "random_mixed_gaps": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,  # 100% - single terrain with all gap sizes
            
            # RANDOM MIXED GAP SETTINGS (ALL SIZES TOGETHER)
            obstacle_height_mode="fixed",       # Fixed depth to ensure only gaps (no obstacles)
            obstacle_width_range=(0.1, 2.0),   # 20cm-2.0m RANDOM gap sizes (small to large)
            obstacle_height_range=(-0.30, -0.07), # NEGATIVE = All gaps 15cm-25cm deep (shallower)
            num_obstacles=15,                   # Total gaps distributed across terrain
            platform_width=1.8,                # 2m central platform for robot spawning
            
            # This creates:
            # - Central flat platform (2m x 2m) for robot spawning
            # - Random gap sizes from 20cm to 2.0m naturally distributed
            # - Shallow 15-25cm depth ensures manageable gaps (no obstacles)
            # - Natural mix: some small step-over gaps, some medium navigation gaps, some large crossing gaps
            # - No predictable patterns - truly random distribution
            # - All LiDAR rays hit gap bottoms within sensor range (no infinite readings)
            # - Robot learns adaptive navigation based on real-time gap size detection
        ),
        
        # 🎯 EXPECTED RESULT: NATURAL RANDOM GAP DISTRIBUTION!
        # - 0% infinite LiDAR readings (all rays hit gap bottoms)
        # - Random gap sizes (20cm-2.0m × 15cm-25cm) naturally distributed
        # - No artificial separation - realistic mixed terrain
        # - Shallow depth (15-25cm) for manageable training difficulty
        # - Complete gap detection: varied sizes with consistent shallow challenge
        # - Perfect for comprehensive height sensor testing with natural scenarios
        # - Robot develops adaptive strategies for unpredictable gap patterns
    },
)

# 🚧 OBSTACLES TERRAIN: Discrete obstacle avoidance terrain (Type 2)
# ENHANCED: Height variations from 0.2cm to 1.7m with increased obstacle density
OBSTACLES_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=8,     # Increased from 5 to 8 for more terrain variety
    num_cols=10,    # Increased from 8 to 10 for more terrain variety     
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # Enable curriculum for progressive difficulty
    sub_terrains={
        # 🚧 RANDOM MIXED OBSTACLES - All heights in single terrain (100% proportion)
        "random_mixed_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,  # 100% - single terrain with all obstacle heights
            
            # RANDOM MIXED OBSTACLE SETTINGS (ALL HEIGHTS TOGETHER)
            obstacle_height_mode="fixed",       # Fixed mode for consistent obstacle generation
            obstacle_width_range=(0.4, 0.8),   # 10cm-1.2m RANDOM obstacle sizes (small to large)
            obstacle_height_range=(0.2, 0.8), # POSITIVE = All obstacles 0.2cm-1.7m high (above ground)
            num_obstacles=25,                   # Total obstacles distributed across terrain
            platform_width=1.8,                # 1m central platform for robot spawning
            
            # This creates:
            # - Central flat platform (1.8m x 1.8m) for robot spawning  
            # - Random obstacle heights from 30cm to 1.2m naturally distributed
            # - Flat base terrain ensures no gaps (robot walks on ground level)
            # - Natural mix: micro obstacles, step-over obstacles, tall barriers, extreme obstacles
            # - No predictable patterns - truly random distribution
            # - Robot encounters obstacles everywhere while walking on flat terrain
            # - Robot learns adaptive navigation based on real-time obstacle height detection
        ),
        
        # 🎯 EXPECTED RESULT: NATURAL RANDOM OBSTACLE DISTRIBUTION!
        # - Flat base terrain for normal walking (no gaps or holes)
        # - Random obstacle heights (30cm-1.2m) naturally distributed across entire environment
        # - No flat safe areas - realistic mixed obstacle field
        # - Full height range (30cm-1.2m) for comprehensive navigation challenge
        # - Complete obstacle navigation: varied heights with consistent ground level
        # - Perfect for comprehensive upper body collision testing
        # - Robot develops adaptive strategies for unpredictable obstacle patterns
    },
)

# 🪜 STAIRS TERRAIN: Dense descending stairs for upward climbing challenge
# 🚀 DESIGN: Robots spawn below stairs and climb upward for optimal sensor comparison
# DESIGN FOR CLIMBING CHALLENGE:
# - 10% flat corridors for approach zones
# - 90% DESCENDING STAIRS: Create upward climbing challenge
# - Robot encounters climbing challenge where sensors provide measurable advantage
# - Environment-aware mode benefits from step preview and foot placement planning
STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=1.0,
    num_rows=5,     
    num_cols=8,     
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # Enable curriculum for progressive difficulty
    sub_terrains={
        "flat_corridors": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.1,  # 10% flat approach areas
            size=(1.5, 1.5),
        ),
        "descending_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(  # DESCENDING for upward climbing
            proportion=0.9,  # 90% stair coverage for intensive climbing practice
            step_height_range=(0.10, 0.10),  # Fixed 10cm steps for consistency
            step_width=0.30,  # Fixed 30cm step width
            platform_width=2.0,  # Wide platforms for stable foot placement
            border_width=0.6,  # Minimal border for maximum stair coverage
        ),
    },
)

# 0..3 supported; types 4..6 removed for focused training

@configclass
class SDSG1FlatWithBoxEnvCfg(SDSG1RoughEnvCfg):
    """SDS Unitree G1 environment with NUMERIC TERRAIN TYPE SELECTION + OPTIONAL environmental sensors.

    TERRAIN TYPE SELECTION SYSTEM:
    - Set TERRAIN_TYPE = 0: Simple terrain (70% flat + 30% gentle bumps) - basic locomotion
    - Set TERRAIN_TYPE = 1: Gaps terrain - mixed flat + gaps (30% flat areas, 35% easy gaps 15-25cm, 35% medium gaps 25-35cm, depth 30-40cm)
    - Set TERRAIN_TYPE = 2: Obstacles terrain - discrete obstacle avoidance
    - Set TERRAIN_TYPE = 3: Stairs terrain - stair climbing and step navigation
    - Uses EXACT same physics, robot, and command settings for all terrain types

    SENSORS CONTROL SYSTEM:
    - Set SENSORS_ENABLED = True: Full environmental sensing (height scanner + lidar + sensor observations)
    - Set SENSORS_ENABLED = False: Completely disable sensors - proprioceptive-only observations
    - Perfect for comparing sensor-based vs. non-sensor learning approaches
    - Ideal for ablation studies and performance comparisons
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

        # 🎯 TERRAIN SELECTION: Numeric switch for different terrain types
        if TERRAIN_TYPE == 0:
            # SIMPLE: Flat terrain with gentle bumps (same as rough_env_cfg)
            self.scene.terrain.terrain_generator = SIMPLE_LEARNING_TERRAIN_CFG
            print("🏞️ TERRAIN TYPE 0: SIMPLE flat terrain with gentle bumps")
        elif TERRAIN_TYPE == 1:
            # GAPS: Jumping parkour with finite depth (progressive gap crossing)
            self.scene.terrain.terrain_generator = GAPS_TERRAIN_CFG
            print("🕳️ TERRAIN TYPE 1: GAPS terrain - mixed flat + gaps (30% flat areas, 35% easy gaps 15-25cm, 35% medium gaps 25-35cm, depth 30-40cm)")
        elif TERRAIN_TYPE == 2:
            # OBSTACLES: Discrete obstacle avoidance
            self.scene.terrain.terrain_generator = OBSTACLES_TERRAIN_CFG
            print("🚧 TERRAIN TYPE 2: OBSTACLES terrain for discrete obstacle avoidance")
        elif TERRAIN_TYPE == 3:
            # STAIRS: Stair climbing and step navigation
            self.scene.terrain.terrain_generator = STAIRS_TERRAIN_CFG
            print("🪜 TERRAIN TYPE 3: STAIRS terrain for stair climbing and step navigation")
            
            # ✅ WAYPOINT NAVIGATION: keep target forward at (3.0, 0.0)
            self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.2)
            self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
            self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)
            print("🎯 WAYPOINT COMMANDS: Forward velocity 0.8-1.2 m/s toward target (0, 3)")
        
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
        
        # === CONDITIONAL ENHANCED SENSING FOR ENVIRONMENTAL ANALYSIS ===
        if SENSORS_ENABLED:
            print("🔧 SENSORS ENABLED: Full environmental sensing capabilities activated")
            
            # 1. HEIGHT SCANNER - For terrain height detection  
            self.scene.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link", 
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.696)),  # ✅ ADJUSTED: 69.6cm above torso to achieve 0.709m sensor height → 0.209m baseline
                attach_yaw_only=True,
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=0.075,  # 🚀 IMPROVED: 7.5cm resolution (was 15cm) - 2x better gap detection!
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
                noise=None,        # ✅ NO NOISE: Clean distance measurements
                clip=(0.1, 5.0),   # ✅ FIXED: Match sensor max_distance=5.0m
                scale=0.204,       # ✅ FIXED: 1.0 / (5.0-0.1) = 0.204 -> maps [0.1,5.0] to [0.02, 1.02] ≈ [0,1]
            )
        else:
            print("🚫 SENSORS DISABLED: Environmental sensing capabilities completely removed")
            print("📝 Robot will rely only on proprioceptive observations (joints, IMU, commands)")
            
            # ✅ CRITICAL FIX: Remove height sensor observations when sensors disabled
            # The base velocity_env_cfg.py includes height_scan by default - we must remove it!
            if hasattr(self.observations.policy, 'height_scan'):
                delattr(self.observations.policy, 'height_scan')
                print("🗑️  Removed height_scan observation from policy")
            
            # ✅ CRITICAL FIX: Remove lidar observations when sensors disabled
            if hasattr(self.observations.policy, 'lidar_range'):
                delattr(self.observations.policy, 'lidar_range')
                print("🗑️  Removed lidar_range observation from policy")
            
            # Note: Contact forces sensor is inherited from base config and used in rewards, so we keep it
            # but disable its visualization when sensors are disabled
            if hasattr(self.scene, 'contact_forces') and self.scene.contact_forces is not None:
                self.scene.contact_forces.debug_vis = False  # Disable visualization
                print("📊 Contact forces sensor kept for rewards but visualization disabled")

        # 📝 NOTE: Everything else (commands, physics, robot, rewards) is IDENTICAL to rough_env_cfg!
        # This config = rough_env_cfg + terrain_selection + conditional_sensors
        # When SENSORS_ENABLED=True: adds height_scanner + lidar + sensor_observations
        # When SENSORS_ENABLED=False: pure proprioceptive observations only


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
        
        # ✅ CONDITIONAL: Sensor visualization settings for play mode
        if SENSORS_ENABLED:
            # ✅ ENSURE: Sensor visualization is enabled for play mode
            self.scene.height_scanner.debug_vis = True
            self.scene.lidar.debug_vis = True
            print("🎯 PLAY MODE: Environmental sensors visualization enabled!")
        else:
            print("🎯 PLAY MODE: No environmental sensors (disabled by SENSORS_ENABLED=False)")
        
        # ✅ CONTACT FORCES: Always available but visualization depends on sensor setting
        if hasattr(self.scene, 'contact_forces') and self.scene.contact_forces is not None:
            if SENSORS_ENABLED:
                self.scene.contact_forces.debug_vis = True
                print("🎯 PLAY MODE: Contact visualization enabled - see real foot contact forces!")
            else:
                self.scene.contact_forces.debug_vis = False
                print("📊 PLAY MODE: Contact forces available for rewards but visualization disabled")
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False 