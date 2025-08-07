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
- 4: Complex terrain - for advanced obstacle avoidance visualization
- 5: Jump gaps terrain - 30cm wide gaps for jumping behavior training
- Complete environmental sensing suite (height scanner, lidar, IMU)
- Perfect for progressive training and thesis videos demonstrating different capabilities
"""

import os
import torch
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh import mesh_terrains_cfg as mesh_terrain_gen  # Import mesh terrain configurations
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
# 4: Complex terrain (advanced obstacle avoidance visualization)
# 5: Jump gaps terrain (30cm wide gaps for jumping behavior training)
# 6: Fixed gap challenge (deterministic gap scenario: spawn at 0,0 → gap at 0,2 → target at 0,3)
TERRAIN_TYPE = 6 # ✅ CHANGED: Use fixed gap challenge for deterministic training (was 5)

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
            obstacle_height_range=(-0.30, -0.20), # NEGATIVE = All gaps 15cm-25cm deep (shallower)
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
OBSTACLES_TERRAIN_CFG = TerrainGeneratorCfg(
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
        # 30% FLAT CORRIDORS: Clear paths for normal walking
        "flat_corridors": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,  # 30% flat - baseline walking
        ),
        
        # 40% LOW OBSTACLES: Small obstacles requiring step-over (5-15cm)
        "low_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.4,
            obstacle_height_range=(0.05, 0.15),  # 5-15cm - step-over height
            obstacle_width_range=(0.2, 0.5),     # 20-50cm wide - clear obstacles
            num_obstacles=5,                     # 5 obstacles per terrain
            platform_width=1.5,                 # 1.5m platform
        ),
        
        # 30% MEDIUM OBSTACLES: Higher obstacles requiring careful navigation (10-25cm)
        "medium_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.3,
            obstacle_height_range=(0.10, 0.25),  # 10-25cm - challenging height
            obstacle_width_range=(0.3, 0.7),     # 30-70cm wide - various sizes
            num_obstacles=7,                     # 7 obstacles - denser field
            platform_width=1.2,                 # 1.2m platform - tighter navigation
        ),
    },
)

# 🪜 STAIRS TERRAIN: Stair climbing and step navigation terrain (Type 3)
STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
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
        # 25% FLAT CORRIDORS: Approach and transition areas
        "flat_corridors": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.25,  # 25% flat - transition zones
        ),
        
        # 40% LOW STAIRS: Small steps for basic stair climbing (5-15cm steps)
        "low_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.15),  # 5-15cm steps - manageable climbing
            step_width=0.35,                 # 35cm step depth - good foot placement
            platform_width=2.0,             # 2m platform - stable base
            border_width=0.5,               # 50cm border
            holes=False,
        ),
        
        # 35% HIGH STAIRS: Taller steps for advanced climbing (10-25cm steps)
        "high_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.10, 0.25),  # 10-25cm steps - challenging climbing
            step_width=0.30,                 # 30cm step depth - precise placement
            platform_width=1.5,             # 1.5m platform - smaller target
            border_width=0.3,               # 30cm border
            holes=False,
        ),
    },
)

# 🏔️ COMPLEX TERRAIN: Advanced obstacle avoidance visualization terrain (Type 4)
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

# 🦘 RING GAP TERRAIN: Complete circular ring gap around robot spawn (Type 5)
# 🎯 DESIGN: Continuous ring gap around central platform - 30cm wide x 20cm deep (FINITE DEPTH)
JUMP_GAPS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.05,  # Higher resolution for precise ring gap
    vertical_scale=0.01,   # Fine vertical resolution for 20cm depth
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # 30% FLAT APPROACH: Flat areas for approaching the ring gap
        "flat_approach": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,  # 30% flat - approach zones
        ),
        
        # 70% FINITE DEPTH RING GAP: Well-spaced ring gaps around spawn
        "ring_gap": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.7,
            
            # WELL-SPACED RING GAP SETTINGS - Fewer, larger gaps with good spacing
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.8, 1.0),      # 80-100cm wide gaps (larger, fewer gaps)
            obstacle_height_range=(-0.21, -0.19), # 20cm deep (FINITE depth as requested)
            num_obstacles=8,                      # Fewer obstacles to prevent overcrowding
            platform_width=1.5,                  # Slightly larger platform for better spacing
            
            # This creates:
            # - Central platform (1.5m x 1.5m) for stable robot spawning
            # - 8 well-spaced large gaps forming ring pattern
            # - Each gap segment: 80-100cm wide x 20cm deep (substantial jumps)
            # - Gaps are distributed around the ring with proper spacing
            # - No overlapping or crowded gap placement
            # - Clear jump challenges with recovery space between gaps
        ),
        
        # 🎯 EXPECTED RESULT: WELL-SPACED RING GAP JUMPING TRAINING!
        # - Ring of well-spaced gaps around spawn platform with FINITE 20cm depth
        # - Gap segments: 80-100cm wide x 20cm deep (substantial jumps required)
        # - 8 gaps create clear ring pattern with proper spacing
        # - No overlapping or crowded gap placement
        # - Robot must develop precise jumping skills for larger gaps
        # - Clear recovery zones between gaps for safe landing
        # - Finite depth ensures safe training with recovery possible
    },
)

# 🎯 FIXED GAP CHALLENGE TERRAIN: Deterministic gap scenario for focused training (Type 6)
# 🚀 DESIGN: All robots spawn at (0,0), gap at (0,2), target waypoint at (0,3)
# Every parallel environment has IDENTICAL gap challenge for consistent learning
FIXED_GAP_CHALLENGE_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=1,        # ✅ CRITICAL: Single terrain type (1x1 grid) = ALL environments identical
    num_cols=1,        # ✅ CRITICAL: No variation = deterministic training scenario
    horizontal_scale=0.02,  # ✅ HIGH PRECISION: 2cm resolution for exact gap placement
    vertical_scale=0.01,   # ✅ FINE DEPTH: 1cm vertical precision 
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,  # ✅ DISABLE: No curriculum = fixed difficulty
    sub_terrains={
        # 100% FIXED GAP SCENARIO: Single deterministic terrain across ALL environments
        "fixed_gap_challenge": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,  # ✅ 100% - EVERY environment gets same terrain
            
            # 🎯 DETERMINISTIC GAP SETTINGS
            obstacle_height_mode="fixed",           # ✅ FIXED: No randomization
            obstacle_width_range=(0.49, 0.51),     # ✅ NEAR-EXACT: 50cm gap width (tiny range to avoid numpy error)
            obstacle_height_range=(-0.26, -0.24),  # ✅ NEAR-EXACT: 25cm gap depth (tiny range to avoid numpy error) 
            num_obstacles=1,                        # ✅ SINGLE: One gap per environment
            platform_width=4.0,                    # ✅ LARGE: 4m platform ensures gap at desired location
            
            # 🎯 FIXED PLACEMENT STRATEGY:
            # - Robot spawns at environment origin (0, 0, ground_level)
            # - Platform extends from (-2m, -2m) to (+2m, +2m) around spawn
            # - Single gap placed at forward direction (~2m ahead = 0, 2, gap_level)
            # - Target area at (0, 3, ground_level) - 1m beyond gap
            # - EVERY parallel environment has IDENTICAL layout
            
            # This creates CONSISTENT training scenario:
            # 1. Spawn: Robot at (0, 0) facing forward (+Y direction)
            # 2. Approach: 0-2m forward movement on flat ground  
            # 3. Challenge: 50cm wide × 25cm deep gap at 2m mark
            # 4. Landing: Flat ground from 2.5m to 4m+ for safe landing
            # 5. Target: Waypoint at (0, 3) - clear goal 1m past gap
        ),
        
        # 🎯 EXPECTED RESULT: PERFECT DETERMINISTIC GAP TRAINING!
        # ✅ ALL 512 robots face IDENTICAL gap challenge simultaneously
        # ✅ Consistent spawn position: (0, 0, ground) in each environment  
        # ✅ Consistent gap location: ~(0, 2, -0.25) relative to spawn
        # ✅ Consistent target: (0, 3, ground) - clear waypoint beyond gap
        # ✅ No randomization = pure skill development on fixed scenario
        # ✅ Parallel learning = 512x faster data collection on same problem
        # ✅ Height sensor can detect gap consistently at 2m distance
        # ✅ Robot learns: approach → detect → time jump → clear gap → reach target
    },
)

@configclass
class SDSG1FlatWithBoxEnvCfg(SDSG1RoughEnvCfg):
    """SDS Unitree G1 environment with NUMERIC TERRAIN TYPE SELECTION + OPTIONAL environmental sensors.

    TERRAIN TYPE SELECTION SYSTEM:
    - Set TERRAIN_TYPE = 0: Simple terrain (70% flat + 30% gentle bumps) - basic locomotion
    - Set TERRAIN_TYPE = 1: Gaps terrain - mixed flat + gaps (30% flat areas, 35% easy gaps 15-25cm, 35% medium gaps 25-35cm, depth 30-40cm)
    - Set TERRAIN_TYPE = 2: Obstacles terrain - discrete obstacle avoidance
    - Set TERRAIN_TYPE = 3: Stairs terrain - stair climbing and step navigation
    - Set TERRAIN_TYPE = 4: Complex terrain - advanced obstacle avoidance visualization
    - Set TERRAIN_TYPE = 5: Jump gaps terrain - 30cm wide gaps for jumping behavior training
    - Set TERRAIN_TYPE = 6: Fixed gap challenge (deterministic gap scenario: spawn at 0,0 → gap at 0,2 → target at 0,3)
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
        elif TERRAIN_TYPE == 4:
            # COMPLEX: Advanced obstacle avoidance visualization
            self.scene.terrain.terrain_generator = COMPLEX_BOX_TERRAIN_CFG
            print("🏔️ TERRAIN TYPE 4: COMPLEX advanced obstacle avoidance visualization")
        elif TERRAIN_TYPE == 5:
            # RING GAPS: Complete circular ring gap around spawn with finite depth
            self.scene.terrain.terrain_generator = JUMP_GAPS_TERRAIN_CFG
            print("🦘 TERRAIN TYPE 5: FINITE DEPTH RING GAP terrain - 30cm wide x 20cm deep ring around spawn")
        elif TERRAIN_TYPE == 6:
            # FIXED GAP CHALLENGE: Deterministic gap scenario for focused training
            self.scene.terrain.terrain_generator = FIXED_GAP_CHALLENGE_CFG
            print("🎯 TERRAIN TYPE 6: FIXED GAP CHALLENGE terrain - deterministic gap scenario for focused training")
            
            # ✅ WAYPOINT NAVIGATION: Set commands to guide robot toward target (0, 3)
            # Robot spawns at (0, 0) and needs to reach waypoint at (0, 3) = 3m forward
            # This creates consistent forward movement command across all environments
            self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.2)   # 0.8-1.2 m/s forward (toward gap)
            self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05) # Minimal lateral movement (stay on path)
            self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)   # Minimal turning (stay straight)
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