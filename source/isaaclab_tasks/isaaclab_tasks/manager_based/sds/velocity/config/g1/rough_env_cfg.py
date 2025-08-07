# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SDS G1 Rough Environment Configuration.

This configuration is specifically designed for the SDS project
using Unitree G1 humanoid robot on SIMPLIFIED rough terrain.
FIXED: Uses minimal obstacles for easy locomotion learning.
"""

from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # Standard Isaac Lab rough terrain 
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from isaaclab_tasks.manager_based.sds.velocity.velocity_env_cfg import SDSVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

# 🎯 SIMPLIFIED TERRAIN: Minimal obstacles for easy learning
SIMPLE_LEARNING_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,     # Fewer terrain variations
    num_cols=8,     # Fewer environments per row  
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # Enable curriculum for progressive learning
    sub_terrains={
        # 70% FLAT for easy learning and confidence building
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.7,  # 70% completely flat terrain
        ),
        # 30% GENTLE BUMPS (2-5cm) - Very forgiving rough terrain
        "gentle_bumps": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, 
            noise_range=(0.02, 0.05),  # Only 2-5cm bumps (very gentle)
            noise_step=0.01, 
            border_width=0.25
        ),
    },
)


@configclass
class SDSG1RoughEnvCfg(SDSVelocityRoughEnvCfg):
    """SDS Unitree G1 SIMPLIFIED rough terrain environment for easy locomotion learning.
    
    FIXED for standing still issue:
    - 50% flat terrain for confidence building
    - 30% gentle bumps (2-5cm) for basic adaptation
    - 20% small boxes (5-10cm) for simple obstacle navigation
    - Curriculum enabled for progressive difficulty
    - Forward-only commands (0.3-0.6 m/s) with minimal standing
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # 🎯 CRITICAL: Use SIMPLIFIED terrain for easy learning!
        self.scene.terrain.terrain_generator = SIMPLE_LEARNING_TERRAIN_CFG
        
        # CHANGE: Use G1 robot configuration with self-collisions enabled
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Enable self-collisions so robot learns to avoid leg crossing naturally
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # G1 HUMANOID SPECIFIC ADAPTATIONS:
        
        # Randomization - adjust for bipedal stability
        self.events.push_robot = None  # Remove random pushes that cause instability
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.5)  # Smaller mass variations
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]  # Only vary torso mass

        # Enhanced stability control for humanoid
        # Reduced reset pose randomization for more stable learning
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-1.57, 1.57)},  # Smaller range
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # 🚀 ENHANCED: Updated velocity ranges for forward-only movement
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)     # Forward-only movement (no backward)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # Increased lateral movement range
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)    # Increased turning range for better maneuverability

        # 🚨 CRITICAL FIX: Minimize standing commands to encourage locomotion
        self.commands.base_velocity.rel_standing_envs = 0.02  # Only 2% standing for active learning

        # rewards - SDS uses only the sds_custom reward, no need to configure others
        # self.rewards.sds_custom is already configured in the base class

        # terminations - use torso instead of base
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"

        # Viewer - updated for G1 CORRECTED height (1.27m)
        self.viewer.eye = (0.0, -3.0, 1.8)  # Raised from 0.9m to 1.8m for 1.27m robot
        self.viewer.lookat = (0.0, 0.0, 1.0)  # Raised from 0.5m to 1.0m (G1 torso level)


@configclass
class SDSG1RoughEnvCfg_PLAY(SDSG1RoughEnvCfg):
    """SDS Unitree G1 rough terrain environment configuration for play/testing WITHOUT GAPS."""
    
    # G1 Humanoid tracking - Bipedal optimized camera (lowered further for optimal framing)
    viewer = ViewerCfg(
        origin_type="asset_root",    # Automatically follow humanoid torso
        asset_name="robot",          # Track the humanoid robot asset
        env_index=0,                # Environment 0 (single humanoid)
        eye=(0.0, -3.0, 0.15),       # Side view for humanoid - further lowered camera for optimal body framing
        lookat=(0.0, 0.0, 0.2),     # Look at humanoid waist level (optimal for 0.74m height robot)
    )
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        
        # spawn the robot randomly in the grid
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Commands - UPDATED: Allow backward movement but keep demo movements conservative
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)  # Allow backward movement + forward velocity for demos
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # Conservative lateral movement for clean demos
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)  # Conservative turning for stable demo walking
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)   # Straight forward heading
        
        # 🚨 EVEN MORE CRITICAL for PLAY: No standing still in demos!
        self.commands.base_velocity.rel_standing_envs = 0.0  # 0% standing for active demos
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None