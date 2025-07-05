# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SDS G1 Rough Environment Configuration.

This configuration is specifically designed for the SDS project
using Unitree G1 humanoid robot on rough terrain.
"""

from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg

from isaaclab_tasks.manager_based.sds.velocity.velocity_env_cfg import SDSVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip


@configclass
class SDSG1RoughEnvCfg(SDSVelocityRoughEnvCfg):
    """SDS Unitree G1 rough terrain environment configuration."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # CHANGE: Use G1 robot configuration with self-collisions enabled
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Enable self-collisions so robot learns to avoid leg crossing naturally
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # G1 HUMANOID SPECIFIC ADAPTATIONS:
        
        # Randomization - adjust for bipedal stability
        self.events.push_robot = None  # More conservative for bipedal robot
        self.events.add_base_mass = None  # Avoid mass perturbations for humanoid
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)  # Smaller range for stability
        
        # External forces - use torso link for humanoid
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        
        # Reset parameters - bipedal specific
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-1.57, 1.57)},  # Smaller range
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # Commands - focused on forward movement with minimal lateral movement
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.2)  # Increased forward velocity for better training variety
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)  # Minimal lateral movement (just a bit of side movement)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)  # Reduced turning for forward-focused training

        # rewards - SDS uses only the sds_custom reward, no need to configure others
        # self.rewards.sds_custom is already configured in the base class

        # terminations - use torso instead of base
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"

        # Viewer - updated for G1 CORRECTED height (1.27m)
        self.viewer.eye = (0.0, -3.0, 1.8)  # Raised from 0.9m to 1.8m for 1.27m robot
        self.viewer.lookat = (0.0, 0.0, 1.0)  # Raised from 0.5m to 1.0m (G1 torso level)


@configclass
class SDSG1RoughEnvCfg_PLAY(SDSG1RoughEnvCfg):
    """SDS Unitree G1 rough terrain environment configuration for play/testing."""
    
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

        # Conservative commands for demonstration
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)  # Slower, more stable walking for demos
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # No lateral movement for clean forward walking
        self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)  # Minimal turning for straight walking demo
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)   # Straight forward heading
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
