# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SDS G1 Flat Environment Configuration.

This configuration is specifically designed for the SDS project
using Unitree G1 humanoid robot on flat terrain.
"""

from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
import isaaclab_assets

from .rough_env_cfg import SDSG1RoughEnvCfg  # Updated class name


@configclass
class SDSG1FlatEnvCfg(SDSG1RoughEnvCfg):
    """SDS Unitree G1 flat terrain environment configuration."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Viewer - updated for G1 CORRECTED height (1.27m)
        self.viewer.eye = (0.0, -3.0, 0.2)  # Raised from 0.9m to 1.8m for 1.27m robot
        self.viewer.lookat = (0.0, 0.0, 0.2)  # Raised from 0.5m to 1.0m (G1 torso level)

        # CRITICAL: For SDS, use ONLY GPT-generated sds_custom reward
        # self.rewards.sds_custom is already configured in the base class

        # Commands - focused on forward movement with minimal lateral movement
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.1)  # Keep higher forward velocity for flat terrain (jumping/sprinting gaits)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)  # Minimal lateral movement (just a bit of side movement)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.05, 0.05)  # Reduced turning for forward-focused training

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class SDSG1FlatEnvCfg_PLAY(SDSG1FlatEnvCfg):
    """SDS Unitree G1 flat terrain environment configuration for play/testing."""
    
    # G1 Humanoid tracking - Bipedal optimized camera (lowered further for optimal framing)
    viewer = ViewerCfg(
        origin_type="asset_root",    # Automatically follow humanoid torso
        asset_name="robot",          # Track the humanoid robot asset
        env_index=0,                # Environment 0 (single humanoid)
        eye=(0.0, -3.0, 0.15),       # Side view for humanoid - further lowered camera for optimal body framing
        lookat=(0.0, 0.0, 0.2),     # Look at humanoid waist level (optimal for 0.74m height robot)
    )
    
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1  # Single humanoid for close tracking
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
