# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SDS Go1 Flat Environment Configuration.

This configuration is specifically designed for the SDS project
using Unitree Go1 robot on flat terrain.
"""

from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from isaaclab.sim import RenderCfg
import isaaclab.sim as sim_utils

from .rough_env_cfg import SDSUnitreeGo1RoughEnvCfg


@configclass
class SDSUnitreeGo1FlatEnvCfg(SDSUnitreeGo1RoughEnvCfg):
    """SDS Unitree Go1 flat terrain environment configuration."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # CRITICAL: For SDS, use ONLY GPT-generated sds_custom reward
        # No other reward terms are defined in the SDS RewardsCfg
        # self.rewards.sds_custom is already configured in the base class

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class SDSUnitreeGo1FlatEnvCfg_PLAY(SDSUnitreeGo1FlatEnvCfg):
    """SDS Unitree Go1 flat terrain environment configuration for play/testing."""
    
    # Add automatic side robot tracking with path tracing
    # SIDE TRACKING - Sports broadcast style (ACTIVE)
    viewer = ViewerCfg(
        origin_type="asset_root",    # Automatically follow robot base
        asset_name="robot",          # Track the robot asset
        env_index=0,                # Environment 0 (single robot)
        eye=(0.0, -2.8, 0.15),       # Side view tracking - 3.5m to the side, lowered camera height
        lookat=(0.0, 0.0, 0.3),     # Look at robot center, slightly lower
    )
    
    # VERY CLOSE - Action shot
    # viewer = ViewerCfg(
    #     origin_type="asset_root",    # Automatically follow robot base
    #     asset_name="robot",          # Track the robot asset
    #     env_index=0,                # Environment 0 (single robot)
    #     eye=(1.0, 1.0, 0.6),        # Very close camera position
    #     lookat=(0.0, 0.0, 0.3),     # Look at robot legs/body
    # )
    
    # MEDIUM CLOSE - Good overall view (RECOMMENDED)
    # viewer = ViewerCfg(
    #     origin_type="asset_root",
    #     asset_name="robot",
    #     env_index=0,
    #     eye=(2.0, 2.0, 1.2),        # Medium distance
    #     lookat=(0.0, 0.0, 0.5),     # Look at robot center
    # )
    
    # CINEMATIC - Wide shot
    # viewer = ViewerCfg(
    #     origin_type="asset_root",
    #     asset_name="robot",
    #     env_index=0,
    #     eye=(3.5, 3.5, 2.0),        # Wider cinematic view
    #     lookat=(0.0, 0.0, 0.8),     # Look slightly up
    # )
    
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1  # Single robot for close tracking
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
