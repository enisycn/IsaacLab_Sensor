# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 Custom Rewards Environment Configuration.

This configuration is specifically designed for testing custom reward functions
on the Unitree G1 humanoid robot without SDS components.
"""

from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from isaaclab.managers import RewardTermCfg as RewTerm

from isaaclab_tasks.manager_based.sds.velocity.velocity_env_cfg import SDSVelocityRoughEnvCfg
from isaaclab_tasks.manager_based.sds.velocity.mdp import custom_rewards

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip


@configclass
class CustomRewardsCfg:
    """Custom reward terms for G1 robot testing."""

    # Custom rewards from the optimization interface
    jump_and_keep_left_hand_high = RewTerm(
        func=custom_rewards.jump_and_keep_left_hand_high, 
        weight=10.0
    )
    
    spin_with_closed_arms = RewTerm(
        func=custom_rewards.spin_with_closed_arms, 
        weight=5.0
    )
    
    move_backward = RewTerm(
        func=custom_rewards.move_backward, 
        weight=3.0
    )
    
    reach_forward_with_both_hands = RewTerm(
        func=custom_rewards.reach_forward_with_both_hands, 
        weight=8.0
    )
    
    dance_pose = RewTerm(
        func=custom_rewards.dance_pose, 
        weight=6.0
    )
    
    walk_forward_with_arm_swing = RewTerm(
        func=custom_rewards.walk_forward_with_arm_swing, 
        weight=4.0
    )
    
    # Helper rewards for stability
    maintain_balance = RewTerm(
        func=custom_rewards.maintain_balance, 
        weight=1.0
    )
    
    avoid_joint_limits = RewTerm(
        func=custom_rewards.avoid_joint_limits, 
        weight=0.5
    )


@configclass
class G1JumpHandHighEnvCfg(SDSVelocityRoughEnvCfg):
    """G1 environment for 'Jump and Keep Left Hand High' task."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Use G1 robot configuration
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Custom rewards for this specific task
        self.rewards = type(self.rewards)()
        self.rewards.jump_and_keep_left_hand_high = CustomRewardsCfg.jump_and_keep_left_hand_high
        self.rewards.maintain_balance = CustomRewardsCfg.maintain_balance
        self.rewards.avoid_joint_limits = CustomRewardsCfg.avoid_joint_limits

        # Task-specific settings
        self.episode_length_s = 15.0
        self.scene.num_envs = 1024  # Smaller for easier monitoring


@configclass
class G1SpinArmsEnvCfg(SDSVelocityRoughEnvCfg):
    """G1 environment for 'Spin with Closed Arms' task."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Use G1 robot configuration
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Custom rewards for this specific task
        self.rewards = type(self.rewards)()
        self.rewards.spin_with_closed_arms = CustomRewardsCfg.spin_with_closed_arms
        self.rewards.maintain_balance = CustomRewardsCfg.maintain_balance
        self.rewards.avoid_joint_limits = CustomRewardsCfg.avoid_joint_limits

        # Task-specific settings
        self.episode_length_s = 10.0
        self.scene.num_envs = 1024


@configclass
class G1MoveBackwardEnvCfg(SDSVelocityRoughEnvCfg):
    """G1 environment for 'Move Backward' task."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Use G1 robot configuration
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Custom rewards for this specific task
        self.rewards = type(self.rewards)()
        self.rewards.move_backward = CustomRewardsCfg.move_backward
        self.rewards.maintain_balance = CustomRewardsCfg.maintain_balance
        self.rewards.avoid_joint_limits = CustomRewardsCfg.avoid_joint_limits

        # Task-specific settings
        self.episode_length_s = 10.0
        self.scene.num_envs = 1024


@configclass
class G1ReachForwardEnvCfg(SDSVelocityRoughEnvCfg):
    """G1 environment for 'Reach Forward with Both Hands' task."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Use G1 robot configuration
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Custom rewards for this specific task
        self.rewards = type(self.rewards)()
        self.rewards.reach_forward_with_both_hands = CustomRewardsCfg.reach_forward_with_both_hands
        self.rewards.maintain_balance = CustomRewardsCfg.maintain_balance
        self.rewards.avoid_joint_limits = CustomRewardsCfg.avoid_joint_limits

        # Task-specific settings
        self.episode_length_s = 15.0
        self.scene.num_envs = 1024


@configclass
class G1DancePoseEnvCfg(SDSVelocityRoughEnvCfg):
    """G1 environment for 'Dance Pose' task."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Use G1 robot configuration
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Custom rewards for this specific task
        self.rewards = type(self.rewards)()
        self.rewards.dance_pose = CustomRewardsCfg.dance_pose
        self.rewards.maintain_balance = CustomRewardsCfg.maintain_balance
        self.rewards.avoid_joint_limits = CustomRewardsCfg.avoid_joint_limits

        # Task-specific settings
        self.episode_length_s = 20.0
        self.scene.num_envs = 1024


@configclass
class G1WalkArmSwingEnvCfg(SDSVelocityRoughEnvCfg):
    """G1 environment for 'Walk Forward with Arm Swing' task."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Use G1 robot configuration
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Custom rewards for this specific task
        self.rewards = type(self.rewards)()
        self.rewards.walk_forward_with_arm_swing = CustomRewardsCfg.walk_forward_with_arm_swing
        self.rewards.maintain_balance = CustomRewardsCfg.maintain_balance
        self.rewards.avoid_joint_limits = CustomRewardsCfg.avoid_joint_limits

        # Task-specific settings
        self.episode_length_s = 15.0
        self.scene.num_envs = 1024


@configclass
class G1CustomAllRewardsEnvCfg(SDSVelocityRoughEnvCfg):
    """G1 environment with all custom rewards active (for general exploration)."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Use G1 robot configuration
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # All custom rewards with lower weights for balanced exploration
        self.rewards = type(self.rewards)()
        self.rewards.jump_and_keep_left_hand_high = RewTerm(
            func=custom_rewards.jump_and_keep_left_hand_high, weight=2.0
        )
        self.rewards.spin_with_closed_arms = RewTerm(
            func=custom_rewards.spin_with_closed_arms, weight=1.0
        )
        self.rewards.move_backward = RewTerm(
            func=custom_rewards.move_backward, weight=1.0
        )
        self.rewards.reach_forward_with_both_hands = RewTerm(
            func=custom_rewards.reach_forward_with_both_hands, weight=2.0
        )
        self.rewards.dance_pose = RewTerm(
            func=custom_rewards.dance_pose, weight=1.5
        )
        self.rewards.walk_forward_with_arm_swing = RewTerm(
            func=custom_rewards.walk_forward_with_arm_swing, weight=2.0
        )
        self.rewards.maintain_balance = CustomRewardsCfg.maintain_balance
        self.rewards.avoid_joint_limits = CustomRewardsCfg.avoid_joint_limits

        # General settings
        self.episode_length_s = 20.0
        self.scene.num_envs = 2048 