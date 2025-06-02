# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab_tasks.manager_based.sam2.mdp as sam2_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: sam2_mdp.PreTrainedPolicyActionCfg = sam2_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=isaaclab_mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=isaaclab_mdp.projected_gravity)
        velocity_command = ObsTerm(func=isaaclab_mdp.generated_commands, params={"command_name": "base_velocity"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # === MAIN GOALS ===
    # Enhanced velocity tracking specifically for forward motion  
    track_velocity_enhanced = RewTerm(
        func=sam2_mdp.velocity_command_tracking_exp,
        weight=4.0,  # Higher weight for main goal
        params={"std": 0.5, "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Reward forward progress (distance traveled)
    forward_progress = RewTerm(
        func=sam2_mdp.forward_velocity,
        weight=1.5,  # Increased weight for forward progress
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === HEADING CONTROL FOR STRAIGHT FORWARD MOTION ===
    # Reward for robot's forward vector aligning with world X-axis
    forward_direction_alignment = RewTerm(
        func=sam2_mdp.forward_direction_reward,
        weight=2.0,  # Strong reward for facing forward
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Penalize heading deviation from forward (0 radians)
    heading_error = RewTerm(
        func=sam2_mdp.forward_heading_error,
        weight=-2.0,  # Strong penalty for heading deviation
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === STRAIGHT FORWARD MOTION CONSTRAINTS ===
    # Strong penalty for ANY lateral movement
    lateral_velocity = RewTerm(
        func=sam2_mdp.lateral_velocity_penalty,
        weight=-8.0,  # Even stronger penalty for sideways movement
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Reward for moving purely in X direction (straight forward)
    straight_motion = RewTerm(
        func=sam2_mdp.straight_motion_reward,
        weight=3.0,  # Increased reward for zero lateral velocity
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Strong penalty for any turning/rotation
    angular_velocity_z = RewTerm(
        func=sam2_mdp.angular_velocity_z_penalty,
        weight=-4.0,  # Stronger penalty for turning
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === STABILITY & SAFETY ===
    # Penalty for falling/termination
    termination_penalty = RewTerm(func=isaaclab_mdp.is_terminated, weight=-500.0)
    
    # Encourage stable upright posture (but allow some natural body movement)
    flat_orientation = RewTerm(
        func=isaaclab_mdp.flat_orientation_l2, 
        weight=-0.3,  # Reduced weight to allow natural walking dynamics
    )
    
    # === EFFICIENCY & SMOOTHNESS ===
    # Encourage smooth actions (reduce jittering)
    action_smoothness = RewTerm(
        func=isaaclab_mdp.action_rate_l2,
        weight=-0.005,  # Reduced to allow necessary control adjustments
    )
    
    # === ENERGY EFFICIENCY ===
    # Penalize excessive joint torques for energy efficiency
    joint_torques = RewTerm(
        func=isaaclab_mdp.joint_torques_l2,
        weight=-0.0005,  # Very light penalty
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # Simple velocity command for forward motion learning
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),  # 5 second episodes
        heading_command=False,
        rel_standing_envs=0.0,  # No standing still
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.8, 1.2),  # Forward velocity between 0.8-1.2 m/s
            lin_vel_y=(0.0, 0.0),  # No lateral movement
            ang_vel_z=(0.0, 0.0),  # No turning
        ),
        debug_vis=True,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=isaaclab_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the simple forward motion navigation environment."""

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = 5.0  # 5 second episodes for forward motion learning

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
