# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import define_collision_properties, activate_contact_sensors, CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab_tasks.manager_based.sam2.mdp as sam2_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()

##
# Modified Scene definition with obstacles
##

@configclass
class NavigationSceneCfg(InteractiveSceneCfg):
    """Navigation scene configuration with obstacles."""
    
    # Import terrain configuration from the parent environment
    terrain = LOW_LEVEL_ENV_CFG.scene.terrain
    
    # Robot configuration with collision enabled for obstacle avoidance
    robot = LOW_LEVEL_ENV_CFG.scene.robot
    robot.spawn.activate_contact_sensors = True  # Enable contact sensors for collision detection
    
    # Contact forces sensor for robot feet (inherited from base config)
    contact_forces = LOW_LEVEL_ENV_CFG.scene.contact_forces
    
    # Lights 
    sky_light = LOW_LEVEL_ENV_CFG.scene.sky_light

    # Add red box obstacle to the side to encourage learning navigation around it
    obstacle_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 0.3, 0.5),  # Moved to side: robots can go around left or right
            rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 1.0),  # Smaller: 0.6m x 0.6m x 1.0m - easier to navigate around
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Make it kinematic so it doesn't move
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1),  # Red color
                metallic=0.0,
                roughness=0.5,
            ),
            activate_contact_sensors=True,  # Enable contact reporting for obstacles
        ),
    )

    # Contact sensor for the entire robot body to detect obstacle collisions
    robot_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # All robot links
        update_period=0.0,  # Update every physics step
        history_length=3,
        debug_vis=True,  # Enable debug visualization to see contact points
        # Don't filter any contacts - we'll handle this in the reward function
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-1.0, 1.0), "yaw": (-0.3, 0.3)},  # More diverse starting positions and orientations
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
        
        # Add robot contact forces as observation for obstacle avoidance
        robot_contacts = ObsTerm(
            func=sam2_mdp.robot_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("robot_contact_forces")},
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # === MAIN GOALS ===
    # Enhanced velocity tracking specifically for forward motion  
    track_velocity_enhanced = RewTerm(
        func=sam2_mdp.velocity_command_tracking_exp,
        weight=4.0,  # Reduced to balance with exploration rewards
        params={"std": 0.5, "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Reward forward progress (distance traveled)
    forward_progress = RewTerm(
        func=sam2_mdp.forward_velocity,
        weight=2.0,  # Reduced to balance with exploration
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === OBSTACLE AVOIDANCE ===
    # Strong penalty for base link hitting obstacles (excluding feet and other links)
    robot_obstacle_collision = RewTerm(
        func=sam2_mdp.robot_obstacle_collision_penalty,
        weight=-50.0,  # Strong penalty to encourage steering around obstacles
        params={
            "sensor_cfg": SceneEntityCfg("robot_contact_forces", body_names=["base"]),
            "threshold": 10.0  # Increased from 1.0N to avoid noise, still sensitive to real collisions
        },
    )
    
    # === ADAPTIVE MOTION CONSTRAINTS ===
    # Allow lateral movement - no penalty for steering to avoid obstacles
    # Robot learns obstacle avoidance through collision penalty feedback only
    
    # === REMOVED HEADING CONSTRAINTS ===
    # Removed forward_direction_alignment and heading_error to allow free exploration
    # Removed angular_velocity_z penalty to allow turning for navigation
    
    # === STABILITY & SAFETY ===
    # Penalty for falling/termination
    termination_penalty = RewTerm(func=isaaclab_mdp.is_terminated, weight=-200.0)  # Reduced penalty
    
    # Encourage stable upright posture (but allow some natural body movement)
    flat_orientation = RewTerm(
        func=isaaclab_mdp.flat_orientation_l2, 
        weight=-0.1,  # Further reduced weight to allow natural walking dynamics
    )
    
    # === EFFICIENCY & SMOOTHNESS ===
    # Encourage smooth actions (reduce jittering)
    action_smoothness = RewTerm(
        func=isaaclab_mdp.action_rate_l2,
        weight=-0.001,  # Very light penalty to allow dynamic movements
    )
    
    # === ENERGY EFFICIENCY ===
    # Penalize excessive joint torques for energy efficiency
    joint_torques = RewTerm(
        func=isaaclab_mdp.joint_torques_l2,
        weight=-0.0001,  # Very light penalty
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # Simple velocity command for forward motion learning
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),  # Longer episodes for navigation learning
        heading_command=False,
        rel_standing_envs=0.0,  # No standing still
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.8, 1.2),  # Forward velocity between 0.8-1.2 m/s
            lin_vel_y=(0.0, 0.0),  # No lateral movement commanded (robot will learn to move laterally for avoidance)
            ang_vel_z=(0.0, 0.0),  # No turning commanded (robot will learn to turn for avoidance)
        ),
        debug_vis=True,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 1) Always terminate on timeout
    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)
    
    # 2) Terminate if the base link hits the ground (robot has fallen over)
    base_contact = DoneTerm(
        func=isaaclab_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]),
            "threshold": 1.0,
        },
    )
    
    # 3) Terminate when base link collides with obstacles (robust threshold above sensor noise)
    robot_obstacle_collision_termination = DoneTerm(
        func=isaaclab_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("robot_contact_forces", body_names=["base"]),
            "threshold": 10.0,  # Increased from 1.0N to avoid false triggers from sensor noise
        },
    )


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment with obstacle avoidance."""

    # environment settings - use our custom scene with obstacles
    scene: NavigationSceneCfg = NavigationSceneCfg(num_envs=4096, env_spacing=4.0)
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
        self.episode_length_s = 15.0  # Longer episodes for navigation learning and exploration

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.robot_contact_forces is not None:
            self.scene.robot_contact_forces.update_period = self.sim.dt


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
