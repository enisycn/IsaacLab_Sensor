# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect terrain-aware policy performance data.

This script loads trained RL policies and collects detailed performance metrics
for comparative analysis between environment-aware and foundation-only modes
across different terrain types.

Terrain Types:
- 0: Simple/Flat terrain with gentle bumps
- 1: Gap navigation terrain (random gaps 20cm-2.0m)  
- 2: Obstacle avoidance terrain
- 3: Stair climbing terrain

.. code-block:: bash

    # Terrain Type 0: Simple terrain (current implementation)
    ./isaaclab.sh -p SDS_ANONYM/collect_policy_data_simple.py \
        --terrain_type 0 \
        --task Isaac-SDS-Velocity-Flat-G1-Enhanced-v0 \
        --checkpoint logs/rsl_rl/g1_enhanced/2025-08-14_00-40-58/model_700.pt \
        --num_envs 50 --steps 1000 --output simple_terrain_data.pkl

    # Terrain Type 1: Gap navigation (placeholders ready for implementation)
    ./isaaclab.sh -p SDS_ANONYM/collect_policy_data_simple.py \
        --terrain_type 1 \
        --task Isaac-SDS-Velocity-Flat-G1-Enhanced-v0 \
        --checkpoint logs/rsl_rl/g1_enhanced/2025-08-14_00-40-58/model_700.pt \
        --num_envs 50 --steps 1000 --output gap_terrain_data.pkl

Note: Before running, manually set TERRAIN_TYPE in flat_with_box_env_cfg.py to match --terrain_type argument.

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# import cli_args from RSL-RL scripts  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reinforcement_learning', 'rsl_rl'))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Comprehensive Policy Data Collection")
parser.add_argument("--task", type=str, default="Isaac-SDS-Velocity-Flat-G1-Enhanced-v0", help="Isaac Lab task")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to collect")
parser.add_argument("--output", type=str, required=True, help="Output file path (.pkl)")
parser.add_argument("--terrain_type", type=int, default=0, choices=[0, 1, 2, 3], help="Terrain type: 0=Simple, 1=Gaps, 2=Obstacles, 3=Stairs")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric and use USD I/O operations")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import pickle
import time
import torch
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
import isaaclab.utils.math as math_utils

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


class TerrainAwareMetricsCollector:
    """Terrain-aware metrics collector with specific metrics for each terrain type."""
    
    def __init__(self, env, num_envs, terrain_type=0):
        self.env = env
        self.num_envs = num_envs
        self.terrain_type = terrain_type
        self.nominal_height = 0.74  # G1 nominal height in meters
        
        # Get actual timestep from environment
        self.dt = self._get_environment_timestep()
        print(f"📊 Environment timestep: {self.dt:.4f}s ({1/self.dt:.1f} Hz)")
        
        # Initialize robot reference
        self._initialize_robot_references()
        
        # Initialize terrain-specific metrics
        self._initialize_terrain_metrics()
        
        # Common disturbance tracking for all terrains
        self.step_counter = 0
        self.disturbance_applied = False
        self.disturbance_step = 150  # Apply disturbance at step 150
        self.post_disturbance_steps = []
        self.disturbance_removed = False
        
        # Enhanced disturbance tracking
        self.pre_disturbance_baseline = None
        self.during_disturbance_data = []
        self.post_disturbance_data = []
        self.recovery_start_step = None
        
        # Command-aware trajectory tracking for lateral displacement
        self.trajectory_tracking = {
            'positions_before_disturbance': [],
            'commands_before_disturbance': [],
            'expected_trajectory': None,
            'disturbance_start_pos': None,
            'disturbance_start_commands': None,
        }
        
        # Contact termination tracking
        self.contact_terminations = 0
        self.total_episodes_started = 0
        
        # Disturbance component tracking
        self.disturbance_components = {
            'height_component': [],
            'lateral_component': [],
            'angular_component': [],
            'recovery_component': []
        }
        
        # Disturbance analysis results storage
        self.disturbance_analysis = {}
        
        print(f"✅ Terrain-aware metrics collector initialized for {num_envs} environments")
        print(f"🎯 Terrain Type {terrain_type}: {self._get_terrain_description()}")
        print(f"📊 Metrics: {list(self.metrics.keys())}")
    
    def _get_terrain_description(self):
        """Get description of current terrain type."""
        descriptions = {
            0: "Simple (flat with gentle bumps)",
            1: "Gaps (random gap navigation 20cm-2.0m)",
            2: "Obstacles (discrete obstacle avoidance)",
            3: "Stairs (stair climbing and steps)"
        }
        return descriptions.get(self.terrain_type, "Unknown")
    
    def _initialize_terrain_metrics(self):
        """Initialize metrics based on terrain type."""
        # UNIVERSAL METRICS SET: All terrain types now use the same comprehensive 6 metrics
        universal_metrics = {
            # ORIGINAL 4 METRICS (smaller is better)
            'height_deviation': [],           # Smaller is better - stability
            'velocity_tracking_error': [],    # Smaller is better - performance  
            'disturbance_resistance': [],     # Smaller is better - robustness (composite)
            'contact_termination_rate': [],   # Smaller is better - fall prevention
            
            # LOCOMOTION QUALITY METRICS (higher is better)
            'balance_stability_score': [],    # Higher is better - body stability
            'gait_smoothness_score': [],      # Higher is better - smooth joint movements
        }
        
        if self.terrain_type == 0:
            # TERRAIN TYPE 0: Simple/Flat terrain
            self.metrics = universal_metrics.copy()
            # ADD TERRAIN-SPECIFIC METRIC: Obstacle collision detection (should be 0 on flat terrain)
            self.metrics['obstacle_collision_count'] = []  # Smaller is better - fewer collisions
            print(f"🏞️ TERRAIN 0 METRICS: Simple terrain with 7 comprehensive metrics + obstacle collision count")
            
            # Initialize obstacle collision tracking
            self.total_obstacle_collisions = 0
            self.collision_threshold = 300.0  # Minimum contact force to count as collision (N)
            
        elif self.terrain_type == 1:
            # TERRAIN TYPE 1: Gap navigation terrain
            self.metrics = universal_metrics.copy()
            # ADD TERRAIN-SPECIFIC METRIC: Obstacle collision detection
            self.metrics['obstacle_collision_count'] = []  # Smaller is better - fewer collisions
            print(f"🕳️ TERRAIN 1 METRICS: Gap navigation with 7 comprehensive metrics + obstacle collision count")
            
            # Initialize obstacle collision tracking
            self.total_obstacle_collisions = 0
            self.collision_threshold = 300.0  # Minimum contact force to count as collision (N)
            
        elif self.terrain_type == 2:
            # TERRAIN TYPE 2: Obstacle avoidance terrain
            self.metrics = universal_metrics.copy()
            # ADD TERRAIN-SPECIFIC METRIC: Obstacle collision detection
            self.metrics['obstacle_collision_count'] = []  # Smaller is better - fewer collisions
            print(f"🚧 TERRAIN 2 METRICS: Obstacle avoidance with 7 comprehensive metrics + obstacle collision count")
            
            # Initialize obstacle collision tracking
            self.total_obstacle_collisions = 0
            self.collision_threshold = 300.0  # Minimum contact force to count as collision (N)
            
        elif self.terrain_type == 3:
            # TERRAIN TYPE 3: Stair climbing terrain
            self.metrics = universal_metrics.copy()
            # ADD TERRAIN-SPECIFIC METRICS: Obstacle collision detection + Stair climbing performance
            self.metrics['obstacle_collision_count'] = []  # Smaller is better - fewer collisions
            self.metrics['stair_climbing_performance'] = []  # Higher is better - ascending progress
            print(f"🪜 TERRAIN 3 METRICS: Stair climbing with 8 comprehensive metrics + obstacle collision count + stair performance")
            
            # Initialize obstacle collision tracking
            self.total_obstacle_collisions = 0
            self.collision_threshold = 300.0  # Minimum contact force to count as collision (N)
            
            # Initialize stair climbing tracking
            self.stair_baseline_height = 0.209  # G1 robot baseline height (m)
            self.previous_height = None  # Track height progress
            self.cumulative_ascent = 0.0  # Track total upward progress
            
        else:
            # Fallback for unknown terrain types
            self.metrics = universal_metrics.copy()
            # ADD TERRAIN-SPECIFIC METRIC: Obstacle collision detection
            self.metrics['obstacle_collision_count'] = []  # Smaller is better - fewer collisions
            print(f"❓ UNKNOWN TERRAIN: Using 7 comprehensive metrics + obstacle collision count")
            
            # Initialize obstacle collision tracking
            self.total_obstacle_collisions = 0
            self.collision_threshold = 300.0  # Minimum contact force to count as collision (N)
    
    def _get_environment_timestep(self):
        """Get timestep from environment with proper error handling."""
        try:
            if hasattr(self.env.unwrapped, 'step_dt'):
                return self.env.unwrapped.step_dt
            elif hasattr(self.env.unwrapped, 'cfg'):
                sim_dt = self.env.unwrapped.cfg.sim.dt
                decimation = self.env.unwrapped.cfg.decimation
                return sim_dt * decimation
            else:
                print("⚠️ Cannot determine timestep, using 50Hz fallback")
                return 0.02
        except Exception as e:
            print(f"⚠️ Error getting timestep: {e}, using 50Hz fallback")
            return 0.02
    
    def _initialize_robot_references(self):
        """Initialize robot reference with proper Isaac Lab patterns."""
        try:
            if hasattr(self.env.unwrapped, 'scene'):
                scene = self.env.unwrapped.scene
                
                if hasattr(scene, 'robot'):
                    self.robot = scene.robot
                    print(f"✅ Robot reference obtained: {self.robot.__class__.__name__}")
                elif hasattr(scene, 'articulations') and 'robot' in scene.articulations:
                    self.robot = scene.articulations['robot']
                    print(f"✅ Robot reference obtained from articulations: {self.robot.__class__.__name__}")
                else:
                    print(f"⚠️ Robot not found in scene")
                    self.robot = None
                
            else:
                raise RuntimeError("Scene not found in environment")
                
        except Exception as e:
            print(f"❌ Error initializing robot references: {e}")
            self.robot = None
    
    def apply_external_disturbance(self):
        """Apply a single external push force for disturbance testing using proper Isaac Lab APIs."""
        if self.robot is None:
            return
            
        try:
            import torch
            
            # Apply a lateral push force (20N to the right for 5 simulation steps = 0.1 seconds)
            push_force = 20.0  # 20 Newtons lateral force
            self.disturbance_duration_steps = 5  # 5 steps = 0.1 seconds at 50Hz
            
            # Create force tensor: [num_envs, num_bodies, 3]
            # Apply force to root body (body_id=0) in world frame
            forces = torch.zeros(self.num_envs, 1, 3, device=self.env.unwrapped.device)
            forces[:, 0, 1] = push_force  # 10N in Y direction (lateral/sideways)
            
            torques = torch.zeros_like(forces)  # No torques, just linear force
            
            # Use Isaac Lab's proper force application API
            self.robot.set_external_force_and_torque(
                forces=forces, 
                torques=torques, 
                body_ids=[0],  # Apply to root body (torso)
                env_ids=None   # Apply to all environments
            )
            
            # CRITICAL: Write forces to simulation (this actually applies them!)
            self.robot.write_data_to_sim()
            
            print(f"🔄 Applied {push_force}N lateral force to root body at step {self.step_counter}")
            self.disturbance_applied = True
            self.disturbance_start_step = self.step_counter
            
        except Exception as e:
            print(f"⚠️ Error applying external force: {e}")
    
    def remove_external_disturbance(self):
        """Remove external forces after disturbance period using proper Isaac Lab APIs."""
        if self.robot is None:
            return
            
        try:
            import torch
            
            # Create zero forces to disable external wrench
            zero_forces = torch.zeros(self.num_envs, 1, 3, device=self.env.unwrapped.device)
            zero_torques = torch.zeros_like(zero_forces)
                    
            # Use Isaac Lab API to remove forces
            self.robot.set_external_force_and_torque(
                forces=zero_forces,
                torques=zero_torques,
                body_ids=[0],
                env_ids=None
            )
            
            # Write to simulation to actually remove forces
            self.robot.write_data_to_sim()
            
            print(f"🛑 Removed external forces at step {self.step_counter}")
            
        except Exception as e:
            print(f"⚠️ Error removing external force: {e}")
    
    def update_metrics(self, obs, actions, rewards, terminated, truncated):
        """Update terrain-specific metrics with current step data."""
        self.step_counter += 1
        
        # Apply external disturbance at specified step (common for all terrains)
        if self.step_counter == self.disturbance_step and not self.disturbance_applied:
            self.apply_external_disturbance()
        
        # Remove external disturbance after duration period
        if (self.disturbance_applied and 
            not self.disturbance_removed and
            hasattr(self, 'disturbance_start_step') and 
            self.step_counter >= self.disturbance_start_step + self.disturbance_duration_steps):
            self.remove_external_disturbance()
            self.disturbance_removed = True
        
        # Get robot state data
        robot_state = self._get_robot_state()
        if robot_state is None:
            return
            
        # Get command data
        commands = self._get_commands()
        
        # Switch to terrain-specific metric updates
        if self.terrain_type == 0:
            self._update_terrain_0_metrics(robot_state, commands, obs, terminated, truncated)
        elif self.terrain_type == 1:
            self._update_terrain_1_metrics(robot_state, commands, obs, terminated, truncated)
        elif self.terrain_type == 2:
            self._update_terrain_2_metrics(robot_state, commands, obs, terminated, truncated)
        elif self.terrain_type == 3:
            self._update_terrain_3_metrics(robot_state, commands, obs, terminated, truncated)
    
    def _update_terrain_0_metrics(self, robot_state, commands, obs, terminated, truncated):
        """Update metrics for Terrain Type 0: Simple/Flat terrain."""
        # ALL 6 COMPREHENSIVE METRICS
        self._update_height_deviation(robot_state)
        self._update_velocity_tracking_error(robot_state, commands)
        self._update_disturbance_resistance(robot_state)
        self._update_contact_termination_rate(terminated, truncated)
        
        # LOCOMOTION QUALITY METRICS
        self._update_balance_stability_score(robot_state)
        self._update_gait_smoothness_score(robot_state)
        
        # TERRAIN-SPECIFIC METRIC: Obstacle collision detection (should be 0 on flat terrain)
        self._update_obstacle_collision_count()
    
    def _update_terrain_1_metrics(self, robot_state, commands, obs, terminated, truncated):
        """Update metrics for Terrain Type 1: Gap navigation terrain."""
        # ALL 6 COMPREHENSIVE METRICS (same as all other terrains)
        self._update_height_deviation(robot_state)
        self._update_velocity_tracking_error(robot_state, commands)
        self._update_disturbance_resistance(robot_state)
        self._update_contact_termination_rate(terminated, truncated)
        
        # LOCOMOTION QUALITY METRICS
        self._update_balance_stability_score(robot_state)
        self._update_gait_smoothness_score(robot_state)
        
        # TERRAIN-SPECIFIC METRIC: Obstacle collision detection
        self._update_obstacle_collision_count()
    
    def _update_terrain_2_metrics(self, robot_state, commands, obs, terminated, truncated):
        """Update metrics for Terrain Type 2: Obstacle avoidance terrain."""
        # ALL 6 COMPREHENSIVE METRICS
        self._update_height_deviation(robot_state)
        self._update_velocity_tracking_error(robot_state, commands)
        self._update_disturbance_resistance(robot_state)
        self._update_contact_termination_rate(terminated, truncated)
        
        # LOCOMOTION QUALITY METRICS
        self._update_balance_stability_score(robot_state)
        self._update_gait_smoothness_score(robot_state)
        
        # TERRAIN-SPECIFIC METRIC: Obstacle collision detection
        self._update_obstacle_collision_count()
    
    def _update_terrain_3_metrics(self, robot_state, commands, obs, terminated, truncated):
        """Update metrics for Terrain Type 3: Stair climbing terrain."""
        # ALL 6 COMPREHENSIVE METRICS
        self._update_height_deviation(robot_state)
        self._update_velocity_tracking_error(robot_state, commands)
        self._update_disturbance_resistance(robot_state)
        self._update_contact_termination_rate(terminated, truncated)
        
        # LOCOMOTION QUALITY METRICS
        self._update_balance_stability_score(robot_state)
        self._update_gait_smoothness_score(robot_state)
        
        # TERRAIN-SPECIFIC METRICS: Obstacle collision detection + Stair climbing performance
        self._update_obstacle_collision_count()
        self._update_stair_climbing_performance(robot_state)
    
    def _get_robot_state(self):
        """Get robot state data using proper Isaac Lab access patterns."""
        if self.robot is None:
            return None
            
        try:
            robot_state = {
                'base_pos': (self.robot.data.root_pos_w - self.env.unwrapped.scene.env_origins).cpu().numpy(),
                'base_quat': self.robot.data.root_quat_w.cpu().numpy(),
                'base_lin_vel': self.robot.data.root_lin_vel_w.cpu().numpy(),
                'base_ang_vel': self.robot.data.root_ang_vel_w.cpu().numpy(),
            }
            return robot_state
        except Exception as e:
            print(f"⚠️ Error getting robot state: {e}")
            return None
    
    def _get_commands(self):
        """Get command data from command manager."""
        try:
            if hasattr(self.env.unwrapped, 'command_manager'):
                return self.env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy()
            return None
        except Exception as e:
            return None
    
    def _update_contact_termination_rate(self, terminated, truncated):
        """Update contact termination rate metric (smaller is better) - measures falls."""
        # Count new episodes starting
        if hasattr(self, '_prev_terminated'):
            # Detect episode resets (when terminated goes from True to False)
            episode_resets = (self._prev_terminated & ~terminated).sum().item()
            self.total_episodes_started += episode_resets
        
        # Count contact terminations (falls/torso contacts)
        contact_terms = terminated.sum().item()
        if contact_terms > 0:
            self.contact_terminations += contact_terms
            print(f"🚨 Contact termination detected: {contact_terms} robots fell/had torso contact")
        
        # Store current rate for this step
        if self.total_episodes_started > 0:
            termination_rate = self.contact_terminations / self.total_episodes_started
        else:
            termination_rate = 0.0
            
        # Store the rate for each environment (broadcast the same rate)
        self.metrics['contact_termination_rate'].extend([termination_rate] * self.num_envs)
        
        # Store for next step comparison
        self._prev_terminated = terminated.clone() if hasattr(terminated, 'clone') else terminated.copy()
    
    def _update_height_deviation(self, robot_state):
        """Update jump trajectory consistency metric (smaller is better) - measures jumping movement quality and consistency."""
        
        # Get current robot height (RELATIVE position for all terrains - better for jumping adaptation)
        # Use relative position for consistent jumping behavior across different terrain heights
        base_pos = robot_state['base_pos']  # Already relative to env_origins
        current_height = base_pos[:, 2]  # Z-component of relative position
        base_lin_vel = robot_state['base_lin_vel']
        
        # Initialize tracking variables if needed
        if not hasattr(self, 'previous_jump_height'):
            self.previous_jump_height = current_height.copy()
            self.height_change_history = []
            self.jump_consistency_tracker = np.zeros(self.num_envs)
        
        # Calculate height change and vertical velocity
        height_change = current_height - self.previous_jump_height
        vertical_velocity = base_lin_vel[:, 2]  # Z-component of velocity
        
        # Track height change patterns for consistency analysis
        self.height_change_history.append(height_change.copy())
        if len(self.height_change_history) > 10:  # Keep last 10 frames for analysis
            self.height_change_history.pop(0)
        
        # Component 1: Vertical movement smoothness (avoid erratic height changes)
        height_change_magnitude = np.abs(height_change)
        smoothness_penalty = np.where(
            height_change_magnitude > 0.05,  # Penalize sudden large changes > 5cm
            height_change_magnitude - 0.05,  # Penalty increases with magnitude
            0.0  # No penalty for smooth movements
        )
        
        # Component 2: Velocity-height consistency (vertical velocity should align with height changes)
        velocity_height_mismatch = np.abs(
            np.sign(height_change) - np.sign(vertical_velocity)
        ) * np.minimum(height_change_magnitude, 0.1)  # Penalize direction mismatches
        
        # Component 3: Movement consistency over time (reward consistent jumping patterns)
        if len(self.height_change_history) >= 5:
            # Calculate consistency of height changes over recent history
            recent_changes = np.array(self.height_change_history[-5:])  # Last 5 frames
            change_std = np.std(recent_changes, axis=0)  # Standard deviation per environment
            consistency_penalty = np.minimum(change_std, 0.05)  # Cap penalty at 5cm std
        else:
            consistency_penalty = np.zeros(self.num_envs)  # No penalty initially
        
        # Component 4: Excessive height variation penalty (prevent wild jumping)
        excessive_height_penalty = np.where(
            height_change_magnitude > 0.15,  # Penalize very large changes > 15cm per step
            (height_change_magnitude - 0.15) * 2.0,  # Strong penalty for excessive changes
            0.0
        )
        
        # Combined trajectory consistency metric (smaller is better)
        # Focus on smooth, consistent jumping movements rather than specific heights
        trajectory_inconsistency = (
            smoothness_penalty * 0.3 +           # 30% - smooth height changes
            velocity_height_mismatch * 0.2 +     # 20% - velocity-height alignment  
            consistency_penalty * 0.3 +          # 30% - temporal consistency
            excessive_height_penalty * 0.2       # 20% - prevent extreme movements
        )
        
        # Store per-environment deviations
        self.metrics['height_deviation'].extend(trajectory_inconsistency.tolist())
        
        # Update tracking
        self.previous_jump_height = current_height.copy()
        
        # Report significant inconsistencies for debugging
        avg_inconsistency = np.mean(trajectory_inconsistency)
        if avg_inconsistency > 0.02:  # Report if average inconsistency > 2cm
            print(f"🔄 Jump trajectory inconsistency: {avg_inconsistency:.3f} "
                  f"(smoothness: {np.mean(smoothness_penalty):.3f}, "
                  f"consistency: {np.mean(consistency_penalty):.3f})")
    
    def _update_velocity_tracking_error(self, robot_state, commands):
        """Update velocity tracking error metric (smaller is better) using REAL robot velocity and command data only."""
        if commands is None or len(commands) == 0:
            raise RuntimeError("❌ Command data required for velocity tracking error calculation - no fallback data allowed")
            
        base_lin_vel = robot_state['base_lin_vel']
        base_quat = robot_state['base_quat']
        
        # Convert to torch for Isaac Lab math operations
        import torch
        import isaaclab.utils.math as math_utils
        
        base_lin_vel_torch = torch.tensor(base_lin_vel, device=self.env.unwrapped.device)
        base_quat_torch = torch.tensor(base_quat, device=self.env.unwrapped.device)
        
        # Get velocity in yaw-aligned frame
        yaw_quat = math_utils.yaw_quat(base_quat_torch)
        base_lin_vel_yaw = math_utils.quat_apply_inverse(yaw_quat, base_lin_vel_torch[:, :3])
        
        # Calculate tracking error
        if commands.shape[-1] >= 2:
            cmd_vel_x = commands[:, 0] if commands.ndim > 1 else commands[0]
            cmd_vel_y = commands[:, 1] if commands.ndim > 1 else commands[1]
            
            # L2 velocity tracking error
            vel_error = np.sqrt((base_lin_vel_yaw[:, 0].cpu().numpy() - cmd_vel_x)**2 + 
                                 (base_lin_vel_yaw[:, 1].cpu().numpy() - cmd_vel_y)**2)
            
            self.metrics['velocity_tracking_error'].extend(vel_error.tolist())
    
    def _update_disturbance_resistance(self, robot_state):
        """Update composite disturbance resistance metric with trajectory-aware lateral displacement."""
        base_pos = robot_state['base_pos']
        base_quat = robot_state['base_quat'] 
        base_ang_vel = robot_state['base_ang_vel']
        
        # Get current commands for trajectory prediction
        commands = self._get_commands()
        
        # 1. PRE-DISTURBANCE: Collect baseline and command patterns (steps 140-149)
        if (self.step_counter >= self.disturbance_step - 10 and 
            self.step_counter < self.disturbance_step and 
            not self.disturbance_applied):
            
            if self.pre_disturbance_baseline is None:
                self.pre_disturbance_baseline = {
                    'height': [],
                    'lateral_pos': [],
                    'angular_stability': []
                }
            
            # Store baseline measurements
            height_dev = np.abs(base_pos[:, 2] - self.nominal_height)
            lateral_pos = base_pos[:, 1]  # Store actual lateral position (not absolute)
            
            # Angular stability (roll/pitch magnitude)
            import torch
            import isaaclab.utils.math as math_utils
            base_quat_torch = torch.tensor(base_quat, device=self.env.unwrapped.device)
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(base_quat_torch)
            angular_stability = torch.sqrt(roll**2 + pitch**2).cpu().numpy()
            
            # Store trajectory data for command-aware analysis
            self.trajectory_tracking['positions_before_disturbance'].append(base_pos.copy())
            if commands is not None and len(commands) > 0:
                self.trajectory_tracking['commands_before_disturbance'].append(commands.copy())
            
            self.pre_disturbance_baseline['height'].extend(height_dev.tolist())
            self.pre_disturbance_baseline['lateral_pos'].extend(lateral_pos.tolist())
            self.pre_disturbance_baseline['angular_stability'].extend(angular_stability.tolist())
        
        # 2. DURING DISTURBANCE: Collect components with trajectory deviation (steps 150-154)
        elif (self.disturbance_applied and 
              hasattr(self, 'disturbance_start_step') and 
              self.step_counter >= self.disturbance_start_step and 
              self.step_counter < self.disturbance_start_step + self.disturbance_duration_steps):
            
            # Store reference point when disturbance starts
            if self.trajectory_tracking['disturbance_start_pos'] is None:
                self.trajectory_tracking['disturbance_start_pos'] = base_pos.copy()
                if commands is not None:
                    self.trajectory_tracking['disturbance_start_commands'] = commands.copy()
            
            # Component 1: Height stability (normalized to baseline)
            height_deviation = np.abs(base_pos[:, 2] - self.nominal_height)
            baseline_height = np.mean(self.pre_disturbance_baseline['height']) if self.pre_disturbance_baseline else 0.02
            height_component = height_deviation / max(baseline_height, 0.01)
            
            # Component 2: TRAJECTORY-AWARE LATERAL DISPLACEMENT
            # Calculate expected lateral position based on commanded velocity
            steps_since_disturbance = self.step_counter - self.disturbance_start_step
            dt = self.dt  # Environment timestep
            
            # Expected lateral displacement from commands (if robot followed commands perfectly)
            expected_lateral_displacement = 0.0
            if (commands is not None and len(commands) > 0 and 
                self.trajectory_tracking['disturbance_start_commands'] is not None):
                
                # Calculate expected movement from commanded lateral velocity
                cmd_vel_y = commands[:, 1] if commands.shape[-1] >= 2 else 0.0
                start_cmd_vel_y = self.trajectory_tracking['disturbance_start_commands'][:, 1] if self.trajectory_tracking['disturbance_start_commands'].shape[-1] >= 2 else 0.0
                
                # Expected lateral movement = time * average_commanded_velocity
                avg_cmd_vel_y = (cmd_vel_y + start_cmd_vel_y) / 2.0
                expected_lateral_displacement = avg_cmd_vel_y * steps_since_disturbance * dt
            
            # Actual lateral displacement from disturbance start
            actual_lateral_displacement = base_pos[:, 1] - self.trajectory_tracking['disturbance_start_pos'][:, 1]
            
            # TRAJECTORY DEVIATION = how much robot deviated from expected path
            trajectory_deviation = np.abs(actual_lateral_displacement - expected_lateral_displacement)
            
            # Normalize by typical lateral movement range
            baseline_lateral_variation = np.std(self.pre_disturbance_baseline['lateral_pos']) if len(self.pre_disturbance_baseline['lateral_pos']) > 1 else 0.2
            lateral_component = trajectory_deviation / max(baseline_lateral_variation, 0.1)
            
            # Component 3: Angular instability (normalized)
            angular_vel_magnitude = np.linalg.norm(base_ang_vel, axis=1)
            angular_component = angular_vel_magnitude / 10.0
            
            # Store components for analysis
            self.disturbance_components['height_component'].extend(height_component.tolist())
            self.disturbance_components['lateral_component'].extend(lateral_component.tolist())
            self.disturbance_components['angular_component'].extend(angular_component.tolist())
            
            # COMPOSITE DISTURBANCE METRIC: Weighted combination
            composite_metric = (0.4 * height_component + 
                              0.4 * lateral_component + 
                              0.2 * angular_component)
            
            self.metrics['disturbance_resistance'].extend(composite_metric.tolist())
            
            print(f"📏 Step {self.step_counter}: Composite disturbance = {np.mean(composite_metric):.3f} "
                  f"(H:{np.mean(height_component):.2f}, L:{np.mean(lateral_component):.2f}, A:{np.mean(angular_component):.2f})")
            print(f"   💫 Trajectory deviation: {np.mean(trajectory_deviation):.3f}m (expected: {np.mean(expected_lateral_displacement):.3f}m, actual: {np.mean(actual_lateral_displacement):.3f}m)")
        
        # 3. POST-DISTURBANCE: Recovery component with trajectory awareness
        elif (self.disturbance_removed and 
              hasattr(self, 'disturbance_start_step') and 
              self.step_counter >= self.disturbance_start_step + self.disturbance_duration_steps and
              self.step_counter < self.disturbance_start_step + self.disturbance_duration_steps + 15):
            
            if self.recovery_start_step is None:
                self.recovery_start_step = self.step_counter
            
            # Recovery: How quickly robot returns to expected trajectory
            recovery_step = self.step_counter - self.recovery_start_step
            baseline_height = np.mean(self.pre_disturbance_baseline['height']) if self.pre_disturbance_baseline else 0.02
            
            current_height_dev = np.abs(base_pos[:, 2] - self.nominal_height)
            height_recovery = current_height_dev / max(baseline_height, 0.01)
            
            # Trajectory recovery: how quickly robot returns to commanded path
            steps_total = self.step_counter - self.disturbance_start_step
            dt = self.dt
            
            # Expected position if robot had followed commands perfectly from disturbance start
            expected_recovery_displacement = 0.0
            if (commands is not None and len(commands) > 0 and 
                self.trajectory_tracking['disturbance_start_commands'] is not None):
                cmd_vel_y = commands[:, 1] if commands.shape[-1] >= 2 else 0.0
                start_cmd_vel_y = self.trajectory_tracking['disturbance_start_commands'][:, 1] if self.trajectory_tracking['disturbance_start_commands'].shape[-1] >= 2 else 0.0
                avg_cmd_vel_y = (cmd_vel_y + start_cmd_vel_y) / 2.0
                expected_recovery_displacement = avg_cmd_vel_y * steps_total * dt
            
            actual_recovery_displacement = base_pos[:, 1] - self.trajectory_tracking['disturbance_start_pos'][:, 1]
            recovery_trajectory_deviation = np.abs(actual_recovery_displacement - expected_recovery_displacement)
            
            baseline_lateral_variation = np.std(self.pre_disturbance_baseline['lateral_pos']) if len(self.pre_disturbance_baseline['lateral_pos']) > 1 else 0.2
            lateral_recovery = recovery_trajectory_deviation / max(baseline_lateral_variation, 0.1)
            
            # Recovery penalty increases with time (should recover quickly)
            time_penalty = 1.0 + (recovery_step * 0.1)
            recovery_component = (height_recovery + lateral_recovery) * time_penalty
            
            self.disturbance_components['recovery_component'].extend(recovery_component.tolist())
            self.metrics['disturbance_resistance'].extend(recovery_component.tolist())
            
            if recovery_step == 0:
                print(f"🔄 Recovery started: measuring trajectory return (deviation: {np.mean(recovery_trajectory_deviation):.3f}m)")
    
    def _update_balance_stability_score(self, robot_state):
        """Update balance stability score (higher is better) - body stability using REAL robot orientation and angular velocity data only."""
        import torch
        import isaaclab.utils.math as math_utils
        
        base_quat = robot_state['base_quat']
        base_ang_vel = robot_state['base_ang_vel']
        
        # Convert to torch for Isaac Lab math operations
        base_quat_torch = torch.tensor(base_quat, device=self.env.unwrapped.device)
        
        # Calculate roll and pitch deviations (body orientation stability)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(base_quat_torch)
        orientation_deviation = torch.abs(roll) + torch.abs(pitch)
        
        # Calculate angular velocity magnitude (rotational stability)
        angular_velocity_magnitude = np.linalg.norm(base_ang_vel, axis=1)
        
        # Combine orientation and rotational stability
        # Higher score = more stable (lower deviations)
        balance_stability = 1.0 / (1.0 + orientation_deviation.cpu().numpy() + angular_velocity_magnitude * 0.1)
        
        self.metrics['balance_stability_score'].extend(balance_stability.tolist())
    
    def _update_gait_smoothness_score(self, robot_state):
        """Update jump coordination smoothness score (higher is better) - measures bilateral jumping coordination using REAL joint velocity data only."""
        # REQUIRE valid robot reference - no fallbacks
        if self.robot is None:
            raise RuntimeError("❌ Robot reference required for jump coordination calculation - no fallback data allowed")
        
        # Get joint velocities for jumping coordination analysis
        joint_velocities = self.robot.data.joint_vel.cpu().numpy()
        base_lin_vel = robot_state['base_lin_vel']
        
        # Focus on leg coordination for jumping (bilateral symmetry)
        # Get leg joint indices (hip, knee, ankle joints for both legs)
        num_joints = joint_velocities.shape[1]
        if num_joints >= 12:  # Ensure we have enough joints for leg analysis
            # Assume first 6 joints are left leg, next 6 are right leg (typical humanoid structure)
            left_leg_vels = joint_velocities[:, :6]  # Left leg joints
            right_leg_vels = joint_velocities[:, 6:12]  # Right leg joints
            
            # Calculate bilateral coordination (how similar left and right leg movements are)
            leg_vel_diff = np.abs(left_leg_vels - right_leg_vels)
            bilateral_coordination = np.mean(leg_vel_diff, axis=1)  # Average difference per environment
            
            # Calculate overall joint velocity magnitude for jumping power assessment
            jump_power_metric = np.linalg.norm(joint_velocities, axis=1)
            
        else:
            # Fallback: use all available joints if structure is different
            bilateral_coordination = np.zeros(self.num_envs)
            jump_power_metric = np.linalg.norm(joint_velocities, axis=1)
        
        # Calculate vertical movement consistency (how well robot maintains upward momentum)
        vertical_velocity = base_lin_vel[:, 2]  # Z-component
        vertical_consistency = np.abs(vertical_velocity)  # Consistent vertical movement
        
        # Jumping coordination score components:
        # 1. Bilateral symmetry (lower leg difference = better coordination)
        symmetry_score = np.exp(-bilateral_coordination / 2.0)  # Moderate tolerance for leg differences
        
        # 2. Power efficiency (moderate joint velocity for effective jumping)
        optimal_jump_velocity = 8.0  # Reasonable joint velocity for jumping
        power_score = np.exp(-np.abs(jump_power_metric - optimal_jump_velocity) / 4.0)
        
        # 3. Vertical consistency (reward consistent vertical movement)
        vertical_score = np.minimum(vertical_consistency / 2.0, 1.0)  # Cap at 2 m/s vertical velocity
        
        # Combined jumping coordination score (higher = better coordination)
        coordination_score = (0.4 * symmetry_score + 0.3 * power_score + 0.3 * vertical_score)
        
        self.metrics['gait_smoothness_score'].extend(coordination_score.tolist())
    
    def _update_stair_climbing_performance(self, robot_state):
        """Update stair climbing performance score (higher is better) - measures ascending progress and stability."""
        # Get current robot height (absolute position for consistent stair measurement)
        current_height = self.robot.data.root_pos_w[:, 2].cpu().numpy()
        
        # Initialize previous height tracking if needed
        if self.previous_height is None:
            self.previous_height = current_height.copy()
            self.cumulative_ascent = np.zeros(self.num_envs)
        
        # Calculate height change (positive = ascent, negative = descent)
        height_change = current_height - self.previous_height
        
        # Track cumulative ascent (only positive gains)
        positive_ascent = np.maximum(height_change, 0.0)
        self.cumulative_ascent += positive_ascent
        
        # Calculate stair climbing performance score
        # Component 1: Height Progress (40%) - Reward upward movement (adapted for jumping)
        ascent_progress = positive_ascent / 0.10  # Normalize by 10cm to match actual step height
        ascent_score = np.clip(ascent_progress, 0.0, 1.0)
            
        # Component 2: Height Stability (30%) - Maintain appropriate height relative to steps
        target_height = self.stair_baseline_height + self.cumulative_ascent
        height_deviation = np.abs(current_height - target_height)
        stability_score = np.exp(-height_deviation / 0.15)  # 15cm tolerance
        
        # Component 3: Consistent Progress (30%) - Reward steady climbing
        if len(self.metrics['stair_climbing_performance']) > 5:
            # Look at recent progress to encourage consistency
            recent_ascent = self.cumulative_ascent / max(len(self.metrics['stair_climbing_performance']) * self.dt, 1.0)
            progress_score = np.minimum(recent_ascent / 0.02, 1.0)  # Target 2cm/second ascent
        else:
            progress_score = np.ones(self.num_envs) * 0.5  # Neutral score initially
        
        # Combined performance score (higher is better)
        performance_score = (0.4 * ascent_score + 
                           0.3 * stability_score + 
                           0.3 * progress_score)
        
        # Store the metric
        self.metrics['stair_climbing_performance'].extend(performance_score.tolist())
        
        # Update tracking
        self.previous_height = current_height.copy()
        
        # Report significant ascent
        total_ascent = np.sum(positive_ascent)
        if total_ascent > 0.01:  # Report if any robot climbed more than 1cm
            print(f"🪜 STAIR ASCENT: {total_ascent:.3f}m total progress, avg performance: {np.mean(performance_score):.3f}")
    
    def _update_obstacle_collision_count(self):
        """Update obstacle collision count (smaller is better) - counts 2*N upper body G1 humanoid contacts with terrain elevation/obstacles (torso, arms, shoulders only - excludes legs)."""
        try:
            # Get contact sensor using Isaac Lab standard approach
            # PRIORITY: collision_sensor (comprehensive) > torso_contact > contact_sensor
            # EXPLICITLY AVOID contact_forces (that's for feet - not obstacles!)
            contact_sensor = self.env.unwrapped.scene.sensors.get("collision_sensor")
            contact_sensor_name = "collision_sensor"
            
            if contact_sensor is None:
                # Try body contact sensors only - NO FOOT SENSORS for obstacle detection
                for sensor_name in ["torso_contact", "contact_sensor", "body_contact"]:
                    contact_sensor = self.env.unwrapped.scene.sensors.get(sensor_name)
                    if contact_sensor is not None:
                        contact_sensor_name = sensor_name
                        break
            
            if contact_sensor is None:
                raise RuntimeError("❌ No body contact sensor found - obstacle collision detection requires body contact sensors (not foot sensors)")
            
            # Use Isaac Lab's proper contact force data with history
            # Shape: [num_envs, history_length, num_bodies, 3] or [num_envs, num_bodies, 3]
            if hasattr(contact_sensor.data, 'net_forces_w_history') and contact_sensor.data.net_forces_w_history is not None:
                # Use force history and take maximum over time dimension (most robust)
                net_contact_forces = contact_sensor.data.net_forces_w_history
                # Get peak forces over time: [num_envs, num_bodies, 3]
                peak_forces = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]  # [num_envs, num_bodies]
            elif hasattr(contact_sensor.data, "net_forces_w") and contact_sensor.data.net_forces_w is not None:
                # Use current forces: [num_envs, num_bodies, 3] 
                net_contact_forces = contact_sensor.data.net_forces_w
                # Calculate force magnitudes: [num_envs, num_bodies]
                peak_forces = torch.norm(net_contact_forces, dim=-1)
            else:
                raise RuntimeError("❌ Contact sensor has no force data available")
            
            # Find upper body collision parts using Isaac Lab G1 humanoid asset names
            # Focus on G1 upper body links that indicate obstacle collisions (torso, arms, shoulders only)
            collision_body_ids = []
            collision_body_names = []
            
            if self.robot is not None:
                try:
                    # Define UPPER BODY collision patterns for G1 humanoid robot
                    # Focus on torso, arms, and shoulders - exclude legs (normal locomotion contact)
                    collision_patterns = [
                        # G1 TORSO COLLISIONS - Body hitting obstacles
                        "pelvis",                  # Main pelvis/base contact
                        "torso_link",              # Main torso contact
                        "pelvis_contour_link",     # Pelvis contour contact
                        
                        # G1 ARM COLLISIONS - Arm hitting obstacles
                        "left_shoulder_pitch_link",  # Left shoulder pitch collisions
                        "right_shoulder_pitch_link", # Right shoulder pitch collisions
                        "left_shoulder_roll_link",   # Left shoulder roll collisions
                        "right_shoulder_roll_link",  # Right shoulder roll collisions
                        "left_shoulder_yaw_link",    # Left shoulder yaw collisions
                        "right_shoulder_yaw_link",   # Right shoulder yaw collisions
                        "left_elbow_pitch_link",     # Left elbow pitch collisions
                        "right_elbow_pitch_link",    # Right elbow pitch collisions
                        "left_elbow_roll_link",      # Left elbow roll collisions
                        "right_elbow_roll_link",     # Right elbow roll collisions
                        
                        # G1 HAND/PALM COLLISIONS - Hand hitting obstacles
                        "left_palm_link",            # Left palm collisions
                        "right_palm_link",           # Right palm collisions
                    ]
                    
                    # Try each body name and collect all collision-prone bodies
                    for body_name in collision_patterns:
                        try:
                            # Use direct body name lookup instead of regex patterns
                            if hasattr(self.robot, 'body_names') and body_name in self.robot.body_names:
                                body_idx = self.robot.body_names.index(body_name)
                                collision_body_ids.append(body_idx)
                                collision_body_names.append(body_name)
                            elif hasattr(self.robot, 'find_bodies'):
                                # Fallback to find_bodies for exact name matching
                                found_ids, found_names = self.robot.find_bodies(body_name)
                                if len(found_ids) > 0:
                                    collision_body_ids.extend(found_ids.tolist())
                                    collision_body_names.extend(found_names)
                        except:
                            continue
                        
                    # Remove duplicates and convert to tensor
                    collision_body_ids = list(set(collision_body_ids))
                    undesired_body_ids = torch.tensor(collision_body_ids, device=peak_forces.device)
                    
                    if len(collision_body_ids) == 0:
                        if peak_forces.shape[1] <= 5:  # Limited sensor coverage
                            undesired_body_ids = torch.arange(peak_forces.shape[1], device=peak_forces.device)
                        else:
                            undesired_body_ids = torch.arange(peak_forces.shape[1], device=peak_forces.device)
                    
                except Exception as e:
                    undesired_body_ids = torch.arange(peak_forces.shape[1], device=peak_forces.device)
            else:
                # Monitor all bodies if robot reference not available
                undesired_body_ids = torch.arange(peak_forces.shape[1], device=peak_forces.device)
            
            # CRITICAL: Check if contact sensor covers all body IDs
            num_bodies_in_sensor = peak_forces.shape[1]
            
            if len(undesired_body_ids) > 0:
                max_body_id = torch.max(undesired_body_ids).item()
                
                if max_body_id >= num_bodies_in_sensor:
                    # Filter out body IDs that exceed sensor coverage
                    valid_mask = undesired_body_ids < num_bodies_in_sensor
                    undesired_body_ids = undesired_body_ids[valid_mask]
                    
                    if len(undesired_body_ids) == 0:
                        undesired_body_ids = torch.arange(num_bodies_in_sensor, device=peak_forces.device)
            else:
                undesired_body_ids = torch.arange(num_bodies_in_sensor, device=peak_forces.device)
            
            # Count collisions using Isaac Lab vectorized approach
            # Check which undesired bodies exceed force threshold: [num_envs, num_undesired_bodies]
            undesired_forces = peak_forces[:, undesired_body_ids]
            is_collision = undesired_forces > self.collision_threshold
            
            # Count collisions per environment: [num_envs]
            collision_counts = torch.sum(is_collision, dim=1)
            
            # Convert to list (no scaling)
            final_collision_counts = collision_counts.cpu().tolist()
            
            # Store collision counts
            self.metrics['obstacle_collision_count'].extend(final_collision_counts)
            
            # Report detected collisions
            total_actual_collisions = torch.sum(collision_counts).item()
            total_reported_collisions = sum(final_collision_counts)
            
            if total_actual_collisions > 0:
                self.total_obstacle_collisions += total_actual_collisions
                print(f"🚧 UPPER BODY TERRAIN/OBSTACLE COLLISIONS: {total_actual_collisions} collisions detected")
                print(f"   📊 Sensor: {contact_sensor_name} | Threshold: {self.collision_threshold}N | Bodies monitored: {len(undesired_body_ids)} | Max force: {torch.max(undesired_forces).item():.3f}N")
                print(f"   🎯 Upper body parts hitting terrain/obstacles (torso, arms, shoulders only)")
                
        except Exception as e:
            print(f"❌ CRITICAL ERROR in collision detection: {e}")
            import traceback
            traceback.print_exc()
            # FAIL FAST - no fallbacks allowed per user request
            raise RuntimeError(f"Collision detection failed: {e}")
    
    def _finalize_disturbance_analysis(self):
        """Compute final disturbance analysis summary."""
        if not self.disturbance_components['height_component']:
            return
            
        try:
            # Component analysis
            height_comp = np.array(self.disturbance_components['height_component'])
            lateral_comp = np.array(self.disturbance_components['lateral_component'])
            angular_comp = np.array(self.disturbance_components['angular_component'])
            recovery_comp = np.array(self.disturbance_components['recovery_component']) if self.disturbance_components['recovery_component'] else np.array([0])
            
            # Store comprehensive analysis
            self.disturbance_analysis.update({
                'composite_metric_mean': np.mean(self.metrics['disturbance_resistance']),
                'height_component_mean': np.mean(height_comp),
                'lateral_component_mean': np.mean(lateral_comp), 
                'angular_component_mean': np.mean(angular_comp),
                'recovery_component_mean': np.mean(recovery_comp),
                'peak_composite_disturbance': np.max(self.metrics['disturbance_resistance']) if self.metrics['disturbance_resistance'] else 0,
                'total_disturbance_measurements': len(self.metrics['disturbance_resistance']),
                'component_weights': {'height': 0.4, 'lateral': 0.4, 'angular': 0.2, 'recovery': 'variable'}
            })
            
            print(f"📊 COMPOSITE DISTURBANCE ANALYSIS:")
            print(f"   • Overall disturbance resistance: {self.disturbance_analysis['composite_metric_mean']:.3f}")
            print(f"   • Height component (40%): {self.disturbance_analysis['height_component_mean']:.3f}") 
            print(f"   • Lateral component (40%): {self.disturbance_analysis['lateral_component_mean']:.3f}")
            print(f"   • Angular component (20%): {self.disturbance_analysis['angular_component_mean']:.3f}")
            print(f"   • Recovery component: {self.disturbance_analysis['recovery_component_mean']:.3f}")
            print(f"   • Peak disturbance: {self.disturbance_analysis['peak_composite_disturbance']:.3f}")
                
        except Exception as e:
            print(f"⚠️ Error in disturbance analysis: {e}")
    
    def get_summary_statistics(self):
        """Compute summary statistics for focused metrics."""
        # Finalize comprehensive disturbance analysis
        self._finalize_disturbance_analysis()
        
        summary = {}
        
        for metric_name, values in self.metrics.items():
                if isinstance(values, list) and len(values) > 0:
                    arr = np.array(values)
                summary[metric_name] = {
                    'count': len(values),
                            'mean': float(np.mean(arr)),
                            'std': float(np.std(arr)),
                            'min': float(np.min(arr)),
                            'max': float(np.max(arr)),
                            'median': float(np.median(arr)),
                }
        
        # Add enhanced disturbance analysis
        if hasattr(self, 'disturbance_analysis') and self.disturbance_analysis:
            summary['comprehensive_disturbance_analysis'] = self.disturbance_analysis
        
        # Add disturbance timing info
        if len(self.post_disturbance_steps) > 0:
            summary['disturbance_timing'] = {
                'disturbance_applied_at_step': self.disturbance_step,
                'force_duration_steps': self.disturbance_duration_steps,
                'measurements_during_force': len(self.post_disturbance_steps),
                'total_post_disturbance_steps': len(self.post_disturbance_data) if hasattr(self, 'post_disturbance_data') else 0,
            }
        
        return summary

    def _extract_height_scan_from_obs(self, obs):
        """Extract height scanner data from observations."""
        try:
            # Height scan is typically at a specific index in the observation vector
            # Based on the environment configuration, height_scan has shape (567,)
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = obs
            
            # Height scan starts after base state observations
            # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + joint_pos(23) + joint_vel(23) + actions(23) = 81
            # So height_scan starts at index 81 and has 567 elements
            height_scan_start = 81
            height_scan_end = height_scan_start + 567
            
            if obs_np.shape[-1] >= height_scan_end:
                height_scan = obs_np[:, height_scan_start:height_scan_end]
                return height_scan
            else:
                return None
        except Exception as e:
            print(f"⚠️ Error extracting height scan: {e}")
            return None
    
    def _extract_lidar_from_obs(self, obs):
        """Extract LiDAR data from observations."""
        try:
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = obs
            
            # LiDAR comes after height scan: 81 + 567 = 648, with 144 elements
            lidar_start = 648
            lidar_end = lidar_start + 144
            
            if obs_np.shape[-1] >= lidar_end:
                lidar_data = obs_np[:, lidar_start:lidar_end]
                return lidar_data
            else:
                return None
        except Exception as e:
            print(f"⚠️ Error extracting LiDAR: {e}")
            return None
    
    def _detect_gaps_from_sensors(self, height_scan_data, lidar_data):
        """Detect gaps from height scanner and LiDAR data."""
        if height_scan_data is None:
            return []
        
        try:
            gaps = []
            # Height scan is a grid pattern, convert to 2D for analysis
            # Grid pattern is typically configured as size=[2.0, 1.5] with resolution=0.075
            # This gives approximately 27x20 = 540 points (close to 567)
            
            grid_width = 27  # approximately 2.0m / 0.075m
            grid_height = 21  # approximately 1.5m / 0.075m
            
            for env_idx in range(min(self.num_envs, height_scan_data.shape[0])):
                env_heights = height_scan_data[env_idx]
                
                # Reshape to grid if possible
                if len(env_heights) >= grid_width * grid_height:
                    height_grid = env_heights[:grid_width * grid_height].reshape(grid_height, grid_width)
                    
                    # Detect gaps as regions significantly below normal height
                    mean_height = np.mean(height_grid)
                    gap_threshold = mean_height - 0.15  # 15cm below average = gap
                    
                    # Find connected regions below threshold
                    gap_mask = height_grid < gap_threshold
                    gap_regions = self._find_gap_regions(gap_mask)
                    
                    # Convert gap regions to world coordinates
                    for region in gap_regions:
                        gap_size = self._calculate_gap_size(region, grid_width, grid_height)
                        if gap_size > 0.1:  # Only consider gaps larger than 10cm
                            gaps.append({
                                'size': gap_size,
                                'position': self._gap_region_to_world_pos(region, grid_width, grid_height),
                                'depth': mean_height - np.mean(height_grid[gap_mask])
                            })
            
            return gaps
        except Exception as e:
            print(f"⚠️ Error detecting gaps: {e}")
            return []
    
    def _find_gap_regions(self, gap_mask):
        """Find connected regions in gap mask."""
        # Simple connected component analysis
        regions = []
        visited = np.zeros_like(gap_mask, dtype=bool)
        
        for i in range(gap_mask.shape[0]):
            for j in range(gap_mask.shape[1]):
                if gap_mask[i, j] and not visited[i, j]:
                    region = self._flood_fill(gap_mask, visited, i, j)
                    if len(region) > 5:  # Only consider regions with at least 5 points
                        regions.append(region)
        
        return regions
    
    def _flood_fill(self, gap_mask, visited, start_i, start_j):
        """Flood fill to find connected gap region."""
        region = []
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            if (i < 0 or i >= gap_mask.shape[0] or j < 0 or j >= gap_mask.shape[1] or 
                visited[i, j] or not gap_mask[i, j]):
                continue
            
            visited[i, j] = True
            region.append((i, j))
            
            # Add neighbors to stack
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((i + di, j + dj))
        
        return region
    
    def _calculate_gap_size(self, region, grid_width, grid_height):
        """Calculate the size of a gap region."""
        if not region:
            return 0.0
        
        # Calculate bounding box of region
        min_i = min(point[0] for point in region)
        max_i = max(point[0] for point in region)
        min_j = min(point[1] for point in region)
        max_j = max(point[1] for point in region)
        
        # Convert to world coordinates (assuming 2.0m x 1.5m scan area)
        width_m = (max_j - min_j) * (2.0 / grid_width)
        height_m = (max_i - min_i) * (1.5 / grid_height)
        
        return max(width_m, height_m)  # Return the larger dimension
    
    def _gap_region_to_world_pos(self, region, grid_width, grid_height):
        """Convert gap region to world position relative to robot."""
        if not region:
            return [0.0, 0.0]
        
        # Calculate center of region
        center_i = np.mean([point[0] for point in region])
        center_j = np.mean([point[1] for point in region])
        
        # Convert to world coordinates (relative to robot)
        # Height scanner is centered on robot with 2.0m x 1.5m scan area
        world_x = (center_j / grid_width - 0.5) * 2.0  # -1.0 to 1.0 meters
        world_y = (center_i / grid_height - 0.5) * 1.5  # -0.75 to 0.75 meters
        
        return [world_x, world_y]
    
    def _count_actual_gaps_in_vicinity(self, robot_state):
        """Count actual gaps in the vicinity (simplified heuristic)."""
        # This is a simplified heuristic - in reality, you'd need ground truth terrain data
        # For now, assume there are typically 1-3 gaps in the scanning area
        return 2  # Placeholder
    
    def _get_foot_positions(self, robot_state):
        """Get current foot positions from robot state."""
        try:
            if self.robot is None:
                return []
            
            # Get foot body positions (left and right ankle roll links)
            foot_bodies = ["left_ankle_roll_link", "right_ankle_roll_link"]
            foot_positions = []
            
            for foot_name in foot_bodies:
                foot_indices = self.robot.find_bodies(foot_name)
                if len(foot_indices) > 0:
                    foot_idx = foot_indices[0].item() if hasattr(foot_indices[0], 'item') else foot_indices[0]
                    foot_pos = self.robot.data.body_pos_w[:, foot_idx, :].cpu().numpy()
                    # Convert to relative position
                    foot_pos_rel = foot_pos - self.env.unwrapped.scene.env_origins.cpu().numpy()
                    foot_positions.append(foot_pos_rel)
            
            return foot_positions
        except Exception as e:
            print(f"⚠️ Error getting foot positions: {e}")
            return []


def main():
    """Main function to collect policy performance data."""
    
    print("🎯 Comprehensive Policy Performance Data Collection")
    print("=" * 60)
    print(f"📁 Checkpoint: {args_cli.checkpoint}")
    print(f"🎮 Task: {args_cli.task}")
    print(f"🔢 Environments: {args_cli.num_envs}")
    print(f"⏱️ Steps: {args_cli.steps}")
    
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    print(f"✅ Environment created: {env.observation_space}")
    
    # load policy if checkpoint provided
    if args_cli.checkpoint:
        print(f"🤖 Loading policy from: {args_cli.checkpoint}")
        
        # get checkpoint path
        resume_path = retrieve_file_path(args_cli.checkpoint)
        
        # wrap environment for RSL-RL
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        # load policy
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        
        # get inference policy
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        print(f"✅ Policy loaded successfully!")
        
    else:
        print("❌ No checkpoint provided!")
        env.close()
        return
    
    # initialize comprehensive metrics collector
    metrics_collector = TerrainAwareMetricsCollector(env, args_cli.num_envs, args_cli.terrain_type)
    
    # initialize data collection
    collected_data = {
        'metadata': {
            'task': args_cli.task,
            'checkpoint': args_cli.checkpoint,
            'num_envs': args_cli.num_envs,
            'collection_steps': args_cli.steps,
            'timestamp': datetime.now().isoformat(),
        },
        'step_data': {
            'rewards': [],
            'observations': [],
            'actions': [],
        },
        'metrics': {},
        'summary': {},
    }
    
    # reset environment
    obs, _ = env.get_observations()
    
    print(f"🚀 Starting comprehensive data collection for {args_cli.steps} steps...")
    
    step_count = 0
    start_time = time.time()
    
    # data collection loop
    with torch.inference_mode():
        while step_count < args_cli.steps:
            # policy inference
            actions = policy(obs)
            
            # environment step
            step_result = env.step(actions)
            if len(step_result) == 5:
                obs, rewards, terminated, truncated, info = step_result
            else:
                obs, rewards, dones, info = step_result
                terminated = dones
                truncated = dones
            
            # store step data
            collected_data['step_data']['rewards'].append(
                rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
            )
            collected_data['step_data']['observations'].append(
                obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
            )
            collected_data['step_data']['actions'].append(
                actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
            )
            
            # update comprehensive metrics
            metrics_collector.update_metrics(obs, actions, rewards, terminated, truncated)
            
            step_count += 1
            
            # progress update
            if step_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  📊 Step {step_count}/{args_cli.steps} ({step_count/args_cli.steps*100:.1f}%) - {elapsed:.1f}s")
    
    # compute final metrics
    print("🧮 Computing comprehensive metrics...")
    collected_data['metrics'] = metrics_collector.get_summary_statistics()
    
    # compute summary statistics
    all_rewards = np.concatenate(collected_data['step_data']['rewards'])
    collected_data['summary'] = {
        'total_steps': step_count,
        'mean_reward': float(np.mean(all_rewards)),
        'total_reward': float(np.sum(all_rewards)),
        'collection_time': time.time() - start_time,
    }
        
    print(f"✅ Data collection completed!")
    print(f"📈 Mean step reward: {collected_data['summary']['mean_reward']:.3f}")
    print(f"📊 Total reward: {collected_data['summary']['total_reward']:.1f}")
        
    # Print comprehensive metrics summary
    if 'height_deviation' in collected_data['metrics']:
        hd = collected_data['metrics']['height_deviation']
        if 'mean' in hd:
            print(f"📐 Mean height deviation: {hd['mean']:.3f} m")
    if 'velocity_tracking_error' in collected_data['metrics']:
        vt = collected_data['metrics']['velocity_tracking_error']
        if 'mean' in vt:
            print(f"🎯 Mean velocity tracking error: {vt['mean']:.3f} m/s")
    if 'disturbance_resistance' in collected_data['metrics']:
        dr = collected_data['metrics']['disturbance_resistance']
        if 'mean' in dr:
            print(f"🛡️ Mean disturbance resistance: {dr['mean']:.3f}")
    
    # save data
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(collected_data, f)
    
    print(f"💾 Comprehensive data saved to: {output_path}")
    
    # close environment
    env.close()
    

if __name__ == "__main__":
    # run the main function
    main() 
    # close sim app
    simulation_app.close() 