#!/usr/bin/env python3
"""
Simple Real Policy Data Collection
==================================

Collects real performance data from trained policies for comparative analysis.
Based on Isaac Lab's play_with_contact_plotting.py script.

This script loads actual trained checkpoints and extracts real metrics.

Usage:
    # Environment-Aware data collection
    ./isaaclab.sh -p SDS_ANONYM/collect_policy_data_simple.py \
        --task Isaac-SDS-Velocity-Flat-G1-Enhanced-v0 \
        --checkpoint logs/rsl_rl/g1_enhanced/2025-08-12_16-33-40/model_499.pt \
        --num_envs 50 --headless --steps 1000 --output environment_aware_data.pkl

    # Foundation-Only data collection  
    ./isaaclab.sh -p SDS_ANONYM/collect_policy_data_simple.py \
        --task Isaac-SDS-Velocity-Flat-G1-Enhanced-v0 \
        --checkpoint logs/rsl_rl/g1_enhanced/2025-08-12_16-13-29/model_499.pt \
        --num_envs 50 --headless --steps 1000 --output foundation_only_data.pkl \
        --foundation_only

Author: SDS Performance Analysis Team
"""

import argparse
import os
import torch
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

# Isaac Lab imports - must come first
from isaaclab.app import AppLauncher

# Initialize the simulation context
parser = argparse.ArgumentParser(description="Simple Real Policy Data Collection")
parser.add_argument("--task", type=str, default="Isaac-SDS-Velocity-Flat-G1-Enhanced-v0", help="Isaac Lab task")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to collect")
parser.add_argument("--output", type=str, required=True, help="Output pickle file")
parser.add_argument("--foundation_only", action="store_true", help="Disable sensors for foundation-only mode")
parser.add_argument("--headless", action="store_true", help="Run headless")
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
# Add missing arguments expected by cli_args.parse_rsl_rl_cfg
parser.add_argument("--resume", default=None, help="Resume from checkpoint")
parser.add_argument("--run_dir", default=None, help="Run directory")
parser.add_argument("--max_iterations", type=int, default=None, help="Max iterations")
parser.add_argument("--load_run", default=None, help="Load run")
parser.add_argument("--load_checkpoint", default=None, help="Load checkpoint")
parser.add_argument("--log_interval", type=int, default=None, help="Log interval")
parser.add_argument("--save_interval", type=int, default=None, help="Save interval")
parser.add_argument("--wandb", action="store_true", help="Use wandb")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--run_name", default=None, help="Run name")
parser.add_argument("--logger", default=None, help="Logger type")
parser.add_argument("--log_project_name", default=None, help="Project name for logging")

args_cli = parser.parse_args()

# Launch the simulation application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import time

import isaaclab
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.dict import print_dict
import isaaclab.utils.math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Add scripts directory to path for cli_args import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reinforcement_learning', 'rsl_rl'))
import cli_args

def set_sensor_mode(foundation_only: bool):
    """Set the sensor mode in the environment configuration."""
    try:
        import isaaclab_tasks.manager_based.sds.velocity.config.g1.flat_with_box_env_cfg as env_cfg_module
        env_cfg_module.SENSORS_ENABLED = not foundation_only
        print(f"🔧 Set SENSORS_ENABLED = {not foundation_only} (foundation_only = {foundation_only})")
        return True
    except Exception as e:
        print(f"⚠️ Could not set sensor mode: {e}")
        return False

class ComprehensiveMetricsCollector:
    """Collects comprehensive metrics for environment-aware vs foundation-only comparison."""
    
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
        self.contact_threshold = 50.0  # Newtons
        self.dt = 0.02  # Environment timestep (50 Hz)
        self.nominal_height = 0.74  # G1 nominal height in meters
        
        # ✅ FIX: Get actual dt from environment instead of hardcoding
        if hasattr(env.unwrapped, 'step_dt'):
            self.dt = env.unwrapped.step_dt
        elif hasattr(env.unwrapped, 'cfg') and hasattr(env.unwrapped.cfg, 'decimation'):
            # dt = sim_dt * decimation  
            sim_dt = getattr(env.unwrapped.cfg.sim, 'dt', 0.005)  # Default physics dt
            decimation = getattr(env.unwrapped.cfg, 'decimation', 4)  # Default decimation
            self.dt = sim_dt * decimation
        else:
            self.dt = 0.02  # Fallback
        print(f"📊 Using environment timestep: {self.dt:.4f}s ({1/self.dt:.1f} Hz)")
        
        # Get foot body indices from contact sensor
        self.foot_body_ids = None
        self._initialize_foot_indices()
        
        # Initialize metric storage
        self.metrics = {
            # 1. Tracking Performance
            'tracking_performance': {
                'velocity_error_xy': [],
                'velocity_error_yaw': [],
                'command_tracking_accuracy': [],
                'velocity_variance': [],
                'heading_drift': [],
            },
            
            # 2. Gait and Cadence Quality
            'gait_quality': {
                'step_frequency': [],
                'stride_length': [],
                'stance_phase_duration': [],
                'swing_phase_duration': [],
                'double_support_time': [],
                'gait_symmetry': [],
                'foot_clearance': [],
                'step_timing_regularity': [],
                'contact_timing_accuracy': [],
            },
            
            # 3. Foot Contact and Sliding
            'foot_contact': {
                'foot_slide_distance': [],
                'contact_force_magnitude': [],
                'contact_stability': [],
                'ground_reaction_forces': [],
                'foot_slip_velocity': [],
                'contact_timing_accuracy': [],
            },
            
            # 4. Stability and Posture
            'stability_posture': {
                'base_height_deviation': [],
                'orientation_error': [],
                'angular_velocity_magnitude': [],
                'com_stability': [],
                'balance_recovery_time': [],
                'postural_sway': [],
                'trunk_inclination': [],
            },
            
            # 5. Velocity/Orientation Stability Indices
            'velocity_stability': {
                'velocity_smoothness': [],
                'acceleration_magnitude': [],
                'jerk_index': [],
                'direction_changes': [],
                'speed_consistency': [],
                'turning_smoothness': [],
            },
            
            # 6. Command Adherence
            'command_adherence': {
                'command_response_time': [],
                'overshoot_magnitude': [],
                'steady_state_error': [],
                'command_following_precision': [],
                'response_delay': [],
            },
            
            # 7. Evaluation Coverage/Sanity
            'evaluation_coverage': {
                'workspace_exploration': [],
                'velocity_range_coverage': [],
                'turning_range_coverage': [],
                'episode_completion_rate': [],
                'termination_reasons': defaultdict(int),
            },
            
            # 8. External Disturbance Robustness
            'disturbance_robustness': {
                'recovery_time': [],
                'stability_margin': [],
                'perturbation_resistance': [],
                'adaptive_response': [],
                'fall_prevention': [],
            },
        }
        
        # State tracking for metric computation
        self.previous_states = {}
        # Initialize per-environment data tracking
        num_feet = len(self.foot_body_ids) if self.foot_body_ids is not None else 2  # Default to 2 feet
        self.env_contact_history = {i: {f"foot_{j}": deque(maxlen=100) for j in range(num_feet)} for i in range(self.num_envs)}
        self.env_foot_position_history = {i: {f"foot_{j}": deque(maxlen=100) for j in range(num_feet)} for i in range(self.num_envs)}  # Increased from 10 to 100
        self.env_velocity_history = {i: deque(maxlen=50) for i in range(self.num_envs)}
        self.env_heading_history = {i: deque(maxlen=50) for i in range(self.num_envs)}
        self.env_contact_force_history = {i: deque(maxlen=20) for i in range(self.num_envs)}
        self.env_stability_history = {i: deque(maxlen=30) for i in range(self.num_envs)}
        
        # Contact timing tracking
        self.env_last_contact_step = {i: {f"foot_{j}": 0 for j in range(num_feet)} for i in range(self.num_envs)}
        self.env_last_contact_state = {i: {f"foot_{j}": False for j in range(num_feet)} for i in range(self.num_envs)}
        
        # Dynamic stance duration estimation (instead of hardcoded 0.3s)
        self.env_stance_durations = {i: deque(maxlen=10) for i in range(self.num_envs)}  # Track recent stance durations
        
        # Step counter and episode tracking
        self.step_counter = 0
        
        # Per-environment episode tracking
        self.env_episode_rewards = [0.0] * num_envs
        self.env_episode_lengths = [0] * num_envs
        self.completed_episodes = []
        
        # Rolling buffers for per-env metrics
        self.env_height_history = [deque(maxlen=20) for _ in range(num_envs)]
        self.env_workspace_min_max = [(np.inf, -np.inf, np.inf, -np.inf) for _ in range(num_envs)]  # min_x, max_x, min_y, max_y
        
        # ✅ NEW: Enhanced tracking for improved metrics
        self.env_heading_history = [deque(maxlen=30) for _ in range(num_envs)]  # Heading angle tracking
        self.env_velocity_smoothness_history = [deque(maxlen=15) for _ in range(num_envs)]  # Velocity for smoothness
        self.env_contact_force_history = [deque(maxlen=20) for _ in range(num_envs)]  # Contact stability over time
        self.env_stability_history = [deque(maxlen=25) for _ in range(num_envs)]  # Per-env disturbance tracking
        self.env_foot_positions_history = [deque(maxlen=10) for _ in range(num_envs)]  # For stride length
        self.env_last_contact_step = [{} for _ in range(num_envs)]  # Contact timing tracking
        self.command_history = [deque(maxlen=10) for _ in range(num_envs)]  # Command history for overshoot detection
    
    def _initialize_foot_indices(self):
        """Initialize foot body indices from contact sensor configuration."""
        try:
            if hasattr(self.env.unwrapped, 'scene') and hasattr(self.env.unwrapped.scene, 'contact_forces'):
                contact_sensor = self.env.unwrapped.scene.contact_forces
                # Get foot body IDs from contact sensor (matches .*_ankle_roll_link pattern)
                foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"] 
                self.foot_body_ids, _ = contact_sensor.find_bodies(foot_names, preserve_order=True)
                self.foot_body_ids = torch.tensor(self.foot_body_ids, dtype=torch.long, device=self.env.unwrapped.device)
                print(f"✅ Found foot body IDs: {self.foot_body_ids.tolist()} for {foot_names}")
            elif hasattr(self.env.unwrapped, 'scene') and hasattr(self.env.unwrapped.scene, 'robot'):
                # Try to get foot indices from robot directly
                robot = self.env.unwrapped.scene.robot
                foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
                self.foot_body_ids, _ = robot.find_bodies(foot_names, preserve_order=True)
                self.foot_body_ids = torch.tensor(self.foot_body_ids, dtype=torch.long, device=self.env.unwrapped.device)
                print(f"✅ Found foot body IDs from robot: {self.foot_body_ids.tolist()} for {foot_names}")
            else:
                print("⚠️ No contact sensor or robot found, foot slip metrics will be unavailable")
                self.foot_body_ids = None
        except Exception as e:
            print(f"⚠️ Could not resolve foot body IDs: {e}")
            self.foot_body_ids = None
            
    def _get_yaw_quaternion(self, quat_w):
        """Extract yaw-only quaternion from full quaternion (w, x, y, z)."""
        # Extract yaw angle from quaternion and create yaw-only rotation
        forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=quat_w.dtype, device=quat_w.device)
        # Expand forward_vec to match batch size
        forward_vec = forward_vec.expand(quat_w.shape[0], -1)
        forward_w = math_utils.quat_apply(quat_w, forward_vec)
        yaw = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        
        # Create yaw-only quaternion (rotation around z-axis)
        zeros = torch.zeros_like(yaw)
        yaw_quat = torch.stack([
            torch.cos(yaw / 2),  # w
            zeros,               # x  
            zeros,               # y
            torch.sin(yaw / 2)   # z
        ], dim=1)
        return yaw_quat
        
    def update_metrics(self, obs, actions, rewards, robot_state, commands, contact_forces, terminated, truncated):
        """Update all metric categories with current step data."""
        self.step_counter += 1
        
        # Convert tensors to numpy for easier computation
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = obs
            
        # 1. TRACKING PERFORMANCE
        self._update_tracking_performance(robot_state, commands)
        
        # 2. GAIT AND CADENCE QUALITY
        self._update_gait_quality(robot_state, contact_forces)
        
        # 3. FOOT CONTACT AND SLIDING
        self._update_foot_contact(robot_state, contact_forces)
        
        # 4. STABILITY AND POSTURE
        self._update_stability_posture(robot_state)
        
        # 5. VELOCITY/ORIENTATION STABILITY
        self._update_velocity_stability(robot_state)
        
        # 6. COMMAND ADHERENCE
        self._update_command_adherence(robot_state, commands)
        
        # 7. EVALUATION COVERAGE
        self._update_evaluation_coverage(robot_state, terminated, truncated)
        
        # 8. EXTERNAL DISTURBANCE ROBUSTNESS
        self._update_disturbance_robustness(robot_state, contact_forces)
        
        # Update per-environment episode tracking
        self._update_episode_tracking(rewards, terminated, truncated)
        
        # Store current state for next iteration
        self.previous_states = {
            'base_pos': robot_state.get('base_pos', np.zeros((self.num_envs, 3))),
            'base_quat': robot_state.get('base_quat', np.zeros((self.num_envs, 4))),
            'base_lin_vel': robot_state.get('base_lin_vel', np.zeros((self.num_envs, 3))),
            'base_ang_vel': robot_state.get('base_ang_vel', np.zeros((self.num_envs, 3))),
            'commands': commands,
            'step': self.step_counter,
        }
    
    def _update_tracking_performance(self, robot_state, commands):
        """Update tracking performance metrics with correct coordinate frame."""
        if commands is None or len(commands) == 0:
            return
            
        base_lin_vel_w = robot_state.get('base_lin_vel', np.zeros((self.num_envs, 3)))
        base_ang_vel_w = robot_state.get('base_ang_vel', np.zeros((self.num_envs, 3)))
        base_quat_w = robot_state.get('base_quat', np.zeros((self.num_envs, 4)))
        
        # Convert to torch tensors for math_utils
        base_lin_vel_w_torch = torch.tensor(base_lin_vel_w, device=self.env.unwrapped.device)
        base_quat_w_torch = torch.tensor(base_quat_w, device=self.env.unwrapped.device)
        
        # ✅ FIX: Convert world frame velocity to yaw-aligned base frame
        # Commands are in yaw-aligned base frame, so we need to rotate velocities accordingly
        yaw_quat = self._get_yaw_quaternion(base_quat_w_torch)
        base_lin_vel_yaw = math_utils.quat_apply_inverse(yaw_quat, base_lin_vel_w_torch[:, :3])
        
        # Velocity tracking error in yaw frame
        if commands.shape[-1] >= 3:  # x, y, yaw commands
            cmd_vel_x = commands[:, 0] if commands.ndim > 1 else commands[0]
            cmd_vel_y = commands[:, 1] if commands.ndim > 1 else commands[1]
            cmd_yaw = commands[:, 2] if commands.ndim > 1 else commands[2]
            
            # ✅ FIX: Correct frame comparison
            vel_error_xy = np.sqrt((base_lin_vel_yaw[:, 0].cpu().numpy() - cmd_vel_x)**2 + 
                                 (base_lin_vel_yaw[:, 1].cpu().numpy() - cmd_vel_y)**2)
            vel_error_yaw = np.abs(base_ang_vel_w[:, 2] - cmd_yaw)
            
            self.metrics['tracking_performance']['velocity_error_xy'].extend(vel_error_xy.tolist())
            self.metrics['tracking_performance']['velocity_error_yaw'].extend(vel_error_yaw.tolist())
            
            # Command tracking accuracy (1 - normalized error)
            max_vel = np.maximum(np.linalg.norm(commands[:, :2], axis=1), 0.1)
            tracking_accuracy = 1.0 - (vel_error_xy / max_vel)
            self.metrics['tracking_performance']['command_tracking_accuracy'].extend(tracking_accuracy.tolist())
            
            # ✅ FIX: Proper velocity variance per environment
            for env_idx in range(self.num_envs):
                vel_magnitude = np.linalg.norm(base_lin_vel_yaw[env_idx, :2].cpu().numpy())
                self.env_velocity_history[env_idx].append(vel_magnitude)
                
                if len(self.env_velocity_history[env_idx]) > 10:
                    vel_var = float(np.var(list(self.env_velocity_history[env_idx])))
                    self.metrics['tracking_performance']['velocity_variance'].append(vel_var)
            
            # ✅ FIX: Proper heading drift - track actual heading angle deviation
            for env_idx in range(self.num_envs):
                # Extract current heading angle from quaternion
                current_yaw = np.arctan2(2.0 * (base_quat_w[env_idx, 0] * base_quat_w[env_idx, 3] + 
                                               base_quat_w[env_idx, 1] * base_quat_w[env_idx, 2]),
                                        1.0 - 2.0 * (base_quat_w[env_idx, 2]**2 + base_quat_w[env_idx, 3]**2))
                
                self.env_heading_history[env_idx].append(current_yaw)
                
                if len(self.env_heading_history[env_idx]) > 10:
                    # Compute heading drift as std of heading angles over time
                    headings = np.array(list(self.env_heading_history[env_idx]))
                    # Handle angle wrapping by computing circular standard deviation
                    mean_heading = np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings)))
                    angular_diffs = np.arctan2(np.sin(headings - mean_heading), np.cos(headings - mean_heading))
                    heading_drift = float(np.std(angular_diffs))
                    self.metrics['tracking_performance']['heading_drift'].append(heading_drift)
    
    def _update_gait_quality(self, robot_state, contact_forces):
        """Update gait and cadence quality metrics with proper step detection."""
        if contact_forces is None or len(contact_forces) == 0 or self.foot_body_ids is None:
            return
            
        # ✅ FIX: Use correct foot body indexing
        contact_forces_np = contact_forces if isinstance(contact_forces, np.ndarray) else contact_forces
        if contact_forces_np.ndim >= 3 and len(self.foot_body_ids) >= 2:
            # Extract foot contact forces using resolved indices
            left_foot_idx = self.foot_body_ids[0].item()
            right_foot_idx = self.foot_body_ids[1].item()
            
            left_forces = np.linalg.norm(contact_forces_np[:, left_foot_idx, :], axis=1)
            right_forces = np.linalg.norm(contact_forces_np[:, right_foot_idx, :], axis=1)
            
            left_contact = left_forces > self.contact_threshold
            right_contact = right_forces > self.contact_threshold
            
            # Track contact history per environment
            for env_idx in range(self.num_envs):
                left_key = 'foot_0'  # Use consistent key names
                right_key = 'foot_1'
                
                self.env_contact_history[env_idx][left_key].append(left_contact[env_idx])
                self.env_contact_history[env_idx][right_key].append(right_contact[env_idx])
                
                # Compute gait metrics when we have sufficient history
                if len(self.env_contact_history[env_idx][left_key]) > 20:
                    left_hist = np.array(list(self.env_contact_history[env_idx][left_key]))
                    right_hist = np.array(list(self.env_contact_history[env_idx][right_key]))
                    
                    # ✅ FIX: Step frequency - count only rising edges (contact initiation)
                    left_steps = np.sum(np.diff(left_hist.astype(int)) == 1)  # False->True transitions
                    right_steps = np.sum(np.diff(right_hist.astype(int)) == 1)
                    total_steps = left_steps + right_steps
                    
                    if total_steps > 0:
                        step_freq = total_steps / (len(left_hist) * self.dt)
                        self.metrics['gait_quality']['step_frequency'].append(step_freq)
                    
                    # Double support time
                    double_support = np.sum(left_hist & right_hist) / len(left_hist)
                    self.metrics['gait_quality']['double_support_time'].append(double_support)
                    
                    # Gait symmetry
                    left_stance_time = np.sum(left_hist) / len(left_hist)
                    right_stance_time = np.sum(right_hist) / len(right_hist)
                    symmetry = 1.0 - abs(left_stance_time - right_stance_time)
                    self.metrics['gait_quality']['gait_symmetry'].append(symmetry)
                    
                    # Stance and swing phase durations
                    left_contacts = self._get_phase_durations(left_hist)
                    right_contacts = self._get_phase_durations(right_hist)
                    
                    if left_contacts['stance'] and right_contacts['stance']:
                        avg_stance = (np.mean(left_contacts['stance']) + np.mean(right_contacts['stance'])) / 2 * self.dt
                        self.metrics['gait_quality']['stance_phase_duration'].append(avg_stance)
                    
                    if left_contacts['swing'] and right_contacts['swing']:
                        avg_swing = (np.mean(left_contacts['swing']) + np.mean(right_contacts['swing'])) / 2 * self.dt
                        self.metrics['gait_quality']['swing_phase_duration'].append(avg_swing)
                    
                    # Step timing regularity (coefficient of variation of step intervals)
                    if len(left_contacts['stance']) > 1 and len(right_contacts['stance']) > 1:
                        all_intervals = left_contacts['stance'] + right_contacts['stance']
                        if len(all_intervals) > 3:
                            regularity = 1.0 - (np.std(all_intervals) / (np.mean(all_intervals) + 1e-6))
                            self.metrics['gait_quality']['step_timing_regularity'].append(max(0.0, regularity))
                    
                    # ✅ NEW: Stride length calculation
                    if hasattr(self.env.unwrapped, 'scene') and hasattr(self.env.unwrapped.scene, 'robot'):
                        robot = self.env.unwrapped.scene.robot
                        if hasattr(robot.data, 'body_pos_w') and len(self.foot_body_ids) >= 2:
                            # Get foot positions
                            foot_positions = robot.data.body_pos_w[:, self.foot_body_ids, :2]  # xy only
                            current_foot_pos = foot_positions[env_idx].cpu().numpy()  # Shape: (2, 2)
                            
                            self.env_foot_positions_history[env_idx].append(current_foot_pos.copy())
                            
                            if len(self.env_foot_positions_history[env_idx]) > 5:
                                # Calculate stride length as distance between successive foot placements
                                foot_hist = list(self.env_foot_positions_history[env_idx])
                                for foot_idx in range(len(self.foot_body_ids)):
                                    # Find contact events for this foot
                                    foot_contacts = left_hist if foot_idx == 0 else right_hist
                                    contact_steps = np.where(np.diff(foot_contacts.astype(int)) == 1)[0]  # Contact initiation steps
                                    
                                    if len(contact_steps) > 1:
                                        # Measure distance between last two contact points
                                        recent_contacts = contact_steps[-2:]
                                        if recent_contacts[1] < len(foot_hist) and recent_contacts[0] >= 0:
                                            pos1 = foot_hist[recent_contacts[0]][foot_idx]
                                            pos2 = foot_hist[recent_contacts[1]][foot_idx]
                                            stride_length = float(np.linalg.norm(pos2 - pos1))
                                            self.metrics['gait_quality']['stride_length'].append(stride_length)
                    
                    # === Foot Clearance ===
                    foot_clearance_values = []
                    if hasattr(self.env.unwrapped, 'scene') and hasattr(self.env.unwrapped.scene, 'robot'):
                        robot = self.env.unwrapped.scene.robot
                        foot_heights = robot.data.body_pos_w[:, self.foot_body_ids, 2]  # Z-coordinates of feet
                        
                        for env_idx in range(self.num_envs):
                            for foot_idx, foot_id in enumerate(self.foot_body_ids):
                                foot_key = f"foot_{foot_idx}"
                                
                                # Check if foot is in swing phase (not in contact)
                                is_in_contact = False
                                if contact_forces_np is not None and contact_forces_np.shape[0] > env_idx and contact_forces_np.shape[1] > foot_idx:
                                    is_in_contact = contact_forces_np[env_idx, foot_idx, 2] > self.contact_threshold
                                
                                if not is_in_contact:  # Swing phase
                                    current_height = foot_heights[env_idx, foot_idx]
                                    
                                    # Estimate ground level from recent contact positions
                                    ground_height = 0.0  # Default ground level
                                    if foot_key in self.env_foot_position_history[env_idx]:
                                        recent_positions = list(self.env_foot_position_history[env_idx][foot_key])
                                        if recent_positions:
                                            # Get Z-coordinates from recent 3D positions where foot was in contact
                                            ground_heights = []
                                            for pos_data in recent_positions:
                                                if len(pos_data) == 4:  # [x, y, z, was_in_contact]
                                                    if pos_data[3]:  # was_in_contact
                                                        ground_heights.append(pos_data[2])  # Z coordinate
                                            
                                            if ground_heights:
                                                ground_height = np.mean(ground_heights)
                                    
                                    clearance = current_height - ground_height
                                    foot_clearance_values.append(max(0.0, clearance))  # Ensure non-negative
                                
                                # Store current foot position with contact state for future ground estimation
                                current_pos = robot.data.body_pos_w[env_idx, foot_id].cpu().numpy()
                                pos_with_contact = [current_pos[0], current_pos[1], current_pos[2], is_in_contact]
                                self.env_foot_position_history[env_idx][foot_key].append(pos_with_contact)
                    
                    foot_clearance = float(np.mean(foot_clearance_values)) if foot_clearance_values else 0.0
                    self.metrics['gait_quality']['foot_clearance'].append(foot_clearance)
                    
                    # ✅ NEW: Contact timing accuracy
                    contact_timing_accuracy_values = []
                    for env_idx in range(self.num_envs):
                        for foot_idx in range(len(self.foot_body_ids)):
                            foot_key = f"foot_{foot_idx}"
                            is_in_contact = False
                            if contact_forces_np is not None and contact_forces_np.shape[0] > env_idx and contact_forces_np.shape[1] > foot_idx:
                                is_in_contact = contact_forces_np[env_idx, foot_idx, 2] > self.contact_threshold
                            
                            # Track contact state changes
                            was_in_contact = self.env_last_contact_state[env_idx][foot_key]
                            
                            if was_in_contact and not is_in_contact:  # End of stance phase
                                stance_duration = (self.step_counter - self.env_last_contact_step[env_idx][foot_key]) * self.dt
                                self.env_stance_durations[env_idx].append(stance_duration)
                            
                            if not was_in_contact and is_in_contact:  # Start of new stance
                                self.env_last_contact_step[env_idx][foot_key] = self.step_counter
                            
                            self.env_last_contact_state[env_idx][foot_key] = is_in_contact
                            
                            # Calculate timing accuracy using dynamic expected duration
                            if is_in_contact:
                                current_stance_duration = (self.step_counter - self.env_last_contact_step[env_idx][foot_key]) * self.dt
                                
                                # Use recent average stance duration as expected, fallback to 0.3s
                                if len(self.env_stance_durations[env_idx]) > 0:
                                    expected_stance_duration = np.mean(list(self.env_stance_durations[env_idx]))
                                else:
                                    expected_stance_duration = 0.3  # Fallback for initial steps
                                
                                if expected_stance_duration > 0:
                                    timing_error = abs(current_stance_duration - expected_stance_duration) / expected_stance_duration
                                    timing_accuracy = max(0.0, 1.0 - timing_error)
                                    contact_timing_accuracy_values.append(timing_accuracy)
                     
                    contact_timing_accuracy = float(np.mean(contact_timing_accuracy_values)) if contact_timing_accuracy_values else 1.0
                    self.metrics['gait_quality']['contact_timing_accuracy'].append(contact_timing_accuracy)
    
    def _get_phase_durations(self, contact_sequence):
        """Extract stance and swing phase durations from contact sequence."""
        stance_durations = []
        swing_durations = []
        
        current_phase = contact_sequence[0]
        phase_start = 0
        
        for i in range(1, len(contact_sequence)):
            if contact_sequence[i] != current_phase:
                duration = i - phase_start
                if current_phase:  # Was in stance
                    stance_durations.append(duration)
                else:  # Was in swing
                    swing_durations.append(duration)
                
                current_phase = contact_sequence[i]
                phase_start = i
        
        return {'stance': stance_durations, 'swing': swing_durations}
    
    def _update_foot_contact(self, robot_state, contact_forces):
        """Update foot contact and sliding metrics with proper foot slip calculation."""
        if contact_forces is None or len(contact_forces) == 0 or self.foot_body_ids is None:
            return
            
        # ✅ FIX: Use foot body velocities for proper slip calculation
        if hasattr(self.env.unwrapped, 'scene') and hasattr(self.env.unwrapped.scene, 'robot'):
            robot = self.env.unwrapped.scene.robot
            
            # Get foot body velocities using resolved indices
            if hasattr(robot.data, 'body_lin_vel_w') and len(self.foot_body_ids) >= 2:
                foot_velocities = robot.data.body_lin_vel_w[:, self.foot_body_ids, :]  # Shape: (num_envs, num_feet, 3)
                
                # Contact forces for feet only
                contact_forces_np = contact_forces if isinstance(contact_forces, np.ndarray) else contact_forces
                if contact_forces_np.ndim >= 3:
                    foot_forces = contact_forces_np[:, self.foot_body_ids, :]  # Shape: (num_envs, num_feet, 3)
                    foot_contacts = np.linalg.norm(foot_forces, axis=2) > self.contact_threshold  # Shape: (num_envs, num_feet)
                    
                    # ✅ FIX: Proper foot slip - horizontal velocity during contact
                    for env_idx in range(self.num_envs):
                        for foot_idx in range(len(self.foot_body_ids)):
                            if foot_contacts[env_idx, foot_idx]:  # Foot is in contact
                                foot_vel_horizontal = foot_velocities[env_idx, foot_idx, :2].cpu().numpy()
                                slip_velocity = float(np.linalg.norm(foot_vel_horizontal))
                                self.metrics['foot_contact']['foot_slip_velocity'].append(slip_velocity)
                    
                    # ✅ FIX: Ground reaction forces - foot bodies only
                    foot_grf = np.sum(np.linalg.norm(foot_forces, axis=2), axis=1)
                    self.metrics['foot_contact']['ground_reaction_forces'].extend(foot_grf.tolist())
                    
                    # Contact force magnitude
                    total_foot_forces = np.mean(np.linalg.norm(foot_forces, axis=2), axis=1)
                    self.metrics['foot_contact']['contact_force_magnitude'].extend(total_foot_forces.tolist())
                    
                    # ✅ FIX: Contact stability - time-window approach per environment
                    for env_idx in range(self.num_envs):
                        env_force_magnitude = np.mean(np.linalg.norm(foot_forces[env_idx], axis=1))
                        self.env_contact_force_history[env_idx].append(env_force_magnitude)
                        
                        if len(self.env_contact_force_history[env_idx]) > 10:
                            # Compute stability as inverse of coefficient of variation over time window
                            force_history = list(self.env_contact_force_history[env_idx])
                            force_mean = np.mean(force_history)
                            force_std = np.std(force_history)
                            if force_mean > 1e-6:  # Avoid division by zero
                                cv = force_std / force_mean  # Coefficient of variation
                                stability = 1.0 / (1.0 + cv)  # Higher stability = lower variation
                            else:
                                stability = 0.0  # No forces = no stability
                            self.metrics['foot_contact']['contact_stability'].append(stability)
    
    def _update_stability_posture(self, robot_state):
        """Update stability and posture metrics."""
        base_pos = robot_state.get('base_pos', np.zeros((self.num_envs, 3)))
        base_quat = robot_state.get('base_quat', np.zeros((self.num_envs, 4)))
        base_ang_vel = robot_state.get('base_ang_vel', np.zeros((self.num_envs, 3)))
        
        # Base height deviation from nominal
        height_deviation = np.abs(base_pos[:, 2] - self.nominal_height)
        self.metrics['stability_posture']['base_height_deviation'].extend(height_deviation.tolist())
        
        # Angular velocity magnitude (stability indicator)
        ang_vel_mag = np.linalg.norm(base_ang_vel, axis=1)
        self.metrics['stability_posture']['angular_velocity_magnitude'].extend(ang_vel_mag.tolist())
        
        # Orientation error from upright
        if 'projected_gravity' in robot_state:
            proj_grav = robot_state['projected_gravity']
            upright_error = np.linalg.norm(proj_grav[:, :2], axis=1)  # xy components should be 0 when upright
            self.metrics['stability_posture']['orientation_error'].extend(upright_error.tolist())
        
        # ✅ FIX: Per-environment postural sway
        for env_idx in range(self.num_envs):
            self.env_height_history[env_idx].append(height_deviation[env_idx])
            
            if len(self.env_height_history[env_idx]) > 10:
                sway = float(np.std(list(self.env_height_history[env_idx])))
                self.metrics['stability_posture']['postural_sway'].append(sway)
        
        # Trunk inclination (roll/pitch from gravity)
        if 'projected_gravity' in robot_state:
            proj_grav = robot_state['projected_gravity']
            trunk_inclination = np.linalg.norm(proj_grav[:, :2], axis=1)
            self.metrics['stability_posture']['trunk_inclination'].extend(trunk_inclination.tolist())
        
        # COM stability approximation (based on angular velocity)
        com_stability = 1.0 / (1.0 + ang_vel_mag)
        self.metrics['stability_posture']['com_stability'].extend(com_stability.tolist())
    
    def _update_velocity_stability(self, robot_state):
        """Update velocity and orientation stability indices with per-environment tracking."""
        base_lin_vel = robot_state.get('base_lin_vel', np.zeros((self.num_envs, 3)))
        
        # ✅ FIX: Per-environment acceleration and jerk computation
        if 'base_lin_vel' in self.previous_states:
            prev_vel = self.previous_states['base_lin_vel']
            acceleration = base_lin_vel - prev_vel
            accel_mag = np.linalg.norm(acceleration, axis=1)
            self.metrics['velocity_stability']['acceleration_magnitude'].extend(accel_mag.tolist())
            
            # Per-environment jerk computation
            if hasattr(self, '_prev_acceleration'):
                jerk = np.linalg.norm(acceleration - self._prev_acceleration, axis=1)
                self.metrics['velocity_stability']['jerk_index'].extend(jerk.tolist())
            
            self._prev_acceleration = acceleration.copy()
        
        # ✅ FIX: Velocity smoothness - measure acceleration consistency per environment
        for env_idx in range(self.num_envs):
            speed = np.linalg.norm(base_lin_vel[env_idx, :2])
            self.env_velocity_history[env_idx].append(speed)
            
            # Compute smoothness as inverse of speed variation (lower variation = smoother)
            if len(self.env_velocity_history[env_idx]) > 5:
                speeds = list(self.env_velocity_history[env_idx])
                speed_changes = np.abs(np.diff(speeds))  # Absolute speed changes
                if len(speed_changes) > 0:
                    smoothness = 1.0 / (1.0 + np.mean(speed_changes))  # Higher value = smoother
                    self.metrics['velocity_stability']['velocity_smoothness'].append(smoothness)
            
            # Speed consistency within environment
            if len(self.env_velocity_history[env_idx]) > 5:
                speeds = list(self.env_velocity_history[env_idx])
                speed_consistency = 1.0 / (1.0 + np.std(speeds))
                self.metrics['velocity_stability']['speed_consistency'].append(speed_consistency)
        
        # Direction changes (angular velocity magnitude)
        base_ang_vel = robot_state.get('base_ang_vel', np.zeros((self.num_envs, 3)))
        direction_changes = np.abs(base_ang_vel[:, 2])  # Yaw rate magnitude
        self.metrics['velocity_stability']['direction_changes'].extend(direction_changes.tolist())
        
        # Turning smoothness (inverse of angular acceleration)
        if 'base_ang_vel' in self.previous_states:
            prev_ang_vel = self.previous_states['base_ang_vel']
            ang_accel = np.abs(base_ang_vel[:, 2] - prev_ang_vel[:, 2])
            turning_smoothness = 1.0 / (1.0 + ang_accel)
            self.metrics['velocity_stability']['turning_smoothness'].extend(turning_smoothness.tolist())
    
    def _update_command_adherence(self, robot_state, commands):
        """Update command adherence metrics."""
        if commands is None or len(commands) == 0:
            return
            
        base_lin_vel = robot_state.get('base_lin_vel', np.zeros((self.num_envs, 3)))
        base_quat_w = robot_state.get('base_quat', np.zeros((self.num_envs, 4)))
        
        # Convert to torch for coordinate transformation
        base_lin_vel_torch = torch.tensor(base_lin_vel, device=self.env.unwrapped.device)
        base_quat_w_torch = torch.tensor(base_quat_w, device=self.env.unwrapped.device)
        yaw_quat = self._get_yaw_quaternion(base_quat_w_torch)
        base_lin_vel_yaw = math_utils.quat_apply_inverse(yaw_quat, base_lin_vel_torch[:, :3])
        
        # Command following precision in correct frame
        if commands.shape[-1] >= 2:
            cmd_vel = commands[:, :2] if commands.ndim > 1 else commands[:2]
            actual_vel = base_lin_vel_yaw[:, :2].cpu().numpy()
            
            precision = 1.0 - np.linalg.norm(actual_vel - cmd_vel, axis=1) / (np.linalg.norm(cmd_vel, axis=1) + 0.1)
            self.metrics['command_adherence']['command_following_precision'].extend(precision.tolist())
            
            # Steady state error
            steady_error = np.linalg.norm(actual_vel - cmd_vel, axis=1)
            self.metrics['command_adherence']['steady_state_error'].extend(steady_error.tolist())
            
            # Command response and overshoot detection
            for env_idx in range(self.num_envs):
                if len(self.command_history[env_idx]) > 1:
                    # Detect command changes
                    prev_cmd = list(self.command_history[env_idx])[-2] if len(self.command_history[env_idx]) > 1 else commands[env_idx, 2]
                    current_cmd = commands[env_idx, 2]
                    
                    if abs(current_cmd - prev_cmd) > 0.1:  # Significant command change
                        # Measure overshoot
                        actual_yaw_rate = robot_state.get('base_ang_vel', np.zeros((self.num_envs, 3)))[env_idx, 2]
                        overshoot = max(0.0, abs(actual_yaw_rate) - abs(current_cmd))
                        self.metrics['command_adherence']['overshoot_magnitude'].append(overshoot)
                        
                        # Response delay (simplified as current error magnitude)
                        response_delay = abs(actual_yaw_rate - current_cmd)
                        self.metrics['command_adherence']['response_delay'].append(response_delay)
                        
                        # Response time (steps to achieve 90% of command)
                        if abs(current_cmd) > 0.05:
                            target_response = 0.9 * abs(current_cmd)
                            if abs(actual_yaw_rate) >= target_response:
                                self.metrics['command_adherence']['command_response_time'].append(1.0)  # Immediate response
                            else:
                                self.metrics['command_adherence']['command_response_time'].append(0.1)  # Delayed response
        
        # Update command history for future comparisons
        for env_idx in range(self.num_envs):
            if commands.ndim > 1:
                self.command_history[env_idx].append(commands[env_idx, 2])  # Yaw command
            else:
                self.command_history[env_idx].append(commands[2])  # Single command case
    
    def _update_evaluation_coverage(self, robot_state, terminated, truncated):
        """Update evaluation coverage and sanity metrics with per-environment tracking."""
        base_pos = robot_state.get('base_pos', np.zeros((self.num_envs, 3)))
        base_lin_vel = robot_state.get('base_lin_vel', np.zeros((self.num_envs, 3)))
        base_ang_vel = robot_state.get('base_ang_vel', np.zeros((self.num_envs, 3)))
        
        # ✅ FIX: Per-environment workspace exploration
        for env_idx in range(self.num_envs):
            min_x, max_x, min_y, max_y = self.env_workspace_min_max[env_idx]
            
            # Update workspace bounds
            pos_x, pos_y = base_pos[env_idx, 0], base_pos[env_idx, 1]
            min_x = min(min_x, pos_x)
            max_x = max(max_x, pos_x)
            min_y = min(min_y, pos_y)
            max_y = max(max_y, pos_y)
            
            self.env_workspace_min_max[env_idx] = (min_x, max_x, min_y, max_y)
            
            # Compute workspace coverage area
            if max_x > min_x and max_y > min_y:
                coverage_area = (max_x - min_x) * (max_y - min_y)
                self.metrics['evaluation_coverage']['workspace_exploration'].append(coverage_area)
        
        # Velocity range coverage
        speed_range = np.ptp(np.linalg.norm(base_lin_vel[:, :2], axis=1))
        self.metrics['evaluation_coverage']['velocity_range_coverage'].append(speed_range)
        
        # Turning range coverage
        yaw_rate_range = np.ptp(base_ang_vel[:, 2])
        self.metrics['evaluation_coverage']['turning_range_coverage'].append(yaw_rate_range)
        
        # Termination tracking
        if isinstance(terminated, torch.Tensor):
            term_np = terminated.cpu().numpy()
        else:
            term_np = terminated
            
        if isinstance(truncated, torch.Tensor):
            trunc_np = truncated.cpu().numpy()
        else:
            trunc_np = truncated
        
        # Count termination reasons
        if np.any(term_np):
            self.metrics['evaluation_coverage']['termination_reasons']['terminated'] += int(np.sum(term_np))
        if np.any(trunc_np):
            self.metrics['evaluation_coverage']['termination_reasons']['truncated'] += int(np.sum(trunc_np))
    
    def _update_disturbance_robustness(self, robot_state, contact_forces):
        """Update external disturbance robustness metrics with per-environment tracking."""
        base_ang_vel = robot_state.get('base_ang_vel', np.zeros((self.num_envs, 3)))
        
        # Stability margin (inverse of angular velocity magnitude)
        ang_vel_mag = np.linalg.norm(base_ang_vel, axis=1)
        stability_margin = 1.0 / (1.0 + ang_vel_mag)
        self.metrics['disturbance_robustness']['stability_margin'].extend(stability_margin.tolist())
        
        # ✅ FIX: Per-environment disturbance tracking
        for env_idx in range(self.num_envs):
            env_stability = stability_margin[env_idx]
            self.env_stability_history[env_idx].append(env_stability)
            
            # Recovery time estimation per environment
            if len(self.env_stability_history[env_idx]) > 15:
                recent_stability = list(self.env_stability_history[env_idx])[-15:]  # Last 15 steps
                stable_threshold = 0.8
                
                # Find periods of instability and recovery
                unstable_indices = [i for i, s in enumerate(recent_stability) if s < stable_threshold]
                if len(unstable_indices) > 0:
                    last_unstable = unstable_indices[-1]
                    recovery_time = len(recent_stability) - last_unstable - 1  # Steps since last instability
                    # Convert to time units
                    recovery_time_seconds = recovery_time * self.dt
                    self.metrics['disturbance_robustness']['recovery_time'].append(recovery_time_seconds)
            
            # Adaptive response per environment
            if len(self.env_stability_history[env_idx]) > 5:
                recent_margins = list(self.env_stability_history[env_idx])[-5:]
                # Measure ability to maintain or improve stability
                stability_trend = recent_margins[-1] - recent_margins[0]  # Positive = improving
                adaptive_response = max(0.0, stability_trend)  # Only count improvements
                self.metrics['disturbance_robustness']['adaptive_response'].append(adaptive_response)
        
        # Perturbation resistance (based on force variance and stability)
        if contact_forces is not None and len(contact_forces) > 0:
            force_magnitude = np.mean(np.linalg.norm(contact_forces.reshape(self.num_envs, -1, 3), axis=2), axis=1)
            resistance = stability_margin * (1.0 / (1.0 + force_magnitude))
            self.metrics['disturbance_robustness']['perturbation_resistance'].extend(resistance.tolist())
        
        # Fall prevention per environment (high stability margin maintenance)
        for env_idx in range(self.num_envs):
            if len(self.env_stability_history[env_idx]) > 10:
                recent_stability = list(self.env_stability_history[env_idx])[-10:]
                high_stability_fraction = np.mean([s > 0.7 for s in recent_stability])
                self.metrics['disturbance_robustness']['fall_prevention'].append(high_stability_fraction)
    
    def _update_episode_tracking(self, rewards, terminated, truncated):
        """Update per-environment episode tracking."""
        if isinstance(rewards, torch.Tensor):
            reward_np = rewards.cpu().numpy()
        else:
            reward_np = rewards
            
        if isinstance(terminated, torch.Tensor):
            term_np = terminated.cpu().numpy()
        else:
            term_np = terminated
            
        if isinstance(truncated, torch.Tensor):
            trunc_np = truncated.cpu().numpy()
        else:
            trunc_np = truncated
        
        # Update per-environment rewards and lengths
        for env_idx in range(self.num_envs):
            self.env_episode_rewards[env_idx] += reward_np[env_idx]
            self.env_episode_lengths[env_idx] += 1
            
            # Check for episode completion
            if term_np[env_idx] or trunc_np[env_idx]:
                episode_data = {
                    'reward': self.env_episode_rewards[env_idx],
                    'length': self.env_episode_lengths[env_idx],
                    'env_id': env_idx,
                    'termination_type': 'terminated' if term_np[env_idx] else 'truncated'
                }
                self.completed_episodes.append(episode_data)
                
                # Reset for this environment
                self.env_episode_rewards[env_idx] = 0.0
                self.env_episode_lengths[env_idx] = 0
        
        # Episode completion rate
        if len(self.completed_episodes) > 0:
            total_episodes = len(self.completed_episodes)
            completed_episodes = sum(1 for ep in self.completed_episodes if ep['termination_type'] == 'truncated')
            completion_rate = completed_episodes / total_episodes
            self.metrics['evaluation_coverage']['episode_completion_rate'].append(completion_rate)
    
    def get_summary_statistics(self):
        """Compute summary statistics for all collected metrics."""
        summary = {}
        
        for category, metrics in self.metrics.items():
            summary[category] = {}
            for metric_name, values in metrics.items():
                if isinstance(values, list) and len(values) > 0:
                    # Convert to numpy array for statistics
                    arr = np.array(values)
                    if arr.size > 0:
                        summary[category][metric_name] = {
                            'count': int(len(values)),
                            'mean': float(np.mean(arr)),
                            'std': float(np.std(arr)),
                            'min': float(np.min(arr)),
                            'max': float(np.max(arr)),
                            'median': float(np.median(arr)),
                            'q25': float(np.percentile(arr, 25)),
                            'q75': float(np.percentile(arr, 75)),
                        }
                elif isinstance(values, defaultdict):
                    # For termination reasons - ensure all values are JSON serializable
                    summary[category][metric_name] = {str(k): int(v) if isinstance(v, (np.integer, np.int64)) else v for k, v in dict(values).items()}
        
        return summary

def collect_real_data(args):
    """Collect real performance data from a trained policy."""
    
    print("🎯 Starting Real Policy Data Collection")
    print("=" * 50)
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"🎮 Task: {args.task}")
    print(f"🔢 Environments: {args.num_envs}")
    print(f"⏱️ Steps: {args.steps}")
    print(f"🌍 Foundation-only mode: {args.foundation_only}")
    
    # Set sensor configuration
    # set_sensor_mode(args.foundation_only) # Disabled for manual control
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args.task, 
        device=args.device, 
        num_envs=args.num_envs, 
        use_fabric=False
    )
    
    # Force headless for data collection
    env_cfg.sim.headless = True
    
    # Disable randomization for consistent evaluation
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False
    
    # Parse agent configuration
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    
    # Create environment
    env = gym.make(args.task, cfg=env_cfg)
    print(f"✅ Environment created with {args.num_envs} parallel environments")
    
    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    print(f"🔄 Loading model checkpoint from: {args.checkpoint}")
    
    # Load trained policy
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    ppo_runner.load(args.checkpoint)
    
    # Get inference policy
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    print("✅ Policy loaded successfully")
    
    # Initialize comprehensive metrics collector
    metrics_collector = ComprehensiveMetricsCollector(env, args.num_envs)
    
    # Initialize data collection
    collected_data = {
        'metadata': {
            'task': args.task,
            'checkpoint': args.checkpoint,
            'foundation_only': args.foundation_only,
            'sensors_enabled': not args.foundation_only,
            'num_envs': args.num_envs,
            'collection_steps': args.steps,
            'timestamp': datetime.now().isoformat(),
            'observation_space': str(env.observation_space),
            'action_space': str(env.action_space),
        },
        'episodes': [],
        'step_data': {
            'observations': [],
            'actions': [],
            'rewards': [],
            'robot_states': [],
            'commands': [],
            'contact_forces': [],
            'terminations': [],
        },
        'comprehensive_metrics': {},  # Will be filled by metrics_collector
        'summary_metrics': {},
    }
    
    print(f"📊 Observation space: {env.observation_space}")
    print(f"🎮 Action space: {env.action_space}")
    
    # Reset environment
    obs, info = env.get_observations()
    
    print(f"🚀 Starting data collection for {args.steps} steps...")
    
    step_count = 0
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0.0
    current_episode_length = 0
    
    # Data collection loop
    with torch.inference_mode():
        while step_count < args.steps:
            # Store observations
            if isinstance(obs, torch.Tensor):
                collected_data['step_data']['observations'].append(obs.cpu().numpy())
            else:
                collected_data['step_data']['observations'].append(obs)
            
            # Get robot state data
            robot_state = {}
            if hasattr(env.unwrapped, 'scene') and hasattr(env.unwrapped.scene, 'robot'):
                robot = env.unwrapped.scene.robot
                robot_state = {
                    'base_pos': robot.data.root_pos_w.cpu().numpy(),
                    'base_quat': robot.data.root_quat_w.cpu().numpy(),
                    'base_lin_vel': robot.data.root_lin_vel_w.cpu().numpy(),
                    'base_ang_vel': robot.data.root_ang_vel_w.cpu().numpy(),
                    'joint_pos': robot.data.joint_pos.cpu().numpy(),
                    'joint_vel': robot.data.joint_vel.cpu().numpy(),
                }
                
                # Add projected gravity if available
                if hasattr(robot.data, 'projected_gravity_b'):
                    robot_state['projected_gravity'] = robot.data.projected_gravity_b.cpu().numpy()
                
                collected_data['step_data']['robot_states'].append(robot_state)
            
            # Get contact force data if available
            contact_forces = None
            if hasattr(env.unwrapped, 'scene') and hasattr(env.unwrapped.scene, 'contact_forces'):
                contact_sensor = env.unwrapped.scene.contact_forces
                if hasattr(contact_sensor, 'data') and hasattr(contact_sensor.data, 'net_forces_w'):
                    contact_forces = contact_sensor.data.net_forces_w.cpu().numpy()
                    collected_data['step_data']['contact_forces'].append(contact_forces)
            
            # Get command data
            commands = None
            if hasattr(env.unwrapped, 'command_manager') and hasattr(env.unwrapped.command_manager, 'get_command'):
                try:
                    commands = env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy()
                    collected_data['step_data']['commands'].append(commands)
                except:
                    # Skip command data if not accessible
                    pass
            elif hasattr(env.unwrapped, 'command_manager') and hasattr(env.unwrapped.command_manager, 'data'):
                try:
                    commands = env.unwrapped.command_manager.data.cpu().numpy()
                    collected_data['step_data']['commands'].append(commands)
                except:
                    # Skip command data if not accessible
                    pass
            
            # Policy inference
            actions = policy(obs)
            if isinstance(actions, torch.Tensor):
                collected_data['step_data']['actions'].append(actions.cpu().numpy())
            else:
                collected_data['step_data']['actions'].append(actions)
            
            # Environment step
            step_result = env.step(actions)
            if len(step_result) == 5:
                obs, rewards, terminated, truncated, info = step_result
            else:
                obs, rewards, dones, info = step_result
                terminated = dones
                truncated = dones
            
            # Store rewards
            if isinstance(rewards, torch.Tensor):
                reward_np = rewards.cpu().numpy()
            else:
                reward_np = rewards
            collected_data['step_data']['rewards'].append(reward_np)
            
            # Store termination info
            term_info = {}
            if isinstance(terminated, torch.Tensor):
                term_info['terminated'] = terminated.cpu().numpy()
            else:
                term_info['terminated'] = terminated
                
            if isinstance(truncated, torch.Tensor):
                term_info['truncated'] = truncated.cpu().numpy()
            else:
                term_info['truncated'] = truncated
                
            collected_data['step_data']['terminations'].append(term_info)
            
            # Update comprehensive metrics
            metrics_collector.update_metrics(
                obs, actions, reward_np, robot_state, commands, 
                contact_forces, terminated, truncated
            )
            
            # Track episodes
            current_episode_reward += np.mean(reward_np)
            current_episode_length += 1
            
            # Check for episode termination
            if isinstance(terminated, torch.Tensor):
                any_terminated = terminated.any().item()
            else:
                any_terminated = np.any(terminated)
                
            if isinstance(truncated, torch.Tensor):
                any_truncated = truncated.any().item()
            else:
                any_truncated = np.any(truncated)
            
            if any_terminated or any_truncated:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                
                episode_data = {
                    'episode_reward': current_episode_reward,
                    'episode_length': current_episode_length,
                    'end_step': step_count,
                    'termination_reason': 'terminated' if any_terminated else 'truncated'
                }
                collected_data['episodes'].append(episode_data)
                
                print(f"  📈 Episode completed at step {step_count}: reward={current_episode_reward:.3f}, length={current_episode_length}")
                
                current_episode_reward = 0.0
                current_episode_length = 0
            
            step_count += 1
            
            # Progress update
            if step_count % 100 == 0:
                print(f"  📊 Step {step_count}/{args.steps} ({step_count/args.steps*100:.1f}%)")
    
    # Compute comprehensive metrics summary
    print("\n🧮 Computing comprehensive metrics...")
    collected_data['comprehensive_metrics'] = metrics_collector.get_summary_statistics()
    
    # Compute basic summary metrics
    if episode_rewards:
        collected_data['summary_metrics'] = {
            'mean_episode_reward': float(np.mean(episode_rewards)),
            'std_episode_reward': float(np.std(episode_rewards)),
            'min_episode_reward': float(np.min(episode_rewards)),
            'max_episode_reward': float(np.max(episode_rewards)),
            'median_episode_reward': float(np.median(episode_rewards)),
            
            'mean_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths)),
            'min_episode_length': float(np.min(episode_lengths)),
            'max_episode_length': float(np.max(episode_lengths)),
            'median_episode_length': float(np.median(episode_lengths)),
            
            'total_episodes': len(episode_rewards),
            'total_steps_collected': step_count,
        }
        
        # Compute per-step metrics
        all_rewards = np.array(collected_data['step_data']['rewards'])
        collected_data['summary_metrics'].update({
            'mean_step_reward': float(np.mean(all_rewards)),
            'std_step_reward': float(np.std(all_rewards)),
            'total_reward': float(np.sum(all_rewards)),
        })
        
        print(f"\n✅ Data collection completed!")
        print(f"📊 Episodes completed: {len(episode_rewards)}")
        print(f"📈 Mean episode reward: {collected_data['summary_metrics']['mean_episode_reward']:.3f}")
        print(f"📏 Mean episode length: {collected_data['summary_metrics']['mean_episode_length']:.1f}")
        print(f"⭐ Mean step reward: {collected_data['summary_metrics']['mean_step_reward']:.3f}")
        
        # Print some key comprehensive metrics
        comp_metrics = collected_data['comprehensive_metrics']
        if 'tracking_performance' in comp_metrics and 'velocity_error_xy' in comp_metrics['tracking_performance']:
            vel_error = comp_metrics['tracking_performance']['velocity_error_xy']['mean']
            print(f"🎯 Mean velocity tracking error: {vel_error:.3f} m/s")
        
        if 'gait_quality' in comp_metrics and 'step_frequency' in comp_metrics['gait_quality']:
            step_freq = comp_metrics['gait_quality']['step_frequency']['mean']
            print(f"👣 Mean step frequency: {step_freq:.2f} Hz")
        
        if 'stability_posture' in comp_metrics and 'base_height_deviation' in comp_metrics['stability_posture']:
            height_dev = comp_metrics['stability_posture']['base_height_deviation']['mean']
            print(f"📐 Mean height deviation: {height_dev:.3f} m")
    
    # Save data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(collected_data, f)
    
    # Also save summary as JSON
    json_path = output_path.with_suffix('.json')
    summary_data = {
        'metadata': collected_data['metadata'],
        'summary_metrics': collected_data['summary_metrics'],
        'comprehensive_metrics': collected_data['comprehensive_metrics'],
        'episodes_summary': collected_data['episodes'],
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"💾 Data saved to: {output_path}")
    print(f"📋 Summary saved to: {json_path}")
    
    # Clean up
    env.close()
    
    return collected_data

def main():
    """Main function."""
    try:
        data = collect_real_data(args_cli)
        print(f"\n🎉 Real policy data collection successful!")
        print("📊 Comprehensive metrics collected:")
        print("   • Tracking Performance")
        print("   • Gait and Cadence Quality") 
        print("   • Foot Contact and Sliding")
        print("   • Stability and Posture")
        print("   • Velocity/Orientation Stability Indices")
        print("   • Command Adherence")
        print("   • Evaluation Coverage/Sanity")
        print("   • External Disturbance Robustness")
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the simulation
        simulation_app.close()

if __name__ == "__main__":
    main() 