# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect policy performance data with proper Isaac Lab format.

This script loads trained RL policies and collects performance metrics
for comparative analysis between different configurations.

.. code-block:: bash

    # Basic data collection
    ./isaaclab.sh -p SDS_ANONYM/collect_policy_data_clean.py \
        --task Isaac-SDS-Velocity-Flat-G1-Enhanced-v0 \
        --checkpoint logs/rsl_rl/g1_enhanced/2025-08-14_00-40-58/model_700.pt \
        --num_envs 50 --steps 1000 --output policy_data.pkl

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
parser = argparse.ArgumentParser(description="Policy Performance Data Collection")
parser.add_argument("--task", type=str, default="Isaac-SDS-Velocity-Flat-G1-Enhanced-v0", help="Isaac Lab task")
parser.add_argument("--num_envs", type=int, default=50, help="Number of environments")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to collect")
parser.add_argument("--output", type=str, required=True, help="Output file path (.pkl)")
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
import isaaclab.utils.math as math_utils

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


class SimpleMetricsCollector:
    """Collects basic performance metrics using proper Isaac Lab data access."""
    
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
        
        # Get timestep from environment
        self.dt = self._get_environment_timestep()
        print(f"📊 Environment timestep: {self.dt:.4f}s ({1/self.dt:.1f} Hz)")
        
        # Initialize robot reference
        self.robot = self._get_robot_reference()
        
        # Initialize metric storage
        self.metrics = {
            'tracking_performance': {
                'velocity_error': [],
                'command_tracking_accuracy': [],
            },
            'stability_metrics': {
                'base_height_deviation': [],
                'angular_velocity_magnitude': [],
            },
            'overall_performance': {
                'episode_rewards': [],
                'episode_lengths': [],
                'termination_rates': defaultdict(int),
            }
        }
        
        # Episode tracking
        self.step_counter = 0
        self.env_episode_data = {'rewards': [0.0] * num_envs, 'lengths': [0] * num_envs}
        
        print(f"✅ Simple metrics collector initialized for {num_envs} environments")
    
    def _get_environment_timestep(self):
        """Get timestep from environment."""
        try:
            if hasattr(self.env.unwrapped, 'step_dt'):
                return self.env.unwrapped.step_dt
            elif hasattr(self.env.unwrapped, 'cfg'):
                sim_dt = self.env.unwrapped.cfg.sim.dt
                decimation = self.env.unwrapped.cfg.decimation
                return sim_dt * decimation
            else:
                return 0.02  # 50Hz fallback
        except:
            return 0.02
    
    def _get_robot_reference(self):
        """Get robot reference from environment."""
        try:
            if hasattr(self.env.unwrapped, 'scene'):
                scene = self.env.unwrapped.scene
                if hasattr(scene, 'robot'):
                    print(f"✅ Robot found: {scene.robot.__class__.__name__}")
                    return scene.robot
                elif hasattr(scene, 'articulations') and 'robot' in scene.articulations:
                    print(f"✅ Robot found in articulations: {scene.articulations['robot'].__class__.__name__}")
                    return scene.articulations['robot']
            print("⚠️ Robot not found - some metrics will be limited")
            return None
        except Exception as e:
            print(f"⚠️ Error getting robot reference: {e}")
            return None
    
    def update_metrics(self, obs, actions, rewards, terminated, truncated):
        """Update metrics with current step data."""
        self.step_counter += 1
        
        # Get robot state if available
        robot_state = self._get_robot_state()
        
        # Get commands if available
        commands = self._get_commands()
        
        # Update metrics
        if robot_state is not None:
            self._update_tracking_performance(robot_state, commands)
            self._update_stability_metrics(robot_state)
        
        self._update_episode_tracking(rewards, terminated, truncated)
    
    def _get_robot_state(self):
        """Get basic robot state data."""
        if self.robot is None:
            return None
        
        try:
            return {
                'base_pos': self.robot.data.root_pos_w.cpu().numpy(),
                'base_quat': self.robot.data.root_quat_w.cpu().numpy(),
                'base_lin_vel': self.robot.data.root_lin_vel_w.cpu().numpy(),
                'base_ang_vel': self.robot.data.root_ang_vel_w.cpu().numpy(),
            }
        except Exception as e:
            print(f"⚠️ Error getting robot state: {e}")
            return None
    
    def _get_commands(self):
        """Get command data."""
        try:
            if hasattr(self.env.unwrapped, 'command_manager'):
                cmd_manager = self.env.unwrapped.command_manager
                if hasattr(cmd_manager, 'get_command'):
                    return cmd_manager.get_command("base_velocity").cpu().numpy()
                elif hasattr(cmd_manager, 'data'):
                    return cmd_manager.data.cpu().numpy()
            return None
        except:
            return None
    
    def _update_tracking_performance(self, robot_state, commands):
        """Update velocity tracking metrics."""
        if commands is None or len(commands) == 0:
            return
        
        base_lin_vel = robot_state['base_lin_vel']
        base_quat = robot_state['base_quat']
        
        # Simple velocity error calculation
        if commands.shape[-1] >= 2:
            cmd_vel_x = commands[:, 0] if commands.ndim > 1 else commands[0]
            cmd_vel_y = commands[:, 1] if commands.ndim > 1 else commands[1]
            
            # Velocity error in world frame (simplified)
            vel_error = np.sqrt((base_lin_vel[:, 0] - cmd_vel_x)**2 + 
                              (base_lin_vel[:, 1] - cmd_vel_y)**2)
            self.metrics['tracking_performance']['velocity_error'].extend(vel_error.tolist())
            
            # Command tracking accuracy
            max_vel = np.maximum(np.linalg.norm(commands[:, :2], axis=1), 0.1)
            accuracy = 1.0 - (vel_error / max_vel)
            self.metrics['tracking_performance']['command_tracking_accuracy'].extend(accuracy.tolist())
    
    def _update_stability_metrics(self, robot_state):
        """Update basic stability metrics."""
        base_pos = robot_state['base_pos']
        base_ang_vel = robot_state['base_ang_vel']
        
        # Base height deviation (assuming 0.74m nominal height for G1)
        nominal_height = 0.74
        height_deviation = np.abs(base_pos[:, 2] - nominal_height)
        self.metrics['stability_metrics']['base_height_deviation'].extend(height_deviation.tolist())
        
        # Angular velocity magnitude
        ang_vel_mag = np.linalg.norm(base_ang_vel, axis=1)
        self.metrics['stability_metrics']['angular_velocity_magnitude'].extend(ang_vel_mag.tolist())
    
    def _update_episode_tracking(self, rewards, terminated, truncated):
        """Update episode performance tracking."""
        # Convert to numpy if needed
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()
        
        # Update per-environment episode data
        for env_idx in range(self.num_envs):
            self.env_episode_data['rewards'][env_idx] += rewards[env_idx]
            self.env_episode_data['lengths'][env_idx] += 1
            
            # Check for episode completion
            if terminated[env_idx] or truncated[env_idx]:
                self.metrics['overall_performance']['episode_rewards'].append(
                    self.env_episode_data['rewards'][env_idx]
                )
                self.metrics['overall_performance']['episode_lengths'].append(
                    self.env_episode_data['lengths'][env_idx]
                )
                
                # Track termination reason
                if terminated[env_idx]:
                    self.metrics['overall_performance']['termination_rates']['terminated'] += 1
                else:
                    self.metrics['overall_performance']['termination_rates']['truncated'] += 1
                
                # Reset for this environment
                self.env_episode_data['rewards'][env_idx] = 0.0
                self.env_episode_data['lengths'][env_idx] = 0
    
    def get_summary_statistics(self):
        """Compute summary statistics for all collected metrics."""
        summary = {}
        
        for category, metrics in self.metrics.items():
            summary[category] = {}
            for metric_name, values in metrics.items():
                if isinstance(values, list) and len(values) > 0:
                    arr = np.array(values)
                    if arr.size > 0:
                        summary[category][metric_name] = {
                            'count': len(values),
                            'mean': float(np.mean(arr)),
                            'std': float(np.std(arr)),
                            'min': float(np.min(arr)),
                            'max': float(np.max(arr)),
                            'median': float(np.median(arr)),
                        }
                elif isinstance(values, defaultdict):
                    summary[category][metric_name] = dict(values)
        
        return summary


def main():
    """Main function to collect policy performance data."""
    
    print("🎯 Policy Performance Data Collection")
    print("=" * 50)
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
    
    # initialize metrics collector
    metrics_collector = SimpleMetricsCollector(env, args_cli.num_envs)
    
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
    
    print(f"🚀 Starting data collection for {args_cli.steps} steps...")
    
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
            
            # update metrics
            metrics_collector.update_metrics(obs, actions, rewards, terminated, truncated)
            
            step_count += 1
            
            # progress update
            if step_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  📊 Step {step_count}/{args_cli.steps} ({step_count/args_cli.steps*100:.1f}%) - {elapsed:.1f}s")
    
    # compute final metrics
    print("🧮 Computing final metrics...")
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
    
    # Print some key metrics if available
    if 'tracking_performance' in collected_data['metrics']:
        tp = collected_data['metrics']['tracking_performance']
        if 'velocity_error' in tp:
            print(f"🎯 Mean velocity error: {tp['velocity_error']['mean']:.3f} m/s")
        if 'command_tracking_accuracy' in tp:
            print(f"📍 Mean tracking accuracy: {tp['command_tracking_accuracy']['mean']:.3f}")
    
    if 'stability_metrics' in collected_data['metrics']:
        sm = collected_data['metrics']['stability_metrics']
        if 'base_height_deviation' in sm:
            print(f"📐 Mean height deviation: {sm['base_height_deviation']['mean']:.3f} m")
    
    # save data
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(collected_data, f)
    
    print(f"💾 Data saved to: {output_path}")
    
    # close environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 