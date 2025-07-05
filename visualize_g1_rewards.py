#!/usr/bin/env python3
"""
Visualize G1 Robot with Custom Reward Functions in Isaac Lab

This script shows the G1 robot performing tasks with custom reward functions
in the Isaac Lab simulation environment with GUI visualization.

Usage:
    python visualize_g1_rewards.py --task jump_hand_high --num_envs 4
    python visualize_g1_rewards.py --task spin_arms --num_envs 8
    python visualize_g1_rewards.py --task all_rewards --num_envs 1
"""

import argparse
import torch
import time

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Visualize G1 Robot with Custom Reward Functions")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate")
parser.add_argument("--task", type=str, default="jump_hand_high", 
                   choices=["jump_hand_high", "spin_arms", "move_backward", "reach_forward", "dance_pose", "all_rewards"],
                   help="Task to visualize")
parser.add_argument("--use_metamotivo", action="store_true", help="Use Meta Motivo for intelligent actions")
parser.add_argument("--action_scale", type=float, default=0.05, help="Action scaling factor")
parser.add_argument("--print_rewards", action="store_true", help="Print reward values")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Everything else follows
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

# Meta Motivo bridge (optional)
try:
    from metamotivo_g1_bridge import G1MetaMotivoBridge
    METAMOTIVO_AVAILABLE = True
except ImportError:
    METAMOTIVO_AVAILABLE = False

# Task to environment mapping
TASK_ENV_MAP = {
    "jump_hand_high": "Isaac-SDS-Velocity-Flat-G1-v0",  # Corrected environment name with -v0 suffix
    "spin_arms": "Isaac-SDS-Velocity-Flat-G1-v0",
    "move_backward": "Isaac-SDS-Velocity-Flat-G1-v0", 
    "reach_forward": "Isaac-SDS-Velocity-Flat-G1-v0",
    "dance_pose": "Isaac-SDS-Velocity-Flat-G1-v0",
    "all_rewards": "Isaac-SDS-Velocity-Flat-G1-v0",
}

# Task descriptions for Meta Motivo
TASK_DESCRIPTIONS = {
    "jump_hand_high": "jump high while raising left hand above head",
    "spin_arms": "spin quickly while keeping arms close to body", 
    "move_backward": "walk backward quickly",
    "reach_forward": "slowly raise both hands up while standing still and balanced",
    "dance_pose": "gently lift both arms up high while maintaining balance",
    "all_rewards": "perform various dynamic movements and poses",
}


def main():
    """Visualize G1 robot with custom reward functions."""
    
    print("🎮 G1 Robot Custom Reward Visualization")
    print("=" * 50)
    print(f"Task: {args_cli.task}")
    print(f"Environments: {args_cli.num_envs}")
    print(f"Meta Motivo: {args_cli.use_metamotivo and METAMOTIVO_AVAILABLE}")
    print(f"Action Scale: {args_cli.action_scale}")
    
    # Get environment name
    env_name = TASK_ENV_MAP[args_cli.task]
    
    # Create environment configuration
    env_cfg = parse_env_cfg(
        env_name, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=True
    )
    
    # CRITICAL FIXES for Meta Motivo compatibility
    # 1. Disable fall termination to prevent robot from becoming "fixed"
    env_cfg.terminations.base_contact = None
    print("🚫 Disabled fall termination for continuous demo")
    
    # 2. Longer episodes for better observation
    env_cfg.episode_length_s = 60.0
    print(f"⏱️  Episode length: {env_cfg.episode_length_s}s")
    
    # 3. Adjust action scaling for better control
    if hasattr(env_cfg, 'actions') and hasattr(env_cfg.actions, 'joint_pos'):
        env_cfg.actions.joint_pos.scale = 1.0  # Increase from 0.25 to 1.0
        print("🔧 Increased action scaling to 1.0")
    
    # Create environment
    env = gym.make(env_name, cfg=env_cfg)
    
    print(f"✅ Environment created: {env_name}")
    print(f"📊 Observation space: {env.observation_space}")
    print(f"🎮 Action space: {env.action_space}")
    
    # Initialize Meta Motivo bridge if requested
    bridge = None
    if args_cli.use_metamotivo and METAMOTIVO_AVAILABLE:
        try:
            bridge = G1MetaMotivoBridge()
            # Test the bridge
            if bridge.test_bridge():
                print("🤖 Meta Motivo bridge initialized and tested successfully")
            else:
                print("⚠️  Meta Motivo bridge test failed, using random actions")
                bridge = None
        except Exception as e:
            print(f"❌ Meta Motivo initialization failed: {e}")
            print("⚠️  Falling back to random actions")
            bridge = None
    elif args_cli.use_metamotivo:
        print("⚠️  Meta Motivo not available, using random actions")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"🔄 Environment reset complete")
    
    # Start simulation
    print(f"\n🚀 Starting simulation...")
    print(f"🎯 Task: {TASK_DESCRIPTIONS[args_cli.task]}")
    print("Press Ctrl+C to stop simulation\n")
    
    step_count = 0
    total_reward = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
    reset_counter = 0
    
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Generate actions
                if bridge is not None:
                    # Use Meta Motivo for intelligent actions
                    # Extract the policy observation tensor from the dictionary
                    if isinstance(obs, dict):
                        obs_tensor = obs["policy"]
                    else:
                        obs_tensor = obs
                    actions = bridge.act(obs_tensor, TASK_DESCRIPTIONS[args_cli.task], context_method="goal")
                else:
                    # Generate random actions
                    actions = (2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1) * args_cli.action_scale
                
                # Apply actions
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                # Track rewards
                total_reward += rewards
                step_count += 1
                
                # Manual reset every 30 seconds to prevent getting stuck
                if step_count % 1500 == 0:  # ~30 seconds at 50Hz
                    print(f"🔄 Manual reset at step {step_count} to refresh simulation")
                    obs, _ = env.reset()
                    reset_counter += 1
                
                # Print reward information
                if args_cli.print_rewards and step_count % 50 == 0:
                    avg_reward = rewards.mean().item()
                    max_reward = rewards.max().item()
                    success_rate = (rewards > 0.1).float().mean().item() * 100
                    print(f"Step {step_count:4d}: Avg Reward = {avg_reward:6.3f}, Max = {max_reward:6.3f}, Success = {success_rate:5.1f}%")
                
                # Small delay to control simulation speed
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print(f"\n⏹️  Simulation stopped after {step_count} steps")
        
        # Print final statistics
        if step_count > 0:
            final_avg_reward = total_reward.mean().item()
            final_max_reward = total_reward.max().item()
            print(f"\n📊 Final Statistics:")
            print(f"   Average total reward: {final_avg_reward:.3f}")
            print(f"   Maximum total reward: {final_max_reward:.3f}")
            print(f"   Best environment: {total_reward.argmax().item()}")
            print(f"   Total resets: {reset_counter}")
            print(f"   Steps per reset: {step_count // max(1, reset_counter)}")
    
    # Close the environment
    env.close()
    print("✅ Environment closed successfully")


if __name__ == "__main__":
    main()
    # Close sim app
    simulation_app.close() 