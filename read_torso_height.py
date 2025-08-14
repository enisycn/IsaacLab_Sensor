# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to read torso height directly from robot articulation data.

This script demonstrates how to load an RL policy and monitor robot torso height
without using sensors, accessing robot articulation data directly.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p read_torso_height.py --task Isaac-SDS-Velocity-Flat-G1-Enhanced-v0 --checkpoint logs/rsl_rl/g1_enhanced/2025-08-14_00-40-58/model_700.pt

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# import cli_args from RSL-RL scripts
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts", "reinforcement_learning", "rsl_rl"))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Simple Torso Height Reader")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-SDS-Velocity-Flat-G1-Enhanced-v0", help="Task name")
parser.add_argument("--duration", type=float, default=10.0, help="Duration to collect data (seconds)")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
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
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Main function to read torso height from robot data."""
    
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    print("📏 SIMPLE TORSO HEIGHT READER")
    print("=" * 50)
    print(f"Environment: {args_cli.task}")
    print(f"Number of robots: {args_cli.num_envs}")
    print(f"Duration: {args_cli.duration}s")
    if args_cli.checkpoint:
        print(f"Checkpoint: {args_cli.checkpoint}")
    print("Reading torso height directly from robot data...")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # get robot reference for direct data access
    robot = env.unwrapped.scene.articulations["robot"]
    print(f"✅ Robot found: {robot.__class__.__name__}")
    
    # find torso body for humanoid robot
    torso_body_names = ["torso_link", "torso", "base_link", "base"]
    torso_body_idx = None
    
    for body_name in torso_body_names:
        try:
            torso_indices = robot.find_bodies(body_name)
            if len(torso_indices) > 0:
                # ensure we get a scalar index
                if isinstance(torso_indices, (list, tuple)):
                    torso_body_idx = torso_indices[0]
                    if hasattr(torso_body_idx, 'item'):
                        torso_body_idx = torso_body_idx.item()
                else:
                    torso_body_idx = torso_indices.item() if hasattr(torso_indices, 'item') else torso_indices
                print(f"✅ Found torso body: '{body_name}' at index {torso_body_idx}")
                break
        except Exception:
            continue
    
    if torso_body_idx is None:
        print("❌ Could not find torso body! Available bodies:")
        try:
            print(f"   All body names: {robot.body_names}")
        except Exception:
            print("   (Could not retrieve body names)")
        env.close()
        return
    
    # load policy if checkpoint is provided
    if args_cli.checkpoint:
        print(f"🤖 Loading policy from: {args_cli.checkpoint}")
        
        # get the checkpoint path
        resume_path = retrieve_file_path(args_cli.checkpoint)
        log_dir = os.path.dirname(resume_path)
        
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        
        # wrap environment for RSL-RL
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        # load policy using OnPolicyRunner
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        
        # get inference policy
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        
        print(f"✅ Policy loaded successfully!")
    else:
        print("⚠️  No checkpoint provided, using random actions")
        policy = None
    
    # reset environment and get initial observations
    obs, _ = env.get_observations()
    
    # monitoring loop
    print(f"⏳ Waiting 2 seconds before starting data collection...")
    time.sleep(2.0)
    
    start_time = time.time()
    print(f"🚀 Starting {args_cli.duration}-second torso height data collection...")
    
    try:
        step_count = 0
        all_torso_heights = []
        
        while simulation_app.is_running():
            current_time = time.time()
            
            # check if collection period is over
            if current_time - start_time >= args_cli.duration:
                print(f"✅ Data collection complete ({args_cli.duration} seconds)")
                break
            
            # get action from policy or random
            with torch.inference_mode():
                if policy is not None:
                    actions = policy(obs)
                else:
                    # random actions for demonstration
                    actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            
            # step environment
            obs, _, _, _ = env.step(actions)
            
            # read torso height directly from robot data
            # ensure scalar index
            if isinstance(torso_body_idx, (list, tuple)):
                torso_body_idx = torso_body_idx[0]
                if hasattr(torso_body_idx, 'item'):
                    torso_body_idx = torso_body_idx.item()
            
            # get torso position in world coordinates
            if len(robot.data.body_pos_w.shape) == 3:
                # shape: [num_envs, num_bodies, 3]
                torso_pos_w = robot.data.body_pos_w[:, torso_body_idx, :]  # [num_envs, 3]
                torso_height_z = torso_pos_w[:, 2]  # Z-coordinate (height)
            elif len(robot.data.body_pos_w.shape) == 2:
                # shape: [num_envs, 3] - single body or different format
                torso_height_z = robot.data.body_pos_w[:, 2]  # Z-coordinate (height)
            else:
                print(f"❌ Unexpected tensor shape: {robot.data.body_pos_w.shape}")
                continue
            
            # store height data
            all_torso_heights.append(torso_height_z.cpu().numpy())
            
            # print current height every 20 steps
            if step_count % 20 == 0:
                min_height = torso_height_z.min().item()
                max_height = torso_height_z.max().item()
                avg_height = torso_height_z.mean().item()
                print(f"Step {step_count:3d}: Torso heights - Min: {min_height:.3f}m, Avg: {avg_height:.3f}m, Max: {max_height:.3f}m")
            
            step_count += 1
        
        # final analysis
        if all_torso_heights:
            all_heights = np.concatenate(all_torso_heights)
            
            print(f"\n📊 FINAL TORSO HEIGHT ANALYSIS")
            print(f"   Total data points: {len(all_heights)}")
            print(f"   Height range: {all_heights.min():.3f}m to {all_heights.max():.3f}m")
            print(f"   Average height: {all_heights.mean():.3f}m")
            print(f"   Height std dev: {all_heights.std():.3f}m")
            print(f"   Steps collected: {step_count}")
            
            # height distribution
            percentiles = [5, 25, 50, 75, 95]
            height_percentiles = np.percentile(all_heights, percentiles)
            print(f"   Height percentiles:")
            for p, h in zip(percentiles, height_percentiles):
                print(f"     {p:2d}th percentile: {h:.3f}m")
            
            print(f"\n✅ Direct torso height reading completed successfully!")
        else:
            print(f"❌ No height data collected")
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 