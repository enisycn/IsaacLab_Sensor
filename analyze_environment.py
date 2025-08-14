#!/usr/bin/env python3

"""
RL Policy Height Scanner Monitor
Loads a trained RL policy and monitors terrain height while robot walks.
Uses Isaac Lab's standard policy loading mechanism.
"""

import argparse
import time
import torch
import gymnasium as gym
import numpy as np
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="RL Policy Height Scanner Monitor")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-SDS-Velocity-Rough-G1-Enhanced-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, default="logs/rsl_rl/g1_enhanced/2025-08-14_00-40-58/model_700.pt", help="Path to RL checkpoint file")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Import RSL-RL components (same as play.py)
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import cli_args from the scripts directory
import sys
sys.path.append("scripts/reinforcement_learning/rsl_rl")
import cli_args

def load_rsl_rl_policy(env, task_name, checkpoint_path):
    """Load RSL-RL policy using Isaac Lab's standard method."""
    print(f"🤖 Loading RSL-RL policy from: {checkpoint_path}")
    
    try:
        # Parse agent configuration (same as play.py)
        # Create minimal args for cli_args.parse_rsl_rl_cfg
        class MinimalArgs:
            def __init__(self):
                self.experiment_name = None
                self.run_name = None
                self.resume = False
                self.load_run = None
                self.checkpoint = checkpoint_path
                self.logger = None
                self.log_project_name = None
                self.device = args_cli.device
        
        minimal_args = MinimalArgs()
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, minimal_args)
        
        # Get the checkpoint path
        resume_path = retrieve_file_path(checkpoint_path)
        log_dir = os.path.dirname(resume_path)
        
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        
        # Wrap environment for RSL-RL (same as play.py)
        env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        # Load policy using OnPolicyRunner (same as play.py)
        ppo_runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        
        # Get inference policy (same as play.py)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        
        print(f"✅ RSL-RL policy loaded successfully!")
        
        return policy, env_wrapped
        
    except Exception as e:
        print(f"❌ Failed to load RSL-RL policy: {e}")
        import traceback
        traceback.print_exc()
        return None, env

def main():
    """Main monitoring function with RL policy."""
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    
    # Create environment (gravity ENABLED for realistic walking)
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Convert to single-agent if needed (same as play.py)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    print("🔍 RL POLICY HEIGHT SCANNER MONITOR")
    print("="*70)
    print(f"Environment: {args_cli.task}")
    print(f"Number of robots: {args_cli.num_envs}")
    print(f"Checkpoint: {args_cli.checkpoint}")
    print("Monitoring terrain height while robot walks with RL policy...")
    print("Press Ctrl+C to stop")
    print("="*70)
    
    # Get height scanner sensor BEFORE wrapping environment
    if hasattr(env.unwrapped.scene, 'sensors') and "height_scanner" in env.unwrapped.scene.sensors:
        height_scanner = env.unwrapped.scene.sensors["height_scanner"]
        print(f"✅ Height scanner found!")
        print(f"   Number of rays per robot: {height_scanner.num_rays}")
        print(f"   Total rays across all robots: {height_scanner.num_rays * args_cli.num_envs}")
    else:
        print("❌ No height scanner found in this environment!")
        env.close()
        simulation_app.close()
        return
    
    # Load RL policy using Isaac Lab's standard method
    policy, env_wrapped = load_rsl_rl_policy(env, args_cli.task, args_cli.checkpoint)
    
    if policy is None:
        print("❌ Failed to load policy! Exiting.")
        env.close()
        simulation_app.close()
        return
    
    # Get the correct number of environments
    num_envs = args_cli.num_envs
    
    # Reset environment and get initial observations (same as play.py)
    obs, _ = env_wrapped.get_observations()
    
    # Monitoring loop
    print("⏳ Waiting 2 seconds before starting sensor readings...")
    time.sleep(2.0)
    
    start_time = time.time()
    collection_duration = 10.0  # Collect data for 10 seconds
    print(f"🚀 Starting {collection_duration}-second sensor data collection...")
    
    try:
        step_count = 0
        episode_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
        
        while simulation_app.is_running():
            current_time = time.time()
            
            # Check if collection period is over
            if current_time - start_time >= collection_duration:
                print(f"✅ Data collection complete ({collection_duration} seconds)")
                break
            
            # Get policy action (same as play.py)
            with torch.inference_mode():
                actions = policy(obs)
            
            # Step environment
            obs, rew, terminated, truncated = env_wrapped.step(actions)
            episode_rewards += rew
            
            # Height sensor analysis - only print once at the end
            if current_time - start_time >= collection_duration - 0.1:  # Near the end
                print(f"📋 COMPREHENSIVE FINAL ENVIRONMENT ANALYSIS FOR AI AGENT")
                
                # Get height scanner data
                sensor = env.unwrapped.scene.sensors["height_scanner"]
                sensor_pos_z = sensor.data.pos_w[:, 2]  # Sensor Z positions
                ray_hits_z = sensor.data.ray_hits_w[..., 2]  # Ray hit Z positions
                
                # Collect ALL rays from ALL robots
                all_sensor_heights = []
                all_height_readings = []

                for env_idx in range(num_envs):
                    sensor_height = sensor_pos_z[env_idx].item()
                    hit_points = ray_hits_z[env_idx]  # All ray hits for this robot

                    # Calculate RL observation height readings (Isaac Lab formula)
                    offset = 0.5
                    height_readings = sensor_height - hit_points - offset

                    # Add all this robot's rays to the combined pool
                    all_height_readings.extend(height_readings.tolist())
                    all_sensor_heights.extend([sensor_height] * len(height_readings))

                # Calculate total rays
                total_rays = len(all_height_readings)

                if all_height_readings:
                    # Convert to tensor for easy processing
                    all_height_readings = torch.tensor(all_height_readings)
                    valid_rays = len(all_height_readings)

                    # ✅ USE ISAAC LAB STANDARD BASELINE (same as reward function)
                    baseline_height_standard = 0.209  # Standard G1 baseline from Isaac Lab guide
                    
                    # ✅ USE DYNAMIC THRESHOLDS (adaptable based on analysis)
                    baseline_threshold = 0.07         # Default threshold for balanced detection
                    gap_threshold = baseline_threshold      # Above baseline = gaps
                    obstacle_threshold = baseline_threshold # Below baseline = obstacles
                    
                    print(f"   🎯 USING ISAAC LAB STANDARD DETECTION:")
                    print(f"     Standard baseline: {baseline_height_standard:.3f}m")
                    print(f"     Classification threshold: ±{baseline_threshold:.2f}m")
                    print(f"     Gap threshold: >{baseline_height_standard + gap_threshold:.3f}m (terrain >{gap_threshold:.2f}m lower)")
                    print(f"     Obstacle threshold: <{baseline_height_standard - obstacle_threshold:.3f}m (terrain >{obstacle_threshold:.2f}m higher)")
                    print(f"     Normal range: {baseline_height_standard - obstacle_threshold:.3f}m to {baseline_height_standard + gap_threshold:.3f}m")

                    # ✅ CORRECTED CLASSIFICATION (same logic as reward function)
                    obstacles = all_height_readings < (baseline_height_standard - obstacle_threshold)
                    gaps = all_height_readings > (baseline_height_standard + gap_threshold)  
                    normal = ~obstacles & ~gaps

                    obstacle_count = obstacles.sum().item()
                    gap_count = gaps.sum().item()
                    normal_count = normal.sum().item()

                    # Calculate percentages of total rays
                    obstacle_pct = (obstacle_count / total_rays) * 100
                    gap_pct = (gap_count / total_rays) * 100
                    normal_pct = (normal_count / total_rays) * 100

                    # Find extreme values across all robots
                    min_height = all_height_readings.min().item()
                    max_height = all_height_readings.max().item()
                    avg_height_reading = all_height_readings.mean().item()

                    # Find deepest gap and highest obstacle
                    deepest_gap = None
                    highest_obstacle = None
                    deepest_gap_reading = None
                    highest_obstacle_reading = None
                    shallowest_gap_reading = None
                    actual_obstacle_height = None
                    actual_gap_depth = None
                    actual_shallowest_gap_depth = None

                    if gap_count > 0:
                        deepest_gap_reading = all_height_readings[gaps].max().item()  # Highest reading = deepest gap
                        shallowest_gap_reading = all_height_readings[gaps].min().item()  # Lowest reading = shallowest gap
                        # Gap depth: how much the reading exceeds the gap threshold
                        actual_gap_depth = deepest_gap_reading - (baseline_height_standard + gap_threshold)
                        actual_shallowest_gap_depth = shallowest_gap_reading - (baseline_height_standard + gap_threshold)

                    if obstacle_count > 0:
                        highest_obstacle_reading = all_height_readings[obstacles].min().item()  # Lowest reading = highest obstacle
                        # Obstacle height: how much the reading falls below the obstacle threshold  
                        actual_obstacle_height = (baseline_height_standard - obstacle_threshold) - highest_obstacle_reading

                    # Simple output
                    print(f"   📊 CORRECTED RAY ANALYSIS (Isaac Lab Standards):")
                    print(f"     Total rays: {total_rays} (from {num_envs} robots)")
                    print(f"     Height readings: {min_height:.3f}m to {max_height:.3f}m (avg: {avg_height_reading:.3f}m)")
                    print(f"     Isaac Lab baseline: {baseline_height_standard:.3f}m (standard)")
                    print(f"     Measured baseline: {avg_height_reading:.3f}m (actual terrain average)")
                    print(f"   ---")
                    print(f"   🔺 OBSTACLES:")
                    print(f"     Count: {obstacle_count} rays ({obstacle_pct:.1f}%)")
                    if actual_obstacle_height is not None:
                        print(f"     Tallest obstacle height above baseline: {actual_obstacle_height:.3f}m")
                        print(f"       (Obstacle sensor reading: {highest_obstacle_reading:.3f}m)")
                        print(f"       (Isaac Lab baseline: {baseline_height_standard:.3f}m)")
                        print(f"       (Threshold: <{baseline_height_standard - obstacle_threshold:.3f}m)")
                        print(f"       (Physical meaning: Terrain is {actual_obstacle_height:.3f}m higher than expected)")
                    
                    print(f"   🕳️  GAPS:")
                    print(f"     Count: {gap_count} rays ({gap_pct:.1f}%)")
                    if actual_gap_depth is not None:
                        print(f"     Deepest gap depth below baseline: {actual_gap_depth:.3f}m")
                        print(f"       (Gap sensor reading: {deepest_gap_reading:.3f}m)")
                        print(f"       (Isaac Lab baseline: {baseline_height_standard:.3f}m)")
                        print(f"       (Threshold: >{baseline_height_standard + gap_threshold:.3f}m)")
                        print(f"       (Physical meaning: Terrain is {actual_gap_depth + gap_threshold:.3f}m lower than expected)")
                        if actual_shallowest_gap_depth >= 0:
                            print(f"     Shallowest gap depth below baseline: {actual_shallowest_gap_depth:.3f}m")
                            print(f"       (Gap sensor reading: {shallowest_gap_reading:.3f}m)")
                            print(f"       (Physical meaning: Terrain is {actual_shallowest_gap_depth + gap_threshold:.3f}m lower than expected)")
                    
                    print(f"   🏞️  NORMAL TERRAIN:")
                    print(f"     Count: {normal_count} rays ({normal_pct:.1f}%)")
                    print(f"     Definition: Terrain between {baseline_height_standard - obstacle_threshold:.3f}m and {baseline_height_standard + gap_threshold:.3f}m")
                    print(f"     Range: ±{baseline_threshold:.2f}m tolerance zone around {baseline_height_standard:.3f}m baseline")
                    
                    # 🌊 TERRAIN ROUGHNESS (within ±0.07m normal range)
                    if normal_count > 0:
                        normal_readings = all_height_readings[normal]
                        terrain_std = normal_readings.std().item()
                        
                        # Simple roughness classification
                        if terrain_std < 0.01:
                            roughness_level = "🟢 SMOOTH"
                        elif terrain_std < 0.025:
                            roughness_level = "🟡 MODERATE"
                        elif terrain_std < 0.05:
                            roughness_level = "🟠 ROUGH"
                        else:
                            roughness_level = "🔴 VERY ROUGH"
                        
                        print(f"   🌊 TERRAIN ROUGHNESS: {terrain_std*100:.1f}cm variation ({roughness_level})")
                    else:
                        print(f"   🌊 TERRAIN ROUGHNESS: No normal terrain data")
                    
                    # Safety assessment with corrected thresholds
                    if gap_pct > 10.0 or obstacle_pct > 30.0:
                        safety = "🔴 DANGEROUS"
                    elif gap_pct > 5.0 or obstacle_pct > 20.0:
                        safety = "🟡 CAUTION"
                    else:
                        safety = "🟢 SAFE"
                    print(f"   🚨 Overall terrain safety: {safety}")
                    print(f"     Reason: {gap_pct:.1f}% gaps (limit: 10.0%), {obstacle_pct:.1f}% obstacles (limit: 30.0%)")
                else:
                    print(f"   ❌ No terrain data available")
                
                break  # Exit after printing analysis
            
            step_count += 1
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown (silent)
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    try:
        main()
    finally:
        # Final GPU memory clearing before simulation app close (silent)
        try:
            import torch
            import gc
            import time
            
            if torch.cuda.is_available():
                time.sleep(1)
                device = torch.cuda.current_device()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            
            gc.collect()
            
        except Exception:
            pass
        
        # Close sim app silently  
        try:
            simulation_app.close()
        except Exception:
            pass 