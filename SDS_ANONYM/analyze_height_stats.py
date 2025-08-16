#!/usr/bin/env python3
"""
🔍 HEIGHT STATISTICS ANALYZER

Runs a checkpoint and analyzes robot height behavior:
- Average robot height across environments
- Height gain/loss from initial position
- Height distribution statistics
- Identifies if robots are above/below initial spawn height
"""

import subprocess
import sys
import os
import time
import tempfile
import numpy as np
from pathlib import Path

def analyze_checkpoint_heights(checkpoint_path: str, num_envs: int = 200, steps: int = 1000):
    """
    Run checkpoint and collect height statistics
    """
    print(f"🔍 ANALYZING HEIGHT STATISTICS")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"🤖 Environments: {num_envs}")
    print(f"⏱️  Steps: {steps}")
    print("=" * 60)
    
    # Create a temporary script to inject height logging
    isaac_lab_root = "/home/enis/IsaacLab"
    
    # Create temporary script that logs height data
    height_logger_script = f"""
import torch
import numpy as np
import json
import sys
import os

# Add Isaac Lab to path
sys.path.append('{isaac_lab_root}/source')

from isaaclab.app import AppLauncher

# Parse arguments
app_launcher = AppLauncher(
    description="Height Statistics Analysis",
    epilog="Height analysis tool for Isaac Lab checkpoints"
)
app_launcher.add_arg("--task", type=str, default="Isaac-SDS-Velocity-Flat-G1-Enhanced-v0")
app_launcher.add_arg("--num_envs", type=int, default={num_envs})
app_launcher.add_arg("--checkpoint", type=str, default="{checkpoint_path}")
app_launcher.add_arg("--steps", type=int, default={steps})
args_cli = app_launcher.parse_args()

# Launch the simulator
app_launcher.launch_app()

import isaaclab_tasks
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

def main():
    \"\"\"Main height analysis function\"\"\"
    print(f"\\n🔍 Starting height analysis...")
    
    # Import the task
    try:
        task_name = args_cli.task
        print(f"📋 Task: {{task_name}}")
        
        # Get environment configuration
        import isaaclab_tasks.manager_based.sds.velocity
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.registry import Registry
        
        # Get the environment
        env_cfg = Registry.get("ManagerBasedRLEnvCfg", task_name)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = "cuda:0"
        
        # Override for analysis
        env_cfg.episode_length_s = 60.0  # Longer episodes
        
        print(f"🌍 Creating environment with {{args_cli.num_envs}} environments...")
        env = ManagerBasedRLEnv(cfg=env_cfg)
        
        # Load the checkpoint
        import torch
        checkpoint = torch.load(args_cli.checkpoint, map_location="cuda:0")
        
        print(f"✅ Loaded checkpoint: {{args_cli.checkpoint}}")
        print(f"📊 Policy network loaded successfully")
        
        # Get the policy network
        policy = checkpoint.get('model', None)
        if policy is None:
            policy = checkpoint.get('actor_critic', None)
        if policy is None:
            print("❌ Could not find policy in checkpoint")
            return
            
        # Move policy to device
        policy = policy.to("cuda:0")
        policy.eval()
        
        # Reset environment
        print("🔄 Resetting environment...")
        obs, _ = env.reset()
        
        # Collect height statistics
        height_data = []
        initial_heights = None
        
        print(f"📏 Collecting height data for {{args_cli.steps}} steps...")
        
        for step in range(args_cli.steps):
            # Get robot heights
            robot = env.scene["robot"]
            current_heights = robot.data.root_pos_w[:, 2].cpu().numpy()
            
            # Store initial heights for reference
            if initial_heights is None:
                initial_heights = current_heights.copy()
                print(f"📍 Initial heights recorded:")
                print(f"   Mean: {{np.mean(initial_heights):.4f}}m")
                print(f"   Std:  {{np.std(initial_heights):.4f}}m")
                print(f"   Min:  {{np.min(initial_heights):.4f}}m")
                print(f"   Max:  {{np.max(initial_heights):.4f}}m")
            
            # Calculate height gains
            height_gains = current_heights - initial_heights
            
            # Store statistics
            height_data.append({{
                'step': step,
                'heights': current_heights.copy(),
                'height_gains': height_gains.copy(),
                'mean_height': np.mean(current_heights),
                'mean_gain': np.mean(height_gains),
                'std_height': np.std(current_heights),
                'std_gain': np.std(height_gains),
                'min_height': np.min(current_heights),
                'max_height': np.max(current_heights),
                'min_gain': np.min(height_gains),
                'max_gain': np.max(height_gains),
                'positive_gains': np.sum(height_gains > 0.05),  # 5cm threshold
                'negative_gains': np.sum(height_gains < -0.05),
                'neutral_gains': np.sum(np.abs(height_gains) <= 0.05)
            }})
            
            # Progress indicator
            if step % 100 == 0:
                current_data = height_data[-1]
                print(f"Step {{step:4d}}: Mean height={{current_data['mean_height']:.4f}}m, "
                      f"Mean gain={{current_data['mean_gain']:.4f}}m, "
                      f"Positive gains={{current_data['positive_gains']}}/{{args_cli.num_envs}}")
            
            # Get action from policy
            with torch.no_grad():
                actions = policy.act(obs)[0]
            
            # Step environment
            obs, _, dones, _ = env.step(actions)
            
            # Reset if needed
            if torch.any(dones):
                reset_ids = torch.where(dones)[0]
                if len(reset_ids) > 0:
                    env.reset(reset_ids)
        
        print("\\n" + "="*60)
        print("📊 FINAL HEIGHT ANALYSIS RESULTS")
        print("="*60)
        
        # Calculate overall statistics
        all_heights = np.concatenate([data['heights'] for data in height_data])
        all_gains = np.concatenate([data['height_gains'] for data in height_data])
        
        # Final statistics
        final_data = height_data[-1]
        
        print(f"\\n🏁 FINAL STATE (Step {{len(height_data)-1}}):")
        print(f"   Mean height: {{final_data['mean_height']:.4f}}m")
        print(f"   Mean gain:   {{final_data['mean_gain']:.4f}}m")
        print(f"   Std height:  {{final_data['std_height']:.4f}}m")
        print(f"   Height range: {{final_data['min_height']:.4f}}m to {{final_data['max_height']:.4f}}m")
        print(f"   Gain range:   {{final_data['min_gain']:.4f}}m to {{final_data['max_gain']:.4f}}m")
        
        print(f"\\n📈 HEIGHT DISTRIBUTION:")
        print(f"   Robots with positive gain (>5cm): {{final_data['positive_gains']:3d}}/{{args_cli.num_envs}} ({{100*final_data['positive_gains']/args_cli.num_envs:.1f}}%)")
        print(f"   Robots with negative gain (<-5cm): {{final_data['negative_gains']:3d}}/{{args_cli.num_envs}} ({{100*final_data['negative_gains']/args_cli.num_envs:.1f}}%)")
        print(f"   Robots neutral (±5cm):            {{final_data['neutral_gains']:3d}}/{{args_cli.num_envs}} ({{100*final_data['neutral_gains']/args_cli.num_envs:.1f}}%)")
        
        print(f"\\n📊 OVERALL STATISTICS (All {{len(height_data)}} steps):")
        print(f"   Overall mean height: {{np.mean(all_heights):.4f}}m")
        print(f"   Overall mean gain:   {{np.mean(all_gains):.4f}}m")
        print(f"   Overall std height:  {{np.std(all_heights):.4f}}m")
        print(f"   Overall std gain:    {{np.std(all_gains):.4f}}m")
        
        # Height trend analysis
        mean_gains_over_time = [data['mean_gain'] for data in height_data]
        initial_mean_gain = np.mean(mean_gains_over_time[:50]) if len(mean_gains_over_time) >= 50 else mean_gains_over_time[0]
        final_mean_gain = np.mean(mean_gains_over_time[-50:]) if len(mean_gains_over_time) >= 50 else mean_gains_over_time[-1]
        
        print(f"\\n📈 HEIGHT TREND ANALYSIS:")
        print(f"   Initial mean gain (first 50 steps): {{initial_mean_gain:.4f}}m")
        print(f"   Final mean gain (last 50 steps):   {{final_mean_gain:.4f}}m")
        print(f"   Trend: {{final_mean_gain - initial_mean_gain:+.4f}}m change")
        
        if final_mean_gain > 0.05:
            print("   ✅ POSITIVE: Robots are generally climbing/gaining height")
        elif final_mean_gain < -0.05:
            print("   ⚠️  NEGATIVE: Robots are generally losing height")
        else:
            print("   ➡️  NEUTRAL: Robots maintain roughly initial height")
        
        # Save data
        output_file = f"height_analysis_{{int(time.time())}}.json"
        with open(output_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_data = []
            for data in height_data:
                json_entry = {{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}}
                json_data.append(json_entry)
            
            analysis_results = {{
                'checkpoint': args_cli.checkpoint,
                'num_envs': args_cli.num_envs,
                'steps': len(height_data),
                'initial_heights': {{
                    'mean': float(np.mean(initial_heights)),
                    'std': float(np.std(initial_heights)),
                    'min': float(np.min(initial_heights)),
                    'max': float(np.max(initial_heights))
                }},
                'final_analysis': {{
                    'mean_height': float(final_data['mean_height']),
                    'mean_gain': float(final_data['mean_gain']),
                    'positive_gains': int(final_data['positive_gains']),
                    'negative_gains': int(final_data['negative_gains']),
                    'neutral_gains': int(final_data['neutral_gains']),
                    'trend_change': float(final_mean_gain - initial_mean_gain)
                }},
                'time_series': json_data
            }}
            
            json.dump(analysis_results, f, indent=2)
        
        print(f"\\n💾 Data saved to: {{output_file}}")
        print("\\n🎯 CONCLUSION:")
        if final_data['mean_gain'] > 0.02:
            print("   The checkpoint shows robots are successfully gaining height (likely climbing)")
        elif final_data['mean_gain'] < -0.02:
            print("   The checkpoint shows robots are losing height (may indicate falling/sinking)")
        else:
            print("   The checkpoint shows robots maintain stable height (good for flat terrain)")
            
        env.close()
        
    except Exception as e:
        print(f"❌ Error during analysis: {{e}}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
    
    # Write the temporary script
    temp_script_path = "/tmp/height_analyzer.py"
    with open(temp_script_path, 'w') as f:
        f.write(height_logger_script)
    
    print(f"📝 Created analysis script: {temp_script_path}")
    print("🚀 Running height analysis...")
    print("-" * 60)
    
    # Run the analysis
    try:
        cmd = [
            "python", temp_script_path,
            "--task", "Isaac-SDS-Velocity-Flat-G1-Enhanced-v0", 
            "--num_envs", str(num_envs),
            "--checkpoint", checkpoint_path,
            "--steps", str(steps)
        ]
        
        result = subprocess.run(
            cmd,
            cwd=isaac_lab_root,
            capture_output=False,  # Show real-time output
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print("\n✅ Height analysis completed successfully!")
        else:
            print(f"\n❌ Analysis failed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Analysis timed out after 10 minutes")
    except Exception as e:
        print(f"\n❌ Error running analysis: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze robot height statistics from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--num_envs", type=int, default=200, help="Number of environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to analyze")
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    analyze_checkpoint_heights(args.checkpoint, args.num_envs, args.steps) 