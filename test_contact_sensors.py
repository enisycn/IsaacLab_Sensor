#!/usr/bin/env python3
"""
Contact Sensor Diagnostic for G1 Robot
Tests if contact sensors are properly configured and working.
"""

import argparse
import torch
import gymnasium as gym
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Contact Sensor Diagnostic")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-SDS-Velocity-Rough-G1-Enhanced-v0", help="Task name")
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

def test_contact_sensors():
    """Test contact sensor functionality."""
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    print("🔍 CONTACT SENSOR DIAGNOSTIC")
    print("="*70)
    print(f"Environment: {args_cli.task}")
    print(f"Number of robots: {args_cli.num_envs}")
    print("="*70)
    
    # Check if contact sensor exists
    if hasattr(env.unwrapped.scene, 'sensors') and "contact_forces" in env.unwrapped.scene.sensors:
        contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        print(f"✅ Contact sensor found!")
        print(f"   Sensor type: {type(contact_sensor)}")
        print(f"   Configuration: {contact_sensor.cfg}")
        print()
        
        # Check robot structure
        robot = env.unwrapped.scene["robot"]
        print(f"🤖 ROBOT STRUCTURE:")
        print(f"   Robot type: {type(robot)}")
        print(f"   Body names: {robot.body_names}")
        print()
        
        # Test contact sensor body finding
        print(f"🔍 CONTACT SENSOR BODY DETECTION:")
        try:
            foot_ids, foot_names = contact_sensor.find_bodies(".*_ankle_roll_link")
            print(f"   Pattern: '.*_ankle_roll_link'")
            print(f"   Found {len(foot_ids)} bodies: {foot_names}")
            print(f"   Body IDs: {foot_ids}")
            
            if len(foot_ids) >= 2:
                print(f"   ✅ SUCCESS: Found {len(foot_ids)} foot contacts")
            else:
                print(f"   ❌ PROBLEM: Only found {len(foot_ids)} foot contacts (need 2)")
                
                # Try alternative patterns
                print(f"\n🔧 TRYING ALTERNATIVE PATTERNS:")
                alternatives = [
                    ".*ankle.*",
                    ".*_ankle.*", 
                    ".*foot.*",
                    ".*_foot.*",
                    "left_ankle_roll_link",
                    "right_ankle_roll_link"
                ]
                
                for pattern in alternatives:
                    try:
                        alt_ids, alt_names = contact_sensor.find_bodies(pattern)
                        print(f"   Pattern '{pattern}': {len(alt_ids)} bodies - {alt_names}")
                    except Exception as e:
                        print(f"   Pattern '{pattern}': ERROR - {e}")
                        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            
    else:
        print("❌ No contact sensor found!")
        if hasattr(env.unwrapped.scene, 'sensors'):
            print(f"   Available sensors: {list(env.unwrapped.scene.sensors.keys())}")
        else:
            print("   No sensors found in scene")
        return
    
    # Reset and run a few steps to test contact forces
    print(f"\n🚀 TESTING CONTACT FORCES:")
    obs, _ = env.reset()
    
    for step in range(5):
        # Random actions
        actions = torch.randn(args_cli.num_envs, env.unwrapped.action_space.shape[0], device=env.unwrapped.device) * 0.1
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Check contact forces
        if len(foot_ids) >= 2:
            foot_ids_tensor = torch.tensor(foot_ids[:2], dtype=torch.long, device=env.unwrapped.device)
            contact_forces = contact_sensor.data.net_forces_w[:, foot_ids_tensor, 2]  # Vertical forces
            
            # Contact detection with different thresholds
            contact_15N = contact_forces > 15.0
            contact_50N = contact_forces > 50.0
            
            both_feet_15N = torch.all(contact_15N, dim=1)
            any_foot_15N = torch.any(contact_15N, dim=1)
            both_feet_50N = torch.all(contact_50N, dim=1)
            any_foot_50N = torch.any(contact_50N, dim=1)
            
            print(f"   Step {step+1}:")
            print(f"     Average forces: {contact_forces.mean(dim=0).cpu().numpy():.1f}N per foot")
            print(f"     Max forces: {contact_forces.max(dim=0)[0].cpu().numpy():.1f}N per foot")
            print(f"     15N threshold: Both={torch.sum(both_feet_15N).item()}/{args_cli.num_envs}, Any={torch.sum(any_foot_15N).item()}/{args_cli.num_envs}")
            print(f"     50N threshold: Both={torch.sum(both_feet_50N).item()}/{args_cli.num_envs}, Any={torch.sum(any_foot_50N).item()}/{args_cli.num_envs}")
    
    print(f"\n🎯 CONTACT SENSOR SUMMARY:")
    if len(foot_ids) >= 2:
        print(f"   ✅ Contact sensors working correctly")
        print(f"   ✅ {len(foot_ids)} foot contacts detected")
        print(f"   ✅ Force readings available")
        print(f"   📝 Use 15.0N threshold for jumping detection")
        print(f"   📝 Use 50.0N threshold for solid contact")
    else:
        print(f"   ❌ Contact sensors have issues")
        print(f"   ❌ Wrong body pattern or missing bodies")
    
    # Clean shutdown
    env.close()

if __name__ == "__main__":
    try:
        test_contact_sensors()
    finally:
        simulation_app.close()
        print("🏁 Contact sensor diagnostic completed") 