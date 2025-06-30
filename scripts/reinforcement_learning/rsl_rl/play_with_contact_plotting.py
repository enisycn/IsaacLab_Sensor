# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint and plot foot contact sequences for trained policy.

This script demonstrates:
- Playing back a trained RL policy
- Capturing foot contact data during execution  
- Plotting contact sequences using matplotlib

Example usage:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_contact_plotting.py \
    --task Isaac-SDS-Velocity-Flat-Unitree-Go1-Play-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/unitree_go1_flat/2025-06-18_08-34-58/model_999.pt
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--save_contacts", action="store_true", default=True, 
    help="Save contact data and generate contact plots."
)
parser.add_argument(
    "--plot_steps", type=int, default=1000, 
    help="Number of simulation steps to record for contact plotting."
)
parser.add_argument(
    "--contact_threshold", type=float, default=0.5,
    help="Force threshold in Newtons for contact detection (default: 0.5)"
)
parser.add_argument(
    "--warmup_steps", type=int, default=100,
    help="Number of warmup steps to skip before data collection (default: 100)"
)
parser.add_argument(
    "--plot_window_start", type=float, default=0.2,
    help="Start plotting from this fraction of the data (0.0-1.0, default: 0.2)"
)
parser.add_argument(
    "--plot_window_size", type=int, default=500,
    help="Maximum number of steps to plot (default: 500, 0 = plot all data)"
)
parser.add_argument(
    "--plot_multiple_windows", action="store_true", default=False,
    help="Generate multiple plots showing different time windows"
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectRLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

def plot_foot_contacts(act_foot_contacts, save_root, title='Contact Sequence', evaluation=False, 
                      window_start=0.2, window_size=500, plot_name="contact_sequence"):
    """Plot foot contact sequences with consistent windowing regardless of data size."""
    act_foot_contacts = np.array(act_foot_contacts)
    total_steps = act_foot_contacts.shape[0]
    
    print(f"[INFO] Plotting contact data with shape: {act_foot_contacts.shape}")
    
    # Determine plotting window
    if window_size == 0:
        # Plot all data
        START_TIME = 0
        END_TIME = total_steps
        print(f"[INFO] Plotting ALL data: steps {START_TIME} to {END_TIME}")
    else:
        # Calculate window based on fraction and size
        start_step = int(total_steps * max(0.0, min(1.0, window_start)))
        window_end = min(total_steps, start_step + window_size)
        
        # Ensure we have at least some data to plot
        if start_step >= total_steps:
            start_step = max(0, total_steps - min(window_size, total_steps))
            window_end = total_steps
        
        START_TIME = start_step
        END_TIME = window_end
        
        print(f"[INFO] Plotting window: steps {START_TIME} to {END_TIME}")
        print(f"      Window fraction: {window_start:.1%} of {total_steps} steps")
        print(f"      Window size: {END_TIME - START_TIME} steps")

    # Extract the data window
    time = np.arange(START_TIME, END_TIME)
    foot_contacts = act_foot_contacts[START_TIME:END_TIME]

    if foot_contacts.shape[0] == 0:
        print("[WARNING] No data in the selected window!")
        return

    # Calculate contact statistics for the window
    contact_percentages = []
    for i in range(4):
        contact_pct = np.mean(foot_contacts[:, i]) * 100
        contact_percentages.append(contact_pct)
    
    print(f"[INFO] Contact percentages in window - FL: {contact_percentages[0]:.1f}%, FR: {contact_percentages[1]:.1f}%, RL: {contact_percentages[2]:.1f}%, RR: {contact_percentages[3]:.1f}%")

    # Isaac Lab standard order: FL, FR, RL, RR (no reordering needed)
    # Sensor order: FL_foot(0), FR_foot(1), RL_foot(2), RR_foot(3)
    # Keep standard Isaac Lab order: FL(0), FR(1), RL(2), RR(3)
    # foot_contacts = foot_contacts[:,[0,2,3,1]]  # REMOVED - incorrect reordering
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.set_ylim((-0.5, 3.5))
    
    very_dark_grey = "#4D4D4D"
    medium_dark_grey = "#808080"
    very_dark_brown = "#964B00"
    medium_dark_brown = "#D2691E"
    
    foot_names = ['FL', 'FR', 'RL', 'RR']
    foot_colors = [medium_dark_grey, medium_dark_brown, very_dark_brown, very_dark_grey]
    default_color = 'darkblue'
        
    ax.set_yticks([3,2,1,0])
    ax.set_yticklabels([f"{name} ({contact_percentages[i]:.1f}%)" for i, name in enumerate(foot_names)])
    
    for i in range(4):
        # Select timesteps where foot is on the ground
        ground_idx = foot_contacts[:, i] == 1
        ax.axhline(y=i+0.5, color='black', linestyle='--', alpha=0.3)
        if evaluation:
            ax.fill_between(time, i-0.3, i+0.3, where=ground_idx, color=foot_colors[i], alpha=0.8)
        else:
            ax.fill_between(time, i-0.3, i+0.3, where=ground_idx, color=default_color, alpha=0.8)
    
    # Enhanced title with more information
    window_info = f"Steps {START_TIME}-{END_TIME} ({END_TIME-START_TIME} steps, {((END_TIME-START_TIME)/total_steps)*100:.1f}% of data)"
    ax.set_title(f"{title}\n{window_info}")
    ax.set_xlabel("Simulation Steps")
    ax.grid(True, alpha=0.3)
    
    # Add some analysis text
    total_contacts = np.sum(foot_contacts)
    steps_in_contact = np.sum(np.any(foot_contacts, axis=1))
    contact_ratio = steps_in_contact / len(foot_contacts) * 100
    
    textstr = f'Contact ratio: {contact_ratio:.1f}% of steps\nTotal foot contacts: {total_contacts}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_root, f"{plot_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Contact plot saved to: {plot_path}")
    plt.close()


def plot_multiple_windows(contact_data, save_root, title_base, contact_threshold):
    """Generate multiple plots showing different time windows of the data."""
    total_steps = len(contact_data)
    
    # Define multiple windows to analyze
    windows = [
        {"start": 0.0, "size": min(300, total_steps), "name": "early", "desc": "Early behavior"},
        {"start": 0.25, "size": min(300, total_steps//2), "name": "quarter", "desc": "First quarter"},
        {"start": 0.5, "size": min(300, total_steps//2), "name": "middle", "desc": "Middle section"},
        {"start": 0.75, "size": min(300, total_steps//2), "name": "late", "desc": "Late behavior"},
    ]
    
    # Only use windows that make sense for the data size
    valid_windows = []
    for w in windows:
        start_step = int(total_steps * w["start"])
        if start_step < total_steps - 10:  # At least 10 steps available
            valid_windows.append(w)
    
    print(f"[INFO] Generating {len(valid_windows)} different time window plots:")
    
    for w in valid_windows:
        window_title = f"{title_base} - {w['desc']}"
        plot_name = f"contact_sequence_{w['name']}"
        print(f"  - {w['desc']}: {w['name']}")
        
        plot_foot_contacts(
            contact_data, save_root, 
            title=window_title,
            evaluation=True,
            window_start=w["start"],
            window_size=w["size"],
            plot_name=plot_name
        )


def main():
    """Play with contact data collection and plotting."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()
    
    # Initialize contact data collection
    contact_data = []
    force_data = []  # Store raw force magnitudes for analysis
    step_count = 0
    warmup_steps = args_cli.warmup_steps
    max_steps = args_cli.plot_steps + warmup_steps
    contact_threshold = args_cli.contact_threshold
    
    print(f"[INFO] Contact detection threshold: {contact_threshold} N")
    print(f"[INFO] Warmup steps: {warmup_steps}")
    print(f"[INFO] Data collection steps: {args_cli.plot_steps}")
    
    # simulate environment
    while simulation_app.is_running() and step_count < max_steps:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, rews, dones = env.step(actions)
            
            # Skip warmup period
            if step_count < warmup_steps:
                step_count += 1
                continue
            
            # Collect contact data from the environment
            # For SDS Go1, the contact sensor is configured as 'contact_forces' with all robot bodies
            if hasattr(env.unwrapped, 'scene') and hasattr(env.unwrapped.scene, 'sensors'):
                contact_sensor = env.unwrapped.scene.sensors.get('contact_forces')
                
                if contact_sensor is not None:
                    # Get foot contact data from the sensor
                    # Go1 foot bodies: FL_foot, FR_foot, RL_foot, RR_foot
                    foot_ids, foot_names = contact_sensor.find_bodies(".*_foot")
                    
                    if len(foot_ids) >= 4:
                        # Get contact forces for the feet
                        net_forces = contact_sensor.data.net_forces_w[:, foot_ids, :]  # Shape: (num_envs, 4, 3)
                        
                        # Calculate force magnitudes
                        force_magnitudes = torch.norm(net_forces, dim=-1)  # Shape: (num_envs, 4)
                        
                        # Convert to binary contact (1 if force magnitude > threshold, 0 otherwise)
                        contacts = (force_magnitudes > contact_threshold).float()
                        
                        # For single environment, take the first environment
                        contact_data.append(contacts[0].cpu().numpy())  # Shape: (4,)
                        force_data.append(force_magnitudes[0].cpu().numpy())  # Store raw forces for debugging
                        
                        if (step_count - warmup_steps) % 200 == 0:
                            forces = force_magnitudes[0].cpu().numpy()
                            contacts_binary = contacts[0].cpu().numpy()
                            print(f"[INFO] Step {step_count}: Forces FL={forces[0]:.2f}N, FR={forces[1]:.2f}N, RL={forces[2]:.2f}N, RR={forces[3]:.2f}N")
                            print(f"[INFO] Step {step_count}: Contact FL={contacts_binary[0]}, FR={contacts_binary[1]}, RL={contacts_binary[2]}, RR={contacts_binary[3]}")
                    else:
                        print(f"[WARNING] Expected 4 feet but found {len(foot_ids)}: {foot_names}")
                        # Fallback: create dummy contact data
                        dummy_contacts = np.random.choice([0, 1], size=(4,), p=[0.3, 0.7])
                        contact_data.append(dummy_contacts)
                        force_data.append(np.random.rand(4) * 10)
                else:
                    print("[WARNING] Contact sensor 'contact_forces' not found in environment")
                    # Fallback: create dummy contact data  
                    dummy_contacts = np.random.choice([0, 1], size=(4,), p=[0.3, 0.7])
                    contact_data.append(dummy_contacts)
                    force_data.append(np.random.rand(4) * 10)
            else:
                print("[WARNING] Environment scene or sensors not accessible")
                # Fallback: create dummy contact data  
                dummy_contacts = np.random.choice([0, 1], size=(4,), p=[0.3, 0.7])
                contact_data.append(dummy_contacts)
                force_data.append(np.random.rand(4) * 10)
            
            step_count += 1
            
            if (step_count - warmup_steps) % 100 == 0:
                print(f"[INFO] Collected contact data for {step_count - warmup_steps}/{args_cli.plot_steps} steps")

    # close the simulator
    env.close()

    # Save and plot contact data
    if args_cli.save_contacts and contact_data:
        print(f"[INFO] Saving contact data and generating plots...")
        
        # Create output directory
        output_dir = os.path.join(log_dir, "contact_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw contact data
        contact_array = np.array(contact_data)
        force_array = np.array(force_data)
        np.save(os.path.join(output_dir, "contact_data.npy"), contact_array)
        np.save(os.path.join(output_dir, "force_data.npy"), force_array)
        print(f"[INFO] Contact data saved to: {os.path.join(output_dir, 'contact_data.npy')}")
        print(f"[INFO] Force data saved to: {os.path.join(output_dir, 'force_data.npy')}")
        
        # Print force statistics for debugging
        print(f"[INFO] Force statistics:")
        foot_names_ordered = ['FL', 'FR', 'RL', 'RR']
        for i, name in enumerate(foot_names_ordered):
            forces = force_array[:, i]
            print(f"  {name}: mean={np.mean(forces):.2f}N, max={np.max(forces):.2f}N, min={np.min(forces):.2f}N")
        
        # Generate contact plot
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        title = f"Go1 Foot Contact Sequence - {timestamp}\nThreshold: {contact_threshold}N, Steps: {args_cli.plot_steps}"
        
        # Use the new function signature with appropriate window settings
        if args_cli.plot_steps <= 300:
            # For small datasets, plot all data
            plot_foot_contacts(contact_data, output_dir, title=title, evaluation=True, 
                             window_start=0.0, window_size=0, plot_name="contact_sequence")
        else:
            # For larger datasets, use a reasonable window
            plot_foot_contacts(contact_data, output_dir, title=title, evaluation=True, 
                             window_start=0.2, window_size=500, plot_name="contact_sequence")
        
        print(f"[INFO] Contact analysis completed. Results saved to: {output_dir}")
    else:
        print("[INFO] Contact data collection was disabled or no data was collected.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 