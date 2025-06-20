#!/usr/bin/env python3
"""
SDS Training Monitor

This script helps monitor SDS training progress by showing:
- Current SDS iteration and reward sample being trained
- IsaacLab training progress (epochs, rewards, loss)
- GPU utilization
- GPT-generated reward components dynamically extracted from code
- Log file updates

Usage:
    python monitor_training.py [--workspace /path/to/workspace]
"""

import os
import time
import argparse
import subprocess
from pathlib import Path
import re
from datetime import datetime, timedelta
import ast
import textwrap

def get_latest_workspace():
    """Find the most recent SDS workspace directory."""
    outputs_dir = Path("outputs/sds")
    if not outputs_dir.exists():
        return None
    
    workspaces = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not workspaces:
        return None
    
    # Always get the most recently modified workspace
    latest = max(workspaces, key=lambda x: x.stat().st_mtime)
    return latest

def get_sds_start_time(workspace_dir):
    """Get SDS start time from the workspace directory creation time."""
    try:
        return datetime.fromtimestamp(workspace_dir.stat().st_ctime)
    except:
        return None

def format_duration(duration):
    """Format duration as HH:MM:SS."""
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)
    seconds = int(duration.total_seconds() % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def extract_gpt_reward_components(workspace_dir):
    """Dynamically extract GPT-generated reward component names from the generated reward files."""
    try:
        # Look for the most recent reward-only file
        reward_files = list(workspace_dir.glob("env_iter*_response*_rewardonly.py"))
        if not reward_files:
            return {}
        
        # Get the most recent reward file
        latest_reward_file = max(reward_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_reward_file, 'r') as f:
            reward_code = f.read()
        
        # Parse the reward function to extract component names and their logic
        components = {}
        
        # Extract variable assignments that look like reward components
        # Look for patterns like: component_name = torch.something or reward += component_name
        lines = reward_code.split('\n')
        
        in_function = False
        current_components = []
        
        for line in lines:
            stripped = line.strip()
            
            # Start tracking when we enter the function
            if stripped.startswith('def sds_custom_reward'):
                in_function = True
                continue
            
            if not in_function:
                continue
                
            # End tracking when we exit the function  
            if stripped.startswith('def ') and not stripped.startswith('def sds_custom_reward'):
                break
                
            if stripped.startswith('return '):
                break
            
            # Look for reward component assignments
            # Pattern 1: component_name = some_calculation
            if '=' in stripped and not stripped.startswith('#'):
                # Exclude simple assignments like device=env.device
                if any(keyword in stripped for keyword in ['reward', 'penalty', 'term', 'component', 'bonus']):
                    var_name = stripped.split('=')[0].strip()
                    # Clean variable name
                    if var_name and var_name.isidentifier():
                        # Extract the calculation part for display
                        calc_part = '='.join(stripped.split('=')[1:]).strip()
                        # Truncate long calculations
                        if len(calc_part) > 60:
                            calc_part = calc_part[:57] + "..."
                        current_components.append({
                            'name': var_name,
                            'calculation': calc_part,
                            'line': stripped
                        })
            
            # Pattern 2: reward += something or reward = reward + something
            elif 'reward' in stripped and ('+=' in stripped or 'reward =' in stripped and '+' in stripped):
                # Extract what's being added to reward
                if '+=' in stripped:
                    added_part = stripped.split('+=')[1].strip()
                elif '=' in stripped and '+' in stripped:
                    # Find what's after the +
                    plus_parts = stripped.split('+')
                    if len(plus_parts) > 1:
                        added_part = '+'.join(plus_parts[1:]).strip()
                    else:
                        continue
                else:
                    continue
                
                # Truncate long expressions
                if len(added_part) > 50:
                    added_part = added_part[:47] + "..."
                    
                current_components.append({
                    'name': f"direct_reward_{len(current_components)}",
                    'calculation': added_part,
                    'line': stripped
                })
        
        # Extract file metadata
        file_parts = latest_reward_file.stem.split('_')
        iteration = "?"
        response = "?"
        for i, part in enumerate(file_parts):
            if part.startswith('iter') and i+1 < len(file_parts):
                iteration = file_parts[i+1]
            elif part.startswith('response') and i+1 < len(file_parts):
                response = file_parts[i+1]
        
        return {
            'file': latest_reward_file.name,
            'iteration': iteration,
            'response': response,
            'components': current_components,
            'total_components': len(current_components)
        }
        
    except Exception as e:
        return {'error': str(e)}

def monitor_sds_log(workspace_dir):
    """Monitor the main SDS log file."""
    sds_log = workspace_dir / "sds.log"
    if not sds_log.exists():
        return "SDS log not found"
    
    try:
        with open(sds_log, 'r') as f:
            content = f.read()
        
            lines = content.splitlines()
        
        # Extract SDS progress information
        sds_info = []
        
        # Get total iterations and samples
        iteration_matches = re.findall(r'Running for (\d+) iterations', content)
        sample_matches = re.findall(r'Generating (\d+) reward function samples', content)
        
        if iteration_matches and sample_matches:
            total_iterations = iteration_matches[0]
            samples_per_iter = sample_matches[0]
            sds_info.append(f"Total: {total_iterations} iterations × {samples_per_iter} samples each")
        
        # Get current iteration
        current_iter_matches = re.findall(r'Iteration (\d+):', content)
        if current_iter_matches:
            current_iter = current_iter_matches[-1]
            sds_info.append(f"Current SDS iteration: {current_iter}")
        
        # Get current sample being processed
        current_sample_matches = re.findall(r'Starting training for response (\d+)', content)
        if current_sample_matches:
            current_sample = current_sample_matches[-1]
            sds_info.append(f"Current sample: {current_sample}")
        
        # Check training environments being used
        env_matches = re.findall(r'Training environments: (\d+)', content)
        if env_matches:
            train_envs = env_matches[0]
            sds_info.append(f"Training with {train_envs} environments")
        
        # Calculate progress if we have current iteration and sample info
        if current_iter_matches and current_sample_matches:
            current_iter = int(current_iter_matches[-1])
            current_sample = int(current_sample_matches[-1])
            
            if iteration_matches and sample_matches:
                total_iterations = int(iteration_matches[0])
                samples_per_iter = int(sample_matches[0])
            
                # Calculate overall progress
                completed_samples = current_iter * samples_per_iter + current_sample
                total_samples = total_iterations * samples_per_iter
                progress_percent = (completed_samples / total_samples) * 100
                
                sds_info.append(f"Overall progress: {completed_samples}/{total_samples} samples ({progress_percent:.1f}%)")
        
        # Show recent log lines
        if lines:
            sds_info.append("Recent logs:")
            latest_lines = lines[-3:]  # Show last 3 lines
            for line in latest_lines:
                if line.strip():  # Only show non-empty lines
                    sds_info.append(f"  {line.strip()}")
        
        return "SDS Status:\n" + "\n".join(sds_info)
        
    except Exception as e:
        return f"Error reading SDS log: {e}"

def monitor_isaaclab_training(workspace_dir):
    """Monitor IsaacLab training progress from training log files."""
    # Look for training output files
    training_files = list(workspace_dir.glob("env_iter*_response*.txt"))
    if not training_files:
        return "No training files found"
    
    # Always get the most recently modified file
    latest_file = max(training_files, key=lambda x: x.stat().st_mtime)
    
    try:
        # Re-read the file each time to get latest content
        with open(latest_file, 'r') as f:
            content = f.read()
        
        # Extract detailed training information
        info = []
        
        # Show file info with last modification time
        file_mtime = datetime.fromtimestamp(latest_file.stat().st_mtime)
        current_time = datetime.now()
        time_since_update = current_time - file_mtime
        
        info.append(f"Training file: {latest_file.name}")
        info.append(f"Last updated: {time_since_update.total_seconds():.0f}s ago")
        
        # Learning iteration progress
        learning_iter_matches = re.findall(r'Learning iteration (\d+)/(\d+)', content)
        if learning_iter_matches:
            current_iter, total_iter = learning_iter_matches[-1]
            info.append(f"Learning iteration: {current_iter}/{total_iter}")
        
        # Computation stats
        computation_matches = re.findall(r'Computation: (\d+) steps/s \(collection: ([\d.]+)s, learning ([\d.]+)s\)', content)
        if computation_matches:
            steps_per_sec, collection_time, learning_time = computation_matches[-1]
            info.append(f"Computation: {steps_per_sec} steps/s (collection: {collection_time}s, learning: {learning_time}s)")
        
        # Mean reward and episode length
        reward_matches = re.findall(r'Mean reward: ([-\d.]+)', content)
        if reward_matches:
            mean_reward = reward_matches[-1]
            info.append(f"Mean reward: {mean_reward}")
        
        episode_length_matches = re.findall(r'Mean episode length: ([\d.]+)', content)
        if episode_length_matches:
            episode_length = episode_length_matches[-1]
            info.append(f"Mean episode length: {episode_length}")
        
        # Loss information
        value_loss_matches = re.findall(r'Mean value_function loss: ([\d.]+)', content)
        if value_loss_matches:
            value_loss = value_loss_matches[-1]
            info.append(f"Value function loss: {value_loss}")
        
        surrogate_loss_matches = re.findall(r'Mean surrogate loss: ([\d.]+)', content)
        if surrogate_loss_matches:
            surrogate_loss = surrogate_loss_matches[-1]
            info.append(f"Surrogate loss: {surrogate_loss}")
        
        # Timing information
        total_timesteps_matches = re.findall(r'Total timesteps: (\d+)', content)
        if total_timesteps_matches:
            total_timesteps = total_timesteps_matches[-1]
            info.append(f"Total timesteps: {total_timesteps}")
        
        iteration_time_matches = re.findall(r'Iteration time: ([\d.]+)s', content)
        if iteration_time_matches:
            iter_time = iteration_time_matches[-1]
            info.append(f"Iteration time: {iter_time}s")
        
        time_elapsed_matches = re.findall(r'Time elapsed: ([\d:]+)', content)
        if time_elapsed_matches:
            time_elapsed = time_elapsed_matches[-1]
            info.append(f"Time elapsed: {time_elapsed}")
        
        eta_matches = re.findall(r'ETA: ([\d:]+)', content)
        if eta_matches:
            eta = eta_matches[-1]
            info.append(f"ETA: {eta}")
        
        # Key reward components (show all Isaac Lab tracked components)
        reward_components = re.findall(r'Episode_Reward/([^:]+): ([-\d.]+)', content)
        if reward_components:
            # Get latest reward components
            latest_rewards = {}
            for component, value in reward_components:
                latest_rewards[component] = float(value)
            
            if latest_rewards:
                info.append("Isaac Lab reward tracking:")
                # Sort by absolute value to show most significant components
                sorted_rewards = sorted(latest_rewards.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
                for component, value in sorted_rewards:
                    info.append(f"  {component}: {value:.4f}")
        
        # Termination info
        termination_matches = re.findall(r'Episode_Termination/([^:]+): ([\d.]+)', content)
        if termination_matches:
            info.append("Terminations:")
            for term_type, value in termination_matches[-3:]:  # Show last 3
                info.append(f"  {term_type}: {value}")
        
        # Check for completion
        if "Training completed" in content or "Saving model" in content:
            info.append("🎉 Training COMPLETED!")
        
        return "IsaacLab Training:\n" + "\n".join(f"  {line}" for line in info)
        
    except Exception as e:
        return f"Error reading training file: {e}"

def display_gpt_reward_components(workspace_dir):
    """Display GPT-generated reward components with their structure."""
    gpt_components = extract_gpt_reward_components(workspace_dir)
    
    if 'error' in gpt_components:
        return f"GPT Reward Analysis: Error - {gpt_components['error']}"
    
    if not gpt_components:
        return "GPT Reward Analysis: No reward files found"
    
    info = []
    info.append(f"📋 GPT Reward Function Analysis")
    info.append(f"   File: {gpt_components.get('file', 'Unknown')}")
    info.append(f"   Iteration: {gpt_components.get('iteration', '?')}, Response: {gpt_components.get('response', '?')}")
    info.append(f"   Components detected: {gpt_components.get('total_components', 0)}")
    
    components = gpt_components.get('components', [])
    if components and isinstance(components, list):
        info.append("   🧠 GPT-Generated Reward Components:")
        for i, comp in enumerate(components[:8], 1):  # Show max 8 components
            if isinstance(comp, dict):
                name = comp.get('name', 'unknown')
                calc = comp.get('calculation', 'unknown')
            else:
                # Fallback for unexpected data structure
                name = f"component_{i}"
                calc = str(comp) if comp else "unknown"
            
            # Truncate component name if too long
            if len(str(name)) > 25:
                name = str(name)[:22] + "..."
            info.append(f"      {i}. {name}")
            info.append(f"         └── {calc}")
        
        if len(components) > 8:
            remaining = len(components) - 8
            info.append(f"      ... and {remaining} more components")
    else:
        info.append("   ⚠️  No clear reward components detected in function")
    
    return "\n".join(info)

def get_gpu_usage():
    """Get GPU utilization information."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                util, mem_used, mem_total = line.split(', ')
                gpu_info.append(f"GPU {i}: {util}% util, {mem_used}/{mem_total}MB")
            return "GPU Status:\n" + "\n".join(f"  {info}" for info in gpu_info)
    except Exception as e:
        return f"GPU info unavailable: {e}"
    
    return "GPU info unavailable"

def get_isaaclab_processes():
    """Check for running IsaacLab processes."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        isaaclab_processes = [line for line in lines if 'isaaclab.sh' in line or 'train.py' in line]
        
        if isaaclab_processes:
            return f"IsaacLab Processes: {len(isaaclab_processes)} running"
        else:
            return "IsaacLab Processes: None running"
    except:
        return "Process check failed"

def main():
    parser = argparse.ArgumentParser(description='Monitor SDS training progress with GPT reward analysis')
    parser.add_argument('--workspace', type=str, help='SDS workspace directory to monitor')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds (default:5)')
    parser.add_argument('--show-gpt-details', action='store_true', help='Show detailed GPT reward component analysis')
    args = parser.parse_args()
    
    if args.workspace:
        workspace_dir = Path(args.workspace)
    else:
        workspace_dir = get_latest_workspace()
    
    if not workspace_dir or not workspace_dir.exists():
        print("❌ No SDS workspace found. Make sure SDS is running.")
        return
    
    print(f"📊 Monitoring SDS workspace: {workspace_dir}")
    print(f"🔄 Update interval: {args.interval} seconds")
    print(f"🧠 GPT analysis: {'Enabled' if args.show_gpt_details else 'Basic'}")
    print("=" * 80)
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Calculate total elapsed time
            sds_start_time = get_sds_start_time(workspace_dir)
            current_time = datetime.now()
            
            if sds_start_time:
                total_elapsed = current_time - sds_start_time
                elapsed_str = format_duration(total_elapsed)
            else:
                elapsed_str = "Unknown"
            
            # Header
            print(f"📊 SDS Training Monitor - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 Workspace: {workspace_dir}")
            print(f"⏱️  Total time elapsed: {elapsed_str}")
            print("=" * 80)
            
            # SDS Status
            print(monitor_sds_log(workspace_dir))
            print("-" * 80)
            
            # GPT Reward Components Analysis
            print(display_gpt_reward_components(workspace_dir))
            print("-" * 80)
            
            # IsaacLab Training Status
            print(monitor_isaaclab_training(workspace_dir))
            print("-" * 80)
            
            # GPU Status
            print(get_gpu_usage())
            print("-" * 80)
            
            # Process Status
            print(get_isaaclab_processes())
            print("-" * 80)
            
            print(f"🔄 Next update in {args.interval} seconds... (Ctrl+C to exit)")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")

if __name__ == "__main__":
    main() 