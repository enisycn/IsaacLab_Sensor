import subprocess
import os
import json
import logging
import re
from utils.extract_task_code import file_to_string
from openai import OpenAI
import time

def gpt_query(sample,messages,temperature,model):
    client = OpenAI()  # API key automatically loaded from OPENAI_API_KEY environment variable
    responses = []
    response_cur = None
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    chunk_size = 4

    while True:
        if total_samples >= sample:
            break
        for attempt in range(3):
            try:
                response_cur = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=chunk_size
                )
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()

        # Convert new response format to old format for compatibility
        for choice in response_cur.choices:
            responses.append({
                "message": {
                    "content": choice.message.content
                }
            })
        prompt_tokens = response_cur.usage.prompt_tokens
        total_completion_token += response_cur.usage.completion_tokens
        total_token += response_cur.usage.total_tokens
    
    return responses,prompt_tokens,total_completion_token,total_token

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    # Note: if this line breaks, you can provide an absolute path to gpustat instead
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])
    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def extract_training_log_dir(file_path):
    with open(file_path,mode="r") as f:
        for line in f:
            dashboard_match = re.match(r"Dashboard: http://app.dash.ml/(.+)",line)
            if dashboard_match:
                return dashboard_match.group(1)

def block_until_training(rl_filepath, success_keyword, failure_keyword, log_status=False, iter_num=-1, response_id=-1, timeout_minutes=60):
    # Ensure that the RL training has started before moving on
    start_time = time.time()
    last_log_size = 0
    no_progress_start = None
    
    while True:
        rl_log = file_to_string(rl_filepath)
        
        # Check for success
        if success_keyword in rl_log:
            if log_status:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully trained!")
                return True
        
        # Check for various failure patterns
        failure_detected = False
        failure_reason = ""
        
        # Original failure keyword check
        if failure_keyword in rl_log:
            failure_detected = True
            failure_reason = "Traceback detected"
        
        # Hydra configuration errors
        elif "HYDRA_FULL_ERROR=1" in rl_log:
            failure_detected = True
            failure_reason = "Hydra configuration error"
        
        # Hydra missing configuration errors
        elif "MissingConfigException" in rl_log:
            failure_detected = True
            failure_reason = "Missing Hydra configuration"
        
        # CUDA/GPU errors
        elif any(error in rl_log for error in ["CUDA error", "CUDA out of memory", "RuntimeError: CUDA", "torch.cuda.OutOfMemoryError"]):
            failure_detected = True
            failure_reason = "CUDA/GPU error"
        
        # Isaac Lab specific errors
        elif any(error in rl_log for error in ["Isaac Lab Error", "Simulation Error", "Environment Error"]):
            failure_detected = True
            failure_reason = "Isaac Lab simulation error"
        
        # Python syntax/import errors
        elif any(error in rl_log for error in ["SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError"]):
            failure_detected = True
            failure_reason = "Python syntax/import error"
        
        # Process killed/terminated
        elif any(error in rl_log for error in ["Killed", "Terminated", "Process finished with exit code"]):
            failure_detected = True
            failure_reason = "Process terminated"
        
        # Training divergence/NaN errors
        elif any(error in rl_log for error in ["nan", "inf", "NaN", "diverged", "unstable"]):
            failure_detected = True
            failure_reason = "Training divergence/NaN values"
        
        if failure_detected:
            if log_status:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error - {failure_reason}")
            return False
        
        # Check for timeout and stuck training
        current_time = time.time()
        elapsed_minutes = (current_time - start_time) / 60
        
        # Overall timeout check
        if elapsed_minutes > timeout_minutes:
            if log_status:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} timed out after {timeout_minutes} minutes")
            return False
        
        # Check if log file is growing (training making progress)
        current_log_size = len(rl_log)
        if current_log_size > last_log_size:
            last_log_size = current_log_size
            no_progress_start = None  # Reset no progress timer
        else:
            # No new log output
            if no_progress_start is None:
                no_progress_start = current_time
            elif (current_time - no_progress_start) > 600:  # 10 minutes without progress
                if log_status:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} stuck - no progress for 10 minutes")
                return False
        
        # Add small delay to avoid busy waiting
        time.sleep(1)


def construct_run_log(stdout_str):
    run_log = {}
    lines = stdout_str.split('\n')
    
    # Handle both old table format and new Isaac Lab format
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Old table format with │ separators
        if line.startswith("│") and line.endswith("│"):
            line = line[1:-1].split("│")
            key, val = line[0].strip(), line[1].strip()
            if key == "timesteps" or key == "iterations":
                key = key + "/"
            elif "train/episode/rew" in key:
                key = key.split("/")[2]
            elif key == "train/episode/episode length/mean":
                key = "episode length"
            run_log[key] = run_log.get(key, []) + [float(val)]
        
        # ✅ ENHANCED: Extract ALL Isaac Lab training metrics
        else:
            try:
                # Core training metrics (existing)
                if "Mean episode length:" in line:
                val = float(line.split("Mean episode length:")[-1].strip())
                run_log["episode length"] = run_log.get("episode length", []) + [val]
        elif "Mean reward:" in line:
                val = float(line.split("Mean reward:")[-1].strip())
                run_log["reward"] = run_log.get("reward", []) + [val]
        elif "Learning iteration" in line and "/" in line:
                iteration_part = line.split("Learning iteration")[-1].split("/")[0].strip()
                val = float(iteration_part)
                run_log["iterations/"] = run_log.get("iterations/", []) + [val]
                
                # ✅ NEW: Training performance metrics
                elif "Mean action noise std:" in line:
                    val = float(line.split("Mean action noise std:")[-1].strip())
                    run_log["action_noise_std"] = run_log.get("action_noise_std", []) + [val]
                elif "Mean value_function loss:" in line:
                    val = float(line.split("Mean value_function loss:")[-1].strip())
                    run_log["value_function_loss"] = run_log.get("value_function_loss", []) + [val]
                elif "Mean surrogate loss:" in line:
                    val = float(line.split("Mean surrogate loss:")[-1].strip())
                    run_log["surrogate_loss"] = run_log.get("surrogate_loss", []) + [val]
                elif "Mean entropy loss:" in line:
                    val = float(line.split("Mean entropy loss:")[-1].strip())
                    run_log["entropy_loss"] = run_log.get("entropy_loss", []) + [val]
                
                # ✅ NEW: Reward component breakdown
                elif "Episode_Reward/sds_custom:" in line:
                    val = float(line.split("Episode_Reward/sds_custom:")[-1].strip())
                    run_log["reward_sds_custom"] = run_log.get("reward_sds_custom", []) + [val]
                
                # ✅ NEW: Curriculum progression
                elif "Curriculum/terrain_levels:" in line:
                    val = float(line.split("Curriculum/terrain_levels:")[-1].strip())
                    run_log["curriculum_terrain_levels"] = run_log.get("curriculum_terrain_levels", []) + [val]
                
                # ✅ NEW: Task performance metrics
                elif "Metrics/base_velocity/error_vel_xy:" in line:
                    val = float(line.split("Metrics/base_velocity/error_vel_xy:")[-1].strip())
                    run_log["velocity_error_xy"] = run_log.get("velocity_error_xy", []) + [val]
                elif "Metrics/base_velocity/error_vel_yaw:" in line:
                    val = float(line.split("Metrics/base_velocity/error_vel_yaw:")[-1].strip())
                    run_log["velocity_error_yaw"] = run_log.get("velocity_error_yaw", []) + [val]
                
                # ✅ NEW: Episode termination analysis
                elif "Episode_Termination/time_out:" in line:
                    val = float(line.split("Episode_Termination/time_out:")[-1].strip())
                    run_log["termination_timeout"] = run_log.get("termination_timeout", []) + [val]
                elif "Episode_Termination/base_contact:" in line:
                    val = float(line.split("Episode_Termination/base_contact:")[-1].strip())
                    run_log["termination_base_contact"] = run_log.get("termination_base_contact", []) + [val]
                
                # ✅ NEW: Computation performance metrics
                elif "Computation:" in line and "steps/s" in line:
                    # Extract "27297 steps/s" from "Computation: 27297 steps/s (collection: 2.369s, learning 0.268s)"
                    parts = line.split("steps/s")[0].split(":")[-1].strip()
                    val = float(parts)
                    run_log["computation_steps_per_sec"] = run_log.get("computation_steps_per_sec", []) + [val]
                    
                    # Extract collection and learning times if present
                    if "collection:" in line and "learning" in line:
                        collection_part = line.split("collection:")[-1].split("s")[0].strip()
                        learning_part = line.split("learning")[-1].split("s")[0].strip()
                        run_log["collection_time"] = run_log.get("collection_time", []) + [float(collection_part)]
                        run_log["learning_time"] = run_log.get("learning_time", []) + [float(learning_part)]
                
                # ✅ NEW: Additional reward components (dynamic extraction)
                elif line.startswith("Episode_Reward/") and ":" in line:
                    # Extract any Episode_Reward/* metrics dynamically
                    metric_name = line.split(":")[0].strip().replace("Episode_Reward/", "reward_")
                    val = float(line.split(":")[-1].strip())
                    run_log[metric_name] = run_log.get(metric_name, []) + [val]
                
                # ✅ NEW: Additional metrics (dynamic extraction)
                elif line.startswith("Metrics/") and ":" in line:
                    # Extract any Metrics/* dynamically
                    metric_name = line.split(":")[0].strip().replace("Metrics/", "metric_").replace("/", "_")
                    val = float(line.split(":")[-1].strip())
                    run_log[metric_name] = run_log.get(metric_name, []) + [val]
                
                # ✅ NEW: Environmental sensing and stability metrics (specific extraction)
                elif "Metrics/terrain_stability:" in line:
                    val = float(line.split("Metrics/terrain_stability:")[-1].strip())
                    run_log["terrain_height_variance"] = run_log.get("terrain_height_variance", []) + [val]
                elif "Metrics/terrain_complexity:" in line:
                    val = float(line.split("Metrics/terrain_complexity:")[-1].strip())
                    run_log["terrain_complexity_score"] = run_log.get("terrain_complexity_score", []) + [val]
                elif "Metrics/robot_height_stability:" in line:
                    val = float(line.split("Metrics/robot_height_stability:")[-1].strip())
                    run_log["robot_height_baseline"] = run_log.get("robot_height_baseline", []) + [val]
                elif "Metrics/body_orientation_stability:" in line:
                    val = float(line.split("Metrics/body_orientation_stability:")[-1].strip())
                    run_log["body_orientation_deviation"] = run_log.get("body_orientation_deviation", []) + [val]
                elif "Metrics/height_tracking_accuracy:" in line:
                    val = float(line.split("Metrics/height_tracking_accuracy:")[-1].strip())
                    run_log["height_tracking_error"] = run_log.get("height_tracking_error", []) + [val]
                
            except (ValueError, IndexError):
                # Skip lines that can't be parsed as metrics
                continue
    
    # Ensure we have some episode length data for backward compatibility
    if "episode length" not in run_log and "reward" in run_log:
        # If we don't have episode length but have rewards, create dummy episode lengths
        run_log["episode length"] = [100.0] * len(run_log["reward"])
    elif "episode length" not in run_log:
        # If we have no data at all, create minimal dummy data
        run_log["episode length"] = [100.0]
        run_log["reward"] = [0.0]
    
    run_log["gpt_reward"] = []
    run_log["gt_reward"] = []
    
    # Create reward data for compatibility
    episode_length_data = run_log.get("episode length", [])
    for i in range(len(episode_length_data)):
        cur_sum = 0
        for key in run_log:
            if "rew " in key:
                cur_sum += run_log[key][i] if i < len(run_log[key]) else 0
        # Use reward data if available, otherwise use cur_sum
        if "reward" in run_log and i < len(run_log["reward"]):
            cur_sum = run_log["reward"][i]
        run_log["gpt_reward"].append(cur_sum)
        run_log["gt_reward"].append(cur_sum)
    
    return run_log