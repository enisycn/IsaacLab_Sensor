import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import re
import subprocess
from pathlib import Path
import shutil
import torch
import gc
import glob
from utils.misc import * 
from utils.extract_task_code import *
from utils.vid_utils import create_grid_image,encode_image,save_grid_image
from utils.easy_vit_pose import vitpose_inference
import cv2
import os
from agents import SUSGenerator, EnhancedSUSGenerator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def find_isaac_lab_root():
    """Find Isaac Lab root directory by looking for isaaclab.sh."""
    from pathlib import Path
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "isaaclab.sh").exists():
            return str(parent)
    raise FileNotFoundError("Could not find Isaac Lab root directory (isaaclab.sh not found)")

# Get SDS root directory from script location, not working directory
# This is important because Hydra changes the working directory
SDS_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = f"{SDS_ROOT_DIR}/.."

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {SDS_ROOT_DIR}")
    logging.info(f"Running for {cfg.iteration} iterations")
    logging.info(f"Training each RF for: {cfg.train_iterations} iterations")
    logging.info(f"Generating {cfg.sample} reward function samples per iteration")

    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info(f"Imitation Task: {cfg.task.description}")

    env_name = cfg.env_name.lower()

    # Updated for Isaac Lab integration - use Isaac Lab environment description
    isaac_lab_root = find_isaac_lab_root()
    task_rew_file = f'{isaac_lab_root}/source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/mdp/rewards.py'
    task_obs_file = f'{SDS_ROOT_DIR}/SDS/envs/isaac_lab_sds_env.py'  # Use Isaac Lab environment description
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_rew_code_string = file_to_string(task_rew_file)
    task_obs_code_string = file_to_string(task_obs_file)
    output_file = task_rew_file  # Write directly to Isaac Lab reward file

    # Loading all text prompts
    prompt_dir = f'{SDS_ROOT_DIR}/SDS/prompts'
    initial_reward_engineer_system = file_to_string(f'{prompt_dir}/initial_reward_engineer_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_reward_engineer_user = file_to_string(f'{prompt_dir}/initial_reward_engineer_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signatures/{env_name}.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    initial_task_evaluator_system = file_to_string(f'{prompt_dir}/initial_task_evaluator_system.txt')
    
    demo_video_name = cfg.task.video
    video_do_crop = cfg.task.crop
    logging.info(f"Demonstration Video: {demo_video_name}, Crop Option: {cfg.task.crop_option}")
    gt_frame_grid = create_grid_image(f'{SDS_ROOT_DIR}/videos/{demo_video_name}',grid_size=(cfg.task.grid_size,cfg.task.grid_size),crop=video_do_crop,crop_option=cfg.task.crop_option)
    save_grid_image(gt_frame_grid,"gt_demo.png")
    
    annotated_video_path = vitpose_inference(f'{SDS_ROOT_DIR}/videos/{demo_video_name}',f"{workspace_dir}/pose-estimate/gt-pose-estimate")
    gt_annotated_frame_grid = create_grid_image(annotated_video_path,grid_size=(cfg.task.grid_size,cfg.task.grid_size),crop=video_do_crop,crop_option=cfg.task.crop_option)
    save_grid_image(gt_annotated_frame_grid,"gt_demo_annotated.png")
    
    eval_script_dir = os.path.join(ROOT_DIR,"forward_locomotion_sds/scripts/play.py")
    
    encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo.png')
    
    # Choose SUS generator based on configuration
    enable_env_analysis = getattr(cfg, 'enable_environment_analysis', True)
    
    if enable_env_analysis:
        # Use enhanced SUS generator with environment awareness
        sus_generator = EnhancedSUSGenerator(cfg, prompt_dir)
        
        # Generate environment-aware SUS prompt
        num_envs_for_analysis = getattr(cfg, 'environment_analysis_robots', 50)
        logging.info(f"Using environment-aware SUS generation with {num_envs_for_analysis} robots for terrain analysis")
        
        SUS_prompt = sus_generator.generate_enhanced_sus_prompt(
            encoded_gt_frame_grid, 
            cfg.task.description,
            num_envs=num_envs_for_analysis
        )
    else:
        # Use original SUS generator (video-only analysis)
        sus_generator = SUSGenerator(cfg, prompt_dir)
        logging.info("Using original SUS generation (video-only analysis)")
        
        SUS_prompt = sus_generator.generate_sus_prompt(
            encoded_gt_frame_grid, 
            cfg.task.description
        )
    
    initial_reward_engineer_system = initial_reward_engineer_system.format(task_reward_signature_string=reward_signature,task_obs_code_string=task_obs_code_string) + code_output_tip

    initial_reward_engineer_user = initial_reward_engineer_user.format(sus_string=SUS_prompt,task_obs_code_string=task_obs_code_string)
    
    initial_task_evaluator_system = initial_task_evaluator_system.format(sus_string=SUS_prompt)


    reward_query_messages = [
        {"role": "system", "content": initial_reward_engineer_system}, 
        {"role": "user", "content": [
          {
            "type": "text",
            "text": initial_reward_engineer_user 
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/png;base64,{encoded_gt_frame_grid}",
              "detail": cfg.image_quality
            }
          }
        ]
        }
    ]

    os.mkdir(f"{workspace_dir}/training_footage")
    os.mkdir(f"{workspace_dir}/contact_sequence")

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    # execute_rates = []
    best_code_paths = []
    max_reward_code_path = None 
    
    best_footage = None
    best_contact = None

    for iter in range(cfg.iteration):

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        responses,prompt_tokens,total_completion_token,total_token = gpt_query(cfg.sample,reward_query_messages,cfg.temperature,cfg.model)
        
        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")

        code_runs = [] 
        rl_runs = []
        footage_grids_dir = []
        contact_pattern_dirs = []
        
        successful_runs_index = []
        
        eval_success = False

        for response_id in range(cfg.sample):
            import re  # Move import to the beginning
            response_cur = responses[response_id]["message"]["content"]
            # print(response_cur)
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            # Find the start of the function definition (skip imports/comments)
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    break
            
            
            def ensure_proper_indentation(code_str):
                """Ensure the function has proper Python indentation (0 for def, preserve nested structure)."""
                lines = code_str.splitlines()
                
                if not lines:
                    return code_str
                
                # Check if first line is the function definition
                first_line = lines[0].strip()
                if not first_line.startswith('def sds_custom_reward'):
                    raise ValueError(f"Generated code must start with 'def sds_custom_reward', got: {first_line}")

                adjusted_lines = []
                
                # Find the base indentation level (function body level)
                base_indent = None
                for line in lines[1:]:  # Skip the def line
                    if line.strip():  # First non-empty line after def
                        base_indent = len(line) - len(line.lstrip())
                        break
                
                if base_indent is None:
                    base_indent = 4  # Default to 4 if no body found
                
                for i, line in enumerate(lines):
                    if i == 0:
                        # Function definition at file level (no indentation)
                        adjusted_lines.append(first_line)
                    elif line.strip():  # Non-empty line
                        # Calculate current indentation relative to base
                        current_indent = len(line) - len(line.lstrip())
                        # Preserve relative indentation, but ensure function body starts at 4 spaces
                        relative_indent = max(0, current_indent - base_indent)
                        new_indent = 4 + relative_indent
                        adjusted_lines.append(" " * new_indent + line.strip())
                    else:
                        # Empty line
                        adjusted_lines.append("")
                
                return '\n'.join(adjusted_lines)
            
            code_string = ensure_proper_indentation(code_string)
            code_runs.append(code_string)
                    
            # Add the SDS Reward Signature to the environment code
            # For Isaac Lab integration, we need to replace the entire sds_custom_reward function
            
            # ROBUST replacement pattern that matches the actual Isaac Lab function signature
            # This pattern matches the function from def to the end marker
            pattern = r'def sds_custom_reward\(env\).*?return reward.*?(?=\n\n# SDS_FUNCTION_END_MARKER|\n\n\n|\nclass|\ndef |\Z)'
            
            # Verify the GPT generated a proper Isaac Lab function
            if not code_string.strip().startswith('def sds_custom_reward'):
                logging.error(f"GPT did not generate a proper sds_custom_reward function. Generated: {code_string[:100]}...")
                raise ValueError("Invalid GPT reward function - must start with 'def sds_custom_reward'")
            
            # Perform the replacement with the robust pattern
            replacement_count = 0
            cur_task_rew_code_string = re.sub(pattern, code_string.strip(), task_rew_code_string, flags=re.DOTALL, count=1)
            
            # Verify replacement was successful by checking if the new code is different
            if cur_task_rew_code_string == task_rew_code_string:
                # If regex failed, try a more aggressive approach
                logging.warning("Primary regex replacement failed, trying alternative replacement method")
                
                # Find the function start and end manually
                start_marker = "def sds_custom_reward(env) -> torch.Tensor:"
                if start_marker in task_rew_code_string:
                    start_idx = task_rew_code_string.find(start_marker)
                    
                    # Find the end of the function by looking for the next function or end marker
                    temp_str = task_rew_code_string[start_idx:]
                    lines = temp_str.split('\n')
                    
                    end_idx = start_idx
                    function_lines = 1  # Count the def line
                    
                    # Find where the function ends (next def/class or significant dedent)
                    for i, line in enumerate(lines[1:], 1):
                        if line.strip() == "# INSERT SDS REWARD HERE":
                            break
                        elif line.startswith('def ') or line.startswith('class ') or (line.strip() and not line.startswith(' ') and not line.startswith('\t')):
                            break
                        function_lines += 1
                        end_idx += len(line) + 1  # +1 for newline
                    
                    # Replace the function
                    before = task_rew_code_string[:start_idx]
                    after = task_rew_code_string[start_idx + end_idx:]
                    cur_task_rew_code_string = before + code_string.strip() + '\n\n' + after
                    logging.info("Alternative replacement method succeeded")
                else:
                    raise ValueError("Could not find sds_custom_reward function in rewards.py to replace")
            else:
                logging.info("Primary regex replacement succeeded")
                replacement_count = 1
            
            # Final verification that replacement occurred
            if code_string.strip() not in cur_task_rew_code_string:
                raise ValueError("Reward function replacement failed - GPT code not found in output")

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(cur_task_rew_code_string + '\n')

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()
            
            # Execute Isaac Lab training - NO FALLBACKS, Isaac Lab only
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            isaac_lab_root = find_isaac_lab_root()
            
            # Use Isaac Lab training command with G1 humanoid task
            # Note: SDS_ANALYSIS_MODE is NOT set, so gravity will be ENABLED for training
            command = [
                f"{isaac_lab_root}/isaaclab.sh",
                "-p", "scripts/reinforcement_learning/rsl_rl/train.py",
                f"--task=Isaac-SDS-Velocity-Flat-G1-Enhanced-v0",
                f"--num_envs={cfg.num_envs}",
                f"--max_iterations={cfg.train_iterations}",
                "--headless"
            ]
            
            logging.info(f"Running Isaac Lab training: {' '.join(command)}")
            
            # SIMPLE RETRY MECHANISM: Try up to 3 times per sample
            training_successful = False
            max_retries = 2  # 3 total attempts (original + 2 retries)
            
            for retry_attempt in range(max_retries + 1):
                # Generate new reward function on retry
                if retry_attempt > 0:
                    logging.info(f"Sample {response_id} failed, generating new reward function (retry {retry_attempt}/{max_retries})")
                    try:
                        # Generate new reward function from GPT
                        new_responses, _, _, _ = gpt_query(1, reward_query_messages, cfg.temperature, cfg.model)
                        new_response_cur = new_responses[0]["message"]["content"]
                        
                        # Extract and process new code the same way as original
                        patterns = [
                            r'```python(.*?)```',
                            r'```(.*?)```',
                            r'"""(.*?)"""',
                            r'""(.*?)""',
                            r'"(.*?)"',
                        ]
                        new_code_string = None
                        for pattern in patterns:
                            match = re.search(pattern, new_response_cur, re.DOTALL)
                            if match is not None:
                                new_code_string = match.group(1).strip()
                                break
                        new_code_string = new_response_cur if not new_code_string else new_code_string
                        
                        # Process new code the same way
                        lines = new_code_string.split("\n")
                        # Find the start of the function definition (skip imports/comments)
                        for i, line in enumerate(lines):
                            if line.strip().startswith("def "):
                                new_code_string = "\n".join(lines[i:])
                                break
                        
                        new_code_string = ensure_proper_indentation(new_code_string)
                        
                        # Replace the failed code with new one
                        code_runs[response_id] = new_code_string
                        
                        # Update environment code with new reward function
                        if not new_code_string.strip().startswith('def sds_custom_reward'):
                            raise ValueError("New reward function must start with 'def sds_custom_reward'")
                        
                        # Replace function in environment code
                        pattern = r'def sds_custom_reward\(env\).*?return reward.*?(?=\n\n# SDS_FUNCTION_END_MARKER|\n\n\n|\nclass|\ndef |\Z)'
                        cur_task_rew_code_string = re.sub(pattern, new_code_string.strip(), task_rew_code_string, flags=re.DOTALL, count=1)
                        
                        if cur_task_rew_code_string == task_rew_code_string:
                            # Fallback replacement method
                            start_marker = "def sds_custom_reward(env) -> torch.Tensor:"
                            if start_marker in task_rew_code_string:
                                start_idx = task_rew_code_string.find(start_marker)
                                temp_str = task_rew_code_string[start_idx:]
                                lines = temp_str.split('\n')
                                end_idx = start_idx
                                for i, line in enumerate(lines[1:], 1):
                                    if line.strip() == "# INSERT SDS REWARD HERE":
                                        break
                                    elif line.startswith('def ') or line.startswith('class ') or (line.strip() and not line.startswith(' ') and not line.startswith('\t')):
                                        break
                                    end_idx += len(line) + 1
                                before = task_rew_code_string[:start_idx]
                                after = task_rew_code_string[start_idx + end_idx:]
                                cur_task_rew_code_string = before + new_code_string.strip() + '\n\n' + after
                        
                        # Save new environment code
                        with open(output_file, 'w') as file:
                            file.writelines(cur_task_rew_code_string + '\n')
                        
                        with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                            file.writelines(new_code_string + '\n')
                        
                        shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")
                        
                        logging.info(f"Generated new reward function for sample {response_id} retry {retry_attempt}")
                    
                    except Exception as e:
                        logging.error(f"Failed to generate new reward function for sample {response_id}: {e}")
                        break  # Skip this sample if we can't generate new code
                
                # Try training with current reward function
                # Set HYDRA_FULL_ERROR=1 to ensure all Hydra errors show full stack trace for better detection
                training_env = os.environ.copy()
                training_env['HYDRA_FULL_ERROR'] = '1'
                
                with open(rl_filepath, 'w') as f:
                    process = subprocess.run(command, stdout=f, stderr=f, cwd=isaac_lab_root, env=training_env)
                
                if process.returncode == 0:
                    # Training succeeded
                    training_successful = True
                    logging.info(f"Sample {response_id} training succeeded" + (f" (retry {retry_attempt})" if retry_attempt > 0 else ""))
                    break
                else:
                    # Training failed
                    logging.error(f"Sample {response_id} training failed (attempt {retry_attempt + 1})")
                    with open(rl_filepath, 'r') as f:
                        error_content = f.read()
                        logging.error(f"Training error: {error_content[-500:]}")  # Last 500 chars
                    
                    # GPU cleanup after failure
                    try:
                        import time
                        time.sleep(2)
                        if torch.cuda.is_available():
                            device = torch.cuda.current_device()
                            torch.cuda.synchronize(device)
                            torch.cuda.empty_cache()
                            logging.info(f"GPU cache cleared after failure for sample {response_id}")
                    except Exception as e:
                        logging.warning(f"GPU cleanup failed: {e}")
                    
                    # If this was the last retry, log final failure
                    if retry_attempt == max_retries:
                        logging.error(f"Sample {response_id} failed after {max_retries + 1} attempts, skipping")
            
            # Only proceed if training was successful
            training_success = False  # Initialize to prevent NameError
            if training_successful:
                training_success = block_until_training(rl_filepath, success_keyword=cfg.success_keyword, failure_keyword=cfg.failure_keyword,
                                     log_status=True, iter_num=iter, response_id=response_id)
                rl_runs.append(process)
            else:
                # Add placeholder for failed sample to maintain indexing
                rl_runs.append(None)
                logging.warning(f"Sample {response_id} failed all attempts, added placeholder")
            
            # REDUCED FREQUENCY: Only clear GPU cache after training, not after every subprocess
            # Let Isaac Lab fully complete before clearing cache
            import time
            time.sleep(1)  # Give Isaac Lab time to finish any background operations
            
            try:
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    torch.cuda.synchronize(device)  # Ensure all CUDA operations complete first
                    torch.cuda.empty_cache()
                    logging.info(f"GPU cache cleared after training completion for sample {response_id}")
                    
                    # Log GPU memory usage immediately after cleanup
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        vram_used = result.stdout.strip()
                        logging.info(f"GPU VRAM after training cleanup for sample {response_id}: {vram_used}MB")
            except Exception as e:
                logging.warning(f"GPU cleanup failed after training: {e}")
            
            if training_success:
                # Isaac Lab post-training evaluation and analysis - NO FALLBACKS
                isaac_lab_root = find_isaac_lab_root()
                logs_dir = os.path.join(isaac_lab_root, "logs", "rsl_rl")
                
                # Get the latest experiment directory from Isaac Lab - find the most recent task directory
                task_dirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
                if not task_dirs:
                    raise RuntimeError(f"No Isaac Lab task directories found in {logs_dir}/")
                
                # Get the most recently modified task directory (should be the one we just trained)
                latest_task_dir = max(task_dirs, key=lambda d: os.path.getmtime(os.path.join(logs_dir, d)))
                experiment_dirs = glob.glob(os.path.join(logs_dir, latest_task_dir, "*"))
                if not experiment_dirs:
                    raise RuntimeError(f"No Isaac Lab experiment directories found in {logs_dir}/{latest_task_dir}/")
                
                latest_experiment = max(experiment_dirs, key=os.path.getmtime)
                logging.info(f"Using latest experiment directory: {latest_experiment}")
                
                # Find the latest checkpoint
                checkpoint_files = glob.glob(os.path.join(latest_experiment, "model_*.pt"))
                if not checkpoint_files:
                    raise RuntimeError(f"No checkpoints found in {latest_experiment}")
                
                latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                logging.info(f"Using latest checkpoint: {latest_checkpoint}")
                
                # Isaac Lab evaluation with video recording
                eval_command = [
                    f"{isaac_lab_root}/isaaclab.sh",
                    "-p", "scripts/reinforcement_learning/rsl_rl/play.py",
                    "--task=Isaac-SDS-Velocity-Flat-G1-Enhanced-Play-v0",
                    "--num_envs=200",
                    f"--checkpoint={latest_checkpoint}",
                    "--video",
                    f"--video_length={cfg.video_length}",
                    "--headless"
                ]
                
                logging.info(f"Running video generation: {' '.join(eval_command)}")
                # Set HYDRA_FULL_ERROR=1 for evaluation as well
                eval_env = os.environ.copy()
                eval_env['HYDRA_FULL_ERROR'] = '1'
                
                result = subprocess.run(eval_command, cwd=isaac_lab_root, capture_output=True, text=True, env=eval_env)
                if result.returncode != 0:
                    logging.error(f"Video generation failed: {result.stderr}")
                    
                    # GPU cleanup on video generation failure
                    try:
                        import time
                        time.sleep(1)
                        if torch.cuda.is_available():
                            device = torch.cuda.current_device()
                            torch.cuda.synchronize(device)
                            torch.cuda.empty_cache()
                            logging.info(f"GPU cache cleared after video generation failure for sample {response_id}")
                    except Exception as e:
                        logging.warning(f"GPU cleanup failed after video generation failure: {e}")
                    
                    raise RuntimeError(f"Video generation failed for iteration {iter}, response {response_id}")
                else:
                    logging.info("Video generation completed successfully")
                
                # Generate contact analysis using Isaac Lab contact plotting
                contact_command = [
                    f"{isaac_lab_root}/isaaclab.sh",
                    "-p", "scripts/reinforcement_learning/rsl_rl/play_with_contact_plotting.py",
                    "--task=Isaac-SDS-Velocity-Flat-G1-Enhanced-Play-v0",
                    "--num_envs=1",
                    f"--checkpoint={latest_checkpoint}",
                    f"--plot_steps={cfg.video_length}",
                    "--contact_threshold=50.0",
                    "--warmup_steps=50",
                    "--save_contacts",
                    "--headless"
                ]
                
                logging.info(f"Running contact analysis: {' '.join(contact_command)}")
                # Set HYDRA_FULL_ERROR=1 for contact analysis as well
                contact_env = os.environ.copy()
                contact_env['HYDRA_FULL_ERROR'] = '1'
                
                result = subprocess.run(contact_command, cwd=isaac_lab_root, capture_output=True, text=True, env=contact_env)
                if result.returncode != 0:
                    logging.error(f"Contact analysis failed: {result.stderr}")
                    
                    # GPU cleanup on contact analysis failure
                    try:
                        import time
                        time.sleep(1)
                        if torch.cuda.is_available():
                            device = torch.cuda.current_device()
                            torch.cuda.synchronize(device)
                            torch.cuda.empty_cache()
                            logging.info(f"GPU cache cleared after contact analysis failure for sample {response_id}")
                    except Exception as e:
                        logging.warning(f"GPU cleanup failed after contact analysis failure: {e}")
                    
                    raise RuntimeError(f"Contact analysis failed for iteration {iter}, response {response_id}")
                else:
                    logging.info("Contact analysis completed successfully")
                
                # Set Isaac Lab result paths
                full_training_log_dir = latest_experiment
                training_footage_dir = os.path.join(latest_experiment, "videos")
                contact_pattern_dir = os.path.join(latest_experiment, "contact_analysis", "contact_sequence.png")
                    
                try:
                    # Find video file - Isaac Lab saves videos in videos/play/ subdirectory
                    video_play_dir = os.path.join(training_footage_dir, "play")
                    
                    if not os.path.exists(video_play_dir):
                        raise RuntimeError(f"Video play directory not found: {video_play_dir}")
                    
                    video_files = glob.glob(os.path.join(video_play_dir, "*.mp4"))
                    if not video_files:
                        raise RuntimeError(f"No video files found in {video_play_dir}")
                    
                    video_file = max(video_files, key=os.path.getmtime)  # Use the newest video file
                    logging.info(f"Found Isaac Lab video file: {video_file}")
                    
                    annotated_video_path = vitpose_inference(video_file, f"{workspace_dir}/pose-estimate/sample-pose-estimate")
                    training_frame_grid = create_grid_image(annotated_video_path, crop=True, crop_option="RobotFocus", training_fixed_length=True)
                    
                    footage_grid_save_dir = f"training_footage/training_frame_{iter}_{response_id}.png"
                    save_grid_image(training_frame_grid, footage_grid_save_dir)
                    
                    contact_sequence_save_dir = f"{workspace_dir}/contact_sequence/contact_sequence_{iter}_{response_id}.png"
                    
                    # Verify contact pattern exists from Isaac Lab contact analysis
                    if not os.path.exists(contact_pattern_dir):
                        raise RuntimeError(f"Contact analysis file not found: {contact_pattern_dir}")
                    
                    shutil.copy(contact_pattern_dir, contact_sequence_save_dir)
                    logging.info(f"Copied Isaac Lab contact pattern: {contact_pattern_dir} -> {contact_sequence_save_dir}")
                    
                    footage_grids_dir.append(footage_grid_save_dir)
                    contact_pattern_dirs.append(contact_sequence_save_dir)

                    successful_runs_index.append(response_id)
                    eval_success = True

                except Exception as e:
                    # No footages saved due to reward run time error
                    logging.info(f"Iteration {iter}: Code Run {response_id} Failed to Evaluate: {str(e)}")
            else:
                logging.info(f"Iteration {iter}: Code Run {response_id} Unstable, Not evaluated")
        
            # CONSERVATIVE GPU MEMORY CLEANUP - Only run once per sample with proper timing
            try:
                # Give Isaac Lab processes time to fully complete
                import time
                time.sleep(2)  # Allow any background processes to finish
                
                # Clear PyTorch GPU cache only once per sample
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    torch.cuda.synchronize(device)  # Wait for all CUDA operations to complete
                    torch.cuda.empty_cache()
                    logging.info(f"Final GPU cache cleared after sample {response_id}")
                
                # Force garbage collection only after GPU cleanup
                gc.collect()
                
                # Log current GPU memory usage with error handling
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=10)  # Increased timeout
                    if result.returncode == 0:
                        vram_used = result.stdout.strip()
                        logging.info(f"Final GPU VRAM after sample {response_id}: {vram_used}MB")
                    else:
                        logging.warning(f"nvidia-smi query failed with return code {result.returncode}")
                except subprocess.TimeoutExpired:
                    logging.warning(f"nvidia-smi query timed out for sample {response_id}")
                except Exception as nvidia_e:
                    logging.warning(f"nvidia-smi query failed: {nvidia_e}")
                    
            except Exception as e:
                logging.warning(f"Final GPU memory cleanup failed: {e}")
        
        # Repeat the iteration if all code generation failed
        if not eval_success and cfg.sample != 1:
            # execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code evaluation failed! Repeat this iteration from the current message checkpoint!")
            continue
        

        code_feedbacks = []
        contents = []
        reward_correlations = []
        code_paths = []
        
        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            
            # SAFETY: Skip failed samples (where rl_run is None)
            if rl_run is None:
                content = execution_error_feedback.format(traceback_msg="Sample failed during training after all retry attempts!")
                content += code_output_tip
                contents.append(content) 
                reward_correlations.append(DUMMY_FAILURE)
                continue
                
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read()
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                run_log = construct_run_log(stdout_str)
                
                train_iterations = np.array(run_log['iterations/']).shape[0]
                epoch_freq = max(int(train_iterations // 10), 1)
                
                epochs_per_log = 10
                content += policy_feedback.format(epoch_freq=epochs_per_log*epoch_freq)
                
                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in run_log and "gpt_reward" in run_log:
                    gt_reward = np.array(run_log["gt_reward"])
                    gpt_reward = np.array(run_log["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in sorted(run_log.keys()):
                    if "/" not in metric:
                        metric_cur = ['{:.2f}'.format(x) for x in run_log[metric][::epoch_freq]]
                        metric_cur_max = max(run_log[metric])
                        metric_cur_mean = sum(run_log[metric]) / len(run_log[metric])

                        metric_cur_min = min(run_log[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            metric_name = metric 
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
               
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # Otherwise, provide execution traceback error feedback

                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 

        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue
        
        
        def compute_similarity_score_gpt(footage_grids_dir,contact_pattern_dirs):
            
            evaluator_query_content = [
            {
                        "type": "text",
                        "text": "You will be rating the following images:"
                    } 
            ]
            
            for footage_dir in footage_grids_dir:
                
                encoded_footage = encode_image(footage_dir)
            
                evaluator_query_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_footage}",
                        "detail": cfg.image_quality
                    }
                    }
                )
        

            contact_evaluator_query_content = [
            {
                        "type": "text",
                        "text": "They have the following corresponding foot contact sequence plots, where L means Left Foot and R means Right Foot for the humanoid robot"
                    } 
            ]
            
            for contact_dir in contact_pattern_dirs:
                
                encoded_contact = encode_image(contact_dir)
            
                contact_evaluator_query_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_contact}",
                        "detail": cfg.image_quality
                    }
                    }
                )
        
            if best_footage is not None:
                evaluator_query_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{best_footage}",
                        "detail": cfg.image_quality
                    }
                    }
                )
                
                contact_evaluator_query_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{best_contact}",
                        "detail": cfg.image_quality
                    }
                    }
                )
                
                successful_runs_index.append(-1)
            
            if cfg.task.use_annotation:
                encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo_annotated.png')
            else:
                encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo.png')
            
            evaluator_query_messages = [
                {"role": "system", "content": initial_task_evaluator_system},
                {"role" : "user", "content":
                    [
                        {
                            "type": "text",
                            "text": "Here is the image demonstrating the ground truth task"
                        },
                        {
                            
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_gt_frame_grid}",
                                "detail": cfg.image_quality
                            }
                        }
                    ]
                },
                None,
                None
            ]
            
            evaluator_query_messages[2] ={"role" : "user", "content": evaluator_query_content}
            
            evaluator_query_messages[3] ={"role" : "user", "content": contact_evaluator_query_content}
            
            logging.info("Evaluating...")
            eval_responses,_,_,_ = gpt_query(1,evaluator_query_messages,cfg.temperature,cfg.model)
        
            eval_responses = eval_responses[0]["message"]["content"]
            
            scores_re = re.findall(r'\[([^\]]*)\](?!.*\[)',eval_responses)
            
      
                
            scores_re = scores_re[-1]

            scores = [float(x) for x in scores_re.split(",")]
            
            # Save evaluator messages regardless of number of samples
            with open(f'evaluator_query_messages_{iter}.json', 'w') as file:
                json.dump(evaluator_query_messages + [{"role": "assistant", "content": eval_responses}], file, indent=4)
            
            if len(scores) == 1:
                logging.info(f"Best Sample Index: {0}")
                return 0, True
            else:
                best_idx_in_successful_runs = np.argmax(scores)
                second_best_idx_in_successful_runs = np.argsort(scores)[-2]
                
                best_idx = successful_runs_index[best_idx_in_successful_runs]
                second_best_idx = successful_runs_index[second_best_idx_in_successful_runs]

                logging.info(f"Best Sample Index: {best_idx}, Second Best Sample Index: {second_best_idx}")
            
                if best_idx == -1:
                    # Best sample is the previous best footage
                    return second_best_idx, False
                
                return best_idx, True
        
        best_sample_idx,improved = compute_similarity_score_gpt(footage_grids_dir,contact_pattern_dirs)

        best_content = contents[best_sample_idx]

        
        if improved:
            logging.info(f"Iteration {iter}: A better reward function has been generated")
            max_reward_code_path = code_paths[best_sample_idx]
            best_footage = encode_image(f'{workspace_dir}/training_footage/training_frame_{iter}_{best_sample_idx}.png')
            
            # Get best contact pattern from Isaac Lab - NO FALLBACKS
            rl_filepath = f"env_iter{iter}_response{best_sample_idx}.txt"
            isaac_lab_root = find_isaac_lab_root()
            logs_dir = os.path.join(isaac_lab_root, "logs", "rsl_rl")
            
            # Find the most recent task directory (same logic as in training)
            task_dirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
            if not task_dirs:
                raise RuntimeError(f"No Isaac Lab task directories found in {logs_dir}/")
            
            latest_task_dir = max(task_dirs, key=lambda d: os.path.getmtime(os.path.join(logs_dir, d)))
            experiment_dirs = glob.glob(os.path.join(logs_dir, latest_task_dir, "*"))
            
            if not experiment_dirs:
                raise RuntimeError(f"No Isaac Lab experiment directories found in {logs_dir}/{latest_task_dir}/")
            
            full_training_log_dir = max(experiment_dirs, key=os.path.getmtime)
            contact_pattern_dir = os.path.join(full_training_log_dir, "contact_analysis", "contact_sequence.png")
            
            if not os.path.exists(contact_pattern_dir):
                raise RuntimeError(f"Contact analysis file not found: {contact_pattern_dir}")
            
            best_contact = encode_image(contact_pattern_dir)
            logging.info(f"Loaded best contact pattern: {contact_pattern_dir}")

        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
            
            
        if len(reward_query_messages) == 2:
            reward_query_messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
            reward_query_messages += [{"role": "user", "content": best_content}]
        else:
            assert len(reward_query_messages) == 4
            reward_query_messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
            reward_query_messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open('reward_query_messages.json', 'w') as file:
            json.dump(reward_query_messages, file, indent=4)
    
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    logging.info(f"Best Reward Code Path: {max_reward_code_path}")

    best_reward = file_to_string(max_reward_code_path)
    with open(output_file, 'w') as file:
        file.writelines(best_reward + '\n')
    
    # Get run directory of best-performing policy
    with open(max_reward_code_path.replace(".py", ".txt"), "r") as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith("Dashboard: "):
            run_dir = line.split(": ")[1].strip()
            run_dir = run_dir.replace("http://app.dash.ml/", f"{ROOT_DIR}/{env_name}/runs/")
            logging.info("Best policy run directory: " + run_dir)

if __name__ == "__main__":
    main()