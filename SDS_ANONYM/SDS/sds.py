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
from utils.plotting import create_sds_training_plots

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def validate_foundation_only_code(code: str):
    """
    Validate reward code for foundation-only compliance.
    Returns: (is_valid, violations, feedback_message)
    """
    violations = []
    
    # Forbidden patterns that indicate environmental sensor usage
    forbidden_patterns = [
        r'env\.scene\.sensors',
        r'height_scanner', 
        r'lidar',
        r'RayCaster',
        r'height_scan',
        r'lidar_range',
        r'ray_caster',
        r'terrain_height',
        r'gap_detection',
        r'obstacle_detection', 
        r'stair_detection',
        r'SceneEntityCfg\(["\']height_scanner["\']\)',
        r'SceneEntityCfg\(["\']lidar["\']\)',
        r'sensor_cfg.*height',
        r'sensor_cfg.*lidar',
        r'terrain_following',
        r'obstacle_avoidance',
        r'gap_crossing',
        r'stair_climbing'
    ]
    
    # Check for forbidden environmental sensor patterns
    for pattern in forbidden_patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        if matches:
            violations.append(f"Forbidden sensor usage: {pattern}")
    
    is_valid = len(violations) == 0
    
    # Generate feedback message
    if not is_valid:
        feedback_msg = "🚫 FOUNDATION-ONLY VIOLATIONS:\n"
        for violation in violations:
            feedback_msg += f"• {violation}\n"
        feedback_msg += "\n✅ ALLOWED: velocity tracking, gait quality, posture, stability, smoothness only"
    else:
        feedback_msg = "✅ Code validates for foundation-only constraints"
    
    return is_valid, violations, feedback_msg

def extract_code_from_response(response_content: str):
    """Extract Python code from LLM response."""
    patterns = [
        r'```python(.*?)```',
        r'```(.*?)```', 
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_content, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return response_content.strip()

def find_isaac_lab_root():
    """Find Isaac Lab root directory by looking for isaaclab.sh."""
    from pathlib import Path
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "isaaclab.sh").exists():
            return str(parent)
    raise FileNotFoundError("Could not find Isaac Lab root directory (isaaclab.sh not found)")

def create_readable_json_version(json_file_path):
    """
    Create a readable version of GPT query JSON files by removing base64 images
    but keeping all text content intact for full analysis.
    
    Args:
        json_file_path (str): Path to the original JSON file
    """
    try:
        # Load the original JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Create readable version
        readable_data = []
        for i, msg in enumerate(data):
            clean_msg = {'role': msg.get('role', 'unknown')}
            content = msg.get('content', '')
            
            if isinstance(content, list):
                # Handle list content (usually contains text + images)
                clean_content = []
                for item in content:
                    if item.get('type') == 'text':
                        # Keep FULL text content (no truncation)
                        clean_content.append({
                            'type': 'text',
                            'text': item.get('text', '')
                        })
                    elif item.get('type') == 'image_url':
                        # Replace base64 image with placeholder
                        image_url = item.get('image_url', {})
                        if isinstance(image_url, dict) and 'url' in image_url:
                            # Check if it's base64 data
                            if image_url['url'].startswith('data:image'):
                                clean_content.append({
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': '[BASE64_IMAGE_DATA_REMOVED_FOR_READABILITY]',
                                        'detail': image_url.get('detail', 'high'),
                                        'original_size_note': f'Original base64 data was {len(image_url["url"])} characters'
                                    }
                                })
                            else:
                                # Keep non-base64 URLs as is
                                clean_content.append(item)
                        else:
                            clean_content.append({
                                'type': 'image_url',
                                'note': '[IMAGE_DATA_REMOVED_FOR_READABILITY]'
                            })
                    else:
                        # Keep other content types as is
                        clean_content.append(item)
                clean_msg['content'] = clean_content
            else:
                # Handle string content - keep full content
                clean_msg['content'] = content
            
            readable_data.append(clean_msg)
        
        # Save readable version with _READABLE suffix
        readable_path = json_file_path.replace('.json', '_READABLE.json')
        with open(readable_path, 'w', encoding='utf-8') as f:
            json.dump(readable_data, f, indent=2, ensure_ascii=False)
        
        # Calculate size savings
        original_size = os.path.getsize(json_file_path)
        readable_size = os.path.getsize(readable_path)
        
        logging.info(f"✅ Created readable JSON: {readable_path}")
        logging.info(f"📊 Size reduction: {original_size/1024/1024:.1f}MB → {readable_size/1024:.1f}KB ({(1-readable_size/original_size)*100:.1f}% smaller)")
        
        return readable_path
        
    except Exception as e:
        logging.error(f"❌ Failed to create readable JSON version for {json_file_path}: {e}")
        return None

def reduce_image_detail_in_messages(messages):
    """Reduce image detail from 'high' to 'low' to decrease payload size for 500 error recovery."""
    modified_messages = []
    
    for message in messages:
        if isinstance(message, dict):
            modified_message = message.copy()
            
            # Check if content is a list (multimodal content)
            if isinstance(message.get('content'), list):
                modified_content = []
                for item in message['content']:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        # Reduce image detail
                        modified_item = item.copy()
                        if 'image_url' in modified_item and isinstance(modified_item['image_url'], dict):
                            modified_item['image_url'] = modified_item['image_url'].copy()
                            modified_item['image_url']['detail'] = 'low'
                        modified_content.append(modified_item)
                    else:
                        modified_content.append(item)
                modified_message['content'] = modified_content
            
            modified_messages.append(modified_message)
        else:
            modified_messages.append(message)
    
    return modified_messages

# Get SDS root directory from script location, not working directory
# This is important because Hydra changes the working directory
SDS_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = f"{SDS_ROOT_DIR}/.."

def capture_environment_image_automatically(workspace_dir):
    """Automatically capture environment image before SDS reward generation."""
    try:
        import subprocess
        import os
        
        # Find Isaac Lab root
        isaac_lab_root = find_isaac_lab_root()
        
        # Build capture command 
        capture_script = os.path.join(isaac_lab_root, "source", "isaaclab_tasks", "isaaclab_tasks", 
                                    "manager_based", "sds", "velocity", "capture_environment_image.py")
        
        if not os.path.exists(capture_script):
            logging.warning(f"⚠️ Environment image capture script not found: {capture_script}")
            logging.warning("📝 Continuing SDS without environment image")
            return False
        
        # Run image capture
        logging.info("📸 Automatically capturing environment image...")
        cmd = [
            os.path.join(isaac_lab_root, "isaaclab.sh"), 
            "-p", capture_script,
            "--checkpoint_dir", str(workspace_dir),
            "--headless", 
            "--enable_cameras"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=isaac_lab_root)
        
        if result.returncode == 0:
            logging.info("✅ Environment image captured successfully!")
            return True
        else:
            logging.warning(f"⚠️ Environment image capture failed: {result.stderr}")
            logging.warning("📝 Continuing SDS without environment image")
            return False
            
    except Exception as e:
        logging.warning(f"⚠️ Failed to capture environment image: {e}")
        logging.warning("📝 Continuing SDS without environment image")
        return False

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {SDS_ROOT_DIR}")
    logging.info(f"Running for {cfg.iteration} iterations")
    logging.info(f"Training each RF for: {cfg.train_iterations} iterations")
    logging.info(f"Generating {cfg.sample} reward function samples per iteration")
    
    # 🎯 ENVIRONMENT-AWARE MODE CONTROL
    env_aware = getattr(cfg, 'env_aware', True)
    if os.environ.get('SDS_ENV_AWARE') is not None:
        env_aware = os.environ.get('SDS_ENV_AWARE').lower() in ['1', 'true', 'on']
    
    logging.info(f"🎯 SDS Environment-Aware Mode: {'ENABLED' if env_aware else 'DISABLED (Foundation-Only)'}")
    
    # Create context dict for prompt injection
    context_dict = {
        'ENV_AWARE': env_aware,
        'SENSORS_AVAILABLE': env_aware,
        'TERRAIN_INFO': 'PROVIDED' if env_aware else 'NOT_PROVIDED',
        'ENVIRONMENT_IMAGE': 'AVAILABLE' if env_aware else 'NONE'
    }
    
    # 🆕 CONDITIONAL ENVIRONMENT IMAGE CAPTURE
    if env_aware and getattr(cfg, 'auto_capture_environment_image', True):
        logging.info("🎯 Starting automatic environment image capture...")
        capture_success = capture_environment_image_automatically(workspace_dir)
        if capture_success:
            logging.info("✅ Environment image ready for reward generation")
            context_dict['ENVIRONMENT_IMAGE'] = 'CAPTURED'
        else:
            logging.info("📝 Proceeding with video-only analysis")
            context_dict['ENVIRONMENT_IMAGE'] = 'FAILED_CAPTURE'
    elif not env_aware:
        logging.info("🚫 Environment image capture DISABLED (Foundation-only mode)")
        context_dict['ENVIRONMENT_IMAGE'] = 'DISABLED'
    else:
        logging.info("⏭️ Auto environment image capture disabled by configuration")

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
    
    encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo.png', max_size=(512, 512), quality=60)
    
    # Check for environment image captured by the image capture script
    # 🎯 CONDITIONAL ENVIRONMENT IMAGE HANDLING
    environment_image_path = f'{workspace_dir}/environment_image.png'
    has_environment_image = os.path.exists(environment_image_path) and env_aware
    
    if env_aware and os.path.exists(environment_image_path):
        encoded_environment_image = encode_image(environment_image_path, max_size=(512, 512), quality=60)
        logging.info(f"✅ Found environment image for SUS generation: {environment_image_path}")
        context_dict['ENVIRONMENT_IMAGE'] = 'AVAILABLE'
    elif not env_aware:
        encoded_environment_image = None
        logging.info(f"🚫 Environment image DISABLED (Foundation-only mode)")
        context_dict['ENVIRONMENT_IMAGE'] = 'DISABLED_FOUNDATION_MODE'
    else:
        encoded_environment_image = None
        logging.info(f"ℹ️ No environment image found at: {environment_image_path}")
        context_dict['ENVIRONMENT_IMAGE'] = 'NOT_FOUND'
    
    # Choose SUS generator based on env_aware configuration
    enable_env_analysis = getattr(cfg, 'enable_environment_analysis', True) and env_aware
    
    if enable_env_analysis and env_aware:
        # Use enhanced SUS generator with environment awareness
        sus_generator = EnhancedSUSGenerator(cfg, prompt_dir)
        
        # Generate environment-aware SUS prompt
        num_envs_for_analysis = getattr(cfg, 'environment_analysis_robots', 100)
        logging.info(f"🌍 Using environment-aware SUS generation with {num_envs_for_analysis} robots for terrain analysis")
        
        SUS_prompt = sus_generator.generate_enhanced_sus_prompt(
            encoded_gt_frame_grid, 
            cfg.task.description,
            num_envs=num_envs_for_analysis,
            encoded_environment_image=encoded_environment_image if has_environment_image else None
        )
    else:
        # Use original SUS generator (video-only analysis) or foundation-only mode
        sus_generator = SUSGenerator(cfg, prompt_dir)
        if not env_aware:
            logging.info("🚫 Using FOUNDATION-ONLY SUS generation (no environmental analysis, gait-focused)")
            # For foundation-only mode, we need to inject foundation explanations instead of environment analysis
            SUS_prompt = sus_generator.generate_foundation_only_sus_prompt(
                encoded_gt_frame_grid, 
                cfg.task.description
            )
        else:
            logging.info("📝 Using original SUS generation (video-only analysis)")
        SUS_prompt = sus_generator.generate_sus_prompt(
            encoded_gt_frame_grid, 
            cfg.task.description,
            encoded_environment_image=encoded_environment_image if has_environment_image else None
        )
    
    # 🔧 COPY ALL AGENT CONVERSATIONS TO OUTPUT DIRECTORY
    copy_agent_conversations_to_output(workspace_dir)
    
    # 🔧 CREATE DATA FLOW SUMMARY
    create_data_flow_summary(workspace_dir, SUS_prompt)

    # 🎯 FOUNDATION-ONLY PROMPT INJECTION
    if not env_aware:
        foundation_constraint = f"""
🚫 CRITICAL CONSTRAINT - FOUNDATION-ONLY MODE:
ENV_AWARE: FALSE
SENSORS_AVAILABLE: FALSE  
ENVIRONMENT_IMAGE: {context_dict['ENVIRONMENT_IMAGE']}
TERRAIN_CONTEXT: NOT PROVIDED

MANDATORY RESTRICTIONS:
• DO NOT import or access env.scene.sensors, height_scanner, lidar, RayCaster
• DO NOT use terrain-specific adaptations or environmental sensing
• FOCUS ONLY on fundamental locomotion: velocity tracking, gait quality, posture, stability, smoothness
• Use ONLY proprioceptive observations (joint states, IMU, commands)
• Design terrain-agnostic rewards that work across all environments

ALLOWED FOUNDATION COMPONENTS ONLY:
• track_lin_vel_xy_yaw_frame_exp (velocity tracking)
• track_ang_vel_z_world_exp (angular velocity control)  
• feet_air_time_positive_biped (gait quality)
• base_height_l2 (postural stability)
• orientation_l2 (balance and uprightness)
• joint_deviation_l1 (natural joint positions)
• action_smoothness_l2 (smooth control)
• feet_slide (contact quality)

FOUNDATION-ONLY EXPLANATION:
This is foundation-only mode where environmental analysis has been disabled.
The SUS (Status Understanding Summary) contains ONLY basic gait and movement analysis from video footage.
NO environmental sensor data, terrain analysis, or adaptive strategies are available.
Focus on creating robust, terrain-agnostic locomotion rewards.

"""
        initial_reward_engineer_system = foundation_constraint + initial_reward_engineer_system
        initial_reward_engineer_user = foundation_constraint + initial_reward_engineer_user
        
        # Also inject explanation into task evaluator
        initial_task_evaluator_system = foundation_constraint + initial_task_evaluator_system

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
            
            # 🚫 FOUNDATION-ONLY CODE VALIDATION DISABLED (per user request)
            if not env_aware:
                logging.info(f"🚫 Foundation-only mode active for sample {response_id} - validation disabled")
            
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
                    
                    # Skip ViTPose processing - use original video directly for faster processing
                    logging.info("Skipping ViTPose annotation for faster post-training capture")
                    training_frame_grid = create_grid_image(video_file, crop=True, crop_option="RobotFocus", training_fixed_length=True)
                    
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
        successful_run_logs = []  # Store run_logs for grading agent
        
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
                successful_run_logs.append(None)  # Keep indices aligned
                continue
                
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read()
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                reward_correlations.append(DUMMY_FAILURE)
                successful_run_logs.append(None)  # Keep indices aligned
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                run_log = construct_run_log(stdout_str)
                successful_run_logs.append(run_log)  # Store for grading agent
                
                train_iterations = np.array(run_log['iterations/']).shape[0]
                epoch_freq = max(int(train_iterations // 5), 1)
                
                epochs_per_log = 5
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
               
                # ✅ CREATE PLOTS: Generate and save training plots after successful completion
                try:
                    logging.info("🎨 Generating training plots...")
                    # Use current SDS workspace directory (managed by Hydra)
                    current_sds_workspace = os.getcwd()
                    plot_success = create_sds_training_plots(run_log, current_sds_workspace)
                    if plot_success:
                        logging.info("✅ Training plots created successfully!")
                        logging.info(f"📁 Plots saved to: {os.path.join(current_sds_workspace, 'plots')}")
                    else:
                        logging.warning("⚠️ Plot creation failed, but training analysis continues")
                except Exception as e:
                    logging.error(f"❌ Error creating plots: {e}")
                    logging.warning("⚠️ Plot creation failed, but training analysis continues")
               
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # Otherwise, provide execution traceback error feedback

                reward_correlations.append(DUMMY_FAILURE)
                successful_run_logs.append(None)  # Keep indices aligned
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
        
        
        def compute_similarity_score_gpt(footage_grids_dir,contact_pattern_dirs,run_logs=None):
            
            evaluator_query_content = [
            {
                        "type": "text",
                        "text": "You will be rating the following images:"
                    } 
            ]
            
            for footage_dir in footage_grids_dir:
                
                encoded_footage = encode_image(footage_dir, max_size=(512, 512), quality=60)
            
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
                
                encoded_contact = encode_image(contact_dir, max_size=(512, 512), quality=60)
            
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
            
            # Add performance metrics for grading evaluation
            if run_logs is not None and len(run_logs) > 0:
                metrics_text = "PERFORMANCE METRICS FOR EVALUATION:\n\n"
                for i, run_log in enumerate(run_logs):
                    if run_log is not None:
                        metrics_text += f"Policy {i+1} Performance:\n"
                        # Core metrics
                        if "reward" in run_log:
                            metrics_text += f"- Mean Reward: {run_log['reward'][-5:]} (last 5 iterations)\n"
                        if "episode_length" in run_log:
                            metrics_text += f"- Episode Length: {run_log['episode_length'][-5:]}\n"
                        # Task performance  
                        if "velocity_error_xy" in run_log:
                            metrics_text += f"- Velocity Error XY: {[f'{x:.3f}' for x in run_log['velocity_error_xy'][-3:]]}\n"
                        if "velocity_error_yaw" in run_log:
                            metrics_text += f"- Velocity Error Yaw: {[f'{x:.3f}' for x in run_log['velocity_error_yaw'][-3:]]}\n"
                        # Termination rates
                        if "termination_base_contact" in run_log:
                            metrics_text += f"- Fall Rate: {[f'{x:.3f}' for x in run_log['termination_base_contact'][-3:]]}\n"
                        if "termination_timeout" in run_log:
                            timeout_rate = [f'{x:.3f}' for x in run_log['termination_timeout'][-3:]]
                            completion_rate = [f'{1.0-x:.3f}' for x in run_log['termination_timeout'][-3:]]
                            metrics_text += f"- Task Completion Rate: {completion_rate}\n"
                        metrics_text += "\n"
                
                # Add metrics as text content
                contact_evaluator_query_content.append({
                    "type": "text",
                    "text": metrics_text + "Consider these performance metrics alongside visual similarity for comprehensive evaluation."
                })
            
            if cfg.task.use_annotation:
                encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo_annotated.png', max_size=(512, 512), quality=60)
            else:
                encoded_gt_frame_grid = encode_image(f'{workspace_dir}/gt_demo.png', max_size=(512, 512), quality=60)
            
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
            evaluator_json_path = f'evaluator_query_messages_{iter}.json'
            with open(evaluator_json_path, 'w') as file:
                json.dump(evaluator_query_messages + [{"role": "assistant", "content": eval_responses}], file, indent=4)
            
            # 🔧 AUTO-GENERATE READABLE VERSION
            create_readable_json_version(evaluator_json_path)
            
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
        
        best_sample_idx,improved = compute_similarity_score_gpt(footage_grids_dir,contact_pattern_dirs,successful_run_logs)

        best_content = contents[best_sample_idx]

        
        if improved:
            logging.info(f"Iteration {iter}: A better reward function has been generated")
            max_reward_code_path = code_paths[best_sample_idx]
            best_footage = encode_image(f'{workspace_dir}/training_footage/training_frame_{iter}_{best_sample_idx}.png', max_size=(512, 512), quality=60)
            
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
            
            best_contact = encode_image(contact_pattern_dir, max_size=(512, 512), quality=60)
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
        reward_json_path = 'reward_query_messages.json'
        with open(reward_json_path, 'w') as file:
            json.dump(reward_query_messages, file, indent=4)
        
        # 🔧 AUTO-GENERATE READABLE VERSION
        create_readable_json_version(reward_json_path)
    
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
    
    # 🏷️ SAVE RUN METADATA FOR ANALYSIS GROUPING
    run_meta = {
        "env_aware": env_aware,
        "mode": "environment_aware" if env_aware else "foundation_only",
        "sensors_available": env_aware,
        "environment_analysis": enable_env_analysis and env_aware,
        "auto_image_capture": getattr(cfg, 'auto_capture_environment_image', True) and env_aware,
        "timestamp": str(workspace_dir).split('/')[-1],  # Extract timestamp from workspace dir
        "terrain_type": getattr(cfg, 'terrain', 'unknown'),
        "model": cfg.model,
        "iterations": cfg.iteration,
        "samples_per_iteration": cfg.sample,
        "train_iterations": cfg.train_iterations
    }
    
    meta_file = workspace_dir / "run_meta.json"
    with open(meta_file, 'w') as f:
        json.dump(run_meta, f, indent=2)
    
    logging.info(f"📊 Run metadata saved: {meta_file}")
    logging.info(f"🎯 Mode: {'Environment-Aware' if env_aware else 'Foundation-Only'}")

def copy_agent_conversations_to_output(workspace_dir):
    """Copy all agent conversation files to the output directory"""
    try:
        import glob
        import shutil
        
        # Find all agent conversation files in current directory
        conversation_files = glob.glob("*_conversation*.json")
        
        if conversation_files:
            logging.info(f"🔧 Copying {len(conversation_files)} agent conversation files to output directory")
            
            for conv_file in conversation_files:
                dest_path = os.path.join(workspace_dir, conv_file)
                shutil.copy2(conv_file, dest_path)
                logging.info(f"🔧 Copied: {conv_file} -> {dest_path}")
                
                # Also copy to Isaac Lab root for easy access
                try:
                    root_dest = os.path.join("/home/enis/IsaacLab", conv_file)
                    shutil.copy2(conv_file, root_dest)
                except:
                    pass  # Don't fail if can't copy to root
        else:
            logging.warning("🔧 No agent conversation files found to copy")
            
    except Exception as e:
        logging.error(f"Failed to copy agent conversations: {e}")

def create_data_flow_summary(workspace_dir, sus_prompt):
    """Create a summary of the data flow for debugging"""
    try:
        summary_file = os.path.join(workspace_dir, "data_flow_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("🔍 DATA FLOW SUMMARY FOR ENVIRONMENT ANALYSIS DEBUGGING\n")
            f.write("=" * 70 + "\n\n")
            
            # Check if SUS prompt contains environment analysis data
            f.write("📊 ENVIRONMENT ANALYSIS DATA IN SUS PROMPT:\n")
            f.write("-" * 50 + "\n")
            
            env_data_indicators = [
                "Total Gaps Detected:",
                "Gaps Detected:",
                "Total Obstacles Detected:",
                "Obstacles Detected:",
                "Average Terrain Roughness:",
                "Terrain Roughness:",
                "Safety Score:"
            ]
            
            found_env_data = False
            for indicator in env_data_indicators:
                if indicator in sus_prompt:
                    # Extract the line containing this indicator
                    lines = sus_prompt.split('\n')
                    for line in lines:
                        if indicator in line:
                            f.write(f"✅ FOUND: {line.strip()}\n")
                            found_env_data = True
                            break
            
            if not found_env_data:
                f.write("❌ NO ENVIRONMENT ANALYSIS DATA FOUND IN SUS PROMPT!\n")
                f.write("This indicates the environment analysis data was lost somewhere in the pipeline.\n")
            
            f.write("\n📂 AGENT CONVERSATION FILES GENERATED:\n")
            f.write("-" * 50 + "\n")
            
            # List available conversation files
            import glob
            conv_files = glob.glob(os.path.join(workspace_dir, "*_conversation*.json"))
            if conv_files:
                for conv_file in conv_files:
                    f.write(f"📄 {os.path.basename(conv_file)}\n")
            else:
                f.write("❌ No conversation files found!\n")
            
            f.write(f"\n🔍 HOW TO DEBUG:\n")
            f.write("-" * 50 + "\n")
            f.write("1. Check environmentawaretaskdescriptor_conversation_READABLE.json\n")
            f.write("2. Check enhancedssusgenerator_conversation_READABLE.json\n")
            f.write("3. Compare environment analysis data between files\n")
            f.write("4. Look for the exact numbers: gaps, obstacles, terrain roughness\n")
            
        logging.info(f"🔧 Created data flow summary: {summary_file}")
        
    except Exception as e:
        logging.error(f"Failed to create data flow summary: {e}")

if __name__ == "__main__":
    main()