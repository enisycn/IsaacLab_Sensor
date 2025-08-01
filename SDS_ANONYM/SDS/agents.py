import logging
from utils.misc import gpt_query
import subprocess
import os
import re

class Conversation():
    def __init__(self,system_prompt:str) -> None:
        self.messages = [
            {
            "role": "system",
            "content": [
                {
                "text": system_prompt,
                "type": "text"
                }
            ]
            }
        ]
        self.conversation_completion_tokens = 0
        self.conversation_prompt_tokens = 0
        self.converstation_total_tokens = 0
    
    def modify_system_prompt(self,new_system_prompt):
        self.messages[0]["content"][0]["text"] = new_system_prompt
    
    def add_user_content(self,content:list):
        self.messages.append(
            {
            "role": "user",
            "content": content
            }
        )
    
    
    def add_assistant_content(self,prompt):
        self.messages.append(
            {
            "role": "assistant",
            "content": [
                {
                "text": prompt,
                "type": "text"
                }
            ]
            }
        )
    
    def add_usage(self,usage):
        self.conversation_prompt_tokens += usage.prompt_tokens
        self.conversation_completion_tokens += usage.completion_tokens
        self.converstation_total_tokens += usage.total_tokens
    
    def get_message(self):
        return self.messages
    
    def get_last_content(self):
        return 

class Agent():
    def __init__(self, system_prompt_file,cfg):
        with open(system_prompt_file,"r") as f:
            self.system_prompt = f.read()
        self.cfg = cfg
        self.conversation = Conversation(self.system_prompt)
        self.last_assistant_content = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Activated Agent {self.__class__.__name__}")
        self.sample = 1
        self.model = cfg.model
        
        # Reasoning models (o1, o3, o4 series) only support temperature=1.0
        reasoning_models = ['o1', 'o1-mini', 'o1-preview', 'o3', 'o3-mini', 'o4-mini']
        if any(self.model.startswith(model) for model in reasoning_models):
            self.temperature = 1.0  # Required for reasoning models
            self.logger.info(f"Using temperature=1.0 for reasoning model {self.model}")
        else:
            self.temperature = 0.8  # Default for other models
    
    def get_conversation(self):
        return self.conversation
    
    def prepare_user_content(self,contents:list):
        full_content = []
        
        for content in contents:
            if content["type"] == "text":
                 full_content.append(
                    {
                    "text": content["data"],
                    "type": "text"
                    }
                )
            elif content["type"] == "image_uri":
                full_content.append(
                    {
                    "type": "image_url",
                    "image_url": 
                        {
                        "url": f"data:image/png;base64,{content['data']}",
                        "detail": "high"
                        }
                    }
                )
            else:
                full_content.append(
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": content["data"],
                        "detail": "high"
                        }
                    }
                )
        
        self.conversation.add_user_content(full_content)
    
    def query(self):
        responses,_,_,_ = gpt_query(sample=self.sample,temperature=self.temperature,model=self.model,messages=self.conversation.get_message())
        assistant_content = responses[0]["message"]["content"]
        self.conversation.add_assistant_content(assistant_content)
        self.last_assistant_content = assistant_content
        return assistant_content
    
    def obtain_results(self):
        return self.last_assistant_content

class ContactSequenceAnalyser(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/contact_sequence_system.txt"
        super().__init__(system_prompt_file, cfg)
    def analyse(self,encoded_frame_grid):
        self.prepare_user_content([{"type":"image_uri","data":encoded_frame_grid}])
        
        contact_sequence = self.query()
        
        self.prepare_user_content([{"type":"text","data":"Revise the contact sequence that you just generated with the provided image containing sequential frames of a video. Check frame by frame to make sure it is correct. If it is, give your reasoning, if not fix the error."}])
        
        return self.query()
    
class TaskRequirementAnalyser(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/task_requirement_system.txt"
        super().__init__(system_prompt_file, cfg)
    def analyse(self,encoded_frame_grid, task_hint=None):
        user_content = [{"type":"image_uri","data":encoded_frame_grid}]
        if task_hint:
            user_content.append({"type":"text","data":f"You are analyzing a video demonstrating: {task_hint}. Please provide requirements specifically for this type of locomotion behavior."})
        self.prepare_user_content(user_content)
        
        return self.query()

class GaitAnalyser(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/gait_pattern_system.txt"
        self.prompt_dir = prompt_dir
        super().__init__(system_prompt_file, cfg)
    
    def analyse(self,encoded_frame_grid,contact_pattern):
        self.prepare_user_content([{"type":"image_uri","data":encoded_frame_grid},{"type":"text","data":f"For the provided sequential frames, you are provided with a likely corresponding feet contact pattern: {contact_pattern}"}])
        gait_pattern_response = self.query()

        return gait_pattern_response
    
class SUSGenerator(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/SUS_generation_prompt.txt"
        self.prompt_dir = prompt_dir
        super().__init__(system_prompt_file, cfg)
    
    def generate_sus_prompt(self,encoded_gt_frame_grid, task_description_hint=None, encoded_environment_image=None):
        task_descriptor = TaskDescriptor(self.cfg,self.prompt_dir)
        task_description = task_descriptor.analyse(encoded_gt_frame_grid, task_description_hint, encoded_environment_image)
        
        contact_sequence_analyser = ContactSequenceAnalyser(self.cfg,self.prompt_dir)
        contact_pattern = contact_sequence_analyser.analyse(encoded_gt_frame_grid)
        
        gait_analyser = GaitAnalyser(self.cfg,self.prompt_dir)
        gait_response = gait_analyser.analyse(encoded_gt_frame_grid,contact_pattern)
        
        task_requirement_analyser = TaskRequirementAnalyser(self.cfg,self.prompt_dir)
        task_requirement_response = task_requirement_analyser.analyse(encoded_gt_frame_grid, task_description_hint)
        
        self.prepare_user_content([{"type":"text","data":task_description},{"type":"text","data":gait_response},{"type":"text","data":task_requirement_response}])
        
        sus_prompt = self.query()
        
        return sus_prompt

class EnvironmentAwareTaskDescriptor(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/environment_aware_task_descriptor_system.txt"
        super().__init__(system_prompt_file, cfg)
        self.prompt_file = system_prompt_file
        
    def inject_environment_analysis(self, environment_analysis):
        """Dynamically inject environment analysis into the prompt file between markers"""
        try:
            # Read the current prompt file
            with open(self.prompt_file, 'r') as f:
                prompt_content = f.read()
            
            # Find the markers and replace content between them
            start_marker = "<!-- ENVIRONMENT_ANALYSIS_START -->"
            end_marker = "<!-- ENVIRONMENT_ANALYSIS_END -->"
            
            if start_marker in prompt_content and end_marker in prompt_content:
                # Replace content between markers
                pattern = f"{re.escape(start_marker)}.*?{re.escape(end_marker)}"
                new_content = f"{start_marker}\n{environment_analysis}\n{end_marker}"
                updated_prompt = re.sub(pattern, new_content, prompt_content, flags=re.DOTALL)
                
                # Write back to file
                with open(self.prompt_file, 'w') as f:
                    f.write(updated_prompt)
            
                # Update the agent's system prompt
                self.conversation.modify_system_prompt(updated_prompt)
                
                self.logger.info("Environment analysis successfully injected into prompt")
                return True
            else:
                self.logger.error("Environment analysis markers not found in prompt file")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to inject environment analysis: {e}")
            return False
    
    def run_environment_analysis(self, num_envs=100):
        """Run the environment analysis script and capture output"""
        try:
            # Log agent activation for consistency with other agents
            self.logger.info("Activated Agent EnvironmentAnalyzer")
            
            # Change to IsaacLab directory and run analysis
            isaac_lab_path = "/home/enis/IsaacLab"
            analysis_script = f"{isaac_lab_path}/analyze_environment.py"
            
            # Run the analysis script with SDS_ANALYSIS_MODE=true
            # Use direct Python call instead of isaaclab.sh wrapper to avoid subprocess issues
            cmd = [
                "bash", "-c", 
                f"source /home/enis/miniconda3/etc/profile.d/conda.sh && conda activate sam2 && cd {isaac_lab_path} && SDS_ANALYSIS_MODE=true python -u analyze_environment.py --task=Isaac-SDS-Velocity-Flat-G1-Enhanced-v0 --headless --num_envs {num_envs}"
            ]
            
            self.logger.info(f"Running environment analysis with {num_envs} robots...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Extract only the comprehensive final analysis section
                output = result.stdout
                
                # Find the comprehensive analysis section
                start_marker = "📋 COMPREHENSIVE FINAL ENVIRONMENT ANALYSIS FOR AI AGENT"
                
                # Multiple possible end markers
                end_markers = [
                    "🔄 CLEANING UP ENVIRONMENT...",
                    "✅ COMPREHENSIVE MULTI-ROBOT ANALYSIS COMPLETE!",
                    "🔄 SHUTTING DOWN SIMULATION...",
                    "================================================================================\n\n🔄"
                ]
                
                if start_marker in output:
                    start_idx = output.find(start_marker)
            
                    # Find the first matching end marker
                    end_idx = len(output)  # Default to end of output
                    for end_marker in end_markers:
                        marker_idx = output.find(end_marker, start_idx)
                        if marker_idx != -1:
                            end_idx = marker_idx
                            break
                    
                    analysis_section = output[start_idx:end_idx].strip()
                    
                    # Clean up any remaining unwanted content
                    lines = analysis_section.split('\n')
                    clean_lines = []
                    for line in lines:
                        # Skip lines that look like system output
                        if any(skip in line for skip in ["[INFO]", "[WARNING]", "[ERROR]", "Exit code:", "Command completed"]):
                            continue
                        clean_lines.append(line)
                    
                    clean_analysis = '\n'.join(clean_lines).strip()
                    
                    self.logger.info(f"Environment analysis extracted successfully ({len(clean_analysis)} chars)")
                    return clean_analysis
                else:
                    self.logger.error("Could not find analysis start marker")
                    return None
            else:
                self.logger.error(f"Environment analysis failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("Environment analysis timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error running environment analysis: {e}")
            return None
    
    def analyse(self, encoded_frame_grid, task_hint=None, num_envs=100, encoded_environment_image=None):
        """Analyze with real-time environment data injection"""
        # Run environment analysis
        environment_analysis = self.run_environment_analysis(num_envs)
        
        if environment_analysis:
            # Inject environment analysis into prompt
            success = self.inject_environment_analysis(environment_analysis)
            if not success:
                self.logger.warning("Failed to inject environment analysis, proceeding without it")
        else:
            self.logger.warning("No environment analysis available, proceeding with video-only analysis")
            
        # Prepare user content for analysis
        user_content = [{"type":"image_uri","data":encoded_frame_grid}]
        
        # Add environment image if available
        if encoded_environment_image:
            user_content.append({"type":"image_uri","data":encoded_environment_image})
            self.logger.info("✅ Including environment image in environment-aware task descriptor analysis")
        
        if task_hint:
            user_content.append({"type":"text","data":f"You are analyzing a video demonstrating: {task_hint}. Focus your analysis on the specific behaviors and movements characteristic of this task while integrating the provided environmental sensor data."})
        
        self.prepare_user_content(user_content)
        return self.query()

class TaskDescriptor(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/task_descriptor_system.txt"
        super().__init__(system_prompt_file, cfg)
    def analyse(self,encoded_frame_grid, task_hint=None, encoded_environment_image=None):
        user_content = [{"type":"image_uri","data":encoded_frame_grid}]
        
        # Add environment image if available
        if encoded_environment_image:
            user_content.append({"type":"image_uri","data":encoded_environment_image})
            self.logger.info("✅ Including environment image in task descriptor analysis")
        
        if task_hint:
            user_content.append({"type":"text","data":f"You are analyzing a video demonstrating: {task_hint}. Focus your analysis on the specific behaviors and movements characteristic of this task."})
        self.prepare_user_content(user_content)
        return self.query()

class EnhancedSUSGenerator(Agent):
    def __init__(self, cfg, prompt_dir):
        system_prompt_file=f"{prompt_dir}/SUS_generation_prompt.txt"
        self.prompt_dir = prompt_dir
        super().__init__(system_prompt_file, cfg)
    
    def generate_enhanced_sus_prompt(self, encoded_gt_frame_grid, task_description_hint=None, num_envs=100, encoded_environment_image=None):
        """Generate SUS prompt with environment awareness"""
        # Use environment-aware task descriptor
        task_descriptor = EnvironmentAwareTaskDescriptor(self.cfg, self.prompt_dir)
        task_description = task_descriptor.analyse(encoded_gt_frame_grid, task_description_hint, num_envs, encoded_environment_image)   
        
        # Continue with regular SUS generation pipeline
        contact_sequence_analyser = ContactSequenceAnalyser(self.cfg, self.prompt_dir)
        contact_pattern = contact_sequence_analyser.analyse(encoded_gt_frame_grid)
        
        gait_analyser = GaitAnalyser(self.cfg, self.prompt_dir)
        gait_response = gait_analyser.analyse(encoded_gt_frame_grid, contact_pattern)
        
        task_requirement_analyser = TaskRequirementAnalyser(self.cfg, self.prompt_dir)
        task_requirement_response = task_requirement_analyser.analyse(encoded_gt_frame_grid, task_description_hint)
        
        self.prepare_user_content([
            {"type":"text","data":task_description},
            {"type":"text","data":gait_response},
            {"type":"text","data":task_requirement_response}
        ])
        
        sus_prompt = self.query()
        
        return sus_prompt