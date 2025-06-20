# Concrete Example: GPT Reward Integration Flow in SDS Isaac Lab

This document provides a **concrete step-by-step example** of how GPT-generated reward functions are integrated into Isaac Lab training within the SDS framework.

## 🎯 **Overview of the Integration Process**

The SDS system replaces Isaac Lab's default rewards with GPT-generated custom reward functions that are specifically designed to imitate demonstrated behaviors from video.

## 📋 **Step-by-Step Concrete Example**

### **Step 1: Initial Configuration State**

**Environment Configuration** (`velocity_env_cfg.py`):
```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # SDS Custom Reward Integration (ACTIVE)
    sds_custom = RewTerm(func=mdp.sds_custom_reward, weight=1.0)
    
    # Base rewards (DISABLED - weight=0.0)
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=0.0, ...)
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.0, ...)
    # ... all other base rewards set to weight=0.0
```

**Initial Reward Function** (`rewards.py`):
```python
def sds_custom_reward(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
    """Placeholder for SDS-generated custom reward function."""
    # Default implementation - returns zero reward
    return torch.zeros(env.num_envs, device=env.device)

# INSERT SDS REWARD HERE
```

### **Step 2: Video Analysis and SUS Generation**

**Input Video**: `trot.mp4` (demonstration of quadruped trotting gait)

**GPT Analysis Chain**:
1. **TaskDescriptor**: "Quadruped performing trotting locomotion with diagonal leg coordination"
2. **ContactSequence**: "FL+RR contact, then FR+RL contact, alternating pattern"
3. **GaitAnalyser**: "Trotting gait with 50% stance phase, clear flight phases"
4. **TaskRequirement**: "Maintain forward velocity while coordinating diagonal leg pairs"

**Generated SUS (See-Use-Synthesize) Prompt**:
```
"The robot should perform a trotting gait with diagonal leg coordination. The front-left and rear-right legs should contact ground simultaneously, followed by front-right and rear-left legs. Maintain steady forward velocity around 0.8-1.2 m/s with minimal vertical oscillation."
```

### **Step 3: GPT Reward Function Generation**

**GPT-4 Input Prompt** (combined system + user prompts):
```
SYSTEM: You are an expert in quadruped locomotion and Isaac Lab reward design...
[Isaac Lab API documentation]
[Reward function signature template]

USER: Based on this video demonstration showing trotting gait:
[Video frames grid image]

SUS Requirement: "The robot should perform a trotting gait with diagonal leg coordination..."

Generate a reward function that encourages this behavior.
```

**GPT-4 Generated Response** (Example):
```python
def sds_custom_reward(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
    """
    Custom reward function for trotting gait with diagonal leg coordination.
    """
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    
    # Get foot contact forces
    contact_forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
    foot_bodies = []
    for i, name in enumerate(contact_sensor.data.body_names):
        if "_foot" in name:
            foot_bodies.append(i)
    
    foot_forces = contact_forces[:, foot_bodies, :]  # [num_envs, 4, 3]
    foot_contact = torch.norm(foot_forces, dim=-1) > 5.0  # [num_envs, 4] - binary contact
    
    # Assuming foot order: [FL_foot, FR_foot, RL_foot, RR_foot]
    FL_contact = foot_contact[:, 0]  # Front-left
    FR_contact = foot_contact[:, 1]  # Front-right  
    RL_contact = foot_contact[:, 2]  # Rear-left
    RR_contact = foot_contact[:, 3]  # Rear-right
    
    # Reward diagonal coordination (FL+RR together, FR+RL together)
    diagonal1_sync = (FL_contact == RR_contact).float()  # FL and RR same state
    diagonal2_sync = (FR_contact == RL_contact).float()  # FR and RL same state
    diagonal_opposite = (FL_contact != FR_contact).float()  # Diagonals opposite states
    
    gait_reward = (diagonal1_sync + diagonal2_sync + diagonal_opposite) / 3.0
    
    # Forward velocity tracking
    commands = env.command_manager.get_command("base_velocity")
    target_vel = commands[:, 0]  # Forward velocity command
    actual_vel = robot.data.root_lin_vel_b[:, 0]  # Forward velocity in body frame
    vel_error = torch.abs(target_vel - actual_vel)
    velocity_reward = torch.exp(-vel_error / 0.5)
    
    # Stability reward (minimize vertical velocity)
    vertical_vel = torch.abs(robot.data.root_lin_vel_b[:, 2])
    stability_reward = torch.exp(-vertical_vel / 0.3)
    
    # Combined reward
    total_reward = 0.5 * gait_reward + 0.3 * velocity_reward + 0.2 * stability_reward
    
    return total_reward
```

### **Step 4: Code Processing and Integration**

**SDS Code Extraction** (`sds.py` line ~210):
```python
# Extract GPT response using regex patterns
patterns = [
    r'```python(.*?)```',
    r'```(.*?)```',
    # ... other patterns
]

code_string = extract_code_from_response(gpt_response)
# Result: Full function definition starting with "def sds_custom_reward..."
```

**Reward Function Replacement** (`sds.py` line ~215):
```python
# Pattern to match the entire sds_custom_reward function
pattern = r'def sds_custom_reward\(env: ManagerBasedRLEnv.*?\n    return torch\.zeros\(env\.num_envs, device=env\.device\)'

# Replace the placeholder function with GPT-generated function
if code_string.strip().startswith('def sds_custom_reward'):
    cur_task_rew_code_string = re.sub(pattern, code_string.strip(), task_rew_code_string, flags=re.DOTALL)
else:
    # Fallback: use placeholder replacement
    cur_task_rew_code_string = task_rew_code_string.replace("# INSERT SDS REWARD HERE", code_string)

# Write the modified reward file
with open(output_file, 'w') as file:
    file.writelines(cur_task_rew_code_string + '\n')
```

**Modified Reward File Result** (`rewards.py`):
```python
def sds_custom_reward(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
    """
    Custom reward function for trotting gait with diagonal leg coordination.
    """
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    
    # ... [GPT-generated reward logic as shown above] ...
    
    return total_reward

# INSERT SDS REWARD HERE
```

### **Step 5: Isaac Lab Training Execution**

**Training Command Execution** (`sds.py` line ~238):
```python
isaac_lab_root = find_isaac_lab_root()  # "/home/enis/IsaacLab"
command = f"{isaac_lab_root}/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-SDS-Velocity-Flat-Unitree-Go1-v0 --num_envs=4096 --max_iterations=1000 --headless"
process = subprocess.run(command.split(" "), stdout=f, stderr=f, cwd=isaac_lab_root)
```

**Isaac Lab Training Process**:
1. **Environment Creation**: Isaac Lab creates 4096 Go1 environments
2. **Reward Manager Setup**: Loads the modified `rewards.py` with GPT function
3. **Training Loop**: Each step calls `sds_custom_reward(env)` 
4. **Reward Computation**: GPT logic computes rewards for all 4096 environments
5. **Policy Updates**: PPO updates policy based on GPT-designed rewards

### **Step 6: Training Results Flow**

**Training Output Structure**:
```
logs/rsl_rl/unitree_go1_flat/2025-01-XX_XX-XX-XX/
├── model_0.pt, model_50.pt, ..., model_999.pt  # Policy checkpoints
├── events.out.tfevents.*                       # TensorBoard logs
├── videos/play/rl-video-step-0.mp4            # Evaluation video
└── contact_analysis/
    ├── contact_sequence.png                    # Gait visualization
    ├── contact_data.npy                       # Binary contact data
    └── force_data.npy                         # Force magnitude data
```

**Reward Function in Action During Training**:
```python
# Called ~200,000 times during 1000 iterations (4096 envs × 50 steps/iter × 1000 iter)
def sds_custom_reward(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
    # [GPT logic executes]
    # Returns: torch.Tensor([reward_env_0, reward_env_1, ..., reward_env_4095])
    #          Shape: [4096] - one reward per environment
```

### **Step 7: Evaluation and Feedback Loop**

**Policy Evaluation** (`sds.py` line ~270):
```python
# Evaluate trained policy
eval_script = f"{isaac_lab_root}/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-SDS-Velocity-Flat-Unitree-Go1-Play-v0 --num_envs=1 --checkpoint={latest_checkpoint} --video --video_length=500 --headless"
subprocess.run(eval_script.split(" "), cwd=isaac_lab_root)
```

**Generated Results**:
1. **Policy Video**: Robot performing the trained behavior
2. **Contact Analysis**: Gait pattern visualization  
3. **ViTPose Analysis**: Pose extraction from policy video
4. **Similarity Scoring**: Compare policy behavior to demonstration

## 🔄 **Complete Information Flow Summary**

```
Demo Video (trot.mp4)
    ↓ [ViTPose + GPT Analysis]
SUS Prompt ("trotting gait with diagonal coordination...")
    ↓ [GPT-4 Reward Generation]  
Isaac Lab Reward Function (def sds_custom_reward...)
    ↓ [Regex Replacement]
Modified rewards.py file 
    ↓ [Isaac Lab Training]
4096 Go1 Environments × 1000 Iterations × GPT Reward Logic
    ↓ [Policy Learning]
Trained Model (model_999.pt)
    ↓ [Evaluation]
Policy Video + Contact Analysis
    ↓ [ViTPose + Similarity Scoring]
Feedback for Next Iteration
```

## 🎯 **Key Integration Points**

### **1. Environment Configuration**
- **Single Active Reward**: Only `sds_custom` has weight=1.0, all others weight=0.0
- **Full Control**: GPT reward has complete control over training signals

### **2. Reward Function Signature**
- **Isaac Lab Compatible**: Matches exact Isaac Lab `ManagerBasedRLEnv` interface
- **Tensor Output**: Returns rewards for all environments simultaneously
- **Device Consistency**: Uses `env.device` for GPU compatibility

### **3. Real-time Replacement**
- **Live Updates**: Reward function replaced during SDS execution
- **No Restart Required**: Isaac Lab loads modified function automatically
- **Iterative Process**: Different reward functions tested across iterations

### **4. Training Integration**
- **Native Isaac Lab**: Uses standard Isaac Lab training scripts
- **Headless Mode**: Runs without GUI for speed
- **Full Logging**: Complete TensorBoard and checkpoint integration

## 🚀 **Performance Impact**

- **Training Speed**: ~2-4 hours for 1000 iterations (4096 environments)
- **Reward Calls**: ~200,000 calls to `sds_custom_reward` per training run
- **GPU Utilization**: Full GPU acceleration for both environments and reward computation
- **Memory Efficiency**: Single reward function handles all 4096 environments simultaneously

This integration provides **seamless GPT-to-training** flow while maintaining **full Isaac Lab performance** and **native RL framework compatibility**. 