# SDS for Humanoid Locomotion

## Project Overview

SDS uses large language models to automatically generate reward functions for humanoid locomotion training in Isaac Lab. The system analyzes demonstration videos, generates reward code, trains policies, and iteratively improves through GPT-04-mini feedback.

---

## System Requirements

### **Software Versions**
| Component | Version | Status |
|-----------|---------|---------|
| **Isaac Lab** | 0.40.1 | 
| **Isaac Sim** | 4.5.0 |       
| **Python** | 3.10.16 |    
| **PyTorch** | 2.8.0.dev20250618+cu128 |   
| **CUDA** | 12.8 |     
| **NVIDIA Driver** | 570.153.02 |  
| **OpenAI** | 1.89.0 |     

### **Verified Hardware Specifications**
| Component | Your Setup | Minimum Requirements |
|-----------|------------|---------------------|
| **GPU** | NVIDIA RTX 5080 Laptop GPU (16GB VRAM) 
| **RAM** | 32GB total system memory 
| **Storage** | 961GB total, 641GB available 
| **OS** | Linux Ubuntu 22.04

---

## Directory Structure

```
IsaacLab/
├── SDS_ANONYM/                    # Main SDS project directory
│   ├── SDS/                       # Core SDS system
│   │   ├── sds.py                 # Main orchestration script
│   │   ├── agents.py              # AI video analysis agents
│   │   ├── evaluator.py           # Policy evaluation system
│   │   ├── prompts/               # GPT prompt templates
│   │   ├── envs/                  # Environment API documentation
│   │   └── cfg/                   # Configuration files
│   ├── easy_ViTPose/              # Pose estimation system
│   └── monitor_training.py        # Training progress monitor
│
└── source/isaaclab_tasks/
    └── manager_based/sds/         # Isaac Lab SDS environments
        └── velocity/              # Velocity-based locomotion tasks
            ├── velocity_env_cfg.py    # Base environment config
            ├── mdp/rewards.py         # Generated reward functions
            └── config/g1/             # G1 humanoid configuration
                ├── flat_env_cfg.py    # Flat terrain config (velocity ranges should be adjusted here)

```

---

## Usage Commands

### **1. Full SDS Training**
```bash
cd SDS_ANONYM
conda activate <your_environment>
export OPENAI_API_KEY="your_openai_api_key_here"

python SDS/sds.py task=jump train_iterations=1000 iteration=4 sample=5 num_envs=4096 video_length=1000
```

### **2. Real Time Monitoring Training Progress (second terminal)**
```bash
cd SDS_ANONYM
conda activate <your_environment>
python monitor_training.py
```

### **3. Manual Training (Without AI)**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-SDS-Velocity-Flat-G1-v0 --num_envs 2048 --headless
```

### **4. Play Trained Policy**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-SDS-Velocity-Flat-G1-v0 --num_envs 5 \
  --checkpoint logs/rsl_rl/g1_flat/YYYY-MM-DD_HH-MM-SS/model_XXX.pt
```

---

## System Workflow

1. **AI Analysis** → GPT-04-mini analyzes demonstration videos and generates reward functions
2. **Training** → Isaac Lab trains policies using generated rewards (4096 parallel environments)  
3. **Evaluation** → Trained policies generate videos for AI analysis and improvement
4. **Iteration** → Process repeats with refined reward functions

---

## Supported Tasks

Available locomotion behaviors for humanoid: `walk`, `jump`

---

## Results and Output Locations

### SDS Experiment Outputs
```
SDS_ANONYM/outputs/YYYY-MM-DD/HH-MM-SS/
├── .hydra/                      # Hydra configuration logs
├── training_footage/            # AI-generated video analysis frames
├── contact_sequence/            # Contact pattern visualization plots
├── pose-estimate/               # Human pose estimation results
├── env_iter*_response*.py       # Generated reward function code
└── env_iter*_response*.txt      # Training log outputs
```

### Isaac Lab Training Logs
```
logs/rsl_rl/
├── g1_flat/                     # Flat terrain experiments
│   └── YYYY-MM-DD_HH-MM-SS/     # Timestamped training run
│       ├── model_*.pt           # Trained policy checkpoints
│       ├── progress.csv         # Training metrics and rewards
│       ├── config.yaml          # Training configuration
│       ├── videos/              # Policy demonstration videos
│       │   └── play/            # Playback video files (.mp4)
│       └── summaries/           # TensorBoard logs
```

### Key Result Files

| Location | Content | Purpose |
|----------|---------|---------|
| `logs/rsl_rl/g1_*/YYYY-MM-DD_HH-MM-SS/videos/play/*.mp4` | Policy demonstration videos | Visual assessment of learned behaviors |
| `logs/rsl_rl/g1_*/YYYY-MM-DD_HH-MM-SS/progress.csv` | Training metrics over time | Performance analysis and convergence monitoring |
| `logs/rsl_rl/g1_*/YYYY-MM-DD_HH-MM-SS/model_*.pt` | Trained neural network weights | Policy checkpoints for evaluation and deployment |
| `SDS_ANONYM/outputs/*/training_footage/*.png` | AI-analyzed video frames | Pose estimation and gait pattern analysis |
| `SDS_ANONYM/outputs/*/contact_sequence/*.png` | Contact force visualizations | Foot contact timing and gait analysis |
| `SDS_ANONYM/outputs/*/env_iter*_response*.py` | Generated reward code | AI-created reward functions for each iteration |

### Accessing Training Videos
After training completes, videos are automatically generated at:
```
logs/rsl_rl/g1_flat/YYYY-MM-DD_HH-MM-SS/videos/play/
```
Each `.mp4` file shows the trained humanoid policy executing the learned locomotion behavior.

---

## Velocity Configuration for Different Actions

**Important**: For different locomotion tasks, you need to adjust the base velocity commands in `source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/velocity_env_cfg.py`:

**Note**: The flat environment configuration in `source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/config/g1/flat_env_cfg.py` also needs velocity range revision for different locomotion tasks (see inline comments in the file).

### **Stationary Actions**
```python
ranges=mdp.UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 0.0),     # Zero forward velocity 
    lin_vel_y=(0.0, 0.0),     # Zero lateral movement
    ang_vel_z=(0.0, 0.0),     # Zero turning
)
```

### **Walking/Running Actions**
```python
ranges=mdp.UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 1.2),     # Forward movement 0-1.2 m/s
    lin_vel_y=(-0.3, 0.3),    # Limited lateral movement
    ang_vel_z=(-0.5, 0.5),    # Conservative turning
)
```

**Why this matters**: The robot tries to track these velocity commands while executing the reward function. For jumping tasks, non-zero velocity commands would conflict with stationary jumping behavior.

---


 