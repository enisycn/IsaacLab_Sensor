# SDS Gait-Specific Tasks Guide

## 🎯 Available Gait Tasks

The SDS system now supports **5 gait-specific tasks** based on the updated humanoid gait classification:

| **Gait** | **Task File** | **Description** | **Video** | **Use Case** |
|----------|---------------|-----------------|-----------|--------------|
| **Walk** | `walk.yaml` | Humanoid Robot Walking | `walk.mp4` | Standard walking locomotion |
| **Jump** | `jump.yaml` | Humanoid Robot Jumping | `jump.mp4` | Synchronized takeoff/landing patterns |
| **March** | `march.yaml` | Humanoid Robot Marching | `march.mp4` | Controlled high knee lift movement |
| **Sprint** | `sprint.yaml` | Humanoid Robot Sprinting | `sprint.mp4` | High-speed extended flight phases |
| **Pace** | `pace.yaml` | Humanoid Robot Pacing | `pace.mp4` | Lateral stepping locomotion |

## 🚀 How to Run Different Gait Tasks

### **Method 1: Command Line Override**
```bash
# Train Walking gait (default)
python sds.py

# Train Jumping gait
python sds.py task=jump

# Train Marching gait  
python sds.py task=march

# Train Sprinting gait
python sds.py task=sprint

# Train Pacing gait
python sds.py task=pace
```

### **Method 2: Config File Modification**
Edit `SDS_ANONYM/SDS/cfg/config.yaml`:
```yaml
defaults:
  - _self_
  - task: jump  # Change this to: walk, jump, march, sprint, or pace
  - override hydra/launcher: local
  - override hydra/output: local
```

## 📁 Task Configuration Structure

Each task YAML file contains identical options for consistency:

```yaml
description: "Humanoid Robot [GAIT_NAME]"  # GPT analysis hint
video: [video_file].mp4                    # Video to analyze
crop: False                                # No video cropping
crop_option: None                          # No crop method
use_annotation: True                       # Use pose estimation
grid_size: 8                              # 8x8 frame grid
```

## 🎬 Video-Task Connections

| **Gait** | **Video File** | **Content Description** |
|----------|---------------|------------------------|
| **Walk** | `walk.mp4` | Standard humanoid walking pattern |
| **Jump** | `jump.mp4` | Jumping/hopping movements |
| **March** | `march.mp4` | Controlled marching with high knee lift |
| **Sprint** | `sprint.mp4` | High-speed running with extended flight phases |
| **Pace** | `pace.mp4` | Lateral stepping locomotion patterns |

## 🔧 Gait-Specific Behavior

### **Walk Task**
- **Contact Pattern**: 1 or 2 feet (alternating single + double support)
- **Characteristics**: Stable alternating locomotion with double support phases
- **GPT Focus**: Contact alternation, step timing, ground contact duration

### **Jump Task**
- **Contact Pattern**: 0 or 2 feet (synchronized takeoff/landing + flight)
- **Characteristics**: Synchronized two-foot movements with flight phases
- **GPT Focus**: Air time consistency, synchronized landing, vertical motion

### **March Task**
- **Contact Pattern**: 1 foot (controlled single support)
- **Characteristics**: Controlled alternating with deliberate high knee lift
- **GPT Focus**: Single support stability, controlled movement timing

### **Sprint Task**
- **Contact Pattern**: 0 or 1 feet (extended flight phases)
- **Characteristics**: Minimal contact time with extended aerial phases
- **GPT Focus**: Flight phase rewards, minimal ground contact

### **Pace Task**
- **Contact Pattern**: 1 or 2 feet (lateral movement stability)
- **Characteristics**: Side-to-side movement with stable contact
- **GPT Focus**: Lateral movement, side-stepping patterns, directional changes

## 📊 Expected Outcomes

Each gait task will generate **gait-appropriate reward functions**:

- **Walk**: Contact alternation rewards, step timing
- **Jump**: Air time consistency, synchronized contact rewards  
- **March**: Single support stability, controlled movement rewards
- **Sprint**: Flight phase emphasis, minimal contact rewards
- **Pace**: Lateral movement rewards, side-stepping patterns

## 🎯 Usage Examples

### Training a Jump Gait
```bash
cd SDS_ANONYM/SDS
python sds.py task=jump iteration=5 sample=8
```

### Training Multiple Gaits in Sequence
```bash
# Train each gait for 3 iterations
python sds.py task=walk iteration=3 sample=5
python sds.py task=jump iteration=3 sample=5  
python sds.py task=march iteration=3 sample=5
python sds.py task=sprint iteration=3 sample=5
python sds.py task=pace iteration=3 sample=5
```

### Custom Video Analysis
```bash
# Use with specific configuration
python sds.py task=jump num_envs=2048 train_iterations=500
```

## 🔄 System Workflow

1. **Task Selection**: Choose gait via command line or config file
2. **Video Processing**: System loads specified video and processes frames
3. **GPT Analysis**: 5-agent pipeline analyzes video with gait-specific hints
4. **Reward Generation**: GPT generates gait-appropriate reward functions
5. **Training**: Isaac Lab trains G1 humanoid with generated rewards
6. **Evaluation**: System evaluates results against original gait pattern

## ✅ Validation

Each task configuration has been **validated** to ensure:
- ✅ Same options as walking for consistency
- ✅ Appropriate video connections
- ✅ Gait-specific descriptions for GPT analysis  
- ✅ Compatible with existing SDS pipeline
- ✅ Proper YAML formatting 