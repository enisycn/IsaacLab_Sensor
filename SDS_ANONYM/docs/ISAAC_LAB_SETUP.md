# SDS + Isaac Lab Setup Guide

## 🎯 **Prerequisites**

### Isaac Lab Installation
Ensure Isaac Lab is properly installed and working:

```bash
# Test Isaac Lab installation
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go1-v0 --headless
```

### Python Environment
SDS requires Python 3.8+ with the following packages:

```bash
# Core dependencies
pip install opencv-python openai requests pyyaml matplotlib

# Video processing
pip install pillow imageio[ffmpeg]

# Optional: Advanced visualization
pip install plotly seaborn
```

## 🚀 **Installation Steps**

### 1. Clone SDS Repository
```bash
cd /path/to/your/IsaacLab
git clone --recursive https://github.com/sdsreview/SDS_ANONYM.git
```

### 2. Install SDS Components
```bash
cd SDS_ANONYM

# Install core SDS package
cd SDS && pip install -e .

# Install ViTPose for pose estimation
cd ../easy_ViTPose && pip install -e . && mkdir checkpoints

# Download ViTPose++ model
wget https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/ap10k/vitpose-h-ap10k.pth -P checkpoints

# Return to main directory
cd ..
```

### 3. Set Up OpenAI API
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 4. Verify Installation
```bash
# Test video processing
python -c "from utils.vid_utils import create_grid_image; print('Video utils working')"

# Test ViTPose
python -c "from utils.easy_vit_pose import vitpose_inference; print('ViTPose working')"

# Test OpenAI connection
python -c "import openai; print('OpenAI connected')"
```

## 🔧 **Isaac Lab Integration**

### Environment Configuration
SDS works with existing Isaac Lab environments. Ensure you have:

```bash
# SDS environments registered
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-SDS-Velocity-Rough-Unitree-Go1-v0
```

### Custom Reward Integration
SDS automatically generates reward functions compatible with Isaac Lab's MDP framework:

```python
# Example auto-generated reward configuration
@configclass
class SDSCustomRewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,  # From GPT analysis
        params={"command_name": "base_velocity", "std": 0.5}
    )
```

## 🎬 **Usage**

### Basic Video Processing
```bash
cd SDS
python sds.py task=custom video=your_demo.mp4
```

### With Isaac Lab Environments
```bash
# Process video and train in Isaac Lab
python sds_isaac_lab.py \
    --video path/to/demo.mp4 \
    --robot go1 \
    --environment rough \
    --num_envs 4096
```

### Batch Processing
```bash
# Process multiple videos
python sds_isaac_lab.py \
    --video_dir path/to/videos/ \
    --batch_process \
    --output_dir results/
```

## 📊 **Expected Workflow**

1. **📹 Video Analysis**: ViTPose++ extracts poses from demonstration video
2. **🧠 GPT Analysis**: GPT-4V analyzes skill and generates reward functions
3. **🔧 Environment Setup**: Auto-create Isaac Lab environment with custom rewards
4. **🏃 Training**: Train policy using Isaac Lab + RSL-RL
5. **📈 Evaluation**: Test learned policy and generate comparison videos
6. **🤖 Deployment**: Export policy for real robot deployment

## 🐛 **Troubleshooting**

### Common Issues

**Import Errors**:
```bash
# Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/IsaacLab/source"
```

**OpenAI API Issues**:
```bash
# Test API access
python -c "import openai; client = openai.OpenAI(); print(client.models.list())"
```

**ViTPose Model Missing**:
```bash
# Re-download model
cd easy_ViTPose
wget https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/ap10k/vitpose-h-ap10k.pth -P checkpoints
```

**Isaac Lab Environment Issues**:
```bash
# Verify Isaac Lab environments
./isaaclab.sh -p scripts/tools/list_envs.py | grep SDS
```

## 🔗 **Additional Resources**

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL Training Guide](https://rsl-rl.readthedocs.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [ViTPose++ Documentation](https://github.com/ViTAE-Transformer/ViTPose)

## 🎯 **Next Steps**

1. Test with provided demo videos in `docs/resources/demo/`
2. Create your own demonstration videos
3. Experiment with different quadruped skills
4. Deploy learned policies on real robots

---

**Ready to teach your robot new tricks from video demonstrations! 🎥→🤖** 