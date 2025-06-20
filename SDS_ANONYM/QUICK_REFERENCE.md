# SDS Isaac Lab Integration - Quick Reference Guide

[![Quick Start](https://img.shields.io/badge/Quick%20Start-Ready-green.svg)](#quick-start)
[![Troubleshooting](https://img.shields.io/badge/Troubleshooting-Available-blue.svg)](#troubleshooting)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)

> **One-page reference for commands, troubleshooting, and system status**

---

## 🚀 **Quick Start**

### **Basic Setup**
```bash
# 1. Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# 2. Navigate to project
cd /home/enis/IsaacLab/SDS_ANONYM

# 3. Run SDS (basic)
python SDS/sds.py task=trot train_iterations=50 iteration=3 sample=4
```

### **Task Options**
- `task=trot` - Trotting gait (most stable)
- `task=pace` - Pacing gait  
- `task=hop` - Hopping behavior
- `task=bound` - Bounding gait

### **Common Parameters**
- `train_iterations=50` - Training duration (50-1000)
- `iteration=3` - SDS optimization iterations (1-5)
- `sample=4` - Samples per iteration (2-8)

---

## ⚡ **Instant Commands**

### **Test System Health**
```bash
# Quick system check
cd /home/enis/IsaacLab
./isaaclab.sh -p scripts/tools/list_envs.py | grep SDS

# Expected output: Isaac-SDS-Velocity-Flat-Unitree-Go1-v0
```

### **Debug Run**
```bash
# Minimal test (fast)
python SDS/sds.py task=trot train_iterations=5 iteration=1 sample=1

# Verbose errors
HYDRA_FULL_ERROR=1 python SDS/sds.py task=trot train_iterations=5 iteration=1 sample=1
```

### **Manual Training Test**
```bash
cd /home/enis/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-SDS-Velocity-Flat-Unitree-Go1-v0 --num_envs=512 --max_iterations=5 --headless
```

---

## 🔧 **Troubleshooting**

### **Training Fails to Start**
```bash
# Problem: Environment not found
# Solution: Check registration
./isaaclab.sh -p scripts/tools/list_envs.py | grep SDS

# Expected: 4 SDS environments listed
```

### **Import Errors**
```bash
# Problem: matrix_from_quat not defined
# Solution: Check import exists
grep "matrix_from_quat" source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/mdp/rewards.py

# Expected: import line present
```

### **GPU Memory Issues**
```bash
# Problem: CUDA out of memory
# Solution: Reduce environments
python SDS/sds.py task=trot train_iterations=50 iteration=2 sample=2 num_envs=2048

# Or reduce batch size in config
```

### **Log Parsing Errors**
```bash
# Problem: KeyError in log parsing
# Solution: Check utils/misc.py has updated construct_run_log()
grep "Mean episode length:" SDS_ANONYM/utils/misc.py

# Expected: Isaac Lab format handling present
```

---

## 📊 **System Status Check**

### **Health Indicators**
```bash
# ✅ All should return success (exit code 0)

# 1. Environment registration
./isaaclab.sh -p scripts/tools/list_envs.py | grep SDS > /dev/null && echo "✅ PASS" || echo "❌ FAIL"

# 2. Import dependencies  
python -c "from isaaclab.utils.math import matrix_from_quat; print('✅ PASS')" 2>/dev/null || echo "❌ FAIL"

# 3. Reward file exists
test -f source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/mdp/rewards.py && echo "✅ PASS" || echo "❌ FAIL"

# 4. OpenAI key set
test -n "$OPENAI_API_KEY" && echo "✅ PASS" || echo "❌ FAIL"
```

### **Performance Check**
```bash
# Check latest successful run
ls -la SDS_ANONYM/outputs/sds/ | tail -1

# Expected: Recent timestamp directory with multiple files
```

---

## 📁 **Key File Locations**

### **Critical Files**
- **Main SDS Engine**: `SDS_ANONYM/SDS/sds.py`
- **Reward Integration**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/mdp/rewards.py`
- **Environment Config**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/velocity_env_cfg.py`
- **Log Parser**: `SDS_ANONYM/utils/misc.py`

### **Configuration Files**
- **Main Config**: `SDS_ANONYM/SDS/cfg/config.yaml`
- **Task Configs**: `SDS_ANONYM/SDS/cfg/task/`
- **Prompts**: `SDS_ANONYM/SDS/prompts/`

### **Output Locations**
- **SDS Results**: `SDS_ANONYM/outputs/sds/`
- **Training Logs**: `/home/enis/IsaacLab/logs/rsl_rl/unitree_go1_flat/`
- **Videos**: `logs/rsl_rl/unitree_go1_flat/*/videos/`

---

## 🎯 **Debug Mode - Individual Rewards**

### **Enable Component Visibility**
```bash
# Edit file:
nano source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/velocity_env_cfg.py

# Change these lines:
sds_velocity_tracking = RewTerm(func=mdp.sds_velocity_tracking, weight=5.0)
sds_height_stability = RewTerm(func=mdp.sds_height_stability, weight=3.0)
sds_orientation_stability = RewTerm(func=mdp.sds_orientation_stability, weight=4.0)
sds_trot_gait = RewTerm(func=mdp.sds_trot_gait, weight=6.0)
sds_action_smoothness = RewTerm(func=mdp.sds_action_smoothness, weight=0.02)
sds_custom = RewTerm(func=mdp.sds_custom_reward, weight=0.0)  # Disable combined

# Then run training to see individual components
```

### **Revert to Combined Mode**
```bash
# Change back to:
sds_velocity_tracking = RewTerm(func=mdp.sds_velocity_tracking, weight=0.0)
sds_height_stability = RewTerm(func=mdp.sds_height_stability, weight=0.0)
sds_orientation_stability = RewTerm(func=mdp.sds_orientation_stability, weight=0.0)
sds_trot_gait = RewTerm(func=mdp.sds_trot_gait, weight=0.0)
sds_action_smoothness = RewTerm(func=mdp.sds_action_smoothness, weight=0.0)
sds_custom = RewTerm(func=mdp.sds_custom_reward, weight=1.0)  # Enable combined
```

---

## 📈 **Performance Expectations**

### **Normal Performance Ranges**
- **Training Speed**: 7,400-19,600 steps/s
- **GPU Memory**: 8-12GB VRAM
- **Episode Length**: >1000 (good), >500 (ok), <200 (problem)
- **Reward Values**: Positive (good), around 0 (ok), negative (problem)

### **Timing Expectations**
- **Single Sample**: 2-4 minutes total
- **2 iterations × 2 samples**: ~15 minutes
- **5 iterations × 4 samples**: ~45 minutes

---

## 🆘 **Emergency Recovery**

### **System Not Working At All**
```bash
# 1. Check Isaac Lab installation
cd /home/enis/IsaacLab && ./isaaclab.sh --help

# 2. Re-source environment
source /path/to/conda/activate sam2

# 3. Check GPU access
nvidia-smi

# 4. Test basic Isaac Lab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Unitree-Go1-v0 --num_envs=128 --max_iterations=1 --headless
```

### **Restore Default Configuration**
```bash
# If configuration is corrupted, restore from working version
git checkout HEAD -- source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/

# Or check backup files if available
```

---

## 📞 **Support Resources**

### **Documentation Priority Order**
1. **This Quick Reference** (immediate help)
2. **`PROJECT_STATUS_SUMMARY.md`** (current system status)
3. **`README_SDS_COMPREHENSIVE.md`** (complete overview)
4. **`INTEGRATION_CHANGELOG.md`** (detailed changes)
5. **Isaac Lab Documentation** (framework help)

### **Common Solutions Index**
- **Training Crashes**: Check imports in `rewards.py`
- **Poor Performance**: Check velocity frame usage (body vs world)
- **Log Errors**: Check `construct_run_log()` in `misc.py`
- **GPU Issues**: Reduce `num_envs` parameter
- **API Errors**: Check environment registration

---

**Last Updated**: 2025-06-19 | **Status**: ✅ Production Ready | **Success Rate**: 100% 