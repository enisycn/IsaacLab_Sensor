# SDS Humanoid Adaptation Plan: Go1 → G1 Migration

[![Status](https://img.shields.io/badge/Status-Implementation%20Ready-green.svg)](#)
[![Robot](https://img.shields.io/badge/Target-Unitree%20G1%20Humanoid-blue.svg)](#)
[![Framework](https://img.shields.io/badge/Framework-Isaac%20Lab-green.svg)](#)
[![Approach](https://img.shields.io/badge/Strategy-Copy%20&%20Adapt-orange.svg)](#)

> **Objective**: Add Unitree G1 humanoid robot support to SDS framework by creating G1-specific environments following Isaac Lab's established patterns.

---

## 🚨 **MAJOR DISCOVERY: Simplified Migration Path**

### **🔍 Critical Findings**
After comprehensive codebase analysis and web verification:

- ✅ **Isaac Lab G1 Fully Supported**: `Isaac-Velocity-Flat-G1-v0` environments exist and active
- ✅ **SDS Has Robot-Specific Structure**: Separate `/sds/velocity/config/go1/` path discovered  
- ❌ **G1 SDS Missing**: No `/sds/velocity/config/g1/` exists yet
- 🎯 **Much Simpler Path**: Copy & adapt existing Go1 SDS → Create G1 SDS
- ⏱️ **50% Time Reduction**: ~1.5 hours vs original 3 hours

### **🎯 New Strategy: Copy & Adapt Approach**
Instead of modifying the entire SDS framework, leverage Isaac Lab's existing patterns:
1. **Create G1 SDS environments** by copying Go1 SDS structure
2. **Adapt robot specifications** (Go1 → G1 parameters)  
3. **Register new environments** in Isaac Lab
4. **Minimal SDS system updates** for robot selection

---

## 📋 **REVISED Implementation Phases**

### **Phase 1: Create G1 SDS Environment** ⏱️ *~45 minutes*

#### **1.1 Directory Structure Creation**
**Command**:
```bash
cd /home/enis/IsaacLab
mkdir -p source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/config/g1/agents
```

#### **1.2 Copy Base Structure**
**Copy Go1 SDS as template**:
```bash
# Copy complete Go1 SDS structure  
cp -r source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/config/go1/* \
      source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/config/g1/
```

#### **1.3 Adapt G1 Flat Environment**
**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/config/g1/flat_env_cfg.py`

**Key Changes**:
```python
"""
SDS G1 Flat Environment Configuration.

This configuration is specifically designed for the SDS project
using Unitree G1 humanoid robot on flat terrain.
"""

from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
import isaaclab_assets

# CHANGE: Import G1 configuration instead of Go1
from isaaclab_assets import G1_MINIMAL_CFG

from .rough_env_cfg import SDSG1RoughEnvCfg  # Updated class name


@configclass  
class SDSG1FlatEnvCfg(SDSG1RoughEnvCfg):
    """SDS Unitree G1 flat terrain environment configuration."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # CRITICAL: For SDS, use ONLY GPT-generated sds_custom reward
        # self.rewards.sds_custom is already configured in the base class

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class SDSG1FlatEnvCfg_PLAY(SDSG1FlatEnvCfg):
    """SDS Unitree G1 flat terrain environment configuration for play/testing."""
    
    # G1 Humanoid tracking - Bipedal optimized camera
    viewer = ViewerCfg(
        origin_type="asset_root",    # Automatically follow robot base
        asset_name="robot",          # Track the robot asset
        env_index=0,                # Environment 0 (single robot)
        eye=(0.0, -3.0, 1.2),       # Side view for humanoid - higher camera for full body
        lookat=(0.0, 0.0, 0.8),     # Look at robot center of mass (higher for humanoid)
    )
    
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1  # Single robot for close tracking
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
```

#### **1.4 Adapt G1 Rough Environment**
**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/config/g1/rough_env_cfg.py`

**Key Changes**:
```python
"""
SDS G1 Rough Environment Configuration.

This configuration is specifically designed for the SDS project  
using Unitree G1 humanoid robot on rough terrain.
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

# CHANGE: Import G1 instead of Go1
from isaaclab_assets import G1_MINIMAL_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Import base SDS reward configuration
from isaaclab_tasks.manager_based.sds.velocity.mdp import SDSRewardsCfg


@configclass
class SDSG1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """SDS Unitree G1 rough terrain environment configuration."""

    rewards: SDSRewardsCfg = SDSRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # CHANGE: Use G1 robot configuration
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # G1 HUMANOID SPECIFIC ADAPTATIONS:
        
        # Termination condition - use torso instead of base
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        
        # Randomization - adjust for bipedal stability
        self.events.push_robot = None  # More conservative for bipedal robot
        self.events.add_base_mass = None  # Avoid mass perturbations for humanoid
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)  # Smaller range for stability
        
        # External forces - use torso link for humanoid
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        
        # Reset parameters - bipedal specific
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-1.57, 1.57)},  # Smaller range
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # Commands - conservative for humanoid
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)  # Max 1 m/s forward
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)  # Limited lateral
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)  # Limited turning


@configclass  
class SDSG1RoughEnvCfg_PLAY(SDSG1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        
        # spawn the robot randomly in the grid
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Conservative commands for demonstration
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
```

#### **1.5 G1 Environment Registration**
**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/config/g1/__init__.py`

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments for SDS project with G1 humanoid.
##

gym.register(
    id="Isaac-SDS-Velocity-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SDSG1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-SDS-Velocity-Flat-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SDSG1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-SDS-Velocity-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SDSG1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-SDS-Velocity-Rough-G1-Play-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:SDSG1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
```

**Validation**:
- [ ] Directory structure created successfully
- [ ] All G1 configurations adapted from Go1 
- [ ] Environment registration follows Isaac Lab patterns
- [ ] Robot specifications updated for humanoid

---

### **Phase 2: Update SDS System Logic** ⏱️ *~30 minutes*

#### **2.1 Core Configuration Addition**
**File**: `SDS/cfg/config.yaml`

**Add robot selection parameters** (around line 15):
```yaml
# Robot and terrain selection
robot: go1              # Options: go1, g1  
terrain: flat          # Options: flat, rough
```

#### **2.2 Environment Detection Logic**
**File**: `SDS/sds.py`

**Add after line 50 in `main()` function**:
```python
env_name = cfg.env_name.lower()

# ADD: Dynamic robot and environment configuration
robot_type = getattr(cfg, 'robot', 'go1').lower()  # Default to go1 for backward compatibility
terrain_type = getattr(cfg, 'terrain', 'flat').lower()

# Generate SDS environment name based on robot type
if robot_type == 'g1':
    sds_env_name = f"Isaac-SDS-Velocity-{terrain_type.title()}-G1-v0"
    sds_play_env_name = f"Isaac-SDS-Velocity-{terrain_type.title()}-G1-Play-v0"
    logging.info(f"Using Unitree G1 Humanoid robot configuration")
elif robot_type == 'go1':
    sds_env_name = f"Isaac-SDS-Velocity-{terrain_type.title()}-Unitree-Go1-v0"
    sds_play_env_name = f"Isaac-SDS-Velocity-{terrain_type.title()}-Unitree-Go1-Play-v0"
    logging.info(f"Using Unitree Go1 Quadruped robot configuration")
else:
    raise ValueError(f"Unsupported robot type: {robot_type}. Supported: go1, g1")

logging.info(f"Robot: {robot_type}, Terrain: {terrain_type}")
logging.info(f"SDS Environment: {sds_env_name}")
```

#### **2.3 Update Environment Commands**
**File**: `SDS/sds.py`

**Replace hardcoded environment names in 3 locations**:

**Location 1** (~line 281 in `train_command`):
```python
# FIND:
f"--task=Isaac-SDS-Velocity-Flat-Unitree-Go1-v0",

# REPLACE WITH:
f"--task={sds_env_name}",
```

**Location 2** (~line 362 in `run_isaac_lab_env`):
```python
# FIND:
"--task=Isaac-SDS-Velocity-Flat-Unitree-Go1-Play-v0",

# REPLACE WITH:
f"--task={sds_play_env_name}",
```

**Location 3** (~line 395 in `run_isaac_lab_env_with_agent`):
```python
# FIND:
"--task=Isaac-SDS-Velocity-Flat-Unitree-Go1-Play-v0",

# REPLACE WITH:
f"--task={sds_play_env_name}",
```

**Validation**:
- [ ] All environment name variables are in scope
- [ ] F-string syntax is correct
- [ ] Backward compatibility maintained with Go1 default

---

### **Phase 3: Prompt and Task Updates** ⏱️ *~15 minutes*

#### **3.1 Task Descriptions** *(Optional)*
**Files**: `SDS/cfg/task/*.yaml`

**Update descriptions for humanoid context**:

**trot.yaml**:
```yaml
description: Humanoid Robot Walking  # FROM: Quadruped Robot Trotting
```

**pace.yaml**: 
```yaml
description: Humanoid Robot Running  # FROM: Horse in pacing gait
```

**hop.yaml**:
```yaml
description: Humanoid Robot Jumping  # FROM: Quadruped Robot Hopping 
```

**bound.yaml**:
```yaml  
description: Humanoid Robot Sprinting  # FROM: Quadruped Robot Bounding
```

#### **3.2 Robot Specification Injection** *(Optional Enhancement)*
**File**: `SDS/sds.py`

**Add after reward signature loading** (~line 65):
```python
# ADD: Dynamic robot specification injection for better prompts
if robot_type == 'g1':
    robot_specs_injection = f"""
# ACTIVE ROBOT: UNITREE G1 HUMANOID
nominal_height = 0.74  # meters
contact_bodies = ".*_ankle_roll_link"  
num_feet = 2
robot_type = "bipedal_humanoid"
critical_stability_factor = 5.0  # Higher than quadruped
balance_priority = "high"  # Essential for bipedal locomotion
"""
else:  # go1 default
    robot_specs_injection = f"""
# ACTIVE ROBOT: UNITREE GO1 QUADRUPED  
nominal_height = 0.34  # meters
contact_bodies = ".*_foot"
num_feet = 4
robot_type = "quadrupedal"
critical_stability_factor = 1.0  # Natural quadruped stability
"""

# Inject robot specifications into prompts
initial_reward_engineer_system = initial_reward_engineer_system.replace(
    "You are a reward engineer", 
    robot_specs_injection + "\n\nYou are a reward engineer"
)

logging.info(f"Injected {robot_type.upper()} robot specifications into prompts")
```

**Validation**:
- [ ] Task descriptions updated (optional)
- [ ] Robot specification injection working (optional)
- [ ] All changes maintain YAML/Python syntax

---

## 🧪 **Testing Integration** 

### **Basic Functionality Tests**

#### **Test G1 Humanoid Configuration**:
```bash
cd /home/enis/IsaacLab/SDS_ANONYM

# Test G1 humanoid walking (interpreted from trot task)
python SDS/sds.py task=trot robot=g1 terrain=flat train_iterations=50 iteration=2 sample=2

# Test G1 on rough terrain
python SDS/sds.py task=pace robot=g1 terrain=rough train_iterations=50 iteration=1 sample=1
```

#### **Test Backward Compatibility**:
```bash
# Test Go1 quadruped (should work exactly as before)  
python SDS/sds.py task=trot robot=go1 terrain=flat train_iterations=50 iteration=2 sample=2

# Test without robot parameter (should default to go1)
python SDS/sds.py task=trot train_iterations=50 iteration=2 sample=2
```

#### **Validate Environment Registration**:
```bash
cd /home/enis/IsaacLab

# Check G1 SDS environments are registered
./isaaclab.sh -p scripts/environments/list_envs.py | grep "SDS.*G1"

# Expected output:
# Isaac-SDS-Velocity-Flat-G1-v0
# Isaac-SDS-Velocity-Flat-G1-Play-v0  
# Isaac-SDS-Velocity-Rough-G1-v0
# Isaac-SDS-Velocity-Rough-G1-Play-v0
```

---

## 📊 **File Creation & Modification Summary**

### **NEW Files Created** *(Following Isaac Lab patterns)*:
| File | Purpose | Size |
|------|---------|------|
| `source/.../sds/velocity/config/g1/__init__.py` | Environment registration | ~2KB |
| `source/.../sds/velocity/config/g1/flat_env_cfg.py` | G1 flat terrain config | ~3KB |
| `source/.../sds/velocity/config/g1/rough_env_cfg.py` | G1 rough terrain config | ~4KB |
| `source/.../sds/velocity/config/g1/agents/` | G1 agent configs | ~5KB |

### **Modified Files** *(Minimal changes)*:
| File | Changes | Risk |
|------|---------|------|
| `SDS/cfg/config.yaml` | Add robot/terrain params | Low |
| `SDS/sds.py` | Environment name logic | Low |
| `SDS/cfg/task/*.yaml` | Update descriptions (optional) | Low |

### **Files NOT Modified**:
- Video files (`videos/*.mp4`) - Reused with new interpretation
- Existing Go1 configurations - Unchanged
- Core SDS utilities - No changes needed
- Isaac Lab base environments - Already exist

---

## 🎯 **Web-Verified Benefits**

### **✅ Confirmed Isaac Lab G1 Support**
- **Official NVIDIA Partnership**: Unitree is NVIDIA's humanoid robotics partner
- **Active Development**: G1 environments in Isaac Lab main branch 
- **Complete Integration**: Contact sensors, joint configurations, physics validated
- **Production Ready**: Used in commercial policy deployment examples

### **✅ Market Position Advantages**  
- **Most Accessible**: $16,000 starting price vs $50K+ competitors
- **Established Ecosystem**: Extensive developer community and resources
- **Proven Platform**: Successfully deployed in research and commercial applications
- **Future Growth**: Leading the humanoid robotics adoption curve

---

## 🚀 **Revised Implementation Timeline**

**Total Estimated Time**: ~1.5 hours (50% reduction from original 3 hours!)

```
Phase 1: Create G1 SDS Environments    [████████████████████████] 45 min
Phase 2: Update SDS System Logic       [████████████████████████] 30 min  
Phase 3: Prompt and Task Updates       [████████████████████████] 15 min
```

### **Implementation Order**:
1. **Phase 1 First** - Creates the foundation G1 environments
2. **Test after Phase 1** - Verify environments load in Isaac Lab
3. **Phase 2 Second** - Enables robot selection in SDS  
4. **Test after Phase 2** - Verify full G1 workflow
5. **Phase 3 Last** - Optional enhancements for better prompts

---

## 🔄 **Rollback Procedure**

**Simple rollback since we only ADD files**:

```bash
# 1. Remove new G1 SDS directory
rm -rf source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/config/g1/

# 2. Reset modified files  
cd /home/enis/IsaacLab/SDS_ANONYM
git checkout -- SDS/cfg/config.yaml
git checkout -- SDS/sds.py  
git checkout -- SDS/cfg/task/

# 3. Verify original functionality
python SDS/sds.py task=trot train_iterations=50 iteration=1 sample=1
```

---

## ✅ **Success Criteria**

### **Phase Completion Validation**:

**Phase 1 Complete**: ✅
- [ ] G1 SDS directory structure created
- [ ] All configuration files adapted from Go1
- [ ] Environment registration successful
- [ ] Isaac Lab recognizes G1 SDS environments

**Phase 2 Complete**: ✅  
- [ ] Robot selection parameter works
- [ ] Environment names generate correctly  
- [ ] Both G1 and Go1 modes functional
- [ ] Backward compatibility maintained

**Phase 3 Complete**: ✅
- [ ] Task descriptions updated (optional)
- [ ] Robot-specific prompts injected (optional)
- [ ] All syntax validated

### **Final Integration Success**:

**System Ready When**:
- ✅ **G1 humanoid training works**: `python SDS/sds.py task=trot robot=g1`
- ✅ **Go1 quadruped still works**: `python SDS/sds.py task=trot robot=go1`  
- ✅ **Default behavior maintained**: `python SDS/sds.py task=trot` (defaults to go1)
- ✅ **All combinations functional**: Both robots × both terrains × all tasks
- ✅ **Environment registration verified**: G1 SDS environments list properly

---

## 📝 **Key Implementation Notes**

### **Design Decisions**
- **Isaac Lab Native**: Follows established SDS environment patterns
- **Copy & Adapt**: Cleaner than framework modification
- **Backward Compatible**: Go1 remains default, no breaking changes
- **Future Extensible**: Easy to add more robots using same pattern

### **Critical Specifications Verified**
- **G1 Contact Bodies**: `.*_ankle_roll_link` (confirmed in Isaac Lab)
- **G1 Robot Config**: `G1_MINIMAL_CFG` (available in isaaclab_assets)
- **Environment Naming**: `Isaac-SDS-Velocity-{Terrain}-G1-v0` pattern
- **Torso Link**: `torso_link` for height scanner and termination

### **Known Advantages**
- **Leverages existing demonstrations** - Same videos, humanoid interpretation
- **Professional integration** - Follows Isaac Lab's established conventions  
- **Reduced maintenance** - Separate configs don't interfere with Go1
- **Immediate testing** - Can validate incrementally at each phase

---

**Ready to Begin?** Start with **Phase 1: Create G1 SDS Environment** using the Copy & Adapt approach! 🚀