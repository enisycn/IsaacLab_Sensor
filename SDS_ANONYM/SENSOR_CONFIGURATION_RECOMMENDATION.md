# Contact Sensor Configuration for Obstacle Collision Detection

## 🔍 **Current Configuration Analysis**

### **Existing Sensors in `velocity_env_cfg.py`:**
```python
# Current sensors (LIMITED coverage)
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",  # Only feet!
    history_length=3, 
    track_air_time=True, 
    force_threshold=50.0
)

torso_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link", 
    history_length=3, 
    track_air_time=False
)
```

### **Problem:**
- `contact_forces` only monitors **feet** (`.*_ankle_roll_link`)
- `torso_contact` only monitors **torso** 
- **Missing coverage** for collision-prone body parts: thighs, shins, arms, etc.

## 🚀 **Recommended Solution**

### **Option 1: Add Comprehensive Collision Sensor (RECOMMENDED)**
```python
# Add this to MySceneCfg in velocity_env_cfg.py:
collision_sensor = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",  # Monitor ALL robot bodies
    history_length=3,
    track_air_time=False,  # Not needed for collision detection
    force_threshold=1.0,   # Lower threshold for obstacle detection
)
```

### **Option 2: Add Specific Body Part Sensors**
```python
# For more granular control, add multiple sensors:
leg_collisions = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*_thigh|.*_shin|.*_calf",
    history_length=3,
    track_air_time=False,
    force_threshold=1.0,
)

arm_collisions = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*_shoulder.*|.*_upper_arm.*|.*_forearm.*|.*_elbow.*",
    history_length=3,
    track_air_time=False, 
    force_threshold=1.0,
)

torso_collisions = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*torso.*|.*pelvis.*",
    history_length=3,
    track_air_time=False,
    force_threshold=1.0,
)
```

## 📊 **Implementation Steps**

### **Step 1: Update Configuration File**
Add the recommended sensor to `source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/velocity_env_cfg.py`:

```python
@configclass  
class MySceneCfg(InteractiveSceneCfg):
    # ... existing sensors ...
    
    # EXISTING: Limited foot contact sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link", 
        history_length=3, 
        track_air_time=True, 
        force_threshold=50.0
    )
    
    # NEW: Comprehensive collision detection sensor
    collision_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # All robot bodies
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,  # Sensitive collision detection
    )
    
    # EXISTING: Torso contact (for termination)
    torso_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link", 
        history_length=3, 
        track_air_time=False
    )
```

### **Step 2: Update Metrics Script**
Modify the collision detection to use the new sensor:

```python
def _update_obstacle_collision_count(self):
    # Try the comprehensive collision sensor first
    contact_sensor = self.env.unwrapped.scene.sensors.get("collision_sensor")
    
    if contact_sensor is None:
        # Fallback to existing sensors
        contact_sensor = self.env.unwrapped.scene.sensors.get("contact_forces")
        # ... existing logic ...
```

## 🎯 **Benefits of Comprehensive Sensor**

### **Option 1 Advantages (Recommended):**
- ✅ **Complete Coverage**: Monitors all robot bodies
- ✅ **Simpler Configuration**: Single sensor covers everything  
- ✅ **Future-Proof**: Works with any robot configuration
- ✅ **Performance**: Efficient single sensor operation
- ✅ **Flexible Filtering**: Can filter collision-prone bodies in code

### **Option 2 Advantages:**
- ✅ **Granular Control**: Separate sensors for different body regions
- ✅ **Targeted Thresholds**: Different force thresholds per body type
- ✅ **Detailed Analytics**: Can analyze collision patterns by body region
- ❌ **More Complex**: Multiple sensors to configure and maintain

## 🛠️ **Current Script Compatibility**

The revised collision detection script will work with **either approach**:

1. **With comprehensive sensor**: Uses `collision_sensor` for full body monitoring
2. **With existing sensors**: Filters collision-prone bodies from `contact_forces`
3. **Fallback gracefully**: Handles missing sensors with appropriate warnings

## 📋 **Recommendation**

**Use Option 1 (Comprehensive Sensor)** because:
- Simpler to implement and maintain
- Better performance (single sensor)
- Complete collision coverage
- Future-proof for different robots
- Already supported by the revised metrics script

The script's body pattern filtering will automatically focus on collision-prone parts while having access to complete contact data. 