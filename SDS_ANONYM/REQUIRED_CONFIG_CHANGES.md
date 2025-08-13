# Required Configuration Changes for Comprehensive Obstacle Collision Detection

## 🚨 **Current Problem**
The SDS environment configuration only has limited contact sensors:

```python
# CURRENT CONFIG in velocity_env_cfg.py (LIMITED COVERAGE)
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",  # FEET ONLY
    history_length=3, 
    track_air_time=True, 
    force_threshold=50.0
)

torso_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # TORSO ONLY  
    history_length=3, 
    track_air_time=False
)
```

**Result**: Only 1 body (torso) available for collision detection ❌

## ✅ **Required Solution**

### **Add Comprehensive Collision Sensor**

Add this to `velocity_env_cfg.py` in the scene sensors section:

```python
# ADD THIS: Comprehensive collision sensor for obstacle detection
collision_sensor = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",  # ALL ROBOT BODIES
    history_length=3,
    track_air_time=False,
    force_threshold=0.5  # Lower threshold for collision detection
)

# OR MORE SPECIFIC: Target collision-prone bodies only
collision_sensor = ContactSensorCfg(
    prim_path=[
        # G1 Collision-prone body patterns
        "{ENV_REGEX_NS}/Robot/pelvis",
        "{ENV_REGEX_NS}/Robot/torso_link", 
        "{ENV_REGEX_NS}/Robot/pelvis_contour_link",
        "{ENV_REGEX_NS}/Robot/left_hip_pitch_link",
        "{ENV_REGEX_NS}/Robot/right_hip_pitch_link",
        "{ENV_REGEX_NS}/Robot/left_hip_roll_link",
        "{ENV_REGEX_NS}/Robot/right_hip_roll_link",
        "{ENV_REGEX_NS}/Robot/left_hip_yaw_link",
        "{ENV_REGEX_NS}/Robot/right_hip_yaw_link",
        "{ENV_REGEX_NS}/Robot/left_knee_link",
        "{ENV_REGEX_NS}/Robot/right_knee_link",
        "{ENV_REGEX_NS}/Robot/left_shoulder_pitch_link",
        "{ENV_REGEX_NS}/Robot/right_shoulder_pitch_link",
        "{ENV_REGEX_NS}/Robot/left_shoulder_roll_link",
        "{ENV_REGEX_NS}/Robot/right_shoulder_roll_link",
        "{ENV_REGEX_NS}/Robot/left_shoulder_yaw_link",
        "{ENV_REGEX_NS}/Robot/right_shoulder_yaw_link",
        "{ENV_REGEX_NS}/Robot/left_elbow_pitch_link",
        "{ENV_REGEX_NS}/Robot/right_elbow_pitch_link",
        "{ENV_REGEX_NS}/Robot/left_elbow_roll_link",
        "{ENV_REGEX_NS}/Robot/right_elbow_roll_link",
        "{ENV_REGEX_NS}/Robot/left_palm_link",
        "{ENV_REGEX_NS}/Robot/right_palm_link",
    ],
    history_length=3,
    track_air_time=False,
    force_threshold=0.5
)
```

## 📍 **Where to Add This**

### **File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/velocity_env_cfg.py`

### **Location**: In the scene sensors section around line 85, after the existing sensors:

```python
@configclass
class VelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
    # ... existing code ...
    
    # scene
    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=4.0)
    
    # ... existing sensors ...
    contact_forces = ContactSensorCfg(...)  # EXISTING - for feet
    torso_contact = ContactSensorCfg(...)   # EXISTING - for torso only
    
    # ADD THIS NEW SENSOR:
    collision_sensor = ContactSensorCfg(    # NEW - for comprehensive collision detection
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        force_threshold=0.5
    )
    
    imu = ImuCfg(...)  # EXISTING
```

## 🎯 **Expected Result After Adding**

Once the `collision_sensor` is added to the configuration:

1. **✅ Full G1 body coverage**: All 23+ collision-prone bodies monitored
2. **✅ Proper obstacle detection**: Terrain elevation and obstacle collisions detected
3. **✅ No more "1 body coverage" warnings**: Comprehensive sensor data available
4. **✅ Meaningful collision metrics**: Accurate terrain navigation performance measurement

## ⚠️ **Alternative Workaround**

If you can't modify the main config file, you could:

1. **Create a custom environment config** that extends the existing one
2. **Override the sensors section** to add the collision sensor
3. **Use that custom config** for your terrain collision analysis

But the cleanest solution is to add the `collision_sensor` directly to the main SDS velocity environment configuration. 