# Terrain/Obstacle Collision Detection - Final Approach

## 🎯 **Objective**
Detect when robot body parts (excluding feet) collide with terrain elevations or obstacles, indicating navigation difficulties or poor terrain adaptation.

## 🚫 **What We DON'T Count**
- **Foot-ground contact**: Normal locomotion behavior
- **Expected terrain contact**: Feet touching the ground is normal

## ✅ **What We DO Count**
- **Body-terrain collisions**: Torso, thighs, knees hitting elevated terrain
- **Arm-obstacle contact**: Arms hitting obstacles while navigating
- **Hip/pelvis contact**: Robot sitting on or hitting obstacles
- **Unexpected body contact**: Any non-foot body part touching terrain/obstacles

## 🔧 **Implementation Strategy**

### **1. Sensor Priority (NO Foot Sensors)**
```python
# CORRECT: Focus on body contact sensors
sensor_priority = [
    "collision_sensor",     # Comprehensive body sensor
    "torso_contact",        # Torso-specific sensor
    "contact_sensor",       # General body sensor
    "body_contact"          # Alternative body sensor
]

# AVOID: Foot sensors for obstacle detection
avoid_sensors = [
    "contact_forces",       # Usually foot sensors
    "foot_contact",         # Explicit foot sensors
    "ankle_contact"         # Ankle/foot sensors
]
```

### **2. G1 Robot Body Parts Monitored**
```python
collision_prone_bodies = [
    # LEGS (non-foot parts that indicate obstacles)
    "left_hip_pitch_link",      # Hip hitting obstacles
    "right_hip_pitch_link",     # Hip hitting obstacles  
    "left_knee_link",           # Knee hitting obstacles
    "right_knee_link",          # Knee hitting obstacles
    
    # TORSO (body hitting terrain elevations)
    "pelvis",                   # Main body contact
    "torso_link",               # Torso hitting obstacles
    "pelvis_contour_link",      # Pelvis contour contact
    
    # ARMS (arms hitting obstacles during navigation)
    "left_shoulder_pitch_link", # Shoulder collisions
    "right_shoulder_pitch_link",# Shoulder collisions
    "left_elbow_pitch_link",    # Elbow collisions
    "right_elbow_pitch_link",   # Elbow collisions
    "left_palm_link",           # Hand collisions
    "right_palm_link",          # Hand collisions
]
```

### **3. Collision Threshold**
- **Force threshold**: 0.5N minimum contact force
- **Rationale**: Light contact may be normal, but >0.5N indicates meaningful collision

### **4. 2*N Scaling**
- **Actual collisions**: Count of body parts exceeding threshold
- **Reported metric**: 2 × actual_collisions (per user requirement)
- **Interpretation**: Higher values = more problematic terrain navigation

## 📊 **Expected Behavior**

### **Terrain Type 2 (Obstacles)**
- **Good navigation**: Low collision counts (body avoids obstacles)
- **Poor navigation**: High collision counts (body hits obstacles frequently)
- **Typical range**: 0-20 collisions per step depending on terrain difficulty

### **Real-World Interpretation**
- **0 collisions**: Robot successfully navigates without body contact
- **2-10 collisions**: Occasional body contact (acceptable)
- **>20 collisions**: Frequent body hitting obstacles (poor navigation)

## 🚀 **Benefits of This Approach**
1. **Terrain-specific**: Focuses on non-foot contact relevant for obstacle terrain
2. **Robot-specific**: Uses actual G1 humanoid body part names
3. **Meaningful metric**: High values indicate real navigation problems
4. **Isaac Lab compliant**: Uses proper sensor access patterns

## ⚠️ **Important Notes**
- This metric is **terrain-specific to Type 2 (Obstacles)**
- Foot contact is **explicitly excluded** as it's normal locomotion
- Only activated when contact sensors provide body-level force data
- Requires proper contact sensor configuration in environment setup 