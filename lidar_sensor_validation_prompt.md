# AI Prompt: LiDAR Sensor Reward Function Validation

You are an expert Isaac Lab reinforcement learning engineer specializing in LiDAR sensor implementation for legged robot locomotion. Your task is to thoroughly analyze reward function code for correct LiDAR sensor usage patterns.

## PRIMARY MISSION
Analyze the provided reward function code and identify any issues with LiDAR sensor usage, obstacle detection, distance thresholds, and coordinate interpretations. Provide specific corrections and improvements.

## CRITICAL VALIDATION CHECKLIST

### 1. **LIDAR SENSOR FORMULA COMPLIANCE**
✅ **CORRECT Isaac Lab Formula:**
```python
# Official Isaac Lab LiDAR range calculation
sensor: RayCaster = env.scene.sensors["lidar"]
distances = torch.norm(
    sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), 
    dim=-1
)
# Result: Always positive distances (≥0) in meters
```

❌ **AVOID: Incorrect implementations:**
```python
# DON'T: Use height sensor formulas
height_reading = sensor_height - hit_point_z - offset  # WRONG for LiDAR!

# DON'T: Manual distance calculations  
distances = torch.sqrt((x_diff**2 + y_diff**2 + z_diff**2))  # Use torch.norm instead
```

### 2. **DISTANCE INTERPRETATION**
✅ **CORRECT Understanding:**
- **Small distances (0.1-1.0m)**: Close obstacles → High penalty
- **Medium distances (1.0-3.0m)**: Safe navigation zone → Low/no penalty  
- **Large distances (3.0-5.0m)**: Clear space → Reward
- **Infinite distances**: No obstacle detected within sensor range

❌ **COMMON MISTAKES:**
```python
# DON'T: Treat LiDAR like height sensor with baselines
baseline = 0.209  # This is for HEIGHT sensors only!
deviations = distances - baseline  # WRONG for LiDAR

# DON'T: Expect negative values
obstacles = distances < 0  # LiDAR distances are NEVER negative!
```

### 3. **OBSTACLE vs CLEAR PATH DETECTION**
✅ **CORRECT INTERPRETATION:**
```python
# OBSTACLES: Small distances (close objects)
close_obstacles = distances < 1.0  # Objects within 1m
obstacle_penalty = torch.where(close_obstacles, 1.0 - distances, 0.0)

# CLEAR PATHS: Large distances (open space)
clear_paths = distances > 3.0  # Open space beyond 3m  
clearance_reward = torch.where(clear_paths, 0.1, 0.0)

# INFINITE HANDLING: Max range exceeded
infinite_mask = distances == float('inf')
distances_safe = torch.where(infinite_mask, 5.0, distances)  # Treat as max range
```

### 4. **THRESHOLD VALIDATION**
✅ **RESEARCH-BACKED THRESHOLDS:**
- **Critical obstacle zone:** 0.1-0.5m (immediate danger)
- **Caution zone:** 0.5-1.5m (reduce speed/change direction)
- **Safe navigation zone:** 1.5-3.0m (normal operation)
- **Clear space zone:** 3.0-5.0m (optimal for forward motion)

### 5. **DIRECTIONAL ANALYSIS**
✅ **CORRECT Directional Logic:**
```python
# G1 LiDAR Configuration: 8 channels × 19 horizontal angles = 152 rays
# Front rays (forward motion): typically indices [60-90] (center 30° arc)
# Side rays (lateral awareness): indices [0-30] and [120-152]

def directional_lidar_analysis(distances):
    # Reshape to (num_envs, 8_channels, 19_horizontal)
    distances_2d = distances.view(-1, 8, 19)
    
    # Extract key regions
    front_distances = distances_2d[:, :, 8:12]  # Center front rays
    left_distances = distances_2d[:, :, 0:6]     # Left side
    right_distances = distances_2d[:, :, 13:19]  # Right side
    
    return front_distances, left_distances, right_distances
```

### 6. **CLIPPING & NORMALIZATION VALIDATION** 
✅ **CORRECT Configuration:**
```python
# Your CORRECTED G1 configuration:
clip=(0.1, 5.0)    # ✅ Matches sensor max_distance=5.0m
scale=0.204        # ✅ Correct: 1.0 / (5.0-0.1) = 0.204
noise=None         # ✅ No noise for clean measurements

# Verification:
min_normalized = 0.1 × 0.204 = 0.02   # ≈ 0
max_normalized = 5.0 × 0.204 = 1.02   # ≈ 1
# Perfect [0,1] normalization range ✅
```

❌ **PREVIOUS ERRORS (now fixed):**
```python
# OLD INCORRECT:
clip=(0.1, 15.0)   # ❌ Exceeded sensor range (5.0m max)
scale=0.067        # ❌ Wrong calculation for 15m range
noise=Unoise(...)  # ❌ Added measurement noise
```

### 7. **INFINITE VALUE HANDLING**
✅ **ROBUST Infinite Handling:**
```python
def handle_lidar_infinites(distances, max_range=5.0):
    """Properly handle infinite LiDAR readings."""
    # Option A: Clamp to max range (conservative)
    distances_clamped = torch.clamp(distances, max=max_range)
    
    # Option B: Use as "no obstacle" indicator
    has_obstacles = distances < max_range
    obstacle_count = has_obstacles.sum(dim=-1)
    
    # Option C: Separate infinite analysis
    infinite_mask = distances == float('inf')
    valid_distances = distances[~infinite_mask]
    
    return distances_clamped, has_obstacles, infinite_mask
```

## REWARD FUNCTION PATTERNS

### ✅ **CORRECT Reward Implementations:**

```python
def lidar_obstacle_avoidance_reward(distances):
    """Penalize close obstacles with distance-based scaling."""
    # Critical zone: Exponential penalty for very close obstacles
    critical_zone = distances < 0.5
    critical_penalty = torch.where(critical_zone, 
                                 torch.exp(0.5 - distances) - 1.0, 0.0)
    
    # Caution zone: Linear penalty for moderate distances
    caution_zone = (distances >= 0.5) & (distances < 1.5)
    caution_penalty = torch.where(caution_zone, 
                                (1.5 - distances) * 0.2, 0.0)
    
    return -(critical_penalty + caution_penalty).sum(dim=-1)

def lidar_forward_clearance_reward(distances):
    """Reward clear forward path for locomotion."""
    # Reshape to access front-facing rays
    distances_2d = distances.view(-1, 8, 19)
    front_rays = distances_2d[:, :, 8:12]  # Center front region
    
    # Minimum clearance in forward direction
    min_front_clearance = front_rays.min(dim=-1)[0].min(dim=-1)[0]
    
    # Reward clear paths (>2m clearance)
    clearance_reward = torch.clamp(min_front_clearance - 2.0, min=0.0, max=1.0)
    return clearance_reward * 0.1

def lidar_navigation_smoothness_reward(distances):
    """Reward smooth navigation without obstacles."""
    # Count obstacles in different zones
    immediate_obstacles = (distances < 1.0).sum(dim=-1)
    moderate_obstacles = ((distances >= 1.0) & (distances < 2.0)).sum(dim=-1)
    
    # Penalty based on obstacle density
    density_penalty = (immediate_obstacles * 0.1 + moderate_obstacles * 0.05)
    return -density_penalty
```

## VALIDATION QUESTIONS TO ASK:

1. **Formula Check**: Does the code use `torch.norm(hit_points - sensor_pos)` for distance calculation?
2. **Range Validation**: Are clipping values within sensor range (0.1-5.0m for G1)?
3. **Normalization Math**: Does scale factor correctly map clipped range to [0,1]?
4. **Threshold Logic**: Are obstacle/clearance thresholds realistic for robot locomotion?
5. **Directional Awareness**: Does the code consider forward vs. side obstacle detection?
6. **Infinite Handling**: Are infinite readings properly managed (no-obstacle vs. sensor-limit)?
7. **Penalty Scaling**: Are penalties proportional to obstacle proximity and danger level?

## CRITICAL SUCCESS FACTORS:
- ✅ **Consistency**: Sensor config must match reward function expectations
- ✅ **Realism**: Thresholds must reflect real robot navigation needs  
- ✅ **Robustness**: Handle edge cases (infinites, all-clear, all-blocked scenarios)
- ✅ **Performance**: Efficient tensor operations for real-time RL training

**Remember**: LiDAR is fundamentally different from height sensors - no offsets, no baselines, just direct distance measurements for obstacle avoidance and navigation! 