# Height Sensor Guide for Isaac Lab RL Rewards

> **⚠️ IMPORTANT**: These are technical explanations. for reward generation you shoul come up correct reward terms suitable for analyzed environment.

## 📐 **Isaac Lab Formula**
```python
# Official height scan observation
height_reading = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
# Default offset = 0.5m (NOT 0.05!)
```

## 🎯 **Core Rules**
- **OBSTACLES** = Lower readings (< baseline - threshold) = Terrain HIGHER than expected
- **GAPS** = Higher readings (> baseline + threshold) = Terrain LOWER than expected  
- **BASELINE** = G1 robot on flat terrain = 0.209m (sensor_height - terrain_z - offset)
- **THRESHOLDS** = ±0.07m (7cm) for balanced detection

## ⚡ **Correct Implementation**
```python
def terrain_classification(height_readings, baseline=0.209):
    obstacle_threshold = 0.07  # 7cm
    gap_threshold = 0.07       # 7cm
    
    obstacles = height_readings < (baseline - obstacle_threshold)    # < 0.139m
    gaps = height_readings > (baseline + gap_threshold)              # > 0.279m  
    normal = ~obstacles & ~gaps                                      # 0.139-0.279m
    extreme_gaps = height_readings == float('inf')                   # No terrain detected
    
    return obstacles, gaps, normal, extreme_gaps
```

## 📊 **Optimized Thresholds**
- **Standard**: 0.07m (7cm) - balanced for most robots
- **Sensitive**: 0.05m (5cm) - careful navigation
- **Relaxed**: 0.10m (10cm) - rough terrain
- **Range**: 0.05-0.15m acceptable, 0.07m optimal

## 🔬 **Height Scanner Specifications**

### **G1 Robot Configuration (Enhanced)**
```python
# Enhanced sensor configuration from flat_with_box_env_cfg.py
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.6)),  # 60cm above torso
    attach_yaw_only=True,  # Only yaw rotation (not pitch/roll)
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.075,  # 7.5cm spacing between rays
        size=[2.0, 1.5],   # 2m forward × 1.5m lateral coverage
    ),
    max_distance=3.0,      # 3m maximum ray distance
    update_period=0.02,    # 50Hz update rate
    mesh_prim_paths=["/World/ground"],
)
```

### **Ray Pattern Analysis**
```python
# Enhanced configuration calculations:
forward_coverage = 2.0m  # Total forward scan distance
lateral_coverage = 1.5m  # Total lateral scan distance  
ray_resolution = 0.075m  # 7.5cm between rays
rays_forward = int(2.0 / 0.075) + 1 = 27 rays  # Forward direction
rays_lateral = int(1.5 / 0.075) + 1 = 21 rays  # Lateral direction
total_rays = 27 × 21 = 567 rays  # Total ray count

# Standard configuration calculations:
standard_forward = 1.6m
standard_lateral = 1.0m
standard_resolution = 0.1m
standard_rays_forward = int(1.6 / 0.1) + 1 = 17 rays
standard_rays_lateral = int(1.0 / 0.1) + 1 = 11 rays
standard_total = 17 × 11 = 187 rays
```

## 🎯 **Relative Height Tracking Rewards**

### **1. ONLY Relative Terrain Navigation**
```python
def relative_terrain_reward(env, sensor_cfg=SceneEntityCfg("height_scanner")):
    """Pure relative height tracking - NO absolute positioning."""
    
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # Isaac Lab formula (RELATIVE measurements only)
    height_readings = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5
    baseline = 0.209  # G1 robot baseline
    
    # Classification using ONLY sensor readings
    obstacles = height_readings < (baseline - 0.07)
    gaps = height_readings > (baseline + 0.07)
    normal_terrain = ~obstacles & ~gaps & (height_readings != float('inf'))
    
    # Count features
    total_rays = height_readings.shape[-1]
    obstacle_count = obstacles.sum(dim=-1)
    gap_count = gaps.sum(dim=-1)
    normal_count = normal_terrain.sum(dim=-1)
    
    # Reward ONLY based on terrain sensing (not absolute height)
    terrain_safety = (normal_count / total_rays) * 0.5
    obstacle_penalty = -(obstacle_count / total_rays) * 2.0
    gap_penalty = -(gap_count / total_rays) * 1.5
    
    return terrain_safety + obstacle_penalty + gap_penalty
```

### **2. Look-Ahead Terrain Preview**
```python
def lookahead_terrain_reward(env, sensor_cfg=SceneEntityCfg("height_scanner")):
    """Forward-looking terrain analysis for proactive navigation."""
    
    sensor = env.scene.sensors[sensor_cfg.name]
    height_readings = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5
    baseline = 0.209
    
    # Reshape to grid pattern for spatial analysis
    # For 567 rays (27×21 grid): rays_forward=27, rays_lateral=21
    rays_forward, rays_lateral = 27, 21
    height_grid = height_readings.view(-1, rays_forward, rays_lateral)
    
    # Split into zones: near (0-0.7m), mid (0.7-1.3m), far (1.3-2.0m)
    near_zone = height_grid[:, :9, :]    # First 9 rays = 0-0.675m
    mid_zone = height_grid[:, 9:18, :]   # Next 9 rays = 0.675-1.35m  
    far_zone = height_grid[:, 18:, :]    # Last 9 rays = 1.35-2.0m
    
    # Analyze each zone for upcoming terrain
    def analyze_zone(zone_data, zone_weight):
        obstacles = (zone_data < (baseline - 0.07)).float().mean(dim=(-1, -2))
        gaps = (zone_data > (baseline + 0.07)).float().mean(dim=(-1, -2))
        return -(obstacles * 2.0 + gaps * 1.5) * zone_weight
    
    # Weight zones: near=highest, far=planning
    near_reward = analyze_zone(near_zone, 1.0)    # Immediate danger
    mid_reward = analyze_zone(mid_zone, 0.5)      # Tactical planning
    far_reward = analyze_zone(far_zone, 0.2)     # Strategic planning
    
    return near_reward + mid_reward + far_reward
```

### **3. Adaptive Baseline Calculation**
```python
def adaptive_baseline_terrain_reward(env, sensor_cfg=SceneEntityCfg("height_scanner")):
    """Dynamic baseline adaptation for varying terrain conditions."""
    
    sensor = env.scene.sensors[sensor_cfg.name]
    height_readings = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5
    
    # Filter out infinite readings for baseline calculation
    finite_mask = height_readings != float('inf')
    finite_readings = height_readings[finite_mask]
    
    if finite_readings.numel() > 0:
        # Dynamic baseline: use median of current readings
        # More robust than mean for terrain with obstacles/gaps
        dynamic_baseline = torch.median(finite_readings.view(-1, -1), dim=-1)[0]
        
        # Adaptive thresholds based on terrain variation
        terrain_std = torch.std(finite_readings.view(-1, -1), dim=-1)
        adaptive_threshold = torch.clamp(terrain_std * 2.0, 0.05, 0.15)  # 2σ rule
        
        # Classification using adaptive parameters
        obstacles = height_readings < (dynamic_baseline.unsqueeze(-1) - adaptive_threshold.unsqueeze(-1))
        gaps = height_readings > (dynamic_baseline.unsqueeze(-1) + adaptive_threshold.unsqueeze(-1))
        
        # Reward based on terrain complexity
        total_rays = height_readings.shape[-1]
        obstacle_ratio = obstacles.sum(dim=-1).float() / total_rays
        gap_ratio = gaps.sum(dim=-1).float() / total_rays
        
        # Penalty scales with terrain difficulty
        complexity_factor = torch.clamp(terrain_std, 0.5, 2.0)
        obstacle_penalty = -obstacle_ratio * 2.0 * complexity_factor
        gap_penalty = -gap_ratio * 1.5 * complexity_factor
        
        return obstacle_penalty + gap_penalty
    else:
        # Fallback for all-infinite readings
        return torch.zeros(env.num_envs, device=env.device)
```

## 🎯 **Complete Reward Example**
```python
def comprehensive_terrain_reward(env, sensor_cfg=SceneEntityCfg("height_scanner")):
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]
    
    # Isaac Lab formula
    height_readings = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5
    baseline = 0.209  # G1 robot baseline
    
    # Classification
    obstacles = height_readings < (baseline - 0.07)
    gaps = height_readings > (baseline + 0.07)
    normal_terrain = ~obstacles & ~gaps & (height_readings != float('inf'))
    infinite_gaps = height_readings == float('inf')
    
    # Count features
    total_rays = height_readings.shape[-1]
    obstacle_count = obstacles.sum(dim=-1)
    gap_count = gaps.sum(dim=-1)
    normal_count = normal_terrain.sum(dim=-1)
    infinite_count = infinite_gaps.sum(dim=-1)
    
    # Percentage-based rewards
    obstacle_penalty = -(obstacle_count / total_rays) * 2.0
    gap_penalty = -(gap_count / total_rays) * 1.5
    stability_reward = (normal_count / total_rays) * 0.5
    cliff_penalty = -(infinite_count / total_rays) * 10.0
    
    return obstacle_penalty + gap_penalty + stability_reward + cliff_penalty
```

### 3. Dynamic Gap Crossing Reward
```python
def dynamic_gap_crossing(env, sensor_cfg=SceneEntityCfg("height_scanner")):
    """Reward function that encourages crossing appropriate gaps."""
    
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]
    
    height_readings = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5
    baseline = 0.209
    
    # Gap classification
    small_gaps = (height_readings > (baseline + 0.05)) & (height_readings < (baseline + 0.15))
    medium_gaps = (height_readings > (baseline + 0.15)) & (height_readings < (baseline + 0.25))
    large_gaps = height_readings > (baseline + 0.25)
    extreme_gaps = height_readings == float('inf')
    
    # Robot capabilities (based on leg length and body size)
    robot_velocity = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    can_jump = robot_velocity > 0.3  # Moving fast enough to jump
    
    # Dynamic gap rewards based on robot state
    small_gap_reward = torch.where(
        can_jump & torch.any(small_gaps, dim=-1),
        0.2,  # Small reward for crossing small gaps when able
        torch.where(torch.any(small_gaps, dim=-1), -0.5, 0.0)  # Penalty if not able
    )
    
    medium_gap_penalty = torch.sum(medium_gaps, dim=-1) * -1.0
    large_gap_penalty = torch.sum(large_gaps, dim=-1) * -3.0
    extreme_gap_penalty = torch.sum(extreme_gaps, dim=-1) * -10.0
    
    return small_gap_reward + medium_gap_penalty + large_gap_penalty + extreme_gap_penalty
```

---

## ✅ **Validation Checklist**

### 1. **FORMULA COMPLIANCE**
✅ **CORRECT**: `sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5`
❌ **REJECT**: Direct use of `ray_hits_w[..., 2]` without sensor position/offset

### 2. **BASELINE UNDERSTANDING** 
✅ **FLAT TERRAIN BASELINE:** ~0.209m
- Calculation: sensor_height(0.709) - terrain_z(0.000) - offset(0.5) = 0.209m
- All thresholds should be RELATIVE to this baseline, not absolute

❌ **REJECT these approaches:**
- Using 0.209m as absolute threshold
- Hardcoded terrain Z values
- Ignoring sensor mounting height variations

### 3. **OBSTACLE vs GAP INTERPRETATION**
✅ **CORRECT INTERPRETATION:**
```python
# OBSTACLES: Negative height readings (terrain higher than expected)
obstacles = height_readings < (baseline - 0.07)  # < 0.139m for 0.209 baseline
obstacle_penalty = torch.where(obstacles, -penalty_value, 0.0)

# GAPS: Positive height readings (terrain lower than expected)  
gaps = height_readings > (baseline + 0.07)  # > 0.279m for 0.209 baseline
gap_penalty = torch.where(gaps, -penalty_value, 0.0)
```

❌ **REJECT these patterns:**
- Treating positive values as obstacles
- Using absolute thresholds without baseline consideration
- Confusing height readings with terrain coordinates

### 4. **THRESHOLD VALIDATION**
✅ **OPTIMIZED THRESHOLDS:**
- **Standard obstacles:** 0.07m above baseline (baseline - 0.07)
- **Standard gaps:** 0.07m below baseline (baseline + 0.07)
- **Acceptable range:** 0.05-0.15m for balanced sensitivity
- **Research validation:** 5-25cm proven successful in academic studies

❌ **REJECT these ranges:**
- Thresholds > 0.30m (too large for most robots)
- Thresholds < 0.03m (too sensitive to noise)
- Same threshold for obstacles and gaps (should be different)

### 5. **INFINITE READING HANDLING**
✅ **PROPER INFINITE HANDLING:**
```python
# Handle max range exceeded
valid_readings = height_readings[height_readings != float('inf')]
infinite_penalty = torch.sum(height_readings == float('inf')) * extreme_gap_penalty
```

❌ **REJECT these approaches:**
- Ignoring infinite readings completely
- Treating infinite as zero
- Not penalizing extreme gaps (cliffs)

### 6. **SENSOR CONFIGURATION VALIDATION**
✅ **CORRECT SENSOR ACCESS:**
```python
sensor: RayCaster = env.scene.sensors["height_scanner"]
sensor_cfg = SceneEntityCfg("height_scanner")
```

❌ **REJECT these patterns:**
- Hardcoded sensor names not matching environment
- Missing sensor existence checks
- Wrong sensor type assumptions

### 7. **CLIPPING & NORMALIZATION VALIDATION**

#### **Observation Clipping**
✅ **CHECK CLIPPING COMPATIBILITY:**
```python

clip=(-0.5, 3.0)      # Custom extended range

### 4. **THRESHOLD VALUES**
✅ **CORRECT**: 0.05-0.15m range, 0.07m optimal
❌ **REJECT**: >0.30m (too large), <0.03m (too sensitive)

### 5. **INFINITE HANDLING**
✅ **CORRECT**: `cliff_penalty = torch.sum(height_readings == float('inf')) * penalty`
❌ **REJECT**: Ignoring infinite readings

### 6. **CLIPPING COMPATIBILITY**
✅ **CHECK**: Thresholds work with clip ranges:
- `clip=(-1.0, 1.0)` or `clip=(-0.5, 3.0)`
- 0.07m thresholds → 0.139m, 0.279m ✅ Within range

### 7. **RELATIVE TRACKING VALIDATION**
✅ **CORRECT RELATIVE TRACKING:**
- Use ONLY height sensor readings for terrain navigation
- NO absolute robot height tracking in rewards
- Dynamic baseline adaptation for varying terrain
- Look-ahead zones for proactive navigation

❌ **REJECT ABSOLUTE TRACKING:**
- Direct use of `robot.data.root_pos_w[:, 2]` in terrain rewards
- Fixed absolute height targets
- Mixing absolute positioning with relative terrain sensing

## ❌ **Common Mistakes**
```python
# ❌ WRONG - Missing sensor position/offset
height = sensor.data.ray_hits_w[..., 2] - 0.5

# ❌ WRONG - Backwards logic
obstacles = height_readings > 0.2  # Positive is gaps!

# ❌ WRONG - Absolute thresholds
obstacles = height_readings < 0.1  # Ignores baseline

# ❌ WRONG - Absolute height tracking in terrain rewards
target_height = 0.7  # Fixed absolute height
height_reward = -torch.square(robot.data.root_pos_w[:, 2] - target_height)

# ✅ CORRECT - Isaac Lab formula + relative thresholds
height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5
obstacles = height_readings < (baseline - 0.07)
gaps = height_readings > (baseline + 0.07)
# ✅ CORRECT - Only relative terrain navigation
terrain_reward = analyze_terrain_features(height_readings, baseline)
```

## 🔧 **Quick Troubleshooting**
- **All negative readings**: Check sensor height, adjust baseline
- **All classified as obstacles**: Baseline too high, use dynamic calculation
- **Robot avoiding terrain**: Thresholds too sensitive, use 0.07m
- **Infinite crashes**: Always filter with `height_readings != float('inf')`
- **Poor look-ahead**: Check ray pattern grid dimensions (27×21 for enhanced config)
- **Baseline drift**: Use median instead of mean for adaptive baseline

## 📊 **Visual Scale**
```
0.139m ← OBSTACLE THRESHOLD (baseline - 0.07)
0.209m ← BASELINE (G1 flat terrain) 
0.279m ← GAP THRESHOLD (baseline + 0.07)

Examples:
0.120m = 7cm obstacle (terrain higher)
0.209m = flat terrain (normal)
0.300m = 9cm gap (terrain lower)
inf    = extreme gap (cliff/void)
```

## 🎯 **Key Points**
- **Default offset = 0.5m** (half meter, NOT 5cm!)
- **Lower readings = obstacles** (terrain closer to sensor)
- **Higher readings = gaps** (terrain farther from sensor)
- **0.209m baseline** for G1 robot specifically
- **±0.07m thresholds** for balanced detection
- **Always handle infinite** readings as dangerous cliffs
- **Use relative thresholds**, never absolute values
- **567 rays total** (27×21 grid) for enhanced configuration
- **3m sensor range** with 7.5cm resolution for detailed scanning
- **Look-ahead zones** enable proactive navigation planning
- **NO absolute height** tracking in terrain-based rewards

**The key insight: smaller height readings mean obstacles, larger height readings mean gaps!** 🎯 