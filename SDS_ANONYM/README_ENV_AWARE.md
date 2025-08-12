# SDS Environment-Aware vs Foundation-Only Comparison System

## Overview

The SDS pipeline now supports a **dual-mode comparison system** that enables direct comparison between:

- **Environment-Aware Mode**: Full environmental sensing + terrain analysis + sensor-based rewards
- **Foundation-Only Mode**: Gait-focused rewards with NO environmental information

This allows for controlled A/B testing to understand the impact of environmental awareness on robot locomotion performance.

## Configuration

### Using Hydra Configuration

Edit `SDS/cfg/config.yaml`:

```yaml
# Environment-Aware Mode (DEFAULT)
env_aware: true   # Full environmental sensing + terrain analysis

# Foundation-Only Mode  
env_aware: false  # Pure gait-based rewards, NO environmental info
```

### Using Environment Variables (Overrides Config)

```bash
# Environment-Aware Mode
export SDS_ENV_AWARE=1
python SDS/sds.py task=walk

# Foundation-Only Mode  
export SDS_ENV_AWARE=0
python SDS/sds.py task=walk
```

## Mode Differences

### Environment-Aware Mode (`env_aware: true`)

**Features Enabled:**
- ✅ Automatic environment image capture
- ✅ Real-time environmental analysis with terrain sensing  
- ✅ Enhanced SUS generation with sensor data integration
- ✅ Sensor-based reward components (gap detection, obstacle avoidance, etc.)
- ✅ Full observation space including height scanner and LiDAR

**Reward Components:**
- Velocity tracking, gait quality, posture (foundation)
- **PLUS** terrain adaptation, sensor utilization, environmental navigation

**Expected Behavior:**
- Robot learns terrain-specific strategies
- Proactive navigation using sensor feedback
- Higher performance on complex terrains

### Foundation-Only Mode (`env_aware: false`)

**Features Disabled:**
- 🚫 NO environment image capture
- 🚫 NO environmental analysis or terrain sensing
- 🚫 NO sensor-based reward components
- 🚫 NO terrain-specific adaptations

**Features Enabled:**
- ✅ Video-based gait analysis only
- ✅ Foundation locomotion rewards only
- ✅ Same observation space (sensors available but not rewarded)

**Reward Components (Foundation Only):**
- `track_lin_vel_xy_yaw_frame_exp` (velocity tracking)
- `track_ang_vel_z_world_exp` (angular velocity control)
- `feet_air_time_positive_biped` (gait quality)
- `base_height_l2` (postural stability)
- `orientation_l2` (balance and uprightness)
- `joint_deviation_l1` (natural joint positions)
- `action_smoothness_l2` (smooth control)
- `feet_slide` (contact quality)

**Expected Behavior:**
- Robot learns terrain-agnostic locomotion
- Reactive strategies based on contact feedback
- May develop conservative movement patterns

## Validation & Safety

### Foundation-Only Code Validation

When `env_aware: false`, the system automatically:

1. **Blocks Environmental Sensor Usage**: Detects and prevents usage of:
   - `env.scene.sensors`
   - `height_scanner`, `lidar`, `RayCaster`
   - `terrain_following`, `obstacle_avoidance`, `gap_crossing`

2. **Auto-Regeneration**: If violations detected, automatically requests new code

3. **Logging**: Provides detailed validation feedback

## Usage Examples

### Run Environment-Aware Training

```bash
cd /home/enis/IsaacLab/SDS_ANONYM
python SDS/sds.py task=walk env_aware=true
```

### Run Foundation-Only Training  

```bash
cd /home/enis/IsaacLab/SDS_ANONYM
python SDS/sds.py task=walk env_aware=false
```

### Override with Environment Variable

```bash
# Force foundation-only mode regardless of config
SDS_ENV_AWARE=0 python SDS/sds.py task=walk

# Force environment-aware mode regardless of config  
SDS_ENV_AWARE=1 python SDS/sds.py task=walk
```

## Run Metadata & Analysis

Each training run automatically saves metadata in `run_meta.json`:

```json
{
  "env_aware": true/false,
  "mode": "environment_aware" or "foundation_only",
  "sensors_available": true/false,
  "environment_analysis": true/false,
  "timestamp": "2025-01-XX_XX-XX-XX",
  "terrain_type": "flat/rough",
  "model": "gpt-5",
  "iterations": 5,
  "samples_per_iteration": 8,
  "train_iterations": 1000
}
```

This enables automatic grouping and comparison of results.

## Performance Comparison

### Expected Performance Patterns

**Simple Terrain (Flat)**:
- Foundation-only: Good baseline performance
- Environment-aware: Similar performance (limited terrain complexity)

**Complex Terrain (Gaps, Obstacles, Stairs)**:
- Foundation-only: Conservative strategies, potential avoidance behaviors
- Environment-aware: Adaptive navigation, higher success rates

### Key Metrics to Compare

1. **Success Rate**: Episode completion percentage
2. **Average Velocity**: Forward locomotion speed
3. **Terrain Coverage**: What % of environment is utilized
4. **Energy Efficiency**: Power consumption per distance
5. **Failure Modes**: How robots fail (falling vs avoidance)

## Terrain-Specific Testing

Configure terrain type in your environment config:

```python
# In flat_with_box_env_cfg.py
TERRAIN_TYPE = 0  # Simple flat terrain
TERRAIN_TYPE = 1  # Gaps terrain  
TERRAIN_TYPE = 2  # Obstacles terrain
TERRAIN_TYPE = 3  # Stairs terrain
```

Test both modes on each terrain type for comprehensive analysis.

## Troubleshooting

### Foundation-Only Validation Errors

If you see foundation-only validation failures:

```
🚫 Foundation-only validation FAILED for sample X:
• Forbidden sensor usage: height_scanner
```

**Solution**: The system will auto-regenerate. If it persists, check your prompt files for environmental references.

### No Environment Analysis Data

If environment-aware mode shows missing analysis:

```
⚠️ No environment analysis available, proceeding with video-only analysis
```

**Solution**: Check that the analyze_environment.py script runs successfully and your checkpoint path is correct.

## Research Applications

This system enables controlled studies to answer:

1. **"How much does environmental awareness improve locomotion?"**
2. **"What behaviors emerge with vs without sensor feedback?"**
3. **"Which terrains benefit most from environmental sensing?"**
4. **"What's the trade-off between safety and performance?"**

Perfect for academic papers comparing sensor-based vs reactive locomotion strategies! 