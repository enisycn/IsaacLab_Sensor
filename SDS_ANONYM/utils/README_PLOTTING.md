# SDS Training Metrics Plotting System

## Overview

This system automatically generates and saves comprehensive training metric plots after every successful SDS training run. Plots are saved in the Isaac Lab checkpoint directory under a `plots/` subfolder.

## 🎯 Features

### Automatic Plot Generation
- **Triggered**: After every successful training completion
- **Location**: `{Isaac Lab checkpoint dir}/plots/`
- **Format**: High-quality PNG files (150 DPI)
- **Integration**: Seamlessly integrated into SDS training loop

### Comprehensive Metrics Coverage

#### 1. Core Training Metrics (Individual Plots)
- `reward_progression.png` - Mean episode reward over training
- `episode length_progression.png` - Episode duration trends
- `termination_base_contact_progression.png` - Contact failure rates

#### 2. Environmental Sensing & Robot Stability (Individual Plots)
From `metrics.py`:
- `terrain_height_variance_progression.png` - Terrain roughness measurement
- `robot_height_baseline_progression.png` - Robot height above terrain
- `body_orientation_deviation_progression.png` - Body tilt from upright (degrees)
- `height_tracking_error_progression.png` - Height control accuracy
- `terrain_complexity_score_progression.png` - Obstacle/gap analysis

#### 3. Training Performance (Combined Plot)
- `training_performance.png` - Value function loss, surrogate loss, entropy loss, action noise

#### 4. Task Performance (Combined Plot)
- `task_performance.png` - Velocity errors, curriculum progression

#### 5. Termination Analysis (Combined Plot)
- `termination_analysis.png` - Timeout vs contact failure rates

#### 6. System Performance (Combined Plot)
- `system_performance.png` - Computation speed, collection/learning times

## 📁 Directory Structure

### Primary SDS Checkpoint Structure
```
/home/enis/IsaacLab/SDS_ANONYM/outputs/sds/
├── 2025-08-06_14-30-15/              # SDS training checkpoint directory
│   ├── sds.log                       # SDS training log
│   ├── .hydra/                       # Hydra configuration
│   ├── plots/                        # ✅ NEW: Auto-generated plots
│   │   ├── reward_progression.png
│   │   ├── episode length_progression.png
│   │   ├── termination_base_contact_progression.png
│   │   ├── terrain_height_variance_progression.png
│   │   ├── robot_height_baseline_progression.png
│   │   ├── body_orientation_deviation_progression.png
│   │   ├── height_tracking_error_progression.png
│   │   ├── terrain_complexity_score_progression.png
│   │   ├── training_performance.png
│   │   ├── task_performance.png
│   │   ├── termination_analysis.png
│   │   └── system_performance.png
│   └── ...
```

### Isaac Lab G1 Enhanced Checkpoint Structure (Alternative)
```
/home/enis/IsaacLab/logs/rsl_rl/g1_enhanced/
├── 2025-08-06_14-30-15/              # Isaac Lab training checkpoint directory
│   ├── model_*.pt                    # Model checkpoints
│   ├── events.out.tfevents.*         # TensorBoard logs
│   ├── plots/                        # ✅ NEW: Auto-generated plots (same as above)
│   └── ...
```

## 🔧 Implementation Details

### Files Modified/Created

1. **`utils/plotting.py`** (NEW)
   - Main plotting module with all functionality
   - Professional plot styling and statistics
   - Error handling and logging

2. **`SDS/sds.py`** (MODIFIED)
   - Added import: `from utils.plotting import create_sds_training_plots`
   - Added plotting call after successful training (line ~888)

3. **`utils/test_plot_integration.py`** (NEW)
   - Test script for plotting functionality
   - Demonstrates integration with real checkpoints

### Integration Points

The plotting is triggered in `sds.py` after successful training completion:

```python
# In sds.py, after successful training (around line 888)
try:
    logging.info("🎨 Generating training plots...")
    plot_success = create_sds_training_plots(run_log)
    if plot_success:
        logging.info("✅ Training plots created successfully!")
    else:
        logging.warning("⚠️ Plot creation failed, but training analysis continues")
except Exception as e:
    logging.error(f"❌ Error creating plots: {e}")
    logging.warning("⚠️ Plot creation failed, but training analysis continues")
```

## 📊 Plot Features

### Individual Metric Plots
- **Line plots** with markers showing progression over training iterations
- **Statistics box** showing Final, Mean±Std, and Range values
- **Professional styling** with grids, proper axes labels, and titles
- **Color coding** for different metric types

### Combined Metric Plots
- **Multi-subplot layout** for related metrics
- **Shared x-axis** (training iterations) for easy comparison
- **Final value annotations** on each subplot
- **Automatic subplot sizing** based on number of metrics

### Plot Quality
- **High DPI** (150) for crisp images
- **Large size** (12x8 inches) for detailed viewing
- **Matplotlib non-interactive backend** for headless server compatibility
- **Proper error handling** to not interrupt training on plot failures

## 🚀 Usage

### Automatic Usage (Recommended)
Plots are **automatically generated** after every successful SDS training. No manual intervention required.

### Manual Testing
```bash
# Test plotting with sample data
cd /home/enis/IsaacLab/SDS_ANONYM/utils
python3 plotting.py

# Test integration with real checkpoints
python3 test_plot_integration.py
```

### Manual Plot Generation
```python
from utils.plotting import create_sds_training_plots
from utils.misc import construct_run_log

# After training, with run_log from construct_run_log(stdout_str)
success = create_sds_training_plots(run_log)
```

## 📈 Metrics Included

### From Isaac Lab Training Logs
- Core metrics: `reward`, `episode length`, `iterations/`
- Training performance: `value_function_loss`, `surrogate_loss`, `entropy_loss`, `action_noise_std`
- Reward components: `reward_sds_custom` and other `Episode_Reward/*`
- Task performance: `velocity_error_xy`, `velocity_error_yaw`
- Termination analysis: `termination_timeout`, `termination_base_contact`
- System performance: `computation_steps_per_sec`, `collection_time`, `learning_time`
- Curriculum: `curriculum_terrain_levels`

### From Enhanced Metrics System (`metrics.py`)
- `terrain_height_variance` - Terrain roughness under robot
- `robot_height_baseline` - Robot height above terrain baseline
- `body_orientation_deviation` - Roll/pitch deviation from upright (degrees)
- `height_tracking_error` - Height control accuracy (meters)
- `terrain_complexity_score` - Weighted terrain complexity (obstacles/gaps)

## 🎯 Benefits

### For Training Analysis
- **Visual progression tracking** of all key metrics
- **Quick identification** of training issues or improvements
- **Statistical summaries** for each metric
- **Professional presentation** suitable for reports/papers

### For GPT Feedback Enhancement
- **Comprehensive data** about training performance
- **Environmental sensing validation** through terrain metrics
- **Robot stability assessment** through orientation and height metrics
- **Clear visualization** of training trends and patterns

### For Development
- **Automatic organization** of training results
- **Easy comparison** between different training runs
- **Persistent storage** with training checkpoints
- **No manual intervention** required

## 🔧 Error Handling

The plotting system is designed to **never interrupt training**:
- All plotting code is wrapped in try-catch blocks
- Plot failures log warnings but allow training to continue
- Missing metrics are handled gracefully (plots skipped)
- Invalid data is detected and logged appropriately

## 📝 Log Output Example

```
🎨 Generating training plots...
📁 Plots saved to: /home/enis/IsaacLab/logs/rsl_rl/g1_flat/2025-08-06_14-30-15/plots
📊 Plot creation complete: 12/12 plots created successfully
✅ Training plots created successfully!
```

This system provides **comprehensive visual analysis** of SDS training performance with **zero manual overhead**! 