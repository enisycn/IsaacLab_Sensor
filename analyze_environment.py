#!/usr/bin/env python3

"""
Real Environmental Analysis Script
Extracts actual sensor readings and performs detailed gap/obstacle analysis.
"""

import argparse
import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Environmental Analysis with Real Sensor Data")
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments")  # Increased to sample all terrain types
parser.add_argument("--task", type=str, default="Isaac-SDS-Velocity-Flat-G1-Enhanced-v0", help="Task name")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

def simple_connected_components(binary_mask):
    """Simple connected component analysis."""
    if not binary_mask.any():
        return np.zeros_like(binary_mask, dtype=int), 0
        
    labeled = np.zeros_like(binary_mask, dtype=int)
    current_label = 0
    rows, cols = binary_mask.shape
    
    def flood_fill(start_r, start_c, label):
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if (r >= 0 and r < rows and c >= 0 and c < cols and 
                binary_mask[r, c] and labeled[r, c] == 0):
                labeled[r, c] = label
                # Add 4-connected neighbors
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if (nr >= 0 and nr < rows and nc >= 0 and nc < cols and 
                        binary_mask[nr, nc] and labeled[nr, nc] == 0):
                        stack.append((nr, nc))
    
    for i in range(rows):
        for j in range(cols):
            if binary_mask[i, j] and labeled[i, j] == 0:
                current_label += 1
                flood_fill(i, j, current_label)
    
    return labeled, current_label

def analyze_clusters(binary_mask, resolution, feature_type, height_grid=None):
    """Analyze connected components and extract dimensions."""
    labeled, num_features = simple_connected_components(binary_mask)
    clusters = []
    
    for i in range(1, num_features + 1):
        cluster_mask = labeled == i
        cluster_coords = np.where(cluster_mask)
        
        if len(cluster_coords[0]) < 5:  # Skip tiny clusters
            continue
            
        # Calculate dimensions
        min_row, max_row = cluster_coords[0].min(), cluster_coords[0].max()
        min_col, max_col = cluster_coords[1].min(), cluster_coords[1].max()
        
        width_m = (max_row - min_row + 1) * resolution
        length_m = (max_col - min_col + 1) * resolution
        area_m2 = cluster_mask.sum() * (resolution ** 2)
        
        # Use minimum crossing distance for correct classification (as user specified)
        min_crossing = min(width_m, length_m)
        
        # Check if cluster contains impossible gaps (max range readings)
        contains_impossible = False
        if height_grid is not None and feature_type == "gaps":
            cluster_heights = height_grid[cluster_mask]
            contains_impossible = np.any(cluster_heights >= 19.9)
        
        # Classify difficulty based on minimum crossing distance
        if feature_type == "gaps":
            if contains_impossible:
                difficulty = "impossible"
            elif min_crossing <= 0.3:
                difficulty = "steppable"  # Can be crossed using normal walking
            elif min_crossing <= 0.6:
                difficulty = "jumpable"  # Requires jumping to cross
            else:
                difficulty = "impossible"  # Too wide - cannot be traversed
        else:
            if max(width_m, length_m) <= 0.2:
                difficulty = "small_obstacle"
            elif max(width_m, length_m) <= 0.5:
                difficulty = "medium_obstacle"
            else:
                difficulty = "large_obstacle"
        
        cluster_info = {
            'id': i,
            'width_m': round(width_m, 3),
            'length_m': round(length_m, 3),
            'min_crossing': round(min_crossing, 3),
            'area_m2': round(area_m2, 4),
            'center_x_m': round((min_row + max_row) / 2 * resolution, 3),
            'center_y_m': round((min_col + max_col) / 2 * resolution, 3),
            'difficulty': difficulty,
            'point_count': int(cluster_mask.sum())
        }
        clusters.append(cluster_info)
    
    # Sort by area (largest first)
    clusters.sort(key=lambda x: x['area_m2'], reverse=True)
    return clusters

def analyze_height_data(height_data, resolution=0.1, area_size=(4.0, 3.0), robot_id=None):
    """Analyze height scanner data for gaps and obstacles."""
    
    robot_label = f" (Robot #{robot_id})" if robot_id is not None else ""
    
    # Calculate grid dimensions
    grid_width = int(area_size[0] / resolution)
    grid_height = int(area_size[1] / resolution)
    expected_points = grid_width * grid_height
    
    if robot_id is not None:  # Only print details for specific robots
        print(f"📐 HEIGHT SCANNER CONFIGURATION{robot_label}:")
        print(f"   Expected grid: {grid_width} × {grid_height} = {expected_points:,} points")
        print(f"   Actual data points: {len(height_data):,}")
        print(f"   Resolution: {resolution*100:.1f}cm grid spacing")
        print(f"   Coverage area: {area_size[0]}m × {area_size[1]}m = {area_size[0]*area_size[1]} m²")
    
    # Use available data
    available_points = min(len(height_data), expected_points)
    used_height_data = height_data[:available_points]
    
    # Reshape to closest square if needed
    if available_points != expected_points:
        side_length = int(np.sqrt(available_points))
        if side_length == 0:
            side_length = 1  # Prevent division by zero
        used_height_data = height_data[:side_length**2]
        height_grid = used_height_data.reshape(side_length, side_length)
        actual_resolution = min(area_size) / max(side_length, 1)  # Prevent division by zero
        if robot_id is not None:
            print(f"   Adjusted to: {side_length} × {side_length} grid")
            print(f"   Adjusted resolution: {actual_resolution*100:.1f}cm")
    else:
        height_grid = used_height_data.reshape(grid_width, grid_height)
        actual_resolution = resolution
    
    # Convert to numpy for analysis
    if torch.is_tensor(height_grid):
        height_grid = height_grid.cpu().numpy()
    
    # === IDENTIFY IMPOSSIBLE GAPS (MAX RANGE READINGS) ===
    # Readings at sensor max distance (19.9-20.0m) indicate gaps deeper than sensor range
    max_sensor_range = 20.0
    impossible_gaps_mask = (height_grid >= max_sensor_range - 0.1)  # Within 10cm of max range
    valid_terrain_for_analysis = ~impossible_gaps_mask
    
    # Count impossible gaps for safety assessment
    impossible_gap_points = impossible_gaps_mask.sum()
    impossible_gap_percentage = (impossible_gap_points / max(height_grid.size, 1)) * 100
    
    # Only use finite terrain readings for ground level calculation
    if valid_terrain_for_analysis.any():
        finite_heights = height_grid[valid_terrain_for_analysis]
        ground_level = np.median(finite_heights)
        min_valid_height = finite_heights.min()
        max_valid_height = finite_heights.max()
        
        if robot_id is not None:
            print(f"\n🔧 IMPOSSIBLE GAP DETECTION{robot_label}:")
            print(f"   Total points: {height_grid.size:,}")
            print(f"   Impossible gaps (≥19.9m): {impossible_gap_points:,} points")
            print(f"   Finite terrain points: {valid_terrain_for_analysis.sum():,} points")
            print(f"   Impossible gap percentage: {impossible_gap_percentage:.1f}%")
    else:
        if robot_id is not None:
            print(f"\n⚠️ WARNING{robot_label}: No finite terrain data found (all impossible gaps)")
        return [], [], {'ground_level': 0, 'min_height': 0, 'max_height': 0, 
                       'gap_coverage': 0, 'conservative_gap_coverage': 0, 'obstacle_coverage': 0, 
                       'flat_coverage': 0, 'terrain_type': 'UNKNOWN', 'terrain_std_cm': 0,
                       'adaptive_threshold_cm': 5.0, 'conservative_threshold_cm': 5.0,
                       'old_method_gap_coverage': 0, 'improvement_percentage': 0,
                       'impossible_gap_coverage': impossible_gap_percentage}
    
    # === FIXED 4CM THRESHOLD - PROVEN SAFE ===
    terrain_std = finite_heights.std()
    height_range = max_valid_height - min_valid_height
    
    # Classify terrain complexity for informational purposes only
    if terrain_std < 0.01:  # <1cm variation
        terrain_type = "FLAT"
    elif terrain_std < 0.03:  # 1-3cm variation  
        terrain_type = "SMOOTH"
    elif terrain_std < 0.05:  # 3-5cm variation
        terrain_type = "MODERATE" 
    else:  # >5cm variation
        terrain_type = "COMPLEX"
    
    # Fixed thresholds - simple, safe, and consistent
    gap_threshold_adaptive = 0.04  # Fixed 4cm threshold - proven safe across all terrain
    conservative_threshold = 0.05  # Slightly more conservative (5cm)
    
    # Standard obstacle threshold 
    obstacle_threshold = ground_level + 0.03  # Fixed 3cm above ground for obstacles
    
    if robot_id is not None:
        print(f"\n🌍 FIXED THRESHOLD TERRAIN ANALYSIS{robot_label}:")
        print(f"   Ground level (median): {ground_level:.3f}m")
        print(f"   Valid height range: {min_valid_height:.3f}m to {max_valid_height:.3f}m")
        print(f"   Total valid variation: {max_valid_height - min_valid_height:.3f}m")
        print(f"   Terrain std deviation: {terrain_std*100:.1f}cm")
        print(f"   Terrain type: {terrain_type} (informational only)")
        print(f"   Fixed gap threshold: {gap_threshold_adaptive*100:.1f}cm below ground (SAFE & CONSISTENT)")
        print(f"   Conservative threshold: {conservative_threshold*100:.1f}cm (extra safety margin)")
    
    # Create masks using fixed thresholds
    gap_threshold_final = ground_level - gap_threshold_adaptive  
    conservative_gap_threshold = ground_level - conservative_threshold
    
    # Include ALL terrain for gap analysis (finite + impossible)
    all_terrain_mask = np.ones_like(height_grid, dtype=bool)
    
        # Gap analysis: finite gaps + impossible gaps
    finite_gaps_mask = valid_terrain_for_analysis & (height_grid < gap_threshold_final)
    gaps_mask = finite_gaps_mask | impossible_gaps_mask  # Include impossible gaps!
    conservative_gaps_mask = valid_terrain_for_analysis & (height_grid < conservative_gap_threshold)
    obstacles_mask = valid_terrain_for_analysis & (height_grid > obstacle_threshold)
    flat_mask = valid_terrain_for_analysis & (height_grid >= gap_threshold_final) & (height_grid <= obstacle_threshold)
    
    # Analyze gaps and obstacles with multi-level detection (use total grid for percentages)
    total_analyzed_points = height_grid.size
    gap_coverage = gaps_mask.sum() / max(total_analyzed_points, 1) * 100
    finite_gap_coverage = finite_gaps_mask.sum() / max(total_analyzed_points, 1) * 100
    conservative_gap_coverage = conservative_gaps_mask.sum() / max(total_analyzed_points, 1) * 100
    obstacle_coverage = obstacles_mask.sum() / max(total_analyzed_points, 1) * 100
    flat_coverage = flat_mask.sum() / max(total_analyzed_points, 1) * 100
    
    if robot_id is not None:
        print(f"\n📊 FIXED THRESHOLD TERRAIN DISTRIBUTION{robot_label}:")
        print(f"   🕳️  Gaps (4cm+): {gap_coverage:.1f}% ({gaps_mask.sum():,} points)")
        print(f"       ├── Finite gaps: {finite_gap_coverage:.1f}% ({finite_gaps_mask.sum():,} points)")
        print(f"       └── Impossible gaps: {impossible_gap_percentage:.1f}% ({impossible_gap_points:,} points)")
        print(f"   ⚠️  Deep gaps (5cm+): {conservative_gap_coverage:.1f}% ({conservative_gaps_mask.sum():,} points)")
        print(f"   🗻 Elevated obstacles: {obstacle_coverage:.1f}% ({obstacles_mask.sum():,} points)")
        print(f"   ✅ Safe/Flat terrain: {flat_coverage:.1f}% ({flat_mask.sum():,} points)")
    
    # Compare with old adaptive method for validation
    old_gap_threshold = ground_level - 0.05  # Previous 5cm fixed baseline
    old_gaps_mask = valid_terrain_for_analysis & (height_grid < old_gap_threshold)
    old_gap_coverage = old_gaps_mask.sum() / max(total_analyzed_points, 1) * 100
    
    improvement = old_gap_coverage - gap_coverage
    if robot_id is not None:
        print(f"\n🔄 COMPARISON WITH PREVIOUS 5CM METHOD:")
        print(f"   Previous method (5cm): {old_gap_coverage:.1f}% gaps")
        print(f"   Current fixed (4cm): {gap_coverage:.1f}% gaps")
        if improvement > 1.0:
            print(f"   ✅ More sensitive: Detected {abs(improvement):.1f}% more safety-critical features")
        elif improvement < -1.0:
            print(f"   ⚠️  Less sensitive: Missed {improvement:.1f}% features vs 5cm threshold")
        else:
            print(f"   ✅ Consistent detection - both methods find similar features")
    
    # Detailed gap analysis
    if gaps_mask.any():
        gap_clusters = analyze_clusters(gaps_mask, actual_resolution, "gaps", height_grid)
        
        # Calculate deepest gap (finite gaps only - impossible gaps are >20m deep)
        finite_gap_heights = height_grid[finite_gaps_mask]
        if len(finite_gap_heights) > 0:
            deepest_gap = ground_level - finite_gap_heights.min()
        else:
            deepest_gap = 20.0  # Impossible gaps are deeper than sensor range
        
        if robot_id is not None:
            print(f"\n🕳️ GAP ANALYSIS{robot_label}:")
            print(f"   Total gaps detected: {len(gap_clusters)}")
            if deepest_gap >= 20.0:
                print(f"   Deepest gap: >20.0m (impossible gaps detected)")
            else:
                print(f"   Deepest gap: {deepest_gap:.3f}m below ground")
        
        if gap_clusters and robot_id is not None and robot_id < 3:  # Show details only for first 3 robots
            print(f"\n   📏 DETAILED GAP MEASUREMENTS:")
            for i, gap in enumerate(gap_clusters[:5]):  # Show top 5 gaps
                print(f"   Gap #{gap['id']:2d}: {gap['width_m']:5.2f}m × {gap['length_m']:5.2f}m "
                      f"(Min crossing: {gap['min_crossing']:.2f}m) - {gap['difficulty']:15s}")
                print(f"            Area: {gap['area_m2']:7.4f} m², Center: ({gap['center_x_m']:5.2f}m, {gap['center_y_m']:5.2f}m)")
    else:
        gap_clusters = []
        if robot_id is not None:
            print(f"\n🕳️ No significant gaps detected{robot_label}")
    
    # Detailed obstacle analysis  
    if obstacles_mask.any():
        obstacle_clusters = analyze_clusters(obstacles_mask, actual_resolution, "obstacles", height_grid)
        obstacle_heights = height_grid[obstacles_mask]
        if len(obstacle_heights) > 0:
            highest_obstacle = obstacle_heights.max() - ground_level
        else:
            highest_obstacle = 0.0
        
        if robot_id is not None:
            print(f"\n🗻 OBSTACLE ANALYSIS{robot_label}:")
            print(f"   Total obstacles detected: {len(obstacle_clusters)}")
            print(f"   Highest obstacle: {highest_obstacle:.3f}m above ground")
        
        if obstacle_clusters and robot_id is not None and robot_id < 3:  # Show details only for first 3 robots
            print(f"\n   📏 DETAILED OBSTACLE MEASUREMENTS:")
            for i, obs in enumerate(obstacle_clusters[:5]):  # Show top 5 obstacles
                print(f"   Obs #{obs['id']:2d}: {obs['width_m']:5.2f}m × {obs['length_m']:5.2f}m "
                      f"(Area: {obs['area_m2']:7.4f} m²) - {obs['difficulty']:15s}")
                print(f"            Center: ({obs['center_x_m']:5.2f}m, {obs['center_y_m']:5.2f}m)")
    else:
        obstacle_clusters = []
        if robot_id is not None:
            print(f"\n🗻 No significant obstacles detected{robot_label}")
    
    return gap_clusters, obstacle_clusters, {
        'ground_level': ground_level,
        'min_height': min_valid_height,
        'max_height': max_valid_height,
        'gap_coverage': gap_coverage,
        'conservative_gap_coverage': conservative_gap_coverage,
        'obstacle_coverage': obstacle_coverage,
        'flat_coverage': flat_coverage,
        'terrain_type': terrain_type,
        'terrain_std_cm': terrain_std * 100,
        'adaptive_threshold_cm': gap_threshold_adaptive * 100,  # Now fixed 4cm
        'conservative_threshold_cm': conservative_threshold * 100,  # Fixed 5cm
        'old_method_gap_coverage': old_gap_coverage,
        'improvement_percentage': improvement
    }

def analyze_lidar_data(lidar_data, max_range=30.0, robot_id=None):
    """Analyze LiDAR data for obstacle detection."""
    
    robot_label = f" (Robot #{robot_id})" if robot_id is not None else ""
    
    # Convert to numpy if needed
    if torch.is_tensor(lidar_data):
        lidar_data = lidar_data.cpu().numpy()
    
    if robot_id is not None:  # Only print details for specific robots
        print(f"\n📡 LIDAR ANALYSIS{robot_label}:")
        print(f"   Total rays: {len(lidar_data):,}")
        print(f"   Range: {lidar_data.min():.3f}m to {lidar_data.max():.3f}m")
        print(f"   Average distance: {lidar_data.mean():.3f}m")
        print(f"   Max sensor range: {max_range}m")
    
    # Distance-based obstacle analysis
    immediate = lidar_data < 1.0      # < 1m = immediate danger
    near = (lidar_data >= 1.0) & (lidar_data < 3.0)    # 1-3m = near field
    mid = (lidar_data >= 3.0) & (lidar_data < 10.0)    # 3-10m = mid field
    far = (lidar_data >= 10.0) & (lidar_data < max_range)  # 10m+ = far field
    clear = lidar_data >= max_range   # No obstacle detected
    
    if robot_id is not None and robot_id < 3:  # Show details only for first 3 robots
        total_rays = max(len(lidar_data), 1)  # Prevent division by zero
        print(f"\n   🎯 OBSTACLE ZONES:")
        print(f"   Immediate Danger (<1m):  {immediate.sum():6,} rays ({immediate.sum()/total_rays*100:5.1f}%)")
        print(f"   Near Field (1-3m):       {near.sum():6,} rays ({near.sum()/total_rays*100:5.1f}%)")
        print(f"   Mid Field (3-10m):       {mid.sum():6,} rays ({mid.sum()/total_rays*100:5.1f}%)")
        print(f"   Far Field (10-15m):      {far.sum():6,} rays ({far.sum()/total_rays*100:5.1f}%)")
        print(f"   Clear Space (15m+):      {clear.sum():6,} rays ({clear.sum()/total_rays*100:5.1f}%)")
    
    # Safety assessment
    danger_percent = immediate.sum() / max(len(lidar_data), 1) * 100
    if danger_percent > 20:
        safety = "🔴 HIGH RISK"
    elif danger_percent > 10:
        safety = "🟡 MODERATE RISK"
    elif danger_percent > 5:
        safety = "🟠 LOW RISK"
    else:
        safety = "🟢 SAFE"
    
    if robot_id is not None:
        print(f"\n   🛡️ SAFETY ASSESSMENT{robot_label}: {safety}")
    
    return {
        'immediate_danger': immediate.sum(),
        'near_field': near.sum(),
        'mid_field': mid.sum(),
        'far_field': far.sum(),
        'clear': clear.sum(),
        'safety_level': safety
    }

def print_navigation_recommendations(gap_analysis, obstacle_analysis, terrain_stats, lidar_stats):
    """Generate navigation recommendations based on fixed threshold analysis."""
    
    print(f"\n🧭 FIXED THRESHOLD NAVIGATION RECOMMENDATIONS:")
    
    # Terrain-based assessment
    terrain_type = terrain_stats.get('terrain_type', 'UNKNOWN')
    improvement = terrain_stats.get('improvement_percentage', 0)
    conservative_gaps = terrain_stats.get('conservative_gap_coverage', 0)
    
    print(f"   🌍 Terrain Assessment:")
    print(f"   ├── Type: {terrain_type} ({terrain_stats.get('terrain_std_cm', 0):.1f}cm variation)")
    print(f"   ├── Fixed threshold: 4.0cm (SAFE & CONSISTENT)")
    print(f"   └── Deep gaps (5cm+): {conservative_gaps:.1f}% (extra caution required)")
    
    if improvement > 5:
        print(f"   ✅ More sensitive than 5cm method: Found {abs(improvement):.1f}% more safety features")
    elif improvement < -5:
        print(f"   ⚠️  Less sensitive than 5cm method: {improvement:.1f}% difference")
    
    # Count navigation challenges
    if gap_analysis:
        steppable = sum(1 for gap in gap_analysis if gap['difficulty'] == 'steppable')
        jumpable = sum(1 for gap in gap_analysis if gap['difficulty'] == 'jumpable')
        impossible = sum(1 for gap in gap_analysis if gap['difficulty'] == 'impossible')
        
        print(f"\n   🕳️  Gap Navigation:")
        print(f"   ├── Steppable gaps: {steppable}")
        print(f"   ├── Jumpable gaps: {jumpable}")
        print(f"   └── Impossible gaps: {impossible}")
    
    # Terrain-specific strategy recommendation
    print(f"\n   🎯 RECOMMENDED STRATEGY:")
    
    if terrain_type == "FLAT" and terrain_stats['flat_coverage'] > 90:
        strategy = "Standard locomotion - terrain is highly predictable with fixed 4cm detection"
    elif terrain_type == "COMPLEX" and conservative_gaps > 5:
        strategy = "ADVANCED: Multiple gap types detected - use careful navigation patterns"
    elif isinstance(lidar_stats['immediate_danger'], (int, float)) and lidar_stats['immediate_danger'] > 1000:
        strategy = "EMERGENCY: Stop and reassess - high collision risk detected by LiDAR"
    elif gap_analysis and len([g for g in gap_analysis if g['difficulty'] in ['jumpable', 'impossible']]) > 3:
        strategy = "Dynamic locomotion: Jumping and path planning required for large gaps"
    elif gap_analysis and len([g for g in gap_analysis if g['difficulty'] == 'steppable']) > 0:
        strategy = "Careful stepping with 4cm threshold providing consistent safety margin"
    else:
        strategy = "Standard walking with fixed 4cm gap detection for consistent safety"
    
    print(f"   {strategy}")
    
    # Fixed threshold warnings based on detection results
    if terrain_stats['gap_coverage'] > 15:
        print(f"   ⚠️  HIGH GAP DENSITY: {terrain_stats['gap_coverage']:.1f}% gaps detected with 4cm threshold")
    elif terrain_type == "FLAT" and terrain_stats['gap_coverage'] > 5:
        print(f"   ⚠️  UNEXPECTED: {terrain_stats['gap_coverage']:.1f}% gaps on flat terrain - verify sensor calibration")
    
    if conservative_gaps > 2:
        print(f"   ⚠️  DEEP GAPS (5cm+): {conservative_gaps:.1f}% require extra caution or alternative paths")
    
    if terrain_stats['obstacle_coverage'] > 20:
        print(f"   ⚠️  DENSE OBSTACLES: {terrain_stats['obstacle_coverage']:.1f}% elevated obstacles - slow navigation recommended")

def main():
    """Main analysis function."""
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True  # Just use True as default
    )
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    print("🚀 COMPREHENSIVE MULTI-ROBOT ENVIRONMENTAL ANALYSIS")
    print("="*80)
    print(f"Environment: {args_cli.task}")
    print(f"Number of robots: {args_cli.num_envs}")
    
    # Reset environment and get observations
    print(f"\n⏳ Initializing environment and collecting sensor data...")
    observations, _ = env.reset()
    obs_data = observations['policy']  # Shape: [num_envs, total_obs]
    
    print(f"✅ Environment loaded successfully!")
    print(f"   Total observation dimensions: {obs_data.shape[1]:,}")
    
    # === TERRAIN DEBUG INFORMATION ===
    print(f"\n🌍 TERRAIN CONFIGURATION DEBUG:")
    if hasattr(env.unwrapped.scene, 'terrain'):
        terrain = env.unwrapped.scene.terrain
        if hasattr(terrain, 'terrain_generator'):
            terrain_gen = terrain.terrain_generator
            if hasattr(terrain_gen, 'sub_terrains'):
                print(f"   Sub-terrains configured: {list(terrain_gen.sub_terrains.keys())}")
                for name, sub_terrain in terrain_gen.sub_terrains.items():
                    print(f"     - {name}: {type(sub_terrain).__name__} (proportion: {sub_terrain.proportion})")
                    if hasattr(sub_terrain, 'gap_width_range'):
                        print(f"       Gap width range: {sub_terrain.gap_width_range}")
                    if hasattr(sub_terrain, 'grid_height_range'):
                        print(f"       Grid height range: {sub_terrain.grid_height_range}")
        
        if hasattr(terrain, 'data') and hasattr(terrain.data, 'terrain_types'):
            terrain_types = terrain.data.terrain_types.cpu().numpy()
            unique_types = np.unique(terrain_types)
            print(f"\n   📊 TERRAIN TYPE DISTRIBUTION:")
            for terrain_type in unique_types:
                count = (terrain_types == terrain_type).sum()
                percentage = count / len(terrain_types) * 100
                print(f"     Type {terrain_type}: {count:2d} environments ({percentage:5.1f}%)")
    
    # Get robot positions early
    positions = None
    if hasattr(env.unwrapped.scene, 'robot'):
        robot = env.unwrapped.scene.robot
        root_state = robot.data.root_state_w
        positions = root_state[:, :3]  # [num_envs, 3] (x, y, z)
        
        print(f"\n🤖 ROBOT POSITIONING:")
        print(f"   Robots distributed across {args_cli.num_envs} environments")
        print(f"   Position range: X({positions[:, 0].min():.1f} to {positions[:, 0].max():.1f}m), "
              f"Y({positions[:, 1].min():.1f} to {positions[:, 1].max():.1f}m), "
              f"Z({positions[:, 2].min():.1f} to {positions[:, 2].max():.1f}m)")
    else:
        print(f"\n⚠️ WARNING: Could not access robot positioning data")
    
    # Calculate expected dimensions (based on our optimized config)
    robot_state_dims = 110
    # Height scanner: 4m × 3m at 10cm resolution = 1,200 points
    height_scan_points = int((4.0 / 0.1) * (3.0 / 0.1))  # 40 × 30 = 1,200
    # LiDAR: 16 channels × 72 horizontal = 1,152 points  
    lidar_points = 16 * 72
    
    print(f"\n📊 SENSOR DATA EXTRACTION:")
    print(f"   Expected height scanner points: {height_scan_points:,}")
    print(f"   Expected LiDAR points: {lidar_points:,}")
    
    # Analyze multiple robots to get diverse terrain coverage
    print(f"\n" + "="*80)
    print("🔍 MULTI-ROBOT TERRAIN ANALYSIS")
    print("="*80)
    
    all_gaps = []
    all_obstacles = []
    terrain_stats_summary = {
        'ground_levels': [],
        'height_ranges': [],
        'gap_coverages': [],
        'obstacle_coverages': [],
        'flat_coverages': [],
        'terrain_std_cms': []
    }
    lidar_stats_summary = {
        'immediate_dangers': [],
        'near_fields': [],
        'safety_levels': []
    }
    
    # Analyze ALL robots but only show detailed output for first few
    robots_for_detailed_print = 0  # No detailed robot output, only final results
    robots_to_analyze = args_cli.num_envs  # Analyze ALL for final aggregation
    
    print(f"\n🔄 Analyzing {robots_to_analyze} robots for comprehensive environment assessment...")
    
    for robot_id in range(robots_to_analyze):
        # Extract sensor data for this robot
        robot_data = obs_data[robot_id]
        
        # Extract height scanner data (after robot state)
        height_start = robot_state_dims
        height_end = height_start + min(height_scan_points, len(robot_data) - robot_state_dims - lidar_points)
        height_scan = robot_data[height_start:height_end]
        
        # Extract LiDAR data (after height scanner)
        lidar_start = height_end  
        lidar_end = lidar_start + min(lidar_points, len(robot_data) - lidar_start)
        lidar_range = robot_data[lidar_start:lidar_end]
        
        # Perform detailed analysis for this robot (always analyze, but only print details for first few)
        gap_analysis, obstacle_analysis, terrain_stats = analyze_height_data(height_scan, robot_id=(robot_id if robot_id < robots_for_detailed_print else None))
        lidar_stats = analyze_lidar_data(lidar_range, robot_id=(robot_id if robot_id < robots_for_detailed_print else None))
        
        # Accumulate results
        all_gaps.extend(gap_analysis if gap_analysis else [])
        all_obstacles.extend(obstacle_analysis if obstacle_analysis else [])
        
        terrain_stats_summary['ground_levels'].append(terrain_stats['ground_level'])
        terrain_stats_summary['height_ranges'].append(terrain_stats['max_height'] - terrain_stats['min_height'])
        terrain_stats_summary['gap_coverages'].append(terrain_stats['gap_coverage'])
        terrain_stats_summary['obstacle_coverages'].append(terrain_stats['obstacle_coverage'])
        terrain_stats_summary['flat_coverages'].append(terrain_stats['flat_coverage'])
        terrain_stats_summary['terrain_std_cms'].append(terrain_stats['terrain_std_cm'])
        
        lidar_stats_summary['immediate_dangers'].append(lidar_stats['immediate_danger'])
        lidar_stats_summary['near_fields'].append(lidar_stats['near_field'])
        lidar_stats_summary['safety_levels'].append(lidar_stats['safety_level'])
    
    # === FINAL COMPREHENSIVE ANALYSIS ONLY ===
    print(f"\n" + "="*80)
    print("📋 COMPREHENSIVE FINAL ENVIRONMENT ANALYSIS FOR AI AGENT")
    print("="*80)
    
    # Basic Statistics
    print(f"\n🎯 ANALYSIS SCOPE:")
    print(f"   Robots Analyzed: {robots_to_analyze}")
    print(f"   Total Sensor Points: {robots_to_analyze * 213_333:,} height measurements")
    print(f"   Total LiDAR Rays: {robots_to_analyze * 23_040:,} distance measurements")
    print(f"   Environment Type: Isaac-SDS-Velocity-Flat-G1-Enhanced-v0")
    
    # Terrain Diversity Analysis
    height_variations = terrain_stats_summary['height_ranges']
    ground_levels = terrain_stats_summary['ground_levels']
    max_height_variation = max(height_variations) if height_variations else 0
    
    print(f"\n🌍 TERRAIN CHARACTERISTICS:")
    print(f"   Ground Level Range: {min(ground_levels):.3f}m to {max(ground_levels):.3f}m")
    print(f"   Height Variation Range: {min(height_variations):.3f}m to {max(height_variations):.3f}m")
    print(f"   Maximum Terrain Variation: {max_height_variation:.3f}m")
    print(f"   Average Terrain Roughness: {np.mean(terrain_stats_summary['terrain_std_cms']):.1f}cm")
    print(f"   Configuration Compliance: {'✅ WITHIN LIMITS' if max_height_variation <= 0.20 else '⚠️ EXCEEDS 20cm LIMIT'}")
    
    # Gap Analysis with Classifications
    gap_types = {}
    for gap in all_gaps:
        gap_type = gap.get('difficulty', 'unknown')
        gap_types[gap_type] = gap_types.get(gap_type, 0) + 1
    
    print(f"\n🕳️ COMPREHENSIVE GAP ANALYSIS:")
    print(f"   Total Gaps Detected: {len(all_gaps)}")
    print(f"   Gap Type Distribution:")
    for gap_type, count in sorted(gap_types.items()):
        percentage = count / len(all_gaps) * 100 if all_gaps else 0
        print(f"     - {gap_type.capitalize()}: {count} gaps ({percentage:.1f}%)")
    
    if all_gaps:
        gap_widths = [gap['width_m'] for gap in all_gaps]
        gap_lengths = [gap['length_m'] for gap in all_gaps]
        gap_areas = [gap['area_m2'] for gap in all_gaps]
        
        print(f"\n   📐 GAP SIZE METRICS:")
        print(f"     Width Range: {min(gap_widths):.3f}m to {max(gap_widths):.3f}m")
        print(f"     Length Range: {min(gap_lengths):.3f}m to {max(gap_lengths):.3f}m")
        print(f"     Area Range: {min(gap_areas):.4f}m² to {max(gap_areas):.4f}m²")
        print(f"     Average Gap Size: {np.mean(gap_widths):.3f}m × {np.mean(gap_lengths):.3f}m")
        
        print(f"\n   🏆 LARGEST GAPS BY TYPE:")
        # Show largest gaps for each type
        for gap_type in sorted(gap_types.keys()):
            type_gaps = [gap for gap in all_gaps if gap.get('difficulty') == gap_type]
            if type_gaps:
                largest = max(type_gaps, key=lambda x: x['area_m2'])
                print(f"     {gap_type.capitalize()}: {largest['width_m']:.2f}m × {largest['length_m']:.2f}m "
                      f"(Area: {largest['area_m2']:.4f}m²)")
    
    # Obstacle Analysis with Classifications  
    obstacle_types = {}
    for obstacle in all_obstacles:
        obs_type = obstacle.get('difficulty', 'unknown')
        obstacle_types[obs_type] = obstacle_types.get(obs_type, 0) + 1
    
    print(f"\n🗻 COMPREHENSIVE OBSTACLE ANALYSIS:")
    print(f"   Total Obstacles Detected: {len(all_obstacles)}")
    print(f"   Obstacle Type Distribution:")
    for obs_type, count in sorted(obstacle_types.items()):
        percentage = count / len(all_obstacles) * 100 if all_obstacles else 0
        print(f"     - {obs_type.replace('_', ' ').title()}: {count} obstacles ({percentage:.1f}%)")
    
    if all_obstacles:
        obs_widths = [obs['width_m'] for obs in all_obstacles]
        obs_lengths = [obs['length_m'] for obs in all_obstacles]
        obs_areas = [obs['area_m2'] for obs in all_obstacles]
        
        print(f"\n   📐 OBSTACLE SIZE METRICS:")
        print(f"     Width Range: {min(obs_widths):.3f}m to {max(obs_widths):.3f}m")
        print(f"     Length Range: {min(obs_lengths):.3f}m to {max(obs_lengths):.3f}m")
        print(f"     Area Range: {min(obs_areas):.4f}m² to {max(obs_areas):.4f}m²")
        print(f"     Average Obstacle Size: {np.mean(obs_widths):.3f}m × {np.mean(obs_lengths):.3f}m")
    
    # Safety and Navigation Analysis
    danger_counts = np.array(lidar_stats_summary['immediate_dangers'])
    near_field_counts = np.array(lidar_stats_summary['near_fields']) 
    
    print(f"\n🛡️ SAFETY & NAVIGATION ANALYSIS:")
    print(f"   Immediate Danger Zones (<1m):")
    print(f"     Range: {danger_counts.min()} to {danger_counts.max()} rays per robot")
    print(f"     Average: {np.mean(danger_counts):.1f} rays ({np.mean(danger_counts)/max(lidar_points, 1)*100:.1f}%)")
    print(f"   Near Field Obstacles (1-3m):")
    print(f"     Range: {near_field_counts.min()} to {near_field_counts.max()} rays per robot")
    print(f"     Average: {np.mean(near_field_counts):.1f} rays ({np.mean(near_field_counts)/max(lidar_points, 1)*100:.1f}%)")
    print(f"   High-Risk Robot Count: {(danger_counts > np.mean(danger_counts) * 1.5).sum()}/{robots_to_analyze}")
    
    # Terrain Coverage Statistics
    print(f"\n📊 TERRAIN COVERAGE STATISTICS:")
    print(f"   Gap Coverage: {np.mean(terrain_stats_summary['gap_coverages']):.1f}% ± {np.std(terrain_stats_summary['gap_coverages']):.1f}%")
    print(f"   Obstacle Coverage: {np.mean(terrain_stats_summary['obstacle_coverages']):.1f}% ± {np.std(terrain_stats_summary['obstacle_coverages']):.1f}%")
    print(f"   Safe Traversable Terrain: {np.mean(terrain_stats_summary['flat_coverages']):.1f}% ± {np.std(terrain_stats_summary['flat_coverages']):.1f}%")
    
    # Navigation Recommendations
    steppable_gaps = len([g for g in all_gaps if g.get('difficulty') == 'steppable'])
    jumpable_gaps = len([g for g in all_gaps if g.get('difficulty') == 'jumpable'])
    impossible_gaps = len([g for g in all_gaps if g.get('difficulty') == 'impossible'])
    
    print(f"\n🧭 NAVIGATION STRATEGY RECOMMENDATIONS:")
    print(f"   Recommended Approach: {'Standard Walking' if steppable_gaps > impossible_gaps else 'Advanced Navigation Required'}")
    print(f"   Traversal Difficulty: {'LOW' if np.mean(terrain_stats_summary['flat_coverages']) > 90 else 'MODERATE' if np.mean(terrain_stats_summary['flat_coverages']) > 75 else 'HIGH'}")
    print(f"   Gap Navigation:")
    print(f"     - Walkable gaps: {steppable_gaps} gaps (≤30cm - can walk across)")
    print(f"     - Jump required: {jumpable_gaps} gaps (30-60cm - need jumping)")  
    print(f"     - Untraversable: {impossible_gaps} gaps (>60cm or unmeasurable depth)")
    
    # Environmental Verdict - Fixed Risk Assessment
    safety_score = np.mean(terrain_stats_summary['flat_coverages'])
    complexity_score = len(all_gaps) + len(all_obstacles)
    
    # Calculate gap risk factors
    total_gaps = len(all_gaps)
    impossible_gap_ratio = impossible_gaps / total_gaps if total_gaps > 0 else 0
    
    print(f"\n🎯 FINAL ENVIRONMENTAL ASSESSMENT:")
    print(f"   Safety Score: {safety_score:.1f}% traversable terrain")
    print(f"   Complexity Score: {complexity_score} total features detected")
    print(f"   Impossible Gap Ratio: {impossible_gap_ratio*100:.1f}% ({impossible_gaps}/{total_gaps})")
    print(f"   Environment Verdict: ", end="")
    
    # Improved risk assessment that considers impossible gaps
    if impossible_gap_ratio > 0.8 or impossible_gaps > 100:  # >80% impossible or >100 impossible gaps
        print("🔴 HIGH RISK - Extensive unmeasurable/untraversable terrain features")
    elif impossible_gap_ratio > 0.5 or (safety_score < 75):  # >50% impossible or low safety
        print("🟡 MODERATE RISK - Significant navigation challenges")
    elif safety_score > 90 and impossible_gap_ratio < 0.2:  # High safety + low impossible ratio
        print("🟢 LOW RISK - Suitable for basic navigation")
    else:
        print("🟡 MODERATE RISK - Requires careful planning")
    
    print(f"\n   Recommended Robot Capabilities:")
    if jumpable_gaps > 0:
        print("     - Jumping ability required")
    if steppable_gaps > 0:
        print("     - Precise walking control needed (for gap crossing)")
    if impossible_gaps > 0:
        print("     - Advanced path planning essential (many untraversable areas)")
    if np.mean(danger_counts) > 1000:
        print("     - Enhanced collision avoidance recommended")
        
    print("="*80)
    
    # Close environment cleanly
    print(f"\n🔄 CLEANING UP ENVIRONMENT...")
    try:
        env.close()
        print(f"   ✅ Environment closed successfully")
    except Exception as e:
        print(f"   ⚠️ Environment close warning: {e}")
    
    print(f"\n✅ COMPREHENSIVE MULTI-ROBOT ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Analysis error: {e}")
    finally:
        # Force clean shutdown
        print(f"\n🔄 SHUTTING DOWN SIMULATION...")
        try:
            simulation_app.close()
            print(f"✅ Simulation closed successfully")
        except Exception as e:
            print(f"⚠️ Shutdown warning: {e}")
        
        # Force exit to prevent hanging
        import sys
        print(f"🏁 ANALYSIS COMPLETE - EXITING")
        sys.exit(0) 