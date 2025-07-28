#!/usr/bin/env python3

"""
Enhanced environmental analysis script for GPT-ready gap detection and terrain analysis.
Extracts specific gap dimensions, heights, obstacle information, and navigation metrics.
"""

import numpy as np
import torch
import gymnasium as gym

# Isaac Lab imports
from isaaclab.app import AppLauncher
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

def simple_connected_components(binary_mask):
    """Simple connected component analysis using numpy."""
    labeled = np.zeros_like(binary_mask, dtype=int)
    current_label = 0
    
    rows, cols = binary_mask.shape
    
    for i in range(rows):
        for j in range(cols):
            if binary_mask[i, j] and labeled[i, j] == 0:
                current_label += 1
                # Simple flood fill
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if (r >= 0 and r < rows and c >= 0 and c < cols and 
                        binary_mask[r, c] and labeled[r, c] == 0):
                        labeled[r, c] = current_label
                        # Add 4-connected neighbors
                        stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
    
    return labeled, current_label

def analyze_gaps_and_terrain(height_data, resolution=0.015, area_size=(8.0, 6.0)):
    """
    Detailed analysis of gaps, obstacles, and terrain features.
    
    Args:
        height_data: Height map data from scanner
        resolution: Grid resolution in meters
        area_size: (width, height) of scanned area in meters
    """
    # Reshape height data into 2D grid
    grid_width = int(area_size[0] / resolution)
    grid_height = int(area_size[1] / resolution) 
    
    try:
        height_grid = height_data[:grid_width * grid_height].reshape(grid_width, grid_height)
    except:
        # Fallback if reshape doesn't work perfectly
        available_points = len(height_data)
        side_length = int(np.sqrt(available_points))
        height_grid = height_data[:side_length**2].reshape(side_length, side_length)
        grid_width = grid_height = side_length
        resolution = min(area_size) / side_length
    
    ground_level = np.median(height_grid)
    
    # Define thresholds
    gap_threshold = ground_level - 0.05  # 5cm below ground = gap
    obstacle_threshold = ground_level + 0.05  # 5cm above ground = obstacle
    
    # Create binary masks
    gaps_mask = height_grid < gap_threshold
    obstacles_mask = height_grid > obstacle_threshold
    flat_mask = (height_grid >= gap_threshold) & (height_grid <= obstacle_threshold)
    
    analysis = {
        'ground_level': float(ground_level),
        'min_height': float(height_grid.min()),
        'max_height': float(height_grid.max()),
        'height_range': float(height_grid.max() - height_grid.min()),
        'resolution_m': resolution,
        'scanned_area_m2': grid_width * grid_height * (resolution ** 2),
        'total_points': int(grid_width * grid_height),
    }
    
    # Analyze gaps
    if gaps_mask.any():
        gap_clusters = analyze_clusters(gaps_mask, resolution, "gaps")
        analysis['gaps'] = gap_clusters
        analysis['gap_coverage_percent'] = float(gaps_mask.sum() / gaps_mask.size * 100)
        analysis['deepest_gap_depth'] = float(ground_level - height_grid[gaps_mask].min())
    else:
        analysis['gaps'] = []
        analysis['gap_coverage_percent'] = 0.0
        analysis['deepest_gap_depth'] = 0.0
    
    # Analyze obstacles  
    if obstacles_mask.any():
        obstacle_clusters = analyze_clusters(obstacles_mask, resolution, "obstacles")
        analysis['obstacles'] = obstacle_clusters
        analysis['obstacle_coverage_percent'] = float(obstacles_mask.sum() / obstacles_mask.size * 100)
        analysis['highest_obstacle'] = float(height_grid[obstacles_mask].max() - ground_level)
    else:
        analysis['obstacles'] = []
        analysis['obstacle_coverage_percent'] = 0.0
        analysis['highest_obstacle'] = 0.0
    
    # Analyze flat/traversable areas
    analysis['flat_coverage_percent'] = float(flat_mask.sum() / flat_mask.size * 100)
    
    return analysis

def analyze_clusters(binary_mask, resolution, feature_type):
    """Analyze connected components (gaps or obstacles) and extract dimensions."""
    labeled, num_features = simple_connected_components(binary_mask)
    clusters = []
    
    for i in range(1, num_features + 1):
        cluster_mask = labeled == i
        cluster_coords = np.where(cluster_mask)
        
        if len(cluster_coords[0]) < 3:  # Skip tiny clusters
            continue
            
        # Calculate dimensions
        min_row, max_row = cluster_coords[0].min(), cluster_coords[0].max()
        min_col, max_col = cluster_coords[1].min(), cluster_coords[1].max()
        
        width_m = (max_row - min_row + 1) * resolution
        length_m = (max_col - min_col + 1) * resolution
        area_m2 = cluster_mask.sum() * (resolution ** 2)
        
        # Determine if gap is jumpable/steppable based on dimensions
        if feature_type == "gaps":
            if width_m <= 0.3:
                difficulty = "steppable"
            elif width_m <= 0.8:
                difficulty = "careful_step"
            elif width_m <= 1.2:
                difficulty = "jumpable"
            else:
                difficulty = "requires_detour"
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

def analyze_lidar_obstacles(lidar_data, max_range=30.0):
    """Analyze LiDAR data for obstacle detection and free space."""
    
    # LiDAR configuration: 64 channels × 360 horizontal
    channels = 64
    horizontal_rays = 360
    
    try:
        lidar_grid = lidar_data[:channels * horizontal_rays].reshape(channels, horizontal_rays)
    except:
        # Fallback 
        available_points = len(lidar_data)
        side_length = int(np.sqrt(available_points))
        lidar_grid = lidar_data[:side_length**2].reshape(side_length, side_length)
        channels = horizontal_rays = side_length
    
    analysis = {
        'total_rays': int(channels * horizontal_rays),
        'max_range_m': max_range,
        'min_distance_m': float(lidar_data.min()),
        'max_distance_m': float(lidar_data.max()),
        'avg_distance_m': float(lidar_data.mean()),
    }
    
    # Distance-based obstacle analysis
    close_obstacles = lidar_data < 1.0     # < 1m = immediate danger
    near_obstacles = (lidar_data >= 1.0) & (lidar_data < 3.0)   # 1-3m = near field
    mid_obstacles = (lidar_data >= 3.0) & (lidar_data < 10.0)   # 3-10m = mid field  
    far_features = (lidar_data >= 10.0) & (lidar_data < max_range)  # 10m+ = far field
    max_range_rays = lidar_data >= max_range  # No obstacle detected
    
    analysis['obstacle_zones'] = {
        'immediate_danger_rays': int(close_obstacles.sum()),
        'near_field_rays': int(near_obstacles.sum()),
        'mid_field_rays': int(mid_obstacles.sum()),
        'far_field_rays': int(far_features.sum()),
        'clear_rays': int(max_range_rays.sum()),
    }
    
    # Calculate percentages
    total_rays = len(lidar_data)
    analysis['obstacle_distribution'] = {
        'immediate_danger_percent': round(close_obstacles.sum() / total_rays * 100, 1),
        'near_field_percent': round(near_obstacles.sum() / total_rays * 100, 1),
        'mid_field_percent': round(mid_obstacles.sum() / total_rays * 100, 1),
        'far_field_percent': round(far_features.sum() / total_rays * 100, 1),
        'clear_percent': round(max_range_rays.sum() / total_rays * 100, 1),
    }
    
    # Find clear directions (useful for path planning)
    if len(lidar_grid.shape) == 2:
        # Average distance per horizontal direction
        avg_per_direction = lidar_grid.mean(axis=0)
        clear_directions = []
        
        for angle in range(0, 360, 30):  # Check every 30 degrees
            if angle < len(avg_per_direction):
                distance = avg_per_direction[angle]
                if distance > 5.0:  # 5m+ clear distance
                    clear_directions.append({
                        'angle_deg': angle,
                        'clear_distance_m': round(float(distance), 2)
                    })
        
        analysis['clear_directions'] = clear_directions
    
    return analysis

def print_gpt_ready_analysis(terrain_analysis, lidar_analysis):
    """Print detailed analysis in GPT-ready format."""
    
    print("\n" + "="*80)
    print("🧠 GPT-READY ENVIRONMENTAL ANALYSIS")
    print("="*80)
    
    print(f"\n📊 TERRAIN OVERVIEW:")
    print(f"   Ground Level: {terrain_analysis['ground_level']:.3f}m")
    print(f"   Height Range: {terrain_analysis['min_height']:.3f}m to {terrain_analysis['max_height']:.3f}m")
    print(f"   Total Variation: {terrain_analysis['height_range']:.3f}m")
    print(f"   Scanned Area: {terrain_analysis['scanned_area_m2']:.1f} m²")
    print(f"   Resolution: {terrain_analysis['resolution_m']*100:.1f}cm grid")
    
    print(f"\n🕳️ GAP ANALYSIS:")
    if terrain_analysis['gaps']:
        print(f"   Total Gaps Found: {len(terrain_analysis['gaps'])}")
        print(f"   Gap Coverage: {terrain_analysis['gap_coverage_percent']:.1f}%")
        print(f"   Deepest Gap: {terrain_analysis['deepest_gap_depth']:.3f}m below ground")
        
        print(f"\n   📏 GAP DETAILS:")
        for i, gap in enumerate(terrain_analysis['gaps'][:5]):  # Show top 5 gaps
            print(f"   Gap #{gap['id']}: {gap['width_m']}m × {gap['length_m']}m "
                  f"(Area: {gap['area_m2']} m²) - {gap['difficulty']}")
            print(f"      Center: ({gap['center_x_m']}m, {gap['center_y_m']}m)")
    else:
        print("   No significant gaps detected")
    
    print(f"\n🗻 OBSTACLE ANALYSIS:")
    if terrain_analysis['obstacles']:
        print(f"   Total Obstacles: {len(terrain_analysis['obstacles'])}")
        print(f"   Obstacle Coverage: {terrain_analysis['obstacle_coverage_percent']:.1f}%")
        print(f"   Highest Obstacle: {terrain_analysis['highest_obstacle']:.3f}m above ground")
        
        print(f"\n   📏 OBSTACLE DETAILS:")
        for i, obs in enumerate(terrain_analysis['obstacles'][:5]):  # Show top 5 obstacles
            print(f"   Obstacle #{obs['id']}: {obs['width_m']}m × {obs['length_m']}m "
                  f"(Area: {obs['area_m2']} m²) - {obs['difficulty']}")
            print(f"      Center: ({obs['center_x_m']}m, {obs['center_y_m']}m)")
    else:
        print("   No significant obstacles detected")
    
    print(f"\n🚶 TRAVERSABLE AREA:")
    print(f"   Flat/Safe Terrain: {terrain_analysis['flat_coverage_percent']:.1f}%")
    
    print(f"\n📡 LIDAR OBSTACLE DETECTION:")
    print(f"   Total LiDAR Rays: {lidar_analysis['total_rays']:,}")
    print(f"   Detection Range: 0.1m to {lidar_analysis['max_range_m']}m")
    print(f"   Average Distance: {lidar_analysis['avg_distance_m']:.2f}m")
    
    print(f"\n   🎯 OBSTACLE ZONES:")
    zones = lidar_analysis['obstacle_zones']
    dist = lidar_analysis['obstacle_distribution']
    print(f"   Immediate Danger (<1m): {zones['immediate_danger_rays']} rays ({dist['immediate_danger_percent']}%)")
    print(f"   Near Field (1-3m): {zones['near_field_rays']} rays ({dist['near_field_percent']}%)")
    print(f"   Mid Field (3-10m): {zones['mid_field_rays']} rays ({dist['mid_field_percent']}%)")
    print(f"   Far Field (10-30m): {zones['far_field_rays']} rays ({dist['far_field_percent']}%)")
    print(f"   Clear Space (30m+): {zones['clear_rays']} rays ({dist['clear_percent']}%)")
    
    if 'clear_directions' in lidar_analysis and lidar_analysis['clear_directions']:
        print(f"\n   🧭 CLEAR NAVIGATION DIRECTIONS:")
        for direction in lidar_analysis['clear_directions'][:8]:  # Show best directions
            print(f"   {direction['angle_deg']}° - Clear for {direction['clear_distance_m']}m")
    
    print(f"\n🤖 NAVIGATION RECOMMENDATIONS:")
    
    # Gap-based recommendations
    steppable_gaps = sum(1 for gap in terrain_analysis.get('gaps', []) if gap['difficulty'] == 'steppable')
    jumpable_gaps = sum(1 for gap in terrain_analysis.get('gaps', []) if gap['difficulty'] in ['careful_step', 'jumpable'])
    detour_gaps = sum(1 for gap in terrain_analysis.get('gaps', []) if gap['difficulty'] == 'requires_detour')
    
    print(f"   Steppable Gaps: {steppable_gaps}")
    print(f"   Jumpable Gaps: {jumpable_gaps}")
    print(f"   Detour Required: {detour_gaps}")
    
    # Safety assessment
    danger_level = dist['immediate_danger_percent']
    if danger_level > 20:
        safety = "HIGH RISK"
    elif danger_level > 10:
        safety = "MODERATE RISK"
    elif danger_level > 5:
        safety = "LOW RISK"
    else:
        safety = "SAFE"
    
    print(f"   Current Safety Level: {safety}")
    print(f"   Recommended Strategy: ", end="")
    
    if terrain_analysis['flat_coverage_percent'] > 70:
        print("Direct navigation with obstacle avoidance")
    elif steppable_gaps > jumpable_gaps:
        print("Careful stepping with height monitoring")
    elif jumpable_gaps > 0:
        print("Dynamic jumping and gap traversal")
    else:
        print("Detour planning around major obstacles")

def analyze_sensor_data():
    """Main analysis function."""
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        task="Isaac-SDS-Velocity-Flat-G1-v0",
        device="cuda:0",
        num_envs=2,  # Just 2 robots for analysis
        use_fabric=True
    )
    
    # Create environment
    env = gym.make("Isaac-SDS-Velocity-Flat-G1-v0", cfg=env_cfg)
    
    print("🚀 ADVANCED ENVIRONMENTAL INTELLIGENCE ANALYSIS")
    print("="*80)
    
    # Reset environment
    observations, _ = env.reset()
    obs_data = observations['policy']  # Shape: [num_envs, total_obs]
    
    # Calculate dimensions based on enhanced config
    height_scan_points = int((8.0 / 0.015) * (6.0 / 0.015))  # ~213,200 points
    lidar_points = 64 * 360  # 23,040 points
    robot_state_dims = 110
    
    print(f"📊 SENSOR DATA DIMENSIONS:")
    print(f"   Total observations per robot: {obs_data.shape[1]:,}")
    print(f"   Robot state: {robot_state_dims}")
    print(f"   Height scanner points: {height_scan_points:,}")
    print(f"   LiDAR distance measurements: {lidar_points:,}")
    
    # Extract sensor data for first robot
    robot_0_data = obs_data[0]
    
    # Extract height scanner data
    height_start = robot_state_dims
    height_end = height_start + min(height_scan_points, len(robot_0_data) - robot_state_dims - lidar_points)
    height_scan = robot_0_data[height_start:height_end]
    
    # Extract LiDAR data 
    lidar_start = height_end
    lidar_end = lidar_start + min(lidar_points, len(robot_0_data) - lidar_start)
    lidar_range = robot_0_data[lidar_start:lidar_end]
    
    print(f"✅ Successfully extracted {len(height_scan):,} height points and {len(lidar_range):,} LiDAR measurements")
    
    # Perform detailed analysis
    terrain_analysis = analyze_gaps_and_terrain(height_scan)
    lidar_analysis = analyze_lidar_obstacles(lidar_range)
    
    # Print comprehensive results
    print_gpt_ready_analysis(terrain_analysis, lidar_analysis)
    
    # Close environment
    env.close()
    print(f"\n✅ ENVIRONMENTAL ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    # Launch Isaac Sim
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    
    try:
        analyze_sensor_data()
    finally:
        simulation_app.close() 