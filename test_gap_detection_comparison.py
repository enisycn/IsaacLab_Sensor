#!/usr/bin/env python3

"""
Test Script: Compare Old vs New Gap Detection Systems
Tests the improved adaptive gap detection against the current fixed threshold approach.
"""

import argparse
import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Gap Detection Comparison Test")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-SDS-Velocity-Flat-G1-v0", help="Task name")
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

# Import the improved gap detection system
from improved_gap_detection import adaptive_gap_detection, generate_terrain_report

def old_gap_detection(height_grid, resolution=0.015):
    """
    Current gap detection method (fixed 5cm threshold).
    """
    
    if torch.is_tensor(height_grid):
        height_grid = height_grid.cpu().numpy()
    
    # Filter sensor artifacts
    valid_terrain = height_grid < 19.9
    valid_heights = height_grid[valid_terrain]
    
    if len(valid_heights) == 0:
        return {"error": "No valid terrain data"}
    
    # Fixed threshold approach (current method)
    ground_level = np.median(valid_heights)
    gap_threshold = 0.05  # Fixed 5cm
    gaps_mask = valid_terrain & (height_grid < (ground_level - gap_threshold))
    
    # Simple connected components
    labeled, num_gaps = simple_connected_components(gaps_mask)
    
    # Basic classification
    steppable = 0
    jumpable = 0
    impossible = 0
    
    for gap_id in range(1, min(num_gaps + 1, 21)):
        gap_region = labeled == gap_id
        gap_coords = np.where(gap_region)
        
        if len(gap_coords[0]) < 2:
            continue
        
        min_row, max_row = gap_coords[0].min(), gap_coords[0].max()
        min_col, max_col = gap_coords[1].min(), gap_coords[1].max()
        
        height_m = (max_row - min_row + 1) * resolution
        width_m = (max_col - min_col + 1) * resolution
        min_crossing = min(height_m, width_m)
        
        if min_crossing <= 0.30:
            steppable += 1
        elif min_crossing <= 0.60:
            jumpable += 1
        else:
            impossible += 1
    
    return {
        "method": "OLD_FIXED_5CM",
        "terrain_std_cm": valid_heights.std() * 100,
        "total_gaps": num_gaps,
        "gap_percentage": gaps_mask.sum() / valid_terrain.sum() * 100,
        "classifications": {
            "steppable": steppable,
            "jumpable": jumpable, 
            "impossible": impossible
        }
    }

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

def compare_methods(robot_id, height_grid):
    """Compare old vs new gap detection methods."""
    
    print(f"\n{'='*80}")
    print(f"🔍 COMPARING GAP DETECTION METHODS - ROBOT #{robot_id}")
    print(f"{'='*80}")
    
    # Test old method
    print("\n📊 OLD METHOD (Fixed 5cm Threshold):")
    print("-" * 50)
    old_results = old_gap_detection(height_grid)
    
    if "error" not in old_results:
        print(f"   Terrain Std: {old_results['terrain_std_cm']:.1f}cm")
        print(f"   Total Gaps: {old_results['total_gaps']}")
        print(f"   Gap Coverage: {old_results['gap_percentage']:.1f}%")
        print(f"   Steppable: {old_results['classifications']['steppable']}")
        print(f"   Jumpable: {old_results['classifications']['jumpable']}")
        print(f"   Impossible: {old_results['classifications']['impossible']}")
    else:
        print(f"   ❌ {old_results['error']}")
    
    # Test new method
    print("\n🚀 NEW METHOD (Adaptive Threshold):")
    print("-" * 50)
    new_results = adaptive_gap_detection(height_grid)
    
    if "error" not in new_results:
        terrain = new_results["terrain_analysis"]
        detection = new_results["detection_results"]
        gaps = new_results["gap_analysis"]
        thresholds = new_results["thresholds"]
        
        print(f"   Terrain Type: {terrain['type']}")
        print(f"   Terrain Std: {terrain['std_dev_cm']:.1f}cm")
        print(f"   Adaptive Threshold: {thresholds['standard_cm']:.1f}cm")
        print(f"   Total Gaps: {gaps['total_gaps']}")
        print(f"   Gap Coverage: {detection['standard_gaps_percent']:.1f}%")
        print(f"   Steppable: {gaps['classifications']['steppable']}")
        print(f"   Jumpable: {gaps['classifications']['jumpable']}")
        print(f"   Impossible: {gaps['classifications']['impossible']}")
        
        print(f"\n🎯 Multi-Level Detection:")
        print(f"   Conservative ({thresholds['conservative_cm']:.1f}cm): {detection['real_gaps_percent']:.1f}%")
        print(f"   Standard ({thresholds['standard_cm']:.1f}cm): {detection['standard_gaps_percent']:.1f}%")
        print(f"   Sensitive ({thresholds['sensitive_cm']:.1f}cm): {detection['all_depressions_percent']:.1f}%")
    else:
        print(f"   ❌ {new_results['error']}")
    
    # Analysis and recommendation
    print(f"\n📈 COMPARISON ANALYSIS:")
    print("-" * 50)
    
    if "error" in old_results or "error" in new_results:
        print("   ❌ Cannot compare due to errors")
        return None
    
    # Compare gap detection sensitivity
    old_gaps = old_results['gap_percentage']
    new_gaps = new_results["detection_results"]['standard_gaps_percent']
    conservative_gaps = new_results["detection_results"]['real_gaps_percent']
    
    terrain_std = new_results["terrain_analysis"]['std_dev_cm']
    
    if terrain_std < 1.0:  # Very flat terrain
        print(f"   🟢 FLAT TERRAIN: Both methods should agree")
        if abs(old_gaps - new_gaps) < 1.0:
            print(f"   ✅ Methods agree ({old_gaps:.1f}% vs {new_gaps:.1f}%)")
        else:
            print(f"   ⚠️ Methods disagree ({old_gaps:.1f}% vs {new_gaps:.1f}%)")
    
    elif terrain_std > 3.0:  # Complex terrain
        print(f"   🟡 COMPLEX TERRAIN: New method should filter better")
        if new_gaps < old_gaps * 0.7:  # New method detects significantly fewer gaps
            print(f"   ✅ Improved filtering: {old_gaps:.1f}% → {new_gaps:.1f}% gaps")
            print(f"   📊 Conservative estimate: {conservative_gaps:.1f}% (likely real obstacles)")
        else:
            print(f"   ⚠️ Similar sensitivity: {old_gaps:.1f}% vs {new_gaps:.1f}%")
    
    else:  # Moderate terrain
        print(f"   🟠 MODERATE TERRAIN: New method should be more selective")
        if new_gaps <= old_gaps:
            print(f"   ✅ Better selectivity: {old_gaps:.1f}% → {new_gaps:.1f}% gaps")
        else:
            print(f"   ⚠️ More sensitive: {old_gaps:.1f}% → {new_gaps:.1f}% gaps")
    
    # Overall recommendation
    print(f"\n🎯 RECOMMENDATION:")
    if terrain_std < 1.0 and abs(old_gaps - new_gaps) < 1.0:
        print("   ✅ KEEP EITHER - Both methods work well for flat terrain")
    elif terrain_std > 2.0 and new_gaps < old_gaps * 0.8:
        print("   🚀 USE NEW METHOD - Much better filtering for complex terrain")
    elif new_gaps <= old_gaps:
        print("   ⬆️ PREFER NEW METHOD - Better or equal performance")
    else:
        print("   ⚠️ INVESTIGATE - New method may be too sensitive")
    
    return {
        "old": old_results,
        "new": new_results,
        "terrain_complexity": terrain_std,
        "improvement": old_gaps - new_gaps
    }

def main():
    """Main test function."""
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True
    )
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    print("🧪 GAP DETECTION COMPARISON TEST")
    print("="*80)
    print("Testing old (fixed 5cm) vs new (adaptive) gap detection methods...")
    
    # Reset environment and get observations
    observations, _ = env.reset()
    obs_data = observations['policy']
    
    # Height scanner configuration
    grid_height, grid_width = 400, 533
    expected_height_points = grid_height * grid_width
    
    results_summary = []
    
    # Test each robot
    for robot_id in range(args_cli.num_envs):
        
        # Extract height scanner data
        robot_data = obs_data[robot_id]
        robot_state_dims = 110
        
        remaining_data = robot_data[robot_state_dims:]
        if len(remaining_data) < expected_height_points:
            print(f"❌ Robot {robot_id}: Insufficient height data")
            continue
        
        height_scan = remaining_data[:expected_height_points]
        height_grid = height_scan.reshape(grid_height, grid_width)
        
        # Compare methods
        comparison_result = compare_methods(robot_id, height_grid)
        if comparison_result:
            results_summary.append(comparison_result)
    
    # Overall summary
    print(f"\n🎯 OVERALL TEST SUMMARY:")
    print("="*80)
    
    if not results_summary:
        print("❌ No valid comparisons completed")
        env.close()
        return
    
    complex_terrains = [r for r in results_summary if r["terrain_complexity"] > 2.0]
    flat_terrains = [r for r in results_summary if r["terrain_complexity"] < 1.0]
    
    print(f"   Total Robots Tested: {len(results_summary)}")
    print(f"   Complex Terrain Robots: {len(complex_terrains)}")
    print(f"   Flat Terrain Robots: {len(flat_terrains)}")
    
    if complex_terrains:
        avg_improvement = np.mean([r["improvement"] for r in complex_terrains])
        print(f"   Avg Improvement on Complex Terrain: {avg_improvement:.1f}% fewer gaps detected")
    
    # Final recommendation
    print(f"\n✅ INTEGRATION RECOMMENDATION:")
    if len(complex_terrains) > 0 and avg_improvement > 5.0:
        print("   🚀 INTEGRATE NEW METHOD - Significant improvement on complex terrain")
    elif all(r["improvement"] >= 0 for r in results_summary):
        print("   ⬆️ INTEGRATE NEW METHOD - Better or equal performance across all terrain types")
    else:
        print("   ⚠️ REVIEW RESULTS - Mixed performance, needs investigation")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close() 