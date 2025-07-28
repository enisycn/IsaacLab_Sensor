#!/usr/bin/env python3

"""
Improved Gap Detection System
Implements adaptive thresholds and terrain-aware gap classification.
"""

import numpy as np
import torch

def adaptive_gap_detection(height_grid, resolution=0.015):
    """
    Improved gap detection with adaptive thresholds based on terrain characteristics.
    
    Args:
        height_grid: 2D array of height measurements
        resolution: Grid resolution in meters per cell
    
    Returns:
        dict: Comprehensive gap analysis results
    """
    
    # Convert to numpy if needed
    if torch.is_tensor(height_grid):
        height_grid = height_grid.cpu().numpy()
    
    # Filter sensor artifacts
    valid_terrain = height_grid < 19.9
    valid_heights = height_grid[valid_terrain]
    
    if len(valid_heights) == 0:
        return {"error": "No valid terrain data"}
    
    # === TERRAIN CHARACTERIZATION ===
    ground_level = np.median(valid_heights)
    terrain_std = valid_heights.std()
    height_range = valid_heights.max() - valid_heights.min()
    
    # Classify terrain complexity
    if terrain_std < 0.01:  # <1cm variation
        terrain_type = "FLAT"
        complexity_factor = 1.0
    elif terrain_std < 0.03:  # 1-3cm variation  
        terrain_type = "SMOOTH"
        complexity_factor = 1.5
    elif terrain_std < 0.05:  # 3-5cm variation
        terrain_type = "MODERATE" 
        complexity_factor = 2.0
    else:  # >5cm variation
        terrain_type = "COMPLEX"
        complexity_factor = 2.5
    
    # === ADAPTIVE THRESHOLD CALCULATION ===
    
    # Base threshold: minimum detectable gap depth
    base_threshold = 0.02  # 2cm minimum
    
    # Adaptive component: scale with terrain variation
    adaptive_component = terrain_std * complexity_factor
    
    # Final threshold with reasonable bounds
    gap_threshold = max(base_threshold, min(0.10, adaptive_component))
    
    # === MULTI-LEVEL GAP DETECTION ===
    
    # Level 1: Conservative (real obstacles only)
    conservative_threshold = max(0.03, terrain_std * 3.0)
    real_gaps = valid_terrain & (height_grid < (ground_level - conservative_threshold))
    
    # Level 2: Standard (adaptive threshold)
    standard_gaps = valid_terrain & (height_grid < (ground_level - gap_threshold))
    
    # Level 3: Sensitive (catch everything)
    sensitive_threshold = max(0.015, terrain_std * 1.0)
    all_depressions = valid_terrain & (height_grid < (ground_level - sensitive_threshold))
    
    # === GAP CLASSIFICATION ===
    results = {
        "terrain_analysis": {
            "type": terrain_type,
            "std_dev_cm": terrain_std * 100,
            "height_range_cm": height_range * 100,
            "ground_level": ground_level,
            "complexity_factor": complexity_factor
        },
        "thresholds": {
            "conservative_cm": conservative_threshold * 100,
            "standard_cm": gap_threshold * 100,
            "sensitive_cm": sensitive_threshold * 100
        },
        "detection_results": {
            "real_gaps_percent": real_gaps.sum() / valid_terrain.sum() * 100,
            "standard_gaps_percent": standard_gaps.sum() / valid_terrain.sum() * 100,
            "all_depressions_percent": all_depressions.sum() / valid_terrain.sum() * 100
        }
    }
    
    # Analyze gap regions with standard threshold
    gap_analysis = analyze_gap_regions(height_grid, standard_gaps, resolution, ground_level)
    results["gap_analysis"] = gap_analysis
    
    return results

def analyze_gap_regions(height_grid, gap_mask, resolution, ground_level):
    """Analyze individual gap regions with proper classification."""
    
    if not gap_mask.any():
        return {"total_gaps": 0, "classifications": {"steppable": 0, "jumpable": 0, "impossible": 0}}
    
    # Find connected components
    labeled, num_gaps = simple_connected_components(gap_mask)
    
    classifications = {"steppable": 0, "jumpable": 0, "impossible": 0}
    gap_details = []
    
    for gap_id in range(1, min(num_gaps + 1, 21)):  # Analyze up to 20 gaps
        gap_region = labeled == gap_id
        gap_coords = np.where(gap_region)
        
        if len(gap_coords[0]) < 2:
            continue
        
        # Calculate dimensions
        min_row, max_row = gap_coords[0].min(), gap_coords[0].max()
        min_col, max_col = gap_coords[1].min(), gap_coords[1].max()
        
        height_m = (max_row - min_row + 1) * resolution
        width_m = (max_col - min_col + 1) * resolution
        
        # CRITICAL: Use minimum dimension for crossing
        min_crossing = min(height_m, width_m)
        
        # Get depth information
        gap_heights = height_grid[gap_region]
        gap_depth = ground_level - gap_heights.max()
        
        # Classify based on humanoid capabilities
        if min_crossing <= 0.30:  # 30cm step capability
            classification = "steppable"
        elif min_crossing <= 0.60:  # 60cm jump capability
            classification = "jumpable"
        else:
            classification = "impossible"
        
        classifications[classification] += 1
        
        gap_details.append({
            "id": gap_id,
            "dimensions_m": (height_m, width_m),
            "min_crossing_m": min_crossing,
            "depth_m": gap_depth,
            "points": gap_region.sum(),
            "classification": classification
        })
    
    return {
        "total_gaps": num_gaps,
        "analyzed_gaps": len(gap_details),
        "classifications": classifications,
        "details": gap_details[:10]  # Return first 10 for display
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

def generate_terrain_report(results):
    """Generate a comprehensive terrain analysis report."""
    
    if "error" in results:
        return f"❌ Error: {results['error']}"
    
    terrain = results["terrain_analysis"]
    thresholds = results["thresholds"]
    detection = results["detection_results"]
    gaps = results["gap_analysis"]
    
    report = f"""
🌍 TERRAIN ANALYSIS REPORT
{'='*50}

📊 Terrain Characteristics:
   Type: {terrain['type']}
   Variation: {terrain['std_dev_cm']:.1f}cm std dev
   Height Range: {terrain['height_range_cm']:.1f}cm
   Complexity Factor: {terrain['complexity_factor']:.1f}x

🎯 Adaptive Thresholds:
   Conservative: {thresholds['conservative_cm']:.1f}cm (real obstacles only)
   Standard: {thresholds['standard_cm']:.1f}cm (recommended)
   Sensitive: {thresholds['sensitive_cm']:.1f}cm (all depressions)

🔍 Detection Results:
   Real Gaps: {detection['real_gaps_percent']:.1f}%
   Standard Gaps: {detection['standard_gaps_percent']:.1f}%
   All Depressions: {detection['all_depressions_percent']:.1f}%

🕳️ Gap Analysis ({gaps['total_gaps']} total regions):
   Steppable (≤30cm): {gaps['classifications']['steppable']}
   Jumpable (30-60cm): {gaps['classifications']['jumpable']} 
   Impossible (>60cm): {gaps['classifications']['impossible']}

✅ Navigation Assessment:
"""
    
    total_analyzed = sum(gaps['classifications'].values())
    if total_analyzed == 0:
        report += "   CLEAR TERRAIN - No significant obstacles detected"
    elif gaps['classifications']['impossible'] == 0:
        if gaps['classifications']['jumpable'] == 0:
            report += "   ALL GAPS STEPPABLE - Standard locomotion sufficient"
        else:
            report += "   MIXED TERRAIN - Stepping and jumping required"
    else:
        report += "   COMPLEX TERRAIN - Requires advanced navigation strategies"
    
    return report

# Example usage function
def analyze_robot_terrain(height_scan_data, robot_id=0):
    """Analyze terrain for a specific robot using improved detection."""
    
    # Reshape height scan to grid (adjust dimensions as needed)
    grid_height, grid_width = 400, 533  # From environment config
    height_grid = height_scan_data[:grid_height*grid_width].reshape(grid_height, grid_width)
    
    # Run improved analysis
    results = adaptive_gap_detection(height_grid)
    
    # Generate report
    report = generate_terrain_report(results)
    
    print(f"\n🤖 ROBOT #{robot_id} TERRAIN ANALYSIS:")
    print(report)
    
    return results 