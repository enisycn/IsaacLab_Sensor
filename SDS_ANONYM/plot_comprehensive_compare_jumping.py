#!/usr/bin/env python3
"""
Comprehensive Jumping Metrics Comparison Plotting Tool
=====================================================

Generates publication-ready comparison plots for ALL 8 STANDARDIZED JUMPING METRICS between 
environment-aware and foundation-only modes with enhanced visualizations.

Jumping Metrics Coverage:
- 5 Smaller-is-Better: height_deviation (jump trajectory consistency), velocity_tracking_error, 
  disturbance_resistance, contact_termination_rate, obstacle_collision_count
- 3 Higher-is-Better: balance_stability_score, gait_smoothness_score (jump coordination smoothness),
  stair_climbing_performance (jumping on stairs - terrain 3 only)

JUMPING-SPECIFIC MODIFICATIONS:
- height_deviation: Measures jump trajectory consistency (smaller = more consistent jumping)
- gait_smoothness_score: Measures jumping coordination smoothness (higher = better bilateral coordination)
- stair_climbing_performance: Adapted for jumping on stairs (higher = better jumping ascent)

All terrain types (0,1,2,3) now have identical metric sets for fair jumping comparison.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, List
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'DejaVu Sans'
})

def load_results(path: str) -> Dict[str, Any]:
    """Load results from a .json or .pkl file."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return {
            "metadata": data.get("metadata", {}),
            "summary_metrics": data.get("summary", {}),  # Fixed: use 'summary' key
            "comprehensive_metrics": data.get("metrics", {}),  # Fixed: use 'metrics' key
            "episodes": data.get("episodes_summary", []),
        }
    elif suffix == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return {
            "metadata": data.get("metadata", {}),
            "summary_metrics": data.get("summary", {}),  # Fixed: use 'summary' key
            "comprehensive_metrics": data.get("metrics", {}),  # Fixed: use 'metrics' key
            "episodes": data.get("episodes", []),
        }
    else:
        raise ValueError(f"Unsupported file type: {path}")


def ensure_outdir(path: str) -> str:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir)


def to_title(text: str) -> str:
    """Convert snake_case to Title Case with better formatting."""
    return text.replace("_", " ").title().replace("Xy", "XY").replace("Com", "CoM").replace("Imu", "IMU")


def get_metric_description(category: str, metric: str) -> str:
    """Get human-readable description for all 7 standardized jumping metrics."""
    descriptions = {
        # JUMPING-SPECIFIC METRICS (smaller is better)
        "height_deviation": "Jump trajectory consistency - Smaller is Better",
        "velocity_tracking_error": "Velocity tracking error (m/s) - Smaller is Better", 
        "disturbance_resistance": "External disturbance resistance - Smaller is Better",
        "contact_termination_rate": "Fall/contact failure rate - Smaller is Better",
        
        # JUMPING COORDINATION QUALITY METRICS (higher is better)
        "balance_stability_score": "Body stability score - Higher is Better",
        "gait_smoothness_score": "Jump coordination smoothness - Higher is Better",
        
        # TERRAIN-SPECIFIC METRICS
        "obstacle_collision_count": "Upper body collision count - Smaller is Better",  # Smaller is better
        "stair_climbing_performance": "Stair jumping performance - Higher is Better",  # Higher is better (adapted for jumping)
        
        # Summary Metrics
        "total_steps": "Total simulation steps",
        "mean_reward": "Mean step reward",
        "total_reward": "Total collected reward",
        "collection_time": "Data collection time (s)",
    }
    return descriptions.get(metric, f"{to_title(metric)} ({to_title(category)})")


def is_higher_better_metric(metric: str) -> bool:
    """Determine if higher values are better for this metric."""
    higher_better_metrics = {
        "balance_stability_score",
        "gait_smoothness_score",
        "stair_climbing_performance"
    }
    return metric in higher_better_metrics


def plot_numeric_comparison(ax, env_name_a: str, env_name_b: str, data_a: Dict[str, Any], 
                          data_b: Dict[str, Any], category: str, metric: str, colors: List[str]):
    """Plot numeric metric comparison with enhanced modern aesthetics."""
    try:
        # Handle correct data structure: metrics are under 'metrics' key, summary under 'summary' key
        if category == "standardized":
            stats_a = data_a.get('metrics', {}).get(metric, {})
            stats_b = data_b.get('metrics', {}).get(metric, {})
        elif category == "summary":
            stats_a = data_a.get('summary', {})
            stats_b = data_b.get('summary', {})
            # For summary metrics, we need to create the stats dict format
            if metric in stats_a and metric in stats_b:
                stats_a = {'mean': float(stats_a[metric]), 'std': 0.0, 'count': 1}
                stats_b = {'mean': float(stats_b[metric]), 'std': 0.0, 'count': 1}
            else:
                return False
        else:
            # Legacy structure fallback
            stats_a = data_a.get(category, {}).get(metric, {})
            stats_b = data_b.get(category, {}).get(metric, {})
        
        if not (isinstance(stats_a, dict) and isinstance(stats_b, dict)):
            return False
            
        # Check if we have numeric statistics
        if not ("mean" in stats_a and "mean" in stats_b):
            return False
            
        mean_a, std_a = float(stats_a.get("mean", 0)), float(stats_a.get("std", 0))
        mean_b, std_b = float(stats_b.get("mean", 0)), float(stats_b.get("std", 0))
        count_a, count_b = int(stats_a.get("count", 0)), int(stats_b.get("count", 0))
        
        # Determine winner based on metric type (needed for direction indicator)
        higher_is_better = is_higher_better_metric(metric)
        
        # Enhanced color scheme with gradients
        if higher_is_better:
            # Blue to green gradient for higher-is-better metrics
            color_a = '#2E86AB'  # Deep blue
            color_b = '#A23B72'  # Deep pink/purple
            gradient_color_a = '#74C69D'  # Light green
            gradient_color_b = '#F4A261'  # Warm orange
        else:
            # Orange to teal gradient for smaller-is-better metrics  
            color_a = '#E76F51'  # Warm red-orange
            color_b = '#2A9D8F'  # Teal
            gradient_color_a = '#F4A261'  # Warm orange
            gradient_color_b = '#74C69D'  # Light green
        
        # Modern bar plot with enhanced styling
        x = np.array([0, 1])
        means = np.array([mean_a, mean_b])
        stds = np.array([std_a, std_b])
        labels = [env_name_a, env_name_b]
        bar_colors = [color_a, color_b]
        
        # Create bars with modern styling
        width = 0.6
        bars = ax.bar(x, means, width=width, yerr=stds, 
                     color=bar_colors, capsize=8,
                     alpha=0.85, edgecolor='white', linewidth=2.5,
                     error_kw={'elinewidth': 2, 'capsize': 8, 'alpha': 0.7})
        
        # Add subtle gradient effect using patches
        from matplotlib.patches import Rectangle
        import matplotlib.patches as mpatches
        
        for i, (bar, color, grad_color) in enumerate(zip(bars, bar_colors, [gradient_color_a, gradient_color_b])):
            # Create gradient effect
            height = bar.get_height()
            if height > 0:
                # Add gradient overlay
                gradient_rect = Rectangle((bar.get_x(), 0), bar.get_width(), height * 0.3,
                                        facecolor=grad_color, alpha=0.3, zorder=3)
                ax.add_patch(gradient_rect)
        
        # Enhanced value labels with better typography and spacing
        max_error = [std_a, std_b]
        max_height = max(means) if max(means) > 0 else max(stds)
        
        for i, (bar, mean, std, count) in enumerate(zip(bars, means, stds, [count_a, count_b])):
            height = bar.get_height()
            
            # Main value label - positioned higher to avoid overlap
            label_y = height + max_error[i] + 0.06 * max_height
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{mean:.4f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color='#2C3E50')
        
        # Enhanced styling with modern design
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold', fontsize=12, color='#2C3E50')
        
        # Improved y-axis label
        metric_desc = get_metric_description(category, metric)
        ax.set_ylabel(metric_desc, fontweight='bold', fontsize=11, color='#34495E')
        
        # Modern title with better spacing
        title_text = f"{to_title(metric)}"
        ax.set_title(title_text, fontweight='bold', fontsize=14, color='#2C3E50', pad=30)
        
        # Enhanced grid
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8, color='#BDC3C7')
        ax.set_axisbelow(True)
        
        # Modern spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add subtle background color
        ax.set_facecolor('#FAFAFA')
        
        # Calculate difference and winner
        diff = mean_a - mean_b
        diff_pct = (diff / max(abs(mean_b), 1e-6)) * 100
        
        if higher_is_better:
            # For higher-is-better metrics
            if mean_a > mean_b:
            winner_text = f"{env_name_a} WINS"
                winner_color = '#27AE60'
                winner_icon = "🏆"
            elif mean_b > mean_a:
            winner_text = f"{env_name_b} WINS"
                winner_color = '#27AE60'
                winner_icon = "🏆"
            else:
                winner_text = "TIE"
                winner_color = '#F39C12'
                winner_icon = "⚖️"
        else:
            # For smaller-is-better metrics
            if mean_a < mean_b:
                winner_text = f"{env_name_a} WINS"
                winner_color = '#27AE60'
                winner_icon = "🏆"
            elif mean_b > mean_a:
                winner_text = f"{env_name_b} WINS"
                winner_color = '#27AE60'
                winner_icon = "🏆"
        else:
            winner_text = "TIE"
                winner_color = '#F39C12'
                winner_icon = "⚖️"
        
        # Enhanced winner annotation positioned at the top of the plot
        ax.text(0.5, 0.95, f'{winner_icon} {winner_text}', transform=ax.transAxes,
                ha='center', va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=winner_color, 
                         alpha=0.2, edgecolor=winner_color, linewidth=2))
        
        # Difference annotation positioned slightly below winner text
        diff_text = f'Δ = {diff:+.4f} ({diff_pct:+.1f}%)'
        ax.text(0.5, 0.88, diff_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, color='#34495E',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         alpha=0.8, edgecolor='#BDC3C7', linewidth=1))
        
        # Add direction indicator badge - repositioned to avoid overlap
        direction_text = "📈 Higher = Better" if higher_is_better else "📉 Smaller = Better"
        direction_color = '#E8F5E8' if higher_is_better else '#FFF2E8'
        badge_edge_color = '#27AE60' if higher_is_better else '#E67E22'
        
        ax.text(0.02, 0.95, direction_text, transform=ax.transAxes, 
                ha='left', va='top', fontsize=9, fontweight='bold', color='#2C3E50',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=direction_color, 
                         alpha=0.9, edgecolor=badge_edge_color, linewidth=1.5))
        
        # Set y-axis limits with better padding to accommodate all text
        y_max = max_height + max(stds) + 0.15 * max_height  # More padding for labels
        ax.set_ylim(0, y_max * 1.3)  # Extra space at top
        
        return True
        
    except Exception as e:
        print(f"Error plotting {category}.{metric}: {e}")
        return False


def plot_dict_comparison(ax, env_name_a: str, env_name_b: str, data_a: Dict[str, Any], 
                        data_b: Dict[str, Any], category: str, metric: str, colors: List[str]):
    """Plot dictionary metric comparison (e.g., termination reasons)."""
    try:
        stats_a = data_a.get(category, {}).get(metric, {})
        stats_b = data_b.get(category, {}).get(metric, {})
        
        if not (isinstance(stats_a, dict) and isinstance(stats_b, dict)):
            return False
            
        # Skip if this is actually numeric data with mean/std
        if "mean" in stats_a or "mean" in stats_b:
            return False
            
        # Get all unique keys and their values
        all_keys = sorted(set(list(stats_a.keys()) + list(stats_b.keys())))
        if not all_keys:
            return False
            
        values_a = [int(stats_a.get(k, 0)) for k in all_keys]
        values_b = [int(stats_b.get(k, 0)) for k in all_keys]
        
        # Grouped bar chart
        x = np.arange(len(all_keys))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, values_a, width, label=env_name_a, color=colors[0], 
                      alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x + width/2, values_b, width, label=env_name_b, color=colors[1], 
                      alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # Add value labels on bars
        for bars, values in [(bars1, values_a), (bars2, values_b)]:
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                           str(value), ha='center', va='bottom', fontsize=8, weight='bold')
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels([to_title(str(k)) for k in all_keys], rotation=45, ha='right')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title(f"{to_title(category)}\n{to_title(metric)}", fontweight='bold', pad=20)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return True
        
    except Exception as e:
        print(f"Error plotting dict {category}.{metric}: {e}")
        return False


def plot_summary_comparison(ax, env_name_a: str, env_name_b: str, data_a: Dict[str, Any], 
                          data_b: Dict[str, Any], metric: str, colors: List[str]):
    """Plot summary metric comparison."""
    try:
        val_a = float(data_a.get("summary_metrics", {}).get(metric, 0))
        val_b = float(data_b.get("summary_metrics", {}).get(metric, 0))
        
        # Bar plot
        x = [0, 1]
        values = [val_a, val_b]
        labels = [env_name_a, env_name_b]
        
        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02*max(values),
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold')
        ax.set_ylabel(to_title(metric), fontweight='bold')
        ax.set_title(f"Summary Metric\n{to_title(metric)}", fontweight='bold', pad=20)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add difference annotation
        diff = val_a - val_b
        diff_pct = (diff / max(abs(val_b), 1e-6)) * 100
        ax.text(0.5, max(values) * 0.1, f'Δ = {diff:+.3f} ({diff_pct:+.1f}%)', 
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        return True
        
    except Exception as e:
        print(f"Error plotting summary {metric}: {e}")
        return False


def create_summary_info_plot(fig, data_a: Dict[str, Any], data_b: Dict[str, Any], 
                           env_name_a: str, env_name_b: str):
    """Create an informative summary comparison plot."""
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Extract metadata
    meta_a = data_a.get("metadata", {})
    meta_b = data_b.get("metadata", {})
    
    # Extract key summary metrics
    summary_a = data_a.get("summary_metrics", {})
    summary_b = data_b.get("summary_metrics", {})
    
    # Extract observation dimensions safely
    def extract_obs_dims(obs_space_str):
        try:
            if '(' in str(obs_space_str) and ')' in str(obs_space_str):
                # Extract dimensions from string like "Box(-inf, inf, (20, 792), float32)"
                parts = str(obs_space_str).split('(')
                if len(parts) >= 3:
                    dim_part = parts[2].split(')')[0].split(',')
                    if len(dim_part) >= 2:
                        return dim_part[1].strip()
            return "N/A"
        except:
            return "N/A"
    
    obs_dims_a = extract_obs_dims(meta_a.get('observation_space', ''))
    obs_dims_b = extract_obs_dims(meta_b.get('observation_space', ''))
    
    # Title
    ax.text(0.5, 0.95, "Environment-Aware vs Foundation-Only Locomotion", 
            ha='center', va='top', fontsize=16, weight='bold')
    
    # Create comparison table
    info_text = f"""
Configuration Comparison:
├─ Task: {meta_a.get('task', 'N/A')}
├─ {env_name_a}: {obs_dims_a} observation dims
├─ {env_name_b}: {obs_dims_b} observation dims
├─ Collection Steps: {meta_a.get('collection_steps', meta_b.get('collection_steps', 'N/A'))}
└─ Environments: {meta_a.get('num_envs', meta_b.get('num_envs', 'N/A'))}

Performance Summary:
├─ Mean Episode Reward:  {summary_a.get('mean_episode_reward', 'N/A')} vs {summary_b.get('mean_episode_reward', 'N/A')}
├─ Mean Episode Length:  {summary_a.get('mean_episode_length', 'N/A')} vs {summary_b.get('mean_episode_length', 'N/A')}
├─ Mean Step Reward:     {summary_a.get('mean_step_reward', 'N/A')} vs {summary_b.get('mean_step_reward', 'N/A')}
└─ Total Episodes:       {summary_a.get('total_episodes', 'N/A')} vs {summary_b.get('total_episodes', 'N/A')}

Key Insights:
• Observation space difference indicates sensor usage
• Higher-dimensional space = environmental sensing enabled
• Performance metrics show learning effectiveness
• Comprehensive metrics below show detailed comparisons
    """
    
    ax.text(0.1, 0.8, info_text, ha='left', va='top', fontsize=10, 
            fontfamily='monospace', linespacing=1.5)
    
    # Add color legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor='#1f77b4', alpha=0.8, label=env_name_a),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='#ff7f0e', alpha=0.8, label=env_name_b)
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)


def main():
    """Generate comprehensive comparison plots with enhanced aesthetics."""
    parser = argparse.ArgumentParser(description="Generate comprehensive metric comparison plots")
    parser.add_argument("--env_aware", required=True, help="Path to environment-aware data file")
    parser.add_argument("--foundation_only", required=True, help="Path to foundation-only data file")
    parser.add_argument("--outdir", default="plots_comprehensive", help="Output directory")
    parser.add_argument("--env_aware_label", default="Environment-Aware", help="Label for environment-aware data")
    parser.add_argument("--foundation_only_label", default="Foundation-Only", help="Label for foundation-only data")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.env_aware} and {args.foundation_only}")
    
    with open(args.env_aware, 'rb') as f:
        env_aware_data = pickle.load(f)
    
    with open(args.foundation_only, 'rb') as f:
        foundation_only_data = pickle.load(f)
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    
    # Set up modern matplotlib style
    plt.style.use('default')  # Start with clean default
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFAFA',
        'axes.edgecolor': '#CCCCCC',
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.8,
        'axes.linewidth': 0.8,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#2C3E50'
    })
    
    # Modern color palette
    modern_colors = ['#3498DB', '#E74C3C']  # Modern blue and red
    
    # Plot all 8 jumping metrics (includes stair jumping for terrain 3)
    standardized_metrics = [
        # UNIVERSAL 6 METRICS
        'height_deviation',
        'velocity_tracking_error', 
        'disturbance_resistance',
        'contact_termination_rate',
        'balance_stability_score',
        'gait_smoothness_score',
        # TERRAIN-SPECIFIC METRICS
        'obstacle_collision_count',
        'stair_climbing_performance'  # Included for terrain 3 jumping
    ]
    
    # Create plots with enhanced styling
    plot_count = 0
    pdf_path = outdir / "comprehensive_8metrics_jumping_comparison.pdf"
    
    with PdfPages(pdf_path) as pdf:
        for metric in standardized_metrics:
            # Create figure with enhanced size and styling
            fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
            fig.patch.set_facecolor('white')
            
            success = plot_numeric_comparison(
                ax, args.env_aware_label, args.foundation_only_label,
                env_aware_data, foundation_only_data, 
                "standardized", metric, modern_colors
            )
            
            if success:
                # Apply additional modern styling
                fig.suptitle("Performance Comparison Study", 
                           fontsize=16, fontweight='bold', color='#2C3E50', y=0.95)
        
                # Add subtle border around the plot
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#E0E0E0')
                    spine.set_linewidth(1)
                
                # Enhance layout
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                
                # Save individual plot with high quality
                filename = f"metric__{metric}.png"
                out_path = outdir / filename
                fig.savefig(out_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none',
                           pad_inches=0.2)
                pdf.savefig(fig, bbox_inches='tight', facecolor='white')
                
                print(f"✅ Generated: {metric}")
                    plot_count += 1
                
                plt.close(fig)
        
        # Plot summary metrics with same enhanced styling
        summary_metrics = ['collection_time', 'mean_reward', 'total_reward', 'total_steps']
        
        for metric in summary_metrics:
            fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
            fig.patch.set_facecolor('white')
                
            success = plot_numeric_comparison(
                ax, args.env_aware_label, args.foundation_only_label,
                env_aware_data, foundation_only_data, 
                "summary", metric, modern_colors
            )
                
                if success:
                fig.suptitle("Performance Comparison Study", 
                           fontsize=16, fontweight='bold', color='#2C3E50', y=0.95)
                
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('#E0E0E0')
                    spine.set_linewidth(1)
                
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                
                    filename = f"summary__{metric}.png"
                out_path = outdir / filename
                fig.savefig(out_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none',
                           pad_inches=0.2)
                pdf.savefig(fig, bbox_inches='tight', facecolor='white')
                
                print(f"✅ Generated: Summary • {metric}")
                    plot_count += 1
                
                plt.close(fig)
    
    # Enhanced final summary
    print(f"\n🎉 Generated {plot_count} comprehensive jumping comparison plots!")
    print(f"📁 Output directory: {outdir}")
    print(f"📄 Combined PDF: {pdf_path}")
    print(f"🖼️  Individual PNGs: {plot_count} files")
    print(f"\n🦘 COMPREHENSIVE JUMPING METRICS ANALYSIS (8 Standardized Metrics):")
    print(f"\n   📉 SMALLER IS BETTER METRICS:")
    print(f"      • Jump Trajectory Consistency: Movement quality (smaller = more consistent jumping)")
    print(f"      • Velocity Tracking Error: Performance measure (smaller = better tracking)")  
    print(f"      • Disturbance Resistance: Robustness measure (smaller = more robust)")
    print(f"      • Contact Termination Rate: Fall prevention (smaller = fewer falls)")
    print(f"      • Obstacle Collision Count: Safety measure (smaller = fewer collisions)")
    print(f"\n   📈 HIGHER IS BETTER METRICS:")
    print(f"      • Balance Stability Score: Body stability (higher = more stable)")
    print(f"      • Jump Coordination Smoothness: Bilateral jumping (higher = better coordination)")
    print(f"      • Stair Jumping Performance: Jumping ascent ability (higher = better)")
    print(f"\n   🦘 All terrain types now have identical 8-metric sets for fair jumping comparison!")
    print(f"\n   ✨ Enhanced with modern visual aesthetics and professional styling!")
    

if __name__ == "__main__":
    main() 