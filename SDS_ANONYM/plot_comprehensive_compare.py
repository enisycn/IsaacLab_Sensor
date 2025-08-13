#!/usr/bin/env python3
"""
Comprehensive Metrics Comparison Plotting Tool
==============================================

Generates publication-ready comparison plots for ALL metrics between 
environment-aware and foundation-only modes with enhanced visualizations.
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
    """Get human-readable description for the focused metrics."""
    descriptions = {
        # Core Performance Metrics (smaller is better for all)
        "height_deviation": "Height deviation from nominal (m) - Smaller is Better",
        "velocity_tracking_error": "Velocity tracking error (m/s) - Smaller is Better", 
        "disturbance_resistance": "External disturbance resistance - Smaller is Better",
        
        # Summary Metrics
        "total_steps": "Total simulation steps",
        "mean_reward": "Mean step reward",
        "total_reward": "Total collected reward",
        "collection_time": "Data collection time (s)",
    }
    return descriptions.get(metric, f"{to_title(metric)} ({to_title(category)})")


def plot_numeric_comparison(ax, env_name_a: str, env_name_b: str, data_a: Dict[str, Any], 
                          data_b: Dict[str, Any], category: str, metric: str, colors: List[str]):
    """Plot numeric metric comparison with enhanced visuals for focused metrics."""
    try:
        # Handle both old structure (category.metric) and new flat structure
        if category == "focused":
            stats_a = data_a.get(metric, {})
            stats_b = data_b.get(metric, {})
        else:
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
        
        # Bar plot with error bars
        x = [0, 1]
        means = [mean_a, mean_b]
        stds = [std_a, std_b]
        labels = [env_name_a, env_name_b]
        
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=6, alpha=0.8, 
                     edgecolor='black', linewidth=0.8)
        
        # Add value labels on bars
        for i, (bar, mean, std, count) in enumerate(zip(bars, means, stds, [count_a, count_b])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02*max(means),
                   f'{mean:.4f}±{std:.4f}\n(n={count})',
                   ha='center', va='bottom', fontsize=8, weight='bold')
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold')
        ax.set_ylabel(get_metric_description(category, metric), fontweight='bold')
        ax.set_title(f"Performance Comparison\n{to_title(metric)}", fontweight='bold', pad=20)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add difference annotation and winner indication
        diff = mean_a - mean_b
        diff_pct = (diff / max(abs(mean_b), 1e-6)) * 100
        
        # Determine winner (smaller is better for our metrics)
        if mean_a < mean_b:
            winner_text = f"{env_name_a} WINS"
            winner_color = 'lightgreen'
        elif mean_b < mean_a:
            winner_text = f"{env_name_b} WINS"
            winner_color = 'lightgreen'
        else:
            winner_text = "TIE"
            winner_color = 'lightblue'
            
        ax.text(0.5, max(means) * 0.1, f'{winner_text}\nΔ = {diff:+.4f} ({diff_pct:+.1f}%)', 
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=winner_color, alpha=0.7))
        
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
    parser = argparse.ArgumentParser(description="Focused metrics comparison with enhanced visualizations.")
    parser.add_argument("--env_aware", required=True, help="Path to env-aware results (.json or .pkl)")
    parser.add_argument("--foundation_only", required=True, help="Path to foundation-only results (.json or .pkl)")
    parser.add_argument("--outdir", default="plots_focused", help="Output directory for plots")
    parser.add_argument("--env_aware_label", default="Environment-Aware", help="Label for env-aware mode")
    parser.add_argument("--foundation_only_label", default="Foundation-Only", help="Label for foundation-only mode")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    
    # Load data
    data_a = load_results(args.env_aware)
    data_b = load_results(args.foundation_only)
    
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for env-aware, orange for foundation-only
    
    pdf_path = os.path.join(outdir, "focused_comparison.pdf")
    plot_count = 0
    
    with PdfPages(pdf_path) as pdf:
        # Summary information page
        fig = plt.figure(figsize=(8.5, 11))
        create_summary_info_plot(fig, data_a, data_b, args.env_aware_label, args.foundation_only_label)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Plot focused metrics (3 key metrics)
        focused_metrics = ['height_deviation', 'velocity_tracking_error', 'disturbance_resistance']
        
        metrics_a = data_a.get("comprehensive_metrics", {})
        metrics_b = data_b.get("comprehensive_metrics", {})
        
        # Plot the 3 focused metrics
        for metric in focused_metrics:
            if metric in metrics_a and metric in metrics_b:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                success = plot_numeric_comparison(ax, args.env_aware_label, args.foundation_only_label,
                                                metrics_a, metrics_b, "focused", metric, colors)
                
                if success:
                    # Save individual plot
                    filename = f"focused__{metric}.png"
                    out_path = os.path.join(outdir, filename)
                    fig.tight_layout()
                    fig.savefig(out_path, dpi=200, bbox_inches='tight')
                    pdf.savefig(fig, bbox_inches='tight')
                    plot_count += 1
                    print(f"✅ Generated: {metric}")
                else:
                    print(f"❌ Failed to plot: {metric}")
                
                plt.close(fig)
            else:
                print(f"❌ Missing data for: {metric}")
        
        # Plot summary metrics
        summary_a = data_a.get("summary_metrics", {})
        summary_b = data_b.get("summary_metrics", {})
        
        for metric in sorted(set(summary_a.keys()) & set(summary_b.keys())):
            if isinstance(summary_a[metric], (int, float)) and isinstance(summary_b[metric], (int, float)):
                fig, ax = plt.subplots(figsize=(6, 4.5))
                
                success = plot_summary_comparison(ax, args.env_aware_label, args.foundation_only_label,
                                                data_a, data_b, metric, colors)
                
                if success:
                    # Save individual plot
                    filename = f"summary__{metric}.png"
                    out_path = os.path.join(outdir, filename)
                    fig.tight_layout()
                    fig.savefig(out_path, dpi=200, bbox_inches='tight')
                    pdf.savefig(fig, bbox_inches='tight')
                    plot_count += 1
                    print(f"✅ Generated: Summary • {metric}")
                
                plt.close(fig)
    
    print(f"\n🎉 Generated {plot_count} focused comparison plots!")
    print(f"📁 Output directory: {outdir}")
    print(f"📄 Combined PDF: {pdf_path}")
    print(f"🖼️  Individual PNGs: {plot_count} files")
    print(f"\n🎯 FOCUSED METRICS ANALYSIS:")
    print(f"   • Height Deviation: Stability measure (smaller = more stable)")
    print(f"   • Velocity Tracking Error: Performance measure (smaller = better tracking)")  
    print(f"   • Disturbance Resistance: Robustness measure (smaller = more robust)")
    print(f"   • All metrics: SMALLER IS BETTER for performance comparison")
    

if __name__ == "__main__":
    main() 