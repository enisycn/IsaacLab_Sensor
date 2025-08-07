#!/usr/bin/env python3

"""
SDS Training Metrics Plotting Module

This module generates and saves plots for SDS training metrics after successful training runs.
Plots are saved in the Isaac Lab checkpoint directory under a 'plots/' subfolder.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import logging
import glob
from typing import Dict, List, Optional, Tuple

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

def setup_plot_style():
    """Setup matplotlib style for professional-looking plots."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def find_latest_sds_checkpoint_dir() -> Optional[str]:
    """
    Find the latest SDS checkpoint directory from recent training runs.
    
    Returns:
        str: Path to the latest SDS checkpoint directory, or None if not found
    """
    sds_outputs = "/home/enis/IsaacLab/SDS_ANONYM/outputs/sds"
    
    if not os.path.exists(sds_outputs):
        logging.warning(f"SDS outputs directory not found: {sds_outputs}")
        return None
    
    # Get all timestamp directories
    checkpoint_dirs = glob.glob(os.path.join(sds_outputs, "????-??-??_??-??-??"))
    
    if not checkpoint_dirs:
        logging.warning(f"No SDS checkpoint directories found in {sds_outputs}")
        return None
    
    # Sort by directory name (timestamp) and get the latest
    latest_dir = sorted(checkpoint_dirs)[-1]
    
    logging.info(f"Found latest SDS checkpoint directory: {latest_dir}")
    return latest_dir

def find_latest_isaac_lab_checkpoint_dir() -> Optional[str]:
    """
    Find the latest Isaac Lab training checkpoint directory for G1 enhanced terrain.
    
    Returns:
        str: Path to the latest Isaac Lab checkpoint directory, or None if not found
    """
    isaac_lab_logs = "/home/enis/IsaacLab/logs/rsl_rl/g1_enhanced"
    
    if not os.path.exists(isaac_lab_logs):
        logging.warning(f"Isaac Lab logs directory not found: {isaac_lab_logs}")
        return None
    
    # Get all timestamp directories
    checkpoint_dirs = glob.glob(os.path.join(isaac_lab_logs, "????-??-??_??-??-??"))
    
    if not checkpoint_dirs:
        logging.warning(f"No checkpoint directories found in {isaac_lab_logs}")
        return None
    
    # Sort by directory name (timestamp) and get the latest
    latest_dir = sorted(checkpoint_dirs)[-1]
    
    # Verify it contains model files
    model_files = glob.glob(os.path.join(latest_dir, "model_*.pt"))
    if not model_files:
        logging.warning(f"No model files found in {latest_dir}")
        return None
    
    logging.info(f"Found latest Isaac Lab checkpoint directory: {latest_dir}")
    return latest_dir

def create_plots_directory(checkpoint_dir: str) -> str:
    """
    Create plots directory in the checkpoint folder.
    
    Args:
        checkpoint_dir: Path to the training checkpoint directory
        
    Returns:
        str: Path to the created plots directory
    """
    plots_dir = os.path.join(checkpoint_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logging.info(f"Created plots directory: {plots_dir}")
    return plots_dir

def plot_metric_progression(
    run_log: Dict[str, List[float]], 
    metric_name: str, 
    plots_dir: str,
    title_override: Optional[str] = None,
    ylabel_override: Optional[str] = None,
    color: str = 'blue'
) -> bool:
    """
    Plot a single metric's progression over training iterations.
    
    Args:
        run_log: Dictionary containing training metrics
        metric_name: Name of the metric to plot
        plots_dir: Directory to save the plot
        title_override: Optional custom title for the plot
        ylabel_override: Optional custom y-axis label
        color: Color for the plot line
        
    Returns:
        bool: True if plot was created successfully, False otherwise
    """
    if metric_name not in run_log:
        logging.warning(f"Metric '{metric_name}' not found in run_log")
        return False
    
    if "iterations/" not in run_log:
        logging.warning("No iteration data found in run_log")
        return False
    
    setup_plot_style()
    
    try:
        iterations = run_log["iterations/"]
        values = run_log[metric_name]
        
        # Ensure same length
        min_length = min(len(iterations), len(values))
        iterations = iterations[:min_length]
        values = values[:min_length]
        
        if len(values) == 0:
            logging.warning(f"No data points for metric '{metric_name}'")
            return False
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(iterations, values, color=color, linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        # Set labels and title
        title = title_override or f"{metric_name.replace('_', ' ').title()} Over Training"
        ylabel = ylabel_override or metric_name.replace('_', ' ').title()
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Training Iterations', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format axes
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        final_val = values[-1]
        
        stats_text = f'Final: {final_val:.4f}\nMean: {mean_val:.4f} ± {std_val:.4f}\nRange: [{min_val:.4f}, {max_val:.4f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save the plot
        filename = f"{metric_name}_progression.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Created plot: {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating plot for '{metric_name}': {e}")
        plt.close()
        return False

def plot_multiple_metrics(
    run_log: Dict[str, List[float]], 
    metric_names: List[str], 
    plots_dir: str,
    title: str = "Multiple Metrics Progression",
    filename: str = "multiple_metrics.png"
) -> bool:
    """
    Plot multiple metrics on the same figure with subplots.
    
    Args:
        run_log: Dictionary containing training metrics
        metric_names: List of metric names to plot
        plots_dir: Directory to save the plot
        title: Title for the overall figure
        filename: Filename for the saved plot
        
    Returns:
        bool: True if plot was created successfully, False otherwise
    """
    if "iterations/" not in run_log:
        logging.warning("No iteration data found in run_log")
        return False
    
    # Filter metrics that exist in run_log
    available_metrics = [m for m in metric_names if m in run_log and len(run_log[m]) > 0]
    
    if not available_metrics:
        logging.warning(f"None of the requested metrics found: {metric_names}")
        return False
    
    setup_plot_style()
    
    try:
        iterations = run_log["iterations/"]
        n_metrics = len(available_metrics)
        
        # Create subplots
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))
        
        for i, (metric_name, color) in enumerate(zip(available_metrics, colors)):
            values = run_log[metric_name]
            
            # Ensure same length
            min_length = min(len(iterations), len(values))
            iter_subset = iterations[:min_length]
            val_subset = values[:min_length]
            
            axes[i].plot(iter_subset, val_subset, color=color, linewidth=2, marker='o', markersize=3)
            axes[i].set_title(f"{metric_name.replace('_', ' ').title()}", fontsize=12)
            axes[i].set_xlabel('Training Iterations', fontsize=10)
            axes[i].set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10)
            axes[i].grid(True, alpha=0.3, linestyle='--')
            
            # Add final value annotation
            if val_subset:
                final_val = val_subset[-1]
                axes[i].text(0.98, 0.98, f'Final: {final_val:.4f}', 
                           transform=axes[i].transAxes, fontsize=9,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # Save the plot
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Created multi-metric plot: {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Error creating multi-metric plot: {e}")
        plt.close()
        return False

def create_sds_training_plots(run_log: Dict[str, List[float]], checkpoint_dir: Optional[str] = None) -> bool:
    """
    Create all SDS training plots and save them to the SDS checkpoint directory.
    
    Args:
        run_log: Dictionary containing training metrics from construct_run_log()
        checkpoint_dir: Optional specific checkpoint directory. If None, finds latest SDS checkpoint.
        
    Returns:
        bool: True if plots were created successfully, False otherwise
    """
    # Find checkpoint directory - prioritize SDS checkpoints
    if checkpoint_dir is None:
        checkpoint_dir = find_latest_sds_checkpoint_dir()
        if checkpoint_dir is None:
            logging.error("Could not find SDS checkpoint directory for plotting")
            return False
    
    # Create plots directory
    plots_dir = create_plots_directory(checkpoint_dir)
    
    # Track successful plots
    success_count = 0
    total_plots = 0
    
    logging.info("🎨 Creating SDS training plots...")
    
    # 1. Core metrics plots (individual)
    core_metrics = [
        ("reward", "Mean Episode Reward", "Reward", "green"),
        ("episode length", "Episode Length Progression", "Episode Length", "blue"),
        ("termination_base_contact", "Base Contact Failures", "Contact Failure Rate", "red"),
    ]
    
    for metric, title, ylabel, color in core_metrics:
        total_plots += 1
        if plot_metric_progression(run_log, metric, plots_dir, title, ylabel, color):
            success_count += 1
    
    # 2. Metrics from metrics.py (individual plots)
    metrics_py_metrics = [
        ("terrain_height_variance", "Terrain Height Variance", "Variance (m²)", "orange"),
        ("robot_height_baseline", "Robot Height Above Terrain", "Height (m)", "purple"),
        ("body_orientation_deviation", "Body Orientation Deviation", "Deviation (degrees)", "brown"),
        ("height_tracking_error", "Height Tracking Error", "Error (m)", "pink"),
        ("terrain_complexity_score", "Terrain Complexity Score", "Complexity Score", "gray"),
    ]
    
    for metric, title, ylabel, color in metrics_py_metrics:
        total_plots += 1
        if plot_metric_progression(run_log, metric, plots_dir, title, ylabel, color):
            success_count += 1
    
    # 3. Training performance metrics (combined plot)
    training_metrics = ["value_function_loss", "surrogate_loss", "entropy_loss", "action_noise_std"]
    total_plots += 1
    if plot_multiple_metrics(run_log, training_metrics, plots_dir, 
                           "Training Performance Metrics", "training_performance.png"):
        success_count += 1
    
    # 4. Task performance metrics (combined plot)
    task_metrics = ["velocity_error_xy", "velocity_error_yaw", "curriculum_terrain_levels"]
    total_plots += 1
    if plot_multiple_metrics(run_log, task_metrics, plots_dir,
                           "Task Performance Metrics", "task_performance.png"):
        success_count += 1
    
    # 5. Termination analysis (combined plot)
    termination_metrics = ["termination_timeout", "termination_base_contact"]
    total_plots += 1
    if plot_multiple_metrics(run_log, termination_metrics, plots_dir,
                           "Episode Termination Analysis", "termination_analysis.png"):
        success_count += 1
    
    # 6. System performance (combined plot)
    system_metrics = ["computation_steps_per_sec", "collection_time", "learning_time"]
    total_plots += 1
    if plot_multiple_metrics(run_log, system_metrics, plots_dir,
                           "System Performance Metrics", "system_performance.png"):
        success_count += 1
    
    # Summary
    logging.info(f"📊 Plot creation complete: {success_count}/{total_plots} plots created successfully")
    logging.info(f"📁 Plots saved to: {plots_dir}")
    
    return success_count > 0

def test_plotting_with_sample_data():
    """Test the plotting functionality with sample data."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample run_log data
    iterations = list(range(0, 1000, 50))  # 0, 50, 100, ..., 950
    n_points = len(iterations)
    
    sample_run_log = {
        "iterations/": iterations,
        "reward": [0.1 + 0.8 * (i/n_points) + 0.1 * np.sin(i/5) for i in range(n_points)],
        "episode length": [100 + 200 * (i/n_points) + 20 * np.sin(i/3) for i in range(n_points)],
        "termination_base_contact": [0.5 - 0.3 * (i/n_points) + 0.1 * np.sin(i/4) for i in range(n_points)],
        "terrain_height_variance": [0.02 + 0.01 * np.sin(i/7) for i in range(n_points)],
        "robot_height_baseline": [1.25 + 0.05 * np.sin(i/6) for i in range(n_points)],
        "body_orientation_deviation": [15 - 10 * (i/n_points) + 2 * np.sin(i/8) for i in range(n_points)],
        "height_tracking_error": [0.1 - 0.08 * (i/n_points) + 0.02 * np.sin(i/9) for i in range(n_points)],
        "terrain_complexity_score": [1.0 + 0.5 * np.sin(i/10) for i in range(n_points)],
    }
    
    # Create test plots in a temporary directory
    test_dir = "/tmp/sds_test_plots"
    os.makedirs(test_dir, exist_ok=True)
    
    result = create_sds_training_plots(sample_run_log, test_dir)
    
    if result:
        print(f"✅ Test plots created successfully in {test_dir}/plots/")
    else:
        print("❌ Test plotting failed")
    
    return result

if __name__ == "__main__":
    test_plotting_with_sample_data() 