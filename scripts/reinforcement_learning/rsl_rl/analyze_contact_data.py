#!/usr/bin/env python3

"""
Script to analyze saved contact data and help debug contact detection issues.

Usage:
python analyze_contact_data.py --data_dir logs/rsl_rl/experiment_name/run_name/contact_analysis
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_contact_data(data_dir, threshold_range=(1.0, 20.0), num_thresholds=10):
    """Analyze contact data with different thresholds."""
    
    # Load data
    contact_file = os.path.join(data_dir, "contact_data.npy")
    force_file = os.path.join(data_dir, "force_data.npy")
    
    if not os.path.exists(contact_file):
        print(f"Contact data file not found: {contact_file}")
        return
    
    if not os.path.exists(force_file):
        print(f"Force data file not found: {force_file}")
        return
    
    contact_data = np.load(contact_file)
    force_data = np.load(force_file)
    
    print(f"Loaded contact data shape: {contact_data.shape}")
    print(f"Loaded force data shape: {force_data.shape}")
    
    foot_names = ['FL', 'FR', 'RL', 'RR']
    
    # Analyze force distributions
    print("\n=== FORCE ANALYSIS ===")
    for i, name in enumerate(foot_names):
        forces = force_data[:, i]
        print(f"{name} foot forces:")
        print(f"  Mean: {np.mean(forces):.2f}N")
        print(f"  Std:  {np.std(forces):.2f}N")
        print(f"  Min:  {np.min(forces):.2f}N")
        print(f"  Max:  {np.max(forces):.2f}N")
        print(f"  25th percentile: {np.percentile(forces, 25):.2f}N")
        print(f"  50th percentile: {np.percentile(forces, 50):.2f}N")
        print(f"  75th percentile: {np.percentile(forces, 75):.2f}N")
        print(f"  95th percentile: {np.percentile(forces, 95):.2f}N")
        print()
    
    # Test different thresholds
    print("=== THRESHOLD ANALYSIS ===")
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    
    print(f"{'Threshold':<12} {'FL_contact%':<12} {'FR_contact%':<12} {'RL_contact%':<12} {'RR_contact%':<12} {'Total_contact%':<15}")
    print("-" * 80)
    
    for threshold in thresholds:
        contact_binary = (force_data > threshold).astype(float)
        contact_percentages = np.mean(contact_binary, axis=0) * 100
        total_contact = np.mean(np.any(contact_binary, axis=1)) * 100
        
        print(f"{threshold:<12.1f} {contact_percentages[0]:<12.1f} {contact_percentages[1]:<12.1f} {contact_percentages[2]:<12.1f} {contact_percentages[3]:<12.1f} {total_contact:<15.1f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Force distribution histograms
    for i, name in enumerate(foot_names):
        ax = axes[i//2, i%2]
        forces = force_data[:, i]
        ax.hist(forces, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(forces), color='red', linestyle='--', label=f'Mean: {np.mean(forces):.1f}N')
        ax.axvline(np.percentile(forces, 95), color='orange', linestyle='--', label=f'95th%: {np.percentile(forces, 95):.1f}N')
        ax.set_xlabel('Force (N)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Foot Force Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "force_distribution_analysis.png"), dpi=150, bbox_inches='tight')
    print(f"\nForce distribution plot saved to: {os.path.join(data_dir, 'force_distribution_analysis.png')}")
    
    # Contact pattern analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot force data over time (sample every 10th point for readability)
    sample_indices = np.arange(0, len(force_data), max(1, len(force_data)//1000))
    sample_forces = force_data[sample_indices]
    sample_time = sample_indices
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, name in enumerate(foot_names):
        ax.plot(sample_time, sample_forces[:, i], label=f'{name} foot', color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Contact Force (N)')
    ax.set_title('Contact Forces Over Time (Sampled)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "force_time_series.png"), dpi=150, bbox_inches='tight')
    print(f"Force time series plot saved to: {os.path.join(data_dir, 'force_time_series.png')}")
    
    # Recommend threshold
    print(f"\n=== THRESHOLD RECOMMENDATIONS ===")
    
    # Calculate noise level (minimum non-zero force)
    all_forces = force_data.flatten()
    non_zero_forces = all_forces[all_forces > 0.1]
    if len(non_zero_forces) > 0:
        noise_level = np.percentile(non_zero_forces, 5)
        print(f"Estimated noise level (5th percentile of non-zero forces): {noise_level:.2f}N")
        
        # Recommend threshold as 2-3x noise level
        recommended_threshold = max(3.0, noise_level * 3)
        print(f"Recommended threshold (3x noise level): {recommended_threshold:.1f}N")
        
        # Check if this gives reasonable contact percentages
        test_contacts = (force_data > recommended_threshold).astype(float)
        test_percentages = np.mean(test_contacts, axis=0) * 100
        print(f"Contact percentages with recommended threshold:")
        for i, name in enumerate(foot_names):
            print(f"  {name}: {test_percentages[i]:.1f}%")
    
    print("\nAnalysis complete!")

def main():
    parser = argparse.ArgumentParser(description="Analyze contact data")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Directory containing contact_data.npy and force_data.npy")
    parser.add_argument("--threshold_min", type=float, default=1.0,
                       help="Minimum threshold to test (default: 1.0)")
    parser.add_argument("--threshold_max", type=float, default=20.0,
                       help="Maximum threshold to test (default: 20.0)")
    parser.add_argument("--num_thresholds", type=int, default=10,
                       help="Number of thresholds to test (default: 10)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Data directory does not exist: {args.data_dir}")
        return
    
    analyze_contact_data(args.data_dir, 
                        threshold_range=(args.threshold_min, args.threshold_max),
                        num_thresholds=args.num_thresholds)

if __name__ == "__main__":
    main() 