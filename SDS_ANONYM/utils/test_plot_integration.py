#!/usr/bin/env python3

"""
Test script to demonstrate SDS plotting integration with real Isaac Lab checkpoints.
This script shows how plots will be created after successful SDS training runs.
"""

import os
import sys
import logging
import numpy as np
from plotting import create_sds_training_plots, find_latest_sds_checkpoint_dir, find_latest_isaac_lab_checkpoint_dir

def create_sample_training_data():
    """Create realistic sample training data that mimics actual SDS training metrics."""
    
    # Simulate training progression over 1000 iterations
    iterations = list(range(0, 1000, 50))  # Every 50 iterations
    n_points = len(iterations)
    
    # Create realistic training progression data
    run_log = {
        "iterations/": iterations,
        
        # Core metrics (improving over time)
        "reward": [0.1 + 0.6 * (i/n_points) + 0.05 * np.sin(i/5) + np.random.normal(0, 0.02) for i in range(n_points)],
        "episode length": [150 + 100 * (i/n_points) + 10 * np.sin(i/3) + np.random.normal(0, 5) for i in range(n_points)],
        
        # Termination metrics (decreasing failure rate)
        "termination_base_contact": [0.4 - 0.3 * (i/n_points) + 0.05 * np.sin(i/4) + np.random.normal(0, 0.02) for i in range(n_points)],
        "termination_timeout": [0.6 + 0.3 * (i/n_points) + 0.05 * np.sin(i/6) + np.random.normal(0, 0.02) for i in range(n_points)],
        
        # Training performance metrics (stabilizing)
        "value_function_loss": [0.01 - 0.008 * (i/n_points) + 0.001 * np.sin(i/8) + np.random.normal(0, 0.0005) for i in range(n_points)],
        "surrogate_loss": [-0.001 - 0.002 * (i/n_points) + 0.0005 * np.sin(i/7) + np.random.normal(0, 0.0002) for i in range(n_points)],
        "entropy_loss": [30 - 5 * (i/n_points) + 2 * np.sin(i/9) + np.random.normal(0, 0.5) for i in range(n_points)],
        "action_noise_std": [1.2 - 0.2 * (i/n_points) + 0.05 * np.sin(i/10) + np.random.normal(0, 0.01) for i in range(n_points)],
        
        # Task performance metrics (improving)
        "velocity_error_xy": [0.05 - 0.03 * (i/n_points) + 0.005 * np.sin(i/11) + np.random.normal(0, 0.002) for i in range(n_points)],
        "velocity_error_yaw": [0.08 - 0.05 * (i/n_points) + 0.008 * np.sin(i/12) + np.random.normal(0, 0.003) for i in range(n_points)],
        "curriculum_terrain_levels": [1.0 + 2.5 * (i/n_points) + 0.2 * np.sin(i/13) for i in range(n_points)],
        
        # Reward components
        "reward_sds_custom": [0.05 + 0.25 * (i/n_points) + 0.02 * np.sin(i/14) + np.random.normal(0, 0.01) for i in range(n_points)],
        
        # System performance metrics
        "computation_steps_per_sec": [45000 + 10000 * (i/n_points) + 2000 * np.sin(i/15) + np.random.normal(0, 1000) for i in range(n_points)],
        "collection_time": [1.2 - 0.3 * (i/n_points) + 0.1 * np.sin(i/16) + np.random.normal(0, 0.05) for i in range(n_points)],
        "learning_time": [0.15 + 0.05 * (i/n_points) + 0.02 * np.sin(i/17) + np.random.normal(0, 0.01) for i in range(n_points)],
        
        # ✅ NEW: Metrics from metrics.py (environmental sensing and robot stability)
        "terrain_height_variance": [0.015 + 0.01 * np.sin(i/7) + np.random.normal(0, 0.002) for i in range(n_points)],
        "robot_height_baseline": [1.24 + 0.03 * np.sin(i/6) + np.random.normal(0, 0.01) for i in range(n_points)],
        "body_orientation_deviation": [12 - 8 * (i/n_points) + 1.5 * np.sin(i/8) + np.random.normal(0, 0.5) for i in range(n_points)],
        "height_tracking_error": [0.08 - 0.06 * (i/n_points) + 0.01 * np.sin(i/9) + np.random.normal(0, 0.005) for i in range(n_points)],
        "terrain_complexity_score": [1.2 + 0.3 * np.sin(i/10) + np.random.normal(0, 0.1) for i in range(n_points)],
    }
    
    return run_log

def test_with_latest_sds_checkpoint():
    """Test plotting with the latest SDS checkpoint directory."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🔍 Testing SDS plotting integration...")
    
    # Find latest SDS checkpoint
    latest_checkpoint = find_latest_sds_checkpoint_dir()
    if latest_checkpoint:
        print(f"📁 Found latest SDS checkpoint: {latest_checkpoint}")
        
        # Create sample training data
        run_log = create_sample_training_data()
        
        # Generate plots
        print("🎨 Creating plots with sample data...")
        success = create_sds_training_plots(run_log, latest_checkpoint)
        
        if success:
            plots_dir = os.path.join(latest_checkpoint, "plots")
            print(f"✅ Plots created successfully!")
            print(f"📊 Location: {plots_dir}")
            print("\n📋 Generated plots:")
            
            if os.path.exists(plots_dir):
                plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
                for i, plot_file in enumerate(plot_files, 1):
                    print(f"   {i}. {plot_file}")
            
            return True
        else:
            print("❌ Plot creation failed")
            return False
    else:
        print("❌ No SDS checkpoint directory found")
        print("💡 Make sure you have run at least one SDS training session")
        return False

def test_with_latest_isaac_lab_checkpoint():
    """Test plotting with the latest Isaac Lab G1 Enhanced checkpoint directory."""
    print("\n🔍 Testing with Isaac Lab G1 Enhanced checkpoint...")
    
    # Find latest Isaac Lab checkpoint
    latest_checkpoint = find_latest_isaac_lab_checkpoint_dir()
    if latest_checkpoint:
        print(f"📁 Found latest Isaac Lab checkpoint: {latest_checkpoint}")
        
        # Create sample training data
        run_log = create_sample_training_data()
        
        # Generate plots
        print("🎨 Creating plots with sample data...")
        success = create_sds_training_plots(run_log, latest_checkpoint)
        
        if success:
            plots_dir = os.path.join(latest_checkpoint, "plots")
            print(f"✅ Plots created successfully!")
            print(f"📊 Location: {plots_dir}")
            
            if os.path.exists(plots_dir):
                plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
                print(f"📈 Created {len(plot_files)} plots")
            
            return True
        else:
            print("❌ Plot creation failed")
            return False
    else:
        print("❌ No Isaac Lab G1 Enhanced checkpoint directory found")
        print("💡 Make sure you have run at least one training with Isaac Lab G1 Enhanced")
        return False

def test_with_custom_directory():
    """Test plotting with a custom test directory."""
    test_dir = "/tmp/sds_checkpoint_test"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n🧪 Testing with custom directory: {test_dir}")
    
    # Create sample training data
    run_log = create_sample_training_data()
    
    # Generate plots
    success = create_sds_training_plots(run_log, test_dir)
    
    if success:
        plots_dir = os.path.join(test_dir, "plots")
        print(f"✅ Test plots created!")
        print(f"📊 Location: {plots_dir}")
        
        if os.path.exists(plots_dir):
            plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
            print(f"📈 Created {len(plot_files)} plots")
        
        return True
    else:
        print("❌ Test plot creation failed")
        return False

if __name__ == "__main__":
    print("🎨 SDS PLOTTING INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: With latest SDS checkpoint
    test1_success = test_with_latest_sds_checkpoint()
    
    # Test 2: With latest Isaac Lab checkpoint (G1 Enhanced)
    test2_success = test_with_latest_isaac_lab_checkpoint()
    
    # Test 3: With custom directory
    test3_success = test_with_custom_directory()
    
    print("\n📊 TEST RESULTS:")
    print(f"   SDS checkpoint test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   Isaac Lab checkpoint test: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"   Custom directory test: {'✅ PASS' if test3_success else '❌ FAIL'}")
    
    if test1_success or test2_success or test3_success:
        print("\n🎯 SDS plotting integration is ready!")
        print("   Plots will be automatically created after each successful training.")
        if test1_success:
            print("   ✅ SDS checkpoint integration working!")
        if test2_success:
            print("   ✅ Isaac Lab G1 Enhanced integration working!")
    else:
        print("\n⚠️ Some tests failed - check the setup.") 