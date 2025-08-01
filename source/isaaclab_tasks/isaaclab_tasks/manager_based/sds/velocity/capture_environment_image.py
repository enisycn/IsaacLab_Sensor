#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment Image Capture for SDS Reward Generation.

This script captures a high-quality image of the SDS Enhanced environment for use in
GPT-based reward generation. The image is saved to the most recent SDS checkpoint
directory for integration with the reward generation pipeline.

Features:
- Camera positioned behind robot to analyze whole scene and environment
- Automatic SDS checkpoint detection and image saving
- Support for SDS Enhanced terrain with complex box-shaped height variations
- High-resolution image capture (1080x1920 pixels - Full HD)
- Robust error handling and proper script termination

Usage:
    ./isaaclab.sh -p capture_environment_image.py --checkpoint_dir /path/to/sds/checkpoint --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from datetime import datetime
from PIL import Image
import cv2
import threading
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Capture SDS Environment Image")
parser.add_argument("--checkpoint_dir", type=str, required=False, 
                   help="SDS checkpoint directory to save image (auto-detects if not provided)")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg


def find_latest_sds_checkpoint():
    """Find the most recent SDS checkpoint directory."""
    # Use absolute path for reliability
    sds_outputs_dir = Path("/home/enis/IsaacLab/SDS_ANONYM/outputs/sds")
    
    if not sds_outputs_dir.exists():
        print(f"❌ SDS outputs directory not found: {sds_outputs_dir}")
        return None
    
    # Look for the most recent subdirectory
    checkpoint_dirs = [d for d in sds_outputs_dir.iterdir() if d.is_dir()]
    if not checkpoint_dirs:
        print(f"❌ No checkpoint directories found in: {sds_outputs_dir}")
        return None
    
    # Get the most recent directory by modification time
    latest_dir = max(checkpoint_dirs, key=lambda d: d.stat().st_mtime)
    print(f"✅ Found latest SDS checkpoint: {latest_dir}")
    return latest_dir


@configclass
class SDSCaptureSceneCfg(InteractiveSceneCfg):
    """Scene configuration for capturing SDS environment images."""
    
    # Add G1 robot to the scene
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.98),  # Standing height for G1
            joint_pos={
                # Natural standing pose
                "left_hip_yaw_joint": 0.0,
                "left_hip_roll_joint": 0.0, 
                "left_hip_pitch_joint": -0.3,
                "left_knee_joint": 0.6,
                "left_ankle_pitch_joint": -0.3,
                "left_ankle_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.3, 
                "right_knee_joint": 0.6,
                "right_ankle_pitch_joint": -0.3,
                "right_ankle_roll_joint": 0.0,
                # Arms - using asset defaults for consistency with training
                "left_shoulder_pitch_joint": 0.35,    # Asset default (arms slightly forward)
                "right_shoulder_pitch_joint": 0.35,   # Asset default (arms slightly forward)
                "left_shoulder_roll_joint": 0.16,     # Asset default (slight outward angle)
                "right_shoulder_roll_joint": -0.16,   # Asset default (slight outward angle)
                ".*_elbow_pitch_joint": 0.87,         # Asset default (natural elbow bend)
            },
            joint_vel={},  # Empty dictionary for zero velocities
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "base_legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness=25.0,
                damping=5.0,
            ),
            "base_arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness=25.0,
                damping=5.0,
            ),
        },
    )
    
    # SDS Environment terrain configuration
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # Use simple plane (can be enhanced with complex terrain)
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        debug_vis=False,
    )
    
    # Camera for capturing environment image - positioned behind robot for scene analysis
    capture_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CaptureCamera",
        update_period=0,  # Update every step for immediate capture
        height=1200,      # Enhanced HD+ resolution for good detail/performance balance
        width=2048,       # Enhanced HD+ resolution for good detail/performance balance
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=10.0,  # Focus on robots at closer distance
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
    )


def apply_sds_enhancements(scene_cfg):
    """Apply SDS Enhanced environment features to the scene configuration."""
    try:
        # Set analysis mode for stable capture
        os.environ['SDS_ANALYSIS_MODE'] = 'true'
        
        # Try to load SDS enhanced terrain configuration
        from isaaclab_tasks.manager_based.sds.velocity.config.g1.flat_with_box_env_cfg import SDSG1FlatWithBoxEnvCfg_PLAY
        sds_env_cfg = SDSG1FlatWithBoxEnvCfg_PLAY()
        
        # Apply SDS terrain if available
        if hasattr(sds_env_cfg.scene, 'terrain'):
            scene_cfg.terrain = sds_env_cfg.scene.terrain
            print("✅ Applied SDS Enhanced terrain with complex box-shaped height variations")
            print("🏔️ TERRAIN MODE: COMPLEX stepping stones, platforms, and obstacles")
        else:
            print("⚠️ Using simple terrain (SDS enhanced terrain config not available)")
                
        print("🚀 SDS ANALYSIS MODE: Gravity DISABLED for stable image capture")
        
    except Exception as e:
        print(f"⚠️ SDS enhancements failed ({e}), using basic configuration")
    
    return scene_cfg


def capture_sds_environment_image(checkpoint_dir: Path) -> bool:
    """Capture SDS environment image using angled overhead camera view."""
    print("🎯 === SDS ENVIRONMENT IMAGE CAPTURE ===")
    
    # Updated camera constants for better robot-focused environment capture
    CAMERA_EYE = [-12.0, 0.0, 8.0]     # Medium distance (12m back, 8m up) for robot focus with environment context
    CAMERA_TARGET = [0.0, 0.0, 1.0]    # Look at robot body height (1m up from ground)
    
    # But we'll apply robot focus positioning dynamically after scene creation
    
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main simulation camera to our defined position
    sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
    print(f"📷 Main simulation camera set to: {CAMERA_EYE} → {CAMERA_TARGET}")
    
    # Create SDS scene configuration with multiple robots for overhead view
    scene_cfg = SDSCaptureSceneCfg(num_envs=16, env_spacing=8.0)  # 16 robots for high overhead view - matches training setup
    
    # Apply SDS enhancements
    scene_cfg = apply_sds_enhancements(scene_cfg)
    
    # Create the scene
    scene = InteractiveScene(scene_cfg)
    print("✅ SDS Enhanced scene created")
    
    # Add lighting for clear robot visibility
    light_cfg = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(1.0, 1.0, 1.0),
    )
    light_cfg.func("/World/skyLight", light_cfg)
    print("✅ Lighting added")
    
    # Reset simulation
    sim.reset()
    print("✅ Simulation reset")
    
    # Access camera from scene
    try:
        camera = scene["capture_camera"]
        print("✅ Camera accessed from scene")
    except KeyError:
        print("❌ Failed to access camera from scene")
        return False
    
    # Access robot for dynamic camera positioning
    try:
        robot = scene["robot"]
        robot_pos = robot.data.root_pos_w  # Get robot positions [num_envs, 3]
        
        # Calculate camera position to focus on robots (like H1 demo)
        # Use first robot as reference for camera focus
        center_robot_pos = robot_pos[0]  # Use first robot as focus point
        
        # Robot-focused camera position: positioned for optimal robot viewing with environment context
        robot_focused_eye = center_robot_pos + torch.tensor([0, -8.0, 3.0], device=sim.device)   # 8m back, 3m higher from robot
        robot_focused_target = center_robot_pos + torch.tensor([0.0, 0.0, 0.8], device=sim.device)  # Look at robot's torso height (0.8m)
        
        # Update simulation camera to focus on robots
        sim.set_camera_view(
            eye=robot_focused_eye.cpu().numpy().tolist(), 
            target=robot_focused_target.cpu().numpy().tolist()
        )
        
        # Set camera positions for all environments to focus on robot center
        num_envs = scene_cfg.num_envs
        robot_eyes = robot_focused_eye.unsqueeze(0).repeat(num_envs, 1)
        robot_targets = robot_focused_target.unsqueeze(0).repeat(num_envs, 1)
        
        camera.set_world_poses_from_view(
            eyes=robot_eyes, 
            targets=robot_targets
        )
    
        print(f"📷 Camera dynamically positioned to focus on robots at {robot_focused_eye.cpu().numpy()} → {robot_focused_target.cpu().numpy()}")
        print(f"🎯 Robot-focused view: 3m above, 8m straight back from robot - good balance of robot detail and environment context")
        
    except Exception as e:
        print(f"⚠️ Could not set robot-focused camera, using default medium view: {e}")
        # Fallback to original positioning
        num_envs = scene_cfg.num_envs
        camera_eyes = torch.tensor([CAMERA_EYE] * num_envs, device=sim.device)
        camera_targets = torch.tensor([CAMERA_TARGET] * num_envs, device=sim.device)
        
        camera.set_world_poses_from_view(
            eyes=camera_eyes, 
            targets=camera_targets
        )
        print(f"📷 Fallback camera positioned at: {CAMERA_EYE} → {CAMERA_TARGET}")
    
    print(f"🤖 Capturing image with {num_envs} G1 robots using scene-focused camera")
    print("✅ Camera positioned behind robot for whole scene analysis")
    
    # Perform simulation steps to initialize everything
    print("⏳ Initializing SDS environment and camera...")
    for _ in range(5):
        sim.step()
    
    # Update camera to ensure data is populated
    print("📷 Finalizing camera data capture...")
    camera.update(dt=sim.get_physics_dt())
    for _ in range(3):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
    
    # Capture image
    try:
        print("📸 Capturing SDS environment image...")
        print(f"🔍 Camera info: {camera}")
        
        # Get the RGB data
        rgb_data = camera.data.output["rgb"][0]  # Get first environment
        print(f"🔍 Camera data keys: {list(camera.data.output.keys())}")
        print(f"📐 RGB tensor shape: {rgb_data.shape}")
        print(f"📊 RGB tensor device: {rgb_data.device}")
        print(f"📊 RGB tensor dtype: {rgb_data.dtype}")
        
        # Convert to numpy for saving
        rgb_array = rgb_data.cpu().numpy()
        print(f"📐 Image shape: {rgb_array.shape}")
        
        # Save the full image without cropping
        print(f"📷 Using full image without cropping for better robot visibility")
        
        # Convert back to PIL Image
        rgb_image = Image.fromarray(rgb_array, mode="RGB")
        
        # Save image
        output_path = checkpoint_dir / "environment_image.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
        
        # Verify save
        if output_path.exists():
            print(f"✅ Environment image saved: {output_path}")
            # Load and check image
            img = Image.open(output_path)
            img_array = np.array(img)
            print(f"📐 Image shape: {img_array.shape}")
            print(f"📊 Image min/max values: {img_array.min()} / {img_array.max()}")
            
            print(f"\n🎉 SUCCESS: SDS Enhanced environment image captured and saved!")
            print(f"📁 Location: {output_path}")
            print(f"🔗 This image will be automatically used by SDS reward generation")
            print(f"🤖 Task descriptor system will analyze this environment image alongside demo videos")
            print(f"📷 Camera perspective: Robot-focused view (4m above, 6m diagonal) - robot centered with environment context")
            
            # 🆕 PROPER CLEANUP: Following build_simulation_context pattern
            print("\n🧹 Cleaning up simulation context...")
            
            # 🚨 ADD 10-SECOND TIMEOUT TO PREVENT HANGING
            def force_exit():
                print("⏰ 10-second timeout reached - forcing exit!")
                import os
                os._exit(0)
            
            # Start timeout timer
            timeout_timer = threading.Timer(10.0, force_exit)
            timeout_timer.start()
            
            try:
                # Stop simulation if not rendering
                if not sim.has_gui():
                    print("🛑 Stopping simulation...")
                    sim.stop()
                # Clear callbacks and instance
                print("🧽 Clearing callbacks...")
                sim.clear_all_callbacks()
                print("🔄 Clearing instance...")
                sim.clear_instance()
                print("✅ Simulation context cleaned up")
                
                # Cancel timeout if we completed successfully
                timeout_timer.cancel()
                
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
                # Cancel timeout even on cleanup failure
                timeout_timer.cancel()
            
            return True
    
    except Exception as e:
        print(f"❌ Failed to save environment image: {e}")
        
        # Clean up even on failure - ADD TIMEOUT HERE TOO
        def force_exit_on_failure():
            print("⏰ Cleanup timeout on failure - forcing exit!")
            import os
            os._exit(1)
        
        # Start timeout timer for failure cleanup
        failure_timeout_timer = threading.Timer(10.0, force_exit_on_failure)
        failure_timeout_timer.start()
        
        try:
            print("🧹 Cleaning up after failure...")
            if not sim.has_gui():
                sim.stop()
            sim.clear_all_callbacks()
            sim.clear_instance()
            print("✅ Failure cleanup completed")
            failure_timeout_timer.cancel()
        except:
            failure_timeout_timer.cancel()
            pass
            
        return False


def main():
    """Main function."""
    try:
        # Determine checkpoint directory
        if args_cli.checkpoint_dir:
            checkpoint_dir = Path(args_cli.checkpoint_dir)
        else:
            checkpoint_dir = find_latest_sds_checkpoint()
        
        if not checkpoint_dir or not checkpoint_dir.exists():
            print("❌ No valid checkpoint directory found. Please specify --checkpoint_dir or ensure SDS has been run.")
            return False
        
        print(f"🎯 Using checkpoint directory: {checkpoint_dir}")
        
        # Capture SDS environment image
        success = capture_sds_environment_image(checkpoint_dir)
        
        if success:
            print("\n🏆 === CAPTURE COMPLETE ===")
            print("✅ SDS environment image successfully captured!")
            print("🚀 Ready for GPT-based reward generation")
            return True
        else:
            print("\n❌ === CAPTURE FAILED ===")
            print("⚠️ Image capture unsuccessful")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # Run the main function
        success = main()
        if not success:
            print("❌ Image capture failed")
    except Exception as e:
        print(f"\n❌ Capture error: {e}")
    finally:
        # Clean shutdown of simulation app with timeout
        print(f"\n🔄 SHUTTING DOWN SIMULATION APP...")
        
        # 🚨 ADD FINAL 10-SECOND TIMEOUT FOR APP SHUTDOWN
        def force_exit_final():
            print("⏰ Final shutdown timeout - forcing immediate exit!")
            import os
            os._exit(0)
        
        # Start final timeout timer
        final_timeout_timer = threading.Timer(10.0, force_exit_final)
        final_timeout_timer.start()
        
        try:
            simulation_app.close() 
            print(f"✅ Simulation app closed successfully")
            final_timeout_timer.cancel()
        except Exception as e:
            print(f"⚠️ App shutdown warning: {e}")
            final_timeout_timer.cancel()
        
        # Force exit to ensure clean termination
        import sys
        print(f"🏁 SCRIPT COMPLETE - EXITING")
        sys.exit(0) 