"""
Isaac Lab Environment Reference for SDS Reward Function Generation

This file describes the Isaac Lab environment structure for GPT to generate compatible reward functions.
The environment uses Isaac Lab's ManagerBasedRLEnv with Unitree G1 humanoid robot and enhanced
environmental sensing capabilities for adaptive locomotion.

⚠️ CRITICAL SENSOR ACCESS WARNING:
Isaac Lab RayCaster sensors (height_scanner, lidar) ONLY have these attributes:
- data.ray_hits_w: Hit points in world frame [num_envs, num_rays, 3]
- data.pos_w: Sensor position [num_envs, 3] 
- data.quat_w: Sensor orientation [num_envs, 4]

❌ DO NOT ACCESS: .distances, .height_measurements, .range_measurements (THESE DON'T EXIST!)
✅ USE ISAAC LAB STANDARD: Direct sensor access for reward functions with raw measurements

Environment Configuration:
- Robot: Unitree G1 Humanoid (23-DOF full-body control, arms fixed at default poses)
- Physics: Fully simulated rigid body dynamics with friction
- Control: Joint position targets with PD control
- Observations: Proprioception + height scanning + LiDAR + contact sensing  
- Action Space: Continuous joint position targets [num_envs, 23]
- Observation Space: Mixed proprioception + environmental sensing (≈800 dims, depends on config)
- Command Space: 2D velocity + angular velocity [num_envs, 3]

Enhanced Sensor Integration (Updated Specifications):
- Contact Forces: Body contact detection for gait analysis
- Height Scanner: 567-ray grid (27×21) for terrain detection - 2.0×1.5m coverage, 7.5cm resolution, 3m range
- LiDAR Range: 152-ray array (8 channels × 19 horizontal) for obstacle detection - 180° FOV, 5m range  
- Environmental Analysis: Real-time terrain feature classification with G1 baseline (0.209m)
- Terrain Features: Mixed gap sizes (15-25cm depth), no infinite readings, controlled difficulty

This environment implements environmental sensing for adaptive locomotion behaviors.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class SDSIsaacLabEnvironment:
    """
    Isaac Lab Manager-Based RL Environment for SDS Humanoid Locomotion with Environmental Sensing
    
    Environment Details:
    - Robot: Unitree G1 humanoid (37 DOF total, 23 DOF controlled for complete humanoid control)
    - Action Space: 23 DOF full body joints for comprehensive humanoid locomotion
    - Task: Velocity tracking locomotion with environmental sensing and adaptive behavior
    - Framework: Isaac Lab ManagerBasedRLEnv
    - Control: 50Hz (20ms timestep, 4x decimation from 200Hz physics)
    
    Enhanced Environmental Sensing (Updated):
    - Height Scanner: 567-ray grid (27×21) terrain analysis - 2.0×1.5m coverage, 7.5cm resolution, 3m range
    - LiDAR Sensor: 152-ray array (8×19) obstacle detection - 180° FOV, 5m range
    - G1 Baseline: 0.209m on flat terrain (sensor_height - terrain_z - 0.5m offset)
    - Terrain Classification: ±0.07m thresholds for obstacles/gaps detection
    - Environmental Observations: Sanitized sensor data for adaptive reward functions
    - Real-time Integration: Sensor data available for terrain-aware locomotion rewards
    
    Controlled Joints for Locomotion (23 DOF):
    - Legs: 12 DOF (hip yaw/roll/pitch, knee, ankle pitch/roll per leg)
    - Torso: 1 DOF (torso_joint)
    - Arms: 10 DOF (shoulder pitch/roll/yaw, elbow pitch/roll per arm) - FIXED at default poses
    
    Fixed Joints (24 DOF total):
    - Hand Fingers: 14 DOF maintained at default poses (zero/one/two/three/four/five/six_joint per hand)
    - Arm Joints: 10 DOF fixed for locomotion focus (arms do not contribute to movement)
    
    Sensor Integration (Updated):
    - Contact Forces: Body contact detection for gait analysis
    - Height Scanner: 567-ray grid (27×21) for terrain detection - 2.0×1.5m coverage, 7.5cm resolution, 3m range
    - LiDAR Range: 152-ray array (8×19) for obstacle detection - 180° FOV, 5m range
    - Environmental Analysis: Real-time terrain feature classification with G1 baseline (0.209m)
    - Data Sanitization: Mandatory filtering of NaN/Inf values for sensor reliability
    """
    
    def __init__(self):
        """Environment initialization."""
        self.num_envs = 4096  # Parallel environments
        self.device = "cuda"  # Device for tensors
        
        # Main components
        self.scene = {
            "robot": self._robot_interface(),
            "sensors": self._sensor_interface()
        }
        # ✅ CORRECT ACCESS PATTERNS from prompts:
        # robot = env.scene["robot"]  # Dictionary-style access (NOT env.scene.robot)
        # contact_sensor = env.scene.sensors["contact_forces"]  
        # height_scanner = env.scene.sensors["height_scanner"]
        # lidar = env.scene.sensors["lidar"] 
        # imu = env.scene.sensors["imu"]
        
        self.command_manager = self._command_interface()
        
    def _robot_interface(self):
        """Isaac Lab robot interface - Unitree G1 data access."""
        class Robot:
            def __init__(self):
                self.data = self._robot_data()
            
            def _robot_data(self):
                """Robot state data in Isaac Lab ArticulationData format."""
                class RobotData:
                    # ✅ CORRECT Isaac Lab ArticulationData properties
                    
                    # Root/Base state (torso body) - [num_instances, dimensions]
                    root_pos_w = None      # [num_envs, 3] Position in world frame
                    root_quat_w = None     # [num_envs, 4] Quaternion (w,x,y,z) in world frame
                    root_lin_vel_b = None  # [num_envs, 3] Linear velocity in BODY frame  
                    root_ang_vel_b = None  # [num_envs, 3] Angular velocity in BODY frame
                    root_state_w = None    # [num_envs, 13] Combined [pos, quat, lin_vel, ang_vel]
                    
                    # Joint state (37 joints for G1 EDU U4)
                    # CONTROLLED JOINTS (23 DOF): 12 legs + 1 torso + 10 arms (all except hand fingers)
                    joint_pos = None       # [num_envs, 37] Joint positions (rad) - ALL joints
                    joint_vel = None       # [num_envs, 37] Joint velocities (rad/s) - ALL joints
                    joint_acc = None       # [num_envs, 37] Joint accelerations (rad/s²) - ALL joints
                    
                    # Joint targets (set by actions)
                    joint_pos_target = None    # [num_envs, 37] Position targets
                    joint_vel_target = None    # [num_envs, 37] Velocity targets  
                    joint_effort_target = None # [num_envs, 37] Effort targets
                    
                    # Joint limits and properties
                    joint_pos_limits = None    # [num_envs, 37, 2] Joint position limits [lower, upper]
                    joint_vel_limits = None    # [num_envs, 37] Joint velocity limits
                    soft_joint_pos_limits = None # [num_envs, 37, 2] Soft position limits
                    
                    # Body states (all robot bodies - links)
                    body_pos_w = None      # [num_envs, num_bodies, 3] Body positions
                    body_quat_w = None     # [num_envs, num_bodies, 4] Body orientations
                    body_lin_vel_w = None  # [num_envs, num_bodies, 3] Body linear velocities
                    body_ang_vel_w = None  # [num_envs, num_bodies, 3] Body angular velocities
                
                return RobotData()
        
        return Robot()
    
    def _sensor_interface(self):
        """Isaac Lab sensor interface with enhanced environmental sensing."""
        class Sensors:
            def __init__(self):
                self.contact_forces = self._contact_sensor()
                self.height_scanner = self._height_scanner()
                self.lidar = self._lidar_sensor()
                self.imu = self._imu_sensor()  # ✅ ADDED: IMU sensor mentioned in prompts
                
            def _contact_sensor(self):
                """Contact force sensor for all robot bodies."""
                class ContactSensor:
                    def __init__(self):
                        self.data = self._contact_data()
                        # ✅ CORRECT Isaac Lab ContactSensor properties
                        self.num_instances = 4096  # Number of environments
                        self.device = "cuda"       # Device for tensors
                        self.body_names = [        # Body names in contact sensor
                            "left_ankle_roll_link", 
                            "right_ankle_roll_link",
                            "torso_link"  # Example additional bodies
                        ]
                    
                    def find_bodies(self, name_keys: str, preserve_order: bool = False):
                        """Find bodies in the contact sensor based on regex pattern.
                        
                        ✅ CRITICAL: This method is consistently used in all SDS prompts!
                        
                        Args:
                            name_keys: Regular expression to match body names (e.g., ".*_ankle_roll_link")
                            preserve_order: Whether to preserve order. Defaults to False.
                            
                        Returns:
                            tuple: (body_indices: list[int], body_names: list[str])
                            
                        Example from prompts:
                            foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
                            foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=env.device)
                        """
                        # Mock implementation for documentation - actual Isaac Lab handles this
                        return ([0, 1], ["left_ankle_roll_link", "right_ankle_roll_link"])
                    
                    def _contact_data(self):
                        """Contact sensor data in Isaac Lab ContactSensorData format."""
                        class ContactData:
                            # ✅ CORRECT Isaac Lab ContactSensorData attributes
                            
                            # Sensor pose (if track_pose=True)
                            pos_w = None                  # [N, 3] | None - Sensor position in world frame
                            quat_w = None                 # [N, 4] | None - Sensor orientation in world frame
                            
                            # Contact forces 
                            net_forces_w = None           # [N, B, 3] | None - Net contact forces per body
                            net_forces_w_history = None   # [N, T, B, 3] | None - Force history (if history enabled)
                            force_matrix_w = None         # [N, B, M, 3] | None - Filtered contact forces
                            
                            # Air time tracking (if track_air_time=True)
                            last_air_time = None          # [N, B] | None - Time in air before last contact
                            current_air_time = None       # [N, B] | None - Current time in air
                            last_contact_time = None      # [N, B] | None - Time in contact before last detach
                            current_contact_time = None   # [N, B] | None - Current time in contact
                            
                            # ❌ THESE LEGACY ATTRIBUTES DON'T EXIST IN ISAAC LAB CONTACTSENSORDATA:
                            # current_contact_time = None   # This is now handled by ContactSensorData directly
                        
                        return ContactData()
                
                return ContactSensor()
            
            def _height_scanner(self):
                """Height scanner for terrain analysis - Grid-based height detection."""
                class HeightScanner:
                    def __init__(self):
                        self.data = self._height_scanner_data()
                    
                    def _height_scanner_data(self):
                        """Height scanner data for gap and terrain detection."""
                        class HeightScannerData:
                            # ✅ CORRECT Isaac Lab RayCaster attributes (ONLY THESE EXIST!)
                            ray_hits_w = None             # [num_envs, 567, 3] Hit points in world frame (27×21 grid)
                            pos_w = None                  # [num_envs, 3] Scanner position in world frame
                            quat_w = None                 # [num_envs, 4] Scanner orientation in world frame
                            
                            # ❌ THESE ATTRIBUTES DO NOT EXIST - WILL CAUSE TRAINING CRASHES!
                            # distances = None              # AttributeError: 'RayCasterData' object has no attribute 'distances'
                            # height_measurements = None    # AttributeError: 'RayCasterData' object has no attribute 'height_measurements'
                            
                            # ✅ CORRECT ACCESS METHOD: Isaac Lab standard formula with G1 baseline
                            # height_measurements = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5
                            # G1 baseline: 0.209m on flat terrain | Obstacles: < 0.139m | Gaps: > 0.279m
                            # MANDATORY: Sanitize with torch.where(torch.isfinite(height_measurements), height_measurements, torch.zeros_like(height_measurements))
                        
                        return HeightScannerData()
                
                return HeightScanner()
            
            def _lidar_sensor(self):
                """LiDAR sensor for front-hemisphere (~180°) environmental awareness."""
                class LiDARSensor:
                    def __init__(self):
                        self.data = self._lidar_data()
                    
                    def _lidar_data(self):
                        """LiDAR data for obstacle detection and navigation."""
                        class LiDARData:
                            # ✅ CORRECT Isaac Lab RayCaster attributes (ONLY THESE EXIST!)
                            ray_hits_w = None             # [num_envs, 152, 3] Hit points in world frame (8×19 channels)
                            pos_w = None                  # [num_envs, 3] LiDAR position in world frame
                            quat_w = None                 # [num_envs, 4] LiDAR orientation in world frame
                            
                            # ❌ THESE ATTRIBUTES DO NOT EXIST - WILL CAUSE TRAINING CRASHES!
                            # distances = None              # AttributeError: 'RayCasterData' object has no attribute 'distances'
                            # range_measurements = None     # AttributeError: 'RayCasterData' object has no attribute 'range_measurements'
                            
                            # ✅ CORRECT ACCESS METHOD: Isaac Lab standard with 180° FOV, 5m range
                            # lidar_distances = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
                            # Range: 0.1-5.0m | Close obstacles: < 2.0m | Clear path: > 3.0m
                            # MANDATORY: Sanitize with torch.where(torch.isfinite(lidar_distances), lidar_distances, torch.ones_like(lidar_distances) * 5.0)
                        
                        return LiDARData()
                
                return LiDARSensor()
        
            def _imu_sensor(self):
                """IMU sensor for orientation and acceleration measurements - mentioned in prompts."""
                class ImuSensor:
                    def __init__(self):
                        self.data = self._imu_data()
                        
                    def _imu_data(self):
                        """IMU sensor data as mentioned in the prompts."""
                        class ImuData:
                            # ✅ CORRECT IMU sensor data from prompts (lines 772-775)
                            
                            # Pose tracking
                            pos_w = None                  # [num_envs, 3] - World position
                            quat_w = None                 # [num_envs, 4] - World orientation quaternion (w,x,y,z)
                            
                            # Body-frame velocities 
                            lin_vel_b = None              # [num_envs, 3] - Linear velocity in body frame
                            ang_vel_b = None              # [num_envs, 3] - Angular velocity in body frame
                            
                            # Body-frame accelerations
                            lin_acc_b = None              # [num_envs, 3] - Linear acceleration in body frame  
                            ang_acc_b = None              # [num_envs, 3] - Angular acceleration in body frame
                            
                            # Configuration from prompts:
                            # - Attached to: "/World/envs/env_.*/Robot/torso_link"
                            # - Update period: 0.02 (50 Hz)
                            # - Gravity bias: [0.0, 0.0, 9.81]
                            
                            # Usage examples from prompts:
                            # - Head-pitch angles: Extract from quat_w using quaternion-to-euler conversion
                            # - Yaw angular velocity: Use ang_vel_b[:, 2] (Z-axis in body frame)
                            # - Spatial awareness: Use pos_w and quat_w for rotation monitoring
                        
                        return ImuData()
                
                return ImuSensor()
        
        return Sensors()
    
    def _command_interface(self):
        """Isaac Lab command manager interface."""
        class CommandManager:
            def get_command(self, command_name: str):
                """Get velocity command using actual Isaac Lab CommandManager.get_command() method.
                
                Args:
                    command_name: Name of the command term (e.g., "base_velocity")
                    
                Returns:
                    torch.Tensor: Command tensor for the specified term
                    
                ✅ CORRECT ACCESS PATTERN:
                    velocity_cmd = env.command_manager.get_command("base_velocity")  # [num_envs, 3] -> (vx, vy, omega_z)
                """
                if command_name == "base_velocity":
                    # Returns [num_envs, 3] -> (vx, vy, omega_z)
                    return None  # Tensor with desired velocities
                return None
        
        return CommandManager()

# Unitree G1 Humanoid Robot Structure
G1_BODY_NAMES = {
    "base": "torso_link",          # Main chassis/torso body
    "feet": [                      # Foot bodies (end effectors)
        "left_ankle_roll_link",    # Left foot contact point
        "right_ankle_roll_link"    # Right foot contact point
    ],
    "legs": {                      # Leg segment bodies
        "thighs": ["left_thigh_link", "right_thigh_link"],
        "shins": ["left_shin_link", "right_shin_link"],
        "hips": ["left_hip_link", "right_hip_link"]
    },
    "arms": {                      # Arm segment bodies
        "upper_arms": ["left_upper_arm_link", "right_upper_arm_link"],
        "forearms": ["left_forearm_link", "right_forearm_link"],
        "hands": ["left_hand_link", "right_hand_link"]
    }
}

# Joint Configuration (37 DOF for G1 EDU U4 - VERIFIED FROM ISAAC LAB SOURCE)
G1_JOINT_NAMES = [
    # Leg and Torso joints (13 DOF - 12 leg + 1 torso, part of 23 controlled DOF) - VERIFIED FROM ISAAC LAB
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", 
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    
    # Arm joints (10 DOF - 5 per arm) - VERIFIED FROM ISAAC LAB
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
    
    # Hand joints (14 DOF - 7 per hand) - VERIFIED FROM ISAAC LAB
    "left_zero_joint", "left_one_joint", "left_two_joint", "left_three_joint", 
    "left_four_joint", "left_five_joint", "left_six_joint",
    "right_zero_joint", "right_one_joint", "right_two_joint", "right_three_joint",
    "right_four_joint", "right_five_joint", "right_six_joint",
]

# Robot specifications based on Isaac Lab implementation and official specs
G1_SPECS = {
    "total_joints": 37,             # VERIFIED: Isaac Lab G1 EDU U4 with dexterous hands (full robot)
    "action_space": 23,             # LOCOMOTION FOCUSED: Controlled joints for locomotion (12 legs + 1 torso) + 10 arms fixed
    "controlled_joints": {
        "legs": 12,                 # Controlled: All leg joints for locomotion
        "torso": 1,                 # Controlled: Torso joint for posture
        "arms": 0,                  # Arms: 10 DOF FIXED at default poses (not controlled)
    },
    "fixed_joints": {
        "hand_fingers": 14,         # Fixed: Hand finger joints at default poses
        "arms": 10,                 # Fixed: All arm joints at default poses for locomotion focus
    },
    "nominal_height": 0.74,         # VERIFIED: Isaac Lab init position z=0.74m
    "contact_threshold": 50.0,      # CORRECTED: Appropriate for 35kg humanoid robot
    "foot_bodies": ["left_ankle_roll_link", "right_ankle_roll_link"],  # FIXED: Contact detection bodies (links not joints)
}

# G1 Body parts for reference (UPDATED to Isaac Lab G1 EDU U4)
G1_BODY_PARTS = {
    "torso": ["torso_joint"],
    "legs": [
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", 
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", 
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
    ],
    "arms": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint", "left_elbow_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint", "right_elbow_roll_joint"
    ],
    "hands": [
        "left_five_joint", "left_three_joint", "left_six_joint", "left_four_joint",
        "left_zero_joint", "left_one_joint", "left_two_joint",
        "right_five_joint", "right_three_joint", "right_six_joint", "right_four_joint",
        "right_zero_joint", "right_one_joint", "right_two_joint"
    ],
    "feet": ["left_ankle_roll_link", "right_ankle_roll_link"]  # Contact points (links not joints)
}

def get_velocity_tracking_error(robot_data, commands):
    """Calculate error between desired and actual velocity."""
    # robot_data.root_lin_vel_b is [num_envs, 3] 
    # commands is [num_envs, 3] (vx, vy, omega_z)
    lin_vel_error = torch.norm(robot_data.root_lin_vel_b[:, :2] - commands[:, :2], dim=1)
    ang_vel_error = torch.abs(robot_data.root_ang_vel_b[:, 2] - commands[:, 2])
    return lin_vel_error, ang_vel_error

def get_foot_contacts(contact_sensor, threshold=50.0):
    """Get binary foot contacts for G1 humanoid using CORRECT Isaac Lab ContactSensor API."""
    # ✅ CORRECT Isaac Lab ContactSensor access pattern from prompts
    contact_forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3] | None
    
    if contact_forces is None:
        # No contact data available - return zeros
        num_envs = contact_sensor.num_instances
        return torch.zeros((num_envs, 2), device=contact_sensor.device)
    
    # ✅ CORRECT: Use find_bodies method EXACTLY as shown in all prompts
    foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
    foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=contact_sensor.device)
    
    if len(foot_ids) == 0:
        # Fallback if no ankle links found
        foot_ids = torch.tensor([0, min(1, contact_forces.shape[1] - 1)], device=contact_sensor.device)
    elif len(foot_ids) == 1:
        # Duplicate if only one found
        foot_ids = torch.tensor([foot_ids[0], foot_ids[0]], device=contact_sensor.device)
    elif len(foot_ids) > 2:
        # Take first 2
        foot_ids = foot_ids[:2]
    
    # Extract foot forces and compute contact
    foot_forces = contact_forces[:, foot_ids, :]  # [num_envs, 2, 3]
    force_magnitudes = foot_forces.norm(dim=-1)  # [num_envs, 2]
    foot_contacts = (force_magnitudes > threshold).float()  # [num_envs, 2]
    
    return foot_contacts  # Returns [left_foot_contact, right_foot_contact]

def get_foot_contact_analysis(contact_sensor, threshold=50.0):
    """Enhanced foot contact analysis for G1 humanoid gait patterns."""
    foot_contacts = get_foot_contacts(contact_sensor, threshold)
    
    # Humanoid gait phase detection (CORRECTED for bipedal locomotion)
    left_contact = foot_contacts[:, 0]  # Left foot
    right_contact = foot_contacts[:, 1]  # Right foot
    
    # Bipedal gait phases (FIXED from quadruped)
    double_support = (left_contact > 0.5) & (right_contact > 0.5)  # Both feet down
    single_support_left = (left_contact > 0.5) & (right_contact < 0.5)  # Only left down
    single_support_right = (left_contact < 0.5) & (right_contact > 0.5)  # Only right down  
    flight_phase = (left_contact < 0.5) & (right_contact < 0.5)  # Both feet up (running)
    
    return {
        "foot_contacts": foot_contacts,
        "gait_phases": {
            "double_support": double_support,
            "single_support_left": single_support_left, 
            "single_support_right": single_support_right,
            "flight_phase": flight_phase
        },
        "contact_count": foot_contacts.sum(dim=-1)  # Number of feet in contact
    }

def extract_foot_contacts(env, contact_threshold=50.0):
    """
    Extract foot contact information for G1 humanoid reward functions.
    
    This is the main function used by SDS reward functions to get contact data.
    
    Args:
        env: Isaac Lab environment instance
        contact_threshold: Force threshold for contact detection (N)
        
    Returns:
        dict: Contact analysis data including binary contacts, forces, and gait metrics
    """
    return get_foot_contact_analysis(env.scene.sensors["contact_forces"], contact_threshold) 

# ========== ENVIRONMENTAL SENSOR INTEGRATION ==========

def get_height_scan_data(self, env):
    """
    Get height scanner data for terrain analysis with G1 baseline.
    
    ✅ ISAAC LAB STANDARD: Raw sensor access for reward functions
    Height measurements in METERS with G1 robot baseline (0.209m on flat terrain)!
        
    Returns:
        torch.Tensor: Height measurements relative to sensor [num_envs, 567]
                         G1 baseline: 0.209m | Obstacles: < 0.139m | Gaps: > 0.279m
                         Physical range: [-0.5m to +3.0m] relative to sensor (clipped)
    """
    # ✅ CORRECT: Isaac Lab standard raw sensor access
    height_sensor = env.scene.sensors["height_scanner"]
    
    # Raw height calculation: sensor_height - hit_point_height - offset
    # This gives terrain height relative to sensor position in METERS
    height_measurements = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2] - 0.5
    
    # ✅ MANDATORY: Sanitize sensor data to prevent NaN/Inf crashes
    height_measurements = torch.where(torch.isfinite(height_measurements), height_measurements, torch.zeros_like(height_measurements))
    
    return height_measurements  # [num_envs, 567] in meters


def get_lidar_range_data(self, env):
    """
    Get LiDAR range data for obstacle detection with updated specifications.
    
    ✅ ISAAC LAB STANDARD: Raw sensor access for reward functions  
    Distance measurements in METERS - 180° FOV, 5m range!
        
    Returns:
        torch.Tensor: Distance measurements [num_envs, 152]
                         Physical range: [0.1m to 5.0m] actual distances (clipped)
    """
    # ✅ CORRECT: Isaac Lab standard raw sensor access
    lidar_sensor = env.scene.sensors["lidar"]
    
    # Raw distance calculation: norm of (hit_points - sensor_position)
    # This gives actual distances to obstacles in METERS
    lidar_distances = torch.norm(
        lidar_sensor.data.ray_hits_w - lidar_sensor.data.pos_w.unsqueeze(1), 
        dim=-1
    )
    
    # ✅ MANDATORY: Sanitize sensor data to prevent NaN/Inf crashes
    lidar_distances = torch.where(torch.isfinite(lidar_distances), lidar_distances, torch.ones_like(lidar_distances) * 5.0)
    
    return lidar_distances  # [num_envs, 152] in meters

def analyze_terrain_features(env, baseline=0.209, threshold=0.07):
    """
    Analyze terrain features using G1 baseline and relative thresholds.
    
    ✅ ISAAC LAB STANDARD: Uses raw sensor measurements with G1 robot baseline
    G1 baseline: 0.209m on flat terrain | ±0.07m thresholds for classification!
    
    Args:
        env: Isaac Lab environment instance
        baseline: G1 robot baseline height on flat terrain (default: 0.209m)
        threshold: Relative threshold for obstacle/gap detection (default: 0.07m)
        
    Returns:
        dict: Comprehensive terrain analysis with G1-specific measurements
    """
    # ✅ CORRECT: Isaac Lab standard raw sensor access
    height_sensor = env.scene.sensors["height_scanner"]
    lidar_sensor = env.scene.sensors["lidar"]
    
    # Raw height measurements: sensor_height - hit_point_height - offset (in meters)
    height_measurements = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2] - 0.5
    
    # ✅ MANDATORY: Sanitize sensor data to prevent NaN/Inf crashes
    height_measurements = torch.where(torch.isfinite(height_measurements), height_measurements, torch.zeros_like(height_measurements))
    
    # Raw distance measurements: actual distances to obstacles (in meters)
    lidar_distances = torch.norm(
        lidar_sensor.data.ray_hits_w - lidar_sensor.data.pos_w.unsqueeze(1), 
        dim=-1
    )
    
    # ✅ MANDATORY: Sanitize sensor data to prevent NaN/Inf crashes
    lidar_distances = torch.where(torch.isfinite(lidar_distances), lidar_distances, torch.ones_like(lidar_distances) * 5.0)
    
    # === GAP DETECTION (using G1 baseline + threshold) ===
    # Gaps: readings higher than baseline + threshold
    gap_threshold_value = baseline + threshold  # 0.209 + 0.07 = 0.279m
    gap_mask = height_measurements > gap_threshold_value  # [num_envs, 567] - gaps deeper than threshold
    gap_count = gap_mask.sum(dim=-1)  # [num_envs] - total gap points per environment
    gap_density = gap_count.float() / height_measurements.shape[1]  # [num_envs] - fraction of gaps
    
    # === OBSTACLE DETECTION (using G1 baseline - threshold) ===  
    # Obstacles: readings lower than baseline - threshold
    obstacle_threshold_value = baseline - threshold  # 0.209 - 0.07 = 0.139m
    obstacle_mask = height_measurements < obstacle_threshold_value  # [num_envs, 567] - obstacles above threshold
    obstacle_count = obstacle_mask.sum(dim=-1)  # [num_envs] - total obstacle points per environment
    obstacle_density = obstacle_count.float() / height_measurements.shape[1]  # [num_envs] - fraction of obstacles
    
    # === FORWARD SAFETY ANALYSIS ===
    # Height scanner: front center section (27×21 = 567 rays grid layout)
    forward_height_indices = slice(200, 350)  # Center-front area of 567-ray grid
    forward_gaps = gap_mask[:, forward_height_indices].any(dim=-1)  # [num_envs] - gaps ahead
    
    # LiDAR: front-facing rays for obstacle proximity (8×19 = 152 rays)
    forward_lidar_indices = slice(60, 92)   # Center 32 rays of 152-ray array  
    close_obstacle_threshold = 2.0  # 2 meters - close obstacle distance
    forward_close_obstacles = (lidar_distances[:, forward_lidar_indices] < close_obstacle_threshold).any(dim=-1)
    
    # Safety assessment for navigation
    safe_forward_path = ~forward_gaps & ~forward_close_obstacles  # [num_envs] - clear forward path
    
    return {
        "height_measurements": height_measurements,  # [num_envs, 567] in meters
        "lidar_distances": lidar_distances,  # [num_envs, 152] in meters
        "gap_detection": {
            "gap_mask": gap_mask,
            "gap_count": gap_count, 
            "gap_density": gap_density,
            "forward_gaps": forward_gaps
        },
        "obstacle_detection": {
            "obstacle_mask": obstacle_mask,
            "obstacle_count": obstacle_count,
            "obstacle_density": obstacle_density, 
            "forward_close_obstacles": forward_close_obstacles
        },
        "navigation_safety": {
            "safe_forward_path": safe_forward_path,
            "terrain_complexity": gap_density + obstacle_density  # Combined challenge metric
        }
    }

def get_environmental_reward_components(env):
    """
    Generate environmental reward components using G1 baseline and updated sensor specs.
    
    ✅ ISAAC LAB STANDARD: Uses raw sensor measurements with G1 robot baseline
    G1 baseline: 0.209m on flat terrain | ±0.07m thresholds for classification!
    
    Args:
        env: Isaac Lab environment instance
        
    Returns:
        dict: Environmental reward components for adaptive locomotion with G1 baseline
    """
    # ✅ CORRECT: Isaac Lab standard raw sensor access
    height_sensor = env.scene.sensors["height_scanner"]
    lidar_sensor = env.scene.sensors["lidar"]
    
    # Raw height measurements: sensor_height - hit_point_height - offset (in meters)
    height_measurements = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2] - 0.5
    
    # ✅ MANDATORY: Sanitize sensor data to prevent NaN/Inf crashes
    height_measurements = torch.where(torch.isfinite(height_measurements), height_measurements, torch.zeros_like(height_measurements))
    
    # Raw distance measurements: actual distances to obstacles (in meters)
    lidar_distances = torch.norm(
        lidar_sensor.data.ray_hits_w - lidar_sensor.data.pos_w.unsqueeze(1), 
        dim=-1
    )
    
    # ✅ MANDATORY: Sanitize sensor data to prevent NaN/Inf crashes
    lidar_distances = torch.where(torch.isfinite(lidar_distances), lidar_distances, torch.ones_like(lidar_distances) * 5.0)
    
    # === GAP AVOIDANCE (using G1 baseline + threshold) ===
    # Detect gaps using G1 baseline: readings above baseline + threshold indicate gaps
    baseline = 0.209  # G1 robot baseline on flat terrain
    threshold = 0.07  # Standard threshold for gap/obstacle classification
    gap_threshold_value = baseline + threshold  # 0.279m = gap detection threshold
    gap_detected = torch.any(height_measurements > gap_threshold_value, dim=1)  # [num_envs] bool
    gap_avoidance_reward = torch.where(gap_detected, -1.0, 0.1)  # Penalty for gaps, bonus for clear
    
    # === OBSTACLE AVOIDANCE (using physical measurements) ===
    # Detect close obstacles: any obstacle within 2 meters
    close_obstacle_threshold = 2.0  # 2 meters = close obstacle requiring avoidance
    close_obstacles = torch.any(lidar_distances < close_obstacle_threshold, dim=1)  # [num_envs] bool
    obstacle_avoidance_reward = torch.where(close_obstacles, -0.5, 0.05)  # Penalty for close obstacles
    
    # === FORWARD PATH CLEAR BONUS ===
    # Check front sector for clear path: front 25% of rays should be >3m clear (updated for 5m max range)
    front_ray_count = lidar_distances.shape[1] // 4  # Front 25% of 152 rays = 38 rays
    front_rays = lidar_distances[:, :front_ray_count]  # Front sector
    clear_path_threshold = 3.0  # 3 meters = clear forward path (adjusted for 5m max range)
    clear_front_path = torch.all(front_rays > clear_path_threshold, dim=1)  # [num_envs] bool
    navigation_bonus = torch.where(clear_front_path, 0.2, 0.0)  # Bonus for clear path ahead
    
    # === TERRAIN ADAPTATION BONUS ===
    # Reward maintaining appropriate height near G1 baseline
    avg_terrain_height = height_measurements.mean(dim=-1)  # [num_envs] - average terrain level
    baseline_deviation = torch.abs(avg_terrain_height - baseline)  # Distance from G1 baseline
    # Reward staying close to G1 baseline (0.209m on flat terrain)
    clearance_threshold = 0.15  # 15cm tolerance from G1 baseline
    terrain_adaptation_reward = torch.exp(-baseline_deviation / clearance_threshold) * 0.1
    
    return {
        "gap_avoidance": gap_avoidance_reward,           # Penalty for gaps
        "obstacle_avoidance": obstacle_avoidance_reward, # Penalty for close obstacles  
        "navigation_bonus": navigation_bonus,            # Bonus for clear forward path
        "terrain_adaptation": terrain_adaptation_reward, # Bonus for appropriate clearance
        "total_environmental": (
            gap_avoidance_reward + 
            obstacle_avoidance_reward + 
            navigation_bonus + 
            terrain_adaptation_reward
        )
    }

# ========== PROVEN ISAAC LAB LOCOMOTION REWARD FUNCTIONS ==========

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    
    Args:
        env: Isaac Lab environment instance
        command_name: Name of velocity command (e.g., "base_velocity")
        sensor_cfg: Contact sensor configuration with body_ids for feet
        threshold: Minimum air time threshold for reward
        
    Returns:
        torch.Tensor: Reward for feet air time [num_envs]
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    
    Args:
        env: Isaac Lab environment instance
        command_name: Name of velocity command (e.g., "base_velocity")
        threshold: Maximum air/contact time threshold for clamping
        sensor_cfg: Contact sensor configuration with body_ids for feet
        
    Returns:
        torch.Tensor: Reward for proper bipedal gait [num_envs]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    
    Args:
        env: Isaac Lab environment instance
        sensor_cfg: Contact sensor configuration with body_ids for feet
        asset_cfg: Robot asset configuration (defaults to "robot")
        
    Returns:
        torch.Tensor: Penalty for feet sliding [num_envs]
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel.
    
    Args:
        env: Isaac Lab environment instance
        std: Standard deviation for exponential kernel
        command_name: Name of velocity command (e.g., "base_velocity")
        asset_cfg: Robot asset configuration (defaults to "robot")
        
    Returns:
        torch.Tensor: Reward for velocity tracking [num_envs]
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel.
    
    Args:
        env: Isaac Lab environment instance
        command_name: Name of velocity command (e.g., "base_velocity")
        std: Standard deviation for exponential kernel
        asset_cfg: Robot asset configuration (defaults to "robot")
        
    Returns:
        torch.Tensor: Reward for angular velocity tracking [num_envs]
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2) 

# ========== CRITICAL ISAAC LAB USAGE PATTERNS ==========

"""
🚨 ISAAC LAB STANDARD SENSOR ACCESS FOR REWARD FUNCTIONS 🚨

✅ REWARD FUNCTIONS USE RAW SENSOR ACCESS:
Isaac Lab reward functions access sensors directly for physically meaningful measurements in meters!

🎯 PROVEN PATTERNS FROM ISAAC LAB SOURCE CODE:

1. RAW SENSOR DATA ACCESS (PREFERRED FOR REWARDS):
   # Height scanner - raw heights in meters
   height_sensor = env.scene.sensors["height_scanner"]
   height_measurements = height_sensor.data.pos_w[:, 2].unsqueeze(1) - height_sensor.data.ray_hits_w[..., 2] - 0.5
   
   # LiDAR - raw distances in meters  
   lidar_sensor = env.scene.sensors["lidar"]
   lidar_distances = torch.norm(lidar_sensor.data.ray_hits_w - lidar_sensor.data.pos_w.unsqueeze(1), dim=-1)

   # Physical thresholds with clear meaning:
   significant_gaps = height_measurements < -0.2    # 20cm below sensor = gap
   close_obstacles = lidar_distances < 2.0          # 2m distance = close obstacle

2. ROBOT STATE ACCESS:
   robot = env.scene["robot"]           # Dictionary access
   pos = robot.data.root_pos_w         # [N, 3] World position
   vel = robot.data.root_lin_vel_b     # [N, 3] Linear velocity

3. CONTACT SENSOR ACCESS:
   contact_sensor = env.scene.sensors["contact_forces"] 
   foot_ids, _ = contact_sensor.find_bodies(".*_ankle_roll_link")
   foot_forces = contact_sensor.data.net_forces_w[env_ids, foot_ids]

4. COMMAND ACCESS:
   velocity_cmd = env.command_manager.get_command("base_velocity")  # [N, 3]

5. MATH UTILITIES:
   from isaaclab.utils.math import quat_apply_inverse, yaw_quat
   quat_yaw = yaw_quat(robot.data.root_quat_w)
   transformed = quat_apply_inverse(quat, vector)

⚠️ UPDATED SENSOR RANGES (Isaac Lab Standard with G1 Baseline):

HEIGHT SCANNER: Physical measurements in meters with G1 baseline
- G1 Baseline: 0.209m on flat terrain (sensor_height - terrain_z - 0.5m offset)
- Obstacles: < 0.139m (baseline - 0.07m threshold)
- Gaps: > 0.279m (baseline + 0.07m threshold)
- Range: [-0.5m to +3.0m] relative to sensor position (clipped)
- Rays: 567 total (27×21 grid), 2.0×1.5m coverage, 7.5cm resolution, 3m max distance

LIDAR RANGE: Physical distances in meters  
- Range: [0.1m to 5.0m] actual distances to obstacles (updated from 15m)
- Rays: 152 total (8×19 channels), 180° FOV
- Use meaningful thresholds: 2.0m = close, 3.0m = clear path

🎯 G1 REWARD FUNCTION THRESHOLDS (Physical Units with Baseline):
height_measurements < 0.139   # G1 baseline - 0.07m = obstacles
height_measurements > 0.279   # G1 baseline + 0.07m = gaps  
lidar_distances < 2.0         # 2m = close obstacle
lidar_distances > 3.0         # 3m = clear path (adjusted for 5m max range)

🚨 MANDATORY SENSOR DATA SANITIZATION:
height_measurements = torch.where(torch.isfinite(height_measurements), height_measurements, torch.zeros_like(height_measurements))
lidar_distances = torch.where(torch.isfinite(lidar_distances), lidar_distances, torch.ones_like(lidar_distances) * 5.0)

✅ This approach follows Isaac Lab conventions and provides physically meaningful reward functions!
""" 