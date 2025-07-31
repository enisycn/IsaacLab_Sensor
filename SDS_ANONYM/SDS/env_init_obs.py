"""
Isaac Lab Environment Reference for SDS Reward Function Generation

This file describes the Isaac Lab environment structure for GPT to generate compatible reward functions.
The environment uses Isaac Lab's ManagerBasedRLEnv with Unitree G1 humanoid robot and enhanced
environmental sensing capabilities for adaptive locomotion.

ENHANCED ENVIRONMENTAL SENSING:
- Height Scanner: 2m x 1.5m grid-based terrain height detection (15cm resolution, 130 points)
- LiDAR Sensor: 180° obstacle detection (8 channels, 144 rays, 5m range)  
- Environmental Observations: Normalized sensor data for gap/obstacle detection
- Adaptive Locomotion: Real-time sensor integration for terrain-aware reward functions
"""

import torch

class SDSIsaacLabEnvironment:
    """
    Isaac Lab Manager-Based RL Environment for SDS Humanoid Locomotion with Environmental Sensing
    
    Environment Details:
    - Robot: Unitree G1 humanoid (37 DOF total, 23 DOF controlled for complete humanoid control)
    - Action Space: 23 DOF full body joints for comprehensive humanoid locomotion
    - Task: Velocity tracking locomotion with environmental sensing and adaptive behavior
    - Framework: Isaac Lab ManagerBasedRLEnv
    - Control: 50Hz (20ms timestep, 4x decimation from 200Hz physics)
    
    Enhanced Environmental Sensing:
    - Height Scanner: Grid-based terrain analysis (gaps, stairs, height variations)
    - LiDAR Sensor: 360° obstacle detection and avoidance
    - Environmental Observations: Processed sensor data for adaptive reward functions
    - Real-time Integration: Sensor data available for terrain-aware locomotion rewards
    
    Controlled Joints for Full Body Locomotion (23 DOF):
    - Legs: 12 DOF (hip yaw/roll/pitch, knee, ankle pitch/roll per leg)
    - Arms: 10 DOF (shoulder pitch/roll/yaw, elbow pitch/roll per arm)
    - Torso: 1 DOF (torso_joint)
    
    Fixed Joints (14 DOF):
    - Hand Fingers: 14 DOF maintained at default poses (zero/one/two/three/four/five/six_joint per hand)
    
    Sensor Integration:
    - Contact Forces: Body contact detection for gait analysis
    - Height Scanner: 130-point grid for gap/stair detection (normalized [0,1])
    - LiDAR Range: 144-ray array for obstacle detection (normalized [0,1])
    - Environmental Analysis: Real-time terrain feature classification
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
        self.command_manager = self._command_interface()
        
    def _robot_interface(self):
        """Isaac Lab robot interface - Unitree G1 data access."""
        class Robot:
            def __init__(self):
                self.data = self._robot_data()
            
            def _robot_data(self):
                """Robot state data in Isaac Lab format."""
                class RobotData:
                    # Root/Base state (torso body)
                    root_pos_w = None      # [num_envs, 3] Position in world frame
                    root_quat_w = None     # [num_envs, 4] Quaternion (w,x,y,z) in world frame
                    root_lin_vel_b = None  # [num_envs, 3] Linear velocity in BODY frame  
                    root_ang_vel_b = None  # [num_envs, 3] Angular velocity in BODY frame
                    
                    # Joint state (37 joints: 12 legs + 1 torso + 10 arms + 14 hand fingers)
                    # CONTROLLED JOINTS (23 DOF): 12 legs + 1 torso + 10 arms (all except hand fingers)
                    joint_pos = None       # [num_envs, 37] Joint positions (rad)
                    joint_vel = None       # [num_envs, 37] Joint velocities (rad/s)
                    joint_acc = None       # [num_envs, 37] Joint accelerations (rad/s²)
                    
                    # Body states (all robot bodies)
                    body_pos_w = None      # [num_envs, num_bodies, 3] Body positions
                    body_quat_w = None     # [num_envs, num_bodies, 4] Body orientations
                    body_lin_vel_w = None  # [num_envs, num_bodies, 3] Body linear velocities
                
                return RobotData()
        
        return Robot()
    
    def _sensor_interface(self):
        """Isaac Lab sensor interface with enhanced environmental sensing."""
        class Sensors:
            def __init__(self):
                self.contact_forces = self._contact_sensor()
                self.height_scanner = self._height_scanner()
                self.lidar = self._lidar_sensor()
                
            def _contact_sensor(self):
                """Contact force sensor for all robot bodies."""
                class ContactSensor:
                    def __init__(self):
                        self.data = self._contact_data()
                    
                    def _contact_data(self):
                        """Contact sensor data."""
                        class ContactData:
                            # Contact forces in world frame for all bodies
                            net_forces_w = None           # [num_envs, num_bodies, 3]
                            
                            # Air time tracking (for gait analysis)
                            last_air_time = None          # [num_envs, num_bodies] 
                            current_air_time = None       # [num_envs, num_bodies]
                            current_contact_time = None   # [num_envs, num_bodies]
                        
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
                            # Grid-based height measurements around robot
                            # Configuration: 2m x 1.5m area, 15cm resolution = ~13x10 grid = 130 points
                            ray_hits_w = None             # [num_envs, 130, 3] Hit points in world frame
                            distances = None              # [num_envs, 130] Distances to ground
                            
                            # Processed height data (normalized to [0,1] range)
                            height_measurements = None    # [num_envs, 130] Height relative to robot
                        
                        return HeightScannerData()
                
                return HeightScanner()
            
            def _lidar_sensor(self):
                """LiDAR sensor for 360° environmental awareness."""
                class LiDARSensor:
                    def __init__(self):
                        self.data = self._lidar_data()
                    
                    def _lidar_data(self):
                        """LiDAR data for obstacle detection and navigation."""
                        class LiDARData:
                            # 180° front coverage: 8 channels x 18 horizontal rays = 144 points  
                            # Configuration: 8 vertical channels, 10° horizontal resolution, 5m range
                            ray_hits_w = None             # [num_envs, 144, 3] Hit points in world frame
                            distances = None              # [num_envs, 144] Range measurements
                            
                            # Processed range data (normalized to [0,1] range) 
                            range_measurements = None     # [num_envs, 144] Obstacle distances
                        
                        return LiDARData()
                
                return LiDARSensor()
        
        return Sensors()
    
    def _command_interface(self):
        """Isaac Lab command manager."""
        class CommandManager:
            def get_command(self, command_name):
                """Get velocity command."""
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
    "action_space": 23,             # FULL BODY: Controlled joints for complete humanoid control (12 legs + 1 torso + 10 arms)
    "controlled_joints": {
        "legs": 12,                 # Controlled: All leg joints for locomotion
        "torso": 1,                 # Controlled: Torso joint for posture
        "arms": 10,                 # Controlled: All arm joints for balance and natural movement
    },
    "fixed_joints": {
        "hand_fingers": 14,         # Fixed: Hand finger joints at default poses
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
    """Get binary foot contacts for G1 humanoid using CORRECTED link-based detection."""
    # Isaac Lab contact sensor detects forces on body links (not joints)
    # Get contact forces for ankle roll links (G1 foot contact bodies)
    contact_forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
    
    # Find foot contact bodies using the correct Isaac Lab pattern
    foot_ids, foot_names = contact_sensor.find_bodies(".*_ankle_roll_link")  # Link names for contact sensor
    
    if len(foot_ids) == 0:
        # Fallback pattern if ankle_roll_link not found
        foot_ids, foot_names = contact_sensor.find_bodies(".*foot.*")
        if len(foot_ids) == 0:
            # Final fallback - try any ankle-related bodies
            foot_ids, foot_names = contact_sensor.find_bodies(".*ankle.*")
    
    foot_forces = contact_forces[:, foot_ids, :]  # [num_envs, num_feet, 3]
    force_magnitudes = foot_forces.norm(dim=-1)  # [num_envs, num_feet]
    foot_contacts = (force_magnitudes > threshold).float()  # [num_envs, num_feet]
    
    # Ensure we have exactly 2 feet (left, right)
    if foot_contacts.shape[-1] != 2:
        # Pad or trim to 2 feet
        num_envs = foot_contacts.shape[0]
        new_contacts = torch.zeros((num_envs, 2), device=foot_contacts.device)
        if foot_contacts.shape[-1] > 0:
            new_contacts[:, :min(2, foot_contacts.shape[-1])] = foot_contacts[:, :min(2, foot_contacts.shape[-1])]
        foot_contacts = new_contacts
    
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

def get_height_scan_data(env):
    """
    Extract height scanner data for terrain analysis and gap detection.
    
    The height scanner provides a 2m x 1.5m grid of height measurements around the robot
    with 15cm resolution, enabling detection of gaps, stairs, and terrain variations.
    
    Args:
        env: Isaac Lab environment instance
        
    Returns:
        torch.Tensor: Height scan data [num_envs, 130] - normalized to [0,1] range
    """
    # Access height scan observation (properly normalized by environment)
    height_scan = env.observation_manager.get_term("height_scan")  # [num_envs, 130]
    return height_scan

def get_lidar_data(env):
    """
    Extract LiDAR range data for obstacle detection and navigation.
    
    The LiDAR provides 180° front coverage with 8 vertical channels and 10° horizontal 
    resolution (144 total rays), enabling detection of obstacles up to 5m range.
    
    Args:
        env: Isaac Lab environment instance
        
    Returns:
        torch.Tensor: LiDAR range data [num_envs, 144] - normalized to [0,1] range
    """
    # Access LiDAR range observation (properly normalized by environment)
    lidar_range = env.observation_manager.get_term("lidar_range")  # [num_envs, 144]
    return lidar_range

def analyze_terrain_features(env, gap_threshold=0.1, obstacle_threshold=0.8):
    """
    Analyze terrain features for adaptive locomotion using height scanner and LiDAR.
    
    This function demonstrates how to process sensor data to detect environmental
    features that require adaptive locomotion behaviors.
    
    Args:
        env: Isaac Lab environment instance
        gap_threshold: Height scan value indicating gaps (normalized, 0.1 = deep gap)
        obstacle_threshold: LiDAR value indicating close obstacles (normalized, 0.8 = close)
        
    Returns:
        dict: Terrain analysis with gap detection, obstacle detection, and safety assessment
    """
    # Get sensor data
    height_scan = get_height_scan_data(env)  # [num_envs, 130]
    lidar_range = get_lidar_data(env)        # [num_envs, 144]
    
    # Gap detection using height scanner (low values = gaps)
    gap_mask = height_scan < gap_threshold   # [num_envs, 130]
    gap_count = gap_mask.sum(dim=-1)         # [num_envs] - number of gap points
    gap_density = gap_count / height_scan.shape[-1]  # [num_envs] - fraction of area with gaps
    
    # Obstacle detection using LiDAR (high values = far, low values = close obstacles)
    obstacle_mask = lidar_range < obstacle_threshold  # [num_envs, 144]  
    obstacle_count = obstacle_mask.sum(dim=-1)        # [num_envs] - number of obstacle rays
    obstacle_density = obstacle_count / lidar_range.shape[-1]  # [num_envs] - fraction with obstacles
    
    # Forward path analysis (front-facing sensors)
    # Height scanner: center portion for forward path
    forward_height_indices = slice(45, 85)  # Center 40 points of 130-point grid
    forward_gaps = gap_mask[:, forward_height_indices].any(dim=-1)  # [num_envs] - gaps ahead
    
    # LiDAR: front-facing rays for obstacle avoidance
    forward_lidar_indices = slice(60, 84)   # Center 24 rays of 144-ray array  
    forward_obstacles = obstacle_mask[:, forward_lidar_indices].any(dim=-1)  # [num_envs] - obstacles ahead
    
    # Safety assessment for navigation
    safe_forward_path = ~forward_gaps & ~forward_obstacles  # [num_envs] - clear forward path
    
    return {
        "height_scan": height_scan,
        "lidar_range": lidar_range,
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
            "forward_obstacles": forward_obstacles
        },
        "navigation_safety": {
            "safe_forward_path": safe_forward_path,
            "terrain_complexity": gap_density + obstacle_density  # Combined challenge metric
        }
    }

def get_environmental_reward_components(env):
    """
    Generate environmental reward components based on sensor analysis.
    
    This function shows how to create reward signals that adapt to environmental
    conditions detected by height scanner and LiDAR sensors.
    
    Args:
        env: Isaac Lab environment instance
        
    Returns:
        dict: Environmental reward components for adaptive locomotion
    """
    terrain_analysis = analyze_terrain_features(env)
    
    # Adaptive gap navigation rewards
    gap_density = terrain_analysis["gap_detection"]["gap_density"]
    forward_gaps = terrain_analysis["gap_detection"]["forward_gaps"]
    
    # Encourage appropriate gap behavior based on detection
    gap_avoidance_reward = torch.where(
        forward_gaps,
        -torch.ones_like(gap_density),  # Penalty for stepping into gaps
        torch.zeros_like(gap_density)   # No penalty when no gaps ahead
    )
    
    # Adaptive obstacle avoidance rewards
    obstacle_density = terrain_analysis["obstacle_detection"]["obstacle_density"]
    forward_obstacles = terrain_analysis["obstacle_detection"]["forward_obstacles"]
    
    # Encourage obstacle avoidance based on detection
    obstacle_avoidance_reward = torch.where(
        forward_obstacles,
        -torch.ones_like(obstacle_density),  # Penalty for colliding with obstacles
        torch.zeros_like(obstacle_density)   # No penalty when clear path
    )
    
    # Safe navigation bonus
    safe_path = terrain_analysis["navigation_safety"]["safe_forward_path"]
    navigation_bonus = torch.where(
        safe_path,
        torch.ones_like(gap_density),     # Bonus for maintaining safe navigation
        torch.zeros_like(gap_density)    # No bonus when path is unsafe
    )
    
    # Terrain complexity adaptation
    terrain_complexity = terrain_analysis["navigation_safety"]["terrain_complexity"]
    complexity_factor = torch.clamp(1.0 - terrain_complexity, 0.1, 1.0)  # Scale other rewards
    
    return {
        "gap_avoidance": gap_avoidance_reward,
        "obstacle_avoidance": obstacle_avoidance_reward, 
        "navigation_bonus": navigation_bonus,
        "complexity_factor": complexity_factor,
        "raw_terrain_analysis": terrain_analysis
    } 