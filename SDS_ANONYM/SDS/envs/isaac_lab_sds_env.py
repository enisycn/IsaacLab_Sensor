"""
Isaac Lab Environment Reference for SDS Reward Function Generation

This file describes the Isaac Lab environment structure for GPT to generate compatible reward functions.
The environment uses Isaac Lab's ManagerBasedRLEnv with Unitree G1 humanoid robot.
"""

import torch

class SDSIsaacLabEnvironment:
    """
    Isaac Lab Manager-Based RL Environment for SDS Humanoid Locomotion
    
    Environment Details:
    - Robot: Unitree G1 humanoid (37 DOF total, 13 DOF controlled for locomotion)
    - Action Space: 13 DOF legs-only joints for cleaner locomotion debugging
    - Task: Velocity tracking locomotion  
    - Framework: Isaac Lab ManagerBasedRLEnv
    - Control: 50Hz (20ms timestep, 4x decimation from 200Hz physics)
    
    Controlled Joints for Locomotion (13 DOF):
    - Legs: 8 DOF (hip yaw/roll/pitch, knee per leg)
    - Ankles: 4 DOF (pitch/roll per ankle)  
    - Torso: 1 DOF (torso_joint)
    
    Fixed Joints (24 DOF):
    - Arms: 10 DOF maintained at default poses for stability
    - Hands: 14 DOF maintained at default poses for walking stability
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
                    
                    # Joint state (37 joints: 12 legs + 1 torso + 10 arms + 14 hands)
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
        """Isaac Lab sensor interface."""
        class Sensors:
            def __init__(self):
                self.contact_forces = self._contact_sensor()
                
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
    # Leg joints (9 DOF - 4 per leg + 1 torso) - VERIFIED FROM ISAAC LAB
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
    "torso_joint",
    
    # Ankle joints (4 DOF - 2 per foot) - VERIFIED FROM ISAAC LAB
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    
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
    "action_space": 13,             # LEGS-ONLY: Controlled joints for locomotion debugging (6 legs + 4 ankles + 1 torso)
    "arm_joints": 10,               # Fixed at default poses during locomotion
    "hand_joints": 14,              # Fixed at default poses during locomotion
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