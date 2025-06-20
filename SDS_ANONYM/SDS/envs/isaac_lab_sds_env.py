"""
Isaac Lab Environment Reference for SDS Reward Function Generation

This file describes the Isaac Lab environment structure for GPT to generate compatible reward functions.
The environment uses Isaac Lab's ManagerBasedRLEnv with Unitree Go1 quadruped robot.
"""

class SDSIsaacLabEnvironment:
    """
    Isaac Lab Manager-Based RL Environment for SDS Quadruped Locomotion
    
    Environment Details:
    - Robot: Unitree Go1 quadruped (12 DOF)
    - Task: Velocity tracking locomotion  
    - Framework: Isaac Lab ManagerBasedRLEnv
    - Control: 50Hz (20ms timestep, 4x decimation from 200Hz physics)
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
        """Isaac Lab robot interface - Unitree Go1 data access."""
        class Robot:
            def __init__(self):
                self.data = self._robot_data()
            
            def _robot_data(self):
                """Robot state data in Isaac Lab format."""
                class RobotData:
                    # Root/Base state (trunk body)
                    root_pos_w = None      # [num_envs, 3] Position in world frame
                    root_quat_w = None     # [num_envs, 4] Quaternion (w,x,y,z) in world frame
                    root_lin_vel_b = None  # [num_envs, 3] Linear velocity in BODY frame  
                    root_ang_vel_b = None  # [num_envs, 3] Angular velocity in BODY frame
                    
                    # Joint state (12 joints: 3 per leg × 4 legs)
                    joint_pos = None       # [num_envs, 12] Joint positions (rad)
                    joint_vel = None       # [num_envs, 12] Joint velocities (rad/s)
                    joint_acc = None       # [num_envs, 12] Joint accelerations (rad/s²)
                    
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

# Unitree Go1 Robot Structure
GO1_BODY_NAMES = {
    "base": "trunk",           # Main chassis/base body
    "feet": [                  # Foot bodies (end effectors)
        "FL_foot",             # Front Left foot
        "FR_foot",             # Front Right foot  
        "RL_foot",             # Rear Left foot
        "RR_foot"              # Rear Right foot
    ],
    "legs": {                  # Leg segment bodies
        "thighs": ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"],
        "calfs": ["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
        "hips": ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
    }
}

# Joint Configuration (12 DOF total)
GO1_JOINT_NAMES = [
    # Front Left leg (3 DOF)
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    # Front Right leg (3 DOF)  
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    # Rear Left leg (3 DOF)
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", 
    # Rear Right leg (3 DOF)
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

# Physical Parameters
GO1_SPECS = {
    "nominal_height": 0.34,   # meters - trunk height above ground
    "mass": 12.0,             # kg - approximate robot mass
    "num_joints": 12,         # Total degrees of freedom
    "num_feet": 4,            # Number of foot contact points
    "control_freq": 50,       # Hz - control frequency  
    "physics_freq": 200       # Hz - physics simulation frequency
}

# Contact Force Analysis Helper
def extract_foot_contacts(env, force_threshold=1.0):
    """
    Extract foot contact information from Isaac Lab contact sensor.
    
    Args:
        env: Isaac Lab environment instance
        force_threshold: Minimum force magnitude for contact detection (N)
        
    Returns:
        dict: Contact states for each foot {foot_name: contact_tensor}
    """
    # Get contact sensor
    contact_sensor = env.scene.sensors["contact_forces"]
    contact_forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
    
    # Get foot body indices correctly
    foot_ids, foot_names = contact_sensor.find_bodies(".*_foot")
    
    # Calculate force magnitudes for feet
    foot_forces = contact_forces[:, foot_ids, :]  # [num_envs, num_feet, 3]
    force_magnitudes = foot_forces.norm(dim=-1)  # [num_envs, num_feet]
    
    # Create contact dict with foot names
    foot_contacts = {}
    for i, foot_name in enumerate(foot_names):
        foot_contacts[foot_name] = force_magnitudes[:, i] > force_threshold
    
    return foot_contacts

# Utility Functions for Isaac Lab Reward Development
def get_base_orientation_error(robot_data):
    """Calculate orientation error from upright pose."""
    # robot_data.root_quat_w is [num_envs, 4] quaternion (w,x,y,z)
    # Calculate rotation from upright pose (identity quaternion)
    pass

def get_velocity_tracking_error(robot_data, commands):
    """Calculate error between desired and actual velocity."""
    # robot_data.root_lin_vel_b is [num_envs, 3] 
    # commands is [num_envs, 3] (vx, vy, omega_z)
    pass

def calculate_joint_power(robot_data):
    """Calculate mechanical power consumption."""
    # Power = torque * velocity (estimated from joint accelerations)
    pass

# Example Isaac Lab Reward Function Template
"""
def sds_custom_reward(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
    '''Custom SDS reward for Isaac Lab locomotion.'''
    
    # Access robot and sensor data
    robot = env.scene["robot"]
    contact_sensor = env.scene.sensors["contact_forces"]
    commands = env.command_manager.get_command("base_velocity")
    
    # Initialize reward tensor
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # Example reward components:
    # 1. Velocity tracking
    # velocity_error = (robot.data.root_lin_vel_b[:, :2] - commands[:, :2]).norm(dim=-1)
    # reward += torch.exp(-velocity_error / 0.5)  # Exponential reward
    
    # 2. Orientation stability
    # gravity_projection = ... # Calculate projected gravity
    # reward -= torch.square(gravity_projection[:, :2]).sum(dim=-1)
    
    # 3. Contact patterns (CORRECTED VERSION)
    # Get foot contact forces correctly
    # foot_ids, foot_names = contact_sensor.find_bodies(".*_foot")
    # contact_forces = contact_sensor.data.net_forces_w[:, foot_ids, :]  # [num_envs, 4, 3]
    # contact_magnitudes = torch.norm(contact_forces, dim=-1)  # [num_envs, 4]
    # foot_contacts = contact_magnitudes > 5.0  # Binary contact detection
    # fl, fr, rl, rr = foot_contacts[:, 0], foot_contacts[:, 1], foot_contacts[:, 2], foot_contacts[:, 3]
    # trot_pattern = (fl & rr & ~fr & ~rl) | (~fl & ~rr & fr & rl)  # Diagonal pairs
    # reward += trot_pattern.float() * 2.0
    
    return reward
""" 

# Detailed Contact Analysis for SDS Reward Functions
def get_foot_contact_analysis(env, contact_threshold=5.0):
    """
    Comprehensive foot contact analysis for reward function development.
    
    Returns detailed contact information including force magnitudes,
    contact states, gait patterns, and temporal analysis.
    """
    contact_sensor = env.scene.sensors["contact_forces"]
    
    # Get foot indices and names
    foot_ids, foot_names = contact_sensor.find_bodies(".*_foot")
    
    # Extract contact forces
    contact_forces = contact_sensor.data.net_forces_w[:, foot_ids, :]  # [num_envs, 4, 3]
    force_magnitudes = torch.norm(contact_forces, dim=-1)  # [num_envs, 4]
    
    # Binary contact detection
    in_contact = force_magnitudes > contact_threshold  # [num_envs, 4]
    
    # Gait pattern analysis
    fl, fr, rl, rr = in_contact[:, 0], in_contact[:, 1], in_contact[:, 2], in_contact[:, 3]
    
    # Trotting: diagonal pairs alternate
    trot_diagonal1 = fl & rr & ~fr & ~rl  # FL+RR, no FR+RL
    trot_diagonal2 = ~fl & ~rr & fr & rl  # FR+RL, no FL+RR  
    trot_pattern = trot_diagonal1 | trot_diagonal2
    
    # Pace: lateral pairs
    pace_left = fl & rl & ~fr & ~rr   # Left legs
    pace_right = ~fl & ~rl & fr & rr  # Right legs
    pace_pattern = pace_left | pace_right
    
    # Bound: front/rear pairs  
    bound_front = fl & fr & ~rl & ~rr  # Front legs
    bound_rear = ~fl & ~fr & rl & rr   # Rear legs
    bound_pattern = bound_front | bound_rear
    
    # Contact count and stability
    num_contacts = torch.sum(in_contact, dim=-1)  # [num_envs]
    stable_contact = (num_contacts >= 2) & (num_contacts <= 3)  # Good stability
    
    return {
        'foot_names': foot_names,
        'foot_ids': foot_ids,
        'force_magnitudes': force_magnitudes,
        'in_contact': in_contact,
        'trot_pattern': trot_pattern,
        'pace_pattern': pace_pattern, 
        'bound_pattern': bound_pattern,
        'num_contacts': num_contacts,
        'stable_contact': stable_contact,
        'contact_forces_raw': contact_forces
    } 