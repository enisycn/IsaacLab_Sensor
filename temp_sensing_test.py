import torch
import logging
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import json

try:
    # Parse environment configuration for SDS  
    env_cfg = parse_env_cfg(
        task_name="Isaac-SDS-Velocity-Rough-G1-v0", 
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        num_envs=1
    )
    
    # Create environment using gym.make
    import gymnasium as gym
    env = gym.make("Isaac-SDS-Velocity-Rough-G1-v0", cfg=env_cfg)
    
    # Reset to get observations
    obs, info = env.reset()
    
    # Extract height scan data
    if "policy" in obs:
        height_scan_start = 3 + 3 + 3 + 3 + 37 + 37 + 23  # = 109
        height_scan_end = height_scan_start + 187  # = 296
        
        if obs["policy"].shape[1] >= height_scan_end:
            height_data = obs["policy"][0, height_scan_start:height_scan_end]
            
            # Quick analysis
            mean_height = torch.mean(height_data).item()
            height_std = torch.std(height_data).item()
            
            # Detect gaps and obstacles
            gaps = height_data < -0.2
            obstacles = height_data > 0.1
            gap_count = torch.sum(gaps).item()
            obstacle_count = torch.sum(obstacles).item()
            
            result = {
                "terrain_type": "mixed_terrain" if gap_count > 10 or obstacle_count > 10 else "rough",
                "roughness": float(height_std),
                "gaps_detected": gap_count > 0,
                "obstacles_detected": obstacle_count > 0,
                "gap_count": gap_count,
                "obstacle_count": obstacle_count,
                "max_gap_depth": abs(torch.min(height_data[gaps]).item()) if gap_count > 0 else 0.0,
                "max_obstacle_height": torch.max(height_data[obstacles]).item() if obstacle_count > 0 else 0.0,
                "robot_stability": "unstable" if gap_count > 15 or obstacle_count > 15 else "moderate",
                "risk_level": "high" if gap_count > 15 or obstacle_count > 15 else "medium"
            }
            
            print("ENVIRONMENT_DATA_START")
            print(json.dumps(result))
            print("ENVIRONMENT_DATA_END")
        else:
            print("ERROR: observation tensor too small")
    else:
        print("ERROR: no policy observations")
    
    env.close()
    
except Exception as e:
    print(f"ENVIRONMENT_SENSING_ERROR: {e}")
    
simulation_app.close()

