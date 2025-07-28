# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common observation functions for SDS environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def lidar_range(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """LiDAR range measurements from the sensor.
    
    This function computes the distance from the sensor to the hit points
    for all rays in the LiDAR sensor.
    
    Args:
        env: The environment.
        sensor_cfg: The sensor configuration.
        
    Returns:
        The distance measurements with shape (num_envs, num_rays).
    """
    # Extract the sensor
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # Compute distance from sensor position to hit points
    distances = torch.norm(
        sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), 
        dim=-1
    )
    
    # Flatten to (num_envs, num_rays)
    return distances.view(env.num_envs, -1) 