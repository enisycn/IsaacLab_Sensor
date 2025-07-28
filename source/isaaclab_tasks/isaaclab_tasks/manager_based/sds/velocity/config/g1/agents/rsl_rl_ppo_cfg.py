# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Optimized PPO configuration for G1 robot with basic locomotion (no sensors).
    
    This configuration is optimized for basic locomotion learning:
    - Robot State: ~81 dimensions (velocities, joints, commands, basic observations)
    - No environmental sensors (height scanner, lidar disabled)
    - Right-sized networks for efficient learning on simple observation space.
    """
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # OPTIMIZED NETWORKS: Right-sized for ~81 dimensional observation space
        actor_hidden_dims=[256, 128, 64],      # 3-layer network (efficient for basic locomotion)
        critic_hidden_dims=[256, 128, 64],     # Matching architecture for value estimation
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        # Keep same network size as rough (same ~81 dimensional observation space)
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [256, 128, 64]


@configclass
class G1EnhancedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Optimized PPO configuration for G1 robot with environmental sensors.
    
    This configuration is optimized for sensor-rich environment navigation:
    - Height Scanner: ~75 dims (2m x 1.5m, 20cm resolution)
    - LiDAR: ~144 dims (8 channels, 180°, 10° resolution)
    - Robot State: ~81 dims (velocities, joints, commands)
    Total observation space: ~300 dimensions - REQUIRES LARGER NETWORKS!
    """
    num_steps_per_env = 24
    max_iterations = 2000  # More iterations for complex sensor learning
    save_interval = 50
    experiment_name = "g1_enhanced"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # FIXED NETWORKS: LARGER networks for ~300 dimensional observation space!
        actor_hidden_dims=[512, 256, 128],     # 3-layer LARGE network for sensor processing
        critic_hidden_dims=[512, 256, 128],    # Matching large architecture for value estimation
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Good exploration for sensor-based navigation
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # Standard learning rate for larger networks
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
