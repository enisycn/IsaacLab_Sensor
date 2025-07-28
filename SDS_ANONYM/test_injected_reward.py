def sds_custom_reward(env) -> torch.Tensor:
    """Test reward function"""
    import torch
    
    robot = env.scene["robot"]
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # Sample reward components
    height_reward = torch.ones(env.num_envs, device=env.device) * 2.0
    velocity_reward = torch.ones(env.num_envs, device=env.device) * 1.5
    stability_bonus = torch.ones(env.num_envs, device=env.device) * 0.8
    contact_penalty = torch.ones(env.num_envs, device=env.device) * -0.3
    
    # Combine rewards
    reward = height_reward + velocity_reward + stability_bonus + contact_penalty
    

# SDS Metrics Collection - Auto-injected
if not hasattr(env, 'sds_metrics'):
    class SDSMetrics:
        def __init__(self):
            self.components = {}
    env.sds_metrics = SDSMetrics()
try:
    comp_val = height_reward
    if torch.is_tensor(comp_val):
        env.sds_metrics.components['height_reward'] = comp_val.detach().cpu().mean().item()
    else:
        env.sds_metrics.components['height_reward'] = float(comp_val)
except:
    env.sds_metrics.components['height_reward'] = 0.0
try:
    comp_val = velocity_reward
    if torch.is_tensor(comp_val):
        env.sds_metrics.components['velocity_reward'] = comp_val.detach().cpu().mean().item()
    else:
        env.sds_metrics.components['velocity_reward'] = float(comp_val)
except:
    env.sds_metrics.components['velocity_reward'] = 0.0
try:
    comp_val = stability_bonus
    if torch.is_tensor(comp_val):
        env.sds_metrics.components['stability_bonus'] = comp_val.detach().cpu().mean().item()
    else:
        env.sds_metrics.components['stability_bonus'] = float(comp_val)
except:
    env.sds_metrics.components['stability_bonus'] = 0.0
try:
    comp_val = contact_penalty
    if torch.is_tensor(comp_val):
        env.sds_metrics.components['contact_penalty'] = comp_val.detach().cpu().mean().item()
    else:
        env.sds_metrics.components['contact_penalty'] = float(comp_val)
except:
    env.sds_metrics.components['contact_penalty'] = 0.0
try:
    env.sds_metrics.components['total_reward'] = reward.detach().cpu().mean().item()
except:
    env.sds_metrics.components['total_reward'] = 0.0

    return reward.clamp(min=0.0, max=10.0)