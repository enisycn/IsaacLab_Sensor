#!/usr/bin/env python3
"""
Analyze Current SDS Reward Function

This script analyzes the current SDS custom reward function and shows:
1. Individual reward components and their weights
2. Expected behavior from each component
3. Visual breakdown of reward structure
"""

import re
from pathlib import Path

def parse_current_reward_function():
    """Parse the current SDS custom reward function to extract components."""
    
    # Read the current reward function
    reward_file = Path("../source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/mdp/rewards.py")
    
    if not reward_file.exists():
        print(f"❌ Reward file not found: {reward_file}")
        return None
    
    with open(reward_file, 'r') as f:
        content = f.read()
    
    # Extract the sds_custom_reward function
    pattern = r'def sds_custom_reward\(env\).*?return reward'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("❌ Could not find sds_custom_reward function")
        return None
    
    reward_function = match.group(0)
    
    print("🎯 Current SDS Custom Reward Function Analysis")
    print("=" * 60)
    
    # Print the actual function code for reference
    print("\n📝 CURRENT REWARD FUNCTION CODE:")
    print("-" * 40)
    lines = reward_function.split('\n')
    for i, line in enumerate(lines[:30]):  # Show first 30 lines
        if line.strip():
            print(f"{i+1:2d}: {line}")
    
    # Parse reward components
    components = []
    
    # 1. Velocity tracking
    if "desired_velocity" in reward_function:
        vel_match = re.search(r'desired_velocity = ([\d.]+)', reward_function)
        if vel_match:
            target_vel = float(vel_match.group(1))
            components.append({
                'name': 'Forward Velocity Tracking',
                'target': f'{target_vel} m/s',
                'weight': '1.0 (exponential)',
                'formula': 'exp(-|actual_vel - target_vel|)',
                'behavior': f'Encourages robot to move forward at {target_vel} m/s'
            })
    
    # 2. Height maintenance
    if "height_error" in reward_function:
        height_match = re.search(r'- ([\d.]+)\).*?Nominal height: ([\d.]+)m', reward_function)
        if height_match:
            target_height = float(height_match.group(2))
            components.append({
                'name': 'Height Maintenance',
                'target': f'{target_height} m',
                'weight': '1.0 (exponential)',
                'formula': 'exp(-|actual_height - target_height|)',
                'behavior': f'Keeps robot trunk at {target_height}m height'
            })
    
    # 3. Orientation stability
    if "gravity_alignment" in reward_function:
        components.append({
            'name': 'Orientation Stability',
            'target': 'Upright posture',
            'weight': '1.0 (exponential)',
            'formula': 'exp(-gravity_alignment²)',
            'behavior': 'Prevents robot from tilting/falling over'
        })
    
    # 4. Contact pattern reward
    if "contact_reward" in reward_function:
        contact_match = re.search(r'([\d.]+) \* contact_reward', reward_function)
        if contact_match:
            contact_weight = float(contact_match.group(1))
            components.append({
                'name': 'Foot Contact Pattern',
                'target': 'At least one foot contact',
                'weight': f'{contact_weight}',
                'formula': 'count(foot_force > 1.0 N)',
                'behavior': 'Encourages stable foot contacts during locomotion'
            })
    
    # 5. Joint velocity penalty
    if "joint_vel_penalty" in reward_function:
        joint_match = re.search(r'([\d.]+) \* joint_vel_penalty', reward_function)
        if joint_match:
            joint_weight = float(joint_match.group(1))
            components.append({
                'name': 'Joint Velocity Penalty',
                'target': 'Smooth motion',
                'weight': f'-{joint_weight}',
                'formula': '-sum(joint_vel²)',
                'behavior': 'Penalizes jerky/excessive joint movements'
            })
    
    return components

def display_reward_breakdown(components):
    """Display a clear breakdown of reward components."""
    
    if not components:
        print("❌ No reward components found")
        return
    
    print("\n📊 REWARD COMPONENTS BREAKDOWN")
    print("=" * 60)
    
    total_positive_weight = 0
    total_negative_weight = 0
    
    for i, comp in enumerate(components, 1):
        print(f"\n{i}. {comp['name']}")
        print(f"   🎯 Target: {comp['target']}")
        print(f"   ⚖️  Weight: {comp['weight']}")
        print(f"   📐 Formula: {comp['formula']}")
        print(f"   🤖 Behavior: {comp['behavior']}")
        
        # Track weights for summary
        weight_str = comp['weight']
        if 'exponential' in weight_str:
            total_positive_weight += 1.0
        elif weight_str.startswith('-'):
            total_negative_weight += abs(float(weight_str.replace('-', '')))
        else:
            try:
                w = float(weight_str)
                if w > 0:
                    total_positive_weight += w
                else:
                    total_negative_weight += abs(w)
            except:
                pass
    
    print("\n📈 REWARD STRUCTURE SUMMARY")
    print("=" * 60)
    print(f"✅ Positive rewards: ~{total_positive_weight:.2f} total weight")
    print(f"❌ Penalties: ~{total_negative_weight:.2f} total weight")
    
    print("\n🎮 EXPECTED ROBOT BEHAVIOR")
    print("=" * 60)
    print("Based on this reward function, the robot should:")
    print("1. 🏃 Move forward at 2.5 m/s consistently")
    print("2. 📏 Maintain trunk height around 0.34m")
    print("3. 🎯 Stay upright and balanced")
    print("4. 🦶 Keep feet in contact with ground")
    print("5. 🔄 Move joints smoothly without jerking")
    
    print("\n⚡ REWARD MAGNITUDES")
    print("=" * 60)
    print("• Forward velocity: HIGH impact (exponential reward)")
    print("• Height maintenance: HIGH impact (exponential reward)")
    print("• Orientation: HIGH impact (exponential reward)")
    print("• Foot contacts: MEDIUM impact (0.1x multiplier)")
    print("• Joint smoothness: LOW impact (0.01x penalty)")

def create_reward_visualization():
    """Create a simple visualization of reward components."""
    print("\n📊 REWARD COMPONENT WEIGHTS VISUALIZATION")
    print("=" * 60)
    
    components = ['Forward Velocity', 'Height Control', 'Orientation', 'Foot Contact', 'Joint Smooth.']
    weights = [1.0, 1.0, 1.0, 0.1, -0.01]
    colors = ['green' if w > 0 else 'red' for w in weights]
    
    # Text-based bar chart
    max_weight = max(abs(w) for w in weights)
    for comp, weight, color in zip(components, weights, colors):
        bar_length = int(abs(weight) / max_weight * 20)
        bar = '█' * bar_length
        sign = '+' if weight > 0 else '-'
        color_symbol = '🟢' if weight > 0 else '🔴'
        print(f"{color_symbol} {comp:<15} {sign}{abs(weight):>5.2f} |{bar:<20}|")

def analyze_reward_function():
    """Main analysis function."""
    print("🤖 SDS Custom Reward Function Analysis")
    print("Isaac Lab Integration - Go1 Quadruped Locomotion")
    print("=" * 60)
    
    components = parse_current_reward_function()
    
    if components:
        display_reward_breakdown(components)
        create_reward_visualization()
        
        print("\n🔧 HOW TO MODIFY")
        print("=" * 60)
        print("To change reward components, modify the sds_custom_reward function in:")
        print("source/isaaclab_tasks/isaaclab_tasks/manager_based/sds/velocity/mdp/rewards.py")
        print("\nOr run SDS optimization to let GPT-4 generate new reward functions!")
        
    else:
        print("❌ Could not analyze reward function")

if __name__ == "__main__":
    analyze_reward_function() 