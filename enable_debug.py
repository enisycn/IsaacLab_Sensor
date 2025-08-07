#!/usr/bin/env python3
"""
Enable reward debugging for SDS training.
Add this to your training script to see detailed reward component breakdowns.
"""

def enable_reward_debugging(env):
    """Enable detailed reward debugging output."""
    env._debug_rewards = True
    print("🔧 REWARD DEBUGGING ENABLED")
    print("📊 Detailed reward component breakdowns will be printed every 100 steps")
    print("🎯 Look for:")
    print("   - Gap detection status (should be > 0 when near gaps)")
    print("   - Height map statistics (positive values = gaps below)")
    print("   - FSM state distribution (0=normal, 1=approaching, 2=jumped, 3=landed)")
    print("   - Forward velocity commands and tracking")

# Example usage in your training script:
# import enable_debug
# enable_debug.enable_reward_debugging(env) 