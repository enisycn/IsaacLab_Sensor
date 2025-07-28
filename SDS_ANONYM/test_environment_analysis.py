#!/usr/bin/env python3
"""
Test script for environment-aware SDS integration
This script tests the dynamic environment analysis injection system
"""

import os
import sys
import logging
from pathlib import Path

# Add SDS to path
sys.path.append('SDS')

from SDS.agents import EnvironmentAwareTaskDescriptor
from omegaconf import DictConfig

def test_environment_analysis_injection():
    """Test the environment analysis injection system"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a minimal config
    cfg = DictConfig({
        'model': 'gpt-4o-mini',
        'temperature': 0.8
    })
    
    # Create environment-aware task descriptor
    prompt_dir = "SDS/prompts"
    agent = EnvironmentAwareTaskDescriptor(cfg, prompt_dir)
    
    print("🧪 Testing Environment Analysis Integration")
    print("="*60)
    
    # Test 1: Check if prompt file exists
    print("✅ Test 1: Checking prompt file...")
    if os.path.exists(agent.prompt_file):
        print(f"   ✓ Prompt file found: {agent.prompt_file}")
    else:
        print(f"   ❌ Prompt file missing: {agent.prompt_file}")
        return False
    
    # Test 2: Check if markers exist in prompt
    print("✅ Test 2: Checking environment analysis markers...")
    with open(agent.prompt_file, 'r') as f:
        content = f.read()
    
    start_marker = "<!-- ENVIRONMENT_ANALYSIS_START -->"
    end_marker = "<!-- ENVIRONMENT_ANALYSIS_END -->"
    
    if start_marker in content and end_marker in content:
        print("   ✓ Environment analysis markers found")
    else:
        print("   ❌ Environment analysis markers missing")
        return False
    
    # Test 3: Test environment analysis execution (with smaller robot count)
    print("✅ Test 3: Testing environment analysis execution...")
    try:
        analysis_result = agent.run_environment_analysis(num_envs=3)
        if analysis_result:
            print("   ✓ Environment analysis executed successfully")
            print(f"   ✓ Analysis length: {len(analysis_result)} characters")
            
            # Show a preview of the analysis
            preview = analysis_result[:200] + "..." if len(analysis_result) > 200 else analysis_result
            print(f"   📊 Analysis preview: {preview}")
        else:
            print("   ❌ Environment analysis failed")
            return False
    except Exception as e:
        print(f"   ❌ Environment analysis error: {e}")
        return False
    
    # Test 4: Test dynamic injection
    print("✅ Test 4: Testing dynamic environment analysis injection...")
    try:
        test_analysis = """
🎯 TEST ANALYSIS SCOPE:
   Robots Analyzed: 3
   Total Gaps Detected: 100
   Environment Verdict: 🟢 LOW RISK - Test successful
        """
        
        success = agent.inject_environment_analysis(test_analysis)
        if success:
            print("   ✓ Environment analysis injection successful")
            
            # Verify injection worked
            with open(agent.prompt_file, 'r') as f:
                updated_content = f.read()
            
            if "TEST ANALYSIS SCOPE" in updated_content:
                print("   ✓ Test analysis found in prompt file")
            else:
                print("   ❌ Test analysis not found in prompt file")
                return False
        else:
            print("   ❌ Environment analysis injection failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Injection error: {e}")
        return False
    
    print("="*60)
    print("🎉 All tests passed! Environment analysis integration is working correctly.")
    print("")
    print("🚀 Usage Instructions:")
    print("1. Set 'enable_environment_analysis: true' in your config.yaml")
    print("2. Adjust 'environment_analysis_robots: 50' for desired analysis depth")
    print("3. Run your SDS process normally - environment analysis will be automatic")
    print("")
    print("📋 What happens during SDS execution:")
    print("   → Environment analysis runs with specified number of robots")
    print("   → Real-time terrain data gets injected into the task descriptor prompt")
    print("   → GPT receives both video frames AND quantitative environment data")
    print("   → Enhanced SUS prompts are generated with terrain-specific context")
    
    return True

if __name__ == "__main__":
    test_environment_analysis_injection() 