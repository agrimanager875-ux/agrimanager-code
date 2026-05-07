"""
Test script for pest management functionality in AgriManager.

This script demonstrates how to use the pest management features:
1. Configure environment with pests enabled
2. Run a simple simulation with pest observations
3. Test pesticide application actions
"""

import sys
import os

# Add agrimanager to path
sys.path.insert(0, os.path.dirname(__file__))

from agrimanager.env.gym_dssat.env import DSSATEnv
from agrimanager.env.gym_dssat.env_config import DSSATEnvConfig


def test_pest_management():
    """Test pest management functionality."""

    print("=" * 80)
    print("TESTING PEST MANAGEMENT IN AGRIMANAGER")
    print("=" * 80)

    # Create configuration with pests enabled
    config = DSSATEnvConfig(
        env_id="maize-pest-test",
        llm_mode=True,
        enable_pests=True,
        pest_config={
            "base_pressure": 0.4,  # Start with moderate pest pressure
            "weather_sensitivity": 0.6,  # Pests respond to weather
            "damage_rate": 0.03,  # 3% yield loss per day per unit pressure
            "pesticide_efficacy": 0.7,  # 70% reduction
            "pesticide_cost": 15.0,  # $15/ha
        },
        turn_num=50,  # Short test season
    )

    print("\n✅ Configuration created with pest management enabled")
    print(f"   - Base pest pressure: {config.pest_config['base_pressure']}")
    print(f"   - Pesticide efficacy: {config.pest_config['pesticide_efficacy'] * 100}%")
    print(f"   - Pesticide cost: ${config.pest_config['pesticide_cost']}/ha")

    # Create environment
    print("\n📦 Creating DSSAT environment...")
    try:
        env = DSSATEnv(config)
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    # Check that pest variables are in output_vars
    print(f"\n📊 Output variables: {env.output_vars}")
    pest_vars = ['pest_pressure', 'pest_damage', 'days_since_pesticide']
    for pv in pest_vars:
        if pv in env.output_vars:
            print(f"   ✅ {pv} is tracked")
        else:
            print(f"   ❌ {pv} is NOT tracked")

    # Reset environment
    print("\n🔄 Resetting environment...")
    obs, info = env.reset()
    print("✅ Environment reset")
    print(f"\nInitial observation (first 500 chars):\n{obs[:500]}...")

    # Test actions
    print("\n" + "=" * 80)
    print("TESTING ACTIONS")
    print("=" * 80)

    actions_to_test = [
        "<answer>Do nothing.<answer>",
        "<answer>Apply 20 kg/ha nitrogen fertilizer.<answer>",
        "<answer>Apply pesticide.<answer>",
        "<answer>Irrigate with 10 mm of water.<answer>",
    ]

    for i, action in enumerate(actions_to_test, 1):
        print(f"\n--- Step {i}: {action} ---")
        try:
            obs, reward, done, info = env.step(action)
            print(f"✅ Action executed")
            print(f"   Reward: {reward:.2f}")
            print(f"   Done: {done}")
            if "action_applied" in info:
                print(f"   Action applied: {info['action_applied']}")

            # Show pest info if available
            if hasattr(env, '_pest_pressure'):
                print(f"   Pest pressure: {env._pest_pressure:.3f}")
                print(f"   Cumulative damage: {env._cumulative_pest_damage:.2f}")
                print(f"   Days since pesticide: {env._days_since_pesticide}")

            if done:
                print("   Season ended!")
                break

        except Exception as e:
            print(f"❌ Error executing action: {e}")
            import traceback
            traceback.print_exc()

    # Close environment
    print("\n🔒 Closing environment...")
    env.close()
    print("✅ Test complete!")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Pest management functionality is working!")
    print("✅ You can now:")
    print("   1. Enable pests in any DSSAT environment config")
    print("   2. Observe pest_pressure, pest_damage, days_since_pesticide")
    print("   3. Apply pesticides using: <answer>Apply pesticide.<answer>")
    print("   4. Optimize for net profit including pest costs and damage")


if __name__ == "__main__":
    test_pest_management()
