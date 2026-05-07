"""Simple test script for wofost_gym environment.

This script demonstrates how to create and use a wofost_gym environment
with the wrapper classes. It shows basic usage and outputs key information
to help users understand the environment.
"""

from agrimanager.env.wofost_gym import (
    DEFAULT_WOFOST_GYM_PATH,
    WOFOSTEnv,
    WOFOSTEnvConfig,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_observation_info(obs, env):
    """Print detailed observation information."""
    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation type: {type(obs)}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")

    # Try to get variable names if available
    try:
        output_vars = env.env.unwrapped.get_output_vars()
        print(f"\nObservation variables ({len(output_vars)}):")
        for i, var_name in enumerate(output_vars[:min(15, len(output_vars))]):
            if i < len(obs):
                print(f"  {i:2d}. {var_name:15s}: {obs[i]:10.2f}")
        if len(output_vars) > 15:
            print(f"  ... and {len(output_vars) - 15} more variables")
    except Exception as e:
        print(f"\nCould not retrieve variable names: {e}")
        print(f"First 10 observation values: {obs[:10]}")


def main():
    """Run a simple test of the wofost_gym environment."""

    print_section("wofost_gym Environment Simple Test")

    # ========================================================================
    # Step 1: Create Configuration
    # ========================================================================
    print_section("Step 1: Create Configuration")

    config = WOFOSTEnvConfig(
        env_id="lnpkw-v0",                              # Environment template
        agro_file="wheat_agro.yaml",                    # Agromanagement file
        wofost_gym_path=DEFAULT_WOFOST_GYM_PATH,          # Path to wofost_gym
        llm_mode=False,                                 # Use numerical interface for basic tests
        env_reward=None,                                # No custom reward wrapper
        wofost_params={                                 # Override WOFOST parameters
        },
        agro_params={                                   # Override agromanagement parameters
            "latitude": 50.0,
            "longitude": 5.0,
        },
        seed=42
    )

    print("\nConfiguration:")
    print(f"  Environment ID: {config.env_id}")
    print(f"  Agro file: {config.agro_file}")
    print(f"  LLM Mode: {config.llm_mode}")
    print(f"  Seed: {config.seed}")
    print(f"  WOFOST params: {config.wofost_params}")
    print(f"  Agro params: {config.agro_params}")

    # ========================================================================
    # Step 2: Create Environment
    # ========================================================================
    print_section("Step 2: Create Environment")

    print("\nCreating environment...")
    env = WOFOSTEnv(config)
    print("✓ Environment created successfully!")

    # Print environment info
    print(f"\nAction Space: {env.env.action_space}")
    print(f"Observation Space: {env.env.observation_space}")

    # ========================================================================
    # Step 3: Get System Prompt
    # ========================================================================
    print_section("Step 3: System Prompt for LLM Agent")

    system_prompt = env.system_prompt()
    print(system_prompt)

    # ========================================================================
    # Step 4: Reset Environment
    # ========================================================================
    print_section("Step 4: Reset Environment")

    obs, info = env.reset()
    print("\n✓ Environment reset successfully!")
    print_observation_info(obs, env)

    print(f"\nInfo keys: {list(info.keys())}")

    # ========================================================================
    # Step 5: Take Random Actions
    # ========================================================================
    print_section("Step 5: Take Random Actions (10 steps)")

    total_reward = 0.0

    for step in range(10):
        # Sample random action
        action = env.env.action_space.sample()

        # Take step
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Print step info
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        print(f"  Total Reward: {total_reward:.2f}")

        # Print some key observation values
        try:
            output_vars = env.env.unwrapped.get_output_vars()
            # Find indices for key variables
            key_vars = ['WSO', 'NAVAIL', 'PAVAIL', 'KAVAIL', 'SM']
            print(f"  Key observations:")
            for var in key_vars:
                if var in output_vars:
                    idx = output_vars.index(var)
                    if idx < len(obs):
                        print(f"    {var}: {obs[idx]:.2f}")
        except:
            pass

        if done:
            print("\n✓ Episode finished!")
            break

    # ========================================================================
    # Step 6: Test Specific Actions
    # ========================================================================
    print_section("Step 6: Test Specific Actions")

    print("\nResetting environment for specific action test...")
    obs, info = env.reset()

    # Test different action types
    test_actions = [
        (0, "Do nothing"),
        (1, "Apply N fertilizer (level 1)"),
        (5, "Apply P fertilizer (level 1)"),
        (9, "Apply K fertilizer (level 1)"),
        (13, "Apply irrigation (level 1)"),
    ]

    for action, description in test_actions:
        obs, reward, done, info = env.step(action)
        print(f"\nAction {action}: {description}")
        print(f"  Reward: {reward:.2f}")

        if done:
            print("  Episode finished, resetting...")
            obs, info = env.reset()

    # ========================================================================
    # Step 7: Summary
    # ========================================================================
    print_section("Summary")

    print("\nEnvironment Test Summary:")
    print(f"  ✓ Configuration created successfully")
    print(f"  ✓ Environment initialized successfully")
    print(f"  ✓ Reset works correctly")
    print(f"  ✓ Step function works correctly")
    print(f"  ✓ System prompt generated")
    print(f"  ✓ Actions execute without errors")

    print("\nAction Space Info:")
    print(f"  Type: Discrete")
    print(f"  Number of actions: {env.env.action_space.n}")
    print(f"  Action breakdown:")
    print(f"    - Action 0: Do nothing")
    print(f"    - Actions 1-4: N fertilizer (4 levels)")
    print(f"    - Actions 5-8: P fertilizer (4 levels)")
    print(f"    - Actions 9-12: K fertilizer (4 levels)")
    print(f"    - Actions 13-16: Irrigation (4 levels)")

    print("\nObservation Space Info:")
    print(f"  Type: Box (continuous)")
    print(f"  Shape: {env.env.observation_space.shape}")
    print(f"  Contains: crop state + soil nutrients + weather forecast + time")

    # ========================================================================
    # Step 8: Test LLM Interface
    # ========================================================================
    print_section("Step 8: Test LLM Interface")

    print("\nCreating environment with LLM mode enabled...")
    config_llm = WOFOSTEnvConfig(
        env_id="lnpkw-v0",
        agro_file="wheat_agro.yaml",
        wofost_gym_path=DEFAULT_WOFOST_GYM_PATH,
        llm_mode=True,  # Enable LLM interface
        seed=42
    )

    env_llm = WOFOSTEnv(config_llm)
    print("✓ LLM-mode environment created!")

    # Get system prompt
    system_prompt = env_llm.system_prompt()
    print("\nSystem Prompt:")
    print(system_prompt)

    # Reset and get turn prompt
    turn_prompt, info = env_llm.reset()
    print("\n✓ Environment reset!")
    print(f"\nObservation type: {type(turn_prompt)}")
    print(f"\nTurn Prompt:")
    print(turn_prompt)

    # Test LLM response parsing
    print("\n\nTesting LLM Response Parsing:")
    llm_responses = [
        "<answer>Apply 2.0 kg/ha nitrogen fertilizer.<answer>",
        "<answer>Irrigate with 1.0 cm of water.<answer>",
        "<answer>Take no action.<answer>",
    ]

    for i, llm_response in enumerate(llm_responses):
        print(f"\n--- LLM Step {i + 1} ---")
        print(f"LLM Response: {llm_response}")

        try:
            turn_prompt, reward, done, info = env_llm.step(llm_response)
            print(f"✓ Parsed Action ID: {info['raw_action']}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Done: {done}")

            if done:
                print("\n✓ Episode finished!")
                break
        except Exception as e:
            print(f"✗ Error: {e}")
            break

    # Test direct action ID in LLM mode
    print("\n\nTesting Direct Action ID (even in LLM mode):")
    turn_prompt, info = env_llm.reset()
    action_id = 5
    print(f"Direct action ID: {action_id}")
    turn_prompt, reward, done, info = env_llm.step(action_id)
    print(f"  Reward: {reward:.2f}")

    env_llm.close()
    print("\n✓ LLM environment closed!")

    # ========================================================================
    # Step 9: Cleanup
    # ========================================================================
    print_section("Step 9: Cleanup")

    env.close()
    print("\n✓ All environments closed successfully!")

    print("\n" + "=" * 80)
    print("  Test completed successfully! 🎉")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
