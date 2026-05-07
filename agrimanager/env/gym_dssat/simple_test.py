"""
Simple test script for gym-dssat environment.

This script demonstrates how to create and use a gym-dssat environment
with the wrapper classes. It shows basic usage and outputs key information
to help users understand the environment.
"""

from agrimanager.env.dssat_gym import (
    DEFAULT_DSSAT_GYM_PATH,
    DSSATEnv,
    DSSATEnvConfig,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_observation_info(obs, env):
    """Print detailed observation information."""
    import numpy as np

    if isinstance(obs, str):
        # LLM-mode
        print("\nObservation is text. (LLM mode enabled)")
        print(obs[:500], "..." if len(obs) > 500 else "")
        return

    print(f"\nObservation type: {type(obs)}")

    if isinstance(obs, (list, tuple)):
        obs = np.array(obs)

    if hasattr(obs, "shape"):
        print(f"Observation shape: {obs.shape}")

    # Print first few entries
    try:
        output_vars = env.env.observation_variables
        print(f"\nObservation variables ({len(output_vars)}):")
        for i, var_name in enumerate(output_vars[:min(20, len(output_vars))]):
            if i < len(obs):
                print(f"  {i:2d}. {var_name:15s}: {obs[i]:10.4f}")
        if len(output_vars) > 20:
            print(f"  ... and {len(output_vars) - 20} more variables")
    except Exception as e:
        print(f"\nCould not retrieve variable names: {e}")
        print("First 10 observation values:\n", obs[:10])


def main():
    """Run a simple test of the gym-dssat environment."""

    print_section("gym-dssat Environment Simple Test")

    # ========================================================================
    # Step 1: Create Configuration
    # ========================================================================
    print_section("Step 1: Create Configuration")

    config = DSSATEnvConfig(
        env_id="maize-irrigation-v0",
        dssat_gym_path=DEFAULT_DSSAT_GYM_PATH,
        llm_mode=False,              # Numeric mode
        seed=42,
        dssat_params={},
        env_params={"mode": "all"},
        turn_num=200,
    )

    print("\nConfiguration:")
    print(f"  Environment ID: {config.env_id}")
    print(f"  DSSAT Gym Path: {config.dssat_gym_path}")
    print(f"  LLM Mode:       {config.llm_mode}")
    print(f"  Seed:           {config.seed}")
    print(f"  turn_num:       {config.turn_num}")

    # ========================================================================
    # Step 2: Create Environment
    # ========================================================================
    print_section("Step 2: Create Environment")

    print("\nCreating environment...")
    env = DSSATEnv(config)
    print("✓ Environment created successfully!")

    # Print environment info (from underlying gym env)
    print(f"\nAction Space:      {env.env.action_space}")
    print(f"Observation Space: {env.env.observation_space}")

    # ========================================================================
    # Step 3: Get System Prompt
    # ========================================================================
    print_section("Step 3: System Prompt")

    print(env.system_prompt())

    # ========================================================================
    # Step 4: Reset Environment
    # ========================================================================
    print_section("Step 4: Reset Environment")

    obs, info = env.reset()
    print("✓ Environment reset successfully!")
    print_observation_info(obs, env)
    print(f"\nInfo keys: {list(info.keys())}")

    # ========================================================================
    # Step 5: Take Random Actions
    # ========================================================================
    print_section("Step 5: Take Random Actions (10 steps)")

    total_reward = 0.0

    for step in range(10):
        action = env.env.action_space.sample()
        obs, reward, done, info = env.step(action)

        total_reward += reward

        print(f"\nStep {step + 1}:")
        print(f"  Action:        {action}")
        print(f"  Reward:        {reward:.4f}")
        print(f"  Done:          {done}")
        print(f"  Total Reward:  {total_reward:.4f}")

        if not isinstance(obs, str):
            try:
                output_vars = env.env.observation_variables
                if "grnwt" in output_vars:
                    idx = output_vars.index("grnwt")
                    print(f"    grnwt: {obs[idx]:.4f}")
                if "swfac" in output_vars:
                    idx = output_vars.index("swfac")
                    print(f"    swfac: {obs[idx]:.4f}")
            except:
                pass

        if done:
            print("\n✓ Episode finished early!")
            break

    # ========================================================================
    # Step 6: Summary
    # ========================================================================
    print_section("Summary")

    print("\nEnvironment Test Summary:")
    print("  ✓ Configuration created")
    print("  ✓ Environment initialized")
    print("  ✓ Reset works")
    print("  ✓ Step works")
    print("  ✓ System prompt printed")
    print("  ✓ Random policy runs without errors")

    # ========================================================================
    # Step 7: Test LLM Mode
    # ========================================================================
    print_section("Step 7: Test LLM Mode")

    config_llm = DSSATEnvConfig(
        env_id="maize-irrigation-v0",
        dssat_gym_path=DEFAULT_DSSAT_GYM_PATH,
        llm_mode=True,    # Enable natural-language mode
        seed=42,
        env_params={"mode": "all"},
    )

    env_llm = DSSATEnv(config_llm)
    print("✓ LLM-mode env created!")

    system_prompt = env_llm.system_prompt()
    print("\nSystem Prompt:")
    print(system_prompt)

    turn_prompt, info = env_llm.reset()
    print("\nTurn Prompt:\n")
    print(turn_prompt[:700], "..." if len(turn_prompt) > 700 else "")

    # Test example LLM responses
    print("\nTesting example LLM responses:")
    examples = [
        "<answer>Take no action.<answer>",
        "<answer>Apply 20 kg/ha nitrogen.<answer>",
        "<answer>Irrigate with 20 mm water.<answer>",
    ]

    for txt in examples:
        print(f"\nLLM response: {txt}")
        try:
            turn_prompt, reward, done, info = env_llm.step(txt)
            print(f"  Reward: {reward:.4f}")
            print(f"  Done:   {done}")
        except Exception as e:
            print(f"  Error: {e}")

    env_llm.close()
    print("✓ LLM environment closed")

    # ========================================================================
    # Step 8: Cleanup
    # ========================================================================
    print_section("Step 8: Cleanup")

    env.close()
    print("\n✓ All environments closed successfully!")

    print("\n" + "=" * 80)
    print("  Test completed successfully! 🎉")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
