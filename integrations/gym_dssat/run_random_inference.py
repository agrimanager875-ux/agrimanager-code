"""
Run random action baseline inference on gym-dssat environment.
Uses the gym DSSAT env directly (via gym.make) with random actions.
"""

import json
import os
import gym
import gym_dssat_pdi
import numpy as np
from pathlib import Path
import argparse

AGRIMANAGER_ROOT = Path(__file__).resolve().parents[2]


def _dssat_runtime_env_args() -> dict:
    """Return DSSAT runtime paths from env vars or repo-local defaults."""
    dssat_gym_path = Path(
        os.environ.get("DSSAT_GYM_PATH", AGRIMANAGER_ROOT / "spack" / "gym-dssat-pdi")
    )
    run_dssat = Path(
        os.environ.get("DSSAT_RUN_DSSAT", dssat_gym_path / "bin" / "run_dssat")
    )
    env_args = {"run_dssat_location": str(run_dssat)}

    profile = Path(
        os.environ.get("DSSAT_PROFILE_PATH", dssat_gym_path / "bin" / "DSSATPRO.L48")
    )
    if profile.is_file():
        env_args["auxiliary_file_paths"] = [str(profile)]
    return env_args


def run_random_inference(
    num_envs: int = 6,
    output_dir: str = "./random_results",
    seed: int = 42,
    max_steps: int = 200,
):
    """Run random action baseline on gym-dssat."""

    print("=" * 80)
    print("Random Action Baseline Inference (gym-dssat)")
    print("=" * 80)
    print(f"Environments: {num_envs}")
    print(f"Max steps: {max_steps}")
    print(f"Base seed: {seed}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    rng = np.random.RandomState(seed)

    results = []

    # Use same seeds as other evaluations for consistency
    eval_seeds = [2000, 2001, 2002, 2003, 2004, 2005]

    for env_idx in range(num_envs):
        env_seed = eval_seeds[env_idx] if env_idx < len(eval_seeds) else seed + env_idx
        print(f"\nEnvironment {env_idx + 1}/{num_envs} (seed={env_seed})")

        env = gym.make(
            'GymDssatPdi-v0',
            **_dssat_runtime_env_args(),
            mode='all',
            seed=env_seed,
            random_weather=True,
        )
        obs = env.reset()

        episode_data = {
            'env_id': env_idx,
            'seed': env_seed,
            'turns': []
        }

        # Record initial state
        metrics = _extract_metrics(obs)
        metrics['reward'] = 0.0
        episode_data['turns'].append({
            'turn': 0,
            'action': None,
            'metrics': metrics,
        })

        cumulative_reward = 0.0
        done = False

        for step in range(1, max_steps + 1):
            if done:
                break

            # Random action: sample from action space ranges
            anfer = float(rng.uniform(0.0, 10.0))
            amir = float(rng.uniform(0.0, 10.0))
            action = {'anfer': anfer, 'amir': amir}

            obs, reward, done, info = env.step(action)
            # reward can be a list [fert_reward, irrig_reward], a number, or None
            if reward is None:
                total_reward = 0.0
            elif isinstance(reward, list):
                total_reward = sum(reward)
            else:
                total_reward = float(reward)
            cumulative_reward += total_reward

            metrics = _extract_metrics(obs)
            metrics['reward'] = cumulative_reward

            episode_data['turns'].append({
                'turn': step,
                'action': action,
                'metrics': metrics,
            })

        final = episode_data['turns'][-1]['metrics']
        print(f"  Final yield: {final.get('grnwt', 0):.0f} kg/ha, "
              f"DAP: {final.get('dap', 0)}, "
              f"Reward: {cumulative_reward:.2f}")

        results.append(episode_data)
        env.close()

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "gym_dssat_maize_phase1_test_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=_json_serialize)
    print(f"\nResults saved to: {output_file}")

    # Summary
    _print_summary(results)


def _extract_metrics(obs):
    """Extract key metrics from observation dict."""
    if obs is None:
        return {}
    m = {}
    for key in ['dap', 'grnwt', 'xlai', 'vstage', 'cumsumfert', 'totir',
                'swfac', 'nstres', 'topwt']:
        if key in obs:
            val = obs[key]
            m[key] = float(val) if hasattr(val, 'item') else val
    return m


def _json_serialize(obj):
    """Handle numpy types for JSON serialization."""
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    final_yields = []
    final_rewards = []
    for r in results:
        last = r['turns'][-1]['metrics']
        if 'grnwt' in last:
            final_yields.append(last['grnwt'])
        if 'reward' in last:
            final_rewards.append(last['reward'])

    if final_yields:
        print(f"Environments: {len(final_yields)}")
        print(f"\nFinal Grain Yield (kg/ha):")
        print(f"  Mean: {np.mean(final_yields):.2f}")
        print(f"  Std:  {np.std(final_yields):.2f}")
        print(f"  Min:  {np.min(final_yields):.2f}")
        print(f"  Max:  {np.max(final_yields):.2f}")

    if final_rewards:
        print(f"\nFinal Rewards:")
        print(f"  Mean: {np.mean(final_rewards):.2f}")
        print(f"  Std:  {np.std(final_rewards):.2f}")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random action baseline (gym-dssat)")
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="results/gym_dssat/maize_phase1/test/random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200)

    args = parser.parse_args()

    run_random_inference(
        num_envs=args.num_envs,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
    )
