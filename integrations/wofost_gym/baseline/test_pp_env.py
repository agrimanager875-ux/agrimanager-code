#!/usr/bin/env python3
"""
Test script for pp-v0 (Potential Production) environment.

This script loads a dataset of environment configurations and tests the pp-v0
environment which represents the theoretical maximum yield (no resource limitations).
It uses the same per-turn results schema as random_rollout.py.

pp-v0 characteristics:
- Limitations: NONE
- N/P/K/Water: Unlimited (always optimal)
- Action space: Discrete(1) - Only 'do nothing' (action=0)
- Use case: Theoretical maximum yield baseline
"""

from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
import json
import yaml
import numpy as np

# Get agrimanager root
agrimanager_root = Path(__file__).parent.parent.parent.parent.parent

from agrimanager.env.base import create_environment as base_create_env


def _infer_env_name(dataset_path: Path, df) -> str:
    """Infer env_name from a dataset path, falling back to data_source."""
    parts = dataset_path.parts
    try:
        data_idx = parts.index("data")
        return parts[data_idx - 1]
    except (ValueError, IndexError):
        pass

    if "data_source" in df.columns and len(df) > 0:
        data_sources = df["data_source"].dropna()
        if len(data_sources) > 0:
            return str(data_sources.iloc[0])

    raise ValueError(
        f"Cannot parse env_name from path or data_source column: {dataset_path}"
    )


def load_dataset(test_files: str) -> tuple:
    """Load dataset configurations from parquet file.

    Returns:
        (configs, env_name) — env_name is parsed from the path.
    """
    import pandas as pd

    dataset_path = Path(test_files)
    if not dataset_path.is_absolute():
        dataset_path = agrimanager_root / dataset_path

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    env_name = _infer_env_name(dataset_path, df)
    configs = [
        row["extra_info"]["interaction_kwargs"]["env_config"]
        for _, row in df.iterrows()
    ]

    print(f"Loading dataset from: {dataset_path}")
    print(f"Loaded {len(configs)} environment configurations")

    return configs, env_name


def create_pp_env(env_name: str, env_config: Dict[str, Any]):
    """Create pp-v0 environment instance from configuration.

    Args:
        env_name: Environment name (should be 'wofost_gym')
        env_config: Environment configuration dictionary

    Returns:
        Environment instance with env_id='pp-v0' and llm_mode=False
    """
    # Force pp-v0 environment and gym interface
    env_config = env_config.copy()
    env_config['env_id'] = 'pp-v0'  # Override to use pp-v0
    env_config['llm_mode'] = False   # Use gym interface

    return base_create_env(env_name, env_config)


def run_pp_test(
    test_files: str,
    output_dir: str,
    intvn_interval: int | None = None,
):
    """Run pp-v0 test on dataset.

    pp-v0 always uses action=0 (do nothing) since there's only one action available.
    This represents the theoretical maximum yield with no resource constraints.

    Args:
        test_files: Path to test parquet file
        output_dir: Directory to save results
        intvn_interval: Override intervention interval (days between decisions)
    """
    print("=" * 80)
    print("pp-v0 (Potential Production) Test Configuration")
    print("=" * 80)
    print(f"Test files: {test_files}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print("\npp-v0 Environment:")
    print("  - Theoretical maximum yield (no resource constraints)")
    print("  - N/P/K/Water: Unlimited (always optimal)")
    print("  - Action space: Discrete(1) - Only 'do nothing' (action=0)")
    print("  - Use case: Upper bound baseline for comparison")
    print("=" * 80)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    env_configs, env_name = load_dataset(test_files)
    num_envs = len(env_configs)

    # Create pp-v0 environments
    print(f"\n[2/4] Creating {num_envs} pp-v0 environments...")
    envs = []
    configs = []

    for i, env_config in enumerate(tqdm(env_configs, desc="Creating pp-v0 envs")):
        if intvn_interval is not None:
            env_config = env_config.copy()
            env_config['intvn_interval'] = intvn_interval
        env, config = create_pp_env(env_name, env_config)
        envs.append(env)
        configs.append(config)

    # Get turn_num from first config
    turn_num = configs[0].turn_num
    print(f"Episodes will run for {turn_num} turns")
    print(f"Action space: {envs[0].env.action_space if hasattr(envs[0], 'env') else envs[0].action_space}")

    # Initialize storage for results
    results = []
    for i in range(num_envs):
        results.append({
            'env_id': i,
            'env_config': env_configs[i],
            'turns': []
        })

    # Reset all environments
    print(f"\n[3/4] Resetting all environments...")
    observations = []
    for i, env in enumerate(tqdm(envs, desc="Resetting")):
        _, info = env.reset()
        observations.append((info or {}).get('observation', {}))

    # Run pp-v0 test for turn_num steps
    # pp-v0 always uses action=0 (do nothing) - this is the only action
    print(f"\n[4/4] Running pp-v0 test for {turn_num} turns...")
    done_flags = [False] * num_envs
    for turn in tqdm(range(turn_num), desc="Turns"):
        active_indices = [i for i, done in enumerate(done_flags) if not done]
        if not active_indices:
            print(f"\nAll environments finished at turn {turn - 1}")
            break

        for i in active_indices:
            try:
                current_observation = observations[i]

                # pp-v0 always uses action=0 (do nothing)
                action_id = 0

                # Step environment
                _, reward, done, info = envs[i].step(action_id)
                info = info or {}
                metrics = info.get('turn_metrics', {})

                # Match inference/baseline unified schema:
                # pre-step observation -> action -> post-step metrics.
                turn_record = {
                    'turn': turn,
                    'turn_prompt': None,
                    'observation': current_observation,
                    'raw_llm_response': None,
                    'llm_reasoning': None,
                    'retries': 0,
                    'executed_action_id': int(info.get('executed_action_id', action_id)),
                    'invalid_action': False,
                    'reward': float(reward),
                    'done': bool(done),
                    'post_turn_metrics': metrics,
                }
                if done:
                    turn_record['trajectory_metrics'] = info.get('trajectory_metrics', {})
                results[i]['turns'].append(turn_record)

                if done:
                    done_flags[i] = True
                    observations[i] = None
                else:
                    observations[i] = info.get('observation', {})

            except Exception as e:
                print(f"\nError in env {i} at turn {turn}: {e}")
                # Store error info
                results[i]['turns'].append({
                    'turn': turn,
                    'turn_prompt': None,
                    'observation': observations[i],
                    'raw_llm_response': None,
                    'llm_reasoning': None,
                    'retries': 0,
                    'executed_action_id': None,
                    'invalid_action': None,
                    'reward': None,
                    'done': True,
                    'error': str(e),
                    'post_turn_metrics': {}
                })
                done_flags[i] = True
                observations[i] = None

        # Early stopping if all environments are done
        if all(done_flags):
            print(f"\nAll environments finished at turn {turn}")
            break

    # Save results
    print(f"\nSaving results to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    metrics = _compute_metrics(results)
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("pp-v0 Test Results Summary")
    print("=" * 80)
    print(f"Number of episodes: {metrics['num_envs']}")
    print(f"Average Final Yield (WSO): {metrics['final_wso_mean']:.1f} kg/ha")
    print(f"WSO Std:                   {metrics['final_wso_std']:.1f} kg/ha")
    print(f"Invalid rate:              {metrics['invalid_action_rate_mean']:.4f}")
    print(f"Average turns:             {metrics['avg_turns']:.2f}")
    print("=" * 80)

    print(f"Results saved to {results_file}")
    print(f"Metrics saved to {metrics_file}")
    print("\n✓ pp-v0 test completed!")


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics using the shared baseline results schema."""
    final_wsos = []
    total_turns_list = []
    per_crop: Dict[str, list] = {}

    for r in results:
        last_turn = r['turns'][-1] if r['turns'] else {}
        traj = last_turn.get('trajectory_metrics', {})
        wso = traj.get('final_wso', last_turn.get('post_turn_metrics', {}).get('wso', 0.0))
        final_wsos.append(wso)
        total_turns_list.append(len(r['turns']))

        crop = r.get('env_config', {}).get('crop_name', 'unknown')
        per_crop.setdefault(crop, []).append(wso)

    metrics = {
        'final_wso_mean': float(np.mean(final_wsos)),
        'final_wso_std': float(np.std(final_wsos)),
        'invalid_action_rate_mean': 0.0,
        'avg_turns': float(np.mean(total_turns_list)),
        'num_envs': len(results),
    }

    crop_summary = {}
    for crop, wsos in sorted(per_crop.items()):
        crop_summary[crop] = {
            'final_wso_mean': float(np.mean(wsos)),
            'final_wso_std': float(np.std(wsos)),
            'count': len(wsos),
        }
    metrics['per_crop'] = crop_summary

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run pp-v0 environment test")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    # All other args are optional overrides
    parser.add_argument("--test-files", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--intvn-interval", type=int)

    args = parser.parse_args()

    # Load YAML config as defaults
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # CLI overrides (only non-None values)
    test_files = args.test_files or cfg["test_files"]
    output_dir = args.output_dir or cfg["output_dir"]
    intvn_interval = args.intvn_interval if args.intvn_interval is not None else cfg.get("intvn_interval")

    run_pp_test(
        test_files=test_files,
        output_dir=output_dir,
        intvn_interval=intvn_interval,
    )


if __name__ == "__main__":
    main()
