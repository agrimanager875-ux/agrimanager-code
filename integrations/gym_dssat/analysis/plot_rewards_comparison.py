"""
Plot rewards comparison across different policies/models.
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def extract_dap_from_prompt(prompt_text):
    """Extract DAP from turn prompt text."""
    if not prompt_text:
        return None
    dap_match = re.search(r'day (\d+) after planting', str(prompt_text))
    if dap_match:
        return float(dap_match.group(1))
    return None


def load_model_results(results_dir):
    """Load all model results and extract rewards."""
    results_dir = Path(results_dir)
    model_data = {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        json_files = list(model_dir.glob("*.json"))
        if not json_files:
            continue

        with open(json_files[0]) as f:
            results = json.load(f)

        final_rewards = []
        reward_curves = []

        for env_result in results:
            rewards = []
            daps = []
            for turn_data in env_result.get('turns', []):
                metrics = turn_data.get('metrics', {})
                reward = None
                dap = None

                # Get reward from metrics
                if 'reward' in metrics:
                    reward = float(metrics['reward'])

                # Get dap from metrics or turn_prompt
                if 'dap' in metrics:
                    dap = float(metrics['dap'])
                else:
                    # Try to extract from turn_prompt
                    dap = extract_dap_from_prompt(turn_data.get('turn_prompt', ''))

                # Add if both are available
                if reward is not None and dap is not None:
                    rewards.append(reward)
                    daps.append(dap)

            if rewards and len(daps) == len(rewards) and len(rewards) > 0:
                rewards_arr = np.array(rewards)

                # Check if rewards are cumulative (monotonically increasing after initial)
                # If not, accumulate them
                if len(rewards_arr) > 10:
                    # Check if rewards increase monotonically (allowing small dips)
                    diffs = np.diff(rewards_arr[5:])  # Skip first few turns
                    is_cumulative = np.sum(diffs < -10) < len(diffs) * 0.1  # Less than 10% decreases

                    if not is_cumulative:
                        # Convert step rewards to cumulative
                        rewards_arr = np.cumsum(rewards_arr)

                final_rewards.append(rewards_arr[-1])
                reward_curves.append((np.array(daps), rewards_arr))

        if final_rewards:
            model_data[model_name] = {
                'final_rewards': final_rewards,
                'reward_curves': reward_curves
            }

    return model_data


def plot_rewards_comparison(model_data, output_path):
    """Create a comparison plot of rewards across models."""
    if not model_data:
        print("No reward data found!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(model_data), 8)))

    # --- Left panel: reward curves over time ---
    ax = axes[0]
    for idx, (model_name, data) in enumerate(sorted(model_data.items())):
        curves = data['reward_curves']
        if not curves:
            continue

        # Interpolate to common DAP grid
        all_daps = np.concatenate([d for (d, _) in curves])
        dap_min, dap_max = int(np.min(all_daps)), int(np.max(all_daps))
        dap_grid = np.arange(dap_min, dap_max + 1)

        interp_rewards = []
        for (daps, rewards) in curves:
            if len(daps) >= 2:
                interp = np.interp(dap_grid, daps, rewards)
                interp_rewards.append(interp)

        if interp_rewards:
            mean_reward = np.mean(interp_rewards, axis=0)
            std_reward = np.std(interp_rewards, axis=0)
            color = colors[idx % len(colors)]
            ax.plot(dap_grid, mean_reward, label=model_name, color=color, linewidth=2)
            ax.fill_between(dap_grid, mean_reward - std_reward, mean_reward + std_reward,
                            alpha=0.15, color=color)

    ax.set_xlabel('Days After Planting', fontsize=11)
    ax.set_ylabel('Cumulative Reward', fontsize=11)
    ax.set_title('Reward Accumulation Over Time (mean ± std)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right panel: final reward bar chart ---
    ax = axes[1]
    names = sorted(model_data.keys())
    means = [np.mean(model_data[n]['final_rewards']) for n in names]
    stds = [np.std(model_data[n]['final_rewards']) for n in names]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, means, xerr=stds, height=0.6, capsize=4,
                   color=[colors[i % len(colors)] for i in range(len(names))])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Final Cumulative Reward', fontsize=11)
    ax.set_title('Final Reward Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 50, i, f'{m:.0f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Print table
    print(f"\n{'Model':<30} {'Mean Reward':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 72)
    for name in names:
        fr = model_data[name]['final_rewards']
        print(f"{name:<30} {np.mean(fr):>12.1f} {np.std(fr):>10.1f} "
              f"{np.min(fr):>10.1f} {np.max(fr):>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rewards comparison across models")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="rewards_comparison.png")
    args = parser.parse_args()

    model_data = load_model_results(args.results_dir)
    if not model_data:
        print(f"No reward data found in {args.results_dir}")
        exit(1)

    print(f"Found reward data for {len(model_data)} models: {', '.join(sorted(model_data.keys()))}")
    plot_rewards_comparison(model_data, args.output)
