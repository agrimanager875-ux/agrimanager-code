"""
Plot yield comparison across all evaluated models.

Usage:
    python integrations/gym_dssat/analysis/plot_yield_comparison.py \
        --results-dir results/gym_dssat/maize_phase1/test \
        --output plot_yield_comparison.png
"""

import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_obs_from_prompt(prompt_text):
    """Extract observation values from turn prompt text."""
    if not prompt_text:
        return {}
    obs = {}
    dap_match = re.search(r'day (\d+) after planting', prompt_text)
    if dap_match:
        obs['dap'] = float(dap_match.group(1))
    lines = prompt_text.split('\n')
    for line in lines:
        if line.strip().startswith('- '):
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    value = float(parts[1].strip())
                    if 'Grain weight' in parts[0]:
                        obs['grnwt'] = value
                except (ValueError, IndexError):
                    pass
    return obs


def load_model_results(results_dir):
    """Load all model results from a results directory.

    Returns dict mapping model_name -> list of per-env yield timeseries.
    Each timeseries is a list of (dap, grnwt) tuples.
    """
    results_dir = Path(results_dir)
    model_data = {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # Find results JSON
        json_files = list(model_dir.glob("*.json"))
        if not json_files:
            continue
        results_file = json_files[0]

        with open(results_file) as f:
            results = json.load(f)

        env_timeseries = []
        for env_result in results:
            daps = []
            yields = []
            for turn_data in env_result.get('turns', []):
                # Try metrics dict first (PPO/random baselines)
                metrics = turn_data.get('metrics', {})
                if 'grnwt' in metrics and 'dap' in metrics:
                    daps.append(float(metrics['dap']))
                    yields.append(float(metrics['grnwt']))
                else:
                    # Fall back to prompt parsing (LLM models)
                    obs = extract_obs_from_prompt(turn_data.get('turn_prompt', ''))
                    if 'dap' in obs and 'grnwt' in obs:
                        daps.append(obs['dap'])
                        yields.append(obs['grnwt'])
            if daps:
                env_timeseries.append((np.array(daps), np.array(yields)))

        if env_timeseries:
            model_data[model_name] = env_timeseries

    return model_data


def plot_yield_comparison(model_data, output_path):
    """Create a comparison plot of yield curves across models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(model_data), 10)))

    # --- Left panel: mean yield curves ---
    ax = axes[0]
    final_yields = {}

    for idx, (model_name, env_series) in enumerate(sorted(model_data.items())):
        # Collect final yields
        finals = [ys[-1] for (_, ys) in env_series]
        final_yields[model_name] = finals

        # Compute mean yield curve (interpolate to common DAP grid)
        all_daps = np.concatenate([d for (d, _) in env_series])
        dap_min, dap_max = int(np.min(all_daps)), int(np.max(all_daps))
        dap_grid = np.arange(dap_min, dap_max + 1)

        interp_yields = []
        for (daps, ys) in env_series:
            if len(daps) >= 2:
                interp = np.interp(dap_grid, daps, ys)
                interp_yields.append(interp)

        if interp_yields:
            mean_yield = np.mean(interp_yields, axis=0)
            std_yield = np.std(interp_yields, axis=0)
            color = colors[idx % len(colors)]
            ax.plot(dap_grid, mean_yield, label=model_name, color=color, linewidth=2)
            ax.fill_between(dap_grid, mean_yield - std_yield, mean_yield + std_yield,
                            alpha=0.15, color=color)

    ax.set_xlabel('Days After Planting')
    ax.set_ylabel('Grain Weight (kg/ha)')
    ax.set_title('Yield Development (mean +/- std)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Right panel: final yield bar chart ---
    ax = axes[1]
    names = sorted(final_yields.keys())
    means = [np.mean(final_yields[n]) for n in names]
    stds = [np.std(final_yields[n]) for n in names]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, means, xerr=stds, height=0.6, capsize=3,
                   color=[colors[i % len(colors)] for i in range(len(names))])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Final Grain Yield (kg/ha)')
    ax.set_title('Final Yield Comparison')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 20, i, f'{m:.0f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Print table
    print(f"\n{'Model':<40} {'Mean Yield':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 82)
    for name in names:
        fy = final_yields[name]
        print(f"{name:<40} {np.mean(fy):>12.1f} {np.std(fy):>10.1f} "
              f"{np.min(fy):>10.1f} {np.max(fy):>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot yield comparison across models")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing per-model result subdirectories")
    parser.add_argument("--output", type=str, default="yield_comparison.png",
                        help="Output plot path")
    args = parser.parse_args()

    model_data = load_model_results(args.results_dir)
    if not model_data:
        print(f"No results found in {args.results_dir}")
        exit(1)

    print(f"Loaded results for {len(model_data)} models: {', '.join(sorted(model_data.keys()))}")
    plot_yield_comparison(model_data, args.output)
