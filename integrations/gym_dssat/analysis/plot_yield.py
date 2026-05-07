import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def extract_observation_from_prompt(prompt_text):
    """Extract observation values from the turn prompt text."""
    if not prompt_text:
        return {}
    
    obs = {}
    
    # Extract day after planting
    dap_match = re.search(r'day (\d+) after planting', prompt_text)
    if dap_match:
        obs['dap'] = float(dap_match.group(1))
    
    # Extract metrics from observation summary
    lines = prompt_text.split('\n')
    for line in lines:
        # Pattern: "- Description: value"
        if line.strip().startswith('- '):
            parts = line.split(':')
            if len(parts) == 2:
                value_str = parts[1].strip()
                try:
                    value = float(value_str)
                    
                    # Map descriptions to variable names
                    if 'Vegetative stage index' in parts[0]:
                        obs['vstage'] = value
                    elif 'Grain weight' in parts[0]:
                        obs['grnwt'] = value
                    elif 'Leaf area index' in parts[0]:
                        obs['xlai'] = value
                    elif 'Total aboveground biomass' in parts[0]:
                        obs['topwt'] = value
                    elif 'Nitrogen stress' in parts[0]:
                        obs['nstres'] = value
                    elif 'Soil water factor' in parts[0]:
                        obs['swfac'] = value
                    elif 'Total fertilizer applied' in parts[0]:
                        obs['cumsumfert'] = value
                    elif 'Growing degree days' in parts[0] or 'Thermal time' in parts[0]:
                        obs['dtt'] = value
                except:
                    pass
    
    return obs

def plot_checkpoint_metrics(results_file):
    """Plot rewards and growth metrics for all environments."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    num_envs = len(results)
    
    # Extract all data
    all_env_data = []
    for env_result in results:
        env_data = {
            'turns': [],
            'rewards': [],
            'dap': [],
            'grnwt': [],
            'xlai': [],
            'vstage': [],
            'cumsumfert': [],
            'swfac': [],
            'nstres': []
        }
        
        for turn_data in env_result['turns']:
            turn = turn_data['turn']
            env_data['turns'].append(turn)
            
            # Get reward
            reward = 0.0
            if 'metrics' in turn_data and 'reward' in turn_data['metrics']:
                reward = turn_data['metrics']['reward']
            env_data['rewards'].append(reward)
            
            # Extract observations: try metrics dict first, fall back to prompt parsing
            metrics = turn_data.get('metrics', {})
            prompt = turn_data.get('turn_prompt', '')
            obs = extract_observation_from_prompt(prompt) if prompt else {}

            # Metrics dict takes priority over prompt parsing
            for key in ('dap', 'grnwt', 'xlai', 'vstage', 'cumsumfert', 'swfac', 'nstres'):
                if key in metrics:
                    obs[key] = metrics[key]

            env_data['dap'].append(obs.get('dap', np.nan))
            env_data['grnwt'].append(obs.get('grnwt', np.nan))
            env_data['xlai'].append(obs.get('xlai', np.nan))
            env_data['vstage'].append(obs.get('vstage', np.nan))
            env_data['cumsumfert'].append(obs.get('cumsumfert', np.nan))
            env_data['swfac'].append(obs.get('swfac', np.nan))
            env_data['nstres'].append(obs.get('nstres', np.nan))
        
        all_env_data.append(env_data)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Checkpoint Performance Metrics (n={num_envs} environments)', fontsize=16)
    
    # Plot 1: Rewards
    ax = axes[0, 0]
    for env_data in all_env_data:
        ax.plot(env_data['turns'], env_data['rewards'], alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Reward')
    ax.set_title('Cumulative Reward')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Grain Weight (Yield)
    ax = axes[0, 1]
    for env_data in all_env_data:
        if not all(np.isnan(env_data['grnwt'])):
            ax.plot(env_data['dap'], env_data['grnwt'], alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Days After Planting')
    ax.set_ylabel('Grain Weight (kg/ha)')
    ax.set_title('Grain Yield Development')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Leaf Area Index
    ax = axes[1, 0]
    for env_data in all_env_data:
        if not all(np.isnan(env_data['xlai'])):
            ax.plot(env_data['dap'], env_data['xlai'], alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Days After Planting')
    ax.set_ylabel('LAI (m²/m²)')
    ax.set_title('Leaf Area Index')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Vegetative Stage
    ax = axes[1, 1]
    for env_data in all_env_data:
        if not all(np.isnan(env_data['vstage'])):
            ax.plot(env_data['dap'], env_data['vstage'], alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Days After Planting')
    ax.set_ylabel('Vegetative Stage (# leaves)')
    ax.set_title('Vegetative Growth Stage')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative Fertilizer
    ax = axes[2, 0]
    for env_data in all_env_data:
        if not all(np.isnan(env_data['cumsumfert'])):
            ax.plot(env_data['dap'], env_data['cumsumfert'], alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Days After Planting')
    ax.set_ylabel('Cumulative Fertilizer (kg/ha)')
    ax.set_title('Total Fertilizer Applied')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Stress Factors
    ax = axes[2, 1]
    # Plot water stress
    for env_data in all_env_data:
        if not all(np.isnan(env_data['swfac'])):
            ax.plot(env_data['dap'], env_data['swfac'], alpha=0.2, linewidth=0.5, color='blue')
    # Plot nitrogen stress
    for env_data in all_env_data:
        if not all(np.isnan(env_data['nstres'])):
            ax.plot(env_data['dap'], env_data['nstres'], alpha=0.2, linewidth=0.5, color='green')
    ax.set_xlabel('Days After Planting')
    ax.set_ylabel('Stress Factor (0-1)')
    ax.set_title('Water (blue) & Nitrogen (green) Stress')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = results_file.replace('.json', '_full_metrics.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    final_rewards = []
    final_yields = []
    final_lai = []
    total_fert = []
    
    for env_data in all_env_data:
        if len(env_data['rewards']) > 0:
            final_rewards.append(env_data['rewards'][-1])
        if len(env_data['grnwt']) > 0 and not np.isnan(env_data['grnwt'][-1]):
            final_yields.append(env_data['grnwt'][-1])
        if len(env_data['xlai']) > 0 and not np.isnan(env_data['xlai'][-1]):
            final_lai.append(env_data['xlai'][-1])
        if len(env_data['cumsumfert']) > 0 and not np.isnan(env_data['cumsumfert'][-1]):
            total_fert.append(env_data['cumsumfert'][-1])
    
    print(f"\nNumber of environments: {num_envs}")
    
    if len(final_rewards) > 0:
        print(f"\nFinal Rewards:")
        print(f"  Mean: {np.mean(final_rewards):.2f}")
        print(f"  Std:  {np.std(final_rewards):.2f}")
        print(f"  Min:  {np.min(final_rewards):.2f}")
        print(f"  Max:  {np.max(final_rewards):.2f}")
    
    if len(final_yields) > 0:
        print(f"\nFinal Grain Yield (kg/ha):")
        print(f"  Mean: {np.mean(final_yields):.2f}")
        print(f"  Std:  {np.std(final_yields):.2f}")
        print(f"  Min:  {np.min(final_yields):.2f}")
        print(f"  Max:  {np.max(final_yields):.2f}")
    
    if len(final_lai) > 0:
        print(f"\nFinal Leaf Area Index:")
        print(f"  Mean: {np.mean(final_lai):.3f}")
        print(f"  Std:  {np.std(final_lai):.3f}")
    
    if len(total_fert) > 0:
        print(f"\nTotal Fertilizer Applied (kg/ha):")
        print(f"  Mean: {np.mean(total_fert):.2f}")
        print(f"  Std:  {np.std(total_fert):.2f}")
        print(f"  Min:  {np.min(total_fert):.2f}")
        print(f"  Max:  {np.max(total_fert):.2f}")
    
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <results_json_file>")
        print("Example: python plot_metrics.py ./results/gym_dssat_maize_phase1_test_results.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    plot_checkpoint_metrics(results_file)