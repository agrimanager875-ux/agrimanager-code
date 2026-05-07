import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_checkpoint_rewards(results_file):
    """Plot rewards for all environments in a checkpoint."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract rewards for each environment
    num_envs = len(results)
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Individual environment rewards over time
    ax1 = axes[0]
    for env_result in results:
        env_id = env_result['env_id']
        rewards = []
        turns = []
        
        for turn_data in env_result['turns']:
            if 'metrics' in turn_data and 'reward' in turn_data['metrics']:
                turns.append(turn_data['turn'])
                rewards.append(turn_data['metrics']['reward'])
        
        if len(rewards) > 0:
            ax1.plot(turns, rewards, alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('Turn')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'Individual Environment Rewards (n={num_envs})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average reward across all environments
    ax2 = axes[1]
    
    # Collect all rewards by turn
    max_turns = max(len(env['turns']) for env in results)
    rewards_by_turn = [[] for _ in range(max_turns)]
    
    for env_result in results:
        for turn_data in env_result['turns']:
            turn = turn_data['turn']
            if 'metrics' in turn_data and 'reward' in turn_data['metrics']:
                rewards_by_turn[turn].append(turn_data['metrics']['reward'])
    
    # Calculate mean and std
    turns = []
    mean_rewards = []
    std_rewards = []
    
    for turn, rewards in enumerate(rewards_by_turn):
        if len(rewards) > 0:
            turns.append(turn)
            mean_rewards.append(np.mean(rewards))
            std_rewards.append(np.std(rewards))
    
    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)
    
    ax2.plot(turns, mean_rewards, 'b-', linewidth=2, label='Mean')
    ax2.fill_between(turns, 
                      mean_rewards - std_rewards, 
                      mean_rewards + std_rewards, 
                      alpha=0.3, label='±1 std')
    ax2.set_xlabel('Turn')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Average Reward Across All Environments')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = results_file.replace('.json', '_rewards.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Reward Summary Statistics")
    print("="*60)
    
    final_rewards = []
    for env_result in results:
        if len(env_result['turns']) > 0:
            last_turn = env_result['turns'][-1]
            if 'metrics' in last_turn and 'reward' in last_turn['metrics']:
                final_rewards.append(last_turn['metrics']['reward'])
    
    if len(final_rewards) > 0:
        print(f"Number of environments: {len(final_rewards)}")
        print(f"Mean final reward: {np.mean(final_rewards):.2f}")
        print(f"Std final reward: {np.std(final_rewards):.2f}")
        print(f"Min final reward: {np.min(final_rewards):.2f}")
        print(f"Max final reward: {np.max(final_rewards):.2f}")
        print(f"Median final reward: {np.median(final_rewards):.2f}")
    
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_rewards.py <results_json_file>")
        print("Example: python plot_rewards.py ./results/gym_dssat_maize_phase1_test_results.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    plot_checkpoint_rewards(results_file)