"""
Run PPO agent inference on test dataset for comparison with LLM agents.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

agrimanager_root = Path(__file__).parent.parent.parent

from agrimanager.env.base import create_environment, load_dataset_configs


def load_ppo_agent(checkpoint_path):
    """Load a trained PPO agent from checkpoint."""
    from stable_baselines3 import PPO
    
    print(f"Loading PPO agent from: {checkpoint_path}")
    model = PPO.load(checkpoint_path)
    return model


def dict_to_array(obs_dict):
    """
    Convert dict observation to array matching PPO's expected 25 features.
    
    Order: 16 scalar features + 9 soil layer features = 25 total
    """
    # Ordered scalar keys (16 features)
    ordered_keys = [
        'dap', 'cumsumfert', 'swfac', 'vstage', 'grnwt', 'xlai', 
        'tmax', 'srad', 'dtt', 'ep', 'istage', 'nstres', 'rtdep',
        'topwt', 'totir', 'wtdep'
    ]
    
    values = []
    
    # Extract scalar values
    for key in ordered_keys:
        if key in obs_dict:
            values.append(float(obs_dict[key]))
        else:
            values.append(0.0)
    
    # Extract soil water array (9 soil layers)
    if 'sw' in obs_dict:
        sw = obs_dict['sw']
        if hasattr(sw, '__len__'):
            values.extend([float(x) for x in sw[:9]])  # Ensure only 9 values
        else:
            values.extend([float(sw)] * 9)
    else:
        values.extend([0.0] * 9)
    
    result = np.array(values, dtype=np.float32)
    
    # Verify shape
    if result.shape[0] != 25:
        raise ValueError(f"Expected 25 features, got {result.shape[0]}")
    
    return result


def action_to_dict(action, anfer_max=10.0, amir_max=10.0):
    """
    Convert PPO action array to DSSAT action dict.
    Reduced max values to prevent over-application.
    """
    if isinstance(action, np.ndarray):
        if action.shape == (2,):
            # Denormalize from [-1, 1] to [0, max] and clip
            anfer = np.clip((action[0] + 1.0) / 2.0 * anfer_max, 0.0, anfer_max)
            amir = np.clip((action[1] + 1.0) / 2.0 * amir_max, 0.0, amir_max)
            return {
                "anfer": float(anfer),
                "amir": float(amir)
            }


def extract_metrics_from_obs(obs_dict):
    """Extract key metrics from observation dict for tracking."""
    return {
        'reward': 0.0,  # Will be filled by actual reward
        'dap': obs_dict.get('dap', 0),
        'grnwt': obs_dict.get('grnwt', 0),
        'xlai': obs_dict.get('xlai', 0),
        'cumsumfert': obs_dict.get('cumsumfert', 0),
        'totir': obs_dict.get('totir', 0),
    }


def run_ppo_inference(
    env_name: str,
    dataset_id: str,
    split: str,
    ppo_checkpoint: str,
    output_dir: str,
):
    """Run PPO agent inference on test dataset."""
    
    print("=" * 80)
    print("PPO Agent Inference")
    print("=" * 80)
    print(f"Environment: {env_name}")
    print(f"Dataset: {dataset_id} ({split})")
    print(f"PPO Checkpoint: {ppo_checkpoint}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Load PPO agent
    print("\n[1/4] Loading PPO agent...")
    agent = load_ppo_agent(ppo_checkpoint)
    print(f"PPO observation space: {agent.observation_space}")
    print(f"PPO action space: {agent.action_space}")
    
    # Load dataset
    print("\n[2/4] Loading dataset...")
    env_configs, _ = load_dataset_configs(
        env_name=env_name,
        dataset_id=dataset_id,
        split=split,
        repo_root=agrimanager_root,
    )
    num_envs = len(env_configs)
    print(f"Loaded {num_envs} environment configurations")
    
    # Run evaluation on each environment
    print(f"\n[3/4] Running PPO inference on {num_envs} environments...")
    
    results = []
    
    for env_idx, env_config in enumerate(tqdm(env_configs, desc="Evaluating")):
        # Create environment
        env, config = create_environment(env_name, env_config)
        
        # CRITICAL: Disable LLM mode to get raw observations
        if hasattr(env, 'llm_mode'):
            env.llm_mode = False
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'llm_mode'):
            env.unwrapped.llm_mode = False
        
        # Run episode
        obs_raw, info = env.reset()
        
        # Convert dict to array (25 features)
        if isinstance(obs_raw, dict):
            obs = dict_to_array(obs_raw)
        else:
            obs = np.array(obs_raw, dtype=np.float32)
        
        done = False
        turn = 0
        cumulative_reward = 0.0
        
        episode_data = {
            'env_id': env_idx,
            'env_config': env_config,
            'turns': []
        }
        
        # Record initial state
        metrics = extract_metrics_from_obs(obs_raw) if isinstance(obs_raw, dict) else {}
        metrics['reward'] = 0.0
        episode_data['turns'].append({
            'turn': 0,
            'action': None,
            'metrics': metrics,
        })
        
        # Run episode
        while not done:
            turn += 1
            
            # Get action from PPO agent
            action, _ = agent.predict(obs, deterministic=True)
            
            # Convert action to dict format for DSSAT
            action_dict = action_to_dict(action)
            
            # Step environment with dict action
            obs_raw, reward, done, info = env.step(action_dict)
            cumulative_reward += reward
            
            # Convert dict to array (25 features)
            if isinstance(obs_raw, dict):
                obs = dict_to_array(obs_raw)
            else:
                obs = np.array(obs_raw, dtype=np.float32)
            
            # Extract metrics from observation
            metrics = extract_metrics_from_obs(obs_raw) if isinstance(obs_raw, dict) else {}
            metrics['reward'] = cumulative_reward
            
            episode_data['turns'].append({
                'turn': turn,
                'action': action.tolist() if hasattr(action, 'tolist') else [float(action)] if np.isscalar(action) else [float(x) for x in action],
                'action_dict': action_dict,  # Also store the denormalized action
                'metrics': metrics,
            })
            
            if done:
                break
        
        results.append(episode_data)
        env.close()
    
    # Save results
    print(f"\n[4/4] Saving results...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = Path(ppo_checkpoint).stem
    output_file = output_path / f"{env_name}_{dataset_id}_{split}_ppo_{checkpoint_name}_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    final_rewards = []
    final_yields = []
    for result in results:
        if len(result['turns']) > 0:
            last_turn = result['turns'][-1]
            if 'metrics' in last_turn:
                if 'reward' in last_turn['metrics']:
                    final_rewards.append(last_turn['metrics']['reward'])
                if 'grnwt' in last_turn['metrics']:
                    final_yields.append(last_turn['metrics']['grnwt'])
    
    if len(final_rewards) > 0:
        print(f"Environments evaluated: {len(final_rewards)}")
        print(f"\nFinal Rewards:")
        print(f"  Mean: {np.mean(final_rewards):.2f}")
        print(f"  Std:  {np.std(final_rewards):.2f}")
        print(f"  Min:  {np.min(final_rewards):.2f}")
        print(f"  Max:  {np.max(final_rewards):.2f}")
    
    if len(final_yields) > 0:
        print(f"\nFinal Grain Yield (kg/ha):")
        print(f"  Mean: {np.mean(final_yields):.2f}")
        print(f"  Std:  {np.std(final_yields):.2f}")
    
    print("=" * 80)
    print("\n✓ PPO inference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO agent inference")
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ppo-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./ppo_results")
    
    args = parser.parse_args()
    
    run_ppo_inference(
        env_name=args.env_name,
        dataset_id=args.dataset_id,
        split=args.split,
        ppo_checkpoint=args.ppo_checkpoint,
        output_dir=args.output_dir,
    )
