"""
Run PPO agent inference with the updated_model_v1.zip model.
Outputs results in the same format as other baselines for comparison.
"""

import json
import os
import gym
import gym_dssat_pdi
import numpy as np
from pathlib import Path
from tqdm import tqdm
from stable_baselines3 import PPO

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

# Helpers for action normalization
def normalize_action(action_space_limits, action):
    """Normalize the action from [low, high] to [-1, 1]"""
    low, high = action_space_limits
    return 2.0 * ((action - low) / (high - low)) - 1.0

def denormalize_action(action_space_limits, action):
    """Denormalize the action from [-1, 1] to [low, high]"""
    low, high = action_space_limits
    return low + (0.5 * (action + 1.0) * (high - low))


class GymDssatWrapper(gym.Wrapper):
    """Wrapper for easy and uniform interfacing with SB3"""
    def __init__(self, env):
        super(GymDssatWrapper, self).__init__(env)

        self.action_low, self.action_high = self._get_action_space_bounds()
        num_actions = len(self.action_keys)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(num_actions,), dtype="float32")

        obs_array = np.array(env.observation_dict_to_array(env.observation), dtype=np.float32).flatten()
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=obs_array.shape,
            dtype="float32"
        )

        self.last_info = {}
        self.last_obs = None
        self.last_obs_dict = None  # Store dict version for metrics

    def _get_action_space_bounds(self):
        action_keys = list(self.env.action_space.spaces.keys())
        lows = []
        highs = []
        for key in action_keys:
            box = self.env.action_space[key]
            lows.append(float(box.low))
            highs.append(float(box.high))
        self.action_keys = action_keys
        return np.array(lows), np.array(highs)

    def _format_action(self, action):
        formatted = {}
        for i, key in enumerate(self.action_keys):
            formatted[key] = action[i]
        return formatted

    def _format_observation(self, observation):
        return self.env.observation_dict_to_array(observation)

    def reset(self, seed=None, options=None):
        raw_obs = self.env.reset()
        obs = self._format_observation(raw_obs)
        obs = np.array(obs, dtype=np.float32).flatten()
        self.last_obs = self.env.observation
        self.last_obs_dict = raw_obs  # Store dict for metrics extraction
        return obs

    def step(self, action):
        denormalized_action = denormalize_action((self.action_low, self.action_high), action)
        formatted_action = self._format_action(denormalized_action)
        obs, reward, done, info = self.env.step(formatted_action)

        if reward is None:
            reward = 0.0
        elif isinstance(reward, (list, np.ndarray)):
            reward = float(np.sum([r for r in reward if r is not None]))
        else:
            reward = float(reward)

        if done:
            obs_dict = self.last_obs_dict
            obs, reward, info = self.last_obs, 0.0, self.last_info
        else:
            self.last_obs = obs
            self.last_obs_dict = obs
            self.last_info = info
            obs_dict = obs

        formatted_observation = self._format_observation(obs)
        formatted_observation = np.array(formatted_observation, dtype=np.float32).flatten()

        return formatted_observation, float(reward), done, info, obs_dict

    def close(self):
        return self.env.close()

    def seed(self, seed):
        self.env.set_seed(seed)


def extract_metrics(obs_dict):
    """Extract metrics from observation dict."""
    if obs_dict is None:
        return {}
    metrics = {}
    for key in ['dap', 'grnwt', 'xlai', 'vstage', 'cumsumfert', 'totir', 'swfac', 'nstres', 'topwt']:
        if key in obs_dict:
            val = obs_dict[key]
            metrics[key] = float(val) if hasattr(val, 'item') else float(val) if val is not None else 0.0
    return metrics


def run_ppo_inference(
    model_path: str,
    output_dir: str,
    n_envs: int = 6,
    seeds: list = None,
):
    """Run PPO agent inference and track yield/rewards."""

    if seeds is None:
        seeds = [2000, 2001, 2002, 2003, 2004, 2005]

    print("=" * 80)
    print("PPO Agent Inference (updated_model_v1)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Number of environments: {n_envs}")
    print(f"Seeds: {seeds[:n_envs]}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Load PPO agent
    print("\nLoading PPO agent...")
    agent = PPO.load(model_path)
    print(f"Model loaded. Observation space: {agent.observation_space}")
    print(f"Action space: {agent.action_space}")

    results = []
    final_yields = []
    final_rewards = []

    for env_idx in tqdm(range(n_envs), desc="Evaluating"):
        seed = seeds[env_idx] if env_idx < len(seeds) else 2000 + env_idx

        # Create environment with same settings as training
        env_args = {
            **_dssat_runtime_env_args(),
            'mode': 'all',
            'seed': seed,
            'random_weather': True,
        }
        env = GymDssatWrapper(gym.make('GymDssatPdi-v0', **env_args))

        # Run episode
        obs = env.reset()
        done = False
        turn = 0
        cumulative_reward = 0.0

        episode_data = {
            'env_id': env_idx,
            'seed': seed,
            'turns': []
        }

        # Record initial state
        metrics = extract_metrics(env.last_obs_dict)
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

            # Step environment - note our wrapper returns extra obs_dict
            obs, reward, done, info, obs_dict = env.step(action)
            cumulative_reward += reward

            # Extract metrics
            metrics = extract_metrics(obs_dict)
            metrics['reward'] = cumulative_reward

            # Denormalize action for logging
            denorm_action = denormalize_action((env.action_low, env.action_high), action)
            action_dict = {
                'anfer': float(denorm_action[0]) if len(denorm_action) > 0 else 0.0,
                'amir': float(denorm_action[1]) if len(denorm_action) > 1 else 0.0,
            }

            episode_data['turns'].append({
                'turn': turn,
                'action': action_dict,
                'metrics': metrics,
            })

        # Get final yield
        if len(episode_data['turns']) > 0:
            final_metrics = episode_data['turns'][-1]['metrics']
            final_yield = final_metrics.get('grnwt', 0)
            final_yields.append(final_yield)
            final_rewards.append(cumulative_reward)

        results.append(episode_data)
        env.close()

    # Save results
    print(f"\nSaving results...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "gym_dssat_maize_phase1_test_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    if len(final_yields) > 0:
        print(f"Environments evaluated: {len(final_yields)}")
        print(f"\nFinal Grain Yield (kg/ha):")
        print(f"  Mean: {np.mean(final_yields):.2f}")
        print(f"  Std:  {np.std(final_yields):.2f}")
        print(f"  Min:  {np.min(final_yields):.2f}")
        print(f"  Max:  {np.max(final_yields):.2f}")

    if len(final_rewards) > 0:
        print(f"\nFinal Rewards:")
        print(f"  Mean: {np.mean(final_rewards):.2f}")
        print(f"  Std:  {np.std(final_rewards):.2f}")

    print("=" * 80)
    print("\nPPO inference completed!")

    return final_yields, final_rewards


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PPO agent inference")
    parser.add_argument("--model", type=str, default=str(AGRIMANAGER_ROOT / "integrations" / "gym_dssat" / "checkpoints" / "updated_model_v1.zip"),
                        help="Path to PPO model checkpoint")
    parser.add_argument("--output-dir", type=str,
                        default="results/gym_dssat/maize_phase1/test/ppo",
                        help="Output directory for results")
    parser.add_argument("--n-envs", type=int, default=6,
                        help="Number of environments to evaluate")

    args = parser.parse_args()

    run_ppo_inference(
        model_path=args.model,
        output_dir=args.output_dir,
        n_envs=args.n_envs,
    )
