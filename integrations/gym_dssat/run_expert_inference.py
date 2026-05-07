"""
Run Expert policy inference on test dataset - tracks yield instead of rewards.
"""

import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gym
import gym_dssat_pdi

agrimanager_root = Path(__file__).parent.parent.parent


def _dssat_runtime_env_args() -> dict:
    """Return DSSAT runtime paths from env vars or repo-local defaults."""
    dssat_gym_path = Path(
        os.environ.get("DSSAT_GYM_PATH", agrimanager_root / "spack" / "gym-dssat-pdi")
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


class ExpertAgent:
    """
    Simple agent using policy of choosing fertilization amount based on days after planting
    """
    def __init__(self, env):
        self.env = env
        # Get observation variables from the underlying DSSAT env
        if hasattr(env, 'observation_variables'):
            obs_vars = env.observation_variables
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'observation_variables'):
            obs_vars = env.unwrapped.observation_variables
        else:
            # Default order
            obs_vars = ['dap', 'vstage', 'grnwt', 'xlai', 'cumsumfert']

        self.observation_variables = obs_vars
        self.dap_index = obs_vars.index('dap') if 'dap' in obs_vars else 0

        # Get mode
        if hasattr(env, 'mode'):
            mode = env.mode
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'mode'):
            mode = env.unwrapped.mode
        else:
            mode = 'all'

        all_policy_dic = {
            "fertilization": {
                40: 27,
                45: 35,
                80: 54,
            },
            "irrigation": {
                6: 13,
                20: 10,
                37: 10,
                50: 13,
                54: 18,
                65: 25,
                69: 25,
                72: 13,
                75: 15,
                77: 19,
                80: 20,
                84: 20,
                91: 15,
                101: 19,
                104: 4,
                105: 25,
            }
        }

        if mode == "all":
            self.fert_policy = all_policy_dic["fertilization"]
            self.irrig_policy = all_policy_dic["irrigation"]
        elif mode == "fertilization":
            self.fert_policy = all_policy_dic["fertilization"]
            self.irrig_policy = {}
        else:
            self.fert_policy = {}
            self.irrig_policy = all_policy_dic["irrigation"]

    def predict(self, obs_dict):
        """Get action based on observation dict."""
        if isinstance(obs_dict, dict):
            dap = int(obs_dict.get('dap', 0))
        else:
            dap = int(obs_dict[self.dap_index])

        anfer = self.fert_policy.get(dap, 0)
        amir = self.irrig_policy.get(dap, 0)

        return {"anfer": float(anfer), "amir": float(amir)}


def run_expert_inference(
    output_dir: str,
    n_envs: int = 5,
    seeds: list = None,
):
    """Run Expert agent inference and track yield."""

    if seeds is None:
        seeds = [2000, 2001, 2002, 2003, 2004]

    print("=" * 80)
    print("Expert Policy Inference")
    print("=" * 80)
    print(f"Number of environments: {n_envs}")
    print(f"Seeds: {seeds[:n_envs]}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    results = []
    final_yields = []

    for env_idx in tqdm(range(n_envs), desc="Evaluating"):
        seed = seeds[env_idx] if env_idx < len(seeds) else 2000 + env_idx

        # Create environment
        env = gym.make(
            'GymDssatPdi-v0',
            **_dssat_runtime_env_args(),
            mode='all',
            seed=seed,
            random_weather=True
        )

        # Create expert agent
        agent = ExpertAgent(env)

        # Run episode
        obs = env.reset()
        done = False
        turn = 0

        episode_data = {
            'env_id': env_idx,
            'seed': seed,
            'turns': []
        }

        # Record initial state
        metrics = {
            'dap': float(obs.get('dap', 0)),
            'grnwt': float(obs.get('grnwt', 0)),
            'xlai': float(obs.get('xlai', 0)),
            'cumsumfert': float(obs.get('cumsumfert', 0)),
        }
        episode_data['turns'].append({
            'turn': 0,
            'action': None,
            'metrics': metrics,
        })

        # Run episode
        while not done:
            turn += 1

            # Get action from expert
            action = agent.predict(obs)

            # Step environment
            obs, reward, done, info = env.step(action)

            if obs is None:
                break

            # Extract metrics
            metrics = {
                'dap': float(obs.get('dap', 0)),
                'grnwt': float(obs.get('grnwt', 0)),
                'xlai': float(obs.get('xlai', 0)),
                'cumsumfert': float(obs.get('cumsumfert', 0)),
            }

            episode_data['turns'].append({
                'turn': turn,
                'action': action,
                'metrics': metrics,
            })

        # Get final yield
        if len(episode_data['turns']) > 0:
            final_yield = episode_data['turns'][-1]['metrics'].get('grnwt', 0)
            final_yields.append(final_yield)

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

    print("=" * 80)
    print("\n✓ Expert policy inference completed!")

    return final_yields


if __name__ == "__main__":
    run_expert_inference(
        output_dir="results/gym_dssat/maize_phase1/test/expert",
        n_envs=5,
    )
