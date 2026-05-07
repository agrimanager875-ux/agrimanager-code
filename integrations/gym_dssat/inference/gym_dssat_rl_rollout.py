"""SB3 PPO rollout for Gym-DSSAT parquet datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf


def _load_env_configs(parquet_path: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    return [
        row["extra_info"]["interaction_kwargs"]["env_config"]
        for _, row in df.iterrows()
    ]


class GymDssatEvalEnv:
    """Single-environment evaluator for SB3 PPO Gym-DSSAT agents."""

    def __init__(self, env_config: Dict[str, Any]):
        from agrimanager.env.base import create_environment

        self.env_config = dict(env_config)
        cfg = dict(self.env_config)
        cfg["llm_mode"] = False
        self.env, self.config = create_environment(cfg.get("env_name", "gym_dssat"), cfg)
        action_space = self.env.env.action_space
        self.action_keys = list(action_space.spaces.keys())
        self.action_low = np.array([float(action_space[key].low) for key in self.action_keys], dtype=np.float32)
        self.action_high = np.array([float(action_space[key].high) for key in self.action_keys], dtype=np.float32)
        self.last_obs = None

    def _obs_to_array(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and hasattr(self.env.env, "observation_dict_to_array"):
            return np.asarray(self.env.env.observation_dict_to_array(obs), dtype=np.float32).flatten()
        return np.asarray(obs, dtype=np.float32).flatten()

    def _denormalize_action(self, action: np.ndarray) -> Dict[str, float]:
        phys = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
        phys = np.clip(phys, self.action_low, self.action_high)
        return {key: float(phys[i]) for i, key in enumerate(self.action_keys)}

    def reset(self):
        obs, info = self.env.reset()
        self.last_obs = obs
        return self._obs_to_array(obs), info

    def step(self, action: np.ndarray):
        action_dict = self._denormalize_action(action)
        obs, reward, done, info = self.env.step(action_dict)
        if obs is not None:
            self.last_obs = obs
        obs_for_model = self._obs_to_array(self.last_obs)
        return obs_for_model, float(reward), bool(done), info or {}, action_dict, obs

    def close(self):
        self.env.close()


@hydra.main(version_base=None)
def main(cfg) -> None:
    from stable_baselines3 import PPO

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    inference_file = cfg_dict["data"]["inference_file"]
    agent_path = cfg_dict["agent"]["path"]
    output_dir = Path(cfg_dict["output"]["dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg_dict["runtime"].get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_configs = _load_env_configs(inference_file)
    model = PPO.load(agent_path)

    results = []
    for env_id, env_config in enumerate(env_configs):
        env = GymDssatEvalEnv(env_config)
        try:
            obs_arr, info = env.reset()
            episode = {
                "env_id": env_id,
                "env_config": env_config,
                "turns": [
                    {
                        "turn": 0,
                        "action": None,
                        "turn_metrics": (info or {}).get("turn_metrics", {}),
                    }
                ],
            }

            done = False
            turn = 0
            while not done:
                turn += 1
                action, _ = model.predict(obs_arr, deterministic=True)
                obs_arr, reward, done, step_info, action_dict, _ = env.step(action)
                record = {
                    "turn": turn,
                    "action": action_dict,
                    "turn_metrics": (step_info or {}).get("turn_metrics", {}),
                }
                if done:
                    record["trajectory_metrics"] = (step_info or {}).get("trajectory_metrics", {})
                episode["turns"].append(record)
            results.append(episode)
        finally:
            env.close()

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved Gym-DSSAT NN rollout results to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
