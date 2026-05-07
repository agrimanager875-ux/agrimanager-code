"""Stable-Baselines3 PPO training for Gym-DSSAT from parquet env configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

try:
    import gymnasium as gym
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Gym-DSSAT NN training requires `gymnasium` in the active Conda env. "
        "Install `gym==0.26.2 gymnasium stable-baselines3` before running the smoke test."
    ) from exc


def _load_env_configs(parquet_path: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    return [
        row["extra_info"]["interaction_kwargs"]["env_config"]
        for _, row in df.iterrows()
    ]


class GymDssatPPOEnv(gym.Env):
    """Continuous-action wrapper that samples reset configs from parquet rows."""

    metadata = {}

    def __init__(self, env_configs: List[Dict[str, Any]], env_slot: int = 0, num_envs: int = 1):
        super().__init__()
        from agrimanager.env.base import create_environment

        if not env_configs:
            raise ValueError("env_configs must not be empty")

        self._create_environment = create_environment
        self._env_configs = [cfg.copy() for cfg in env_configs]
        self._env_slot = env_slot
        self._num_envs = max(1, int(num_envs))
        self._cursor = env_slot
        self._env = None
        self._obs_dict = None
        self._action_keys: List[str] = []
        self._action_low: np.ndarray | None = None
        self._action_high: np.ndarray | None = None

        initial_env = self._build_env(self._env_configs[self._cursor % len(self._env_configs)])
        try:
            action_space = initial_env.env.action_space
            self._action_keys = list(action_space.spaces.keys())
            self._action_low = np.array(
                [float(action_space[key].low) for key in self._action_keys],
                dtype=np.float32,
            )
            self._action_high = np.array(
                [float(action_space[key].high) for key in self._action_keys],
                dtype=np.float32,
            )

            obs, _ = initial_env.reset()
            obs_arr = self._obs_to_array(initial_env, obs)
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self._action_keys),),
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_arr.shape,
                dtype=np.float32,
            )
        finally:
            initial_env.close()

    def _build_env(self, env_config: Dict[str, Any]):
        cfg = dict(env_config)
        cfg["llm_mode"] = False
        env, _ = self._create_environment(cfg.get("env_name", "gym_dssat"), cfg)
        return env

    def _obs_to_array(self, env, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and hasattr(env.env, "observation_dict_to_array"):
            arr = env.env.observation_dict_to_array(obs)
            return np.asarray(arr, dtype=np.float32).flatten()
        return np.asarray(obs, dtype=np.float32).flatten()

    def _next_config(self) -> Dict[str, Any]:
        cfg = self._env_configs[self._cursor % len(self._env_configs)]
        self._cursor += self._num_envs
        return cfg

    def _denormalize_action(self, action: np.ndarray) -> Dict[str, float]:
        lo = self._action_low
        hi = self._action_high
        assert lo is not None and hi is not None
        phys = lo + 0.5 * (action + 1.0) * (hi - lo)
        phys = np.clip(phys, lo, hi)
        return {key: float(phys[i]) for i, key in enumerate(self._action_keys)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._env is not None:
            self._env.close()
        self._env = self._build_env(self._next_config())
        obs, info = self._env.reset()
        self._obs_dict = obs
        return self._obs_to_array(self._env, obs), info

    def step(self, action):
        assert self._env is not None
        action_dict = self._denormalize_action(np.asarray(action, dtype=np.float32))
        obs, reward, done, info = self._env.step(action_dict)
        if obs is None:
            obs_arr = self._obs_to_array(self._env, self._obs_dict)
        else:
            self._obs_dict = obs
            obs_arr = self._obs_to_array(self._env, obs)
        return obs_arr, float(reward), bool(done), False, info or {}

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


@hydra.main(version_base=None)
def main(cfg) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gym-DSSAT NN training requires `stable-baselines3` in the active "
            "Conda env. Install `gym==0.26.2 gymnasium stable-baselines3` "
            "before running the smoke test."
        ) from exc

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    train_files = cfg_dict["data"]["train_files"]
    env_configs = _load_env_configs(train_files)
    num_envs = int(cfg_dict["agent"].get("num_envs", 1))
    save_root = Path(cfg_dict["output"]["save_folder"]).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    def _make_env(slot: int):
        def factory():
            return Monitor(GymDssatPPOEnv(env_configs, env_slot=slot, num_envs=num_envs))
        return factory

    vec_env = VecMonitor(DummyVecEnv([_make_env(i) for i in range(num_envs)]))

    model = PPO(
        cfg_dict["agent"].get("policy", "MlpPolicy"),
        vec_env,
        learning_rate=float(cfg_dict["agent"].get("learning_rate", 3e-4)),
        gamma=float(cfg_dict["agent"].get("gamma", 1.0)),
        gae_lambda=float(cfg_dict["agent"].get("gae_lambda", 0.99)),
        n_steps=int(cfg_dict["agent"].get("n_steps", 128)),
        batch_size=int(cfg_dict["agent"].get("batch_size", 64)),
        verbose=1,
        seed=int(cfg_dict["agent"].get("seed", 0)),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, int(cfg_dict["agent"].get("checkpoint_frequency", 512)) // max(1, num_envs)),
        save_path=str(save_root / "checkpoints"),
        name_prefix="agent",
    )

    model.learn(
        total_timesteps=int(cfg_dict["agent"].get("total_timesteps", 4096)),
        callback=checkpoint_cb,
        progress_bar=False,
    )

    final_path = save_root / "agent"
    model.save(str(final_path))

    metadata = {
        "train_files": train_files,
        "num_envs": num_envs,
        "total_timesteps": int(cfg_dict["agent"].get("total_timesteps", 4096)),
        "agent_type": cfg_dict["agent"].get("type", "PPO"),
    }
    with open(save_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    vec_env.close()
    print(f"Saved PPO checkpoint to {final_path}.zip")


if __name__ == "__main__":
    main()
