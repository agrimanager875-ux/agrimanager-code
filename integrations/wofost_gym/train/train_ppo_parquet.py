"""PPO/DQN/SAC training with parquet-defined reset config sampling.

Patches WOFOSTGym's rl_utils.make_env to insert a ParquetResetWrapper
so that each env.reset() uses parquet-provided reset settings
(`agro_file`, `year`, `lat`, `lon`) rather than random_reset=True.

Usage:
    python3 integrations/wofost_gym/train/train_ppo_parquet.py \
        --config-path entrypoints/train/config \
        --config-name wofost_gym_nn \
        [agent.total_timesteps=...] [data.train_files=...]
"""

import sys
import os
import json
import copy
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from argparse import Namespace

import numpy as np
import gymnasium as gym
import hydra
from omegaconf import OmegaConf

AGRIMANAGER_ROOT = Path(__file__).resolve().parents[4]
if str(AGRIMANAGER_ROOT) not in sys.path:
    sys.path.insert(0, str(AGRIMANAGER_ROOT))

from agrimanager.env.base import load_env_configs_from_parquet
from agrimanager.env.wofost_gym.env_config import DEFAULT_WOFOST_GYM_PATH, WOFOSTEnvConfig
from agrimanager.env.wofost_gym.crop_traits_observation import (
    CropTraitEncoder,
    CropTraitsObservationWrapper,
)
from parquet_reset_wrapper import ParquetResetWrapper

WOFOST_RL_INFERENCE_DIR = AGRIMANAGER_ROOT / "integrations" / "wofost_gym" / "inference"


def _resolve_local_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = AGRIMANAGER_ROOT / p
    return p


def _resolve_save_folder(path: str) -> str:
    """Resolve save folder to an absolute path with trailing separator."""
    resolved = _resolve_local_path(path)
    save_folder = str(resolved)
    if not save_folder.endswith(os.sep):
        save_folder += os.sep
    return save_folder


def infer_runtime_features(env_config: dict[str, object]) -> dict[str, object]:
    normalized = WOFOSTEnvConfig(**env_config).to_dict()
    return {
        "include_crop_traits": bool(normalized.get("include_crop_traits", False)),
        "crop_traits_dir": normalized.get("crop_traits_dir"),
        "trait_schema": normalized.get("trait_schema"),
        "wofost_gym_path": normalized.get("wofost_gym_path"),
    }


def _normalize_primary_config(cfg: dict) -> dict:
    normalized = copy.deepcopy(cfg)

    data_cfg = normalized.get("data") or {}
    for key in ("train_files", "val_files"):
        if key in data_cfg:
            normalized.setdefault(key, data_cfg[key])

    agent_cfg = normalized.get("agent") or {}
    if agent_cfg.get("type"):
        normalized.setdefault("agent_type", agent_cfg["type"])
    for key, value in agent_cfg.items():
        if key != "type":
            normalized.setdefault(key, value)

    runtime_cfg = normalized.get("runtime") or {}
    if "sample_with_replacement" in runtime_cfg:
        normalized.setdefault("sample_with_replacement", runtime_cfg["sample_with_replacement"])
    validation_cfg = runtime_cfg.get("validation") or {}
    if "enabled" in validation_cfg:
        normalized.setdefault("val_enabled", validation_cfg["enabled"])
    if "split" in validation_cfg:
        normalized.setdefault("val_split", validation_cfg["split"])
    if "device" in validation_cfg:
        normalized.setdefault("val_device", validation_cfg["device"])
    if "seed" in validation_cfg:
        normalized.setdefault("val_seed", validation_cfg["seed"])
    if "poll_interval_sec" in validation_cfg:
        normalized.setdefault("val_poll_interval_sec", validation_cfg["poll_interval_sec"])
    if "every_n_checkpoints" in validation_cfg:
        normalized.setdefault("val_every_n_checkpoints", validation_cfg["every_n_checkpoints"])
    if "log_to_wandb" in validation_cfg:
        normalized.setdefault("val_log_to_wandb", validation_cfg["log_to_wandb"])

    output_cfg = normalized.get("output") or {}
    if "save_folder" in output_cfg:
        normalized.setdefault("save_folder", output_cfg["save_folder"])
    if "validation_dir" in output_cfg:
        normalized.setdefault("val_output_dir", output_cfg["validation_dir"])

    return normalized


def _extract_final_wso(result_item: dict) -> float | None:
    turns = result_item.get("turns") or []
    for turn in reversed(turns):
        metrics = turn.get("turn_metrics") or {}
        wso = metrics.get("wso")
        if wso is not None:
            return float(wso)
    return None


def _extract_crop_name(result_item: dict) -> str:
    env_config = result_item.get("env_config") or {}
    agro_file = str(env_config.get("agro_file", "") or "")
    if agro_file.endswith("_agro.yaml"):
        return agro_file[:-10]
    if agro_file.endswith(".yaml"):
        return agro_file[:-5]
    return agro_file or "unknown"


def _compute_final_wso_stats(
    results_path: Path,
) -> tuple[float | None, int, dict[str, dict[str, float | int]]]:
    with open(results_path, "r") as f:
        results = json.load(f)

    final_wsos = []
    by_crop: dict[str, list[float]] = {}
    for item in results:
        final_wso = _extract_final_wso(item)
        if final_wso is not None:
            final_wsos.append(final_wso)
            crop = _extract_crop_name(item)
            by_crop.setdefault(crop, []).append(final_wso)

    per_crop_stats: dict[str, dict[str, float | int]] = {}
    for crop, values in by_crop.items():
        if not values:
            continue
        per_crop_stats[crop] = {
            "mean_final_wso": float(np.mean(values)),
            "num_envs": len(values),
        }

    if not final_wsos:
        return None, 0, per_crop_stats
    return float(np.mean(final_wsos)), len(final_wsos), per_crop_stats


class PeriodicValidationRunner:
    """Run validation rollout on checkpoint updates and log mean final_wso."""

    def __init__(
        self,
        checkpoint_root: Path,
        output_dir: Path,
        agent_type: str,
        device: str = "cpu",
        seed: int = 0,
        poll_interval_sec: float = 30.0,
        every_n_checkpoints: int = 10,
        log_to_wandb: bool = True,
        include_crop_traits: bool = False,
        crop_traits_dir: str | None = None,
        crop_trait_schema: str | None = None,
        test_files: Path | None = None,
        env_configs: list[dict[str, object]] | None = None,
        env_name: str = "wofost_gym",
    ):
        self.test_files = test_files
        self.env_configs = env_configs
        self.env_name = env_name
        self.checkpoint_root = checkpoint_root
        self.output_dir = output_dir
        self.agent_type = agent_type
        self.device = device
        self.seed = seed
        self.poll_interval_sec = poll_interval_sec
        self.every_n_checkpoints = max(1, int(every_n_checkpoints))
        self.log_to_wandb = log_to_wandb
        self.include_crop_traits = include_crop_traits
        self.crop_traits_dir = crop_traits_dir
        self.crop_trait_schema = crop_trait_schema

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._last_mtime = -1.0
        self._seen_checkpoints = 0
        self._wandb_metric_defined = False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "final_wso_history.jsonl"

    def start(self):
        source = self.test_files if self.test_files is not None else f"{len(self.env_configs or [])} in-memory envs"
        print(
            f"[VAL] enabled: source={source}, every_n_checkpoints={self.every_n_checkpoints}, "
            f"device={self.device}"
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def run_once(self, reason: str = "manual"):
        checkpoint = self._latest_checkpoint()
        if checkpoint is None:
            print(f"[VAL] skip ({reason}): no checkpoint found under {self.checkpoint_root}")
            return
        self._run_validation(checkpoint, reason=reason)

    def _latest_checkpoint(self) -> Path | None:
        candidates = list(self.checkpoint_root.rglob("agent.pt"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _watch(self):
        while not self._stop.is_set():
            try:
                checkpoint = self._latest_checkpoint()
                if checkpoint is not None:
                    mtime = checkpoint.stat().st_mtime
                    if mtime > self._last_mtime:
                        self._last_mtime = mtime
                        self._seen_checkpoints += 1
                        if self._seen_checkpoints % self.every_n_checkpoints == 0:
                            self._run_validation(checkpoint, reason="periodic")
                        else:
                            print(
                                f"[VAL] checkpoint update #{self._seen_checkpoints} detected, "
                                f"skip (every_n={self.every_n_checkpoints})"
                            )
            except Exception as exc:
                print(f"[VAL] watcher error: {exc}")
            self._stop.wait(self.poll_interval_sec)

    def _run_validation(self, checkpoint: Path, reason: str):
        if str(WOFOST_RL_INFERENCE_DIR) not in sys.path:
            sys.path.insert(0, str(WOFOST_RL_INFERENCE_DIR))
        from wofost_rl_rollout import run_rl_rollout

        eval_idx = self._seen_checkpoints
        run_dir = self.output_dir / f"ckpt_{eval_idx:06d}"

        print(f"[VAL] start ({reason}): checkpoint={checkpoint}")
        run_rl_rollout(
            output_dir=str(run_dir),
            agent_path=str(checkpoint),
            agent_type=self.agent_type,
            seed=self.seed,
            device=self.device,
            include_crop_traits=self.include_crop_traits,
            crop_traits_dir=self.crop_traits_dir,
            crop_trait_schema=self.crop_trait_schema,
            test_files=None if self.test_files is None else str(self.test_files),
            env_configs=None if self.env_configs is None else copy.deepcopy(self.env_configs),
            env_name=self.env_name,
        )

        results_path = run_dir / "results.json"
        mean_final_wso, n_envs, per_crop_stats = _compute_final_wso_stats(results_path)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "checkpoint_index": eval_idx,
            "checkpoint_path": str(checkpoint),
            "mean_final_wso": mean_final_wso,
            "num_envs": n_envs,
            "mean_final_wso_by_crop": per_crop_stats,
            "results_path": str(results_path),
        }
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if self.log_to_wandb:
            try:
                import wandb
                if wandb.run is not None and mean_final_wso is not None:
                    if not self._wandb_metric_defined:
                        # Keep validation charts on their own x-axis so they do not
                        # conflict with trainer-managed global step values.
                        wandb.define_metric("val/checkpoint_index")
                        wandb.define_metric("val/*", step_metric="val/checkpoint_index")
                        self._wandb_metric_defined = True
                    payload = {
                        "val/checkpoint_index": eval_idx,
                        "val/mean_final_wso": mean_final_wso,
                        "val/num_envs": n_envs,
                    }
                    for crop, stats in sorted(per_crop_stats.items()):
                        payload[f"val/mean_final_wso/{crop}"] = stats["mean_final_wso"]
                        payload[f"val/num_envs/{crop}"] = stats["num_envs"]
                    wandb.log(payload)
            except Exception as exc:
                print(f"[VAL] wandb log error: {exc}")

        per_crop_str = ", ".join(
            f"{crop}={stats['mean_final_wso']:.4f}(n={stats['num_envs']})"
            for crop, stats in sorted(per_crop_stats.items())
        )
        print(
            f"[VAL] done: checkpoint_index={eval_idx}, mean_final_wso={mean_final_wso}, "
            f"num_envs={n_envs}, per_crop=[{per_crop_str}], log={self.history_path}"
        )


def load_parquet_configs(train_files: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract reset configs and env defaults from a dataset artifact."""
    env_configs, env_name, path = load_env_configs_from_parquet(train_files)
    if env_name != "wofost_gym":
        raise ValueError(f"Expected wofost_gym dataset, got {env_name!r} from {path}")
    if not env_configs:
        raise ValueError(f"Dataset contains no environment configs: {path}")

    defaults = WOFOSTEnvConfig(**env_configs[0]).to_dict()
    configs: list[dict[str, object]] = []
    for row_idx, env_config in enumerate(env_configs):
        normalized = WOFOSTEnvConfig(**env_config).to_dict()
        agro_params = normalized.get("agro_params") or {}
        try:
            year = int(agro_params["year"])
            lat = float(agro_params["latitude"])
            lon = float(agro_params["longitude"])
        except KeyError as exc:
            raise ValueError(
                f"Missing agro_params field {exc!s} in dataset row index={row_idx}: {path}"
            ) from exc

        agro_file = normalized.get("agro_file")
        if not agro_file:
            raise ValueError(
                f"Missing env_config.agro_file in dataset row index={row_idx}: {path}"
            )

        configs.append(
            {
                "agro_file": str(agro_file),
                "year": year,
                "location": (lat, lon),
                "seed": normalized.get("seed"),
                "crop_name": normalized.get("crop_name"),
                "scenario_id": normalized.get("scenario_id"),
            }
        )

    print(f"Loaded {len(configs)} (agro_file, year, location) configs from {path}")
    return configs, defaults


def patch_make_env(
    configs: list[dict[str, object]],
    seed: int = 0,
    sample_with_replacement: bool = False,
    include_crop_traits: bool = False,
    crop_traits_dir: str | None = None,
    crop_trait_schema: str | None = None,
):
    """Patch rl_utils.make_env to insert ParquetResetWrapper.

    The patched make_env wraps the env chain as:
        make_gym_env → wrap_env_reward → ParquetResetWrapper
        → RecordEpisodeStatistics → NormalizeObservation
        → CropTraitsObservationWrapper (optional) → NormalizeReward
    """
    from rl_algs import rl_utils
    from pcse_gym import wrappers

    # Import utils from WOFOSTGym (already on sys.path)
    import utils as wofost_utils

    rng = np.random.default_rng(seed)
    trait_encoder = (
        CropTraitEncoder(
            traits_dir=crop_traits_dir,
            trait_schema=crop_trait_schema,
        )
        if include_crop_traits
        else None
    )

    def patch_eval_policy_for_crop_traits():
        """Keep WOFOSTGym's periodic eval observation shape aligned with training."""

        def eval_policy_with_crop_traits(
            policy,
            eval_env: gym.Env,
            kwargs: Namespace,
            device,
            eval_episodes: int = 5,
        ) -> float:
            import torch

            avg_reward = 0.0

            if isinstance(eval_env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
                if isinstance(eval_env, gym.vector.AsyncVectorEnv):
                    env_constr = eval_env.get_attr("unwrapped")[0].__class__
                    args = eval_env.get_attr("args")[0]
                    base_fpath = eval_env.get_attr("base_fpath")[0]
                    agro_fpath = eval_env.get_attr("agro_fpath")[0]
                    site_fpath = eval_env.get_attr("site_fpath")[0]
                    crop_fpath = eval_env.get_attr("crop_fpath")[0]
                    name_fpath = eval_env.get_attr("name_fpath")[0]
                    unit_fpath = eval_env.get_attr("unit_fpath")[0]
                    range_fpath = eval_env.get_attr("range_fpath")[0]
                    render_mode = eval_env.get_attr("render_mode")[0]
                    config = eval_env.get_attr("config")[0]
                else:
                    unwrapped = eval_env.envs[0].unwrapped
                    env_constr = type(unwrapped)
                    args = unwrapped.args
                    base_fpath = unwrapped.base_fpath
                    agro_fpath = unwrapped.agro_fpath
                    site_fpath = unwrapped.site_fpath
                    crop_fpath = unwrapped.crop_fpath
                    name_fpath = unwrapped.name_fpath
                    unit_fpath = unwrapped.unit_fpath
                    range_fpath = unwrapped.range_fpath
                    render_mode = unwrapped.render_mode
                    config = unwrapped.config
            else:
                unwrapped = eval_env.unwrapped
                env_constr = type(unwrapped)
                args = unwrapped.args
                base_fpath = unwrapped.base_fpath
                agro_fpath = unwrapped.agro_fpath
                site_fpath = unwrapped.site_fpath
                crop_fpath = unwrapped.crop_fpath
                name_fpath = unwrapped.name_fpath
                unit_fpath = unwrapped.unit_fpath
                range_fpath = unwrapped.range_fpath
                render_mode = unwrapped.render_mode
                config = unwrapped.config

            new_args = copy.deepcopy(args)
            new_args.random_reset = True
            new_args.train_reset = False
            new_args.domain_rand = False
            env = env_constr(
                new_args,
                base_fpath,
                agro_fpath,
                site_fpath,
                crop_fpath,
                name_fpath,
                unit_fpath,
                range_fpath,
                render_mode,
                config,
            )

            env = wofost_utils.wrap_env_reward(env, kwargs)
            env = wrappers.NormalizeObservation(env)
            env = CropTraitsObservationWrapper(
                env,
                encoder=trait_encoder,
                agro_file=agro_fpath,
            )
            env = wrappers.NormalizeReward(env)

            for _ in range(eval_episodes):
                state, _, term, trunc = *env.reset(), False, False
                while not (term or trunc):
                    if isinstance(state, np.ndarray):
                        state = torch.Tensor(state).reshape((-1, *env.observation_space.shape)).to(device)
                    action = policy.get_action(state)
                    state, reward, term, trunc, _ = env.step(action.detach().cpu().numpy())

                    if isinstance(eval_env, gym.vector.SyncVectorEnv):
                        avg_reward += eval_env.envs[0].unnormalize(reward)
                    elif isinstance(eval_env, gym.vector.AsyncVectorEnv):
                        avg_reward += eval_env.call("unnormalize", reward)[0]
                    else:
                        avg_reward += eval_env.unnormalize(reward)

            return avg_reward / eval_episodes

        rl_utils.eval_policy = eval_policy_with_crop_traits
        for module_name in ("rl_algs.PPO", "rl_algs.DQN", "rl_algs.SAC", "rl_algs.BCQ"):
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, "eval_policy"):
                module.eval_policy = eval_policy_with_crop_traits

    def make_env_parquet(kwargs: Namespace, idx: int = 1,
                         capture_video: bool = False, run_name: str = None):
        worker_seed = int(rng.integers(0, np.iinfo(np.uint32).max))

        def make_wrapped_env_for_agro_file(agro_file: str | None):
            env_kwargs = copy.deepcopy(kwargs)
            if agro_file:
                env_kwargs.agro_file = agro_file

            if capture_video and idx == 0:
                built_env = wofost_utils.make_gym_env(env_kwargs, run_name=run_name)
                built_env = gym.wrappers.RecordVideo(built_env, f"videos/{run_name}")
            else:
                built_env = wofost_utils.make_gym_env(env_kwargs, run_name=run_name)

            return wofost_utils.wrap_env_reward(built_env, env_kwargs)

        def thunk():
            initial_agro_file = str(getattr(kwargs, "agro_file", "") or "") or None
            env = make_wrapped_env_for_agro_file(initial_agro_file)
            env = ParquetResetWrapper(
                env,
                configs,
                rng=np.random.default_rng(worker_seed),
                sample_with_replacement=sample_with_replacement,
                make_env_for_agro_file=make_wrapped_env_for_agro_file,
                current_agro_file=initial_agro_file,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = wrappers.NormalizeObservation(env)
            if trait_encoder is not None:
                env = CropTraitsObservationWrapper(
                    env,
                    encoder=trait_encoder,
                    agro_file=initial_agro_file,
                )
            env = wrappers.NormalizeReward(env)
            return env
        return thunk

    # Patch the module-level make_env so that setup() sees it
    rl_utils.make_env = make_env_parquet
    if trait_encoder is not None:
        patch_eval_policy_for_crop_traits()
    sample_mode = "with replacement" if sample_with_replacement else "without replacement"
    print(f"Patched rl_utils.make_env with ParquetResetWrapper ({sample_mode}, seed={seed})")
    if trait_encoder is not None:
        print(
            "Enabled crop-trait observation features: "
            f"schema={trait_encoder.trait_schema}, dim={trait_encoder.dim}, "
            f"traits_dir={trait_encoder.traits_dir}, "
            f"crops={list(trait_encoder.crop_names)}"
        )


def build_agent_args(cfg: dict):
    """Build AgentArgs namespace using tyro from the resolved Hydra config.

    Returns the parsed AgentArgs namespace that WOFOSTGym's train() expects.
    """
    import tyro

    # We need to import AgentArgs; train_agent.py lives in WOFOSTGym root
    # which is already on sys.path.
    from train_agent import AgentArgs

    # Build a flat list of tyro-compatible CLI args from the config dict
    tyro_args = []

    agent_type = cfg.get("agent_type", "PPO")

    # Map flat config keys to tyro dotted paths
    key_map = {
        "env_id": "--env-id",
        "agro_file": "--agro-file",
        "save_folder": "--save-folder",
        "env_reward": "--env-reward",
        "agent_type": "--agent-type",
        "wandb_project_name": f"--{agent_type}.wandb-project-name",
        "wandb_entity": f"--{agent_type}.wandb-entity",
        "intvn_interval": "--npk.intvn-interval",
        "random_reset": "--npk.random-reset",
        "fert_amount": "--npk.fert-amount",
        "irrig_amount": "--npk.irrig-amount",
        # PPO/DQN/SAC hyperparameters (nested under agent type)
        "num_envs": f"--{agent_type}.num-envs",
        "total_timesteps": f"--{agent_type}.total-timesteps",
        "num_steps": f"--{agent_type}.num-steps",
        "minibatch_size": f"--{agent_type}.minibatch-size",
        "num_minibatches": f"--{agent_type}.num-minibatches",
        "learning_rate": f"--{agent_type}.learning-rate",
        "gamma": f"--{agent_type}.gamma",
        "gae_lambda": f"--{agent_type}.gae-lambda",
        "seed": f"--{agent_type}.seed",
        "checkpoint_frequency": f"--{agent_type}.checkpoint-frequency",
        "exp_name": f"--{agent_type}.exp-name",
        "use_simple_exp_name": f"--{agent_type}.use-simple-exp-name",
        "update_epochs": f"--{agent_type}.update-epochs",
        "anneal_lr": f"--{agent_type}.anneal-lr",
        "clip_coef": f"--{agent_type}.clip-coef",
        "ent_coef": f"--{agent_type}.ent-coef",
        "vf_coef": f"--{agent_type}.vf-coef",
        "max_grad_norm": f"--{agent_type}.max-grad-norm",
    }

    def to_tyro_false_flag(tyro_flag: str) -> str:
        """Convert a tyro boolean flag to its explicit False form.

        Examples:
            --npk.random-reset -> --npk.no-random-reset
            --track -> --no-track
        """
        flag = tyro_flag.lstrip("-")
        if "." in flag:
            head, tail = flag.rsplit(".", 1)
            return f"--{head}.no-{tail}"
        return f"--no-{flag}"

    for cfg_key, tyro_flag in key_map.items():
        if cfg_key in cfg:
            val = cfg[cfg_key]
            # tyro expects booleans as flags
            if isinstance(val, bool):
                if val:
                    tyro_args.append(tyro_flag)
                else:
                    tyro_args.append(to_tyro_false_flag(tyro_flag))
            else:
                tyro_args.extend([tyro_flag, str(val)])

    # Force random_reset=False (parquet controls sampling)
    if "random_reset" not in cfg:
        tyro_args.extend(["--npk.no-random-reset"])

    # track flag
    if cfg.get("track", False):
        tyro_args.append("--track")

    args = tyro.cli(AgentArgs, args=tyro_args)
    return args


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(config) -> None:
    OmegaConf.resolve(config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError(f"Expected dict config, got {type(config_dict)!r}")
    cfg = _normalize_primary_config(config_dict)

    if "train_files" not in cfg:
        raise ValueError("Config must specify data.train_files or train_files.")
    if "save_folder" in cfg and cfg["save_folder"]:
        cfg["save_folder"] = _resolve_save_folder(str(cfg["save_folder"]))

    agent_type = cfg.get("agent_type", "PPO")
    configs, defaults = load_parquet_configs(cfg["train_files"])
    for key in ("env_id", "agro_file", "intvn_interval", "env_reward"):
        if defaults.get(key) is not None:
            cfg.setdefault(key, defaults[key])

    runtime_features = infer_runtime_features(defaults)
    include_crop_traits = bool(cfg.get("include_crop_traits", runtime_features["include_crop_traits"]))
    crop_traits_dir = cfg.get("crop_traits_dir") or runtime_features["crop_traits_dir"]
    crop_trait_schema = (
        cfg.get("crop_trait_schema")
        or cfg.get("trait_schema")
        or runtime_features["trait_schema"]
    )

    # Add WOFOSTGym to sys.path
    wofost_gym_path = str(runtime_features.get("wofost_gym_path") or DEFAULT_WOFOST_GYM_PATH)
    if wofost_gym_path not in sys.path:
        sys.path.insert(0, wofost_gym_path)
    # Also add pcse_gym to path for wrappers
    pcse_gym_path = os.path.join(wofost_gym_path, "pcse_gym")
    if pcse_gym_path not in sys.path:
        sys.path.insert(0, pcse_gym_path)

    # Change cwd to WOFOSTGym so that relative paths (env_config/) resolve
    original_cwd = os.getcwd()
    os.chdir(wofost_gym_path)
    val_runner = None

    try:
        patch_make_env(
            configs,
            seed=int(cfg.get("seed", 0)),
            sample_with_replacement=bool(cfg.get("sample_with_replacement", False)),
            include_crop_traits=include_crop_traits,
            crop_traits_dir=crop_traits_dir,
            crop_trait_schema=crop_trait_schema,
        )

        # Build AgentArgs and train
        agent_args = build_agent_args(cfg)

        val_enabled = bool(cfg.get("val_enabled", False))
        if val_enabled:
            checkpoint_root = _resolve_local_path(agent_args.save_folder)
            val_output_dir = _resolve_local_path(
                cfg.get(
                    "val_output_dir",
                    f"{agent_args.save_folder.rstrip('/')}/validation",
                )
            )

            val_files = cfg.get("val_files")
            if not val_files:
                raise ValueError("val_enabled=true but val_files is not set.")

            val_runner = PeriodicValidationRunner(
                checkpoint_root=checkpoint_root,
                output_dir=val_output_dir,
                agent_type=agent_type,
                device=str(cfg.get("val_device", "cpu")),
                seed=int(cfg.get("val_seed", 0)),
                poll_interval_sec=float(cfg.get("val_poll_interval_sec", 30)),
                every_n_checkpoints=int(cfg.get("val_every_n_checkpoints", 10)),
                log_to_wandb=bool(cfg.get("val_log_to_wandb", True)),
                include_crop_traits=include_crop_traits,
                crop_traits_dir=crop_traits_dir,
                crop_trait_schema=crop_trait_schema,
                test_files=_resolve_local_path(val_files),
                env_configs=None,
                env_name="wofost_gym",
            )
            val_runner.start()

        # Get the trainer function
        import utils as wofost_utils
        trainers = wofost_utils.get_valid_trainers()
        if agent_type not in trainers:
            raise ValueError(
                f"Unknown agent_type '{agent_type}'. Available: {list(trainers.keys())}"
            )

        print(f"\nStarting {agent_type} training with parquet dataset artifact")
        print(f"  Train file: {_resolve_local_path(cfg['train_files'])}")
        print(f"  Num configs: {len(configs)}")
        print(f"  Save folder: {agent_args.save_folder}")
        if val_runner is not None:
            print(f"  Val file: {_resolve_local_path(cfg['val_files'])}")
        print()

        trainers[agent_type](agent_args)
        if val_runner is not None:
            val_runner.run_once(reason="final")
    finally:
        if val_runner is not None:
            val_runner.stop()
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
