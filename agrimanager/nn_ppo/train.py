"""Framework-native parquet-driven PPO training for numeric environments."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
import torch

from agrimanager.adapter.trainer.validation_metrics import add_env_metrics
from agrimanager.adapter.trainer.validation_sets import (
    annotate_env_config_with_validation_set,
    normalize_val_sets,
)
from agrimanager.env.base import create_nn_env_adapter, load_env_configs_from_parquet
from agrimanager.nn_ppo.common import (
    ScenarioCyclingEnv,
    collect_trajectory_metrics,
    evaluate_model_on_dataset,
    resolve_repo_path,
)


def _ensure_experiment_dir(path_str: str) -> Path:
    exp_dir = resolve_repo_path(path_str)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _partition_env_configs_for_workers(
    env_configs: list[dict[str, Any]],
    num_workers: int,
    *,
    shard_across_workers: bool,
) -> list[list[dict[str, Any]]]:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}.")
    if not env_configs:
        raise ValueError("env_configs must not be empty.")
    if not shard_across_workers or num_workers == 1:
        return [list(env_configs) for _ in range(num_workers)]

    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    for idx, env_config in enumerate(env_configs):
        shards[idx % num_workers].append(dict(env_config))

    empty_shards = [idx for idx, shard in enumerate(shards) if not shard]
    if empty_shards:
        raise ValueError(
            "Cannot shard train scenarios across env workers because some workers "
            f"would receive no scenarios: {empty_shards}."
        )
    return shards


def _resolve_target_train_episodes(
    train_epochs_cfg: Any,
    num_train_scenarios: int,
    *,
    sample_with_replacement: bool,
    num_envs: int,
    shard_train_scenarios: bool,
) -> int | None:
    if train_epochs_cfg is None:
        return None

    train_epochs = int(train_epochs_cfg)
    if train_epochs <= 0:
        raise ValueError(f"runtime.train_epochs must be positive, got {train_epochs}.")
    if sample_with_replacement:
        raise ValueError("runtime.train_epochs requires runtime.sample_with_replacement=false.")
    if num_envs > 1 and not shard_train_scenarios:
        raise ValueError(
            "runtime.train_epochs with num_envs > 1 requires "
            "runtime.shard_train_scenarios_across_envs=true so epochs refer to the "
            "global dataset instead of per-worker duplicated passes."
        )
    return int(num_train_scenarios) * train_epochs


def _resolve_total_timesteps(
    agent_cfg: dict[str, Any],
    adapter,
    train_env_configs: list[dict[str, Any]],
    *,
    total_train_episodes: int | None,
    num_envs: int,
) -> int:
    configured_total_timesteps = agent_cfg.get("total_timesteps")
    if configured_total_timesteps is not None:
        total_timesteps = int(configured_total_timesteps)
        if total_timesteps <= 0:
            raise ValueError(f"agent.total_timesteps must be positive, got {total_timesteps}.")
        return total_timesteps

    if total_train_episodes is None:
        raise ValueError(
            "Config must specify either agent.total_timesteps or runtime.train_epochs."
        )

    hints: list[int] = []
    for env_config in train_env_configs:
        hint = adapter.episode_length_hint(env_config)
        if hint is None:
            raise ValueError(
                "runtime.train_epochs requires either agent.total_timesteps or "
                "BaseNNEnvAdapter.episode_length_hint(env_config) for every scenario."
            )
        hints.append(int(hint))

    max_episode_length = max(hints)
    n_steps = int(agent_cfg.get("n_steps", agent_cfg.get("num_steps", 128)))
    return int(total_train_episodes) * max_episode_length + int(num_envs) * n_steps


def _resolve_epoch_timestep_frequency(
    adapter,
    train_env_configs: list[dict[str, Any]],
    *,
    intervals_per_epoch: int,
    num_envs: int,
    setting_name: str,
) -> int:
    intervals_per_epoch = int(intervals_per_epoch)
    if intervals_per_epoch <= 0:
        return 0

    hints: list[int] = []
    for env_config in train_env_configs:
        hint = adapter.episode_length_hint(env_config)
        if hint is None:
            raise ValueError(
                f"{setting_name} requires BaseNNEnvAdapter.episode_length_hint(env_config) "
                "for every scenario."
            )
        hints.append(int(hint))

    epoch_timesteps = len(train_env_configs) * max(hints)
    raw_frequency = max(1, int(math.ceil(epoch_timesteps / intervals_per_epoch)))
    vector_stride = max(1, int(num_envs))
    return int(math.ceil(raw_frequency / vector_stride) * vector_stride)


def _resolve_checkpoint_frequency(
    agent_cfg: dict[str, Any],
) -> int:
    checkpoint_mode = _resolve_checkpoint_mode(agent_cfg)
    if checkpoint_mode != "timestep":
        return 0

    configured_frequency = agent_cfg.get("checkpoint_frequency")
    if configured_frequency is None:
        raise ValueError(
            "agent.checkpoint_mode=timestep requires agent.checkpoint_frequency."
        )
    return max(0, int(configured_frequency))


def _resolve_checkpoint_mode(agent_cfg: dict[str, Any]) -> str:
    configured_mode = agent_cfg.get("checkpoint_mode")
    legacy_frequency = agent_cfg.get("checkpoint_frequency")
    if configured_mode is None:
        return "timestep" if legacy_frequency is not None else "epoch"

    mode = str(configured_mode).strip().lower().replace("-", "_")
    aliases = {
        "per_epoch": "epoch",
        "every_epoch": "epoch",
        "episode": "epoch",
        "episodes": "epoch",
        "step": "timestep",
        "steps": "timestep",
        "timesteps": "timestep",
        "off": "none",
        "disabled": "none",
        "false": "none",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"epoch", "timestep", "none"}:
        raise ValueError(
            "agent.checkpoint_mode must be one of epoch, timestep, or none; "
            f"got {configured_mode!r}."
        )
    return mode


def _resolve_checkpoint_episode_frequency(
    agent_cfg: dict[str, Any],
    *,
    train_epochs: int | None,
    num_train_scenarios: int,
) -> int:
    checkpoint_mode = _resolve_checkpoint_mode(agent_cfg)
    if checkpoint_mode != "epoch":
        return 0
    if train_epochs is None:
        raise ValueError("agent.checkpoint_mode=epoch requires runtime.train_epochs.")
    return max(1, int(num_train_scenarios))


def _resolve_validation_frequency(
    validation_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    adapter,
    train_env_configs: list[dict[str, Any]],
    *,
    train_epochs: int | None,
    num_envs: int,
) -> int:
    configured_frequency = validation_cfg.get("frequency")
    if configured_frequency is not None:
        return max(0, int(configured_frequency))

    evals_per_epoch_cfg = validation_cfg.get("evals_per_epoch")
    if evals_per_epoch_cfg is None or train_epochs is None:
        fallback_frequency = agent_cfg.get("checkpoint_frequency")
        if fallback_frequency is None:
            return 0
        return max(0, int(fallback_frequency))

    evals_per_epoch = int(evals_per_epoch_cfg)
    return _resolve_epoch_timestep_frequency(
        adapter,
        train_env_configs,
        intervals_per_epoch=evals_per_epoch,
        num_envs=num_envs,
        setting_name="runtime.validation.frequency=null with runtime.validation.evals_per_epoch",
    )


def _load_named_val_env_configs(
    raw_val_sets: Any,
) -> tuple[list[dict[str, Any]], str, Path] | None:
    val_sets = normalize_val_sets(raw_val_sets)
    if not val_sets:
        return None

    all_configs: list[dict[str, Any]] = []
    env_name: str | None = None
    first_path: Path | None = None
    for validation_set, files in val_sets.items():
        configs, set_env_name, dataset_path = load_env_configs_from_parquet(files)
        if env_name is None:
            env_name = set_env_name
            first_path = dataset_path
        elif set_env_name != env_name:
            raise ValueError(
                f"NN validation env_name mismatch across data.val_sets: "
                f"{env_name!r} vs {set_env_name!r}"
            )
        all_configs.extend(
            annotate_env_config_with_validation_set(config, validation_set)
            for config in configs
        )
    assert env_name is not None and first_path is not None
    return all_configs, env_name, first_path


DEFAULT_VEC_ENV = "subproc"
DEFAULT_SUBPROC_START_METHOD = "fork"


def _resolve_vec_env_settings(runtime_cfg: dict[str, Any]) -> tuple[str, str | None]:
    raw_vec_env = str(
        runtime_cfg.get("vec_env", DEFAULT_VEC_ENV) or DEFAULT_VEC_ENV
    ).strip().lower()
    aliases = {
        "dummy": "dummy",
        "sync": "dummy",
        "subproc": "subproc",
        "subprocess": "subproc",
    }
    vec_env = aliases.get(raw_vec_env)
    if vec_env is None:
        raise ValueError(
            f"Unsupported runtime.vec_env={raw_vec_env!r}; expected 'dummy' or 'subproc'."
        )

    default_start_method = DEFAULT_SUBPROC_START_METHOD if vec_env == "subproc" else None
    raw_start_method = runtime_cfg.get("subproc_start_method", default_start_method)
    if raw_start_method is None:
        start_method = None
    else:
        start_method = str(raw_start_method).strip()
        if not start_method or start_method.lower() in {"none", "null"}:
            start_method = None

    return vec_env, start_method


def _save_resolved_config(path: Path, config_yaml: str) -> None:
    with open(path / "config.yaml", "w", encoding="utf-8") as f:
        f.write(config_yaml)


def _model_save_stem(save_dir: Path) -> str:
    return str((save_dir / "agent").resolve())


def _vecnormalize_path(save_dir: Path) -> Path:
    return save_dir / "vecnormalize.pkl"


def _trainer_state_path(save_dir: Path) -> Path:
    return save_dir / "trainer_state.json"


def _save_model_bundle(
    model: PPO,
    vecnormalize: VecNormalize | None,
    save_dir: Path,
    *,
    trainer_state: dict[str, Any] | None = None,
    config_yaml: str | None = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(_model_save_stem(save_dir))
    if vecnormalize is not None:
        vecnormalize.save(str(_vecnormalize_path(save_dir)))
    if trainer_state is not None:
        with open(_trainer_state_path(save_dir), "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, indent=2, sort_keys=True)
    if config_yaml is not None:
        _save_resolved_config(save_dir, config_yaml)


def _checkpoint_is_complete(path: Path) -> bool:
    model_path = path / "agent.zip"
    return (
        path.is_dir()
        and model_path.is_file()
        and _vecnormalize_path(path).is_file()
        and _trainer_state_path(path).is_file()
        and (path / "config.yaml").is_file()
    )


def _checkpoint_step(path: Path) -> int | None:
    try:
        return int(path.name.split("_")[-1])
    except (TypeError, ValueError):
        return None


def _checkpoint_rank(path: Path) -> int | None:
    step = _checkpoint_step(path)
    if step is not None:
        return step
    try:
        state = _load_trainer_state(path)
        return int(state.get("global_step"))
    except (OSError, TypeError, ValueError):
        return None


def _resolve_resume_checkpoint(
    experiment_dir: Path,
    resume_cfg: dict[str, Any] | None,
) -> Path | None:
    cfg = dict(resume_cfg or {})
    mode = str(cfg.get("mode", "never") or "never")
    if mode == "never":
        return None
    if mode == "auto_latest":
        checkpoint_root = experiment_dir / "checkpoints"
        candidate_paths = []
        if checkpoint_root.is_dir():
            candidate_paths.extend(checkpoint_root.glob("step_*"))
        candidate_paths.append(experiment_dir)
        candidates = [
            path
            for path in candidate_paths
            if _checkpoint_is_complete(path) and _checkpoint_rank(path) is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: int(_checkpoint_rank(path) or 0))
    if mode == "path":
        raw_path = cfg.get("path")
        if not raw_path:
            raise ValueError("runtime.resume.mode=path requires runtime.resume.path.")
        path = resolve_repo_path(raw_path)
        if path.is_file() and path.name == "agent.zip":
            path = path.parent
        if not _checkpoint_is_complete(path):
            raise FileNotFoundError(
                "Resume checkpoint must contain agent.zip, vecnormalize.pkl, and trainer_state.json: "
                f"{path}"
            )
        return path
    raise ValueError(f"Unsupported runtime.resume.mode={mode!r}; expected never, auto_latest, or path.")


def _load_trainer_state(checkpoint_dir: Path) -> dict[str, Any]:
    with open(_trainer_state_path(checkpoint_dir), encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid trainer_state.json in {checkpoint_dir}")
    return state


def _training_signature(
    *,
    env_name: str,
    train_dataset_path: Path,
    val_dataset_path: Path | None,
    num_train_scenarios: int,
    num_val_scenarios: int,
    num_envs: int,
    sample_with_replacement: bool,
    shard_train_scenarios: bool,
    train_epochs: int | None,
) -> dict[str, Any]:
    return {
        "env_name": env_name,
        "train_dataset_path": str(train_dataset_path.resolve()),
        "val_dataset_path": None if val_dataset_path is None else str(val_dataset_path.resolve()),
        "num_train_scenarios": int(num_train_scenarios),
        "num_val_scenarios": int(num_val_scenarios),
        "num_envs": int(num_envs),
        "sample_with_replacement": bool(sample_with_replacement),
        "shard_train_scenarios_across_envs": bool(shard_train_scenarios),
        "train_epochs": train_epochs,
    }


def _validate_resume_state(
    resume_state: dict[str, Any],
    current_signature: dict[str, Any],
) -> None:
    previous_signature = resume_state.get("training_signature")
    if not isinstance(previous_signature, dict):
        raise ValueError("Resume checkpoint is missing a valid NN PPO training signature.")

    previous_comparable = dict(previous_signature)
    current_comparable = dict(current_signature)
    previous_train_epochs = previous_comparable.pop("train_epochs", None)
    current_train_epochs = current_comparable.pop("train_epochs", None)

    if previous_comparable != current_comparable:
        raise ValueError(
            "Resume checkpoint does not match current NN PPO training configuration. "
            "Start a new run or use the same dataset/env overrides/worker sharding settings."
        )
    if previous_train_epochs == current_train_epochs:
        return
    if previous_train_epochs is None or current_train_epochs is None:
        raise ValueError(
            "Resume checkpoint changes runtime.train_epochs between configured and unset. "
            "Start a new run or resume with an explicit epoch target."
        )
    if int(current_train_epochs) < int(previous_train_epochs):
        raise ValueError(
            "Resume checkpoint was created for a larger runtime.train_epochs target. "
            "Start a new run or resume with an equal/larger epoch target."
        )


class ValidationAndCheckpointCallback(BaseCallback):
    """Trainer-native validation and checkpointing for NN PPO runs."""

    def __init__(
        self,
        *,
        adapter,
        val_env_configs: list[dict[str, Any]] | None,
        experiment_dir: Path,
        checkpoint_frequency: int,
        checkpoint_episode_frequency: int,
        validation_enabled: bool,
        val_before_train: bool,
        validation_frequency: int,
        validation_seed: int,
        validation_num_repeats: int,
        validation_axis: str | None,
        validation_log_to_wandb: bool,
        deterministic_eval: bool,
        track_wandb: bool,
        total_train_episodes: int | None,
        train_epochs: int | None,
        initial_completed_train_episodes: int,
        initial_last_checkpoint_step: int,
        initial_last_checkpoint_episode: int,
        initial_last_validation_step: int,
        base_trainer_state: dict[str, Any],
        config_yaml: str,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.val_env_configs = list(val_env_configs or [])
        self.experiment_dir = experiment_dir
        self.checkpoint_frequency = max(0, int(checkpoint_frequency))
        self.checkpoint_episode_frequency = max(0, int(checkpoint_episode_frequency))
        self.validation_enabled = bool(validation_enabled) and bool(self.val_env_configs)
        self.val_before_train = bool(val_before_train)
        self.validation_frequency = max(0, int(validation_frequency))
        self.validation_seed = int(validation_seed)
        self.validation_num_repeats = max(1, int(validation_num_repeats))
        self.validation_axis = str(validation_axis or "").strip() or None
        self.validation_log_to_wandb = bool(validation_log_to_wandb)
        self.deterministic_eval = bool(deterministic_eval)
        self.track_wandb = bool(track_wandb)
        self.total_train_episodes = (
            None if total_train_episodes is None else max(1, int(total_train_episodes))
        )
        self.train_epochs = train_epochs
        self._last_checkpoint_step = int(initial_last_checkpoint_step)
        self._last_checkpoint_episode = int(initial_last_checkpoint_episode)
        self._last_validation_step = int(initial_last_validation_step)
        self._validation_history_path = self.experiment_dir / "validation_history.jsonl"
        self._train_env_infos: dict[str, list[Any]] = defaultdict(list)
        self._completed_train_episodes = int(initial_completed_train_episodes)
        self.base_trainer_state = dict(base_trainer_state)
        self.config_yaml = config_yaml

    def _count_completed_train_episodes(self) -> None:
        dones = self.locals.get("dones")
        if dones is None:
            return
        done_count = int(np.asarray(dones, dtype=np.int64).sum())
        if done_count <= 0:
            return
        self._completed_train_episodes += done_count
        self.logger.record("train/episodes_completed", int(self._completed_train_episodes))

    def _should_stop_for_episode_budget(self) -> bool:
        if self.total_train_episodes is None:
            return False
        return self._completed_train_episodes >= self.total_train_episodes

    def _collect_train_episode_metrics(self) -> None:
        infos = self.locals.get("infos") or []
        for info in infos:
            if not isinstance(info, dict):
                continue
            trajectory_metrics = collect_trajectory_metrics(info)
            if not trajectory_metrics:
                continue
            env_config = dict(info.get("env_config") or {})
            for key, value in trajectory_metrics.items():
                self._train_env_infos[key].append(value)
            for group_name, group_value in self.adapter.group_labels(env_config, info).items():
                self._train_env_infos[f"group_label/{group_name}"].append(group_value)

    def _flush_train_episode_metrics(self) -> None:
        if not self._train_env_infos:
            return

        payload: dict[str, float | int] = {
            "global_step": int(self.num_timesteps),
        }
        add_env_metrics(payload, self._train_env_infos, prefix="train-env", include_grouped=True)
        for key, value in payload.items():
            if key != "global_step":
                self.logger.record(key, value)

        if self.track_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(payload, step=int(self.num_timesteps))
            except Exception as exc:
                print(f"[nn_ppo] wandb train metric log error: {exc}")

        self._train_env_infos.clear()

    def trainer_state(self) -> dict[str, Any]:
        state = dict(self.base_trainer_state)
        state.update(
            {
                "global_step": int(self.num_timesteps),
                "completed_train_episodes": int(self._completed_train_episodes),
                "target_train_episodes": self.total_train_episodes,
                "train_epochs": self.train_epochs,
                "last_checkpoint_step": int(self._last_checkpoint_step),
                "last_checkpoint_episode": int(self._last_checkpoint_episode),
                "last_validation_step": int(self._last_validation_step),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        return state

    def _write_validation_record(self, record: dict[str, Any]) -> None:
        with open(self._validation_history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _run_validation(self, stage: str) -> None:
        if not self.validation_enabled:
            return
        vecnormalize = self.model.get_vec_normalize_env()
        _, _, metric_dict = evaluate_model_on_dataset(
            self.model,
            self.adapter,
            self.val_env_configs,
            vecnormalize=vecnormalize,
            deterministic=self.deterministic_eval,
            seed=self.validation_seed,
            num_repeats=self.validation_num_repeats,
            validation_axis=self.validation_axis,
        )

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "global_step": int(self.num_timesteps),
            "metrics": metric_dict,
        }
        self._write_validation_record(record)

        for key, value in metric_dict.items():
            self.logger.record(key, value)

        if self.track_wandb and self.validation_log_to_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    payload = {"global_step": int(self.num_timesteps)}
                    payload.update(metric_dict)
                    wandb.log(payload, step=int(self.num_timesteps))
            except Exception as exc:
                print(f"[nn_ppo] wandb validation log error: {exc}")

    def _maybe_checkpoint(self) -> None:
        if self.checkpoint_episode_frequency > 0:
            if (
                self._completed_train_episodes - self._last_checkpoint_episode
                < self.checkpoint_episode_frequency
            ):
                return
            self._last_checkpoint_episode = int(self._completed_train_episodes)
            self._last_checkpoint_step = int(self.num_timesteps)
            checkpoint_dir = self.experiment_dir / "checkpoints" / f"step_{self.num_timesteps:08d}"
            _save_model_bundle(
                self.model,
                self.model.get_vec_normalize_env(),
                checkpoint_dir,
                trainer_state=self.trainer_state(),
                config_yaml=self.config_yaml,
            )
            return

        if self.checkpoint_frequency <= 0:
            return
        if self.num_timesteps - self._last_checkpoint_step < self.checkpoint_frequency:
            return
        self._last_checkpoint_step = int(self.num_timesteps)
        checkpoint_dir = self.experiment_dir / "checkpoints" / f"step_{self.num_timesteps:08d}"
        _save_model_bundle(
            self.model,
            self.model.get_vec_normalize_env(),
            checkpoint_dir,
            trainer_state=self.trainer_state(),
            config_yaml=self.config_yaml,
        )

    def _maybe_validate(self) -> None:
        if not self.validation_enabled or self.validation_frequency <= 0:
            return
        if self.num_timesteps - self._last_validation_step < self.validation_frequency:
            return
        self._last_validation_step = int(self.num_timesteps)
        self._run_validation(stage="periodic")

    def _on_training_start(self) -> None:
        if self.total_train_episodes is not None:
            print(
                f"[nn_ppo] target completed train episodes: {self.total_train_episodes}",
                flush=True,
            )
        if self.validation_enabled and self.val_before_train:
            self._last_validation_step = int(self.num_timesteps)
            self._run_validation(stage="before_train")

    def _on_step(self) -> bool:
        self._count_completed_train_episodes()
        self._collect_train_episode_metrics()
        self._maybe_checkpoint()
        self._maybe_validate()
        if self._should_stop_for_episode_budget():
            print(
                "[nn_ppo] reached target completed train episodes: "
                f"{self._completed_train_episodes}",
                flush=True,
            )
            return False
        return True

    def _on_rollout_end(self) -> None:
        self._flush_train_episode_metrics()

    def _on_training_end(self) -> None:
        self._flush_train_episode_metrics()
        if self.validation_enabled:
            self._run_validation(stage="final")


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(config) -> None:
    OmegaConf.resolve(config)
    print("[nn_ppo] resolved Hydra config", flush=True)
    config_yaml = OmegaConf.to_yaml(config, resolve=True)
    cfg = OmegaConf.to_container(config, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict config, got {type(cfg)!r}")

    data_cfg = cfg.get("data") or {}
    agent_cfg = cfg.get("agent") or {}
    runtime_cfg = cfg.get("runtime") or {}
    output_cfg = cfg.get("output") or {}
    validation_cfg = runtime_cfg.get("validation") or {}
    resume_cfg = runtime_cfg.get("resume") or {}

    train_files = data_cfg.get("train_files")
    if not train_files:
        raise ValueError("Config must specify data.train_files.")
    val_files = data_cfg.get("val_files")
    val_sets = data_cfg.get("val_sets")

    print(f"[nn_ppo] loading train parquet: {train_files}", flush=True)
    train_env_configs, env_name, train_dataset_path = load_env_configs_from_parquet(train_files)
    adapter = create_nn_env_adapter(env_name)

    val_env_configs: list[dict[str, Any]] | None = None
    named_val_load = _load_named_val_env_configs(val_sets)
    if named_val_load is not None:
        val_env_configs, val_env_name, val_dataset_path = named_val_load
        print(f"[nn_ppo] loading named validation sets: {list(normalize_val_sets(val_sets).keys())}", flush=True)
        if val_env_name != env_name:
            raise ValueError(
                f"train/val env_name mismatch: train={env_name!r}, val={val_env_name!r}"
            )
    elif val_files:
        print(f"[nn_ppo] loading val parquet: {val_files}", flush=True)
        val_env_configs, val_env_name, val_dataset_path = load_env_configs_from_parquet(val_files)
        if val_env_name != env_name:
            raise ValueError(
                f"train/val env_name mismatch: train={env_name!r}, val={val_env_name!r}"
            )
    else:
        val_dataset_path = None

    experiment_dir = _ensure_experiment_dir(str(output_cfg.get("save_folder") or "results/nn_train/nn_ppo_default"))
    print(f"[nn_ppo] experiment dir: {experiment_dir}", flush=True)
    _save_resolved_config(experiment_dir, config_yaml)

    exp_name = str(agent_cfg.get("exp_name") or experiment_dir.name)
    seed = int(agent_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_envs = int(agent_cfg.get("num_envs", 1))
    vec_env_kind, subproc_start_method = _resolve_vec_env_settings(runtime_cfg)
    sample_with_replacement = bool(runtime_cfg.get("sample_with_replacement", False))
    shard_train_scenarios = bool(runtime_cfg.get("shard_train_scenarios_across_envs", False))
    train_epochs_cfg = runtime_cfg.get("train_epochs")
    train_epochs = None if train_epochs_cfg is None else int(train_epochs_cfg)
    total_train_episodes = _resolve_target_train_episodes(
        train_epochs_cfg,
        len(train_env_configs),
        sample_with_replacement=sample_with_replacement,
        num_envs=num_envs,
        shard_train_scenarios=shard_train_scenarios,
    )
    total_timesteps = _resolve_total_timesteps(
        agent_cfg,
        adapter,
        train_env_configs,
        total_train_episodes=total_train_episodes,
        num_envs=num_envs,
    )
    validation_frequency = _resolve_validation_frequency(
        validation_cfg,
        agent_cfg,
        adapter,
        train_env_configs,
        train_epochs=train_epochs,
        num_envs=num_envs,
    )
    checkpoint_frequency = _resolve_checkpoint_frequency(
        agent_cfg,
    )
    checkpoint_episode_frequency = _resolve_checkpoint_episode_frequency(
        agent_cfg,
        train_epochs=train_epochs,
        num_train_scenarios=len(train_env_configs),
    )

    training_signature = _training_signature(
        env_name=env_name,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        num_train_scenarios=len(train_env_configs),
        num_val_scenarios=len(val_env_configs or []),
        num_envs=num_envs,
        sample_with_replacement=sample_with_replacement,
        shard_train_scenarios=shard_train_scenarios,
        train_epochs=train_epochs,
    )

    resume_checkpoint_dir = _resolve_resume_checkpoint(experiment_dir, resume_cfg)
    resume_state: dict[str, Any] | None = None
    if resume_checkpoint_dir is not None:
        resume_state = _load_trainer_state(resume_checkpoint_dir)
        _validate_resume_state(resume_state, training_signature)
        print(f"[nn_ppo] resuming from checkpoint: {resume_checkpoint_dir}", flush=True)

    worker_env_configs = _partition_env_configs_for_workers(
        train_env_configs,
        num_envs,
        shard_across_workers=shard_train_scenarios,
    )

    def make_env_factory(worker_idx: int):
        worker_seed = seed + worker_idx
        worker_train_env_configs = worker_env_configs[worker_idx]

        def _factory():
            return ScenarioCyclingEnv(
                adapter,
                worker_train_env_configs,
                sample_with_replacement=sample_with_replacement,
                seed=worker_seed,
            )

        return _factory

    print(f"[nn_ppo] building training VecEnv: {vec_env_kind}", flush=True)
    env_fns = [make_env_factory(i) for i in range(num_envs)]
    if vec_env_kind == "subproc":
        base_vec_env = SubprocVecEnv(env_fns, start_method=subproc_start_method)
    else:
        base_vec_env = DummyVecEnv(env_fns)
    monitored_vec_env = VecMonitor(base_vec_env)
    gamma = float(agent_cfg.get("gamma", 0.99))
    if resume_checkpoint_dir is not None:
        vec_env = VecNormalize.load(str(_vecnormalize_path(resume_checkpoint_dir)), monitored_vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(
            monitored_vec_env,
            norm_obs=True,
            norm_reward=True,
            gamma=gamma,
        )

    n_steps = int(agent_cfg.get("n_steps", agent_cfg.get("num_steps", 128)))
    if "batch_size" in agent_cfg and agent_cfg.get("batch_size") is not None:
        batch_size = int(agent_cfg["batch_size"])
    else:
        num_minibatches = int(agent_cfg.get("num_minibatches", 4))
        rollout_batch = num_envs * n_steps
        if rollout_batch % num_minibatches != 0:
            raise ValueError(
                f"num_envs * n_steps must be divisible by num_minibatches, got {rollout_batch} and {num_minibatches}."
            )
        batch_size = rollout_batch // num_minibatches

    tensorboard_log = str((experiment_dir / "tensorboard").resolve())
    if resume_checkpoint_dir is not None:
        print("[nn_ppo] loading SB3 PPO model", flush=True)
        model = PPO.load(
            str(resume_checkpoint_dir / "agent.zip"),
            env=vec_env,
            device=str(runtime_cfg.get("device", "auto")),
        )
    else:
        print("[nn_ppo] constructing SB3 PPO model", flush=True)
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=float(agent_cfg.get("learning_rate", 2.5e-4)),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(agent_cfg.get("n_epochs", agent_cfg.get("update_epochs", 4))),
            gamma=gamma,
            gae_lambda=float(agent_cfg.get("gae_lambda", 0.95)),
            clip_range=float(agent_cfg.get("clip_range", agent_cfg.get("clip_coef", 0.2))),
            ent_coef=float(agent_cfg.get("ent_coef", 0.0)),
            vf_coef=float(agent_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(agent_cfg.get("max_grad_norm", 0.5)),
            seed=seed,
            verbose=int(agent_cfg.get("verbose", 1)),
            tensorboard_log=tensorboard_log,
            device=str(runtime_cfg.get("device", "auto")),
        )

    wandb_run = None
    track = bool(agent_cfg.get("track", False))
    if track:
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(agent_cfg.get("wandb_project_name", "agrimanager")),
                entity=agent_cfg.get("wandb_entity"),
                name=exp_name,
                config=cfg,
                sync_tensorboard=False,
                dir=str(experiment_dir),
            )
        except Exception as exc:
            print(f"[nn_ppo] wandb init failed, continuing without wandb: {exc}")

    initial_completed_train_episodes = int(
        (resume_state or {}).get("completed_train_episodes", 0)
    )
    initial_last_checkpoint_step = int((resume_state or {}).get("last_checkpoint_step", 0))
    initial_last_checkpoint_episode = int((resume_state or {}).get("last_checkpoint_episode", 0))
    initial_last_validation_step = int((resume_state or {}).get("last_validation_step", 0))

    base_trainer_state = {
        "env_name": env_name,
        "train_dataset_path": str(train_dataset_path.resolve()),
        "val_dataset_path": None if val_dataset_path is None else str(val_dataset_path.resolve()),
        "num_train_scenarios": len(train_env_configs),
        "num_val_scenarios": len(val_env_configs or []),
        "training_signature": training_signature,
        "resume_from": None if resume_checkpoint_dir is None else str(resume_checkpoint_dir.resolve()),
    }

    callback = ValidationAndCheckpointCallback(
        adapter=adapter,
        val_env_configs=val_env_configs,
        experiment_dir=experiment_dir,
        checkpoint_frequency=checkpoint_frequency,
        checkpoint_episode_frequency=checkpoint_episode_frequency,
        validation_enabled=bool(validation_cfg.get("enabled", True)),
        val_before_train=bool(validation_cfg.get("val_before_train", True)),
        validation_frequency=validation_frequency,
        validation_seed=int(validation_cfg.get("seed", 0)),
        validation_num_repeats=int(validation_cfg.get("num_repeats", 1)),
        validation_axis=data_cfg.get("validation_axis"),
        validation_log_to_wandb=bool(validation_cfg.get("log_to_wandb", True)),
        deterministic_eval=bool(validation_cfg.get("deterministic", False)),
        track_wandb=bool(wandb_run is not None),
        total_train_episodes=total_train_episodes,
        train_epochs=train_epochs,
        initial_completed_train_episodes=initial_completed_train_episodes,
        initial_last_checkpoint_step=initial_last_checkpoint_step,
        initial_last_checkpoint_episode=initial_last_checkpoint_episode,
        initial_last_validation_step=initial_last_validation_step,
        base_trainer_state=base_trainer_state,
        config_yaml=config_yaml,
    )

    print("=" * 80)
    print("AgriManager NN PPO Training")
    print("=" * 80)
    print(f"Env name: {env_name}")
    print(f"Train dataset: {train_dataset_path}")
    if val_dataset_path is not None:
        print(f"Val dataset: {val_dataset_path}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Num train scenarios: {len(train_env_configs)}")
    print(f"Num val scenarios: {len(val_env_configs or [])}")
    print(f"Num envs: {num_envs}")
    if shard_train_scenarios:
        worker_sizes = [len(items) for items in worker_env_configs]
        print(
            "[nn_ppo] sharding train scenarios across env workers: "
            f"{worker_sizes}",
            flush=True,
        )
    if total_train_episodes is not None:
        print(f"Train epochs: {int(train_epochs)}")
        print(f"Target completed train episodes: {total_train_episodes}")
        print(f"Completed train episodes at start: {initial_completed_train_episodes}")
    if checkpoint_episode_frequency > 0:
        print(
            "Checkpoint mode: epoch "
            f"(every {checkpoint_episode_frequency} completed train episodes)"
        )
    elif checkpoint_frequency > 0:
        print(f"Checkpoint frequency: every {checkpoint_frequency} timesteps")
    else:
        print("Checkpoint frequency: disabled")
    print(f"Total timesteps cap: {total_timesteps}")
    print("=" * 80)

    try:
        if (
            total_train_episodes is not None
            and initial_completed_train_episodes >= total_train_episodes
        ):
            print("[nn_ppo] checkpoint already reached target train episodes", flush=True)
            _save_model_bundle(
                model,
                vec_env,
                experiment_dir,
                trainer_state=callback.trainer_state(),
                config_yaml=config_yaml,
            )
            return

        current_timesteps = int(model.num_timesteps)
        if current_timesteps >= total_timesteps:
            if total_train_episodes is not None:
                raise RuntimeError(
                    "Resume checkpoint already reached the timestep safety cap before "
                    "completing the requested train epochs. Increase agent.total_timesteps "
                    "or provide a larger adapter.episode_length_hint(env_config)."
                )
            print("[nn_ppo] checkpoint already reached total timestep cap", flush=True)
            _save_model_bundle(
                model,
                vec_env,
                experiment_dir,
                trainer_state=callback.trainer_state(),
                config_yaml=config_yaml,
            )
            return

        learn_timesteps = total_timesteps
        if resume_checkpoint_dir is not None:
            learn_timesteps = total_timesteps - current_timesteps
        print("[nn_ppo] starting PPO.learn()", flush=True)
        model.learn(
            total_timesteps=learn_timesteps,
            callback=callback,
            tb_log_name=exp_name,
            progress_bar=bool(runtime_cfg.get("progress_bar", False)),
            reset_num_timesteps=resume_checkpoint_dir is None,
        )
        print("[nn_ppo] training complete, saving final bundle", flush=True)
        _save_model_bundle(
            model,
            vec_env,
            experiment_dir,
            trainer_state=callback.trainer_state(),
            config_yaml=config_yaml,
        )
    finally:
        if wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        vec_env.close()


if __name__ == "__main__":
    main()
