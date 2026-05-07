"""Scripted baseline rollouts for no-action and random-action policies.

The output schema intentionally matches ``inference_rollout.py`` so scripted
baselines can be normalized against LLM/training validation metrics split by the
same validation axis.
"""

from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import faulthandler
import hydra
import math
import random
import numpy as np
from omegaconf import OmegaConf

agrimanager_root = Path(__file__).parent.parent.parent.parent

from agrimanager.env.base import create_environment
from agrimanager.rollout.inference.inference_rollout import (
    _build_summary,
    _dump_json_atomic,
    _extract_trial_metrics,
    _extract_validation_metric_dict,
    load_dataset,
)

VALID_MODES = {"random", "no_action"}
MODE_ALIASES = {
    "no-action": "no_action",
    "noaction": "no_action",
    "zero": "no_action",
    "zero_action": "no_action",
}


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _flatten_numeric(prefix: str, payload: Dict[str, Any]) -> Dict[str, float | int]:
    """Return numeric leaves from a nested mapping with slash-separated keys."""
    flattened: Dict[str, float | int] = {}
    for key, value in payload.items():
        output_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_numeric(output_key, value))
            continue
        if isinstance(value, bool):
            flattened[output_key] = int(value)
            continue
        if isinstance(value, (int, float)):
            value_float = float(value)
            if math.isfinite(value_float):
                flattened[output_key] = int(value) if isinstance(value, int) else value_float
    return flattened


def _init_wandb_run(
    wandb_cfg: Dict[str, Any] | None,
    *,
    output_path: Path,
    mode: str,
    seed: int,
    num_trials: int,
    validation_axis: str | None,
    test_files: str | list[str] | None,
):
    cfg = dict(wandb_cfg or {})
    if not bool(cfg.get("enabled", False)):
        return None

    try:
        import wandb
    except Exception as exc:
        print(f"[baseline_eval] wandb import failed, continuing without wandb: {exc}")
        return None

    run_name = (
        _clean_str(cfg.get("name"))
        or _clean_str(cfg.get("run_name"))
        or _clean_str(cfg.get("experiment_name"))
        or f"baseline_{mode}_seed{seed}"
    )
    project = (
        _clean_str(cfg.get("project_name"))
        or _clean_str(cfg.get("project"))
        or "agrimanager"
    )
    tags = cfg.get("tags") or ["baseline", mode]
    if isinstance(tags, str):
        tags = [tags]

    run_config = {
        "baseline_mode": mode,
        "seed": int(seed),
        "num_trials": int(num_trials),
        "validation_axis": validation_axis,
        "test_files": test_files,
        "output_dir": str(output_path),
    }

    init_kwargs = {
        "project": project,
        "entity": cfg.get("entity"),
        "name": run_name,
        "group": cfg.get("group"),
        "tags": tags,
        "config": run_config,
        "dir": str(output_path),
    }
    wandb_mode = _clean_str(cfg.get("mode"))
    if wandb_mode:
        init_kwargs["mode"] = wandb_mode

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[baseline_eval] wandb init failed, continuing without wandb: {exc}")
        return None


def _log_trial_to_wandb(
    wandb_run,
    *,
    trial_metrics: Dict[str, Any],
    validation_metrics: Dict[str, Any],
    mode: str,
    trial: int,
    seed: int,
    step: int,
) -> None:
    if wandb_run is None:
        return
    try:
        import wandb

        payload: Dict[str, Any] = {
            "global_step": int(step),
            "baseline/trial": int(trial),
            "baseline/seed": int(seed),
            "baseline/is_random": int(mode == "random"),
            "baseline/is_no_action": int(mode == "no_action"),
        }
        payload.update(_flatten_numeric("baseline", trial_metrics))
        payload.update(_flatten_numeric("", validation_metrics))
        wandb.log(payload, step=int(step))
    except Exception as exc:
        print(f"[baseline_eval] wandb metric log failed: {exc}")


def create_env(env_name: str, env_config: Dict[str, Any]):
    """Create environment instance from configuration.

    Args:
        env_name: Environment name (e.g., 'wofost_gym')
        env_config: Environment configuration dictionary

    Returns:
        Environment instance with AgriManager wrapper metrics enabled.
    """
    # Keep llm_mode=True so every adapter returns the same wrapper metrics that
    # LLM evaluation/training validation consume. Numeric actions are still
    # accepted by the adapters.
    env_config = env_config.copy()
    env_config["llm_mode"] = True

    return create_environment(env_name, env_config)


def get_action_space(env):
    """Get the action space from environment.

    Args:
        env: Environment instance (with llm_mode=False)

    Returns:
        Action space object (typically gym.spaces.Discrete)
    """
    if hasattr(env, "env") and hasattr(env.env, "action_space"):
        return env.env.action_space
    if hasattr(env, "worker") and hasattr(env.worker, "action_space"):
        return env.worker.action_space
    if hasattr(env, "action_space"):
        return env.action_space
    raise AttributeError("Environment does not have an action_space attribute")


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "random").strip().lower().replace(" ", "_")
    normalized = MODE_ALIASES.get(normalized, normalized)
    if normalized not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES | set(MODE_ALIASES)))
        raise ValueError(f"Unsupported baseline mode {mode!r}. Expected one of: {valid}")
    return normalized


def _zero_action(action_space):
    """Return the native no-op action shape for a gym/gymnasium space."""
    if hasattr(action_space, "spaces"):
        spaces = action_space.spaces
        if isinstance(spaces, dict):
            return {key: _zero_action(space) for key, space in spaces.items()}
        return tuple(_zero_action(space) for space in spaces)

    if hasattr(action_space, "nvec"):
        return np.zeros_like(np.asarray(action_space.nvec), dtype=np.int64)

    if hasattr(action_space, "shape") and getattr(action_space, "shape", None):
        dtype = getattr(action_space, "dtype", np.float32)
        action = np.zeros(action_space.shape, dtype=dtype)
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

    return 0


def _select_action(mode: str, action_space):
    if mode == "no_action":
        return _zero_action(action_space)
    return action_space.sample()


def _seed_action_spaces(action_spaces: List[Any], seed: int) -> None:
    for idx, action_space in enumerate(action_spaces):
        seed_fn = getattr(action_space, "seed", None)
        if callable(seed_fn):
            seed_fn(int(seed) + idx)


def _executed_action(info: Dict[str, Any], fallback: Any) -> Any:
    if "executed_action_id" in info:
        return info["executed_action_id"]
    if "executed_action" in info:
        return info["executed_action"]
    if "action_applied" in info:
        return info["action_applied"]
    return fallback


def _scalar_action_id(action: Any) -> int | None:
    try:
        array = np.asarray(action)
        if array.shape == ():
            return int(array.item())
    except Exception:
        pass
    try:
        return int(action)
    except Exception:
        return None


def _run_single_trial(
    *,
    envs,
    env_configs: List[Dict[str, Any]],
    action_spaces: List[Any],
    turn_num: int,
    mode: str,
) -> List[Dict[str, Any]]:
    results = [
        {
            "env_id": i,
            "env_config": env_configs[i],
            "turns": [],
        }
        for i in range(len(envs))
    ]

    observations = []
    done_flags = []
    for env in tqdm(envs, desc="Resetting"):
        _, info = env.reset()
        observations.append((info or {}).get("observation", {}))
        done_flags.append(False)

    for turn in tqdm(range(turn_num), desc="Turns", leave=False):
        active_indices = [i for i, done in enumerate(done_flags) if not done]
        if not active_indices:
            break

        for i in active_indices:
            current_observation = observations[i]
            action = _select_action(mode, action_spaces[i])

            try:
                _, reward, done, info = envs[i].step(action)
                info = info or {}
                executed_action = _executed_action(info, action)
                action_id = _scalar_action_id(executed_action)

                turn_record = {
                    "turn": turn,
                    "turn_prompt": None,
                    "observation": current_observation,
                    "raw_llm_response": None,
                    "llm_reasoning": None,
                    "retries": 0,
                    "executed_action": executed_action,
                    "executed_action_id": action_id,
                    "invalid_action": bool(info.get("invalid_action", False)),
                    "reward": float(reward),
                    "done": bool(done),
                    "post_turn_metrics": info.get("turn_metrics", {}),
                }
                if done:
                    turn_record["trajectory_metrics"] = info.get("trajectory_metrics", {})
                results[i]["turns"].append(turn_record)

                if done:
                    done_flags[i] = True
                    observations[i] = None
                else:
                    observations[i] = info.get("observation", {})

            except Exception as exc:
                print(f"\nError in env {i} at turn {turn}: {exc}")
                results[i]["turns"].append(
                    {
                        "turn": turn,
                        "turn_prompt": None,
                        "observation": current_observation,
                        "raw_llm_response": None,
                        "llm_reasoning": None,
                        "retries": 0,
                        "executed_action": action,
                        "executed_action_id": _scalar_action_id(action),
                        "invalid_action": None,
                        "reward": None,
                        "done": True,
                        "error": str(exc),
                        "post_turn_metrics": {},
                    }
                )
                done_flags[i] = True
                observations[i] = None

        if all(done_flags):
            break

    return results


def run_random_rollout(
    output_dir: str,
    seed: int = 42,
    mode: str = "random",
    num_trials: int = 1,
    validation_axis: str | None = None,
    wandb_config: Dict[str, Any] | None = None,
    *,
    test_files: str | list[str] | None = None,
    env_configs: List[Dict[str, Any]] | None = None,
    env_name: str = "wofost_gym",
):
    """Run a scripted baseline rollout on a dataset.

    Args:
        output_dir: Directory to save results
        seed: Random seed for reproducibility
        mode: Action selection mode ('random' or 'no_action')
        num_trials: Number of repeated rollout trials
        validation_axis: Optional training validation axis for val-env metrics
        wandb_config: Optional experiment-tracking logging configuration. When enabled, only
            trajectory-level aggregate scalars are logged; per-turn results are
            kept local in results.json and are not uploaded.
    """
    mode = _normalize_mode(mode)
    num_trials = max(1, int(num_trials))

    print("=" * 80)
    print("Baseline Rollout Configuration")
    print("=" * 80)
    if test_files:
        print(f"Test files: {test_files}")
    else:
        print("Dataset source: in-memory dataset artifact")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"Mode: {mode}")
    print(f"Num trials: {num_trials}")
    if validation_axis:
        print(f"Validation axis: {validation_axis}")
    print("=" * 80)

    random.seed(seed)
    np.random.seed(seed)

    print("\n[1/4] Loading dataset...")
    if env_configs is None:
        if not test_files:
            raise ValueError("run_random_rollout requires test_files or env_configs.")
        env_configs, env_name = load_dataset(test_files)
    num_envs = len(env_configs)

    # Create environments
    print(f"\n[2/4] Creating {num_envs} environments...")
    envs = []
    configs = []
    action_spaces = []

    for env_config in tqdm(env_configs, desc="Creating envs"):
        row_env_name = env_config.get("env_name", env_name)
        if row_env_name == "__mixed__":
            raise ValueError("Mixed datasets require every env_config to include env_name.")
        env, config = create_env(row_env_name, env_config)
        envs.append(env)
        configs.append(config)
        action_spaces.append(get_action_space(env))

    turn_num = max(int(config.turn_num) for config in configs)
    print(f"Episodes will run for {turn_num} turns")
    print(f"Action space: {action_spaces[0]}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_trial_metrics = []
    wandb_run = _init_wandb_run(
        wandb_config,
        output_path=output_path,
        mode=mode,
        seed=int(seed),
        num_trials=num_trials,
        validation_axis=validation_axis,
        test_files=test_files,
    )
    wandb_step = int((wandb_config or {}).get("step", 0))

    try:
        print(f"\n[3/4] Preparing scripted policy...")
        print(f"Policy mode: {mode}")

        print(f"\n[4/4] Running {'rollout' if num_trials == 1 else f'{num_trials} trials'}...")
        for trial in range(num_trials):
            trial_seed = int(seed) + trial
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            _seed_action_spaces(action_spaces, trial_seed)

            if num_trials > 1:
                print(f"\n{'-' * 60}")
                print(f"Trial {trial + 1}/{num_trials} (seed={trial_seed})")
                print(f"{'-' * 60}")

            results = _run_single_trial(
                envs=envs,
                env_configs=env_configs,
                action_spaces=action_spaces,
                turn_num=turn_num,
                mode=mode,
            )

            trial_dir = output_path if num_trials == 1 else output_path / f"trial_{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            _dump_json_atomic(trial_dir / "results.json", results, "baseline results")
            trial_metrics = _extract_trial_metrics(results)
            trial_metrics["baseline_mode"] = mode
            trial_metrics["seed"] = trial_seed
            all_trial_metrics.append(trial_metrics)
            _dump_json_atomic(trial_dir / "metrics.json", trial_metrics, "baseline metrics")

            validation_metrics = _extract_validation_metric_dict(
                results,
                validation_axis=validation_axis,
            )
            _dump_json_atomic(
                trial_dir / "validation_metrics.json",
                validation_metrics,
                "training-style validation metrics",
            )
            _log_trial_to_wandb(
                wandb_run,
                trial_metrics=trial_metrics,
                validation_metrics=validation_metrics,
                mode=mode,
                trial=trial,
                seed=trial_seed,
                step=wandb_step + trial,
            )

            primary_metric = trial_metrics["primary_metric"]
            print(
                f"  {primary_metric['label']} mean: {primary_metric['mean']:.2f} "
                f"+/- {primary_metric['std']:.2f}"
            )
            print(f"  Invalid rate: {trial_metrics['invalid_action_rate_mean']:.4f}")

        if num_trials > 1:
            summary = _build_summary(all_trial_metrics)
            summary["baseline_mode"] = mode
            summary["seed"] = int(seed)
            _dump_json_atomic(output_path / "summary.json", summary, "cross-trial summary")
            primary_metric = summary["primary_metric"]
            print(f"\nCross-trial {primary_metric['label']}: {primary_metric['mean']:.2f} +/- {primary_metric['std']:.2f}")
        else:
            metrics = all_trial_metrics[0]
            primary_metric = metrics["primary_metric"]
            print(f"\n{'=' * 80}")
            print("Baseline Rollout Summary")
            print(f"{'=' * 80}")
            print(f"Total environments: {metrics['num_envs']}")
            print(
                f"{primary_metric['label']} mean: "
                f"{primary_metric['mean']:.2f} +/- {primary_metric['std']:.2f}"
            )
            print(f"Invalid rate: {metrics['invalid_action_rate_mean']:.4f}")
            print(f"{'=' * 80}")

    finally:
        for env in envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        if wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception as exc:
                print(f"[baseline_eval] wandb finish failed: {exc}")

    print(f"\nResults saved to: {output_path}")


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from baseline rollout results."""
    final_wsos = []
    total_turns_list = []
    per_crop: Dict[str, list] = {}

    for r in results:
        last_turn = r['turns'][-1] if r['turns'] else {}
        traj = last_turn.get('trajectory_metrics', {})
        wso = traj.get('final_wso', last_turn.get('post_turn_metrics', {}).get('wso', 0.0))
        final_wsos.append(wso)
        total_turns_list.append(len(r['turns']))

        crop = r.get('env_config', {}).get('crop_name', 'unknown')
        per_crop.setdefault(crop, []).append(wso)

    metrics = {
        'final_wso_mean': float(np.mean(final_wsos)),
        'final_wso_std': float(np.std(final_wsos)),
        'invalid_action_rate_mean': 0.0,  # baselines use numerical actions, no invalid actions
        'avg_turns': float(np.mean(total_turns_list)),
        'num_envs': len(results),
    }

    crop_summary = {}
    for crop, wsos in sorted(per_crop.items()):
        crop_summary[crop] = {
            'final_wso_mean': float(np.mean(wsos)),
            'final_wso_std': float(np.std(wsos)),
            'count': len(wsos),
        }
    metrics['per_crop'] = crop_summary

    return metrics


@hydra.main(
    config_path="../../../entrypoints/eval/config",
    config_name="random",
    version_base=None,
)
def main(config) -> None:
    faulthandler.enable(all_threads=True)
    OmegaConf.resolve(config)
    cfg = OmegaConf.to_container(config, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict config, got {type(cfg)!r}")

    data_cfg = cfg.get("data") or {}
    runtime_cfg = cfg.get("runtime") or {}
    output_cfg = cfg.get("output") or {}
    wandb_cfg = cfg.get("wandb") or {}

    inference_file = data_cfg.get("inference_file") or cfg.get("test_files")
    if not inference_file:
        raise ValueError("Config must specify data.inference_file.")

    run_random_rollout(
        test_files=inference_file,
        output_dir=str(output_cfg.get("dir") or cfg.get("output_dir", "./rollout_results")),
        seed=int(runtime_cfg.get("seed", cfg.get("seed", 42))),
        mode=str(runtime_cfg.get("mode", cfg.get("mode", "random"))),
        num_trials=int(runtime_cfg.get("num_trials", cfg.get("num_trials", 1))),
        validation_axis=data_cfg.get("validation_axis"),
        wandb_config=wandb_cfg,
    )


if __name__ == "__main__":
    main()
