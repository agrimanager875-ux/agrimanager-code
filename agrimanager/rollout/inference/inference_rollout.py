"""Inference rollout script for batch environment testing with LLM agents.

This script loads a dataset of environment configurations, creates environments,
and performs parallel batch inference using a language model.

Supports repeated evaluation: the model is loaded once, environments are reset
per trial, and a summary with mean/std is written at the end.
"""

import faulthandler
import json
import os
from pathlib import Path
import re
import traceback
from typing import List, Dict, Any

import hydra
from tqdm import tqdm
from omegaconf import OmegaConf

agrimanager_root = Path(__file__).parent.parent.parent.parent

from agrimanager.env.base import create_environment, load_env_configs_from_parquet
from agrimanager.model_interface.model_factory import create_model
from agrimanager.adapter.trainer.validation_metrics import (
    add_axis_env_metrics,
    add_env_metrics,
)


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _dump_json_atomic(path: Path, payload: Any, label: str):
    """Write JSON atomically so crashes do not leave a truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    print(f"Writing {label} to: {path}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        print(f"Failed while writing {label} to: {path}")
        print(traceback.format_exc())
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    print(f"Finished writing {label} to: {path}")


def _as_file_list(test_files: Any) -> list[str]:
    if isinstance(test_files, (list, tuple)):
        return [str(item) for item in test_files]
    return [str(test_files)]


def load_dataset(test_files: str | list[str]) -> tuple:
    """Load dataset configurations from parquet file.

    Returns:
        (configs, env_name) — env_name is parsed from the path or data_source.
    """
    files = _as_file_list(test_files)
    all_configs: list[dict[str, Any]] = []
    env_names: list[str] = []
    dataset_paths: list[str] = []
    for file_path in files:
        configs, env_name, dataset_path = load_env_configs_from_parquet(file_path)
        for env_config in configs:
            env_config.setdefault("env_name", env_name)
        all_configs.extend(configs)
        env_names.append(env_name)
        dataset_paths.append(str(dataset_path))
    print(f"Loading dataset from: {dataset_paths if len(dataset_paths) > 1 else dataset_paths[0]}")
    print(f"Loaded {len(all_configs)} environment configurations")
    env_name = env_names[0] if len(set(env_names)) == 1 else "__mixed__"
    return all_configs, env_name

def _build_step_input(response, config):
    """Build the string to pass to env.step() and extract reasoning.

    Handles two response shapes:

    1. **dict** — returned when ``return_metadata=True`` (model provider
       already separated ``content`` and ``reasoning``).
    2. **plain str** — returned otherwise.  If the string contains a
       ``<think>`` block, it means a reasoning model is used without
       ``reasoning_effort`` in the model config, which is a
       configuration error.
    """
    if isinstance(response, dict):
        content = response.get("content", "")
        reasoning = response.get("reasoning")
        return content, reasoning
    
    # Plain string — should NOT contain <think> tags
    # if "<think>" in response or "</think>" in response:
    #     raise ValueError(
    #         "Model returned a plain string containing <think> tags. "
    #         "This indicates a reasoning model is being used without "
    #         "'reasoning_effort' set in the model config YAML. "
    #         "Please add 'reasoning_effort: low/medium/high' to your "
    #         "model config so that thinking content is properly extracted."
    #     )

    if not isinstance(response, str):
        return response, None

    
    # Plain strings may still include explicit reasoning blocks because our
    # prompts can require <tool_call>...</tool_call><answer>...</answer> or
    # similar configured tags.
    think_tag = getattr(config, "think_tag", None)
    candidate_tags = []
    if isinstance(think_tag, str) and think_tag.strip():
        candidate_tags.append(think_tag.strip())
    if "think" not in candidate_tags:
        candidate_tags.append("think")

    for tag in candidate_tags:
        escaped = re.escape(tag)
        match = re.search(
            rf"<{escaped}>\s*(.*?)\s*</{escaped}>",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            reasoning = match.group(1).strip() or None
            content = re.sub(
                rf"<{escaped}>\s*.*?\s*</{escaped}>",
                "",
                response,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            return content, reasoning

    return response, None


def _run_single_trial(
    envs,
    configs,
    env_configs,
    system_prompts,
    model,
    turn_num: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> List[Dict[str, Any]]:
    """Run one complete rollout trial. Returns the results list."""
    num_envs = len(envs)

    # Initialize storage
    results = []
    for i in range(num_envs):
        results.append({
            'env_id': i,
            'env_config': env_configs[i],
            'turns': []
        })

    # Reset all environments
    turn_prompts = []
    observations = []
    done_flags = []
    for i, env in enumerate(envs):
        obs, info = env.reset()
        turn_prompts.append(obs)
        observations.append((info or {}).get('observation', {}))
        done_flags.append(False)

    # Run rollout
    for turn in tqdm(range(turn_num), desc="Turns", leave=False):
        active_indices = [i for i, done in enumerate(done_flags) if not done]
        if not active_indices:
            break

        # Build messages for active envs
        messages_list = []
        for i in active_indices:
            messages_list.append([
                {"role": "system", "content": system_prompts[i]},
                {"role": "user", "content": turn_prompts[i]}
            ])

        # Batch generate
        llm_responses = model.generate(
            messages_list, temperature=temperature, max_tokens=max_tokens
        )

        new_turn_prompts = list(turn_prompts)
        new_observations = list(observations)

        for idx, i in enumerate(active_indices):
            llm_response = llm_responses[idx]
            messages = messages_list[idx]
            attempt = 0
            current_prompt = turn_prompts[i]
            current_observation = observations[i]

            while True:
                try:
                    step_input, llm_reasoning = _build_step_input(llm_response, configs[i])
                    obs, reward, done, info = envs[i].step(step_input)
                    info = info or {}
                    raw_llm_response = (
                        info.get('raw_llm_response', step_input)
                    )

                    # Each turn records one full transition:
                    # pre-step prompt/observation -> action -> post-step metrics.
                    turn_record = {
                        'turn': turn,
                        'turn_prompt': current_prompt,
                        'observation': current_observation,
                        'raw_llm_response': raw_llm_response,
                        'llm_reasoning': llm_reasoning,
                        'retries': attempt,
                        'executed_action_id': info.get('executed_action_id'),
                        'invalid_action': info.get('invalid_action', False),
                        'reward': float(reward),
                        'done': bool(done),
                        'post_turn_metrics': info.get('turn_metrics', {}),
                    }
                    if done:
                        turn_record['trajectory_metrics'] = info.get('trajectory_metrics', {})
                    results[i]['turns'].append(turn_record)

                    if done:
                        done_flags[i] = True
                        new_turn_prompts[i] = None
                        new_observations[i] = None
                    else:
                        new_turn_prompts[i] = obs
                        new_observations[i] = info.get('observation', {})
                    break

                except Exception as e:
                    attempt += 1
                    if attempt > max_retries:
                        print(f"\nError in env {i} at turn {turn} after {attempt} attempts: {e}")
                        results[i]['turns'].append({
                            'turn': turn,
                            'turn_prompt': current_prompt,
                            'observation': current_observation,
                            'raw_llm_response': llm_response if isinstance(llm_response, str) else None,
                            'llm_reasoning': None,
                            'retries': attempt,
                            'executed_action_id': None,
                            'invalid_action': None,
                            'reward': None,
                            'done': True,
                            'post_turn_metrics': {},
                            'error': str(e),
                        })
                        done_flags[i] = True
                        new_turn_prompts[i] = None
                        new_observations[i] = None
                        break

                    print(f"\nInvalid response in env {i} at turn {turn} (attempt {attempt}/{max_retries}), regenerating...")
                    llm_response = model.generate(
                        [messages], temperature=temperature, max_tokens=max_tokens
                    )[0]

        turn_prompts = new_turn_prompts
        observations = new_observations
        if all(done_flags):
            break

    return results


def _extract_trial_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract aggregate metrics from a single trial's results."""
    import numpy as np

    def _extract_result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
        turns = result.get("turns", [])
        traj = {}
        for turn in turns:
            if "trajectory_metrics" in turn:
                traj = turn["trajectory_metrics"] or {}
                break

        last_turn = turns[-1] if turns else {}
        post_turn = last_turn.get("post_turn_metrics", {}) or {}

        reward_values = [
            float(turn["reward"])
            for turn in turns
            if turn.get("reward") is not None
        ]
        cumulative_reward = float(sum(reward_values))
        episode_reward = traj.get("episode_reward")
        if episode_reward is None:
            episode_reward = cumulative_reward

        objective_reward = traj.get("objective_reward")
        if objective_reward is not None:
            try:
                objective_reward = float(objective_reward)
                if not np.isfinite(objective_reward):
                    objective_reward = None
            except Exception:
                objective_reward = None

        target_yield = traj.get("target_yield")
        if target_yield is not None:
            try:
                target_yield = float(target_yield)
                if not np.isfinite(target_yield):
                    target_yield = None
            except Exception:
                target_yield = None

        final_wso = traj.get("final_wso")
        if final_wso is None and "wso" in post_turn:
            final_wso = post_turn["wso"]

        group = (
            result.get("env_config", {}).get("crop_name")
            or result.get("env_config", {}).get("env_id")
            or "unknown"
        )
        env_config = result.get("env_config", {}) or {}
        group_labels = env_config.get("trajectory_group_labels") or {}
        simulator = (
            group_labels.get("simulator")
            or env_config.get("env_name")
            or "unknown"
        )

        return {
            "trajectory_metrics": traj,
            "episode_reward": float(episode_reward),
            "objective_reward": objective_reward,
            "target_yield": target_yield,
            "final_wso": float(final_wso) if final_wso is not None else None,
            "invalid_action_rate": float(traj.get("invalid_action_rate", 0.0)),
            "total_steps": int(traj.get("total_steps", len(turns))),
            "group": str(group),
            "simulator": str(simulator),
        }

    def _primary_metric_descriptor(result_summaries: List[Dict[str, Any]]) -> tuple[str, str, str]:
        if result_summaries and all(summary["objective_reward"] is not None for summary in result_summaries):
            return "objective_reward", "Objective reward", "objective_reward"
        if result_summaries and all(summary["target_yield"] is not None for summary in result_summaries):
            return "target_yield", "Target yield", "target_yield"
        if any(summary["trajectory_metrics"].get("episode_reward") is not None for summary in result_summaries):
            return "episode_reward", "Episode reward", "episode_reward"
        if result_summaries and all(summary["final_wso"] is not None for summary in result_summaries):
            return "final_wso", "Final WSO", "final_wso"
        return "episode_reward", "Cumulative reward", "episode_reward"

    result_summaries = [_extract_result_summary(result) for result in results]
    primary_key, primary_label, source_key = _primary_metric_descriptor(result_summaries)

    episode_rewards = [summary["episode_reward"] for summary in result_summaries]
    objective_rewards = [
        summary["objective_reward"]
        for summary in result_summaries
        if summary["objective_reward"] is not None
    ]
    final_wsos = [summary["final_wso"] for summary in result_summaries if summary["final_wso"] is not None]
    invalid_rates = [summary["invalid_action_rate"] for summary in result_summaries]
    total_turns_list = [summary["total_steps"] for summary in result_summaries]

    primary_values = [summary[source_key] for summary in result_summaries]
    # Group by crop when present, otherwise fall back to env_id for non-WOFOST envs.
    per_group: Dict[str, list] = {}
    per_group_final_wsos: Dict[str, list] = {}
    per_simulator: Dict[str, list] = {}
    per_simulator_invalid_rates: Dict[str, list] = {}
    for summary in result_summaries:
        group = summary["group"]
        per_group.setdefault(group, []).append(summary[source_key])
        if summary["final_wso"] is not None:
            per_group_final_wsos.setdefault(group, []).append(summary["final_wso"])
        simulator = summary["simulator"]
        per_simulator.setdefault(simulator, []).append(summary[source_key])
        per_simulator_invalid_rates.setdefault(simulator, []).append(summary["invalid_action_rate"])

    metrics = {
        "primary_metric": {
            "key": primary_key,
            "label": primary_label,
            "mean": float(np.mean(primary_values)),
            "std": float(np.std(primary_values)),
        },
        "episode_reward_mean": float(np.mean(episode_rewards)),
        "episode_reward_std": float(np.std(episode_rewards)),
        "invalid_action_rate_mean": float(np.mean(invalid_rates)),
        "avg_turns": float(np.mean(total_turns_list)),
        "num_envs": len(results),
    }

    if final_wsos:
        metrics["final_wso_mean"] = float(np.mean(final_wsos))
        metrics["final_wso_std"] = float(np.std(final_wsos))
    if objective_rewards:
        metrics["objective_reward_mean"] = float(np.mean(objective_rewards))
        metrics["objective_reward_std"] = float(np.std(objective_rewards))

    group_summary = {}
    for group, values in sorted(per_group.items()):
        group_metrics = {
            "primary_metric_mean": float(np.mean(values)),
            "primary_metric_std": float(np.std(values)),
            "count": len(values),
        }
        group_final_wsos = per_group_final_wsos.get(group, [])
        if group_final_wsos:
            group_metrics["final_wso_mean"] = float(np.mean(group_final_wsos))
            group_metrics["final_wso_std"] = float(np.std(group_final_wsos))
        group_summary[group] = group_metrics
    metrics["per_group"] = group_summary
    metrics["per_crop"] = group_summary

    simulator_summary = {}
    for simulator, values in sorted(per_simulator.items()):
        simulator_summary[simulator] = {
            "primary_metric_mean": float(np.mean(values)),
            "primary_metric_std": float(np.std(values)),
            "invalid_action_rate_mean": float(np.mean(per_simulator_invalid_rates[simulator])),
            "count": len(values),
        }
    metrics["per_simulator"] = simulator_summary

    return metrics


def _terminal_trajectory_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    for turn in result.get("turns", []):
        traj = turn.get("trajectory_metrics")
        if isinstance(traj, dict) and traj:
            return traj
    return {}


def _result_cumulative_reward(result: Dict[str, Any]) -> float:
    rewards = [
        float(turn["reward"])
        for turn in result.get("turns", [])
        if turn.get("reward") is not None
    ]
    return float(sum(rewards))


def _extract_validation_env_infos(results: List[Dict[str, Any]]) -> Dict[str, list[Any]]:
    """Build trainer-style env_infos from standalone rollout results.

    Training validation aggregates environment metrics from reward-extra fields.
    Standalone evaluation does not use that reward path, so this reconstructs
    the same rectangular metric/label dictionary from terminal trajectory
    metrics and dataset env_config labels.
    """
    metric_rows: list[dict[str, float]] = []
    label_rows: list[dict[str, str]] = []
    metric_keys: set[str] = set()
    label_keys: set[str] = set()

    for result in results:
        env_config = result.get("env_config", {}) or {}
        terminal_metrics = _terminal_trajectory_metrics(result)

        metrics: dict[str, float] = {"reward": _result_cumulative_reward(result)}
        for key, value in terminal_metrics.items():
            if isinstance(value, bool):
                continue
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        labels = {
            str(key): str(value)
            for key, value in (env_config.get("trajectory_group_labels") or {}).items()
            if str(key or "").strip() and str(value or "").strip()
        }
        scenario_id = str(env_config.get("scenario_id", "") or "").strip()
        if scenario_id:
            labels.setdefault("scenario_id", scenario_id)
        env_name = str(env_config.get("env_name", "") or "").strip()
        if env_name:
            labels.setdefault("simulator", env_name)
        objective_id = str(env_config.get("objective_id", "") or "").strip()
        if objective_id:
            labels.setdefault("objective_id", objective_id)
        validation_set = str(env_config.get("validation_set", "") or "").strip()
        if validation_set:
            labels.setdefault("validation_set", validation_set)
        crop_name = str(env_config.get("crop_name", "") or "").strip()
        if crop_name:
            labels.setdefault("crop", crop_name)

        metric_rows.append(metrics)
        label_rows.append(labels)
        metric_keys.update(metrics)
        label_keys.update(labels)

    env_infos: dict[str, list[Any]] = {}
    for key in sorted(metric_keys):
        env_infos[key] = [row.get(key, float("nan")) for row in metric_rows]
    for key in sorted(label_keys):
        env_infos[f"group_label/{key}"] = [row.get(key, "") for row in label_rows]
    return env_infos


def _extract_validation_metric_dict(
    results: List[Dict[str, Any]],
    validation_axis: str | None = None,
) -> Dict[str, float]:
    """Return validation metrics with the same val-env prefixes as training."""
    env_infos = _extract_validation_env_infos(results)
    metric_dict: dict[str, float] = {}

    axis = str(validation_axis or "").strip()
    if axis and add_axis_env_metrics(metric_dict, env_infos, axis=axis, prefix_base="val-env"):
        return metric_dict

    validation_sets = [
        str(value or "").strip()
        for value in env_infos.get("group_label/validation_set", [])
    ]
    if validation_sets and any(validation_sets):
        add_env_metrics(metric_dict, env_infos, prefix="val-env/all", include_grouped=False)
        for validation_set in sorted({value for value in validation_sets if value}):
            indices = [idx for idx, value in enumerate(validation_sets) if value == validation_set]
            subset_infos = {
                key: [values[idx] for idx in indices]
                for key, values in env_infos.items()
                if len(values) > max(indices, default=-1)
            }
            add_env_metrics(
                metric_dict,
                subset_infos,
                prefix=f"val-env/{validation_set}",
                include_grouped=False,
            )
        return metric_dict

    add_env_metrics(metric_dict, env_infos, prefix="val-env", include_grouped=True)
    return metric_dict


def _build_summary(all_trial_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build cross-trial summary with mean/std over trials."""
    import numpy as np

    n = len(all_trial_metrics)

    primary = all_trial_metrics[0]["primary_metric"]
    primary_means = [m["primary_metric"]["mean"] for m in all_trial_metrics]
    inv_rates = [m["invalid_action_rate_mean"] for m in all_trial_metrics]

    summary = {
        "num_trials": n,
        "primary_metric": {
            "key": primary["key"],
            "label": primary["label"],
            "trial_means": primary_means,
            "mean": float(np.mean(primary_means)),
            "std": float(np.std(primary_means)),
        },
        "invalid_action_rate": {
            "trial_means": inv_rates,
            "mean": float(np.mean(inv_rates)),
            "std": float(np.std(inv_rates)),
        },
    }

    if all("episode_reward_mean" in m for m in all_trial_metrics):
        reward_means = [m["episode_reward_mean"] for m in all_trial_metrics]
        summary["episode_reward"] = {
            "trial_means": reward_means,
            "mean": float(np.mean(reward_means)),
            "std": float(np.std(reward_means)),
        }
    if all("final_wso_mean" in m for m in all_trial_metrics):
        wso_means = [m["final_wso_mean"] for m in all_trial_metrics]
        summary["final_wso"] = {
            "trial_means": wso_means,
            "mean": float(np.mean(wso_means)),
            "std": float(np.std(wso_means)),
        }

    all_groups = set()
    all_simulators = set()
    for m in all_trial_metrics:
        all_groups.update(m.get("per_group", {}).keys())
        all_simulators.update(m.get("per_simulator", {}).keys())

    per_group_summary = {}
    for group in sorted(all_groups):
        group_means = [
            m["per_group"][group]["primary_metric_mean"]
            for m in all_trial_metrics
            if group in m.get("per_group", {})
        ]
        group_summary = {
            "trial_means": group_means,
            "mean": float(np.mean(group_means)),
            "std": float(np.std(group_means)),
        }
        group_wso_means = [
            m["per_group"][group]["final_wso_mean"]
            for m in all_trial_metrics
            if group in m.get("per_group", {})
            and "final_wso_mean" in m["per_group"][group]
        ]
        if group_wso_means:
            group_summary["final_wso_trial_means"] = group_wso_means
            group_summary["final_wso_mean"] = float(np.mean(group_wso_means))
            group_summary["final_wso_std"] = float(np.std(group_wso_means))
        per_group_summary[group] = group_summary
    summary["per_group"] = per_group_summary
    summary["per_crop"] = per_group_summary

    per_simulator_summary = {}
    for simulator in sorted(all_simulators):
        sim_means = [
            m["per_simulator"][simulator]["primary_metric_mean"]
            for m in all_trial_metrics
            if simulator in m.get("per_simulator", {})
        ]
        per_simulator_summary[simulator] = {
            "trial_means": sim_means,
            "mean": float(np.mean(sim_means)),
            "std": float(np.std(sim_means)),
        }
    summary["per_simulator"] = per_simulator_summary

    return summary


def run_inference_rollout(
    model_config_path: str,
    output_dir: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_retries: int = 2,
    model_path: str = None,
    num_trials: int = 1,
    validation_axis: str | None = None,
    *,
    test_files: str | list[str] | None = None,
    env_configs: List[Dict[str, Any]] | None = None,
    env_name: str = "wofost_gym",
):
    """Run batch inference rollout on dataset.

    Args:
        model_config_path: Path to model configuration YAML file
        output_dir: Directory to save results
        temperature: Sampling temperature for LLM
        max_tokens: Maximum tokens to generate
        max_retries: Maximum retries per turn on invalid output
        model_path: Optional model path override
        num_trials: Number of repeated evaluation trials (default 1)
        validation_axis: Optional training validation axis for val-env metrics
    """
    print("=" * 80)
    print("Inference Rollout Configuration")
    print("=" * 80)
    if test_files:
        print(f"Test files: {test_files}")
    else:
        print("Dataset source: in-memory dataset artifact")
    print(f"Model config: {model_config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Max retries per turn: {max_retries}")
    print(f"Num trials: {num_trials}")
    if validation_axis:
        print(f"Validation axis: {validation_axis}")
    print("=" * 80)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    if env_configs is None:
        if not test_files:
            raise ValueError("run_inference_rollout requires test_files or env_configs.")
        env_configs, env_name = load_dataset(test_files)
    num_envs = len(env_configs)

    # Create environments
    print(f"\n[2/4] Creating {num_envs} environments...")
    envs = []
    configs = []
    system_prompts = []
    for env_config in tqdm(env_configs, desc="Creating envs"):
        row_env_name = env_config.get("env_name", env_name)
        if row_env_name == "__mixed__":
            raise ValueError("Mixed datasets require every env_config to include env_name.")
        env, config = create_environment(row_env_name, env_config)
        envs.append(env)
        configs.append(config)
        system_prompts.append(env.system_prompt())

    turn_num = max(int(config.turn_num) for config in configs)
    print(f"Episodes will run for {turn_num} turns")

    # Load model (once)
    overrides = {"model_name": model_path} if model_path else None
    print(f"\n[3/4] Loading model from {model_path or model_config_path}...")
    model = create_model(model_config_path, overrides=overrides)

    # Run trials
    print(f"\n[4/4] Running {'rollout' if num_trials == 1 else f'{num_trials} trials'}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_trial_metrics = []

    for trial in range(num_trials):
        if num_trials > 1:
            print(f"\n{'─' * 60}")
            print(f"Trial {trial + 1}/{num_trials}")
            print(f"{'─' * 60}")

        results = _run_single_trial(
            envs=envs,
            configs=configs,
            env_configs=env_configs,
            system_prompts=system_prompts,
            model=model,
            turn_num=turn_num,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

        # Save trial results
        if num_trials == 1:
            trial_dir = output_path
        else:
            trial_dir = output_path / f"trial_{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)

        _dump_json_atomic(trial_dir / "results.json", results, "trial results")

        trial_metrics = _extract_trial_metrics(results)
        all_trial_metrics.append(trial_metrics)

        _dump_json_atomic(trial_dir / "metrics.json", trial_metrics, "trial metrics")

        validation_metrics = _extract_validation_metric_dict(
            results,
            validation_axis=validation_axis,
        )
        _dump_json_atomic(
            trial_dir / "validation_metrics.json",
            validation_metrics,
            "training-style validation metrics",
        )

        # Print trial summary
        primary_metric = trial_metrics["primary_metric"]
        print(
            f"  {primary_metric['label']} mean: {primary_metric['mean']:.2f} "
            f"± {primary_metric['std']:.2f}"
        )
        print(f"  Invalid rate: {trial_metrics['invalid_action_rate_mean']:.4f}")

    # Write cross-trial summary
    if num_trials > 1:
        summary = _build_summary(all_trial_metrics)
        _dump_json_atomic(output_path / "summary.json", summary, "cross-trial summary")

        print(f"\n{'=' * 80}")
        print(f"Cross-trial Summary ({num_trials} trials)")
        print(f"{'=' * 80}")
        primary_metric = summary["primary_metric"]
        print(
            f"{primary_metric['label']}:  "
            f"{primary_metric['mean']:.2f} ± {primary_metric['std']:.2f}"
        )
        print(f"Invalid rate: {summary['invalid_action_rate']['mean']:.4f} ± {summary['invalid_action_rate']['std']:.4f}")
        for group, cs in summary['per_group'].items():
            print(f"  {group}: {cs['mean']:.2f} ± {cs['std']:.2f}")
        for simulator, cs in summary.get("per_simulator", {}).items():
            print(f"  {simulator}: {cs['mean']:.2f} ± {cs['std']:.2f}")
        print(f"{'=' * 80}")
    else:
        # Single trial — also write metrics.json at top level (already done above)
        print(f"\n{'=' * 80}")
        print("Rollout Summary")
        print(f"{'=' * 80}")
        m = all_trial_metrics[0]
        print(f"Total environments: {m['num_envs']}")
        primary_metric = m["primary_metric"]
        print(
            f"{primary_metric['label']} mean: "
            f"{primary_metric['mean']:.2f} ± {primary_metric['std']:.2f}"
        )
        print(f"Invalid rate: {m['invalid_action_rate_mean']:.4f}")
        for group, cs in m.get("per_group", {}).items():
            print(
                f"  {group}: {cs['primary_metric_mean']:.2f} "
                f"± {cs['primary_metric_std']:.2f} (n={cs['count']})"
            )
        for simulator, cs in m.get("per_simulator", {}).items():
            print(
                f"  {simulator}: {cs['primary_metric_mean']:.2f} "
                f"± {cs['primary_metric_std']:.2f} (n={cs['count']})"
            )
        print(f"{'=' * 80}")

    # Close all environments
    for env in envs:
        env.close()

    print(f"\nResults saved to: {output_path}")


@hydra.main(
    config_path="../../../entrypoints/eval/config",
    config_name="default",
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
    model_cfg = cfg.get("model") or {}

    inference_file = data_cfg.get("inference_file") or cfg.get("test_files")
    if not inference_file:
        raise ValueError("Config must specify data.inference_file.")

    model_config_path = model_cfg.get("config")
    if not model_config_path:
        raise ValueError("Config must specify model.config.")

    run_inference_rollout(
        test_files=inference_file,
        model_config_path=str(model_config_path),
        output_dir=str(output_cfg.get("dir") or cfg.get("output_dir", "./rollout_results")),
        temperature=float(runtime_cfg.get("temperature", cfg.get("temperature", 0.7))),
        max_tokens=int(runtime_cfg.get("max_tokens", cfg.get("max_tokens", 512))),
        max_retries=int(runtime_cfg.get("max_retries", cfg.get("max_retries", 5))),
        model_path=model_cfg.get("path"),
        num_trials=int(runtime_cfg.get("num_trials", cfg.get("num_trials", 1))),
        validation_axis=data_cfg.get("validation_axis"),
    )


if __name__ == "__main__":
    main()
