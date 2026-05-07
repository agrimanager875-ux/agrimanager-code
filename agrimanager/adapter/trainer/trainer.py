# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
AgriTrainer — standalone copy of RayPPOTrainer with env metrics logging.

Copied from verl/verl/trainer/ppo/ray_trainer.py and modified to:
- Replace ValidationGenerationsLogger with AgriGenerationsLogger
- Log env trajectory metrics (val-env/ and train-env/ prefixes) to the tracker
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from agrimanager.adapter.trainer.rollout_filter import (
    RewardVarianceFilterConfig,
    filter_batch_by_reward_variance,
)
from agrimanager.adapter.trainer.stepwise_ppo import (
    build_prompt_value_batch,
    compute_stepwise_gae_advantage_cross_turn,
    compute_stepwise_gae_advantage_cross_turn_from_scalar_values,
    extract_first_valid_token_scalar,
    extract_last_valid_token_scalar,
)
from agrimanager.adapter.trainer.validation_metrics import (
    add_axis_env_metrics,
    add_env_metrics,
    add_training_env_metrics,
    add_validation_env_metrics,
    filter_finite_validation_infos,
)
from agrimanager.adapter.trainer.validation_sets import create_named_validation_dataset
from agrimanager.adapter.utils import AgriGenerationsLogger

# ---------------------------------------------------------------------------
# Trajectory formatting helpers (used by _log_agri_generations)
# ---------------------------------------------------------------------------
_ROLE_RE = re.compile(r"(system|user|assistant)\n")


class _SkipFullyFilteredStep(Exception):
    """Internal control-flow exception for rollout-filtered empty batches."""


def _as_list(values: Any) -> list[Any]:
    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _slice_info_dict(infos: dict[str, Any], indices: list[int]) -> dict[str, list[Any]]:
    sliced: dict[str, list[Any]] = {}
    for key, values in infos.items():
        values_list = _as_list(values)
        if len(values_list) < max(indices, default=-1) + 1:
            continue
        sliced[key] = [values_list[i] for i in indices]
    return sliced


def _group_label_values(env_infos: dict[str, Any], group_name: str, length: int) -> list[str]:
    group_name = str(group_name or "").strip()
    if not group_name:
        return [""] * length
    values = env_infos.get(f"group_label/{group_name}")
    if values is None and group_name == "validation_set":
        values = env_infos.get("validation_set")
    if values is None:
        return [""] * length
    values_list = _as_list(values)
    if len(values_list) != length:
        return [""] * length
    return [str(value or "").strip() for value in values_list]


def _add_processed_validation_metrics(
    metric_dict: dict[str, float],
    *,
    scope: str,
    sample_uids: list[Any],
    reward_infos: dict[str, Any],
    core_prefix: str = "val-core",
    aux_prefix: str = "val-aux",
) -> None:
    if not sample_uids:
        return
    scoped_data_sources = np.array([scope] * len(sample_uids), dtype=object)
    data_src2var2metric2val = process_validation_metrics(
        scoped_data_sources,
        sample_uids,
        filter_finite_validation_infos(reward_infos),
    )
    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            if not metric2val:
                continue
            n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
            for metric_name, metric_val in metric2val.items():
                if (
                    (var_name == core_var)
                    and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                    and (f"@{n_max}" in metric_name)
                ):
                    metric_sec = core_prefix
                else:
                    metric_sec = aux_prefix
                metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = metric_val


def _add_axis_validation_metrics(
    metric_dict: dict[str, float],
    *,
    sample_uids: list[Any],
    env_infos: dict[str, Any],
    validation_axis: str,
) -> bool:
    axis_name = str(validation_axis or "").strip()
    if not axis_name:
        return False
    axis_values = _group_label_values(env_infos, axis_name, len(sample_uids))
    named_values = sorted({value for value in axis_values if value})
    core_prefix = f"val-core-{axis_name}"
    aux_prefix = f"val-aux-{axis_name}"

    _add_processed_validation_metrics(
        metric_dict,
        scope="all",
        sample_uids=[
            f"{axis_value}::{uid}" if axis_value else uid
            for uid, axis_value in zip(sample_uids, axis_values)
        ],
        reward_infos=env_infos,
        core_prefix=core_prefix,
        aux_prefix=aux_prefix,
    )
    add_axis_env_metrics(metric_dict, env_infos, axis=axis_name, prefix_base="val-env")

    for axis_value in named_values:
        indices = [idx for idx, value in enumerate(axis_values) if value == axis_value]
        subset_infos = _slice_info_dict(env_infos, indices)
        subset_uids = [sample_uids[idx] for idx in indices]
        _add_processed_validation_metrics(
            metric_dict,
            scope=axis_value,
            sample_uids=subset_uids,
            reward_infos=subset_infos,
            core_prefix=core_prefix,
            aux_prefix=aux_prefix,
        )
    return True


def _add_named_validation_metrics(
    metric_dict: dict[str, float],
    *,
    sample_uids: list[Any],
    env_infos: dict[str, Any],
) -> bool:
    validation_sets = _group_label_values(env_infos, "validation_set", len(sample_uids))
    named_values = sorted({value for value in validation_sets if value})
    if not named_values:
        return False

    _add_processed_validation_metrics(
        metric_dict,
        scope="all",
        sample_uids=[
            f"{validation_set}::{uid}" if validation_set else uid
            for uid, validation_set in zip(sample_uids, validation_sets)
        ],
        reward_infos=env_infos,
    )
    add_env_metrics(metric_dict, env_infos, prefix="val-env/all", include_grouped=False)

    for validation_set in named_values:
        indices = [idx for idx, value in enumerate(validation_sets) if value == validation_set]
        subset_infos = _slice_info_dict(env_infos, indices)
        subset_uids = [sample_uids[idx] for idx in indices]
        _add_processed_validation_metrics(
            metric_dict,
            scope=validation_set,
            sample_uids=subset_uids,
            reward_infos=subset_infos,
        )
        add_env_metrics(
            metric_dict,
            subset_infos,
            prefix=f"val-env/{validation_set}",
            include_grouped=False,
        )
    return True


def _format_trajectory(text: str) -> str:
    """Format a multi-turn conversation by highlighting role markers.

    Converts raw decoded text like:
        system\nYou are...\nuser\nWe are growing...\nassistant\nApply...
    Into:
        ══ SYSTEM ══
        You are...

        ══ TURN 1 · USER ══
        We are growing...

        ══ TURN 1 · ASSISTANT ══
        Apply...

        ══ TURN 2 · USER ══
        ...
    """
    parts = _ROLE_RE.split(text)
    # parts = [preamble, role1, content1, role2, content2, ...]
    # If no role markers found, return as-is
    if len(parts) < 3:
        return text.strip()

    lines = []
    turn = 0
    # Skip any preamble before the first role marker
    for i in range(1, len(parts), 2):
        role = parts[i].upper()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if role == "SYSTEM":
            lines.append(f"══ {role} ══\n{content}")
        else:
            if role == "USER":
                turn += 1
            lines.append(f"══ TURN {turn} · {role} ══\n{content}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Free functions (unchanged from ray_trainer.py)
# ---------------------------------------------------------------------------


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_grpo_trajectory_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    uid: np.ndarray,
    trajectory_id: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GRPO advantage at the trajectory level for per-turn training.

    In per-turn mode, all turns of a trajectory share the same uid and get
    the same reward.  Standard GRPO would group all turns by uid, see
    identical rewards within each trajectory, and compute std=0 → advantage=0.

    This function instead:
    1. Groups samples by uid (same prompt, different rollouts).
    2. De-duplicates by trajectory_id to get one score per trajectory.
    3. Computes GRPO advantage across trajectories within each uid group.
    4. Broadcasts the trajectory-level advantage back to all turns.
    """
    scores = (token_level_rewards * response_mask).sum(dim=-1)

    # Map trajectory_id → list of sample indices
    traj_indices: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(trajectory_id):
        traj_indices[tid].append(i)

    # One score per trajectory (all turns have the same reward, take the first)
    traj_score = {tid: scores[idxs[0]].item() for tid, idxs in traj_indices.items()}
    traj_uid = {tid: uid[idxs[0]] for tid, idxs in traj_indices.items()}

    # Group trajectories by uid
    uid_trajs: dict[str, list[str]] = defaultdict(list)
    for tid, u in traj_uid.items():
        uid_trajs[u].append(tid)

    # Compute per-trajectory advantage within each uid group
    traj_adv: dict[str, float] = {}
    with torch.no_grad():
        for u, tids in uid_trajs.items():
            rewards = torch.tensor([traj_score[tid] for tid in tids])
            if len(tids) == 1:
                mean_r = torch.tensor(0.0)
                std_r = torch.tensor(1.0)
            else:
                mean_r = rewards.mean()
                std_r = rewards.std()
            for tid, r in zip(tids, rewards):
                if norm_adv_by_std_in_grpo:
                    traj_adv[tid] = ((r - mean_r) / (std_r + epsilon)).item()
                else:
                    traj_adv[tid] = (r - mean_r).item()

    # Broadcast trajectory-level advantage to all turns
    advantages = torch.zeros_like(token_level_rewards)
    for tid, idxs in traj_indices.items():
        for idx in idxs:
            advantages[idx] = traj_adv[tid] * response_mask[idx]

    return advantages, advantages.clone()


def _stepwise_cfg_get(stepwise_config: Optional[dict[str, Any]], key: str, default: Any = None) -> Any:
    if stepwise_config is None:
        return default
    if isinstance(stepwise_config, dict):
        return stepwise_config.get(key, default)
    if hasattr(stepwise_config, "get"):
        try:
            return stepwise_config.get(key, default)
        except Exception:
            pass
    try:
        return getattr(stepwise_config, key)
    except Exception:
        return default


def _rollout_filter_cfg_get(rollout_filter_config: Optional[dict[str, Any]], key: str, default: Any = None) -> Any:
    if rollout_filter_config is None:
        return default
    if isinstance(rollout_filter_config, dict):
        return rollout_filter_config.get(key, default)
    if hasattr(rollout_filter_config, "get"):
        try:
            return rollout_filter_config.get(key, default)
        except Exception:
            pass
    try:
        return getattr(rollout_filter_config, key)
    except Exception:
        return default


def _resolve_stepwise_gae_params(stepwise_config: Optional[dict[str, Any]]) -> tuple[float, float]:
    """Resolve stepwise GAE hyperparameters from stepwise config only."""
    gamma = float(_stepwise_cfg_get(stepwise_config, "gamma", 1.0))
    lam = float(_stepwise_cfg_get(stepwise_config, "lam", _stepwise_cfg_get(stepwise_config, "lambda", 0.97)))
    return gamma, lam


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    stepwise_config: Optional[dict[str, Any]] = None,
) -> DataProto:
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    stepwise_enabled = bool(_stepwise_cfg_get(stepwise_config, "enable", False))

    if stepwise_enabled:
        mode = str(_stepwise_cfg_get(stepwise_config, "mode", "per_step"))
        adv_name = str(_stepwise_cfg_get(stepwise_config, "adv_estimator", "gae")).lower()
        whiten_advantages = bool(_stepwise_cfg_get(stepwise_config, "whiten_advantages", True))
        stepwise_gamma, stepwise_lam = _resolve_stepwise_gae_params(stepwise_config)
        if mode != "per_step":
            raise NotImplementedError(f"Only stepwise mode='per_step' is supported for now, got: {mode}")
        if adv_name != "gae":
            raise NotImplementedError(
                f"Only stepwise adv_estimator='gae' is supported for now, got: {adv_name}"
            )
        if adv_estimator != AdvantageEstimator.GAE:
            raise ValueError(
                "stepwise_advantage.enable=True requires algorithm.adv_estimator=gae "
                f"(got {adv_estimator})"
            )
        if "trajectory_id" not in data.non_tensor_batch:
            raise ValueError("Missing non_tensor_batch['trajectory_id'] for stepwise cross-turn GAE")
        if "step_idx" not in data.non_tensor_batch:
            raise ValueError("Missing non_tensor_batch['step_idx'] for stepwise cross-turn GAE")
        if "state_values" in data.batch:
            step_rewards = extract_last_valid_token_scalar(
                data.batch["token_level_rewards"], data.batch["response_mask"]
            )
            advantages, returns, _, state_returns = compute_stepwise_gae_advantage_cross_turn_from_scalar_values(
                step_rewards=step_rewards,
                state_values=data.batch["state_values"],
                response_mask=data.batch["response_mask"],
                trajectory_id=data.non_tensor_batch["trajectory_id"],
                step_idx=data.non_tensor_batch["step_idx"],
                gamma=stepwise_gamma,
                lam=stepwise_lam,
                whiten_advantages=whiten_advantages,
            )
            data.batch["state_returns"] = state_returns
        else:
            if "values" not in data.batch:
                raise ValueError("Missing batch['values'] or batch['state_values'] for stepwise cross-turn GAE")
            advantages, returns = compute_stepwise_gae_advantage_cross_turn(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                trajectory_id=data.non_tensor_batch["trajectory_id"],
                step_idx=data.non_tensor_batch["step_idx"],
                gamma=stepwise_gamma,
                lam=stepwise_lam,
                whiten_advantages=whiten_advantages,
            )
            data.batch["state_returns"] = extract_first_valid_token_scalar(returns, data.batch["response_mask"])
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        if "trajectory_id" in data.non_tensor_batch:
            # Per-turn mode: compute advantage at trajectory level
            advantages, returns = compute_grpo_trajectory_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                uid=data.non_tensor_batch["uid"],
                trajectory_id=data.non_tensor_batch["trajectory_id"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


# ---------------------------------------------------------------------------
# AgriTrainer
# ---------------------------------------------------------------------------

class AgriTrainer:
    """Standalone PPO trainer with AgriManager-specific env metrics logging.

    Based on verl's RayPPOTrainer with the following modifications:
    - Uses AgriGenerationsLogger instead of ValidationGenerationsLogger
    - Logs val-env/ trajectory metrics in _validate()
    - Logs train-env/ trajectory metrics in fit()
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # Per-turn training mode
        self.per_turn_training = config.trainer.get("per_turn_training", False)
        self.stepwise_advantage_cfg = config.trainer.get("stepwise_advantage", {})
        self.stepwise_advantage_enable = bool(_stepwise_cfg_get(self.stepwise_advantage_cfg, "enable", False))
        self.stepwise_critic_value_mode = str(
            _stepwise_cfg_get(self.stepwise_advantage_cfg, "critic_value_mode", "token_aligned")
        ).lower()
        self.stepwise_prompt_value_wrapper_enable = False

        if self.stepwise_advantage_enable:
            mode = str(_stepwise_cfg_get(self.stepwise_advantage_cfg, "mode", "per_step"))
            adv_name = str(_stepwise_cfg_get(self.stepwise_advantage_cfg, "adv_estimator", "gae")).lower()
            reward_source = str(_stepwise_cfg_get(self.stepwise_advantage_cfg, "reward_source", "env_step"))
            token_reward_shape = str(
                _stepwise_cfg_get(self.stepwise_advantage_cfg, "token_reward_shape", "last_token_sparse")
            )
            gae_scope = str(_stepwise_cfg_get(self.stepwise_advantage_cfg, "gae_scope", "cross_turn"))
            if not self.per_turn_training:
                raise ValueError("trainer.stepwise_advantage.enable=True requires trainer.per_turn_training=True")
            if mode != "per_step":
                raise NotImplementedError(f"Only stepwise mode='per_step' is supported, got: {mode}")
            if adv_name != "gae":
                raise NotImplementedError(
                    f"Only stepwise adv_estimator='gae' is supported in first version, got: {adv_name}"
                )
            if reward_source != "env_step":
                raise NotImplementedError(
                    f"Only stepwise reward_source='env_step' is supported in first version, got: {reward_source}"
                )
            if token_reward_shape != "last_token_sparse":
                raise NotImplementedError(
                    "Only stepwise token_reward_shape='last_token_sparse' is supported in first version, "
                    f"got: {token_reward_shape}"
                )
            if gae_scope != "cross_turn":
                raise NotImplementedError(
                    f"Only stepwise gae_scope='cross_turn' is supported in first version, got: {gae_scope}"
                )
            if self.stepwise_critic_value_mode not in {"token_aligned", "prompt_only_wrapper"}:
                raise NotImplementedError(
                    "Only stepwise critic_value_mode in {'token_aligned', 'prompt_only_wrapper'} "
                    f"is supported, got: {self.stepwise_critic_value_mode}"
                )
            if self.config.algorithm.adv_estimator != AdvantageEstimator.GAE:
                raise ValueError(
                    "trainer.stepwise_advantage.enable=True requires algorithm.adv_estimator=gae "
                    f"(got: {self.config.algorithm.adv_estimator})"
                )
            if not self.use_critic:
                raise ValueError("trainer.stepwise_advantage.enable=True requires critic (GAE path)")
            self.stepwise_prompt_value_wrapper_enable = self.stepwise_critic_value_mode == "prompt_only_wrapper"

        self.rollout_filter_cfg = config.trainer.get("rollout_filter", {})
        self.rollout_filter_enable = bool(_rollout_filter_cfg_get(self.rollout_filter_cfg, "enable", False))
        self.rollout_filter_max_consecutive_all_filtered_steps = int(
            _rollout_filter_cfg_get(self.rollout_filter_cfg, "max_consecutive_all_filtered_steps", 10)
        )
        self.rollout_filter_runtime_cfg = RewardVarianceFilterConfig(
            value=float(_rollout_filter_cfg_get(self.rollout_filter_cfg, "value", 0.9)),
            include_zero=bool(_rollout_filter_cfg_get(self.rollout_filter_cfg, "include_zero", True)),
            score_key=str(_rollout_filter_cfg_get(self.rollout_filter_cfg, "score_key", "traj_score")),
            zero_eps=float(_rollout_filter_cfg_get(self.rollout_filter_cfg, "zero_eps", 1e-10)),
            top_p_prob_mode=str(_rollout_filter_cfg_get(self.rollout_filter_cfg, "top_p_prob_mode", "softmax")),
            selection_eps=float(_rollout_filter_cfg_get(self.rollout_filter_cfg, "selection_eps", 0.01)),
        )

        if self.rollout_filter_enable:
            metric = str(_rollout_filter_cfg_get(self.rollout_filter_cfg, "metric", "reward_variance")).lower()
            strategy = str(_rollout_filter_cfg_get(self.rollout_filter_cfg, "strategy", "top_p")).lower()
            filter_type = str(_rollout_filter_cfg_get(self.rollout_filter_cfg, "filter_type", "largest")).lower()
            value = float(_rollout_filter_cfg_get(self.rollout_filter_cfg, "value", 0.9))

            if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                pass
            elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
                if not self.stepwise_advantage_enable:
                    raise ValueError(
                        "trainer.rollout_filter.enable=True with algorithm.adv_estimator=gae "
                        "requires trainer.stepwise_advantage.enable=True"
                    )
            else:
                raise ValueError(
                    "trainer.rollout_filter.enable=True requires either "
                    "algorithm.adv_estimator=grpo or "
                    "(algorithm.adv_estimator=gae and trainer.stepwise_advantage.enable=True) "
                    f"(got: {self.config.algorithm.adv_estimator})"
                )
            if metric != "reward_variance":
                raise NotImplementedError(
                    "First version only supports trainer.rollout_filter.metric=reward_variance "
                    f"(got: {metric})"
                )
            if strategy != "top_p":
                raise NotImplementedError(
                    "First version only supports trainer.rollout_filter.strategy=top_p "
                    f"(got: {strategy})"
                )
            if filter_type != "largest":
                raise NotImplementedError(
                    "First version only supports trainer.rollout_filter.filter_type=largest "
                    f"(got: {filter_type})"
                )
            if not (0.0 < value <= 1.0):
                raise ValueError(
                    "trainer.rollout_filter.value must be in (0, 1] for top_p strategy "
                    f"(got: {value})"
                )
            if self.rollout_filter_max_consecutive_all_filtered_steps < 1:
                raise ValueError(
                    "trainer.rollout_filter.max_consecutive_all_filtered_steps must be >= 1 "
                    f"(got: {self.rollout_filter_max_consecutive_all_filtered_steps})"
                )
            if self.config.reward_model.launch_reward_fn_async:
                raise ValueError(
                    "trainer.rollout_filter.enable=True requires reward_model.launch_reward_fn_async=False "
                    "to keep reward-filter ordering deterministic"
                )

        # AgriManager: use AgriGenerationsLogger instead of ValidationGenerationsLogger
        self.agri_generations_logger = AgriGenerationsLogger()

        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        test_freq = self.config.trainer.get("test_freq", 0) or 0
        validation_requested = self.val_reward_fn is not None and (
            bool(self.config.trainer.get("val_before_train", True))
            or bool(test_freq > 0)
            or bool(self.config.trainer.get("val_only", False))
        )

        if val_dataset is None and validation_requested:
            val_dataset = create_named_validation_dataset(
                self.config.data.get("val_sets", None),
                self.config.data,
                self.tokenizer,
                self.processor,
                create_rl_dataset,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
            if val_dataset is None:
                val_dataset = create_rl_dataset(
                    self.config.data.val_files,
                    self.config.data,
                    self.tokenizer,
                    self.processor,
                    max_samples=self.config.data.get("val_max_samples", -1),
                )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        self.val_dataloader = None
        if self.val_dataset is not None:
            val_batch_size = self.config.data.val_batch_size
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        if self.val_dataloader is not None:
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        val_dataloader_size = len(self.val_dataloader) if self.val_dataloader is not None else 0
        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, "
            f"Size of val dataloader: {val_dataloader_size}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                if sum_reward:
                    return reward_tensor
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        reward_extra_fill_values: dict[str, Any] = {}

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_trajectory_ids = []
        sample_per_turn_data = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )

            if self.per_turn_training:
                # Per-turn: worker expansion changes output size, so skip
                # pad/unpad.  AgriAgentLoopManager.generate_sequences handles
                # batch_size < num_workers by using fewer workers.
                if not self.async_rollout_mode:
                    test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
                else:
                    test_output_gen_batch = self.async_rollout_manager.generate_sequences(test_gen_batch)
            else:
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                        test_gen_batch_padded
                    )
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            if self.per_turn_training:
                # Per-turn: skip test_batch.union(), use gen output directly
                test_batch_for_reward = test_output_gen_batch
                test_batch_for_reward.meta_info["validate"] = True
            else:
                test_batch = test_batch.union(test_output_gen_batch)
                test_batch.meta_info["validate"] = True
                test_batch_for_reward = test_batch

            input_ids = test_batch_for_reward.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch_for_reward.non_tensor_batch["uid"])

            if "trajectory_id" in test_batch_for_reward.non_tensor_batch:
                sample_trajectory_ids.extend(test_batch_for_reward.non_tensor_batch["trajectory_id"])

            if "per_turn_data" in test_batch_for_reward.non_tensor_batch:
                # Per-turn: all turns of a trajectory carry the same per_turn_data.
                # De-duplicate by trajectory_id so each trajectory is logged once.
                seen_tids = set()
                for j in range(len(test_batch_for_reward)):
                    tid_j = test_batch_for_reward.non_tensor_batch["trajectory_id"][j]
                    if tid_j not in seen_tids:
                        seen_tids.add(tid_j)
                        sample_per_turn_data.append(test_batch_for_reward.non_tensor_batch["per_turn_data"][j])

            result = self._compute_or_extract_reward(
                test_batch_for_reward, reward_fn=self.val_reward_fn, return_dict=True
            )
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            previous_count = len(sample_scores) - len(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                values_list = values.tolist() if isinstance(values, np.ndarray) else (
                    values if isinstance(values, list) else [values]
                )
                if key not in reward_extra_fill_values:
                    first_value = values_list[0] if values_list else float("nan")
                    fill_value = "" if isinstance(first_value, str) else float("nan")
                    reward_extra_fill_values[key] = fill_value
                    reward_extra_infos_dict[key].extend([fill_value] * previous_count)
                reward_extra_infos_dict[key].extend(values_list)
            for key, fill_value in reward_extra_fill_values.items():
                if key not in reward_extra_info:
                    reward_extra_infos_dict[key].extend([fill_value] * len(scores))

            if "__num_turns__" in test_batch_for_reward.non_tensor_batch:
                sample_turns.append(test_batch_for_reward.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch_for_reward.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        # AgriManager per-episode trajectory logging
        if sample_trajectory_ids and sample_per_turn_data:
            # Per-turn: de-duplicate all logging data to trajectory level so that
            # each logged episode shows the full multi-turn trajectory, not a
            # single turn.  sample_per_turn_data is already de-duplicated.
            seen_tids_log: set[str] = set()
            traj_inputs_log: list[str] = []
            traj_outputs_log: list[str] = []
            traj_scores_log: list[float] = []
            traj_extra_log: dict[str, list] = defaultdict(list)
            for i, tid in enumerate(sample_trajectory_ids):
                if tid not in seen_tids_log:
                    seen_tids_log.add(tid)
                    traj_inputs_log.append(sample_inputs[i])
                    traj_outputs_log.append(sample_outputs[i])
                    traj_scores_log.append(sample_scores[i])
                    for k, v in reward_extra_infos_dict.items():
                        if i < len(v):
                            traj_extra_log[k].append(v[i])
            self._log_agri_generations(
                sample_inputs=traj_inputs_log,
                sample_outputs=traj_outputs_log,
                sample_scores=traj_scores_log,
                reward_extra_infos_dict=dict(traj_extra_log),
                per_turn_data=sample_per_turn_data,
            )
        else:
            self._log_agri_generations(
                sample_inputs=sample_inputs,
                sample_outputs=sample_outputs,
                sample_scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                per_turn_data=None,
            )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        if sample_trajectory_ids:
            # Per-turn mode: de-duplicate to one entry per trajectory so that
            # process_validation_metrics computes pass@N at the trajectory level
            # (N = number of rollouts) instead of the turn level (N = total turns).
            seen_tids = set()
            traj_indices = []
            for i, tid in enumerate(sample_trajectory_ids):
                if tid not in seen_tids:
                    seen_tids.add(tid)
                    traj_indices.append(i)
            traj_data_sources = data_sources[traj_indices]
            traj_uids = [sample_uids[i] for i in traj_indices]
            traj_infos = {k: [v[i] for i in traj_indices] for k, v in reward_extra_infos_dict.items()}
            metric_uids = traj_uids
            metric_infos = traj_infos
            metric_data_sources = traj_data_sources
        else:
            metric_uids = sample_uids
            metric_infos = reward_extra_infos_dict
            metric_data_sources = data_sources

        metric_dict = {}
        validation_axis = str(self.config.data.get("validation_axis", "") or "").strip()
        has_axis_validation = False
        if validation_axis:
            has_axis_validation = _add_axis_validation_metrics(
                metric_dict,
                sample_uids=metric_uids,
                env_infos=metric_infos,
                validation_axis=validation_axis,
            )
        has_named_validation = False
        if not has_axis_validation:
            has_named_validation = _add_named_validation_metrics(
                metric_dict,
                sample_uids=metric_uids,
                env_infos=metric_infos,
            )
        if not has_axis_validation and not has_named_validation:
            data_src2var2metric2val = process_validation_metrics(
                metric_data_sources,
                metric_uids,
                filter_finite_validation_infos(metric_infos),
            )
            for data_source, var2metric2val in data_src2var2metric2val.items():
                core_var = "acc" if "acc" in var2metric2val else "reward"
                for var_name, metric2val in var2metric2val.items():
                    if not metric2val:
                        continue
                    n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                    for metric_name, metric_val in metric2val.items():
                        if (
                            (var_name == core_var)
                            and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                            and (f"@{n_max}" in metric_name)
                        ):
                            metric_sec = "val-core"
                        else:
                            metric_sec = "val-aux"
                        pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                        metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Log val-env/ trajectory metrics (de-duplicated to trajectory level).
        # Optional grouped aggregations are driven by "group_label/<name>" fields
        # emitted by the environment/reward path, not by trainer-side env-specific logic.
        env_infos = metric_infos
        if not has_axis_validation:
            add_validation_env_metrics(metric_dict, env_infos)

        return metric_dict

    def _format_per_turn_trajectory(self, per_turn):
        """Build full trajectory text from per_turn_data for tracker table logging.

        Decodes every turn's prompt_ids/response_ids faithfully — each turn's
        full prompt (including repeated system message) is shown as-is, matching
        what the model actually sees during inference.
        """
        sections = []

        for t_idx, turn in enumerate(per_turn):
            prompt_text = self.tokenizer.decode(turn["prompt_ids"], skip_special_tokens=True)
            response_text = self.tokenizer.decode(turn["response_ids"], skip_special_tokens=True)
            reward = turn.get("reward", 0.0)

            # Parse prompt role sections directly (don't use _format_trajectory
            # because it resets turn numbering and includes trailing empty assistant)
            parts = _ROLE_RE.split(prompt_text)
            for j in range(1, len(parts), 2):
                role = parts[j]
                content = parts[j + 1].strip() if j + 1 < len(parts) else ""
                if role == "system":
                    sections.append(f"══ SYSTEM ══\n{content}")
                elif role == "user":
                    sections.append(f"══ TURN {t_idx + 1} · USER ══\n{content}")
                # skip trailing empty "assistant" marker from chat template

            sections.append(
                f"══ TURN {t_idx + 1} · ASSISTANT ══\n{response_text.strip()}\n[reward: {reward:.3f}]"
            )

        return "\n\n".join(sections)

    def _log_agri_generations(self, sample_inputs, sample_outputs, sample_scores,
                              reward_extra_infos_dict, per_turn_data=None):
        """Log per-episode trajectory data via AgriGenerationsLogger."""
        generations_to_log = self.config.trainer.get("log_val_generations", 0)
        if generations_to_log == 0:
            return

        n = len(sample_scores)
        logged_scores = reward_extra_infos_dict.get("score", sample_scores)
        num_steps_list = reward_extra_infos_dict.get("num_steps", [0] * n)
        validation_axis = str(self.config.data.get("validation_axis", "") or "").strip()

        def info_value(index: int, *keys: str) -> str:
            for key in keys:
                values = reward_extra_infos_dict.get(key)
                if values is None:
                    continue
                values_list = _as_list(values)
                if index < len(values_list):
                    value = values_list[index]
                    if value is not None:
                        text = str(value).strip()
                        if text:
                            return text
            return ""

        episodes = []
        for i in range(n):
            if per_turn_data and i < len(per_turn_data) and per_turn_data[i]:
                trajectory = self._format_per_turn_trajectory(per_turn_data[i])
                num_steps = len(per_turn_data[i])
            else:
                inp = sample_inputs[i] if i < len(sample_inputs) else ""
                out = sample_outputs[i] if i < len(sample_outputs) else ""
                trajectory = _format_trajectory(inp + out)
                num_steps = num_steps_list[i] if i < len(num_steps_list) else 0

            episodes.append({
                "validation_set": info_value(i, "group_label/validation_set", "validation_set"),
                "validation_axis": validation_axis,
                "validation_axis_value": (
                    info_value(i, f"group_label/{validation_axis}") if validation_axis else ""
                ),
                "weather_regime": info_value(i, "group_label/weather_regime", "weather_regime"),
                "crop_regime": info_value(i, "group_label/crop_regime", "crop_regime"),
                "scenario_id": info_value(i, "group_label/scenario_id", "scenario_id"),
                "crop_name": info_value(i, "crop_name", "group_label/crop"),
                "trajectory": trajectory,
                "reward": logged_scores[i] if i < len(logged_scores) else 0.0,
                "num_steps": num_steps,
            })

        if len(episodes) > generations_to_log:
            episodes.sort(key=lambda x: x["trajectory"])
            rng = np.random.RandomState(42)
            rng.shuffle(episodes)
            episodes = episodes[:generations_to_log]

        self.agri_generations_logger.log(self.config.trainer.logger, episodes, self.global_steps)

    # ------------------------------------------------------------------
    # Worker initialization
    # ------------------------------------------------------------------

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        if not self.use_reward_loop:
            if self.use_rm:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        self.async_rollout_mode = True

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            # Skip loading dataloader state at epoch boundaries — the saved state
            # has the sampler exhausted, so iterating would yield no batches.
            if self.global_steps % len(self.train_dataloader) == 0:
                print(
                    f"Checkpoint at epoch boundary (step {self.global_steps}), "
                    "skipping dataloader state restore (starting fresh epoch)"
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    def _start_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        if keep_minibatch:
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    # ------------------------------------------------------------------
    # Compute helpers
    # ------------------------------------------------------------------

    def _compute_token_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.stepwise_prompt_value_wrapper_enable:
            state_batch = build_prompt_value_batch(batch)
            values = self._compute_token_values(state_batch)
            values.rename(old_keys="values", new_keys="state_values")
        else:
            values = self._compute_token_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            log_probs = tu.get(output, "log_probs")
            log_probs = no_padding_2_padding(log_probs, batch_td)
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _pad_batch_for_training(self, batch: DataProto) -> DataProto:
        """Pad batch to be divisible by ppo_mini_batch_size.

        After per-turn expansion in the worker, the batch size may not be
        evenly divisible.  This pads by randomly duplicating samples.
        """
        mini_bs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        divisor = mini_bs
        total = len(batch)
        remainder = total % divisor
        if remainder == 0:
            return batch
        pad_count = divisor - remainder
        pad_indices = np.random.RandomState(42).choice(total, size=pad_count, replace=True)
        pad_batch = batch[pad_indices]
        print(
            f"[per-turn pad] Padded {pad_count} samples: {total} -> {total + pad_count} "
            f"(divisible by {divisor})"
        )
        return DataProto.concat([batch, pad_batch])

    def _build_training_batch_from_rollout(
        self,
        base_batch: DataProto,
        gen_batch_output: DataProto,
        metrics: dict[str, float],
    ) -> DataProto:
        """Build update batch from rollout output for per-turn and standard modes."""
        defer_padding = self.stepwise_advantage_enable or self.rollout_filter_enable

        if self.per_turn_training:
            # Per-turn: worker already returns expanded rows.
            batch = gen_batch_output
            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)
            if defer_padding:
                # Keep original order/rows before advantage when using stepwise or filtering.
                metrics["per_turn/raw_batch_size"] = batch.batch.batch_size[0]
            else:
                batch = self._pad_batch_for_training(batch)
                per_turn_mini_bs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                metrics["per_turn/batch_size"] = batch.batch.batch_size[0]
                metrics["per_turn/num_updates"] = (
                    max(batch.batch.batch_size[0] // per_turn_mini_bs, 1)
                ) * self.config.actor_rollout_ref.actor.ppo_epochs
        else:
            batch = base_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch = batch.union(gen_batch_output)
            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        return batch

    @staticmethod
    def _filter_reward_extra_infos_by_uid(
        reward_extra_infos_dict: dict[str, list[Any]],
        source_uid: np.ndarray,
        kept_uid: np.ndarray,
    ) -> tuple[dict[str, list[Any]], np.ndarray]:
        """Filter reward extra infos with the same uid-level mask as rollout filtering."""
        kept_uid_set = {str(u) for u in kept_uid.tolist()}
        row_mask = np.asarray([str(u) in kept_uid_set for u in source_uid], dtype=bool)
        filtered_infos: dict[str, list[Any]] = {}
        for key, values in reward_extra_infos_dict.items():
            values_arr = np.asarray(values, dtype=object).reshape(-1)
            if values_arr.shape[0] == row_mask.shape[0]:
                filtered_infos[key] = values_arr[row_mask].tolist()
            else:
                # Skip keys that are not per-sample aligned with row_mask.
                continue
        return filtered_infos, row_mask

    @staticmethod
    def _build_last_token_sparse_rewards(batch: DataProto, step_rewards: list[float]) -> torch.Tensor:
        """Build token-level rewards with non-zero value only on each row's last valid response token."""
        if len(step_rewards) != len(batch):
            raise ValueError(
                f"step_reward length mismatch: got {len(step_rewards)} entries for batch size {len(batch)}"
            )
        response_mask = batch.batch["response_mask"]
        response_shape = batch.batch["responses"].shape
        step_rewards_tensor = torch.zeros(
            response_shape,
            dtype=torch.float32,
            device=batch.batch["responses"].device,
        )
        valid_lens = response_mask.sum(dim=-1).long()
        for i in range(len(batch)):
            vlen = int(valid_lens[i].item())
            if vlen <= 0:
                continue
            step_rewards_tensor[i, vlen - 1] = float(step_rewards[i])
        return step_rewards_tensor

    @staticmethod
    def _get_stepwise_state_values(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Extract per-row state values aligned to action start (0 for empty rows)."""
        return extract_first_valid_token_scalar(values, response_mask)

    @staticmethod
    def _compute_trajectory_len_mean(batch: DataProto) -> Optional[float]:
        if "trajectory_id" not in batch.non_tensor_batch or "step_num" not in batch.non_tensor_batch:
            return None
        seen: set[str] = set()
        traj_lens: list[float] = []
        for tid, step_num in zip(
            batch.non_tensor_batch["trajectory_id"], batch.non_tensor_batch["step_num"], strict=False
        ):
            tid = str(tid)
            if tid in seen:
                continue
            seen.add(tid)
            traj_lens.append(float(step_num))
        if not traj_lens:
            return None
        return float(np.mean(traj_lens))

    @staticmethod
    def _compute_prompt_only_value_metrics(batch: DataProto) -> dict[str, float]:
        if "state_values" not in batch.batch or "state_returns" not in batch.batch:
            return {}

        state_values = batch.batch["state_values"].float()
        state_returns = batch.batch["state_returns"].float()
        if state_values.dim() == 2 and state_values.size(-1) == 1:
            state_values = state_values.squeeze(-1)
        if state_returns.dim() == 2 and state_returns.size(-1) == 1:
            state_returns = state_returns.squeeze(-1)
        if state_values.numel() == 0:
            return {}

        return_diff_var = torch.var(state_returns - state_values)
        return_var = torch.var(state_returns)
        return {
            "critic/values/mean": torch.mean(state_values).detach().item(),
            "critic/values/max": torch.max(state_values).detach().item(),
            "critic/values/min": torch.min(state_values).detach().item(),
            "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        }

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        critic_batch = batch
        if self.stepwise_prompt_value_wrapper_enable:
            if "state_values" not in batch.batch or "state_returns" not in batch.batch:
                raise ValueError(
                    "Prompt-only stepwise critic wrapper requires batch['state_values'] and batch['state_returns']"
                )
            critic_batch = build_prompt_value_batch(
                batch,
                values=batch.batch["state_values"],
                returns=batch.batch["state_returns"],
            )
        if self.use_legacy_worker_impl == "disable":
            batch_td = critic_batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(critic_batch)
        return critic_output

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        consecutive_all_filtered_steps = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                gen_batch.meta_info["global_steps"] = self.global_steps
                is_last_step = self.global_steps >= self.total_training_steps
                skip_due_to_filtered_batch = False
                early_stop_due_to_filtered_batch = False
                try:
                    with marked_timer("step", timing_raw):
                        reward_tensor: torch.Tensor
                        reward_extra_infos_dict: dict[str, list[Any]] = {}
                        future_reward = None

                        if self.rollout_filter_enable:
                            gen_batch_output = gen_batch.repeat(
                                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                            )
                            with marked_timer("gen", timing_raw, color="red"):
                                if not self.async_rollout_mode:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                                else:
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                                timing_raw.update(gen_batch_output.meta_info["timing"])
                                gen_batch_output.meta_info.pop("timing", None)

                            working_batch = self._build_training_batch_from_rollout(
                                base_batch=batch,
                                gen_batch_output=gen_batch_output,
                                metrics=metrics,
                            )

                            with marked_timer("reward", timing_raw, color="yellow"):
                                if self.use_rm and "rm_scores" not in working_batch.batch.keys():
                                    if not self.use_reward_loop:
                                        rm_scores = self.rm_wg.compute_rm_score(working_batch)
                                    else:
                                        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                        rm_scores = self.reward_loop_manager.compute_rm_score(working_batch)
                                    working_batch = working_batch.union(rm_scores)

                                reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                    working_batch, reward_fn=self.reward_fn, return_dict=False
                                )

                            with marked_timer("filter", timing_raw, color="purple"):
                                source_uid = np.asarray(working_batch.non_tensor_batch["uid"], dtype=object)
                                filtered_batch, filter_metrics = filter_batch_by_reward_variance(
                                    batch=working_batch,
                                    reward_tensor=reward_tensor,
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    config=self.rollout_filter_runtime_cfg,
                                )
                                metrics.update(filter_metrics)

                                reward_extra_infos_dict, reward_row_mask = self._filter_reward_extra_infos_by_uid(
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    source_uid=source_uid,
                                    kept_uid=np.asarray(filtered_batch.non_tensor_batch["uid"], dtype=object),
                                )
                                reward_row_mask_t = torch.from_numpy(reward_row_mask).to(
                                    device=reward_tensor.device, dtype=torch.bool
                                )
                                reward_tensor = reward_tensor[reward_row_mask_t]
                                working_batch = filtered_batch
                                working_batch.meta_info["filter_kept_ratio"] = filter_metrics.get(
                                    "rollout/filter_kept_ratio", 1.0
                                )

                            if len(working_batch) == 0:
                                consecutive_all_filtered_steps += 1
                                max_skipped = self.rollout_filter_max_consecutive_all_filtered_steps
                                early_stop_due_to_filtered_batch = consecutive_all_filtered_steps >= max_skipped
                                print(
                                    f"[rollout-filter] Step {self.global_steps}: all samples filtered out; "
                                    f"skipping optimizer update ({consecutive_all_filtered_steps}/{max_skipped} "
                                    "consecutive skipped steps)."
                                )
                                if early_stop_due_to_filtered_batch:
                                    print(
                                        "[rollout-filter] Early stopping because rollout filtering removed all "
                                        f"samples for {consecutive_all_filtered_steps} consecutive steps."
                                    )

                                metrics.update(
                                    {
                                        "rollout/filter_all_skipped": 1.0,
                                        "rollout/consecutive_all_filtered_steps": float(consecutive_all_filtered_steps),
                                        "training/skipped_update": 1.0,
                                        "training/early_stop": 1.0 if early_stop_due_to_filtered_batch else 0.0,
                                        "training/early_stop_due_to_rollout_filter": (
                                            1.0 if early_stop_due_to_filtered_batch else 0.0
                                        ),
                                    }
                                )
                                raise _SkipFullyFilteredStep

                            consecutive_all_filtered_steps = 0
                            metrics.update(
                                {
                                    "rollout/filter_all_skipped": 0.0,
                                    "rollout/consecutive_all_filtered_steps": 0.0,
                                    "training/skipped_update": 0.0,
                                    "training/early_stop": 0.0,
                                    "training/early_stop_due_to_rollout_filter": 0.0,
                                }
                            )
                            batch = working_batch
                        else:
                            gen_batch_output = gen_batch.repeat(
                                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                            )
                            with marked_timer("gen", timing_raw, color="red"):
                                if not self.async_rollout_mode:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                                else:
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                                timing_raw.update(gen_batch_output.meta_info["timing"])
                                gen_batch_output.meta_info.pop("timing", None)

                            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                                if self.reward_fn is None:
                                    raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                                with marked_timer("gen_max", timing_raw, color="purple"):
                                    gen_baseline_batch = deepcopy(gen_batch)
                                    gen_baseline_batch.meta_info["do_sample"] = False
                                    if not self.async_rollout_mode:
                                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                    else:
                                        gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                    batch = batch.union(gen_baseline_output)
                                    rm_scores = None
                                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                                        if not self.use_reward_loop:
                                            rm_scores = self.rm_wg.compute_rm_score(batch)
                                        else:
                                            assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                            rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                        batch = batch.union(rm_scores)

                                    reward_baseline_tensor = self._compute_or_extract_reward(
                                        batch, reward_fn=self.reward_fn, sum_reward=True
                                    )

                                    keys_to_pop = set(gen_baseline_output.batch.keys())
                                    if rm_scores is not None:
                                        keys_to_pop.update(rm_scores.batch.keys())
                                    batch.pop(batch_keys=list(keys_to_pop))

                                    batch.batch["reward_baselines"] = reward_baseline_tensor

                                    del rm_scores, gen_baseline_batch, gen_baseline_output

                            batch = self._build_training_batch_from_rollout(
                                base_batch=batch,
                                gen_batch_output=gen_batch_output,
                                metrics=metrics,
                            )

                            with marked_timer("reward", timing_raw, color="yellow"):
                                if self.use_rm and "rm_scores" not in batch.batch.keys():
                                    if not self.use_reward_loop:
                                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                                    else:
                                        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                        reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                                    batch = batch.union(reward_tensor)

                                if self.config.reward_model.launch_reward_fn_async:
                                    future_reward = compute_reward_async.remote(
                                        data=batch, config=self.config, tokenizer=self.tokenizer
                                    )
                                else:
                                    reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                        batch, reward_fn=self.reward_fn, return_dict=False
                                    )

                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                            "bypass_mode", False
                        )
                        if bypass_recomputing_logprobs:
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                actor_config = self.config.actor_rollout_ref.actor
                                entropy_agg = agg_loss(
                                    loss_mat=entropys,
                                    loss_mask=response_masks,
                                    loss_agg_mode=actor_config.loss_agg_mode,
                                    loss_scale_factor=actor_config.loss_scale_factor,
                                )
                                old_log_prob_metrics = {
                                    "actor/entropy": entropy_agg.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                ref_log_prob = self._compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self._compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            if self.stepwise_advantage_enable:
                                step_rewards_raw = reward_extra_infos_dict.get("step_reward")
                                if step_rewards_raw is None:
                                    raise ValueError(
                                        "trainer.stepwise_advantage.enable=True requires reward extra info "
                                        "field 'step_reward' from reward function"
                                    )
                                step_rewards_arr = np.asarray(step_rewards_raw, dtype=np.float32).reshape(-1)
                                if step_rewards_arr.shape[0] != len(batch):
                                    raise ValueError(
                                        "Length mismatch for step rewards: "
                                        f"{step_rewards_arr.shape[0]} vs batch size {len(batch)}"
                                    )
                                step_rewards = self._build_last_token_sparse_rewards(
                                    batch=batch, step_rewards=step_rewards_arr.tolist()
                                )
                                batch.batch["step_rewards"] = step_rewards
                                batch.batch["traj_rewards"] = reward_tensor
                                batch.batch["token_level_scores"] = step_rewards
                                metrics["train-stepwise/step_reward_mean"] = float(step_rewards_arr.mean())
                                traj_len_mean = self._compute_trajectory_len_mean(batch)
                                if traj_len_mean is not None:
                                    metrics["train-stepwise/trajectory_len_mean"] = traj_len_mean
                            else:
                                batch.batch["token_level_scores"] = reward_tensor

                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import (
                                    compute_rollout_correction_and_add_to_batch,
                                )

                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                    batch, rollout_corr_config
                                )
                                metrics.update(is_metrics)

                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                                stepwise_config=self.stepwise_advantage_cfg,
                            )

                            if self.stepwise_advantage_enable:
                                adv = batch.batch["advantages"]
                                valid_mask = batch.batch["response_mask"].bool()
                                if valid_mask.any():
                                    metrics["train-stepwise/adv_abs_mean"] = float(adv[valid_mask].abs().mean().item())
                                if self.use_critic and "state_values" in batch.batch:
                                    metrics["train-stepwise/value_start_mean"] = float(
                                        batch.batch["state_values"].float().mean().item()
                                    )
                                elif self.use_critic and "values" in batch.batch:
                                    value_start = self._get_stepwise_state_values(
                                        batch.batch["values"], batch.batch["response_mask"]
                                    )
                                    metrics["train-stepwise/value_start_mean"] = float(value_start.mean().item())

                        # Log per-uid reward std to diagnose GRPO learning signal
                        if "uid" in batch.non_tensor_batch:
                            _scores = (batch.batch["token_level_scores"] * batch.batch["response_mask"]).sum(-1)
                            _uid_groups: dict[str, list[float]] = defaultdict(list)
                            for _i, _u in enumerate(batch.non_tensor_batch["uid"]):
                                _uid_groups[_u].append(_scores[_i].item())
                            _stds = [np.std(v) for v in _uid_groups.values() if len(v) > 1]
                            if _stds:
                                metrics["train-diag/reward_std_per_uid"] = float(np.mean(_stds))
                            _adv = batch.batch["advantages"]
                            metrics["train-diag/advantage_abs_mean"] = float(_adv.abs().mean().item())

                        if self.per_turn_training and (self.stepwise_advantage_enable or self.rollout_filter_enable):
                            batch = self._pad_batch_for_training(batch)
                            batch.meta_info["global_token_num"] = (
                                torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                            )
                            per_turn_mini_bs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                            metrics["per_turn/batch_size"] = batch.batch.batch_size[0]
                            metrics["per_turn/num_updates"] = (
                                max(batch.batch.batch_size[0] // per_turn_mini_bs, 1)
                            ) * self.config.actor_rollout_ref.actor.ppo_epochs

                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self._update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self._update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir and not self.per_turn_training:
                            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                except _SkipFullyFilteredStep:
                    skip_due_to_filtered_batch = True

                if skip_due_to_filtered_batch:
                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = (
                            self.global_steps + 1 in self.config.global_profiler.steps
                            if self.config.global_profiler.steps is not None
                            else False
                        )
                        self._stop_profiling(
                            curr_step_profile and not next_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = timing_raw.get("step", 0.0)
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if early_stop_due_to_filtered_batch or is_last_step:
                        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                        if early_stop_due_to_filtered_batch:
                            pprint("Training ended early due to consecutive fully-filtered rollouts.")
                        progress_bar.close()
                        return

                    continue

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                use_critic_for_data_metrics = self.use_critic and not (
                    self.stepwise_prompt_value_wrapper_enable and "state_values" in batch.batch
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=use_critic_for_data_metrics))
                if self.use_critic and self.stepwise_prompt_value_wrapper_enable:
                    metrics.update(self._compute_prompt_only_value_metrics(batch))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Log train-env/ trajectory metrics from non_tensor_batch,
                # including grouped rollups such as train-env-by-dataset_split/*.
                add_training_env_metrics(metrics, batch.non_tensor_batch)

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
