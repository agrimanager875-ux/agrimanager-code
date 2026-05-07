"""Rollout filtering utilities for AgriTrainer.

First version focuses on RAGEN-style reward variance filtering:
- metric: reward_variance
- strategy: top_p
- filter_type: largest
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto


@dataclass
class RewardVarianceFilterConfig:
    """Configuration for reward variance rollout filtering."""

    value: float = 0.9
    include_zero: bool = True
    score_key: str = "traj_score"
    zero_eps: float = 1e-10
    top_p_prob_mode: str = "softmax"  # "softmax" or "linear"
    selection_eps: float = 0.01


def select_top_p_groups(
    scores: torch.Tensor,
    top_p: float,
    include_zero: bool = True,
    zero_eps: float = 1e-10,
    mode: str = "softmax",
    selection_eps: float = 0.01,
) -> torch.Tensor:
    """Select groups by top-p filtering.

    Args:
        mode: "softmax" uses softmax probability mass; "linear" uses raw score sum.
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {tuple(scores.shape)}")

    scores = scores.float()
    indices = torch.arange(scores.numel(), device=scores.device)

    if not include_zero:
        non_zero_mask = torch.abs(scores) > float(zero_eps)
        scores = scores[non_zero_mask]
        indices = indices[non_zero_mask]
        if indices.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=indices.device)

    if top_p >= 1.0:
        return indices

    if mode == "softmax":
        probs = torch.softmax(scores, dim=0)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff_index = int(torch.searchsorted(cumulative_probs, top_p).item())
        k = min(cutoff_index + 1, indices.numel())
        k = max(k, 1)
        top_groups_local_indices = sorted_indices[:k]
        return indices[top_groups_local_indices]

    elif mode == "linear":
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        threshold = top_p * scores.sum() - selection_eps
        cumulative_score = 0.0
        selected_count = 0

        for score in sorted_scores:
            if cumulative_score >= threshold:
                break
            if score.item() <= 0:
                break
            cumulative_score += score.item()
            selected_count += 1

        if cumulative_score >= threshold:
            top_groups_local_indices = sorted_indices[:selected_count]
            return indices[top_groups_local_indices]
        return torch.empty(0, dtype=torch.long, device=indices.device)

    else:
        raise ValueError(
            f"Unknown top_p_prob_mode: {mode}. Expected one of {{'linear', 'softmax'}}."
        )


def _extract_row_scores(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    reward_extra_infos_dict: dict[str, list[Any]],
    score_key: str,
) -> torch.Tensor:
    """Extract row-level trajectory scores with fallback order.

    Priority:
    1) reward_extra_infos_dict[score_key]
    2) reward_extra_infos_dict["score"]
    3) token reward sum over response_mask
    """
    batch_size = len(batch)
    device = reward_tensor.device

    for key in (score_key, "score"):
        values = reward_extra_infos_dict.get(key)
        if values is None:
            continue
        values_arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if values_arr.shape[0] == batch_size:
            return torch.tensor(values_arr, dtype=torch.float32, device=device)

    if "response_mask" not in batch.batch:
        raise ValueError("Missing batch['response_mask'] for reward score fallback")
    return (reward_tensor * batch.batch["response_mask"]).sum(dim=-1).float()


def compute_group_reward_std(
    uid: np.ndarray,
    row_scores: torch.Tensor,
    trajectory_id: np.ndarray | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-uid std/mean/max on trajectory-level scores.

    If trajectory_id is provided, de-duplicate rows by first trajectory occurrence
    before per-uid grouping to avoid turn-count bias in per-turn mode.
    """
    if row_scores.ndim != 1:
        raise ValueError(f"row_scores must be 1-D, got shape {tuple(row_scores.shape)}")
    if uid.shape[0] != row_scores.shape[0]:
        raise ValueError(f"uid length mismatch: {uid.shape[0]} vs {row_scores.shape[0]}")

    if trajectory_id is not None:
        if trajectory_id.shape[0] != row_scores.shape[0]:
            raise ValueError(
                f"trajectory_id length mismatch: {trajectory_id.shape[0]} vs {row_scores.shape[0]}"
            )
        seen_tids: set[str] = set()
        keep_indices: list[int] = []
        for i, tid in enumerate(trajectory_id):
            tid_key = str(tid)
            if tid_key in seen_tids:
                continue
            seen_tids.add(tid_key)
            keep_indices.append(i)
        if keep_indices:
            uid = uid[np.asarray(keep_indices, dtype=np.int64)]
            keep_idx_t = torch.tensor(keep_indices, dtype=torch.long, device=row_scores.device)
            row_scores = row_scores[keep_idx_t]
        else:
            uid = np.asarray([], dtype=object)
            row_scores = row_scores[:0]

    uid_order: list[str] = []
    grouped: dict[str, list[float]] = {}
    for i, u in enumerate(uid):
        key = str(u)
        if key not in grouped:
            grouped[key] = []
            uid_order.append(key)
        grouped[key].append(float(row_scores[i].item()))

    if not uid_order:
        empty = torch.tensor([], dtype=torch.float32, device=row_scores.device)
        return uid_order, empty, empty, empty

    group_std = torch.zeros(len(uid_order), dtype=torch.float32, device=row_scores.device)
    group_mean = torch.zeros(len(uid_order), dtype=torch.float32, device=row_scores.device)
    group_max = torch.zeros(len(uid_order), dtype=torch.float32, device=row_scores.device)
    for i, key in enumerate(uid_order):
        vals = torch.tensor(grouped[key], dtype=torch.float32, device=row_scores.device)
        group_mean[i] = vals.mean()
        group_max[i] = vals.max()
        if vals.numel() <= 1:
            group_std[i] = torch.tensor(0.0, dtype=torch.float32, device=row_scores.device)
        else:
            group_std[i] = vals.std()

    return uid_order, group_std, group_mean, group_max


def filter_batch_by_reward_variance(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    reward_extra_infos_dict: dict[str, list[Any]],
    config: RewardVarianceFilterConfig,
) -> tuple[DataProto, dict[str, float]]:
    """Filter batch by reward variance over prompt groups (uid)."""
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("Missing non_tensor_batch['uid'] required by reward variance filtering")

    uid = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
    if uid.shape[0] != len(batch):
        raise ValueError(f"uid length mismatch: {uid.shape[0]} vs batch {len(batch)}")

    trajectory_id = None
    if "trajectory_id" in batch.non_tensor_batch:
        trajectory_id = np.asarray(batch.non_tensor_batch["trajectory_id"], dtype=object)

    row_scores = _extract_row_scores(
        batch=batch,
        reward_tensor=reward_tensor,
        reward_extra_infos_dict=reward_extra_infos_dict,
        score_key=config.score_key,
    )
    uid_order, group_std, group_mean, group_max = compute_group_reward_std(
        uid=uid,
        row_scores=row_scores,
        trajectory_id=trajectory_id,
    )

    selected_group_idx = select_top_p_groups(
        scores=group_std,
        top_p=float(config.value),
        include_zero=bool(config.include_zero),
        zero_eps=float(config.zero_eps),
        mode=config.top_p_prob_mode,
        selection_eps=float(config.selection_eps),
    )

    selected_uid_set = {uid_order[int(i)] for i in selected_group_idx.tolist()}
    row_mask = np.asarray([str(u) in selected_uid_set for u in uid], dtype=bool)
    filtered_batch = batch.select_idxs(row_mask)

    if group_std.numel() > 0:
        in_group_reward_std = float(group_std.mean().item())
        in_group_reward_mean = float(group_mean.mean().item())
        in_group_reward_max = float(group_max.mean().item())
        filter_zero_count = float((torch.abs(group_std) <= float(config.zero_eps)).sum().item())
    else:
        in_group_reward_std = 0.0
        in_group_reward_mean = 0.0
        in_group_reward_max = 0.0
        filter_zero_count = 0.0

    if selected_group_idx.numel() > 0:
        chosen_in_group_reward_std = float(group_std[selected_group_idx].mean().item())
        chosen_in_group_reward_mean = float(group_mean[selected_group_idx].mean().item())
        chosen_in_group_reward_max = float(group_max[selected_group_idx].mean().item())
    else:
        chosen_in_group_reward_std = 0.0
        chosen_in_group_reward_mean = 0.0
        chosen_in_group_reward_max = 0.0

    total_groups = max(len(uid_order), 1)
    metrics = {
        "rollout/in_group_reward_std": in_group_reward_std,
        "rollout/in_group_reward_mean": in_group_reward_mean,
        "rollout/in_group_reward_max": in_group_reward_max,
        "rollout/chosen_in_group_reward_std": chosen_in_group_reward_std,
        "rollout/chosen_in_group_reward_mean": chosen_in_group_reward_mean,
        "rollout/chosen_in_group_reward_max": chosen_in_group_reward_max,
        "rollout/filter_kept_count": float(selected_group_idx.numel()),
        "rollout/filter_kept_ratio": float(selected_group_idx.numel() / total_groups),
        "rollout/filter_zero_count": filter_zero_count,
    }
    return filtered_batch, metrics
