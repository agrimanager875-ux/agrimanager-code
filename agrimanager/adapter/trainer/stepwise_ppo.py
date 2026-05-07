"""Helpers for stepwise PPO.

This module keeps stepwise PPO utilities independent from the full trainer so
they can be unit-tested without importing the entire PPO stack.

It contains:
- prompt-only critic batch wrapping for state-value estimation
- cross-turn stepwise GAE helpers
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils import torch_functional as verl_F


def _ensure_column_tensor(values: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if values.dim() == 1:
        if values.shape[0] != batch_size:
            raise ValueError(f"{name} length mismatch: {values.shape[0]} vs batch size {batch_size}")
        return values.unsqueeze(-1)
    if values.dim() == 2 and values.shape[1] == 1:
        if values.shape[0] != batch_size:
            raise ValueError(f"{name} length mismatch: {values.shape[0]} vs batch size {batch_size}")
        return values
    raise ValueError(f"{name} must have shape [batch] or [batch, 1], got {tuple(values.shape)}")


def build_prompt_value_batch(
    batch: DataProto,
    *,
    values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
) -> DataProto:
    """Build a critic batch that predicts only the prompt-state value.

    The wrapper keeps the left-padded prompt and the first response token, so
    VERL's existing critic implementation returns exactly one scalar aligned to
    the prompt state.
    """

    required_keys = {"input_ids", "responses", "attention_mask", "position_ids", "response_mask"}
    missing = sorted(required_keys - set(batch.batch.keys()))
    if missing:
        raise ValueError(f"Missing required batch keys for prompt-value wrapper: {missing}")

    input_ids = batch.batch["input_ids"]
    responses = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]
    position_ids = batch.batch["position_ids"]
    response_mask = batch.batch["response_mask"]

    if responses.size(-1) < 1:
        raise ValueError("Prompt-value wrapper requires responses with length >= 1")

    prompt_padded_length = input_ids.size(-1) - responses.size(-1)
    prefix_length = prompt_padded_length + 1
    batch_size = input_ids.size(0)

    wrapped_tensors = {
        "input_ids": input_ids[:, :prefix_length],
        "responses": responses[:, :1],
        "attention_mask": attention_mask[:, :prefix_length],
        "position_ids": position_ids[..., :prefix_length],
        "response_mask": (response_mask.sum(dim=-1, keepdim=True) > 0).to(dtype=response_mask.dtype),
    }
    if values is not None:
        wrapped_tensors["values"] = _ensure_column_tensor(values, batch_size, "values")
    if returns is not None:
        wrapped_tensors["returns"] = _ensure_column_tensor(returns, batch_size, "returns")

    wrapped_non_tensors = {}
    if "multi_modal_inputs" in batch.non_tensor_batch:
        wrapped_non_tensors["multi_modal_inputs"] = batch.non_tensor_batch["multi_modal_inputs"]

    return DataProto.from_dict(
        tensors=wrapped_tensors,
        non_tensors=wrapped_non_tensors,
        meta_info=dict(batch.meta_info),
    )


def extract_first_valid_token_scalar(token_tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Extract the scalar at the first valid response token for each sample."""
    valid_pos = response_mask.to(dtype=torch.bool)
    first_pos = valid_pos.to(dtype=torch.long).argmax(dim=-1)
    rows = torch.arange(token_tensor.size(0), device=token_tensor.device)
    first_vals = token_tensor[rows, first_pos]

    empty_rows = response_mask.sum(dim=-1) <= 0
    if empty_rows.any():
        first_vals = first_vals.clone()
        first_vals[empty_rows] = 0.0
    return first_vals


def extract_last_valid_token_scalar(token_tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Extract the scalar at the last valid response token for each sample."""
    valid_lens = response_mask.sum(dim=-1).long()
    rows = torch.arange(token_tensor.size(0), device=token_tensor.device)
    last_pos = torch.clamp(valid_lens - 1, min=0)
    last_vals = token_tensor[rows, last_pos]

    empty_rows = valid_lens <= 0
    if empty_rows.any():
        last_vals = last_vals.clone()
        last_vals[empty_rows] = 0.0
    return last_vals


def broadcast_scalar_to_response_tokens(scalars: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Broadcast one scalar per row onto all valid response-token positions."""
    if scalars.dim() == 2 and scalars.size(-1) == 1:
        scalars = scalars.squeeze(-1)
    if scalars.dim() != 1:
        raise ValueError(f"scalars must have shape [batch] or [batch, 1], got {tuple(scalars.shape)}")
    if scalars.size(0) != response_mask.size(0):
        raise ValueError(f"batch mismatch: {scalars.size(0)} vs {response_mask.size(0)}")
    return scalars.unsqueeze(-1) * response_mask


def whiten_state_advantages(state_advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Whiten one scalar advantage per row before broadcasting to response tokens."""
    if state_advantages.dim() != 1:
        raise ValueError(f"state_advantages must have shape [batch], got {tuple(state_advantages.shape)}")
    if state_advantages.size(0) != response_mask.size(0):
        raise ValueError(f"batch mismatch: {state_advantages.size(0)} vs {response_mask.size(0)}")

    valid_rows = (response_mask.sum(dim=-1, keepdim=True) > 0).to(dtype=state_advantages.dtype)
    valid_count = int(valid_rows.sum().item())
    if valid_count <= 1:
        return state_advantages

    whitened = verl_F.masked_whiten(state_advantages.unsqueeze(-1), valid_rows).squeeze(-1)
    return torch.where(valid_rows.squeeze(-1).bool(), whitened, torch.zeros_like(whitened))


def compute_stepwise_gae_advantage_cross_turn_from_scalar_values(
    step_rewards: torch.Tensor,
    state_values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_id: np.ndarray,
    step_idx: np.ndarray,
    gamma: float,
    lam: float,
    whiten_advantages: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute cross-turn GAE from scalar state values and broadcast to tokens."""
    if step_rewards.dim() == 2 and step_rewards.size(-1) == 1:
        step_rewards = step_rewards.squeeze(-1)
    if state_values.dim() == 2 and state_values.size(-1) == 1:
        state_values = state_values.squeeze(-1)
    if step_rewards.dim() != 1:
        raise ValueError(f"step_rewards must have shape [batch] or [batch, 1], got {tuple(step_rewards.shape)}")
    if state_values.dim() != 1:
        raise ValueError(f"state_values must have shape [batch] or [batch, 1], got {tuple(state_values.shape)}")
    if len(trajectory_id) != step_rewards.size(0):
        raise ValueError(f"trajectory_id length mismatch: {len(trajectory_id)} vs batch {step_rewards.size(0)}")
    if len(step_idx) != step_rewards.size(0):
        raise ValueError(f"step_idx length mismatch: {len(step_idx)} vs batch {step_rewards.size(0)}")
    if state_values.size(0) != step_rewards.size(0):
        raise ValueError(f"state_values length mismatch: {state_values.size(0)} vs batch {step_rewards.size(0)}")

    state_advantages = torch.zeros_like(step_rewards)
    state_returns = torch.zeros_like(step_rewards)

    traj_indices: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(trajectory_id):
        traj_indices[str(tid)].append(i)
    for tid in traj_indices:
        traj_indices[tid].sort(key=lambda i: int(step_idx[i]))

    with torch.no_grad():
        for idxs in traj_indices.values():
            if not idxs:
                continue

            next_adv = torch.tensor(0.0, dtype=step_rewards.dtype, device=step_rewards.device)
            for local_pos in reversed(range(len(idxs))):
                global_idx = idxs[local_pos]
                r_t = step_rewards[global_idx]
                v_t = state_values[global_idx]
                if local_pos < len(idxs) - 1:
                    next_global_idx = idxs[local_pos + 1]
                    v_tp1 = state_values[next_global_idx]
                else:
                    v_tp1 = torch.tensor(0.0, dtype=step_rewards.dtype, device=step_rewards.device)

                delta = r_t + gamma * v_tp1 - v_t
                next_adv = delta + gamma * lam * next_adv
                state_advantages[global_idx] = next_adv
                state_returns[global_idx] = next_adv + v_t

    if whiten_advantages:
        state_advantages = whiten_state_advantages(state_advantages, response_mask)

    advantages = broadcast_scalar_to_response_tokens(state_advantages, response_mask)
    returns = broadcast_scalar_to_response_tokens(state_returns, response_mask)
    return advantages, returns, state_advantages, state_returns


def compute_stepwise_gae_advantage_cross_turn(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_id: np.ndarray,
    step_idx: np.ndarray,
    gamma: float,
    lam: float,
    whiten_advantages: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute turn-level GAE across a trajectory, then broadcast to turn tokens.

    Semantics:
    - ``r_t`` is the reward on the turn's last valid response token.
    - ``V_t`` is the critic value at the turn's first valid response token,
      which is the position aligned with the state before generating the action.

    For each trajectory, with rows ordered by ``step_idx``:

    ``delta_t = r_t + gamma * V_{t+1} - V_t``
    ``A_t = delta_t + gamma * lam * A_{t+1}``
    ``R_t = A_t + V_t``
    """
    if len(trajectory_id) != token_level_rewards.size(0):
        raise ValueError(
            f"trajectory_id length mismatch: {len(trajectory_id)} vs batch {token_level_rewards.size(0)}"
        )
    if len(step_idx) != token_level_rewards.size(0):
        raise ValueError(f"step_idx length mismatch: {len(step_idx)} vs batch {token_level_rewards.size(0)}")

    step_rewards = extract_last_valid_token_scalar(token_level_rewards, response_mask)
    state_values = extract_first_valid_token_scalar(values, response_mask)

    advantages, returns, _, _ = compute_stepwise_gae_advantage_cross_turn_from_scalar_values(
        step_rewards=step_rewards,
        state_values=state_values,
        response_mask=response_mask,
        trajectory_id=trajectory_id,
        step_idx=step_idx,
        gamma=gamma,
        lam=lam,
        whiten_advantages=whiten_advantages,
    )
    return advantages, returns
