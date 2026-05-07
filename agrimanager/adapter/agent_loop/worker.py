"""Custom AgentLoop Worker and Manager for per-turn expansion at the worker level.

This module expands per-turn trajectories into individual
samples inside the worker's generate_sequences() call.  The trainer then
receives a flat batch of per-turn samples and can follow the standard
old_log_probs -> ref_log_probs -> reward -> advantage -> actor update order.
"""

import asyncio
import logging
import numbers
import os
import uuid
from copy import deepcopy
from typing import Any

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorkerBase,
    DictConfigWrap,
    _agent_loop_registry,
    get_trajectory_info,
)
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr
from verl.utils.transferqueue_utils import tqbridge

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _missing_reward_extra_value(values: list[Any]) -> Any:
    """Return a typed placeholder for absent reward-extra fields."""
    for value in values:
        if isinstance(value, str):
            return ""
        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            return float("nan")
    return None


def _rectangularize_reward_extra_infos(outputs: list[Any]) -> None:
    """Make reward_extra_info dictionaries share the same keys before VERL merge.

    VERL's agent-loop postprocess takes keys from the first sample and then
    indexes every sample with those keys. Mixed Agri simulators can emit
    simulator-specific trajectory metrics, so fill missing numeric metrics with
    NaN and missing string labels with an empty string in the Agri adapter.
    """
    if not outputs:
        return

    key_values: dict[str, list[Any]] = {}
    key_order: list[str] = []
    for output in outputs:
        reward_extra_info = output.extra_fields.get("reward_extra_info", {})
        if not isinstance(reward_extra_info, dict):
            continue
        for key, value in reward_extra_info.items():
            if key not in key_values:
                key_values[key] = []
                key_order.append(key)
            key_values[key].append(value)

    if not key_order:
        return

    fill_values = {
        key: _missing_reward_extra_value(values)
        for key, values in key_values.items()
    }
    for output in outputs:
        reward_extra_info = output.extra_fields.get("reward_extra_info", {})
        if not isinstance(reward_extra_info, dict):
            reward_extra_info = {}
        output.extra_fields["reward_extra_info"] = {
            key: reward_extra_info.get(key, fill_values[key])
            for key in key_order
        }


def _rectangularize_dataproto_non_tensor_batches(outputs: list[DataProto]) -> None:
    """Make worker-level DataProto non-tensor schemas compatible for concat."""
    if not outputs:
        return

    key_values: dict[str, list[Any]] = {}
    key_order: list[str] = []
    reward_extra_key_order: list[str] = []
    for output in outputs:
        for key in output.meta_info.get("reward_extra_keys", []):
            if key not in reward_extra_key_order:
                reward_extra_key_order.append(key)
        for key, values in output.non_tensor_batch.items():
            if key not in key_values:
                key_values[key] = []
                key_order.append(key)
            if isinstance(values, np.ndarray):
                key_values[key].extend(values.reshape(-1).tolist())
            else:
                key_values[key].append(values)

    fill_values = {
        key: _missing_reward_extra_value(values)
        for key, values in key_values.items()
    }
    for output in outputs:
        batch_size = len(output)
        for key in key_order:
            if key in output.non_tensor_batch:
                continue
            fill_value = fill_values[key]
            missing = np.empty(batch_size, dtype=object)
            missing[:] = [fill_value] * batch_size
            output.non_tensor_batch[key] = missing
        if reward_extra_key_order:
            output.meta_info["reward_extra_keys"] = reward_extra_key_order


class AgriAgentLoopWorkerBase(AgentLoopWorkerBase):
    """Worker that expands per-turn trajectories into flat per-turn samples.

    Overrides ``generate_sequences`` and ``_run_agent_loop`` so that when the
    agent loop returns ``per_turn_data`` in its ``extra_fields``, each turn is
    independently post-processed into an ``_InternalAgentLoopOutput``.  The
    resulting flat list is then assembled into a single ``DataProto`` batch by
    the inherited ``_postprocess``.
    """

    def _pad_sequence_batch(
        self,
        tensors: list[torch.Tensor],
        *,
        padding_value: int | float,
        left_pad: bool = False,
    ) -> torch.Tensor:
        """Pad [1, seq_len] tensors before concatenating per-turn samples."""
        max_len = max(tensor.size(1) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            pad_len = max_len - tensor.size(1)
            if pad_len <= 0:
                padded_tensors.append(tensor)
                continue
            pad = (pad_len, 0) if left_pad else (0, pad_len)
            padded_tensors.append(torch.nn.functional.pad(tensor, pad, value=padding_value))
        return torch.cat(padded_tensors, dim=0)

    def _pad_dim1_batch(self, tensors: list[torch.Tensor], *, padding_value: int | float = 0) -> torch.Tensor:
        """Pad tensors whose variable sequence dimension is dim=1."""
        max_len = max(tensor.size(1) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            pad_len = max_len - tensor.size(1)
            if pad_len <= 0:
                padded_tensors.append(tensor)
                continue
            pad = [0, 0] * (tensor.dim() - 2) + [0, pad_len]
            padded_tensors.append(torch.nn.functional.pad(tensor, tuple(pad), value=padding_value))
        return torch.cat(padded_tensors, dim=0)

    def _pad_position_ids_batch(self, tensors: list[torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        """Pad or recompute position ids after prompt/input padding."""
        if all(tensor.size(-1) == tensors[0].size(-1) for tensor in tensors):
            return torch.cat(tensors, dim=0)
        if tensors[0].dim() == 2:
            return compute_position_id_with_mask(attention_mask)

        max_len = max(tensor.size(-1) for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            pad_len = max_len - tensor.size(-1)
            if pad_len <= 0:
                padded_tensors.append(tensor)
                continue
            padded_tensors.append(torch.nn.functional.pad(tensor, (0, pad_len), value=0))
        return torch.cat(padded_tensors, dim=0)

    def _postprocess(self, inputs: list[Any]) -> DataProto:
        """Combine per-turn outputs while allowing different prompt lengths.

        The upstream verl implementation concatenates prompt/input tensors
        directly. AgriManager per-turn prompts include growing trajectory
        context, so prompt lengths differ across turns. Padding here keeps the
        fix local to AgriManager instead of carrying a dirty verl submodule.
        """
        pad_token_id = self.tokenizer.pad_token_id or 0
        prompt_ids = self._pad_sequence_batch(
            [input.prompt_ids for input in inputs],
            padding_value=pad_token_id,
            left_pad=True,
        )
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = self._pad_sequence_batch(
            [input.attention_mask for input in inputs],
            padding_value=0,
            left_pad=True,
        )
        input_ids = self._pad_sequence_batch(
            [input.input_ids for input in inputs],
            padding_value=pad_token_id,
            left_pad=True,
        )
        position_ids = self._pad_position_ids_batch([input.position_ids for input in inputs], attention_mask)

        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = self._pad_dim1_batch(
                [input.routed_experts for input in inputs],
                padding_value=0,
            )

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array(
                [default_agent_loop] * len(batch), dtype=object
            )

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1),
            index.tolist(),
            batch.meta_info.get("validate", False),
        )

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(
                        sampling_params,
                        trajectory_info[i],
                        trace=trace_this_sample,
                        **kwargs,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)

        # Flatten: each output may be a single _InternalAgentLoopOutput or a list
        flat_outputs = []
        for output in outputs:
            if isinstance(output, list):
                flat_outputs.extend(output)
            else:
                flat_outputs.append(output)

        _rectangularize_reward_extra_infos(flat_outputs)
        return self._postprocess(flat_outputs)

    async def _agent_loop_postprocess(self, output, **kwargs):
        internal = await super()._agent_loop_postprocess(output, **kwargs)
        return self._normalize_internal_output_lengths(internal)

    def _normalize_internal_output_lengths(self, output):
        """Keep Agri per-turn samples compatible across mixed simulator workers.

        VERL pads each worker independently.  If an environment-generated prompt
        exceeds ``rollout.prompt_length``, tokenizer.pad returns that longer
        prompt instead of truncating it.  Mixed simulator validation can then
        produce worker batches with different sequence lengths, which cannot be
        concatenated.  Normalize here in the Agri wrapper so VERL stays untouched.
        """
        rollout_config = self.config.actor_rollout_ref.rollout
        prompt_length = int(rollout_config.prompt_length)
        response_length = int(rollout_config.response_length)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        original_prompt_length = output.prompt_ids.shape[-1]
        prompt_attention = output.attention_mask[..., :original_prompt_length]
        response_attention = output.attention_mask[..., original_prompt_length:]

        output.prompt_ids = self._fit_last_dim(
            output.prompt_ids,
            prompt_length,
            pad_token_id,
            keep="right",
        )
        prompt_attention = self._fit_last_dim(
            prompt_attention,
            prompt_length,
            0,
            keep="right",
        )

        output.response_ids = self._fit_last_dim(
            output.response_ids,
            response_length,
            pad_token_id,
            keep="left",
        )
        output.response_mask = self._fit_last_dim(
            output.response_mask,
            response_length,
            0,
            keep="left",
        )
        response_attention = self._fit_last_dim(
            response_attention,
            response_length,
            0,
            keep="left",
        )
        if output.response_logprobs is not None:
            output.response_logprobs = self._fit_last_dim(
                output.response_logprobs,
                response_length,
                0.0,
                keep="left",
            )

        output.input_ids = torch.cat([output.prompt_ids, output.response_ids], dim=-1)
        output.attention_mask = torch.cat([prompt_attention, response_attention], dim=-1)
        output.position_ids = self._compute_position_ids(
            output.input_ids,
            output.attention_mask,
            output.multi_modal_inputs or {},
        )

        if output.routed_experts is not None:
            output.routed_experts = self._fit_dim(
                output.routed_experts,
                dim=1,
                target_length=prompt_length + response_length,
                pad_value=0,
                keep="right",
            )

        return output

    @staticmethod
    def _fit_last_dim(tensor: torch.Tensor, target_length: int, pad_value, *, keep: str) -> torch.Tensor:
        return AgriAgentLoopWorkerBase._fit_dim(
            tensor,
            dim=tensor.dim() - 1,
            target_length=target_length,
            pad_value=pad_value,
            keep=keep,
        )

    @staticmethod
    def _fit_dim(
        tensor: torch.Tensor,
        *,
        dim: int,
        target_length: int,
        pad_value,
        keep: str,
    ) -> torch.Tensor:
        current_length = tensor.shape[dim]
        if current_length == target_length:
            return tensor

        if current_length > target_length:
            start = current_length - target_length if keep == "right" else 0
            return tensor.narrow(dim, start, target_length)

        pad_shape = list(tensor.shape)
        pad_shape[dim] = target_length - current_length
        padding = tensor.new_full(pad_shape, pad_value)
        return torch.cat([padding, tensor], dim=dim) if keep == "right" else torch.cat([tensor, padding], dim=dim)

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ):
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

            per_turn_data = output.extra_fields.get("per_turn_data")
            if per_turn_data is None:
                # Non per-turn mode: use base class post-processing
                return await self._agent_loop_postprocess(output, **kwargs)

            # --- Per-turn expansion ---
            traj_id = str(uuid.uuid4())
            turn_scores = output.extra_fields.get("turn_scores", [])
            interaction_metrics = output.extra_fields.get("interaction_metrics", [])
            response_length = self.config.actor_rollout_ref.rollout.response_length

            # Build shared trajectory metadata injected into extra_info.
            # Keep list copies to avoid accidental cross-turn mutation.
            traj_extra_info = deepcopy(dict(kwargs.get("extra_info", {}) or {}))
            traj_extra_info["turn_scores"] = list(turn_scores)
            traj_extra_info["interaction_metrics"] = list(interaction_metrics)

            results = []
            total_steps = len(per_turn_data)
            for turn_idx, turn in enumerate(per_turn_data):
                response_ids = turn["response_ids"][:response_length]
                logprobs = turn["logprobs"][:response_length] if turn["logprobs"] else None

                turn_output = AgentLoopOutput(
                    prompt_ids=turn["prompt_ids"],
                    response_ids=response_ids,
                    response_mask=[1] * min(len(turn["response_ids"]), response_length),
                    response_logprobs=logprobs,
                    num_turns=1,
                    metrics=AgentLoopMetrics(),
                    extra_fields={},
                )

                # Each turn's kwargs carry full trajectory metadata
                turn_kwargs = dict(kwargs)
                turn_extra_info = deepcopy(traj_extra_info)
                turn_extra_info["step_idx"] = turn_idx
                turn_extra_info["step_num"] = total_steps
                turn_extra_info["is_last_step"] = turn_idx == (total_steps - 1)
                turn_kwargs["extra_info"] = turn_extra_info

                internal = await self._agent_loop_postprocess(turn_output, **turn_kwargs)

                # Inject non_tensor_batch fields into extra_fields;
                # _postprocess() merges extra_fields into non_tensor_batch.
                internal.extra_fields["data_source"] = kwargs.get("data_source")
                internal.extra_fields["reward_model"] = kwargs.get("reward_model")
                internal.extra_fields["uid"] = kwargs.get("uid")
                internal.extra_fields["trajectory_id"] = traj_id
                internal.extra_fields["step_idx"] = turn_idx
                internal.extra_fields["step_num"] = total_steps
                internal.extra_fields["is_last_step"] = turn_idx == (total_steps - 1)
                internal.extra_fields["extra_info"] = turn_extra_info
                internal.extra_fields["per_turn_data"] = per_turn_data

                results.append(internal)

            return results


@ray.remote
class AgriAgentLoopWorker(AgriAgentLoopWorkerBase):
    """Ray remote wrapper for AgriAgentLoopWorkerBase."""

    pass


def _select_even_worker_count(batch_size: int, max_workers: int) -> int:
    """Choose a worker count that DataProto.chunk can split evenly."""
    if batch_size <= 0 or max_workers <= 0:
        return 0
    for worker_count in range(min(batch_size, max_workers), 0, -1):
        if batch_size % worker_count == 0:
            return worker_count
    return 1


class AgriAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that uses AgriAgentLoopWorker for per-turn expansion.

    Also overrides ``generate_sequences`` to handle validation batches that do
    not divide the configured worker count by dispatching to a smaller even
    divisor instead of requiring padding.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rm_resource_pool: RayResourcePool = None,
    ):
        # Set custom worker class before super().__init__() which checks
        # ``if not hasattr(self, "agent_loop_workers_class")``.
        self.agent_loop_workers_class = AgriAgentLoopWorker
        super().__init__(config, worker_group, rm_resource_pool)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        num_workers = len(self.agent_loop_workers)
        batch_size = len(prompts)

        num_workers = _select_even_worker_count(batch_size, num_workers)
        workers = self.agent_loop_workers[:num_workers]

        chunks = prompts.chunk(num_workers)
        outputs = ray.get([
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(workers, chunks, strict=True)
        ])
        _rectangularize_dataproto_non_tensor_batches(outputs)
        output = DataProto.concat(outputs)

        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        metrics = [output_i.meta_info.pop("metrics") for output_i in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output
