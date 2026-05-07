# Per-Turn Training: Design Notes

## Architecture

Per-turn expansion happens at the **Worker** level (`AgriAgentLoopWorkerBase`),
not in the trainer.  This allows the trainer to use the standard computation
order for all modes.

### Data Flow

```
Dataset(B) → repeat(n) → AgentLoop (per-turn inference)
  → Worker expands: B*n trajectories → B*n*∑K per-turn samples
  → gen_batch_output = DataProto (B*n*∑K samples, with trajectory_id/step_idx/step_num/is_last_step)
  → Trainer uses gen_batch_output directly (skips repeat + union)
  → old_log_probs → ref_log_probs → reward → advantage → actor update
```

### Computation Order

Both per-turn and original modes follow the same order:

```
old_log_probs → ref_log_probs → reward → KL (optional) → advantage → actor update
```

This is possible because per-turn samples are already fully formed (each with
its own prompt_ids, response_ids, attention_mask etc.) by the time they reach
the trainer.

### Key Components

| Component | File | Role |
|-----------|------|------|
| `AgriAgentLoopWorkerBase` | `agent_loop/worker.py` | Overrides `generate_sequences()` and `_run_agent_loop()` to flatten per-turn data |
| `AgriAgentLoopWorker` | `agent_loop/worker.py` | Ray remote wrapper |
| `AgriAgentLoopManager` | `agent_loop/worker.py` | Sets custom worker class before `super().__init__()` |
| `AgriTrainer.fit()` | `trainer/trainer.py` | Per-turn branch: skips repeat+union, uses standard order |
| `AgriTrainer._pad_batch_for_training()` | `trainer/trainer.py` | Pads expanded batch to be divisible by mini_batch_size * n |

## Advantage Paths

### 1) Per-turn + GRPO (legacy path)

- `trainer.stepwise_advantage.enable = false`
- `algorithm.adv_estimator = grpo`
- Reward uses trajectory-level `score`
- Advantage is trajectory-level and broadcast to each turn/token

Optional rollout filtering for GRPO or stepwise PPO:
- `trainer.rollout_filter.enable = true`
- supported advantage modes:
  - `algorithm.adv_estimator = grpo`
  - `algorithm.adv_estimator = gae` with `trainer.stepwise_advantage.enable = true`
- first version supports only:
  - `metric = reward_variance`
  - `strategy = top_p`
  - `filter_type = largest`
- filtering score is per-`uid` in-group reward std over trajectory-level scores
- if a step filters out all samples, trainer skips that optimizer update and moves to the next batch
- trainer exits early if this happens for `max_consecutive_all_filtered_steps` consecutive steps

### 2) Per-turn + Stepwise GAE (new path)

- `trainer.stepwise_advantage.enable = true`
- `trainer.stepwise_advantage.mode = per_step`
- `trainer.stepwise_advantage.adv_estimator = gae`
- `trainer.stepwise_advantage.gamma = 1.0`
- `trainer.stepwise_advantage.lam = 0.97`
- `trainer.stepwise_advantage.whiten_advantages = true`
- `trainer.stepwise_advantage.reward_source = env_step`
- `trainer.stepwise_advantage.token_reward_shape = last_token_sparse`
- `trainer.stepwise_advantage.gae_scope = cross_turn`
- `algorithm.adv_estimator = gae` and critic enabled

Per-turn scalar definitions:
- `r_t`: step reward on the turn's last response token
- `V_t`: critic value on the turn's first response token position, which aligns with the pre-action state for that turn

Cross-turn GAE inside each trajectory (sorted by `step_idx`):
- `delta_t = r_t + gamma * V_{t+1} - V_t`
- `A_t = delta_t + gamma * lambda * A_{t+1}`
- `R_t = A_t + V_t`

If `trainer.stepwise_advantage.whiten_advantages = true`, the scalar `A_t`
values are whitened across valid rows before broadcasting. Then each scalar
`A_t/R_t` is broadcast to all valid response tokens of that turn.

### How Worker Expansion Works

1. `AgriToolAgentLoop._run_per_turn()` returns a single `AgentLoopOutput` with
   `extra_fields["per_turn_data"]` containing all turns' prompt_ids, response_ids,
   logprobs, and rewards.

2. `AgriAgentLoopWorkerBase._run_agent_loop()` detects `per_turn_data` and:
   - Creates a separate `AgentLoopOutput` for each turn
   - Calls `_agent_loop_postprocess()` on each (padding, position_ids, etc.)
   - Injects `data_source`, `reward_model`, `uid`, `trajectory_id`,
     `step_idx`, `step_num`, `is_last_step`, `extra_info` into each sample's
     `extra_fields`
   - Returns a list of `_InternalAgentLoopOutput` instead of a single one

3. `generate_sequences()` flattens nested lists and calls `_postprocess()` on
   the flat list, producing a single `DataProto` batch.

### Padding Timing in Stepwise/Filter Modes

In stepwise mode and GRPO-filter mode, random padding is delayed until after advantage computation:
- Before advantage: keep the true turn order (avoid duplicated rows changing temporal recursion).
- After advantage: apply `_pad_batch_for_training()` only for actor/critic update batch alignment.

### Validation

Per-turn mode skips `pad_dataproto_to_divisor` / `unpad_dataproto` during
validation because the worker expansion changes the output size relative to
the input.  Wandb table logging de-duplicates by `uid` so each trajectory
is logged once.
