# Adapter: Connecting AgriManager to VERL

This module is the bridge between AgriManager environments and [VERL](https://github.com/volcengine/verl)'s RL training framework. It implements the interfaces VERL expects so that any AgriManager environment can be used for PPO/GRPO training.

## Training Flow

```
train.sh
  └─ main_ppo.py (AgriTaskRunner)
       └─ AgriTrainer.fit()  ← training loop
            │
            ├─ Rollout (generate_sequences)
            │    └─ AgriAgentLoopManager
            │         └─ AgriAgentLoopWorker (per GPU)
            │              └─ AgriToolAgentLoop.run()
            │                   ├─ LLM generates action
            │                   ├─ AgriInteraction.generate_response()
            │                   │    └─ env.step(action) → obs, reward, done
            │                   └─ Loop until done or max turns
            │
            ├─ Reward (compute_score)
            │    └─ Aggregates turn_scores from interaction
            │
            ├─ Advantage (GRPO or stepwise GAE)
            │
            └─ Actor update (PPO loss)
```

### Key steps

1. **Rollout**: `AgriAgentLoopManager` dispatches prompts to workers. Each worker runs `AgriToolAgentLoop`, which generates LLM actions and feeds them to the environment through `AgriInteraction`. The environment returns observations, rewards, and done flags.

2. **Reward**: `compute_score()` receives `turn_scores` (per-step rewards from env) and only aggregates them. The adapter contract is generic: environments decide what each turn reward means, and the reward function just returns the summed trajectory score from those turn rewards plus the current turn's `step_reward`.

3. **Advantage & Update**:
- GRPO path: computes trajectory-level GRPO advantages.
- Optional rollout filter path: before policy update, keeps prompt groups with high in-group reward variance (`top_p + largest`) to focus updates on informative groups.
- Stepwise path: computes cross-turn GAE (`per_step`) using each turn's action-start value and step reward, then broadcasts scalar advantage/return to that turn's response tokens.
- Actor/critic are updated with standard PPO losses.

## Reward Protocol

The framework uses a single reward interface for all environments:

- `env.step(action)` returns one scalar `reward` on every turn.
- Trainer and reward adapter only consume this per-turn reward stream.
- There is no separate framework-level "trajectory reward channel".

If an environment conceptually uses trajectory reward, it should encode it
through the same per-turn stream:

- Early turns: return `0.0` or any environment-defined shaping / format reward.
- Final turn: return the trajectory reward, optionally plus any final-step shaping reward.

The adapter then applies a fixed generic aggregation:

- `turn_scores = [r_0, r_1, ..., r_T]`
- `step_reward = turn_scores[step_idx]`
- `score = traj_score = sum(turn_scores)`

Examples:

- Dense per-turn reward: `[0.1, 0.2, 0.3]` gives `traj_score = 0.6`.
- Sparse trajectory reward: `[0.0, 0.0, 0.8]` gives `traj_score = 0.8`, and only the last turn has non-zero `step_reward`.

`turn_metrics` and `trajectory_metrics` are optional logging / diagnostic fields.
They are not part of the reward contract and should not be required to recover
the training reward.

### Per-turn training mode

When `trainer.per_turn_training: True`, each turn in a multi-turn episode is treated as an independent training sample:
- Agent loop gives each turn a fresh prompt (system + current observation)
- Each turn gets its own `prompt_ids` / `response_ids`
- Worker injects `trajectory_id`, `step_idx`, `step_num`, `is_last_step` into each sample
- Advantage can be:
  - GRPO trajectory-level advantage, broadcast to each turn
  - Stepwise cross-turn GAE advantage, then broadcast to each turn tokens
- This allows training on individual decision steps rather than full episodes

## Advantage Path Switch

Use the following config switch:

```yaml
trainer:
  per_turn_training: true
  stepwise_advantage:
    enable: true          # false -> keep GRPO path
    mode: per_step
    adv_estimator: gae
    reward_source: env_step
    token_reward_shape: last_token_sparse
    gae_scope: cross_turn
algorithm:
  adv_estimator: gae      # required when stepwise_advantage.enable=true
```

- `stepwise_advantage.enable=false`: keeps existing per-turn + GRPO behavior.
- `stepwise_advantage.enable=true`: uses `env.step` reward per turn + cross-turn GAE.

## Reward Variance Filter (Optional)

For GRPO or stepwise PPO training, you can enable RAGEN-style rollout filtering:

```yaml
trainer:
  rollout_filter:
    enable: true
    metric: reward_variance
    strategy: top_p
    value: 0.9
    filter_type: largest
    include_zero: true
    max_consecutive_all_filtered_steps: 10
    score_key: traj_score
    zero_eps: 1e-10
```

- Scope (first version): `algorithm.adv_estimator=grpo`, or `algorithm.adv_estimator=gae` with `trainer.stepwise_advantage.enable=true`.
- Selection: compute in-group reward std per `uid`, softmax over groups, keep top nucleus mass `value`.
- Empty filtered batch: skip that optimizer update and continue to the next batch.
- Early stop: if batches are fully filtered out for `max_consecutive_all_filtered_steps` consecutive steps, training exits early.
- Not included in first version: `top_k/top_k_abs/min_p`, entropy/length metrics.

## Module Structure

```
adapter/
├── configs/
│   ├── agri_agent_loop.yaml          # Registers AgriToolAgentLoop
│   └── agri_interaction_config.yaml  # Registers AgriInteraction
├── agent_loop/
│   ├── agent_loop.py                 # AgriToolAgentLoop (LLM ↔ env loop)
│   └── worker.py                     # AgriAgentLoopManager + Worker (per-turn expansion)
├── interactions/
│   └── agri_interaction.py           # AgriInteraction (BaseInteraction → BaseEnv)
├── reward/
│   └── agri_reward.py                # compute_score()
├── trainer/
│   ├── main_ppo.py                   # Entry point (AgriTaskRunner)
│   └── trainer.py                    # AgriTrainer (extended RayPPOTrainer)
└── utils.py                          # Logging utilities
```

## How VERL Connects to AgriManager Environments

VERL knows nothing about AgriManager environments directly. The connection happens through two config files and three adapter classes:

### 1. Interaction Config (`configs/agri_interaction_config.yaml`)

```yaml
interaction:
  - name: "agri"
    class_name: "agrimanager.adapter.interactions.agri_interaction.AgriInteraction"
    config:
      default_env_type: wofost_gym
```

This tells VERL: "when the dataset says `interaction_name=agri`, use `AgriInteraction`".

### 2. Agent Loop Config (`configs/agri_agent_loop.yaml`)

```yaml
- name: agri_tool_agent
  _target_: agrimanager.adapter.agent_loop.agent_loop.AgriToolAgentLoop
```

This tells VERL: "when the dataset says `agent_name=agri_tool_agent`, use `AgriToolAgentLoop`".

### 3. AgriInteraction (`interactions/agri_interaction.py`)

Implements VERL's `BaseInteraction` interface:

| VERL method | What it does |
|---|---|
| `start_interaction()` | Calls `create_environment()` + `env.reset()` |
| `generate_response(messages)` | Parses the last assistant message, calls `env.step()` |
| `get_current_observation()` | Returns current env observation |
| `finalize_interaction()` | Calls `env.close()` |

This is the only class that touches AgriManager's `BaseEnv` API.

## Adding a New Environment

To train with a new environment (e.g., `my_env`), you need to:

### 1. Implement the environment

Create `agrimanager/env/my_env/` with:

```python
# env.py
from agrimanager.env.base import BaseEnv, BaseEnvConfig

class MyEnvConfig(BaseEnvConfig):
    my_param: str = "default"

class MyEnv(BaseEnv):
    def reset(self) -> tuple[str, dict]:
        """Return (observation_text, info_dict)"""
        ...

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Parse LLM action, step simulator, return (obs, reward, done, info).

        Reward protocol:
        - return one scalar reward on every turn
        - if the environment uses trajectory reward, inject it on the final turn

        info may optionally contain:
        - turn_metrics: dict of per-step metrics
        - trajectory_metrics: dict of episode-level metrics (when done=True)
        """
        ...

    def system_prompt(self) -> str:
        """Return the system prompt that tells the LLM how to interact."""
        ...
```

The `action` string in `step()` is the raw LLM output. Your environment is responsible for parsing it (e.g., extracting `<answer>...</answer>` tags).

### 2. Register the environment

Add `__init__.py` in your env directory so `create_environment("my_env", config)` can discover it. Follow the pattern in `agrimanager/env/wofost_gym/`.

### 3. Update the interaction config (if needed)

If your environment uses a different interaction name, add it to `agri_interaction_config.yaml`:

```yaml
interaction:
  - name: "agri"
    class_name: "agrimanager.adapter.interactions.agri_interaction.AgriInteraction"
    config:
      default_env_type: wofost_gym
  - name: "my_env_interaction"
    class_name: "agrimanager.adapter.interactions.agri_interaction.AgriInteraction"
    config:
      default_env_type: my_env
```

In most cases, `AgriInteraction` works for any environment since it delegates to `create_environment()`. You only need a custom interaction class if your environment requires a different LLM-to-action parsing flow.

### 4. Prepare the dataset

Generate a parquet dataset where each row contains the prompt and env config:

```python
# Each row needs at minimum:
{
    "data_source": "wofost_gym/wheat_test",
    "prompt": [{"role": "system", "content": "..."}],
    "extra_info": {
        "interaction_kwargs": {
            "name": "agri",           # matches interaction config
            "env_config": {
                "env_name": "my_env", # passed to create_environment()
                "turn_num": 10,       # number of interaction turns
                # ... any other env-specific params
            }
        }
    }
}
```

**Required `env_config` fields:**

| Field | Used by | Description |
|---|---|---|
| `turn_num` | `AgriToolAgentLoop`, `compute_score` | Max interaction turns per episode. Agent loop uses it to bound the rollout loop; reward function records it as metadata and falls back to the observed episode length if it is missing. |

All other fields (`env_name`, `seed`, `llm_mode`, etc.) have defaults and are optional.

### 5. Train

```bash
TRAIN_FILE=smoke_tests/wofost_gym/data/wofost_smoke_llm/train.parquet
VAL_FILE=smoke_tests/wofost_gym/data/wofost_smoke_llm/val.parquet

bash entrypoints/train/train.sh \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    trainer.experiment_name=my_env-grpo
```

No changes to the adapter code are needed -- the adapter is environment-agnostic.
