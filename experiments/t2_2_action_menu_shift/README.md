# T2.2 Action-Menu Shift

This experiment operationalizes action-schema shift with native WOFOST-Gym
management modes. The first runnable setting is Family A: train on full-coverage
menus and evaluate zero-shot on held-out restricted menus.

The checked-in dataset configuration uses the paper-scale Family A counts:
1600 train scenarios per source menu and 128 validation scenarios per menu. The
training script follows the standard LLM GRPO run shape: `Qwen/Qwen3-4B-Instruct-2507`,
4 rollouts per prompt, 2 GPUs, experiment-tracker logging, and named validation sets by
`action_menu`.

Generated parquet files are written to
`data/t22_family_a_no_think/` from the single dataset
config `config/t22_family_a_no_think.yaml`. Runtime logs, checkpoints,
and cache files are written under `${TMPDIR:-/tmp}/agrimanager_t22_run` by
default; override this with `T22_WORK_DIR=/path`.

## Family A Split

| Role | Menus | Dataset splits |
| --- | --- | --- |
| Train | `lnpkw-v0`, `lnpk-v0` | `train_lnpkw`, `train_lnpk` |
| Seen-menu checks | `lnpkw-v0`, `lnpk-v0` | `val_lnpkw`, `val_lnpk` |
| Held-out restricted checks | `lnw-v0`, `ln-v0`, `lw-v0` | `val_lnw`, `val_ln`, `val_lw` |

## What Was Implemented

- WOFOST prompts now derive their available actions from the native `env_id`.
- The parser maps text actions into the native action ids for each menu.
- The main reward is defined by `objective_id: profit_max`; resource costs are
  part of that objective, and no separate WOFOST-Gym `env_reward` is required.

## Commands

Train the Family A multi-menu no-think policy:

```bash
bash experiments/t2_2_action_menu_shift/run_t22_family_a_llm_no_think_train.sh
```

The train script checks the dataset files and builds them inline when needed.

## Metrics To Report

- Final WSO by target menu.
- Invalid action rate and rejected/no-op fallback rate.
- Action distribution by menu.
- Total N and total irrigation; add P/K instrumentation before strong P/K claims.
- Optional gap to separately trained target-menu specialists.
