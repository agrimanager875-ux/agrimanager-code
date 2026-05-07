# T2.3 Reward-Formulation Shift

This experiment family implements the WOFOST-Gym reward-schema shift from
yield/profit/water objectives to held-out nutrient stewardship. The current
working setup is maize with WOFOST-Gym `lnpkw-v0`.

The paper-facing experiment definition is documented in
`research/paper/experiment_cards/T2_3_reward_formulation_shift.md`.

## Objective Reward Probe

Before training, run the scripted-policy probe under a CPU allocation. It derives
per-scenario `Y_ref` values and checks whether the four objective rewards induce
different but non-degenerate policy rankings.

```bash
conda run -n agrimanager python \
  experiments/t2_3_reward_formulation_shift/analysis/objective_reward_probe.py \
  --config experiments/t2_3_reward_formulation_shift/config/t23_reward_formulation_shift.yaml \
  --split val_nutrient_stewardship \
  --max-scenarios 8
```

Outputs are written to:

```text
experiments/t2_3_reward_formulation_shift/analysis/objective_reward_probe/
```

The key files are:

- `calibrated_y_ref.json`
- `objective_reward_scores.csv`
- `objective_reward_top_by_scenario.csv`
- `objective_reward_zero_high_input_ranks.csv`
- `objective_reward_probe_report.md`

The probe reads scenarios from the same unified dataset config used by training,
but runs scripted policies with `llm_mode=false`.

For the main T2.3 reward-formulation training run, disable rollout filtering:

```text
trainer.rollout_filter.enable=false
```

GRPO still normalizes advantages within each prompt group. Disabling the filter
avoids objective-dependent sampling bias from raw reward-variance filtering.

## Training Datasets and Runs

The formal CropGrowth setup uses one dataset config:

```text
config/t23_reward_formulation_shift.yaml
```

That config materializes the paper-facing train and validation split files. The
no-think and think training scripts read the same scenario/objective files and
set the response format through fixed runtime `env_config` overrides. Both train
on the three in-distribution objectives
(`yield_max`, `profit_max`, `water_stewardship`) and validate on all four
objectives, including held-out `nutrient_stewardship`:

```bash
bash experiments/t2_3_reward_formulation_shift/run_wofost_reward_formulation_shift_maize_llm_no_think_train.sh
bash experiments/t2_3_reward_formulation_shift/run_wofost_reward_formulation_shift_maize_llm_think_train.sh
```

## Yield-Only Reward-Diversity Ablation

The yield-only ablation tests whether reward-schema diversity during RL training
is necessary for prompt-conditioned objective switching. It trains only on
`train_yield_max.parquet` and validates on the same four reward-formulation
sets as the main T2.3 runs. The yield file is repeated in `data.train_files` so
the training row count stays close to the main yield/profit/water setup while
the reward form remains yield-only.

```bash
bash experiments/t2_3_reward_formulation_shift/run_wofost_reward_formulation_shift_maize_llm_yield_only_no_think_train.sh
bash experiments/t2_3_reward_formulation_shift/run_wofost_reward_formulation_shift_maize_llm_yield_only_think_train.sh
```

The config includes a conservative fallback `y_ref: 8500.0` so dataset
construction is not blocked while iterating on prompts. When a per-scenario
calibration map exists, dataset generation injects it into both
`env_config.y_ref` and `env_config.reward_params.y_ref`. The expected map path
is:

```text
experiments/t2_3_reward_formulation_shift/analysis/t23_y_ref/calibrated_y_ref.json
```

## Live Prompt Capture

To refresh the live prompt examples in the paper card:

```bash
conda run -n agrimanager python \
  experiments/t2_3_reward_formulation_shift/analysis/capture_t2_3_live_prompts.py
```

Outputs are written to:

```text
experiments/t2_3_reward_formulation_shift/analysis/t2_3_live_prompt_capture/
```
