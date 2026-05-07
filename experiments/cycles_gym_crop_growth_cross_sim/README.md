# CycleGym Crop-Growth Cross-Simulator

This folder contains the non-smoke CycleGym-side scripts for the
cross-simulator maize crop-growth experiment.

The task is single-season maize/corn nitrogen management with a final-yield
objective:

```text
reward_t = format reward
reward_T = final maize grain yield + format reward
```

The dataset scale is:

- train: 3200 scenarios
- val: 128 scenarios
- test: 512 scenarios

The scripts cover the CycleGym training-source side only. The full
cross-simulator matrix still needs matching WOFOST-Gym and DSSAT-Gym final-yield
reward modes and target evaluation scripts.

## CPU Dataset Build

```bash
bash experiments/cycles_gym_crop_growth_cross_sim/run_build_datasets.sh
```

## GPU LLM Training

Default training uses 2 GPUs, GRPO `n=4`, `GEN_BATCH_SIZE=8`, and
`TOTAL_TRAINING_STEPS=200`. Override these from the shell if needed.

```bash
ray stop --force
bash experiments/cycles_gym_crop_growth_cross_sim/run_llm_train_crop_growth_yield_think.sh

ray stop --force
bash experiments/cycles_gym_crop_growth_cross_sim/run_llm_train_crop_growth_yield_no_think.sh
```

For a longer run:

```bash
TOTAL_TRAINING_STEPS=400 GEN_BATCH_SIZE=8 \
  bash experiments/cycles_gym_crop_growth_cross_sim/run_llm_train_crop_growth_yield_think.sh
```

## GPU Base-Model Evaluation

These eval scripts use the offline Qwen3-4B model config on the held-out
CycleGym test split.

```bash
ray stop --force
bash experiments/cycles_gym_crop_growth_cross_sim/run_llm_eval_qwen3_4b_instruct_crop_growth_yield_think.sh

ray stop --force
bash experiments/cycles_gym_crop_growth_cross_sim/run_llm_eval_qwen3_4b_instruct_crop_growth_yield_no_think.sh
```
