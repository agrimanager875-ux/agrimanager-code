# WOFOST Level 2: Cross-Crop Generalization

This experiment family keeps the simulator and weather-pool rule fixed, and
varies the training crop coverage under a fixed cross-crop generalization setup.

Checked-in dataset configs live under `config/`. Generated parquet artifacts,
logs, and results stay local to this experiment directory.

## Previous NN Results

Previous NN artifacts from the experiment round before the next rerun on
`2026-04-23` were moved out of the active result paths:

- `results/previous_nn_train_20260423/`
- `results/previous_nn_eval_20260423/`

Previous NN datasets were also moved out of the active data paths:

- `data/cross_crop_4id_nn_traits_v1_23d_previous_dataset_20260423/`
- `data/cross_crop_8id_nn_traits_v1_23d_previous_dataset_20260423/`
- `data/cross_crop_16id_nn_traits_v1_23d_previous_dataset_20260423/`
- `data/cross_crop_16id_nn_from_old_llm_traits_v1_23d_previous_dataset_20260423/`

The active `results/nn_train/` and `results/nn_eval/` paths are intentionally
left for new NN runs. The active NN dataset paths are also left missing so the
next NN run rebuilds them from the checked-in dataset config. Do not move the
previous result directories back before launching with
`runtime.resume.mode=auto_latest`, unless the goal is to resume those old
checkpoints.

## Dataset Configs

This folder keeps the active fixed `23D`-trait configs for `4ID / 8ID / 16ID`:

- `config/cross_crop_4id_llm_traits_v1_23d_think.yaml`
- `config/cross_crop_4id_llm_traits_v1_23d_no_think.yaml`
- `config/cross_crop_4id_nn_traits_v1_23d.yaml`
- `config/cross_crop_8id_llm_traits_v1_23d_think.yaml`
- `config/cross_crop_8id_llm_traits_v1_23d_no_think.yaml`
- `config/cross_crop_8id_nn_traits_v1_23d.yaml`
- `config/cross_crop_16id_llm_traits_v1_23d_think.yaml`
- `config/cross_crop_16id_llm_traits_v1_23d_no_think.yaml`
- `config/cross_crop_16id_nn_traits_v1_23d.yaml`

## Fixed OOD Crops

The shared fixed OOD evaluation crops are:

- `barley`
- `groundnut`
- `pigeonpea`
- `seed_onion`

## Budgets

- Train: fixed total budget `1600`
- Validation: fixed total budget `768`
- Test: fixed `128` per crop

Train crop budgets:

- `4ID = 4 x 400`
- `8ID = 8 x 200`
- `16ID = 16 x 100`

Validation crop budgets:

- `4ID`: `4 x 96` ID + `4 x 96` OOD = `768`
- `8ID`: `8 x 48` ID + `4 x 96` OOD = `768`
- `16ID`: `16 x 24` ID + `4 x 96` OOD = `768`

Validation metrics are grouped by crop, not by the aggregate `heldout`
bucket. The experiment tracker should show keys such as
`val-core-crop/maize/final_profit_mean` and
`val-env-crop/chickpea/final_profit_mean`; `id_crops` and `heldout_crops`
remain validation-set labels for filtering.

## Fixed Runs

This folder exposes fixed cross-crop experiments that follow the repository rule:

- one fixed run script per experiment
- no external override arguments

### 4ID

```bash
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_4id_llm_think_train.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_4id_llm_no_think_train.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_4id_nn_train_n1.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_4id_nn_train_n8.sh
```

### 8ID

```bash
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_8id_llm_think_train.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_8id_llm_no_think_train.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_8id_nn_train_n1.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_8id_nn_train_n8.sh
```

### 16ID

```bash
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_16id_llm_think_train.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_16id_llm_no_think_train.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_16id_nn_train_n1.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_16id_nn_train_n8.sh
```

The run scripts build the dataset first if the parquet files are missing. All
active LLM train scripts use `trainer.total_epochs=2`. All active NN train
scripts use `runtime.train_epochs=8`.

Both NN settings are retained. `nn_train_n1` uses `agent.n_epochs=1`, and
`nn_train_n8` uses `agent.n_epochs=8`; this controls PPO optimization passes per
rollout buffer, not the dataset-level train epoch count.
