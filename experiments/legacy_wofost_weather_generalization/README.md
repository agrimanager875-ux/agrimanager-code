# WOFOST Level 1: Weather Generalization

This experiment family fixes one crop at a time and varies only the weather
episode.

Checked-in dataset configs live under `config/`. Generated parquet artifacts,
logs, and results stay local to this experiment directory.

## Previous NN Results

Previous NN artifacts from the experiment round before the next rerun on
`2026-04-23` were moved out of the active result paths:

- `results/previous_nn_train_20260423/`
- `results/previous_nn_eval_20260423/`

Previous NN datasets were also moved out of the active data paths:

- `data/weather_wheat_nn_without_traits_previous_dataset_20260423/`
- `data/weather_maize_nn_without_traits_previous_dataset_20260423/`

The active `results/nn_train/` and `results/nn_eval/` paths are intentionally
left for new NN runs. The active NN dataset paths are also left missing so the
next NN run rebuilds them from the checked-in dataset config. Do not move the
previous result directories back before launching with
`runtime.resume.mode=auto_latest`, unless the goal is to resume those old
checkpoints.

## Dataset Configs

This folder keeps the checked-in weather generalization configs, all without
crop traits:

- `config/weather_maize_llm_without_traits_think.yaml`
- `config/weather_maize_llm_without_traits_no_think.yaml`
- `config/weather_maize_nn_without_traits.yaml`
- `config/weather_wheat_llm_without_traits_think.yaml`
- `config/weather_wheat_llm_without_traits_no_think.yaml`
- `config/weather_wheat_nn_without_traits.yaml`

## Fixed Wheat Runs

This folder exposes fixed wheat experiments that follow the repository rule:

- one fixed run script per experiment
- no external override arguments

### LLM Train: Think

```bash
bash experiments/legacy_wofost_weather_generalization/run_wofost_generalization_weather_wheat_llm_think_train.sh
```

### LLM Train: No Think

```bash
bash experiments/legacy_wofost_weather_generalization/run_wofost_generalization_weather_wheat_llm_no_think_train.sh
```

### NN Train

```bash
bash experiments/legacy_wofost_weather_generalization/run_wofost_generalization_weather_wheat_nn_train_n1.sh
```

The NN train script builds the dataset first if the parquet files are missing.
It uses `agent.n_epochs=1` and `runtime.train_epochs=16` for the `3200`
scenario train split.

## Fixed Maize Runs

This folder also exposes the matching fixed maize experiments:

- one fixed run script per experiment
- no external override arguments

### LLM Train: Think

```bash
bash experiments/legacy_wofost_weather_generalization/run_wofost_generalization_weather_maize_llm_think_train.sh
```

### LLM Train: No Think

```bash
bash experiments/legacy_wofost_weather_generalization/run_wofost_generalization_weather_maize_llm_no_think_train.sh
```

### NN Train

```bash
bash experiments/legacy_wofost_weather_generalization/run_wofost_generalization_weather_maize_nn_train_n1.sh
```

The NN train script builds the dataset first if the parquet files are missing.
It uses `agent.n_epochs=1` and `runtime.train_epochs=16` for the `3200`
scenario train split.
