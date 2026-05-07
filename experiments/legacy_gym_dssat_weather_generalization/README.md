# Gym-DSSAT Level 1: Weather Generalization

This experiment family fixes one crop at a time and varies only the Gym-DSSAT
random weather episode.

Checked-in dataset configs live under `config/`. Generated parquet artifacts,
logs, and results stay local to this experiment directory.

## Dataset Configs

Each config sets `env_params.random_weather: true` and uses non-overlapping
seed ranges. The seed count controls how many distinct random weather
conditions are generated for each split.

- `config/weather_maize_llm_think.yaml`
- `config/weather_maize_llm_no_think.yaml`
- `config/weather_rice_llm_think.yaml`
- `config/weather_rice_llm_no_think.yaml`
- `config/weather_cotton_llm_think.yaml`
- `config/weather_cotton_llm_no_think.yaml`

## Fixed Maize Runs

```bash
bash experiments/legacy_gym_dssat_weather_generalization/run_gym_dssat_generalization_weather_maize_llm_think_train.sh
bash experiments/legacy_gym_dssat_weather_generalization/run_gym_dssat_generalization_weather_maize_llm_no_think_train.sh
```

## Fixed Rice Runs

```bash
bash experiments/legacy_gym_dssat_weather_generalization/run_gym_dssat_generalization_weather_rice_llm_think_train.sh
bash experiments/legacy_gym_dssat_weather_generalization/run_gym_dssat_generalization_weather_rice_llm_no_think_train.sh
```

## Fixed Cotton Runs

```bash
bash experiments/legacy_gym_dssat_weather_generalization/run_gym_dssat_generalization_weather_cotton_llm_think_train.sh
bash experiments/legacy_gym_dssat_weather_generalization/run_gym_dssat_generalization_weather_cotton_llm_no_think_train.sh
```
