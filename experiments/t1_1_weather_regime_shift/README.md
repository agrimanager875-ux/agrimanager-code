# WOFOST Weather-Regime OOD

This experiment family trains one crop at a time on normal viable weather and
validates on interpretable weather-regime shifts: `drought`, `wet`, `hot`, and
`cold`.

The canonical experiment definition is documented in
`research/paper/experiment_cards/T1_1_weather_regime_shift.md`. The weather scenarios come
from the Hugging Face weather pool:

```text
agrimanager/wofost-weather-regime-pool
```

Generated parquet artifacts, logs, and results stay local to this experiment
directory. The weather pool itself is downloaded through the shared
`weather_pool.ensure_pool()` cache path.

## Dataset Configs

This folder keeps the checked-in weather-regime configs, all without crop
traits. `chickpea` and `potato` are separate datasets and separate training
runs.

- `config/weather_regime_chickpea_llm_without_traits_think.yaml`
- `config/weather_regime_chickpea_llm_without_traits_no_think.yaml`
- `config/weather_regime_chickpea_nn_without_traits.yaml`
- `config/weather_regime_potato_llm_without_traits_think.yaml`
- `config/weather_regime_potato_llm_without_traits_no_think.yaml`
- `config/weather_regime_potato_nn_without_traits.yaml`

Each config builds these parquet files:

- `train.parquet`: `1600` normal scenarios for that crop
- `val.parquet`: combined validation, `640` scenarios for that crop
- `val_id.parquet`: `128` held-out normal scenarios for that crop
- `val_drought.parquet`: `128` drought scenarios for that crop
- `val_wet.parquet`: `128` wet scenarios for that crop
- `val_hot.parquet`: `128` hot scenarios for that crop
- `val_cold.parquet`: `128` cold scenarios for that crop

There is no `test.parquet` for this experiment.

## Weather-Regime Definition

The Hugging Face pool is treated as the canonical materialized source for these
splits. It was constructed with equal-size, crop-specific weather-regime filters:
each validation regime keeps `128` scenarios, i.e. `20%` of the `640`-scenario
combined validation set for that crop.

For a crop-specific candidate weather set `S_c`, regime assignment uses the
WOFOST growing-season weather statistics:

- `season_rain(s)`: cumulative `RAIN` over the crop growing window.
- `season_temp(s)`: mean daily `TEMP` over the crop growing window.

With `p = 0.20`, the regime predicates are:

- `drought(s)`: `season_rain(s)` is in the lowest `p` fraction of `S_c`.
- `wet(s)`: `season_rain(s)` is in the highest `p` fraction of `S_c`.
- `hot(s)`: `season_temp(s)` is in the highest `p` fraction of `S_c`.
- `cold(s)`: `season_temp(s)` is in the lowest `p` fraction of `S_c`.
- `id/normal(s)`: none of the four extreme predicates is true.

The local dataset builder does not recompute these filters. It consumes the
already materialized source splits `val_id`, `val_drought`, `val_wet`,
`val_hot`, and `val_cold` and carries their `weather_regime` labels into the
generated parquet artifacts.

## Fixed Runs

The run scripts follow the repository rule: one fixed script per concrete run,
with no external override arguments.

### Chickpea

```bash
bash experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_chickpea_llm_think_train.sh
bash experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_chickpea_llm_no_think_train.sh
bash experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_chickpea_nn_train_n1.sh
bash experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_chickpea_nn_train_n8.sh
```

### Potato

```bash
bash experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_potato_llm_think_train.sh
bash experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_potato_llm_no_think_train.sh
bash experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_potato_nn_train_n1.sh
bash experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_potato_nn_train_n8.sh
```

The run scripts build the dataset first if the required parquet files are
missing or older than the checked-in config. The NN train scripts use
`runtime.train_epochs=8`; `n1` and `n8` variants set `agent.n_epochs` to `1`
and `8`, respectively.
