# WOFOST-Gym Dataset Sources

This note centralizes the WOFOST-Gym weather-pool definitions used by the
experiment folders. The experiment configs consume materialized Hugging Face
sources; they do not recompute the source-level filtering logic at training time.

## Canonical Pools

| Pool | Hugging Face source | Main use |
| --- | --- | --- |
| Weather-regime pool | `agrimanager/wofost-weather-regime-pool` | T1.1 weather-regime shift. |
| 20-crop weather pool | `agrimanager/wofost-weather-pool` | T1.2 cross-crop trait shift and WOFOST crop-support runs. |
| Maize weather pool | `agrimanager/weather_pool_maize` | WOFOST maize schema, action, reward, and cross-simulator experiments. |

For the broader distinction between these hosted WOFOST datasets and
executable CycleGym/DSSAT-Gym generators, see
`docs/dataset_and_generator_sources.md`.

## Generation Seed

All active WOFOST-Gym experiment configs that consume these pools use
`generation_seed: 42` for deterministic train/validation sampling.

At the source-pool level:

- The 20-crop weather pool per-crop configs all use `generation_seed: 42`.
- The maize weather pool is derived from the same deterministic maize source:
  its `train` split matches the 20-crop `maize/train` shard, and its `val`
  split is the union of the 20-crop `maize/val` and `maize/test` shards.
- The weather-regime pool is consumed as a materialized Hugging Face source; the
  local T1.1 configs use `generation_seed: 42` when building train and named
  validation artifacts from that source.

## Weather-Regime Pool

The weather-regime pool is a materialized validation source for `chickpea` and
`potato`. Each crop has:

- `train`: `1600` normal weather scenarios.
- `val`: `640` combined validation scenarios.
- `val_id`, `val_drought`, `val_wet`, `val_hot`, `val_cold`: `128` scenarios each.

The named validation regimes are equal-size, crop-specific `20%` buckets. For a
crop-specific candidate weather set `S_c`, define:

- `season_rain(s)`: cumulative `RAIN` over the WOFOST growing season.
- `season_temp(s)`: mean daily `TEMP` over the WOFOST growing season.

With `p = 0.20`, the intended filtering predicates are:

```text
drought(s) = season_rain(s) is in the lowest  p fraction of S_c
wet(s)     = season_rain(s) is in the highest p fraction of S_c
hot(s)     = season_temp(s) is in the highest p fraction of S_c
cold(s)    = season_temp(s) is in the lowest  p fraction of S_c
normal(s)  = not (drought(s) or wet(s) or hot(s) or cold(s))
```

The local T1.1 dataset builder reads the already materialized `val_*` splits and
propagates their `weather_regime` labels into generated parquet artifacts.

## 20-Crop Weather Pool

The 20-crop weather pool is generated from the per-crop configs under
`integrations/wofost_gym/dataset_tools/weather_pool_configs/pool_crop_*.yaml`.
Each crop config uses:

- `generation_seed: 42`
- `year_range: [1984, 2019]`
- `min_dvs_threshold: 1.5`
- split sizes: `train=3200`, `val=128`, `test=512`

The crop set is:

```text
barley, chickpea, cotton, cowpea, fababean, groundnut, maize, millet,
mungbean, pigeonpea, potato, rapeseed, rice, seed_onion, sorghum, soybean,
sugarbeet, sunflower, sweetpotato, wheat
```

Each candidate scenario is a `(crop, year, latitude, longitude)` tuple. It is
kept only if a full no-action WOFOST rollout succeeds and satisfies all viability
checks:

```text
weather_complete(s) = rollout has no missing-weather/cache errors
phenology_ok(s)     = max_DVS(s) >= 1.5
yield_ok(s)         = max_WSO(s) > 0
keep(s)             = weather_complete(s) and phenology_ok(s) and yield_ok(s)
```

Failed scenarios are discarded and replaced before writing the per-crop pool
parquets.

## Maize Weather Pool

The maize weather pool uses the same WOFOST viability filter as the 20-crop
weather pool:

```text
weather_complete(s) and max_DVS(s) >= 1.5 and max_WSO(s) > 0
```

It is a maize-only materialized source with:

- `train`: `3200` maize scenarios.
- `val`: `640` maize scenarios.
- no separate materialized `test` split in the current source.

WOFOST maize experiments draw from this pool and then render the same base
weather rows under the experiment-specific observation schema, action menu,
reward formulation, or simulator-transfer setting.
