# Dataset And Generator Sources

This file summarizes the dataset/generator plan used by the experiment cards. Detailed dataset provenance and external simulator citations are maintained in `docs/dataset_and_generator_sources.md`.

## Source Types

| Source | Treatment |
| --- | --- |
| `agrimanager/wofost-weather-pool` | Hosted WOFOST 20-crop weather-pool dataset. |
| `agrimanager/wofost-weather-regime-pool` | Hosted WOFOST weather-regime dataset for chickpea and potato. |
| `agrimanager/weather_pool_maize` | Hosted WOFOST maize-only weather-pool dataset. |
| CycleGym | External deterministic generator unless fixed generated parquets are explicitly released. |
| DSSAT-Gym | External deterministic generator unless fixed generated parquets are explicitly released. |

The three hosted WOFOST datasets require dataset cards, public hosting, Croissant metadata, RAI metadata, and reviewer-inspection samples when required by size. CycleGym and DSSAT-Gym are external simulator sources; AgriManager provides configs/adapters that use them.

## Shared Split Rules

| Rule | Standard |
| --- | --- |
| Training split | `train` |
| Held-out checks | Named validation groups |
| Canonical seed | `sampling.generation_seed=42` |
| Leakage control | Keep validation labels and source row IDs in generated rows/manifests. |
| Paired comparisons | Reuse base source scenarios when only schema/action/reward changes. |

## Experiment Plan

| Experiment | Source | Train | Validation |
| --- | --- | ---: | --- |
| T1.1 Weather-regime shift | `agrimanager/wofost-weather-regime-pool` | `1600` scenarios per crop | `640` per crop: `128` each for `val_id`, `val_drought`, `val_wet`, `val_hot`, `val_cold` |
| T1.2 Cross-crop trait shift | `agrimanager/wofost-weather-pool` | `1600` total per coverage setting | `768`: `384` ID-crop rows and `384` held-out-crop rows |
| T1.3 Price-regime shift | CycleGym deterministic generator | `1600` crop-planning episodes | `512`: `128` base scenarios rendered under four price regimes |
| T2.1 Observation-schema shift | `agrimanager/weather_pool_maize` | `1600` maize rows under the full schema | `1024`: `128` rows for each schema condition |
| T2.2 Action-menu shift | `agrimanager/weather_pool_maize` | `1600` maize rows across train menus | `640`: `128` rows for each action-menu condition |
| T2.3 Reward-formulation shift | `agrimanager/weather_pool_maize` | `1600` maize rows across seen reward objectives | `512`: `128` rows for each reward formulation |
| T3.1 Cross-simulator transfer | WOFOST hosted source plus DSSAT/CycleGym generators | `1600` rows per source simulator | `128` rows per target simulator |
| T3.2 Unified prompt-conditioned policy | WOFOST hosted source plus DSSAT/CycleGym generators | `3200` rendered rows across schema tuples | `128` rows per held-out validation tuple |

## T1.2 Crop Coverage

| Setting | ID crops | Train budget |
| --- | --- | ---: |
| `4ID` | cotton, millet, rapeseed, rice | `400` rows per crop |
| `8ID` | cotton, millet, rapeseed, rice, sugarbeet, fababean, maize, potato | `200` rows per crop |
| `16ID` | chickpea, cotton, cowpea, fababean, maize, millet, mungbean, potato, rapeseed, rice, sorghum, soybean, sugarbeet, sunflower, sweetpotato, wheat | `100` rows per crop |

## Reporting Labels

Generated rows should preserve labels for the experiment axis: `weather_regime`, `crop_regime`, `price_regime`, `observation_schema`, `action_menu`, `reward_formulation`, `simulator`, or schema tuple labels as appropriate.
