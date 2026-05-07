# Experiments

This directory stores runnable experiment definitions and local outputs. The
directory names now follow the V2 paper-facing experiment-card structure when a
V2 card exists.

## Training Status

See [`experiments.md`](experiments.md) for the compact T1/T2/T3 training-script
status table and completed experiment-tracker run links.

## Dataset Sources

See [`wofost_gym_datasets.md`](wofost_gym_datasets.md) for the centralized
definition of the WOFOST-Gym weather-regime, 20-crop, and maize weather pools.

## Main V2 Experiments

| Card | Experiment directory | Experiment card |
| --- | --- | --- |
| T1.1 | `t1_1_weather_regime_shift/` | `research/paper/experiment_cards/T1_1_weather_regime_shift.md` |
| T1.2 | `t1_2_cross_crop_trait_shift/` | `research/paper/experiment_cards/T1_2_cross_crop_trait_shift.md` |
| T3.1 | `t3_1_cross_simulator_maize_transfer/` | `research/paper/experiment_cards/T3_1_cross_simulator_maize_transfer.md` |

## Appendix And Support

| Role | Experiment directory | Experiment card |
| --- | --- | --- |
| APP1 | `app1_wofost_variety_ood/` | `research/paper/experiment_cards/APP1_wofost_variety_ood_appendix.md` |
| T1.2 support | `support_t1_2_cross_crop_trait_selection/` | `research/paper/experiment_cards/T1_2_cross_crop_trait_shift.md` |

## Legacy

| Experiment directory | Experiment card |
| --- | --- |
| `legacy_wofost_weather_generalization/` | none |
| `legacy_gym_dssat_weather_generalization/` | none |

V2 experiments without paper-facing cards are intentionally absent for now. Add
the card first, then instantiate a concrete experiment directory, for example
`experiments/t1_1_weather_regime_shift/`.
