# Training Script Conventions

This document records reproducibility expectations for `experiments/*/run_*.sh` scripts. It should not contain local paths, tracker links, private checkpoints, or cluster output.

## Script Boundary

Each run script should define one reproducible run. It may set a run name, dataset config, generated dataset directory, validation files, model settings, and Hydra overrides. It should not require edit-in-place personal paths.

Prefer environment variables for machine-specific paths:

| Variable | Use |
| --- | --- |
| `AGRIMANAGER_EXTERNAL_ROOT` | Optional root for external simulator checkouts. |
| `PCSE_METEO_CACHE_DIR` | Optional WOFOST/PCSE weather-cache directory. |
| `DSSAT_HOME` | Optional native DSSAT runtime location. |
| `DSSAT_PDI_PATH` | Optional DSSAT-PDI checkout/runtime location. |
| `CYCLES_GYM_CYCLESGYM_ROOT` | Optional CycleGym source checkout. |
| `CYCLES_GYM_CYCLES_ROOT` | Optional native Cycles runtime checkout. |

Scripts should use repo-relative defaults where possible.

## Dataset Build Pattern

Training scripts should either point to existing parquet files or build them from the referenced dataset config if files are missing or stale. The config, not the shell script, should define source pool, split labels, crops, generators, and seeds.

Validation should use named validation sets when the experiment reports multiple ID/OOD/schema groups.

## Training Families

| Policy family | Entry point | Config |
| --- | --- | --- |
| LLM GRPO | `entrypoints/train/train.sh` | `entrypoints/train/config/agri_grpo.yaml` |
| NN PPO | experiment NN wrapper or `entrypoints/train/nn_train.py` | `entrypoints/train/config/nn.yaml` |

Default LLM and NN settings are summarized in `research/paper/experiment_cards/overview/defaults.md`. Experiment scripts may override them only when the card or experiment README documents the reason.

## Evaluation

Evaluation scripts should load a specific checkpoint, validation parquet set, and environment adapter. They should preserve named validation labels in output metrics. Do not commit evaluation outputs, checkpoints, logs, or tracker links inside the source tree.

## Double-Blind Safety

Scripts and examples must avoid personal usernames, institutions, local scratch paths, private remotes, SSH URLs, and hosted run dashboards. If a path must be machine-specific, document it as an environment variable rather than a literal default.
