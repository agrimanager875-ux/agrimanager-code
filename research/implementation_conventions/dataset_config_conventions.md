# Dataset Config Conventions

Dataset configs define which episode rows are generated or loaded for training and validation. They should not contain model checkpoint paths, tracker links, GPU resources, personal paths, or cluster-specific locations.

## Required Fields

| Field | Meaning |
| --- | --- |
| `dataset_id` | Stable artifact name. Generated files normally live under a concrete experiment data directory such as `data/weather_regime_chickpea_llm_no_think/`. |
| `env.name` | Environment adapter family, such as `wofost_gym`, `gym_dssat`, or `cycles_gym`. |
| `source.kind` | Source type: hosted weather pool, deterministic simulator generator, or explicit local parquet source. |
| `source.path` | Hosted dataset ID, generator name, or local path supplied by the user. |
| `sampling.generation_seed` | Deterministic sampling seed. Canonical benchmark configs use `42` unless explicitly documented otherwise. |
| `sampling.splits` | Physical split outputs such as `train`, `val_id`, `val_drought`, or `val_heldout_crops`. |
| `labels` | Experiment-axis labels copied into generated rows. |

## Hosted WOFOST Sources

| Dataset | Use |
| --- | --- |
| `agrimanager/wofost-weather-pool` | General 20-crop WOFOST weather source. |
| `agrimanager/wofost-weather-regime-pool` | Chickpea/potato weather-regime source. |
| `agrimanager/weather_pool_maize` | Maize-only WOFOST source for schema, action, reward, and cross-simulator WOFOST experiments. |

CycleGym and DSSAT-Gym rows are generated from external simulators unless fixed generated parquet rows are explicitly released. If fixed generated rows are released as datasets, they need the same dataset-card, Croissant, RAI, and hosting treatment as the WOFOST pools.

## Split Semantics

Use `train` for optimization rows. Use named validation splits for ID checks, OOD checks, schema variants, reward variants, simulator targets, and corruption probes. A separate `test` split should appear only when an experiment explicitly defines a final held-out test pass.

Each validation row should preserve:

- `dataset_split`: physical split/file name;
- `dataset_role`: `validation`;
- `validation_set`: reportable group name;
- the experiment-axis label used for aggregation.

For paired schema/action/reward experiments, multiple rendered validation files may share the same base WOFOST scenario rows. Preserve source row IDs or equivalent source metadata so paired comparisons can be verified.

## Generated Artifacts

A generated dataset directory should contain parquet split files and a manifest.
The manifest should record source, config, split files, row counts, seeds, labels, schema version, and any source row identifiers needed to reproduce the split.

Generated rows should include enough metadata to recover their source:

- dataset ID and split;
- source dataset or generator name;
- source split and source row index when available;
- crop, year, latitude, longitude, and weather-cache file for WOFOST pools;
- simulator, environment ID, seed, year window, price regime, or task setting for simulator-generated rows;
- validation labels used by training/evaluation callbacks.

## Leakage Controls

Do not mix validation rows into training. Do not report aggregate validation metrics without preserving named validation-set labels. Do not use local cache paths as durable identifiers; record source dataset IDs, source row indexes, cache filenames, and split labels instead.
