# WOFOST Dataset Contract

This page defines the WOFOST-specific dataset artifact contract.

Read this after [experiment_conventions.md](./experiment_conventions.md)
when you need the ownership rules and row-level shape for WOFOST parquet
artifacts.

Companion docs:

- [experiment_conventions.md](./experiment_conventions.md) covers the
  repository-wide run-script and artifact layout rules.
- [environment_adapter_contract.md](./environment_adapter_contract.md)
  covers the generic env adapter contract.

## Artifact flow

```text
experiment dataset config
  -> bash entrypoints/dataset/build.sh --config smoke_tests/wofost_gym/config/wofost_smoke_llm.yaml
  -> smoke_tests/wofost_gym/data/wofost_smoke_llm/train.parquet
  -> smoke_tests/wofost_gym/data/wofost_smoke_llm/val.parquet
  -> runtime entrypoints consume parquet only
```

## Ownership

Dataset config fields:

- `source`: input asset location, such as a weather pool
- `sampling`: deterministic scenario selection for each split
- `env`: immutable environment template baked into every parquet row

Runtime entrypoints:

- `entrypoints/train/train.sh`
- `entrypoints/eval/eval.sh`
- `entrypoints/train/nn_train.sh`
- `entrypoints/eval/nn_eval.sh`

These runtime entrypoints select parquet files, models or agents, and output
locations.

## Row contract

Each parquet row is a VERL-compatible record with:

- `extra_info.interaction_kwargs.env_config`: fully materialized environment config
- provenance fields in `env_config`:
  - `dataset_id`
  - `dataset_split`
  - `scenario_id`
  - `seed`

## Minimal row shape

```json
{
  "prompt": "...",
  "uid": "...",
  "extra_info": {
    "interaction_kwargs": {
      "env_config": {
        "env_name": "wofost_gym",
        "dataset_id": "example_dataset",
        "dataset_split": "train",
        "scenario_id": "scenario_0001",
        "seed": 7
      }
    }
  }
}
```

The exact row carries more task-specific fields, but the runtime path relies
on the embedded `env_config` to reconstruct environment state and provenance.

## Practical Rule

If environment behavior changes, create a new dataset config and rebuild the
dataset artifact. Training and inference should continue to consume parquet
artifacts directly.
