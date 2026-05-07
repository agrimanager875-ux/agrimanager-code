# Experiment Conventions

This page defines the repository conventions for `entrypoints/`,
`experiments/`, `smoke_tests/`, and `sbatch_*.slurm`.

Read this when you are:

- adding a new experiment or smoke test,
- deciding whether a change belongs in an entrypoint or a `run_*.sh`,
- reviewing whether a cluster launch file stays within scope.

Companion docs:

- [architecture.md](./architecture.md) explains why these layers exist.
- [environment_adapter_contract.md](./environment_adapter_contract.md)
  explains how env-specific code plugs into the framework.
- [entrypoints/README.md](../entrypoints/README.md) is the direct command
  reference for the stable public scripts.

## Choose the right layer

| If you need to change... | Put it in... | Why |
|---|---|---|
| stable build/train/eval behavior shared across runs | `entrypoints/` | this is the public execution API |
| one concrete dataset/model/output combination | `experiments/t1_1_weather_regime_shift/run_*.sh` or `smoke_tests/wofost_gym/run_*.sh` | each script is one reproducible run definition |
| resources, partition, wall clock, and launch command | `sbatch_*.slurm` | cluster wrappers should request resources only |

## The five atomic operations

| Operation | Script | Main inputs | Main outputs |
|---|---|---|---|
| Dataset build | `entrypoints/dataset/build.sh` | dataset config YAML | named parquet splits, `manifest.json` |
| LLM train | `entrypoints/train/train.sh` | `data.train_files`, `data.val_files` or `data.val_sets`, Hydra overrides | logs, checkpoints |
| LLM eval | `entrypoints/eval/eval.sh` | `data.inference_file`, model/runtime overrides | rollout results |
| NN train | `entrypoints/train/nn_train.sh` | `data.train_files`, `data.val_files` or `data.val_sets`, algorithm/runtime overrides | checkpoints, validation history |
| NN eval | `entrypoints/eval/nn_eval.sh` | `data.inference_file`, `agent.path` | rollout results, metrics |

Example invocations — [entrypoints/README.md](../entrypoints/README.md).

## Experiment folder layout

```text
experiments/t1_1_weather_regime_shift/        # or smoke_tests/wofost_gym/
├── README.md
├── config/           # dataset configs (env_name, scenario, sampling)
├── run_*.sh          # one per concrete run; fixed; calls one entrypoint
├── sbatch_*.slurm    # one per run_*.sh
├── data/weather_regime_chickpea_llm_no_think/    train.parquet · val.parquet · named val/test splits · manifest.json
├── results/
└── logs/
```

Canonical example: [`smoke_tests/wofost_gym/`](../smoke_tests/wofost_gym/).

## Rules

### `run_*.sh` is a fixed definition

One `run_*.sh` = one concrete run. It hardcodes: dataset config path,
parquet paths, model, log / checkpoint / result locations, and every
Hydra override.

Do **not** expose:

- `--dataset-config`, `--crop`, `--trait-schema`, `--run-name`
- any free-form override passthrough

Need a variant? Add a second `run_*.sh`.

### Named validation suites

OOD experiments should prefer `data.val_sets` over an anonymous list of
validation files:

```bash
"data.val_files=null" \
"data.validation_axis=weather_regime" \
"data.val_sets.id=${VAL_ID_FILE}" \
"data.val_sets.drought=${VAL_DROUGHT_FILE}" \
"data.val_sets.wet=${VAL_WET_FILE}"
```

`data.val_sets` is a logging and aggregation interface, not a request to run
validation sets serially. Training flattens the named files into one validation
dataloader and runs rollout once. `data.validation_axis` names the experiment's
primary OOD dimension and keeps tracker namespaces compact. For example,
`data.validation_axis=weather_regime` logs:

```text
val-core-weather_regime/all/...
val-core-weather_regime/drought/...
val-env-weather_regime/all/target_yield/mean
val-env-weather_regime/drought/target_yield/mean
```

Other labels stay in row metadata for debugging and offline analysis, but they
are not all expanded into separate tracker namespaces when `data.validation_axis` is
set.

Dataset configs may annotate splits with lightweight metadata:

```yaml
validation_axis: weather_regime
sampling:
  splits:
    val_drought:
      role: validation
      validation_set: drought
      crops: [potato]
      num_samples: 128
      labels:
        weather_regime: drought
```

The builder infers missing `role` and `validation_set` from split names such as
`val_drought`, injects those labels into trajectory metadata, and writes a
`manifest.json` next to the generated parquet files.

### `sbatch_*.slurm` does three things

Request resources, activate `agrimanager`, run one fixed `run_*.sh`.
Nothing else — no overrides, no crop selection, no array dispatch over
experiment variants.

In this repository, local `sbatch_*.slurm` files are treated as user-specific
cluster wrappers and should not be committed to Git. Their purpose is only to
request resources and launch one fixed experiment, while the exact account,
email, partition, wall-clock time, and resource choices can differ across
users and Delta setups.

## Review checklist

- `entrypoints/` changes improve shared build, train, or eval behavior.
- each `run_*.sh` names one concrete dataset, model or agent, and output set.
- each `sbatch_*.slurm` launches one fixed `run_*.sh`.
- the script layout under `experiments/` or `smoke_tests/` stays reproducible
  without extra CLI parameters.

## Typical workflow

1. Add a dataset config under `config/`.
2. Write `run_*.sh` with every override hardcoded.
3. Add `sbatch_*.slurm` if running on the cluster.
4. Build the dataset, run the experiment, check `results/` and `logs/`.
