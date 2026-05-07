# Entrypoints

This folder contains the stable public execution entrypoints for AgriManager.

This README is the command reference for those scripts.

For repository-level rules on experiment wrappers and cluster launch files,
see [docs/experiment_conventions.md](../docs/experiment_conventions.md).
For the full documentation map, see [docs/README.md](../docs/README.md).

Use these scripts directly for ad hoc runs, or call them from fixed experiment
wrappers under `experiments/` and `smoke_tests/`.

## Layout

```text
entrypoints/
├── dataset/
│   └── build.sh
├── train/
│   ├── train.sh
│   ├── nn_train.sh
│   └── config/
├── eval/
│   ├── eval.sh
│   ├── nn_eval.sh
│   └── config/
└── tools/
    ├── merge.sh
    └── vllm_launch.sh
```

## Atomic Operations

The main atomic operations are:

- `entrypoints/dataset/build.sh`
- `entrypoints/train/train.sh`
- `entrypoints/eval/eval.sh`
- `entrypoints/train/nn_train.sh`
- `entrypoints/eval/nn_eval.sh`

These are the shared building blocks used by experiment-level `run_*.sh`
wrappers.

## Default Configs

Default runtime configs live next to the entrypoints:

- `entrypoints/train/config/`
- `entrypoints/eval/config/`

Current checked-in defaults include:

- `entrypoints/train/config/agri_grpo.yaml`
- `entrypoints/train/config/agri_ppo.yaml`
- `entrypoints/train/config/nn.yaml`
- `entrypoints/eval/config/default.yaml`
- `entrypoints/eval/config/nn.yaml`

The WOFOST default paths in these configs point to the canonical smoke-test
artifacts under `smoke_tests/wofost_gym/`.

## Usage

Examples:

```bash
# Build a dataset artifact
bash entrypoints/dataset/build.sh --config experiments/my_exp/config/my_dataset.yaml

# LLM training
bash entrypoints/train/train.sh

# LLM evaluation
bash entrypoints/eval/eval.sh

# Framework-native NN training
bash entrypoints/train/nn_train.sh --config-name nn

# Framework-native NN evaluation
bash entrypoints/eval/nn_eval.sh --config-name nn
```

## Overrides

The entrypoints are intentionally flexible. Training and evaluation can be
overridden directly from the command line.

Fixed `run_*.sh` wrappers under `experiments/` and `smoke_tests/` usually just
predefine these overrides for one reproducible run.

### LLM Train

`entrypoints/train/train.sh` uses Hydra `key=value` overrides:

```bash
bash entrypoints/train/train.sh \
  --config-name agri_ppo \
  data.train_files=experiments/my_exp/data/my_dataset/train.parquet \
  data.val_files=experiments/my_exp/data/my_dataset/val.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
  trainer.experiment_name=my_llm_train
```

### LLM Eval

`entrypoints/eval/eval.sh` also accepts Hydra overrides:

```bash
bash entrypoints/eval/eval.sh \
  data.inference_file=experiments/my_exp/data/my_dataset/test.parquet \
  model.config=agrimanager/model_interface/configs/vllm_offline/default.yaml \
  model.path=checkpoints/my_run/global_step_100/hf \
  output.dir=experiments/my_exp/results/my_eval
```

### NN Train

`entrypoints/train/nn_train.sh` runs the framework-native parquet-driven
NN trainer. The current default algorithm is SB3 PPO, selected by
`agent.type=PPO` in `entrypoints/train/config/nn.yaml`:

```bash
bash entrypoints/train/nn_train.sh \
  --config-name nn \
  data.train_files=experiments/my_exp/data/my_dataset/train.parquet \
  data.val_files=experiments/my_exp/data/my_dataset/val.parquet \
  runtime.train_epochs=2 \
  runtime.resume.mode=auto_latest \
  agent.exp_name=my_nn_train \
  output.save_folder=experiments/my_exp/results/nn_train/my_nn_train
```

Environment behavior comes from the dataset parquet, which is produced from the
dataset config. Change environment parameters in the dataset config and rebuild
the dataset; keep NN overrides for training/runtime settings only.
By default, NN validation uses `runtime.validation.evals_per_epoch=4` and
derives the timestep interval from the dataset epoch length.

### NN Eval

`entrypoints/eval/nn_eval.sh` evaluates a saved NN policy on a fixed parquet
split. For the current PPO backend, `agent.path` points to `agent.zip`:

```bash
bash entrypoints/eval/nn_eval.sh \
  --config-name nn \
  data.inference_file=experiments/my_exp/data/my_dataset/test.parquet \
  agent.path=experiments/my_exp/results/nn_train/my_nn_train/agent.zip \
  output.dir=experiments/my_exp/results/nn_eval
```
