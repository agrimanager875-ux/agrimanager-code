# CyclesGym Smoke Test

This is the canonical environment-level smoke test for `cycles_gym`.

It exercises the CyclesGym entrypoints through fixed `run_*.sh` scripts and
keeps everything local to this smoke-test directory:

- dataset configs in `config/`
- generated parquet artifacts in `data/`
- run logs in `logs/`
- model outputs and evaluation results in `results/`

## Entry Points Covered

- `entrypoints/dataset/build.sh`
- `entrypoints/train/train.sh`
- `entrypoints/eval/eval.sh`
- `entrypoints/train/nn_train.sh`
- `entrypoints/eval/nn_eval.sh`

## Fixed Run Scripts

Corn fertilization smoke suite:
- `run_build_datasets.sh`
- `run_llm_train.sh`
- `run_llm_eval_qwen3_4b_instruct.sh`
- `run_nn_train.sh`
- `run_nn_eval.sh`

Crop-planning smoke suite:
- `run_build_datasets_crop_planning.sh`
- `run_llm_train_crop_planning.sh`
- `run_llm_eval_qwen3_4b_instruct_crop_planning.sh`
- `run_nn_train_crop_planning.sh`
- `run_nn_eval_crop_planning.sh`

Cross-simulator maize crop-growth smoke suite:
- `run_build_datasets_crop_growth_yield.sh`
- `run_llm_train_crop_growth_yield_think.sh`
- `run_llm_train_crop_growth_yield_no_think.sh`
- `run_llm_eval_qwen3_4b_instruct_crop_growth_yield_think.sh`
- `run_llm_eval_qwen3_4b_instruct_crop_growth_yield_no_think.sh`

Each script is a complete smoke-test definition. It does not accept external
overrides. Running the script is the experiment.

## Typical Usage

Build both dataset variants:

```bash
bash smoke_tests/cycles_gym/run_build_datasets.sh
```

Run the LLM training smoke on a GPU node:

```bash
bash smoke_tests/cycles_gym/run_llm_train.sh
```

Run the LLM evaluation smoke on a GPU node:

```bash
bash smoke_tests/cycles_gym/run_llm_eval_qwen3_4b_instruct.sh
```

Run CyclesGym NN training and evaluation on a CPU node:

```bash
bash smoke_tests/cycles_gym/run_nn_train.sh
bash smoke_tests/cycles_gym/run_nn_eval.sh
```

These NN scripts use the shared framework-native entrypoints and discover the
CyclesGym numeric adapter from `agrimanager/env/cycles_gym/nn_adapter.py`.

Run the crop-planning smoke suite:

```bash
bash smoke_tests/cycles_gym/run_build_datasets_crop_planning.sh
bash smoke_tests/cycles_gym/run_nn_train_crop_planning.sh
bash smoke_tests/cycles_gym/run_nn_eval_crop_planning.sh
bash smoke_tests/cycles_gym/run_llm_train_crop_planning.sh
bash smoke_tests/cycles_gym/run_llm_eval_qwen3_4b_instruct_crop_planning.sh
```

Run the CycleGym maize crop-growth yield-only suite for the cross-simulator task:

```bash
bash smoke_tests/cycles_gym/run_build_datasets_crop_growth_yield.sh
bash smoke_tests/cycles_gym/run_llm_train_crop_growth_yield_think.sh
bash smoke_tests/cycles_gym/run_llm_train_crop_growth_yield_no_think.sh
bash smoke_tests/cycles_gym/run_llm_eval_qwen3_4b_instruct_crop_growth_yield_think.sh
bash smoke_tests/cycles_gym/run_llm_eval_qwen3_4b_instruct_crop_growth_yield_no_think.sh
```

These crop-growth scripts use `reward_mode: final_yield`: intermediate reward
is only the optional action-format bonus, and the terminal reward adds final
maize grain yield from the CycleGym corn simulation.
