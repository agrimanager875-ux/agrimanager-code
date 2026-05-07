# WOFOST-Gym Smoke Test

This is the canonical environment-level smoke test for WOFOST-Gym.

It exercises the WOFOST-Gym entrypoints through fixed `run_*.sh` scripts and
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

- `run_build_datasets.sh`
- `run_llm_train.sh`
- `run_llm_eval_qwen25_3b_instruct.sh`
- `run_nn_train.sh`
- `run_nn_eval.sh`

Each script is a complete smoke-test definition. It does not accept external
overrides. Running the script is the experiment.

## Typical Usage

Build both dataset variants:

```bash
bash smoke_tests/wofost_gym/run_build_datasets.sh
```

Run the LLM training smoke on a GPU node:

```bash
bash smoke_tests/wofost_gym/run_llm_train.sh
```

This smoke run performs one validation pass before training, runs 5 training
steps, and triggers a final validation pass on the last step.

Run the LLM evaluation smoke on a GPU node:

```bash
bash smoke_tests/wofost_gym/run_llm_eval_qwen25_3b_instruct.sh
```

Run framework-native NN training and evaluation:

```bash
bash smoke_tests/wofost_gym/run_nn_train.sh
bash smoke_tests/wofost_gym/run_nn_eval.sh
```
