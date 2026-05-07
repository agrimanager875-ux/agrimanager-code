# Gym-DSSAT Smoke Tests

Commands assume you are running from the repository root after completing the
standard AgriManager install and the optional DSSAT-Gym setup in
`docs/gym_dssat_setup.md`. Supported crops are `maize`, `rice`, and `cotton`;
smoke scripts default to `maize` if no crop is passed.

## Smoke Tests

```bash
source smoke_tests/gym_dssat/_activate_spack.sh

# Replace maize with rice or cotton as needed.
CROP=maize

bash smoke_tests/gym_dssat/run_build_datasets.sh "$CROP"
bash smoke_tests/gym_dssat/run_llm_train.sh "$CROP"
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh "$CROP"
```

Run all three crops manually:

```bash
bash smoke_tests/gym_dssat/run_build_datasets.sh maize
bash smoke_tests/gym_dssat/run_llm_train.sh maize
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh maize

bash smoke_tests/gym_dssat/run_build_datasets.sh rice
bash smoke_tests/gym_dssat/run_llm_train.sh rice
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh rice

bash smoke_tests/gym_dssat/run_build_datasets.sh cotton
bash smoke_tests/gym_dssat/run_llm_train.sh cotton
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh cotton
```

## Outputs

Smoke outputs: `smoke_tests/gym_dssat/{data,logs,results}/`

## Common Fixes

DSSAT runtime not found: complete `docs/gym_dssat_setup.md`, set
`DSSAT_GYM_PATH`, and rerun:

```bash
source smoke_tests/gym_dssat/_activate_spack.sh
```

Ray socket path too long:

```bash
export RAY_LOG_DIR=$HOME/ray
```

LLM eval says required arguments are missing: run the wrapper, for example:

```bash
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh maize
```
