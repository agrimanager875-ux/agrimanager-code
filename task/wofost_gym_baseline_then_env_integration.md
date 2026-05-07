# Task: WOFOST-Gym Baseline Then Environment Integration

This document outlines the current environment-integration task for team members.

## 1. Read the Documents First

Before starting any implementation work, please read the following documents in
full:

- [README.md](../README.md)
- [docs/README.md](../docs/README.md)
- [docs/architecture.md](../docs/architecture.md)
- [docs/experiment_conventions.md](../docs/experiment_conventions.md)
- [docs/repository_rules.md](../docs/repository_rules.md)
- [docs/environment_adapter_contract.md](../docs/environment_adapter_contract.md)
- [docs/wofost_dataset_contract.md](../docs/wofost_dataset_contract.md)

Before you start coding, please make sure you understand these boundaries:

- what code belongs in the AgriManager repository
- what code belongs in the external environment repository
- what may be committed to git
- what must stay out of git
- the fixed rules for `run_*.sh`, `sbatch_*.slurm`, and `smoke_tests/`
- the generic NN path: new environments should use `BaseNNEnvAdapter` and the
  shared `nn_train.sh` / `nn_eval.sh` entrypoints, not environment-specific NN
  trainers

## 2. Clean Install and Smoke Test

### Goal

Please show that your integration works in a clean setup, not only in a local
environment that was already configured before.

That means:

- cloning the project in a new location
- installing the environment from scratch
- running the smoke tests from scratch

At this stage, the first target is our existing `wofost_gym` path.

To help us validate consistently, please follow this order:

1. First run the existing `wofost_gym` smoke tests in a clean test environment.
2. Only after that should you extend the repository with your own environment
   integration and tests.

### Suggested Validation Steps

1. Please find a new working directory.
2. Please re-clone the AgriManager repository.
3. Please create a new Conda environment named `test_wofost_gym` using the
   reproducible install path.
4. Please activate that new environment.
5. Please run all current `wofost_gym` smoke tests first.
6. Only after those pass should you extend the repo and run your own
   environment smoke tests.

### Resource request requirement

Please request the compute resources needed to run the tests before you begin.

- If the task needs CPU resources, please request CPU resources before running.
- If the task needs GPU resources, please request interactive GPU resources before running.

Recommended command form:

```bash
git clone https://github.com/agrimanager875-ux/agrimanager-code.git
cd AgriManager

bash install_repro.sh test_wofost_gym
conda activate test_wofost_gym

bash smoke_tests/wofost_gym/run_build_datasets.sh
bash smoke_tests/wofost_gym/run_llm_train.sh
bash smoke_tests/wofost_gym/run_llm_eval_qwen25_3b_instruct.sh
bash smoke_tests/wofost_gym/run_nn_train.sh
bash smoke_tests/wofost_gym/run_nn_eval.sh
```

Please move on to your own environment integration tests only after this full
`wofost_gym` smoke-test set passes.

If a new environment package adds smoke tests, also run that environment's
`run_build_datasets.sh`, `run_llm_train.sh`, and `run_llm_eval_*.sh` scripts in
the later stage.

## 3. Environment Integration Guidance

Please integrate your environment according to the current AgriManager
documentation and repository boundaries.

### Current task scope

- First run the existing `wofost_gym` smoke tests successfully.
- After that, complete the minimum viable integration for your own
  environment.
- If your environment supports NN training, use the generic NN framework. Do
  not add a custom environment-specific NN training stack.
- Please do not bring in every past task right now.
- Additional tasks will be added later in `experiments/`.

### Current core objective

- First please show that the existing `wofost_gym` path installs cleanly and runs in
  the new `test_wofost_gym` environment.
- Then please integrate your environment without breaking the current AgriManager
  workflow.
- After integration, please add the corresponding smoke tests.
- After integration, please verify that the relevant endpoints work correctly.

### Code placement boundary

External environment repositories live under:

```text
../AgriManagerExternal/{env_name}
```

AgriManager-side environment and integration code lives under:

```text
agrimanager/env/{env_name}
```

For NN support, the AgriManager-side code should be:

```text
agrimanager/env/{env_name}/nn_adapter.py
```

This adapter must expose a `BaseNNEnvAdapter` implementation. It should build a
Gymnasium-compatible numeric environment from the parquet `env_config`. The
shared trainer will discover it automatically from `env_name`.

Please do **not** create new environment-specific NN entrypoints or trainer
folders such as:

```text
entrypoints/train/{env_name}_nn_train.sh
entrypoints/eval/{env_name}_nn_eval.sh
entrypoints/train/config/{env_name}_nn.yaml
entrypoints/eval/config/{env_name}_nn.yaml
integrations/{env_name}/train
integrations/{env_name}/inference
```

Those patterns are legacy for NN work. New NN integrations should use:

```text
entrypoints/train/nn_train.sh
entrypoints/eval/nn_eval.sh
entrypoints/train/config/nn.yaml
entrypoints/eval/config/nn.yaml
```

Please be explicit about the difference between:

- simulator logic, core internal environment code, and large upstream changes
  in the original environment repository
- AgriManager-side adapter code, prompt parsing, dataset tooling, and
  training/inference integration
- environment-specific NN conversion logic, which belongs in
  `agrimanager/env/{env_name}/nn_adapter.py`

If you need to make a large change to the original environment repository,
please:

- please do not place that large change directly in AgriManager
- please create your own repo or fork and make that change in the external
  environment repository
- please keep AgriManager focused on integrating that environment

### Git rules

Please do not commit the following to git:

- `results`
- `logs`
- generated data
- checkpoints
- analysis scripts
- plotting scripts
- figures

Please keep only:

- source code
- config
- fixed `run_*.sh`
- fixed `sbatch_*.slurm`
- fixed smoke tests
- necessary documentation

## 4. Smoke Tests and Endpoint Coverage

After the environment integration is complete, please add and verify at least
the following:

- dataset build path
- LLM train path
- LLM eval path
- if NN is supported, NN train and NN eval paths as well

Please make sure your smoke tests cover the endpoints you actually integrated.

Please keep the following in mind:

- please place smoke tests under a directory such as `smoke_tests/wofost_gym/`
- please keep them as fixed definitions, not free-form parameterized tools
- please first run the existing `wofost_gym` smoke tests in `test_wofost_gym`
- if NN is supported, please make your smoke-test scripts call the shared
  `entrypoints/train/nn_train.sh` and `entrypoints/eval/nn_eval.sh`
- please keep all environment semantics in the dataset config and materialized
  parquet `env_config`; NN train/eval overrides should only change runtime,
  algorithm, logging, checkpoint, and output settings
- please log both `LLM` and `NN` small-size training runs to the experiment tracker
- please then validate your new environment smoke tests in a clean environment
- please explain in the PR exactly which smoke tests you ran

## 5. Second Validation Pass

After your local integration and local smoke tests succeed, please run one more
independent validation pass:

1. Please find a new location.
2. Please re-clone the project.
3. Please reinstall the environment.
4. Please run all relevant smoke tests from scratch.
5. Please confirm that the full path from clone to smoke-test completion passes.

Please include this second validation pass as part of the process.

## 6. PR and Merge Requirements

Please open a PR only after all of the following are true:

- you have read the current documentation set
- you have passed the existing `wofost_gym` smoke tests in the
  `test_wofost_gym` environment
- your environment has been integrated according to the repository rules
- NN support, if added, goes through `BaseNNEnvAdapter` and the shared NN
  entrypoints
- your smoke tests have been written
- your smoke tests have passed in your working environment
- your smoke tests have passed again in a fresh clone/location

After you open the PR, please:

- ask the other two team members to run the same smoke tests
- wait for their verification
- merge to `main` only after all verification passes

## 7. Final Pre-Submission Checklist

Before submitting, please check the following:

- Did I put a large environment-source change into AgriManager by mistake?
- Did I place code in the wrong repo boundary?
- If I added NN support, did I use `BaseNNEnvAdapter` instead of creating
  environment-specific NN trainers or entrypoints?
- Did I keep environment semantics in the dataset config / parquet `env_config`
  instead of NN train/eval overrides?
- Did I commit `results`, `logs`, generated data, or figures?
- Did I add the required smoke tests?
- Did I perform a fresh clone/install/smoke-test validation pass?
- Did I describe the validation steps and results clearly in the PR?
