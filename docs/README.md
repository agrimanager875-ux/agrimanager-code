# Documentation Guide

This folder is organized by reader task.

If you are new to the repository, read in this order:

1. [README.md](../README.md)
2. [architecture.md](./architecture.md)
3. [experiment_conventions.md](./experiment_conventions.md)

After that, jump to the task-specific references below.

## Which doc should I read?

| Question | Read |
|---|---|
| What are the main design boundaries in AgriManager? | [architecture.md](./architecture.md) |
| Where should a change live: `entrypoints/`, `experiments/`, or `sbatch_*.slurm`? | [architecture.md](./architecture.md) |
| How should I structure a new experiment or smoke test? | [experiment_conventions.md](./experiment_conventions.md) |
| What are the fixed rules for `run_*.sh` and `sbatch_*.slurm`? | [experiment_conventions.md](./experiment_conventions.md) |
| What are the repository rules for git-tracked artifacts and env ownership? | [repository_rules.md](./repository_rules.md) |
| Which hosted datasets and simulator generators does AgriManager use? | [dataset_and_generator_sources.md](./dataset_and_generator_sources.md) |
| How are scripted baseline policies defined and bounded? | [baseline_policy_definitions.md](./baseline_policy_definitions.md) |
| Which upstream sources still need formal paper citations? | [upstream_citations.md](./upstream_citations.md) |
| How do I set up optional DSSAT-Gym/DSSAT-PDI runtime support? | [gym_dssat_setup.md](./gym_dssat_setup.md) |
| How do I add a new environment under `agrimanager/env/`? | [environment_adapter_contract.md](./environment_adapter_contract.md) |
| What dataset contract does WOFOST-Gym rely on? | [wofost_dataset_contract.md](./wofost_dataset_contract.md) |
| How does the reward variance filter work? | [reward_variance_filter.md](./reward_variance_filter.md) |
| I just need the direct CLI usage for stable scripts. | [entrypoints/README.md](../entrypoints/README.md) |

## How the docs are split

- Concept docs explain why the repository is structured the way it is.
  - [architecture.md](./architecture.md)
- Convention docs explain how reproducible runs should be laid out.
  - [experiment_conventions.md](./experiment_conventions.md)
- Repository rules define what belongs in git and how ownership is split.
  - [repository_rules.md](./repository_rules.md)
- Dataset/source docs separate hosted static datasets from executable
  simulator generators and document how each source is used.
  - [dataset_and_generator_sources.md](./dataset_and_generator_sources.md)
- Baseline docs define scripted policy roles and claim boundaries.
  - [baseline_policy_definitions.md](./baseline_policy_definitions.md)
- Citation draft docs track upstream citation work that must be finalized in
  the paper bibliography.
  - [upstream_citations.md](./upstream_citations.md)
- Optional runtime setup docs cover slow external dependencies that are not
  part of the default install path.
  - [gym_dssat_setup.md](./gym_dssat_setup.md)
- Extension docs explain how to plug in a new subsystem.
  - [environment_adapter_contract.md](./environment_adapter_contract.md)
- Feature and component references explain one narrow area in detail.
  - [wofost_dataset_contract.md](./wofost_dataset_contract.md)
  - [reward_variance_filter.md](./reward_variance_filter.md)

## Quick rule of thumb

- If you are deciding repository boundaries, start with [architecture.md](./architecture.md).
- If you are writing or reviewing scripts under `experiments/`, `smoke_tests/`, or cluster launch files, use [experiment_conventions.md](./experiment_conventions.md).
- If you are deciding whether an artifact or environment change belongs in this repo at all, use [repository_rules.md](./repository_rules.md).
- If you are touching an environment adapter such as `agrimanager/env/wofost_gym/`, use [environment_adapter_contract.md](./environment_adapter_contract.md).
