# Repository Rules

This page defines the repository-level storage and ownership rules for
AgriManager.

Read this together with the existing execution and integration docs:

- [architecture.md](./architecture.md)
- [experiment_conventions.md](./experiment_conventions.md)
- [environment_adapter_contract.md](./environment_adapter_contract.md)

## What belongs in git

Git stores the source of truth for code, configs, fixed experiment
definitions, and documentation.

Keep these in git:

- source code under `agrimanager/`, `integrations/`, `entrypoints/`, and
  other maintained modules
- dataset configs and experiment configs
- fixed `run_*.sh` definitions
- docs that explain architecture, contracts, and reproducible workflows

## Generated artifacts

Git stores the instructions that reproduce generated artifacts, not the
artifacts themselves.

Keep out of git:

- any `results/` output
- any log output
- any generated dataset or other generated data artifact
- checkpoints, caches, and other runtime byproducts
- local `sbatch_*.slurm` files used only to request cluster resources and
  launch one experiment
- analysis scripts
- plotting scripts
- generated figures and plots
- notebooks or scratch files used only for local inspection

Keep in git:

- the config files that define how generated data is built
- the fixed scripts that reproduce training, evaluation, and dataset builds
- documentation diagrams that are intentionally maintained as part of the
  docs

`sbatch_*.slurm` files are treated as user-specific cluster wrappers rather
than maintained experiment definitions. Their account, email, partition,
wall-clock time, and resource settings often differ across users and Delta
setups, so they should stay local instead of being versioned.

## Environment Ownership Boundary

AgriManager and external environment repositories have different roles.

External environment repositories live under:

```text
../AgriManagerExternal/{env_name}
```

AgriManager-side environment code lives under:

```text
agrimanager/env/{env_name}
integrations/{env_name}
```

Use this split:

| Change type | Home |
|---|---|
| environment simulator logic, core environment internals, domain assets, large upstream refactors | external environment repo |
| env adapters, prompt parsing, AgriManager wrappers, training/inference glue, dataset tooling, evaluation integration | this repository |

Within AgriManager:

- `agrimanager/env/{env_name}` holds the environment adapter contract and
  runtime-facing env code.
- `integrations/{env_name}` holds integration code for training,
  inference, and environment-specific tooling on the AgriManager side.

If an environment needs a large change to its original upstream codebase,
create and maintain that work in your own environment repo or fork first.
AgriManager should integrate with that repo rather than absorb the large
upstream modification here.

## Operational Rules From Existing Project Guidance

These repository rules work together with the existing project guidance:

- use the `agrimanager` Conda environment for project tasks
- keep `verl/` unchanged in this repository
- keep experiment logic in fixed `run_*.sh` scripts instead of local
  `sbatch_*.slurm` wrappers
- use external environment repos for environment-source changes and keep
  AgriManager integration code on the AgriManager side

## Quick Review Checklist

- this change adds source code, config, or maintained documentation
- this change avoids committing generated data, logs, results, and figures
- this change keeps environment-source modifications in the external repo
- this change keeps AgriManager integration code in the AgriManager repo
