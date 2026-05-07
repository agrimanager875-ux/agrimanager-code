# Research Reference Documents

This directory contains compact experiment-definition notes that support the AgriManager benchmark documentation. It should not contain generated outputs, paper drafts, metric tables, tracker links, figures, logs, checkpoints, or machine-specific paths.

## Contents

| Path | Purpose |
| --- | --- |
| `implementation_conventions/` | Conventions for dataset configs, prompts, and training scripts. |
| `paper/experiment_cards/` | Compact experiment cards defining benchmark axes, sources, splits, metrics, and claim boundaries. |
| `paper/experiment_cards/overview/` | Shared dataset, default-training, and reward definitions used by the cards. |
| `paper/experiment_cards/prompts/` | Prompt-surface reference notes. These are not generated prompt dumps. |

## Documentation Boundary

Runnable commands and setup instructions belong in `README.md`, `docs/`, `experiments/`, and `smoke_tests/`. These research notes explain benchmark intent and expected metadata semantics.

Do not commit generated prompt captures, tracker links, analysis tables, local cache paths, cluster output, checkpoints, or run logs here.

## Reproducibility Focus

The retained notes document:

- hosted WOFOST dataset sources and deterministic simulator-generator sources;
- train/validation split definitions and seeds;
- metadata that should be preserved in generated parquet rows and manifests;
- prompt, action, observation, and reward schema axes;
- experiment-level metrics and claim boundaries.
