# Experiment Definition Notes

This folder contains compact benchmark-definition notes. It should not contain runnable configs, generated outputs, figures, metric tables, tracker links, or paper-draft material.

## Contents

| Path | Purpose |
| --- | --- |
| `experiment_cards/` | One compact card per benchmark experiment. |
| `experiment_cards/overview/datasets.md` | Shared dataset and generator source plan. |
| `experiment_cards/overview/defaults.md` | Shared training, evaluation, and WOFOST defaults. |
| `experiment_cards/overview/rewards.md` | Reward objective definitions and experiment mappings. |
| `experiment_cards/prompts/` | Prompt-surface reference notes. |

The runnable source of truth for commands remains under `experiments/` and `smoke_tests/`. These notes define intent, split semantics, and reporting boundaries.
