# T2.2 Action-Menu Shift

## Purpose

Evaluate whether policies trained with one WOFOST action menu can adapt to related menus with missing or recombined management actions.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | WOFOST-Gym through AgriManager |
| Dataset or generator source | `agrimanager/weather_pool_maize` |
| Benchmark axis | maize action-menu variants |
| Training split | `1600` rows across training action menus |
| Validation split | `640` rows: `128` for each evaluated action menu |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | `action_menu` |

## Metrics

Report reward/profit by menu, invalid-action rate for LLM policies, action-use distribution, and final WSO.

## Claim Boundary

Strong results support menu-level action-schema robustness in WOFOST maize. They do not imply transfer to actions absent from the simulator or to real management recommendations.
