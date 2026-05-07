# T1.1 Weather-Regime Shift

## Purpose

Evaluate WOFOST policy robustness when crop and task schema stay fixed but the weather distribution shifts.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | WOFOST-Gym through AgriManager |
| Dataset or generator source | `agrimanager/wofost-weather-regime-pool` |
| Benchmark axis | chickpea, potato |
| Training split | `1600` non-extreme scenarios per crop |
| Validation split | `128` scenarios per crop for each of `val_id`, `val_drought`, `val_wet`, `val_hot`, `val_cold` |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | `weather_regime` |

## Metrics

Report reward/profit on each named validation group, OOD drop relative to ID, and final WSO or yield as a secondary agronomic readout.

## Claim Boundary

Strong results support robustness to curated weather-regime shifts within the WOFOST task setup. They do not establish robustness to arbitrary climates, unseen management actions, or real farm deployment.
