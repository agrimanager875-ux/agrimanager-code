# T1.3 Price-Regime Shift

## Purpose

Evaluate whether CycleGym crop-planning policies adapt to held-out crop-price regimes.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | CycleGym plus native Cycles simulator |
| Dataset or generator source | external deterministic CycleGym generator |
| Benchmark axis | crop-planning tasks |
| Training split | `1600` episodes: two locations, four year windows, and 200 seeds per combination |
| Validation split | `512` paired rows: `128` base scenarios rendered under four price regimes |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | `price_regime` |

## Metrics

Report cumulative gross revenue by price regime, crop-choice distribution, and OOD drop relative to the ID price regime.

## Claim Boundary

Strong results support price-conditioned decision adaptation in the configured CycleGym planning task. They do not establish validity of the native Cycles simulator or real market behavior.
