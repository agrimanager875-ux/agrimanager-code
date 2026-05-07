# T3.1 Cross-Simulator Maize Transfer

## Purpose

Evaluate single-source policy transfer across WOFOST-Gym, DSSAT-Gym, and CycleGym maize-management tasks under an aligned crop-growth objective.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | WOFOST-Gym, DSSAT-Gym, CycleGym |
| Dataset or generator source | WOFOST uses `agrimanager/weather_pool_maize`; DSSAT and CycleGym use external deterministic generators |
| Benchmark axis | maize transfer targets |
| Training split | `1600` rows per source simulator |
| Validation split | `128` rows per target simulator |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | `simulator` |

## Metrics

Report target-simulator reward/profit, normalized score when anchors are available, final yield/biomass readouts, and action-pattern diagnostics.

## Claim Boundary

Strong results support transfer across the documented simulator interfaces and aligned objectives. They do not validate the simulators themselves or imply real-world agronomic transfer.
