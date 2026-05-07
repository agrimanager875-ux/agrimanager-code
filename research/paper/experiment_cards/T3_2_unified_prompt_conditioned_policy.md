# T3.2 Unified Prompt-Conditioned Policy

## Purpose

Evaluate whether one policy can condition on simulator, observation, action, and reward schema tuples, then generalize to held-out combinations.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | WOFOST-Gym, DSSAT-Gym, CycleGym |
| Dataset or generator source | WOFOST uses `agrimanager/weather_pool_maize`; DSSAT and CycleGym use external deterministic generators |
| Benchmark axis | schema tuples |
| Training split | `3200` rendered rows across eight schema tuples |
| Validation split | `128` rows per held-out validation tuple |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | schema tuple, simulator, action menu, reward formulation |

## Metrics

Report reward/profit by tuple, in-family versus held-out tuple gap, target simulator breakdown, invalid-action rate, and final yield/WSO where available.

## Claim Boundary

Strong results support compositional conditioning over the documented schema tuples. They do not prove open-ended transfer to arbitrary simulators or schema definitions.
