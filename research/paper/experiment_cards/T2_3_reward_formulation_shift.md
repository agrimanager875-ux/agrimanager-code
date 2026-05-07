# T2.3 Reward-Formulation Shift

## Purpose

Evaluate whether policies can condition behavior on different management objectives while crop, simulator, and base weather source stay fixed.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | WOFOST-Gym through AgriManager |
| Dataset or generator source | `agrimanager/weather_pool_maize` |
| Benchmark axis | maize reward variants |
| Training split | `1600` rows across `yield_max`, `profit_max`, and `water_stewardship` |
| Validation split | `512` rows: `128` each for those three objectives plus held-out `nutrient_stewardship` |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | `reward_formulation` |

## Metrics

Report objective-specific reward, final WSO/yield, total N/P/K inputs, total irrigation, and OOD drop on the held-out nutrient objective.

## Claim Boundary

Strong results support conditioning on documented reward formulations in WOFOST maize. They do not prove that the chosen reward constants are agronomically optimal or transferable outside the simulator.
