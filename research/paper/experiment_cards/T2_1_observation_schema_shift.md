# T2.1 Observation-Schema Shift

## Purpose

Evaluate whether policies remain effective when WOFOST observation fields are renamed, removed, compacted, or expanded while task dynamics stay fixed.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | WOFOST-Gym through AgriManager |
| Dataset or generator source | `agrimanager/weather_pool_maize` |
| Benchmark axis | maize |
| Training split | `1600` rows rendered with the full current schema |
| Validation split | `1024` rows: `128` for each schema condition |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | `observation_schema` |

## Metrics

Report reward/profit by schema condition, OOD drop relative to the native schema, final WSO, and failure modes caused by missing or renamed observation fields.

## Claim Boundary

Strong results support robustness to documented observation-schema shifts in the maize WOFOST task. They do not show robustness to arbitrary sensor sets or unvalidated observation semantics.
