# T1.2 Cross-Crop Trait Shift

## Purpose

Evaluate whether policies generalize across crops and whether crop-trait conditioning improves held-out crop performance.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | WOFOST-Gym through AgriManager |
| Dataset or generator source | `agrimanager/wofost-weather-pool` |
| Benchmark axis | 4ID, 8ID, 16ID crop coverage settings |
| Training split | `1600` rows per coverage setting |
| Validation split | `384` ID-crop rows plus `384` held-out-crop rows |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | crop coverage and ID/OOD crop group |

## Metrics

Report ID-crop reward, held-out-crop reward, OOD gap, final WSO, and per-crop breakdowns. Trait ablations should compare otherwise matched configs with and without crop-trait text.

## Claim Boundary

Strong results support cross-crop generalization within the WOFOST crop/weather pool. They do not prove generalization to crops or environments outside the released WOFOST support set.
