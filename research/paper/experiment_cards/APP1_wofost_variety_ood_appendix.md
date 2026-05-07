# APP1 WOFOST Same-Crop Variety OOD

## Purpose

Optional appendix experiment for same-crop variety generalization.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | WOFOST-Gym through AgriManager |
| Dataset or generator source | WOFOST weather pool plus crop-variety configs |
| Benchmark axis | held-out variety within the same crop |
| Training split | configured by the appendix experiment |
| Validation split | ID and held-out variety validation groups |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | variety group |

## Metrics

Report ID variety reward, OOD variety reward, OOD gap, and final WSO.

## Claim Boundary

This appendix can support claims about variety-level sensitivity within WOFOST. It should not be used as the main evidence for cross-crop generalization.
