# T2.4 Schema-Corruption Suite

## Purpose

Optional diagnostic suite for prompt/schema mismatch failures. These probes are not mixed into clean validation counts.

## Definition

| Field | Value |
| --- | --- |
| Simulator/source family | WOFOST-Gym through AgriManager |
| Dataset or generator source | parent T2 source datasets |
| Benchmark axis | observation/action/reward corruption probes |
| Training split | no additional training split by default |
| Validation split | separately labeled validation variants |
| Canonical seed | `sampling.generation_seed=42` unless the runnable config documents a different seed |
| Report label | corruption type plus parent validation label |

## Metrics

Report the parent task metric together with mismatch type, invalid-action rate when applicable, and degradation relative to the clean matching schema.

## Claim Boundary

These probes diagnose sensitivity to schema corruption. They should be reported separately from the main clean benchmark result.
