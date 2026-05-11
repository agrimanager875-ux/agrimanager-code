# Baseline Policy Definitions

This page defines scripted baseline policies used by AgriManager evaluation.
The goal is to make baseline claims explicit: a crop-management script should
not be described as agronomic expertise unless its rules come from published
guidance or are clearly labeled as a simulator-only diagnostic.

## Baseline Roles

| Role | Name | Use |
| --- | --- | --- |
| Lower anchor | `no_action` | Stable do-nothing baseline for profit and yield tables. |
| Random reference | `random` | Action-space sanity check; not an agronomic policy. |
| Source-backed scripted reference | `ag_heuristic` | Defensible scripted agronomic reference policy; not an upper bound. |
| Simulator attainable reference | `profit_grid_oracle` | Optional normalization or diagnostic reference only; not a fair competing method unless selected only on ID/train scenarios. |
| Yield-potential diagnostic | `ppv0_yield_potential` | Potential-production yield reference only; not a profit upper bound. |
| Stress-test script | `crop_blind_high_input` | Legacy high-input/stage-budgeted behavior; diagnostic only. |

Do not use the legacy stage-budgeted behavior as an agronomic expert or as the
`1.0` normalized anchor for profit results. It is crop-blind and input-heavy,
so it is useful only as a stress test.

## Current Implementation

The current source-informed scripted rollout is implemented in:

```text
agrimanager/rollout/inference/ag_heuristic_rollout.py
```

Supported modes:

- `no_action`: uses the native zero/no-op action for each environment.
- `ag_heuristic`: dispatches to a conservative scripted policy.

The `ag_heuristic` implementation is simulator-specific:

- WOFOST-Gym: uses a development-stage rule over the active action menu. It
  applies early phosphorus, potassium, and nitrogen before mid-stage nitrogen,
  triggers irrigation only when the water rule says to irrigate, allows late
  small nitrogen only for `yield_max`, and projects to the nearest available
  menu action.
- DSSAT-Gym: uses the built-in expert-style fixed schedule for fertilizer and
  irrigation. The implementation is interval-aware, so a decision interval
  that steps from DAP 39 to DAP 46 still catches the DAP-40 target.
- CycleGym: uses the crop-planning rotation heuristic. It chooses the
  highest-priced crop, but avoids repeating the previous crop. When the top
  crop repeats, it switches to soybean if available; otherwise it switches to
  the next-best crop.

The implementation is a reproducible baseline, not a claim that the policy is
optimal or a substitute for crop-specific expert management.

## Source Stack For `ag_heuristic`

The source-informed policy should be defined from a small set of real
agronomic decision frameworks rather than an uncited heuristic recipe. Rules
should cite one of the sources below or a crop-specific extension source added
here.

### Fertilizer

- 4R Nutrient Stewardship: decisions are organized around right source, right
  rate, right time, and right place.
  Source: `https://4rcertified.org/what-are-the-4rs/`
- North Dakota Fertilizer Recommendation Tables and Equations, NDSU Extension
  SF882: crop recommendations based on field research, soil testing, nutrient
  needs, grower history, and economic modifiers for some nitrogen rates.
  Source: `https://www.ndsu.edu/agriculture/extension/publications/north-dakota-fertilizer-recommendation-tables-and-equations`
- Chickpea and pulse legumes: inoculated chickpea should not receive heavy
  nitrogen; soil testing guides fertility needs, and phosphorus/potassium are
  soil-test driven.
  Sources:
  `https://www.saskatchewan.ca/business/agriculture-natural-resources-and-industry/agribusiness-farmers-and-ranchers/crops-and-irrigation/field-crops/pulse-crop-bean-chickpea-faba-bean-lentils/chickpea/fertilizer-considerations`
  and
  `https://www.ndsu.edu/agriculture/extension/publications/soil-fertility-recommendations-field-pea-lentil-and-chickpea-north-dakota`
- Potato: nutrient programs should depend on soil test, tissue test, variety,
  harvest timing, yield goal, previous crop, and explicit nitrogen timing.
  Sources:
  `https://extension.umn.edu/crop-specific-needs/potato-fertilization-irrigated-soils`
  and
  `https://www.ndsu.edu/agriculture/extension/publications/fertilizing-potato-north-dakota`
- Maize/corn: nitrogen rates should account for expected yield, soil credits,
  and management credits; phosphorus and potassium are soil-test driven.
  Source: `https://extension.colostate.edu/resource/fertilizing-irrigated-corn/`

### Irrigation

- FAO CROPWAT: computes crop water requirements and irrigation requirements
  from soil, climate, and crop data, and develops irrigation schedules using a
  daily soil-water balance.
  Source: `https://www.fao.org/land-water/databases-and-software/cropwat/en/`
- FAO Crop Water Information: crop-specific water requirement and yield
  response references.
  Source: `https://www.fao.org/land-water/databases-and-software/crop-information/en/`
- Potato irrigation: monitor soil-water status and maintain tighter depletion
  thresholds during tuber initiation and bulking.
  Source: `https://extension.usu.edu/vegetableguide/potato/irrigation`

### Yield Potential

Use WOFOST-Gym `pp-v0` only as a yield-potential diagnostic. It represents
potential production without nitrogen, phosphorus, potassium, or water
limitations and without management actions. That is useful for interpreting
yield gaps, but it is not a profit policy and cannot be reached by a costed
`lnpkw-v0` management policy.

WOFOST-Gym environment semantics are defined by the installed WOFOST-Gym
dependency, including `ENVIRONMENT_CONFIG.md` when present under
`WOFOST_GYM_PATH`.

## AG-Heuristic Policy Contract

The `ag_heuristic` baseline is a frozen scripted policy. It is a reference
policy, not an expert policy or performance bound.

### WOFOST-Gym

Inputs available from WOFOST-Gym include:

- crop identity and scenario metadata;
- development stage `DVS`;
- current biomass/yield proxy `WSO`;
- soil moisture `SM`, rainfall `RAIN`, and optional stress variables such as
  `RFTRA` when exposed;
- available nutrient pools `NAVAIL`, `PAVAIL`, `KAVAIL`;
- cumulative applied inputs `TOTN`, `TOTP`, `TOTK`, `TOTIRRIG`.

WOFOST rule:

1. If `DVS < 0.20`, no-op.
2. If `0.20 <= DVS < 0.45`, apply medium phosphorus if phosphorus is available
   and total phosphorus is zero; otherwise medium potassium if potassium is
   available and total potassium is zero; otherwise medium nitrogen if nitrogen
   is available and total nitrogen is below the early cap.
3. If `0.45 <= DVS < 0.90`, apply medium/high nitrogen if nitrogen is
   available and total nitrogen is below the mid cap; otherwise irrigate medium
   if irrigation is available and the water policy says to irrigate.
4. If `0.90 <= DVS < 1.20`, irrigate medium if irrigation is available and the
   water policy says to irrigate; otherwise apply small nitrogen only when the
   objective is `yield_max` and total nitrogen is below the late cap.
5. If `DVS >= 1.20`, no-op.
6. Default caps: `early_N_cap=50`, `mid_N_cap=120`, `late_N_cap=160`. For
   `profit_max`, `late_N_cap=130`. For `yield_max`, `late_N_cap=170`. For
   `nutrient_stewardship`, use reduced nutrient caps. Action choices are
   projected to the nearest available action-menu level; if unavailable, no-op.

Conflict rule:

- If several actions are due in one WOFOST decision turn, the priority is
  phosphorus, then potassium, then nitrogen in early development; nitrogen
  before irrigation in mid development; irrigation before late yield-only
  nitrogen in late development. This is an implementation constraint from the
  discrete single-action menu, not an agronomic claim.

### DSSAT-Gym

DSSAT uses the expert-style fixed schedule:

| Target day after planting | Irrigation | Fertilization |
| ---: | ---: | ---: |
| 20 | 25 mm | 0 kg/ha N |
| 40 | 0 mm | 27 kg/ha N |
| 45 | 0 mm | 35 kg/ha N |
| 50 | 30 mm | 0 kg/ha N |
| 80 | 25 mm | 54 kg/ha N |

The runner catches schedule days within the current decision interval rather
than requiring exact equality with the observed DAP.

### CycleGym

CycleGym uses the crop-planning heuristic in the environment wrapper. It ranks
available rotation crops by current price. If the highest-price crop would
repeat the previous crop, it switches to soybean when soybean is available;
otherwise it selects the next-best crop. With two crops this behaves as a
year-to-year crop-rotation heuristic.

## Profit-Grid Oracle Contract

`profit_grid_oracle` is not an expert baseline. It is a simulator reference for
normalization and diagnosis.

Definition:

- Construct a small family of source-backed templates from the `ag_heuristic`
  records, such as conservative, medium, and high nutrient caps plus
  conservative and normal irrigation triggers.
- Evaluate the grid on the exact scenario split being diagnosed.
- Report the best terminal `profit_ge_kg_ha` as an attainable simulator
  reference.

Allowed use:

- normalization denominator;
- attainable scripted reference;
- diagnostic analysis of simulator difficulty.

Disallowed use:

- fair method-comparison row unless the template selection is frozen using only
  ID/train scenarios before OOD evaluation.

## Reporting Language

Allowed:

- "source-informed scripted agronomic reference";
- "ag-heuristic policy";
- "attainable scripted profit reference";
- "yield-potential diagnostic".

Avoid describing the baseline as an expert policy, a performance bound, an
agronomic common-sense anchor, or comparable to an agronomic expert when the
policy is the legacy high-input/stage-budgeted script.

For profit-risk reporting, include the percentage of validation splits where
each method is worse than `no_action`. This is more stable than normalizing
against a crop-blind high-input script.

## Current Implementation And Usage

The implementation lives in
`agrimanager/rollout/inference/ag_heuristic_rollout.py`.

Concrete WOFOST smoke usage after building the WOFOST smoke dataset:

```bash
bash smoke_tests/wofost_gym/run_build_datasets.sh

python -m agrimanager.rollout.inference.ag_heuristic_rollout \
  --data smoke_tests/wofost_gym/data/wofost_smoke_llm/val.parquet \
  --mode no_action \
  --output-dir smoke_tests/wofost_gym/results/baseline_eval/no_action

python -m agrimanager.rollout.inference.ag_heuristic_rollout \
  --data smoke_tests/wofost_gym/data/wofost_smoke_llm/val.parquet \
  --mode ag_heuristic \
  --output-dir smoke_tests/wofost_gym/results/baseline_eval/ag_heuristic
```

Current smoke/probe result:

| Probe | No-action profit | Ag-heuristic profit | Delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Chickpea cold, 1 row | 45.40 | 45.40 | 0.00 | The heuristic applies no inputs and avoids the legacy high-input loss mode. |
| Maize validation, 5 rows | 390.14 | 876.01 | +485.87 | The heuristic improves this small maize profit probe. |
| Eight-crop probe, 1 row per crop | 270.16 mean | 463.70 mean | +193.54 mean | Mean improves, but 4/8 crop scenarios are below no-action. |

The eight-crop probe is intentionally not a paper-level estimate. It is enough
to establish the claim boundary: `ag_heuristic` is a defensible scripted
reference, but it still should not be used as a performance bound or assumed to
dominate `no_action` under distribution shift.

Generated rollout outputs are local analysis artifacts. Do not commit rollout
results, logs, checkpoints, tracker links, or machine-specific paths.
