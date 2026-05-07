# T2.1 Observation-Schema Shift

This folder contains the unified-format WOFOST implementation of the T2.1
observation-schema shift experiment.

The experiment keeps the simulator backend, action menu, and `profit_max`
objective fixed. Only the prompt-side observation interface changes.

## Canonical Files

The root files are the authoritative implementation:

- `config/t21_observation_schema_shift_llm_no_think.yaml`
- `config/t21_observation_schema_shift_llm_think.yaml`
- `run_build_datasets.sh`
- `run_t21_observation_schema_shift_llm_no_think_train.sh`
- `run_t21_observation_schema_shift_llm_think_train.sh`

The older `s1/` to `s5/` subfolders remain as legacy branch artifacts and are
not the canonical launch path for the new framework.

## Unified Dataset Design

This implementation follows the new dataset/validation framework:

- `validation_axis: observation_schema`
- one train split rendered with `S1_full_current`
- named validation sets through `data.val_sets`
- shared `scenario_sets` so schema variants reuse the same maize weather rows

The current local implementation uses `128` shared validation scenarios from
the weather-pool `val` split. Every validation condition re-renders those same
128 base maize scenarios with its own schema projection, so the validation set
contains `1024` rendered rows total: `128` each for `S1`, the four `S2`
component-drop masks, `S3`, `S4`, and `S5`.

## Schemas

| Validation set | Schema role | Prompt change |
| --- | --- | --- |
| `s1_full_current` | ID reference | All 15 baseline no-traits fields |
| `s2a_no_stage_time` | Missing-information | Remove `DVS`, `DAYS` |
| `s2b_no_resource_state` | Missing-information | Remove `NAVAIL`, `PAVAIL`, `KAVAIL`, `SM` |
| `s2c_no_management_history` | Missing-information | Remove `TOTN`, `TOTP`, `TOTK`, `TOTIRRIG` |
| `s2d_no_weather_context` | Missing-information | Remove `IRRAD`, `TEMP`, `RAIN` |
| `s3_domain_synonym_rename` | Semantic rename | Keep values fixed, replace prompt labels with agronomic synonyms plus glossary |
| `s4_compact_growth_superset` | Strict superset | Add `LAI`, `TAGP`, `RD`, `RFTRA`, `NUPTAKETOTAL`, `PUPTAKETOTAL`, `KUPTAKETOTAL` |
| `s5_anonymous_label_rename` | Label-grounding probe | Keep values fixed, replace prompt labels with `A` to `O` plus glossary |

All schema variants use `include_crop_traits: false`.

## Build

```bash
bash experiments/t2_1_observation_schema_shift/run_build_datasets.sh
```

This builds both think and no-think dataset families under:

- `experiments/t2_1_observation_schema_shift/data/t21_observation_schema_shift_llm_no_think/`
- `experiments/t2_1_observation_schema_shift/data/t21_observation_schema_shift_llm_think/`

## Train

No-think:

```bash
bash experiments/t2_1_observation_schema_shift/run_t21_observation_schema_shift_llm_no_think_train.sh
```

Think:

```bash
bash experiments/t2_1_observation_schema_shift/run_t21_observation_schema_shift_llm_think_train.sh
```

## Experiment Tracker Expectations

Because the training scripts set:

- `data.validation_axis=observation_schema`
- named `data.val_sets.*`

The experiment tracker should expose compact grouped validation metrics such as:

- `val-env-observation_schema/S1_full_current/...`
- `val-env-observation_schema/S2a_no_stage_time/...`
- `val-env-observation_schema/S3_domain_synonym_rename/...`

The split metadata also carries `observation_schema_family`, so offline analysis
can aggregate the four `S2` component-drop conditions as one family if needed.
