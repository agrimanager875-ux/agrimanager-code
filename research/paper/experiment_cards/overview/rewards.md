# Reward Objectives

Reward objectives are part of the task definition. Dataset rows may carry objective metadata for reproduction, but semantic definitions belong here and in the experiment cards.

## Config Contract

| Field | Meaning |
| --- | --- |
| `env.objective_id` | True environment reward objective. |
| `env.prompt_objective_id` | Optional prompt-only objective override for corruption probes. |
| `env.reward_params` | Objective constants. |
| `reward_formulation` | Validation label used when reward form is the experiment axis. |

In clean runs, the prompt-facing objective and true environment objective should match. Mismatches are allowed only for explicit corruption probes.

## Objective IDs

| Objective | Meaning | Main use |
| --- | --- | --- |
| `profit_max` | Maximize WSO-equivalent or grain-equivalent profit after input costs. | Default WOFOST objective, T2.2, T3.1. |
| `yield_max` | Maximize terminal yield or WSO. | T2.3 and yield-only baselines. |
| `water_stewardship` | Preserve acceptable yield while penalizing irrigation and water-budget violations. | T2.3 seen training objective. |
| `nutrient_stewardship` | Preserve acceptable yield while avoiding nutrient over-application and poor terminal nutrient status. | T2.3 held-out objective. |
| CycleGym gross revenue | Maximize `yield x crop price` over crop-planning episodes. | T1.3 price-regime shift. |

## WOFOST Profit Default

WOFOST `profit_max` uses WSO-equivalent input costs:

```text
profit_wso_kg_ha =
  final_WSO_kg_ha
  - c_N * total_N_kg_ha
  - c_P * total_P_kg_ha
  - c_K * total_K_kg_ha
  - c_W * total_irrig_mm
```

Default constants:

| Parameter | Value | Unit |
| --- | ---: | --- |
| `c_N` | `3.5` | kg WSO-equivalent per kg nitrogen |
| `c_P` | `5.5` | kg WSO-equivalent per kg phosphorus |
| `c_K` | `2.25` | kg WSO-equivalent per kg potassium |
| `c_W` | `0.05` | kg WSO-equivalent per mm irrigation |

## Experiment Mapping

| Experiment | Reward objective |
| --- | --- |
| T1.1 Weather-regime shift | `profit_max` |
| T1.2 Cross-crop trait shift | `profit_max` |
| T1.3 Price-regime shift | CycleGym gross revenue |
| T2.1 Observation-schema shift | `profit_max` |
| T2.2 Action-menu shift | `profit_max` |
| T2.3 Reward-formulation shift | Train on `yield_max`, `profit_max`, `water_stewardship`; validate also on held-out `nutrient_stewardship`. |
| T2.4 Corruption suite | Parent experiment objective, with prompt-side corruption only when labeled. |
| T3.1 Cross-simulator transfer | Aligned profit-style objective where supported by the target simulator. |
| T3.2 Unified prompt-conditioned policy | Mixed objective tuples with held-out validation combinations. |
