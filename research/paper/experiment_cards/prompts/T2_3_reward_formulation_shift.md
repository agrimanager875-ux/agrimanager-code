# T2.3 Reward-Formulation Shift Prompts

T2.3 changes the management-objective block while keeping the WOFOST maize source and action menu fixed. Clean runs should have matching true and prompt-facing objectives.

## Objective Contract

| Case | True objective | Prompt objective |
| --- | --- | --- |
| clean run | `env.objective_id` | same objective text |
| corruption probe | `env.objective_id` | `env.prompt_objective_id` or explicit override text |

Corruption probes must be separately labeled and should not be mixed into clean T2.3 validation counts.

## `yield_max`

```text
<management objective id="yield_max">
Goal: maximize final harvestable biomass / yield.

Decision rule: apply inputs when they are expected to increase final yield enough to justify the intervention under the environment dynamics.
</management objective>
```

## `profit_max`

```text
<management objective id="profit_max">
Goal: maximize terminal WSO-equivalent profit, not raw yield.

Score at the end of the season:
profit_wso_kg_ha =
  final_WSO_kg_ha
  - 3.5 * total_N_kg_ha
  - 5.5 * total_P_kg_ha
  - 2.25 * total_K_kg_ha
  - 0.5 * total_irrig_cm

Decision rule: apply an input only if that input is expected to increase final WSO by more than its WSO-equivalent cost.
</management objective>
```

## `water_stewardship`

```text
<management objective id="water_stewardship">
Goal: maintain acceptable terminal yield while conserving irrigation water.

Decision rule: avoid irrigation unless water stress is likely to reduce yield below the acceptable-yield target. Prefer lower water use when yield outcomes are similar.
</management objective>
```

## `nutrient_stewardship`

```text
<management objective id="nutrient_stewardship">
Goal: maintain acceptable terminal yield while avoiding N/P/K over-application and terminal nutrient imbalance.

Decision rule: apply nutrients only when the crop state and soil availability indicate a likely yield benefit. Avoid excess terminal nutrient stock.
</management objective>
```

## Shared Prompt Blocks

The observation block, action menu, and response format are inherited from the WOFOST maize config. The old live captures repeated those blocks for every objective; this file keeps only the objective blocks because those are the experimental axis.
