# T1.1 Weather-Regime Shift Prompts

T1.1 changes the weather scenario distribution while keeping the prompt schema fixed. The prompt should not reveal the validation regime as privileged information; `weather_regime` is an evaluation label, not a normal prompt field.

## Covered Variants

| Variant | Prompt change |
| --- | --- |
| crop: `chickpea`, `potato` | crop name and crop-specific WOFOST state values change. |
| split: `val_id`, `val_drought`, `val_wet`, `val_hot`, `val_cold` | weather/state values change; prompt structure stays fixed. |
| think vs no-think | response-format block changes. |

## Stable Prompt Contract

| Block | T1.1 behavior |
| --- | --- |
| System prompt | common agricultural-management system prompt. |
| Objective | `profit_max`. |
| Crop traits | not part of the default T1.1 prompt. |
| Observation schema | native WOFOST observation fields. |
| Action menu | WOFOST action menu configured by the experiment YAML. |
| Evaluation label | `weather_regime`, stored in metadata, not shown as a privileged prompt label. |

## Representative User Prompt Skeleton

```text
We are growing chickpea from sow to maturity. The planned growing window spans around ... days, actions are taken every 10 days, and today is day ... of the season.

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

<current observation>
[Crop status]
- Finish flag: ...
- Development stage index: ...
- Storage organ dry matter: ...
[Soil nutrients]
- Available soil nitrogen: ...
- Available soil phosphorus: ...
- Available soil potassium: ...
[Soil & water]
- Root-zone soil moisture: ...
[Weather]
- Daily solar radiation: ...
- Mean air temperature: ...
- Daily rainfall: ...
</current observation>

Available actions (pick exactly one):
- Do nothing.
- Apply nitrogen fertilizer (... kg/ha).
- Apply phosphorus fertilizer (... kg/ha).
- Apply potassium fertilizer (... kg/ha).
- Irrigate with ... cm of water.
```

## Why The Full Captures Are Not Stored

The original live captures repeated the same prompt blocks for every crop, regime, and think/no-think combination. That is useful for debugging but too noisy for source documentation. The concise contract above is enough to understand the benchmark; exact prompts can be regenerated from the corresponding config and parquet row.
