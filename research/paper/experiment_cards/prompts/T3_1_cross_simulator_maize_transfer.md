# T3.1 Cross-Simulator Maize Transfer Prompts

T3.1 compares WOFOST-Gym, DSSAT-Gym, and CycleGym through AgriManager adapters. The prompt surfaces differ because each simulator exposes different observations and actions, but the task should preserve an aligned crop-growth objective where supported.

AgriManager does not claim to have created WOFOST-Gym, DSSAT-Gym, DSSAT-PDI, CycleGym, Cycles, or their underlying simulator data. These prompts document how AgriManager adapters present those external simulator tasks to a policy.

## Simulator Prompt Differences

| Simulator | Observation surface | Action surface | Objective surface |
| --- | --- | --- | --- |
| WOFOST-Gym | WOFOST crop status, N/P/K, soil moisture, cumulative actions, weather | WOFOST fertilizer and irrigation actions, depending on `env_id` | `profit_max` or aligned WSO-equivalent objective |
| DSSAT-Gym | DSSAT maize state, growth/yield indicators, soil/water/nitrogen fields exposed by adapter | DSSAT-compatible nitrogen and irrigation management actions | aligned profit/yield-style objective where supported |
| CycleGym | Cycles/CycleGym maize state, soil nitrogen, biomass/yield state, weather/context fields exposed by adapter | CycleGym crop-growth management actions | aligned profit/yield-style objective where supported |

## WOFOST-Gym Example Blocks

```text
<current observation>
[Crop status]
- Development stage index: ...
- Storage organ dry matter: ...
[Soil nutrients]
- Available soil nitrogen: ...
[Soil & water]
- Root-zone soil moisture: ...
[Weather]
- Mean air temperature: ...
- Daily rainfall: ...
</current observation>

Available actions:
- Do nothing.
- Apply nitrogen fertilizer (... kg/ha).
- Irrigate with ... cm of water.
```

## DSSAT-Gym Example Blocks

```text
<current observation>
[DSSAT maize state]
- Growth stage / day state: ...
- Grain or biomass proxy: ...
[Soil and water]
- Available water or stress proxy: ...
[Management history]
- Cumulative nitrogen or irrigation actions: ...
</current observation>

Available actions:
- Do nothing.
- Apply nitrogen fertilizer (... kg/ha).
- Irrigate with available DSSAT-compatible water action.
```

## CycleGym Example Blocks

```text
<current observation>
[CycleGym maize state]
- Crop growth or biomass state: ...
- Soil nitrogen state: ...
- Weather/context fields: ...
</current observation>

Available actions:
- Do nothing.
- Apply nitrogen fertilizer according to the CycleGym action menu.
```

## Response Format

T3.1 LLM-think runs use the same two-block response contract:

```text
<tool_call>brief reasoning</tool_call> <answer>final action</answer>
```

## Documentation Boundary

The original live captures were useful for adapter debugging, but full turn-by-turn prompts are bulky and state-dependent. Keep this file focused on the simulator-level interface differences; regenerate exact prompts from config/parquet rows when validating a specific run.
