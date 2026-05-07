# T2.2 Action-Menu Shift Prompts

T2.2 changes the available-action block while keeping the WOFOST maize source and `profit_max` objective fixed. The prompt should make unavailable actions explicit so invalid-action behavior can be evaluated.

## Action Menus

| Menu | Available actions in prompt | Unavailable actions |
| --- | --- | --- |
| `lnpkw-v0` | N, P, K, irrigation, do nothing | none from the WOFOST N/P/K/water set |
| `lnpk-v0` | N, P, K, do nothing | irrigation |
| `lnw-v0` | N, irrigation, do nothing | P and K |
| `ln-v0` | N, do nothing | P, K, irrigation |
| `lw-v0` | irrigation, do nothing | N, P, K |

## Full Menu Example

```text
Available actions (pick exactly one):
- Do nothing.
- Apply nitrogen fertilizer (20.0, 40.0, 60.0, or 80.0 kg/ha).
- Apply phosphorus fertilizer (10.0, 20.0, 30.0, or 40.0 kg/ha).
- Apply potassium fertilizer (10.0, 20.0, 30.0, or 40.0 kg/ha).
- Irrigate with 5.0, 10.0, 15.0, or 20.0 cm of water.
```

## Restricted Menu Examples

```text
# lnpk-v0
Available actions:
- Do nothing.
- Apply nitrogen fertilizer (... kg/ha).
- Apply phosphorus fertilizer (... kg/ha).
- Apply potassium fertilizer (... kg/ha).
Unavailable actions: irrigation.

# lnw-v0
Available actions:
- Do nothing.
- Apply nitrogen fertilizer (... kg/ha).
- Irrigate with ... cm of water.
Unavailable actions: phosphorus fertilizer or potassium fertilizer.

# ln-v0
Available actions:
- Do nothing.
- Apply nitrogen fertilizer (... kg/ha).
Unavailable actions: phosphorus fertilizer, potassium fertilizer, or irrigation.

# lw-v0
Available actions:
- Do nothing.
- Irrigate with ... cm of water.
Unavailable actions: nitrogen, phosphorus, or potassium fertilizer.
```

## Clean Versus Corrupted Prompt Schema

| Case | True environment menu | Prompt menu |
| --- | --- | --- |
| clean run | `env.env_id` | same as `env.env_id` |
| corruption probe | `env.env_id` | `env.prompt_action_schema_env_id` |

Corruption probes must be separately labeled. They should not be mixed into the clean T2.2 validation counts.

## What To Regenerate

Exact turn-by-turn prompts depend on WOFOST state and prior actions. Regenerate those from config/parquet when debugging; source docs only need the menu contract and representative action blocks.
