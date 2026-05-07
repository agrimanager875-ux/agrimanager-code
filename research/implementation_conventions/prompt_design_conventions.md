# Prompt Design Conventions

This document records prompt fields that are part of the benchmark definition. The implementation is the source of truth; generated prompt captures should not be committed as documentation artifacts.

## Code References

| Component | Location |
| --- | --- |
| WOFOST prompt rendering | `agrimanager/env/wofost_gym/prompt.py` |
| Objective text rendering | `agrimanager/env/base/objective_prompt.py` |
| Experiment prompt/config controls | `experiments/*/config/*.yaml` |

## Prompt Surface

Normal WOFOST prompts contain:

- a stable system prompt describing the agricultural-management role;
- episode context: crop, season horizon, decision interval, and current day;
- management objective text derived from `objective_id`;
- optional crop traits when `include_crop_traits=true`;
- current observation fields and units;
- available actions for the active action menu;
- response-format instructions.

## Config Fields

| Field | Meaning |
| --- | --- |
| `env.objective_id` | True reward objective used by the environment. |
| `env.prompt_objective_id` | Optional prompt-only objective override for explicit corruption probes. |
| `env.prompt_objective_text` | Optional free-form objective text; avoid in canonical runs unless documented. |
| `env.include_crop_traits` | Enables crop-trait context. |
| `env.env_id` | Selects the WOFOST action menu and observation/action surface. |
| `env.prompt_action_schema_env_id` | Optional prompt-only action schema override for explicit corruption probes. |

In clean runs, prompt-facing fields should match the true environment fields. Mismatches are valid only when an experiment explicitly defines a corruption probe.

## Experiment Axes

| Axis | Prompt/config treatment |
| --- | --- |
| Weather-regime shift | Prompt schema stays fixed; scenario source changes. |
| Cross-crop trait shift | Crop identity and optional trait text vary by crop. |
| Observation-schema shift | Observation fields, names, or units vary while base scenarios can stay paired. |
| Action-menu shift | Available actions vary while objective and base scenarios stay fixed. |
| Reward-formulation shift | Objective text and true reward vary by `objective_id`. |
| Cross-simulator transfer | Prompt surfaces expose simulator-specific observations/actions while preserving the aligned task objective. |

## Documentation Rule

Experiment cards should describe the prompt/schema axis at the level needed to interpret results. Full prompts should be regenerated from code/config when needed, not stored as paper-development artifacts.
