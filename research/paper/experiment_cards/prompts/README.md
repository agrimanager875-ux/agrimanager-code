# Prompt Documentation

This directory documents how AgriManager prompts are constructed for the main benchmark experiments. It is intentionally a reference layer, not a dump of every generated prompt instance.

The goals are:

- show the system prompt and user-prompt structure;
- identify which prompt blocks change for each experiment axis;
- provide representative examples for the benchmark conditions;
- avoid committing exhaustive generated captures, local paths, run IDs, tracker links, or machine-specific artifacts.

## Common Prompt Shape

Most LLM environments use the same high-level structure:

```text
System prompt

User prompt:
  episode context
  <management objective id="profit_max"> maximize terminal profit </management objective>
  optional <crop traits> maize trait summary </crop traits>
  <current observation> WOFOST state fields for the current decision day </current observation>
  available actions
  response format
```

The common WOFOST system prompt is:

```text
You are an agricultural management expert. Optimize the management objective specified in the user message. Use only the available actions and respond in the required format.
```

Think-mode prompts additionally require a reasoning block before the final answer:

```text
<tool_call>brief reasoning</tool_call> <answer>final action</answer>
```

No-think prompts require only the final action answer format specified by the environment.

## What To Keep Here

| Content | Keep? | Reason |
| --- | --- | --- |
| Prompt contracts and block structure | yes | Needed to understand the benchmark interface. |
| Representative objective/action/observation snippets | yes | Shows how the prompt works without bloating the repo. |
| Every generated prompt row | no | Regenerate from config/code if needed. |
| Generated JSON or markdown captures | no | These are run artifacts, not source documentation. |
| Machine-specific or execution-specific identifiers | no | Not needed and not double-blind safe. |

## Regeneration

Full prompts can be regenerated from the dataset parquet rows and environment configs by loading the stored `env_config`, instantiating the matching AgriManager environment adapter, and calling `system_prompt()`, `reset()`, and optional `step(...)` calls.
