# Experiment Cards

These cards compactly define the benchmark experiments. They are documentation for readers and maintainers, not generated reports.

Each card should include:

- the benchmark axis and simulator family;
- hosted dataset or deterministic generator source;
- train and validation split sizes;
- labels used for reporting;
- main metrics and claim boundaries;
- any required external simulator caveats.

Detailed install and run commands belong in the corresponding experiment
README, such as `experiments/t1_1_weather_regime_shift/README.md`, and
top-level `docs/` pages.

## Index

| Card | Purpose |
| --- | --- |
| `T1_1_weather_regime_shift.md` | Weather-regime OOD in WOFOST. |
| `T1_2_cross_crop_trait_shift.md` | Cross-crop and trait-conditioning OOD in WOFOST. |
| `T1_3_price_regime_shift.md` | Price-regime OOD in CycleGym crop planning. |
| `T2_1_observation_schema_shift.md` | Observation-schema shift in WOFOST maize. |
| `T2_2_action_menu_shift.md` | Action-menu shift in WOFOST maize. |
| `T2_3_reward_formulation_shift.md` | Reward-formulation shift in WOFOST maize. |
| `T2_4_schema_corruption_suite.md` | Optional prompt/schema corruption probes. |
| `T3_1_cross_simulator_maize_transfer.md` | Single-source transfer across WOFOST, DSSAT, and CycleGym. |
| `T3_2_unified_prompt_conditioned_policy.md` | Unified prompt-conditioned policy over held-out tuples. |
| `APP1_wofost_variety_ood_appendix.md` | Optional same-crop variety OOD appendix. |

## Shared References

| File | Role |
| --- | --- |
| `overview/datasets.md` | Dataset and generator source plan. |
| `overview/defaults.md` | Shared training/evaluation defaults. |
| `overview/rewards.md` | Reward objective definitions. |
| `prompts/` | Prompt-surface reference notes. |
