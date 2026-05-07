# WOFOST Same-Crop Variety Generalization

This experiment family is gated by a simulator diagnostic. We do not train a
variety OOD model until WOFOST shows that same-crop varieties produce separable
outcomes under controlled weather and action sequences.

## Diagnostic Gate

The fixed diagnostic runs replay the same action sequence under the same crop
weather scenarios while changing only `agro_params.crop_variety`.

```bash
bash experiments/app1_wofost_variety_ood/run_wofost_variety_diagnostic_wheat.sh
bash experiments/app1_wofost_variety_ood/run_wofost_variety_diagnostic_rice.sh
bash experiments/app1_wofost_variety_ood/run_wofost_variety_diagnostic_potato.sh
```

Default wheat setup:

- crop: `wheat`
- varieties: `wheat_1` to `wheat_8`; `wheat_9` is excluded because it duplicates `wheat_8`
- environment: `lnpkw-v0`, 10-day interval, 24 turns, final WSO reward
- scenarios: 64 wheat weather scenarios from the weather pool train split
- action controls: 3 random action sequences plus one no-op sequence

Rice and potato use the same setup with all nine unique varieties for each
crop. Their initial candidate pair is `*_1` vs `*_9`; final split choices should
still use the full `pair_summary.csv`.

Outputs are written under:

- `results/diagnostics/wheat_variant_separability_v1/rollouts.csv`
- `results/diagnostics/wheat_variant_separability_v1/summary_by_weather_action.csv`
- `results/diagnostics/wheat_variant_separability_v1/pair_summary.csv`
- `results/diagnostics/wheat_variant_separability_v1/gate_decision.json`

Generated outputs stay local and should not be committed.

## Decision Rule

Proceed to variety OOD training only if the random-policy diagnostic passes all
of the following:

- median across-variety relative yield range is at least 5%
- at least 75% of random `(weather, action_seed)` groups have at least 2%
  relative range
- the candidate OOD pair `wheat_1` vs `wheat_7` has at least 5% median paired
  relative yield difference

If the median across-variety relative range is below 2%, stop the wheat variety
OOD experiment and run the same diagnostic for another crop such as `rice` or
`maize` before designing training runs.

## Rice Variety OOD Runs

Rice is the main same-crop variety OOD experiment after the diagnostic gate.

Split:

- ID: `rice_1`, `rice_3`, `rice_4`, `rice_6`, `rice_7`, `rice_8`, `rice_9`
- OOD: `rice_2`, `rice_5`

Budgets:

- train: ID only, `7 x 448 = 3136`
- validation: all varieties, `9 x 96 = 864`
- test: all varieties, `9 x 128 = 1152`

Fixed train runs:

```bash
bash experiments/app1_wofost_variety_ood/run_wofost_variety_generalization_rice_llm_think_train.sh
bash experiments/app1_wofost_variety_ood/run_wofost_variety_generalization_rice_llm_no_think_train.sh
bash experiments/app1_wofost_variety_ood/run_wofost_variety_generalization_rice_nn_train_n1.sh
```

All rice variety runs use `rice_variety_traits_v1`, whose artifacts are stored
under `agrimanager/env/wofost_gym/crop_traits/rice_variety_traits_v1/`.

The NN train script uses `agent.n_epochs=1` and `runtime.train_epochs=16`.

## Analysis

Keep local analysis outputs outside the committed code artifact. If run metadata
is exported from an experiment tracker, store only sanitized config, summary,
metadata, and selected validation metric history needed to reproduce the tables.
