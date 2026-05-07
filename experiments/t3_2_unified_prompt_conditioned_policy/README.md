# T3.2 Unified Prompt-Conditioned Policy

This experiment implements the full-observation T3.2 capstone design from
`research/paper/experiment_cards/T3_2_unified_prompt_conditioned_policy.md`.

The main training run uses `3200` rendered training rows: eight schema tuples
with `400` rows each. Observation schema is fixed to the full/native observation
interface. The experimental shift is compositional transfer across simulator,
action-menu, and objective tuples.

## Dataset Design

Training tuples:

- WOFOST `S1/lnpkw/yield_max`
- WOFOST `S1/lnpkw/profit_max`
- WOFOST `S1/lnpkw/water_stewardship`
- WOFOST `S1/lnpk/profit_max`
- WOFOST `S1/lnpk/nutrient_stewardship`
- WOFOST `S1/lnw/yield_max`
- Gym-DSSAT `yield_max`
- CycleGym `yield_max`

Held-out callback validation tuples:

- WOFOST `S1/lnw/profit_max`
- WOFOST `S1/lnpkw/nutrient_stewardship`
- Gym-DSSAT `profit_max`
- CycleGym `profit_max`

The training scripts also validate seen controls for WOFOST, DSSAT, and
CycleGym. All validation sets use `128` rows and are logged with
`data.validation_axis=schema_tuple`.

## Train

Think:

```bash
bash experiments/t3_2_unified_prompt_conditioned_policy/run_t32_unified_llm_think_train.sh
```

No-think:

```bash
bash experiments/t3_2_unified_prompt_conditioned_policy/run_t32_unified_llm_no_think_train.sh
```

Both scripts build missing datasets inline before launching training.
They are self-contained fixed run definitions, following the T3.1
Cross-Simulator script style: each script spells out the dataset configs,
parquet paths, named validation sets, model, and trainer overrides directly.
Validation is passed through `data.val_sets` with
`data.validation_axis=schema_tuple`. The fixed training scripts run validation
before training and every `10` trainer steps, with `4` validation generations
logged per validation pass.

## Speed Probe

The one-step no-think speed probe has its own fixed run script:

```bash
bash experiments/t3_2_unified_prompt_conditioned_policy/run_t32_speed_probe_no_think_workers12.sh
```
