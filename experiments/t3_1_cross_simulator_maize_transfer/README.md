# T3.1 Cross-Simulator Maize Transfer

This experiment family implements the canonical T3.1 single-source
cross-simulator maize transfer matrix. The scientific definition is:

```text
research/paper/experiment_cards/T3_1_cross_simulator_maize_transfer.md
```

T3.1 is intentionally single-source only. WOFOST-Gym, DSSAT-Gym, and CycleGym
each train one source specialist, and every trained policy is validated on all
three target simulators. Joint mixed-simulator training is reserved
for T3.2-style unified prompt-conditioned policy experiments.

## Dataset Configs

The formal configs are:

```text
config/t31_cross_sim_wofost.yaml
config/t31_cross_sim_gym_dssat.yaml
config/t31_cross_sim_cycles_gym.yaml
```

Each config materializes local `train.parquet` and `val.parquet` artifacts.
The WOFOST config uses `agrimanager/weather_pool_maize`, matching the T2.3
reward formulation shift data source. DSSAT and CycleGym use external
deterministic seed generators; their generated parquet rows are not claimed as
separate hosted datasets unless explicitly released. The split sizes are:

| Simulator | Train | Validation |
| --- | ---: | ---: |
| WOFOST-Gym | 1600 | 128 |
| DSSAT-Gym | 1600 | 128 |
| CycleGym | 1600 | 128 |

All configs use `objective_id: profit_max` with the aligned maize profit
constants from the T3.1 card.

## Training Runs

The fixed LLM runs are:

```bash
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_wofost_llm_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_wofost_llm_no_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_gym_dssat_llm_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_gym_dssat_llm_no_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_cycles_gym_llm_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_cycles_gym_llm_no_think_train.sh
```

The scripts follow the T2.3 reward-formulation pattern: they build required
datasets inline, pass named validation sets with `data.validation_axis=simulator`,
and set think/no-think response format through fixed `data.env_config_overrides`.

Outputs are written under:

```text
experiments/t3_1_cross_simulator_maize_transfer/logs/
experiments/t3_1_cross_simulator_maize_transfer/results/llm_train/
```

## Latest-Checkpoint Evaluation

After training, evaluate the latest checkpoint for each source policy on the
three target validation files:

```bash
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_latest_eval_all.sh
```

By default this reads checkpoints from:

```text
experiments/t3_1_cross_simulator_maize_transfer/results/llm_train/
```

Set `CROSS_SIM_CHECKPOINT_ROOT` only when evaluating checkpoints stored
elsewhere.
