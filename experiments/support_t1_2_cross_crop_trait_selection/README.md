# WOFOST Cross-Crop Strategy Validation

This folder contains diagnostic gates for the cross-crop trait claim. It is
separate from `experiments/t1_2_cross_crop_trait_shift/`, which holds the
main policy-training experiments.

The validation question is:

1. Do different crops require measurably different strategies?
2. Can crop traits predict those strategy differences?

## Data Source

The specialist-transfer evaluation reuses the main cross-crop no-traits test set:

```bash
experiments/t1_2_cross_crop_trait_shift/data/cross_crop_16id_nn_without_traits/test.parquet
```

This keeps the 20-crop evaluation scenarios aligned with the main experiment
while keeping validation scripts and outputs in this folder.

## Two-Specialist Gate

The first gate reuses existing maize/wheat no-traits weather specialists:

```bash
python experiments/support_t1_2_cross_crop_trait_selection/tools/diagnose_existing_specialist_transfer.py
```

The completed full output is currently stored under:

```bash
experiments/support_t1_2_cross_crop_trait_selection/analysis/existing_specialist_transfer/
```

## Eight-Specialist Gate

The eight specialist crops are:

- `maize`
- `wheat`
- `cotton`
- `rice`
- `potato`
- `sugarbeet`
- `barley`
- `seed_onion`

Existing maize/wheat no-traits weather specialists can be reused as anchors. The
six additional no-traits specialist scripts are:

```bash
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_cotton_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_rice_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_potato_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_sugarbeet_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_barley_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_seed_onion_nn_train_n1.sh
```

Submit the six new specialists as a CPU array:

```bash
sbatch experiments/support_t1_2_cross_crop_trait_selection/sbatch_wofost_specialist_transfer_new6_nn_train_array.slurm
```

After all specialist `agent.zip` and `vecnormalize.pkl` bundles exist, run the
8x20 transfer evaluation:

```bash
bash experiments/support_t1_2_cross_crop_trait_selection/tools/run_diagnose_eight_specialist_transfer.sh
```

or submit it on CPU:

```bash
sbatch experiments/support_t1_2_cross_crop_trait_selection/sbatch_wofost_specialist_transfer_eval_8x20.slurm
```

The 8x20 transfer output is written to:

```bash
experiments/support_t1_2_cross_crop_trait_selection/analysis/eight_specialist_transfer/
```

## Trait Selection

After the 8x20 transfer matrix is available, run nested crop-held-out trait
schema selection:

```bash
bash experiments/support_t1_2_cross_crop_trait_selection/tools/run_select_traits_from_eight_specialist_transfer.sh
```

This writes:

```bash
experiments/support_t1_2_cross_crop_trait_selection/analysis/eight_specialist_trait_selection/
```

That script compares existing schemas. To discover a new compact trait set from
raw and derived WOFOST candidate features, run:

```bash
bash experiments/support_t1_2_cross_crop_trait_selection/tools/run_discover_strategy_traits_from_eight_specialist_transfer.sh
```

This writes the strategy-supervised discovery output to:

```bash
experiments/support_t1_2_cross_crop_trait_selection/analysis/strategy_trait_discovery/
```

and generates a training-ready schema under:

```bash
agrimanager/env/wofost_gym/crop_traits/traits_strategy_selected_v1/
```
