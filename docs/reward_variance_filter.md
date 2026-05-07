# Reward Variance (RV) Filter

This page is a feature reference for the rollout reward-variance filter.

It is intentionally narrow: use it when tuning or reviewing RV filtering,
not for general training layout or repository conventions.

The RV filter selects prompt groups (identified by `uid`) based on within-group reward variance before the policy update. Groups with higher reward variance carry more learning signal for GRPO — keeping only the most informative groups reduces noise and improves training efficiency.

## Recommended Configurations

### 1. Top-P Linear (recommended)

```yaml
trainer:
  rollout_filter:
    enable: True
    value: 0.9
    top_p_prob_mode: linear
    include_zero: False
    selection_eps: 0.01
```

### 2. Top-P Softmax

```yaml
trainer:
  rollout_filter:
    enable: True
    value: 0.6
    top_p_prob_mode: softmax
    include_zero: False
```

### 3. No Filter (baseline)

```yaml
trainer:
  rollout_filter:
    enable: False
```

Or equivalently, set `value: 1.0` with `include_zero: True` — this keeps all groups.

## How Each Mode Works

### Top-P Linear

Treats the per-group reward standard deviations as raw scores and accumulates them from largest to smallest until the running sum reaches a fraction `value` of the total score mass.

**Algorithm:**
1. Compute reward std for each prompt group.
2. Remove zero-std groups (when `include_zero: False`).
3. Sort groups by std in descending order.
4. Set threshold = `value * sum(all_stds) - selection_eps`.
5. Walk down the sorted list, accumulating scores. Stop when the cumulative sum reaches the threshold, or when a non-positive score is encountered.
6. Keep the selected groups; discard the rest.

**Characteristics:**
- Score-proportional: a group with 2x the std takes 2x the "budget". High-variance groups are strongly preferred.
- The `selection_eps` (default 0.01) prevents the threshold from being unreachable due to floating-point precision.
- Typically keeps fewer groups than softmax at the same `value`, because high-std groups consume budget faster.

### Top-P Softmax

Converts per-group reward standard deviations into a probability distribution via softmax, then applies nucleus-style (top-p) sampling — keeping the smallest set of groups whose cumulative probability mass exceeds `value`.

**Algorithm:**
1. Compute reward std for each prompt group.
2. Remove zero-std groups (when `include_zero: False`).
3. Apply `softmax` over the std scores to get a probability vector.
4. Sort probabilities in descending order.
5. Accumulate probabilities until the cumulative sum >= `value`.
6. Keep those groups; discard the rest.

**Characteristics:**
- Probability-normalized: softmax compresses the score distribution, so the gap between high-std and low-std groups is smaller than in linear mode.
- At the same `value`, softmax tends to keep *more* groups than linear because each group's probability is more uniform.
- This is why a lower `value` (e.g., 0.6) is recommended for softmax to achieve a similar filtering strength as linear at 0.9.

### No Filter

All prompt groups are kept. The policy update uses the full batch without any variance-based selection. This is the default behavior and serves as the baseline.

## Full Configuration Reference

```yaml
trainer:
  rollout_filter:
    enable: False              # Whether to enable RV filtering
    metric: reward_variance    # Metric to compute per group (only reward_variance supported)
    strategy: top_p            # Selection strategy (only top_p supported)
    filter_type: largest       # Keep groups with largest metric (only largest supported)
    value: 0.9                 # Top-p threshold in (0, 1]
    top_p_prob_mode: softmax   # "softmax" or "linear"
    selection_eps: 0.01        # Epsilon for linear mode threshold
    include_zero: True         # Whether to keep zero-variance groups
    max_consecutive_all_filtered_steps: 10  # Early-stop patience for fully filtered batches
    score_key: traj_score      # Key in reward_extra_infos_dict for trajectory scores
    zero_eps: 1e-10            # Threshold below which a std is considered zero
```

## Logged Metrics

When the filter is enabled, the following metrics are reported to the logger:

| Metric | Description |
|---|---|
| `rollout/in_group_reward_std` | Mean reward std across all groups (before filtering) |
| `rollout/in_group_reward_mean` | Mean reward mean across all groups |
| `rollout/in_group_reward_max` | Mean reward max across all groups |
| `rollout/chosen_in_group_reward_std` | Mean reward std of kept groups |
| `rollout/chosen_in_group_reward_mean` | Mean reward mean of kept groups |
| `rollout/chosen_in_group_reward_max` | Mean reward max of kept groups |
| `rollout/filter_kept_count` | Number of groups kept |
| `rollout/filter_kept_ratio` | Fraction of groups kept |
| `rollout/filter_zero_count` | Number of zero-variance groups (before filtering) |
| `rollout/filter_all_skipped` | Whether the current batch was skipped because all groups were filtered out |
| `rollout/consecutive_all_filtered_steps` | Number of consecutive skipped batches due to full filtering |
| `training/skipped_update` | Whether the optimizer update was skipped on this step |
| `training/early_stop` | Whether training terminated early on this step |
