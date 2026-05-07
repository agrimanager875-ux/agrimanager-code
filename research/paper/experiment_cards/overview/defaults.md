# Experiment Defaults

These defaults apply unless an experiment card or runnable config explicitly overrides them.

## Split Defaults

| Rule | Default |
| --- | --- |
| Optimization split | `train` |
| Held-out checks | Named validation groups, usually `val_*` |
| Separate test split | Not used unless a card explicitly defines one |
| Validation size | `128` rows per named group unless the card states otherwise |
| Canonical generation seed | `42` |

## Training Defaults

| Policy family | Algorithm | Default train rows | Default validation rows | Runtime passes |
| --- | --- | ---: | ---: | --- |
| LLM | GRPO | `1600` unless overridden | `128` per group | `2` train epochs |
| NN | PPO | `1600` unless overridden | `128` per group | `8` train epochs for 1600-row splits |

LLM defaults are implemented through `entrypoints/train/config/agri_grpo.yaml`. NN defaults are implemented through `entrypoints/train/config/nn.yaml`.

## Evaluation Defaults

| Policy family | Default |
| --- | --- |
| LLM | Deterministic decoding with temperature `0` and one repeat. |
| NN | Policy evaluation with one repeat unless the experiment says otherwise. |

## WOFOST Defaults

| Field | Default |
| --- | --- |
| Environment family | WOFOST-Gym through the AgriManager adapter. |
| Decision interval | `10` days |
| Horizon | `24` turns |
| Default action mode | `lnpkw-v0` |
| Default objective | `profit_max` |
| Action amount scaling | Scale by decision interval |
| Crop traits | Disabled unless a trait experiment enables them |

Cards should document only the defaults they override.
