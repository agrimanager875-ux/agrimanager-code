# Dataset and Generator Sources

This page is the source map for AgriManager's dataset and generator artifacts.
It distinguishes the three hosted WOFOST weather-pool datasets from executable
CycleGym and DSSAT-Gym generators, because static datasets and executable
environments have different hosting, documentation, and reproduction
requirements.

## NeurIPS 2026 Boundary

The NeurIPS 2026 E&D call says dataset and code artifacts must be hosted,
accessible, and clearly documented when they are submitted or released. It also
says new datasets and dataset collections should be hosted on a dedicated ML
hosting site, datasets larger than 4 GB should include a small inspection
sample, and dataset authors must provide Croissant metadata with Responsible AI
fields.

The hosting guidance makes the operational split explicit: if the contribution
is an executable environment or codebase rather than a static dataset, dataset
hosting is not required. If a dataset is part of the contribution, it must be
hosted and documented with Croissant metadata. The FAQ also states that
multiple datasets need one valid Croissant file per dataset, and synthetic
datasets follow the same hosting and documentation rules.

For AgriManager, this means:

| Artifact | Documentation treatment | Reason |
| --- | --- | --- |
| `agrimanager/wofost-weather-pool` | Hosted dataset | Static WOFOST-Gym weather scenario pool. |
| `agrimanager/wofost-weather-regime-pool` | Hosted dataset | Static WOFOST-Gym weather-regime benchmark pool. |
| `agrimanager/weather_pool_maize` | Hosted dataset | Static maize-only WOFOST-Gym weather scenario pool. |
| CycleGym configs and adapters | Executable generator/code | AgriManager generates rows from the external CycleGym simulator; no separate static dataset is claimed unless fixed generated parquet rows are released. |
| DSSAT-Gym configs and adapters | Executable generator/code | AgriManager generates rows from the external DSSAT-Gym/DSSAT-PDI simulator; no separate static dataset is claimed unless fixed generated parquet rows are released. |

## Hosted WOFOST Datasets

These three Hugging Face datasets are the static dataset artifacts used by the
WOFOST experiments:

| Dataset | Main code use | Required dataset materials |
| --- | --- | --- |
| `agrimanager/wofost-weather-regime-pool` | T1.1 weather-regime shift for `chickpea` and `potato`. | README/data card, previewable parquet files, `meteo_cache.tar.gz`, small sample if the runtime artifact is over 4 GB, Croissant JSON with RAI fields. |
| `agrimanager/wofost-weather-pool` | T1.2 cross-crop trait shift and WOFOST crop-support experiments. | README/data card, previewable parquet files, `meteo_cache.tar.gz`, small sample if the runtime artifact is over 4 GB, Croissant JSON with RAI fields. |
| `agrimanager/weather_pool_maize` | T2/T3 maize WOFOST schema, action, reward, and cross-simulator experiments. | README/data card, previewable parquet files, `meteo_cache.tar.gz`, small sample if the runtime artifact is over 4 GB, Croissant JSON with RAI fields. |

The hosted WOFOST rows are simulator-derived benchmark scenarios. The
meteorological inputs come from NASA POWER via the WOFOST-Gym/PCSE weather data
provider, but the released rows are curated and sampled by a deterministic
WOFOST-based pipeline. For Croissant RAI metadata, these datasets should set
`rai:hasSyntheticData` to `true` and describe the deterministic simulator-derived
scenario construction.

The runtime cache is part of the dataset artifact because it is needed to run
the exact same weather scenarios without re-querying NASA POWER. AgriManager's
WOFOST loader downloads parquet files and `meteo_cache.tar.gz`, extracts the
cache, and passes the extracted cache directory into the WOFOST environment.

Code references:

| Purpose | File |
| --- | --- |
| WOFOST weather-pool download/extraction/loading | `agrimanager/env/wofost_gym/weather_pool.py` |
| WOFOST artifact-first dataset builder | `agrimanager/env/wofost_gym/create_dataset.py` |
| Weather-pool generation configs | `integrations/wofost_gym/dataset_tools/weather_pool_configs/pool_crop_*.yaml` |
| Weather-pool build script | `integrations/wofost_gym/dataset_tools/build_weather_pool.py` |
| Weather-cache packaging | `integrations/wofost_gym/dataset_tools/package_weather_pool.py` |
| Hugging Face upload helper | `integrations/wofost_gym/dataset_tools/upload_weather_pool.py` |
| WOFOST dataset contract | `docs/wofost_dataset_contract.md` |
| WOFOST pool definitions | `experiments/wofost_gym_datasets.md` |

## WOFOST Pool Construction Summary

The 20-crop weather pool is generated from per-crop configs under
`integrations/wofost_gym/dataset_tools/weather_pool_configs/`. Each crop config
uses `generation_seed: 42`, `year_range: [1984, 2019]`, and
`min_dvs_threshold: 1.5`. The generated split sizes are `train=3200`,
`val=128`, and `test=512` per crop. A candidate scenario is a
`(crop, year, latitude, longitude)` tuple and is retained only if a full
no-action WOFOST rollout has complete weather, reaches `max_DVS >= 1.5`, and
produces `max_WSO > 0`.

The weather-regime pool is a materialized benchmark source for `chickpea` and
`potato`. Its named validation regimes are crop-specific 20 percent buckets:
`drought` is the lowest 20 percent by growing-season cumulative `RAIN`, `wet`
is the highest 20 percent by growing-season cumulative `RAIN`, `cold` is the
lowest 20 percent by growing-season mean daily `TEMP`, and `hot` is the highest
20 percent by growing-season mean daily `TEMP`. The `normal` group excludes all
four named extremes.

The maize weather pool is the maize-only WOFOST pool used by maize schema,
action, reward, and cross-simulator experiments. It uses the same WOFOST
viability filter as the 20-crop pool. Its current source has `train=3200`,
`val=640`, and no separate materialized test split.

## External Simulator Sources

AgriManager does not create CycleGym or DSSAT-Gym. It contributes configs,
adapters, prompt/rendering logic, dataset generators, training entrypoints, and
evaluation glue that use those external simulators.

| External source | AgriManager role | Code references |
| --- | --- | --- |
| WOFOST-Gym and PCSE/WOFOST | External crop simulator used by the three hosted WOFOST weather-pool datasets and WOFOST experiment rows. AgriManager wraps it through a text/numeric adapter and builds fixed parquet rows from hosted weather pools. | `agrimanager/env/wofost_gym/`, `integrations/wofost_gym/` |
| CycleGym | External Cycles-based RL simulator. AgriManager uses CycleGym to generate deterministic crop-planning and maize/corn rows from config-defined environments, seeds, year windows, locations, and price regimes. | `agrimanager/env/cycles_gym/`, `experiments/cycles_gym_price_regime/`, `experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_cycles_gym.yaml` |
| DSSAT-Gym / DSSAT-PDI / DSSAT-CSM | External DSSAT simulator interface. AgriManager uses DSSAT-Gym configs to generate deterministic maize, rice, and cotton rows from seeds, crop/task settings, decision intervals, and DSSAT environment parameters. | `agrimanager/env/gym_dssat/`, `smoke_tests/gym_dssat/`, `experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_gym_dssat.yaml` |
| VERL | External RL training framework used by AgriManager's training adapter. It is code infrastructure, not a dataset source. | `verl/`, `agrimanager/adapter/`, `entrypoints/train/` |

Cite the upstream WOFOST-Gym, PCSE/WOFOST, CycleGym, DSSAT-Gym/DSSAT-PDI,
DSSAT-CSM, NASA POWER, and VERL projects separately from AgriManager. Do not
claim that AgriManager created those simulators or their underlying simulator
assets.

## Upstream Versions And Licenses

Record upstream attribution separately from the install source used by this
repository. For double-blind execution, `install.sh` uses anonymous dependency
snapshots for WOFOST-Gym and CycleGym. These snapshots preserve the exact source
contents needed by AgriManager while this documentation continues to cite the
real upstream projects and licenses.

| Component | Upstream source to cite | Exact source used here | License | Notes |
| --- | --- | --- | --- | --- |
| WOFOST-Gym | `https://github.com/Intelligent-Reliable-Autonomous-Systems/WOFOSTGym` | `install.sh` defaults to the anonymous snapshot `https://github.com/agrimanager875-ux/WOFOSTGym.git` at commit `2a79a287b1d84789763e16f4367f510b5f7c9f6c`. | MIT | WOFOST-Gym package metadata reports version `1.1.0`. AgriManager uses WOFOST-Gym through configs/adapters and hosted weather pools; it does not create WOFOST-Gym. |
| PCSE / WOFOST | `https://github.com/ajwdewit/pcse` and the WOFOST crop-model literature cited by WOFOST-Gym | Vendored through the pinned WOFOST-Gym source above, under `pcse/` and `pcse_gym/`. | MIT for the bundled PCSE code inspected in WOFOST-Gym | AgriManager uses PCSE/WOFOST through WOFOST-Gym. The WOFOST weather inputs are derived from NASA POWER through the WOFOST-Gym/PCSE weather provider. |
| CycleGym | `https://github.com/kora-labs/cyclesgym` | `install.sh` defaults to the anonymous snapshot `https://github.com/agrimanager875-ux/cyclesgym.git` at commit `e91dba78060a05b402c0414b1cb238174adff311`. | BSD-3-Clause | CycleGym package metadata reports version `0.1.0`. AgriManager uses CycleGym as an external executable generator. |
| Cycles crop simulator | `https://github.com/PSUmodeling/Cycles` and `https://plantscience.psu.edu/research/labs/kemanian/models-and-tools/cycles` | Installed by CycleGym's `install_cycles.py` from the `v0.12.9-alpha` release archive: `Cycles_debian_0.12.9-alpha.zip`, `Cycles_macos_0.12.9-alpha.zip`, or `Cycles_win_0.12.9-alpha.zip`. | Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (`CC-BY-NC-ND-4.0`) | Native Cycles simulator copyright Armen Kemanian, 2023. The native simulator license is more restrictive than the BSD-3-Clause CycleGym wrapper license. AgriManager does not distribute or claim the Cycles simulator as an AgriManager-created dataset. |
| DSSAT-Gym / gym-DSSAT-PDI | `https://gitlab.inria.fr/rgautron/gym_dssat_pdi.git` | Optional DSSAT setup observed at commit `63f2c529e0bd339b4553beb9aa56d56af83b5e2b`; the upstream remote reports this as `HEAD` and `refs/heads/stable`. The Spack package recipe records `gym-dssat` version `0.0.8`. | BSD-3-Clause | DSSAT-Gym is optional and is not installed by `install.sh`; see `docs/gym_dssat_setup.md`. AgriManager uses it as an external executable generator. |
| DSSAT-PDI / DSSAT-CSM | `https://gitlab.inria.fr/rgautron/dssat-csm-os.git` and `https://github.com/DSSAT/dssat-csm-os` | Optional Spack runtime records `dssat-pdi` version `4.8.0.24_2`; the DSSAT-Gym source also records `dssat-csm-data` tag `v4.8.0.28`. | BSD-3-Clause for DSSAT-CSM/DSSAT-PDI sources | The DSSAT native runtime is an external simulator runtime. AgriManager only supplies configs/adapters/generation/training/evaluation glue. |
| VERL | `https://github.com/verl-project/verl.git` | Git submodule commit `7522bef0eb5c5761500fa8652e7ed45936f5323d`; `git describe` reports `v0.7.0-1-g7522bef0`. | Apache-2.0 | VERL is the external RL training framework used by the AgriManager training adapter. |

### DSSAT Spack Runtime Details

The optional DSSAT runtime is not installed by the standard `install.sh` path.
When DSSAT-backed experiments are enabled, the local Spack recipe used by the
AgriManager DSSAT setup records the following source artifacts:

| Package/resource | Source artifact | Version/tag | Checksum or ref |
| --- | --- | --- | --- |
| `gym-dssat` | `https://gitlab.inria.fr/rgautron/gym_dssat_pdi/-/archive/v0.0.8/gym_dssat_pdi-v0.0.8.tar.bz2` | `0.0.8` | SHA256 `a97276f43ce0b6ea9e543367fde9cc1f74c20a6155cc652048593c3376e6b9f6` |
| `dssat-pdi` | `https://gitlab.inria.fr/rgautron/dssat-csm-os/-/archive/4.8.0.24_2/dssat-csm-os-4.8.0.24_2.tar.bz2` | `4.8.0.24_2` | SHA256 `410193834540d41831c37b847a3219f6f70386135647389e8429c0c62bfb3a27` |
| `dssat-csm-data` | `https://github.com/DSSAT/dssat-csm-data.git` | tag `v4.8.0.28` | Resolved commit `c1a31af16fc82659e0b024d58e40b021981b182b`. |

The same recipe records earlier `dssat-pdi` versions, but AgriManager's
DSSAT-backed setup should cite the `4.8.0.24_2` runtime unless a different
runtime is explicitly installed and used.

## CycleGym and DSSAT-Gym Generators

CycleGym and DSSAT-Gym rows are generated by executable configs rather than
hosted as standalone HF datasets. They become static dataset artifacts only if
the fixed generated parquet rows are released and claimed as datasets.
Otherwise, the codebase must provide the configs, pinned environment setup, and
reproduction commands needed to regenerate them.

CycleGym generator behavior:

| Item | Documentation |
| --- | --- |
| Generator | `agrimanager/env/cycles_gym/create_dataset.py` |
| Environment wrapper | `agrimanager/env/cycles_gym/env.py` and `agrimanager/env/cycles_gym/env_config.py` |
| Price-regime experiment | `experiments/cycles_gym_price_regime/config/price_regime_llm_no_think.yaml` and paired think/NN configs |
| Cross-simulator maize config | `experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_cycles_gym.yaml` |
| Main controls | `env_id`, `year_windows`, `seeds_per_combo`, `seed_start`, `rotation_crops`, `crop_price_sampling`, paired price-regime overrides, objective and reward parameters |

DSSAT-Gym generator behavior:

| Item | Documentation |
| --- | --- |
| Generator | `agrimanager/env/gym_dssat/create_dataset.py` |
| Environment wrapper | `agrimanager/env/gym_dssat/env.py` and `agrimanager/env/gym_dssat/env_config.py` |
| Smoke configs | `smoke_tests/gym_dssat/config/` |
| Cross-simulator maize config | `experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_gym_dssat.yaml` |
| Main controls | `env_id`, `crop_name`, `train_seeds`, `val_seeds`, `decision_interval`, `num_seasons`, `env_params`, `dssat_params`, optional `soil_variation`, objective and reward parameters |

## Fresh-Checkout Reproduction Commands

Run these commands from the repository root after the standard install in
`README.md` has completed:

```bash
git clone https://github.com/agrimanager875-ux/agrimanager-code.git AgriManager
cd AgriManager
bash install.sh
```

The smoke-test commands are the fastest checks that dataset generation,
training, and evaluation entrypoints are wired correctly. The full experiment
examples use the same checked-in configs and scripts as the main experiment
runs.

### WOFOST-Gym Dataset-Backed Commands

The WOFOST commands consume the hosted Hugging Face datasets listed above and
materialize local parquet files from those fixed weather-pool rows.

Smoke generation:

```bash
bash smoke_tests/wofost_gym/run_build_datasets.sh
```

Smoke training:

```bash
bash smoke_tests/wofost_gym/run_llm_train.sh
bash smoke_tests/wofost_gym/run_nn_train.sh
```

Smoke evaluation:

```bash
bash smoke_tests/wofost_gym/run_llm_eval_qwen25_3b_instruct.sh
bash smoke_tests/wofost_gym/run_nn_eval.sh
```

Representative full WOFOST generation through the shared dataset entrypoint:

```bash
bash entrypoints/dataset/build.sh \
  --config experiments/t1_1_weather_regime_shift/config/weather_regime_chickpea_llm_without_traits_no_think.yaml \
  --num-workers 8

bash entrypoints/dataset/build.sh \
  --config experiments/t1_2_cross_crop_trait_shift/config/cross_crop_4id_llm_traits_v1_23d_no_think.yaml \
  --num-workers 8

bash entrypoints/dataset/build.sh \
  --config experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_wofost.yaml \
  --num-workers 8
```

Representative full WOFOST training and evaluation:

```bash
bash experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_chickpea_llm_no_think_train.sh
bash experiments/t1_2_cross_crop_trait_shift/run_wofost_generalization_cross_crop_4id_llm_no_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_wofost_llm_no_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_latest_eval_all.sh
```

The T1.1 and T1.2 scripts validate during training on their configured
validation splits. The T3.1 cross-simulator scripts additionally provide a
post-training latest-checkpoint evaluation script.

### CycleGym Generator Commands

CycleGym rows are generated from executable CycleGym configs. They are not
hosted as a separate static dataset unless fixed generated parquet rows are
explicitly released.

Smoke generation:

```bash
bash smoke_tests/cycles_gym/run_build_datasets.sh
bash smoke_tests/cycles_gym/run_build_datasets_crop_planning.sh
bash smoke_tests/cycles_gym/run_build_datasets_crop_growth_yield.sh
```

Smoke training:

```bash
bash smoke_tests/cycles_gym/run_llm_train.sh
bash smoke_tests/cycles_gym/run_nn_train.sh
bash smoke_tests/cycles_gym/run_llm_train_crop_planning.sh
bash smoke_tests/cycles_gym/run_nn_train_crop_planning.sh
bash smoke_tests/cycles_gym/run_llm_train_crop_growth_yield_think.sh
bash smoke_tests/cycles_gym/run_llm_train_crop_growth_yield_no_think.sh
```

Smoke evaluation:

```bash
bash smoke_tests/cycles_gym/run_llm_eval_qwen3_4b_instruct.sh
bash smoke_tests/cycles_gym/run_nn_eval.sh
bash smoke_tests/cycles_gym/run_llm_eval_qwen3_4b_instruct_crop_planning.sh
bash smoke_tests/cycles_gym/run_nn_eval_crop_planning.sh
bash smoke_tests/cycles_gym/run_llm_eval_qwen3_4b_instruct_crop_growth_yield_think.sh
bash smoke_tests/cycles_gym/run_llm_eval_qwen3_4b_instruct_crop_growth_yield_no_think.sh
```

Representative full CycleGym generation, training, and evaluation:

```bash
bash experiments/cycles_gym_price_regime/run_build_datasets.sh
bash experiments/cycles_gym_price_regime/run_llm_train_price_regime_think.sh
bash experiments/cycles_gym_price_regime/run_llm_train_price_regime_no_think.sh
bash experiments/cycles_gym_price_regime/run_nn_train_price_regime_n1.sh
bash experiments/cycles_gym_price_regime/run_nn_train_price_regime_n8.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_cycles_gym_llm_no_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_latest_eval_all.sh
```

The CycleGym price-regime experiment reports ID and OOD regime metrics through
training-time validation curves. The T3.1 cross-simulator script performs
latest-checkpoint post-training evaluation.

### DSSAT-Gym Generator Commands

DSSAT-Gym rows are generated from executable DSSAT-Gym/DSSAT-PDI configs. The
DSSAT stack is optional because it has a longer native setup path; install it
with `docs/gym_dssat_setup.md` before running these commands.

Activate the DSSAT environment for the current shell:

```bash
source smoke_tests/gym_dssat/_activate_spack.sh
```

Smoke generation:

```bash
bash smoke_tests/gym_dssat/run_build_datasets.sh maize
```

Smoke training:

```bash
bash smoke_tests/gym_dssat/run_llm_train.sh maize
```

Smoke evaluation:

```bash
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh maize
```

Representative full DSSAT-Gym generation through the shared dataset entrypoint:

```bash
bash entrypoints/dataset/build.sh \
  --config experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_gym_dssat.yaml \
  --num-workers 1
```

Representative full DSSAT-Gym training and evaluation:

```bash
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_gym_dssat_llm_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_gym_dssat_llm_no_think_train.sh
bash experiments/t3_1_cross_simulator_maize_transfer/run_cross_sim_latest_eval_all.sh
```

Generated parquet files remain local to the relevant `data/` directory. Logs
and training outputs remain local to the corresponding `logs/` and `results/`
directories.

## Release And Maintenance

An anonymized version of the AgriManager codebase will be available during
review. The full public GitHub repository will be released upon acceptance and
no later than the camera-ready deadline. We will maintain a stable release
corresponding to the paper and plan to support the repository for at least two
years after publication by addressing critical issues, fixing reproducibility
bugs, and updating documentation when necessary. No essential component
required to reproduce the benchmark will remain private.

## Dataset And Generator Checklist

For each hosted WOFOST dataset:

- Host the dataset publicly on Hugging Face.
- Provide previewable tabular files where possible.
- Include `meteo_cache.tar.gz` if needed for exact runtime reproduction.
- Include a small sample if the dataset or runtime artifact exceeds 4 GB.
- Include a dataset card with provenance, schema, splits, seeds, limitations,
  intended use, synthetic/simulator-derived status, and license.
- Generate one Croissant JSON file and add the required RAI fields.
- Validate each Croissant file before release or submission.
- Check that README files, Croissant files, sample files, manifests, HF
  metadata, and code docs do not leak author identity when double-blind
  evaluation is required.

For CycleGym and DSSAT-Gym:

- Do not present the upstream simulators or their underlying data as
  AgriManager-created artifacts.
- Cite the upstream simulators and any dependency snapshots/commits used.
- Document exact env names, configs, seeds, year windows, crop/task settings,
  price regimes, and reproduction commands.
- If fixed generated parquet rows are released as static artifacts, host those
  rows as datasets and provide Croissant metadata for them. Otherwise, the
  generator configs and code documentation are sufficient.

## NeurIPS References

- NeurIPS 2026 Evaluations and Datasets call:
  `https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets`
- NeurIPS 2026 dataset hosting guidelines:
  `https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting`
- NeurIPS 2026 Evaluations and Datasets FAQ:
  `https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ`
- NeurIPS 2026 Responsible AI metadata blog post:
  `https://blog.neurips.cc/2026/05/04/responsible-ai-metadata-requirements-for-the-evaluations-and-datasets-track-neurips-2026/`
