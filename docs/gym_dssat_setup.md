# Optional DSSAT-Gym Setup

This page documents the optional DSSAT-Gym/DSSAT-PDI runtime setup used by
AgriManager's DSSAT-backed experiments. It is intentionally separate from
`install.sh`: the DSSAT native runtime can take 1-2 hours to install and is
only required for DSSAT-backed dataset generation, training, and evaluation.

Treat DSSAT-Gym, DSSAT-PDI, and DSSAT-CSM as external simulator/runtime
sources. AgriManager does not create those upstream simulators or their
underlying simulator assets; it provides configs, adapters,
dataset-generation scripts, training wrappers, and evaluation commands that
use them.

Exact source, version, commit, and license information for DSSAT-Gym,
DSSAT-PDI, DSSAT-CSM, and the other external simulators is recorded in
`docs/dataset_and_generator_sources.md`.

## When This Setup Is Needed

You need this setup only for commands that use `env_name: gym_dssat`, including:

- `smoke_tests/gym_dssat/*`
- `experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_gym_dssat.yaml`
- `experiments/t3_2_unified_prompt_conditioned_policy/config/t32_unified_gym_dssat_*.yaml`
- `experiments/cross_simulator_crop_growth_ood/config/*gym_dssat*.yaml`

You can skip this setup for WOFOST-Gym and CycleGym smoke tests.

## Prerequisites

First complete the standard AgriManager install from the repository root:

```bash
git clone https://github.com/agrimanager875-ux/agrimanager-code.git AgriManager
cd AgriManager

conda create -n agrimanager python=3.12 -y
conda activate agrimanager
bash install.sh
```

Then install or unpack a compatible DSSAT-Gym/DSSAT-PDI runtime using the
upstream runtime instructions or a provided runtime bundle. AgriManager expects
that runtime to expose a `run_dssat` executable under:

```text
$DSSAT_GYM_PATH/bin/run_dssat
```

The default lookup path is:

```text
spack/gym-dssat-pdi
```

If your runtime is elsewhere, set `DSSAT_GYM_PATH`.

## Environment Variables

| Variable | Required? | Purpose |
| --- | --- | --- |
| `DSSAT_GYM_PATH` | Required unless using `spack/gym-dssat-pdi` | Root of the DSSAT-Gym/DSSAT-PDI runtime containing `bin/run_dssat`. |
| `DSSAT_PDI_BRIDGE_DIR` | Optional | Path to a local `gym_dssat_pdi` Python bridge checkout when it is not discoverable from the default external-runtime paths. |
| `DSSAT_SPACK_PYTHONPATH` | Optional | Python 3.10 site-packages paths for DSSAT subprocesses. Usually inferred by `smoke_tests/gym_dssat/_activate_spack.sh`. |
| `AGRI_DSSAT_ENABLE_SPACK_ACTIVATE` | Optional | Set to `1` to let `_activate_spack.sh` activate a local Spack environment before configuring runtime paths. |
| `SPACK_USER_CONFIG_PATH` | Optional | Repo-local Spack user config path. Defaults to `.spack-user-config`. |
| `SPACK_USER_CACHE_PATH` | Optional | Repo-local Spack cache path. Defaults to `.spack-user-cache`. |
| `WANDB_MODE=offline` | Optional | Keep experiment-tracker logging local during smoke runs. |

Typical external-runtime setup:

```bash
export DSSAT_GYM_PATH="$PWD/spack/gym-dssat-pdi"
source smoke_tests/gym_dssat/_activate_spack.sh
```

## Validate The Runtime

From the repository root, run:

```bash
conda activate agrimanager
source smoke_tests/gym_dssat/_activate_spack.sh

python - <<'PY'
import os
from pathlib import Path

dssat_gym_path = Path(os.environ["DSSAT_GYM_PATH"])
run_dssat = dssat_gym_path / "bin" / "run_dssat"

print("DSSAT_GYM_PATH:", dssat_gym_path)
print("run_dssat:", run_dssat)
assert run_dssat.exists(), f"Missing DSSAT runner: {run_dssat}"
PY
```

Then build the smallest DSSAT smoke dataset:

```bash
bash smoke_tests/gym_dssat/run_build_datasets.sh maize
```

If that succeeds, continue with the smoke-test commands in
`smoke_tests/gym_dssat/README.md`.

## Operational Notes

- DSSAT-Gym setup is optional because it is a slow native runtime dependency.
- DSSAT-Gym generated parquet rows are local executable outputs unless they are
  explicitly released as static datasets.
- Do not commit DSSAT runtime directories, generated DSSAT outputs, logs,
  checkpoints, or experiment-tracker folders to the code repository.
- If a DSSAT-backed command fails, include the failing command,
  `DSSAT_GYM_PATH`, and whether `bin/run_dssat` exists under that path when
  reporting or debugging the issue.
