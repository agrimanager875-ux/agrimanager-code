#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPACK_ROOT="$PROJECT_ROOT/spack"
SPACK_SETUP="$SPACK_ROOT/share/spack/setup-env.sh"
SPACK_ENV_DIR="$SPACK_ROOT/var/spack/environments/gym-dssat-pdi"
export SPACK_USER_CONFIG_PATH="${SPACK_USER_CONFIG_PATH:-$PROJECT_ROOT/.spack-user-config}"
export SPACK_USER_CACHE_PATH="${SPACK_USER_CACHE_PATH:-$PROJECT_ROOT/.spack-user-cache}"
ORIGINAL_PATH="${PATH:-}"
ORIGINAL_PYTHONPATH="${PYTHONPATH:-}"
ORIGINAL_CONDA_PREFIX="${CONDA_PREFIX:-}"
ORIGINAL_PYTHON3="$(command -v python3 || true)"
ORIGINAL_CONDA_PYTHON=""
if [[ -n "$ORIGINAL_CONDA_PREFIX" && -x "$ORIGINAL_CONDA_PREFIX/bin/python3" ]]; then
    ORIGINAL_CONDA_PYTHON="$ORIGINAL_CONDA_PREFIX/bin/python3"
fi

if [[ -f "$SPACK_SETUP" && -d "$SPACK_ENV_DIR" && "${AGRI_DSSAT_ENABLE_SPACK_ACTIVATE:-0}" == "1" ]]; then
    # shellcheck disable=SC1090
    source "$SPACK_SETUP"
    if ! spack env activate "$SPACK_ENV_DIR"; then
        echo "Warning: failed to activate Spack env; using packed DSSAT runtime paths directly." >&2
    fi
fi

if [[ -z "${DSSAT_GYM_PATH:-}" && -d "$SPACK_ROOT/gym-dssat-pdi" ]]; then
    export DSSAT_GYM_PATH="$SPACK_ROOT/gym-dssat-pdi"
fi

if [[ -z "${DSSAT_SPACK_PYTHONPATH:-}" && -d "${DSSAT_GYM_PATH:-}" ]]; then
    DSSAT_SPACK_PYTHONPATH="$(
        find "$DSSAT_GYM_PATH" "$SPACK_ROOT/opt/spack" \
            -path '*/lib/python3.10/site-packages' -type d -print 2>/dev/null \
            | awk 'BEGIN{sep=""} {printf "%s%s", sep, $0; sep=":"}'
    )"
    if [[ -n "$DSSAT_SPACK_PYTHONPATH" ]]; then
        export DSSAT_SPACK_PYTHONPATH
    fi
fi

BRIDGE_SOURCE="$PROJECT_ROOT/AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi"

# Keep the active Conda Python in front so AgriManager/VERL imports still come
# from the working training environment, while preserving the DSSAT runtime
# variables added by Spack activation.
if [[ -n "$ORIGINAL_CONDA_PREFIX" && -d "$ORIGINAL_CONDA_PREFIX/bin" ]]; then
    export PATH="$ORIGINAL_CONDA_PREFIX/bin:$PATH"
elif [[ -n "$ORIGINAL_PATH" ]]; then
    export PATH="$ORIGINAL_PATH"
fi
hash -r

if [[ -n "$ORIGINAL_CONDA_PYTHON" ]]; then
    export AGRIMANAGER_PYTHON="$ORIGINAL_CONDA_PYTHON"
elif [[ -n "$ORIGINAL_PYTHON3" ]]; then
    export AGRIMANAGER_PYTHON="$ORIGINAL_PYTHON3"
fi

for runtime_lib in \
    "${ORIGINAL_CONDA_PREFIX:+$ORIGINAL_CONDA_PREFIX/lib}" \
    "${DSSAT_GYM_PATH:+$DSSAT_GYM_PATH/lib64}" \
    "${DSSAT_GYM_PATH:+$DSSAT_GYM_PATH/lib}"
do
    if [[ -n "$runtime_lib" && -d "$runtime_lib" ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$runtime_lib:"*) ;;
            *) export LD_LIBRARY_PATH="$runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
done

# Do not keep Spack's Python site-packages on PYTHONPATH for AgriManager's main
# interpreter. The DSSAT runtime subprocesses get their own Python 3.10 path in
# agrimanager/env/gym_dssat/env.py. Keeping Spack's site-packages here causes
# ABI mismatches when the Conda Python tries to import Spack-built NumPy.
if [[ -d "$BRIDGE_SOURCE/gym_dssat_pdi" ]]; then
    if [[ -n "$ORIGINAL_PYTHONPATH" ]]; then
        export PYTHONPATH="$BRIDGE_SOURCE:$ORIGINAL_PYTHONPATH"
    else
        export PYTHONPATH="$BRIDGE_SOURCE"
    fi
elif [[ -n "$ORIGINAL_PYTHONPATH" ]]; then
    export PYTHONPATH="$ORIGINAL_PYTHONPATH"
else
    unset PYTHONPATH || true
fi
