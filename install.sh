#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

DEFAULT_LOCK_FILE="$REPO_ROOT/requirements/locks/agrimanager-py312-cu128.lock.txt"
DEFAULT_WOFOST_GYM_REF="2a79a287b1d84789763e16f4367f510b5f7c9f6c"
DEFAULT_CYCLES_GYM_REF="e91dba78060a05b402c0414b1cb238174adff311"
DEFAULT_CYCLES_GYM_LEGACY_GYM_VERSION="0.26.2"
DEFAULT_FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

WOFOST_GYM_REMOTE="${WOFOST_GYM_REMOTE:-https://github.com/agrimanager875-ux/WOFOSTGym.git}"
WOFOST_GYM_DIR="${WOFOST_GYM_DIR:-$REPO_ROOT/../AgriManagerExternal/WOFOSTGym}"
WOFOST_GYM_REF="${WOFOST_GYM_REF:-$DEFAULT_WOFOST_GYM_REF}"
CYCLES_GYM_REMOTE="${CYCLES_GYM_REMOTE:-https://github.com/agrimanager875-ux/cyclesgym.git}"
CYCLES_GYM_DIR="${CYCLES_GYM_DIR:-$REPO_ROOT/../AgriManagerExternal/CyclesGym}"
CYCLES_GYM_REF="${CYCLES_GYM_REF:-$DEFAULT_CYCLES_GYM_REF}"
CYCLES_GYM_LEGACY_GYM_VERSION="${CYCLES_GYM_LEGACY_GYM_VERSION:-$DEFAULT_CYCLES_GYM_LEGACY_GYM_VERSION}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-$DEFAULT_FLASH_ATTN_WHEEL_URL}"
VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-cu128}"
AGRI_USE_LOCKFILE="${AGRI_USE_LOCKFILE:-1}"
AGRI_SKIP_VERL_BOOTSTRAP="${AGRI_SKIP_VERL_BOOTSTRAP:-0}"

mkdir -p "$(dirname "$WOFOST_GYM_DIR")"
mkdir -p "$(dirname "$CYCLES_GYM_DIR")"

LOCK_FILE_TO_APPLY=""
if [[ -n "${AGRI_PIP_CONSTRAINT:-}" ]]; then
    LOCK_FILE_TO_APPLY="$AGRI_PIP_CONSTRAINT"
elif [[ "$AGRI_USE_LOCKFILE" != "0" ]]; then
    LOCK_FILE_TO_APPLY="$DEFAULT_LOCK_FILE"
fi

if [[ -n "$LOCK_FILE_TO_APPLY" ]]; then
    export PIP_CONSTRAINT="$LOCK_FILE_TO_APPLY"
fi

if [[ -n "${PIP_CONSTRAINT:-}" ]]; then
    if [[ ! -f "$PIP_CONSTRAINT" ]]; then
        echo "Constraint file not found: $PIP_CONSTRAINT" >&2
        exit 1
    fi
    echo "=== Using pip constraint file: $PIP_CONSTRAINT ==="
fi

if [[ -n "$LOCK_FILE_TO_APPLY" ]]; then
    echo "=== Applying locked package manifest ==="
    python -m pip install --upgrade --no-deps -r "$LOCK_FILE_TO_APPLY"
    echo "=== Installing pinned FlashAttention wheel ==="
    python -m pip install --upgrade --no-deps "$FLASH_ATTN_WHEEL_URL"
fi

echo "=== Using WOFOST-Gym ref: $WOFOST_GYM_REF ==="
echo "=== Using CyclesGym ref: $CYCLES_GYM_REF ==="

if [[ -d "$REPO_ROOT/.git" ]]; then
    git submodule update --init --recursive
elif [[ -d "$REPO_ROOT/verl" ]]; then
    echo "=== Skipping git submodule update; using packed verl snapshot ==="
else
    echo "Missing verl/ and no .git metadata is available to fetch submodules." >&2
    exit 1
fi

echo "=== Installing verl ==="
cd verl
if [[ -n "$LOCK_FILE_TO_APPLY" || "$AGRI_SKIP_VERL_BOOTSTRAP" == "1" ]]; then
    echo "=== Skipping VERL bootstrap; relying on the pinned package stack ==="
else
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
fi
python -m pip install --no-deps -e .
cd ..

echo "=== Cloning or updating WOFOST-Gym ==="
if [[ -d "$WOFOST_GYM_DIR/.git" ]]; then
    git -C "$WOFOST_GYM_DIR" fetch --tags --prune --all
    git -C "$WOFOST_GYM_DIR" checkout --detach "$WOFOST_GYM_REF"
    echo "=== WOFOST-Gym HEAD: $(git -C "$WOFOST_GYM_DIR" rev-parse HEAD) ==="
elif [[ -d "$WOFOST_GYM_DIR" ]]; then
    echo "=== Using packed WOFOST-Gym snapshot: $WOFOST_GYM_DIR ==="
    if [[ ! -d "$WOFOST_GYM_DIR/pcse" || ! -d "$WOFOST_GYM_DIR/pcse_gym" ]]; then
        echo "Packed WOFOST-Gym snapshot is incomplete: $WOFOST_GYM_DIR" >&2
        exit 1
    fi
else
    git clone "$WOFOST_GYM_REMOTE" "$WOFOST_GYM_DIR"
    git -C "$WOFOST_GYM_DIR" fetch --tags --prune --all
    git -C "$WOFOST_GYM_DIR" checkout --detach "$WOFOST_GYM_REF"
    echo "=== WOFOST-Gym HEAD: $(git -C "$WOFOST_GYM_DIR" rev-parse HEAD) ==="
fi

echo "=== Installing WOFOST-Gym core dependencies ==="
pushd "$WOFOST_GYM_DIR" >/dev/null
if [[ -n "$LOCK_FILE_TO_APPLY" ]]; then
    python -m pip install --no-deps -e pcse -e pcse_gym
    python -m pip install --no-deps tyro huggingface_sb3
    python -m pip install --no-deps -e imitation -e stable-baselines3
else
    python -m pip install -e pcse -e pcse_gym
    python -m pip install tyro huggingface_sb3
    python -m pip install -e imitation -e stable-baselines3
fi
popd >/dev/null

echo "=== Cloning or updating CyclesGym ==="
if [[ -d "$CYCLES_GYM_DIR/.git" ]]; then
    git -C "$CYCLES_GYM_DIR" fetch --tags --prune --all
    git -C "$CYCLES_GYM_DIR" checkout --detach "$CYCLES_GYM_REF"
    echo "=== CyclesGym HEAD: $(git -C "$CYCLES_GYM_DIR" rev-parse HEAD) ==="
elif [[ -d "$CYCLES_GYM_DIR" ]]; then
    echo "=== Using packed CyclesGym snapshot: $CYCLES_GYM_DIR ==="
    if [[ ! -f "$CYCLES_GYM_DIR/setup.py" || ! -f "$CYCLES_GYM_DIR/install_cycles.py" ]]; then
        echo "Packed CyclesGym snapshot is incomplete: $CYCLES_GYM_DIR" >&2
        exit 1
    fi
else
    git clone "$CYCLES_GYM_REMOTE" "$CYCLES_GYM_DIR"
    git -C "$CYCLES_GYM_DIR" fetch --tags --prune --all
    git -C "$CYCLES_GYM_DIR" checkout --detach "$CYCLES_GYM_REF"
    echo "=== CyclesGym HEAD: $(git -C "$CYCLES_GYM_DIR" rev-parse HEAD) ==="
fi

echo "=== Installing CyclesGym runtime and package ==="
pushd "$CYCLES_GYM_DIR" >/dev/null
# CyclesGym still imports legacy OpenAI Gym directly during module import.
python -m pip install "gym==$CYCLES_GYM_LEGACY_GYM_VERSION"

# Install the Python package first, but skip setup.py's post-install hook.
# The native runtime bootstrap is invoked explicitly below so failures are visible.
if [[ -n "$LOCK_FILE_TO_APPLY" ]]; then
    CYCLES_GYM_SKIP_INSTALL=1 python -m pip install --no-deps -e .
else
    CYCLES_GYM_SKIP_INSTALL=1 python -m pip install -e .
fi

# Run the native Cycles bootstrap explicitly instead of relying on setup.py install hooks.
python install_cycles.py

# Fail fast if bootstrap did not populate the runtime tree required by env construction.
required_cycles_files=(
    "cycles/Cycles"
    "cycles/input/ContinuousCorn.operation"
    "cycles/input/GenericCrops.crop"
    "cycles/input/GenericHagerstown.soil"
    "cycles/input/RockSprings.weather"
)
for required_file in "${required_cycles_files[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "CyclesGym bootstrap incomplete: missing $CYCLES_GYM_DIR/$required_file" >&2
        exit 1
    fi
done
popd >/dev/null

echo "=== Installing AgriManager ==="
if [[ -n "$LOCK_FILE_TO_APPLY" ]]; then
    python -m pip install --no-deps -e .
else
    python -m pip install -e .
fi
echo "=== Done ==="
