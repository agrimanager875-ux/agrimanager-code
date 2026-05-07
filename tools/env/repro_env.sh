#!/usr/bin/env bash

set -euo pipefail

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

usage() {
    cat <<'EOF'
Usage:
  bash tools/env/repro_env.sh create <conda-env-name-or-prefix>
  bash tools/env/repro_env.sh sync   <conda-env-name-or-prefix>

Behavior:
  create  Create a fresh Python 3.12 conda env, install the checked-in lock
          file, then run install.sh.
  sync    Remove non-editable packages not present in the checked-in lock
          file, re-apply the lock, then run install.sh.

Environment overrides:
  AGRI_PIP_CONSTRAINT   Alternate lock file path.
  PIP_VERSION           Pip version to install before applying the lock.
  PYTHON_VERSION        Python version for `create` (default: 3.12).
  PYTHONNOUSERSITE      Defaults to 1 to block user-site package leakage.
EOF
}

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 1
fi

ACTION="$1"
ENV_TARGET="$2"

case "$ACTION" in
    create|sync) ;;
    *)
        usage >&2
        exit 1
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK_FILE="${AGRI_PIP_CONSTRAINT:-$REPO_ROOT/requirements/locks/agrimanager-py312-cu128.lock.txt}"
PIP_VERSION="${PIP_VERSION:-25.3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if [[ ! -f "$LOCK_FILE" ]]; then
    echo "Lock file not found: $LOCK_FILE" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ "$ENV_TARGET" == /* ]]; then
    CONDA_CREATE_ARGS=(-p "$ENV_TARGET")
    CONDA_RUN_ARGS=(-p "$ENV_TARGET")
else
    CONDA_CREATE_ARGS=(-n "$ENV_TARGET")
    CONDA_RUN_ARGS=(-n "$ENV_TARGET")
fi

env_exists() {
    conda run "${CONDA_RUN_ARGS[@]}" python -c "import sys" >/dev/null 2>&1
}

run_in_env() {
    conda run --no-capture-output "${CONDA_RUN_ARGS[@]}" "$@"
}

sync_non_editable_packages() {
    local remove_file

    remove_file="$(mktemp)"

    run_in_env python - <<'PY' "$LOCK_FILE" "$remove_file"
from pathlib import Path
import subprocess
import sys


def canonicalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


lock_file = Path(sys.argv[1])
remove_file = Path(sys.argv[2])

keep = set()
for line in lock_file.read_text().splitlines():
    item = line.strip()
    if not item or item.startswith("#"):
        continue
    name = item.split("==", 1)[0].split("[", 1)[0]
    keep.add(canonicalize(name))

freeze = subprocess.check_output(
    [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
    text=True,
)

remove = []
for line in freeze.splitlines():
    item = line.strip()
    if not item or item.startswith("#"):
        continue
    if " @ " in item:
        name = item.split(" @ ", 1)[0]
    elif "==" in item:
        name = item.split("==", 1)[0]
    else:
        continue
    if canonicalize(name) not in keep:
        remove.append(name)

remove_file.write_text("\n".join(sorted(set(remove), key=str.lower)))
PY

    if [[ -s "$remove_file" ]]; then
        echo "=== Removing packages not present in lock file ==="
        while IFS= read -r package; do
            [[ -z "$package" ]] && continue
            run_in_env python -m pip uninstall -y "$package"
        done <"$remove_file"
    else
        echo "=== No extra non-editable packages to remove ==="
    fi

    rm -f "$remove_file"
}

echo "=== Repo root: $REPO_ROOT ==="
echo "=== Lock file: $LOCK_FILE ==="
echo "=== Pip version: $PIP_VERSION ==="

if [[ "$ACTION" == "create" ]]; then
    if env_exists; then
        echo "Environment already exists: $ENV_TARGET" >&2
        echo "Use \`sync\` to repair an existing environment." >&2
        exit 1
    fi
    conda create "${CONDA_CREATE_ARGS[@]}" "python=$PYTHON_VERSION" -y
elif ! env_exists; then
    echo "Environment does not exist: $ENV_TARGET" >&2
    exit 1
fi

run_in_env python -m pip install --upgrade "pip==$PIP_VERSION"

if [[ "$ACTION" == "sync" ]]; then
    sync_non_editable_packages
fi

run_in_env env AGRI_PIP_CONSTRAINT="$LOCK_FILE" bash "$REPO_ROOT/install.sh"

echo "=== Environment is now aligned to the checked-in lock file ==="
