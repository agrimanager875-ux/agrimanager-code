#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash tools/env/generate_lock_from_env.sh <conda-env-name-or-prefix> [output-file]

This script snapshots a known-good environment into a pip constraint file that
can be consumed by install.sh or tools/env/repro_env.sh.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 1
fi

ENV_TARGET="$1"
OUTPUT_FILE="${2:-requirements/locks/agrimanager-py312-cu128.lock.txt}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$OUTPUT_FILE" == /* ]]; then
    OUTPUT_PATH="$OUTPUT_FILE"
else
    OUTPUT_PATH="$REPO_ROOT/$OUTPUT_FILE"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ "$ENV_TARGET" == /* ]]; then
    CONDA_RUN_ARGS=(-p "$ENV_TARGET")
else
    CONDA_RUN_ARGS=(-n "$ENV_TARGET")
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

conda run "${CONDA_RUN_ARGS[@]}" python - <<'PY' "$OUTPUT_PATH"
from datetime import date
from pathlib import Path
import packaging
import subprocess
import sys

output_path = Path(sys.argv[1])
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)

lines = [
    "# Generated from a known-good environment.",
    f"# Source date: {date.today().isoformat()}",
    "#",
    "# This file is intended for use via PIP_CONSTRAINT or `pip install -c`.",
    "# It intentionally excludes editable installs and local-path requirements",
    "# because those are installed explicitly by install.sh.",
    "",
]

for raw_line in freeze.splitlines():
    line = raw_line.strip()
    if not line:
        continue
    if line.startswith("-e "):
        continue
    if line.startswith("packaging @ file://"):
        lines.append(f"packaging=={packaging.__version__}")
        continue
    if " @ file://" in line:
        continue
    lines.append(line)

output_path.write_text("\n".join(lines) + "\n")
PY
