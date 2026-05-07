#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV_NAME="agrimanager"

ENV_TARGET="${1:-$DEFAULT_ENV_NAME}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

cd "$REPO_ROOT"
bash tools/env/repro_env.sh create "$ENV_TARGET"
