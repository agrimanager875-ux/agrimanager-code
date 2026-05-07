#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  tools/deploy/pack_deploy.sh <conda-env-name-or-prefix> [output-dir]
  tools/deploy/pack_deploy.sh --runtime-only [output-dir]

Examples:
  tools/deploy/pack_deploy.sh agrimanager_bash_install_smoke
  tools/deploy/pack_deploy.sh /path/to/conda/envs/agrimanager_bash_install_smoke /tmp/agri-pack
  tools/deploy/pack_deploy.sh --runtime-only /tmp/agri-pack

The script writes two archives:
  - agrimanager-conda-*.tar.gz: conda-pack environment archive
  - agrimanager-code-runtimes-*.tar.gz: AgriManager repo plus local external runtimes

With --runtime-only, the script skips conda-pack and writes only the
code/runtime archive plus the manifest and unpack helper. Use this when the
target machine will create or sync its own Conda environment.

By default the runtime archive includes AgriManager's top-level .git directory,
so the unpacked copy is still a git repository. Set
AGRI_DEPLOY_INCLUDE_GIT=0 to make a pure runtime snapshot without .git.

Environment overrides:
  AGRI_DEPLOY_RUNTIME_ONLY=1   Same as --runtime-only.
  AGRI_DEPLOY_ENV=<env>        Default Conda env for full env+runtime packs.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
RUNTIME_ONLY="${AGRI_DEPLOY_RUNTIME_ONLY:-0}"
INCLUDE_GIT="${AGRI_DEPLOY_INCLUDE_GIT:-1}"
args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --runtime-only|--skip-conda)
            RUNTIME_ONLY=1
            shift
            ;;
        --)
            shift
            args+=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

case "${RUNTIME_ONLY,,}" in
    1|true|yes|y) RUNTIME_ONLY=1 ;;
    *) RUNTIME_ONLY=0 ;;
esac

if [[ "$RUNTIME_ONLY" == "1" ]]; then
    if [[ "${#args[@]}" -gt 1 ]]; then
        echo "--runtime-only accepts at most one positional argument: [output-dir]." >&2
        usage >&2
        exit 1
    fi
    ENV_TARGET=""
    ENV_PREFIX=""
    OUT_DIR="${args[0]:-$REPO_ROOT/deploy_artifacts}"
else
    if [[ "${#args[@]}" -gt 2 ]]; then
        usage >&2
        exit 1
    fi
    ENV_TARGET="${args[0]:-${AGRI_DEPLOY_ENV:-${CONDA_DEFAULT_ENV:-}}}"
    OUT_DIR="${args[1]:-$REPO_ROOT/deploy_artifacts}"

    if [[ -z "$ENV_TARGET" || "$ENV_TARGET" == "base" ]]; then
        echo "Please pass the Conda environment name or prefix to package." >&2
        usage >&2
        exit 1
    fi

    if [[ -x "$ENV_TARGET/bin/python" ]]; then
        ENV_PREFIX="$(cd "$ENV_TARGET" && pwd)"
    else
        if ! command -v conda >/dev/null 2>&1; then
            echo "conda is required to resolve environment name: $ENV_TARGET" >&2
            exit 1
        fi
        ENV_PREFIX="$(conda run -n "$ENV_TARGET" python -c 'import sys; print(sys.prefix)')"
    fi

    if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
        echo "Resolved environment is not usable: $ENV_PREFIX" >&2
        exit 1
    fi

    CONDA_PACK_BIN="${CONDA_PACK_BIN:-$(command -v conda-pack || true)}"
    if [[ -z "$CONDA_PACK_BIN" ]]; then
        cat >&2 <<EOF
conda-pack is required but was not found.
Install it in the environment or base Conda first, for example:
  conda install -c conda-forge conda-pack -y
EOF
        exit 1
    fi
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_ARCHIVE=""
if [[ "$RUNTIME_ONLY" != "1" ]]; then
    ENV_LABEL="$(basename "$ENV_PREFIX" | tr -c 'A-Za-z0-9_.-' '_')"
    ENV_ARCHIVE="$OUT_DIR/agrimanager-conda-${ENV_LABEL}-${STAMP}.tar.gz"
fi
RUNTIME_ARCHIVE="$OUT_DIR/agrimanager-code-runtimes-${STAMP}.tar.gz"
MANIFEST="$OUT_DIR/agrimanager-deploy-${STAMP}.manifest"
UNPACK_SCRIPT="$OUT_DIR/unpack_deploy.sh"
VALIDATE_SCRIPT="$OUT_DIR/validate_deploy.sh"
README_FILE="$OUT_DIR/README.deploy.md"
RUNTIME_ARCHIVE_TMP=""

cleanup() {
    if [[ -n "${RUNTIME_ARCHIVE_TMP:-}" ]]; then
        rm -f "$RUNTIME_ARCHIVE_TMP"
    fi
}
trap cleanup EXIT

if [[ "$RUNTIME_ONLY" == "1" ]]; then
    echo "=== Skipping Conda env packaging (--runtime-only) ==="
else
    echo "=== Packaging Conda env: $ENV_PREFIX ==="
    "$CONDA_PACK_BIN" \
        -p "$ENV_PREFIX" \
        -o "$ENV_ARCHIVE" \
        --force \
        --ignore-editable-packages \
        --ignore-missing-files
fi

echo "=== Packaging AgriManager code and local runtimes ==="
paths=("AgriManager")
if [[ -d "$SOURCE_ROOT/AgriManagerExternal" ]]; then
    paths+=("AgriManagerExternal")
fi

vcs_excludes=()
if [[ "$INCLUDE_GIT" == "0" ]]; then
    vcs_excludes+=(--exclude='*/.git')
else
    # Keep only AgriManager's top-level .git history. External/runtime repos
    # are packed as code snapshots to avoid carrying unrelated Git metadata.
    vcs_excludes+=(
        --exclude='AgriManager/*/.git'
        --exclude='AgriManager/*/*/.git'
        --exclude='AgriManager/*/*/*/.git'
        --exclude='AgriManager/*/*/*/*/.git'
        --exclude='AgriManagerExternal/.git'
        --exclude='AgriManagerExternal/*/.git'
        --exclude='AgriManagerExternal/*/*/.git'
        --exclude='AgriManagerExternal/*/*/*/.git'
    )
fi
vcs_excludes+=(--exclude='*/.hg' --exclude='*/.svn')

RUNTIME_ARCHIVE_TMP="$OUT_DIR/.agrimanager-code-runtimes-${STAMP}.tmp.tar.gz"
(
    cd "$SOURCE_ROOT"
    tar \
        --create \
        --file - \
        "${vcs_excludes[@]}" \
        --exclude='AgriManager/.cache' \
        --exclude='AgriManager/.pytest_cache' \
        --exclude='AgriManager/.spack-user-cache' \
        --exclude='AgriManager/agrimanager.egg-info' \
        --exclude='AgriManager/deploy_artifacts' \
        --exclude='AgriManager/spack/gym-dssat-pdi/dssat_logs' \
        --exclude='AgriManager/spack/var/spack/cache' \
        --exclude='AgriManager/spack/var/spack/stage' \
        --exclude='AgriManager/install_logs' \
        --exclude='AgriManager/inference_rollout.log' \
        --exclude='AgriManager/outputs' \
        --exclude='AgriManager/wandb' \
        --exclude='AgriManager/*/wandb' \
        --exclude='AgriManager/*/*/wandb' \
        --exclude='AgriManager/*/*/*/wandb' \
        --exclude='AgriManager/smoke_tests/*/data' \
        --exclude='AgriManager/smoke_tests/*/logs' \
        --exclude='AgriManager/smoke_tests/*/results' \
        --exclude='AgriManager/experiments/*/logs' \
        --exclude='AgriManager/experiments/*/results' \
        --exclude='AgriManager/experiments/*/analysis' \
        "${paths[@]}"
) | gzip -c > "$RUNTIME_ARCHIVE_TMP"
mv "$RUNTIME_ARCHIVE_TMP" "$RUNTIME_ARCHIVE"
RUNTIME_ARCHIVE_TMP=""

{
    echo "created_at=$STAMP"
    echo "source_root=$SOURCE_ROOT"
    echo "repo_root=$REPO_ROOT"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "runtime_only=$RUNTIME_ONLY"
    if [[ "$RUNTIME_ONLY" != "1" ]]; then
        echo "env_prefix=$ENV_PREFIX"
        echo "env_archive=$ENV_ARCHIVE"
    fi
    echo "runtime_archive=$RUNTIME_ARCHIVE"
    echo "include_git=$INCLUDE_GIT"
    if [[ -x "$REPO_ROOT/spack/gym-dssat-pdi/bin/run_dssat" ]]; then
        echo "dssat_gym_path=$REPO_ROOT/spack/gym-dssat-pdi"
    fi
} > "$MANIFEST"
cp "$REPO_ROOT/tools/deploy/unpack_deploy.sh" "$UNPACK_SCRIPT"
chmod +x "$UNPACK_SCRIPT"
cp "$REPO_ROOT/tools/deploy/validate_deploy.sh" "$VALIDATE_SCRIPT"
chmod +x "$VALIDATE_SCRIPT"
cp "$REPO_ROOT/tools/deploy/README.md" "$README_FILE"

echo "=== Done ==="
if [[ "$RUNTIME_ONLY" != "1" ]]; then
    echo "Conda env archive: $ENV_ARCHIVE"
fi
echo "Code/runtime archive: $RUNTIME_ARCHIVE"
echo "Manifest: $MANIFEST"
echo "Unpack helper: $UNPACK_SCRIPT"
echo "Validation helper: $VALIDATE_SCRIPT"
echo "Deployment README: $README_FILE"
echo
if [[ "$RUNTIME_ONLY" == "1" ]]; then
    echo "Transfer the code/runtime archive and unpack helper to the target server."
    echo "After creating or activating the target Conda env, run:"
    echo "  ./unpack_deploy.sh --runtime-only <code-runtime-archive> <deploy-dir> [python]"
else
    echo "Transfer these files to the target server, then run:"
    echo "  ./unpack_deploy.sh <env-archive> <code-runtime-archive> <deploy-dir>"
fi
