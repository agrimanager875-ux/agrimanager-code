#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  tools/deploy/unpack_deploy.sh <conda-env-archive.tar.gz> <code-runtime-archive.tar.gz> <deploy-dir>
  tools/deploy/unpack_deploy.sh --runtime-only <code-runtime-archive.tar.gz> <deploy-dir> [python]

Example:
  tools/deploy/unpack_deploy.sh \
    agrimanager-conda-agrimanager_bash_install_smoke-20260429_160000.tar.gz \
    agrimanager-code-runtimes-20260429_160000.tar.gz \
    /workspace/am

After unpacking:
  source <deploy-dir>/activate_agrimanager.sh

With --runtime-only, create or activate the target Conda environment first.
The helper uses [python], or the first python found on PATH, to install editable
packages against the unpacked code and runtimes.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

RUNTIME_ONLY=0
if [[ "${1:-}" == "--runtime-only" || "${1:-}" == "--use-current-env" ]]; then
    RUNTIME_ONLY=1
    shift
fi

if [[ "$RUNTIME_ONLY" == "1" ]]; then
    if [[ $# -lt 2 || $# -gt 3 ]]; then
        usage >&2
        exit 1
    fi
    ENV_ARCHIVE=""
    RUNTIME_ARCHIVE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    DEPLOY_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"
    ENV_DIR=""
    PYTHON="${3:-$(command -v python || true)}"
else
    if [[ $# -ne 3 ]]; then
        usage >&2
        exit 1
    fi
    ENV_ARCHIVE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    RUNTIME_ARCHIVE="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
    DEPLOY_DIR="$(mkdir -p "$3" && cd "$3" && pwd)"
    ENV_DIR="$DEPLOY_DIR/env"
    PYTHON="$ENV_DIR/bin/python"
fi

if [[ "$RUNTIME_ONLY" != "1" && ! -f "$ENV_ARCHIVE" ]]; then
    echo "Missing Conda env archive: $ENV_ARCHIVE" >&2
    exit 1
fi
if [[ ! -f "$RUNTIME_ARCHIVE" ]]; then
    echo "Missing code/runtime archive: $RUNTIME_ARCHIVE" >&2
    exit 1
fi

DSSAT_MAX_PATH_LENGTH="${AGRI_DSSAT_MAX_PATH_LENGTH:-70}"
DSSAT_CLD_PATH="$DEPLOY_DIR/AgriManager/spack/gym-dssat-pdi/bin/Weather/Climate"
if (( ${#DSSAT_CLD_PATH} > DSSAT_MAX_PATH_LENGTH )); then
    cat >&2 <<EOF
Deployment path is too long for DSSAT's legacy path buffers:
  CLD path length: ${#DSSAT_CLD_PATH}
  max safe length: $DSSAT_MAX_PATH_LENGTH
  CLD path: $DSSAT_CLD_PATH

Use a shorter deploy directory, for example:
  /tmp/am
  /workspace/am
  /root/am
EOF
    exit 1
fi

if [[ "$RUNTIME_ONLY" == "1" ]]; then
    if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
        echo "Runtime-only unpack needs an executable Python from the target Conda env." >&2
        echo "Activate the env first or pass its python path as the third argument." >&2
        exit 1
    fi
    echo "=== Using existing Python: $PYTHON ==="
else
    mkdir -p "$ENV_DIR"

    echo "=== Unpacking Conda env to $ENV_DIR ==="
    tar --no-same-owner -xzf "$ENV_ARCHIVE" -C "$ENV_DIR"
    if [[ -x "$ENV_DIR/bin/conda-unpack" ]]; then
        "$ENV_DIR/bin/conda-unpack"
    fi
fi

echo "=== Unpacking code and runtimes to $DEPLOY_DIR ==="
tar --no-same-owner -xzf "$RUNTIME_ARCHIVE" -C "$DEPLOY_DIR"

REPO_ROOT="$DEPLOY_DIR/AgriManager"
EXTERNAL_ROOT="$DEPLOY_DIR/AgriManagerExternal"

if [[ ! -x "$PYTHON" ]]; then
    echo "Unpacked Python is missing: $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$REPO_ROOT" ]]; then
    echo "Unpacked AgriManager repo is missing: $REPO_ROOT" >&2
    exit 1
fi
if [[ -d "$REPO_ROOT/.git" ]] && command -v git >/dev/null 2>&1; then
    git config --global --add safe.directory "$REPO_ROOT" || true
fi

rebind_absolute_spack_symlinks() {
    local max_depth="$1"
    local root
    local link target suffix new_target
    local find_args=()

    shift
    if [[ "$max_depth" != "recursive" ]]; then
        find_args=(-maxdepth "$max_depth")
    fi

    for root in "$@"; do
        [[ -d "$root" ]] || continue

        while IFS= read -r -d '' link; do
            target="$(readlink "$link")"
            [[ "$target" == /* ]] || continue

            new_target=""
            if [[ "$target" == */AgriManager/spack/* ]]; then
                suffix="${target#*/AgriManager/spack/}"
                new_target="$REPO_ROOT/spack/$suffix"
            elif [[ "$target" == */spack/opt/spack/* ]]; then
                suffix="${target#*/spack/opt/spack/}"
                new_target="$REPO_ROOT/spack/opt/spack/$suffix"
            fi

            if [[ -n "$new_target" && -e "$new_target" ]]; then
                ln -sfnT "$new_target" "$link"
            fi
        done < <(find "$root" "${find_args[@]}" -type l -print0)
    done
}

write_run_dssat_wrapper() {
    local wrapper="$1"

    mkdir -p "$(dirname "$wrapper")"
    cat > "$wrapper" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/dscsm048" "$@"
EOF
    chmod +x "$wrapper"
}

rewrite_dssat_profiles() {
    local dssat_bin="$1"
    local dssat_package="${2:-}"

    "$PYTHON" - "$dssat_bin" "$dssat_package" <<'PY'
from pathlib import Path
import re
import sys

dssat_bin = Path(sys.argv[1])
dssat_package = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 and sys.argv[2] else None

profiles = []
for candidate in [dssat_bin / "DSSATPRO.L48", dssat_package / "DSSATPRO.L48" if dssat_package else None]:
    if candidate and candidate.is_file() and candidate not in profiles:
        profiles.append(candidate)

patterns = [
    r"/\S*/AgriManager/spack/opt/spack/\S*/dssat-pdi-[^/\s]+",
    r"/\S*/AgriManager/spack/gym-dssat-pdi/bin",
    r"/\S*/AgriManager/spack/\._gym-dssat-pdi/[^/\s]+/bin",
    r"/\S*/AgriManager/spack/opt/spack/\S*/gcc-[^/\s]+",
]

for profile in profiles:
    text = profile.read_text()
    for pattern in patterns:
        text = re.sub(pattern, str(dssat_bin), text)
    profile.write_text(text)
    print(f"rewrote DSSAT profile: {profile}")
PY
}

rewrite_spack_configs() {
    "$PYTHON" - "$REPO_ROOT" <<'PY'
from pathlib import Path
import re
import sys

repo_root = Path(sys.argv[1]).resolve()
files = [
    repo_root / ".spack-user-config" / "repos.yaml",
    repo_root / "spack" / "var" / "spack" / "environments" / "gym-dssat-pdi" / "spack.yaml",
]

patterns = {
    r"/\S*/AgriManager/spack/var/spack/repos": str(repo_root / "spack" / "var" / "spack" / "repos"),
    r"/\S*/AgriManager/spack/gym-dssat-pdi": str(repo_root / "spack" / "gym-dssat-pdi"),
    r"/\S*/AgriManager": str(repo_root),
}

for path in files:
    if not path.is_file():
        continue
    text = path.read_text()
    updated = text
    for pattern, replacement in patterns.items():
        updated = re.sub(pattern, replacement, updated)
    if updated != text:
        path.write_text(updated)
        print(f"rewrote Spack config: {path}")
PY
}

echo "=== Rewriting Spack runtime config paths ==="
rewrite_spack_configs

echo "=== Rebinding DSSAT/Spack runtime symlinks ==="
DSSAT_VIEW_ROOT="$REPO_ROOT/spack/._gym-dssat-pdi"
if [[ -d "$DSSAT_VIEW_ROOT" ]]; then
    DSSAT_VIEW_TARGET="$(find "$DSSAT_VIEW_ROOT" -mindepth 1 -maxdepth 1 -type d -print | sort | head -n 1)"
    if [[ -n "$DSSAT_VIEW_TARGET" ]]; then
        DSSAT_PACKAGE_TARGET="$(find "$REPO_ROOT/spack/opt/spack" -path '*/dssat-pdi-*' -type d -print 2>/dev/null | sort | head -n 1)"
        ln -sfnT "$DSSAT_VIEW_TARGET" "$REPO_ROOT/spack/gym-dssat-pdi"
        rebind_absolute_spack_symlinks recursive \
            "$DSSAT_VIEW_TARGET/bin"
        rebind_absolute_spack_symlinks recursive \
            "$DSSAT_VIEW_TARGET/lib/python3.10"
        rebind_absolute_spack_symlinks 1 \
            "$DSSAT_VIEW_TARGET/lib" \
            "$DSSAT_VIEW_TARGET/lib64"

        DSSAT_BIN="$REPO_ROOT/spack/gym-dssat-pdi/bin"
        if [[ -n "$DSSAT_PACKAGE_TARGET" && -d "$DSSAT_PACKAGE_TARGET" ]]; then
            write_run_dssat_wrapper "$DSSAT_PACKAGE_TARGET/run_dssat"
        fi
        rm -f "$DSSAT_BIN/run_dssat"
        write_run_dssat_wrapper "$DSSAT_BIN/run_dssat"
        rewrite_dssat_profiles "$DSSAT_BIN" "$DSSAT_PACKAGE_TARGET"
    fi
fi

echo "=== Rebinding editable Python packages to this deploy path ==="
"$PYTHON" -m pip install --no-deps -e "$REPO_ROOT"
if [[ -d "$REPO_ROOT/verl" ]]; then
    "$PYTHON" -m pip install --no-deps -e "$REPO_ROOT/verl"
fi
if [[ -d "$EXTERNAL_ROOT/WOFOSTGym" ]]; then
    pushd "$EXTERNAL_ROOT/WOFOSTGym" >/dev/null
    [[ -d pcse && -d pcse_gym ]] && "$PYTHON" -m pip install --no-deps -e pcse -e pcse_gym
    [[ -d imitation && -d stable-baselines3 ]] && "$PYTHON" -m pip install --no-deps -e imitation -e stable-baselines3
    popd >/dev/null
fi
if [[ -d "$EXTERNAL_ROOT/CyclesGym" ]]; then
    CYCLES_GYM_SKIP_INSTALL=1 "$PYTHON" -m pip install --no-deps -e "$EXTERNAL_ROOT/CyclesGym"
fi

ACTIVATE_FILE="$DEPLOY_DIR/activate_agrimanager.sh"
if [[ "$RUNTIME_ONLY" == "1" ]]; then
    cat > "$ACTIVATE_FILE" <<EOF
#!/usr/bin/env bash
# Activate the intended Conda environment before sourcing this file.
export PYTHONNOUSERSITE=1
export AGRIMANAGER_ROOT="$REPO_ROOT"
export AGRIMANAGER_EXTERNAL_ROOT="$EXTERNAL_ROOT"
export WOFOST_GYM_DIR="$EXTERNAL_ROOT/WOFOSTGym"
export CYCLES_GYM_DIR="$EXTERNAL_ROOT/CyclesGym"
export DSSAT_GYM_PATH="$REPO_ROOT/spack/gym-dssat-pdi"
export DSSAT_PDI_BRIDGE_DIR="$EXTERNAL_ROOT/gym_dssat_pdi/gym-dssat-pdi"
_agrimanager_prepend_pythonpath() {
    if [[ -n "\$1" && -d "\$1" ]]; then
        case ":\${PYTHONPATH:-}:" in
            *":\$1:"*) ;;
            *) export PYTHONPATH="\$1\${PYTHONPATH:+:\$PYTHONPATH}" ;;
        esac
    fi
}
_agrimanager_prepend_pythonpath "$REPO_ROOT"
_agrimanager_prepend_pythonpath "$REPO_ROOT/verl"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/WOFOSTGym/pcse"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/WOFOSTGym/pcse_gym"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/WOFOSTGym/imitation"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/WOFOSTGym/stable-baselines3"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/CyclesGym"
_agrimanager_prepend_pythonpath "\$DSSAT_PDI_BRIDGE_DIR"
cd "$REPO_ROOT"
EOF
else
    cat > "$ACTIVATE_FILE" <<EOF
#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
source "$ENV_DIR/bin/activate"
export AGRIMANAGER_ROOT="$REPO_ROOT"
export AGRIMANAGER_EXTERNAL_ROOT="$EXTERNAL_ROOT"
export WOFOST_GYM_DIR="$EXTERNAL_ROOT/WOFOSTGym"
export CYCLES_GYM_DIR="$EXTERNAL_ROOT/CyclesGym"
export DSSAT_GYM_PATH="$REPO_ROOT/spack/gym-dssat-pdi"
export DSSAT_PDI_BRIDGE_DIR="$EXTERNAL_ROOT/gym_dssat_pdi/gym-dssat-pdi"
_agrimanager_prepend_pythonpath() {
    if [[ -n "\$1" && -d "\$1" ]]; then
        case ":\${PYTHONPATH:-}:" in
            *":\$1:"*) ;;
            *) export PYTHONPATH="\$1\${PYTHONPATH:+:\$PYTHONPATH}" ;;
        esac
    fi
}
_agrimanager_prepend_pythonpath "$REPO_ROOT"
_agrimanager_prepend_pythonpath "$REPO_ROOT/verl"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/WOFOSTGym/pcse"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/WOFOSTGym/pcse_gym"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/WOFOSTGym/imitation"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/WOFOSTGym/stable-baselines3"
_agrimanager_prepend_pythonpath "$EXTERNAL_ROOT/CyclesGym"
_agrimanager_prepend_pythonpath "\$DSSAT_PDI_BRIDGE_DIR"
cd "$REPO_ROOT"
EOF
fi
chmod +x "$ACTIVATE_FILE"

echo "=== Smoke checks ==="
(
    cd "$DEPLOY_DIR"
    "$PYTHON" - "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys
import torch
import agrimanager

repo_root = Path(sys.argv[1]).resolve()
agrimanager_path = Path(agrimanager.__file__).resolve()
if repo_root not in agrimanager_path.parents:
    raise SystemExit(
        f"agrimanager imported from {agrimanager_path}, expected under {repo_root}"
    )

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("agrimanager:", agrimanager.__file__)
PY
)

if [[ -x "$REPO_ROOT/spack/gym-dssat-pdi/bin/run_dssat" ]]; then
    echo "DSSAT run_dssat found: $REPO_ROOT/spack/gym-dssat-pdi/bin/run_dssat"
else
    echo "Warning: DSSAT run_dssat was not found or is not executable." >&2
fi

cat <<EOF
=== Done ===
Activate this deployment with:
  source "$ACTIVATE_FILE"

Note:
  The packed DSSAT/Spack runtime is fastest and safest when unpacked on a
  similar Linux server at the same filesystem path. Spack-built binaries may
  contain absolute RPATHs; if DSSAT fails after moving paths or OS versions,
  rebuild DSSAT with the source install path.
EOF
