#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

usage() {
    cat <<'EOF'
Usage:
  tools/deploy/validate_deploy.sh --archive <runtime-archive.tar.gz>
  tools/deploy/validate_deploy.sh --deploy-dir <deploy-dir> [--python <python>]
  tools/deploy/validate_deploy.sh <deploy-dir> [python]

Options:
  --archive <file>       Validate the packed runtime archive structure.
  --deploy-dir <dir>     Validate an unpacked deployment directory.
  --python <path>        Python from the target Conda env. Defaults to python on PATH.
  --timeout <seconds>    Timeout for the DSSAT smoke test. Default: 240.
  --skip-dssat-smoke     Skip the live DSSATEnv.reset() smoke test.
  -h, --help             Show this help.

Examples:
  bash tools/deploy/validate_deploy.sh \
    --archive /workspace/agrimanager_pack/agrimanager-code-runtimes-20260503_025117.tar.gz

  bash /workspace/am/AgriManager/tools/deploy/validate_deploy.sh \
    --deploy-dir /workspace/am \
    --python "$(which python)"
EOF
}

log() {
    printf '=== %s ===\n' "$*"
}

warn() {
    printf 'Warning: %s\n' "$*" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || die "Missing file: $path"
}

require_dir() {
    local path="$1"
    [[ -d "$path" ]] || die "Missing directory: $path"
}

require_exec() {
    local path="$1"
    [[ -x "$path" ]] || die "Missing executable: $path"
}

run_internal_dssat_smoke() {
    local repo_root="$1"
    local python_bin="$2"

    export PYTHONNOUSERSITE=1
    export DSSAT_GYM_PATH="${DSSAT_GYM_PATH:-$repo_root/spack/gym-dssat-pdi}"
    cd "$repo_root"

    if [[ -f smoke_tests/gym_dssat/_activate_spack.sh ]]; then
        # shellcheck disable=SC1091
        source smoke_tests/gym_dssat/_activate_spack.sh
    fi

    export PYTHONPATH="$repo_root:$repo_root/verl${PYTHONPATH:+:$PYTHONPATH}"
    "$python_bin" - "$repo_root" <<'PY'
from pathlib import Path
import os
import sys

repo_root = Path(sys.argv[1]).resolve()
from agrimanager.env.gym_dssat.env import DSSATEnv
from agrimanager.env.gym_dssat.env_config import DSSATEnvConfig

cfg = DSSATEnvConfig(
    env_id="maize-irrigation-v0",
    env_params={"mode": "all"},
    turn_num=5,
    decision_interval=1,
    seed=7300,
)
env = DSSATEnv(cfg)
obs, info = env.reset()
print("DSSAT_SMOKE_OK", type(obs).__name__, sorted(list(info.keys()))[:8])
close = getattr(env, "close", None)
if close is not None:
    close()
PY
}

if [[ "${1:-}" == "--_dssat-smoke" ]]; then
    [[ $# -eq 3 ]] || die "Internal DSSAT smoke usage error."
    run_internal_dssat_smoke "$2" "$3"
    exit 0
fi

TMP_FILES=()
cleanup() {
    local tmp
    for tmp in "${TMP_FILES[@]:-}"; do
        rm -f "$tmp"
    done
}
trap cleanup EXIT

ARCHIVE=""
DEPLOY_DIR=""
PYTHON_BIN=""
TIMEOUT_SECONDS="${AGRI_DEPLOY_VALIDATE_TIMEOUT:-240}"
RUN_DSSAT_SMOKE=1
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --archive)
            ARCHIVE="${2:?Missing value for --archive}"
            shift 2
            ;;
        --deploy-dir)
            DEPLOY_DIR="${2:?Missing value for --deploy-dir}"
            shift 2
            ;;
        --python)
            PYTHON_BIN="${2:?Missing value for --python}"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SECONDS="${2:?Missing value for --timeout}"
            shift 2
            ;;
        --skip-dssat-smoke)
            RUN_DSSAT_SMOKE=0
            shift
            ;;
        --)
            shift
            POSITIONAL+=("$@")
            break
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ "${#POSITIONAL[@]}" -gt 0 && -z "$DEPLOY_DIR" ]]; then
    DEPLOY_DIR="${POSITIONAL[0]}"
fi
if [[ "${#POSITIONAL[@]}" -gt 1 && -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${POSITIONAL[1]}"
fi
if [[ "${#POSITIONAL[@]}" -gt 2 ]]; then
    usage >&2
    die "Too many positional arguments."
fi

if [[ -z "$ARCHIVE" && -z "$DEPLOY_DIR" ]]; then
    usage >&2
    die "Pass --archive, --deploy-dir, or both."
fi

if [[ -n "$PYTHON_BIN" ]]; then
    require_exec "$PYTHON_BIN"
else
    PYTHON_BIN="$(command -v python || true)"
fi

require_archive_pattern() {
    local listing="$1"
    local pattern="$2"
    local description="$3"
    if ! grep -Eq "$pattern" "$listing"; then
        die "Archive is missing $description"
    fi
}

validate_archive() {
    local archive="$1"
    local listing

    require_file "$archive"
    listing="$(mktemp)"
    TMP_FILES+=("$listing")

    log "Reading archive listing: $archive"
    tar -tzf "$archive" > "$listing"

    require_archive_pattern "$listing" '^AgriManager/install\.sh$' "AgriManager/install.sh"
    require_archive_pattern "$listing" '^AgriManager/tools/deploy/unpack_deploy\.sh$' "unpack_deploy.sh"
    require_archive_pattern "$listing" '^AgriManager/verl/' "VERL snapshot"
    require_archive_pattern "$listing" '^AgriManager/spack/(gym-dssat-pdi|._gym-dssat-pdi|opt/spack/)' "DSSAT/Spack runtime"
    require_archive_pattern "$listing" '^AgriManager/spack/opt/spack/.*/dssat-pdi-[^/]+/dscsm048$' "DSSAT dscsm048 binary"
    require_archive_pattern "$listing" '^AgriManagerExternal/WOFOSTGym/pcse/' "WOFOST-Gym pcse snapshot"
    require_archive_pattern "$listing" '^AgriManagerExternal/WOFOSTGym/pcse_gym/' "WOFOST-Gym pcse_gym snapshot"
    require_archive_pattern "$listing" '^AgriManagerExternal/CyclesGym/setup\.py$' "CyclesGym snapshot"

    if grep -Eq '(^|/)\.git(/|$)' "$listing"; then
        warn "Archive contains .git metadata. This is allowed, but runtime-only transfer packages usually exclude it."
    else
        log "Archive contains no .git metadata"
    fi

    log "Archive structure looks usable"
}

validate_dssat_profile() {
    local repo_root="$1"
    local dssat_bin="$2"
"$PYTHON_BIN" - "$repo_root" "$dssat_bin" <<'PY'
from pathlib import Path
import os
import sys

repo_root = Path(sys.argv[1]).resolve()
dssat_bin = Path(sys.argv[2])
profile = dssat_bin / "DSSATPRO.L48"
model_err = dssat_bin / "MODEL.ERR"

if not profile.is_file():
    raise SystemExit(f"Missing DSSAT profile: {profile}")
if not model_err.is_file():
    raise SystemExit(f"Missing DSSAT MODEL.ERR: {model_err}")

entries = {}
for line in profile.read_text(errors="replace").splitlines():
    if "//" not in line:
        continue
    key = line[:3].strip()
    value = line.split("//", 1)[1].strip()
    if not key or not value:
        continue
    path = value.split()[0]
    if path.startswith("/"):
        entries[key] = Path(path)

required = ["DDB", "CDD", "WED", "CLD"]
for key in required:
    if key not in entries:
        raise SystemExit(f"DSSATPRO.L48 missing {key} path")
    resolved = entries[key].resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(
            f"DSSATPRO.L48 {key} points outside deploy tree: {entries[key]}"
        ) from exc

climate_dir = entries["CLD"]
max_path = int(os.environ.get("AGRI_DSSAT_MAX_PATH_LENGTH", "70"))
if len(str(climate_dir)) > max_path:
    raise SystemExit(
        f"DSSAT CLD path is too long ({len(str(climate_dir))} > {max_path}): "
        f"{climate_dir}. Use a shorter deploy path such as /workspace/am."
    )

ufga = climate_dir / "UFGA.CLI"
if not ufga.exists():
    raise SystemExit(f"Missing climate file from CLD path: {ufga}")

print("DSSAT_PROFILE_OK", profile)
print("DSSAT_CLIMATE_OK", ufga)
PY
}

validate_python_env() {
    local repo_root="$1"
    "$PYTHON_BIN" - "$repo_root" <<'PY'
from pathlib import Path
import importlib
import sys

repo_root = Path(sys.argv[1]).resolve()
external_root = repo_root.parent / "AgriManagerExternal"
import_paths = [
    repo_root,
    repo_root / "verl",
    external_root / "WOFOSTGym" / "pcse",
    external_root / "WOFOSTGym" / "pcse_gym",
    external_root / "WOFOSTGym" / "imitation",
    external_root / "WOFOSTGym" / "stable-baselines3",
    external_root / "CyclesGym",
]
for path in reversed(import_paths):
    if path.exists():
        sys.path.insert(0, str(path))

modules = ["agrimanager", "verl", "pcse", "pcse_gym", "cyclesgym"]
for module in modules:
    imported = importlib.import_module(module)
    location = getattr(imported, "__file__", "<namespace>")
    print(f"IMPORT_OK {module} {location}")

expected_roots = {
    "agrimanager": repo_root,
    "verl": repo_root,
    "pcse": external_root / "WOFOSTGym",
    "pcse_gym": external_root / "WOFOSTGym",
    "cyclesgym": external_root / "CyclesGym",
}
for module, expected_root in expected_roots.items():
    imported = sys.modules[module]
    module_file = getattr(imported, "__file__", None)
    if module_file is None:
        continue
    module_path = Path(module_file).resolve()
    try:
        module_path.relative_to(expected_root.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"{module} imported from {module_path}, expected under {expected_root}"
        ) from exc

agrimanager_path = Path(sys.modules["agrimanager"].__file__).resolve()
if repo_root not in agrimanager_path.parents:
    raise SystemExit(
        f"agrimanager imported from {agrimanager_path}, expected under {repo_root}"
    )

import torch
print("PYTHON_OK", sys.executable)
print("TORCH_OK", torch.__version__, "cuda_available=", torch.cuda.is_available())
PY
}

validate_activation_env() {
    local deploy_dir="$1"
    local repo_root="$2"
    local activate_file="$deploy_dir/activate_agrimanager.sh"

    require_file "$activate_file"
    bash - "$activate_file" "$PYTHON_BIN" "$repo_root" <<'BASH'
set -euo pipefail
activate_file="$1"
python_bin="$2"
repo_root="$3"

# shellcheck disable=SC1090
source "$activate_file"
"$python_bin" - "$repo_root" <<'PY'
from pathlib import Path
import importlib
import sys

repo_root = Path(sys.argv[1]).resolve()
external_root = repo_root.parent / "AgriManagerExternal"
expected = {
    "agrimanager": repo_root,
    "verl": repo_root / "verl",
    "gym_dssat_pdi": external_root / "gym_dssat_pdi" / "gym-dssat-pdi",
}

for name, expected_root in expected.items():
    module = importlib.import_module(name)
    module_file = Path(getattr(module, "__file__", "")).resolve()
    try:
        module_file.relative_to(expected_root.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"{name} imported from {module_file}, expected under {expected_root}"
        ) from exc
    print(f"ACTIVATION_IMPORT_OK {name} {module_file}")
PY
BASH
}

validate_deploy_dir() {
    local deploy_dir="$1"
    local repo_root
    local external_root
    local dssat_root
    local dssat_bin
    local timeout_cmd=()

    require_dir "$deploy_dir"
    deploy_dir="$(cd "$deploy_dir" && pwd)"
    repo_root="$deploy_dir/AgriManager"
    external_root="$deploy_dir/AgriManagerExternal"
    dssat_root="$repo_root/spack/gym-dssat-pdi"
    dssat_bin="$dssat_root/bin"

    require_dir "$repo_root"
    require_file "$repo_root/install.sh"
    require_dir "$repo_root/verl"
    require_dir "$external_root/WOFOSTGym"
    require_dir "$external_root/CyclesGym"
    require_dir "$dssat_bin"
    require_exec "$dssat_bin/run_dssat"
    require_exec "$dssat_bin/dscsm048"

    if [[ -L "$dssat_root" ]]; then
        local target
        target="$(readlink "$dssat_root")"
        case "$target" in
            "$repo_root"/spack/*) ;;
            *) die "DSSAT view symlink points outside deploy tree: $dssat_root -> $target" ;;
        esac
    fi

    if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
        die "Deployment validation needs an executable target Conda Python. Pass --python."
    fi

    log "Validating DSSAT profile paths"
    validate_dssat_profile "$repo_root" "$dssat_bin"

    log "Validating Python imports from deploy tree"
    validate_python_env "$repo_root"

    log "Validating activation script imports"
    validate_activation_env "$deploy_dir" "$repo_root"

    if [[ "$RUN_DSSAT_SMOKE" == "1" ]]; then
        log "Running DSSATEnv reset smoke test"
        if command -v timeout >/dev/null 2>&1; then
            timeout_cmd=(timeout "${TIMEOUT_SECONDS}s")
        else
            warn "timeout command not found; DSSAT smoke test will run without a timeout."
        fi
        "${timeout_cmd[@]}" bash "$SCRIPT_SELF" --_dssat-smoke "$repo_root" "$PYTHON_BIN"
    else
        log "Skipping DSSAT smoke test"
    fi

    log "Deployment validation passed: $deploy_dir"
}

if [[ -n "$ARCHIVE" ]]; then
    validate_archive "$ARCHIVE"
fi

if [[ -n "$DEPLOY_DIR" ]]; then
    validate_deploy_dir "$DEPLOY_DIR"
fi

log "All requested validation checks passed"
