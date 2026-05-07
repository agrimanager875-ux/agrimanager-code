#!/bin/bash
# Wrapper script that sets DSSAT environment variables without changing Python
# This allows using miniconda Python (with vLLM) while DSSAT subprocess can still work

AGRIMANAGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# DSSAT subprocess needs these for PDI pycall plugin (the subprocess uses spack Python)
# DO NOT set PYTHONPATH - it would conflict with miniconda's packages
# The subprocess inherits its own Python environment via PDI pycall

DSSAT_GYM_PATH="${DSSAT_GYM_PATH:-$AGRIMANAGER_ROOT/spack/gym-dssat-pdi}"
export PDI_PLUGIN_PATH="${PDI_PLUGIN_PATH:-$DSSAT_GYM_PATH/lib}"
export LD_LIBRARY_PATH="$DSSAT_GYM_PATH/lib:$DSSAT_GYM_PATH/lib64:${LD_LIBRARY_PATH:-}"

# Add spack bins to PATH (for run_dssat, dscsm048)
export PATH="$DSSAT_GYM_PATH/bin:${PATH}"

# Use the active Python unless the caller provides an explicit interpreter.
PYTHON="${AGRIMANAGER_PYTHON:-$(command -v python3)}"

echo "Using Python: $PYTHON"
echo "LD_LIBRARY_PATH set for DSSAT subprocess"

# Run the command
exec "$PYTHON" "$@"
