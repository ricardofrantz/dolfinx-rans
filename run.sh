#!/bin/bash
# run.sh — Run dolfinx-rans k-ω channel flow solver
#
# Usage:
#   ./run.sh                       # Serial, Re_τ=590 (default)
#   ./run.sh 8                     # 8 MPI processes, Re_τ=590
#   ./run.sh path/to/config.json   # Serial, custom config
#   ./run.sh 4 path/to/config.json # 4 MPI processes, custom config

set -e  # Exit on error

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="fenicsx"

# ─────────────────────────────────────────────────────────────────────────────
# Activate conda environment
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  dolfinx-rans: RANS k-ω Channel Flow Solver"
echo "═══════════════════════════════════════════════════════════════════════"

# Find and source conda
if [ -f "/opt/homebrew/bin/conda" ]; then
    eval "$(/opt/homebrew/bin/conda shell.bash hook)"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda installation"
    exit 1
fi

echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

# ─────────────────────────────────────────────────────────────────────────────
# Install package if needed
# ─────────────────────────────────────────────────────────────────────────────
if ! python -c "import dolfinx_rans" 2>/dev/null; then
    echo "Installing dolfinx-rans..."
    pip install -e "$SCRIPT_DIR" --quiet
fi

# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments: [NPROCS] [CONFIG]
# ─────────────────────────────────────────────────────────────────────────────
NPROCS=1
CONFIG_ARG=""

# Check if first arg is a number (MPI processes)
if [[ "$1" =~ ^[0-9]+$ ]]; then
    NPROCS="$1"
    CONFIG_ARG="${2:-re590}"
else
    CONFIG_ARG="${1:-re590}"
fi

# Select config file
case "$CONFIG_ARG" in
    re590)
        CONFIG="$SCRIPT_DIR/examples/channel_re590.json"
        echo "Running: Re_τ=590 channel flow"
        ;;
    *.json)
        CONFIG="$CONFIG_ARG"
        echo "Running: Custom config $CONFIG"
        ;;
    *)
        echo "ERROR: Unknown config '$CONFIG_ARG'"
        echo "Usage: ./run.sh [NPROCS] [re590|path/to/config.json]"
        exit 1
        ;;
esac

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "Config: $CONFIG"
if [ "$NPROCS" -gt 1 ]; then
    echo "MPI processes: $NPROCS"
fi
echo "───────────────────────────────────────────────────────────────────────"

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────
echo "Running pre-flight checks..."

# 1. Syntax check
if ! python -m py_compile "$SCRIPT_DIR/src/dolfinx_rans/solver.py" 2>/dev/null; then
    echo "ERROR: Python syntax error in solver.py"
    python -m py_compile "$SCRIPT_DIR/src/dolfinx_rans/solver.py"
    exit 1
fi

# 2. Import check
python -c "
import sys
errors = []

# Core dependencies
try:
    from mpi4py import MPI
except ImportError as e:
    errors.append(f'mpi4py: {e}')

try:
    from petsc4py import PETSc
except ImportError as e:
    errors.append(f'petsc4py: {e}')

try:
    import dolfinx
except ImportError as e:
    errors.append(f'dolfinx: {e}')

try:
    import dolfinx_mpc
except ImportError as e:
    errors.append(f'dolfinx_mpc: {e}')

# Package import
try:
    import dolfinx_rans
except ImportError as e:
    errors.append(f'dolfinx_rans: {e}')

if errors:
    print('Import errors:')
    for err in errors:
        print(f'  ✗ {err}')
    sys.exit(1)
else:
    print('  ✓ All imports OK')
" || exit 1

# 3. Config validation
python -c "
import json
import sys
with open('$CONFIG') as f:
    cfg = json.load(f)
required = ['geom', 'nondim', 'turb', 'solve']
missing = [k for k in required if k not in cfg]
if missing:
    print(f'ERROR: Config missing sections: {missing}')
    sys.exit(1)
print('  ✓ Config valid')
" || exit 1

# 4. Quick MPI check (only if using multiple processes)
if [ "$NPROCS" -gt 1 ]; then
    mpirun -np 2 python -c "from mpi4py import MPI; print(f'  ✓ MPI rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size} OK')" 2>/dev/null || {
        echo "ERROR: MPI not working properly"
        exit 1
    }
fi

echo "Pre-flight checks passed!"
echo "───────────────────────────────────────────────────────────────────────"

# ─────────────────────────────────────────────────────────────────────────────
# Run solver
# ─────────────────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
if [ "$NPROCS" -gt 1 ]; then
    mpirun -np "$NPROCS" python -m dolfinx_rans "$CONFIG"
else
    python -m dolfinx_rans "$CONFIG"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Post-run regression check (Re_τ = 590 reference case)
# ─────────────────────────────────────────────────────────────────────────────
python - <<PY || exit 1
import csv
import json
import sys
from pathlib import Path

cfg = json.loads(Path("$CONFIG").read_text())
re_tau = float(cfg["nondim"]["Re_tau"])
if abs(re_tau - 590.0) > 1e-12:
    print(f"Skipping regression gate (Re_τ={re_tau:g}, gate targets Re_τ=590).")
    sys.exit(0)

out_dir = Path(cfg["solve"]["out_dir"])
history = out_dir / "history.csv"
if not history.exists():
    print(f"ERROR: Regression gate failed: missing history file {history}")
    sys.exit(1)

with history.open() as f:
    rows = list(csv.DictReader(f))
if not rows:
    print(f"ERROR: Regression gate failed: empty history file {history}")
    sys.exit(1)

last = rows[-1]
u_bulk = float(last["U_bulk"])
tau_wall = float(last["tau_wall"])

u_bounds = (14.0, 18.0)
tau_bounds = (0.90, 1.10)

violations = []
if not (u_bounds[0] <= u_bulk <= u_bounds[1]):
    violations.append(f"U_bulk={u_bulk:.4f} not in [{u_bounds[0]:.2f}, {u_bounds[1]:.2f}]")
if not (tau_bounds[0] <= tau_wall <= tau_bounds[1]):
    violations.append(f"tau_wall={tau_wall:.4f} not in [{tau_bounds[0]:.2f}, {tau_bounds[1]:.2f}]")

if violations:
    print("ERROR: Regression gate failed:")
    for v in violations:
        print(f"  - {v}")
    print(f"  Source row: iter={last.get('iter', '?')}, dt={last.get('dt', '?')}")
    sys.exit(1)

print(
    "Regression gate passed: "
    f"U_bulk={u_bulk:.4f}, tau_wall={tau_wall:.4f}"
)
PY

echo "───────────────────────────────────────────────────────────────────────"
echo "Done! Check results in the output directory specified in config."
