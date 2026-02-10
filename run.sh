#!/bin/bash
# run.sh — Run dolfinx-rans k-ω channel flow solver
#
# Usage:
#   ./run.sh                       # Serial, high-Re Nek-like benchmark (default)
#   ./run.sh 8                     # 8 MPI processes, high-Re Nek-like benchmark
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
    CONFIG_ARG="${2:-nek}"
else
    CONFIG_ARG="${1:-nek}"
fi

# Select config file
case "$CONFIG_ARG" in
    nek|highre|re125k)
        CONFIG="$SCRIPT_DIR/examples/channel_nek_re125k_like.json"
        echo "Running: high-Re Nek-like channel benchmark"
        ;;
    re590)
        CONFIG="$SCRIPT_DIR/examples/channel_re590.json"
        echo "Running: Re_τ=590 channel flow (legacy DNS-oriented case)"
        ;;
    *.json)
        CONFIG="$CONFIG_ARG"
        echo "Running: Custom config $CONFIG"
        ;;
    *)
        echo "ERROR: Unknown config '$CONFIG_ARG'"
        echo "Usage: ./run.sh [NPROCS] [nek|re590|path/to/config.json]"
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
# Post-run regression checks (config-driven)
# ─────────────────────────────────────────────────────────────────────────────
python - <<PY || exit 1
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

cfg_path = Path("$CONFIG")
cfg = json.loads(cfg_path.read_text())
out_dir = Path(cfg["solve"]["out_dir"])
re_tau = float(cfg["nondim"]["Re_tau"])
bench = dict(cfg.get("benchmark", {}))


def parse_bounds(val, key):
    if val is None:
        return None
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        raise ValueError(f"benchmark.{key} must be a [min, max] list")
    lo, hi = float(val[0]), float(val[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError(f"benchmark.{key} must satisfy finite min <= max")
    return lo, hi


u_bounds = parse_bounds(bench.get("gate_u_bulk_bounds"), "gate_u_bulk_bounds")
tau_bounds = parse_bounds(bench.get("gate_tau_wall_bounds"), "gate_tau_wall_bounds")
ref_csv = bench.get("reference_profile_csv")
if isinstance(ref_csv, str) and not ref_csv.strip():
    ref_csv = None
u_rmse_max = bench.get("u_plus_rmse_max")
k_rmse_max = bench.get("k_plus_rmse_max")

# Backward-compatible default gate for legacy Re_tau=590 config.
if u_bounds is None and tau_bounds is None and ref_csv is None and abs(re_tau - 590.0) < 1e-12:
    u_bounds = (14.0, 18.0)
    tau_bounds = (0.90, 1.10)

if u_bounds is None and tau_bounds is None and ref_csv is None:
    print("Skipping regression gate (no benchmark thresholds configured).")
    sys.exit(0)

violations = []
summary_parts = []

if u_bounds is not None or tau_bounds is not None:
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
    summary_parts.append(f"U_bulk={u_bulk:.4f}")
    summary_parts.append(f"tau_wall={tau_wall:.4f}")

    if u_bounds is not None and not (u_bounds[0] <= u_bulk <= u_bounds[1]):
        violations.append(
            f"U_bulk={u_bulk:.4f} not in [{u_bounds[0]:.2f}, {u_bounds[1]:.2f}]"
        )
    if tau_bounds is not None and not (tau_bounds[0] <= tau_wall <= tau_bounds[1]):
        violations.append(
            f"tau_wall={tau_wall:.4f} not in [{tau_bounds[0]:.2f}, {tau_bounds[1]:.2f}]"
        )

if ref_csv is not None:
    profiles_path = out_dir / "profiles.csv"
    if not profiles_path.exists():
        print(f"ERROR: Regression gate failed: missing profile file {profiles_path}")
        sys.exit(1)

    ref_path = Path(ref_csv)
    if not ref_path.is_absolute():
        ref_path = (cfg_path.parent / ref_path).resolve()
    if not ref_path.exists():
        print(f"ERROR: Regression gate failed: reference profile not found {ref_path}")
        sys.exit(1)

    with profiles_path.open() as f:
        ours = list(csv.DictReader(f))
    with ref_path.open() as f:
        ref = list(csv.DictReader(f))
    if not ours:
        raise ValueError(f"{profiles_path} is empty")
    if not ref:
        raise ValueError(f"{ref_path} is empty")

    for req in ("y_plus", "u_plus"):
        if req not in ours[0]:
            raise ValueError(f"{profiles_path} missing required column '{req}'")
        if req not in ref[0]:
            raise ValueError(f"{ref_path} missing required column '{req}'")

    y_ours = np.array([float(r["y_plus"]) for r in ours])
    u_ours = np.array([float(r["u_plus"]) for r in ours])
    y_ref = np.array([float(r["y_plus"]) for r in ref])
    u_ref = np.array([float(r["u_plus"]) for r in ref])

    y_min = max(float(np.min(y_ours)), float(np.min(y_ref)))
    y_max = min(float(np.max(y_ours)), float(np.max(y_ref)))
    mask = (y_ref >= y_min) & (y_ref <= y_max)
    if int(np.count_nonzero(mask)) < 5:
        raise ValueError("Insufficient y_plus overlap between computed and reference profiles")

    u_interp = np.interp(y_ref[mask], y_ours, u_ours)
    u_rmse = float(np.sqrt(np.mean((u_interp - u_ref[mask]) ** 2)))
    summary_parts.append(f"u_plus_rmse={u_rmse:.4f}")

    if u_rmse_max is not None and u_rmse > float(u_rmse_max):
        violations.append(
            f"u_plus RMSE={u_rmse:.4f} exceeds limit {float(u_rmse_max):.4f}"
        )

    if k_rmse_max is not None:
        if "k_plus" not in ref[0]:
            raise ValueError(f"{ref_path} missing required column 'k_plus' for k RMSE gate")
        if "k_plus" not in ours[0]:
            raise ValueError(f"{profiles_path} missing required column 'k_plus' for k RMSE gate")
        k_ours = np.array([float(r["k_plus"]) for r in ours])
        k_ref = np.array([float(r["k_plus"]) for r in ref])
        k_interp = np.interp(y_ref[mask], y_ours, k_ours)
        k_rmse = float(np.sqrt(np.mean((k_interp - k_ref[mask]) ** 2)))
        summary_parts.append(f"k_plus_rmse={k_rmse:.4f}")
        if k_rmse > float(k_rmse_max):
            violations.append(
                f"k_plus RMSE={k_rmse:.4f} exceeds limit {float(k_rmse_max):.4f}"
            )

if violations:
    print("ERROR: Regression gate failed:")
    for item in violations:
        print(f"  - {item}")
    sys.exit(1)

details = ", ".join(summary_parts) if summary_parts else "no checks run"
print(f"Regression gate passed: {details}")
PY

echo "───────────────────────────────────────────────────────────────────────"
echo "Done! Check results in the output directory specified in config."
