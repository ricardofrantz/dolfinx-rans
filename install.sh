#!/bin/bash
# install.sh — Fully automatic installer for dolfinx-rans
# Supports: macOS ARM64, Ubuntu x86_64
# Uses: conda-forge for FEniCSx, uv for Python package management

set -e  # Exit on error

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="fenicsx"
MINIFORGE_VERSION="24.11.3-2"
PYTHON_VERSION="3.11"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

detect_os() {
    OS=""
    ARCH=""
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        ARCH=$(uname -m)
        if [[ "$ARCH" != "arm64" ]]; then
            log_warn "Detected macOS but not ARM64 ($ARCH). Proceeding anyway..."
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        ARCH=$(uname -m)
        if [[ "$ARCH" != "x86_64" ]]; then
            log_warn "Detected Linux but not x86_64 ($ARCH). Proceeding anyway..."
        fi
    else
        log_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
    
    log_info "Detected: $OS $ARCH"
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

# ─────────────────────────────────────────────────────────────────────────────
# UV Installation
# ─────────────────────────────────────────────────────────────────────────────
install_uv() {
    if check_command uv; then
        log_success "uv already installed: $(uv --version)"
        return 0
    fi
    
    log_info "Installing uv..."
    
    if [[ "$OS" == "macos" ]]; then
        if check_command brew; then
            brew install uv
        else
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
    else
        # Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add to PATH for this session
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    if ! check_command uv; then
        log_error "Failed to install uv. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    
    log_success "uv installed: $(uv --version)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Conda Installation
# ─────────────────────────────────────────────────────────────────────────────
find_conda() {
    # Check various conda locations
    if check_command conda; then
        echo "$(which conda)"
        return 0
    elif check_command mamba; then
        echo "$(which mamba)"
        return 0
    elif [[ -f "/opt/homebrew/bin/conda" ]]; then
        echo "/opt/homebrew/bin/conda"
        return 0
    elif [[ -f "/opt/homebrew/bin/mamba" ]]; then
        echo "/opt/homebrew/bin/mamba"
        return 0
    elif [[ -f "$HOME/miniforge3/bin/conda" ]]; then
        echo "$HOME/miniforge3/bin/conda"
        return 0
    elif [[ -f "$HOME/miniforge3/bin/mamba" ]]; then
        echo "$HOME/miniforge3/bin/mamba"
        return 0
    elif [[ -f "$HOME/mambaforge/bin/conda" ]]; then
        echo "$HOME/mambaforge/bin/conda"
        return 0
    elif [[ -f "$HOME/mambaforge/bin/mamba" ]]; then
        echo "$HOME/mambaforge/bin/mamba"
        return 0
    elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
        echo "$HOME/anaconda3/bin/conda"
        return 0
    elif [[ -f "$HOME/miniconda3/bin/conda" ]]; then
        echo "$HOME/miniconda3/bin/conda"
        return 0
    fi
    return 1
}

install_miniforge() {
    log_info "Installing Miniforge (conda-forge distribution)..."
    
    cd "$HOME"
    
    # Determine the correct installer
    if [[ "$OS" == "macos" ]]; then
        if [[ "$ARCH" == "arm64" ]]; then
            INSTALLER="Miniforge3-${MINIFORGE_VERSION}-MacOSX-arm64.sh"
        else
            INSTALLER="Miniforge3-${MINIFORGE_VERSION}-MacOSX-x86_64.sh"
        fi
    else
        # Linux
        if [[ "$ARCH" == "x86_64" ]]; then
            INSTALLER="Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
        else
            INSTALLER="Miniforge3-${MINIFORGE_VERSION}-Linux-aarch64.sh"
        fi
    fi
    
    log_info "Downloading $INSTALLER..."
    curl -L -o "$INSTALLER" "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${INSTALLER}"
    
    log_info "Running installer..."
    bash "$INSTALLER" -b -p "$HOME/miniforge3"
    rm "$INSTALLER"
    
    log_success "Miniforge installed to $HOME/miniforge3"
}

setup_conda() {
    log_info "Setting up conda..."
    
    CONDA_BIN=$(find_conda) || true
    
    if [[ -z "$CONDA_BIN" ]]; then
        install_miniforge
        CONDA_BIN="$HOME/miniforge3/bin/conda"
    else
        log_success "Found conda: $CONDA_BIN"
    fi
    
    # Initialize conda for bash
    if [[ "$CONDA_BIN" == *"mamba"* ]]; then
        CONDA_CMD="mamba"
        eval "$("$CONDA_BIN" shell hook --shell bash)"
    else
        CONDA_CMD="conda"
        eval "$("$CONDA_BIN" shell.bash hook)"
    fi
    
    export CONDA_CMD
    log_success "Conda initialized"
}

# ─────────────────────────────────────────────────────────────────────────────
# FEniCSx Environment Setup
# ─────────────────────────────────────────────────────────────────────────────
create_fenicsx_env() {
    log_info "Setting up FEniCSx environment: $CONDA_ENV"
    
    # Check if environment exists
    if $CONDA_CMD env list | grep -q "^${CONDA_ENV} "; then
        log_warn "Environment '$CONDA_ENV' already exists"
        read -p "Recreate it? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing environment..."
            $CONDA_CMD env remove -n "$CONDA_ENV" -y
        else
            log_info "Using existing environment"
            $CONDA_CMD activate "$CONDA_ENV"
            return 0
        fi
    fi
    
    log_info "Creating new conda environment with FEniCSx..."
    
    # Create environment with all FEniCSx dependencies
    $CONDA_CMD create -n "$CONDA_ENV" -y \
        -c conda-forge \
        python="${PYTHON_VERSION}" \
        fenics-dolfinx \
        dolfinx_mpc \
        petsc4py \
        mpi4py \
        numpy \
        matplotlib
    
    $CONDA_CMD activate "$CONDA_ENV"
    
    log_success "FEniCSx environment created and activated"
}

# ─────────────────────────────────────────────────────────────────────────────
# Install dolfinx-rans
# ─────────────────────────────────────────────────────────────────────────────
install_dolfinx_rans() {
    log_info "Installing dolfinx-rans..."
    
    cd "$SCRIPT_DIR"
    
    # Use uv pip inside the conda environment
    # This is the key: uv manages Python packages, conda manages the system deps
    uv pip install -e "."
    
    log_success "dolfinx-rans installed"
}

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
verify_installation() {
    log_info "Verifying installation..."
    
    python -c "
import sys
errors = []
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
    print('All imports OK!')
    print(f'  ✓ DOLFINx version: {dolfinx.__version__}')
    sys.exit(0)
" || {
        log_error "Installation verification failed"
        exit 1
    }
    
    log_success "All dependencies verified!"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  dolfinx-rans Automatic Installer"
    echo "  OS: $(uname -s) | Arch: $(uname -m)"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    
    # Step 1: Detect OS
    detect_os
    
    # Step 2: Install uv
    install_uv
    
    # Step 3: Setup conda (for FEniCSx)
    setup_conda
    
    # Step 4: Create FEniCSx environment
    create_fenicsx_env
    
    # Step 5: Install dolfinx-rans with uv
    install_dolfinx_rans
    
    # Step 6: Verify
    verify_installation
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  Installation Complete!"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Usage:"
    echo "  conda activate $CONDA_ENV"
    echo "  ./run.sh                    # Run with default config (Re_τ=590)"
    echo "  ./run.sh 4                  # Run with 4 MPI processes"
    echo "  ./run.sh path/to/config.json # Run with custom config"
    echo ""
    echo "Or directly:"
    echo "  python -m dolfinx_rans examples/channel_re590.json"
    echo ""
}

main "$@"
