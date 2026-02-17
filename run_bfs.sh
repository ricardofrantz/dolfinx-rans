#!/bin/bash
# run_bfs.sh — Run dolfinx-rans backward-facing step solver
#
# Usage:
#   ./run_bfs.sh                           # Serial, default BFS config
#   ./run_bfs.sh 8                         # 8 MPI processes, default BFS config
#   ./run_bfs.sh path/to/config.jsonc        # Serial, custom config
#   ./run_bfs.sh 4 path/to/config.jsonc      # 4 MPI processes, custom config

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="$SCRIPT_DIR/bfs/bfs.jsonc"

# Parse arguments: [NPROCS] [CONFIG]
NPROCS=1
CONFIG_ARG=""

if [[ "$1" =~ ^[0-9]+$ ]]; then
    NPROCS="$1"
    CONFIG_ARG="${2:-$DEFAULT_CONFIG}"
else
    CONFIG_ARG="${1:-$DEFAULT_CONFIG}"
fi

if [[ "$CONFIG_ARG" == "default" || "$CONFIG_ARG" == "bfs" ]]; then
    CONFIG="$DEFAULT_CONFIG"
else
    CONFIG="$CONFIG_ARG"
fi

if [[ "$CONFIG" != /* ]]; then
    CONFIG="$SCRIPT_DIR/$CONFIG"
fi
CONFIG="$(realpath "$CONFIG")"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

echo "Using BFS config: $CONFIG"

echo "Running through run_channel.sh..."
"$SCRIPT_DIR/run_channel.sh" "$NPROCS" "$CONFIG"
