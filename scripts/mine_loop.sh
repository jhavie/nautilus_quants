#!/bin/bash
# mine_loop.sh — Infinite alpha mining loop
#
# Runs `mine --rounds N` in a loop until Ctrl+C or quota exhausted.
# Each iteration is a single session with parallel generation + serial analysis.
#
# Usage:
#   cd ~/Sync/worktrees/feature-040-llm-alpha-mining
#   bash scripts/mine_loop.sh
#
# Stop: Ctrl+C

set -uo pipefail

CONFIG="config/cs/alpha_mining.yaml"
ROUNDS_PER_SESSION=5    # rounds per mine invocation
PYTHONPATH="src"
ITERATION=1

export PYTHONPATH

echo "================================================"
echo "Alpha Mining Loop — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config: ${CONFIG}"
echo "Rounds/session: ${ROUNDS_PER_SESSION}"
echo "Parallel: from YAML mining.parallel"
echo "Ctrl+C to stop"
echo "================================================"
echo ""

while true; do
    echo "--- Session ${ITERATION} ($(date '+%H:%M:%S')) ---"

    python -m nautilus_quants.alpha mine "$CONFIG" --rounds "$ROUNDS_PER_SESSION" 2>&1
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] Session ${ITERATION} failed (rc=${rc}). Waiting 30s..."
        sleep 30
    fi

    ITERATION=$((ITERATION + 1))
    echo ""
    sleep 5
done
