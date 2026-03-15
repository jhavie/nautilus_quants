#!/bin/bash
# Cross-platform backtest result comparison
# Usage: ./scripts/verify_cross_platform.sh <baseline_dir> <docker_dir>
#
# Compares factor_values_report.csv and account_report.csv
# between two backtest output directories.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <baseline_dir> <docker_dir>"
    echo "Example: $0 logs/performance/20260315_160510 logs/performance/20260315_170000"
    exit 1
fi

BASELINE="$1"
DOCKER_OUTPUT="$2"

# Validate directories
for dir in "$BASELINE" "$DOCKER_OUTPUT"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Directory not found: $dir"
        exit 1
    fi
done

echo "=== Baseline:  $BASELINE ==="
echo "=== Docker:    $DOCKER_OUTPUT ==="
echo ""

# --- factor_values_report.csv ---
FACTOR_BASE="$BASELINE/factor_values_report.csv"
FACTOR_DOCKER="$DOCKER_OUTPUT/factor_values_report.csv"

if [ -f "$FACTOR_BASE" ] && [ -f "$FACTOR_DOCKER" ]; then
    echo "=== Comparing factor_values_report.csv ==="
    BASE_LINES=$(wc -l < "$FACTOR_BASE")
    DOCKER_LINES=$(wc -l < "$FACTOR_DOCKER")
    echo "  Baseline lines: $BASE_LINES"
    echo "  Docker lines:   $DOCKER_LINES"

    DIFF_OUTPUT=$(diff <(sort "$FACTOR_BASE") <(sort "$FACTOR_DOCKER") || true)
    if [ -z "$DIFF_OUTPUT" ]; then
        echo "  RESULT: IDENTICAL"
    else
        DIFF_LINES=$(echo "$DIFF_OUTPUT" | wc -l)
        echo "  RESULT: DIFFERS ($DIFF_LINES diff lines)"
        echo "  First 20 diff lines:"
        echo "$DIFF_OUTPUT" | head -20
    fi
else
    echo "WARNING: factor_values_report.csv missing in one or both directories"
    [ ! -f "$FACTOR_BASE" ] && echo "  Missing: $FACTOR_BASE"
    [ ! -f "$FACTOR_DOCKER" ] && echo "  Missing: $FACTOR_DOCKER"
fi

echo ""

# --- account_report.csv ---
ACCOUNT_BASE="$BASELINE/account_report.csv"
ACCOUNT_DOCKER="$DOCKER_OUTPUT/account_report.csv"

if [ -f "$ACCOUNT_BASE" ] && [ -f "$ACCOUNT_DOCKER" ]; then
    echo "=== Comparing account_report.csv ==="
    BASE_LINES=$(wc -l < "$ACCOUNT_BASE")
    DOCKER_LINES=$(wc -l < "$ACCOUNT_DOCKER")
    echo "  Baseline lines: $BASE_LINES"
    echo "  Docker lines:   $DOCKER_LINES"

    # Compare first 5 columns (timestamp, total, locked, free, ...)
    DIFF_OUTPUT=$(diff \
        <(cut -d',' -f1-5 "$ACCOUNT_BASE") \
        <(cut -d',' -f1-5 "$ACCOUNT_DOCKER") || true)
    if [ -z "$DIFF_OUTPUT" ]; then
        echo "  RESULT: IDENTICAL (columns 1-5)"
    else
        DIFF_LINES=$(echo "$DIFF_OUTPUT" | wc -l)
        echo "  RESULT: DIFFERS ($DIFF_LINES diff lines in columns 1-5)"
        echo "  First 20 diff lines:"
        echo "$DIFF_OUTPUT" | head -20
    fi
else
    echo "WARNING: account_report.csv missing in one or both directories"
    [ ! -f "$ACCOUNT_BASE" ] && echo "  Missing: $ACCOUNT_BASE"
    [ ! -f "$ACCOUNT_DOCKER" ] && echo "  Missing: $ACCOUNT_DOCKER"
fi

echo ""
echo "=== Verification Complete ==="
