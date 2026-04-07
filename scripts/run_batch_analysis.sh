#!/bin/bash
# Batch factor analysis runner — runs all factor libraries sequentially
# to avoid DuckDB lock conflicts.
#
# Usage: bash scripts/run_batch_analysis.sh [alpha101|ta|alpha158|alpha191|all]

set -euo pipefail
cd "$(dirname "$0")/.."

ENV="test"
TARGET="${1:-all}"

run_analysis() {
    local name="$1"
    local config="$2"
    echo "========================================"
    echo "Starting $name analysis..."
    echo "========================================"
    python3 -m nautilus_quants.alpha analyze "$config" --env "$ENV" -v
    echo ""
    echo "$name analysis completed."
    echo ""
}

if [[ "$TARGET" == "alpha101" || "$TARGET" == "all" ]]; then
    run_analysis "Alpha101 (52 factors)" "config/alpha_batch/analyze_alpha101.yaml"
fi

if [[ "$TARGET" == "ta" || "$TARGET" == "all" ]]; then
    run_analysis "TA (29 factors)" "config/alpha_batch/analyze_ta.yaml"
fi

if [[ "$TARGET" == "alpha158" || "$TARGET" == "all" ]]; then
    run_analysis "Alpha158 (157 factors)" "config/alpha_batch/analyze_alpha158.yaml"
fi

if [[ "$TARGET" == "alpha191" || "$TARGET" == "all" ]]; then
    run_analysis "Alpha191 (128 factors)" "config/alpha_batch/analyze_alpha191.yaml"
fi

echo "========================================"
echo "All analyses complete. Verifying registry..."
echo "========================================"
python3 -m nautilus_quants.alpha list --env "$ENV"
echo ""
echo "Total factors in registry:"
python3 -c "
from nautilus_quants.alpha.registry.database import RegistryDatabase
from nautilus_quants.alpha.registry.environment import resolve_env
env = resolve_env('$ENV')
db = RegistryDatabase.for_environment(env)
row = db.fetch_one('SELECT COUNT(*) FROM factors')
metrics = db.fetch_one('SELECT COUNT(*) FROM alpha_analysis_metrics')
print(f'  Factors: {row[0]}')
print(f'  Metric records: {metrics[0]}')
db.close()
"
