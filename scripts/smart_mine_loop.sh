#!/bin/bash
# smart_mine_loop.sh — Usage-aware alpha mining loop
#
# Queries Anthropic OAuth Usage API before/after each mine run,
# calculates per-run consumption, and loops until quota approaches limit.
#
# Requires:
#   - macOS Keychain with Claude Code credentials
#   - HTTP proxy (aws-sg-server) at 127.0.0.1:8888
#   - python3, curl, jq-like parsing via python3
#
# Usage:
#   cd ~/Sync/worktrees/feature-040-llm-alpha-mining
#   bash scripts/smart_mine_loop.sh
#   bash scripts/smart_mine_loop.sh --rounds 5 --max-util 90
#   bash scripts/smart_mine_loop.sh --dry-run
#
# Stop: Ctrl+C

set -uo pipefail

# ── Defaults ──
CONFIG="config/cs/alpha_mining.yaml"
ROUNDS=10
MAX_UTIL=85
QUOTA_KEY="five_hour"
PROXY="http://127.0.0.1:8888"
DRY_RUN=false
PYTHONPATH="src"
export PYTHONPATH

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2"; shift 2 ;;
        --rounds)   ROUNDS="$2"; shift 2 ;;
        --max-util) MAX_UTIL="$2"; shift 2 ;;
        --quota)    QUOTA_KEY="$2"; shift 2 ;;
        --proxy)    PROXY="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --config PATH       Mining config (default: $CONFIG)"
            echo "  --rounds N          Rounds per mine run (default: $ROUNDS)"
            echo "  --max-util PCT      Stop at this utilization % (default: $MAX_UTIL)"
            echo "  --quota KEY         Quota window: five_hour|seven_day (default: $QUOTA_KEY)"
            echo "  --proxy URL         HTTP proxy (default: $PROXY)"
            echo "  --dry-run           Query usage only, don't mine"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Keychain token ──
get_token() {
    local cred
    cred=$(security find-generic-password -s "Claude Code-credentials" -w 2>/dev/null) || {
        echo "ERROR: Cannot read Claude credentials from Keychain." >&2
        echo "Run 'claude login' first." >&2
        return 1
    }
    echo "$cred" | python3 -c "import sys,json; print(json.load(sys.stdin)['claudeAiOauth']['accessToken'])" 2>/dev/null || {
        echo "ERROR: Failed to parse access token from credentials." >&2
        return 1
    }
}

# ── Query Usage API ──
# Output: "utilization resets_at"  e.g. "67.0 2026-04-04T15:00:00Z"
query_usage() {
    local token="$1"
    local raw
    raw=$(curl -s --proxy "$PROXY" \
        "https://api.anthropic.com/api/oauth/usage" \
        -H "Authorization: Bearer $token" \
        -H "Accept: application/json" \
        -H "anthropic-beta: oauth-2025-04-20" 2>/dev/null)

    if [ -z "$raw" ]; then
        echo "ERROR: Empty response from usage API" >&2
        return 1
    fi

    echo "$raw" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)

if 'error' in d:
    print(f\"ERROR: {d['error'].get('message', d['error'])}\", file=sys.stderr)
    sys.exit(1)

key = '$QUOTA_KEY'
q = d.get(key)
if q is None:
    print(f'ERROR: quota key \"{key}\" not found in response', file=sys.stderr)
    sys.exit(1)

util = q.get('utilization', 0) or 0
resets = q.get('resets_at', '')
print(f'{util} {resets}')
" 2>/dev/null
}

# ── Format reset time as "in Xh Ym" ──
format_reset() {
    local resets_at="$1"
    python3 -c "
from datetime import datetime, timezone
try:
    r = '$resets_at'.replace('+00:00', '+0000').replace('Z', '+0000')
    # Handle fractional seconds
    if '.' in r:
        fmt = '%Y-%m-%dT%H:%M:%S.%f%z'
    else:
        fmt = '%Y-%m-%dT%H:%M:%S%z'
    dt = datetime.strptime(r, fmt)
    now = datetime.now(timezone.utc)
    delta = dt - now
    secs = max(int(delta.total_seconds()), 0)
    h, m = secs // 3600, (secs % 3600) // 60
    if h > 0:
        print(f'{h}h {m}m')
    elif m > 0:
        print(f'{m}m')
    else:
        print('soon')
except Exception as e:
    print('unknown')
" 2>/dev/null
}

# ── Main ──
echo "=== Smart Alpha Mining Loop ==="
echo "Config: ${CONFIG}"
echo "Rounds: ${ROUNDS} | Max utilization: ${MAX_UTIL}% | Quota: ${QUOTA_KEY}"
echo "Proxy: ${PROXY}"
echo ""

# Get token
TOKEN=$(get_token) || exit 1

# Initial usage check
USAGE_LINE=$(query_usage "$TOKEN") || { echo "Failed to query usage."; exit 1; }
UTIL=$(echo "$USAGE_LINE" | awk '{print $1}')
RESETS_AT=$(echo "$USAGE_LINE" | awk '{print $2}')
RESET_IN=$(format_reset "$RESETS_AT")

echo "Current usage: ${UTIL}% (resets in ${RESET_IN})"

# Check if already over limit
OVER=$(python3 -c "print('yes' if $UTIL >= $MAX_UTIL else 'no')")
if [ "$OVER" = "yes" ]; then
    echo "Already at ${UTIL}% >= ${MAX_UTIL}% limit. Wait for reset (in ${RESET_IN})."
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    HEADROOM=$(python3 -c "print(f'{$MAX_UTIL - $UTIL:.1f}')")
    echo "Headroom: ${HEADROOM}% to limit"
    echo "[DRY RUN] No mining executed."
    exit 0
fi

ITERATION=1
AVG_DELTA=0
TOTAL_SESSIONS=0

while true; do
    echo ""
    echo "--- Session ${ITERATION} ---"

    # Pre-run usage
    USAGE_LINE=$(query_usage "$TOKEN") || { echo "Usage query failed. Stopping."; break; }
    PRE_UTIL=$(echo "$USAGE_LINE" | awk '{print $1}')
    RESETS_AT=$(echo "$USAGE_LINE" | awk '{print $2}')
    RESET_IN=$(format_reset "$RESETS_AT")
    echo "[${ITERATION}] Usage: ${PRE_UTIL}% (resets in ${RESET_IN})"

    # Check limit before running
    OVER=$(python3 -c "print('yes' if $PRE_UTIL >= $MAX_UTIL else 'no')")
    if [ "$OVER" = "yes" ]; then
        echo "[${ITERATION}] Reached ${PRE_UTIL}% >= ${MAX_UTIL}% limit. Stopping."
        echo "    Resets in: ${RESET_IN}"
        break
    fi

    # Run mine
    echo "[${ITERATION}] Running mine (${ROUNDS} rounds)..."
    START_TS=$(date +%s)

    python -m nautilus_quants.alpha mine "$CONFIG" --rounds "$ROUNDS" 2>&1
    RC=$?

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))
    echo "[${ITERATION}] Done in ${ELAPSED}s (exit code: ${RC})"

    if [ $RC -ne 0 ]; then
        echo "[${ITERATION}] Mine failed. Waiting 30s before retry..."
        sleep 30
    fi

    # Post-run usage
    USAGE_LINE=$(query_usage "$TOKEN") || { echo "Usage query failed after run. Stopping."; break; }
    POST_UTIL=$(echo "$USAGE_LINE" | awk '{print $1}')
    RESETS_AT=$(echo "$USAGE_LINE" | awk '{print $2}')
    RESET_IN=$(format_reset "$RESETS_AT")

    # Calculate delta
    DELTA=$(python3 -c "print(f'{$POST_UTIL - $PRE_UTIL:.1f}')")
    HEADROOM=$(python3 -c "print(f'{$MAX_UTIL - $POST_UTIL:.1f}')")

    # Update running average
    TOTAL_SESSIONS=$((TOTAL_SESSIONS + 1))
    AVG_DELTA=$(python3 -c "
prev_avg = $AVG_DELTA
n = $TOTAL_SESSIONS
delta = $POST_UTIL - $PRE_UTIL
# Exponential moving average (weight recent more)
if n == 1:
    print(f'{delta:.1f}')
else:
    ema = 0.6 * delta + 0.4 * prev_avg
    print(f'{ema:.1f}')
")

    # Estimate remaining runs
    if python3 -c "exit(0 if float('$AVG_DELTA') > 0.1 else 1)"; then
        EST_REMAINING=$(python3 -c "
headroom = $MAX_UTIL - $POST_UTIL
avg = $AVG_DELTA
est = int(headroom / avg) if avg > 0 else 0
print(est)
")
    else
        EST_REMAINING="?"
    fi

    echo "[${ITERATION}] Usage: ${POST_UTIL}% | delta: ${DELTA}% | avg_delta: ${AVG_DELTA}%"
    echo "[${ITERATION}] Remaining: ~${EST_REMAINING} runs (${HEADROOM}% headroom to ${MAX_UTIL}%)"

    # Check if next run would exceed limit
    WOULD_EXCEED=$(python3 -c "
next_est = $POST_UTIL + float('$AVG_DELTA')
print('yes' if next_est >= $MAX_UTIL else 'no')
")
    if [ "$WOULD_EXCEED" = "yes" ]; then
        echo ""
        echo "[${ITERATION}] Next run would exceed ${MAX_UTIL}% limit. Stopping."
        echo "    Resets in: ${RESET_IN}"
        break
    fi

    ITERATION=$((ITERATION + 1))
    sleep 3
done

echo ""
echo "=== Summary ==="
echo "Total sessions: ${TOTAL_SESSIONS}"
echo "Avg delta per session: ${AVG_DELTA}%"
echo "Final usage: ${POST_UTIL:-$UTIL}%"
echo "==============="
