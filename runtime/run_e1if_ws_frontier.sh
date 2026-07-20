#!/usr/bin/env bash
# 17th vertical: 8K FP8-KV frontier ascent with a chosen graph-pool scope.
# Usage: ./run_e1if_ws_frontier.sh <scope> <bl...>   e.g. global 60 64 72 80
# Stops at first failure (OOM boundary) and cleans wedged ranks.
set -uo pipefail
cd "$(dirname "$0")"
SCOPE=${1:?pool scope required}
shift

cleanup() {
  for h in titan064 titan065; do
    ssh "$h" 'pkill -f e1if_interleaved_bench.py 2>/dev/null; sleep 2; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2' || true
  done
}

for bl in "$@"; do
  echo "=== WS-FRONTIER scope=$SCOPE bl=$bl ctx=8192 ==="
  if ./run_e1if_ws_dual.sh timed fp8 fp8 "$SCOPE" 4 "$bl" 3 300 8192 132; then
    echo "=== WS-RESULT scope=$SCOPE bl=$bl OK ==="
  else
    echo "=== WS-RESULT scope=$SCOPE bl=$bl FAILED (likely OOM) ==="
    cleanup
    break
  fi
done
echo WS_FRONTIER_DONE
