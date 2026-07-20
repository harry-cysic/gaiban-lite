#!/usr/bin/env bash
# FP8-KV E1IF frontier scan driver (fifteenth vertical).
# 8K: bl 56/64/72/80 ascending until first failure (OOM boundary), then 2K 96/128,
# plus one idx-bf16 attribution point at 8K bl64.  Cleans up wedged ranks after
# a failed run (E1F lesson: OOM leaves the 16-rank collective stuck).
set -uo pipefail
cd "$(dirname "$0")"

cleanup() {
  for h in titan064 titan065; do
    ssh "$h" 'pkill -f e1if_interleaved_bench.py 2>/dev/null; sleep 2; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2' || true
  done
}

run_cfg() {  # kv idx bl start
  local kv=$1 idx=$2 bl=$3 start=$4
  echo "=== FRONTIER kv=$kv idx=$idx bl=$bl ctx=$start ==="
  if ./run_e1if_kv_dual.sh timed "$kv" "$idx" 4 "$bl" 3 300 "$start" 132; then
    echo "=== RESULT kv=$kv idx=$idx bl=$bl ctx=$start OK ==="
    return 0
  else
    echo "=== RESULT kv=$kv idx=$idx bl=$bl ctx=$start FAILED (likely OOM) ==="
    cleanup
    return 1
  fi
}

# 8K ascending to OOM
for bl in 56 64 72 80; do
  run_cfg fp8 fp8 "$bl" 8192 || break
done
# 2K high-bl points
for bl in 96 128; do
  run_cfg fp8 fp8 "$bl" 2048 || true
done
# attribution point: latent-fp8 only (indexer bf16) at 8K bl64
run_cfg fp8 bf16 64 8192 || true
echo FRONTIER_SCAN_DONE
