#!/usr/bin/env bash
# E1F B sweep driver: runs runtime/run_e1f_dual.sh once per local-batch value.
# Check plan: bitwise graph-vs-eager twin at B=1 and at the largest B whose
# eager twin still fits (memory permitting); feasibility-first (check off)
# for the larger points.  Usage: ./run_sweep.sh [B values...]
set -uo pipefail
cd "$(dirname "$0")/../../runtime"

declare -A CHECK=( [1]=bitwise [8]=off [32]=off [64]=off [128]=bitwise [192]=off [256]=off )
BS=("$@")
[ ${#BS[@]} -eq 0 ] && BS=(8 32 64 128 192 256)

for b in "${BS[@]}"; do
  mode=${CHECK[$b]:-off}
  echo "===== E1F sweep: B=$b check=$mode ====="
  ./run_e1f_dual.sh "$b" "$mode" 3 300 2048 || echo "===== B=$b FAILED (continuing) ====="
done
echo SWEEP_DONE
