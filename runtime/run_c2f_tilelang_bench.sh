#!/usr/bin/env bash
# C2F 21st vertical: same-口径 prefill throughput re-measurement with the
# tilelang prefill sparse core (11 layers L11-21, iters 5 / warmup 2).
# Usage: ./run_c2f_tilelang_bench.sh <chunk> <sparse-backend> <round> [extra args...]
#   e.g. ./run_c2f_tilelang_bench.sh 8192 tilelang 1
set -euo pipefail
cd "$(dirname "$0")"

CHUNK=${1:?chunk}
BACKEND=${2:?backend: torch|tilelang}
ROUND=${3:-1}
shift 3 || true
HOST=titan064
TAG="c2f-tl-chunk${CHUNK}-${BACKEND}-r${ROUND}"
OUT="out-$TAG"
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime + reference kernel to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c2f_prefill_stage_bench.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

echo "== nvidia-smi before =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

# all-on arm: W4A8 Marlin MoE + D0b fused indexer (the 16.6k baseline form).
rc=0
ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; $TR --standalone --nproc_per_node=4 c2f_prefill_stage_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --chunk $CHUNK --moe-mode w4a8 --indexer fused --sparse-backend $BACKEND $*" \
  2>&1 | tee "$TAG-titan064.log" || rc=$?
echo "== bench exit code: $rc =="

echo "== fetch results =="
mkdir -p "$OUT"
rsync -a "$HOST:e0f-runtime/$OUT/" "$OUT/" || true

echo "== nvidia-smi after =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
