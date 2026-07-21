#!/usr/bin/env bash
# C4F 27th vertical: C2F-口径 prefill throughput with the fused indexer QAT
# lever on/off.  Identical to run_c2f_tilelang_bench.sh (11 layers L11-21,
# chunk 8192, iters 5 / warmup 2, W4A8 Marlin + D0b fused indexer + tilelang
# sparse core) except for DSV4_INDEXER_QAT.
# Usage: ./run_c4f_c2f_bench.sh <chunk> <qat: ref|fused> <round> [extra args...]
set -euo pipefail
cd "$(dirname "$0")"

CHUNK=${1:?chunk}
QAT=${2:?qat mode: ref|fused}
ROUND=${3:-1}
shift 3 || true
HOST=${C4F_HOST:-titan064}
TAG="c4f-chunk${CHUNK}-qat${QAT}-r${ROUND}"
OUT="out-$TAG"
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime + reference kernel to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c2f_prefill_stage_bench.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"

ENV_BASE="export CUDA_HOME=/usr/local/cuda-13.2; export PATH=\$CUDA_HOME/bin:\$PATH; export LD_LIBRARY_PATH=\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_P2P_LEVEL=SYS; export DSV4_INDEXER_QAT=$QAT"

echo "== nvidia-smi before =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

rc=0
ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; $TR --standalone --nproc_per_node=4 c2f_prefill_stage_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --chunk $CHUNK --moe-mode w4a8 --indexer fused --sparse-backend tilelang $*" \
  2>&1 | tee "$TAG-$HOST.log" || rc=$?
echo "== bench exit code: $rc =="

echo "== fetch results =="
mkdir -p "$OUT"
rsync -a "$HOST:e0f-runtime/$OUT/" "$OUT/" || true

echo "== nvidia-smi after =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
