#!/usr/bin/env bash
# C2F 26th vertical: segmented (真分段) vs whole-sequence prefill, same 口径.
#
# The 25th vertical (C3F) measured incremental chunked prefill 1.5-2.3x faster
# than whole-sequence prefill -- but in the E2E golden-gate 口径, not the C2F
# throughput 口径.  This launcher runs the C2F stage bench over a (total
# length, segment length) matrix so the claim is tested as input tok/s against
# the frozen tilelang arm (25,308 @ chunk 8192).
#
# Environment is byte-identical to run_c2f_moe_alloc.sh, which produced the
# frozen 25,308: expandable_segments + NCCL_P2P_LEVEL=SYS.  The latter is
# mandatory -- without it the TP4 MoE collectives fall back to SHM (4 GB/s
# instead of 24) and every number is meaningless.  Each result JSON carries its
# own moe_collective_selfcheck so this is checkable after the fact.
#
# Usage: ./run_c2f_chunked_prefill.sh <tag> <rounds> <TOTAL:SEG> [TOTAL:SEG ...]
#   e.g. ./run_c2f_chunked_prefill.sh matrix 3 8192:0 8192:1024 4096:0
# SEG 0 == whole-sequence control arm (the frozen path, unchanged).
set -euo pipefail
cd "$(dirname "$0")"

TAG_IN=${1:?tag}
ROUNDS=${2:?rounds}
shift 2
if [ "$#" -eq 0 ]; then
  echo "need at least one TOTAL:SEG spec" >&2
  exit 2
fi
SPECS=("$@")

HOST=${HOST:-titan064}
TAG="c2f-chunked-${TAG_IN}"
OUT="out-$TAG"
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime + reference kernel to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c2f_prefill_stage_bench.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_P2P_LEVEL=SYS'

echo "== nvidia-smi before =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

# Build the remote loop: one torchrun process per (round, total, segment) so
# every point gets a cold allocator and its own MoE shape registration.
REMOTE="cd ~/e0f-runtime && $ENV_BASE"
REMOTE="$REMOTE; echo \"PYTORCH_CUDA_ALLOC_CONF=[\${PYTORCH_CUDA_ALLOC_CONF}] NCCL_P2P_LEVEL=[\${NCCL_P2P_LEVEL}]\""
for round in $(seq 1 "$ROUNDS"); do
  for spec in "${SPECS[@]}"; do
    total=${spec%%:*}
    seg=${spec##*:}
    REMOTE="$REMOTE; echo \"===== round $round total=$total seg=$seg =====\""
    REMOTE="$REMOTE; $TR --standalone --nproc_per_node=4 c2f_prefill_stage_bench.py"
    REMOTE="$REMOTE --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT/r$round"
    REMOTE="$REMOTE --chunk $total --prefill-chunk $seg"
    REMOTE="$REMOTE --moe-mode w4a8 --indexer fused --sparse-backend tilelang"
  done
done

rc=0
ssh "$HOST" "$REMOTE" 2>&1 | tee "$TAG-$HOST.log" || rc=$?
echo "== bench exit code: $rc =="

echo "== fetch results =="
mkdir -p "$OUT"
rsync -a "$HOST:e0f-runtime/$OUT/" "$OUT/" || true

echo "== nvidia-smi after =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
