#!/usr/bin/env bash
# C2F: single-stage (TP4, L11-L21) chunked-prefill bench on titan064.
# Usage: ./run_c2f_prefill_titan.sh <chunk> <moe-mode> <indexer> [extra args...]
#   e.g. ./run_c2f_prefill_titan.sh 4096 w4a16 ref --gate-indexer
set -euo pipefail
cd "$(dirname "$0")"

CHUNK=${1:?chunk}
MOE=${2:-w4a16}
IDX=${3:-ref}
shift 3 || true
TAG="c2f-chunk${CHUNK}-${MOE}-${IDX}"
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime to titan064 =="
ssh titan064 'mkdir -p ~/e0f-runtime'
rsync -a --exclude __pycache__ dsv4_direct c2f_prefill_stage_bench.py titan064:e0f-runtime/

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_P2P_LEVEL=SYS'

echo "== nvidia-smi before =="
ssh titan064 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

echo "== C2F prefill bench (4 ranks, chunk=$CHUNK moe=$MOE indexer=$IDX $*) =="
rc=0
ssh titan064 "cd ~/e0f-runtime && $ENV_BASE; $TR --standalone --nproc_per_node=4 c2f_prefill_stage_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-c2f --chunk $CHUNK --moe-mode $MOE --indexer $IDX $*" \
  2>&1 | tee "$TAG-titan064.log" || rc=$?
echo "== bench exit code: $rc =="

echo "== fetch results =="
mkdir -p out-c2f
rsync -a "titan064:e0f-runtime/out-c2f/" out-c2f/ || true

echo "== nvidia-smi after =="
ssh titan064 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
