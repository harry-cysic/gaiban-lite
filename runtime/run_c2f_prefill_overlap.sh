#!/usr/bin/env bash
# C2F 23rd vertical: prefill levers A (HC boundary fusion) and B (MoE
# collective/compute overlap), same 口径 as the 22nd vertical's 25,308 tok/s
# tilelang arm (chunk 8192, 11 layers L11-21, iters 5 / warmup 2, W4A8 Marlin
# MoE + D0b fused indexer + tilelang prefill sparse core).
#
# Usage: ./run_c2f_prefill_overlap.sh <tag> <hc-backend> <moe-overlap> [extra args...]
#   e.g. ./run_c2f_prefill_overlap.sh hcfused-r1 fused off
#        ./run_c2f_prefill_overlap.sh moeovl-r1  default on
#        ./run_c2f_prefill_overlap.sh both-r1    fused  on
set -euo pipefail
cd "$(dirname "$0")"

TAG_IN=${1:?tag}
HCB=${2:-default}
MOEOVL=${3:-off}
shift 3 || true
CHUNK=${CHUNK:-8192}
HOST=${HOST:-titan064}
TAG="c2f-ovl-${TAG_IN}"
OUT="out-$TAG"
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime + reference kernel to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c2f_prefill_stage_bench.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"

# NCCL_P2P_LEVEL=SYS is mandatory: without it the TP4 MoE collectives fall back
# to SHM (4.1 vs 23.8 GB/s) silently.  Every result JSON carries
# moe_collective_selfcheck so this can be confirmed after the fact.
ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_P2P_LEVEL=SYS'

echo "== nvidia-smi before =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

rc=0
ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; $TR --standalone --nproc_per_node=4 c2f_prefill_stage_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --chunk $CHUNK --moe-mode w4a8 --indexer fused --sparse-backend tilelang --hc-backend $HCB --moe-overlap $MOEOVL $*" \
  2>&1 | tee "$TAG-$HOST.log" || rc=$?
echo "== bench exit code: $rc =="

echo "== fetch results =="
mkdir -p "$OUT"
rsync -a "$HOST:e0f-runtime/$OUT/" "$OUT/" || true

echo "== nvidia-smi after =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
