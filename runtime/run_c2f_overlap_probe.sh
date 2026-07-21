#!/usr/bin/env bash
# C2F 23rd vertical lever B attribution: NCCL/compute concurrency probe.
set -euo pipefail
cd "$(dirname "$0")"
TAG=${1:-overlapprobe}
shift || true
HOST=${HOST:-titan065}
OUT="out-c2f-$TAG"
TR='~/Workspace/venvs/sglang/bin/torchrun'
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c2f_moe_overlap_probe.py "$HOST:e0f-runtime/"
ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_P2P_LEVEL=SYS'
rc=0
ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; $TR --standalone --nproc_per_node=4 c2f_moe_overlap_probe.py --out-dir $OUT $*" 2>&1 | tee "c2f-$TAG-$HOST.log" || rc=$?
mkdir -p "$OUT"; rsync -a "$HOST:e0f-runtime/$OUT/" "$OUT/" || true
echo ALL_DONE rc=$rc
