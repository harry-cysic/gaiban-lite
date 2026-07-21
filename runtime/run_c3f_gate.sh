#!/usr/bin/env bash
# C3F 25th vertical: single-layer chunked-prefill equivalence gate (one GPU).
set -euo pipefail
cd "$(dirname "$0")"
TAG=${1:-gate}
shift || true
HOST=${HOST:-titan064}
OUT="out-c3f-$TAG"
PY='~/Workspace/venvs/sglang/bin/python'
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c3f_chunked_prefill_gate.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"
# NCCL_P2P_LEVEL=SYS is set even though this gate is single-process: leaving it
# unset silently falls back to SHM on these boxes, and every launcher in this
# tree carries it so a copy-paste never loses it.
ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_P2P_LEVEL=SYS'
rc=0
ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; $PY c3f_chunked_prefill_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT $*" 2>&1 | tee "c3f-$TAG-$HOST.log" || rc=$?
mkdir -p "$OUT"; rsync -a "$HOST:e0f-runtime/$OUT/" "$OUT/" || true
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader' || true
echo ALL_DONE rc=$rc
