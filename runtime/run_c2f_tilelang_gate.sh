#!/usr/bin/env bash
# C2F 21st vertical: single-operator numeric gate for the tilelang prefill
# sparse core (single GPU, titan065 by default -- no checkpoint needed).
# Usage: ./run_c2f_tilelang_gate.sh [host] [extra args...]
set -euo pipefail
cd "$(dirname "$0")"

HOST=${1:-titan065}
shift || true
PY='~/Workspace/venvs/sglang/bin/python'

echo "== sync runtime + reference kernel to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c2f_tilelang_sparse_gate.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_P2P_LEVEL=SYS'

echo "== nvidia-smi before =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

rc=0
ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; $PY c2f_tilelang_sparse_gate.py --out c2f-tilelang-op-gate.json $*" \
  2>&1 | tee "c2f-tilelang-op-gate-$HOST.log" || rc=$?
echo "== gate exit code: $rc =="

echo "== fetch results =="
mkdir -p out-c2f-tilelang
rsync -a "$HOST:e0f-runtime/c2f-tilelang-op-gate.json" out-c2f-tilelang/ || true

echo "== nvidia-smi after =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
