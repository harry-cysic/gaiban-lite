#!/usr/bin/env bash
# C2F 23rd vertical, lever A: prefill-shape HC boundary fusion micro-gate.
# Single GPU (no collectives), but the standard ENV_BASE is kept verbatim --
# including NCCL_P2P_LEVEL=SYS, whose omission was the 22nd vertical's silent
# SHM fallback.
# Usage: ./run_c2f_hc_gate.sh [tag] [extra args...]
set -euo pipefail
cd "$(dirname "$0")"

TAG=${1:-hcgate}
shift || true
HOST=${HOST:-titan065}
OUT="out-c2f-$TAG"
PY='~/Workspace/venvs/sglang/bin/python'

echo "== sync runtime to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c2f_hc_prefill_gate.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_P2P_LEVEL=SYS'

echo "== nvidia-smi before =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

rc=0
ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; CUDA_VISIBLE_DEVICES=0 $PY c2f_hc_prefill_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT $*" \
  2>&1 | tee "c2f-$TAG-$HOST.log" || rc=$?
echo "== gate exit code: $rc =="

echo "== fetch results =="
mkdir -p "$OUT"
rsync -a "$HOST:e0f-runtime/$OUT/" "$OUT/" || true

echo "== nvidia-smi after =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
