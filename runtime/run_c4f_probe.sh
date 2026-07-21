#!/usr/bin/env bash
# C4F 27th vertical: single-GPU ratio-4 phase probe on titan064.
# Usage: ./run_c4f_probe.sh <mode> <tag> [extra args...]
#   e.g. ./run_c4f_probe.sh profile base --chunk 8192
set -euo pipefail
cd "$(dirname "$0")"

MODE=${1:?mode: profile|micro|ab}
TAG=${2:?tag}
shift 2 || true
HOST=${C4F_HOST:-titan064}
OUT="out-c4f-$TAG"
PY='~/Workspace/venvs/sglang/bin/python'

echo "== sync runtime + reference kernel to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c4f_ratio4_phase_probe.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export NCCL_P2P_LEVEL=SYS; export CUDA_VISIBLE_DEVICES=${C4F_GPU:-0}'

rc=0
ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; $PY c4f_ratio4_phase_probe.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --mode $MODE --tag $TAG $*" \
  2>&1 | tee "c4f-$MODE-$TAG-$HOST.log" || rc=$?
echo "== probe exit code: $rc =="

mkdir -p "$OUT"
rsync -a "$HOST:e0f-runtime/$OUT/" "$OUT/" || true
echo ALL_DONE rc=$rc
