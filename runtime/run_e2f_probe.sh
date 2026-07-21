#!/usr/bin/env bash
# E2F: single-node TP4 decode super-stage phase probe (M4 latency line).
#   ./run_e2f_probe.sh <host> <layers> <tag> [extra args...]
# e.g. ./run_e2f_probe.sh titan065 0-10 stage0 --steps 200
# Products land in ../experiments/E2F-decode-latency-profile/results/out-e2f-<tag>/
# plus logs/e2f-<tag>-<host>.log and the before/after nvidia-smi snapshots.
set -euo pipefail
cd "$(dirname "$0")"

HOST=${1:?usage: run_e2f_probe.sh <host> <layers> <tag> [extra args]}
LAYERS=${2:?layer range, e.g. 0-10}
TAG=${3:?tag}
shift 3

TR='~/Workspace/venvs/sglang/bin/torchrun'
RESULTS=../experiments/E2F-decode-latency-profile/results
OUT="out-e2f-${TAG}"
mkdir -p "$RESULTS/logs" "$RESULTS/$OUT"
LOG="$RESULTS/logs/e2f-${TAG}-${HOST}.log"

echo "== sync runtime to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime'
rsync -a --exclude __pycache__ dsv4_direct e1f_full_decode_bench.py \
  e2f_decode_phase_probe.py "$HOST:e0f-runtime/"

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

{
  echo "== nvidia-smi BEFORE ($HOST) =="
  ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
} | tee "$RESULTS/logs/e2f-${TAG}-smi-before.txt"

echo "== E2F $TAG layers=$LAYERS on $HOST: $* =="
rc=0
ssh "$HOST" "cd ~/e0f-runtime && rm -rf $OUT && $ENV_BASE; $TR --standalone --nproc-per-node 4 e2f_decode_phase_probe.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --layers $LAYERS $*" 2>&1 | tee "$LOG" || rc=$?

echo "== fetch results =="
rsync -a "$HOST:e0f-runtime/$OUT/" "$RESULTS/$OUT/" || true

{
  echo "== nvidia-smi AFTER ($HOST) =="
  ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
} | tee "$RESULTS/logs/e2f-${TAG}-smi-after.txt"

echo "== exit code: $rc =="
exit $rc
