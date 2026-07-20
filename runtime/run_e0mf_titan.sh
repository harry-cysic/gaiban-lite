#!/usr/bin/env bash
# E0mf: MTP block (mtp.0) forward vs fp32 oracle gate (titan064, TP4).
# Usage: ./run_e0mf_titan.sh [kv_dtype]   (default bf16; fp8 arm optional)
set -euo pipefail
cd "$(dirname "$0")"

KV=${1:-bf16}
TAG="e0mf"
if [ "$KV" != "bf16" ]; then TAG="e0mf-$KV"; fi
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime to titan064 =="
ssh titan064 'mkdir -p ~/e0f-runtime'
rsync -a --exclude __pycache__ dsv4_direct e0mf_mtp_block_oracle.py titan064:e0f-runtime/

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'

echo "== nvidia-smi before =="
ssh titan064 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

echo "== E0mf gate (4 ranks, kv=$KV) =="
rc=0
ssh titan064 "cd ~/e0f-runtime && $ENV_BASE; $TR --standalone --nproc_per_node=4 e0mf_mtp_block_oracle.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-$TAG --kv-dtype $KV" \
  2>&1 | tee "$TAG-titan064.log" || rc=$?
echo "== gate exit code: $rc =="

echo "== fetch results =="
mkdir -p "out-$TAG"
rsync -a "titan064:e0f-runtime/out-$TAG/" "out-$TAG/" || true

echo "== nvidia-smi after =="
ssh titan064 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
