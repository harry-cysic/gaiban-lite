#!/usr/bin/env bash
# C2F 21st vertical: layer-level gate for the tilelang prefill sparse core.
# Re-runs the frozen e0ef (ratio-128) and e0wf (window) single-layer oracles
# with DSV4_PREFILL_SPARSE_BACKEND set, at the *original* tolerances.
# Usage: ./run_c2f_tilelang_oracles.sh <torch|tilelang>
set -euo pipefail
cd "$(dirname "$0")"

BACKEND=${1:?backend: torch|tilelang}
HOST=titan064
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime + reference kernel to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct \
  e0ef_ratio128_attention_oracle.py e0wf_window_attention_oracle.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"

ENV_BASE="export CUDA_HOME=/usr/local/cuda-13.2; export PATH=\$CUDA_HOME/bin:\$PATH; export LD_LIBRARY_PATH=\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export DSV4_PREFILL_SPARSE_BACKEND=$BACKEND"

echo "== nvidia-smi before =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

rc=0
for spec in "e0ef:e0ef_ratio128_attention_oracle.py" "e0wf:e0wf_window_attention_oracle.py"; do
  tag=${spec%%:*}; script=${spec##*:}
  out="out-$tag-tl-$BACKEND"
  echo "== $tag oracle (backend=$BACKEND) =="
  ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; $TR --standalone --nproc_per_node=4 $script --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $out" \
    2>&1 | tee "$tag-tl-$BACKEND-titan064.log" || rc=$?
  mkdir -p "$out"
  rsync -a "$HOST:e0f-runtime/$out/" "$out/" || true
done
echo "== oracle exit code: $rc =="

echo "== nvidia-smi after =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
