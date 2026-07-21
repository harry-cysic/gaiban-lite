#!/usr/bin/env bash
# C2F 22nd vertical: prefill MoE allocator bimodality -- probe + A/B launcher.
#
# The 20th/21st verticals recorded the same code/params producing two MoE
# regimes (0.48 s/pass vs 1.37 s/pass).  Neither run recorded its allocator
# configuration, so this launcher makes PYTORCH_CUDA_ALLOC_CONF an explicit,
# logged knob and runs the bench with --alloc-probe.
#
# Usage: ./run_c2f_moe_alloc.sh <tag> <alloc-conf|none> <backend> [extra args...]
#   e.g. ./run_c2f_moe_alloc.sh probe-expandable expandable_segments:True torch --alloc-probe
#        ./run_c2f_moe_alloc.sh probe-default none torch --alloc-probe
set -euo pipefail
cd "$(dirname "$0")"

TAG_IN=${1:?tag}
ALLOC=${2:?alloc conf, or "none" to leave PYTORCH_CUDA_ALLOC_CONF unset}
BACKEND=${3:-torch}
shift 3 || true
CHUNK=${CHUNK:-8192}
HOST=${HOST:-titan064}
TAG="c2f-moe-${TAG_IN}"
OUT="out-$TAG"
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime + reference kernel to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime/reference/inference'
rsync -a --exclude __pycache__ dsv4_direct c2f_prefill_stage_bench.py "$HOST:e0f-runtime/"
rsync -a ../reference/inference/kernel.py "$HOST:e0f-runtime/reference/inference/"

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'
if [ "$ALLOC" = "none" ]; then
  ENV_ALLOC='unset PYTORCH_CUDA_ALLOC_CONF'
else
  ENV_ALLOC="export PYTORCH_CUDA_ALLOC_CONF=$ALLOC"
fi

# THE fix for the prefill MoE bimodality (22nd vertical).  GPU0-3 sit at NODE
# distance (PCIe host bridges within one NUMA node); NCCL's default P2P level
# excludes that, so it falls back to SHM/direct (host-staged) and the TP4 MoE
# collectives run at 4.1 GB/s instead of 23.8 GB/s.  Every other launcher in
# this runtime already exports NCCL_P2P_LEVEL=SYS -- the C2F launchers did not,
# which is exactly the fast/slow split recorded in the 20th/21st verticals.
P2P_LEVEL=${P2P_LEVEL:-SYS}
if [ "$P2P_LEVEL" = "none" ]; then
  ENV_P2P='unset NCCL_P2P_LEVEL'
else
  ENV_P2P="export NCCL_P2P_LEVEL=$P2P_LEVEL"
fi

echo "== nvidia-smi before =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

# all-on arm (W4A8 Marlin MoE + D0b fused indexer) == the 16.6k baseline form.
rc=0
ssh "$HOST" "cd ~/e0f-runtime && $ENV_BASE; $ENV_ALLOC; $ENV_P2P; echo \"PYTORCH_CUDA_ALLOC_CONF=[\${PYTORCH_CUDA_ALLOC_CONF:-<unset>}] NCCL_P2P_LEVEL=[\${NCCL_P2P_LEVEL:-<unset>}]\"; $TR --standalone --nproc_per_node=4 c2f_prefill_stage_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --chunk $CHUNK --moe-mode w4a8 --indexer fused --sparse-backend $BACKEND $*" \
  2>&1 | tee "$TAG-$HOST.log" || rc=$?
echo "== bench exit code: $rc =="

echo "== fetch results =="
mkdir -p "$OUT"
rsync -a "$HOST:e0f-runtime/$OUT/" "$OUT/" || true

echo "== nvidia-smi after =="
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
echo ALL_DONE rc=$rc
