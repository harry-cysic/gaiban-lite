#!/usr/bin/env bash
# E0e2e FP8-KV arm: full-model golden-token E2E gate with --kv-dtype.
# Usage: ./run_e0e2e_kv_dual.sh <kv_dtype> [indexer_kv_dtype] [hc_backends]
#   kv_dtype:  fp8 | fp8_rope_bf16 | bf16
#   indexer:   bf16 (default) | fp8
#   backends:  default "fused" (baseline eager/fused already recorded 468/482)
# Topology and env identical to run_e0e2e_dual.sh.  Products:
#   out-e0e2e-<kv>[-idx<idx>]/ + e0e2e-<kv>[-idx]-node{0,1}.log +
#   out-e0e2e-selfcheck-<kv>/ under runtime/.
set -euo pipefail
cd "$(dirname "$0")"

KV=${1:?usage: run_e0e2e_kv_dual.sh <kv_dtype> [indexer_kv_dtype] [hc_backends]}
IDX=${2:-bf16}
BACKENDS=${3:-fused}
TAG="$KV"
if [ "$IDX" != "bf16" ]; then TAG="$KV-idx$IDX"; fi

MASTER=10.234.1.64
PORT=29645
TR='~/Workspace/venvs/sglang/bin/torchrun'
PY='~/Workspace/venvs/sglang/bin/python'
ORACLE_SRC=../experiments/D0-reference-oracle/results/oracle-mp8.json

echo "== sync runtime + golden oracle to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py \
    e0e2e_ratio4_selfcheck.py fp8_kv_gate_common.py "$ORACLE_SRC" "$h:e0f-runtime/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

echo "== ratio-4 full-position selfcheck ($TAG, titan064, single GPU) =="
ssh titan064 "cd ~/e0f-runtime && $ENV_BASE; CUDA_VISIBLE_DEVICES=0 $PY e0e2e_ratio4_selfcheck.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0e2e-selfcheck-$TAG --kv-dtype $KV --indexer-kv-dtype $IDX" \
  2>&1 | tee "e0e2e-selfcheck-$TAG.log"

echo "== E2E golden gate ($TAG, 16 ranks, hc=$BACKENDS) =="
launch() {  # node_rank
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0ef2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-mp8.json --out-dir out-e0e2e-$TAG --hc-backends $BACKENDS --kv-dtype $KV --indexer-kv-dtype $IDX"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "e0e2e-$TAG-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "e0e2e-$TAG-node0.log" || rc0=$?
wait "$pid" || rc1=$?
echo "== gate exit codes: node0=$rc0 node1=$rc1 =="

echo "== fetch results =="
mkdir -p "out-e0e2e-$TAG" "out-e0e2e-selfcheck-$TAG"
rsync -a "titan064:e0f-runtime/out-e0e2e-$TAG/" "out-e0e2e-$TAG/" || true
rsync -a "titan065:e0f-runtime/out-e0e2e-$TAG/" "out-e0e2e-$TAG/" || true
rsync -a "titan064:e0f-runtime/out-e0e2e-selfcheck-$TAG/" "out-e0e2e-selfcheck-$TAG/" || true

echo "== GPU memory check =="
for h in titan064 titan065; do
  echo "--- $h"
  ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done
echo ALL_DONE
