#!/usr/bin/env bash
# C2F E2E regression arms: e0ef2e golden gate with a prefill lever enabled.
# Usage: ./run_e0e2e_c2f_arm.sh <tag> [gate args...]
#   ./run_e0e2e_c2f_arm.sh fusedidx --ratio4-index-mode fused --fuse-min-seqlen 8
#   ./run_e0e2e_c2f_arm.sh w4a8 --moe-input-dtype fp8
# Baseline for comparison: out-e0e2e (eager 468/482).  Arms run eager HC only.
set -euo pipefail
cd "$(dirname "$0")"

TAG=${1:?tag}
shift
ARM_ARGS="$*"
MASTER=10.234.1.64
PORT=29653
TR='~/Workspace/venvs/sglang/bin/torchrun'
ORACLE_SRC=../experiments/D0-reference-oracle/results/oracle-mp8.json
OUT="out-e0e2e-c2f-$TAG"

echo "== sync runtime + golden oracle to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py \
    "$ORACLE_SRC" "$h:e0f-runtime/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

echo "== GPU memory before =="
for h in titan064 titan065; do
  echo "--- $h"; ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done

echo "== E2E golden gate arm '$TAG' (16 ranks, eager HC, $ARM_ARGS) =="
launch() {  # node_rank
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0ef2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-mp8.json --out-dir $OUT --hc-backends eager $ARM_ARGS"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "e0e2e-c2f-$TAG-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "e0e2e-c2f-$TAG-node0.log" || rc0=$?
wait "$pid" || rc1=$?
echo "== gate exit codes: node0=$rc0 node1=$rc1 =="

echo "== fetch results =="
mkdir -p "$OUT"
rsync -a "titan064:e0f-runtime/$OUT/" "$OUT/" || true
rsync -a "titan065:e0f-runtime/$OUT/" "$OUT/" || true

echo "== GPU memory after =="
for h in titan064 titan065; do
  echo "--- $h"; ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done
echo ALL_DONE rc0=$rc0 rc1=$rc1
