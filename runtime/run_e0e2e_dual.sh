#!/usr/bin/env bash
# E0e2e: full-model 43-layer TP4xPP4 dual-node golden-token E2E driver
# (titan064 stages 0-1 GPU0-7 + titan065 stages 2-3 GPU0-7, IB boundary
# between stage 1 and stage 2).  Runs from the workstation over the ssh
# aliases titan064/titan065 (ProxyJump earth), E0qf no-GDR NCCL config.
# Sequence: ratio-4 full-position selfcheck on titan064, then the 16-rank
# golden gate (eager + fused HC boundary modes in one process).
# Products: e0e2e-node{0,1}.log, e0e2e-selfcheck.log, out-e0e2e/,
# out-e0e2e-selfcheck/.
set -euo pipefail
cd "$(dirname "$0")"

MASTER=10.234.1.64
PORT=29641
TR='~/Workspace/venvs/sglang/bin/torchrun'
PY='~/Workspace/venvs/sglang/bin/python'
ORACLE_SRC=../experiments/D0-reference-oracle/results/oracle-mp8.json

echo "== sync runtime + golden oracle to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py \
    e0e2e_ratio4_selfcheck.py "$ORACLE_SRC" "$h:e0f-runtime/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

echo "== ratio-4 full-position selfcheck (titan064, single GPU) =="
ssh titan064 "cd ~/e0f-runtime && $ENV_BASE; CUDA_VISIBLE_DEVICES=0 $PY e0e2e_ratio4_selfcheck.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0e2e-selfcheck" \
  2>&1 | tee e0e2e-selfcheck.log

echo "== E2E golden gate (16 ranks, eager + fused) =="
launch() {  # node_rank
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0ef2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-mp8.json --out-dir out-e0e2e --hc-backends eager,fused"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > e0e2e-node1.log 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee e0e2e-node0.log || rc0=$?
wait "$pid" || rc1=$?
echo "== gate exit codes: node0=$rc0 node1=$rc1 =="

echo "== fetch results =="
mkdir -p out-e0e2e out-e0e2e-selfcheck
rsync -a "titan064:e0f-runtime/out-e0e2e/" out-e0e2e/ || true
rsync -a "titan065:e0f-runtime/out-e0e2e/" out-e0e2e/ || true
rsync -a "titan064:e0f-runtime/out-e0e2e-selfcheck/" out-e0e2e-selfcheck/ || true

echo "== GPU memory check =="
for h in titan064 titan065; do
  echo "--- $h"
  ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done
echo ALL_DONE
