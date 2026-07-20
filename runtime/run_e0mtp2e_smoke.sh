#!/usr/bin/env bash
# E0mtp2e smoke: 1 prompt, 8 steps, all arms (debug driver).
set -uo pipefail
cd "$(dirname "$0")"

MASTER=10.234.1.64
PORT=29647
TR='~/Workspace/venvs/sglang/bin/torchrun'
ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

for h in titan064 titan065; do
  rsync -a --exclude __pycache__ dsv4_direct e0mtp2e_golden_gate.py "$h:e0f-runtime/"
done

launch() {
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0mtp2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-mp8.json --out-dir out-e0mtp2e-smoke --max-prompts 1 --max-steps 8 --kv-dtype fp8 --hc-backend fused"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > e0mtp2e-smoke-node1.log 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" > e0mtp2e-smoke-node0.log 2>&1 || rc0=$?
wait "$pid" || rc1=$?
mkdir -p out-e0mtp2e-smoke
rsync -a titan064:e0f-runtime/out-e0mtp2e-smoke/ out-e0mtp2e-smoke/ 2>/dev/null
rsync -a titan065:e0f-runtime/out-e0mtp2e-smoke/ out-e0mtp2e-smoke/ 2>/dev/null
echo "SMOKE_DONE rc0=$rc0 rc1=$rc1"
