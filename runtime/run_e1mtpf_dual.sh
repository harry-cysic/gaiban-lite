#!/usr/bin/env bash
# E1MTPF: MTP verify-round timing bench at ctx 2048 (16 ranks, eager fullpos).
# Usage: ./run_e1mtpf_dual.sh [kv_dtype] [hc_backend]
set -euo pipefail
cd "$(dirname "$0")"

KV=${1:-fp8}
HC=${2:-fused}
TAG="e1mtpf-$KV-$HC"

MASTER=10.234.1.64
PORT=29649
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime'
  rsync -a --exclude __pycache__ dsv4_direct e0mtp2e_golden_gate.py \
    e1mtpf_verify_bench.py "$h:e0f-runtime/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

echo "== nvidia-smi before =="
for h in titan064 titan065; do
  echo "--- $h"; ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done

launch() {
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e1mtpf_verify_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-$TAG --kv-dtype $KV --hc-backend $HC"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "$TAG-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "$TAG-node0.log" || rc0=$?
wait "$pid" || rc1=$?
echo "== bench exit codes: node0=$rc0 node1=$rc1 =="

mkdir -p "out-$TAG"
rsync -a "titan064:e0f-runtime/out-$TAG/" "out-$TAG/" || true
rsync -a "titan065:e0f-runtime/out-$TAG/" "out-$TAG/" || true

echo "== nvidia-smi after =="
for h in titan064 titan065; do
  echo "--- $h"; ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done
echo ALL_DONE rc0=$rc0 rc1=$rc1
