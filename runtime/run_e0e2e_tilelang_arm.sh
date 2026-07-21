#!/usr/bin/env bash
# C2F 21st vertical: e0ef2e golden-token gate with the tilelang prefill sparse
# core (prefill only -- decode keeps the torch core).  16 ranks, eager HC.
# Usage: ./run_e0e2e_tilelang_arm.sh [tag] [extra gate args...]
# Baseline for comparison: out-e0e2e (eager 468/482).
set -euo pipefail
cd "$(dirname "$0")"

TAG=${1:-tilelang}
shift || true
ARM_ARGS="$*"
MASTER=10.234.1.64
PORT=29657
TR='~/Workspace/venvs/sglang/bin/torchrun'
PY='~/Workspace/venvs/sglang/bin/python'
ORACLE_SRC=../experiments/D0-reference-oracle/results/oracle-mp8.json
OUT="out-e0e2e-tl-$TAG"

echo "== sync runtime + reference kernel + golden oracle to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime/reference/inference'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py \
    "$ORACLE_SRC" "$h:e0f-runtime/"
  rsync -a ../reference/inference/kernel.py "$h:e0f-runtime/reference/inference/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1; export DSV4_PREFILL_SPARSE_BACKEND=tilelang'

# Warm the tilelang JIT cache once per node so the 8 ranks/node do not race on
# a cold compile of the same (h=16, d=512, scale) kernel.
echo "== warm tilelang JIT cache (single process per node) =="
for h in titan064 titan065; do
  ssh "$h" "cd ~/e0f-runtime && $ENV_BASE; $PY -c \"
import torch
from dsv4_direct.ops.tilelang_sparse import tilelang_sparse_attention, reference_kernel_path
d=torch.device('cuda:0'); torch.cuda.set_device(d)
q=torch.zeros(1,8,64,512,dtype=torch.bfloat16,device=d)
kv=torch.zeros(1,16,512,dtype=torch.bfloat16,device=d)
sink=torch.zeros(64,dtype=torch.float32,device=d)
idx=torch.zeros(1,8,64,dtype=torch.int32,device=d)
tilelang_sparse_attention(q,kv,sink,idx,512**-0.5)
print('WARM', reference_kernel_path())
\"" 2>&1 | grep -E "WARM|Error|error" || true
done

echo "== GPU memory before =="
for h in titan064 titan065; do
  echo "--- $h"; ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done

echo "== E2E golden gate arm 'tilelang-prefill' (16 ranks, eager HC, $ARM_ARGS) =="
launch() {  # node_rank
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0ef2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-mp8.json --out-dir $OUT --hc-backends eager $ARM_ARGS"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "e0e2e-tl-$TAG-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "e0e2e-tl-$TAG-node0.log" || rc0=$?
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
