#!/usr/bin/env bash
# C4F 27th vertical: D0L long-prompt golden gate with the fused indexer QAT
# lever.  Byte-for-byte the same instrument as run_e0l2e_long_arm.sh (frozen
# baseline 494/512, tilelang prefill sparse + eager HC + sequential MoE); the
# only difference is DSV4_INDEXER_QAT.
# Usage: ./run_c4f_long_gate.sh <tag> <qat: ref|fused> [extra args...]
set -euo pipefail
cd "$(dirname "$0")"

TAG=${1:-c4f}
QAT=${2:-fused}
shift 2 || true
ARM_ARGS="$*"
MASTER=10.234.1.64
PORT=29661
TR='~/Workspace/venvs/sglang/bin/torchrun'
PY='~/Workspace/venvs/sglang/bin/python'
ORACLE_SRC=../experiments/D0L-long-prompt-oracle/results/oracle-long.json
OUT="out-c4f-long-$TAG"

MAX_SEQ_LEN=4224
MAX_STEPS=64

echo "== sync runtime + reference kernel + long golden oracle to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime/reference/inference'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py "$h:e0f-runtime/"
  rsync -a "$ORACLE_SRC" "$h:e0f-runtime/oracle-long.json"
  rsync -a ../reference/inference/kernel.py "$h:e0f-runtime/reference/inference/"
done

ENV_BASE="export CUDA_HOME=/usr/local/cuda-13.2; export PATH=\$CUDA_HOME/bin:\$PATH; export LD_LIBRARY_PATH=\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1; export DSV4_PREFILL_SPARSE_BACKEND=tilelang; export DSV4_INDEXER_QAT=$QAT"

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
from dsv4_direct.ops.indexer_qat import fused_hadamard_fp4
fused_hadamard_fp4(torch.zeros(1,8,64,128,dtype=torch.bfloat16,device=d))
print('WARM', reference_kernel_path())
\"" 2>&1 | grep -E "WARM|Error|error" || true
done

echo "== GPU memory before =="
for h in titan064 titan065; do
  echo "--- $h"; ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done

echo "== C4F long golden gate arm '$TAG' (qat=$QAT $ARM_ARGS) =="
launch() {  # node_rank
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0ef2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-long.json --out-dir $OUT --hc-backends eager --fused-scope decode --max-seq-len $MAX_SEQ_LEN --max-steps $MAX_STEPS --share-moe-buffers $ARM_ARGS"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "c4f-long-$TAG-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "c4f-long-$TAG-node0.log" || rc0=$?
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
