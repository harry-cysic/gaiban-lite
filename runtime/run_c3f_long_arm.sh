#!/usr/bin/env bash
# C3F (25th vertical): the D0L long golden gate run with *incremental chunked*
# prefill.  Same oracle, same topology, same frozen config as
# run_e0l2e_long_arm.sh -- the only difference is --prefill-chunk, which turns
# the single whole-sequence start_pos=0 forward into consecutive multi-token
# forwards at start_pos > 0 (the capability the 24th vertical found missing).
#
# Usage: ./run_c3f_long_arm.sh <tag> <chunk> [extra args...]
#   whole-sequence control : ./run_c3f_long_arm.sh chunk0    0
#   4 x 1024 / 2 x 1024 ... : ./run_c3f_long_arm.sh chunk1024 1024
#   non-aligned chunk       : ./run_c3f_long_arm.sh chunk1000 1000
set -euo pipefail
cd "$(dirname "$0")"

TAG=${1:-chunk1024}
CHUNK=${2:-1024}
shift 2 || true
ARM_ARGS="$*"
MASTER=10.234.1.64
PORT=29661
TR='~/Workspace/venvs/sglang/bin/torchrun'
PY='~/Workspace/venvs/sglang/bin/python'
ORACLE_SRC=../experiments/D0L-long-prompt-oracle/results/oracle-long.json
OUT="out-c3f-$TAG"

# Frozen D0L baseline configuration (README section 3.1).
MAX_SEQ_LEN=4224
MAX_STEPS=64

echo "== sync runtime + reference kernel + long golden oracle to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime/reference/inference'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py "$h:e0f-runtime/"
  rsync -a "$ORACLE_SRC" "$h:e0f-runtime/oracle-long.json"
  rsync -a ../reference/inference/kernel.py "$h:e0f-runtime/reference/inference/"
done

# NCCL_P2P_LEVEL=SYS is mandatory here: leaving it unset silently falls back to
# SHM on these boxes and the numbers stop being comparable.
ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1; export DSV4_PREFILL_SPARSE_BACKEND=tilelang'

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

echo "== C3F long golden gate arm '$TAG' (prefill-chunk=$CHUNK $ARM_ARGS) =="
launch() {  # node_rank
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0ef2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-long.json --out-dir $OUT --hc-backends eager --fused-scope decode --max-seq-len $MAX_SEQ_LEN --max-steps $MAX_STEPS --share-moe-buffers --prefill-chunk $CHUNK $ARM_ARGS"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "c3f-$TAG-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "c3f-$TAG-node0.log" || rc0=$?
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
