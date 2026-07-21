#!/usr/bin/env bash
# D0L (24th vertical): e0ef2e golden-token gate driven by the *long-prompt*
# oracle (experiments/D0L-long-prompt-oracle/results/oracle-long.json), so the
# single start_pos=0 prefill forward runs at 1024/2048/4096 rows per lane --
# the C2F "chunk" regime -- instead of the 10-22 rows the D0 short oracle
# produced.  This is the instrument the 23rd vertical asked for: the prefill
# levers (fused HC boundary, row-blocked MoE collective overlap) only change
# behaviour above 896/even row counts, which the short gate could never reach.
#
# Usage: ./run_e0l2e_long_arm.sh <tag> <hc-backend> <fused-scope> [extra args...]
#   baseline : ./run_e0l2e_long_arm.sh base   eager  decode
#   lever A  : ./run_e0l2e_long_arm.sh leverA fused  prefill
#   lever B  : ./run_e0l2e_long_arm.sh leverB eager  decode --moe-overlap-blocks 2
set -euo pipefail
cd "$(dirname "$0")"

TAG=${1:-base}
HCB=${2:-eager}
SCOPE=${3:-decode}
shift 3 || true
ARM_ARGS="$*"
MASTER=10.234.1.64
PORT=29659
TR='~/Workspace/venvs/sglang/bin/torchrun'
PY='~/Workspace/venvs/sglang/bin/python'
# Overridable so a run can be gated against an extended golden without
# editing the script; defaults to the frozen 8-prompt / 512-position set.
ORACLE_SRC=${D0L_ORACLE:-../experiments/D0L-long-prompt-oracle/results/oracle-long.json}
OUT="out-e0l2e-$TAG"

# Longest D0L prompt 4096 + 64 compared decode steps - 1 = 4159 -> 4224 (x128).
# Sized for the frozen 8-prompt set (longest 4096).  The extended golden
# adds 8192-token prompts, which need >= 8255, so it is overridable rather
# than silently too small -- the gate does check and refuse, loudly.
MAX_SEQ_LEN=${D0L_MAX_SEQ_LEN:-4224}
MAX_STEPS=64

echo "== sync runtime + reference kernel + long golden oracle to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime/reference/inference'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py "$h:e0f-runtime/"
  rsync -a "$ORACLE_SRC" "$h:e0f-runtime/oracle-long.json"
  rsync -a ../reference/inference/kernel.py "$h:e0f-runtime/reference/inference/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1; export DSV4_PREFILL_SPARSE_BACKEND=tilelang'"; ${GATE_EXTRA_ENV:-:}"

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

echo "== D0L long golden gate arm '$TAG' (hc=$HCB scope=$SCOPE $ARM_ARGS) =="
launch() {  # node_rank
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0ef2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-long.json --out-dir $OUT --hc-backends $HCB --fused-scope $SCOPE --max-seq-len $MAX_SEQ_LEN --max-steps $MAX_STEPS --share-moe-buffers $ARM_ARGS"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "e0l2e-$TAG-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "e0l2e-$TAG-node0.log" || rc0=$?
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
