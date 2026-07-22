#!/usr/bin/env bash
# E8F: minimal single-path serving loop (section 10 Phase 1) -- resident 16-rank
# job that prefills a real prompt, hands off, free-running graph-decodes until
# EOS, and times the whole request (framework-caliber discount vs 39.2 bare).
#
# One run = one fixed prompt length (the plan/graph are built once and reused;
# a differing length is a hard error).  Bucket by length: one run per length.
#
# Usage: ./run_e8f_serving.sh <tag> <plen> [extra args...]
#   1024 bucket: ./run_e8f_serving.sh b1024 1024
#   2048 bucket: ./run_e8f_serving.sh b2048 2048
set -euo pipefail
cd "$(dirname "$0")"

TAG=${1:-b1024}
PLEN=${2:-1024}
shift 2 || true
ARM_ARGS="$*"
MASTER=10.234.1.64
PORT=29667
TR='~/Workspace/venvs/sglang/bin/torchrun'
PY='~/Workspace/venvs/sglang/bin/python'
ORACLE_SRC=${E8F_ORACLE:-../experiments/D0L-long-prompt-oracle/results/oracle-long-v2.json}
OUT="out-e8f-$TAG"
MAX_SEQ_LEN=${E8F_MAX_SEQ_LEN:-8320}

echo "== sync runtime + reference kernel + oracle =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime/reference/inference'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py \
    e7f_handoff_gate.py e1f_full_decode_bench.py e8f_serving_loop.py "$h:e0f-runtime/"
  rsync -a "$ORACLE_SRC" "$h:e0f-runtime/oracle-long-v2.json"
  rsync -a ../reference/inference/kernel.py "$h:e0f-runtime/reference/inference/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1; export DSV4_PREFILL_SPARSE_BACKEND=tilelang'"; ${GATE_EXTRA_ENV:-:}"

echo "== precheck: both nodes idle (section 3.8) =="
for h in titan064 titan065; do
  used=$(ssh "$h" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -8 | awk '{s+=\$1} END {print s}'")
  echo "--- $h total used ${used} MiB"
  [ "${used:-9999}" -gt 32 ] && { echo "ABORT: $h not idle (${used} MiB)"; exit 9; }
done

echo "== warm tilelang JIT =="
for h in titan064 titan065; do
  ssh "$h" "cd ~/e0f-runtime && $ENV_BASE; $PY -c \"
import torch
from dsv4_direct.ops.tilelang_sparse import tilelang_sparse_attention, reference_kernel_path
d=torch.device('cuda:0'); torch.cuda.set_device(d)
q=torch.zeros(1,8,64,512,dtype=torch.bfloat16,device=d); kv=torch.zeros(1,16,512,dtype=torch.bfloat16,device=d)
sink=torch.zeros(64,dtype=torch.float32,device=d); idx=torch.zeros(1,8,64,dtype=torch.int32,device=d)
tilelang_sparse_attention(q,kv,sink,idx,512**-0.5); print('WARM', reference_kernel_path())
\"" 2>&1 | grep -E "WARM|Error" || true
done

echo "== E8F serving '$TAG' plen=$PLEN ($ARM_ARGS) =="
GATE_ARGS="--stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-long-v2.json --out-dir $OUT --max-seq-len $MAX_SEQ_LEN --prefill-chunk 4096 --prompt-min-tokens $PLEN --prompt-max-tokens $PLEN $ARM_ARGS"
launch_detached() {  # node_rank host
  local r=$1 h=$2
  ssh "$h" "cd ~/e0f-runtime && $ENV_BASE; rm -f ${OUT}.done.${r} e8f-${TAG}-node${r}.log; setsid bash -c '$TR --nnodes 2 --node-rank $r --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e8f_serving_loop.py $GATE_ARGS > e8f-${TAG}-node${r}.log 2>&1; echo \$? > ${OUT}.done.${r}' >/dev/null 2>&1 < /dev/null & echo LAUNCHED_${r}"
}
launch_detached 0 titan064 || true
launch_detached 1 titan065 || true

echo "== poll (detached; survives earth drops) =="
DEADLINE=$(( SECONDS + ${E8F_DEADLINE:-2400} ))
rc0="" rc1=""
while true; do
  rc0=$(ssh titan064 "cat e0f-runtime/${OUT}.done.0 2>/dev/null" 2>/dev/null || true)
  rc1=$(ssh titan065 "cat e0f-runtime/${OUT}.done.1 2>/dev/null" 2>/dev/null || true)
  [ -n "$rc0" ] && [ -n "$rc1" ] && { echo "== finished: node0=$rc0 node1=$rc1 =="; break; }
  if [ "$SECONDS" -gt "$DEADLINE" ]; then
    echo "== TIMEOUT; killing remotes =="
    for h in titan064 titan065; do ssh "$h" 'pkill -9 -f "e8f_serving_loo[p]"; pkill -9 -f "torchru[n]"' 2>/dev/null || true; done
    rc0=${rc0:-TIMEOUT} rc1=${rc1:-TIMEOUT}; break
  fi
  echo "  [poll +$(( SECONDS ))s] $(ssh titan064 "tail -1 e0f-runtime/e8f-${TAG}-node0.log 2>/dev/null" 2>/dev/null || true)"
  sleep 25
done

echo "== fetch =="
mkdir -p "$OUT"
rsync -a "titan064:e0f-runtime/$OUT/" "$OUT/" 2>/dev/null || true
rsync -a "titan065:e0f-runtime/$OUT/" "$OUT/" 2>/dev/null || true
rsync -a "titan064:e0f-runtime/e8f-${TAG}-node0.log" ./ 2>/dev/null || true
echo ALL_DONE rc0=$rc0 rc1=$rc1
