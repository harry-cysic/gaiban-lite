#!/usr/bin/env bash
# E7F step 3: the D0L golden-token gate with the stateful (serving) decode arm.
#
# Same instrument and frozen config as the v2 640-position baseline
# (run_e0l2e_long_arm.sh: oracle-long-v2.json, --prefill-chunk 4096,
# --max-seq-len 8320, --share-moe-buffers, tilelang prefill, sharding default),
# but with --with-stateful appended.  The eager arm reproduces the frozen
# non-stateful baseline on the >= 2047 subset (a live reproduction, TARGET
# reuse-before-cite); the stateful arm runs the *serving* decode path on the
# same prompts, so the D0L criterion transfers.
#
# Extra vs run_e0l2e_long_arm.sh: it also syncs e7f_handoff_gate.py and
# e1f_full_decode_bench.py, which the stateful arm lazy-imports.
#
# Usage: ./run_e7f_golden_stateful.sh <tag> [extra e0ef2e args...]
#   smoke (3x 2048): ./run_e7f_golden_stateful.sh smoke --prompt-max-tokens 2048
#   full  (7 >=2047): ./run_e7f_golden_stateful.sh full
set -euo pipefail
cd "$(dirname "$0")"

TAG=${1:-full}
shift 1 || true
ARM_ARGS="$*"
MASTER=10.234.1.64
PORT=29663
TR='~/Workspace/venvs/sglang/bin/torchrun'
PY='~/Workspace/venvs/sglang/bin/python'
ORACLE_SRC=${D0L_ORACLE:-../experiments/D0L-long-prompt-oracle/results/oracle-long-v2.json}
OUT="out-e7f-golden-$TAG"
MAX_SEQ_LEN=${D0L_MAX_SEQ_LEN:-8320}
MAX_STEPS=64

echo "== sync runtime + reference kernel + v2 golden oracle to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime/reference/inference'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py \
    e7f_handoff_gate.py e1f_full_decode_bench.py "$h:e0f-runtime/"
  rsync -a "$ORACLE_SRC" "$h:e0f-runtime/oracle-long-v2.json"
  rsync -a ../reference/inference/kernel.py "$h:e0f-runtime/reference/inference/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1; export DSV4_PREFILL_SPARSE_BACKEND=tilelang'"; ${GATE_EXTRA_ENV:-:}"

echo "== precheck: both nodes must be idle before we start (section 3.8) =="
for h in titan064 titan065; do
  used=$(ssh "$h" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -8 | awk '{s+=\$1} END {print s}'")
  echo "--- $h total used ${used} MiB"
  if [ "${used:-9999}" -gt 32 ]; then
    echo "ABORT: $h not idle (${used} MiB); refusing to start on dirty GPUs"
    exit 9
  fi
done

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

echo "== E7F golden stateful gate '$TAG' ($ARM_ARGS) =="
# Detach the remote torchrun with setsid so a workstation<->titan (earth
# ProxyJump) drop cannot orphan-and-hang the job: it writes to a remote log
# file and, on exit, an exit-code sentinel.  We then poll the sentinel -- a
# *product*, not process liveness (section 9.12) -- with a bounded deadline
# that kills any survivor rather than leaving it spinning at 95 W.
GATE_ARGS="--stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-long-v2.json --out-dir $OUT --hc-backends eager --max-seq-len $MAX_SEQ_LEN --max-steps $MAX_STEPS --prefill-chunk 4096 --share-moe-buffers --prompt-min-tokens 128 --with-stateful $ARM_ARGS"
launch_detached() {  # node_rank host
  local r=$1 h=$2
  ssh "$h" "cd ~/e0f-runtime && $ENV_BASE; rm -f ${OUT}.done.${r} e7f-golden-${TAG}-node${r}.log; setsid bash -c '$TR --nnodes 2 --node-rank $r --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0ef2e_golden_gate.py $GATE_ARGS > e7f-golden-${TAG}-node${r}.log 2>&1; echo \$? > ${OUT}.done.${r}' >/dev/null 2>&1 < /dev/null & echo LAUNCHED_${r}"
}
launch_detached 0 titan064 || true
launch_detached 1 titan065 || true

echo "== poll for completion (detached; survives earth drops) =="
DEADLINE=$(( SECONDS + ${E7F_DEADLINE:-3000} ))
rc0="" rc1=""
while true; do
  rc0=$(ssh titan064 "cat e0f-runtime/${OUT}.done.0 2>/dev/null" 2>/dev/null || true)
  rc1=$(ssh titan065 "cat e0f-runtime/${OUT}.done.1 2>/dev/null" 2>/dev/null || true)
  if [ -n "$rc0" ] && [ -n "$rc1" ]; then
    echo "== both nodes finished: node0=$rc0 node1=$rc1 =="; break
  fi
  if [ "$SECONDS" -gt "$DEADLINE" ]; then
    echo "== TIMEOUT after $(( SECONDS ))s; killing remotes =="
    for h in titan064 titan065; do ssh "$h" 'pkill -9 -f "e0ef2e_golden_gat[e]"; pkill -9 -f "torchru[n]"' 2>/dev/null || true; done
    rc0=${rc0:-TIMEOUT} rc1=${rc1:-TIMEOUT}; break
  fi
  peek=$(ssh titan064 "tail -1 e0f-runtime/e7f-golden-${TAG}-node0.log 2>/dev/null" 2>/dev/null || true)
  echo "  [poll +$(( SECONDS ))s] $peek"
  sleep 25
done

echo "== fetch results + node logs =="
mkdir -p "$OUT"
rsync -a "titan064:e0f-runtime/$OUT/" "$OUT/" 2>/dev/null || true
rsync -a "titan065:e0f-runtime/$OUT/" "$OUT/" 2>/dev/null || true
rsync -a "titan064:e0f-runtime/e7f-golden-${TAG}-node0.log" ./ 2>/dev/null || true
rsync -a "titan065:e0f-runtime/e7f-golden-${TAG}-node1.log" ./ 2>/dev/null || true

echo "== GPU memory after =="
for h in titan064 titan065; do
  echo "--- $h"; ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done
echo ALL_DONE rc0=$rc0 rc1=$rc1
