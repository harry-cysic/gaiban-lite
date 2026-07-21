#!/usr/bin/env bash
# E1F: full-config TP4xPP4 stateful-graph closed-loop decode bench driver.
# One local-batch value per invocation:
#   ./run_e1f_dual.sh <local_batch> <check_mode: off|bitwise> [rounds] [steps] [start_pos]
# Topology (e0ef2e): titan064 stages 0-1 (GPU0-7), titan065 stages 2-3
# (GPU0-7); IB boundary between stage 1 and stage 2; E0qf no-GDR NCCL env.
# Products land in ../experiments/E1F-full-decode-throughput/results/:
#   out-e1f-bl<B>/rank*.json + result.json, logs/e1f-bl<B>-node{0,1}.log
set -euo pipefail
cd "$(dirname "$0")"

B=${1:?usage: run_e1f_dual.sh <local_batch> <check_mode> [rounds] [steps] [start_pos]}
CHECK=${2:?check_mode off|bitwise}
ROUNDS=${3:-3}
STEPS=${4:-300}
START=${5:-2048}

MASTER=10.234.1.64
PORT=29651
TR='~/Workspace/venvs/sglang/bin/torchrun'
# Both overridable so a reproduction run can be filed elsewhere without
# clobbering a frozen artifact; defaults are the original ones.
RESULTS=${E1F_RESULTS:-../experiments/E1F-full-decode-throughput/results}
OUT=${E1F_OUT:-out-e1f-bl${B}}
mkdir -p "$RESULTS/logs" "$RESULTS/$OUT"

echo "== sync runtime to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime'
  rsync -a --exclude __pycache__ dsv4_direct e1f_full_decode_bench.py "$h:e0f-runtime/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'"; ${E1F_EXTRA_ENV:-:}"

launch() {  # node_rank
  echo "cd ~/e0f-runtime && rm -rf $OUT && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e1f_full_decode_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --local-batch $B --check-mode $CHECK --rounds $ROUNDS --steps $STEPS --start-position $START --config-tag nogdr"
}

echo "== E1F bl=$B check=$CHECK rounds=$ROUNDS steps=$STEPS start=$START =="
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "$RESULTS/logs/e1f-bl${B}-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "$RESULTS/logs/e1f-bl${B}-node0.log" || rc0=$?
wait "$pid" || rc1=$?
echo "== exit codes: node0=$rc0 node1=$rc1 =="

echo "== fetch results =="
rsync -a "titan064:e0f-runtime/$OUT/" "$RESULTS/$OUT/" || true
rsync -a "titan065:e0f-runtime/$OUT/" "$RESULTS/$OUT/" || true

echo "== GPU memory check =="
for h in titan064 titan065; do
  echo "--- $h"
  ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done
exit $(( rc0 > rc1 ? rc0 : rc1 ))
