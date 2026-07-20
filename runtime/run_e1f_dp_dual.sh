#!/usr/bin/env bash
# E1F DP caliber: full-config TP4xPP4 stateful-graph closed-loop decode bench
# under true DP-attention sequence split (--b-semantics dp, E0dpf-gated).
# One global-batch value per invocation:
#   ./run_e1f_dp_dual.sh <global_batch> <check_mode: off|bitwise> [rounds] [steps] [start_pos]
# local_batch per GPU = global_batch / 4.  Topology and NCCL env identical to
# run_e1f_dual.sh (titan064 stages 0-1, titan065 stages 2-3, no-GDR).
# Products land in ../experiments/E1F-full-decode-throughput/results/dp/:
#   out-e1f-dp-bg<BG>-ctx<START>/rank*.json + result.json,
#   logs/e1f-dp-bg<BG>-ctx<START>-node{0,1}.log
set -euo pipefail
cd "$(dirname "$0")"

BG=${1:?usage: run_e1f_dp_dual.sh <global_batch> <check_mode> [rounds] [steps] [start_pos]}
CHECK=${2:?check_mode off|bitwise}
ROUNDS=${3:-3}
STEPS=${4:-300}
START=${5:-2048}
if (( BG % 4 != 0 )); then
  echo "global_batch must be divisible by 4, got $BG" >&2
  exit 2
fi
B=$(( BG / 4 ))

MASTER=10.234.1.64
PORT=29653
TR='~/Workspace/venvs/sglang/bin/torchrun'
RESULTS=../experiments/E1F-full-decode-throughput/results/dp
OUT="out-e1f-dp-bg${BG}-ctx${START}"
mkdir -p "$RESULTS/logs" "$RESULTS/$OUT"

echo "== sync runtime to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime'
  rsync -a --exclude __pycache__ dsv4_direct e1f_full_decode_bench.py "$h:e0f-runtime/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

launch() {  # node_rank
  echo "cd ~/e0f-runtime && rm -rf $OUT && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e1f_full_decode_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --local-batch $B --b-semantics dp --check-mode $CHECK --rounds $ROUNDS --steps $STEPS --start-position $START --config-tag nogdr-dp"
}

echo "== E1F DP B_global=$BG (bl=$B) check=$CHECK rounds=$ROUNDS steps=$STEPS start=$START =="
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "$RESULTS/logs/e1f-dp-bg${BG}-ctx${START}-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "$RESULTS/logs/e1f-dp-bg${BG}-ctx${START}-node0.log" || rc0=$?
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
