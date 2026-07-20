#!/usr/bin/env bash
# E1IF: interleaved microbatch PP4 decode bench (thirteenth vertical).
# Modes:
#   ./run_e1if_dual.sh gate  [mb=2] [bl=32] [gate_cycles=132] [start=2048]
#   ./run_e1if_dual.sh timed [mb=4] [bl=32] [rounds=3] [steps=300] [start=2048] [settle=132]
# Topology and NCCL env identical to run_e1f_dp_dual.sh (titan064 stages 0-1,
# titan065 stages 2-3, no-GDR).  Products land in
# ../experiments/E1F-full-decode-throughput/results/interleaved/:
#   out-e1if-<mode>-mb<M>-bl<B>-ctx<START>/rank*.json + result.json,
#   logs/e1if-<mode>-mb<M>-bl<B>-ctx<START>-node{0,1}.log
set -euo pipefail
cd "$(dirname "$0")"

MODE=${1:?usage: run_e1if_dual.sh <gate|timed> [...]}
case "$MODE" in
  gate)
    MB=${2:-2}; BL=${3:-32}; GATE_CYCLES=${4:-132}; START=${5:-2048}
    EXTRA="--check-mode gate --gate-cycles $GATE_CYCLES"
    ;;
  timed)
    MB=${2:-4}; BL=${3:-32}; ROUNDS=${4:-3}; STEPS=${5:-300}; START=${6:-2048}; SETTLE=${7:-132}
    EXTRA="--check-mode off --rounds $ROUNDS --steps $STEPS --settle-cycles $SETTLE"
    ;;
  *) echo "mode must be gate or timed" >&2; exit 2 ;;
esac

MASTER=10.234.1.64
PORT=29671
TR='~/Workspace/venvs/sglang/bin/torchrun'
RESULTS=../experiments/E1F-full-decode-throughput/results/interleaved
OUT="out-e1if-${MODE}-mb${MB}-bl${BL}-ctx${START}"
mkdir -p "$RESULTS/logs" "$RESULTS/$OUT"

echo "== sync runtime to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime'
  rsync -a --exclude __pycache__ dsv4_direct e1f_full_decode_bench.py \
    e1if_interleaved_bench.py "$h:e0f-runtime/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

launch() {  # node_rank
  echo "cd ~/e0f-runtime && rm -rf $OUT && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e1if_interleaved_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --local-batch $BL --mb-count $MB --start-position $START $EXTRA --config-tag nogdr-dp-interleaved"
}

echo "== E1IF $MODE mb=$MB bl=$BL start=$START extra: $EXTRA =="
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "$RESULTS/logs/e1if-${MODE}-mb${MB}-bl${BL}-ctx${START}-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "$RESULTS/logs/e1if-${MODE}-mb${MB}-bl${BL}-ctx${START}-node0.log" || rc0=$?
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
