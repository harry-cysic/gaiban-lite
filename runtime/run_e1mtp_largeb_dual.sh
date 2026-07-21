#!/usr/bin/env bash
# E1MTPLB: large-B chained MTP interleaved bench (18th vertical).
# Modes:
#   ./run_e1mtp_largeb_dual.sh gate  <kv> <idx> [mb=2] [bl=8]  [gate_rounds=132] [start=2048]
#   ./run_e1mtp_largeb_dual.sh timed <kv> <idx> [mb=2] [bl=144] [segments=3] [steps=300] [start=8192] [settle=16]
# Products land in ../experiments/E1F-full-decode-throughput/results/mtp-largeb/.
set -euo pipefail
cd "$(dirname "$0")"

MODE=${1:?usage: run_e1mtp_largeb_dual.sh <gate|timed> <kv_dtype> <indexer_dtype> [...]}
KV=${2:?kv_dtype required}
IDX=${3:?indexer dtype required}
case "$MODE" in
  gate)
    MB=${4:-2}; BL=${5:-8}; GATE_ROUNDS=${6:-132}; START=${7:-2048}; ACCEPT_MODE=${8:-normal}; HCB=${9:-fused}
    EXTRA="--check-mode gate --gate-rounds $GATE_ROUNDS --accept-mode $ACCEPT_MODE --hc-backend $HCB --trace-rows ${TRACE_ROWS:-0}"
    ;;
  timed)
    MB=${4:-2}; BL=${5:-144}; SEGMENTS=${6:-3}; STEPS=${7:-300}; START=${8:-8192}; SETTLE=${9:-16}
    EXTRA="--check-mode off --rounds $SEGMENTS --steps $STEPS --settle-rounds $SETTLE"
    ;;
  *) echo "mode must be gate or timed" >&2; exit 2 ;;
esac
TAG="$KV"
if [ "$IDX" != "bf16" ]; then TAG="$KV-idx$IDX"; fi

MASTER=10.234.1.64
PORT=29681
TR='~/Workspace/venvs/sglang/bin/torchrun'
RESULTS=../experiments/E1F-full-decode-throughput/results/mtp-largeb
AMTAG=""
if [ "${ACCEPT_MODE:-normal}" != "normal" ]; then AMTAG="-${ACCEPT_MODE}"; fi
if [ "${HCB:-fused}" != "fused" ]; then AMTAG="${AMTAG}-${HCB}"; fi
OUT="out-e1mtplb-${MODE}-${TAG}-mb${MB}-bl${BL}-ctx${START}${AMTAG}"
mkdir -p "$RESULTS/logs" "$RESULTS/$OUT"

echo "== sync runtime to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime'
  rsync -a --exclude __pycache__ dsv4_direct e1f_full_decode_bench.py \
    e1mtp_largeb_bench.py "$h:e0f-runtime/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

launch() {  # node_rank
  echo "cd ~/e0f-runtime && rm -rf $OUT && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e1mtp_largeb_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --local-batch $BL --mb-count $MB --start-position $START --kv-dtype $KV --indexer-kv-dtype $IDX $EXTRA --config-tag mtp-largeb-chained-$TAG"
}

echo "== E1MTPLB $MODE kv=$TAG mb=$MB bl=$BL start=$START extra: $EXTRA =="
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "$RESULTS/logs/e1mtplb-${MODE}-${TAG}-mb${MB}-bl${BL}-ctx${START}-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "$RESULTS/logs/e1mtplb-${MODE}-${TAG}-mb${MB}-bl${BL}-ctx${START}-node0.log" || rc0=$?
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
