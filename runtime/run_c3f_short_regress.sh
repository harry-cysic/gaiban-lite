#!/usr/bin/env bash
# C3F (25th vertical): D0 *short* golden gate regression.
# The 25th vertical changes shared code (the three attention __call__ paths and
# run_prompt's step plan), so the frozen short gate must still score 472/482.
# Short prompts are 10-22 tokens, so --prefill-chunk never splits anything here;
# that is the point -- this arm proves the whole-sequence path is unperturbed.
set -euo pipefail
cd "$(dirname "$0")"

# $2 selects the prefill sparse core, which is what separates the two frozen
# short-gate baselines: torch -> 468/482 (out-e0e2e), tilelang -> 472/482
# (out-e0e2e-tl-tilelang).  Both must be reproduced exactly.
TAG=${1:-shortregress}
BACKEND=${2:-torch}
shift 2 || true
ARM_ARGS="$*"
MASTER=10.234.1.64
PORT=29662
TR='~/Workspace/venvs/sglang/bin/torchrun'
ORACLE_SRC=../experiments/D0-reference-oracle/results/oracle-mp8.json
OUT="out-c3f-$TAG"

echo "== sync runtime + short golden oracle to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime/reference/inference'
  rsync -a --exclude __pycache__ dsv4_direct e0ef2e_golden_gate.py "$h:e0f-runtime/"
  rsync -a "$ORACLE_SRC" "$h:e0f-runtime/oracle-mp8.json"
  rsync -a ../reference/inference/kernel.py "$h:e0f-runtime/reference/inference/"
done

# NCCL_P2P_LEVEL=SYS is mandatory: unset silently falls back to SHM.
ENV_BASE="export CUDA_HOME=/usr/local/cuda-13.2; export PATH=\$CUDA_HOME/bin:\$PATH; export LD_LIBRARY_PATH=\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1; export DSV4_PREFILL_SPARSE_BACKEND=$BACKEND"

launch() {  # node_rank
  echo "cd ~/e0f-runtime && $ENV_BASE; $TR --nnodes 2 --node-rank $1 --nproc-per-node 8 --master-addr $MASTER --master-port $PORT e0ef2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --oracle-json oracle-mp8.json --out-dir $OUT --hc-backends eager,fused $ARM_ARGS"
}
rc0=0 rc1=0
ssh titan065 "$(launch 1)" > "c3f-$TAG-node1.log" 2>&1 &
pid=$!
ssh titan064 "$(launch 0)" 2>&1 | tee "c3f-$TAG-node0.log" || rc0=$?
wait "$pid" || rc1=$?
echo "== gate exit codes: node0=$rc0 node1=$rc1 =="

mkdir -p "$OUT"
rsync -a "titan064:e0f-runtime/$OUT/" "$OUT/" || true
rsync -a "titan065:e0f-runtime/$OUT/" "$OUT/" || true
for h in titan064 titan065; do
  echo "--- $h"; ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done
echo ALL_DONE rc0=$rc0 rc1=$rc1
