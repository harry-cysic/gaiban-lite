#!/usr/bin/env bash
# B1-recal allreduce sweep. Run ON titan064 / titan065 (NOT dsv4exp).
# venv: ~/Workspace/venvs/sglang. NCCL_P2P_LEVEL=SYS forces P2P across SYS
# (cross-socket) links; baseline run disables P2P to confirm patch payoff.
set -u
cd "$(dirname "$0")"
TR=${TR:-$HOME/Workspace/venvs/sglang/bin/torchrun}

run() {  # tag  devices  nproc  extra_env
  echo "### CONFIG: $1  (GPUs=$2, n=$3, DIM=${B1_DIM:-7168})"
  CUDA_VISIBLE_DEVICES=$2 B1_TAG="$1" NCCL_P2P_LEVEL=SYS $4 \
    "$TR" --standalone --nnodes=1 --nproc_per_node=$3 b1_allreduce.py
}

echo "=== P2P-usage check (NCCL_DEBUG=INFO, intra-socket pair) ==="
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_LEVEL=SYS NCCL_DEBUG=INFO B1_TAG=probe B1_DIM=4096 \
  "$TR" --standalone --nnodes=1 --nproc_per_node=2 b1_allreduce.py 2>&1 \
  | grep -iE "via P2P|direct pointer|P2P/IPC|Connected all" | head -8

for DIM in 7168 4096; do
  export B1_DIM=$DIM
  run "TP4 intra-socket0 (0-3 PCIe) DIM=$DIM"   0,1,2,3         4 ""
  run "TP4 intra-socket1 (4-7 PCIe) DIM=$DIM"   4,5,6,7         4 ""
  run "TP4 CROSS (0,1,4,5 xGMI) DIM=$DIM"       0,1,4,5         4 ""
  run "TP8 CROSS (0-7 xGMI) DIM=$DIM"           0,1,2,3,4,5,6,7 8 ""
done

echo "### CONFIG: TP4 intra-socket0 — P2P DISABLED baseline (DIM=4096)"
CUDA_VISIBLE_DEVICES=0,1,2,3 B1_TAG="TP4 intra P2P-DISABLED" B1_DIM=4096 NCCL_P2P_DISABLE=1 \
  "$TR" --standalone --nnodes=1 --nproc_per_node=4 b1_allreduce.py 2>/dev/null
echo ALL_DONE
