#!/usr/bin/env bash
# E0qf: dual-node TP4xPP2 gate driver (titan064 stage0 GPU0-3 + titan065
# stage1 GPU0-3, cross-machine handoff over IB NCCL).  Runs from the
# workstation over the ssh aliases titan064/titan065 (ProxyJump earth).
# Two transport configs: no-GDR (default) then GDR (libcuda-onebyte-patch
# LD_LIBRARY_PATH opt-in + NCCL_NET_GDR_LEVEL=SYS; B2-recal calibration).
# Products: e0qf-node{0,1}-{nogdr,gdr}.log and out-e0qf-{nogdr,gdr}/.
set -euo pipefail
cd "$(dirname "$0")"

MASTER=10.234.1.64
TR='~/Workspace/venvs/sglang/bin/torchrun'

echo "== sync runtime to both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/e0f-runtime'
  rsync -a --exclude __pycache__ dsv4_direct e0qf_pp2_dual_gate.py "$h:e0f-runtime/"
done

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'
ENV_GDR='export LD_LIBRARY_PATH=/home/cysic/libcuda-onebyte-patch:$LD_LIBRARY_PATH NCCL_NET_GDR_LEVEL=SYS'

run_gate() {  # tag extra_env port
  local tag=$1 extra=$2 port=$3 rc0=0 rc1=0
  echo "== gate $tag (port $port) =="
  local env_remote="$ENV_BASE"
  if [ -n "$extra" ]; then env_remote="$env_remote; $extra"; fi
  launch() {  # node_rank
    echo "cd ~/e0f-runtime && $env_remote; CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --nnodes 2 --node-rank $1 --nproc-per-node 4 --master-addr $MASTER --master-port $port e0qf_pp2_dual_gate.py --config-tag $tag --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0qf-$tag"
  }
  ssh titan065 "$(launch 1)" > "e0qf-node1-$tag.log" 2>&1 &
  local pid=$!
  ssh titan064 "$(launch 0)" 2>&1 | tee "e0qf-node0-$tag.log" || rc0=$?
  wait "$pid" || rc1=$?
  echo "== gate $tag exit codes: node0=$rc0 node1=$rc1 =="
}

run_gate nogdr "" 29631
run_gate gdr "$ENV_GDR" 29632

echo "== fetch results =="
for tag in nogdr gdr; do
  mkdir -p "out-e0qf-$tag"
  rsync -a "titan064:e0f-runtime/out-e0qf-$tag/" "out-e0qf-$tag/"
  rsync -a "titan065:e0f-runtime/out-e0qf-$tag/" "out-e0qf-$tag/"
done

echo "== GPU memory check =="
for h in titan064 titan065; do
  echo "--- $h"
  ssh "$h" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
done
echo ALL_DONE
