#!/usr/bin/env bash
# B2-recal: titan064<->titan065 IB NCCL 标定复跑(no-GDR + GDR)。
# 在开发工作站上执行(经 ProxyJump earth 的 ssh 别名 titan064/titan065)。
# 源码复用 ../gaiban/experiments/rdma-nccl-qualification/*.cu(只读复用,不改动)。
# 严禁涉及 dsv4exp。
set -eu
cd "$(dirname "$0")"
TS=${TS:-$(date +%Y%m%d-%H%M%S)}
OUT=results/$TS
mkdir -p "$OUT"
SRC=${SRC:-$HOME/gaiban/experiments/rdma-nccl-qualification}

echo "== build on both nodes =="
for h in titan064 titan065; do
  ssh "$h" 'mkdir -p ~/b2-ib-recal'
  scp -q "$SRC/nccl_pair_bench.cu" "$SRC/nccl_pp_sendrecv_bench.cu" "$h:b2-ib-recal/"
  ssh "$h" 'cd ~/b2-ib-recal && /usr/local/cuda/bin/nvcc -O2 -arch=sm_89 nccl_pair_bench.cu -o nccl_pair_bench -lnccl && /usr/local/cuda/bin/nvcc -O2 -arch=sm_89 nccl_pp_sendrecv_bench.cu -o nccl_pp_sendrecv_bench -lnccl && echo BUILD_OK'
done

ENV_COMMON="NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET CUDA_VISIBLE_DEVICES=0"
ENV_GDR="LD_LIBRARY_PATH=/home/cysic/libcuda-onebyte-patch NCCL_NET_GDR_LEVEL=SYS"
MASTER=10.234.1.64

run_pair() {  # tag  extra_env  port
  local tag=$1 extra=$2 port=$3
  echo "== pair_bench $tag (port $port) =="
  ssh titan064 "cd ~/b2-ib-recal && env $ENV_COMMON $extra ./nccl_pair_bench --rank 0 --nranks 2 --master $MASTER --port $port" \
    > "$OUT/pair_${tag}_rank0.log" 2>&1 &
  local pid=$!
  ssh titan065 "cd ~/b2-ib-recal && env $ENV_COMMON $extra ./nccl_pair_bench --rank 1 --nranks 2 --master $MASTER --port $port" \
    > "$OUT/pair_${tag}_rank1.log" 2>&1
  wait "$pid"
}

run_pp() {  # tag  extra_env  port  sizes
  local tag=$1 extra=$2 port=$3 sizes=$4
  echo "== pp_sendrecv $tag (port $port) =="
  ssh titan064 "cd ~/b2-ib-recal && env $ENV_COMMON $extra ./nccl_pp_sendrecv_bench --rank 0 --master $MASTER --port $port --sizes $sizes" \
    > "$OUT/pp_${tag}_rank0.log" 2>&1 &
  local pid=$!
  ssh titan065 "cd ~/b2-ib-recal && env $ENV_COMMON $extra ./nccl_pp_sendrecv_bench --rank 1 --master $MASTER --port $port --sizes $sizes" \
    > "$OUT/pp_${tag}_rank1.log" 2>&1
  wait "$pid"
}

# Flash PP payload 关键点:32KB/行 => B=128:4.2MB, B=256:8.4MB, B=512:16.8MB
PP_SIZES=458752,524288,786432,4194304,8388608,16777216,33554432,67108864

run_pair no-gdr ""         21001
run_pair gdr    "$ENV_GDR" 21002
run_pp   no-gdr ""         21003 "$PP_SIZES"
run_pp   gdr    "$ENV_GDR" 21004 "$PP_SIZES"

echo "== transport summary =="
grep -l "" "$OUT"/*_rank0.log | while read -r f; do
  echo "--- $f"
  grep -iE "GDRDMA|GDR [01]|via NET|Using network" "$f" | sort -u | head -5
done
echo ALL_DONE
