#!/usr/bin/env bash
# E2F: nsys kernel-level trace of the decode super-stage graph replay.
#   ./run_e2f_nsys.sh <host> <layers> <tag> [extra probe args...]
# Only LOCAL_RANK 0 is traced; the other three ranks run bare, so the traced
# rank sees the same collectives it sees in an untraced run.  The probe emits
# cudaProfilerStart/Stop around the timed segment (--cuda-profiler-range), so
# load/warmup/capture stay out of the report.  --cuda-graph-trace=node is what
# makes individual kernel nodes visible inside a replay instead of one opaque
# graph range.
set -euo pipefail
cd "$(dirname "$0")"

HOST=${1:?usage: run_e2f_nsys.sh <host> <layers> <tag> [extra args]}
LAYERS=${2:?layer range}
TAG=${3:?tag}
shift 3

PY='~/Workspace/venvs/sglang/bin/python'
TR='~/Workspace/venvs/sglang/bin/torchrun'
RESULTS=../experiments/E2F-decode-latency-profile/results
OUT="out-e2f-nsys-${TAG}"
mkdir -p "$RESULTS/logs" "$RESULTS/$OUT"

echo "== sync runtime to $HOST =="
ssh "$HOST" 'mkdir -p ~/e0f-runtime'
rsync -a --exclude __pycache__ dsv4_direct e1f_full_decode_bench.py \
  e2f_decode_phase_probe.py "$HOST:e0f-runtime/"

ssh "$HOST" 'cat > ~/e0f-runtime/e2f_nsys_wrap.sh' <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
if [ "${LOCAL_RANK:-0}" = "0" ]; then
  exec nsys profile -t cuda,nvtx --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --sample=none --cpuctxsw=none \
    -o "$NSYS_OUT" --force-overwrite true "$@"
fi
exec "$@"
WRAP

ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

echo "== nvidia-smi BEFORE ($HOST) ==" | tee "$RESULTS/logs/e2f-nsys-${TAG}-smi-before.txt"
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader' \
  | tee -a "$RESULTS/logs/e2f-nsys-${TAG}-smi-before.txt"

rc=0
ssh "$HOST" "cd ~/e0f-runtime && rm -rf $OUT && chmod +x e2f_nsys_wrap.sh && $ENV_BASE; export NSYS_OUT=~/e0f-runtime/$OUT/e2f-${TAG}; mkdir -p $OUT; $TR --standalone --nproc-per-node 4 --no-python bash e2f_nsys_wrap.sh $PY e2f_decode_phase_probe.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir $OUT --layers $LAYERS --cuda-profiler-range $*" \
  2>&1 | tee "$RESULTS/logs/e2f-nsys-${TAG}-${HOST}.log" || rc=$?

echo "== kernel summary =="
ssh "$HOST" "cd ~/e0f-runtime/$OUT && $ENV_BASE; nsys stats --report cuda_gpu_kern_sum --format csv --output . e2f-${TAG}.nsys-rep >/dev/null 2>&1; nsys stats --report cuda_gpu_trace --format csv --output . e2f-${TAG}.nsys-rep >/dev/null 2>&1; ls -la" || true

echo "== fetch results (csv + json only; .nsys-rep stays on $HOST) =="
rsync -a --exclude '*.nsys-rep' --exclude '*.sqlite' "$HOST:e0f-runtime/$OUT/" "$RESULTS/$OUT/" || true

echo "== nvidia-smi AFTER ($HOST) ==" | tee "$RESULTS/logs/e2f-nsys-${TAG}-smi-after.txt"
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader' \
  | tee -a "$RESULTS/logs/e2f-nsys-${TAG}-smi-after.txt"

echo "== exit code: $rc =="
exit $rc
