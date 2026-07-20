#!/usr/bin/env bash
# 17th vertical: single-layer oracle gates on titan064 after the sparse-core
# workspace slimming (and optionally with DSV4_R4_HALF_ACCUM=1 for the
# leverage-3 numeric probe).  Runs e0wf/e0ef (bf16+fp8), e0ff, e0kf on 4 GPUs.
# Usage: ./run_ws_gates.sh <label> [HALF]   -> results/workspace/gates-<label>/
set -euo pipefail
cd "$(dirname "$0")"
LABEL=${1:?label required}
HALF=${2:-}

RESULTS=../experiments/E1F-full-decode-throughput/results/workspace/gates-$LABEL
mkdir -p "$RESULTS"

echo "== sync runtime to titan064 =="
ssh titan064 'mkdir -p ~/e0f-runtime'
rsync -a --exclude __pycache__ dsv4_direct e0wf_window_attention_oracle.py \
  e0ef_ratio128_attention_oracle.py e0ff_ratio4_attention_oracle.py \
  e0kf_fp8_ratio4_paired_gate.py fp8_kv_gate_common.py titan064:e0f-runtime/

HALF_ENV=""
if [ "$HALF" = "HALF" ]; then HALF_ENV="export DSV4_R4_HALF_ACCUM=1;"; fi

ssh titan064 "set -x; cd ~/e0f-runtime && export CUDA_HOME=/usr/local/cuda-13.2 && export PATH=\$CUDA_HOME/bin:\$PATH && export LD_LIBRARY_PATH=\$CUDA_HOME/lib64 && $HALF_ENV TR=~/Workspace/venvs/sglang/bin/torchrun; SR=~/Workspace/DeepSeek-V4-Flash; rc=0
for arm in bf16 fp8; do
  rm -rf out-ws-e0wf-\$arm; \$TR --standalone --nproc_per_node=4 e0wf_window_attention_oracle.py --stage-root \$SR --out-dir out-ws-e0wf-\$arm --kv-dtype \$arm || rc=1
  rm -rf out-ws-e0ef-\$arm; \$TR --standalone --nproc_per_node=4 e0ef_ratio128_attention_oracle.py --stage-root \$SR --out-dir out-ws-e0ef-\$arm --kv-dtype \$arm || rc=1
done
rm -rf out-ws-e0ff; \$TR --standalone --nproc_per_node=4 e0ff_ratio4_attention_oracle.py --stage-root \$SR --out-dir out-ws-e0ff || rc=1
rm -rf out-ws-e0kf; \$TR --standalone --nproc_per_node=4 e0kf_fp8_ratio4_paired_gate.py --stage-root \$SR --out-dir out-ws-e0kf || rc=1
exit \$rc" 2>&1 | tee "$RESULTS/gates.log"
GRC=${PIPESTATUS[0]}

echo "== fetch =="
for d in out-ws-e0wf-bf16 out-ws-e0wf-fp8 out-ws-e0ef-bf16 out-ws-e0ef-fp8 out-ws-e0ff out-ws-e0kf; do
  rsync -a "titan064:e0f-runtime/$d/" "$RESULTS/$d/" || true
done
echo "== accepted summary =="
for d in "$RESULTS"/out-ws-*; do
  python3 - "$d" <<'EOF'
import json,sys,glob,os
d=sys.argv[1]
f=os.path.join(d,'result.json')
if os.path.exists(f):
    r=json.load(open(f))
    print(os.path.basename(d), '->', r.get('accepted'))
else:
    print(os.path.basename(d), '-> result.json missing')
EOF
done
exit "$GRC"
