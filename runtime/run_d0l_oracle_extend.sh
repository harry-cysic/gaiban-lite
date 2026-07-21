#!/usr/bin/env bash
# D0L: regenerate the long-prompt golden over the FULL prompt set (10 prompts,
# including the two 8192-token ones the original run had to skip).
#
#   ./run_d0l_oracle_extend.sh <host> <tag> [extra generator args...]
#
# Why this is now possible: hc_post is chunked along s (bitwise, see
# experiments/D0L-long-prompt-oracle/hc_post_chunk_selfcheck.py), which frees
# exactly the 2.00 GiB single allocation that made an 8192-token prefill OOM at
# MP=8, and --drop-mtp gives back another 0.52 GiB of never-read residency.
#
# Runs from a *copy* of the reference inference code, not from the one inside
# ~/Workspace/DeepSeek-V4-Flash/inference/ -- other work reads that tree, and a
# generator should not be able to perturb it.
#
# The acceptance question this run answers is not just "does 8192 fit": the
# eight prompts that already have goldens must come back **bitwise identical**,
# which is what verifies the hc_post change on the real model rather than on
# synthetic tensors.
set -euo pipefail
cd "$(dirname "$0")"

HOST=${1:?usage: run_d0l_oracle_extend.sh <host> <tag> [extra args]}
TAG=${2:?tag}
shift 2

# NOT ~/d0l-gen: an unquoted ~ expands *locally* (to /home/harry) before it
# ever reaches the remote, whose user is different.  A relative path is
# resolved against the remote home by both ssh and rsync.
REMOTE=d0l-gen
RESULTS=../experiments/D0L-long-prompt-oracle/results
PY='~/Workspace/venvs/sglang/bin/torchrun'
mkdir -p "$RESULTS"

echo "== stage the patched reference + generator on $HOST =="
ssh "$HOST" "rm -rf $REMOTE && mkdir -p $REMOTE"
rsync -a ../reference/inference/model.py ../reference/inference/generate.py \
  ../reference/inference/kernel.py ../reference/inference/config.json \
  "$HOST:$REMOTE/"
rsync -a ../experiments/D0L-long-prompt-oracle/oracle_long_generate.py \
  ../experiments/D0L-long-prompt-oracle/long_prompts.json "$HOST:$REMOTE/"

# encoding_dsv4 (the tokenizer) ships in the checkpoint tree, not with the
# inference code, so it has to be on PYTHONPATH explicitly.
ENV_BASE='export CUDA_HOME=/usr/local/cuda-13.2; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export PYTHONPATH=$HOME/Workspace/DeepSeek-V4-Flash/encoding${PYTHONPATH:+:$PYTHONPATH}; export NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1'

echo "== nvidia-smi BEFORE ==" | tee "$RESULTS/oracle-$TAG-smi-before.txt"
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader' \
  | tee -a "$RESULTS/oracle-$TAG-smi-before.txt"

rc=0
ssh "$HOST" "cd $REMOTE && $ENV_BASE; $PY --standalone --nproc-per-node 8 oracle_long_generate.py --ckpt-path ~/Workspace/DeepSeek-V4-Flash-mp8 --config config.json --prompts long_prompts.json --drop-mtp --out oracle-$TAG.json $*" \
  2>&1 | tee "$RESULTS/oracle-$TAG.log" || rc=$?

echo "== fetch =="
rsync -a "$HOST:$REMOTE/oracle-$TAG.json" "$RESULTS/" || true

echo "== nvidia-smi AFTER ==" | tee "$RESULTS/oracle-$TAG-smi-after.txt"
ssh "$HOST" 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader' \
  | tee -a "$RESULTS/oracle-$TAG-smi-after.txt"

echo "== exit code: $rc =="
exit $rc
