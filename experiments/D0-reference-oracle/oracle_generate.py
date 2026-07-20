"""D0 golden-token oracle: batch greedy decode with the official reference
implementation, recording token IDs + environment fingerprint as JSON.

Run from reference/inference/ (imports model.py/generate.py from cwd):
  torchrun --standalone --nproc-per-node 8 oracle_generate.py \
    --ckpt-path ~/Workspace/DeepSeek-V4-Flash-mp8 --config config.json \
    --input-file oracle_prompts.txt --max-new-tokens 128 --out oracle.json

Semantics identical to generate.py batch path with temperature=0 (argmax);
max_batch_size is raised to the prompt count. The produced JSON is the frozen
golden reference for later runtime golden-token comparison.
"""
import hashlib
import json
import os
import socket
import sys
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from generate import generate

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../encoding")))
from encoding_dsv4 import encode_messages


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    with open(args.config) as f:
        margs = ModelArgs(**json.load(f))

    with open(args.input_file) as f:
        prompts = [p for p in f.read().split("\n\n") if p.strip()]
    margs.max_batch_size = len(prompts)

    with torch.device("cuda"):
        model = Transformer(margs)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    ckpt_file = os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    load_model(model, ckpt_file, strict=False)
    torch.set_default_device("cuda")

    prompt_tokens = [
        tokenizer.encode(encode_messages([{"role": "user", "content": p}], thinking_mode="chat"))
        for p in prompts
    ]
    completion_tokens = generate(
        model, prompt_tokens, args.max_new_tokens, tokenizer.eos_token_id, temperature=0.0
    )

    if rank == 0:
        record = {
            "experiment": "D0-reference-oracle",
            "decode": {"temperature": 0.0, "mode": "argmax", "max_new_tokens": args.max_new_tokens},
            "environment": {
                "hostname": socket.gethostname(),
                "world_size": world_size,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "config_md5": file_md5(args.config),
                "input_file_md5": file_md5(args.input_file),
                "checkpoint": {
                    "path": args.ckpt_path,
                    "files": sorted(
                        (fn, os.path.getsize(os.path.join(args.ckpt_path, fn)))
                        for fn in os.listdir(args.ckpt_path)
                        if fn.endswith(".safetensors")
                    ),
                },
            },
            "prompts": [
                {
                    "prompt": p,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "completion_text": tokenizer.decode(ct),
                }
                for p, pt, ct in zip(prompts, prompt_tokens, completion_tokens)
            ],
        }
        with open(args.out, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=1)
        print(f"WROTE {args.out}")
        for r in record["prompts"]:
            print("---", r["prompt"][:40])
            print(r["completion_text"])

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
