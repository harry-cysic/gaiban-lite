"""D0L long-prompt golden-token oracle: reference MP=8 greedy decode over the
D0L long prompt set, recording token IDs + environment fingerprint as JSON.

Differences from ``D0-reference-oracle/oracle_generate.py`` (and why):

- **Input format is JSON, not blank-line-separated text.**  The long prompts are
  excerpts of real documents and contain blank lines, so the D0 ``"\\n\\n"``
  split cannot represent them.
- **One prompt per ``generate`` call (batch = 1).**  ``generate.py`` prefills
  only ``min(prompt_lens)`` tokens and then advances one token at a time with
  ``prompt_mask`` overriding, so a mixed batch of 1024- and 8192-token prompts
  would run ~7000 single-token forwards instead of a prefill.  Batch 1 also
  gives every prompt a full-length ``start_pos = 0`` prefill -- exactly the
  shape the E2E gate replays -- and keeps the activation peak at its minimum,
  which matters because the MP=8 residency leaves only ~1.9 GB per card.
- **State is explicitly reset between prompts.**  The reference KV/compressor
  buffers are ``register_buffer`` state that persists across ``generate``
  calls.  For prompt lengths that are multiples of the compress ratio the
  ``start_pos == 0`` branches happen to overwrite everything that is later
  read, but relying on that is fragile; ``reset_model_state`` restores the
  exact ``__init__`` initial values (``kv_state`` 0, ``score_state`` -inf,
  ``kv_cache`` 0) so each prompt is bitwise equivalent to a fresh process.
- **max_seq_len is raised** from the config default 4096 to cover the longest
  prompt plus the decode budget, and per-prompt peak memory is recorded so the
  feasible ceiling is documented rather than assumed.

Semantics otherwise identical to the D0 oracle: greedy (temperature = 0 ->
argmax), reference implementation, MP = 8 single node.

Run from reference/inference/ (imports model.py/generate.py from cwd):
  torchrun --standalone --nproc-per-node 8 oracle_long_generate.py \
    --ckpt-path ~/Workspace/DeepSeek-V4-Flash-mp8 --config config.json \
    --prompts long_prompts.json --max-new-tokens 64 \
    --max-seq-len 8448 --out oracle-long.json
"""

import hashlib
import json
import os
import socket
import sys
import time
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


def reset_model_state(model):
    """Restore every persistent inference buffer to its constructor value.

    ``Compressor`` initialises ``kv_state`` to zeros and ``score_state`` to
    -inf (model.py:303-304); ``Attention``/``Indexer`` KV caches start at zero
    (model.py:399, 474).  ``freqs_cis`` is a constant table and is left alone.
    """

    for name, buffer in model.named_buffers():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "score_state":
            buffer.fill_(float("-inf"))
        elif leaf in ("kv_state", "kv_cache"):
            buffer.zero_()


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="0 = longest prompt + max_new_tokens rounded up to 128",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="skip prompts longer than this (0 = no cap); used to walk the "
        "feasible ceiling without editing the prompt set",
    )
    parser.add_argument(
        "--drop-mtp",
        action="store_true",
        help="build the model with n_mtp_layers=0.  ModelArgs defaults it to 1 "
        "and the Flash config does not override it, so a whole extra MTP block "
        "(attention + 32 local FP4 experts) is allocated and loaded -- but "
        "Transformer.forward never touches self.mtp, so on the generate path it "
        "is pure dead residency.  Dropping it buys back the headroom that "
        "decides whether a 4096-token prefill fits in 24 GB.  Token-identical "
        "by construction; verified against the MTP-resident run.",
    )
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
        config_payload = json.load(f)
    with open(args.prompts, encoding="utf-8") as f:
        prompt_payload = json.load(f)
    prompts = [entry["prompt"] for entry in prompt_payload["prompts"]]

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    prompt_tokens = [
        tokenizer.encode(
            encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
        )
        for p in prompts
    ]
    if args.max_prompt_tokens > 0:
        keep = [
            i for i, t in enumerate(prompt_tokens) if len(t) <= args.max_prompt_tokens
        ]
    else:
        keep = list(range(len(prompt_tokens)))
    longest = max(len(prompt_tokens[i]) for i in keep)

    margs = ModelArgs(**config_payload)
    margs.max_batch_size = 1
    if args.drop_mtp:
        margs.n_mtp_layers = 0
    if args.max_seq_len > 0:
        margs.max_seq_len = args.max_seq_len
    else:
        need = longest + args.max_new_tokens
        margs.max_seq_len = ((need + 127) // 128) * 128
    if rank == 0:
        print(
            f"[D0L] {len(keep)}/{len(prompt_tokens)} prompts, longest "
            f"{longest} tokens, max_seq_len {margs.max_seq_len}, "
            f"max_new_tokens {args.max_new_tokens}",
            flush=True,
        )

    with torch.device("cuda"):
        model = Transformer(margs)
    ckpt_file = os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    load_model(model, ckpt_file, strict=False)
    torch.set_default_device("cuda")

    free_after_load, total_bytes = torch.cuda.mem_get_info()
    if rank == 0:
        print(
            f"[D0L] loaded: free {free_after_load / 2**30:.2f} GiB / "
            f"{total_bytes / 2**30:.2f} GiB",
            flush=True,
        )

    def dump(records):
        """Write the JSON after every prompt.

        Prompts run shortest-first and the 8192 arm is the one at risk of
        exhausting the ~1.9 GB that MP=8 residency leaves free, so an OOM on a
        late prompt must not destroy the golden tokens already produced -- the
        partial file *is* the recorded ceiling."""

        payload = {
            "experiment": "D0L-long-prompt-oracle",
            "decode": {
                "temperature": 0.0,
                "mode": "argmax",
                "max_new_tokens": args.max_new_tokens,
                "batch_size": 1,
                "state_reset_between_prompts": True,
            },
            "environment": {
                "hostname": socket.gethostname(),
                "world_size": world_size,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "max_seq_len": margs.max_seq_len,
                "n_mtp_layers": margs.n_mtp_layers,
                "config_md5": file_md5(args.config),
                "prompts_md5": file_md5(args.prompts),
                "total_bytes_per_gpu": int(total_bytes),
                "free_bytes_after_load": int(free_after_load),
                "checkpoint": {
                    "path": args.ckpt_path,
                    "files": sorted(
                        (fn, os.path.getsize(os.path.join(args.ckpt_path, fn)))
                        for fn in os.listdir(args.ckpt_path)
                        if fn.endswith(".safetensors")
                    ),
                },
            },
            "length_histogram": {},
            "prompts": records,
        }
        histogram = {}
        for record in records:
            key = str(record["prompt_len"])
            histogram[key] = histogram.get(key, 0) + 1
        payload["length_histogram"] = histogram
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    records = []
    for index in keep:
        tokens = prompt_tokens[index]
        reset_model_state(model)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        completion = generate(
            model, [tokens], args.max_new_tokens, tokenizer.eos_token_id,
            temperature=0.0,
        )[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        free_bytes, _ = torch.cuda.mem_get_info()
        record = {
            "index": index,
            "prompt": prompts[index],
            "prompt_tokens": tokens,
            "completion_tokens": completion,
            "completion_text": tokenizer.decode(completion),
            "prompt_len": len(tokens),
            "completion_len": len(completion),
            "wall_seconds": elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "free_bytes_after": int(free_bytes),
        }
        records.append(record)
        if rank == 0:
            dump(records)
            print(
                f"[D0L] prompt {index}: {len(tokens)} prompt tokens -> "
                f"{len(completion)} completion tokens in {elapsed:.1f}s "
                f"(peak alloc {record['peak_allocated_bytes'] / 2**30:.2f} GiB, "
                f"reserved {record['peak_reserved_bytes'] / 2**30:.2f} GiB, "
                f"free {free_bytes / 2**30:.2f} GiB)",
                flush=True,
            )
            print("      " + record["completion_text"][:160].replace("\n", " "), flush=True)

    if rank == 0:
        dump(records)
        print(f"WROTE {args.out}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
