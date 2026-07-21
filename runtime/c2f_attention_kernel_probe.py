"""C2F attention lever probe: tilelang sparse_attn vs the runtime torch path.

Re-attribution showed prefill is 58% attention, and the runtime still uses the
torch masked-einsum correctness implementation while the reference tilelang
`sparse_attn` kernel was never wired in.  This probe measures both at prefill
shapes (identical signature) and checks numerics, deciding whether the lever is
real before any integration work.

sm89 fits at most 16 heads per sparse_attn launch (A4F), so the tilelang arm
runs a head loop; the torch arm takes all heads at once as the runtime does.

  <venv>/bin/python c2f_attention_kernel_probe.py --seqlen 8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + os.pathsep + os.environ.get("PATH", "")

import torch

from dsv4_direct.attention import torch_sparse_attention

HEADS, HEAD_DIM, WINDOW, TOPK = 64, 512, 128, 512


def bench(fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default=None,
                        help="directory holding the reference kernel.py")
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--head-chunk", type=int, default=16)
    parser.add_argument("--numeric-seqlen", type=int, default=512)
    parser.add_argument("--row-block", type=int, default=1024,
                        help="row blocking the runtime applies to the torch path")
    parser.add_argument("--out", default="c2f-attn-kernel.json")
    args = parser.parse_args()

    reference_dir = args.reference_dir
    if reference_dir is None:
        for candidate in ("reference/inference", "../reference/inference", "flash-oracle/reference/inference"):
            path = Path(os.path.expanduser("~")) / candidate
            if path.exists():
                reference_dir = str(path)
                break
    if reference_dir is None or not Path(reference_dir).exists():
        print("reference kernel.py directory not found; pass --reference-dir")
        return 1
    sys.path.insert(0, reference_dir)
    from kernel import sparse_attn as tilelang_sparse_attn

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260721)
    scale = HEAD_DIM ** -0.5
    candidates = WINDOW + TOPK

    def build(seqlen: int):
        kv_rows = WINDOW + seqlen // args.ratio
        query = torch.randn(
            1, seqlen, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=generator
        ) * 0.05
        latent = torch.randn(
            1, kv_rows, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=generator
        ) * 0.05
        sink = torch.randn(HEADS, device=device, dtype=torch.float32, generator=generator)
        # per-row candidate sets, padded with -1 like the runtime's early positions
        picks = torch.randint(
            0, kv_rows, (1, seqlen, candidates), device=device, dtype=torch.int32, generator=generator
        )
        limit = torch.arange(1, seqlen + 1, device=device).view(1, seqlen, 1) // args.ratio + WINDOW
        picks = torch.where(picks < limit.to(torch.int32), picks, torch.full_like(picks, -1))
        return query, latent, sink, picks

    report: dict = {
        "experiment": "C2F-attention-kernel",
        "gpu": torch.cuda.get_device_name(0),
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "candidates": candidates,
        "head_chunk": args.head_chunk,
    }

    # ---- numerics on a small sequence ----
    query, latent, sink, picks = build(args.numeric_seqlen)
    reference_out = torch_sparse_attention(query, latent, sink, picks, scale)
    chunks = []
    for start in range(0, HEADS, args.head_chunk):
        stop = start + args.head_chunk
        chunks.append(
            tilelang_sparse_attn(
                query[:, :, start:stop].contiguous(),
                latent,
                sink[start:stop].contiguous(),
                picks,
                scale,
            )
        )
    tilelang_out = torch.cat(chunks, dim=2)
    delta = (tilelang_out.float() - reference_out.float()).abs()
    report["numerics"] = {
        "seqlen": args.numeric_seqlen,
        "rel_fro": float(
            torch.linalg.norm(tilelang_out.float() - reference_out.float())
            / torch.linalg.norm(reference_out.float())
        ),
        "max_abs": float(delta.max()),
    }
    print(json.dumps(report["numerics"]))
    del query, latent, sink, picks, reference_out, tilelang_out, chunks
    torch.cuda.empty_cache()

    # ---- throughput at prefill shape ----
    query, latent, sink, picks = build(args.seqlen)

    def torch_arm():
        block = args.row_block or args.seqlen
        outputs = []
        for begin in range(0, args.seqlen, block):
            end = min(begin + block, args.seqlen)
            outputs.append(
                torch_sparse_attention(
                    query[:, begin:end].contiguous(),
                    latent,
                    sink,
                    picks[:, begin:end].contiguous(),
                    scale,
                )
            )
        return torch.cat(outputs, dim=1)

    def tilelang_arm():
        outputs = []
        for start in range(0, HEADS, args.head_chunk):
            stop = start + args.head_chunk
            outputs.append(
                tilelang_sparse_attn(
                    query[:, :, start:stop].contiguous(),
                    latent,
                    sink[start:stop].contiguous(),
                    picks,
                    scale,
                )
            )
        return torch.cat(outputs, dim=2)

    torch.cuda.reset_peak_memory_stats()
    tilelang_ms = bench(tilelang_arm)
    tilelang_peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    torch_ms = bench(torch_arm, iters=3, warmup=1)
    torch_peak = torch.cuda.max_memory_allocated() / 2**30
    report["throughput"] = {
        "seqlen": args.seqlen,
        "kv_rows": WINDOW + args.seqlen // args.ratio,
        "row_block": args.row_block,
        "torch_ms": round(torch_ms, 3),
        "tilelang_ms": round(tilelang_ms, 3),
        "speedup": round(torch_ms / tilelang_ms, 3),
        "torch_peak_gib": round(torch_peak, 2),
        "tilelang_peak_gib": round(tilelang_peak, 2),
    }
    print(json.dumps(report["throughput"]))
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
