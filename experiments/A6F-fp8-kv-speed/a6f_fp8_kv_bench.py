"""A6F: FP8 KV decode-attention speed microbench on Flash geometry (sm89, 1 GPU).

Measures the runtime torch sparse-attention decode core (the only part FP8 KV
changes) for the three Flash attention families, comparing KV cache storage:

  - bf16        : latent [bl, N, 512] bf16 (current runtime, dsv4_direct)
  - fp8_cast    : latent [bl, N, 512] float8_e4m3fn, dequant at read via
                  .to(bfloat16) (constant implicit scale)
  - fp8_scale   : fp8 cache + per-token fp32 scale [bl, N, 1], gathered and
                  multiplied after the cast (per-token-scale upper bound)

The core mirrors runtime/dsv4_direct/attention.py
`_torch_sparse_decode_padded_prevalidated` (ratio-128 / ratio-4, masked) and
window_attention.py `_window_sparse_decode_prevalidated` (window, unmasked):
gather -> (dequant) -> fp32 einsum QK -> sink softmax -> fp32 einsum PV.
Timing is CUDA-graph replay (capture once, replay ITERS, 3 rounds, CUDA events).

Families (decode step, seqlen=1, h=64, d=512):
  window   : K = 128,              cache N = 128
  ratio128 : K = 128 + ctx//128,   cache N = 128 + ctx//128
  ratio4   : K = 128 + 512,        cache N = 128 + ctx//4  (topk 512 random)

Usage: python a6f_fp8_kv_bench.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import torch

LATENT_DIM = 512
WINDOW = 128
HEADS = 64
TOPK4 = 512
ITERS = 50
ROUNDS = 3


def sparse_core(query_f32, selected, sink, valid_mask, scale_sel, softmax_scale):
    """Mirror of the runtime decode sparse core, from gathered rows onward."""
    if selected.dtype == torch.float8_e4m3fn:
        selected = selected.to(torch.bfloat16)
        if scale_sel is not None:
            selected = selected * scale_sel
    if valid_mask is not None:
        selected = selected.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    scores = torch.einsum(
        "bshd,bskd->bshk", query_f32, selected.float()
    ) * softmax_scale
    if valid_mask is not None:
        valid = valid_mask.unsqueeze(2)
        scores = scores.masked_fill(~valid, float("-inf"))
    maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    exponent = torch.exp(scores - maximum)
    if valid_mask is not None:
        exponent = exponent.masked_fill(~valid, 0.0)
    denominator = exponent.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
    probabilities = exponent / denominator
    output = torch.einsum("bshk,bskd->bshd", probabilities, selected.float())
    return output.to(torch.bfloat16)


def full_step(query_f32, latent, sink, batch_idx, gather_idx, valid_mask,
              scale_cache, softmax_scale):
    """Gather (the KV read) + core. This is what one decode layer runs."""
    selected = latent[batch_idx, gather_idx]
    scale_sel = None
    if scale_cache is not None:
        scale_sel = scale_cache[batch_idx, gather_idx]
    return sparse_core(query_f32, selected, sink, valid_mask, scale_sel,
                       softmax_scale)


def build_inputs(family, bl, ctx, variant, device, generator):
    if family == "window":
        cache_n, k = WINDOW, WINDOW
        masked = False
    elif family == "ratio128":
        cache_n = k = WINDOW + ctx // 128
        masked = True
    elif family == "ratio4":
        cache_n = WINDOW + ctx // 4
        k = WINDOW + TOPK4
        masked = True
    else:
        raise ValueError(family)

    query = torch.randn(bl, 1, HEADS, LATENT_DIM, device=device,
                        dtype=torch.float32, generator=generator)
    query_f32 = (query / LATENT_DIM ** 0.5)
    sink = torch.randn(HEADS, device=device, dtype=torch.float32,
                       generator=generator).view(1, 1, HEADS, 1)

    kv_bf16 = torch.randn(bl, cache_n, LATENT_DIM, device=device,
                          dtype=torch.float32, generator=generator
                          ).to(torch.bfloat16)
    scale_cache = None
    if variant == "bf16":
        latent = kv_bf16
    else:
        latent = kv_bf16.to(torch.float8_e4m3fn)
        if variant == "fp8_scale":
            scale_cache = torch.rand(bl, cache_n, 1, device=device,
                                     dtype=torch.float32, generator=generator
                                     ) + 0.5

    if family == "ratio4":
        window_idx = torch.arange(WINDOW, device=device).view(1, 1, WINDOW)
        compressed = torch.stack([
            torch.randperm(ctx // 4, device=device, generator=generator)[:TOPK4]
            for _ in range(bl)
        ]).view(bl, 1, TOPK4) + WINDOW
        gather_idx = torch.cat(
            (window_idx.expand(bl, 1, WINDOW), compressed), dim=-1
        ).to(torch.int64).contiguous()
    else:
        gather_idx = (
            torch.arange(k, device=device).view(1, 1, k)
            .expand(bl, 1, k).to(torch.int64).contiguous()
        )
    batch_idx = (
        torch.arange(bl, device=device, dtype=torch.int64)
        .view(bl, 1, 1).expand(bl, 1, k).contiguous()
    )
    valid_mask = (
        torch.ones(bl, 1, k, dtype=torch.bool, device=device) if masked else None
    )
    return dict(query_f32=query_f32, latent=latent, sink=sink,
                batch_idx=batch_idx, gather_idx=gather_idx,
                valid_mask=valid_mask, scale_cache=scale_cache,
                softmax_scale=LATENT_DIM ** -0.5)


def bench_one(inputs):
    run = lambda: full_step(**inputs)
    # eager warmup
    for _ in range(3):
        out = run()
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(ITERS):
            out = run()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(ROUNDS):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(stop) * 1000.0 / ITERS)
    del graph, out
    return {
        "p50_us": statistics.median(times_us),
        "rounds_us": [round(t, 2) for t in times_us],
    }


def kv_bytes(family, bl, ctx, variant):
    if family == "window":
        cache_n = WINDOW
    elif family == "ratio128":
        cache_n = WINDOW + ctx // 128
    else:
        cache_n = WINDOW + ctx // 4
    if variant == "bf16":
        return bl * cache_n * LATENT_DIM * 2
    scale = bl * cache_n * 4 if variant == "fp8_scale" else 0
    return bl * cache_n * LATENT_DIM * 1 + scale


def measure_alloc(family, bl, ctx, variant, device, generator):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    inputs = build_inputs(family, bl, ctx, variant, device, generator)
    keep = [inputs["latent"]]
    if inputs["scale_cache"] is not None:
        keep.append(inputs["scale_cache"])
    cache_bytes = sum(t.numel() * t.element_size() for t in keep)
    del inputs
    torch.cuda.empty_cache()
    return cache_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260720)

    results = []
    for family in ("window", "ratio128", "ratio4"):
        for ctx in (2048, 8192):
            if family == "window" and ctx != 2048:
                continue  # window is ctx-independent
            for bl in (32, 64, 128):
                row = {"family": family, "ctx": ctx, "bl": bl}
                for variant in ("bf16", "fp8_cast", "fp8_scale"):
                    inputs = build_inputs(family, bl, ctx, variant, device,
                                          generator)
                    # numeric sanity vs bf16 dequant chain (not a quality gate)
                    timing = bench_one(inputs)
                    row[variant] = timing
                    row[variant]["cache_bytes"] = kv_bytes(family, bl, ctx,
                                                           variant)
                    del inputs
                    torch.cuda.empty_cache()
                row["fp8_cast_ratio"] = round(
                    row["fp8_cast"]["p50_us"] / row["bf16"]["p50_us"], 4)
                row["fp8_scale_ratio"] = round(
                    row["fp8_scale"]["p50_us"] / row["bf16"]["p50_us"], 4)
                results.append(row)
                print(json.dumps(row), flush=True)

    # allocation confirmation at the reference point
    alloc = {}
    for variant in ("bf16", "fp8_cast", "fp8_scale"):
        alloc[variant] = measure_alloc("ratio4", 64, 8192, variant, device,
                                       generator)
    print("alloc ratio4/bl64/ctx8192:", json.dumps(alloc), flush=True)

    payload = {
        "meta": {
            "host": platform.node(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device),
            "iters": ITERS,
            "rounds": ROUNDS,
            "heads": HEADS,
            "latent_dim": LATENT_DIM,
        },
        "rows": results,
        "alloc_ratio4_bl64_ctx8192": alloc,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
