#!/usr/bin/env python3
"""C4F: phase-resolved profile of one real-weight ratio-4 prefill layer.

Twenty-seventh vertical.  C2F's frozen prefill配置 (tilelang sparse core +
whole-segment prefill, 25,307-25,308 input tok/s) attributes 34.0% of the
stage pass (0.3635 s over 5 ratio-4 layers) to ratio-4 attention -- the
largest non-MoE bucket.  Before touching anything, this probe splits that
number into its phases.

Why a single-GPU single-process probe: attention in this runtime is DP-form,
i.e. every TP lane runs *all* 64 heads over its own B=1 sequence with fully
replicated attention weights (``load_replicated_block_weights``).  The
ratio-4 layer therefore has no collective and no TP dependency, so one GPU
reproduces exactly what one lane of the C2F bench does, at a fraction of the
load time.  The MoE/HC buckets are out of scope here by construction.

Timing: ``dsv4_direct.phase_timer.PhaseRecorder`` records CUDA events on the
stream and synchronizes once *after* the pass, so no per-phase device barrier
is inserted (the C2F component-wall instrumentation costs up to +14.7% on
short forwards; this one is measured against an uninstrumented p50 in the
same run -- see ``instrumentation_overhead``).

Modes:
  --mode profile   phase table + uninstrumented layer p50 (default)
  --mode micro     operator micro-benchmarks at the profiled shapes
  --mode ab        numeric A/B of a candidate variant against the frozen path

Variants (``--variant``, comma separated; frozen path = none):
  comp_tf32        allow TF32 for the FP32 compressor projections
  comp_cast_hoist  compute the FP32 compressor cast once instead of four times
  qat_fused        fused Triton Hadamard + FP4 QAT for the indexer query
  --mode kernel    bitwise + speed gate for the fused QAT kernel
  sparse_head32    tilelang head chunk 32 instead of 16

Run (titan064, one GPU):
  ~/Workspace/venvs/sglang/bin/python c4f_ratio4_phase_probe.py \
      --stage-root ~/Workspace/DeepSeek-V4-Flash --chunk 8192 \
      --out-dir out-c4f-profile
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import traceback
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch

from dsv4_direct.attention import (
    apply_rotary_emb,
    precompute_freqs_cis,
)
from dsv4_direct.block_weights import load_replicated_block_weights
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.phase_timer import PhaseRecorder
from dsv4_direct.ratio4_attention import (
    Ratio4AttentionConfig,
    fp4_quant_dequant,
    hadamard_transform,
    prepare_ratio4_attention_weights,
)
from dsv4_direct.ratio4_fullpos import Ratio4FullPositionAttention


RATIO4_STAGE1_LAYERS = (12, 14, 16, 18, 20)
TP_WORLD = 4


# --------------------------------------------------------------------------


def build_layer(
    *,
    stage_root: Path,
    layer_id: int,
    device: torch.device,
    max_seq_len: int,
    variants: frozenset[str],
) -> tuple[Ratio4FullPositionAttention, Any, dict[str, Any]]:
    contract = inspect_stage_checkpoint(stage_root, [layer_id], TP_WORLD)
    if not contract["ok"]:
        raise RuntimeError(f"checkpoint contract failed: {contract['errors'][:3]}")
    config_payload = json.loads(
        (stage_root / "config.json").read_text(encoding="utf-8")
    )
    raw_block = load_replicated_block_weights(
        stage_root=stage_root,
        rank=0,
        world_size=TP_WORLD,
        layer_id=layer_id,
        device=device,
        checkpoint_id=contract["checkpoint_id"],
    )
    config = Ratio4AttentionConfig.from_model_config(
        config_payload, layer_id=layer_id, max_seq_len=max_seq_len
    )
    weights = prepare_ratio4_attention_weights(
        raw_block.attention,
        layer_id=layer_id,
        rank=0,
        world_size=TP_WORLD,
        checkpoint_id=contract["checkpoint_id"],
    )
    attention = Ratio4FullPositionAttention(
        config,
        weights,
        batch_size=1,
        device=device,
        kv_dtype="bf16",
        indexer_dtype="bf16",
        # C2F frozen prefill form (the 25,307-25,308 tok/s arm)
        index_score_mode="fused",
        fuse_min_seqlen=1024,
        sparse_row_block=1024,
        prefill_sparse_backend="tilelang",
    )
    apply_variants(attention, variants)
    provenance = {
        "checkpoint_id": contract["checkpoint_id"],
        "layer_id": layer_id,
        "block_resident_bytes": int(raw_block.resident_bytes),
        "prepared_attention_bytes": int(weights.resident_bytes),
    }
    return attention, config, provenance


def apply_variants(
    attention: Ratio4FullPositionAttention, variants: frozenset[str]
) -> None:
    """Attach the requested candidate levers to one built layer."""

    unknown = variants - KNOWN_VARIANTS
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")

    def require(hook: str) -> None:
        if not hasattr(attention, hook):
            raise NotImplementedError(
                f"runtime has no {hook!r} hook; the lever is not wired yet"
            )

    if "qat_fused" in variants:
        require("indexer_qat_mode")
        attention.indexer_qat_mode = "fused"
    if "comp_cast_hoist" in variants:
        require("compressor_cast_mode")
        attention.compressor_cast_mode = "hoist"
    if "sparse_head32" in variants:
        os.environ["DSV4_PREFILL_SPARSE_HEAD_CHUNK"] = "32"


KNOWN_VARIANTS = frozenset(
    {"comp_tf32", "comp_cast_hoist", "qat_fused", "sparse_head32"}
)


def make_hidden(
    *, seqlen: int, hidden_size: int, device: torch.device, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return (
        torch.randn(
            1,
            seqlen,
            hidden_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.02
    ).to(torch.bfloat16)


def fresh_state(attention: Ratio4FullPositionAttention) -> None:
    attention.raw.zero_()
    attention.compressed.zero_()
    attention.indexer_kv.zero_()
    attention.main_kv_state.zero_()
    attention.main_score_state.fill_(float("-inf"))
    attention.index_kv_state.zero_()
    attention.index_score_state.fill_(float("-inf"))
    attention.next_position = 0
    attention.compressed_count = 0


# --------------------------------------------------------------------------
# mode: profile


def run_profile(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    variants = frozenset(v for v in args.variant.split(",") if v)
    if "comp_tf32" in variants:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    attention, config, provenance = build_layer(
        stage_root=args.stage_root,
        layer_id=args.layer,
        device=device,
        max_seq_len=args.max_seq_len,
        variants=variants,
    )
    hidden = make_hidden(
        seqlen=args.chunk,
        hidden_size=config.hidden_size,
        device=device,
        seed=args.seed,
    )

    def one_pass() -> torch.Tensor:
        fresh_state(attention)
        return attention(hidden, start_pos=0)

    # ---- uninstrumented headline (no phase marks attached) ----
    for _ in range(args.warmup):
        one_pass()
    torch.cuda.synchronize(device)
    plain: list[float] = []
    for _ in range(args.iters):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        one_pass()
        torch.cuda.synchronize(device)
        plain.append(time.perf_counter() - started)

    # ---- instrumented pass ----
    recorder = PhaseRecorder(device)
    attention.phase_recorder = recorder
    for _ in range(max(1, args.warmup)):
        recorder.begin()
        one_pass()
        recorder.end()
    recorder.passes.clear()
    instrumented: list[float] = []
    for _ in range(args.iters):
        recorder.begin()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        one_pass()
        torch.cuda.synchronize(device)
        instrumented.append(time.perf_counter() - started)
        recorder.end()
    attention.phase_recorder = None

    summary = recorder.summary()
    phase_sum = statistics.median(recorder.pass_totals_ms())
    layer_p50_ms = statistics.median(plain) * 1e3
    table = []
    for name, stats in sorted(
        summary.items(), key=lambda item: -item[1]["p50_ms"]
    ):
        table.append(
            {
                "phase": name,
                "p50_ms": stats["p50_ms"],
                "share_of_layer": stats["p50_ms"] / layer_p50_ms,
                "min_ms": stats["min_ms"],
                "max_ms": stats["max_ms"],
            }
        )
    return {
        "mode": "profile",
        "provenance": provenance,
        "variants": sorted(variants),
        "chunk": args.chunk,
        "layer_wall_p50_ms": layer_p50_ms,
        "layer_wall_p50_instrumented_ms": statistics.median(instrumented) * 1e3,
        "instrumentation_overhead": (
            statistics.median(instrumented) / statistics.median(plain) - 1.0
        ),
        "phase_event_sum_p50_ms": phase_sum,
        "phase_coverage": phase_sum / layer_p50_ms,
        "phases": table,
        "five_layer_projection_s": 5 * layer_p50_ms / 1e3,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


# --------------------------------------------------------------------------
# mode: micro -- operator benchmarks at the profiled shapes


def timed_op(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples) * 1e3


def run_micro(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seqlen = args.chunk
    hidden_size = 4096
    results: dict[str, float] = {}
    warmup, iters = args.warmup, args.iters
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    hidden = (
        torch.randn(
            1, seqlen, hidden_size, dtype=torch.float32,
            device=device, generator=generator,
        )
        * 0.02
    ).to(torch.bfloat16)

    # --- compressor projections: fp32 vs tf32 vs bf16 ---
    for name, out_dim in (("main", 1024), ("index", 256)):
        weight32 = torch.randn(
            out_dim, hidden_size, dtype=torch.float32,
            device=device, generator=generator,
        )
        weight16 = weight32.to(torch.bfloat16)
        hidden32 = hidden.float()
        torch.backends.cuda.matmul.allow_tf32 = False
        results[f"comp_{name}_fp32_ms"] = timed_op(
            lambda: torch.nn.functional.linear(hidden.float(), weight32),
            warmup=warmup, iters=iters, device=device,
        )
        results[f"comp_{name}_fp32_precast_ms"] = timed_op(
            lambda: torch.nn.functional.linear(hidden32, weight32),
            warmup=warmup, iters=iters, device=device,
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        results[f"comp_{name}_tf32_ms"] = timed_op(
            lambda: torch.nn.functional.linear(hidden32, weight32),
            warmup=warmup, iters=iters, device=device,
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        results[f"comp_{name}_bf16_ms"] = timed_op(
            lambda: torch.nn.functional.linear(hidden, weight16),
            warmup=warmup, iters=iters, device=device,
        )
    results["hidden_float_cast_ms"] = timed_op(
        lambda: hidden.float(), warmup=warmup, iters=iters, device=device
    )

    # --- indexer query chain ---
    index_query = torch.randn(
        1, seqlen, 64, 128, dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)
    results["hadamard_ms"] = timed_op(
        lambda: hadamard_transform(index_query),
        warmup=warmup, iters=iters, device=device,
    )
    hadamard_out = hadamard_transform(index_query)
    results["fp4_quant_dequant_ms"] = timed_op(
        lambda: fp4_quant_dequant(hadamard_out),
        warmup=warmup, iters=iters, device=device,
    )
    matrix = build_hadamard_matrix(128, device)
    results["hadamard_matmul_ms"] = timed_op(
        lambda: (index_query.float().reshape(-1, 128) @ matrix)
        .reshape(index_query.shape)
        .to(index_query.dtype),
        warmup=warmup, iters=iters, device=device,
    )

    # --- wo_a grouped einsum forms ---
    grouped = torch.randn(
        1, seqlen, 8, 4096, dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)
    wo_a = torch.randn(
        8, 1024, 4096, dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)
    results["wo_a_einsum_ms"] = timed_op(
        lambda: torch.einsum("bsgd,grd->bsgr", grouped, wo_a),
        warmup=warmup, iters=iters, device=device,
    )
    results["wo_a_bmm_ms"] = timed_op(
        lambda: wo_a_bmm(grouped, wo_a),
        warmup=warmup, iters=iters, device=device,
    )
    results["wo_a_blockdiag_ms"] = timed_op(
        lambda: torch.nn.functional.linear(
            grouped.reshape(1, seqlen, 8 * 4096),
            block_diagonal(wo_a),
        ),
        warmup=warmup, iters=iters, device=device,
    )

    # --- topk over the indexer scores ---
    scores = torch.randn(
        1, seqlen, seqlen // 4, dtype=torch.float32, device=device, generator=generator
    )
    results["topk512_ms"] = timed_op(
        lambda: scores.topk(512, dim=-1).indices,
        warmup=warmup, iters=iters, device=device,
    )
    results["topk512_sorted_false_ms"] = timed_op(
        lambda: scores.topk(512, dim=-1, sorted=False).indices,
        warmup=warmup, iters=iters, device=device,
    )
    results["topk512_bf16_ms"] = timed_op(
        lambda: scores.to(torch.bfloat16).topk(512, dim=-1).indices,
        warmup=warmup, iters=iters, device=device,
    )

    # --- rope on the query rope tail and the output tail ---
    freqs = precompute_freqs_cis(
        dim=64,
        seqlen=args.max_seq_len,
        original_seq_len=65536,
        base=10000,
        factor=16,
        beta_fast=32,
        beta_slow=1,
        device=device,
    )[:seqlen]
    query = torch.randn(
        1, seqlen, 64, 512, dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)
    results["rope_tail_ms"] = timed_op(
        lambda: apply_rotary_emb(query[..., -64:], freqs),
        warmup=warmup, iters=iters, device=device,
    )
    tail = query[..., -64:].contiguous()
    results["rope_tail_contiguous_ms"] = timed_op(
        lambda: apply_rotary_emb(tail, freqs),
        warmup=warmup, iters=iters, device=device,
    )

    # --- tilelang sparse core: kernel vs wrapper (head-chunk copies) ---
    try:
        from dsv4_direct.ops.tilelang_sparse import (
            load_reference_kernel_module,
            tilelang_sparse_attention,
        )

        heads, head_dim, window, topk = 64, 512, 128, 512
        kv_rows = seqlen + seqlen // 4
        sparse_q = torch.randn(
            1, seqlen, heads, head_dim, dtype=torch.float32,
            device=device, generator=generator,
        ).to(torch.bfloat16)
        sparse_kv = torch.randn(
            1, kv_rows, head_dim, dtype=torch.float32,
            device=device, generator=generator,
        ).to(torch.bfloat16)
        sink = torch.zeros(heads, dtype=torch.float32, device=device)
        indices = torch.randint(
            0, kv_rows, (1, seqlen, window + topk), dtype=torch.int32,
            device=device, generator=generator,
        )
        scale = head_dim**-0.5
        results["sparse_wrapper_ms"] = timed_op(
            lambda: tilelang_sparse_attention(
                sparse_q, sparse_kv, sink, indices, scale
            ),
            warmup=warmup, iters=iters, device=device,
        )
        kernel = load_reference_kernel_module().sparse_attn
        chunks = [
            sparse_q[:, :, start : start + 16].contiguous()
            for start in range(0, heads, 16)
        ]
        sinks = [
            sink[start : start + 16].contiguous()
            for start in range(0, heads, 16)
        ]

        def kernel_only() -> None:
            for piece, piece_sink in zip(chunks, sinks):
                kernel(piece, sparse_kv, piece_sink, indices, scale)

        results["sparse_kernel_only_ms"] = timed_op(
            kernel_only, warmup=warmup, iters=iters, device=device
        )
        results["sparse_headslice_copy_ms"] = timed_op(
            lambda: [
                sparse_q[:, :, start : start + 16].contiguous()
                for start in range(0, heads, 16)
            ],
            warmup=warmup, iters=iters, device=device,
        )
        results["sparse_flops_g"] = (
            4.0 * seqlen * heads * (window + topk) * head_dim / 1e9
        )
    except Exception as error:  # pragma: no cover - environment dependent
        results["sparse_probe_error"] = repr(error)

    torch.backends.cuda.matmul.allow_tf32 = False
    return {"mode": "micro", "chunk": seqlen, "results": results}


def build_hadamard_matrix(width: int, device: torch.device) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float32, device=device)
    while matrix.shape[0] < width:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * (width**-0.5)


def wo_a_bmm(grouped: torch.Tensor, wo_a: torch.Tensor) -> torch.Tensor:
    batch, seqlen, groups, width = grouped.shape
    left = grouped.permute(2, 0, 1, 3).reshape(groups, batch * seqlen, width)
    out = torch.bmm(left, wo_a.transpose(1, 2))
    return out.reshape(groups, batch, seqlen, -1).permute(1, 2, 0, 3)


def block_diagonal(wo_a: torch.Tensor) -> torch.Tensor:
    groups, rank, width = wo_a.shape
    dense = wo_a.new_zeros(groups * rank, groups * width)
    for index in range(groups):
        dense[
            index * rank : (index + 1) * rank,
            index * width : (index + 1) * width,
        ] = wo_a[index]
    return dense


# --------------------------------------------------------------------------
# mode: kernel -- standalone gate for the fused indexer QAT kernel


def run_kernel_gate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from dsv4_direct.ops.indexer_qat import bitwise_selfcheck, fused_hadamard_fp4

    shapes = (
        (1, args.chunk, 64, 128),
        (1, 1024, 64, 128),
        (1, 4096, 64, 128),
        (1, 97, 3, 128),
        (1, 1, 64, 128),
    )
    selfcheck = bitwise_selfcheck(device=device, shapes=shapes, seed=args.seed)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    sample = torch.randn(
        1, args.chunk, 64, 128, dtype=torch.float32,
        device=device, generator=generator,
    ).to(torch.bfloat16)
    eager_ms = timed_op(
        lambda: fp4_quant_dequant(hadamard_transform(sample)),
        warmup=args.warmup, iters=args.iters, device=device,
    )
    fused_ms = timed_op(
        lambda: fused_hadamard_fp4(sample),
        warmup=args.warmup, iters=args.iters, device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    fp4_quant_dequant(hadamard_transform(sample))
    eager_peak = int(torch.cuda.max_memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    fused_hadamard_fp4(sample)
    fused_peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "mode": "kernel",
        "chunk": args.chunk,
        "selfcheck": selfcheck,
        "eager_ms": eager_ms,
        "fused_ms": fused_ms,
        "speedup": eager_ms / fused_ms,
        "eager_peak_bytes": eager_peak,
        "fused_peak_bytes": fused_peak,
        "moved_bytes": int(2 * sample.numel() * sample.element_size()),
        "fused_effective_gbps": (
            2 * sample.numel() * sample.element_size() / (fused_ms * 1e-3) / 1e9
        ),
    }


# --------------------------------------------------------------------------
# mode: ab -- numeric A/B of a candidate against the frozen path


def run_ab(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    variants = frozenset(v for v in args.variant.split(",") if v)
    if not variants:
        raise ValueError("--mode ab requires --variant")
    baseline, config, provenance = build_layer(
        stage_root=args.stage_root,
        layer_id=args.layer,
        device=device,
        max_seq_len=args.max_seq_len,
        variants=frozenset(),
    )
    hidden = make_hidden(
        seqlen=args.chunk,
        hidden_size=config.hidden_size,
        device=device,
        seed=args.seed,
    )
    fresh_state(baseline)
    reference = baseline(hidden, start_pos=0).float()
    reference_state = {
        "compressed": baseline.compressed.clone(),
        "indexer_kv": baseline.indexer_kv.clone(),
        "raw": baseline.raw.clone(),
    }
    del baseline
    torch.cuda.empty_cache()

    if "comp_tf32" in variants:
        torch.backends.cuda.matmul.allow_tf32 = True
    candidate_layer, _, _ = build_layer(
        stage_root=args.stage_root,
        layer_id=args.layer,
        device=device,
        max_seq_len=args.max_seq_len,
        variants=variants,
    )
    fresh_state(candidate_layer)
    candidate = candidate_layer(hidden, start_pos=0).float()

    difference = candidate - reference
    result = {
        "mode": "ab",
        "provenance": provenance,
        "variants": sorted(variants),
        "chunk": args.chunk,
        "bitwise_equal": bool(torch.equal(candidate, reference)),
        "rel_fro": float(
            torch.linalg.norm(difference) / torch.linalg.norm(reference)
        ),
        "max_abs_diff": float(difference.abs().max()),
        "reference_abs_max": float(reference.abs().max()),
        "finite": bool(torch.isfinite(candidate).all()),
        "state_bitwise": {
            name: bool(torch.equal(getattr(candidate_layer, name), value))
            for name, value in reference_state.items()
        },
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    return result


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("out-c4f"))
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument(
        "--mode", choices=("profile", "micro", "ab", "kernel"), default="profile"
    )
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    if args.layer not in RATIO4_STAGE1_LAYERS and args.mode not in ("micro", "kernel"):
        raise ValueError(
            f"--layer must be a stage-1 ratio-4 layer {RATIO4_STAGE1_LAYERS}"
        )
    args.stage_root = args.stage_root.expanduser().resolve()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False

    envelope: dict[str, Any] = {
        "tag": args.tag,
        "argv": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(device),
            "hostname": platform.node(),
            "NCCL_P2P_LEVEL": os.environ.get("NCCL_P2P_LEVEL"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
    }
    for package in ("triton", "tilelang"):
        try:
            envelope["env"][package] = package_version(package)
        except Exception:
            envelope["env"][package] = None

    try:
        if args.mode == "profile":
            envelope["result"] = run_profile(args, device)
        elif args.mode == "micro":
            envelope["result"] = run_micro(args, device)
        elif args.mode == "kernel":
            envelope["result"] = run_kernel_gate(args, device)
        else:
            envelope["result"] = run_ab(args, device)
        envelope["ok"] = True
    except Exception:
        envelope["ok"] = False
        envelope["error"] = traceback.format_exc()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"c4f-{args.mode}-{args.tag or 'run'}.json"
    path = args.out_dir / name
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    print(json.dumps(envelope, indent=2))
    return 0 if envelope["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
