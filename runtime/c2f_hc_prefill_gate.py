#!/usr/bin/env python3
"""C2F 23rd vertical, lever A: prefill-shape HC boundary fusion micro-gate.

A5F quantified ``mhc_fused_post_pre_tilelang`` (vLLM C2g path) at *decode*
shapes (s=1, B<=512): 2.92x at B=512, post/comb <= ~1e-5, residual/hidden at
one bf16 ULP.  The prefill boundary has the same op signature but a very
different shape: [1, chunk, hc=4, 4096] with chunk in the thousands, and the
prefill chain is eager (no CUDA graph).  The C2F component walls put the HC
bucket at 0.3134 s of a 1.3002 s tilelang-arm stage pass (24.1%, 44 eager HC
ops = 21 fusable boundaries + 2 stage-edge leftovers), so this is the second
largest bucket after MoE.

This script is the判活/判死 micro-gate: for each prefill chunk it runs the
exact boundary the stage chain would run --

    hc_post(branch, residual, post, comb) -> hc_pre(...) -> rms_norm(...)

-- through both backends on identical inputs, and reports wall time plus
rel_fro / max_abs for all four outputs.  Single GPU; no distributed state.

Run:
  <venv>/bin/python c2f_hc_prefill_gate.py --out-dir out-c2f-hcgate
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from dsv4_direct.attention import rms_norm
from dsv4_direct.hc_boundary_backend import (
    EagerHCBoundaryBackend,
    FusedTilelangHCBoundaryBackend,
)
from dsv4_direct.hyper_connections import hc_pre


# DeepSeek-V4-Flash config.json
HIDDEN = 4096
HC_MULT = 4
SINKHORN_ITERS = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def delta(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.float()
    candidate = candidate.float()
    difference = candidate - reference
    reference_norm = torch.linalg.norm(reference)
    return {
        "rel_fro": float(difference.norm() / reference_norm.clamp_min(1e-30)),
        "max_abs": float(difference.abs().max()),
        "reference_max_abs": float(reference.abs().max()),
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "nonfinite": int((~torch.isfinite(candidate)).sum()),
    }


def load_real_hc_weights(
    stage_root: Path, layer_id: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Real checkpoint HC weights for one layer.

    The synthetic first cut of this gate used ``hc_scale = 1`` and a tiny
    ``hc_base``; the real checkpoint has ``hc_scale ~ 0.03-0.20`` with
    ``hc_base`` spread over roughly [-5, +6].  That difference is not cosmetic:
    ``hc_scale`` multiplies the GEMM logits *before* the sigmoid/sinkhorn, so
    it sets how much a GEMM discrepancy can move ``post``/``comb`` at all.
    Numerical claims about this boundary are only meaningful on real weights.
    """

    from safetensors import safe_open

    index = json.loads(
        (stage_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    wanted = {
        "attn_fn": f"layers.{layer_id}.hc_attn_fn",
        "attn_scale": f"layers.{layer_id}.hc_attn_scale",
        "attn_base": f"layers.{layer_id}.hc_attn_base",
        "ffn_fn": f"layers.{layer_id}.hc_ffn_fn",
        "ffn_scale": f"layers.{layer_id}.hc_ffn_scale",
        "ffn_base": f"layers.{layer_id}.hc_ffn_base",
        "attn_norm": f"layers.{layer_id}.attn_norm.weight",
        "ffn_norm": f"layers.{layer_id}.ffn_norm.weight",
    }
    out: dict[str, torch.Tensor] = {}
    for name, key in wanted.items():
        with safe_open(stage_root / index[key], framework="pt") as handle:
            out[name] = handle.get_tensor(key).to(device).contiguous()
    return out


def make_inputs(
    rows: int,
    device: torch.device,
    seed: int,
    scale: float,
    weights: dict[str, torch.Tensor] | None,
    *,
    decode_shape: bool = False,
):
    """Build one boundary's inputs.

    ``decode_shape`` reproduces the A5F layout ([B, 1, hc, d], B rows) instead
    of the prefill layout ([1, chunk, hc, d]) so the same harness and the same
    weights can be run at both shapes -- the control that separates "the
    prefill shape breaks the kernel" from "this harness differs from A5F".
    """

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = (rows, 1, HC_MULT, HIDDEN) if decode_shape else (1, rows, HC_MULT, HIDDEN)
    branch_shape = (rows, 1, HIDDEN) if decode_shape else (1, rows, HIDDEN)
    residual = (
        torch.randn(
            *shape, dtype=torch.bfloat16, device=device, generator=generator
        )
        * scale
    ).contiguous()
    branch = (
        torch.randn(
            *branch_shape, dtype=torch.bfloat16, device=device, generator=generator
        )
        * scale
    ).contiguous()
    mix_features = (2 + HC_MULT) * HC_MULT
    if weights is None:
        # block_weights.py pins hc_*_fn to F32 in the checkpoint contract, and
        # the vLLM kernel asserts fn.dtype == float32.
        hc_fn = (
            torch.randn(
                mix_features, HC_MULT * HIDDEN,
                dtype=torch.float32, device=device, generator=generator,
            )
            * 0.02
        ).contiguous()
        hc_scale = torch.ones(3, dtype=torch.float32, device=device)
        hc_base = (
            torch.randn(
                mix_features, dtype=torch.float32, device=device,
                generator=generator,
            )
            * 0.02
        ).contiguous()
        norm_weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)
        prev_fn, prev_scale, prev_base = hc_fn, hc_scale, hc_base
    else:
        # The intra-layer boundary: the *previous* half-layer's post/comb come
        # from the attention-side hc_pre, and the boundary's own hc_pre uses
        # the FFN-side parameters -- exactly the chain's pairing.
        prev_fn = weights["attn_fn"]
        prev_scale = weights["attn_scale"]
        prev_base = weights["attn_base"]
        hc_fn = weights["ffn_fn"]
        hc_scale = weights["ffn_scale"]
        hc_base = weights["ffn_base"]
        norm_weight = weights["ffn_norm"]
    # The boundary consumes the previous half-layer's post/comb, which come
    # from an hc_pre on the same residual -- build them the same way the chain
    # does rather than inventing values.
    _, post, comb = hc_pre(
        residual, prev_fn, prev_scale, prev_base,
        norm_eps=NORM_EPS, sinkhorn_iters=SINKHORN_ITERS, hc_eps=HC_EPS,
    )
    return residual, branch, hc_fn, hc_scale, hc_base, norm_weight, post, comb


def timed(fn, device: torch.device, iters: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1e3)
    return {
        "p50_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def peak_delta_bytes(fn, device: torch.device) -> int:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    fn()
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device) - base)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunks", default="1024,4096,8192")
    parser.add_argument(
        "--stage-root", type=Path, default=None,
        help="checkpoint root; when given, the boundary runs on that layer's "
        "real HC weights instead of synthetic ones",
    )
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument(
        "--decode-control", default="512",
        help="comma-separated row counts to also run at the A5F decode layout "
        "[B, 1, hc, d]; the control that separates a shape problem from a "
        "harness problem (empty to skip)",
    )
    parser.add_argument(
        "--synthetic-control", action="store_true",
        help="additionally run the synthetic hc_scale=1 weights, which make "
        "post/comb maximally sensitive to a GEMM discrepancy",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--input-scale", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "C2F-prefill-hc-boundary-gate",
        "lever": "A",
        "koujing": (
            "one HC boundary (hc_post -> hc_pre -> rms_norm) at prefill shapes "
            "[1, chunk, 4, 4096]; eager vs mhc_fused_post_pre_tilelang on "
            "identical inputs; host walls around torch.cuda.synchronize; "
            "single GPU, no CUDA graph (the prefill chain has none)"
        ),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hidden": HIDDEN,
        "hc_mult": HC_MULT,
        "sinkhorn_iters": SINKHORN_ITERS,
        "iters": args.iters,
        "warmup": args.warmup,
        "seed": args.seed,
        "errors": [],
        "per_chunk": [],
    }

    try:
        result["vllm"] = __import__(
            "importlib.metadata", fromlist=["version"]
        ).version("vllm")
    except Exception:
        result["vllm"] = "unknown"

    eager = EagerHCBoundaryBackend()
    try:
        fused = FusedTilelangHCBoundaryBackend()
        # The unsplit backend is kept as a diagnostic arm so the >= 1024-row
        # `hc_prenorm_gemm_block_m_tilelang` divergence stays on the record
        # next to the shipped (row-blocked) numbers.
        fused_nosplit = FusedTilelangHCBoundaryBackend(max_rows=None)
        result["fused_backend_import"] = "ok"
        result["fused_max_rows"] = fused.max_rows
    except Exception:
        fused = None
        fused_nosplit = None
        result["fused_backend_import"] = traceback.format_exc()

    weights = None
    if args.stage_root is not None:
        weights = load_real_hc_weights(
            args.stage_root.expanduser().resolve(), args.layer, device
        )
        result["weights"] = "real"
        result["layer"] = args.layer
        result["hc_scale_ffn"] = [float(v) for v in weights["ffn_scale"]]
        result["hc_scale_attn"] = [float(v) for v in weights["attn_scale"]]
    else:
        result["weights"] = "synthetic"

    cases: list[tuple[str, int, bool, dict[str, torch.Tensor] | None]] = []
    for chunk in (int(v) for v in args.chunks.split(",") if v.strip()):
        cases.append(("prefill", chunk, False, weights))
    for rows in (int(v) for v in args.decode_control.split(",") if v.strip()):
        cases.append(("decode_control", rows, True, weights))
    if args.synthetic_control:
        for chunk in (int(v) for v in args.chunks.split(",") if v.strip()):
            cases.append(("prefill_synthetic_scale1", chunk, False, None))

    for layout, rows, decode_shape, case_weights in cases:
        chunk = rows
        entry: dict[str, Any] = {
            "layout": layout, "rows": rows, "chunk": chunk,
            "weights": "synthetic" if case_weights is None else "real",
        }
        try:
            (
                residual, branch, hc_fn, hc_scale, hc_base,
                norm_weight, post, comb,
            ) = make_inputs(
                rows, device, args.seed + rows, args.input_scale, case_weights,
                decode_shape=decode_shape,
            )
            kwargs = dict(
                hc_fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                norm_weight=norm_weight,
                norm_eps=NORM_EPS,
                sinkhorn_iters=SINKHORN_ITERS,
                hc_eps=HC_EPS,
            )

            def run_eager():
                return eager.post_pre_norm(branch, residual, post, comb, **kwargs)

            reference = run_eager()
            entry["eager"] = timed(run_eager, device, args.iters, args.warmup)
            entry["eager"]["peak_delta_bytes"] = peak_delta_bytes(run_eager, device)

            if fused is None:
                entry["fused"] = {"error": "backend import failed"}
            else:
                def run_fused():
                    return fused.post_pre_norm(
                        branch, residual, post, comb, **kwargs
                    )

                candidate = run_fused()
                names = ("residual", "hidden", "post", "comb")
                entry["numerics"] = {
                    name: delta(reference[index], candidate[index])
                    for index, name in enumerate(names)
                }
                entry["shapes"] = {
                    name: list(candidate[index].shape)
                    for index, name in enumerate(names)
                }
                entry["fused"] = timed(run_fused, device, args.iters, args.warmup)
                entry["fused"]["peak_delta_bytes"] = peak_delta_bytes(
                    run_fused, device
                )
                entry["speedup"] = (
                    entry["eager"]["p50_ms"] / entry["fused"]["p50_ms"]
                )
                del candidate

                def run_nosplit():
                    return fused_nosplit.post_pre_norm(
                        branch, residual, post, comb, **kwargs
                    )

                raw = run_nosplit()
                entry["numerics_nosplit"] = {
                    name: delta(reference[index], raw[index])
                    for index, name in enumerate(names)
                }
                entry["fused_nosplit"] = timed(
                    run_nosplit, device, args.iters, args.warmup
                )
                del raw
            del reference, residual, branch, hc_fn, hc_base, post, comb
            torch.cuda.empty_cache()
        except Exception:
            entry["error"] = traceback.format_exc()
            torch.cuda.empty_cache()
        result["per_chunk"].append(entry)
        print(json.dumps(
            {
                "layout": layout,
                "weights": entry["weights"],
                "rows": rows,
                "eager_p50_ms": entry.get("eager", {}).get("p50_ms"),
                "fused_p50_ms": entry.get("fused", {}).get("p50_ms"),
                "speedup": entry.get("speedup"),
                "worst_rel_fro": (
                    max(v["rel_fro"] for v in entry["numerics"].values())
                    if "numerics" in entry else None
                ),
                "worst_rel_fro_nosplit": (
                    max(v["rel_fro"] for v in entry["numerics_nosplit"].values())
                    if "numerics_nosplit" in entry else None
                ),
                "fused_nosplit_p50_ms": entry.get("fused_nosplit", {}).get("p50_ms"),
                "error": entry.get("error", "").splitlines()[-1:] or None,
            },
            indent=2,
        ), flush=True)

    result["ok"] = all("error" not in entry for entry in result["per_chunk"])
    write_json(args.out_dir / "c2f-hc-prefill-gate.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
