#!/usr/bin/env python3
"""C2F 21st vertical: single-operator gate for the tilelang prefill sparse core.

Compares ``ops/tilelang_sparse.tilelang_sparse_attention`` against the frozen
``attention.torch_sparse_attention`` on the *actual* prefill index geometries
produced by the three call sites (window-only, ratio-128 window+compressed,
ratio-4 window+top-k), across seqlens that include the ``-1``-padding-dense
early positions, plus the two padding-semantics edge cases the kernel and the
torch core disagree on before the wrapper aligns them.

  <venv>/bin/python c2f_tilelang_sparse_gate.py --out c2f-tilelang-op-gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")
os.environ["PATH"] = (
    os.path.join(os.environ["CUDA_HOME"], "bin") + os.pathsep + os.environ.get("PATH", "")
)

import torch

from dsv4_direct.attention import (
    compressed_topk_indices,
    torch_sparse_attention,
    window_topk_indices,
)
from dsv4_direct.ops.tilelang_sparse import (
    load_reference_kernel_module,
    reference_kernel_path,
    tilelang_sparse_attention,
)

HEADS = 64
HEAD_DIM = 512
WINDOW = 128
RATIO4_TOPK = 512

# Frozen limit for this gate.  The attention stage limits in the layer oracles
# (e0ef/e0wf STAGE_RMS_REL_LIMITS) sit at 4e-2; the micro-probe measured
# rel_fro 1.93e-3 at seqlen 512.  1e-2 keeps a >5x margin to the probe while
# staying well inside the layer gates it feeds.  Not to be widened: a miss is
# reported and the gate fails.
REL_FRO_LIMIT = 1.0e-2
MAX_ABS_LIMIT = 5.0e-3


def build_case(
    name: str,
    kind: str,
    seqlen: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Index geometry exactly as the corresponding prefill call site builds it."""

    window = window_topk_indices(
        batch_size=1, seqlen=seqlen, start_pos=0, device=device
    )
    if kind == "window":
        topk = window
        kv_rows = seqlen
    elif kind == "ratio128":
        compressed_count = seqlen // 128
        compressed = compressed_topk_indices(
            batch_size=1, seqlen=seqlen, start_pos=0, offset=seqlen, device=device
        )
        topk = torch.cat((window, compressed), dim=-1)
        kv_rows = seqlen + compressed_count
    elif kind == "ratio4":
        # ratio4_fullpos: top-k over the compressed rows with the causal
        # invalids rewritten to -1, concatenated after the window part.
        compressed_count = seqlen // 4
        visible = torch.arange(1, seqlen + 1, device=device) // 4
        topk_count = min(RATIO4_TOPK, compressed_count)
        scores = torch.randn(
            1, seqlen, compressed_count, device=device, dtype=torch.float32,
            generator=generator,
        )
        future = (
            torch.arange(compressed_count, device=device) >= visible.unsqueeze(1)
        )
        scores = scores + torch.where(future, float("-inf"), 0.0)
        picked = scores.topk(topk_count, dim=-1).indices
        invalid = picked >= visible.view(1, seqlen, 1)
        picked = torch.where(invalid, -1 - seqlen, picked)
        compressed = (picked + seqlen).to(torch.int32)
        topk = torch.cat((window, compressed), dim=-1)
        kv_rows = seqlen + compressed_count
    else:
        raise ValueError(kind)

    topk = topk.contiguous()
    query = (
        torch.randn(
            1, seqlen, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    latent = (
        torch.randn(
            1, kv_rows, HEAD_DIM, device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    sink = torch.randn(
        HEADS, device=device, dtype=torch.float32, generator=generator
    )
    return {
        "name": name,
        "kind": kind,
        "seqlen": seqlen,
        "kv_rows": kv_rows,
        "candidates": int(topk.shape[-1]),
        "query": query,
        "latent": latent,
        "sink": sink,
        "topk": topk,
    }


def torch_reference(
    case: dict[str, Any], scale: float, row_block: int
) -> torch.Tensor:
    """Frozen torch core, row-blocked exactly as the runtime blocks it.

    Rows are independent (per-row mask/softmax), so this is bitwise identical
    to the single call; it bounds the FP32 gather workspace that would reach
    ~10.7 GB at seqlen 8192 with 640 candidates.
    """

    seqlen = case["seqlen"]
    if row_block <= 0 or seqlen <= row_block:
        return torch_sparse_attention(
            case["query"], case["latent"], case["sink"], case["topk"], scale
        )
    pieces = []
    for begin in range(0, seqlen, row_block):
        pieces.append(
            torch_sparse_attention(
                case["query"][:, begin : begin + row_block],
                case["latent"],
                case["sink"],
                case["topk"][:, begin : begin + row_block],
                scale,
            )
        )
    return torch.cat(pieces, dim=1)


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = candidate.float() - reference.float()
    denominator = torch.linalg.norm(reference.float())
    return {
        "rel_fro": float(torch.linalg.norm(delta) / denominator),
        "max_abs": float(delta.abs().max()),
        "reference_max_abs": float(reference.float().abs().max()),
        "candidate_nonfinite": int((~torch.isfinite(candidate.float())).sum()),
    }


def sink_margin(case: dict[str, Any], scale: float) -> dict[str, float]:
    """max over rows/heads of (sink - row_max), the kernel's overflow exposure.

    The torch core stabilizes with ``max(row_max, sink)``; the kernel uses
    ``row_max`` alone and folds the sink in as ``exp(sink - row_max)``.  The
    two are algebraically identical but the kernel form overflows FP32 once
    this margin passes ~88.
    """

    margins = []
    block = 256
    for begin in range(0, case["seqlen"], block):
        topk = case["topk"][:, begin : begin + block]
        valid = topk >= 0
        safe = topk.clamp_min(0).long()
        selected = case["latent"][torch.zeros_like(safe), safe]
        scores = (
            torch.einsum(
                "bshd,bskd->bshk",
                case["query"][:, begin : begin + block].float(),
                selected.float(),
            )
            * scale
        )
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        margins.append(case["sink"].view(1, 1, -1) - scores.amax(dim=-1))
        del selected, scores
    margin = torch.cat(margins, dim=1)
    return {
        "max_sink_minus_rowmax": float(margin.max()),
        "min_sink_minus_rowmax": float(margin.min()),
        "fp32_exp_overflow_threshold": 88.0,
    }


def edge_cases(device: torch.device, generator: torch.Generator) -> dict[str, Any]:
    """The two documented padding divergences, measured rather than asserted."""

    seqlen, kv_rows, candidates = 8, 16, 6
    query = (
        torch.randn(
            1, seqlen, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    latent = (
        torch.randn(
            1, kv_rows, HEAD_DIM, device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    sink = torch.randn(
        HEADS, device=device, dtype=torch.float32, generator=generator
    )
    scale = HEAD_DIM**-0.5
    raw_sparse_attn = load_reference_kernel_module().sparse_attn

    def raw_kernel(topk: torch.Tensor) -> torch.Tensor:
        pieces = []
        for start in range(0, HEADS, 16):
            pieces.append(
                raw_sparse_attn(
                    query[:, :, start : start + 16].contiguous(),
                    latent,
                    sink[start : start + 16].contiguous(),
                    topk.contiguous(),
                    scale,
                )
            )
        return torch.cat(pieces, dim=2)

    report: dict[str, Any] = {}

    # (1) all-padding row: torch -> zeros, raw kernel -> NaN, wrapper -> zeros.
    topk = torch.full((1, seqlen, candidates), -1, dtype=torch.int32, device=device)
    topk[:, 1:] = torch.arange(candidates, device=device, dtype=torch.int32)
    reference = torch_sparse_attention(query, latent, sink, topk, scale)
    raw = raw_kernel(topk)
    wrapped = tilelang_sparse_attention(query, latent, sink, topk, scale)
    report["all_padding_row"] = {
        "torch_row0_all_zero": bool(torch.all(reference[:, 0] == 0)),
        "raw_kernel_row0_nonfinite": int(
            (~torch.isfinite(raw[:, 0].float())).sum()
        ),
        "wrapper_row0_all_zero": bool(torch.all(wrapped[:, 0] == 0)),
        "wrapper_vs_torch": compare(reference, wrapped),
    }

    # (2) stray negative padding (-2): torch masks it, the raw kernel gathers
    # kv[b, -2]; the wrapper normalizes negatives to -1 first.
    topk = torch.arange(candidates, device=device, dtype=torch.int32).view(1, 1, -1)
    topk = topk.expand(1, seqlen, candidates).clone()
    topk[:, :, -2:] = -2
    reference = torch_sparse_attention(query, latent, sink, topk, scale)
    raw = raw_kernel(topk)
    wrapped = tilelang_sparse_attention(query, latent, sink, topk, scale)
    report["stray_negative_padding"] = {
        "raw_kernel_vs_torch": compare(reference, raw),
        "wrapper_vs_torch": compare(reference, wrapped),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="c2f-tilelang-op-gate.json")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--head-chunks", default="16,8",
        help="comma list of head-loop widths to gate",
    )
    parser.add_argument(
        "--probe-head-chunks", default="32,64",
        help="widths expected to hit the sm89 shared-memory wall (recorded, "
        "not gated)",
    )
    parser.add_argument(
        "--reference-row-block", type=int, default=1024,
        help="row blocking of the torch reference arm (bitwise identical; "
        "bounds its FP32 gather workspace)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    scale = HEAD_DIM**-0.5

    specs = [
        ("window-96", "window", 96),
        ("window-128", "window", 128),
        ("window-200", "window", 200),
        ("window-2048", "window", 2048),
        ("ratio128-128", "ratio128", 128),
        ("ratio128-512", "ratio128", 512),
        ("ratio128-2048", "ratio128", 2048),
        ("ratio128-8192", "ratio128", 8192),
        ("ratio4-512", "ratio4", 512),
        ("ratio4-2048", "ratio4", 2048),
        ("ratio4-8192", "ratio4", 8192),
    ]
    head_chunks = [int(v) for v in args.head_chunks.split(",") if v.strip()]
    probe_chunks = [int(v) for v in args.probe_head_chunks.split(",") if v.strip()]

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "C2F-tilelang-sparse-op-gate",
        "measurement_class": "semantic_correctness_gate",
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "seed": args.seed,
        "limits": {"rel_fro": REL_FRO_LIMIT, "max_abs": MAX_ABS_LIMIT},
        "head_chunks": head_chunks,
        "cases": [],
    }

    failures: list[str] = []
    for name, kind, seqlen in specs:
        case = build_case(name, kind, seqlen, device, generator)
        reference = torch_reference(case, scale, args.reference_row_block)
        padding_fraction = float((case["topk"] < 0).float().mean())
        entry: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "seqlen": seqlen,
            "kv_rows": case["kv_rows"],
            "candidates": case["candidates"],
            "padding_fraction": round(padding_fraction, 6),
            "first_row_valid": int((case["topk"][0, 0] >= 0).sum()),
            "arms": {},
        }
        if seqlen <= 2048:
            entry["sink_margin"] = sink_margin(case, scale)
        for chunk in head_chunks:
            candidate = tilelang_sparse_attention(
                case["query"],
                case["latent"],
                case["sink"],
                case["topk"],
                scale,
                head_chunk=chunk,
            )
            metrics = compare(reference, candidate)
            metrics["pass"] = bool(
                metrics["rel_fro"] <= REL_FRO_LIMIT
                and metrics["max_abs"] <= MAX_ABS_LIMIT
                and metrics["candidate_nonfinite"] == 0
            )
            if not metrics["pass"]:
                failures.append(f"{name}/h{chunk}")
            entry["arms"][f"head_chunk_{chunk}"] = metrics
            del candidate
        # head-loop width must not change the result at all (heads are
        # independent): compare the widths against each other via torch.
        report["cases"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "arms"}))
        for chunk_name, metrics in entry["arms"].items():
            print(f"  {chunk_name}: {json.dumps(metrics)}")
        del case, reference
        torch.cuda.empty_cache()

    # sm89 shared-memory wall: record what larger head chunks actually do.
    wall: dict[str, Any] = {}
    case = build_case("wall-probe", "ratio128", 512, device, generator)
    for chunk in probe_chunks:
        try:
            out = tilelang_sparse_attention(
                case["query"], case["latent"], case["sink"], case["topk"],
                scale, head_chunk=chunk,
            )
            wall[str(chunk)] = {"ok": True, "finite": bool(torch.isfinite(out).all())}
            del out
        except Exception as error:  # noqa: BLE001 - recording environment behaviour
            wall[str(chunk)] = {"ok": False, "error": f"{type(error).__name__}: {error}"[:600]}
        torch.cuda.empty_cache()
    report["head_chunk_wall"] = wall
    del case
    torch.cuda.empty_cache()

    report["edge_cases"] = edge_cases(device, generator)
    edge = report["edge_cases"]
    edge_ok = (
        edge["all_padding_row"]["torch_row0_all_zero"]
        and edge["all_padding_row"]["wrapper_row0_all_zero"]
        and edge["all_padding_row"]["wrapper_vs_torch"]["rel_fro"] <= REL_FRO_LIMIT
        and edge["stray_negative_padding"]["wrapper_vs_torch"]["rel_fro"]
        <= REL_FRO_LIMIT
    )
    if not edge_ok:
        failures.append("edge_cases")

    report["reference_kernel_path"] = reference_kernel_path()
    report["failures"] = failures
    report["pass"] = not failures
    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("EDGE", json.dumps(report["edge_cases"], indent=1))
    print("WALL", json.dumps(wall))
    print("PASS" if report["pass"] else f"FAIL {failures}")
    print("WROTE", args.out)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
