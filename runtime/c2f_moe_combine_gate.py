#!/usr/bin/env python3
"""C2F 22nd vertical: exhaustive numeric gate for the MoE combine rewrite.

``TP4MoE.__call__`` combined the routed and shared partials as::

    buffers.combined.copy_((routed.float() + shared.float()).to(torch.bfloat16))

which materialises three FP32 [global_rows, hidden] temporaries plus a BF16
one -- 1.88 GiB per call at chunk 8192 -- for what is an elementwise add of two
BF16 tensors.  The rewrite is::

    torch.add(routed, shared, out=buffers.combined)

ATen's CUDA elementwise add promotes BF16 to ``opmath_t = float``, adds in
FP32, and rounds once on store, so the two forms should be *bitwise* identical
rather than merely close.  BF16 has only 2**16 values, so this gate does not
sample: it checks **all 2**32 ordered BF16 pairs** for bit-exact equality.

Run (single GPU):
  python c2f_moe_combine_gate.py --out-dir out-moe-combine
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import torch


BF16_VALUES = 1 << 16


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def all_bf16(device: torch.device) -> torch.Tensor:
    """Every BF16 bit pattern, in ascending bit-pattern order."""

    bits = torch.arange(BF16_VALUES, dtype=torch.int64, device=device)
    return bits.to(torch.int16).view(torch.bfloat16)


def exhaustive_pair_check(device: torch.device, rows_per_chunk: int) -> dict[str, Any]:
    values = all_bf16(device)
    mismatches = 0
    finite_mismatches = 0
    checked = 0
    combined = torch.empty(
        rows_per_chunk, BF16_VALUES, dtype=torch.bfloat16, device=device
    )
    for begin in range(0, BF16_VALUES, rows_per_chunk):
        left = values[begin : begin + rows_per_chunk]
        rows = left.numel()
        view = combined[:rows]
        a = left.view(-1, 1).expand(rows, BF16_VALUES).contiguous()
        b = values.view(1, -1).expand(rows, BF16_VALUES).contiguous()
        reference = (a.float() + b.float()).to(torch.bfloat16)
        torch.add(a, b, out=view)
        same_bits = view.view(torch.int16) == reference.view(torch.int16)
        bad = ~same_bits
        mismatches += int(bad.sum())
        finite_inputs = torch.isfinite(a) & torch.isfinite(b)
        finite_mismatches += int((bad & finite_inputs).sum())
        checked += rows * BF16_VALUES
        del a, b, reference
    return {
        "pairs_checked": checked,
        "expected_pairs": BF16_VALUES * BF16_VALUES,
        "bit_mismatches": mismatches,
        "bit_mismatches_finite_inputs": finite_mismatches,
        "bitwise_identical": mismatches == 0,
        "bitwise_identical_on_finite_inputs": finite_mismatches == 0,
    }


def shape_timing(
    device: torch.device, rows: int, hidden: int, iters: int
) -> dict[str, Any]:
    """Cost of both forms at the real chunk-8192 MoE partial shape."""

    routed = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
    shared = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
    combined = torch.empty_like(routed)

    def old_form() -> None:
        combined.copy_((routed.float() + shared.float()).to(torch.bfloat16))

    def new_form() -> None:
        torch.add(routed, shared, out=combined)

    timings = {}
    for name, fn in (("fp32_temporaries", old_form), ("bf16_add_out", new_form)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) / iters
        timings[name] = {
            "ms": elapsed * 1e3,
            "peak_transient_bytes": int(
                torch.cuda.max_memory_allocated(device) - baseline
            ),
        }
    old_form()
    reference = combined.clone()
    new_form()
    timings["bitwise_identical_at_shape"] = bool(
        torch.equal(combined.view(torch.int16), reference.view(torch.int16))
    )
    timings["rows"] = rows
    timings["hidden"] = hidden
    timings["speedup"] = (
        timings["fp32_temporaries"]["ms"] / timings["bf16_add_out"]["ms"]
    )
    del routed, shared, combined, reference
    return timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--shape-rows", type=int, default=32768)
    parser.add_argument("--shape-hidden", type=int, default=4096)
    parser.add_argument("--shape-iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "C2F-moe-combine-gate",
        "form": "torch.add(routed, shared, out=combined) vs "
        "combined.copy_((routed.float() + shared.float()).to(bfloat16))",
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "errors": [],
    }
    try:
        started = time.perf_counter()
        result["exhaustive_bf16_pairs"] = exhaustive_pair_check(
            device, args.rows_per_chunk
        )
        result["exhaustive_seconds"] = time.perf_counter() - started
        result["shape_timing"] = shape_timing(
            device, args.shape_rows, args.shape_hidden, args.shape_iters
        )
        result["accepted"] = bool(
            result["exhaustive_bf16_pairs"]["bitwise_identical"]
            and result["shape_timing"]["bitwise_identical_at_shape"]
        )
        result["ok"] = True
    except Exception:
        result["ok"] = False
        result["accepted"] = False
        result["errors"].append(traceback.format_exc())

    write_json(args.out_dir / "moe-combine-gate.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=2))
    for error in result["errors"]:
        print(error)
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
