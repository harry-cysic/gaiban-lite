#!/usr/bin/env python3
"""C2F 23rd vertical, lever B attribution: can NCCL and compute overlap here?

The row-blocked MoE pipeline hides only ~1.9 ms of the ~8.9 ms it should be
able to hide at 2 blocks, and at 8 blocks it is *slower* than the sequential
path.  Two explanations are compatible with that: (a) per-block overhead
(extra launches, a less efficient Marlin at smaller M) eats the gain, or
(b) the collectives simply do not run concurrently with compute on this box.

This probe separates them with no MoE machinery at all.  It measures, at the
real MoE payload:

  1. ``all_gather_into_tensor`` alone,
  2. a compute load alone (a BF16 GEMM sized to saturate the SMs),
  3. both issued together (collective async, compute on the default stream).

If the hardware overlaps, (3) ~ max(1, 2).  If it serializes, (3) ~ (1) + (2).
The achieved overlap fraction is reported directly.

Run (4 ranks, single node):
  torchrun --standalone --nproc_per_node=4 c2f_moe_overlap_probe.py \
    --out-dir out-c2f-overlap-probe
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
import torch.distributed as dist


HIDDEN = 4096
WORLD = 4


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def timed(fn, device, iters: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        dist.barrier()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1e3)
    return {
        "p50_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--local-rows", type=int, default=8192)
    parser.add_argument("--gemm-n", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    torch.set_grad_enabled(False)

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "C2F-moe-overlap-probe",
        "lever": "B",
        "koujing": (
            "all_gather_into_tensor at the real MoE payload vs a BF16 GEMM "
            "load vs both issued together; host walls around "
            "torch.cuda.synchronize with dist.barrier fencing"
        ),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "rank": rank,
        "nccl_p2p_level": os.environ.get("NCCL_P2P_LEVEL", ""),
        "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "local_rows": args.local_rows,
        "errors": [],
    }

    try:
        group = dist.new_group(ranks=list(range(WORLD)), backend="nccl")
        local = torch.zeros(
            args.local_rows, HIDDEN, dtype=torch.bfloat16, device=device
        )
        gathered = torch.zeros(
            args.local_rows * WORLD, HIDDEN, dtype=torch.bfloat16, device=device
        )
        # GEMM load sized so its solo time is close to the collective's, which
        # is the regime the MoE pipeline actually runs in.
        gemm_a = torch.randn(
            args.local_rows * WORLD, HIDDEN, dtype=torch.bfloat16, device=device
        )
        gemm_b = torch.randn(
            HIDDEN, args.gemm_n, dtype=torch.bfloat16, device=device
        )
        gemm_out = torch.empty(
            args.local_rows * WORLD, args.gemm_n,
            dtype=torch.bfloat16, device=device,
        )

        def collective_only():
            dist.all_gather_into_tensor(gathered, local, group=group)

        def compute_only():
            torch.mm(gemm_a, gemm_b, out=gemm_out)

        def both():
            work = dist.all_gather_into_tensor(
                gathered, local, group=group, async_op=True
            )
            torch.mm(gemm_a, gemm_b, out=gemm_out)
            work.wait()

        result["collective_only"] = timed(
            collective_only, device, args.iters, args.warmup
        )
        result["compute_only"] = timed(
            compute_only, device, args.iters, args.warmup
        )
        result["concurrent"] = timed(both, device, args.iters, args.warmup)

        collective = result["collective_only"]["p50_ms"]
        compute = result["compute_only"]["p50_ms"]
        concurrent = result["concurrent"]["p50_ms"]
        serial = collective + compute
        ideal = max(collective, compute)
        result["analysis"] = {
            "serial_sum_ms": serial,
            "perfect_overlap_ms": ideal,
            "measured_ms": concurrent,
            # 1.0 == the shorter op was fully hidden, 0.0 == pure serialization
            "overlap_fraction": (serial - concurrent) / max(serial - ideal, 1e-9),
            "hidden_ms": serial - concurrent,
            "hideable_ms": serial - ideal,
        }
        payload = gathered.numel() * gathered.element_size()
        result["all_gather_bus_gbps"] = (
            (WORLD - 1) / WORLD * payload / (collective / 1e3) / 1e9
        )
        result["ok"] = True
    except Exception:
        result["ok"] = False
        result["errors"].append(traceback.format_exc())

    if rank == 0:
        write_json(args.out_dir / "c2f-moe-overlap-probe.json", result)
        print(json.dumps(
            {
                "collective_ms": result.get("collective_only", {}).get("p50_ms"),
                "compute_ms": result.get("compute_only", {}).get("p50_ms"),
                "concurrent_ms": result.get("concurrent", {}).get("p50_ms"),
                "analysis": result.get("analysis"),
                "ag_bus_gbps": result.get("all_gather_bus_gbps"),
                "ok": result.get("ok"),
            },
            indent=2,
        ))
    dist.barrier()
    dist.destroy_process_group()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
