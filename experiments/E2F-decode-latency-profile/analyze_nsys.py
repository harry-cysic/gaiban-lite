#!/usr/bin/env python3
"""E2F: turn an nsys kernel summary into the per-category replay budget.

Input is ``<tag>_cuda_gpu_kern_sum.csv`` as produced by
``nsys stats --report cuda_gpu_kern_sum`` on a trace taken with
``--cuda-graph-trace=node`` (without that flag a graph replay is one opaque
range and none of this exists).  The trace covers exactly the replays inside
the probe's ``cudaProfilerStart/Stop`` window, so dividing by the replay count
gives per-replay figures directly.

Categories are assigned from the kernel name.  They are coarse on purpose: the
question this profile answers is "which *kind* of work owns the 8 ms", not
which kernel, and the elementwise tail has no single owner -- 71 distinct
kernel types, none above 0.45 ms/replay.

Usage:
  ./analyze_nsys.py results/out-e2f-nsys-stage0/e2f-stage0_cuda_gpu_kern_sum.csv 24
"""

from __future__ import annotations

import csv
import json
import sys


def category(name: str) -> str:
    lowered = name.lower()
    if "gemvx" in name:
        return "dense projection: cublas GEMV"
    if "cutlass" in name and "gemm" in name:
        return "dense projection: cutlass GEMM"
    if "sgemm" in name:
        return "dense projection: fp32 SGEMM"
    if "marlin" in lowered:
        return "MoE: Marlin fp4"
    if "nccl" in lowered:
        return "collective: NCCL"
    if "tilelang" in name:
        return "fused tilelang (HC / sparse attn)"
    if "direct_copy" in name:
        return "elementwise: copy"
    if "reduce_kernel" in name or "Reduce" in name:
        return "elementwise: reduce"
    if "elementwise" in name or "vectorized" in name:
        return "elementwise: other"
    if any(key in lowered for key in ("sort", "topk", "radix")):
        return "sort / topk"
    if any(key in lowered for key in ("index", "gather", "scatter")):
        return "index / gather / scatter"
    return "other"


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <kern_sum.csv> <replay_count>")
    path, replays = sys.argv[1], int(sys.argv[2])
    rows = list(csv.DictReader(open(path, encoding="utf-8")))

    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        key = category(row["Name"])
        entry = buckets.setdefault(key, {"ns": 0.0, "instances": 0, "types": 0})
        entry["ns"] += float(row["Total Time (ns)"])
        entry["instances"] += int(row["Instances"])
        entry["types"] += 1

    total_ns = sum(entry["ns"] for entry in buckets.values())
    total_instances = sum(entry["instances"] for entry in buckets.values())
    ordered = sorted(buckets.items(), key=lambda item: -item[1]["ns"])

    print(
        f"{'category':36s} {'ms/replay':>10s} {'pct':>7s} "
        f"{'kernels/replay':>15s} {'avg_us':>8s} {'types':>6s}"
    )
    for key, entry in ordered:
        print(
            f"{key:36s} {entry['ns'] / 1e6 / replays:10.3f} "
            f"{entry['ns'] / total_ns * 100:6.2f}% "
            f"{entry['instances'] / replays:15.1f} "
            f"{entry['ns'] / entry['instances'] / 1e3:8.2f} "
            f"{int(entry['types']):6d}"
        )
    print(
        f"{'TOTAL':36s} {total_ns / 1e6 / replays:10.3f} {100.0:6.2f}% "
        f"{total_instances / replays:15.1f} "
        f"{total_ns / total_instances / 1e3:8.2f} {len(rows):6d}"
    )

    summary = {
        "replays": replays,
        "kernel_ms_per_replay": total_ns / 1e6 / replays,
        "kernels_per_replay": total_instances / replays,
        "mean_kernel_us": total_ns / total_instances / 1e3,
        "by_category": {
            key: {
                "ms_per_replay": entry["ns"] / 1e6 / replays,
                "share": entry["ns"] / total_ns,
                "kernels_per_replay": entry["instances"] / replays,
                "mean_us": entry["ns"] / entry["instances"] / 1e3,
                "kernel_types": int(entry["types"]),
            }
            for key, entry in ordered
        },
    }
    out = path.rsplit("_cuda_gpu_kern_sum.csv", 1)[0] + "_category_budget.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
