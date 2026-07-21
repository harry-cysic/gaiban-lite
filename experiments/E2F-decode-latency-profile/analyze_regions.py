#!/usr/bin/env python3
"""E2F: split the kernel trace into per-layer attention and MoE regions.

The phase marks (GraphPhaseRecorder) cost real device time, so they are not
the right tool for quantitative work.  The trace needs no marks: within one
replay the kernel sequence is fixed, and the MoE collectives delimit it.

  - ``ncclDevKernel_ReduceScatter`` fires exactly once per layer, at the end
    of that layer's MoE -- it is the layer delimiter.
  - the first ``ncclDevKernel_AllGather`` inside a layer segment is the MoE
    input gather -- it splits the segment into attention+HC and MoE.

Both counts are verified against the layer count before anything is reported,
so a change in the MoE collective structure fails loudly instead of silently
mis-segmenting.

Usage:
  ./analyze_regions.py results/out-e2f-nsys-stage0/e2f-stage0_cuda_gpu_trace.csv 11
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from analyze_nsys import category  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <cuda_gpu_trace.csv> <layers_per_stage>")
    path, layers = sys.argv[1], int(sys.argv[2])
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["Name"].strip()]

    delim = [i for i, r in enumerate(rows) if "ReduceScatter" in r["Name"]]
    if len(delim) % layers:
        raise SystemExit(
            f"{len(delim)} ReduceScatter kernels is not a multiple of {layers} "
            "layers -- the MoE collective structure changed, segmentation invalid"
        )
    replays = len(delim) // layers
    # drop the first and last replay: capture/teardown neighbours are not steady state
    window = delim[layers : len(delim) - layers]

    totals: dict[str, dict[str, list[float]]] = {
        "attention+HC": defaultdict(lambda: [0.0, 0.0]),
        "MoE": defaultdict(lambda: [0.0, 0.0]),
    }
    segments = 0
    for start, end in zip(window, window[1:]):
        segment = rows[start + 1 : end + 1]
        gathers = [j for j, r in enumerate(segment) if "AllGather" in r["Name"]]
        if not gathers:
            continue
        segments += 1
        for region, kernels in (
            ("attention+HC", segment[: gathers[0]]),
            ("MoE", segment[gathers[0] :]),
        ):
            for row in kernels:
                bucket = totals[region][category(row["Name"])]
                bucket[0] += float(row["Duration (ns)"])
                bucket[1] += 1

    if not segments:
        raise SystemExit("no layer segment contained an AllGather")

    print(f"{replays} replays, {segments} layer segments (steady-state window)\n")
    summary: dict[str, dict] = {}
    for region in ("attention+HC", "MoE"):
        region_ns = sum(v[0] for v in totals[region].values())
        region_kernels = sum(v[1] for v in totals[region].values())
        print(
            f"--- {region}: {region_ns / segments / 1e3:.1f} us/layer, "
            f"{region_kernels / segments:.0f} kernels/layer"
        )
        entries = sorted(totals[region].items(), key=lambda item: -item[1][0])
        for name, (ns, count) in entries:
            print(
                f"    {name:36s} {ns / segments / 1e3:8.2f} us "
                f"{count / segments:7.1f} k/layer {ns / region_ns * 100:5.1f}%"
            )
        summary[region] = {
            "us_per_layer": region_ns / segments / 1e3,
            "kernels_per_layer": region_kernels / segments,
            "by_category": {
                name: {
                    "us_per_layer": ns / segments / 1e3,
                    "kernels_per_layer": count / segments,
                    "share_of_region": ns / region_ns,
                }
                for name, (ns, count) in entries
            },
        }
        print()

    elementwise = {
        region: sum(
            entry["us_per_layer"]
            for name, entry in summary[region]["by_category"].items()
            if name.startswith("elementwise")
        )
        for region in summary
    }
    total = sum(elementwise.values())
    print(
        f"elementwise tail: {total:.1f} us/layer, "
        f"{elementwise['attention+HC'] / total * 100:.0f}% in attention, "
        f"{elementwise['MoE'] / total * 100:.0f}% in MoE"
    )
    summary["elementwise_us_per_layer"] = elementwise

    out = path.rsplit("_cuda_gpu_trace.csv", 1)[0] + "_region_split.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
