#!/usr/bin/env python3
"""Summarize the 26th vertical's segmented-prefill matrix.

Reads the per-round result JSONs written by ``c2f_prefill_stage_bench.py``
(``--prefill-chunk``) and prints the throughput matrix, the round-to-round
spread, the component-wall split, the memory columns and the MoE collective
self-check that proves every point ran on P2P rather than the SHM fallback.

  python3 summarize.py r1/*.json r2/*.json r3/*.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

GIB = 1024 ** 3


def load(paths: list[str]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    points: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not data.get("ok") or not all(data.get("per_rank_ok", [])):
            print(f"!! not ok: {path}", file=sys.stderr)
            continue
        points[(int(data["chunk"]), int(data["prefill_chunk"]))].append(data)
    return points


def spread(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return (max(values) - min(values)) / statistics.fmean(values)


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    points = load(paths)

    baselines = {
        total: statistics.fmean(
            [d["input_tok_s_per_stage_dp4"] for d in runs]
        )
        for (total, seg), runs in points.items()
        if seg == 0
    }

    print("\n=== throughput (input tok/s per stage, DP4 == 16-card projection) ===")
    header = (
        f"{'total':>6} {'segment':>8} {'fwds':>5} {'tok/s mean':>11} "
        f"{'rounds':>28} {'spread':>7} {'vs whole':>9} {'wall p50 s':>11}"
    )
    print(header)
    print("-" * len(header))
    for total, seg in sorted(points, key=lambda k: (-k[0], k[1])):
        runs = points[(total, seg)]
        toks = [d["input_tok_s_per_stage_dp4"] for d in runs]
        walls = [d["stage_pass_wall_p50_s"] for d in runs]
        mean = statistics.fmean(toks)
        ratio = mean / baselines[total] if total in baselines else float("nan")
        rounds = " ".join(f"{v:8.0f}" for v in toks)
        label = "whole" if seg == 0 else str(seg)
        print(
            f"{total:>6} {label:>8} {runs[0]['prefill_forwards']:>5} "
            f"{mean:>11.0f} {rounds:>28} {spread(toks) * 100:>6.2f}% "
            f"{ratio:>8.3f}x {statistics.fmean(walls):>11.4f}"
        )

    print("\n=== component walls (instrumented pass, round 1, seconds) ===")
    keys = ["attention_ratio4", "attention_ratio128", "moe", "hc", "norm",
            "total_instrumented"]
    header = f"{'total':>6} {'segment':>8} " + " ".join(f"{k[:13]:>13}" for k in keys)
    print(header)
    print("-" * len(header))
    for total, seg in sorted(points, key=lambda k: (-k[0], k[1])):
        walls = points[(total, seg)][0]["component_walls_s"]
        label = "whole" if seg == 0 else str(seg)
        row = " ".join(f"{walls.get(k, 0.0):>13.4f}" for k in keys)
        print(f"{total:>6} {label:>8} {row}")

    print("\n=== memory (GiB) + MoE shape buffers ===")
    header = (
        f"{'total':>6} {'segment':>8} {'rows':>7} {'buffers':>8} {'after load':>11} "
        f"{'prefill peak':>13} {'activation':>11} {'proc peak':>10} {'free end':>9}"
    )
    print(header)
    print("-" * len(header))
    for total, seg in sorted(points, key=lambda k: (-k[0], k[1])):
        d = points[(total, seg)][0]
        label = "whole" if seg == 0 else str(seg)
        print(
            f"{total:>6} {label:>8} {d['moe_registered_global_rows'][0]:>7} "
            f"{d['moe_shape_buffer_bytes'] / GIB:>8.3f} "
            f"{d['memory_allocated_bytes_after_load'] / GIB:>11.3f} "
            f"{d['prefill_peak_allocated_bytes'] / GIB:>13.3f} "
            f"{d['prefill_activation_peak_bytes'] / GIB:>11.3f} "
            f"{d['max_memory_allocated_bytes'] / GIB:>10.3f} "
            f"{d['driver_free_bytes_end'] / GIB:>9.3f}"
        )

    print("\n=== MoE collective self-check (must be ~22-24 GB/s; ~4 == SHM) ===")
    header = (
        f"{'total':>6} {'segment':>8} {'rows':>7} {'ag GB/s':>9} {'rs GB/s':>9} "
        f"{'seg rows':>9} {'seg ag':>8} {'seg rs':>8} {'p2p':>5}"
    )
    print(header)
    print("-" * len(header))
    worst = 1e9
    for total, seg in sorted(points, key=lambda k: (-k[0], k[1])):
        for d in points[(total, seg)]:
            sc = d["moe_collective_selfcheck"]
            seg_sc = d.get("moe_collective_selfcheck_segment")
            worst = min(
                worst, sc["all_gather_bus_gbps"], sc["reduce_scatter_bus_gbps"]
            )
            if seg_sc:
                worst = min(
                    worst,
                    seg_sc["all_gather_bus_gbps"],
                    seg_sc["reduce_scatter_bus_gbps"],
                )
        d = points[(total, seg)][0]
        sc = d["moe_collective_selfcheck"]
        seg_sc = d.get("moe_collective_selfcheck_segment")
        label = "whole" if seg == 0 else str(seg)
        seg_cols = (
            f"{seg_sc['rows_global']:>9} {seg_sc['all_gather_bus_gbps']:>8.1f} "
            f"{seg_sc['reduce_scatter_bus_gbps']:>8.1f}"
            if seg_sc else f"{'-':>9} {'-':>8} {'-':>8}"
        )
        print(
            f"{total:>6} {label:>8} {sc['rows_global']:>7} "
            f"{sc['all_gather_bus_gbps']:>9.2f} "
            f"{sc['reduce_scatter_bus_gbps']:>9.2f} {seg_cols} "
            f"{'YES' if d['nccl_p2p_level'] == 'SYS' else 'NO':>5}"
        )
    # The SHM fallback lands near 4 GB/s and is systematic (every probe in a
    # run is slow), so a single sample between 6 and 15 is a probe hiccup, not
    # a transport change.  Report the distribution and only flag the SHM band.
    samples: list[float] = []
    for runs in points.values():
        for d in runs:
            for key in (
                "moe_collective_selfcheck", "moe_collective_selfcheck_segment"
            ):
                sc = d.get(key)
                if sc:
                    samples.append(sc["all_gather_bus_gbps"])
                    samples.append(sc["reduce_scatter_bus_gbps"])
    shm = [v for v in samples if v < 6.0]
    print(
        f"\ncollective bus bandwidth over {len(samples)} probes: "
        f"min {min(samples):.2f} / p50 {statistics.median(samples):.2f} / "
        f"max {max(samples):.2f} GB/s; "
        f"{len(shm)} in the SHM band (<6 GB/s) -> "
        f"{'P2P everywhere' if not shm else 'SHM FALLBACK PRESENT'}"
    )

    print("\n=== superlinear size penalty of the whole-sequence arm ===")
    print("per-token cost of one whole prefill, normalized to the 1024 segment")
    for total in sorted(baselines, reverse=True):
        runs = points[(total, 0)]
        wall = statistics.fmean([d["stage_pass_wall_p50_s"] for d in runs])
        print(f"  whole L={total:>5}: {wall / total * 1e6:>8.3f} us/token/lane "
              f"({statistics.fmean([d['input_tok_s_per_stage_dp4'] for d in runs]):>8.0f} tok/s)")
    for (total, seg) in sorted(points, key=lambda k: (-k[0], k[1])):
        if seg == 0:
            continue
        runs = points[(total, seg)]
        wall = statistics.fmean([d["stage_pass_wall_p50_s"] for d in runs])
        print(f"  L={total:>5} seg={seg:>5}: {wall / total * 1e6:>8.3f} us/token/lane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
