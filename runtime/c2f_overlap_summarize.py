#!/usr/bin/env python3
"""Summarize the C2F 23rd vertical throughput rounds (levers A and B)."""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path


BASELINE_TOK_S = 25308.0  # 22nd vertical tilelang arm, 3-round mean
DECODE_D = 8733.0  # single-pool decode ceiling used by 1/T = 1/D + 8/P


def load(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            out.append((path, json.load(handle)))
    return out


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    arms = {
        "leverA (hc fused)": "out-c2f-ovl-leverA-r*/c2f-chunk8192-w4a8-fused.json",
        "leverB (moe overlap b2)": "out-c2f-ovl-leverB-r*/c2f-chunk8192-w4a8-fused.json",
        "leverAB (both)": "out-c2f-ovl-leverAB-r*/c2f-chunk8192-w4a8-fused.json",
    }
    print(f"{'arm':26s} {'round':6s} {'tok/s':>9s} {'moe':>8s} {'hc':>8s} "
          f"{'norm':>8s} {'attn4':>8s} {'attn128':>8s} {'total':>8s} "
          f"{'ag':>6s} {'rs':>6s}")
    summary = {}
    for name, pattern in arms.items():
        rows = load(str(root / pattern))
        values = []
        for index, (path, data) in enumerate(rows, start=1):
            cw = data["component_walls_s"]
            tok = data["input_tok_s_per_stage_dp4"]
            sc = data["moe_collective_selfcheck"]
            values.append((tok, cw))
            print(f"{name:26s} {'r'+str(index):6s} {tok:9.0f} "
                  f"{cw.get('moe', 0):8.4f} {cw.get('hc', 0):8.4f} "
                  f"{cw.get('norm', 0):8.4f} "
                  f"{cw.get('attention_ratio4', 0):8.4f} "
                  f"{cw.get('attention_ratio128', 0):8.4f} "
                  f"{cw.get('total_instrumented', 0):8.4f} "
                  f"{sc['all_gather_bus_gbps']:6.1f} "
                  f"{sc['reduce_scatter_bus_gbps']:6.1f}")
        if not values:
            continue
        toks = [v[0] for v in values]
        mean = statistics.fmean(toks)
        spread = (max(toks) - min(toks)) / mean * 100 if len(toks) > 1 else 0.0
        buckets = {}
        for key in ("moe", "hc", "norm", "attention_ratio4",
                    "attention_ratio128", "total_instrumented"):
            buckets[key] = statistics.fmean(v[1].get(key, 0.0) for v in values)
        pool = 1.0 / (1.0 / DECODE_D + 8.0 / mean)
        summary[name] = (mean, spread, buckets, pool)
        print(f"{name:26s} {'MEAN':6s} {mean:9.0f} "
              f"{buckets['moe']:8.4f} {buckets['hc']:8.4f} "
              f"{buckets['norm']:8.4f} {buckets['attention_ratio4']:8.4f} "
              f"{buckets['attention_ratio128']:8.4f} "
              f"{buckets['total_instrumented']:8.4f}"
              f"   spread {spread:.2f}%")
        print()

    print(f"{'arm':26s} {'mean tok/s':>11s} {'vs 25308':>10s} "
          f"{'hc+norm':>9s} {'single-pool T':>14s}")
    base_pool = 1.0 / (1.0 / DECODE_D + 8.0 / BASELINE_TOK_S)
    print(f"{'baseline (22nd vertical)':26s} {BASELINE_TOK_S:11.0f} "
          f"{'--':>10s} {0.3134 + 0.0322:9.4f} {base_pool:14.0f}")
    for name, (mean, spread, buckets, pool) in summary.items():
        print(f"{name:26s} {mean:11.0f} "
              f"{(mean / BASELINE_TOK_S - 1) * 100:+9.1f}% "
              f"{buckets['hc'] + buckets['norm']:9.4f} {pool:14.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
