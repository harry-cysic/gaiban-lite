#!/usr/bin/env python3
"""Aggregate the FP8-KV E1IF frontier runs into one table.

Scans out-e1if-timed-*/rank0.json (stage-0 representative; aggregate
throughput is identical across ranks by construction) and prints
ctx / kv form / bl_mb / best-round aggregate tok/s + replay p50.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    for out_dir in sorted(ROOT.glob("out-e1if-timed-*")):
        rank0 = out_dir / "rank0.json"
        if not rank0.exists():
            continue
        r = json.loads(rank0.read_text())
        rounds = r.get("round_results") or []
        if not rounds:
            rows.append((out_dir.name, r.get("kv_dtype"), r.get("local_batch"),
                         r.get("start_position"), None, None, r.get("accepted"),
                         (r.get("errors") or ["?"])[0].strip().splitlines()[-1]
                         if r.get("errors") else ""))
            continue
        best = max(rounds, key=lambda x: x["aggregate_tok_s_wall"])
        replay = best.get("timing_ms", {}).get("replay", {}).get("p50_ms")
        rows.append((out_dir.name, r.get("kv_dtype"), r.get("local_batch"),
                     r.get("start_position"), best["aggregate_tok_s_wall"],
                     replay, r.get("accepted"), ""))
    header = f"{'run':58s} {'kv':14s} {'bl':>4s} {'ctx':>6s} {'tok/s':>8s} {'replay':>7s} acc"
    print(header)
    for name, kv, bl, ctx, tps, replay, acc, err in rows:
        tps_s = f"{tps:8.0f}" if tps else "     OOM"
        rp_s = f"{replay:7.2f}" if replay else "      -"
        print(f"{name:58s} {str(kv):14s} {bl:>4} {ctx:>6} {tps_s} {rp_s} {acc} {err[:60]}")


if __name__ == "__main__":
    main()
