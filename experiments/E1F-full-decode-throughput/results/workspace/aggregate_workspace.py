#!/usr/bin/env python3
"""Aggregate 17th-vertical (workspace slimming) E1IF runs into a frontier table.

Scans out-e1if-timed-* directories in this folder, printing per point:
kv/idx dtype, pool scope, bl, ctx, best-round and per-round aggregate tok/s,
and the memory ladder (after_build / after_warmup / after_settle free GiB,
min across gathered stage representatives).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def gib(value: int | None) -> str:
    return "-" if value is None else f"{value / 2**30:.2f}"


def main() -> None:
    rows = []
    for out_dir in sorted(ROOT.glob("out-e1if-timed-*")):
        result_file = out_dir / "result.json"
        if not result_file.exists():
            rows.append((out_dir.name, None))
            continue
        result = json.loads(result_file.read_text())
        reps = result.get("stage_representatives", [])
        memory_min: dict[str, int] = {}
        for rep in reps:
            for label, snap in rep.get("memory", {}).items():
                free = int(snap["free_bytes"])
                if label not in memory_min or free < memory_min[label]:
                    memory_min[label] = free
        per_round = []
        for rep in reps:
            if rep.get("stage") == 0:
                per_round = [
                    round(r["aggregate_tok_s_wall"], 1)
                    for r in rep.get("round_results", [])
                ]
                scope = rep.get("graph_pool_scope", "lane_family(default)")
                break
        else:
            scope = "?"
        rows.append(
            (
                out_dir.name,
                {
                    "accepted": result.get("accepted"),
                    "kv": f"{result.get('kv_dtype')}/{result.get('indexer_kv_dtype')}",
                    "scope": scope,
                    "bl": result.get("local_batch"),
                    "rounds_tok_s": per_round,
                    "best_tok_s": max(per_round) if per_round else None,
                    "free_after_build": memory_min.get("after_build"),
                    "free_after_warmup": memory_min.get("after_warmup"),
                    "free_after_settle": memory_min.get("after_settle"),
                    "free_at_end": memory_min.get("at_end"),
                },
            )
        )
    header = (
        f"{'run':58s} {'ok':3s} {'kv':9s} {'scope':11s} {'bl':>3s} "
        f"{'best tok/s':>10s} {'build':>6s} {'warm':>6s} {'settle':>6s} {'end':>6s}"
    )
    print(header)
    print("-" * len(header))
    for name, info in rows:
        if info is None:
            print(f"{name:58s} (no result.json)")
            continue
        print(
            f"{name:58s} {str(bool(info['accepted']))[0]:3s} {info['kv']:9s} "
            f"{str(info['scope']):11s} {info['bl']:>3d} "
            f"{info['best_tok_s'] if info['best_tok_s'] is not None else 0:>10.1f} "
            f"{gib(info['free_after_build']):>6s} {gib(info['free_after_warmup']):>6s} "
            f"{gib(info['free_after_settle']):>6s} {gib(info['free_at_end']):>6s}"
        )
        print(f"{'':58s}     rounds: {info['rounds_tok_s']}")


if __name__ == "__main__":
    main()
