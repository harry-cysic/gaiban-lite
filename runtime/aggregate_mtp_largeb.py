#!/usr/bin/env python3
"""Aggregate E1MTPLB timed results from rank JSONs (18th vertical).

For each out-e1mtplb-timed-* directory: pull per-segment walls from the
rank-0 record and per-segment accepts from every tail-stage rank (12-15),
compute batch acceptance and effective throughput, and print a table row:

  effective tok/s = (B_total * rounds + accepts_total) / wall
                  = B_total * (1 + alpha_batch) * rounds / wall
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def aggregate(out_dir: Path) -> dict:
    rank0 = json.loads((out_dir / "rank0.json").read_text())
    tail = [
        json.loads((out_dir / f"rank{r}.json").read_text()) for r in (12, 13, 14, 15)
    ]
    mb = rank0["mb_count"]
    bl = rank0["local_batch"]
    b_total = rank0["global_batch"]
    segments = []
    for index, seg0 in enumerate(rank0["round_results"]):
        rounds = seg0["rounds"]
        wall = seg0["wall_s"]
        accepts = sum(r["round_results"][index]["accepts_local"] for r in tail)
        base_tokens = b_total * rounds
        alpha = accepts / base_tokens
        effective = (base_tokens + accepts) / wall
        segments.append(
            {
                "segment": index,
                "rounds": rounds,
                "wall_s": round(wall, 3),
                "round_wall_ms": round(wall * 1e3 / rounds, 2),
                "alpha_batch": round(alpha, 4),
                "effective_tok_s": round(effective, 1),
                "base_tok_s_passA_only": round(base_tokens / wall, 1),
            }
        )
    best = max(segments, key=lambda s: s["effective_tok_s"])
    return {
        "dir": out_dir.name,
        "config": {
            "mb": mb,
            "bl": bl,
            "B_total": b_total,
            "ctx": rank0["start_position"],
            "kv": rank0["kv_dtype"],
            "accepted": rank0["accepted"],
        },
        "segments": segments,
        "best_effective_tok_s": best["effective_tok_s"],
        "best_alpha": best["alpha_batch"],
        "effective_at_alpha_086": round(
            best["effective_tok_s"] / (1 + best["alpha_batch"]) * 1.86, 1
        ),
    }


def main() -> int:
    for arg in sys.argv[1:]:
        record = aggregate(Path(arg))
        print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
