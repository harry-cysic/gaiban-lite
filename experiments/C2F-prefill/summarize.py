#!/usr/bin/env python3
"""Aggregate C2F prefill bench JSONs into the README table.

Usage: python3 summarize.py <results-dir>
"""

import json
import sys
from pathlib import Path

D_DECODE = 8733.0  # E1F/17th-vertical 8K bf16 frontier, output tok/s


def pooled_T(P: float, D: float = D_DECODE) -> float:
    """8K/1K single-pool aggregate: 1/T = 1/D + 8/P."""

    return 1.0 / (1.0 / D + 8.0 / P)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = []
    for path in sorted(root.glob("c2f-chunk*.json")):
        d = json.loads(path.read_text())
        ok = all(d.get("per_rank_ok", []))
        rows.append(
            (
                int(d["chunk"]),
                d["moe_mode"],
                d["indexer"],
                d.get("stage_pass_wall_p50_s") if ok else None,
                d.get("input_tok_s_per_stage_dp4") if ok else None,
                d,
            )
        )
    moe_order = {"w4a16": 0, "w4a8": 1}
    idx_order = {"ref": 0, "fused": 1}
    rows.sort(
        key=lambda r: (r[0], moe_order.get(r[1], 9), idx_order.get(r[2], 9))
    )
    base = {
        r[0]: r[3] for r in rows if r[1] == "w4a16" and r[2] == "ref" and r[3]
    }
    print(
        f"{'chunk':>6} {'moe':>6} {'indexer':>7} {'wall_p50_s':>10} "
        f"{'tok/s_dp4':>10} {'vs_base':>8} {'T_8K/1K':>8} {'memGB':>6}"
    )
    for chunk, moe, idx, wall, tput, d in rows:
        if wall is None:
            print(f"{chunk:>6} {moe:>6} {idx:>7}     FAILED")
            continue
        speedup = base[chunk] / wall if chunk in base else float("nan")
        mem = max(d["per_rank_max_memory_allocated_bytes"]) / 2**30
        print(
            f"{chunk:>6} {moe:>6} {idx:>7} {wall:>10.3f} {tput:>10,.0f} "
            f"{speedup:>7.3f}x {pooled_T(tput):>8.0f} {mem:>6.1f}"
        )


if __name__ == "__main__":
    main()
