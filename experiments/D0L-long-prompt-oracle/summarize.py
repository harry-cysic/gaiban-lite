#!/usr/bin/env python3
"""D0L: compare the long-prompt golden gate arms (baseline vs the two prefill
levers) and print the prefill-coverage evidence that makes the comparison
meaningful.

Usage:
  python3 summarize.py results/e2e-long-base.json results/e2e-long-leverA.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2

    rows = []
    for path in paths:
        payload = load(path)
        summaries = payload.get("mode_summaries") or {}
        for mode, summary in summaries.items():
            if mode == "eager_vs_fused":
                continue
            coverage = summary.get("prefill_coverage") or {}
            rows.append(
                {
                    "arm": path.stem,
                    "mode": mode,
                    "accepted": payload.get("accepted"),
                    "matched": summary["matched_tokens"],
                    "total": summary["total_tokens"],
                    "rate": summary["match_rate"],
                    "per_prompt": [
                        (p["prompt_len"], p["matched"], p["compare_steps"])
                        for p in summary["per_prompt"]
                    ],
                    "gap_max": (summary["mismatch_top2_gap"] or {}).get("max"),
                    "gap_median": (summary["mismatch_top2_gap"] or {}).get("median"),
                    "deficit_max": (summary["mismatch_golden_deficit"] or {}).get(
                        "max"
                    ),
                    "deficit_median": (
                        summary["mismatch_golden_deficit"] or {}
                    ).get("median"),
                    "lanes": summary["lane_argmax_agreement"],
                    "coverage": coverage,
                    "fused_scope": payload.get("fused_scope"),
                    "overlap_blocks": payload.get("moe_overlap_blocks"),
                    "peak_gib": (
                        (payload.get("memory_at_end") or {}).get(
                            "peak_allocated_bytes"
                        )
                        or 0
                    )
                    / 2**30,
                }
            )

    print(f"{'arm':<26}{'mode':<7}{'score':>12}{'rate':>9}"
          f"{'gapmax':>9}{'defmax':>9}{'peakGiB':>9}  per-prompt")
    for row in rows:
        per = " ".join(f"{m}/{c}" for _, m, c in row["per_prompt"])
        print(
            f"{row['arm']:<26}{row['mode']:<7}"
            f"{row['matched']:>6}/{row['total']:<5}{row['rate']:>9.4f}"
            f"{(row['gap_max'] or 0):>9.4f}{(row['deficit_max'] or 0):>9.4f}"
            f"{row['peak_gib']:>9.2f}  {per}"
        )

    print("\n-- prefill coverage evidence (per arm/mode) --")
    for row in rows:
        cov = row["coverage"]
        print(
            f"{row['arm']:<26}{row['mode']:<7} lengths={cov.get('prompt_lengths')} "
            f"hc_calls={cov.get('hc_calls')} hc_split={cov.get('hc_split_calls')} "
            f"moe_overlapped={cov.get('moe_overlapped_calls')} "
            f"moe_sequential={cov.get('moe_sequential_calls')} "
            f"scope={row['fused_scope']} blocks={row['overlap_blocks']} "
            f"lanes_agree={row['lanes']} accepted={row['accepted']}"
        )

    base = next((r for r in rows if "base" in r["arm"]), None)
    if base is not None:
        print("\n-- delta vs baseline --")
        for row in rows:
            if row is base:
                continue
            delta = row["matched"] - base["matched"]
            verdict = "PASS (not worse)" if delta >= 0 else f"FAIL ({delta})"
            print(
                f"{row['arm']:<26}{row['mode']:<7}"
                f"{row['matched']:>6}/{row['total']:<5} vs baseline "
                f"{base['matched']}/{base['total']}  delta {delta:+d}  {verdict}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
