#!/usr/bin/env python3
"""C3F: compare chunked-prefill arms of the D0L long golden gate.

Reads the per-rank JSON (``rank0.json``) rather than ``result.json`` because
only the per-rank file carries ``prefill_chunk`` and the per-prompt
``prefill_evidence`` (chunk lengths, prefill wall clock, activation peak) that
this vertical is measuring.

Usage:
  python3 summarize.py results/long-chunk0.json results/long-chunk1024.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(path: Path) -> list[dict]:
    payload = load(path)
    rows = []
    for mode, summary in (payload.get("mode_summaries") or {}).items():
        if mode == "eager_vs_fused":
            continue
        per_prompt = summary["per_prompt"]
        rows.append(
            {
                "arm": path.stem,
                "mode": mode,
                "chunk": payload.get("prefill_chunk", 0),
                "matched": summary["matched_tokens"],
                "total": summary["total_tokens"],
                "rate": summary["match_rate"],
                "gap_max": (summary["mismatch_top2_gap"] or {}).get("max"),
                "gap_median": (summary["mismatch_top2_gap"] or {}).get("median"),
                "lanes": summary["lane_argmax_agreement"],
                "per_prompt": per_prompt,
                "shapes": payload.get("global_row_shapes"),
            }
        )
    return rows


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    rows = [row for path in paths for row in collect(path)]

    print(f"{'arm':<22}{'mode':<7}{'chunk':>7}{'score':>12}{'rate':>9}"
          f"{'gapmax':>9}  per-prompt")
    for row in rows:
        per = " ".join(str(p["matched"]) for p in row["per_prompt"])
        print(
            f"{row['arm']:<22}{row['mode']:<7}{row['chunk']:>7}"
            f"{row['matched']:>6}/{row['total']:<5}{row['rate']:>9.4f}"
            f"{(row['gap_max'] or 0):>9.4f}  {per}"
        )

    print("\n-- prefill shape / wall clock / activation peak (per prompt) --")
    print("(prompt 0 carries the tilelang JIT cold start; ignore its wall_ms)")
    print(f"{'arm':<22}{'len':>6}{'fwds':>6}  {'chunks':<14}"
          f"{'wall_ms':>10}{'peakGiB':>9}{'matched':>8}")
    for row in rows:
        for prompt in row["per_prompt"]:
            evidence = prompt.get("prefill_evidence") or {}
            chunks = evidence.get("chunk_lengths") or []
            label = (
                f"{len(chunks)}x{chunks[0]}"
                if chunks and len(set(chunks)) == 1
                else str(chunks)
            )
            print(
                f"{row['arm']:<22}{prompt['prompt_len']:>6}"
                f"{evidence.get('prefill_forwards', 1):>6}  {label:<14}"
                f"{evidence.get('wall_ms', 0):>10.1f}"
                f"{evidence.get('peak_allocated_gib', 0):>9.3f}"
                f"{prompt['matched']:>8}"
            )

    base = next((r for r in rows if r["chunk"] == 0), None)
    if base is not None:
        print("\n-- delta vs whole-sequence control (chunk=0) --")
        by_prompt_base = {
            p["prompt_index"]: p["matched"] for p in base["per_prompt"]
        }
        for row in rows:
            if row is base:
                continue
            delta = row["matched"] - base["matched"]
            per_delta = " ".join(
                f"{p['matched'] - by_prompt_base[p['prompt_index']]:+d}"
                for p in row["per_prompt"]
            )
            print(
                f"{row['arm']:<22}chunk={row['chunk']:<6}"
                f"{row['matched']:>4}/{row['total']:<5} vs {base['matched']}"
                f"  delta {delta:+d}   per-prompt {per_delta}"
            )
        have_tokens = all(
            p.get("predicted_tokens") for p in base["per_prompt"]
        )
        if have_tokens:
            print("\n-- position-level agreement with the control --")
            print("(argmax stream compared directly: this is 'does chunked "
                  "prefill produce the same tokens', independent of golden)")
            base_tokens = {
                p["prompt_index"]: p["predicted_tokens"]
                for p in base["per_prompt"]
            }
            for row in rows:
                if row is base or not all(
                    p.get("predicted_tokens") for p in row["per_prompt"]
                ):
                    continue
                same = total = 0
                per_prompt = []
                for prompt in row["per_prompt"]:
                    mine = prompt["predicted_tokens"]
                    theirs = base_tokens[prompt["prompt_index"]]
                    agree = sum(1 for a, b in zip(mine, theirs) if a == b)
                    per_prompt.append(f"{agree}/{len(mine)}")
                    same += agree
                    total += len(mine)
                print(
                    f"{row['arm']:<22}chunk={row['chunk']:<6}"
                    f"{same}/{total} identical ({same / total:.4f})   "
                    + " ".join(per_prompt)
                )

        print("\n-- prefill speedup / peak vs control (per prompt length) --")
        base_by_len: dict[int, dict] = {}
        for prompt in base["per_prompt"]:
            if prompt["prompt_index"] == 0:
                continue  # tilelang JIT cold start
            evidence = prompt.get("prefill_evidence") or {}
            base_by_len[prompt["prompt_len"]] = evidence
        for row in rows:
            if row is base:
                continue
            for prompt in row["per_prompt"]:
                if prompt["prompt_index"] == 0:
                    continue  # tilelang JIT cold start
                evidence = prompt.get("prefill_evidence") or {}
                reference = base_by_len.get(prompt["prompt_len"]) or {}
                whole_ms = reference.get("wall_ms") or 0.0
                chunk_ms = evidence.get("wall_ms") or 0.0
                if whole_ms <= 0 or chunk_ms <= 0:
                    continue
                whole_peak = reference.get("peak_allocated_gib") or 0.0
                chunk_peak = evidence.get("peak_allocated_gib") or 0.0
                print(
                    f"{row['arm']:<22}len={prompt['prompt_len']:<6}"
                    f"whole {whole_ms:>8.1f} ms / {whole_peak:>6.3f} GiB  ->  "
                    f"chunked {chunk_ms:>8.1f} ms / {chunk_peak:>6.3f} GiB   "
                    f"speedup {whole_ms / chunk_ms:>5.2f}x  "
                    f"peak {chunk_peak - whole_peak:+.3f} GiB"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
