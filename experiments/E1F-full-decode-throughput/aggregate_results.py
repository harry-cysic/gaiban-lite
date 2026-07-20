#!/usr/bin/env python3
"""Aggregate E1F per-B results into one sweep table (results/sweep-table.json).

Reads results/out-e1f-bl*/rank{0,4,8,12}.json (stage representatives) and
emits, per B and per round: step-wall p50/p95 (rank0, the closed-loop
canonical), per-stage replay p50, sender handoff walls, embed/head/token
walls, replicated throughput, and the DP-equivalent conversion.  Also prints
a flat text table to stdout.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
STAGE_RANKS = {0: 0, 1: 4, 2: 8, 3: 12}


def load(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _ctx_sensitivity():
    """B=64 pair at seeded ctx 2048 vs 8192 (results/ctx8192-bl64)."""

    path = RESULTS / "ctx8192-bl64"
    if not (path / "rank0.json").exists():
        return None
    record = {"local_batch": 64}
    for label, directory in (("ctx2048", "out-e1f-bl64"), ("ctx8192", "ctx8192-bl64")):
        ranks = {s: load(RESULTS / directory / f"rank{r}.json") for s, r in STAGE_RANKS.items()}
        rounds = ranks[0]["round_results"]
        mid = sorted(rounds, key=lambda r: r["timing_ms"]["step_wall"]["p50_ms"])[
            len(rounds) // 2
        ]
        record[label] = {
            "start_position": ranks[0]["start_position"],
            "max_seq_len": ranks[0]["max_seq_len"],
            "accepted": all(r["accepted"] for r in ranks.values()),
            "step_wall_p50_ms": mid["timing_ms"]["step_wall"]["p50_ms"],
            "step_wall_p95_ms": mid["timing_ms"]["step_wall"]["p95_ms"],
            "stage_replay_p50_ms": {
                s: ranks[s]["round_results"][1]["timing_ms"]["replay"]["p50_ms"]
                for s in STAGE_RANKS
            },
        }
    record["step_wall_delta_pct"] = (
        (record["ctx8192"]["step_wall_p50_ms"] / record["ctx2048"]["step_wall_p50_ms"])
        - 1.0
    ) * 100.0
    return record


def main() -> None:
    rows = []
    for out_dir in sorted(
        RESULTS.glob("out-e1f-bl*"),
        key=lambda p: int(re.search(r"bl(\d+)", p.name).group(1)),
    ):
        b = int(re.search(r"bl(\d+)", out_dir.name).group(1))
        ranks = {}
        missing = False
        for stage, rank in STAGE_RANKS.items():
            path = out_dir / f"rank{rank}.json"
            if not path.exists():
                missing = True
                break
            ranks[stage] = load(path)
        if missing:
            rows.append({"local_batch": b, "status": "incomplete"})
            continue
        r0 = ranks[0]
        entry = {
            "local_batch": b,
            "status": "accepted" if all(r["accepted"] for r in ranks.values()) else "FAILED",
            "check_mode": r0["check_mode"],
            "start_position": r0["start_position"],
            "max_seq_len": r0["max_seq_len"],
            "settle": {
                stage: {
                    "bitwise_steps": ranks[stage]["settle"]["bitwise_steps"],
                    "steps": ranks[stage]["settle"]["steps"],
                    "capture_order": ranks[stage]["settle"]["capture_order"],
                    "final_state_digests_equal": ranks[stage]["settle"].get(
                        "final_state_digests_equal"
                    ),
                    "output_lanes_bitwise": ranks[stage]["settle"].get(
                        "output_lanes_bitwise"
                    ),
                }
                for stage in STAGE_RANKS
            },
            "rounds": [],
            "memory_free_gib_at_end": {
                stage: round(
                    ranks[stage]["memory"]["at_end"]["free_bytes"] / 2**30, 2
                )
                for stage in STAGE_RANKS
                if "at_end" in ranks[stage].get("memory", {})
            },
            "errors": {
                stage: ranks[stage]["errors"][:1]
                for stage in STAGE_RANKS
                if ranks[stage]["errors"]
            },
        }
        for round_index, r0_round in enumerate(r0.get("round_results", [])):
            wall = r0_round["timing_ms"].get("step_wall", {})
            record = {
                "round": round_index,
                "step_wall_p50_ms": wall.get("p50_ms"),
                "step_wall_p95_ms": wall.get("p95_ms"),
                "step_wall_mean_ms": wall.get("mean_ms"),
                "throughput_tok_s_p50": r0_round["throughput_tok_s_p50"],
                "throughput_tok_s_mean": r0_round["throughput_tok_s_mean"],
                "dp_equivalent_tok_s_mean": r0_round["dp_equivalent_tok_s_mean"],
                "embed_p50_ms": r0_round["timing_ms"].get("embed", {}).get("p50_ms"),
                "token_wait_p50_ms": r0_round["timing_ms"]
                .get("token_wait", {})
                .get("p50_ms"),
                "stage_replay_p50_ms": {},
                "stage_send_p50_ms": {},
            }
            for stage in STAGE_RANKS:
                stage_round = ranks[stage]["round_results"][round_index]
                record["stage_replay_p50_ms"][stage] = (
                    stage_round["timing_ms"].get("replay", {}).get("p50_ms")
                )
                record["stage_send_p50_ms"][stage] = (
                    stage_round["timing_ms"].get("send", {}).get("p50_ms")
                )
                if stage == 3:
                    record["head_p50_ms"] = (
                        stage_round["timing_ms"].get("head", {}).get("p50_ms")
                    )
                    record["token_send_p50_ms"] = (
                        stage_round["timing_ms"].get("token_send", {}).get("p50_ms")
                    )
                    record["logits_finite"] = stage_round.get("logits_finite")
            entry["rounds"].append(record)
        if entry["rounds"]:
            walls = [r["step_wall_p50_ms"] for r in entry["rounds"]]
            entry["step_wall_p50_ms_over_rounds"] = {
                "min": min(walls),
                "median": statistics.median(walls),
                "max": max(walls),
            }
        rows.append(entry)

    # capacity-model comparison: the revised ~14k tok/s estimate (root README,
    # C1F + C2g revision) is DP-attention caliber at B_global=512 (bl=128 per
    # GPU, 8K ctx) with 4-deep pipelining: throughput = B_global / t_stage.
    # Replicated bl=B per-rank compute equals DP at B_global=4B, so the
    # measured max stage replay p50 converts directly.
    capacity = []
    for entry in rows:
        if not entry.get("rounds"):
            continue
        mid = sorted(entry["rounds"], key=lambda r: r["step_wall_p50_ms"])[
            len(entry["rounds"]) // 2
        ]
        replays = [v for v in mid["stage_replay_p50_ms"].values() if v]
        if not replays:
            continue
        t_stage_max = max(replays)
        b = entry["local_batch"]
        capacity.append(
            {
                "local_batch_bl": b,
                "dp_global_batch_equivalent": 4 * b,
                "t_stage_max_p50_ms": t_stage_max,
                "dp_pipelined_estimate_tok_s": 4 * b / t_stage_max * 1e3,
                "dp_pipelined_with_2ms_handoff_tok_s": 4 * b / (t_stage_max + 2.0)
                * 1e3,
                "measured_serial_closed_loop_tok_s": mid["throughput_tok_s_p50"],
            }
        )
    table = {
        "experiment": "E1F-full-decode-throughput",
        "caliber_note": (
            "B fully replicated across TP4 (bl per GPU = B = distinct global "
            "batch); throughput = B/step_wall; dp_equivalent = 4x (model-"
            "derived); do not mix with C1F numbers"
        ),
        "capacity_model_comparison": {
            "reference": (
                "revised estimate ~14k tok/s @ B_global=512 DP caliber, 8K "
                "ctx, 4-deep pipelining (root README / feasibility 5.2 + C1F "
                "+ C2g revision); this run is ctx~2K seeded residency, so KV/"
                "attention terms are slightly lighter than the 8K model point"
            ),
            "conversion": (
                "replicated bl=B per-rank compute == DP B_global=4B; DP+"
                "pipelined estimate = 4B / max-stage replay p50"
            ),
            "points": capacity,
        },
        "ctx_sensitivity": _ctx_sensitivity(),
        "b_max_finding": (
            "B=256 (rows 1024) OOMs during eager warmup on stage-0/1 GPUs "
            "(ratio-4 stateful attention einsum, ratio4_attention.py:1116, "
            "320 MiB request with ~0.3 GiB free); B=192 is the replicated "
            "capacity ceiling at ctx-2048 seeded residency with 4 MoE slot "
            "shapes; min free headroom at B=192 after settle: ~1.6 GiB"
        ),
        "rows": rows,
    }
    out = RESULTS / "sweep-table.json"
    out.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}\n")
    header = (
        f"{'B':>4} {'status':>10} {'p50 ms':>8} {'p95 ms':>8} "
        f"{'tok/s':>8} {'dp-eq':>9} {'s0':>7} {'s1':>7} {'s2':>7} {'s3':>7} "
        f"{'send0':>6} {'send1':>6} {'send2':>6} {'head':>6}"
    )
    print(header)
    for entry in rows:
        if entry.get("status") == "incomplete" or not entry.get("rounds"):
            print(f"{entry['local_batch']:>4} {entry.get('status', '?'):>10}")
            continue
        walls = sorted(entry["rounds"], key=lambda r: r["step_wall_p50_ms"])
        mid = walls[len(walls) // 2]
        print(
            f"{entry['local_batch']:>4} {entry['status']:>10} "
            f"{mid['step_wall_p50_ms']:>8.2f} {mid['step_wall_p95_ms']:>8.2f} "
            f"{mid['throughput_tok_s_p50']:>8.1f} "
            f"{mid['dp_equivalent_tok_s_mean']:>9.1f} "
            + " ".join(
                f"{mid['stage_replay_p50_ms'][s]:>7.2f}" for s in STAGE_RANKS
            )
            + " "
            + " ".join(
                (
                    f"{mid['stage_send_p50_ms'][s]:>6.2f}"
                    if mid["stage_send_p50_ms"][s] is not None
                    else f"{'-':>6}"
                )
                for s in (0, 1, 2)
            )
            + f" {mid.get('head_p50_ms', 0) or 0:>6.2f}"
        )


if __name__ == "__main__":
    main()
