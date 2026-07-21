#!/usr/bin/env python3
"""E4F: does C4F's fused indexer QAT kernel survive at the decode shape?

E2F measured the eager ``fp4_quant_dequant(hadamard_transform(index_query))``
chain at ~66 us per ratio-4 layer at B=1 -- 4.3% of the stage replay, an order
of magnitude above the 0.2-0.4% prior in TARGET section 2.  It also showed the
mechanism is different from prefill: at B=1 the chain is ~68 kernels of ~1 us
each sitting on the 4090 minimum kernel duration, so a fusion wins by removing
launches, not by removing bandwidth.  C4F's 90.5x is an 8192-row prefill number
and does not carry over.

This is a **judge-alive-or-dead** microbenchmark, per TARGET section 9.4:
a micro number never converts into a projected in-layer gain.  If the fusion
survives here, the real number has to come from an in-layer A/B.

Caliber (TARGET section 9.1): serial A/B on a 4090 is untrustworthy, so both
arms are captured as CUDA graphs and replayed **back-to-back in alternating
order**, which is also what production does -- the chain runs inside the decode
graph, so device-side kernel time is exactly what matters and host launch cost
is correctly excluded from both arms.

Run (titan065, one GPU):
  ~/Workspace/venvs/sglang/bin/python e4f_indexer_qat_decode_micro.py \
      --out-dir out-e4f-micro
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from dsv4_direct.ops.indexer_qat import bitwise_selfcheck, fused_hadamard_fp4
from dsv4_direct.ratio4_attention import fp4_quant_dequant, hadamard_transform


def eager_chain(value: torch.Tensor) -> torch.Tensor:
    return fp4_quant_dequant(hadamard_transform(value))


def capture(fn: Any, source: torch.Tensor, pool: Any) -> tuple[Any, torch.Tensor]:
    stream = torch.cuda.Stream(device=source.device)
    stream.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(source)
    torch.cuda.current_stream(source.device).wait_stream(stream)
    torch.cuda.synchronize(source.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        output = fn(source)
    return graph, output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    shape = (args.batch, 1, args.heads, args.width)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    source = (
        (torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.05)
        .to(torch.bfloat16)
        .to(device)
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E4F-indexer-qat-decode",
        "measurement_class": "paired_alternating_graph_replay_microbenchmark",
        "judgment_scope": (
            "alive-or-dead only; TARGET 9.4 forbids converting a micro number "
            "into a projected in-layer gain"
        ),
        "caliber": (
            "both arms captured as CUDA graphs and replayed back-to-back in "
            "alternating order within each iteration (TARGET 9.1); the decode "
            "chain runs inside the decode graph in production, so device-side "
            "kernel time is the right quantity"
        ),
        "shape": list(shape),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "errors": [],
    }

    try:
        # ---- numeric gate first: a faster wrong kernel is not a result.
        reference = eager_chain(source)
        candidate = fused_hadamard_fp4(source)
        result["bitwise_equal"] = bool(torch.equal(reference, candidate))
        result["max_abs_diff"] = float(
            (reference.float() - candidate.float()).abs().max().item()
        )
        result["selfcheck"] = bitwise_selfcheck(device=device)
        if not result["bitwise_equal"]:
            raise RuntimeError("fused kernel is not bitwise equal at the decode shape")

        pool = torch.cuda.graph_pool_handle()
        eager_graph, eager_out = capture(eager_chain, source, pool)
        fused_graph, fused_out = capture(fused_hadamard_fp4, source, pool)
        # Null arm: one trivial kernel.  replay() + event pair + synchronize has
        # a floor of several us that both real arms pay, so the *difference*
        # between arms is trustworthy while the *ratio* understates the true
        # kernel-time ratio.  Measuring the floor makes that explicit.
        sink = torch.zeros(1, device=device)
        null_graph, _ = capture(lambda _t: sink.add_(0.0), source, pool)
        torch.cuda.synchronize(device)
        result["graph_outputs_bitwise_equal"] = bool(torch.equal(eager_out, fused_out))

        rounds: list[dict[str, float]] = []
        for round_index in range(args.rounds):
            samples: dict[str, list[float]] = {"eager": [], "fused": [], "null": []}
            for iteration in range(args.iters):
                # rotate which arm goes first, step by step
                arms = [
                    ("eager", eager_graph),
                    ("fused", fused_graph),
                    ("null", null_graph),
                ]
                shift = iteration % len(arms)
                order = arms[shift:] + arms[:shift]
                for label, graph in order:
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    stop.record()
                    torch.cuda.synchronize(device)
                    samples[label].append(start.elapsed_time(stop))
            rounds.append(
                {
                    "round": round_index,
                    "eager_p50_us": 1e3 * statistics.median(samples["eager"]),
                    "fused_p50_us": 1e3 * statistics.median(samples["fused"]),
                    "null_p50_us": 1e3 * statistics.median(samples["null"]),
                }
            )

        eager_p50 = statistics.median(r["eager_p50_us"] for r in rounds)
        fused_p50 = statistics.median(r["fused_p50_us"] for r in rounds)
        null_p50 = statistics.median(r["null_p50_us"] for r in rounds)
        spread = lambda key: (  # noqa: E731
            100.0
            * (max(r[key] for r in rounds) - min(r[key] for r in rounds))
            / statistics.median(r[key] for r in rounds)
        )
        result["rounds"] = rounds
        result["eager_p50_us"] = eager_p50
        result["fused_p50_us"] = fused_p50
        result["speedup"] = eager_p50 / fused_p50 if fused_p50 else None
        result["saved_us_per_call"] = eager_p50 - fused_p50
        result["null_p50_us"] = null_p50
        result["measurement_floor_us"] = null_p50
        result["eager_floor_corrected_us"] = eager_p50 - null_p50
        result["fused_floor_corrected_us"] = fused_p50 - null_p50
        # The fused arm can land inside the null arm's noise; dividing by that
        # yields a meaningless ratio.  Only report a ratio when the fused arm is
        # resolvable above the floor, otherwise report it as a lower bound.
        fused_above_floor = fused_p50 - null_p50
        resolvable = fused_above_floor > 1.0  # us
        result["fused_resolvable_above_floor"] = bool(resolvable)
        result["speedup_floor_corrected"] = (
            (eager_p50 - null_p50) / fused_above_floor if resolvable else None
        )
        result["speedup_floor_corrected_lower_bound"] = (
            None if resolvable else (eager_p50 - null_p50) / 1.0
        )
        result["round_spread_pct"] = {
            "eager": spread("eager_p50_us"),
            "fused": spread("fused_p50_us"),
            "null": spread("null_p50_us"),
        }
        result["verdict"] = (
            "alive" if eager_p50 / max(fused_p50, 1e-9) >= 2.0 else "dead"
        )
        result["accepted"] = bool(
            result["bitwise_equal"]
            and result["graph_outputs_bitwise_equal"]
            and max(result["round_spread_pct"].values()) < 5.0
        )
        print(
            f"[E4F] shape {tuple(shape)}  eager {eager_p50:.2f} us  "
            f"fused {fused_p50:.2f} us  null {null_p50:.2f} us  "
            f"| chain above floor {result['eager_floor_corrected_us']:.2f} us, "
            f"fused {'%.2f us' % result['fused_floor_corrected_us'] if resolvable else 'below floor resolution (<1 us)'}, "
            f"saves {result['saved_us_per_call']:.1f} us/call  "
            f"bitwise={result['bitwise_equal']}  verdict={result['verdict']}",
            flush=True,
        )
    except Exception:
        import traceback

        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
        print(f"[E4F] FAILED\n{result['errors'][0]}", flush=True)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
