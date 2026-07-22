#!/usr/bin/env python3
"""E9F micro-benchmark: does the fused index-score kernel help at DECODE shapes?

E2F 5b names `index_topk_done` (44.5 us/layer, roofline 0 -- pure launch-bound
elementwise) as the biggest remaining tail-fold target after E4F/E5F.  Its core
is the eager chain

    scores = einsum("bshd,btd->bsht", q, kv)      # [b,1,h,t]
    scores = scores.relu_().mul_(w[...,None]).sum(dim=2).float()   # [b,1,t]

which `ops/indexer_fused.fused_index_score` already fuses into one kernel
(sum_h relu(q_h @ kv^T) * w_h), but only the PREFILL path (ratio4_fullpos)
wires it; decode (ratio4_attention) still runs it eager.  Wiring it into decode
is the E4F pattern -- but that kernel was tuned for large-s prefill, so whether
it is faster at decode s=1 is unknown.  This is the go/no-go BEFORE touching the
frozen decode path (E4F micro-benchmarked first): paired-alternating timing
(9.1) of eager vs fused at decode shapes, plus the numeric delta and whether the
top-k selection changes (the non-bitwise risk that would send it to the D0L soft
gate).

Run (single GPU, titan065):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES=0 ~/Workspace/venvs/sglang/bin/python e9f_index_score_micro.py \
    --out-dir out-e9f-micro
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from dsv4_direct.ops.indexer_fused import fused_index_score

# Flash ratio-4 indexer geometry (TARGET 3): index_n_heads 64, index_head_dim
# 128, index_topk 512.  Decode is b=1, s=1; candidate_width is the compressed
# rows -- ~512 at saturation, larger for long context.
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512


def eager_index_score(q, kv, w):
    """The frozen decode chain (ratio4_attention.py, _HALF_ACCUM branch)."""
    scores = torch.einsum("bshd,btd->bsht", q, kv)
    return scores.relu_().mul_(w.unsqueeze(-1)).sum(dim=2).float()


def paired_timing(fn_a, fn_b, *, rounds, iters, device):
    """Paired-alternating CUDA-event timing (TARGET 9.1): both lanes back to
    back per iteration, so clock/thermal drift hits both equally."""
    a_ms, b_ms = [], []
    for _ in range(rounds):
        # warm
        for _ in range(5):
            fn_a(); fn_b()
        torch.cuda.synchronize(device)
        ea0, ea1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        eb0, eb1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ea0.record()
        for _ in range(iters):
            fn_a()
        ea1.record()
        eb0.record()
        for _ in range(iters):
            fn_b()
        eb1.record()
        torch.cuda.synchronize(device)
        a_ms.append(ea0.elapsed_time(ea1) / iters)
        b_ms.append(eb0.elapsed_time(eb1) / iters)
    return a_ms, b_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--candidate-widths", type=str, default="512,1024,2080",
        help="decode candidate_width values to sweep (512 saturated, up to msl/4)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    result: dict[str, Any] = {
        "experiment": "E9F-index-score-decode-micro",
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "geometry": {"n_heads": INDEX_N_HEADS, "head_dim": INDEX_HEAD_DIM, "topk": INDEX_TOPK},
        "rounds": args.rounds, "iters": args.iters,
        "sweeps": [], "errors": [], "accepted": False,
    }
    try:
        for t in (int(x) for x in args.candidate_widths.split(",")):
            # decode-shape inputs (b=1, s=1), bf16 q/kv as the kernel requires
            q = (torch.randn(1, 1, INDEX_N_HEADS, INDEX_HEAD_DIM, device=device) * 0.1).to(torch.bfloat16)
            kv = (torch.randn(1, t, INDEX_HEAD_DIM, device=device) * 0.1).to(torch.bfloat16)
            w = (torch.randn(1, 1, INDEX_N_HEADS, device=device) * 0.1).to(torch.bfloat16)

            s_eager = eager_index_score(q, kv, w)
            s_fused = fused_index_score(q, kv, w)
            delta = (s_fused - s_eager).abs()
            # top-k agreement (the decode topk that feeds sparse attention)
            k = min(INDEX_TOPK, t)
            tk_e = s_eager.topk(k, dim=-1).indices.sort(dim=-1).values
            tk_f = s_fused.topk(k, dim=-1).indices.sort(dim=-1).values
            topk_equal = bool(torch.equal(tk_e, tk_f))

            eager_ms, fused_ms = paired_timing(
                lambda: eager_index_score(q, kv, w),
                lambda: fused_index_score(q, kv, w),
                rounds=args.rounds, iters=args.iters, device=device,
            )
            em, fm = statistics.median(eager_ms), statistics.median(fused_ms)
            sweep = {
                "candidate_width": t,
                "eager_us_p50": em * 1e3,
                "fused_us_p50": fm * 1e3,
                "speedup": em / fm if fm > 0 else None,
                "saved_us_per_call": (em - fm) * 1e3,
                "eager_spread_pct": (max(eager_ms) - min(eager_ms)) / em * 100,
                "fused_spread_pct": (max(fused_ms) - min(fused_ms)) / fm * 100,
                "score_max_abs_delta": float(delta.max().item()),
                "score_mean_abs_delta": float(delta.mean().item()),
                "topk_equal": topk_equal,
            }
            result["sweeps"].append(sweep)
            print(
                f"[E9F] t={t}: eager {sweep['eager_us_p50']:.1f}us fused "
                f"{sweep['fused_us_p50']:.1f}us speedup {sweep['speedup']:.2f}x "
                f"(saved {sweep['saved_us_per_call']:.1f}us) | topk_equal={topk_equal} "
                f"max_delta={sweep['score_max_abs_delta']:.2e}",
                flush=True,
            )
        result["accepted"] = bool(result["sweeps"])
    except Exception as error:  # noqa: BLE001
        result["errors"].append(
            {"type": type(error).__name__, "message": str(error),
             "traceback": traceback.format_exc()}
        )
        traceback.print_exc()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
