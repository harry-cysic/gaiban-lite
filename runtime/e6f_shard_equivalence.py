#!/usr/bin/env python3
"""E6F step 1: is the sharded o-path algebraically the full o-path?

Before any runtime plumbing for attention TP4 sharding (design note:
``docs/design-attention-tp4-sharding.md``, variant A), check the one thing the
whole plan rests on, on **real weights**: that splitting the output path by head
and summing the four partials reproduces the unsharded result, differing only
by summation order.

The full path (``Ratio4TorchAttention``, decode):

    sparse_output [b,1,64,512]
      -> reshape [b,1,o_groups=8,4096]            # 8 heads per group
      -> einsum("bsgd,grd->bsgr", ., wo_a[8,1024,4096]) -> [b,1,8,1024]
      -> flatten -> [b,1,8192]
      -> linear(wo_b[4096,8192]) -> [b,1,4096]

The sharded path: rank r owns heads [16r,16r+16) = o_groups [2r,2r+2), computes
its 2 groups through its slice of ``wo_a``, produces 2048 of the 8192 lora
values, multiplies by its **column slice** of ``wo_b``, and the four partials
are summed.

What this establishes and what it does not:

- **Establishes**: the index algebra is right (group/head alignment, which
  columns of ``wo_b`` pair with which groups) and the numeric gap is only
  reordering, not a different function.
- **Does not establish**: that the model-level quality gate passes.  Per
  TARGET 9.6 a changed summation order can never be bitwise, so this is
  measured as a relative difference and the real verdict is the D0L soft gate
  (score not lower, ``top2_gap`` within envelope).  A tiny gap here is
  necessary, not sufficient.

The FP32 reference is the arbiter: both paths are also evaluated in float64 so
the BF16 paths can be compared against a common exact answer rather than
against each other, which would confound "which one moved".

Run (titan065, one GPU):
  ~/Workspace/venvs/sglang/bin/python e6f_shard_equivalence.py \\
      --stage-root ~/Workspace/DeepSeek-V4-Flash --layer 4 --out-dir out-e6f
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch

from dsv4_direct.block_weights import load_replicated_block_weights
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.ratio4_attention import (
    Ratio4AttentionConfig,
    prepare_ratio4_attention_weights,
)

TP = 4


def full_o_path(
    sparse_output: torch.Tensor, wo_a: torch.Tensor, wo_b: torch.Tensor, cfg: Any
) -> torch.Tensor:
    """Exactly the shipped decode output path."""

    grouped = sparse_output.reshape(
        sparse_output.shape[0],
        1,
        cfg.o_groups,
        cfg.num_heads * cfg.head_dim // cfg.o_groups,
    )
    wo_a3 = wo_a.reshape(
        cfg.o_groups, cfg.o_lora_rank, cfg.num_heads * cfg.head_dim // cfg.o_groups
    )
    projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a3)
    return torch.nn.functional.linear(projected.flatten(2), wo_b)


def sharded_o_path(
    sparse_output: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    cfg: Any,
    *,
    reduce_dtype: torch.dtype = torch.bfloat16,
    gemm_fp32: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """The same, split over TP ranks and summed -- what the runtime would do.

    Two separable roundings are added by sharding, and they need separate names:

    - ``reduce_dtype``: the dtype the four partials are summed in.  Upcasting
      to FP32 before the all-reduce costs twice the collective bytes (16 KB vs
      8 KB per layer, irrelevant at ~10 us of latency-bound NCCL).
    - ``gemm_fp32``: whether the ``wo_b`` product itself is computed in FP32.
      **Upcasting after the fact does not undo this one** -- ``F.linear`` on
      BF16 inputs returns BF16, so each partial is already rounded at full
      output magnitude before anything sees it.  This is the rounding the
      unsharded path never pays, because it accumulates all 8192 products
      inside a single GEMM.  Running it in FP32 bounds the best case but
      doubles the ``wo_b`` bytes read, which is the whole point of sharding,
      so it is a floor to know rather than a shipping option.
    """

    group_width = cfg.num_heads * cfg.head_dim // cfg.o_groups
    groups_per_rank = cfg.o_groups // TP
    heads_per_rank = cfg.num_heads // TP
    lora_per_rank = groups_per_rank * cfg.o_lora_rank
    wo_a3 = wo_a.reshape(cfg.o_groups, cfg.o_lora_rank, group_width)

    partials = []
    for rank in range(TP):
        heads = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
        groups = slice(rank * groups_per_rank, (rank + 1) * groups_per_rank)
        columns = slice(rank * lora_per_rank, (rank + 1) * lora_per_rank)

        local_heads = sparse_output[:, :, heads, :]
        grouped = local_heads.reshape(
            sparse_output.shape[0], 1, groups_per_rank, group_width
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a3[groups])
        lora = projected.flatten(2)
        local_wo_b = wo_b[:, columns]
        if gemm_fp32:
            partial = torch.nn.functional.linear(lora.float(), local_wo_b.float())
        else:
            partial = torch.nn.functional.linear(lora, local_wo_b)
        partials.append(partial.to(reduce_dtype))
    total = partials[0]
    for part in partials[1:]:
        total = total + part
    return total.to(sparse_output.dtype), [
        heads_per_rank,
        groups_per_rank,
        lora_per_rank,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    stage_root = args.stage_root.expanduser().resolve()
    contract = inspect_stage_checkpoint(stage_root, [args.layer], TP)
    if not contract["ok"]:
        raise SystemExit(f"checkpoint contract failed: {contract['errors'][:3]}")
    config_payload = json.loads((stage_root / "config.json").read_text("utf-8"))
    raw = load_replicated_block_weights(
        stage_root=stage_root,
        rank=0,
        world_size=TP,
        layer_id=args.layer,
        device=device,
        checkpoint_id=contract["checkpoint_id"],
    )
    cfg = Ratio4AttentionConfig.from_model_config(
        config_payload, layer_id=args.layer, max_seq_len=4096
    )
    weights = prepare_ratio4_attention_weights(
        raw.attention,
        layer_id=args.layer,
        rank=0,
        world_size=TP,
        checkpoint_id=contract["checkpoint_id"],
    )

    result: dict[str, Any] = {
        "experiment": "E6F-attention-tp-shard",
        "step": "o_path_algebra_equivalence",
        "scope": (
            "index algebra + reordering magnitude only; the model-level verdict "
            "is the D0L soft gate (TARGET 1.3), never this number"
        ),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "layer": args.layer,
        "geometry": {
            "num_heads": cfg.num_heads,
            "head_dim": cfg.head_dim,
            "o_groups": cfg.o_groups,
            "o_lora_rank": cfg.o_lora_rank,
            "tp": TP,
        },
        "trials": [],
        "errors": [],
    }

    try:
        for trial in range(args.trials):
            generator = torch.Generator(device="cpu").manual_seed(args.seed + trial)
            # real-range attention output: post-softmax convex combinations of
            # latent rows, so O(1) magnitudes with occasional larger heads
            sparse_output = (
                (
                    torch.randn(
                        args.batch,
                        1,
                        cfg.num_heads,
                        cfg.head_dim,
                        generator=generator,
                        dtype=torch.float32,
                    )
                    * 0.35
                )
                .to(torch.bfloat16)
                .to(device)
            )
            full = full_o_path(sparse_output, weights.wo_a, weights.wo_b, cfg)
            shard, layout = sharded_o_path(
                sparse_output, weights.wo_a, weights.wo_b, cfg
            )
            shard32, _ = sharded_o_path(
                sparse_output,
                weights.wo_a,
                weights.wo_b,
                cfg,
                reduce_dtype=torch.float32,
            )
            shard_floor, _ = sharded_o_path(
                sparse_output,
                weights.wo_a,
                weights.wo_b,
                cfg,
                reduce_dtype=torch.float32,
                gemm_fp32=True,
            )
            exact = full_o_path(
                sparse_output.double(),
                weights.wo_a.double(),
                weights.wo_b.double(),
                cfg,
            )

            def rel(value: torch.Tensor) -> float:
                delta = (value.double() - exact).abs().max().item()
                return float(delta / exact.abs().max().item())

            result["trials"].append(
                {
                    "trial": trial,
                    "bitwise_equal": bool(torch.equal(full, shard)),
                    "max_abs_diff_full_vs_shard": float(
                        (full.float() - shard.float()).abs().max().item()
                    ),
                    "rel_err_full_vs_fp64": rel(full),
                    "rel_err_shard_bf16reduce_vs_fp64": rel(shard),
                    "rel_err_shard_upcast_reduce_vs_fp64": rel(shard32),
                    "rel_err_shard_fp32gemm_floor_vs_fp64": rel(shard_floor),
                    "output_absmax": float(exact.abs().max().item()),
                }
            )
        result["layout"] = {
            "heads_per_rank": layout[0],
            "groups_per_rank": layout[1],
            "lora_cols_per_rank": layout[2],
        }
        worst_full = max(t["rel_err_full_vs_fp64"] for t in result["trials"])
        worst_bf16 = max(
            t["rel_err_shard_bf16reduce_vs_fp64"] for t in result["trials"]
        )
        worst_fp32 = max(
            t["rel_err_shard_upcast_reduce_vs_fp64"] for t in result["trials"]
        )
        worst_floor = max(
            t["rel_err_shard_fp32gemm_floor_vs_fp64"] for t in result["trials"]
        )
        result["worst_rel_err_shard_fp32gemm_floor"] = worst_floor
        result["fp32gemm_floor_ratio_vs_full"] = worst_floor / worst_full
        result["worst_rel_err_full"] = worst_full
        result["worst_rel_err_shard_bf16reduce"] = worst_bf16
        result["shipping_choice"] = "upcast partials to fp32 before all-reduce"
        result["worst_rel_err_shard_fp32reduce"] = worst_fp32
        result["bf16reduce_ratio_vs_full"] = worst_bf16 / worst_full
        result["fp32reduce_ratio_vs_full"] = worst_fp32 / worst_full
        # What this experiment can decide is the **algebra**: with the BF16
        # roundings removed (fp32 GEMM + fp32 reduce) the sharded path must
        # land on the unsharded path's own error, because then the only
        # remaining difference is association order in exact-ish arithmetic.
        # It cannot decide whether the shipping variant's extra BF16 rounding
        # is acceptable -- that is the D0L soft gate's call, by construction.
        result["algebra_verified"] = bool(
            0.95 <= worst_floor / worst_full <= 1.05
        )
        result["shipping_cost_ratio"] = worst_fp32 / worst_full
        result["accepted"] = result["algebra_verified"]
        print(
            f"[E6F] layer {args.layer} heads/rank {layout[0]} groups/rank {layout[1]} "
            f"| rel err vs fp64: full {worst_full:.3e} "
            f"shard(bf16 reduce) {worst_bf16:.3e} ({worst_bf16/worst_full:.2f}x) "
            f"shard(upcast reduce) {worst_fp32:.3e} ({worst_fp32/worst_full:.2f}x) "
            f"shard(fp32 gemm floor) {worst_floor:.3e} ({worst_floor/worst_full:.2f}x) "
            f"| algebra {'VERIFIED' if result['accepted'] else 'BROKEN'}",
            flush=True,
        )
    except Exception:
        import traceback

        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
        print(f"[E6F] FAILED\n{result['errors'][0]}", flush=True)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "o_path_equivalence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
