#!/usr/bin/env python3
"""E6F step 2: is the head slicing exact, so the whole-layer delta reduces to step 1?

Step 1 measured the o-path reduction on real weights.  This closes the gap
between "the o-path" and "the whole layer" without running the full forward,
by establishing that everything else sharding touches is *exact*.

The argument has two halves, and only one of them needs measuring.

**By construction (read off the einsum subscripts, not assumed):** the main
attention contracts are ``bshd,bskd->bshk`` and ``bshk,bskd->bshd``.  ``h``
appears in the output of both and is **never reduced**, so each head's result
depends only on its own query and the shared latent -- slicing heads across
ranks cannot change any head's output.  The one contraction that *does* reduce
over heads is the indexer's ``scores.sum(dim=2)``, and variant A deliberately
leaves the indexer unsharded (every rank computes all 64 index heads), which is
exactly why variant B would need a score all-reduce.

**By measurement (this script):** that the *slicing itself* is bitwise.  Row
slicing a GEMM's weight ought to give the matching slice of its output, but
"ought to" is not a guarantee -- cuBLAS may pick a different kernel or a
different split-K for a 8192-row operand than for a 32768-row one, and split-K
changes the reduction order inside each dot product.  So it is measured.

If both hold, the whole-layer numeric consequence of variant A **is** step 1's
o-path number, with nothing else added.

Run (titan065, one GPU):
  ~/Workspace/venvs/sglang/bin/python e6f_slice_exactness.py \
      --stage-root ~/Workspace/DeepSeek-V4-Flash --layer 4 --out-dir out-e6f-slice
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dsv4_direct.block_weights import load_replicated_block_weights
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.ratio4_attention import (
    Ratio4AttentionConfig,
    prepare_ratio4_attention_weights,
    shard_ratio4_attention_weights,
)

TP = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    root = args.stage_root.expanduser().resolve()
    contract = inspect_stage_checkpoint(root, [args.layer], TP)
    if not contract["ok"]:
        raise SystemExit(f"checkpoint contract failed: {contract['errors'][:3]}")
    payload = json.loads((root / "config.json").read_text("utf-8"))
    raw = load_replicated_block_weights(
        stage_root=root, rank=0, world_size=TP, layer_id=args.layer,
        device=device, checkpoint_id=contract["checkpoint_id"],
    )
    cfg = Ratio4AttentionConfig.from_model_config(
        payload, layer_id=args.layer, max_seq_len=3328
    )
    full = prepare_ratio4_attention_weights(
        raw.attention, layer_id=args.layer, rank=0, world_size=TP,
        checkpoint_id=contract["checkpoint_id"],
    )
    shards = [
        shard_ratio4_attention_weights(full, tp_rank=r, tp_size=TP, config=cfg)
        for r in range(TP)
    ]

    result: dict[str, Any] = {
        "experiment": "E6F-attention-tp-shard",
        "step": "slice_exactness",
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "layer": args.layer,
        "head_independence_by_construction": {
            "main_attention_subscripts": ["bshd,bskd->bshk", "bshk,bskd->bshd"],
            "h_reduced": False,
            "indexer_subscript": "bshd,btd->bsht",
            "indexer_reduces_h_via": "scores.sum(dim=2)",
            "indexer_sharded_in_variant_A": False,
        },
        "checks": [],
        "errors": [],
    }

    try:
        heads = cfg.num_heads // TP
        head_rows = heads * cfg.head_dim
        lora_cols = (cfg.o_groups // TP) * cfg.o_lora_rank

        # 1. weight slices must reconstruct the originals exactly
        result["checks"].append({
            "name": "wq_b concat == full",
            "bitwise": bool(torch.equal(
                torch.cat([s.wq_b for s in shards], dim=0), full.wq_b)),
        })
        result["checks"].append({
            "name": "wo_b concat == full (columns)",
            "bitwise": bool(torch.equal(
                torch.cat([s.wo_b for s in shards], dim=1), full.wo_b)),
        })
        result["checks"].append({
            "name": "attn_sink concat == full",
            "bitwise": bool(torch.equal(
                torch.cat([s.attn_sink for s in shards], dim=0), full.attn_sink)),
        })
        wo_a_full3 = full.wo_a.reshape(cfg.o_groups, cfg.o_lora_rank, cfg.group_width)
        wo_a_cat = torch.cat(
            [s.wo_a.reshape(cfg.o_groups // TP, cfg.o_lora_rank, cfg.group_width)
             for s in shards], dim=0)
        result["checks"].append({
            "name": "wo_a concat == full (groups)",
            "bitwise": bool(torch.equal(wo_a_cat, wo_a_full3)),
        })

        # 2. the q projection under sliced weights must be bitwise equal to the
        #    matching slice of the unsliced projection -- the split-K question
        mismatches = 0
        worst = 0.0
        for trial in range(args.trials):
            gen = torch.Generator(device="cpu").manual_seed(args.seed + trial)
            q_lora = (torch.randn(1, 1, cfg.q_lora_rank, generator=gen,
                                  dtype=torch.float32) * 0.05
                      ).to(torch.bfloat16).to(device)
            full_q = F.linear(q_lora, full.wq_b)
            for r, s in enumerate(shards):
                part = F.linear(q_lora, s.wq_b)
                ref = full_q[..., r * head_rows : (r + 1) * head_rows]
                if not torch.equal(part, ref):
                    mismatches += 1
                    worst = max(worst,
                                float((part.float() - ref.float()).abs().max().item()))
        result["checks"].append({
            "name": "wq_b sliced projection == slice of full projection",
            "bitwise": mismatches == 0,
            "mismatching_slices": mismatches,
            "trials": args.trials * TP,
            "worst_abs_diff": worst,
        })

        result["accepted"] = all(c["bitwise"] for c in result["checks"])
        result["conclusion"] = (
            "whole-layer delta == step 1 o-path delta"
            if result["accepted"]
            else "a slice is not exact; the whole-layer delta is NOT reducible to step 1"
        )
        for c in result["checks"]:
            print(f"[E6F]   {c['name']:52s} bitwise={c['bitwise']}", flush=True)
        print(f"[E6F] {result['conclusion']}", flush=True)
    except Exception:
        import traceback
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
        print(f"[E6F] FAILED\n{result['errors'][0]}", flush=True)

    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "slice_exactness.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
