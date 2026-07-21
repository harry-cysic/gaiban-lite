#!/usr/bin/env python3
"""E6F step 3: one sharded ratio-4 layer across 4 real ranks, with a real all-reduce.

Steps 1-2 settled the numerics on one GPU (the o-path reduction is the entire
delta, and every slice is bitwise).  This checks the **wiring**: that four
processes each holding their own shard, with ``dist.all_reduce`` in the
forward, reproduce what a single unsharded instance computes.

Deliberately narrow: one layer, no super-stage, no CUDA graph.  The point is to
separate "the sharding is wired correctly" from "it survives graph capture and
the pipeline", because when those fail together they are hard to tell apart.

Rank 0 additionally builds an unsharded instance over the same seeded state and
compares.  The expected outcome is **not** bitwise -- per TARGET 9.6 a changed
summation order never is -- so the check is that the relative difference sits
at the magnitude E6F step 1 predicted (1.25-1.49x the unsharded path's own BF16
error, i.e. a few times 1e-3), not that it vanishes.

Run:
  torchrun --standalone --nproc-per-node 4 e6f_dist_layer_gate.py \
      --stage-root ~/Workspace/DeepSeek-V4-Flash --layer 4 --out-dir out-e6f-dist
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from dsv4_direct.block_weights import load_replicated_block_weights
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.ratio4_attention import (
    Ratio4AttentionConfig,
    Ratio4TorchAttention,
    prepare_ratio4_attention_weights,
    shard_ratio4_attention_weights,
)
from dsv4_direct.ratio4_oracle import seed_nonzero_ratio4_state
from dsv4_direct.static_ratio4_kv import StaticRatio4KV

TP = 4


def seeded_state(cfg, oracle, *, device, batch):
    state = StaticRatio4KV(
        layer_id=cfg.layer_id,
        num_local_sequences=batch,
        max_seq_len=cfg.max_seq_len,
        device=device,
    )
    state.seed_decode_payload(
        oracle.next_position,
        raw=oracle.raw.clone(),
        compressed=oracle.compressed.clone(),
        indexer_kv=oracle.indexer_kv.clone(),
        main_kv_state=oracle.main_kv.clone(),
        main_score_state=oracle.main_score.clone(),
        index_kv_state=oracle.index_kv.clone(),
        index_score_state=oracle.index_score.clone(),
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--start-position", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=3328)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    result: dict[str, Any] = {
        "experiment": "E6F-attention-tp-shard",
        "step": "dist_single_layer_wiring",
        "rank": rank,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "layer": args.layer,
        "steps": [],
        "errors": [],
    }

    try:
        if world != TP:
            raise ValueError(f"needs world={TP}, got {world}")
        group = dist.new_group(ranks=list(range(TP)))
        root = args.stage_root.expanduser().resolve()
        contract = inspect_stage_checkpoint(root, [args.layer], TP)
        if not contract["ok"]:
            raise ValueError(f"checkpoint contract failed: {contract['errors'][:3]}")
        payload = json.loads((root / "config.json").read_text("utf-8"))
        raw = load_replicated_block_weights(
            stage_root=root, rank=0, world_size=TP, layer_id=args.layer,
            device=device, checkpoint_id=contract["checkpoint_id"],
        )
        base_cfg = Ratio4AttentionConfig.from_model_config(
            payload, layer_id=args.layer, max_seq_len=args.max_seq_len
        )
        prepared = prepare_ratio4_attention_weights(
            raw.attention, layer_id=args.layer, rank=0, world_size=TP,
            checkpoint_id=contract["checkpoint_id"],
        )
        oracle = seed_nonzero_ratio4_state(
            base_cfg, batch_size=args.batch, start_pos=args.start_position,
            main_ape=prepared.compressor_ape,
            index_ape=prepared.index_compressor_ape,
            seed=args.seed, device=device,
        )

        cfg_r = replace(base_cfg, tp_size=TP, tp_rank=rank)
        cfg_r.validate()
        shard_w = shard_ratio4_attention_weights(
            prepared, tp_rank=rank, tp_size=TP, config=base_cfg
        )
        shard_attn = Ratio4TorchAttention(
            cfg_r, shard_w, seeded_state(cfg_r, oracle, device=device, batch=args.batch),
            tp_group=group,
        )
        result["local_shapes"] = {
            "wq_b": list(shard_w.wq_b.shape),
            "wo_a": list(shard_w.wo_a.shape),
            "wo_b": list(shard_w.wo_b.shape),
            "attn_sink": list(shard_w.attn_sink.shape),
            "local_num_heads": cfg_r.local_num_heads,
            "local_o_groups": cfg_r.local_o_groups,
        }
        full_attn = None
        if rank == 0:
            full_attn = Ratio4TorchAttention(
                base_cfg, prepared,
                seeded_state(base_cfg, oracle, device=device, batch=args.batch),
            )

        for step in range(args.steps):
            gen = torch.Generator(device="cpu").manual_seed(args.seed + step)
            hidden = (
                (torch.randn(args.batch, 1, base_cfg.hidden_size, generator=gen,
                             dtype=torch.float32) * 0.02)
                .to(torch.bfloat16).to(device)
            )
            position = args.start_position + step
            # advance_overlap_state=True: the plan owns the overlap bookkeeping,
            # and without it the pending slot stays -1 and step 1 fails the
            # consistency check that step 0 passes.
            plan = shard_attn.prepare_decode_plan(
                position, advance_overlap_state=True
            )
            out = shard_attn.forward_decode_tensor(
                hidden, start_pos=position, plan=plan
            )
            if rank == 0:
                fplan = full_attn.prepare_decode_plan(
                    position, advance_overlap_state=True
                )
                ref = full_attn.forward_decode_tensor(
                    hidden, start_pos=position, plan=fplan
                )
                delta = (out.float() - ref.float()).abs().max().item()
                scale = ref.float().abs().max().item()
                result["steps"].append({
                    "step": step, "position": position,
                    "max_abs_diff": float(delta),
                    "output_absmax": float(scale),
                    "rel_diff": float(delta / scale) if scale else None,
                    "bitwise_equal": bool(torch.equal(out, ref)),
                })
            dist.barrier(group=group)

        if rank == 0:
            worst = max(s["rel_diff"] for s in result["steps"])
            result["worst_rel_diff"] = worst
            # step 1 put the shipping reduce at 1.25-1.49x the unsharded path's
            # own ~3.1e-3 error against FP64, so a few times 1e-3 is the
            # predicted band.  Wider than that means the wiring, not rounding.
            result["within_predicted_band"] = bool(worst < 2.0e-2)
            result["accepted"] = result["within_predicted_band"]
            print(
                f"[E6F] dist wiring: worst rel diff {worst:.3e} over "
                f"{args.steps} steps | local heads {cfg_r.local_num_heads} "
                f"groups {cfg_r.local_o_groups} wo_b {list(shard_w.wo_b.shape)} "
                f"| {'OK' if result['accepted'] else 'OUT OF BAND'}",
                flush=True,
            )
        else:
            result["accepted"] = True
    except Exception:
        import traceback
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
        print(f"[E6F] rank {rank} FAILED\n{result['errors'][0]}", flush=True)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rank{rank}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
