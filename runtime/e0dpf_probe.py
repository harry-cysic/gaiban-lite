#!/usr/bin/env python3
"""E0dpf probe: localize DP-vs-replication divergence within one block.

Loads a single layer, builds a replicated block (batch 8) and a DP block
(batch 2, this rank's rows), runs one fixed-position decode step, and
compares every intermediate stage row-slice.  Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from e0dpf_dp_gate import (
    EXPECTED_MOE_RESIDENT_BYTES,  # noqa: F401  (import keeps constants aligned)
    LayerAssets,
    deterministic_tensor,
    global_input_ids,
    global_residual,
)
from dsv4_direct.dp_caliber import dp_row_bounds, dp_row_slice
from dsv4_direct.hyper_connections import hc_post

START_POSITION = 8192
GLOBAL_BATCH = 8


def metrics(observed: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    left = observed.detach().float()
    right = expected.detach().float()
    diff = left - right
    rms = float(diff.square().mean().sqrt().item())
    ref = float(right.square().mean().sqrt().item())
    return {
        "bitwise": bool(torch.equal(observed, expected)),
        "max_abs": float(diff.abs().max().item()),
        "rms_rel": rms / max(ref, 1e-12),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    stage_root = args.stage_root.expanduser().resolve()
    config = json.loads((stage_root / "config.json").read_text(encoding="utf-8"))
    from dsv4_direct.checkpoint import inspect_stage_checkpoint

    checkpoint = inspect_stage_checkpoint(stage_root, [args.layer_id], world)
    checkpoint_id = checkpoint["checkpoint_id"]

    dp_batch = GLOBAL_BATCH // world
    lo, hi = dp_row_bounds(rank, dp_batch)

    asset = LayerAssets(
        layer_id=args.layer_id,
        model_config=config,
        stage_root=stage_root,
        rank=rank,
        world=world,
        checkpoint_id=checkpoint_id,
        device=device,
        global_batch=GLOBAL_BATCH,
        global_row_shapes=(GLOBAL_BATCH * world, GLOBAL_BATCH),
        progress_every=1024,
    )
    asset.build_seed_payload(seed=args.seed)
    rep_block = asset.new_block(
        model_config=config, local_batch=GLOBAL_BATCH, dp_rank=None
    )
    dp_block = asset.new_block(
        model_config=config, local_batch=dp_batch, dp_rank=rank
    )

    report: dict[str, Any] = {"rank": rank, "layer_id": args.layer_id}

    # 0. state row-slice parity at seed time
    rep_items = dict(rep_block.attention.state._owned_tensor_items())
    state_parity = {}
    for name, dp_tensor in dp_block.attention.state._owned_tensor_items():
        rep_tensor = rep_items[name]
        if rep_tensor.ndim >= 1 and rep_tensor.shape[0] == GLOBAL_BATCH:
            state_parity[name] = metrics(dp_tensor, rep_tensor[lo:hi])
    report["seed_state_parity"] = state_parity

    residual_g = global_residual(
        seed=args.seed, position=START_POSITION, batch=GLOBAL_BATCH, device=device
    )
    ids_g = global_input_ids(
        seed=args.seed, position=START_POSITION, batch=GLOBAL_BATCH, device=device
    )
    residual_dp = dp_row_slice(residual_g, rank, dp_batch)
    ids_dp = dp_row_slice(ids_g, rank, dp_batch)

    # 1. attention-side hc_pre + norm
    rep_h, rep_post, rep_comb = rep_block.prepare_attention(residual_g)
    dp_h, dp_post, dp_comb = dp_block.prepare_attention(residual_dp)
    report["prepare_attention_hidden"] = metrics(dp_h, rep_h[lo:hi])
    report["prepare_attention_post"] = metrics(dp_post, rep_post[lo:hi])
    report["prepare_attention_comb"] = metrics(dp_comb, rep_comb[lo:hi])

    # 2. attention branch (fixed-position decode plan)
    if rep_block.compression_ratio == 4:
        rep_plan = rep_block.attention.prepare_decode_plan(
            START_POSITION, advance_overlap_state=True
        )
        dp_plan = dp_block.attention.prepare_decode_plan(
            START_POSITION, advance_overlap_state=True
        )
    else:
        rep_plan = rep_block.attention.prepare_decode_plan(START_POSITION)
        dp_plan = dp_block.attention.prepare_decode_plan(START_POSITION)
    rep_attn = rep_block.attention.forward_decode_tensor(
        rep_h, start_pos=START_POSITION, plan=rep_plan
    )
    dp_attn = dp_block.attention.forward_decode_tensor(
        dp_h, start_pos=START_POSITION, plan=dp_plan
    )
    report["attention_branch"] = metrics(dp_attn, rep_attn[lo:hi])

    # 3. hc_post + FFN-side hc_pre + norm
    rep_after = hc_post(rep_attn, residual_g, rep_post, rep_comb)
    dp_after = hc_post(dp_attn, residual_dp, dp_post, dp_comb)
    report["after_attention"] = metrics(dp_after, rep_after[lo:hi])
    rep_ffn_h, rep_fpost, rep_fcomb = rep_block.prepare_ffn(rep_after)
    dp_ffn_h, dp_fpost, dp_fcomb = dp_block.prepare_ffn(dp_after)
    report["ffn_hidden"] = metrics(dp_ffn_h, rep_ffn_h[lo:hi])

    # 3b. cross-check: feed the *same* rows into the DP MoE as the rep rows
    dp_ffn_from_rep = rep_ffn_h[lo:hi].contiguous()
    report["ffn_hidden_dp_vs_repslice_bitwise"] = bool(
        torch.equal(dp_ffn_h, dp_ffn_from_rep)
    )

    # 4. MoE (collective; both lanes on all ranks in the same order)
    moe_kwargs_rep: dict[str, Any] = {"slot": 0}
    moe_kwargs_dp: dict[str, Any] = {"slot": 1}
    if rep_block.route_kind == "hash":
        moe_kwargs_rep["input_ids_local"] = ids_g
        moe_kwargs_dp["input_ids_local"] = ids_dp
    rep_moe = rep_block.moe.forward_tensor(rep_ffn_h, **moe_kwargs_rep)
    dp_moe = dp_block.moe.forward_tensor(dp_ffn_h, **moe_kwargs_dp)
    report["moe_output"] = metrics(dp_moe, rep_moe[lo:hi])

    # 5. final hc_post
    rep_out = hc_post(rep_moe, rep_after, rep_fpost, rep_fcomb)
    dp_out = hc_post(dp_moe, dp_after, dp_fpost, dp_fcomb)
    report["block_output"] = metrics(dp_out, rep_out[lo:hi])

    torch.cuda.synchronize(device)
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, report)
    if rank == 0:
        for record in gathered:
            print(json.dumps(record, indent=1, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
