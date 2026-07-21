#!/usr/bin/env python3
"""Probe: is a row's TP4 Marlin-MoE output bitwise independent of the batch
composition around it?  (18th vertical prerequisite: decides whether the
large-B MTP-on vs MTP-off gate can demand exact per-row equality.)

Single node, 4 ranks (TP4), one real routed MoE layer with a fabricated
learned gate (the routing values are irrelevant; the question is about the
grouped expert GEMM).  For B_global = 32 rows:

  arm0: baseline batch, forward twice           -> determinism check
  arm1: rows 1..31 replaced with fresh noise    -> composition check on row 0
  arm2: rows shuffled so row 0 sits elsewhere   -> position-in-batch check

Row 0's local output must be bitwise identical across all arms for
composition independence to hold.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from dsv4_direct.block_weights import ResidentGateWeights
from dsv4_direct.moe_runtime import TP4MoE, TP4MoEConfig
from dsv4_direct.ops.marlin_moe import load_resident_moe_layer


def det(seed, shape, device, dtype=torch.bfloat16, scale=0.05):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32) * scale
    return value.to(dtype).to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=33)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    config = json.loads(
        (args.stage_root.expanduser() / "config.json").read_text(encoding="utf-8")
    )
    ckpt = "0" * 64
    resident = load_resident_moe_layer(
        stage_root=args.stage_root.expanduser(),
        layer_id=args.layer,
        rank=rank,
        world_size=4,
        hidden_size=int(config["hidden_size"]),
        intermediate_size=int(config["moe_intermediate_size"]),
        n_experts=int(config["n_routed_experts"]),
        device=device,
        checkpoint_id=ckpt,
        progress=(lambda m: print(f"[probe] {m}", flush=True)) if rank == 0 else None,
    )
    if os.environ.get("PROBE_REAL_GATE", "0") == "1":
        from dsv4_direct.block_weights import load_replicated_block_weights

        raw_block = load_replicated_block_weights(
            stage_root=args.stage_root.expanduser(),
            rank=rank,
            world_size=4,
            layer_id=args.layer,
            device=device,
            checkpoint_id=ckpt,
        )
        gate = raw_block.gate
    else:
        gate = ResidentGateWeights(
            weight=det(1234, (int(config["n_routed_experts"]), 4096), device),
            bias=det(1235, (int(config["n_routed_experts"]),), device, torch.float32),
            layer_id=args.layer,
            rank=rank,
            world_size=4,
            checkpoint_id=ckpt,
        )
    moe = TP4MoE(
        config=TP4MoEConfig(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["moe_intermediate_size"]),
            experts=int(config["n_routed_experts"]),
            topk=int(config["num_experts_per_tok"]),
            route_scale=float(config["routed_scaling_factor"]),
            clamp_limit=float(config["swiglu_limit"]),
            world_size=4,
        ),
        resident=resident,
        gate=gate,
        rank=rank,
        device=device,
        global_row_shapes=(32,),
        group=None,
        slots_per_shape=1,
    )

    local = 8
    base_global = det(777, (32, 1, 4096), device, scale=0.02)
    noise_global = det(778, (32, 1, 4096), device, scale=0.02)

    def local_rows(global_hidden):
        return global_hidden[rank * local : (rank + 1) * local].contiguous()

    # arm0 twice: determinism
    out_a = moe.forward_tensor(local_rows(base_global), slot=0).clone()
    out_b = moe.forward_tensor(local_rows(base_global), slot=0).clone()

    # arm1: keep global row 0, replace everything else
    mixed = noise_global.clone()
    mixed[0] = base_global[0]
    out_c = moe.forward_tensor(local_rows(mixed), slot=0).clone()

    # arm1m: keep MIDDLE global row 17 (rank 2, local 1), replace all others.
    # Row 0 always sorts first inside its expert groups, so arm1 cannot see
    # offset-within-group sensitivity; a middle row can.
    mixed_mid = noise_global.clone()
    mixed_mid[17] = base_global[17]
    out_e = moe.forward_tensor(local_rows(mixed_mid), slot=0).clone()

    if rank == 2:
        results_mid_equal = bool(torch.equal(out_a[1], out_e[1]))
        results_mid_diff = float((out_a[1].float() - out_e[1].float()).abs().max())
    # arm2: shuffle rows so global row 0 lands at global position 17 (rank 2)
    perm = torch.roll(torch.arange(32), shifts=17)
    shuffled = base_global[perm].contiguous()
    out_d = moe.forward_tensor(local_rows(shuffled), slot=0).clone()

    deterministic = bool(torch.equal(out_a, out_b))
    row0_rank = 0
    results = {"rank": rank, "deterministic": deterministic}
    if rank == row0_rank:
        results["composition_row0_equal"] = bool(
            torch.equal(out_a[0], out_c[0])
        )
        diff = (out_a[0].float() - out_c[0].float()).abs()
        results["composition_row0_max_abs"] = float(diff.max())
        results["row0_scale_max_abs"] = float(out_a[0].float().abs().max())
    # arm2: global row 0 was moved to global index 17 -> rank 2, local row 1
    if rank == 2:
        target = out_d[1]
        results["note"] = "shuffle places old row0 at rank2 local1"
        gathered_ref: list = [None] * 4
    # broadcast rank0's out_a[0] for the shuffle compare
    ref = out_a[0].contiguous() if rank == 0 else torch.empty_like(out_a[0])
    dist.broadcast(ref, src=0)
    if rank == 2:
        results["middle_row17_equal"] = results_mid_equal
        results["middle_row17_max_abs"] = results_mid_diff
        results["shuffle_row0_equal"] = bool(torch.equal(ref, out_d[1]))
        diff = (ref.float() - out_d[1].float()).abs()
        results["shuffle_row0_max_abs"] = float(diff.max())

    gathered: list = [None] * 4
    dist.all_gather_object(gathered, results)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "result.json").write_text(
            json.dumps(
                {"experiment": "moe-composition-probe", "layer": args.layer,
                 "ranks": gathered},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(gathered, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
