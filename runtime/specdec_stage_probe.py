#!/usr/bin/env python3
"""T5: single-node TP4 stage-level reproduction of the chained-MTP desync
divergence (18th vertical debug).

Teacher-forced: the committed input residual for commit index t is a
deterministic function f(t) (per global row), so no embed/head/pipeline is
needed.  OFF arm = family eager steps at uniform positions.  ON arm =
chained rounds (row-position pass A/B + shadow restore) with a scripted
accept pattern: rows [0, bl/2) always reject (stay lockstep), rows
[bl/2, bl) follow a pseudo-random pattern (desync).  Every committed ON
output row must be bitwise identical to the OFF output at that row's commit
index.

Usage: torchrun --nproc-per-node 4 specdec_stage_probe.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --layers 0 1 2 \
    --rounds 24 [--hc-backend fused]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.physical_stage import EXPECTED_TP_SIZE, build_physical_stage
from dsv4_direct.specdec import (
    _RATIO4_SHADOW_NAMES,
    Ratio4RowWS,
    forward_spec_stage,
    prepare_spec_stage_plan,
)
from dsv4_direct.stateful_decode import (
    StatefulDecodeCursor,
    build_decode_schedule,
)
from dsv4_direct.superstage import TP4DecodeStage

from e1f_full_decode_bench import (
    HC_MULT,
    HIDDEN,
    build_seed_payload,
    forward_eager_prevalidated,
    seed_state,
    tensor_sha256,
)


START = 2048


def det_hidden(seed, index, mb_global, device):
    generator = torch.Generator(device="cpu").manual_seed(
        (seed * 1_000_003 + index * 7_919) & ((1 << 62) - 1)
    )
    value = (
        torch.randn(mb_global, 1, HC_MULT, HIDDEN, generator=generator,
                    dtype=torch.float32) * 0.02
    )
    return value.to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--local-batch", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--hc-backend", type=str, default="fused",
                        choices=("fused", "eager"))
    parser.add_argument("--kv-dtype", type=str, default="fp8")
    parser.add_argument("--indexer-kv-dtype", type=str, default="fp8")
    parser.add_argument("--trace-moe", action="store_true",
                        help="record MoE input/output per call for divergence attribution")
    parser.add_argument("--stub-moe", action="store_true",
                        help="replace MoE with row-local arithmetic in both arms")
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    bl = args.local_batch
    mb_global = bl * EXPECTED_TP_SIZE
    rounds = args.rounds
    off_steps = 2 * rounds + 2
    stop = START + off_steps + 4
    max_seq_len = ((stop + 127) // 128 + 1) * 128
    stage_root = args.stage_root.expanduser()
    config = json.loads((stage_root / "config.json").read_text(encoding="utf-8"))
    ckpt = "0" * 64

    tp_group = dist.group.WORLD
    material = build_physical_stage(
        stage_id=0,
        layer_ids=tuple(args.layers),
        model_config=config,
        stage_root=stage_root,
        tp_rank=rank,
        tp_group=tp_group,
        tp_global_ranks=(0, 1, 2, 3),
        device=device,
        checkpoint_id=ckpt,
        max_seq_len=max_seq_len,
        global_row_shapes=(mb_global,),
        slots_per_shape=4,
        progress=(lambda m: print(f"[T5] {m}", flush=True)) if rank == 0 else None,
        kv_dtype=args.kv_dtype,
        indexer_kv_dtype=args.indexer_kv_dtype,
    )
    backend = resolve_hc_boundary_backend(args.hc_backend)

    payloads = {
        m.layer_id: build_seed_payload(
            m, seed=args.seed, local_batch=bl, start_position=START,
            device=device, dp_tp_rank=rank,
        )
        for m in material.materials
    }

    def build_stage():
        blocks = []
        for m in material.materials:
            state = m.new_state(num_local_sequences=bl)
            seed_state(m, state, payloads[m.layer_id], start_position=START)
            blocks.append(m.new_block(state))
        stage = TP4DecodeStage(blocks, hc_boundary_backend=backend)
        if args.stub_moe:
            for block in stage.blocks:
                block.moe.forward_tensor = (
                    lambda hidden, **kw: hidden * 0.5
                )
        elif args.trace_moe:
            for block in stage.blocks:
                original = block.moe.forward_tensor

                def wrapped(hidden, *, _original=original, _sink=stage, **kw):
                    output = _original(hidden, **kw)
                    getattr(_sink, "_moe_trace").append(
                        (hidden.detach().clone(), output.detach().clone())
                    )
                    return output

                block.moe.forward_tensor = wrapped
            stage._moe_trace = []
        return stage

    ids_zero = torch.zeros((bl, 1), dtype=torch.int64, device=device)
    input_ids_of = {}

    def committed_ids(index):
        # deterministic committed token ids (feeds hash routing)
        if index not in input_ids_of:
            generator = torch.Generator(device="cpu").manual_seed(
                (args.seed * 31 + index) & ((1 << 62) - 1)
            )
            input_ids_of[index] = torch.randint(
                0, 129280, (mb_global, 1), generator=generator
            )
        value = input_ids_of[index]
        return value[rank * bl : (rank + 1) * bl].to(device)

    def committed_hidden(index):
        value = det_hidden(args.seed, index, mb_global, device)
        return value[rank * bl : (rank + 1) * bl].to(device)

    # ---------------- OFF arm (family eager, uniform positions)
    off_stage = build_stage()
    cursor = StatefulDecodeCursor(start_position=START, device=device)
    off_plan = off_stage.prepare_stateful_decode_plan(
        cursor, start_position=START, stop_position=stop,
        graph_moe_slots=(1, 2, 3),
    )
    schedule = build_decode_schedule(START, off_steps)
    off_out = []
    for index, step in enumerate(schedule):
        off_plan.input_residual_buffer.copy_(committed_hidden(index))
        off_plan.input_ids_buffer.copy_(committed_ids(index))
        forward_eager_prevalidated(off_stage, off_plan, graph_family=step.family)
        cursor.advance_host(step.family)
        off_out.append(
            [tensor_sha256(off_plan.output_buffer[b]) for b in range(bl)]
        )
    off_moe_trace = getattr(off_stage, "_moe_trace", [])
    del off_stage, off_plan
    torch.cuda.empty_cache()

    # ---------------- ON arm (chained rounds, scripted accepts)
    on_stage = build_stage()
    sp = prepare_spec_stage_plan(
        on_stage, batch_size=bl, start_position=START, stop_position=stop,
        moe_slot_a=1, moe_slot_b=2, device=device,
    )
    commit_index = torch.zeros(bl, dtype=torch.int64)  # host, per local row

    def accept_pattern(round_index):
        generator = torch.Generator(device="cpu").manual_seed(
            args.seed + 4243 * round_index
        )
        value = (torch.rand(mb_global, generator=generator) < 0.6)
        value[: mb_global // 2] = False  # global rows first half forced reject
        return value[rank * bl : (rank + 1) * bl]

    mismatches = []
    accept_host = torch.zeros(bl, dtype=torch.bool)
    for round_index in range(rounds):
        # round head buffers
        if round_index == 0:
            sp.advance.zero_()
            sp.accept.fill_(1)
        else:
            sp.accept.copy_(accept_host.to(torch.int64).to(device))
            sp.advance.copy_((1 + accept_host.to(torch.int64)).to(device))
        # pass A input: per-row committed hidden at that row's commit index
        rows_a = torch.stack(
            [committed_hidden(int(commit_index[b]))[b] for b in range(bl)]
        )
        ids_a = torch.stack(
            [committed_ids(int(commit_index[b]))[b] for b in range(bl)]
        )
        sp.input_residual_buffer.copy_(rows_a)
        sp.input_ids_buffer.copy_(ids_a)
        forward_spec_stage(on_stage, sp, pass_b=False)
        out_a = sp.output_buffer.clone()
        # compare committed pass-A rows
        for b in range(bl):
            index = int(commit_index[b])
            if tensor_sha256(out_a[b]) != off_out[index][b]:
                entry = {"round": round_index, "row": b, "pass": "a",
                         "commit_index": index}
                if args.trace_moe and len(args.layers) == 1:
                    on_in, on_out_t = on_stage._moe_trace[2 * round_index]
                    off_in, off_out_t = off_moe_trace[index]
                    entry["moe_input_row_equal"] = bool(
                        torch.equal(on_in[b], off_in[b])
                    )
                    entry["moe_output_row_equal"] = bool(
                        torch.equal(on_out_t[b], off_out_t[b])
                    )
                    entry["moe_output_row_max_abs"] = float(
                        (on_out_t[b].float() - off_out_t[b].float()).abs().max()
                    )
                mismatches.append(entry)
        pattern = accept_pattern(round_index)
        # pass B input: next committed hidden for to-be-accepted rows, junk else
        rows_b = torch.stack(
            [committed_hidden(int(commit_index[b]) + 1)[b] for b in range(bl)]
        )
        junk = det_hidden(args.seed + 999, 100000 + round_index, mb_global, device)[
            rank * bl : (rank + 1) * bl
        ].to(device)
        rows_b = torch.where(
            pattern.to(device).view(-1, 1, 1, 1), rows_b, junk
        )
        ids_b = torch.stack(
            [committed_ids(int(commit_index[b]) + 1)[b] for b in range(bl)]
        )
        sp.input_residual_buffer.copy_(rows_b)
        sp.input_ids_buffer.copy_(ids_b)
        forward_spec_stage(on_stage, sp, pass_b=True)
        out_b = sp.output_buffer.clone()
        for b in range(bl):
            if bool(pattern[b]):
                index = int(commit_index[b]) + 1
                if tensor_sha256(out_b[b]) != off_out[index][b]:
                    mismatches.append(
                        {"round": round_index, "row": b, "pass": "b",
                         "commit_index": index}
                    )
        commit_index += 1 + pattern.to(torch.int64)
        accept_host = pattern
        if rank == 0 and (mismatches or round_index % 8 == 0):
            print(
                f"[T5] round {round_index}: mismatches {len(mismatches)}",
                flush=True,
            )
        flag = torch.tensor([1 if len(mismatches) > 12 else 0], device=device)
        dist.all_reduce(flag)
        if int(flag.item()):
            break

    gathered = [None] * 4
    dist.all_gather_object(gathered, mismatches[:12])
    if rank == 0:
        print(json.dumps({
            "layers": args.layers,
            "hc_backend": args.hc_backend,
            "rounds": rounds,
            "per_rank_mismatches": gathered,
            "accepted": all(not m for m in gathered),
        }, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
