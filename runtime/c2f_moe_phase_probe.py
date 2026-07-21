"""C2F phase probe: per-phase timing of the real TP4MoE at prefill row counts.

The component probe showed the routed Marlin GEMM costs ~18 ms for 32768
gathered rows (~135 TFLOPS, ~82% of the 4090 BF16 peak), while the C2F stage
bench attributed ~131 ms/layer to "MoE".  This probe runs the actual
TP4MoE.__call__ across 4 ranks with its built-in stage markers so the
attribution is measured rather than inferred.

  torchrun --standalone --nproc-per-node 4 c2f_moe_phase_probe.py \
      --stage-root ~/Workspace/DeepSeek-V4-Flash --chunk 8192
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from dsv4_direct.block_weights import load_replicated_block_weights
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.moe_runtime import TP4MoE, TP4MoEConfig
from dsv4_direct.ops.marlin_moe import load_resident_moe_layer

HIDDEN, INTER, EXPERTS, TOPK, TP = 4096, 2048, 256, 6, 4
ROUTE_SCALE, CLAMP = 1.5, 10.0

PHASES = (
    "moe_inputs_ready",
    "moe_hidden_all_gather_done",
    "moe_route_done",
    "moe_routed_done",
    "moe_shared_done",
    "moe_combine_done",
    "moe_reduce_scatter_done",
    "moe_finalize_done",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--layer-id", type=int, default=11)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--collect-trace", action="store_true")
    parser.add_argument("--no-barrier", action="store_true",
                        help="emulate the C2F component-wall caliber (per-rank sync, no barrier)")
    parser.add_argument("--out", default="c2f-moe-phase.json")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.set_grad_enabled(False)

    stage_root = Path(args.stage_root).expanduser().resolve()
    envelope = inspect_stage_checkpoint(stage_root, layer_ids=(args.layer_id,), tp_size=TP)
    checkpoint_id = envelope["checkpoint_id"]
    block = load_replicated_block_weights(
        stage_root=stage_root,
        rank=rank,
        world_size=TP,
        layer_id=args.layer_id,
        device=device,
        checkpoint_id=checkpoint_id,
    )
    resident = load_resident_moe_layer(
        stage_root=stage_root,
        layer_id=args.layer_id,
        rank=rank,
        world_size=TP,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        n_experts=EXPERTS,
        device=device,
        checkpoint_id=checkpoint_id,
    )
    global_rows = args.chunk * TP
    moe = TP4MoE(
        config=TP4MoEConfig(
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            experts=EXPERTS,
            topk=TOPK,
            route_scale=ROUTE_SCALE,
            clamp_limit=CLAMP,
            world_size=TP,
        ),
        resident=resident,
        gate=block.gate,
        rank=rank,
        device=device,
        global_row_shapes=(global_rows,),
        group=dist.group.WORLD,
        slots_per_shape=1,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(20260721 + rank)
    hidden = (
        torch.randn(1, args.chunk, HIDDEN, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.05
    )

    events: dict[str, torch.cuda.Event] = {}
    start = torch.cuda.Event(True)

    def marker(name: str) -> None:
        event = events.get(name)
        if event is None:
            event = torch.cuda.Event(True)
            events[name] = event
        event.record()

    totals = {name: 0.0 for name in PHASES}
    wall_total = 0.0
    host_total = 0.0
    import time as _time
    for iteration in range(args.iters + 2):
        torch.cuda.synchronize()
        if not args.no_barrier:
            dist.barrier()
        host_started = _time.perf_counter()
        start.record()
        moe(hidden, collect_trace=args.collect_trace, stage_marker=marker)
        torch.cuda.synchronize()
        host_elapsed = (_time.perf_counter() - host_started) * 1e3
        if iteration < 2:  # warmup
            continue
        host_total += host_elapsed
        previous = start
        for name in PHASES:
            event = events.get(name)
            if event is None:
                continue
            totals[name] += previous.elapsed_time(event)
            previous = event
        wall_total += start.elapsed_time(previous)

    iters = args.iters
    report = {
        "rank": rank,
        "layer_id": args.layer_id,
        "chunk": args.chunk,
        "global_rows": global_rows,
        "iters": iters,
        "collect_trace": bool(args.collect_trace),
        "phase_ms": {name: round(totals[name] / iters, 3) for name in PHASES},
        "call_ms": round(wall_total / iters, 3),
        "host_wall_ms": round(host_total / iters, 3),
        "barrier": not args.no_barrier,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    if rank == 0:
        print(json.dumps(report, indent=1))
        Path(args.out).write_text(json.dumps(report, indent=1))
        print("WROTE", args.out)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
