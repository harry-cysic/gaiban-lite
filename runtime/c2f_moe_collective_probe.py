#!/usr/bin/env python3
"""C2F 22nd vertical: TP4 MoE collective transport probe.

The per-phase allocator probe (``c2f_prefill_stage_bench.py --alloc-probe``)
showed the prefill MoE bucket is ~80% NCCL: at chunk 8192 the all-gather takes
52.6 ms and the reduce-scatter 47.3 ms per layer, while every allocator counter
(``num_alloc_retries`` / ``num_device_alloc`` / ``num_device_free`` / ``num_ooms``)
is exactly 0.  So the "MoE bimodality" is a collective-bandwidth question, not
an allocator question.

This probe isolates the two collectives at the exact MoE shapes, reports
achieved bus bandwidth, and records the P2P capability matrix -- enough to say
whether the slow regime is the no-P2P (host-staged) fallback.

Shapes (chunk 8192, TP4, hidden 4096, bf16):
  all_gather_into_tensor : [8192, 4096] -> [32768, 4096]   (67 MiB in, 268 MiB out)
  reduce_scatter_tensor  : [32768, 4096] -> [8192, 4096]

Run (single node, GPUs 0-3):
  torchrun --standalone --nproc_per_node=4 c2f_moe_collective_probe.py \
      --out-dir out-moe-coll --tag default
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


HIDDEN = 4096
WORLD = 4


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def time_collective(fn, *, iters: int, warmup: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    walls = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        dist.barrier()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        walls.append(time.perf_counter() - started)
    return walls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", default="default")
    parser.add_argument(
        "--local-rows", type=int, nargs="+", default=[8192],
        help="rows per rank; 8192 == the chunk-8192 MoE shape",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "C2F-moe-collective-probe",
        "tag": args.tag,
        "rank": rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "env": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("NCCL_") or key.startswith("PYTORCH_CUDA")
            or key in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
        },
        "errors": [],
    }

    try:
        if world != WORLD:
            raise ValueError("probe requires exactly 4 ranks")

        # P2P capability matrix (what the driver reports, before NCCL choices).
        peers = {}
        for src in range(torch.cuda.device_count()):
            for dst in range(torch.cuda.device_count()):
                if src != dst:
                    peers[f"{src}->{dst}"] = bool(
                        torch.cuda.can_device_access_peer(src, dst)
                    )
        result["can_device_access_peer"] = peers
        result["p2p_all_true_first4"] = all(
            peers[f"{s}->{d}"] for s in range(4) for d in range(4) if s != d
        )

        # The bench uses a dedicated TP subgroup; probe both communicators.
        tp_group = dist.new_group(ranks=list(range(WORLD)), backend="nccl")

        measurements = []
        for local_rows in args.local_rows:
            global_rows = local_rows * WORLD
            local_tensor = torch.randn(
                local_rows, HIDDEN, dtype=torch.bfloat16, device=device
            )
            gathered = torch.empty(
                global_rows, HIDDEN, dtype=torch.bfloat16, device=device
            )
            reduced = torch.empty(
                local_rows, HIDDEN, dtype=torch.bfloat16, device=device
            )
            payload_bytes = gathered.numel() * gathered.element_size()

            for group_name, group in (("world", None), ("tp_subgroup", tp_group)):
                ag = time_collective(
                    lambda: dist.all_gather_into_tensor(
                        gathered, local_tensor, group=group
                    ),
                    iters=args.iters, warmup=args.warmup, device=device,
                )
                rs = time_collective(
                    lambda: dist.reduce_scatter_tensor(
                        reduced, gathered, op=dist.ReduceOp.SUM, group=group
                    ),
                    iters=args.iters, warmup=args.warmup, device=device,
                )
                ag_p50 = statistics.median(ag)
                rs_p50 = statistics.median(rs)
                # bus bandwidth for ring all-gather / reduce-scatter:
                # each rank moves (N-1)/N * payload
                bus_factor = (WORLD - 1) / WORLD * payload_bytes
                measurements.append(
                    {
                        "local_rows": local_rows,
                        "global_rows": global_rows,
                        "group": group_name,
                        "payload_bytes": payload_bytes,
                        "all_gather_p50_ms": ag_p50 * 1e3,
                        "all_gather_bus_gbps": bus_factor / ag_p50 / 1e9,
                        "reduce_scatter_p50_ms": rs_p50 * 1e3,
                        "reduce_scatter_bus_gbps": bus_factor / rs_p50 / 1e9,
                        "all_gather_walls_ms": [w * 1e3 for w in ag],
                        "reduce_scatter_walls_ms": [w * 1e3 for w in rs],
                    }
                )
            del local_tensor, gathered, reduced

        result["measurements"] = measurements
        result["ok"] = True
    except Exception:
        result["ok"] = False
        result["errors"].append(traceback.format_exc())

    gathered_results: list[Any] = [None] * world
    dist.all_gather_object(gathered_results, result)
    if rank == 0:
        summary = dict(gathered_results[0])
        summary["per_rank_ok"] = [e.get("ok") for e in gathered_results]
        summary["per_rank_errors"] = [e.get("errors") for e in gathered_results]
        write_json(args.out_dir / f"moe-collective-{args.tag}.json", summary)
        for entry in summary.get("measurements", []):
            print(
                f"[{args.tag}] rows={entry['local_rows']:>5} "
                f"group={entry['group']:<12} "
                f"all_gather {entry['all_gather_p50_ms']:7.2f} ms "
                f"({entry['all_gather_bus_gbps']:6.2f} GB/s)  "
                f"reduce_scatter {entry['reduce_scatter_p50_ms']:7.2f} ms "
                f"({entry['reduce_scatter_bus_gbps']:6.2f} GB/s)",
                flush=True,
            )
        print(f"[{args.tag}] p2p_all_true_first4={summary.get('p2p_all_true_first4')}")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
