#!/usr/bin/env python3
"""C2F diagnostic: reproduce the W4A8 E2E hang on decode/short-prefill shapes.

Loads one hash layer (L0) and one learned layer (L3) with W4A8 Marlin on TP4,
registers the E2E-like row shapes (4 = decode, 48/88 = short prefill), and
runs forward_tensor on each shape with loud per-rank exception reporting.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.physical_stage import build_physical_layer_material

FP8 = torch.float8_e4m3fn


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    torch.set_grad_enabled(False)
    stage_root = Path(os.path.expanduser("~/Workspace/DeepSeek-V4-Flash"))
    tp_group = dist.new_group(ranks=[0, 1, 2, 3], backend="nccl")
    config = json.loads((stage_root / "config.json").read_text())
    contract = inspect_stage_checkpoint(stage_root, tp_size=4)
    assert contract["ok"]

    shapes = (4, 48, 88)
    for layer_id in (3, 0):
        try:
            material = build_physical_layer_material(
                layer_id=layer_id,
                model_config=config,
                stage_root=stage_root,
                tp_rank=rank,
                tp_group=tp_group,
                tp_global_ranks=(0, 1, 2, 3),
                device=device,
                checkpoint_id=contract["checkpoint_id"],
                max_seq_len=256,
                global_row_shapes=shapes,
                slots_per_shape=1,
                moe_marlin_input_dtype=FP8,
            )
            if rank == 0:
                print(f"layer {layer_id} loaded (route={material.route_kind})", flush=True)
            for rows in shapes:
                seqlen = rows // 4
                hidden = torch.randn(
                    1, seqlen, 4096, dtype=torch.bfloat16, device=device
                ) * 0.02
                ids = None
                if material.route_kind == "hash":
                    ids = torch.randint(
                        0, 1000, (1, seqlen), dtype=torch.int64, device=device
                    )
                out = material.moe.forward_tensor(hidden, input_ids_local=ids, slot=0)
                torch.cuda.synchronize(device)
                if rank == 0:
                    print(
                        f"layer {layer_id} rows {rows}: OK, "
                        f"finite={bool(torch.isfinite(out).all().item())}, "
                        f"rms={out.float().square().mean().sqrt().item():.4f}",
                        flush=True,
                    )
            del material
            torch.cuda.empty_cache()
        except Exception:
            print(f"RANK {rank} layer {layer_id} FAILED:\n{traceback.format_exc()}", flush=True)
            dist.destroy_process_group()
            return 1
    dist.barrier()
    if rank == 0:
        print("REPRO_ALL_OK", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
