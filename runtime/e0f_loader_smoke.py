"""E0F loader smoke: load replicated block weights + itp MoE slices from the
real Flash checkpoint and assert resident shapes/dtypes per layer type.

Layers: 0 (window+hash), 2 (ratio-4+hash), 3 (ratio-128+learned).
Ranks 0..3 are simulated in one process (no distributed init needed for
loading). Run on titan064:
  ~/Workspace/venvs/sglang/bin/python e0f_loader_smoke.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash [--layers 0,2,3] [--ranks 0,3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dsv4_direct.block_weights import load_replicated_block_weights
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.model_contract import SUPPORTED_LAYER_SPECS
from dsv4_direct.ops.marlin_moe import load_resident_moe_layer

E8M0 = torch.float8_e8m0fnu
FP8 = torch.float8_e4m3fn

HIDDEN = 4096
INTER = 2048
EXPERTS = 256
TP = 4
MOE_BUDGET_BYTES = int(2.5 * 2**30)


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def smoke_layer(stage_root: Path, layer_id: int, rank: int, checkpoint_id: str,
                device: torch.device) -> dict:
    spec = SUPPORTED_LAYER_SPECS[layer_id]
    failures: list[str] = []

    block = load_replicated_block_weights(
        stage_root=stage_root, rank=rank, world_size=TP, layer_id=layer_id,
        device=device, checkpoint_id=checkpoint_id,
    )
    att = block.attention
    check(att.wq_a.weight.shape == (1024, HIDDEN) and att.wq_a.weight.dtype == FP8,
          f"wq_a {att.wq_a.weight.shape} {att.wq_a.weight.dtype}", failures)
    check(att.wq_b.weight.shape == (32768, 1024), f"wq_b {att.wq_b.weight.shape}", failures)
    check(att.wkv.weight.shape == (512, HIDDEN), f"wkv {att.wkv.weight.shape}", failures)
    check(att.wo_a.weight.shape == (8192, HIDDEN), f"wo_a {att.wo_a.weight.shape}", failures)
    check(att.wo_b.weight.shape == (HIDDEN, 8192), f"wo_b {att.wo_b.weight.shape}", failures)
    check(att.wq_a.scale.dtype == E8M0, f"wq_a scale {att.wq_a.scale.dtype}", failures)
    ratio = int(spec["compress_ratio"])
    if ratio == 0:
        check(att.compressor is None, "window layer has compressor", failures)
        check(att.indexer is None, "window layer has indexer", failures)
    else:
        dim = 512 * (2 if ratio == 4 else 1)
        check(att.compressor is not None
              and att.compressor.ape.shape == (ratio, dim), "compressor ape", failures)
        check((att.indexer is not None) == (ratio == 4), "indexer presence", failures)
        if att.indexer is not None:
            check(att.indexer.wq_b.weight.shape == (8192, 1024),
                  f"indexer wq_b {att.indexer.wq_b.weight.shape}", failures)
    if spec["route_kind"] == "hash":
        check(block.gate.tid2eid is not None
              and block.gate.tid2eid.shape == (129280, 6)
              and block.gate.bias is None, "hash gate tensors", failures)
    else:
        check(block.gate.bias is not None and block.gate.bias.shape == (EXPERTS,)
              and block.gate.tid2eid is None, "learned gate tensors", failures)

    moe = load_resident_moe_layer(
        stage_root=stage_root, layer_id=layer_id, rank=rank, world_size=TP,
        hidden_size=HIDDEN, intermediate_size=INTER, n_experts=EXPERTS,
        device=device, checkpoint_id=checkpoint_id, progress=None,
    )
    local = INTER // TP
    check(moe.routed.w13_q.shape[0] == EXPERTS, "routed expert count", failures)
    check(moe.routed.w13_q.dtype == torch.int32, f"w13_q dtype {moe.routed.w13_q.dtype}", failures)
    # marlin repack layout: w13_q [E, K/16, 2*local*2], w2_q [E, local/16, hidden*2]
    check(moe.routed.w13_q.shape[1:] == (HIDDEN // 16, 2 * local * 2),
          f"w13_q shape {tuple(moe.routed.w13_q.shape)}", failures)
    check(moe.routed.w2_q.shape[1:] == (local // 16, HIDDEN * 2),
          f"w2_q shape {tuple(moe.routed.w2_q.shape)}", failures)
    check(moe.shared.w1.shape == (local, HIDDEN), f"shared w1 {moe.shared.w1.shape}", failures)
    check(moe.shared.w2.shape == (HIDDEN, local), f"shared w2 {moe.shared.w2.shape}", failures)
    check(moe.intermediate_start == rank * local and moe.intermediate_end == (rank + 1) * local,
          "itp slice bounds", failures)
    check(moe.resident_bytes <= MOE_BUDGET_BYTES,
          f"moe resident {moe.resident_bytes} > budget {MOE_BUDGET_BYTES}", failures)

    result = {
        "layer": layer_id, "rank": rank, "attn_kind": spec["attn_kind"],
        "route_kind": spec["route_kind"],
        "block_bytes": block.resident_bytes,
        "moe_bytes": moe.resident_bytes,
        "moe_load_seconds": round(moe.load_seconds, 1),
        "failures": failures,
    }
    del block, moe
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--layers", default="0,2,3")
    parser.add_argument("--ranks", default="0,3")
    args = parser.parse_args()
    stage_root = Path(args.stage_root).expanduser().resolve()
    device = torch.device("cuda:0")

    gate = inspect_stage_checkpoint(stage_root, tp_size=TP)
    if not gate["ok"]:
        print("checkpoint gate FAILED:", gate["errors"][:5])
        return 1
    checkpoint_id = gate["checkpoint_id"]
    print(f"checkpoint_id={checkpoint_id}")

    ok = True
    for layer_id in (int(x) for x in args.layers.split(",")):
        for rank in (int(x) for x in args.ranks.split(",")):
            result = smoke_layer(stage_root, layer_id, rank, checkpoint_id, device)
            status = "PASS" if not result["failures"] else "FAIL"
            ok = ok and not result["failures"]
            print(json.dumps({"status": status, **result}))
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


