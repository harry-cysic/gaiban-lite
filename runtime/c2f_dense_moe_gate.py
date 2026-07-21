"""C2F-dense: numeric gate + microbench for the dequant->BF16 dense MoE path.

Single GPU, one real layer's intermediate-TP slice (rank 0 of a TP4 split).
Three arms on identical weights and routing:
  oracle  - FP32 dequant + FP32 per-expert GEMM (small M only)
  marlin  - the frozen W4A16 grouped Marlin path (what prefill uses today)
  dense   - dequant-to-BF16 + sorted per-expert dense GEMM (this vertical)

Run on titan064/065:
  <venv>/bin/python c2f_dense_moe_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.moe_forward import dequant_mxfp4
from dsv4_direct.ops.dense_moe import DenseRoutedExecutor, load_dense_moe_layer
from dsv4_direct.ops.marlin_moe import load_resident_moe_layer

HIDDEN = 4096
INTER = 2048
EXPERTS = 256
TOPK = 6
TP = 4
CLAMP = 10.0
ROUTE_SCALE = 1.5


def make_routing(rows: int, generator: torch.Generator, device: torch.device):
    logits = torch.rand(rows, EXPERTS, device=device, generator=generator)
    ids = logits.topk(TOPK, dim=-1)[1].to(torch.int32)
    weights = torch.rand(rows, TOPK, device=device, generator=generator) + 0.1
    weights = weights / weights.sum(dim=-1, keepdim=True) * ROUTE_SCALE
    return weights.float(), ids


def oracle_routed(gathered, routed, weights, ids, local_inter):
    """FP32 dequant + FP32 GEMM reference for the routed half."""
    out = torch.zeros(gathered.shape[0], HIDDEN, dtype=torch.float32, device=gathered.device)
    x = gathered.float()
    touched = torch.unique(ids).tolist()
    for expert in touched:
        mask = (ids == expert)
        rows, slots = mask.nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        w13 = dequant_mxfp4(routed.w13_packed[expert], routed.w13_scale[expert])
        w2 = dequant_mxfp4(routed.w2_packed[expert], routed.w2_scale[expert])
        projected = x.index_select(0, rows) @ w13.t()
        gate = projected[:, :local_inter].clamp(max=CLAMP)
        up = projected[:, local_inter:].clamp(min=-CLAMP, max=CLAMP)
        hidden = F.silu(gate) * up
        contribution = (hidden @ w2.t()) * weights[rows, slots].unsqueeze(1)
        out.index_add_(0, rows, contribution)
    return out


def rel_fro(reference: torch.Tensor, other: torch.Tensor) -> float:
    reference = reference.float()
    other = other.float()
    return float(torch.linalg.norm(other - reference) / torch.linalg.norm(reference))


def bench(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    stop = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rows", default="4096,8192,16384,32768")
    parser.add_argument("--gate-rows", type=int, default=256)
    parser.add_argument("--expert-chunk", type=int, default=8)
    parser.add_argument("--out", default="c2f-dense-gate.json")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    stage_root = Path(args.stage_root).expanduser().resolve()
    gate = inspect_stage_checkpoint(stage_root, layer_ids=(args.layer_id,), tp_size=TP)
    if not gate["ok"]:
        print("checkpoint gate FAILED:", gate["errors"][:3])
        return 1
    checkpoint_id = gate["checkpoint_id"]
    local_inter = INTER // TP

    common = dict(
        stage_root=stage_root,
        layer_id=args.layer_id,
        rank=args.rank,
        world_size=TP,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        n_experts=EXPERTS,
        device=device,
        checkpoint_id=checkpoint_id,
    )
    t0 = time.perf_counter()
    dense_resident = load_dense_moe_layer(**common)
    dense_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    marlin_resident = load_resident_moe_layer(**common)
    marlin_load = time.perf_counter() - t0
    print(
        f"resident bytes dense={dense_resident.resident_bytes} "
        f"marlin={marlin_resident.resident_bytes} "
        f"load_s dense={dense_load:.1f} marlin={marlin_load:.1f}"
    )

    executor = DenseRoutedExecutor(
        dense_resident.routed,
        n_experts=EXPERTS,
        hidden_size=HIDDEN,
        local_intermediate=local_inter,
        topk=TOPK,
        clamp_limit=CLAMP,
        expert_chunk=args.expert_chunk,
    )

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.scalar_type import scalar_types

    workspace = marlin_make_workspace_new(device, 4)
    quant_id = scalar_types.float4_e2m1f.id

    def marlin_call(gathered, weights, ids, cache13, cache2, output):
        return fused_marlin_moe(
            gathered,
            marlin_resident.routed.w13_q,
            marlin_resident.routed.w2_q,
            None,
            None,
            marlin_resident.routed.w13_s,
            marlin_resident.routed.w2_s,
            topk_weights=weights,
            topk_ids=ids,
            quant_type_id=quant_id,
            activation=MoEActivation.SILU,
            workspace=workspace,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
            output=output,
            input_dtype=None,
            clamp_limit=CLAMP,
            global_num_experts=EXPERTS,
            expert_map=None,
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(20260721)
    report: dict = {
        "experiment": "C2F-dense-moe",
        "layer_id": args.layer_id,
        "rank": args.rank,
        "checkpoint_id": checkpoint_id,
        "expert_chunk": args.expert_chunk,
        "resident_bytes": {
            "dense": dense_resident.resident_bytes,
            "marlin": marlin_resident.resident_bytes,
        },
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }

    # ---- numeric gate (small M so the FP32 oracle is affordable) ----
    rows = args.gate_rows
    gathered = (
        torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.05
    )
    weights, ids = make_routing(rows, generator, device)
    cache13 = torch.empty(rows * TOPK * max(2 * local_inter, HIDDEN), dtype=torch.bfloat16, device=device)
    cache2 = torch.empty(rows * TOPK * local_inter, dtype=torch.bfloat16, device=device)
    output = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)

    oracle = oracle_routed(gathered, dense_resident.routed, weights, ids, local_inter)
    dense_out = executor(gathered, weights, ids)
    # the public fused_marlin_moe already reduces the topk axis
    marlin_out = marlin_call(gathered, weights, ids, cache13, cache2, output)
    gate_metrics = {
        "rows": rows,
        "dense_vs_oracle": rel_fro(oracle, dense_out),
        "marlin_vs_oracle": rel_fro(oracle, marlin_out),
        "dense_vs_marlin": rel_fro(marlin_out, dense_out),
    }
    # dequantization itself must be exact in BF16
    exp0 = dequant_mxfp4(
        dense_resident.routed.w13_packed[0], dense_resident.routed.w13_scale[0]
    )
    from dsv4_direct.ops.dense_moe import _byte_pair_table, dequant_mxfp4_bf16

    exp0_bf16 = dequant_mxfp4_bf16(
        dense_resident.routed.w13_packed[0],
        dense_resident.routed.w13_scale[0],
        _byte_pair_table(device),
    )
    gate_metrics["dequant_bf16_exact"] = bool(torch.equal(exp0_bf16.float(), exp0))
    report["gate"] = gate_metrics
    print(json.dumps(gate_metrics, indent=1))
    del oracle, dense_out, marlin_out, cache13, cache2, output, gathered
    torch.cuda.empty_cache()

    # ---- throughput ----
    timings = []
    for rows in (int(value) for value in args.rows.split(",")):
        gathered = (
            torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16, generator=generator)
            * 0.05
        )
        weights, ids = make_routing(rows, generator, device)
        cache13 = torch.empty(
            rows * TOPK * max(2 * local_inter, HIDDEN), dtype=torch.bfloat16, device=device
        )
        cache2 = torch.empty(rows * TOPK * local_inter, dtype=torch.bfloat16, device=device)
        output = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
        marlin_ms = bench(lambda: marlin_call(gathered, weights, ids, cache13, cache2, output))
        dense_ms = bench(lambda: executor(gathered, weights, ids))
        entry = {
            "rows": rows,
            "marlin_ms": round(marlin_ms, 3),
            "dense_ms": round(dense_ms, 3),
            "speedup": round(marlin_ms / dense_ms, 3),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        timings.append(entry)
        print(json.dumps(entry))
        del gathered, weights, ids, cache13, cache2, output
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    report["timings"] = timings
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
