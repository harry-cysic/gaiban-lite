"""C2F probe: where does the runtime's prefill MoE time actually go?

The dense-GEMM microbench measured the *public* fused_marlin_moe at 20.4 ms for
32768 gathered rows, while the C2F stage bench attributed ~131 ms/layer to the
runtime MoE.  This probe times the runtime's own components on identical inputs
to locate the gap: deterministic alignment, the private Marlin call, and the
public wrapper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.deterministic_moe_align import (
    allocate_deterministic_moe_alignment,
    deterministic_moe_align_block_size,
)
from dsv4_direct.moe_runtime import _marlin_block_size_m
from dsv4_direct.ops.marlin_moe import load_resident_moe_layer

HIDDEN, INTER, EXPERTS, TOPK, TP, CLAMP = 4096, 2048, 256, 6, 4, 10.0


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
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
    parser.add_argument("--rows", default="8192,32768")
    parser.add_argument("--out", default="c2f-moe-probe.json")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    stage_root = Path(args.stage_root).expanduser().resolve()
    gate = inspect_stage_checkpoint(stage_root, layer_ids=(args.layer_id,), tp_size=TP)
    resident = load_resident_moe_layer(
        stage_root=stage_root,
        layer_id=args.layer_id,
        rank=0,
        world_size=TP,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        n_experts=EXPERTS,
        device=device,
        checkpoint_id=gate["checkpoint_id"],
    )
    local_inter = INTER // TP

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        _fused_marlin_moe,
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.scalar_type import scalar_types

    workspace = marlin_make_workspace_new(device, 4)
    quant_type = scalar_types.float4_e2m1f
    generator = torch.Generator(device=device)
    generator.manual_seed(20260721)
    results = []

    for rows in (int(v) for v in args.rows.split(",")):
        gathered = (
            torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16, generator=generator)
            * 0.05
        )
        logits = torch.rand(rows, EXPERTS, device=device, generator=generator)
        ids = logits.topk(TOPK, dim=-1)[1].to(torch.int32)
        weights = torch.rand(rows, TOPK, device=device, generator=generator).float()
        block = _marlin_block_size_m(rows=rows, topk=TOPK, experts=EXPERTS)
        alignment = allocate_deterministic_moe_alignment(
            ids, block_size=block, num_experts=EXPERTS
        )
        cache13 = torch.empty(
            rows * TOPK * max(2 * local_inter, HIDDEN), dtype=torch.bfloat16, device=device
        )
        cache2 = torch.empty(rows * TOPK * local_inter, dtype=torch.bfloat16, device=device)
        output = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)

        align_ms = bench(
            lambda: deterministic_moe_align_block_size(
                ids, block_size=block, num_experts=EXPERTS, output=alignment
            )
        )
        deterministic_moe_align_block_size(
            ids, block_size=block, num_experts=EXPERTS, output=alignment
        )

        def private_call():
            return _fused_marlin_moe(
                hidden_states=gathered,
                w1=resident.routed.w13_q,
                w2=resident.routed.w2_q,
                bias1=None,
                bias2=None,
                w1_scale=resident.routed.w13_s,
                w2_scale=resident.routed.w2_s,
                topk_weights=weights,
                num_topk=TOPK,
                quant_type=quant_type,
                apply_router_weight_on_input=False,
                expert_map=None,
                block_size_m=block,
                sorted_token_ids=alignment.sorted_token_ids,
                expert_ids=alignment.expert_ids,
                num_tokens_post_padded=alignment.num_tokens_post_padded,
                activation=MoEActivation.SILU,
                workspace=workspace,
                intermediate_cache13=cache13,
                intermediate_cache2=cache2,
                output=None,
                input_dtype=None,
                is_k_full=True,
                clamp_limit=CLAMP,
            )

        private_ms = bench(private_call)
        contributions = private_call()
        reduce_ms = bench(
            lambda: torch.sum(contributions.view(rows, TOPK, HIDDEN), dim=1, out=output)
        )
        public_ms = bench(
            lambda: fused_marlin_moe(
                gathered,
                resident.routed.w13_q,
                resident.routed.w2_q,
                None,
                None,
                resident.routed.w13_s,
                resident.routed.w2_s,
                topk_weights=weights,
                topk_ids=ids,
                quant_type_id=quant_type.id,
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
        )
        entry = {
            "rows": rows,
            "block_size_m": block,
            "align_ms": round(align_ms, 3),
            "private_marlin_ms": round(private_ms, 3),
            "topk_reduce_ms": round(reduce_ms, 3),
            "runtime_total_ms": round(align_ms + private_ms + reduce_ms, 3),
            "public_marlin_ms": round(public_ms, 3),
        }
        results.append(entry)
        print(json.dumps(entry))
        del gathered, logits, ids, weights, alignment, cache13, cache2, output, contributions
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps({"probe": results}, indent=1))
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
