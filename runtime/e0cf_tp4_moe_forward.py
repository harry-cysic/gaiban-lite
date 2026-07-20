#!/usr/bin/env python3
"""E0cf: real-weight single-layer TP4 MoE component correctness gate (V4-Flash).

Ported from gaiban E0c (e0c_tp4_moe_forward.py) with the DeepSeek-V4-Flash
geometry: hidden 4096, moe_intermediate 2048, 256 routed experts, topk 6,
routed_scaling_factor 1.5, swiglu clamp 10.0.  Layer 3 is the first
learned-router layer on Flash too (num_hash_layers 3), so the gaiban process
form is unchanged: torchrun TP4, one intermediate slice per rank, learned
sqrt-softplus routing on the all-gathered rows, Marlin MXFP4 routed experts +
BF16-dequantized FP8 shared expert, reduce-scatter output.  All numeric
tolerances are gaiban's E0c values, unmodified.

Deliberate deviations from gaiban E0c (with rationale):
- Shard resolution goes through model.safetensors.index.json (ShardReader),
  not the Pro "model-%05d-of-00064" numbering: the Flash checkpoint has 46
  shards and the index weight_map is this repo's only resolution mechanism.
- The E0 fingerprint-fixture / operator-version preflight is dropped: the
  lite repo carries no E0 fixture.  Resident sample fingerprints are recorded
  in the per-rank JSON instead, and the checkpoint identity gate is the
  ported inspect_stage_checkpoint contract.
- A TP4MoE runtime section is added: gaiban E0c predates moe_runtime.py, but
  the target of this port IS the TP4MoE runtime, so the same resident
  weights + gate are also driven through dsv4_direct.moe_runtime.TP4MoE
  (deterministic alignment + private _fused_marlin_moe path) and its local
  output is compared against the same FP32 dequant oracle (gaiban "combined"
  tolerance) and against the manual reduce-scatter path (gaiban
  reduce-scatter tolerance, since both are BF16 reductions of the same
  partials).

Run (titan064):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=4 \
    e0cf_tp4_moe_forward.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0cf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dsv4_direct.block_weights import ResidentGateWeights
from dsv4_direct.checkpoint import inspect_stage_checkpoint, load_weight_map
from dsv4_direct.moe_forward import (
    dequant_fp8_block,
    dequant_mxfp4,
    error_metrics,
    gate_forward,
)
from dsv4_direct.moe_runtime import TP4MoE, TP4MoEConfig
from dsv4_direct.ops.marlin_moe import ShardReader, load_resident_moe_layer


# Frozen from the verified Flash loader smoke run on titan064
# (runtime/loader-smoke-titan064.log: layer 3 moe_bytes).
EXPECTED_LAYER_RESIDENT_BYTES = 861_931_008
EXPECTED_WORLD_SIZE = 4
T = TypeVar("T")


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    metadata = f"{list(tensor.shape)}|{tensor.dtype}|".encode()
    return hashlib.sha256(metadata + payload).hexdigest()


def memory_summary(device: torch.device) -> dict[str, float]:
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def synchronized_local_step(
    name: str,
    fn: Callable[[], T],
    *,
    device: torch.device,
    world: int,
) -> T:
    """Run a non-collective step, then make any rank-local error fail all ranks."""

    value: T | None = None
    local_error: str | None = None
    try:
        value = fn()
    except Exception:
        local_error = traceback.format_exc()

    failed = torch.tensor(int(local_error is not None), device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        errors: list[str | None] = [None for _ in range(world)]
        dist.all_gather_object(errors, local_error)
        details = "\n".join(
            f"rank {rank}:\n{error}" for rank, error in enumerate(errors) if error
        )
        raise RuntimeError(f"{name} failed before the next collective:\n{details}")
    return value  # type: ignore[return-value]


def time_cuda(fn: Callable[[], T], device: torch.device) -> tuple[T, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def load_gate(
    stage_root: Path,
    layer_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_map, _ = load_weight_map(stage_root)
    prefix = f"layers.{layer_id}.ffn.gate"
    with ShardReader(stage_root, weight_map) as handle:
        weight = handle.get_tensor(f"{prefix}.weight").to(device).contiguous()
        bias = handle.get_tensor(f"{prefix}.bias").to(device).float().contiguous()
    return weight, bias


def make_marlin_context(
    device: torch.device,
    *,
    rows: int,
    hidden_size: int,
    local_intermediate: int,
    topk: int,
) -> dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.scalar_type import scalar_types

    return {
        "fused": fused_marlin_moe,
        "activation": MoEActivation.SILU,
        "quant_type_id": scalar_types.float4_e2m1f.id,
        "workspace": marlin_make_workspace_new(device, EXPECTED_WORLD_SIZE),
        "cache13": torch.empty(
            rows * topk * max(2 * local_intermediate, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        ),
        "cache2": torch.empty(
            rows * topk * local_intermediate,
            dtype=torch.bfloat16,
            device=device,
        ),
        "output": torch.empty(rows, hidden_size, dtype=torch.bfloat16, device=device),
    }


def marlin_partial(
    x_full: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_ids: torch.Tensor,
    resident: Any,
    context: dict[str, Any],
    clamp_limit: float,
) -> torch.Tensor:
    # Each rank owns every expert and one intermediate slice, so expert_map is invalid here.
    return context["fused"](
        x_full,
        resident.routed.w13_q,
        resident.routed.w2_q,
        None,
        None,
        resident.routed.w13_s,
        resident.routed.w2_s,
        topk_weights=routing_weights.contiguous(),
        topk_ids=routing_ids.contiguous(),
        quant_type_id=context["quant_type_id"],
        activation=context["activation"],
        workspace=context["workspace"],
        intermediate_cache13=context["cache13"],
        intermediate_cache2=context["cache2"],
        output=context["output"],
        input_dtype=None,
        clamp_limit=clamp_limit,
    )


def shared_candidate_and_oracle(
    x_full: torch.Tensor,
    shared: Any,
    clamp_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sharded shared-expert BF16 fallback and independent FP32 oracle."""

    w1 = dequant_fp8_block(shared.w1, shared.s1)
    w3 = dequant_fp8_block(shared.w3, shared.s3)
    w2 = dequant_fp8_block(shared.w2, shared.s2)

    x_fp32 = x_full.float()
    gate_ref = F.linear(x_fp32, w1).clamp(max=clamp_limit)
    up_ref = F.linear(x_fp32, w3).clamp(min=-clamp_limit, max=clamp_limit)
    hidden_ref = F.silu(gate_ref) * up_ref
    oracle = F.linear(hidden_ref, w2)

    gate = F.linear(x_full, w1.to(torch.bfloat16)).float().clamp(max=clamp_limit)
    up = (
        F.linear(x_full, w3.to(torch.bfloat16))
        .float()
        .clamp(min=-clamp_limit, max=clamp_limit)
    )
    hidden = (F.silu(gate) * up).to(torch.bfloat16)
    candidate = F.linear(hidden, w2.to(torch.bfloat16))
    return candidate, oracle


def routed_oracle_row_zero(
    *,
    stage_root: Path,
    layer_id: int,
    rank: int,
    local_intermediate: int,
    x_row: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_ids: torch.Tensor,
    clamp_limit: float,
    device: torch.device,
) -> torch.Tensor:
    """Dequantize the selected raw experts one at a time for an independent oracle."""

    start = rank * local_intermediate
    end = start + local_intermediate
    prefix = f"layers.{layer_id}.ffn.experts"
    output = torch.zeros(1, x_row.shape[-1], dtype=torch.float32, device=device)
    x_fp32 = x_row.float()

    weight_map, _ = load_weight_map(stage_root)
    with ShardReader(stage_root, weight_map) as handle:
        for kth, expert_id in enumerate(routing_ids[0].tolist()):
            expert = f"{prefix}.{expert_id}"
            w1 = dequant_mxfp4(
                handle.get_slice(f"{expert}.w1.weight")[start:end].contiguous().to(device),
                handle.get_slice(f"{expert}.w1.scale")[start:end].contiguous().to(device),
            )
            gate = F.linear(x_fp32, w1).clamp(max=clamp_limit)
            del w1

            w3 = dequant_mxfp4(
                handle.get_slice(f"{expert}.w3.weight")[start:end].contiguous().to(device),
                handle.get_slice(f"{expert}.w3.scale")[start:end].contiguous().to(device),
            )
            up = F.linear(x_fp32, w3).clamp(min=-clamp_limit, max=clamp_limit)
            del w3

            hidden = F.silu(gate) * up
            hidden.mul_(routing_weights[0, kth].float())
            w2 = dequant_mxfp4(
                handle.get_slice(f"{expert}.w2.weight")[
                    :, start // 2 : end // 2
                ].contiguous().to(device),
                handle.get_slice(f"{expert}.w2.scale")[
                    :, start // 32 : end // 32
                ].contiguous().to(device),
            )
            output.add_(F.linear(hidden, w2))
            del gate, up, hidden, w2
    return output


def validate_config(config: dict[str, Any], layer_id: int, world: int) -> dict[str, Any]:
    expected = {
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "num_hash_layers": 3,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "norm_topk_prob": True,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"unsupported model config: {mismatches}")
    if world != EXPECTED_WORLD_SIZE:
        raise ValueError(f"E0cf requires TP4, got world={world}")
    if layer_id != int(config["num_hash_layers"]):
        raise ValueError(
            f"E0cf fixes the first learned-router layer {config['num_hash_layers']}, got {layer_id}"
        )
    if int(config["moe_intermediate_size"]) % world:
        raise ValueError("MoE intermediate size must divide TP world size")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--rows-per-rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--routed-rms-rel", type=float, default=0.03)
    parser.add_argument("--shared-rms-rel", type=float, default=0.02)
    parser.add_argument("--combined-rms-rel", type=float, default=0.03)
    parser.add_argument("--reduce-scatter-rms-rel", type=float, default=0.01)
    parser.add_argument("--reduce-scatter-max-abs", type=float, default=1e-4)
    parser.add_argument("--progress-every", type=int, default=64)
    args = parser.parse_args()

    process_started = time.perf_counter()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(device)

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0cf-tp4-moe-forward",
        "scope": "single-layer TP4 MoE component correctness; not full block, stage, or performance",
        "measurement_class": "correctness_only",
        "rank": rank,
        "world": world,
        "local_rank": local_rank,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": package_version("vllm"),
        "safetensors": package_version("safetensors"),
        "layer_id": args.layer_id,
        "rows_per_rank": args.rows_per_rank,
        "seed": args.seed,
        "input_scale": args.input_scale,
        "errors": [],
        "validations": [],
        "diagnostic_seconds": {},
    }

    try:
        if args.rows_per_rank < 1:
            raise ValueError("rows-per-rank must be positive")

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config = json.loads((stage_root / "config.json").read_text(encoding="utf-8"))
                expected_config = validate_config(config, args.layer_id, world)
                contract = inspect_stage_checkpoint(stage_root, tp_size=world)
                if not contract["ok"]:
                    raise ValueError(f"checkpoint contract failed: {contract['errors'][:3]}")
                envelope_holder[0] = {
                    "ok": True,
                    "config": config,
                    "expected_config": expected_config,
                    "checkpoint_id": contract["checkpoint_id"],
                }
            except Exception:
                envelope_holder[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        if not envelope["ok"]:
            raise ValueError(f"rank-0 preflight failed:\n{envelope['error']}")

        config = envelope["config"]
        result["checkpoint_id"] = envelope["checkpoint_id"]
        result["config_contract"] = envelope["expected_config"]

        hidden_size = int(config["hidden_size"])
        intermediate_size = int(config["moe_intermediate_size"])
        local_intermediate = intermediate_size // world
        experts = int(config["n_routed_experts"])
        topk = int(config["num_experts_per_tok"])
        clamp_limit = float(config["swiglu_limit"])
        route_scale = float(config["routed_scaling_factor"])

        def load_weights_and_gate() -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
            resident = load_resident_moe_layer(
                stage_root=stage_root,
                layer_id=args.layer_id,
                rank=rank,
                world_size=world,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                n_experts=experts,
                device=device,
                progress_every=args.progress_every,
                progress=lambda message: print(message, flush=True) if rank == 0 else None,
                checkpoint_id=result["checkpoint_id"],
            )
            gate_weight, gate_bias = load_gate(stage_root, args.layer_id, device)
            resident_summary = resident.summary()
            if resident.resident_bytes != EXPECTED_LAYER_RESIDENT_BYTES:
                raise ValueError(
                    f"resident bytes {resident.resident_bytes} != {EXPECTED_LAYER_RESIDENT_BYTES}"
                )
            return resident, gate_weight, gate_bias, resident_summary

        (resident, gate_weight, gate_bias, resident_summary), load_seconds = time_cuda(
            lambda: synchronized_local_step(
                "load weights and gate", load_weights_and_gate, device=device, world=world
            ),
            device,
        )
        result["diagnostic_seconds"]["load_and_repack"] = load_seconds
        result["resident"] = resident_summary

        def prepare_input() -> torch.Tensor:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + rank)
            return (
                torch.randn(
                    args.rows_per_rank,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                * args.input_scale
            ).contiguous()

        # Only input allocation is local; routing happens after all-gather.
        x_local = synchronized_local_step(
            "prepare input", prepare_input, device=device, world=world
        )
        rows = args.rows_per_rank * world
        x_full = synchronized_local_step(
            "allocate all-gather output",
            lambda: torch.empty(rows, hidden_size, dtype=torch.bfloat16, device=device),
            device=device,
            world=world,
        )
        dist.all_gather_into_tensor(x_full, x_local)

        def prepare_routes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
            weights, ids, margin = gate_forward(
                x_full,
                gate_weight,
                gate_bias,
                topk=topk,
                route_scale=route_scale,
            )
            weights = weights.contiguous()
            ids = ids.contiguous()
            margin = margin.contiguous()
            digest = tensor_sha256(ids) + ":" + tensor_sha256(weights)
            return weights, ids, margin, digest

        routing_weights, routing_ids, route_margin, route_digest = synchronized_local_step(
            "learned routing", prepare_routes, device=device, world=world
        )
        route_digests: list[str] = ["" for _ in range(world)]
        dist.all_gather_object(route_digests, route_digest)

        def validate_routes() -> tuple[bool, dict[str, Any]]:
            route_sums = routing_weights.sum(dim=-1)
            route_unique = all(len(set(row)) == topk for row in routing_ids.cpu().tolist())
            digest_match = len(set(route_digests)) == 1
            route_valid = (
                digest_match
                and route_unique
                and int(routing_ids.min().item()) >= 0
                and int(routing_ids.max().item()) < experts
                and bool(torch.isfinite(routing_weights).all().item())
                and bool(torch.isfinite(route_margin).all().item())
                and float(route_margin.min().item()) >= 0.0
                and bool(
                    torch.allclose(
                        route_sums,
                        torch.full_like(route_sums, route_scale),
                        atol=1e-5,
                    )
                )
            )
            record = {
                "algorithm": "learned sqrtsoftplus; bias selection-only; normalized top-k",
                "ids_row_zero": routing_ids[0].cpu().tolist(),
                "weights_row_zero": routing_weights[0].cpu().tolist(),
                "weight_sum_min": float(route_sums.min().item()),
                "weight_sum_max": float(route_sums.max().item()),
                "top6_top7_margin_min": float(route_margin.min().item()),
                "digest": route_digest,
                "all_rank_digest_match": digest_match,
            }
            return route_valid, record

        route_valid, result["routing"] = synchronized_local_step(
            "validate learned routing", validate_routes, device=device, world=world
        )
        if not route_valid:
            raise ValueError("learned routing invariant failed")

        context = synchronized_local_step(
            "allocate Marlin workspace",
            lambda: make_marlin_context(
                device,
                rows=rows,
                hidden_size=hidden_size,
                local_intermediate=local_intermediate,
                topk=topk,
            ),
            device=device,
            world=world,
        )
        routed_partial, routed_seconds = time_cuda(
            lambda: synchronized_local_step(
                "Marlin routed partial",
                lambda: marlin_partial(
                    x_full,
                    routing_weights,
                    routing_ids,
                    resident,
                    context,
                    clamp_limit,
                ).clone(),
                device=device,
                world=world,
            ),
            device,
        )
        result["diagnostic_seconds"]["marlin_first_call"] = routed_seconds

        (shared_partial, shared_oracle_partial), shared_seconds = time_cuda(
            lambda: synchronized_local_step(
                "shared BF16 fallback and FP32 oracle",
                lambda: shared_candidate_and_oracle(x_full, resident.shared, clamp_limit),
                device=device,
                world=world,
            ),
            device,
        )
        result["diagnostic_seconds"]["shared_bf16_fallback_and_oracle"] = shared_seconds

        routed_oracle_partial, oracle_seconds = time_cuda(
            lambda: synchronized_local_step(
                "raw-checkpoint routed oracle",
                lambda: routed_oracle_row_zero(
                    stage_root=stage_root,
                    layer_id=args.layer_id,
                    rank=rank,
                    local_intermediate=local_intermediate,
                    x_row=x_full[:1],
                    routing_weights=routing_weights,
                    routing_ids=routing_ids,
                    clamp_limit=clamp_limit,
                    device=device,
                ),
                device=device,
                world=world,
            ),
            device,
        )
        result["diagnostic_seconds"]["raw_checkpoint_oracle"] = oracle_seconds

        def prepare_reduce_scatter() -> tuple[torch.Tensor, torch.Tensor]:
            combined = (routed_partial.float() + shared_partial.float()).to(torch.bfloat16)
            reduced = torch.empty(
                args.rows_per_rank, hidden_size, dtype=torch.bfloat16, device=device
            )
            return combined, reduced

        combined_partial, reduced_local = synchronized_local_step(
            "prepare reduce-scatter buffers",
            prepare_reduce_scatter,
            device=device,
            world=world,
        )
        dist.reduce_scatter_tensor(reduced_local, combined_partial, op=dist.ReduceOp.SUM)
        combined_allreduce = synchronized_local_step(
            "prepare rank-order all-reduce buffer",
            combined_partial.clone,
            device=device,
            world=world,
        )
        dist.all_reduce(combined_allreduce, op=dist.ReduceOp.SUM)
        expected_local = combined_allreduce[
            rank * args.rows_per_rank : (rank + 1) * args.rows_per_rank
        ]

        def prepare_oracle_reductions() -> tuple[torch.Tensor, ...]:
            return (
                routed_partial[:1].float(),
                shared_partial[:1].float(),
                combined_partial[:1].float(),
                routed_oracle_partial.float(),
                shared_oracle_partial[:1].float(),
            )

        oracle_tensors = synchronized_local_step(
            "prepare oracle all-reduce buffers",
            prepare_oracle_reductions,
            device=device,
            world=world,
        )
        (
            routed_observed,
            shared_observed,
            combined_observed,
            routed_reference,
            shared_reference,
        ) = oracle_tensors
        for tensor in (
            routed_observed,
            shared_observed,
            combined_observed,
            routed_reference,
            shared_reference,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # --- TP4MoE runtime section (Flash port target; see module docstring) ---

        def build_runtime_moe() -> TP4MoE:
            gate = ResidentGateWeights(
                weight=gate_weight,
                bias=gate_bias,
                layer_id=args.layer_id,
                rank=rank,
                world_size=world,
                checkpoint_id=result["checkpoint_id"],
            )
            return TP4MoE(
                config=TP4MoEConfig(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    experts=experts,
                    topk=topk,
                    route_scale=route_scale,
                    clamp_limit=clamp_limit,
                    world_size=world,
                ),
                resident=resident,
                gate=gate,
                rank=rank,
                device=device,
                global_row_shapes=(rows,),
            )

        runtime_moe, runtime_build_seconds = time_cuda(
            lambda: synchronized_local_step(
                "build TP4MoE runtime", build_runtime_moe, device=device, world=world
            ),
            device,
        )
        result["diagnostic_seconds"]["tp4moe_build"] = runtime_build_seconds
        runtime_output, runtime_trace = runtime_moe(
            x_local.view(1, args.rows_per_rank, hidden_size),
            collect_trace=True,
        )
        torch.cuda.synchronize(device)
        runtime_local = synchronized_local_step(
            "flatten TP4MoE output",
            lambda: runtime_output.reshape(args.rows_per_rank, hidden_size),
            device=device,
            world=world,
        )

        def calculate_metrics() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
            combined_reference = routed_reference + shared_reference
            observed_metrics = {
                "routed": error_metrics(routed_observed, routed_reference),
                "shared_bf16_fallback": error_metrics(shared_observed, shared_reference),
                "combined": error_metrics(combined_observed, combined_reference),
            }
            reduce_scatter_metrics = error_metrics(reduced_local, expected_local)
            runtime_metrics: dict[str, Any] = {
                "vs_manual_reduce_scatter": error_metrics(runtime_local, reduced_local),
                "vs_rank_order_all_reduce": error_metrics(runtime_local, expected_local),
                "route_ids_row_zero": list(runtime_trace.route_ids_row_zero),
                "route_weights_row_zero": list(runtime_trace.route_weights_row_zero),
                "route_ids_match_manual": (
                    list(runtime_trace.route_ids_row_zero)
                    == routing_ids[0].cpu().tolist()
                ),
                "route_margin_min": runtime_trace.route_margin_min,
                "buffer_slot": runtime_trace.buffer_slot,
                "shared_path": runtime_trace.shared_path,
            }
            if rank == 0:
                # Global row 0 lands in rank 0's reduce-scatter slice, so the
                # runtime output row is directly comparable to the FP32 oracle.
                runtime_metrics["row_zero_vs_fp32_oracle"] = error_metrics(
                    runtime_local[:1].float(), combined_reference
                )
            finite = all(
                bool(torch.isfinite(tensor.float()).all().item())
                for tensor in (
                    x_full,
                    routing_weights,
                    routed_partial,
                    shared_partial,
                    shared_oracle_partial,
                    routed_oracle_partial,
                    combined_partial,
                    reduced_local,
                    expected_local,
                    runtime_local,
                )
            )
            return observed_metrics, reduce_scatter_metrics, runtime_metrics, finite

        metrics, reduce_scatter_metrics, runtime_metrics, candidate_finite = (
            synchronized_local_step(
                "calculate numeric metrics", calculate_metrics, device=device, world=world
            )
        )
        thresholds = {
            "routed": args.routed_rms_rel,
            "shared_bf16_fallback": args.shared_rms_rel,
            "combined": args.combined_rms_rel,
        }
        validations = []
        finite_flag = torch.tensor(int(candidate_finite), device=device)
        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        validations.append(
            {
                "name": "all_intermediates_finite",
                "passed": bool(finite_flag.item()),
            }
        )
        for name, threshold in thresholds.items():
            metric = metrics[name]
            passed = bool(
                metric["finite"]
                and metric["rms_rel"] is not None
                and metric["rms_rel"] <= threshold
            )
            validations.append(
                {
                    "name": f"{name}_rms_rel",
                    "passed": passed,
                    "observed": metric["rms_rel"],
                    "threshold": threshold,
                }
            )
        validations.append(
            {
                "name": "reduce_scatter_rank_order",
                "passed": bool(
                    reduce_scatter_metrics["finite"]
                    and reduce_scatter_metrics["rms_rel"] is not None
                    and reduce_scatter_metrics["rms_rel"] <= args.reduce_scatter_rms_rel
                    and reduce_scatter_metrics["max_abs"] is not None
                    and reduce_scatter_metrics["max_abs"] <= args.reduce_scatter_max_abs
                ),
                "observed_rms_rel": reduce_scatter_metrics["rms_rel"],
                "observed_max_abs": reduce_scatter_metrics["max_abs"],
                "reference_rms": reduce_scatter_metrics["reference_rms"],
                "rms_rel_threshold": args.reduce_scatter_rms_rel,
                "max_abs_threshold": args.reduce_scatter_max_abs,
            }
        )
        runtime_vs_manual = runtime_metrics["vs_manual_reduce_scatter"]
        validations.append(
            {
                # The TP4MoE path reduces the same BF16 partials as the manual
                # path, so gaiban's reduce-scatter tolerance pair applies.
                "name": "tp4moe_runtime_vs_manual",
                "passed": bool(
                    runtime_vs_manual["finite"]
                    and runtime_vs_manual["rms_rel"] is not None
                    and runtime_vs_manual["rms_rel"] <= args.reduce_scatter_rms_rel
                    and runtime_vs_manual["max_abs"] is not None
                    and runtime_vs_manual["max_abs"] <= args.reduce_scatter_max_abs
                ),
                "observed_rms_rel": runtime_vs_manual["rms_rel"],
                "observed_max_abs": runtime_vs_manual["max_abs"],
                "rms_rel_threshold": args.reduce_scatter_rms_rel,
                "max_abs_threshold": args.reduce_scatter_max_abs,
            }
        )
        validations.append(
            {
                "name": "tp4moe_route_ids_match_manual",
                "passed": bool(runtime_metrics["route_ids_match_manual"]),
            }
        )
        if rank == 0:
            runtime_oracle = runtime_metrics["row_zero_vs_fp32_oracle"]
            validations.append(
                {
                    # Same tolerance as the manual combined-vs-oracle gate.
                    "name": "tp4moe_row_zero_vs_fp32_oracle",
                    "passed": bool(
                        runtime_oracle["finite"]
                        and runtime_oracle["rms_rel"] is not None
                        and runtime_oracle["rms_rel"] <= args.combined_rms_rel
                    ),
                    "observed": runtime_oracle["rms_rel"],
                    "threshold": args.combined_rms_rel,
                }
            )
        result["metrics"] = metrics
        result["tp4moe_runtime"] = runtime_metrics
        result["validations"] = validations
        result["reduce_scatter"] = {
            "input_shape": list(combined_partial.shape),
            "output_shape": list(reduced_local.shape),
            "rank_order_metrics": reduce_scatter_metrics,
        }
        result["output_sha256"] = tensor_sha256(reduced_local)
        result["tp4moe_output_sha256"] = tensor_sha256(runtime_local)
        if not all(item["passed"] for item in validations):
            raise ValueError(f"numeric validation failed: {validations}")
        result["ok"] = True
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["ok"] = False

    result["memory"] = memory_summary(device)
    result["process_seconds"] = time.perf_counter() - process_started
    write_json(out_dir / f"rank-{rank:02d}.json", result)

    gathered: list[Any] | None = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(result, gathered, dst=0)
    overall_ok = result["ok"]
    if rank == 0:
        assert gathered is not None
        overall_ok = all(bool(item["ok"]) for item in gathered)
        summary = {
            "schema_version": 1,
            "experiment": "E0cf-tp4-moe-forward",
            "scope": "single-layer TP4 MoE component correctness; not full block, stage, or performance",
            "measurement_class": "correctness_only",
            "ok": overall_ok,
            "checkpoint_id": gathered[0].get("checkpoint_id"),
            "world": world,
            "layer_id": args.layer_id,
            "rows_per_rank": args.rows_per_rank,
            "ranks": gathered,
        }
        write_json(out_dir / "summary.json", summary)
        print(f"{'PASS' if overall_ok else 'FAIL'} E0cf TP4 MoE component correctness", flush=True)

    status_holder = [overall_ok]
    dist.broadcast_object_list(status_holder, src=0)
    dist.destroy_process_group()
    return 0 if status_holder[0] else 1


if __name__ == "__main__":
    sys.exit(main())
