#!/usr/bin/env python3
"""E0mf: real-weight MTP block (mtp.0) forward vs fp32 component oracles.

Sixteenth vertical, gate (a): the runtime MTP block lane
(``dsv4_direct.mtp_block.MTPLane``: enorm/hnorm/e_proj/h_proj bridge ->
HC block core with window attention + learned-router Marlin MoE -> MTP-owned
hc_head/norm -> shared head logits) is driven with real mtp.0 checkpoint
weights on TP4 and compared per stage against independent fp32 oracles,
following the E0df composed-stage teacher-forced form:

- bridge stages (enorm_embedded / hnorm_hidden / bridged): fp32 recomputation
  from raw checkpoint tensors (FP8 block-dequant e_proj/h_proj, fp32 RMS,
  fp32 embedding rows) -- BF16-control-vs-FP32-projection comparison class,
  E0ef/E0wf ``query_lora`` limit 0.012.
- attention branch + ring state: the E0wf raw-FP32 window oracle lane,
  teacher-forced on the candidate's own ``attn_hidden`` (mtp.0 weights; the
  operator implementation is the already-E0wf-verified WindowTorchAttention).
- HC stages (attn_hidden / after_attention / ffn_hidden / block_output):
  fp32 hc_pre/hc_post/rms_norm recomputation, post/comb bitwise (E0df form).
- MoE: learned noaux_tc gate oracle (route IDs exact, weights 2e-5) + fp32
  routed(MXFP4-dequant)+shared(FP8-dequant) oracle on the candidate's
  gathered ffn hidden under the ``mtp.0.ffn`` namespace, E0cf combined
  limit 0.03.
- head stages (hc_head_collapsed / final_norm / logits): fp32 recomputation
  with the MTP block's own hc_head/norm parameters and the fp32-widened
  shared head; logits argmax agreement recorded per phase.

Workload: per-rank deterministic HC residuals + token ids; prefill 96 at
position 0 (bridge/attn/HC/head stages; MoE fp32 oracle is decode-only, as in
E0df) then 3 teacher-forced decode positions 96..98 with the full stage set.

Run (titan064):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=4 \
    e0mf_mtp_block_oracle.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0mf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dsv4_direct.attention_oracle import yarn_rope_table
from dsv4_direct.checkpoint import inspect_stage_checkpoint, load_weight_map
from dsv4_direct.head_stage import load_embed_head_material
from dsv4_direct.hyper_connections import hc_post, hc_pre
from dsv4_direct.attention import rms_norm
from dsv4_direct.model_contract import MTP_LAYER_ID
from dsv4_direct.moe_forward import (
    dequant_fp8_block,
    dequant_mxfp4,
    gate_forward_with_boundary,
)
from dsv4_direct.mtp_block import MTPLane, build_mtp_layer_material
from dsv4_direct.ops.marlin_moe import ShardReader
from dsv4_direct.window_oracle import (
    init_window_oracle_state,
    oracle_prepare_window_attention_weights,
    oracle_window_attention_step,
)


EXPECTED_WORLD = 4
PREFILL_LEN = 96
DECODE_STEPS = 3
VOCAB = 129280

# Comparison-class limits carried over unchanged from the component gates:
# E0ef/E0wf BF16-control-vs-FP32 projection stages (0.012), E0wf branch/state
# (0.040/0.020), E0cf MoE combined (0.03), E0ff route weights (2e-5).
HC_STAGE_LIMIT = 0.012
BRIDGE_LIMIT = 0.012
BRANCH_LIMIT = 0.040
STATE_RAW_LIMIT = 0.020
MOE_COMBINED_LIMIT = 0.03
ROUTE_WEIGHT_LIMIT = 2e-5
HEAD_STAGE_LIMIT = 0.012
LOGITS_LIMIT = 0.012

SEMANTIC_CONTRACT = {
    "model": "deepseek-v4-flash",
    "block": "mtp.0 (layer id 43)",
    "reference": "model.py MTPBlock :738-766 + Transformer :789-793",
    "bridge": "enorm(embed(ids)) -> e_proj; hnorm(hc hidden) -> h_proj; sum",
    "attention": "window ratio-0, no-YaRN base rope_theta 10000 (E0wf lane)",
    "router": "learned noaux_tc (43 >= num_hash_layers)",
    "head": "shared head.weight through MTP-owned hc_head (sigmoid) + norm",
    "logits": "last-position fp32 (ParallelHead.get_logits :716)",
    "measurement_scope": "semantic_correctness_not_performance",
}

IMPLEMENTATION_FILES = (
    "e0mf_mtp_block_oracle.py",
    "dsv4_direct/attention.py",
    "dsv4_direct/attention_oracle.py",
    "dsv4_direct/block_weights.py",
    "dsv4_direct/checkpoint.py",
    "dsv4_direct/head_stage.py",
    "dsv4_direct/hyper_connections.py",
    "dsv4_direct/model_contract.py",
    "dsv4_direct/moe_forward.py",
    "dsv4_direct/moe_runtime.py",
    "dsv4_direct/mtp_block.py",
    "dsv4_direct/ops/marlin_moe.py",
    "dsv4_direct/static_window_kv.py",
    "dsv4_direct/window_attention.py",
    "dsv4_direct/window_oracle.py",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def implementation_sha256(source_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in sorted(IMPLEMENTATION_FILES):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((source_root / relative).read_bytes())
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    metadata = f"{list(tensor.shape)}|{tensor.dtype}|".encode("utf-8")
    return hashlib.sha256(metadata + value).hexdigest()


def deterministic_tensor(
    *, seed: int, shape: tuple[int, ...], device: torch.device, scale: float = 0.02
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32)
    return (value * scale).to(torch.bfloat16).to(device)


def deterministic_token_ids(
    *, seed: int, count: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        0, VOCAB, (1, count), generator=generator, dtype=torch.int64
    ).to(device)


def tensor_metric(
    observed: torch.Tensor, expected: torch.Tensor, *, declared_limit: float
) -> dict[str, Any]:
    if tuple(observed.shape) != tuple(expected.shape):
        raise ValueError(
            f"tensor shape mismatch: {tuple(observed.shape)} != {tuple(expected.shape)}"
        )
    observed_fp32 = observed.detach().to(torch.float32)
    expected_fp32 = expected.detach().to(torch.float32)
    finite = bool(
        torch.isfinite(observed_fp32).all().item()
        and torch.isfinite(expected_fp32).all().item()
    )
    result: dict[str, Any] = {
        "shape": list(observed.shape),
        "finite": finite,
        "declared_limit": declared_limit,
        "rms_rel": None,
        "row_rms_rel_max": None,
        "max_abs": None,
        "accepted": False,
    }
    if not finite:
        return result
    difference = observed_fp32 - expected_fp32
    rms_abs = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_rms = float(torch.sqrt(torch.mean(expected_fp32.square())).item())
    rms_rel = rms_abs / max(reference_rms, 1e-12)
    row_rms_abs = torch.sqrt(torch.mean(difference.square(), dim=-1))
    row_reference = torch.sqrt(torch.mean(expected_fp32.square(), dim=-1))
    row_rms_rel_max = float((row_rms_abs / row_reference.clamp_min(1e-12)).max().item())
    result.update(
        {
            "rms_rel": rms_rel,
            "row_rms_rel_max": row_rms_rel_max,
            "max_abs": float(difference.abs().max().item()),
            "accepted": (
                math.isfinite(rms_rel)
                and math.isfinite(row_rms_rel_max)
                and rms_rel <= declared_limit
                and row_rms_rel_max <= declared_limit * 4.0
            ),
        }
    )
    return result


# --------------------------------------------------------------------------
# fp32 oracles


def oracle_bridge(
    *,
    raw_block: Any,
    embed_weight: torch.Tensor,
    input_ids: torch.Tensor,
    hidden_hc: torch.Tensor,
    norm_eps: float,
) -> dict[str, torch.Tensor]:
    """fp32 recomputation of model.py:760-763 from raw checkpoint tensors."""

    mtp = raw_block.mtp
    e_proj = dequant_fp8_block(mtp.e_proj.weight, mtp.e_proj.scale)
    h_proj = dequant_fp8_block(mtp.h_proj.weight, mtp.h_proj.scale)

    def fp32_rms(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        value = value.float()
        inverse = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + norm_eps)
        return value * inverse * weight.float()

    embedded = F.embedding(input_ids, embed_weight).float()
    enorm_embedded = fp32_rms(embedded, mtp.enorm)
    hnorm_hidden = fp32_rms(hidden_hc, mtp.hnorm)
    bridged = F.linear(enorm_embedded, e_proj).unsqueeze(2) + F.linear(
        hnorm_hidden, h_proj
    )
    return {
        "enorm_embedded": enorm_embedded,
        "hnorm_hidden": hnorm_hidden,
        "bridged": bridged,
    }


def oracle_hc_pre_norm(
    residual: torch.Tensor,
    *,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    sinkhorn_iters: int,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden, post, comb = hc_pre(
        residual.float(),
        hc_fn,
        hc_scale,
        hc_base,
        norm_eps=norm_eps,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
    )
    return rms_norm(hidden, norm_weight, eps=norm_eps), post, comb


def moe_fp32_oracle_partial(
    *,
    stage_root: Path,
    experts_prefix: str,
    rank: int,
    local_intermediate: int,
    x_full: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_ids: torch.Tensor,
    shared: Any,
    clamp_limit: float,
    device: torch.device,
) -> torch.Tensor:
    """E0df's fp32 routed+shared oracle, prefix-parameterized for mtp.0.ffn."""

    rows, hidden_size = x_full.shape
    start = rank * local_intermediate
    end = start + local_intermediate
    x_fp32 = x_full.float()
    output = torch.zeros(rows, hidden_size, dtype=torch.float32, device=device)

    occurrences: dict[int, list[tuple[int, int]]] = {}
    for row, row_ids in enumerate(routing_ids.cpu().tolist()):
        for kth, expert_id in enumerate(row_ids):
            occurrences.setdefault(int(expert_id), []).append((row, kth))

    weight_map, _ = load_weight_map(stage_root)
    with ShardReader(stage_root, weight_map) as handle:
        for expert_id in sorted(occurrences):
            expert = f"{experts_prefix}.{expert_id}"
            w1 = dequant_mxfp4(
                handle.get_slice(f"{expert}.w1.weight")[start:end].contiguous().to(device),
                handle.get_slice(f"{expert}.w1.scale")[start:end].contiguous().to(device),
            )
            w3 = dequant_mxfp4(
                handle.get_slice(f"{expert}.w3.weight")[start:end].contiguous().to(device),
                handle.get_slice(f"{expert}.w3.scale")[start:end].contiguous().to(device),
            )
            w2 = dequant_mxfp4(
                handle.get_slice(f"{expert}.w2.weight")[
                    :, start // 2 : end // 2
                ].contiguous().to(device),
                handle.get_slice(f"{expert}.w2.scale")[
                    :, start // 32 : end // 32
                ].contiguous().to(device),
            )
            for row, kth in occurrences[expert_id]:
                x_row = x_fp32[row : row + 1]
                gate = F.linear(x_row, w1).clamp(max=clamp_limit)
                up = F.linear(x_row, w3).clamp(min=-clamp_limit, max=clamp_limit)
                hidden = F.silu(gate) * up
                hidden.mul_(routing_weights[row, kth].float())
                output[row : row + 1].add_(F.linear(hidden, w2))
            del w1, w3, w2

    shared_w1 = dequant_fp8_block(shared.w1, shared.s1)
    shared_w3 = dequant_fp8_block(shared.w3, shared.s3)
    shared_w2 = dequant_fp8_block(shared.w2, shared.s2)
    shared_gate = F.linear(x_fp32, shared_w1).clamp(max=clamp_limit)
    shared_up = F.linear(x_fp32, shared_w3).clamp(min=-clamp_limit, max=clamp_limit)
    output.add_(F.linear(F.silu(shared_gate) * shared_up, shared_w2))
    return output


def oracle_head_stages(
    *,
    block_output: torch.Tensor,
    bridge: Any,
    head_weight: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> dict[str, torch.Tensor]:
    """fp32 hc_head sigmoid collapse + MTP norm + shared head (model.py:718-735)."""

    flattened = block_output.float().flatten(2)
    inverse_rms = torch.rsqrt(
        flattened.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    mixes = F.linear(flattened, bridge.hc_head_fn) * inverse_rms
    pre = torch.sigmoid(mixes * bridge.hc_head_scale + bridge.hc_head_base) + hc_eps
    collapsed = torch.sum(
        pre.unsqueeze(-1) * flattened.view(block_output.shape).float(), dim=2
    )
    normed = collapsed * torch.rsqrt(
        collapsed.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    normed = normed * bridge.norm
    logits = F.linear(normed[:, -1], head_weight)
    return {"hc_head_collapsed": collapsed, "final_norm": normed, "logits": logits}


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument(
        "--kv-dtype", type=str, default="bf16", choices=("bf16", "fp8", "fp8_rope_bf16")
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    source_root = Path(__file__).resolve().parent
    out_dir = args.out_dir.expanduser().resolve()
    stage_root = args.stage_root.expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0mf-mtp-block-oracle",
        "measurement_class": "semantic_correctness_gate",
        "semantic_contract": SEMANTIC_CONTRACT,
        "implementation_sha256": implementation_sha256(source_root),
        "rank": rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "kv_dtype": args.kv_dtype,
        "workload": {
            "prefill_len": PREFILL_LEN,
            "decode_steps": DECODE_STEPS,
            "seed": args.seed,
            "max_seq_len": args.max_seq_len,
            "limits": {
                "bridge": BRIDGE_LIMIT,
                "hc_stage": HC_STAGE_LIMIT,
                "branch": BRANCH_LIMIT,
                "state_raw": STATE_RAW_LIMIT,
                "moe_combined": MOE_COMBINED_LIMIT,
                "route_weights": ROUTE_WEIGHT_LIMIT,
                "head_stage": HEAD_STAGE_LIMIT,
                "logits": LOGITS_LIMIT,
            },
        },
        "checkpoint_id": None,
        "inputs": {},
        "stage_metrics": {},
        "exact_checks": {},
        "diagnostics": {},
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }
    stage_metrics: dict[str, Any] = result["stage_metrics"]
    exact_checks: dict[str, Any] = result["exact_checks"]
    started = time.perf_counter()

    try:
        if world != EXPECTED_WORLD:
            raise ValueError(f"E0mf requires TP4, got world={world}")
        tp_group = dist.new_group(ranks=[0, 1, 2, 3], backend="nccl")
        warm = torch.ones(1, device=device)
        dist.all_reduce(warm, group=tp_group)
        torch.cuda.synchronize(device)

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root, [MTP_LAYER_ID], world
                )
                if not checkpoint["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {checkpoint['errors'][:4]}"
                    )
                envelope_holder[0] = {
                    "ok": True,
                    "config": config_payload,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                }
            except Exception:
                envelope_holder[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        if not envelope["ok"]:
            raise ValueError(f"rank-0 preflight failed:\n{envelope['error']}")
        result["checkpoint_id"] = envelope["checkpoint_id"]
        model_config = envelope["config"]
        norm_eps = float(model_config["rms_norm_eps"])
        sinkhorn_iters = int(model_config["hc_sinkhorn_iters"])
        hc_eps = float(model_config["hc_eps"])
        clamp_limit = float(model_config["swiglu_limit"])
        route_scale = float(model_config["routed_scaling_factor"])
        topk = int(model_config["num_experts_per_tok"])
        hidden_size = int(model_config["hidden_size"])
        local_intermediate = int(model_config["moe_intermediate_size"]) // world

        load_started = time.perf_counter()
        material = build_mtp_layer_material(
            model_config=model_config,
            stage_root=stage_root,
            tp_rank=rank,
            tp_group=tp_group,
            tp_global_ranks=(0, 1, 2, 3),
            device=device,
            checkpoint_id=result["checkpoint_id"],
            max_seq_len=args.max_seq_len,
            global_row_shapes=(world, world * PREFILL_LEN),
            slots_per_shape=1,
            kv_dtype=args.kv_dtype,
            progress=(
                (lambda message: print(f"[E0mf] {message}", flush=True))
                if rank == 0
                else None
            ),
        )
        embed_head = load_embed_head_material(
            stage_root=stage_root,
            device=device,
            checkpoint_id=result["checkpoint_id"],
            load_embed=True,
            load_head=True,
        )
        result["diagnostic_seconds"]["load"] = time.perf_counter() - load_started
        result["block_resident_bytes"] = material.raw_block.resident_bytes
        result["moe_evidence"] = dict(material.evidence)

        evidence: dict[str, torch.Tensor] = {}
        lane = MTPLane(
            material,
            embed_weight=embed_head.embed_weight,
            head_weight=embed_head.head_weight,
            batch_size=1,
            device=device,
            evidence_sink=evidence,
        )
        twin = MTPLane(
            material,
            embed_weight=embed_head.embed_weight,
            head_weight=embed_head.head_weight,
            batch_size=1,
            device=device,
        )
        oracle_attention_weights = oracle_prepare_window_attention_weights(
            material.raw_block.attention
        )
        oracle_state = init_window_oracle_state(
            material.attention_config, batch_size=1, device=device
        )
        oracle_rope = yarn_rope_table(
            dim=material.attention_config.rope_dim,
            seqlen=material.attention_config.max_seq_len,
            original_seq_len=material.attention_config.original_seq_len,
            base=material.attention_config.rope_theta,
            factor=material.attention_config.rope_factor,
            beta_fast=material.attention_config.beta_fast,
            beta_slow=material.attention_config.beta_slow,
            device=device,
        )
        hc = material.raw_block.hyper_connection

        def compare_phase(
            *,
            phase: str,
            hidden_hc: torch.Tensor,
            input_ids: torch.Tensor,
            start_pos: int,
            with_moe_oracle: bool,
        ) -> None:
            nonlocal oracle_state
            evidence.clear()
            with material.moe.observe_route_tensors(
                capture_local_input=True
            ) as routes:
                logits = lane.forward(
                    hidden_hc.clone(), input_ids.clone(), start_pos=start_pos
                )
            twin_logits = twin.forward(
                hidden_hc.clone(), input_ids.clone(), start_pos=start_pos
            )
            exact_checks[f"{phase}.twin_logits_bitwise"] = bool(
                torch.equal(logits, twin_logits)
            )
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError(f"non-finite MTP logits at {phase}")

            # bridge oracles
            bridge_oracle = oracle_bridge(
                raw_block=material.raw_block,
                embed_weight=embed_head.embed_weight,
                input_ids=input_ids,
                hidden_hc=hidden_hc,
                norm_eps=norm_eps,
            )
            for name in ("enorm_embedded", "hnorm_hidden", "bridged"):
                stage_metrics[f"{phase}.{name}"] = tensor_metric(
                    evidence[name], bridge_oracle[name], declared_limit=BRIDGE_LIMIT
                )

            # HC pre (attn) oracle on the candidate's bridged tensor
            oracle_attn_hidden, oracle_post, oracle_comb = oracle_hc_pre_norm(
                evidence["bridged"],
                hc_fn=hc.attn_fn,
                hc_scale=hc.attn_scale,
                hc_base=hc.attn_base,
                norm_weight=material.raw_block.attn_norm,
                norm_eps=norm_eps,
                sinkhorn_iters=sinkhorn_iters,
                hc_eps=hc_eps,
            )
            stage_metrics[f"{phase}.attn_hidden"] = tensor_metric(
                evidence["attn_hidden"], oracle_attn_hidden,
                declared_limit=HC_STAGE_LIMIT,
            )
            exact_checks[f"{phase}.hc_attn_post_comb_exact"] = bool(
                torch.equal(evidence["attn_post"], oracle_post)
                and torch.equal(evidence["attn_comb"], oracle_comb)
            )

            # window attention oracle lane (E0wf), teacher-forced
            oracle_step = oracle_window_attention_step(
                material.attention_config,
                oracle_attention_weights,
                evidence["attn_hidden"].clone(),
                start_pos=start_pos,
                state=oracle_state,
                rope_table=oracle_rope,
            )
            oracle_state = oracle_step.state
            stage_metrics[f"{phase}.attn_branch"] = tensor_metric(
                evidence["attn_branch"],
                oracle_step.trace.branch,
                declared_limit=BRANCH_LIMIT,
            )
            stage_metrics[f"{phase}.state.raw"] = tensor_metric(
                lane.state.dequantized_latent(),
                oracle_state.raw,
                declared_limit=STATE_RAW_LIMIT,
            )
            exact_checks[f"{phase}.state.next_position"] = (
                lane.state.next_position == int(oracle_state.next_position)
            )

            # HC post/pre (ffn) oracles on candidate tensors
            oracle_after_attention = hc_post(
                evidence["attn_branch"].float(),
                evidence["bridged"].float(),
                evidence["attn_post"],
                evidence["attn_comb"],
            )
            stage_metrics[f"{phase}.after_attention"] = tensor_metric(
                evidence["after_attention"],
                oracle_after_attention,
                declared_limit=HC_STAGE_LIMIT,
            )
            oracle_ffn_hidden, oracle_ffn_post, oracle_ffn_comb = oracle_hc_pre_norm(
                evidence["after_attention"],
                hc_fn=hc.ffn_fn,
                hc_scale=hc.ffn_scale,
                hc_base=hc.ffn_base,
                norm_weight=material.raw_block.ffn_norm,
                norm_eps=norm_eps,
                sinkhorn_iters=sinkhorn_iters,
                hc_eps=hc_eps,
            )
            stage_metrics[f"{phase}.ffn_hidden"] = tensor_metric(
                evidence["ffn_hidden"], oracle_ffn_hidden,
                declared_limit=HC_STAGE_LIMIT,
            )
            exact_checks[f"{phase}.hc_ffn_post_comb_exact"] = bool(
                torch.equal(evidence["ffn_post"], oracle_ffn_post)
                and torch.equal(evidence["ffn_comb"], oracle_ffn_comb)
            )

            # MoE oracle (decode phases; learned gate route + fp32 experts)
            route = routes[0]
            local_flat = (
                evidence["ffn_hidden"].reshape(-1, hidden_size).contiguous()
            )
            exact_checks[f"{phase}.moe_local_input_capture_equal"] = bool(
                route.local_input is not None
                and torch.equal(route.local_input, local_flat)
            )
            if with_moe_oracle:
                rows = local_flat.shape[0]
                x_full = torch.empty(
                    world * rows, hidden_size, dtype=torch.bfloat16, device=device
                )
                dist.all_gather_into_tensor(x_full, local_flat)
                gate_result = gate_forward_with_boundary(
                    x_full,
                    material.raw_block.gate.weight,
                    material.raw_block.gate.bias,
                    topk=topk,
                    route_scale=route_scale,
                )
                exact_checks[f"{phase}.route_ids_match_oracle"] = bool(
                    torch.equal(route.ids, gate_result.routing_ids)
                )
                stage_metrics[f"{phase}.route_weights"] = tensor_metric(
                    route.weights,
                    gate_result.routing_weights,
                    declared_limit=ROUTE_WEIGHT_LIMIT,
                )
                result["diagnostics"][f"{phase}.route_margin_min"] = float(
                    gate_result.margin.min().item()
                )
                oracle_partial = moe_fp32_oracle_partial(
                    stage_root=stage_root,
                    experts_prefix="mtp.0.ffn.experts",
                    rank=rank,
                    local_intermediate=local_intermediate,
                    x_full=x_full,
                    routing_weights=gate_result.routing_weights,
                    routing_ids=gate_result.routing_ids,
                    shared=material.moe.resident.shared,
                    clamp_limit=clamp_limit,
                    device=device,
                )
                dist.all_reduce(oracle_partial, op=dist.ReduceOp.SUM)
                oracle_moe_local = oracle_partial[rank * rows : (rank + 1) * rows]
                stage_metrics[f"{phase}.moe_local"] = tensor_metric(
                    evidence["moe_output"].reshape(-1, hidden_size),
                    oracle_moe_local,
                    declared_limit=MOE_COMBINED_LIMIT,
                )

            # final hc_post oracle
            oracle_block_output = hc_post(
                evidence["moe_output"].float(),
                evidence["after_attention"].float(),
                evidence["ffn_post"],
                evidence["ffn_comb"],
            )
            stage_metrics[f"{phase}.block_output"] = tensor_metric(
                evidence["block_output"],
                oracle_block_output,
                declared_limit=HC_STAGE_LIMIT,
            )

            # head oracles (MTP-owned hc_head/norm + shared head)
            head_oracle = oracle_head_stages(
                block_output=evidence["block_output"],
                bridge=material.bridge,
                head_weight=embed_head.head_weight,
                norm_eps=norm_eps,
                hc_eps=hc_eps,
            )
            stage_metrics[f"{phase}.hc_head_collapsed"] = tensor_metric(
                evidence["hc_head_collapsed"],
                head_oracle["hc_head_collapsed"],
                declared_limit=HEAD_STAGE_LIMIT,
            )
            stage_metrics[f"{phase}.final_norm"] = tensor_metric(
                evidence["final_norm"],
                head_oracle["final_norm"],
                declared_limit=HEAD_STAGE_LIMIT,
            )
            stage_metrics[f"{phase}.logits"] = tensor_metric(
                logits, head_oracle["logits"], declared_limit=LOGITS_LIMIT
            )
            result["diagnostics"][f"{phase}.argmax_agreement"] = bool(
                torch.equal(
                    torch.argmax(logits, dim=-1),
                    torch.argmax(head_oracle["logits"], dim=-1),
                )
            )
            result["diagnostics"][f"{phase}.candidate_argmax"] = int(
                torch.argmax(logits[0]).item()
            )

        # ---- prefill phase ----
        prefill_hidden = deterministic_tensor(
            seed=args.seed + rank * 100_003,
            shape=(1, PREFILL_LEN, 4, hidden_size),
            device=device,
        )
        prefill_ids = deterministic_token_ids(
            seed=args.seed + rank * 7_919, count=PREFILL_LEN, device=device
        )
        result["inputs"]["prefill"] = {
            "hidden_sha256": tensor_sha256(prefill_hidden),
            "ids_sha256": tensor_sha256(prefill_ids),
        }
        phase_started = time.perf_counter()
        compare_phase(
            phase="prefill",
            hidden_hc=prefill_hidden,
            input_ids=prefill_ids,
            start_pos=0,
            with_moe_oracle=False,
        )
        result["diagnostic_seconds"]["prefill"] = time.perf_counter() - phase_started

        # ---- decode phases ----
        for step in range(DECODE_STEPS):
            position = PREFILL_LEN + step
            phase = f"decode_pos{position:03d}"
            hidden = deterministic_tensor(
                seed=args.seed + rank * 100_003 + 50_000 + step * 977,
                shape=(1, 1, 4, hidden_size),
                device=device,
            )
            ids = deterministic_token_ids(
                seed=args.seed + rank * 7_919 + 90_000 + step * 131, count=1,
                device=device,
            )
            result["inputs"][phase] = {
                "hidden_sha256": tensor_sha256(hidden),
                "ids_sha256": tensor_sha256(ids),
            }
            phase_started = time.perf_counter()
            compare_phase(
                phase=phase,
                hidden_hc=hidden,
                input_ids=ids,
                start_pos=position,
                with_moe_oracle=True,
            )
            result["diagnostic_seconds"][phase] = time.perf_counter() - phase_started

        result["accepted"] = bool(
            stage_metrics
            and exact_checks
            and all(metric["accepted"] for metric in stage_metrics.values())
            and all(exact_checks.values())
        )
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
    result["diagnostic_seconds"]["process"] = time.perf_counter() - started

    write_json(out_dir / f"rank-{rank:02d}.json", result)
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, result)
    summary = None
    if rank == 0:
        rank_results = sorted(gathered, key=lambda value: value["rank"])
        worst: dict[str, Any] = {}
        for value in rank_results:
            for name, metric in value["stage_metrics"].items():
                record = worst.setdefault(
                    name,
                    {
                        "rms_rel_max": 0.0,
                        "declared_limit": metric["declared_limit"],
                        "accepted": True,
                    },
                )
                if metric["rms_rel"] is not None:
                    record["rms_rel_max"] = max(
                        record["rms_rel_max"], float(metric["rms_rel"])
                    )
                record["accepted"] = record["accepted"] and bool(metric["accepted"])
        summary = {
            "schema_version": 1,
            "experiment": "E0mf-mtp-block-oracle",
            "measurement_class": "semantic_correctness_gate",
            "accepted": all(value["accepted"] for value in rank_results),
            "semantic_contract": SEMANTIC_CONTRACT,
            "checkpoint_id": rank_results[0]["checkpoint_id"],
            "implementation_sha256": result["implementation_sha256"],
            "kv_dtype": args.kv_dtype,
            "workload": result["workload"],
            "stage_metrics_worst": worst,
            "exact_checks_all": {
                name: all(
                    value["exact_checks"].get(name) is True for value in rank_results
                )
                for name in rank_results[0]["exact_checks"]
            },
            "diagnostics_rank0": rank_results[0]["diagnostics"],
            "errors": [
                error for value in rank_results for error in value["errors"]
            ],
        }
        write_json(out_dir / "summary.json", summary)
        print(
            f"[E0mf] overall: {'PASS' if summary['accepted'] else 'FAIL'}",
            flush=True,
        )
    holder: list[Any] = [summary["accepted"] if rank == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    dist.destroy_process_group()
    return 0 if holder[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
