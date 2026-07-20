#!/usr/bin/env python3
"""E0ef: independent real-weight ratio-128 attention semantic gate (V4-Flash).

Ported from gaiban E0e (e0e_tp4_attention_oracle.py) with the DeepSeek-V4-Flash
geometry: hidden 4096, 64 heads, head_dim 512, q_lora 1024, o_groups 8.  The
candidate (Ratio128TorchAttention, BF16 dequantized-weight control) and the
independent FP32 raw-checkpoint oracle both derive every shape from the
checkpoint config, so this script only re-freezes the identity constants
(layer, resident bytes, implementation file set).  Process form is identical
to gaiban E0e: torchrun TP4, one replicated-weight candidate per rank.

Run (titan064):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=4 \
    e0ef_ratio128_attention_oracle.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0ef
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
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from dsv4_direct.attention import (
    Ratio128AttentionConfig,
    Ratio128TorchAttention,
    prepare_attention_weights,
)
from dsv4_direct.attention_oracle import (
    init_ratio128_oracle_state,
    oracle_prepare_attention_weights,
    oracle_ratio128_attention_step,
    oracle_sparse_attention,
    yarn_rope_table,
)
from dsv4_direct.block_weights import (
    inspect_replicated_block_contract,
    load_replicated_block_weights,
)
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.static_kv import COMPRESS_RATIO, WINDOW_SIZE, StaticLayerKV
from fp8_kv_gate_common import (
    FP8_STAGE_RMS_REL_OVERRIDES,
    fp8_qdq_error_stats,
    latent_amplitude_stats,
)


EXPECTED_WORLD = 4
# Flash layer 3 is the first ratio-128 layer (odd layers 3..41 are ratio-128).
EXPECTED_LAYER = 3
# Frozen from the verified Flash loader smoke run on titan064
# (runtime/loader-smoke-titan064.log: layer 3 block_bytes).
EXPECTED_BLOCK_RESIDENT_BYTES = 120_876_888
CASE_SPECS = (
    ("prefill128_decode128", 128),
    ("prefill127_decode127_boundary", 127),
)
IMPLEMENTATION_FILES = (
    "e0ef_ratio128_attention_oracle.py",
    "dsv4_direct/attention.py",
    "dsv4_direct/attention_oracle.py",
    "dsv4_direct/block_weights.py",
    "dsv4_direct/checkpoint.py",
    "dsv4_direct/moe_forward.py",
    "dsv4_direct/model_contract.py",
    "dsv4_direct/static_kv.py",
    "dsv4_direct/stateful_decode.py",
)

# These limits are fixed before the first real-checkpoint run. They compare a
# BF16 control path with an independent raw-checkpoint FP32 projection oracle.
STAGE_RMS_REL_LIMITS = {
    "query_lora": 0.012,
    "query": 0.020,
    "raw_latent": 0.012,
    "projected_kv": 0.00002,
    "projected_score": 0.00002,
    "compression_pooled": 0.00005,
    "compression_finalized": 0.020,
    "attention_kv": 0.020,
    "sparse_output": 0.030,
    "sparse_control": 0.003,
    "inverse_rotated": 0.030,
    "output_lora": 0.035,
    "branch": 0.040,
    "state.raw": 0.020,
    "state.compressed": 0.020,
    "state.compressor_kv": 0.00002,
    "state.compressor_score": 0.00002,
}
MAIN_STAGE_SUFFIXES = {
    "query_lora",
    "query",
    "raw_latent",
    "projected_kv",
    "projected_score",
    "attention_kv",
    "sparse_output",
    "sparse_control",
    "inverse_rotated",
    "output_lora",
    "branch",
}
BASE_STATE_SUFFIXES = {"state.raw", "state.compressed"}
COMPRESSION_SUFFIXES = {"compression_pooled", "compression_finalized"}
PENDING_STATE_SUFFIXES = {"state.compressor_kv", "state.compressor_score"}
EXPECTED_EXACT_CHECKS = {
    f"{phase}.{suffix}"
    for phase in ("prefill", "decode")
    for suffix in (
        "compression_presence",
        "topk",
        "next_position",
        "compressed_count",
    )
}

SEMANTIC_CONTRACT = {
    "model": "deepseek-v4-flash",
    "geometry": "hidden4096_heads64_headdim512_qlora1024_ogroups8",
    "layer": 3,
    "compress_ratio": 128,
    "nope_quant": "qat_intended_e4m3_ue8m0",
    "nope_decision": (
        "the model comment, scale_fmt=ue8m0 config, and quantizer API define QAT "
        "E4M3 quantize/dequantize intent; the current executable inplace BF16 "
        "nested cast is treated as a reference implementation defect"
    ),
    "weight_oracle": "raw_checkpoint_fp8_e8m0_block_dequant_fp32",
    "candidate_projection": "bf16_dequantized_weight_control",
    "compressor_oracle": "independent_fp32_ape_softmax_pool",
    "sparse_oracle": "independent_fp32_sink_softmax",
    "measurement_scope": "semantic_correctness_not_performance",
}


def expected_case_stage_keys(case_name: str) -> set[str]:
    if case_name not in {name for name, _ in CASE_SPECS}:
        raise ValueError(f"unsupported E0ef case {case_name}")
    keys = {
        f"{phase}.{suffix}"
        for phase in ("prefill", "decode")
        for suffix in MAIN_STAGE_SUFFIXES | BASE_STATE_SUFFIXES
    }
    compression_phase = (
        "prefill" if case_name == "prefill128_decode128" else "decode"
    )
    pending_phase = "decode" if compression_phase == "prefill" else "prefill"
    keys.update(f"{compression_phase}.{suffix}" for suffix in COMPRESSION_SUFFIXES)
    keys.update(f"{pending_phase}.{suffix}" for suffix in PENDING_STATE_SUFFIXES)
    return keys


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def implementation_sha256(source_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in sorted(IMPLEMENTATION_FILES):
        path = source_root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    metadata = f"{list(tensor.shape)}|{tensor.dtype}|".encode("utf-8")
    return hashlib.sha256(metadata + value).hexdigest()


def deterministic_hidden(
    *, seed: int, batch: int, seqlen: int, hidden_size: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(
        batch,
        seqlen,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    )
    return (value * 0.02).to(torch.bfloat16).to(device)


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
        "observed_dtype": str(observed.dtype),
        "oracle_dtype": str(expected.dtype),
        "finite": finite,
        "declared_limit": declared_limit,
        "rms_abs": None,
        "rms_rel": None,
        "row_rms_rel_max": None,
        "max_abs": None,
        "declared_row_limit": declared_limit * 4.0,
        "accepted": False,
    }
    if not finite:
        return result
    difference = observed_fp32 - expected_fp32
    rms_abs = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_rms = float(torch.sqrt(torch.mean(expected_fp32.square())).item())
    rms_rel = rms_abs / max(reference_rms, 1e-12)
    row_rms_abs = torch.sqrt(torch.mean(difference.square(), dim=-1))
    row_reference_rms = torch.sqrt(torch.mean(expected_fp32.square(), dim=-1))
    row_rms_rel_max = float(
        (row_rms_abs / row_reference_rms.clamp_min(1e-12)).max().item()
    )
    max_abs = float(difference.abs().max().item())
    row_limit = declared_limit * 4.0
    result.update(
        {
            "rms_abs": rms_abs,
            "rms_rel": rms_rel,
            "row_rms_rel_max": row_rms_rel_max,
            "max_abs": max_abs,
            "accepted": (
                math.isfinite(rms_rel)
                and math.isfinite(row_rms_rel_max)
                and rms_rel <= declared_limit
                and row_rms_rel_max <= row_limit
            ),
        }
    )
    return result


def add_metric(
    metrics: dict[str, dict[str, Any]],
    name: str,
    observed: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    suffix = name.split(".", 1)[1]
    limit = STAGE_RMS_REL_LIMITS[suffix]
    metrics[name] = tensor_metric(observed, expected, declared_limit=limit)


def integer_value(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("oracle integer state must be scalar")
        return int(value.item())
    return int(value)


def compare_phase(
    *,
    phase: str,
    candidate_evidence: dict[str, torch.Tensor],
    candidate_state: StaticLayerKV,
    oracle_step: Any,
    stage_metrics: dict[str, dict[str, Any]],
    exact_checks: dict[str, bool],
) -> None:
    trace = oracle_step.trace
    pairs = {
        "query_lora": ("query_lora", "query_lora"),
        "query": ("query", "query"),
        "raw_latent": ("raw_latent", "raw_latent"),
        "projected_kv": ("projected_kv", "projected_kv"),
        "projected_score": ("projected_score", "projected_score"),
        "attention_kv": ("attention_kv", "attention_kv"),
        "sparse_output": ("sparse_output", "sparse_output"),
        "inverse_rotated": ("inverse_rope_output", "inverse_rotated"),
        "output_lora": ("output_lora", "output_lora"),
        "branch": ("branch", "branch"),
    }
    for metric_name, (candidate_name, oracle_name) in pairs.items():
        add_metric(
            stage_metrics,
            f"{phase}.{metric_name}",
            candidate_evidence[candidate_name],
            getattr(trace, oracle_name),
        )
    add_metric(
        stage_metrics,
        f"{phase}.sparse_control",
        candidate_evidence["sparse_output"],
        candidate_evidence["sparse_control_oracle"],
    )

    candidate_has_compression = "compression_pooled" in candidate_evidence
    oracle_has_compression = trace.compression_pooled is not None
    exact_checks[f"{phase}.compression_presence"] = (
        candidate_has_compression == oracle_has_compression
    )
    if candidate_has_compression and oracle_has_compression:
        add_metric(
            stage_metrics,
            f"{phase}.compression_pooled",
            candidate_evidence["compression_pooled"],
            trace.compression_pooled,
        )
        add_metric(
            stage_metrics,
            f"{phase}.compression_finalized",
            candidate_evidence["compression_finalized"],
            trace.compression_finalized,
        )

    candidate_topk = candidate_evidence["topk"]
    oracle_topk = trace.topk_indices
    exact_checks[f"{phase}.topk"] = (
        candidate_topk.dtype == oracle_topk.dtype
        and tuple(candidate_topk.shape) == tuple(oracle_topk.shape)
        and torch.equal(candidate_topk, oracle_topk)
    )

    oracle_state = oracle_step.state
    exact_checks[f"{phase}.next_position"] = (
        candidate_state.next_position == integer_value(oracle_state.next_position)
    )
    candidate_compressed_count = int(
        candidate_state.metadata()["compressed_count"][0]
    )
    exact_checks[f"{phase}.compressed_count"] = (
        candidate_compressed_count
        == integer_value(oracle_state.compressed_count)
    )

    # FP8 KV: compare the BF16 values decode actually reads (identity views
    # for bf16 storage, so the frozen comparison is unchanged there).
    candidate_dequant = candidate_state.dequantized_latent()
    state_pairs = {
        "state.raw": (candidate_dequant[:, :WINDOW_SIZE], oracle_state.raw),
        "state.compressed": (
            candidate_dequant[:, WINDOW_SIZE:],
            oracle_state.compressed,
        ),
    }
    pending = candidate_state.next_position % COMPRESS_RATIO
    if pending:
        state_pairs["state.compressor_kv"] = (
            candidate_state.kv_state[:, :pending],
            oracle_state.compressor_kv[:, :pending],
        )
        state_pairs["state.compressor_score"] = (
            candidate_state.score_state[:, :pending],
            oracle_state.compressor_score[:, :pending],
        )
    for metric_name, (candidate_value, oracle_value) in state_pairs.items():
        add_metric(
            stage_metrics,
            f"{phase}.{metric_name}",
            candidate_value,
            oracle_value,
        )


def run_case(
    *,
    name: str,
    prefill_len: int,
    rank: int,
    seed: int,
    config: Ratio128AttentionConfig,
    candidate_weights: Any,
    oracle_weights: Any,
    device: torch.device,
    kv_dtype: str = "bf16",
) -> dict[str, Any]:
    candidate_state = StaticLayerKV(
        num_local_sequences=1,
        max_seq_len=config.max_seq_len,
        layer_id=EXPECTED_LAYER,
        device=device,
        kv_dtype=kv_dtype,
    )
    candidate = Ratio128TorchAttention(
        config,
        candidate_weights,
        candidate_state,
        nope_quant_mode="qat_intended_e4m3",
    )
    oracle_state = init_ratio128_oracle_state(
        config, batch_size=1, device=device
    )
    oracle_rope = yarn_rope_table(
        dim=config.rope_dim,
        seqlen=config.max_seq_len,
        original_seq_len=config.original_seq_len,
        base=config.rope_theta,
        factor=config.rope_factor,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
        device=device,
    )
    stage_metrics: dict[str, dict[str, Any]] = {}
    exact_checks: dict[str, bool] = {}
    inputs: dict[str, dict[str, Any]] = {}
    fp8_diagnostics: dict[str, Any] = {}

    phase_specs = (
        ("prefill", 0, prefill_len, seed + rank * 100_003),
        ("decode", prefill_len, 1, seed + rank * 100_003 + 50_000),
    )
    for phase, start_pos, seqlen, phase_seed in phase_specs:
        canonical_hidden = deterministic_hidden(
            seed=phase_seed,
            batch=1,
            seqlen=seqlen,
            hidden_size=config.hidden_size,
            device=device,
        )
        inputs[phase] = {
            "shape": list(canonical_hidden.shape),
            "dtype": str(canonical_hidden.dtype),
            "start_pos": start_pos,
            "seed": phase_seed,
            "sha256": tensor_sha256(canonical_hidden),
        }
        candidate_hidden = canonical_hidden.clone()
        oracle_hidden = canonical_hidden.clone()
        candidate_evidence: dict[str, torch.Tensor] = {}
        candidate_branch, _ = candidate(
            candidate_hidden, start_pos=start_pos, evidence=candidate_evidence
        )
        if not torch.equal(candidate_hidden, canonical_hidden):
            raise AssertionError("candidate attention mutated its hidden input")
        if not torch.equal(candidate_branch, candidate_evidence["branch"]):
            raise AssertionError("candidate evidence branch does not match return value")
        candidate_evidence["sparse_control_oracle"] = oracle_sparse_attention(
            candidate_evidence["query"],
            candidate_evidence["attention_kv"],
            candidate_weights.attn_sink,
            candidate_evidence["topk"],
            config.head_dim**-0.5,
        )
        oracle_step = oracle_ratio128_attention_step(
            config,
            oracle_weights,
            oracle_hidden,
            start_pos=start_pos,
            state=oracle_state,
            rope_table=oracle_rope,
        )
        if not torch.equal(oracle_hidden, canonical_hidden):
            raise AssertionError("oracle attention mutated its hidden input")
        oracle_state = oracle_step.state
        if kv_dtype != "bf16":
            fp8_diagnostics[phase] = {
                "raw_latent_amplitude": latent_amplitude_stats(
                    candidate_evidence["raw_latent"], rope_dim=config.rope_dim
                ),
                "raw_latent_qdq_error": fp8_qdq_error_stats(
                    candidate_evidence["raw_latent"], rope_dim=config.rope_dim
                ),
            }
        compare_phase(
            phase=phase,
            candidate_evidence=candidate_evidence,
            candidate_state=candidate_state,
            oracle_step=oracle_step,
            stage_metrics=stage_metrics,
            exact_checks=exact_checks,
        )

    accepted = (
        set(stage_metrics) == expected_case_stage_keys(name)
        and all(metric["accepted"] for metric in stage_metrics.values())
        and set(exact_checks) == EXPECTED_EXACT_CHECKS
        and all(exact_checks.values())
    )
    return {
        "accepted": accepted,
        "prefill_len": prefill_len,
        "decode_start_pos": prefill_len,
        "kv_dtype": kv_dtype,
        "inputs": inputs,
        "exact_checks": exact_checks,
        "stage_metrics": stage_metrics,
        "fp8_diagnostics": fp8_diagnostics,
        "errors": [],
    }


def aggregate_results(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    case_aggregates: dict[str, Any] = {}
    for case_name, _ in CASE_SPECS:
        rank_cases = [rank["cases"][case_name] for rank in ranks]
        exact_names = sorted(rank_cases[0]["exact_checks"])
        metric_names = sorted(rank_cases[0]["stage_metrics"])
        case_aggregates[case_name] = {
            "accepted_ranks": [
                rank["rank"] for rank in ranks if rank_cases[rank["rank"]]["accepted"]
            ],
            "exact_checks": {
                name: all(case["exact_checks"].get(name) is True for case in rank_cases)
                for name in exact_names
            },
            "stage_metrics": {
                name: {
                    "finite": all(
                        case["stage_metrics"][name]["finite"] for case in rank_cases
                    ),
                    "rms_rel_max": max(
                        float(case["stage_metrics"][name]["rms_rel"])
                        for case in rank_cases
                    ),
                    "declared_limit": float(
                        rank_cases[0]["stage_metrics"][name]["declared_limit"]
                    ),
                    "accepted": all(
                        case["stage_metrics"][name]["accepted"]
                        for case in rank_cases
                    ),
                }
                for name in metric_names
            },
        }
    return {"cases": case_aggregates}


def render_readme(summary: dict[str, Any]) -> str:
    status = "PASS" if summary["accepted"] else "FAIL"
    lines = [
        "# E0ef V4-Flash TP4 ratio-128 attention oracle",
        "",
        "Experiment: `E0ef-ratio128-attention-oracle`",
        "",
        f"Status: **{status}**",
        "",
        "This is a real-checkpoint semantic correctness gate, not a performance run.",
        "It compares the direct BF16 control against an independent raw-checkpoint",
        "FP32 projection, compressor, RoPE, QDQ, sparse-softmax, and output oracle.",
        "",
        "NoPE semantics are fixed to intended E4M3/UE8M0 QAT. Exact checks cover",
        "top-k indices, next position, compression presence, and compressed-row count.",
        "",
        f"Checkpoint: `{summary.get('checkpoint_id')}`",
        f"Implementation: `{summary.get('implementation_sha256')}`",
        "",
    ]
    for case_name, case in summary["aggregates"]["cases"].items():
        case_ok = len(case["accepted_ranks"]) == EXPECTED_WORLD
        if case["stage_metrics"]:
            worst_name, worst = max(
                case["stage_metrics"].items(),
                key=lambda item: item[1]["rms_rel_max"],
            )
            lines.append(
                f"- `{case_name}`: {'PASS' if case_ok else 'FAIL'}; worst rms_rel "
                f"`{worst_name}={worst['rms_rel_max']:.6g}` "
                f"(limit `{worst['declared_limit']:.6g}`)"
            )
        else:
            lines.append(f"- `{case_name}`: FAIL; no complete stage metrics")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument(
        "--kv-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp8", "fp8_rope_bf16"),
        help="candidate latent KV storage dtype (oracle stays reference BF16)",
    )
    args = parser.parse_args()
    if args.kv_dtype != "bf16":
        # FP8 KV semantic-change arm (E0hf form): cache-derived stages get
        # magnitude-recording ceilings; every other stage keeps its frozen
        # limit.  Final adjudication is the E2E golden gate.
        for key, value in FP8_STAGE_RMS_REL_OVERRIDES.items():
            if key in STAGE_RMS_REL_LIMITS:
                STAGE_RMS_REL_LIMITS[key] = value

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    source_root = Path(__file__).resolve().parent
    out_dir = args.out_dir.expanduser().resolve()
    stage_root = args.stage_root.expanduser().resolve()
    implementation_id = implementation_sha256(source_root)
    workload = {
        "local_batch": 1,
        "max_seq_len": args.max_seq_len,
        "kv_dtype": args.kv_dtype,
        "stage_rms_rel_limits": dict(STAGE_RMS_REL_LIMITS),
        "seed": args.seed,
        "cases": [name for name, _ in CASE_SPECS],
        "input_distribution": "CPU FP32 normal * 0.02, cast BF16, deterministic per rank",
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0ef-ratio128-attention-oracle",
        "measurement_class": "semantic_correctness_gate",
        "semantic_contract": SEMANTIC_CONTRACT,
        "implementation_sha256": implementation_id,
        "rank": rank,
        "local_rank": local_rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "workload": workload,
        "checkpoint_id": None,
        "block_contract_id": None,
        "identity": None,
        "cases": {},
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }

    started = time.perf_counter()
    try:
        if world != EXPECTED_WORLD:
            raise ValueError(f"E0ef requires TP4, got world={world}")
        if args.max_seq_len != 256:
            raise ValueError("E0ef oracle shape is fixed to max_seq_len=256")

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root, [EXPECTED_LAYER], world
                )
                if not checkpoint["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {checkpoint['errors'][:3]}"
                    )
                block_contract = inspect_replicated_block_contract(
                    stage_root, rank=0, world_size=world
                )
                if not block_contract["ok"]:
                    raise ValueError(
                        f"block contract failed: {block_contract['errors'][:3]}"
                    )
                envelope_holder[0] = {
                    "ok": True,
                    "config": config_payload,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "block_contract_id": block_contract["contract_id"],
                }
            except Exception:
                envelope_holder[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        if not envelope["ok"]:
            raise ValueError(f"rank-0 preflight failed:\n{envelope['error']}")
        result["checkpoint_id"] = envelope["checkpoint_id"]
        result["block_contract_id"] = envelope["block_contract_id"]
        result["identity"] = {
            "layer": EXPECTED_LAYER,
            "world": EXPECTED_WORLD,
            "checkpoint_id": result["checkpoint_id"],
            "block_contract_id": result["block_contract_id"],
        }

        load_started = time.perf_counter()
        raw_block = load_replicated_block_weights(
            stage_root=stage_root,
            rank=rank,
            world_size=world,
            device=device,
            checkpoint_id=result["checkpoint_id"],
        )
        if raw_block.resident_bytes != EXPECTED_BLOCK_RESIDENT_BYTES:
            raise ValueError("replicated block resident-byte contract failed")
        if raw_block.contract_id != result["block_contract_id"]:
            raise ValueError("replicated block contract identity mismatch")
        config = Ratio128AttentionConfig.from_model_config(
            envelope["config"], layer_id=EXPECTED_LAYER, max_seq_len=args.max_seq_len
        )
        candidate_weights = prepare_attention_weights(
            raw_block.attention,
            layer_id=EXPECTED_LAYER,
            rank=rank,
            world_size=world,
            checkpoint_id=result["checkpoint_id"],
        )
        oracle_weights = oracle_prepare_attention_weights(raw_block.attention)
        result["diagnostic_seconds"]["load_and_prepare"] = (
            time.perf_counter() - load_started
        )

        for case_index, (case_name, prefill_len) in enumerate(CASE_SPECS):
            try:
                result["cases"][case_name] = run_case(
                    name=case_name,
                    prefill_len=prefill_len,
                    rank=rank,
                    seed=args.seed + case_index * 1_000_003,
                    config=config,
                    candidate_weights=candidate_weights,
                    oracle_weights=oracle_weights,
                    device=device,
                    kv_dtype=args.kv_dtype,
                )
            except Exception:
                result["cases"][case_name] = {
                    "accepted": False,
                    "prefill_len": prefill_len,
                    "decode_start_pos": prefill_len,
                    "inputs": {},
                    "exact_checks": {},
                    "stage_metrics": {},
                    "errors": [traceback.format_exc()],
                }
        result["accepted"] = (
            set(result["cases"]) == {name for name, _ in CASE_SPECS}
            and all(case["accepted"] for case in result["cases"].values())
        )
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
    result["diagnostic_seconds"]["process"] = time.perf_counter() - started
    try:
        write_json(out_dir / f"rank-{rank:02d}.json", result)
    except Exception:
        result["accepted"] = False
        result["errors"].append(
            "rank artifact write failed:\n" + traceback.format_exc()
        )

    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, result)
    summary: dict[str, Any] | None = None
    if rank == 0:
        try:
            rank_results = sorted(gathered, key=lambda value: value["rank"])
            identities_match = all(
                value["world"] == EXPECTED_WORLD
                and value["checkpoint_id"] == rank_results[0]["checkpoint_id"]
                and value["block_contract_id"] == rank_results[0]["block_contract_id"]
                and value["implementation_sha256"]
                == rank_results[0]["implementation_sha256"]
                and value["workload"] == rank_results[0]["workload"]
                and value["semantic_contract"] == rank_results[0]["semantic_contract"]
                for value in rank_results
            )
            aggregates = aggregate_results(rank_results)
            accepted = (
                identities_match
                and all(value["accepted"] for value in rank_results)
                and all(
                    len(case["accepted_ranks"]) == EXPECTED_WORLD
                    and all(case["exact_checks"].values())
                    and all(
                        metric["accepted"] and metric["finite"]
                        for metric in case["stage_metrics"].values()
                    )
                    for case in aggregates["cases"].values()
                )
            )
            summary = {
                "schema_version": 1,
                "experiment": "E0ef-ratio128-attention-oracle",
                "measurement_class": "semantic_correctness_gate",
                "accepted": accepted,
                "semantic_contract": SEMANTIC_CONTRACT,
                "checkpoint_id": rank_results[0]["checkpoint_id"],
                "block_contract_id": rank_results[0]["block_contract_id"],
                "identity": rank_results[0]["identity"],
                "implementation_sha256": implementation_id,
                "world": world,
                "workload": workload,
                "rank_files": [f"rank-{value['rank']:02d}.json" for value in rank_results],
                "ranks": rank_results,
                "identity_checks": {"all_ranks_match": identities_match},
                "aggregates": aggregates,
                "errors": [
                    error
                    for value in rank_results
                    for error in (
                        value["errors"]
                        + [
                            case_error
                            for case in value["cases"].values()
                            for case_error in case["errors"]
                        ]
                    )
                ],
            }
        except Exception:
            summary = {
                "schema_version": 1,
                "experiment": "E0ef-ratio128-attention-oracle",
                "measurement_class": "semantic_correctness_gate",
                "accepted": False,
                "semantic_contract": SEMANTIC_CONTRACT,
                "checkpoint_id": result.get("checkpoint_id"),
                "block_contract_id": result.get("block_contract_id"),
                "identity": result.get("identity"),
                "implementation_sha256": implementation_id,
                "world": world,
                "workload": workload,
                "rank_files": [f"rank-{value['rank']:02d}.json" for value in gathered],
                "ranks": gathered,
                "identity_checks": {"all_ranks_match": False},
                "aggregates": {"cases": {}},
                "errors": [traceback.format_exc()],
            }
        try:
            readme = render_readme(summary)
            write_json(out_dir / "summary.json", summary)
            (out_dir / "README.md").write_text(readme, encoding="utf-8")
        except Exception:
            summary["accepted"] = False
            summary["errors"].append(
                "rank-0 artifact finalization failed:\n" + traceback.format_exc()
            )
            try:
                write_json(out_dir / "summary.json", summary)
            except Exception:
                pass

    accepted_holder: list[Any] = [summary["accepted"] if rank == 0 else None]
    dist.broadcast_object_list(accepted_holder, src=0)
    dist.destroy_process_group()
    return 0 if accepted_holder[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
