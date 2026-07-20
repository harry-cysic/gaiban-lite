#!/usr/bin/env python3
"""E0kf: real-weight ratio-4 FP8 KV paired gate (fifteenth vertical).

Why paired instead of a direct oracle diff: the E0ff dual-oracle gate
teacher-forces its state chain through exact byte digests
(``pre_step_state_exact``), which an FP8 candidate cannot satisfy by
construction (the cache bytes are e4m3).  The E0hf form applies instead:
the **bf16 candidate is the reference lane** (itself E0ff-gated against the
raw-FP32 oracle within frozen limits), and each FP8 arm runs the identical
teacher-forced input stream over the same seeded saturated state, isolating
exactly the semantic change under test -- quantized cache reads.

Arms (all sharing one prepared weight set, one seed payload):
  - ``bf16``            reference lane (E0ff-verified semantics)
  - ``fp8``             latent e4m3 full row (A6F fp8_cast form)
  - ``fp8_rope_bf16``   latent e4m3 + BF16 rope tail side tensor
  - ``fp8_idx``         latent e4m3 + indexer_kv e4m3 (full capacity form)

Judgment per arm, over ``--steps`` saturated decode steps (positions
8192..8192+steps, crossing ratio-4 boundaries every 4 steps):
  - exact: compressed_indices/topk identical to the bf16 lane for arms whose
    indexer_kv stays BF16 (the indexer path never touches the latent cache);
    recorded (overlap fraction) but not required for the indexer-fp8 arm.
  - numeric: per-step branch rms_rel vs bf16 lane <= --branch-limit
    (default 0.10, magnitude-recording ceiling), trajectory recorded to
    expose accumulation; selected_kv / sparse_output rms_rel recorded.
  - amplitude: real-weight latent row stats vs the e4m3 dynamic range
    (decides the A6F open point: constant-scale direct cast is valid only if
    nothing clips at 448 and the sub-subnormal mass is negligible).

Single GPU (attention weights are TP-replicated; no collectives in this
path).  Run on titan064:
  CUDA_VISIBLE_DEVICES=0 ~/Workspace/venvs/sglang/bin/python \
    e0kf_fp8_ratio4_paired_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash \
    --out-dir out-e0kf
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from dsv4_direct.block_weights import load_replicated_block_weights
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.ratio4_attention import (
    Ratio4AttentionConfig,
    Ratio4TorchAttention,
    prepare_ratio4_attention_weights,
)
from dsv4_direct.ratio4_oracle import seed_nonzero_ratio4_state
from dsv4_direct.static_ratio4_kv import StaticRatio4KV
from fp8_kv_gate_common import fp8_qdq_error_stats, latent_amplitude_stats


LAYER_ID = 2
MAX_SEQ_LEN = 8448
START_POSITION = 8192
ARM_SPECS: dict[str, dict[str, str]] = {
    "bf16": {"kv_dtype": "bf16", "indexer_dtype": "bf16"},
    "fp8": {"kv_dtype": "fp8", "indexer_dtype": "bf16"},
    "fp8_rope_bf16": {"kv_dtype": "fp8_rope_bf16", "indexer_dtype": "bf16"},
    "fp8_idx": {"kv_dtype": "fp8", "indexer_dtype": "fp8"},
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def deterministic_hidden(seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(1, 1, 4096, generator=generator, dtype=torch.float32)
    return (value * 0.02).to(torch.bfloat16).to(device)


def rms_rel(observed: torch.Tensor, expected: torch.Tensor) -> float:
    difference = observed.float() - expected.float()
    rms_abs = float(torch.sqrt(torch.mean(difference.square())).item())
    reference = float(torch.sqrt(torch.mean(expected.float().square())).item())
    return rms_abs / max(reference, 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--branch-limit", type=float, default=0.10)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0kf-fp8-ratio4-paired-gate",
        "measurement_class": "semantic_change_paired_gate",
        "judgment": (
            "bf16 candidate lane (E0ff-gated vs raw FP32 oracle) is the "
            "reference; each FP8 arm consumes the identical teacher-forced "
            "hidden stream over the same seeded saturated state; exact topk "
            "for latent-only arms, branch rms_rel ceiling "
            f"{args.branch_limit} per step, trajectory recorded"
        ),
        "layer": LAYER_ID,
        "start_position": START_POSITION,
        "steps": args.steps,
        "seed": args.seed,
        "branch_limit": args.branch_limit,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "checkpoint_id": None,
        "seed_payload": {},
        "arms": {},
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }
    started = time.perf_counter()
    try:
        checkpoint = inspect_stage_checkpoint(stage_root, [LAYER_ID], 4)
        if not checkpoint["ok"]:
            raise ValueError(f"checkpoint contract failed: {checkpoint['errors'][:3]}")
        result["checkpoint_id"] = checkpoint["checkpoint_id"]
        config_payload = json.loads(
            (stage_root / "config.json").read_text(encoding="utf-8")
        )
        config = Ratio4AttentionConfig.from_model_config(
            config_payload, layer_id=LAYER_ID, max_seq_len=MAX_SEQ_LEN
        )
        raw_block = load_replicated_block_weights(
            stage_root=stage_root,
            rank=0,
            world_size=4,
            layer_id=LAYER_ID,
            device=device,
            checkpoint_id=result["checkpoint_id"],
        )
        prepared = prepare_ratio4_attention_weights(
            raw_block.attention,
            layer_id=LAYER_ID,
            rank=0,
            world_size=4,
            checkpoint_id=result["checkpoint_id"],
        )
        result["diagnostic_seconds"]["load"] = time.perf_counter() - started

        oracle_state = seed_nonzero_ratio4_state(
            config,
            batch_size=1,
            start_pos=START_POSITION,
            main_ape=prepared.compressor_ape,
            index_ape=prepared.index_compressor_ape,
            seed=args.seed,
            device=device,
        )
        result["seed_payload"] = {
            "raw_amplitude": latent_amplitude_stats(
                oracle_state.raw, rope_dim=config.rope_dim
            ),
            "raw_qdq_error": fp8_qdq_error_stats(
                oracle_state.raw, rope_dim=config.rope_dim
            ),
            "compressed_amplitude": latent_amplitude_stats(
                oracle_state.compressed[:, : START_POSITION // 4],
                rope_dim=config.rope_dim,
            ),
            "compressed_qdq_error": fp8_qdq_error_stats(
                oracle_state.compressed[:, : START_POSITION // 4],
                rope_dim=config.rope_dim,
            ),
        }

        lanes: dict[str, tuple[StaticRatio4KV, Ratio4TorchAttention]] = {}
        for arm, spec in ARM_SPECS.items():
            state = StaticRatio4KV(
                num_local_sequences=1,
                max_seq_len=MAX_SEQ_LEN,
                layer_id=LAYER_ID,
                device=device,
                kv_dtype=spec["kv_dtype"],
                indexer_dtype=spec["indexer_dtype"],
            )
            state.seed_decode_payload(
                START_POSITION,
                raw=oracle_state.raw.clone(),
                compressed=oracle_state.compressed.clone(),
                indexer_kv=oracle_state.indexer_kv.clone(),
                main_kv_state=oracle_state.main_kv.clone(),
                main_score_state=oracle_state.main_score.clone(),
                index_kv_state=oracle_state.index_kv.clone(),
                index_score_state=oracle_state.index_score.clone(),
            )
            lanes[arm] = (state, Ratio4TorchAttention(config, prepared, state))
            result["arms"][arm] = {
                "kv_dtype": spec["kv_dtype"],
                "indexer_dtype": spec["indexer_dtype"],
                "seed_install_latent_rms_rel": (
                    None
                    if spec["kv_dtype"] == "bf16"
                    else {
                        "raw": rms_rel(
                            state.dequantized_latent()[:, :128], oracle_state.raw
                        ),
                        "compressed": rms_rel(
                            state.dequantized_latent()[
                                :, 128 : 128 + START_POSITION // 4
                            ],
                            oracle_state.compressed[:, : START_POSITION // 4],
                        ),
                    }
                ),
                "steps": [],
                "boundary_steps": 0,
                "topk_equal_steps": 0,
                "branch_rms_rel_max": 0.0,
                "branch_rms_rel_last": None,
                "accepted": None,
            }

        # teacher-forced identical hidden stream through every lane
        for step in range(args.steps):
            position = START_POSITION + step
            hidden = deterministic_hidden(args.seed + 7_919 * position, device)
            step_records: dict[str, dict[str, Any]] = {}
            reference: dict[str, torch.Tensor] = {}
            for arm, (state, attention) in lanes.items():
                plan = attention.prepare_decode_plan(
                    position, advance_overlap_state=True
                )
                with attention.observe_evidence() as observed:
                    branch = attention.forward_decode_tensor(
                        hidden.clone(), start_pos=position, plan=plan
                    )
                evidence = observed[0]
                if arm == "bf16":
                    reference = {
                        "branch": branch,
                        "sparse_output": evidence.sparse_output,
                        "selected_kv": evidence.selected_kv,
                        "topk": evidence.topk_indices,
                        "compressed_indices": evidence.compressed_indices,
                    }
                    if step == 0:
                        result["seed_payload"]["step0_raw_latent_amplitude"] = (
                            latent_amplitude_stats(
                                evidence.raw_latent, rope_dim=config.rope_dim
                            )
                        )
                    continue
                record = {
                    "position": position,
                    "boundary": bool(plan.boundary),
                    "branch_rms_rel": rms_rel(branch, reference["branch"]),
                    "branch_max_abs": float(
                        (branch.float() - reference["branch"].float())
                        .abs()
                        .max()
                        .item()
                    ),
                    "sparse_output_rms_rel": rms_rel(
                        evidence.sparse_output, reference["sparse_output"]
                    ),
                    "topk_equal": bool(
                        torch.equal(evidence.topk_indices, reference["topk"])
                    ),
                    "compressed_indices_overlap": float(
                        torch.isin(
                            evidence.compressed_indices,
                            reference["compressed_indices"],
                        )
                        .float()
                        .mean()
                        .item()
                    ),
                }
                if record["topk_equal"]:
                    # same gather rows -> selected_kv delta is pure cache
                    # quantization error over the actually-read rows
                    record["selected_kv_rms_rel"] = rms_rel(
                        evidence.selected_kv, reference["selected_kv"]
                    )
                step_records[arm] = record
            for arm, record in step_records.items():
                arm_result = result["arms"][arm]
                arm_result["steps"].append(record)
                arm_result["boundary_steps"] += int(record["boundary"])
                arm_result["topk_equal_steps"] += int(record["topk_equal"])
                arm_result["branch_rms_rel_max"] = max(
                    arm_result["branch_rms_rel_max"], record["branch_rms_rel"]
                )
                arm_result["branch_rms_rel_last"] = record["branch_rms_rel"]

        accepted = True
        for arm, arm_result in result["arms"].items():
            if arm == "bf16":
                arm_result["accepted"] = True
                continue
            requires_exact_topk = ARM_SPECS[arm]["indexer_dtype"] == "bf16"
            arm_ok = bool(
                len(arm_result["steps"]) == args.steps
                and arm_result["branch_rms_rel_max"] <= args.branch_limit
                and (
                    not requires_exact_topk
                    or arm_result["topk_equal_steps"] == args.steps
                )
            )
            arm_result["accepted"] = arm_ok
            accepted = accepted and arm_ok
        result["accepted"] = accepted
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
    result["diagnostic_seconds"]["process"] = time.perf_counter() - started

    # compact per-arm summary for the log
    for arm, arm_result in result.get("arms", {}).items():
        if arm == "bf16" or not isinstance(arm_result, dict):
            continue
        print(
            f"[E0kf] {arm}: branch rms_rel max "
            f"{arm_result.get('branch_rms_rel_max', float('nan')):.5f} "
            f"last {arm_result.get('branch_rms_rel_last')}, topk equal "
            f"{arm_result.get('topk_equal_steps')}/{args.steps}, accepted "
            f"{arm_result.get('accepted')}",
            flush=True,
        )
    write_json(out_dir / "result.json", result)
    print(f"[E0kf] overall: {'PASS' if result['accepted'] else 'FAIL'}", flush=True)
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
