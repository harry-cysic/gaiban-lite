#!/usr/bin/env python3
"""E0df: real-weight single-layer DirectDecodeBlock decode gate (V4-Flash).

Fifth port vertical: the gaiban ``DirectDecodeBlock`` assembly (attention
half-layer + Hyper-Connections + TP4 MoE half-layer) ported to Flash with the
three-way window/ratio-4/ratio-128 attention dispatch.  One block per layer
type is driven through real-weight full-layer decode on TP4 and compared
against the already-verified component oracles.

What gaiban E0d actually compares (clarified before porting): E0d is a
**deterministic dataflow regression** -- it replays a fixed workload and
compares exact SHA-256 digests plus token-stream numeric signatures against a
fixture captured from the *same implementation*
(``semantic_correctness: not_evaluated``; ``NUMERIC_SIGNATURE_ATOL = 2e-6``
only absorbs Marlin prefill nondeterminism).  No such fixture exists for
Flash, and a replay regression would not gate the port's semantics, so this
script instead composes the four verified component oracles (E0ef/E0ff/E0wf
attention lanes, E0cf FP32 MoE, fp32 Hyper-Connections) into a per-stage
teacher-forced block gate:

- ``candidate`` is ``DirectDecodeBlock.forward_decode_tensor`` (black box).
- A second block instance ("composed") built from the same weights runs the
  identical stage decomposition; candidate output must equal the composed
  output **bitwise** (the decode path is deterministic; gaiban E0d's decode
  digests were exact), which binds every compared stage to the black-box
  output.
- Each composed stage is then compared against an oracle for that stage given
  the candidate's own upstream tensors (teacher-forced), so every tolerance
  is a component tolerance carried over unchanged:
  * attention branch + KV state: the layer's verified attention oracle lane
    (E0wf raw-FP32 window / E0ff BF16-control ratio-4 / E0ef raw-FP32
    ratio-128) with that script's ``branch``/``state.*`` limits;
  * HC/RMSNorm-only stages (attn_hidden, after_attention, ffn_hidden,
    block_output): fp32 ``hc_pre``/``hc_post``/``rms_norm`` recomputation --
    same comparison class as the E0ef/E0wf BF16-control-vs-FP32 projection
    stages, so their 0.012 limit (``query_lora``/``raw_latent``) applies;
  * MoE: raw-checkpoint FP32 routed+shared oracle on the candidate's gathered
    ``ffn_hidden`` with E0cf's combined limit 0.03; route IDs must match the
    runtime exactly and route weights are held to E0ff's gate limit 2e-5.

Cases (one per Flash layer type, per the acceptance matrix):
- layer 0: window (ratio 0) + hash gate; block-level prefill-free attention
  prefill (128 tokens) then plan-based decode at 128..130.
- layer 2: ratio-4 + hash gate; E0ff-style seeded state at 8192 then decode
  at 8192..8194 (full index_topk=512 participation).
- layer 3: ratio-128 + learned gate; attention prefill 128 then decode at
  128..130.

Run (titan064):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=4 \
    e0df_block_oracle.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0df
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
import torch.nn.functional as F

from dsv4_direct.attention import (
    Ratio128AttentionConfig,
    Ratio128TorchAttention,
    prepare_attention_weights,
    rms_norm,
)
from dsv4_direct.attention_oracle import (
    init_ratio128_oracle_state,
    oracle_prepare_attention_weights,
    oracle_ratio128_attention_step,
    yarn_rope_table,
)
from dsv4_direct.block import DirectDecodeBlock
from dsv4_direct.block_weights import (
    inspect_replicated_block_contract,
    load_replicated_block_weights,
)
from dsv4_direct.checkpoint import inspect_stage_checkpoint, load_weight_map
from dsv4_direct.hyper_connections import hc_post, hc_pre
from dsv4_direct.moe_forward import (
    dequant_fp8_block,
    dequant_mxfp4,
    gate_forward_with_boundary,
    hash_gate_forward,
)
from dsv4_direct.moe_runtime import TP4MoE, TP4MoEConfig
from dsv4_direct.ops.marlin_moe import ShardReader, load_resident_moe_layer
from dsv4_direct.ratio4_attention import (
    Ratio4AttentionConfig,
    Ratio4TorchAttention,
    prepare_ratio4_attention_weights,
)
from dsv4_direct.ratio4_oracle import (
    OracleRatio4State,
    oracle_prepare_ratio4_bf16_control_weights,
    oracle_ratio4_bf16_control_step,
    seed_nonzero_ratio4_state,
)
from dsv4_direct.static_kv import StaticLayerKV
from dsv4_direct.static_ratio4_kv import StaticRatio4KV
from dsv4_direct.static_window_kv import StaticWindowKV
from dsv4_direct.window_attention import (
    WindowAttentionConfig,
    WindowTorchAttention,
    prepare_window_attention_weights,
)
from dsv4_direct.window_oracle import (
    init_window_oracle_state,
    oracle_prepare_window_attention_weights,
    oracle_window_attention_step,
)


EXPECTED_WORLD = 4
EXPECTED_VOCAB = 129280
# Frozen from the verified per-layer loaders (e0wf/e0ff/e0ef constants and the
# titan064 loader smoke run).
EXPECTED_BLOCK_RESIDENT_BYTES = {
    0: 118_429_528,
    2: 148_336_216,
    3: 120_876_888,
}
EXPECTED_MOE_RESIDENT_BYTES = 861_931_008

# Component tolerances carried over unchanged:
# - window branch/state: E0wf STAGE_RMS_REL_LIMITS ("branch" 0.040,
#   "state.raw" 0.020) -- raw-FP32 oracle lane.
# - ratio-128 branch/state: E0ef STAGE_RMS_REL_LIMITS ("branch" 0.040,
#   "state.raw"/"state.compressed" 0.020) -- raw-FP32 oracle lane.
# - ratio-4 branch/state: E0ff CONTROL_STAGE_RMS_REL_LIMITS ("branch" 0.010,
#   "state.raw" 0.003) -- BF16-control lane (the PASS-deciding lane in E0ff;
#   raw-FP32 independence for this layer type was established there).
# - HC/RMSNorm-only stages: 0.012, the E0ef/E0wf limit class for
#   BF16-executed-vs-FP32-recomputed stages with identical inputs
#   ("query_lora"/"raw_latent").
# - MoE local output vs raw-checkpoint FP32 oracle: E0cf --combined-rms-rel
#   0.03.  Route weights vs fp32 gate recomputation: E0ff HASH_RMS_REL_LIMIT
#   2e-5.
HC_STAGE_LIMIT = 0.012
MOE_COMBINED_LIMIT = 0.03
ROUTE_WEIGHT_LIMIT = 0.00002

CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "layer0_window_hash",
        "layer_id": 0,
        "attn_kind": "window",
        "route_kind": "hash",
        "max_seq_len": 256,
        "prefill_len": 128,
        "decode_positions": (128, 129, 130),
        "branch_limit": 0.040,
        "state_limits": {"state.raw": 0.020},
        "oracle_lane": "e0wf_window_raw_fp32",
    },
    {
        "name": "layer2_ratio4_hash",
        "layer_id": 2,
        "attn_kind": "ratio4",
        "route_kind": "hash",
        "max_seq_len": 8448,
        "seed_start_pos": 8192,
        "decode_positions": (8192, 8193, 8194),
        "branch_limit": 0.010,
        # E0ff CONTROL_STAGE_RMS_REL_LIMITS state stages, unchanged.  E0ff
        # gates candidate-vs-control state parity through these metrics plus
        # an exact finite-mask check on the score states (its
        # _score_state_metric), NOT through full-payload digest equality;
        # the digest is recorded as a diagnostic only (the first titan064 run
        # showed one bf16-rounding-level digest divergence on rank 1 at the
        # third step while every state metric passed).
        "state_limits": {
            "state.raw": 0.003,
            "state.compressed": 0.003,
            "state.indexer_kv": 0.003,
            "state.main_kv": 0.00002,
            "state.index_kv": 0.00002,
        },
        "score_state_limits": {
            "state.main_score": 0.00002,
            "state.index_score": 0.00002,
        },
        "oracle_lane": "e0ff_ratio4_bf16_control",
    },
    {
        "name": "layer3_ratio128_learned",
        "layer_id": 3,
        "attn_kind": "ratio128",
        "route_kind": "learned",
        "max_seq_len": 256,
        "prefill_len": 128,
        "decode_positions": (128, 129, 130),
        "branch_limit": 0.040,
        "state_limits": {"state.raw": 0.020, "state.compressed": 0.020},
        "oracle_lane": "e0ef_ratio128_raw_fp32",
    },
)

IMPLEMENTATION_FILES = (
    "e0df_block_oracle.py",
    "dsv4_direct/__init__.py",
    "dsv4_direct/attention.py",
    "dsv4_direct/attention_oracle.py",
    "dsv4_direct/block.py",
    "dsv4_direct/block_weights.py",
    "dsv4_direct/checkpoint.py",
    "dsv4_direct/deterministic_moe_align.py",
    "dsv4_direct/hyper_connections.py",
    "dsv4_direct/model_contract.py",
    "dsv4_direct/moe_forward.py",
    "dsv4_direct/moe_runtime.py",
    "dsv4_direct/ops/marlin_moe.py",
    "dsv4_direct/ratio4_attention.py",
    "dsv4_direct/ratio4_oracle.py",
    "dsv4_direct/static_kv.py",
    "dsv4_direct/static_ratio4_kv.py",
    "dsv4_direct/static_window_kv.py",
    "dsv4_direct/window_attention.py",
    "dsv4_direct/window_oracle.py",
)

SEMANTIC_CONTRACT = {
    "model": "deepseek-v4-flash",
    "geometry": "hidden4096_heads64_headdim512_qlora1024_ogroups8_hc4",
    "layers": [0, 2, 3],
    "candidate": "DirectDecodeBlock.forward_decode_tensor (trace-free decode plan path)",
    "assembly_gate": (
        "candidate output bitwise-equal to the composed stage decomposition; "
        "each stage teacher-forced against its verified component oracle"
    ),
    "attention_oracles": {
        "layer0": "window_oracle raw-checkpoint FP32 (E0wf lane + limits)",
        "layer2": "ratio4_oracle BF16 control (E0ff PASS lane + limits)",
        "layer3": "attention_oracle raw-checkpoint FP32 (E0ef lane + limits)",
    },
    "hc_oracle": "fp32 hc_pre/hc_post/rms_norm recomputation (limit 0.012)",
    "moe_oracle": (
        "raw-checkpoint FP32 routed (MXFP4 dequant) + shared (FP8 dequant) "
        "partials, all-reduced across ranks (E0cf combined limit 0.03)"
    ),
    "measurement_scope": "semantic_correctness_not_performance",
}


# --------------------------------------------------------------------------
# generic helpers (E0wf/E0ff process form)


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


def tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous()
        digest.update(f"{list(value.shape)}|{value.dtype}|".encode("ascii"))
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def deterministic_tensor(
    *, seed: int, shape: tuple[int, ...], device: torch.device, scale: float = 0.02
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32)
    return (value * scale).to(torch.bfloat16).to(device)


def deterministic_token_id(*, seed: int, rank: int, step: int) -> int:
    mixed = (seed * 2654435761 + rank * 1000003 + step * 7919) & ((1 << 63) - 1)
    return mixed % EXPECTED_VOCAB


def tensor_metric(
    observed: torch.Tensor, expected: torch.Tensor, *, declared_limit: float
) -> dict[str, Any]:
    """E0wf tensor_metric, unchanged (row limit = 4x the stage limit)."""

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
        "declared_row_limit": declared_limit * 4.0,
        "rms_abs": None,
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
    row_reference_rms = torch.sqrt(torch.mean(expected_fp32.square(), dim=-1))
    row_rms_rel_max = float(
        (row_rms_abs / row_reference_rms.clamp_min(1e-12)).max().item()
    )
    result.update(
        {
            "rms_abs": rms_abs,
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


def synchronized_local_step(
    name: str, fn: Any, *, device: torch.device, world: int
) -> Any:
    value: Any = None
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
    return value


# --------------------------------------------------------------------------
# state digests (assembly-parity exact checks)


def attention_state_digest(state: Any) -> str:
    if isinstance(state, StaticWindowKV):
        return tensor_digest(state.latent, state._next_position, state._raw_positions)
    if isinstance(state, StaticLayerKV):
        return tensor_digest(
            state.latent,
            state.kv_state,
            state.score_state,
            state._next_position,
            state._raw_positions,
            state._state_positions,
            state._compressed_count,
        )
    if isinstance(state, StaticRatio4KV):
        return tensor_digest(*state._owned_tensors())
    raise TypeError(f"unsupported attention state {type(state)!r}")


def ratio4_state_payload_digest(state: StaticRatio4KV | OracleRatio4State) -> str:
    """E0ff state_payload_digest, unchanged (candidate/control exact parity)."""

    if isinstance(state, StaticRatio4KV):
        values = (
            state.raw,
            state.compressed,
            state.indexer_kv,
            state.main_kv_state,
            state.main_score_state,
            state.index_kv_state,
            state.index_score_state,
        )
        next_position = state.next_position
        compressed_count = int(state._compressed_count[0].item())
    elif isinstance(state, OracleRatio4State):
        values = (
            state.raw,
            state.compressed,
            state.indexer_kv,
            state.main_kv,
            state.main_score,
            state.index_kv,
            state.index_score,
        )
        next_position = state.next_position
        compressed_count = state.compressed_count
    else:
        raise TypeError("unsupported ratio-4 state")
    digest = hashlib.sha256()
    digest.update(tensor_digest(*values).encode("ascii"))
    digest.update(f"{next_position}|{compressed_count}".encode("ascii"))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# fp32 HC oracle stages


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
    """FP32 recomputation of one HC-pre + RMSNorm stage.

    ``hc_pre`` computes internally in fp32 and returns hidden in the residual
    dtype; passing the fp32-cast residual keeps the whole stage fp32.
    """

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


# --------------------------------------------------------------------------
# fp32 MoE oracle (E0cf routed_oracle_row_zero generalized to all rows)


def moe_fp32_oracle_partial(
    *,
    stage_root: Path,
    layer_id: int,
    rank: int,
    local_intermediate: int,
    x_full: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_ids: torch.Tensor,
    shared: Any,
    clamp_limit: float,
    device: torch.device,
) -> torch.Tensor:
    """This rank's FP32 routed+shared partial from raw checkpoint weights.

    Routed experts are dequantized (MXFP4) one at a time for this rank's
    intermediate slice, exactly like E0cf's row-zero oracle but applied to
    every gathered row; the shared expert uses the FP8 block dequant oracle.
    Summing the per-rank partials (all-reduce outside) yields the full
    combined FP32 reference.
    """

    rows, hidden_size = x_full.shape
    if tuple(routing_ids.shape) != tuple(routing_weights.shape) or routing_ids.shape[0] != rows:
        raise ValueError("routing tensors do not match the gathered rows")
    start = rank * local_intermediate
    end = start + local_intermediate
    prefix = f"layers.{layer_id}.ffn.experts"
    x_fp32 = x_full.float()
    output = torch.zeros(rows, hidden_size, dtype=torch.float32, device=device)

    occurrences: dict[int, list[tuple[int, int]]] = {}
    ids_host = routing_ids.cpu().tolist()
    for row, row_ids in enumerate(ids_host):
        for kth, expert_id in enumerate(row_ids):
            occurrences.setdefault(int(expert_id), []).append((row, kth))

    weight_map, _ = load_weight_map(stage_root)
    with ShardReader(stage_root, weight_map) as handle:
        for expert_id in sorted(occurrences):
            expert = f"{prefix}.{expert_id}"
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


# --------------------------------------------------------------------------
# per-layer attention lane (candidate x2 + oracle state/step)


class AttentionLane:
    """Two candidate attention instances plus the layer's verified oracle lane."""

    def __init__(
        self,
        *,
        spec: dict[str, Any],
        model_config: dict[str, Any],
        raw_block: Any,
        rank: int,
        world: int,
        checkpoint_id: str,
        seed: int,
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.device = device
        layer_id = int(spec["layer_id"])
        kind = str(spec["attn_kind"])
        max_seq_len = int(spec["max_seq_len"])
        identity = {
            "layer_id": layer_id,
            "rank": rank,
            "world_size": world,
            "checkpoint_id": checkpoint_id,
        }
        if kind == "window":
            self.config = WindowAttentionConfig.from_model_config(
                model_config, layer_id=layer_id, max_seq_len=max_seq_len
            )
            self.candidate_weights = prepare_window_attention_weights(
                raw_block.attention, **identity
            )
            self.oracle_weights = oracle_prepare_window_attention_weights(
                raw_block.attention
            )
            self.oracle_rope = yarn_rope_table(
                dim=self.config.rope_dim,
                seqlen=self.config.max_seq_len,
                original_seq_len=self.config.original_seq_len,
                base=self.config.rope_theta,
                factor=self.config.rope_factor,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                device=device,
            )
            self.states = tuple(
                StaticWindowKV(
                    num_local_sequences=1,
                    max_seq_len=max_seq_len,
                    layer_id=layer_id,
                    device=device,
                )
                for _ in range(2)
            )
            self.attentions = tuple(
                WindowTorchAttention(
                    self.config,
                    self.candidate_weights,
                    state,
                    nope_quant_mode="qat_intended_e4m3",
                )
                for state in self.states
            )
            self.oracle_state = init_window_oracle_state(
                self.config, batch_size=1, device=device
            )
        elif kind == "ratio128":
            self.config = Ratio128AttentionConfig.from_model_config(
                model_config, layer_id=layer_id, max_seq_len=max_seq_len
            )
            self.candidate_weights = prepare_attention_weights(
                raw_block.attention, **identity
            )
            self.oracle_weights = oracle_prepare_attention_weights(
                raw_block.attention
            )
            self.oracle_rope = yarn_rope_table(
                dim=self.config.rope_dim,
                seqlen=self.config.max_seq_len,
                original_seq_len=self.config.original_seq_len,
                base=self.config.rope_theta,
                factor=self.config.rope_factor,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                device=device,
            )
            self.states = tuple(
                StaticLayerKV(
                    num_local_sequences=1,
                    max_seq_len=max_seq_len,
                    layer_id=layer_id,
                    device=device,
                )
                for _ in range(2)
            )
            self.attentions = tuple(
                Ratio128TorchAttention(
                    self.config,
                    self.candidate_weights,
                    state,
                    nope_quant_mode="qat_intended_e4m3",
                )
                for state in self.states
            )
            self.oracle_state = init_ratio128_oracle_state(
                self.config, batch_size=1, device=device
            )
        elif kind == "ratio4":
            self.config = Ratio4AttentionConfig.from_model_config(
                model_config, layer_id=layer_id, max_seq_len=max_seq_len
            )
            self.candidate_weights = prepare_ratio4_attention_weights(
                raw_block.attention, **identity
            )
            self.oracle_weights = oracle_prepare_ratio4_bf16_control_weights(
                raw_block.attention
            )
            self.oracle_rope = None
            start_pos = int(spec["seed_start_pos"])
            self.oracle_state = seed_nonzero_ratio4_state(
                self.config,
                batch_size=1,
                start_pos=start_pos,
                main_ape=self.oracle_weights.compressor_ape,
                index_ape=self.oracle_weights.index_compressor_ape,
                seed=seed,
                device=device,
            )
            self.states = tuple(
                StaticRatio4KV(
                    num_local_sequences=1,
                    max_seq_len=max_seq_len,
                    layer_id=layer_id,
                    device=device,
                )
                for _ in range(2)
            )
            for state in self.states:
                state.seed_decode_payload(
                    self.oracle_state.next_position,
                    raw=self.oracle_state.raw.clone(),
                    compressed=self.oracle_state.compressed.clone(),
                    indexer_kv=self.oracle_state.indexer_kv.clone(),
                    main_kv_state=self.oracle_state.main_kv.clone(),
                    main_score_state=self.oracle_state.main_score.clone(),
                    index_kv_state=self.oracle_state.index_kv.clone(),
                    index_score_state=self.oracle_state.index_score.clone(),
                )
            self.attentions = tuple(
                Ratio4TorchAttention(self.config, self.candidate_weights, state)
                for state in self.states
            )
            for state in self.states:
                if ratio4_state_payload_digest(state) != ratio4_state_payload_digest(
                    self.oracle_state
                ):
                    raise AssertionError(
                        "seeded candidate state differs from the control payload"
                    )
        else:
            raise ValueError(f"unsupported attention kind {kind!r}")
        self.kind = kind

    def prefill(self, hidden: torch.Tensor) -> dict[str, Any]:
        """Prefill both candidate states and the oracle state with one hidden."""

        if self.kind == "ratio4":
            raise AssertionError("ratio-4 lane is seeded, not prefilled")
        branches = []
        for attention in self.attentions:
            local_hidden = hidden.clone()
            branch, _ = attention(local_hidden, start_pos=0)
            branches.append(branch)
        oracle_step = self.oracle_step(hidden, start_pos=0)
        self.oracle_state = oracle_step.state
        return {
            "candidate_branch": branches[0],
            "candidate_branches_equal": bool(torch.equal(branches[0], branches[1])),
            "oracle_branch": oracle_step.trace.branch,
        }

    def oracle_step(self, hidden: torch.Tensor, *, start_pos: int) -> Any:
        if self.kind == "window":
            return oracle_window_attention_step(
                self.config,
                self.oracle_weights,
                hidden,
                start_pos=start_pos,
                state=self.oracle_state,
                rope_table=self.oracle_rope,
            )
        if self.kind == "ratio128":
            return oracle_ratio128_attention_step(
                self.config,
                self.oracle_weights,
                hidden,
                start_pos=start_pos,
                state=self.oracle_state,
                rope_table=self.oracle_rope,
            )
        return oracle_ratio4_bf16_control_step(
            self.config,
            self.oracle_weights,
            hidden,
            start_pos=start_pos,
            state=self.oracle_state,
        )

    def prepare_plans(self, start_pos: int) -> tuple[Any, Any]:
        if self.kind == "ratio4":
            return tuple(
                attention.prepare_decode_plan(start_pos, advance_overlap_state=True)
                for attention in self.attentions
            )
        return tuple(
            attention.prepare_decode_plan(start_pos)
            for attention in self.attentions
        )

    def state_metric_pairs(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        candidate = self.states[0]
        oracle = self.oracle_state
        pairs = {"state.raw": (candidate.raw, oracle.raw)}
        if self.kind == "ratio128":
            pairs["state.compressed"] = (candidate.compressed, oracle.compressed)
        if self.kind == "ratio4":
            pairs.update(
                {
                    "state.compressed": (candidate.compressed, oracle.compressed),
                    "state.indexer_kv": (candidate.indexer_kv, oracle.indexer_kv),
                    "state.main_kv": (candidate.main_kv_state, oracle.main_kv),
                    "state.index_kv": (candidate.index_kv_state, oracle.index_kv),
                }
            )
        return pairs

    def score_state_pairs(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Ratio-4 score states carry ``-inf`` in unfilled overlap slots.

        E0ff compares them via an exact finite-mask check plus a metric over
        the finite elements (``_score_state_metric``); mirrored here.
        """

        if self.kind != "ratio4":
            return {}
        candidate = self.states[0]
        oracle = self.oracle_state
        return {
            "state.main_score": (candidate.main_score_state, oracle.main_score),
            "state.index_score": (candidate.index_score_state, oracle.index_score),
        }


# --------------------------------------------------------------------------
# one case


def run_case(
    *,
    spec: dict[str, Any],
    model_config: dict[str, Any],
    stage_root: Path,
    rank: int,
    world: int,
    checkpoint_id: str,
    seed: int,
    device: torch.device,
    progress_every: int,
) -> dict[str, Any]:
    layer_id = int(spec["layer_id"])
    route_kind = str(spec["route_kind"])
    branch_limit = float(spec["branch_limit"])
    hidden_size = int(model_config["hidden_size"])
    clamp_limit = float(model_config["swiglu_limit"])
    route_scale = float(model_config["routed_scaling_factor"])
    topk = int(model_config["num_experts_per_tok"])
    local_intermediate = int(model_config["moe_intermediate_size"]) // world
    result: dict[str, Any] = {
        "layer_id": layer_id,
        "route_kind": route_kind,
        "oracle_lane": spec["oracle_lane"],
        "decode_positions": list(spec["decode_positions"]),
        "inputs": {},
        "stage_metrics": {},
        "exact_checks": {},
        "diagnostics": {},
        "errors": [],
        "accepted": False,
    }
    stage_metrics: dict[str, dict[str, Any]] = result["stage_metrics"]
    exact_checks: dict[str, bool] = result["exact_checks"]

    def load_all() -> tuple[Any, Any]:
        raw_block = load_replicated_block_weights(
            stage_root=stage_root,
            rank=rank,
            world_size=world,
            layer_id=layer_id,
            device=device,
            checkpoint_id=checkpoint_id,
        )
        if raw_block.resident_bytes != EXPECTED_BLOCK_RESIDENT_BYTES[layer_id]:
            raise ValueError(
                f"layer-{layer_id} block resident bytes {raw_block.resident_bytes} "
                f"!= {EXPECTED_BLOCK_RESIDENT_BYTES[layer_id]}"
            )
        if raw_block.gate.route_kind != route_kind:
            raise ValueError(
                f"layer-{layer_id} gate route {raw_block.gate.route_kind} != {route_kind}"
            )
        moe_resident = load_resident_moe_layer(
            stage_root=stage_root,
            layer_id=layer_id,
            rank=rank,
            world_size=world,
            hidden_size=hidden_size,
            intermediate_size=int(model_config["moe_intermediate_size"]),
            n_experts=int(model_config["n_routed_experts"]),
            device=device,
            progress_every=progress_every,
            progress=(
                (lambda message: print(message, flush=True)) if rank == 0 else None
            ),
            checkpoint_id=checkpoint_id,
        )
        if moe_resident.resident_bytes != EXPECTED_MOE_RESIDENT_BYTES:
            raise ValueError(
                f"layer-{layer_id} MoE resident bytes {moe_resident.resident_bytes} "
                f"!= {EXPECTED_MOE_RESIDENT_BYTES}"
            )
        return raw_block, moe_resident

    raw_block, moe_resident = synchronized_local_step(
        f"load layer-{layer_id} weights", load_all, device=device, world=world
    )

    def build_runtime() -> tuple[AttentionLane, TP4MoE, DirectDecodeBlock, DirectDecodeBlock]:
        lane = AttentionLane(
            spec=spec,
            model_config=model_config,
            raw_block=raw_block,
            rank=rank,
            world=world,
            checkpoint_id=checkpoint_id,
            seed=seed + 777,
            device=device,
        )
        moe = TP4MoE(
            config=TP4MoEConfig(
                hidden_size=hidden_size,
                intermediate_size=int(model_config["moe_intermediate_size"]),
                experts=int(model_config["n_routed_experts"]),
                topk=topk,
                route_scale=route_scale,
                clamp_limit=clamp_limit,
                world_size=world,
            ),
            resident=moe_resident,
            gate=raw_block.gate,
            rank=rank,
            device=device,
            global_row_shapes=(world,),
        )
        block_candidate = DirectDecodeBlock(
            weights=raw_block,
            attention=lane.attentions[0],
            moe=moe,
            norm_eps=float(model_config["rms_norm_eps"]),
            sinkhorn_iters=int(model_config["hc_sinkhorn_iters"]),
            hc_eps=float(model_config["hc_eps"]),
        )
        block_composed = DirectDecodeBlock(
            weights=raw_block,
            attention=lane.attentions[1],
            moe=moe,
            norm_eps=float(model_config["rms_norm_eps"]),
            sinkhorn_iters=int(model_config["hc_sinkhorn_iters"]),
            hc_eps=float(model_config["hc_eps"]),
        )
        return lane, moe, block_candidate, block_composed

    lane, moe, block_candidate, block_composed = synchronized_local_step(
        f"build layer-{layer_id} runtime", build_runtime, device=device, world=world
    )
    hc = raw_block.hyper_connection
    norm_eps = float(model_config["rms_norm_eps"])
    sinkhorn_iters = int(model_config["hc_sinkhorn_iters"])
    hc_eps = float(model_config["hc_eps"])

    # ---- state preparation -------------------------------------------------
    if lane.kind != "ratio4":
        prefill_len = int(spec["prefill_len"])
        prefill_hidden = deterministic_tensor(
            seed=seed + rank * 100_003,
            shape=(1, prefill_len, hidden_size),
            device=device,
        )
        result["inputs"]["prefill"] = {
            "shape": list(prefill_hidden.shape),
            "sha256": tensor_sha256(prefill_hidden),
        }
        prefill = synchronized_local_step(
            f"layer-{layer_id} attention prefill",
            lambda: lane.prefill(prefill_hidden),
            device=device,
            world=world,
        )
        exact_checks["prefill.candidate_pair_branch_equal"] = prefill[
            "candidate_branches_equal"
        ]
        exact_checks["prefill.candidate_pair_state_equal"] = (
            attention_state_digest(lane.states[0])
            == attention_state_digest(lane.states[1])
        )
        stage_metrics["prefill.branch"] = tensor_metric(
            prefill["candidate_branch"],
            prefill["oracle_branch"],
            declared_limit=branch_limit,
        )
        for name, (candidate_value, oracle_value) in lane.state_metric_pairs().items():
            stage_metrics[f"prefill.{name}"] = tensor_metric(
                candidate_value,
                oracle_value,
                declared_limit=float(spec["state_limits"][name]),
            )
    else:
        exact_checks["seed.candidate_pair_state_equal"] = (
            attention_state_digest(lane.states[0])
            == attention_state_digest(lane.states[1])
        )
        exact_checks["seed.candidate_matches_control"] = (
            ratio4_state_payload_digest(lane.states[0])
            == ratio4_state_payload_digest(lane.oracle_state)
        )

    # ---- decode steps ------------------------------------------------------
    for step_index, position in enumerate(spec["decode_positions"]):
        phase = f"decode_pos{position:05d}"
        residual = deterministic_tensor(
            seed=seed + rank * 100_003 + 50_000 + step_index * 977,
            shape=(1, 1, 4, hidden_size),
            device=device,
        )
        input_ids = None
        if route_kind == "hash":
            token_id = deterministic_token_id(
                seed=seed, rank=rank, step=step_index
            )
            input_ids = torch.tensor(
                [[token_id]], dtype=torch.int64, device=device
            )
        result["inputs"][phase] = {
            "residual_sha256": tensor_sha256(residual),
            "token_id": None if input_ids is None else int(input_ids.item()),
        }

        plan_candidate, plan_composed = synchronized_local_step(
            f"{phase} prepare plans",
            lambda: lane.prepare_plans(position),
            device=device,
            world=world,
        )

        # candidate: the acceptance surface, black box.
        block_arguments: dict[str, Any] = {
            "start_pos": position,
            "attention_plan": plan_candidate,
        }
        if input_ids is not None:
            block_arguments["input_ids_local"] = input_ids
        with moe.observe_route_tensors(capture_local_input=True) as candidate_routes:
            candidate_output = block_candidate.forward_decode_tensor(
                residual, **block_arguments
            )

        # composed: identical stage decomposition on the twin block.
        def composed_stages() -> dict[str, torch.Tensor]:
            attn_hidden, attn_post, attn_comb = block_composed._hc_pre(
                residual, branch="attn"
            )
            attn_hidden = rms_norm(
                attn_hidden, raw_block.attn_norm, eps=norm_eps
            )
            branch = block_composed.attention.forward_decode_tensor(
                attn_hidden, start_pos=position, plan=plan_composed
            )
            after_attention = hc_post(branch, residual, attn_post, attn_comb)
            ffn_hidden, ffn_post, ffn_comb = block_composed.prepare_ffn(
                after_attention
            )
            return {
                "attn_hidden": attn_hidden,
                "attn_post": attn_post,
                "attn_comb": attn_comb,
                "branch": branch,
                "after_attention": after_attention,
                "ffn_hidden": ffn_hidden,
                "ffn_post": ffn_post,
                "ffn_comb": ffn_comb,
            }

        stages = synchronized_local_step(
            f"{phase} composed stages", composed_stages, device=device, world=world
        )
        moe_arguments: dict[str, Any] = {}
        if input_ids is not None:
            moe_arguments["input_ids_local"] = input_ids
        with moe.observe_route_tensors(capture_local_input=True) as composed_routes:
            moe_output = moe.forward_tensor(stages["ffn_hidden"], **moe_arguments)
        composed_output = synchronized_local_step(
            f"{phase} composed hc-post",
            lambda: hc_post(
                moe_output,
                stages["after_attention"],
                stages["ffn_post"],
                stages["ffn_comb"],
            ),
            device=device,
            world=world,
        )

        # assembly parity: black box == composition, bitwise.
        candidate_route = candidate_routes[0]
        composed_route = composed_routes[0]
        exact_checks[f"{phase}.candidate_equals_composed"] = bool(
            torch.equal(candidate_output, composed_output)
        )
        exact_checks[f"{phase}.ffn_hidden_capture_equal"] = bool(
            candidate_route.local_input is not None
            and torch.equal(
                candidate_route.local_input,
                stages["ffn_hidden"].reshape(-1, hidden_size),
            )
        )
        exact_checks[f"{phase}.route_ids_pair_equal"] = bool(
            torch.equal(candidate_route.ids, composed_route.ids)
        )
        exact_checks[f"{phase}.attention_state_pair_equal"] = (
            attention_state_digest(lane.states[0])
            == attention_state_digest(lane.states[1])
        )

        # oracle lane 1: fp32 HC recomputation of the attention-pre stage.
        oracle_attn_hidden, oracle_attn_post, oracle_attn_comb = oracle_hc_pre_norm(
            residual,
            hc_fn=hc.attn_fn,
            hc_scale=hc.attn_scale,
            hc_base=hc.attn_base,
            norm_weight=raw_block.attn_norm,
            norm_eps=norm_eps,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=hc_eps,
        )
        stage_metrics[f"{phase}.attn_hidden"] = tensor_metric(
            stages["attn_hidden"], oracle_attn_hidden, declared_limit=HC_STAGE_LIMIT
        )
        exact_checks[f"{phase}.hc_attn_post_comb_exact"] = bool(
            torch.equal(stages["attn_post"], oracle_attn_post)
            and torch.equal(stages["attn_comb"], oracle_attn_comb)
        )

        # oracle lane 2: verified attention oracle, teacher-forced on the
        # candidate's own attention hidden (bit-identical inputs, exactly the
        # component-gate calibration).
        oracle_hidden = stages["attn_hidden"].clone()
        oracle_step = synchronized_local_step(
            f"{phase} attention oracle",
            lambda: lane.oracle_step(oracle_hidden, start_pos=position),
            device=device,
            world=world,
        )
        lane.oracle_state = oracle_step.state
        stage_metrics[f"{phase}.branch"] = tensor_metric(
            stages["branch"], oracle_step.trace.branch, declared_limit=branch_limit
        )
        for name, (candidate_value, oracle_value) in lane.state_metric_pairs().items():
            stage_metrics[f"{phase}.{name}"] = tensor_metric(
                candidate_value,
                oracle_value,
                declared_limit=float(spec["state_limits"][name]),
            )
        for name, (observed, expected) in lane.score_state_pairs().items():
            observed_mask = torch.isfinite(observed)
            expected_mask = torch.isfinite(expected)
            mask_equal = bool(torch.equal(observed_mask, expected_mask))
            exact_checks[f"{phase}.{name}.finite_mask"] = mask_equal
            if mask_equal:
                stage_metrics[f"{phase}.{name}"] = tensor_metric(
                    observed[observed_mask],
                    expected[expected_mask],
                    declared_limit=float(spec["score_state_limits"][name]),
                )
        if lane.kind == "ratio4":
            # Diagnostic only: E0ff gates state parity through the metrics
            # above, not bitwise digest equality (see CASE_SPECS comment).
            result["diagnostics"][f"{phase}.state_digest_matches_control"] = (
                ratio4_state_payload_digest(lane.states[0])
                == ratio4_state_payload_digest(lane.oracle_state)
            )

        # oracle lane 3: fp32 hc_post / hc_pre+norm on candidate tensors.
        oracle_after_attention = hc_post(
            stages["branch"].float(),
            residual.float(),
            stages["attn_post"],
            stages["attn_comb"],
        )
        stage_metrics[f"{phase}.after_attention"] = tensor_metric(
            stages["after_attention"],
            oracle_after_attention,
            declared_limit=HC_STAGE_LIMIT,
        )
        oracle_ffn_hidden, oracle_ffn_post, oracle_ffn_comb = oracle_hc_pre_norm(
            stages["after_attention"],
            hc_fn=hc.ffn_fn,
            hc_scale=hc.ffn_scale,
            hc_base=hc.ffn_base,
            norm_weight=raw_block.ffn_norm,
            norm_eps=norm_eps,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=hc_eps,
        )
        stage_metrics[f"{phase}.ffn_hidden"] = tensor_metric(
            stages["ffn_hidden"], oracle_ffn_hidden, declared_limit=HC_STAGE_LIMIT
        )
        exact_checks[f"{phase}.hc_ffn_post_comb_exact"] = bool(
            torch.equal(stages["ffn_post"], oracle_ffn_post)
            and torch.equal(stages["ffn_comb"], oracle_ffn_comb)
        )

        # oracle lane 4: fp32 MoE on the candidate's gathered ffn hidden.
        local_flat = stages["ffn_hidden"].reshape(-1, hidden_size).contiguous()
        x_full = torch.empty(
            world, hidden_size, dtype=torch.bfloat16, device=device
        )
        dist.all_gather_into_tensor(x_full, local_flat)
        if route_kind == "hash":
            gathered_ids = torch.empty(world, dtype=torch.int64, device=device)
            dist.all_gather_into_tensor(gathered_ids, input_ids.view(-1))
            gate_result = hash_gate_forward(
                x_full,
                raw_block.gate.weight,
                raw_block.gate.tid2eid,
                gathered_ids,
                route_scale=route_scale,
            )
            oracle_route_ids = gate_result.routing_ids
            oracle_route_weights = gate_result.routing_weights
        else:
            gate_result = gate_forward_with_boundary(
                x_full,
                raw_block.gate.weight,
                raw_block.gate.bias,
                topk=topk,
                route_scale=route_scale,
            )
            oracle_route_ids = gate_result.routing_ids
            oracle_route_weights = gate_result.routing_weights
            result["diagnostics"][f"{phase}.route_margin_min"] = float(
                gate_result.margin.min().item()
            )
        exact_checks[f"{phase}.route_ids_match_oracle"] = bool(
            torch.equal(candidate_route.ids, oracle_route_ids)
        )
        stage_metrics[f"{phase}.route_weights"] = tensor_metric(
            candidate_route.weights,
            oracle_route_weights,
            declared_limit=ROUTE_WEIGHT_LIMIT,
        )
        oracle_partial = synchronized_local_step(
            f"{phase} fp32 MoE oracle partial",
            lambda: moe_fp32_oracle_partial(
                stage_root=stage_root,
                layer_id=layer_id,
                rank=rank,
                local_intermediate=local_intermediate,
                x_full=x_full,
                routing_weights=oracle_route_weights,
                routing_ids=oracle_route_ids,
                shared=moe_resident.shared,
                clamp_limit=clamp_limit,
                device=device,
            ),
            device=device,
            world=world,
        )
        dist.all_reduce(oracle_partial, op=dist.ReduceOp.SUM)
        oracle_moe_local = oracle_partial[rank : rank + 1]
        stage_metrics[f"{phase}.moe_local"] = tensor_metric(
            moe_output.reshape(-1, hidden_size),
            oracle_moe_local,
            declared_limit=MOE_COMBINED_LIMIT,
        )

        # oracle lane 5: fp32 final hc_post on candidate tensors.
        oracle_block_output = hc_post(
            moe_output.reshape(1, 1, hidden_size).float(),
            stages["after_attention"].float(),
            stages["ffn_post"],
            stages["ffn_comb"],
        )
        stage_metrics[f"{phase}.block_output"] = tensor_metric(
            candidate_output, oracle_block_output, declared_limit=HC_STAGE_LIMIT
        )

    result["accepted"] = bool(
        all(metric["accepted"] for metric in stage_metrics.values())
        and all(exact_checks.values())
        and len(stage_metrics) > 0
        and len(exact_checks) > 0
    )

    # release per-case memory before the next layer loads.
    del lane, moe, block_candidate, block_composed, raw_block, moe_resident
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------


def aggregate_results(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for spec in CASE_SPECS:
        name = spec["name"]
        rank_cases = [rank["cases"][name] for rank in ranks]
        metric_names = sorted(rank_cases[0].get("stage_metrics", {}))
        exact_names = sorted(rank_cases[0].get("exact_checks", {}))
        cases[name] = {
            "accepted_ranks": [
                rank["rank"]
                for rank in ranks
                if rank["cases"][name].get("accepted") is True
            ],
            "exact_checks": {
                check: all(
                    case.get("exact_checks", {}).get(check) is True
                    for case in rank_cases
                )
                for check in exact_names
            },
            "stage_metrics": {
                metric: {
                    "rms_rel_max": max(
                        float(case["stage_metrics"][metric]["rms_rel"])
                        for case in rank_cases
                        if case["stage_metrics"][metric]["rms_rel"] is not None
                    )
                    if all(
                        case["stage_metrics"][metric]["rms_rel"] is not None
                        for case in rank_cases
                    )
                    else None,
                    "declared_limit": float(
                        rank_cases[0]["stage_metrics"][metric]["declared_limit"]
                    ),
                    "accepted": all(
                        case["stage_metrics"][metric]["accepted"] is True
                        for case in rank_cases
                    ),
                }
                for metric in metric_names
            },
        }
    return {"cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--progress-every", type=int, default=64)
    args = parser.parse_args()

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
    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    implementation_id = implementation_sha256(source_root)
    workload = {
        "local_batch": 1,
        "seed": args.seed,
        "cases": [
            {
                "name": spec["name"],
                "layer_id": spec["layer_id"],
                "decode_positions": list(spec["decode_positions"]),
            }
            for spec in CASE_SPECS
        ],
        "input_distribution": "CPU FP32 normal * 0.02, cast BF16, deterministic per rank",
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0df-block-oracle",
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
        "cases": {},
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }

    started = time.perf_counter()
    try:
        if world != EXPECTED_WORLD:
            raise ValueError(f"E0df requires TP4, got world={world}")
        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                layer_ids = [int(spec["layer_id"]) for spec in CASE_SPECS]
                checkpoint = inspect_stage_checkpoint(stage_root, layer_ids, world)
                if not checkpoint["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {checkpoint['errors'][:3]}"
                    )
                contracts = {}
                for layer_id in layer_ids:
                    block_contract = inspect_replicated_block_contract(
                        stage_root, layer_id=layer_id, rank=0, world_size=world
                    )
                    if not block_contract["ok"]:
                        raise ValueError(
                            f"layer-{layer_id} block contract failed: "
                            f"{block_contract['errors'][:3]}"
                        )
                    contracts[str(layer_id)] = block_contract["contract_id"]
                envelope_holder[0] = {
                    "ok": True,
                    "config": config_payload,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "block_contract_ids": contracts,
                }
            except Exception:
                envelope_holder[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        if not envelope["ok"]:
            raise ValueError(f"rank-0 preflight failed:\n{envelope['error']}")
        result["checkpoint_id"] = envelope["checkpoint_id"]
        result["block_contract_ids"] = envelope["block_contract_ids"]
        model_config = envelope["config"]

        for case_index, spec in enumerate(CASE_SPECS):
            case_started = time.perf_counter()
            case_name = spec["name"]
            if rank == 0:
                print(f"[E0df] case {case_name} starting", flush=True)
            result["cases"][case_name] = run_case(
                spec=spec,
                model_config=model_config,
                stage_root=stage_root,
                rank=rank,
                world=world,
                checkpoint_id=result["checkpoint_id"],
                seed=args.seed + case_index * 1_000_003,
                device=device,
                progress_every=args.progress_every,
            )
            result["diagnostic_seconds"][f"case.{case_name}"] = (
                time.perf_counter() - case_started
            )
            if rank == 0:
                status = "PASS" if result["cases"][case_name]["accepted"] else "FAIL"
                print(f"[E0df] case {case_name}: {status}", flush=True)
        result["accepted"] = bool(
            set(result["cases"]) == {spec["name"] for spec in CASE_SPECS}
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
    summary_accepted = False
    if rank == 0:
        try:
            rank_results = sorted(gathered, key=lambda value: value["rank"])
            identities_match = all(
                value["checkpoint_id"] == rank_results[0]["checkpoint_id"]
                and value["implementation_sha256"]
                == rank_results[0]["implementation_sha256"]
                and value["workload"] == rank_results[0]["workload"]
                for value in rank_results
            )
            aggregates = aggregate_results(rank_results)
            summary_accepted = bool(
                identities_match
                and all(value["accepted"] for value in rank_results)
                and all(
                    len(case["accepted_ranks"]) == EXPECTED_WORLD
                    for case in aggregates["cases"].values()
                )
            )
            summary = {
                "schema_version": 1,
                "experiment": "E0df-block-oracle",
                "measurement_class": "semantic_correctness_gate",
                "accepted": summary_accepted,
                "semantic_contract": SEMANTIC_CONTRACT,
                "checkpoint_id": rank_results[0]["checkpoint_id"],
                "block_contract_ids": rank_results[0].get("block_contract_ids"),
                "implementation_sha256": implementation_id,
                "world": world,
                "workload": workload,
                "identity_checks": {"all_ranks_match": identities_match},
                "aggregates": aggregates,
                "ranks": rank_results,
                "errors": [
                    error
                    for value in rank_results
                    for error in value["errors"]
                    + [
                        case_error
                        for case in value["cases"].values()
                        for case_error in case.get("errors", [])
                    ]
                ],
            }
            write_json(out_dir / "summary.json", summary)
            print(
                f"{'PASS' if summary_accepted else 'FAIL'} E0df DirectDecodeBlock "
                "decode gate (layers 0/2/3)",
                flush=True,
            )
            for case_name, case in aggregates["cases"].items():
                worst_name, worst = None, None
                for metric_name, metric in case["stage_metrics"].items():
                    if metric["rms_rel_max"] is None:
                        continue
                    ratio = metric["rms_rel_max"] / max(metric["declared_limit"], 1e-12)
                    if worst is None or ratio > worst:
                        worst, worst_name = ratio, metric_name
                exact_ok = all(case["exact_checks"].values())
                case_ok = (
                    len(case["accepted_ranks"]) == EXPECTED_WORLD and exact_ok
                )
                detail = ""
                if worst_name is not None:
                    metric = case["stage_metrics"][worst_name]
                    detail = (
                        f"; worst {worst_name}: rms_rel {metric['rms_rel_max']:.3e}"
                        f" <= {metric['declared_limit']:.3g}"
                    )
                print(
                    f"  {case_name}: {'PASS' if case_ok else 'FAIL'}"
                    f" (exact_checks {'ok' if exact_ok else 'FAILED'}{detail})",
                    flush=True,
                )
        except Exception:
            summary_accepted = False
            print("FAIL E0df summary aggregation:\n" + traceback.format_exc(), flush=True)

    accepted_holder: list[Any] = [summary_accepted if rank == 0 else None]
    dist.broadcast_object_list(accepted_holder, src=0)
    dist.destroy_process_group()
    return 0 if accepted_holder[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
