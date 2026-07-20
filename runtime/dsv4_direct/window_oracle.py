"""Independent CPU-friendly pure sliding-window attention mathematics.

Semantic oracle for Flash ``compress_ratio == 0`` layers (L0/L1).  It reuses
the generic FP32 primitives of :mod:`dsv4_direct.attention_oracle` (RoPE
table, real-pair rotation, RMSNorm, E4M3/UE8M0 QDQ, scalar-loop sparse
attention, window indices, FP8 block dequant) and deliberately does not
import the candidate ``dsv4_direct.window_attention`` module.

Reference semantics (``/home/harry/gaiban/references/inference/model.py``):
- ``Attention.__init__`` :466-471: no compressor, no indexer for ratio 0.
- :473: ``kv_cache_size == window_size`` (128 rows only).
- :477-481: RoPE table built with ``original_seq_len = 0`` and the base
  ``args.rope_theta`` (10000) -- YaRN disabled.  ``yarn_rope_table`` already
  degrades identically: its correction branch is guarded by
  ``original_seq_len > 0`` exactly like ``precompute_freqs_cis``
  (model.py:221), so factor/beta inputs are inert.
- ``forward`` :496-543 with the ratio-0 branchs: q/kv projections, window
  top-k only (:507, :515; :508-514 skipped), prefill ring write :518-523 and
  dense attention over the full prefill latent :528, decode ring write :530
  and attention over the ring :533, inverse RoPE :534, grouped wo_a einsum
  and wo_b :537-542.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .attention_oracle import (
    RopeTable,
    _config_value,
    _linear_bf16,
    e4m3_ue8m0_qdq,
    oracle_apply_rope,
    oracle_dequant_fp8_block,
    oracle_rms_norm,
    oracle_sparse_attention,
    oracle_window_topk_indices,
    yarn_rope_table,
)


WINDOW = 128


@dataclass
class OracleWindowAttentionWeights:
    """FP32 oracle view of raw checkpoint window-layer attention weights."""

    attn_sink: torch.Tensor
    wq_a: torch.Tensor
    q_norm: torch.Tensor
    wq_b: torch.Tensor
    wkv: torch.Tensor
    kv_norm: torch.Tensor
    wo_a: torch.Tensor
    wo_b: torch.Tensor


@dataclass(frozen=True)
class WindowOracleState:
    """Functional snapshot of the 128-row raw ring."""

    raw: torch.Tensor
    next_position: int
    max_seq_len: int

    def clone(self) -> "WindowOracleState":
        return WindowOracleState(
            raw=self.raw.clone(),
            next_position=self.next_position,
            max_seq_len=self.max_seq_len,
        )


@dataclass(frozen=True)
class WindowAttentionOracleTrace:
    """Numerical boundaries exposed for independent candidate attribution."""

    query_lora: torch.Tensor
    query: torch.Tensor
    raw_latent: torch.Tensor
    attention_kv: torch.Tensor
    topk_indices: torch.Tensor
    sparse_output: torch.Tensor
    inverse_rotated: torch.Tensor
    output_lora: torch.Tensor
    branch: torch.Tensor


@dataclass(frozen=True)
class WindowAttentionOracleStep:
    trace: WindowAttentionOracleTrace
    state: WindowOracleState


def oracle_prepare_window_attention_weights(
    weights: Any,
) -> OracleWindowAttentionWeights:
    """Create an independent FP32 view of resident raw checkpoint weights."""

    if isinstance(weights, OracleWindowAttentionWeights):
        return weights

    def dequant(name: str) -> torch.Tensor:
        try:
            linear = getattr(weights, name)
            matrix = linear.weight
            scales = linear.scale
        except AttributeError as exc:
            raise TypeError(
                f"attention weights are missing quantized linear {name}"
            ) from exc
        return oracle_dequant_fp8_block(matrix, scales)

    try:
        # Reference model.py:466-471: ratio-0 layers own no compressor and no
        # indexer; refuse to build a window oracle over a layer that has them.
        if getattr(weights, "compressor") is not None:
            raise TypeError("window oracle rejects layers with compressor weights")
        if getattr(weights, "indexer") is not None:
            raise TypeError("window oracle rejects layers with indexer weights")
        result = OracleWindowAttentionWeights(
            attn_sink=weights.attn_sink.to(torch.float32).contiguous().clone(),
            wq_a=dequant("wq_a"),
            q_norm=weights.q_norm.to(torch.float32).contiguous().clone(),
            wq_b=dequant("wq_b"),
            wkv=dequant("wkv"),
            kv_norm=weights.kv_norm.to(torch.float32).contiguous().clone(),
            wo_a=dequant("wo_a"),
            wo_b=dequant("wo_b"),
        )
    except AttributeError as exc:
        raise TypeError(
            "resident attention weights do not satisfy the raw contract"
        ) from exc
    return result


def _window_oracle_dimensions(config: Any) -> dict[str, int | float]:
    integer_names = (
        "hidden_size",
        "num_heads",
        "head_dim",
        "rope_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "beta_fast",
        "beta_slow",
        "original_seq_len",
        "max_seq_len",
    )
    values: dict[str, int | float] = {
        name: int(_config_value(config, name)) for name in integer_names
    }
    for name in ("norm_eps", "rope_theta", "rope_factor"):
        values[name] = float(_config_value(config, name))

    positive = (
        "hidden_size",
        "num_heads",
        "head_dim",
        "rope_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "beta_fast",
        "beta_slow",
        "max_seq_len",
    )
    if any(int(values[name]) <= 0 for name in positive):
        raise ValueError("oracle attention dimensions must be positive")
    # Fail-closed no-YaRN rule for pure sliding-window layers
    # (model.py:477-479): original_seq_len must be exactly zero.
    if int(values["original_seq_len"]) != 0:
        raise ValueError(
            "window oracle requires original_seq_len == 0 (YaRN disabled)"
        )
    if int(values["rope_dim"]) % 2:
        raise ValueError("rope_dim must be even")
    nope_dim = int(values["head_dim"]) - int(values["rope_dim"])
    if nope_dim <= 0 or nope_dim % 64:
        raise ValueError("NoPE width must be a positive multiple of 64")
    total_head_dim = int(values["num_heads"]) * int(values["head_dim"])
    if total_head_dim % int(values["o_groups"]):
        raise ValueError("heads times head_dim must divide evenly into output groups")
    if int(values["max_seq_len"]) < WINDOW:
        raise ValueError(f"max_seq_len must be at least the window size {WINDOW}")
    if any(
        not math.isfinite(float(values[name])) or float(values[name]) <= 0
        for name in ("norm_eps", "rope_theta", "rope_factor")
    ):
        raise ValueError(
            "oracle attention numerical constants must be positive and finite"
        )
    return values


def _validate_prepared_window_weights(
    weights: OracleWindowAttentionWeights,
    dimensions: Mapping[str, int | float],
) -> None:
    hidden = int(dimensions["hidden_size"])
    heads = int(dimensions["num_heads"])
    head_dim = int(dimensions["head_dim"])
    q_rank = int(dimensions["q_lora_rank"])
    o_rank = int(dimensions["o_lora_rank"])
    groups = int(dimensions["o_groups"])
    grouped_width = heads * head_dim // groups
    expected = {
        "attn_sink": (heads,),
        "wq_a": (q_rank, hidden),
        "q_norm": (q_rank,),
        "wq_b": (heads * head_dim, q_rank),
        "wkv": (head_dim, hidden),
        "kv_norm": (head_dim,),
        "wo_a": (groups * o_rank, grouped_width),
        "wo_b": (hidden, groups * o_rank),
    }
    devices: set[torch.device] = set()
    for name, shape in expected.items():
        value = getattr(weights, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"prepared attention weight {name} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"prepared attention weight {name} shape "
                f"{tuple(value.shape)} != {shape}"
            )
        if value.dtype != torch.float32:
            raise TypeError(f"prepared attention weight {name} must be float32")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"prepared attention weight {name} must be finite")
        devices.add(value.device)
    if len(devices) != 1:
        raise ValueError("prepared attention weights must share one device")


def init_window_oracle_state(
    config: Any,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> WindowOracleState:
    """Allocate an empty functional window-ring state snapshot."""

    dimensions = _window_oracle_dimensions(config)
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("oracle batch_size must be a positive integer")
    raw = torch.zeros(
        batch_size,
        WINDOW,
        int(dimensions["head_dim"]),
        dtype=torch.bfloat16,
        device=torch.device(device),
    )
    return WindowOracleState(
        raw=raw,
        next_position=0,
        max_seq_len=int(dimensions["max_seq_len"]),
    )


def _validate_window_oracle_state(
    state: WindowOracleState,
    *,
    batch_size: int,
    head_dim: int,
    max_seq_len: int,
    device: torch.device,
) -> None:
    if not isinstance(state, WindowOracleState):
        raise TypeError("state must be a WindowOracleState")
    if (
        tuple(state.raw.shape) != (batch_size, WINDOW, head_dim)
        or state.raw.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"oracle state raw must have shape {(batch_size, WINDOW, head_dim)} "
            "and dtype bfloat16"
        )
    if state.raw.device != device:
        raise ValueError("oracle state and hidden input must share a device")
    if state.max_seq_len != max_seq_len:
        raise ValueError("oracle state and config capacities differ")
    if not 0 <= state.next_position <= max_seq_len:
        raise ValueError("oracle state next_position is outside capacity")
    if not bool(torch.isfinite(state.raw).all()):
        raise ValueError("oracle latent state must be finite")


def oracle_window_attention_step(
    config: Any,
    weights: Any,
    hidden: torch.Tensor,
    *,
    start_pos: int,
    state: WindowOracleState | None = None,
    rope_table: RopeTable | None = None,
) -> WindowAttentionOracleStep:
    """Evaluate one functional prefill or decode step from raw resident weights."""

    dimensions = _window_oracle_dimensions(config)
    prepared = oracle_prepare_window_attention_weights(weights)
    _validate_prepared_window_weights(prepared, dimensions)
    if hidden.ndim != 3 or hidden.dtype != torch.bfloat16:
        raise ValueError("hidden must be a rank-3 BF16 tensor")
    if not isinstance(start_pos, int) or isinstance(start_pos, bool) or start_pos < 0:
        raise ValueError("start_pos must be a non-negative integer")
    batch, seqlen, hidden_size = hidden.shape
    head_dim = int(dimensions["head_dim"])
    rope_dim = int(dimensions["rope_dim"])
    max_seq_len = int(dimensions["max_seq_len"])
    if batch <= 0 or seqlen <= 0 or hidden_size != int(dimensions["hidden_size"]):
        raise ValueError("hidden shape does not match the oracle config")
    if start_pos > 0 and seqlen != 1:
        raise ValueError("decode oracle steps require exactly one token")
    if start_pos + seqlen > max_seq_len:
        raise ValueError("attention step exceeds the oracle state capacity")
    if prepared.wq_a.device != hidden.device:
        raise ValueError("prepared weights and hidden input must share a device")

    if state is None:
        if start_pos != 0:
            raise ValueError("decode oracle steps require an explicit prior state")
        working = init_window_oracle_state(config, batch, hidden.device)
    else:
        _validate_window_oracle_state(
            state,
            batch_size=batch,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            device=hidden.device,
        )
        if state.next_position != start_pos:
            raise ValueError(
                f"start_pos {start_pos} does not match state position "
                f"{state.next_position}"
            )
        working = state.clone()

    if rope_table is None:
        # model.py:477-481: original_seq_len 0 + base rope_theta -> plain
        # base-theta table (YaRN branch skipped in yarn_rope_table exactly as
        # in the reference precompute_freqs_cis, model.py:221).
        rope_table = yarn_rope_table(
            dim=rope_dim,
            seqlen=max_seq_len,
            original_seq_len=int(dimensions["original_seq_len"]),
            base=float(dimensions["rope_theta"]),
            factor=float(dimensions["rope_factor"]),
            beta_fast=int(dimensions["beta_fast"]),
            beta_slow=int(dimensions["beta_slow"]),
            device=hidden.device,
        )
    rope_table.validate()
    if (
        rope_table.cos.device != hidden.device
        or rope_table.cos.shape[1] * 2 != rope_dim
    ):
        raise ValueError("RoPE table device or width does not match the attention step")
    if rope_table.cos.shape[0] < start_pos + seqlen:
        raise ValueError("RoPE table does not cover the attention step")
    step_table = RopeTable(
        cos=rope_table.cos[start_pos : start_pos + seqlen],
        sin=rope_table.sin[start_pos : start_pos + seqlen],
    )
    eps = float(dimensions["norm_eps"])

    # q path (model.py:496-499).
    query_lora = oracle_rms_norm(
        _linear_bf16(hidden, prepared.wq_a),
        prepared.q_norm,
        eps=eps,
    )
    query = _linear_bf16(query_lora, prepared.wq_b).reshape(
        batch,
        seqlen,
        int(dimensions["num_heads"]),
        head_dim,
    )
    query_fp32 = query.to(torch.float32)
    query_fp32 = query_fp32 / torch.sqrt(
        torch.mean(query_fp32 * query_fp32, dim=-1, keepdim=True) + eps
    )
    query = query_fp32.to(torch.bfloat16)
    query[..., -rope_dim:] = oracle_apply_rope(query[..., -rope_dim:], step_table)

    # kv path (model.py:502-506): kv_norm, RoPE, NoPE E4M3/UE8M0 QDQ (QAT
    # intent, same fixed decision as E0ef).
    raw_latent = oracle_rms_norm(
        _linear_bf16(hidden, prepared.wkv),
        prepared.kv_norm,
        eps=eps,
    ).clone()
    raw_latent[..., -rope_dim:] = oracle_apply_rope(
        raw_latent[..., -rope_dim:], step_table
    )
    raw_qdq = e4m3_ue8m0_qdq(raw_latent[..., :-rope_dim], group_size=64)
    raw_latent[..., :-rope_dim] = raw_qdq.dequantized

    if start_pos == 0:
        # Prefill (model.py:518-523, 528): keep the last min(seqlen, window)
        # tokens at ring slots position % window; attend over the full
        # prefill latent with absolute-position indices.
        working.raw.zero_()
        kept = min(seqlen, WINDOW)
        absolute = torch.arange(
            seqlen - kept,
            seqlen,
            dtype=torch.int64,
            device=hidden.device,
        )
        working.raw.index_copy_(1, absolute.remainder(WINDOW), raw_latent[:, -kept:])
        attention_kv = raw_latent
        next_position = seqlen
    else:
        # Decode (model.py:530, 533): ring write, attend over the 128 rows.
        working.raw[:, start_pos % WINDOW].copy_(raw_latent[:, 0])
        attention_kv = working.raw
        next_position = start_pos + 1

    topk_indices = oracle_window_topk_indices(
        batch_size=batch,
        seqlen=seqlen,
        start_pos=start_pos,
        device=hidden.device,
    )
    sparse_output = oracle_sparse_attention(
        query,
        attention_kv,
        prepared.attn_sink,
        topk_indices,
        head_dim**-0.5,
    )
    inverse_rotated = sparse_output.clone()
    inverse_rotated[..., -rope_dim:] = oracle_apply_rope(
        inverse_rotated[..., -rope_dim:], step_table, inverse=True
    )

    # Output projection (model.py:537-542).
    groups = int(dimensions["o_groups"])
    o_rank = int(dimensions["o_lora_rank"])
    grouped_width = int(dimensions["num_heads"]) * head_dim // groups
    grouped = inverse_rotated.reshape(batch, seqlen, groups, grouped_width)
    wo_a = prepared.wo_a.reshape(groups, o_rank, grouped_width)
    projected_groups = [
        torch.matmul(
            grouped[:, :, group].to(torch.float32),
            wo_a[group].transpose(0, 1),
        ).to(torch.bfloat16)
        for group in range(groups)
    ]
    output_lora = torch.stack(projected_groups, dim=2)
    branch = _linear_bf16(output_lora.flatten(2), prepared.wo_b)

    post_state = WindowOracleState(
        raw=working.raw,
        next_position=next_position,
        max_seq_len=max_seq_len,
    )
    trace = WindowAttentionOracleTrace(
        query_lora=query_lora,
        query=query,
        raw_latent=raw_latent,
        attention_kv=attention_kv,
        topk_indices=topk_indices,
        sparse_output=sparse_output,
        inverse_rotated=inverse_rotated,
        output_lora=output_lora,
        branch=branch,
    )
    return WindowAttentionOracleStep(trace=trace, state=post_state)


__all__ = [
    "OracleWindowAttentionWeights",
    "WindowAttentionOracleStep",
    "WindowAttentionOracleTrace",
    "WindowOracleState",
    "init_window_oracle_state",
    "oracle_prepare_window_attention_weights",
    "oracle_window_attention_step",
]
