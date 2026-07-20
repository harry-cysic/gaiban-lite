"""Independent CPU-friendly ratio-128 attention/compressor mathematics.

This module is a semantic oracle, not a performance implementation.  It uses
only PyTorch primitives and deliberately does not import the candidate
``dsv4_direct.attention`` module.  The implementation also takes different
mathematical paths where practical: RoPE is evaluated with explicit real
cosine/sine pairs and sparse attention uses a scalar head loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch


RATIO128 = 128
E4M3_MAX = 448.0


@dataclass(frozen=True)
class RopeTable:
    """Real-valued YaRN RoPE table with one cosine/sine pair per complex dim."""

    cos: torch.Tensor
    sin: torch.Tensor

    def validate(self) -> None:
        if self.cos.dtype != torch.float32 or self.sin.dtype != torch.float32:
            raise TypeError("RoPE cosine and sine tables must be float32")
        if self.cos.ndim != 2 or self.cos.shape != self.sin.shape:
            raise ValueError("RoPE cosine and sine tables must have equal rank-2 shapes")
        if self.cos.shape[0] <= 0 or self.cos.shape[1] <= 0:
            raise ValueError("RoPE table dimensions must be positive")
        if not bool(torch.isfinite(self.cos).all() and torch.isfinite(self.sin).all()):
            raise ValueError("RoPE table must be finite")


@dataclass(frozen=True)
class E4M3UE8M0QDQ:
    """Quantized values, stored UE8M0 scales, and dequantized BF16 values."""

    quantized: torch.Tensor
    scales: torch.Tensor
    dequantized: torch.Tensor


@dataclass(frozen=True)
class Ratio128Compression:
    """Intermediate and finalized rows emitted by ratio-128 compression."""

    pooled: torch.Tensor
    finalized: torch.Tensor
    group_starts: torch.Tensor
    nope_scales: torch.Tensor


@dataclass
class OracleAttentionWeights:
    """FP32 oracle view of raw checkpoint attention/compressor weights."""

    attn_sink: torch.Tensor
    wq_a: torch.Tensor
    q_norm: torch.Tensor
    wq_b: torch.Tensor
    wkv: torch.Tensor
    kv_norm: torch.Tensor
    wo_a: torch.Tensor
    wo_b: torch.Tensor
    compressor_ape: torch.Tensor
    compressor_wkv: torch.Tensor
    compressor_wgate: torch.Tensor
    compressor_norm: torch.Tensor


@dataclass(frozen=True)
class Ratio128OracleState:
    """Functional snapshot of raw/compressed KV and pending compressor state."""

    raw: torch.Tensor
    compressed: torch.Tensor
    compressor_kv: torch.Tensor
    compressor_score: torch.Tensor
    next_position: int
    compressed_count: int
    max_seq_len: int

    def clone(self) -> "Ratio128OracleState":
        return Ratio128OracleState(
            raw=self.raw.clone(),
            compressed=self.compressed.clone(),
            compressor_kv=self.compressor_kv.clone(),
            compressor_score=self.compressor_score.clone(),
            next_position=self.next_position,
            compressed_count=self.compressed_count,
            max_seq_len=self.max_seq_len,
        )


@dataclass(frozen=True)
class Ratio128AttentionTrace:
    """Numerical boundaries exposed for independent candidate attribution."""

    query_lora: torch.Tensor
    query: torch.Tensor
    raw_latent: torch.Tensor
    projected_kv: torch.Tensor
    projected_score: torch.Tensor
    compression_pooled: torch.Tensor | None
    compression_finalized: torch.Tensor | None
    attention_kv: torch.Tensor
    topk_indices: torch.Tensor
    sparse_output: torch.Tensor
    inverse_rotated: torch.Tensor
    output_lora: torch.Tensor
    branch: torch.Tensor


@dataclass(frozen=True)
class Ratio128AttentionStep:
    trace: Ratio128AttentionTrace
    state: Ratio128OracleState


def oracle_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Evaluate RMSNorm in FP32 and return the input activation dtype."""

    if value.ndim < 1 or weight.ndim != 1 or value.shape[-1] != weight.numel():
        raise ValueError("RMSNorm value and weight shapes are incompatible")
    if not value.is_floating_point() or not weight.is_floating_point():
        raise TypeError("RMSNorm inputs must be floating point")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be positive and finite")
    value_fp32 = value.to(torch.float32)
    mean_square = torch.mean(value_fp32 * value_fp32, dim=-1, keepdim=True)
    normalized = value_fp32 / torch.sqrt(mean_square + eps)
    return (normalized * weight.to(torch.float32)).to(value.dtype)


def yarn_rope_table(
    *,
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device | str = "cpu",
) -> RopeTable:
    """Build the DeepSeek YaRN frequency table without complex tensors."""

    if dim <= 0 or dim % 2 or seqlen <= 0:
        raise ValueError("RoPE dim must be positive/even and seqlen must be positive")
    if original_seq_len < 0:
        raise ValueError("original_seq_len must be non-negative")
    if not math.isfinite(base) or base <= 1:
        raise ValueError("RoPE base must be finite and greater than one")
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("RoPE factor must be positive and finite")
    if beta_fast <= 0 or beta_slow <= 0:
        raise ValueError("YaRN beta values must be positive")

    target = torch.device(device)
    pair_indices = torch.arange(0, dim, 2, dtype=torch.float32, device=target)
    inverse_frequency = torch.pow(
        torch.tensor(base, dtype=torch.float32, device=target),
        -pair_indices / float(dim),
    )

    if original_seq_len > 0:
        def correction_dim(rotations: float) -> float:
            return dim * math.log(
                original_seq_len / (rotations * 2.0 * math.pi)
            ) / (2.0 * math.log(base))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        high_value = float(high) if low != high else float(high) + 0.001
        ramp = (
            torch.arange(dim // 2, dtype=torch.float32, device=target) - float(low)
        ) / (high_value - float(low))
        smooth = 1.0 - ramp.clamp(0.0, 1.0)
        inverse_frequency = (
            inverse_frequency / factor * (1.0 - smooth)
            + inverse_frequency * smooth
        )

    positions = torch.arange(seqlen, dtype=torch.float32, device=target)
    phase = positions[:, None] * inverse_frequency[None, :]
    table = RopeTable(cos=torch.cos(phase), sin=torch.sin(phase))
    table.validate()
    return table


def oracle_apply_rope(
    value: torch.Tensor,
    table: RopeTable,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE using explicit real two-vector rotations."""

    table.validate()
    if value.ndim < 3 or value.shape[-1] != table.cos.shape[-1] * 2:
        raise ValueError("RoPE value width does not match the table")
    if value.shape[1] != table.cos.shape[0]:
        raise ValueError("RoPE value sequence length does not match the table")
    if not value.is_floating_point():
        raise TypeError("RoPE input must be floating point")
    if value.device != table.cos.device:
        raise ValueError("RoPE input and table must share a device")

    pairs = value.to(torch.float32).reshape(*value.shape[:-1], -1, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    table_shape = [1, value.shape[1]] + [1] * (value.ndim - 3) + [table.cos.shape[1]]
    cosine = table.cos.view(*table_shape)
    sine = table.sin.view(*table_shape)
    if inverse:
        sine = -sine
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine),
        dim=-1,
    ).flatten(-2)
    return rotated.to(value.dtype)


def e4m3_ue8m0_qdq(
    value: torch.Tensor,
    *,
    group_size: int = 64,
) -> E4M3UE8M0QDQ:
    """Per-group E4M3 quantize/dequantize with power-of-two UE8M0 scales."""

    if value.dtype != torch.bfloat16:
        raise TypeError("E4M3/UE8M0 activation QDQ requires BF16 input")
    if value.ndim < 1 or group_size <= 0 or value.shape[-1] % group_size:
        raise ValueError("QDQ group size must positively divide the last dimension")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("QDQ input must be finite")

    grouped = value.to(torch.float32).reshape(*value.shape[:-1], -1, group_size)
    absolute_max = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale_exponent = torch.ceil(torch.log2(absolute_max / E4M3_MAX))
    exact_scale = torch.pow(2.0, scale_exponent)
    stored_scale = exact_scale.squeeze(-1).to(torch.float8_e8m0fnu)
    scale = stored_scale.to(torch.float32).unsqueeze(-1)
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("QDQ scale is outside the finite UE8M0 range")

    normalized = torch.clamp(grouped / scale, -E4M3_MAX, E4M3_MAX)
    quantized_grouped = normalized.to(torch.float8_e4m3fn)
    dequantized = (quantized_grouped.to(torch.float32) * scale).reshape_as(value)
    return E4M3UE8M0QDQ(
        quantized=quantized_grouped.reshape_as(value),
        scales=stored_scale,
        dequantized=dequantized.to(value.dtype),
    )


def oracle_compressor_pool(
    projected_kv: torch.Tensor,
    projected_score: torch.Tensor,
    ape: torch.Tensor,
    *,
    ratio: int = RATIO128,
) -> torch.Tensor:
    """Apply per-dimension APE logits and pool each complete ratio-sized group."""

    if projected_kv.dtype != torch.float32 or projected_score.dtype != torch.float32:
        raise TypeError("compressor KV and score projections must be float32")
    if projected_kv.ndim != 3 or projected_kv.shape != projected_score.shape:
        raise ValueError("compressor projections must have equal [batch, seq, dim] shapes")
    if ape.dtype != torch.float32 or ape.ndim != 2:
        raise TypeError("compressor APE must be a rank-2 float32 tensor")
    batch, seqlen, width = projected_kv.shape
    if ratio <= 0 or seqlen <= 0 or seqlen % ratio:
        raise ValueError("compressor sequence length must be a positive ratio multiple")
    if tuple(ape.shape) != (ratio, width):
        raise ValueError("compressor APE shape does not match ratio and latent width")
    if projected_kv.device != projected_score.device or projected_kv.device != ape.device:
        raise ValueError("compressor inputs must share a device")
    if not bool(
        torch.isfinite(projected_kv).all()
        and torch.isfinite(projected_score).all()
        and torch.isfinite(ape).all()
    ):
        raise ValueError("compressor inputs must be finite")

    groups = seqlen // ratio
    grouped_kv = projected_kv.reshape(batch, groups, ratio, width)
    grouped_logits = projected_score.reshape(batch, groups, ratio, width)
    grouped_logits = grouped_logits + ape.view(1, 1, ratio, width)
    probabilities = torch.softmax(grouped_logits, dim=2)
    return torch.sum(grouped_kv * probabilities, dim=2)


def oracle_finalize_compressed(
    pooled: torch.Tensor,
    norm_weight: torch.Tensor,
    table: RopeTable,
    group_starts: torch.Tensor,
    *,
    rope_dim: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS-normalize, rotate, and QDQ the NoPE portion of compressed rows."""

    if pooled.dtype != torch.float32 or pooled.ndim != 3:
        raise TypeError("pooled compressor rows must be rank-3 float32")
    if group_starts.dtype != torch.int64 or group_starts.ndim != 1:
        raise TypeError("group starts must be a rank-1 int64 tensor")
    if group_starts.numel() != pooled.shape[1]:
        raise ValueError("group starts must contain one position per compressed row")
    if group_starts.device != pooled.device or table.cos.device != pooled.device:
        raise ValueError("compressed rows, positions, and RoPE table must share a device")
    width = pooled.shape[-1]
    if rope_dim <= 0 or rope_dim % 2 or rope_dim >= width:
        raise ValueError("rope_dim must be positive/even and leave a non-empty NoPE prefix")
    if bool((group_starts < 0).any()) or bool((group_starts >= table.cos.shape[0]).any()):
        raise ValueError("compressed group start is outside the RoPE table")
    if table.cos.shape[1] * 2 != rope_dim:
        raise ValueError("compressed RoPE table width does not match rope_dim")

    normalized = oracle_rms_norm(
        pooled.to(torch.bfloat16), norm_weight, eps=eps
    ).clone()
    row_table = RopeTable(
        cos=table.cos.index_select(0, group_starts),
        sin=table.sin.index_select(0, group_starts),
    )
    normalized[..., -rope_dim:] = oracle_apply_rope(
        normalized[..., -rope_dim:], row_table
    )
    qdq = e4m3_ue8m0_qdq(normalized[..., :-rope_dim], group_size=64)
    normalized[..., :-rope_dim] = qdq.dequantized
    return normalized.contiguous(), qdq.scales


def oracle_ratio128_compress(
    projected_kv: torch.Tensor,
    projected_score: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    table: RopeTable,
    *,
    rope_dim: int,
    eps: float = 1e-6,
) -> Ratio128Compression:
    """Evaluate all complete ratio-128 compressor groups from position zero."""

    pooled = oracle_compressor_pool(
        projected_kv, projected_score, ape, ratio=RATIO128
    )
    starts = torch.arange(
        0,
        projected_kv.shape[1],
        RATIO128,
        dtype=torch.int64,
        device=projected_kv.device,
    )
    finalized, scales = oracle_finalize_compressed(
        pooled,
        norm_weight,
        table,
        starts,
        rope_dim=rope_dim,
        eps=eps,
    )
    return Ratio128Compression(
        pooled=pooled,
        finalized=finalized,
        group_starts=starts,
        nope_scales=scales,
    )


def oracle_sparse_attention(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Small-shape sparse attention using an explicit batch/sequence/head loop."""

    if query.ndim != 4 or latent_kv.ndim != 3 or topk_indices.ndim != 3:
        raise ValueError("query, latent KV, and top-k indices require ranks 4, 3, and 3")
    batch, seqlen, heads, width = query.shape
    if latent_kv.shape[0] != batch or latent_kv.shape[2] != width:
        raise ValueError("query and latent KV shapes are incompatible")
    if tuple(topk_indices.shape[:2]) != (batch, seqlen):
        raise ValueError("top-k batch/sequence shape does not match query")
    if tuple(attn_sink.shape) != (heads,):
        raise ValueError("attention sink must contain one scalar per head")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("top-k indices must use int32 or int64")
    if not query.is_floating_point() or not latent_kv.is_floating_point():
        raise TypeError("query and latent KV must be floating point")
    if not attn_sink.is_floating_point():
        raise TypeError("attention sink must be floating point")
    if query.device != latent_kv.device or query.device != topk_indices.device:
        raise ValueError("sparse-attention tensors must share a device")
    if attn_sink.device != query.device:
        raise ValueError("attention sink must share the query device")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax scale must be positive and finite")

    valid_indices = topk_indices[topk_indices >= 0]
    if valid_indices.numel() and bool((valid_indices >= latent_kv.shape[1]).any()):
        raise ValueError("top-k index exceeds latent KV capacity")

    output = torch.zeros_like(query)
    kv_fp32 = latent_kv.to(torch.float32)
    for batch_index in range(batch):
        for sequence_index in range(seqlen):
            indices = topk_indices[batch_index, sequence_index]
            indices = indices[indices >= 0].to(torch.int64)
            if indices.numel() == 0:
                continue
            selected = kv_fp32[batch_index].index_select(0, indices)
            for head_index in range(heads):
                scores = torch.mv(
                    selected,
                    query[batch_index, sequence_index, head_index].to(torch.float32),
                ) * softmax_scale
                logits = torch.cat(
                    (scores, attn_sink[head_index : head_index + 1].to(torch.float32))
                )
                probabilities = torch.softmax(logits, dim=0)[:-1]
                attended = torch.sum(probabilities[:, None] * selected, dim=0)
                output[batch_index, sequence_index, head_index].copy_(
                    attended.to(output.dtype)
                )
    return output


def oracle_sparse_attention_batched(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Full-batch sparse-attention oracle using batched matrix products.

    This is the scalable counterpart to :func:`oracle_sparse_attention`.  It
    deliberately lives in the independent oracle module and uses ``matmul``
    rather than the candidate/control einsum implementation.  Promotion
    experiments should bind a predeclared subset of rows and heads to the
    scalar oracle above, then use this function for the complete workload.
    """

    if query.ndim != 4 or latent_kv.ndim != 3 or topk_indices.ndim != 3:
        raise ValueError("query, latent KV, and top-k indices require ranks 4, 3, and 3")
    batch, seqlen, heads, width = query.shape
    if latent_kv.shape[0] != batch or latent_kv.shape[2] != width:
        raise ValueError("query and latent KV shapes are incompatible")
    if tuple(topk_indices.shape[:2]) != (batch, seqlen):
        raise ValueError("top-k batch/sequence shape does not match query")
    if tuple(attn_sink.shape) != (heads,):
        raise ValueError("attention sink must contain one scalar per head")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("top-k indices must use int32 or int64")
    if not query.is_floating_point() or not latent_kv.is_floating_point():
        raise TypeError("query and latent KV must be floating point")
    if not attn_sink.is_floating_point():
        raise TypeError("attention sink must be floating point")
    if (
        query.device != latent_kv.device
        or query.device != topk_indices.device
        or query.device != attn_sink.device
    ):
        raise ValueError("sparse-attention tensors must share a device")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax scale must be positive and finite")

    valid = topk_indices >= 0
    valid_indices = topk_indices[valid]
    if valid_indices.numel() and bool((valid_indices >= latent_kv.shape[1]).any()):
        raise ValueError("top-k index exceeds latent KV capacity")
    safe = topk_indices.clamp_min(0).to(torch.int64)
    batch_indices = (
        torch.arange(batch, dtype=torch.int64, device=query.device)
        .view(batch, 1, 1)
        .expand_as(safe)
    )
    selected = latent_kv[batch_indices, safe].to(torch.float32)
    scores = torch.matmul(
        query.to(torch.float32), selected.transpose(-1, -2)
    ) * float(softmax_scale)
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.to(torch.float32).view(1, 1, heads, 1)
    logits = torch.cat(
        (scores, sink.expand(batch, seqlen, -1, -1)), dim=-1
    )
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.matmul(probabilities[..., :-1], selected)
    return output.to(query.dtype)


def oracle_dequant_fp8_block(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """Decode a block-scaled FP8 matrix without using runtime helpers."""

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 block weight and scale must both be rank 2")
    if block_size <= 0:
        raise ValueError("FP8 block size must be positive")
    if not weight.is_floating_point():
        raise TypeError("FP8 block weight must be floating point")

    rows, columns = weight.shape
    expected = (
        math.ceil(rows / block_size),
        math.ceil(columns / block_size),
    )
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"FP8 block scale shape {tuple(scale.shape)} does not match {expected}"
        )
    decoded_scale = scale.contiguous()
    if decoded_scale.dtype == torch.uint8:
        decoded_scale = decoded_scale.view(torch.float8_e8m0fnu)
    if decoded_scale.dtype == torch.float8_e8m0fnu:
        decoded_scale = decoded_scale.to(torch.float32)
    elif decoded_scale.is_floating_point():
        decoded_scale = decoded_scale.to(torch.float32)
    else:
        raise TypeError("FP8 block scale must be E8M0, uint8, or decoded floating point")
    if not bool(torch.isfinite(decoded_scale).all()):
        raise ValueError("FP8 block scale must decode to finite values")

    expanded = decoded_scale.repeat_interleave(block_size, dim=0)
    expanded = expanded.repeat_interleave(block_size, dim=1)[:rows, :columns]
    return (weight.to(torch.float32) * expanded).contiguous()


def oracle_prepare_attention_weights(weights: Any) -> OracleAttentionWeights:
    """Create an independent FP32 view of resident raw checkpoint weights."""

    if isinstance(weights, OracleAttentionWeights):
        return weights

    def dequant(name: str) -> torch.Tensor:
        try:
            linear = getattr(weights, name)
            matrix = linear.weight
            scales = linear.scale
        except AttributeError as exc:
            raise TypeError(f"attention weights are missing quantized linear {name}") from exc
        return oracle_dequant_fp8_block(matrix, scales)

    try:
        compressor = weights.compressor
        result = OracleAttentionWeights(
            attn_sink=weights.attn_sink.to(torch.float32).contiguous().clone(),
            wq_a=dequant("wq_a"),
            q_norm=weights.q_norm.to(torch.float32).contiguous().clone(),
            wq_b=dequant("wq_b"),
            wkv=dequant("wkv"),
            kv_norm=weights.kv_norm.to(torch.float32).contiguous().clone(),
            wo_a=dequant("wo_a"),
            wo_b=dequant("wo_b"),
            compressor_ape=compressor.ape.to(torch.float32).contiguous().clone(),
            compressor_wkv=compressor.wkv.to(torch.float32).contiguous().clone(),
            compressor_wgate=compressor.wgate.to(torch.float32).contiguous().clone(),
            compressor_norm=compressor.norm.to(torch.float32).contiguous().clone(),
        )
    except AttributeError as exc:
        raise TypeError("resident attention weights do not satisfy the raw contract") from exc
    return result


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        try:
            return config[name]
        except KeyError as exc:
            raise ValueError(f"oracle config is missing {name}") from exc
    try:
        return getattr(config, name)
    except AttributeError as exc:
        raise ValueError(f"oracle config is missing {name}") from exc


def _oracle_dimensions(config: Any) -> dict[str, int | float]:
    # All dimensions are derived from the caller's config; nothing is frozen
    # here.  The Flash ratio-128 geometry (hidden 4096, 64 heads, head_dim 512,
    # rope 64, q_lora 1024, o_lora 1024, o_groups 8) satisfies every generic
    # constraint below: NoPE width 448 is a positive multiple of 64 and
    # heads*head_dim = 32768 divides evenly into 8 output groups.
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
    if int(values["original_seq_len"]) < 0:
        raise ValueError("original_seq_len must be non-negative")
    if int(values["rope_dim"]) % 2:
        raise ValueError("rope_dim must be even")
    nope_dim = int(values["head_dim"]) - int(values["rope_dim"])
    if nope_dim <= 0 or nope_dim % 64:
        raise ValueError("NoPE width must be a positive multiple of 64")
    total_head_dim = int(values["num_heads"]) * int(values["head_dim"])
    if total_head_dim % int(values["o_groups"]):
        raise ValueError("heads times head_dim must divide evenly into output groups")
    if int(values["max_seq_len"]) < RATIO128 or int(values["max_seq_len"]) % RATIO128:
        raise ValueError("max_seq_len must be a positive multiple of 128")
    if any(
        not math.isfinite(float(values[name])) or float(values[name]) <= 0
        for name in ("norm_eps", "rope_theta", "rope_factor")
    ):
        raise ValueError("oracle attention numerical constants must be positive and finite")
    return values


def _validate_prepared_weights(
    weights: OracleAttentionWeights,
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
        "compressor_ape": (RATIO128, head_dim),
        "compressor_wkv": (head_dim, hidden),
        "compressor_wgate": (head_dim, hidden),
        "compressor_norm": (head_dim,),
    }
    devices: set[torch.device] = set()
    for name, shape in expected.items():
        value = getattr(weights, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"prepared attention weight {name} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"prepared attention weight {name} shape {tuple(value.shape)} != {shape}"
            )
        if value.dtype != torch.float32:
            raise TypeError(f"prepared attention weight {name} must be float32")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"prepared attention weight {name} must be finite")
        devices.add(value.device)
    if len(devices) != 1:
        raise ValueError("prepared attention weights must share one device")


def init_ratio128_oracle_state(
    config: Any,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> Ratio128OracleState:
    """Allocate an empty functional ratio-128 state snapshot."""

    dimensions = _oracle_dimensions(config)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("oracle batch_size must be a positive integer")
    target = torch.device(device)
    head_dim = int(dimensions["head_dim"])
    max_seq_len = int(dimensions["max_seq_len"])
    raw = torch.zeros(
        batch_size,
        RATIO128,
        head_dim,
        dtype=torch.bfloat16,
        device=target,
    )
    compressed = torch.zeros(
        batch_size,
        max_seq_len // RATIO128,
        head_dim,
        dtype=torch.bfloat16,
        device=target,
    )
    compressor_kv = torch.zeros(
        batch_size,
        RATIO128,
        head_dim,
        dtype=torch.float32,
        device=target,
    )
    compressor_score = torch.full_like(compressor_kv, float("-inf"))
    return Ratio128OracleState(
        raw=raw,
        compressed=compressed,
        compressor_kv=compressor_kv,
        compressor_score=compressor_score,
        next_position=0,
        compressed_count=0,
        max_seq_len=max_seq_len,
    )


def _validate_oracle_state(
    state: Ratio128OracleState,
    *,
    batch_size: int,
    head_dim: int,
    max_seq_len: int,
    device: torch.device,
) -> None:
    if not isinstance(state, Ratio128OracleState):
        raise TypeError("state must be a Ratio128OracleState")
    expected = {
        "raw": ((batch_size, RATIO128, head_dim), torch.bfloat16),
        "compressed": (
            (batch_size, max_seq_len // RATIO128, head_dim),
            torch.bfloat16,
        ),
        "compressor_kv": ((batch_size, RATIO128, head_dim), torch.float32),
        "compressor_score": ((batch_size, RATIO128, head_dim), torch.float32),
    }
    for name, (shape, dtype) in expected.items():
        value = getattr(state, name)
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"oracle state {name} must have shape {shape} and dtype {dtype}"
            )
        if value.device != device:
            raise ValueError("oracle state and hidden input must share a device")
    if state.max_seq_len != max_seq_len:
        raise ValueError("oracle state and config capacities differ")
    if not 0 <= state.next_position <= max_seq_len:
        raise ValueError("oracle state next_position is outside capacity")
    if state.compressed_count != state.next_position // RATIO128:
        raise ValueError("oracle state compressed_count is inconsistent")
    if not bool(torch.isfinite(state.raw).all() and torch.isfinite(state.compressed).all()):
        raise ValueError("oracle latent state must be finite")
    if not bool(torch.isfinite(state.compressor_kv).all()):
        raise ValueError("oracle compressor KV state must be finite")
    score_valid = torch.isfinite(state.compressor_score) | torch.isneginf(
        state.compressor_score
    )
    if not bool(score_valid.all()):
        raise ValueError("oracle compressor scores must be finite or negative infinity")


def oracle_window_topk_indices(
    *,
    batch_size: int,
    seqlen: int,
    start_pos: int,
    device: torch.device | str,
    window_size: int = RATIO128,
) -> torch.Tensor:
    """Construct chronological raw-window indices with explicit integer loops."""

    if batch_size <= 0 or seqlen <= 0 or start_pos < 0 or window_size <= 0:
        raise ValueError("invalid raw-window index dimensions")
    if start_pos > 0 and seqlen != 1:
        raise ValueError("incremental raw-window indices require one token")
    rows: list[list[int]] = []
    if start_pos == 0:
        width = min(seqlen, window_size)
        for position in range(seqlen):
            oldest = max(0, position - window_size + 1)
            visible = list(range(oldest, position + 1))
            rows.append(visible + [-1] * (width - len(visible)))
    elif start_pos < window_size - 1:
        visible = list(range(start_pos + 1))
        rows.append(visible + [-1] * (window_size - len(visible)))
    else:
        ring = start_pos % window_size
        rows.append(list(range(ring + 1, window_size)) + list(range(ring + 1)))
    result = torch.tensor(rows, dtype=torch.int32, device=device)
    return result.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def oracle_compressed_topk_indices(
    *,
    batch_size: int,
    seqlen: int,
    start_pos: int,
    offset: int,
    device: torch.device | str,
    ratio: int = RATIO128,
) -> torch.Tensor:
    """Construct visible completed-compressor indices independently."""

    if batch_size <= 0 or seqlen <= 0 or start_pos < 0 or offset < 0 or ratio <= 0:
        raise ValueError("invalid compressed-index dimensions")
    rows: list[list[int]] = []
    if start_pos > 0:
        if seqlen != 1:
            raise ValueError("incremental compressed indices require one token")
        rows.append([offset + row for row in range((start_pos + 1) // ratio)])
    else:
        width = seqlen // ratio
        for position in range(seqlen):
            visible = (position + 1) // ratio
            rows.append(
                [offset + row if row < visible else -1 for row in range(width)]
            )
    if rows and not rows[0]:
        result = torch.empty((len(rows), 0), dtype=torch.int32, device=device)
    else:
        result = torch.tensor(rows, dtype=torch.int32, device=device)
    return result.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def _linear_bf16(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(value.to(torch.float32), weight.transpose(0, 1)).to(
        torch.bfloat16
    )


def oracle_ratio128_attention_step(
    config: Any,
    weights: Any,
    hidden: torch.Tensor,
    *,
    start_pos: int,
    state: Ratio128OracleState | None = None,
    rope_table: RopeTable | None = None,
) -> Ratio128AttentionStep:
    """Evaluate one functional prefill or decode step from raw resident weights."""

    dimensions = _oracle_dimensions(config)
    prepared = oracle_prepare_attention_weights(weights)
    _validate_prepared_weights(prepared, dimensions)
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
        working = init_ratio128_oracle_state(config, batch, hidden.device)
    else:
        _validate_oracle_state(
            state,
            batch_size=batch,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            device=hidden.device,
        )
        if state.next_position != start_pos:
            raise ValueError(
                f"start_pos {start_pos} does not match state position {state.next_position}"
            )
        working = state.clone()

    if rope_table is None:
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
    if rope_table.cos.device != hidden.device or rope_table.cos.shape[1] * 2 != rope_dim:
        raise ValueError("RoPE table device or width does not match the attention step")
    if rope_table.cos.shape[0] < start_pos + seqlen:
        raise ValueError("RoPE table does not cover the attention step")
    step_table = RopeTable(
        cos=rope_table.cos[start_pos : start_pos + seqlen],
        sin=rope_table.sin[start_pos : start_pos + seqlen],
    )
    eps = float(dimensions["norm_eps"])

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
    query[..., -rope_dim:] = oracle_apply_rope(
        query[..., -rope_dim:], step_table
    )

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

    projected_kv = torch.matmul(
        hidden.to(torch.float32), prepared.compressor_wkv.transpose(0, 1)
    )
    projected_score = torch.matmul(
        hidden.to(torch.float32), prepared.compressor_wgate.transpose(0, 1)
    )
    compression_pooled: torch.Tensor | None = None
    compression_finalized: torch.Tensor | None = None

    if start_pos == 0:
        working.raw.zero_()
        working.compressed.zero_()
        working.compressor_kv.zero_()
        working.compressor_score.fill_(float("-inf"))
        completed = seqlen // RATIO128
        cutoff = completed * RATIO128
        if cutoff:
            compression = oracle_ratio128_compress(
                projected_kv[:, :cutoff],
                projected_score[:, :cutoff],
                prepared.compressor_ape,
                prepared.compressor_norm,
                rope_table,
                rope_dim=rope_dim,
                eps=eps,
            )
            compression_pooled = compression.pooled
            compression_finalized = compression.finalized
            working.compressed[:, :completed].copy_(compression.finalized)

        kept = min(seqlen, RATIO128)
        absolute = torch.arange(
            seqlen - kept,
            seqlen,
            dtype=torch.int64,
            device=hidden.device,
        )
        working.raw.index_copy_(1, absolute.remainder(RATIO128), raw_latent[:, -kept:])
        remainder = seqlen - cutoff
        if remainder:
            working.compressor_kv[:, :remainder].copy_(projected_kv[:, cutoff:])
            adjusted = projected_score[:, cutoff:] + prepared.compressor_ape[:remainder]
            working.compressor_score[:, :remainder].copy_(adjusted)

        attention_kv = torch.cat(
            (raw_latent, working.compressed[:, :completed]), dim=1
        )
        compressed_indices = oracle_compressed_topk_indices(
            batch_size=batch,
            seqlen=seqlen,
            start_pos=0,
            offset=seqlen,
            device=hidden.device,
        )
        next_position = seqlen
        compressed_count = completed
    else:
        expected_compressed = start_pos // RATIO128
        if working.compressed_count != expected_compressed:
            raise ValueError("decode state has an inconsistent compressed-row count")
        raw_slot = start_pos % RATIO128
        working.raw[:, raw_slot].copy_(raw_latent[:, 0])
        working.compressor_kv[:, raw_slot].copy_(projected_kv[:, 0])
        working.compressor_score[:, raw_slot].copy_(
            projected_score[:, 0] + prepared.compressor_ape[raw_slot]
        )
        completed_now = (start_pos + 1) % RATIO128 == 0
        compressed_count = expected_compressed
        if completed_now:
            probabilities = torch.softmax(working.compressor_score, dim=1)
            compression_pooled = torch.sum(
                working.compressor_kv * probabilities, dim=1, keepdim=True
            )
            group_start = torch.tensor(
                [start_pos + 1 - RATIO128],
                dtype=torch.int64,
                device=hidden.device,
            )
            compression_finalized, _ = oracle_finalize_compressed(
                compression_pooled,
                prepared.compressor_norm,
                rope_table,
                group_start,
                rope_dim=rope_dim,
                eps=eps,
            )
            working.compressed[
                :, expected_compressed : expected_compressed + 1
            ].copy_(compression_finalized)
            compressed_count += 1
        attention_kv = torch.cat((working.raw, working.compressed), dim=1)
        compressed_indices = oracle_compressed_topk_indices(
            batch_size=batch,
            seqlen=1,
            start_pos=start_pos,
            offset=RATIO128,
            device=hidden.device,
        )
        next_position = start_pos + 1

    window_indices = oracle_window_topk_indices(
        batch_size=batch,
        seqlen=seqlen,
        start_pos=start_pos,
        device=hidden.device,
    )
    topk_indices = torch.cat((window_indices, compressed_indices), dim=-1).contiguous()
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

    post_state = Ratio128OracleState(
        raw=working.raw,
        compressed=working.compressed,
        compressor_kv=working.compressor_kv,
        compressor_score=working.compressor_score,
        next_position=next_position,
        compressed_count=compressed_count,
        max_seq_len=max_seq_len,
    )
    trace = Ratio128AttentionTrace(
        query_lora=query_lora,
        query=query,
        raw_latent=raw_latent,
        projected_kv=projected_kv,
        projected_score=projected_score,
        compression_pooled=compression_pooled,
        compression_finalized=compression_finalized,
        attention_kv=attention_kv,
        topk_indices=topk_indices,
        sparse_output=sparse_output,
        inverse_rotated=inverse_rotated,
        output_lora=output_lora,
        branch=branch,
    )
    return Ratio128AttentionStep(trace=trace, state=post_state)


__all__ = [
    "E4M3UE8M0QDQ",
    "OracleAttentionWeights",
    "Ratio128AttentionStep",
    "Ratio128AttentionTrace",
    "Ratio128Compression",
    "Ratio128OracleState",
    "RopeTable",
    "e4m3_ue8m0_qdq",
    "init_ratio128_oracle_state",
    "oracle_apply_rope",
    "oracle_compressed_topk_indices",
    "oracle_compressor_pool",
    "oracle_dequant_fp8_block",
    "oracle_finalize_compressed",
    "oracle_prepare_attention_weights",
    "oracle_ratio128_attention_step",
    "oracle_ratio128_compress",
    "oracle_rms_norm",
    "oracle_sparse_attention",
    "oracle_sparse_attention_batched",
    "oracle_window_topk_indices",
    "yarn_rope_table",
]
