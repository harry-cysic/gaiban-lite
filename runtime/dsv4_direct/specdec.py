"""Per-sequence-position (row-position) stateful decode for chained MTP.

Eighteenth vertical: large-B chained MTP speculative decoding on the E1IF
interleaved pipeline.  Design lineage: results/mtp/DESIGN-largeB-graph.md
(per-sequence position vector + masked boundary + shadow-commit rollback),
with the **chained dual-pass** form chosen over the fixed 2-token verify
graph:

- E0mtp2e proved the chained protocol is bitwise lossless (8/8 prompts
  identical to MTP-off) while the fused seqlen-2 GEMM form is not (near-tie
  argmax flips).  The hard acceptance for this vertical is per-sequence
  bitwise equality of MTP-on vs MTP-off output streams, which only the
  chained form can meet.
- Each pass keeps the exact 1-token operator chain of the verified family
  graphs; the only change is that every position-derived quantity becomes a
  per-row (``positions[B]``) gather/scatter and the two compressor
  boundaries become masked commits ("compute always, commit where
  ``phase == ratio-1``").  At any step where every row sits at the same
  position, each committed value and each attention output is bitwise
  identical to the family path (same kernels, same operand values, same
  reduction widths -- the ratio-128 sparse bucket width is caller-fixed so
  it can match the family plan's width exactly).

Round structure (per lane, per round; ``positions[B]`` = pass-A write
position):

1. round head (inside the pass-A graph): ``positions += advance`` where
   ``advance = 1 + accept_{r-1}`` (0 for the very first round); then for
   every ratio-4 layer restore the overlap-compressor state from the
   post-pass-A shadow of the previous round on rejected rows (the only
   destructive mutation of pass B is the ratio-4 boundary shift; window
   rings, the ratio-128 compressor, and all compressed rows heal by the
   refeed itself because pass A rewrites the same slots with corrected
   values before anything reads them).
2. pass A: feed the pending token at ``positions`` (verify pass).
3. pass-A tail: snapshot every ratio-4 overlap state into its shadow.
4. pass B: feed the draft token at ``positions + 1`` (speculative pass;
   real commits, provisional semantics).

Accept/reject (computed at the tail stage from the pass-A argmax vs the
round's draft) only changes ``advance`` and which token becomes pending --
never any shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .attention import (
    Ratio128TorchAttention,
    _torch_sparse_decode_padded_prevalidated,
    apply_rotary_emb,
    fp8_quant_dequant,
    rms_norm,
)
from .block import BLOCK_HC_MULT, BLOCK_HIDDEN_SIZE, DirectDecodeBlock
from .head_stage import hc_head_collapse_tensors
from .hyper_connections import hc_post, hc_pre
from .ratio4_attention import (
    Ratio4TorchAttention,
    fp4_quant_dequant,
    hadamard_transform,
)
from .static_kv import (
    LATENT_ROPE_DIM,
    StaticLayerKV,
    quantize_latent_rows,
)
from .static_ratio4_kv import (
    INDEX_DIM,
    StaticRatio4KV,
)
from .static_ratio4_kv import COMPRESS_RATIO as RATIO4
from .static_ratio4_kv import LATENT_DIM as R4_LATENT_DIM
from .static_window_kv import StaticWindowKV
from .static_kv import COMPRESS_RATIO as RATIO128
from .static_kv import WINDOW_SIZE
from .superstage import TP4DecodeStage
from .window_attention import (
    WindowTorchAttention,
    _window_sparse_decode_prevalidated,
)


class SpecDecodeError(ValueError):
    """Raised when the row-position spec-decode path is driven off-contract."""


# --------------------------------------------------------------------------
# row-position primitives


def gather_row_freqs(
    freqs_cis: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """``freqs_cis[positions]`` as ``[B, 1, half]`` complex (per-row RoPE)."""

    return freqs_cis.index_select(0, positions).unsqueeze(1)


def apply_rotary_emb_rows(
    value: torch.Tensor,
    row_freqs: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Per-row RoPE: same complex product as ``apply_rotary_emb`` with the
    frequency row gathered per sequence.  ``value`` is ``[B, 1, ..., d]``;
    ``row_freqs`` is ``[B, 1, d//2]`` complex.  At equal positions the result
    is bitwise identical to the shared-frequency path (identical elementwise
    complex multiply on identical operand values)."""

    if value.shape[-1] != row_freqs.shape[-1] * 2 or value.shape[1] != 1:
        raise SpecDecodeError("row RoPE value/frequency geometry mismatch")
    complex_value = torch.view_as_complex(
        value.float().reshape(*value.shape[:-1], -1, 2)
    )
    frequencies = row_freqs.conj() if inverse else row_freqs
    view_shape = (
        [value.shape[0], 1]
        + [1] * (value.ndim - 3)
        + [row_freqs.shape[-1]]
    )
    rotated = torch.view_as_real(
        complex_value * frequencies.view(*view_shape)
    ).flatten(-2)
    return rotated.to(value.dtype)


def _row_index(row: torch.Tensor, width: int) -> torch.Tensor:
    """``[B]`` row indices -> ``[B, 1, width]`` scatter/gather index."""

    return row.view(-1, 1, 1).expand(-1, 1, width)


def scatter_state_rows_(
    destination: torch.Tensor, row: torch.Tensor, source: torch.Tensor
) -> None:
    """Per-row dim-1 row write (``destination[b, row[b]] = source[b, 0]``).

    FP8 storage is written through a same-itemsize uint8 reinterpret, the
    same byte movement as ``static_kv.index_copy_rows``.
    """

    width = destination.shape[-1]
    index = _row_index(row, width)
    if destination.dtype == torch.float8_e4m3fn:
        destination.view(torch.uint8).scatter_(
            1, index, source.contiguous().view(torch.uint8)
        )
    else:
        destination.scatter_(1, index, source.to(destination.dtype))


def masked_scatter_state_rows_(
    destination: torch.Tensor,
    row: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Commit ``source`` at per-row rows only where ``mask``; elsewhere the
    destination row is rewritten with its own current bytes (no-op)."""

    width = destination.shape[-1]
    index = _row_index(row, width)
    mask_view = mask.view(-1, 1, 1)
    if destination.dtype == torch.float8_e4m3fn:
        current = destination.view(torch.uint8).gather(1, index)
        blended = torch.where(
            mask_view, source.contiguous().view(torch.uint8), current
        )
        destination.view(torch.uint8).scatter_(1, index, blended)
    else:
        current = destination.gather(1, index)
        blended = torch.where(mask_view, source.to(destination.dtype), current)
        destination.scatter_(1, index, blended)


def scatter_position_rows_(
    destination: torch.Tensor, slot: torch.Tensor, value: torch.Tensor
) -> None:
    """``destination[b, slot[b]] = value[b]`` for int64 metadata rows."""

    destination.scatter_(1, slot.view(-1, 1), value.view(-1, 1))


def build_padded_ratio128_sparse_indices_rows(
    positions: torch.Tensor,
    *,
    bucket_width: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Vectorized ``build_padded_ratio128_sparse_indices`` over ``positions[B]``.

    Row ``b`` of the result equals the scalar builder at ``positions[b]``
    (identical closed forms), so at equal positions the padded index set --
    and therefore the fixed-width masked sparse core -- matches the family
    plan bitwise.
    """

    columns = torch.arange(
        bucket_width, dtype=torch.int64, device=positions.device
    )
    position = positions.view(-1, 1)
    position_mod = position.remainder(RATIO128)
    full_raw = (columns + position_mod + 1).remainder(RATIO128)
    partial_raw = torch.where(
        columns <= position,
        columns.expand_as(full_raw),
        torch.full_like(full_raw, -1),
    )
    raw = torch.where(position >= WINDOW_SIZE - 1, full_raw, partial_raw)
    completed = torch.div(position, RATIO128, rounding_mode="floor")
    completed = completed + position_mod.eq(RATIO128 - 1).to(torch.int64)
    compressed_column = columns - WINDOW_SIZE
    compressed = torch.where(
        compressed_column < completed,
        columns.expand_as(full_raw),
        torch.full_like(full_raw, -1),
    )
    row = torch.where(columns < WINDOW_SIZE, raw, compressed)
    row = torch.where(position >= 0, row, torch.full_like(row, -1))
    out.copy_(row.unsqueeze(1).to(torch.int32))
    return out


# --------------------------------------------------------------------------
# per-layer row-position workspaces


@dataclass
class WindowRowWS:
    gather_indices: torch.Tensor  # [B, 1, 128] int64
    batch_indices: torch.Tensor  # [B, 1, 128] int64
    valid_mask: torch.Tensor | None = None  # unused; window ring is full


@dataclass
class Ratio128RowWS:
    bucket_width: int
    topk_indices: torch.Tensor  # [B, 1, W] int32
    gather_indices: torch.Tensor  # [B, 1, W] int64
    valid_mask: torch.Tensor  # [B, 1, W] bool
    batch_indices: torch.Tensor  # [B, 1, W] int64


@dataclass
class Ratio4RowWS:
    candidate_width: int
    index_topk_count: int
    window_columns: torch.Tensor  # [128] int64
    compressed_columns: torch.Tensor  # [W] int64
    topk_indices: torch.Tensor  # [B, 1, 128 + topk] int64
    batch_indices: torch.Tensor  # [B, 1, 128 + topk] int64
    shadow: dict[str, torch.Tensor] = field(default_factory=dict)


_RATIO4_SHADOW_NAMES = (
    "main_kv_state",
    "main_score_state",
    "index_kv_state",
    "index_score_state",
    "_main_state_positions",
    "_index_state_positions",
)


def build_layer_row_ws(
    attention: Any,
    *,
    batch: int,
    stop_position: int,
    ratio128_bucket_width: int,
) -> WindowRowWS | Ratio128RowWS | Ratio4RowWS:
    device = attention.state.device
    if isinstance(attention, WindowTorchAttention):
        shape = (batch, 1, WINDOW_SIZE)
        return WindowRowWS(
            gather_indices=torch.zeros(shape, dtype=torch.int64, device=device),
            batch_indices=torch.arange(batch, dtype=torch.int64, device=device)
            .view(batch, 1, 1)
            .expand(shape)
            .contiguous(),
        )
    if isinstance(attention, Ratio128TorchAttention):
        shape = (batch, 1, ratio128_bucket_width)
        return Ratio128RowWS(
            bucket_width=ratio128_bucket_width,
            topk_indices=torch.full(shape, -1, dtype=torch.int32, device=device),
            gather_indices=torch.zeros(shape, dtype=torch.int64, device=device),
            valid_mask=torch.zeros(shape, dtype=torch.bool, device=device),
            batch_indices=torch.arange(batch, dtype=torch.int64, device=device)
            .view(batch, 1, 1)
            .expand(shape)
            .contiguous(),
        )
    if isinstance(attention, Ratio4TorchAttention):
        cfg = attention.config
        candidate_width = stop_position // RATIO4
        if candidate_width > attention.state.compressed_capacity:
            raise SpecDecodeError("ratio-4 candidate width exceeds capacity")
        total = WINDOW_SIZE + cfg.index_topk
        state = attention.state
        shadow = {
            name: torch.empty_like(getattr(state, name))
            for name in _RATIO4_SHADOW_NAMES
        }
        return Ratio4RowWS(
            candidate_width=candidate_width,
            index_topk_count=cfg.index_topk,
            window_columns=torch.arange(
                WINDOW_SIZE, dtype=torch.int64, device=device
            ),
            compressed_columns=torch.arange(
                candidate_width, dtype=torch.int64, device=device
            ),
            topk_indices=torch.empty(
                batch, 1, total, dtype=torch.int64, device=device
            ),
            batch_indices=torch.arange(batch, dtype=torch.int64, device=device)
            .view(batch, 1, 1)
            .expand(batch, 1, total)
            .contiguous(),
            shadow=shadow,
        )
    raise SpecDecodeError(f"unsupported attention type {type(attention)!r}")


# --------------------------------------------------------------------------
# per-layer row-position forwards (mirrors of forward_stateful_decode_tensor)


def _output_projection(
    attention: Any, output: torch.Tensor, row_freqs: torch.Tensor
) -> torch.Tensor:
    cfg = attention.config
    weights = attention.weights
    output[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
        output[..., -cfg.rope_dim :], row_freqs, inverse=True
    )
    grouped = output.reshape(
        output.shape[0],
        1,
        cfg.o_groups,
        cfg.num_heads * cfg.head_dim // cfg.o_groups,
    )
    wo_a = weights.wo_a.reshape(
        cfg.o_groups,
        cfg.o_lora_rank,
        cfg.num_heads * cfg.head_dim // cfg.o_groups,
    )
    projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
    return F.linear(projected.flatten(2), weights.wo_b)


def window_rowpos_forward(
    attention: WindowTorchAttention,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    ws: WindowRowWS,
) -> torch.Tensor:
    """Per-row-position mirror of ``WindowTorchAttention.forward_stateful_decode_tensor``."""

    cfg = attention.config
    weights = attention.weights
    state = attention.state
    batch = hidden.shape[0]
    row_freqs = gather_row_freqs(attention.freqs_cis, positions)

    query_lora = rms_norm(
        F.linear(hidden, weights.wq_a), weights.q_norm, eps=cfg.norm_eps
    )
    query = F.linear(query_lora, weights.wq_b).reshape(
        batch, 1, cfg.num_heads, cfg.head_dim
    )
    query *= torch.rsqrt(query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps)
    query[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
        query[..., -cfg.rope_dim :], row_freqs
    )

    raw_latent = rms_norm(
        F.linear(hidden, weights.wkv), weights.kv_norm, eps=cfg.norm_eps
    )
    raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
        raw_latent[..., -cfg.rope_dim :], row_freqs
    )
    raw_latent[..., : -cfg.rope_dim] = attention._nope_control(
        raw_latent[..., : -cfg.rope_dim]
    )

    slot = positions.remainder(WINDOW_SIZE)
    scatter_state_rows_(
        state.latent, slot, quantize_latent_rows(raw_latent, state.latent.dtype)
    )
    if state.latent_rope is not None:
        scatter_state_rows_(
            state.latent_rope, slot, raw_latent[..., -LATENT_ROPE_DIM:].contiguous()
        )
    scatter_position_rows_(state._raw_positions, slot, positions)
    state._next_position.copy_(positions + 1)

    chronological = (
        torch.arange(WINDOW_SIZE, dtype=torch.int64, device=positions.device)
        + slot.view(-1, 1)
        + 1
    ).remainder(WINDOW_SIZE)
    ws.gather_indices.copy_(chronological.unsqueeze(1))
    output = _window_sparse_decode_prevalidated(
        query,
        state.latent,
        weights.attn_sink,
        ws,
        cfg.head_dim**-0.5,
        latent_rope=state.latent_rope,
    )
    return _output_projection(attention, output, row_freqs)


def ratio128_rowpos_forward(
    attention: Ratio128TorchAttention,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    ws: Ratio128RowWS,
) -> torch.Tensor:
    """Per-row-position, masked-boundary mirror of the ratio-128 stateful step.

    The compressor pooling + finalization runs every step; the compressed-row
    commit happens only where ``positions % 128 == 127`` (per row).  At a
    boundary the committed row is computed from exactly the same state as the
    family boundary graph, so it is bitwise identical.
    """

    cfg = attention.config
    weights = attention.weights
    state = attention.state
    batch = hidden.shape[0]
    row_freqs = gather_row_freqs(attention.freqs_cis, positions)
    slot = positions.remainder(RATIO128)
    compressor_ape = weights.compressor_ape.index_select(0, slot)  # [B, 512]

    query_lora = rms_norm(
        F.linear(hidden, weights.wq_a), weights.q_norm, eps=cfg.norm_eps
    )
    query = F.linear(query_lora, weights.wq_b).reshape(
        batch, 1, cfg.num_heads, cfg.head_dim
    )
    query *= torch.rsqrt(query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps)
    query[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
        query[..., -cfg.rope_dim :], row_freqs
    )

    raw_latent = rms_norm(
        F.linear(hidden, weights.wkv), weights.kv_norm, eps=cfg.norm_eps
    )
    raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
        raw_latent[..., -cfg.rope_dim :], row_freqs
    )
    raw_latent[..., : -cfg.rope_dim] = attention._nope_control(
        raw_latent[..., : -cfg.rope_dim]
    )
    projected_kv = F.linear(hidden.float(), weights.compressor_wkv)
    projected_score = F.linear(hidden.float(), weights.compressor_wgate)
    adjusted_score = projected_score[:, 0] + compressor_ape

    # ring + compressor-state writes (assignment slots, per row)
    scatter_state_rows_(
        state.raw, slot, quantize_latent_rows(raw_latent, state.latent.dtype)
    )
    raw_rope = state.raw_rope
    if raw_rope is not None:
        scatter_state_rows_(
            raw_rope, slot, raw_latent[..., -LATENT_ROPE_DIM:].contiguous()
        )
    scatter_state_rows_(state.kv_state, slot, projected_kv)
    scatter_state_rows_(state.score_state, slot, adjusted_score.unsqueeze(1))
    scatter_position_rows_(state._raw_positions, slot, positions)
    scatter_position_rows_(state._state_positions, slot, positions)

    # masked boundary commit: pool always, commit where phase == 127
    boundary = slot.eq(RATIO128 - 1)
    pooled = (state.kv_state * state.score_state.softmax(dim=1)).sum(
        dim=1, keepdim=True
    )
    group_start = positions + 1 - RATIO128
    finalized = rms_norm(
        pooled.to(torch.bfloat16), weights.compressor_norm, eps=cfg.norm_eps
    )
    group_freqs = gather_row_freqs(attention.freqs_cis, group_start.clamp_min(0))
    finalized[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
        finalized[..., -cfg.rope_dim :], group_freqs
    )
    finalized[..., : -cfg.rope_dim] = attention._nope_control(
        finalized[..., : -cfg.rope_dim]
    )
    finalized = finalized.contiguous()
    compressed_row = torch.div(positions, RATIO128, rounding_mode="floor")
    masked_scatter_state_rows_(
        state.compressed,
        compressed_row,
        quantize_latent_rows(finalized, state.latent.dtype),
        boundary,
    )
    compressed_rope = state.compressed_rope
    if compressed_rope is not None:
        masked_scatter_state_rows_(
            compressed_rope,
            compressed_row,
            finalized[..., -LATENT_ROPE_DIM:].contiguous(),
            boundary,
        )
    current_starts = state._compressed_group_starts.gather(
        1, compressed_row.view(-1, 1)
    )
    state._compressed_group_starts.scatter_(
        1,
        compressed_row.view(-1, 1),
        torch.where(boundary.view(-1, 1), group_start.view(-1, 1), current_starts),
    )
    state._compressed_count.copy_(
        torch.div(positions + 1, RATIO128, rounding_mode="floor")
    )
    state._next_position.copy_(positions + 1)

    # fixed-width masked sparse attention (family bucket width)
    build_padded_ratio128_sparse_indices_rows(
        positions, bucket_width=ws.bucket_width, out=ws.topk_indices
    )
    ws.valid_mask.copy_(ws.topk_indices.ge(0))
    ws.gather_indices.copy_(ws.topk_indices)
    ws.gather_indices.clamp_min_(0)
    output = _torch_sparse_decode_padded_prevalidated(
        query,
        state.latent,
        weights.attn_sink,
        ws,
        cfg.head_dim**-0.5,
        latent_rope=state.latent_rope,
    )
    return _output_projection(attention, output, row_freqs)


def ratio4_rowpos_forward(
    attention: Ratio4TorchAttention,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    ws: Ratio4RowWS,
) -> torch.Tensor:
    """Per-row-position, masked-boundary mirror of the ratio-4 stateful step."""

    cfg = attention.config
    weights = attention.weights
    state = attention.state
    batch = hidden.shape[0]
    row_freqs = gather_row_freqs(attention.freqs_cis, positions)
    phase = positions.remainder(RATIO4)
    raw_slot = positions.remainder(WINDOW_SIZE)
    overlap_slot = phase + RATIO4
    main_ape = weights.compressor_ape.index_select(0, phase)  # [B, 1024]
    index_ape = weights.index_compressor_ape.index_select(0, phase)  # [B, 256]

    query_lora = rms_norm(
        F.linear(hidden, weights.wq_a), weights.q_norm, eps=cfg.norm_eps
    )
    query = F.linear(query_lora, weights.wq_b).reshape(
        batch, 1, cfg.num_heads, cfg.head_dim
    )
    query *= torch.rsqrt(query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps)
    query[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
        query[..., -cfg.rope_dim :], row_freqs
    )

    raw_latent = rms_norm(F.linear(hidden, weights.wkv), weights.kv_norm, eps=cfg.norm_eps)
    raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
        raw_latent[..., -cfg.rope_dim :], row_freqs
    )
    raw_latent[..., : -cfg.rope_dim] = fp8_quant_dequant(
        raw_latent[..., : -cfg.rope_dim], group_size=64
    )

    main_projected = F.linear(hidden.float(), weights.compressor_wkv)
    main_score = F.linear(hidden.float(), weights.compressor_wgate)
    main_adjusted = main_score[:, 0] + main_ape
    index_projected = F.linear(hidden.float(), weights.index_compressor_wkv)
    index_score = F.linear(hidden.float(), weights.index_compressor_wgate)
    index_adjusted = index_score[:, 0] + index_ape

    # ring + overlap-state writes
    scatter_state_rows_(
        state.raw, raw_slot, quantize_latent_rows(raw_latent, state.latent.dtype)
    )
    raw_rope = state.raw_rope
    if raw_rope is not None:
        scatter_state_rows_(
            raw_rope, raw_slot, raw_latent[..., -LATENT_ROPE_DIM:].contiguous()
        )
    scatter_state_rows_(state.main_kv_state, overlap_slot, main_projected)
    scatter_state_rows_(
        state.main_score_state, overlap_slot, main_adjusted.unsqueeze(1)
    )
    scatter_state_rows_(state.index_kv_state, overlap_slot, index_projected)
    scatter_state_rows_(
        state.index_score_state, overlap_slot, index_adjusted.unsqueeze(1)
    )
    scatter_position_rows_(state._raw_positions, raw_slot, positions)
    scatter_position_rows_(state._main_state_positions, overlap_slot, positions)
    scatter_position_rows_(state._index_state_positions, overlap_slot, positions)

    # masked boundary: overlap pooling always, commit + shift where phase == 3
    boundary = phase.eq(RATIO4 - 1)
    group_start = positions + 1 - RATIO4
    group_freqs = gather_row_freqs(attention.freqs_cis, group_start.clamp_min(0))

    def pool_and_finalize(
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        output_dim: int,
        norm_weight: torch.Tensor,
        index_form: bool,
    ) -> torch.Tensor:
        values = torch.cat(
            (
                kv_state[:, :RATIO4, :output_dim],
                kv_state[:, RATIO4:, output_dim:],
            ),
            dim=1,
        )
        scores = torch.cat(
            (
                score_state[:, :RATIO4, :output_dim],
                score_state[:, RATIO4:, output_dim:],
            ),
            dim=1,
        )
        pooled = (values * scores.softmax(dim=1)).sum(dim=1, keepdim=True)
        value = rms_norm(pooled.to(torch.bfloat16), norm_weight, eps=cfg.norm_eps)
        value[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
            value[..., -cfg.rope_dim :], group_freqs
        )
        if index_form:
            return fp4_quant_dequant(hadamard_transform(value)).contiguous()
        value[..., : -cfg.rope_dim] = fp8_quant_dequant(
            value[..., : -cfg.rope_dim], group_size=64
        )
        return value.contiguous()

    finalized_main = pool_and_finalize(
        state.main_kv_state,
        state.main_score_state,
        R4_LATENT_DIM,
        weights.compressor_norm,
        index_form=False,
    )
    finalized_index = pool_and_finalize(
        state.index_kv_state,
        state.index_score_state,
        INDEX_DIM,
        weights.index_compressor_norm,
        index_form=True,
    )
    compressed_row = torch.div(positions, RATIO4, rounding_mode="floor")
    masked_scatter_state_rows_(
        state.compressed,
        compressed_row,
        quantize_latent_rows(finalized_main, state.latent.dtype),
        boundary,
    )
    compressed_rope = state.compressed_rope
    if compressed_rope is not None:
        masked_scatter_state_rows_(
            compressed_rope,
            compressed_row,
            finalized_main[..., -LATENT_ROPE_DIM:].contiguous(),
            boundary,
        )
    masked_scatter_state_rows_(
        state.indexer_kv,
        compressed_row,
        quantize_latent_rows(finalized_index, state.indexer_kv.dtype),
        boundary,
    )
    current_starts = state._compressed_group_starts.gather(
        1, compressed_row.view(-1, 1)
    )
    state._compressed_group_starts.scatter_(
        1,
        compressed_row.view(-1, 1),
        torch.where(boundary.view(-1, 1), group_start.view(-1, 1), current_starts),
    )
    boundary_rows = boundary.view(-1, 1, 1)
    for tensor in (
        state.main_kv_state,
        state.main_score_state,
        state.index_kv_state,
        state.index_score_state,
    ):
        tensor[:, :RATIO4].copy_(
            torch.where(boundary_rows, tensor[:, RATIO4:], tensor[:, :RATIO4])
        )
    boundary_meta = boundary.view(-1, 1)
    for tensor in (state._main_state_positions, state._index_state_positions):
        tensor[:, :RATIO4].copy_(
            torch.where(boundary_meta, tensor[:, RATIO4:], tensor[:, :RATIO4])
        )
    state._compressed_count.copy_(
        torch.div(positions + 1, RATIO4, rounding_mode="floor")
    )
    state._next_position.copy_(positions + 1)

    # indexer scoring + top-k (per-row visibility mask)
    index_query = F.linear(query_lora, weights.index_wq_b).reshape(
        batch, 1, cfg.index_n_heads, cfg.index_head_dim
    )
    index_query[..., -cfg.rope_dim :] = apply_rotary_emb_rows(
        index_query[..., -cfg.rope_dim :], row_freqs
    )
    index_query = fp4_quant_dequant(hadamard_transform(index_query))
    index_weights = F.linear(hidden, weights.index_weights_proj) * (
        cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5
    )
    index_kv = state.indexer_kv[:, : ws.candidate_width]
    scores = torch.einsum("bshd,btd->bsht", index_query.float(), index_kv.float())
    scores = scores.relu_().mul_(index_weights.float().unsqueeze(-1)).sum(dim=2)
    compressed_after = torch.div(positions + 1, RATIO4, rounding_mode="floor")
    visible = ws.compressed_columns.view(1, -1) < compressed_after.view(-1, 1)
    scores.masked_fill_(
        ~visible.view(batch, 1, ws.candidate_width), float("-inf")
    )
    compressed_indices = scores.topk(ws.index_topk_count, dim=-1).indices
    window = (
        ws.window_columns.view(1, -1) + raw_slot.view(-1, 1) + 1
    ).remainder(WINDOW_SIZE)
    ws.topk_indices[..., :WINDOW_SIZE].copy_(window.unsqueeze(1))
    ws.topk_indices[..., WINDOW_SIZE:].copy_(compressed_indices + WINDOW_SIZE)

    # sparse core (17th-vertical single-FP32-materialization form)
    selected = state.latent[ws.batch_indices, ws.topk_indices].float()
    if state.latent_rope is not None:
        selected[..., -LATENT_ROPE_DIM:] = state.latent_rope[
            ws.batch_indices, ws.topk_indices
        ]
    attention_scores = torch.einsum(
        "bshd,bskd->bshk", query.float(), selected
    ) * (cfg.head_dim**-0.5)
    sink = weights.attn_sink.float().view(1, 1, cfg.num_heads, 1)
    maximum = torch.maximum(attention_scores.amax(dim=-1, keepdim=True), sink)
    exponent = attention_scores.sub_(maximum).exp_()
    denominator = exponent.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
    probabilities = exponent.div_(denominator)
    sparse_output = torch.einsum(
        "bshk,bskd->bshd", probabilities, selected
    ).to(query.dtype)
    return _output_projection(attention, sparse_output, row_freqs)


def rowpos_attention_forward(
    block: DirectDecodeBlock,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    ws: WindowRowWS | Ratio128RowWS | Ratio4RowWS,
) -> torch.Tensor:
    if block.compression_ratio == 0:
        return window_rowpos_forward(block.attention, hidden, positions, ws)
    if block.compression_ratio == 4:
        return ratio4_rowpos_forward(block.attention, hidden, positions, ws)
    return ratio128_rowpos_forward(block.attention, hidden, positions, ws)


# --------------------------------------------------------------------------
# stage-level plan and passes


@dataclass
class SpecStagePlan:
    """One lane's fixed-address row-position workspace on one stage."""

    start_position: int
    stop_position: int
    batch_size: int
    ratio128_bucket_width: int
    positions: torch.Tensor  # [B] int64: pass-A write position
    advance: torch.Tensor  # [B] int64: applied at pass-A head
    accept: torch.Tensor  # [B] int64 (1 accept / 0 reject), previous round
    input_residual_buffer: torch.Tensor
    input_ids_buffer: torch.Tensor
    output_buffer: torch.Tensor
    layer_ws: tuple[Any, ...]
    moe_slot_a: int
    moe_slot_b: int

    @property
    def resident_bytes(self) -> int:
        total = 0
        for tensor in (
            self.positions,
            self.advance,
            self.accept,
            self.input_residual_buffer,
            self.input_ids_buffer,
            self.output_buffer,
        ):
            total += tensor.numel() * tensor.element_size()
        for ws in self.layer_ws:
            for value in vars(ws).values():
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
                elif isinstance(value, dict):
                    total += sum(
                        v.numel() * v.element_size() for v in value.values()
                    )
        return total


def prepare_spec_stage_plan(
    stage: TP4DecodeStage,
    *,
    batch_size: int,
    start_position: int,
    stop_position: int,
    moe_slot_a: int,
    moe_slot_b: int,
    device: torch.device,
) -> SpecStagePlan:
    from .stateful_decode import ratio128_sparse_bucket_width

    bucket_width = ratio128_sparse_bucket_width(start_position, stop_position - 1)
    layer_ws = tuple(
        build_layer_row_ws(
            block.attention,
            batch=batch_size,
            stop_position=stop_position,
            ratio128_bucket_width=bucket_width,
        )
        for block in stage.blocks
    )
    residual_shape = (batch_size, 1, BLOCK_HC_MULT, BLOCK_HIDDEN_SIZE)
    return SpecStagePlan(
        start_position=start_position,
        stop_position=stop_position,
        batch_size=batch_size,
        ratio128_bucket_width=bucket_width,
        positions=torch.full(
            (batch_size,), start_position, dtype=torch.int64, device=device
        ),
        advance=torch.zeros(batch_size, dtype=torch.int64, device=device),
        accept=torch.ones(batch_size, dtype=torch.int64, device=device),
        input_residual_buffer=torch.empty(
            residual_shape, dtype=torch.bfloat16, device=device
        ),
        input_ids_buffer=torch.zeros(
            (batch_size, 1), dtype=torch.int64, device=device
        ),
        output_buffer=torch.empty(
            residual_shape, dtype=torch.bfloat16, device=device
        ),
        layer_ws=layer_ws,
        moe_slot_a=moe_slot_a,
        moe_slot_b=moe_slot_b,
    )


def spec_round_head(stage: TP4DecodeStage, plan: SpecStagePlan) -> None:
    """Pass-A graph head: advance positions; restore ratio-4 shadows on reject."""

    plan.positions.add_(plan.advance)
    keep = plan.accept.ne(0)
    keep_rows = keep.view(-1, 1, 1)
    keep_meta = keep.view(-1, 1)
    for block, ws in zip(stage.blocks, plan.layer_ws, strict=True):
        if not isinstance(ws, Ratio4RowWS):
            continue
        state = block.attention.state
        for name in _RATIO4_SHADOW_NAMES:
            live = getattr(state, name)
            mask = keep_meta if live.ndim == 2 else keep_rows
            live.copy_(torch.where(mask, live, ws.shadow[name]))


def spec_snapshot(stage: TP4DecodeStage, plan: SpecStagePlan) -> None:
    """Pass-A graph tail: shadow the post-pass-A ratio-4 overlap states."""

    for block, ws in zip(stage.blocks, plan.layer_ws, strict=True):
        if not isinstance(ws, Ratio4RowWS):
            continue
        state = block.attention.state
        for name in _RATIO4_SHADOW_NAMES:
            ws.shadow[name].copy_(getattr(state, name))


def forward_spec_stage(
    stage: TP4DecodeStage,
    plan: SpecStagePlan,
    *,
    pass_b: bool,
    moe_slot_override: int | None = None,
) -> torch.Tensor:
    """One row-position pass over the stage (graph body).

    Pass A additionally runs the round head (advance + masked restore) before
    any layer and the shadow snapshot after the last layer.  The layer chain
    mirrors ``TP4DecodeStage._forward_stateful_fused_chain`` exactly (same HC
    boundary ops, same MoE call sites), with the family attention step
    replaced by its row-position mirror.
    """

    if not pass_b:
        spec_round_head(stage, plan)
        positions = plan.positions
        moe_slot = plan.moe_slot_a
    else:
        positions = plan.positions + 1
        moe_slot = plan.moe_slot_b
    if moe_slot_override is not None:
        moe_slot = moe_slot_override
    backend = stage.hc_boundary_backend
    blocks = stage.blocks
    residual = plan.input_residual_buffer
    last_index = len(blocks) - 1
    current_residual = residual
    attention_hidden, post, comb = blocks[0].prepare_attention(residual)
    output: torch.Tensor | None = None
    for index, (block, ws) in enumerate(
        zip(blocks, plan.layer_ws, strict=True)
    ):
        branch_output = rowpos_attention_forward(
            block, attention_hidden, positions, ws
        )
        after_attention, ffn_hidden, ffn_post, ffn_comb = block.ffn_boundary(
            branch_output,
            current_residual,
            post,
            comb,
            backend=backend,
        )
        moe_arguments: dict[str, Any] = {"slot": moe_slot}
        if block.route_kind == "hash":
            moe_arguments["input_ids_local"] = plan.input_ids_buffer
        moe_output = block.moe.forward_tensor(ffn_hidden, **moe_arguments)
        if index == last_index:
            output = hc_post(moe_output, after_attention, ffn_post, ffn_comb)
        else:
            current_residual, attention_hidden, post, comb = blocks[
                index + 1
            ].attention_boundary(
                moe_output,
                after_attention,
                ffn_post,
                ffn_comb,
                backend=backend,
            )
    if output is None:
        raise SpecDecodeError("spec stage chain produced no output")
    plan.output_buffer.copy_(output)
    if not pass_b:
        spec_snapshot(stage, plan)
    return plan.output_buffer


# --------------------------------------------------------------------------
# MTP block (tail stage) row-position forward


@dataclass
class MTPSpecPlan:
    """Per-lane MTP workspace: bridge inputs, window WS, draft output."""

    batch_size: int
    input_residual_buffer: torch.Tensor  # [B, 1, 4, 4096] bf16 (pre-head HC)
    input_ids_buffer: torch.Tensor  # [B, 1] int64 (committed next token)
    draft_buffer: torch.Tensor  # [B] int64 (argmax draft out)
    window_ws: WindowRowWS
    moe_slot_a: int
    moe_slot_b: int


def prepare_mtp_spec_plan(
    mtp_lane: Any,
    *,
    batch_size: int,
    moe_slot_a: int,
    moe_slot_b: int,
    device: torch.device,
) -> MTPSpecPlan:
    ws = build_layer_row_ws(
        mtp_lane.attention,
        batch=batch_size,
        stop_position=mtp_lane.material.max_seq_len,
        ratio128_bucket_width=WINDOW_SIZE,  # unused for window layers
    )
    return MTPSpecPlan(
        batch_size=batch_size,
        input_residual_buffer=torch.empty(
            (batch_size, 1, BLOCK_HC_MULT, BLOCK_HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
        ),
        input_ids_buffer=torch.zeros(
            (batch_size, 1), dtype=torch.int64, device=device
        ),
        draft_buffer=torch.zeros(batch_size, dtype=torch.int64, device=device),
        window_ws=ws,
        moe_slot_a=moe_slot_a,
        moe_slot_b=moe_slot_b,
    )


def forward_mtp_spec(
    mtp_lane: Any,
    plan: MTPSpecPlan,
    positions: torch.Tensor,
    *,
    second: bool,
    moe_slot_override: int | None = None,
) -> torch.Tensor:
    """Graph-safe MTP block forward at per-row positions.

    ``positions`` is the main lane's pass-A position vector; MTP pass 1 runs
    at ``positions`` (the pair committed by pass A), pass 2 at
    ``positions + 1``.  The dataflow mirrors ``MTPLane.forward`` (bridge ->
    HC block core with window attention -> own hc_head/norm -> shared head)
    with the attention step replaced by the row-position window mirror and
    the trailing argmax written into ``plan.draft_buffer``.
    """

    material = mtp_lane.material
    positions_eff = positions + 1 if second else positions
    moe_slot = plan.moe_slot_b if second else plan.moe_slot_a
    if moe_slot_override is not None:
        moe_slot = moe_slot_override

    embedded = F.embedding(plan.input_ids_buffer, mtp_lane.embed_weight)
    embedded = rms_norm(embedded, material.bridge.enorm, eps=material.norm_eps)
    normed_hidden = rms_norm(
        plan.input_residual_buffer, material.bridge.hnorm, eps=material.norm_eps
    )
    residual = F.linear(embedded, material.bridge.e_proj).unsqueeze(2) + F.linear(
        normed_hidden, material.bridge.h_proj
    )

    hc = material.raw_block.hyper_connection
    hidden, post, comb = hc_pre(
        residual,
        hc.attn_fn,
        hc.attn_scale,
        hc.attn_base,
        norm_eps=material.norm_eps,
        sinkhorn_iters=material.sinkhorn_iters,
        hc_eps=material.hc_eps,
    )
    hidden = rms_norm(hidden, material.raw_block.attn_norm, eps=material.norm_eps)
    branch = window_rowpos_forward(
        mtp_lane.attention, hidden, positions_eff, plan.window_ws
    )
    residual = hc_post(branch, residual, post, comb)
    hidden, post, comb = hc_pre(
        residual,
        hc.ffn_fn,
        hc.ffn_scale,
        hc.ffn_base,
        norm_eps=material.norm_eps,
        sinkhorn_iters=material.sinkhorn_iters,
        hc_eps=material.hc_eps,
    )
    hidden = rms_norm(hidden, material.raw_block.ffn_norm, eps=material.norm_eps)
    moe_output = material.moe.forward_tensor(
        hidden, input_ids_local=None, slot=moe_slot
    )
    residual = hc_post(moe_output, residual, post, comb)

    collapsed = hc_head_collapse_tensors(
        residual,
        hc_head_fn=material.bridge.hc_head_fn,
        hc_head_base=material.bridge.hc_head_base,
        hc_head_scale=material.bridge.hc_head_scale,
        norm_eps=material.norm_eps,
        hc_eps=material.hc_eps,
    )
    value = collapsed.float()
    value = value * torch.rsqrt(
        value.square().mean(dim=-1, keepdim=True) + material.norm_eps
    )
    normed = (material.bridge.norm * value).to(collapsed.dtype)
    logits = F.linear(normed[:, -1].float(), mtp_lane.head_weight)
    plan.draft_buffer.copy_(logits.argmax(dim=-1))
    return plan.draft_buffer


__all__ = [
    "MTPSpecPlan",
    "Ratio128RowWS",
    "Ratio4RowWS",
    "SpecDecodeError",
    "SpecStagePlan",
    "WindowRowWS",
    "apply_rotary_emb_rows",
    "build_layer_row_ws",
    "build_padded_ratio128_sparse_indices_rows",
    "forward_mtp_spec",
    "forward_spec_stage",
    "gather_row_freqs",
    "masked_scatter_state_rows_",
    "prepare_mtp_spec_plan",
    "prepare_spec_stage_plan",
    "ratio128_rowpos_forward",
    "ratio4_rowpos_forward",
    "rowpos_attention_forward",
    "scatter_position_rows_",
    "scatter_state_rows_",
    "spec_round_head",
    "spec_snapshot",
    "window_rowpos_forward",
]
