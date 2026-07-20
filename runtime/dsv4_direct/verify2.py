"""Fused two-token verify-step decode and state snapshot/rollback helpers.

MTP draft-verify (standard speculative decoding, one draft token): the main
model verifies ``[next_token, draft]`` as one two-position decode step.  This
module supplies

- **fused seqlen-2 decode steps** for the three eager full-position attention
  lanes (window ``WindowTorchAttention``, ratio-128 ``Ratio128TorchAttention``,
  ratio-4 ``Ratio4FullPositionAttention``).  All hidden-side GEMMs (q/kv/
  compressor/indexer projections, output projection) run once over both
  tokens -- at batch 1 these weight-bound GEMMs cost the same as one token,
  which is where the MTP speedup comes from -- while cache writes, compressor
  ingestion, top-k selection, and the sparse core run per position in
  reference order (position ``p+1`` must see position ``p``'s KV, and the
  ring/compressed visibility differs per position).  The per-position operator
  chain is copied from each lane's verified single-token decode path.

- **state snapshot/rollback**: on draft rejection the second position's state
  mutations must be undone.  Window rings and the ratio-128 compressor state
  heal by re-feeding the corrected token (pure assignments), but the ratio-4
  overlap compressor's boundary shift (``kv_state[:ratio] = kv_state[ratio:]``,
  reference model.py:353-354) destroys the previous-window rows, so rollback
  restores a pre-step snapshot of every mutable state tensor (uniformly for
  all layer kinds; B=1 correctness path, not a performance path).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .attention import (
    Ratio128TorchAttention,
    apply_rotary_emb,
    compressed_topk_indices,
    fp8_quant_dequant,
    rms_norm,
    torch_sparse_attention,
    window_topk_indices,
)
from .ratio4_attention import fp4_quant_dequant, hadamard_transform
from .ratio4_fullpos import Ratio4FullPositionAttention
from .static_kv import LATENT_ROPE_DIM, WINDOW_SIZE
from .static_ratio4_kv import COMPRESS_RATIO, INDEX_DIM, LATENT_DIM
from .window_attention import WindowTorchAttention


class Verify2Error(ValueError):
    """Raised when a fused verify-2 step is driven off-contract."""


# --------------------------------------------------------------------------
# state snapshot / rollback


_RATIO4_FULLPOS_TENSORS = (
    "raw",
    "compressed",
    "raw_rope",
    "compressed_rope",
    "indexer_kv",
    "main_kv_state",
    "main_score_state",
    "index_kv_state",
    "index_score_state",
)
_RATIO4_FULLPOS_SCALARS = ("next_position", "compressed_count")


def snapshot_decode_state(state: Any) -> dict[str, Any]:
    """Clone every mutable tensor (plus scalar cursors) of one lane state."""

    if hasattr(state, "_owned_tensor_items"):
        # StaticWindowKV / StaticLayerKV: tensors incl. device cursors.
        return {
            "kind": "owned",
            "tensors": {
                name: tensor.clone() for name, tensor in state._owned_tensor_items()
            },
        }
    if isinstance(state, Ratio4FullPositionAttention):
        tensors = {}
        for name in _RATIO4_FULLPOS_TENSORS:
            value = getattr(state, name)
            if value is not None:
                tensors[name] = value.clone()
        return {
            "kind": "ratio4_fullpos",
            "tensors": tensors,
            "scalars": {
                name: getattr(state, name) for name in _RATIO4_FULLPOS_SCALARS
            },
        }
    raise Verify2Error(f"unsupported decode state type {type(state)!r}")


def restore_decode_state(state: Any, snapshot: dict[str, Any]) -> None:
    if snapshot["kind"] == "owned":
        current = dict(state._owned_tensor_items())
        if set(current) != set(snapshot["tensors"]):
            raise Verify2Error("state snapshot does not match the live tensor set")
        for name, saved in snapshot["tensors"].items():
            current[name].copy_(saved)
        return
    if snapshot["kind"] == "ratio4_fullpos":
        for name, saved in snapshot["tensors"].items():
            getattr(state, name).copy_(saved)
        for name, saved in snapshot["scalars"].items():
            setattr(state, name, saved)
        return
    raise Verify2Error(f"unknown snapshot kind {snapshot['kind']!r}")


# --------------------------------------------------------------------------
# fused two-token decode steps


def _validate_two_token(hidden: torch.Tensor, hidden_size: int) -> tuple[int, int]:
    if hidden.ndim != 3 or hidden.shape[1] != 2 or hidden.shape[-1] != hidden_size:
        raise Verify2Error(
            f"verify-2 hidden must be [batch, 2, {hidden_size}], "
            f"got {tuple(hidden.shape)}"
        )
    if hidden.dtype != torch.bfloat16:
        raise Verify2Error("verify-2 hidden must be BF16")
    return hidden.shape[0], hidden.shape[1]


def _fused_output_projection(
    per_token_outputs: list[torch.Tensor],
    frequencies: torch.Tensor,
    *,
    cfg: Any,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
) -> torch.Tensor:
    output = torch.cat(per_token_outputs, dim=1)
    output[..., -cfg.rope_dim :] = apply_rotary_emb(
        output[..., -cfg.rope_dim :], frequencies, inverse=True
    )
    grouped = output.reshape(
        output.shape[0],
        output.shape[1],
        cfg.o_groups,
        cfg.num_heads * cfg.head_dim // cfg.o_groups,
    )
    reshaped_wo_a = wo_a.reshape(
        cfg.o_groups,
        cfg.o_lora_rank,
        cfg.num_heads * cfg.head_dim // cfg.o_groups,
    )
    projected = torch.einsum("bsgd,grd->bsgr", grouped, reshaped_wo_a)
    return F.linear(projected.flatten(2), wo_b)


def window_decode2(
    attention: WindowTorchAttention,
    hidden: torch.Tensor,
    *,
    start_pos: int,
    snapshot_out: list[tuple[Any, dict[str, Any]]] | None = None,
) -> torch.Tensor:
    """Two-position window decode (per-position mirror of ``__call__`` decode)."""

    cfg = attention.config
    weights = attention.weights
    state = attention.state
    batch, _ = _validate_two_token(hidden, cfg.hidden_size)
    if start_pos != state.next_position or start_pos <= 0:
        raise Verify2Error(
            f"window verify-2 start_pos {start_pos} != state position "
            f"{state.next_position} (or is a prefill position)"
        )
    if start_pos + 2 > cfg.max_seq_len:
        raise Verify2Error("window verify-2 exceeds static KV capacity")
    frequencies = attention.freqs_cis[start_pos : start_pos + 2]

    query_lora = rms_norm(
        F.linear(hidden, weights.wq_a), weights.q_norm, eps=cfg.norm_eps
    )
    query = F.linear(query_lora, weights.wq_b).reshape(
        batch, 2, cfg.num_heads, cfg.head_dim
    )
    query *= torch.rsqrt(query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps)
    query[..., -cfg.rope_dim :] = apply_rotary_emb(
        query[..., -cfg.rope_dim :], frequencies
    )

    raw_latent = rms_norm(
        F.linear(hidden, weights.wkv), weights.kv_norm, eps=cfg.norm_eps
    )
    raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
        raw_latent[..., -cfg.rope_dim :], frequencies
    )
    raw_latent[..., : -cfg.rope_dim] = attention._nope_control(
        raw_latent[..., : -cfg.rope_dim]
    )

    outputs = []
    for token in range(2):
        position = start_pos + token
        state.decode_write(raw_latent[:, token : token + 1])
        if token == 0 and snapshot_out is not None:
            # Post-first-token state: rollback target on draft rejection.
            snapshot_out.append((state, snapshot_decode_state(state)))
        attention_kv = state.dequantized_latent()
        topk = window_topk_indices(
            batch_size=batch, seqlen=1, start_pos=position, device=hidden.device
        )
        outputs.append(
            torch_sparse_attention(
                query[:, token : token + 1],
                attention_kv,
                weights.attn_sink,
                topk,
                cfg.head_dim**-0.5,
            )
        )
    return _fused_output_projection(
        outputs, frequencies, cfg=cfg, wo_a=weights.wo_a, wo_b=weights.wo_b
    )


def ratio128_decode2(
    attention: Ratio128TorchAttention,
    hidden: torch.Tensor,
    *,
    start_pos: int,
    snapshot_out: list[tuple[Any, dict[str, Any]]] | None = None,
) -> torch.Tensor:
    """Two-position ratio-128 decode (per-position mirror of ``__call__``)."""

    cfg = attention.config
    weights = attention.weights
    state = attention.state
    batch, _ = _validate_two_token(hidden, cfg.hidden_size)
    if start_pos != state.next_position or start_pos <= 0:
        raise Verify2Error(
            f"ratio-128 verify-2 start_pos {start_pos} != state position "
            f"{state.next_position} (or is a prefill position)"
        )
    if start_pos + 2 > cfg.max_seq_len:
        raise Verify2Error("ratio-128 verify-2 exceeds static KV capacity")
    frequencies = attention.freqs_cis[start_pos : start_pos + 2]

    query_lora = rms_norm(
        F.linear(hidden, weights.wq_a), weights.q_norm, eps=cfg.norm_eps
    )
    query = F.linear(query_lora, weights.wq_b).reshape(
        batch, 2, cfg.num_heads, cfg.head_dim
    )
    query *= torch.rsqrt(query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps)
    query[..., -cfg.rope_dim :] = apply_rotary_emb(
        query[..., -cfg.rope_dim :], frequencies
    )

    raw_latent = rms_norm(
        F.linear(hidden, weights.wkv), weights.kv_norm, eps=cfg.norm_eps
    )
    raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
        raw_latent[..., -cfg.rope_dim :], frequencies
    )
    raw_latent[..., : -cfg.rope_dim] = attention._nope_control(
        raw_latent[..., : -cfg.rope_dim]
    )
    projected_kv = F.linear(hidden.float(), weights.compressor_wkv)
    projected_score = F.linear(hidden.float(), weights.compressor_wgate)

    outputs = []
    for token in range(2):
        position = start_pos + token
        state.decode_write(
            raw_latent[:, token : token + 1],
            projected_kv=projected_kv[:, token : token + 1],
            projected_score=projected_score[:, token : token + 1],
            ape=weights.compressor_ape,
            finalize_compressed=attention._compress_finalizer,
        )
        if token == 0 and snapshot_out is not None:
            snapshot_out.append((state, snapshot_decode_state(state)))
        attention_kv = state.dequantized_latent()
        window = window_topk_indices(
            batch_size=batch, seqlen=1, start_pos=position, device=hidden.device
        )
        compressed = compressed_topk_indices(
            batch_size=batch,
            seqlen=1,
            start_pos=position,
            offset=WINDOW_SIZE,
            device=hidden.device,
        )
        topk = torch.cat((window, compressed), dim=-1).contiguous()
        outputs.append(
            torch_sparse_attention(
                query[:, token : token + 1],
                attention_kv,
                weights.attn_sink,
                topk,
                cfg.head_dim**-0.5,
            )
        )
    return _fused_output_projection(
        outputs, frequencies, cfg=cfg, wo_a=weights.wo_a, wo_b=weights.wo_b
    )


def ratio4_decode2(
    attention: Ratio4FullPositionAttention,
    hidden: torch.Tensor,
    *,
    start_pos: int,
    snapshot_out: list[tuple[Any, dict[str, Any]]] | None = None,
) -> torch.Tensor:
    """Two-position ratio-4 decode (per-position mirror of the fullpos step)."""

    cfg = attention.config
    weights = attention.weights
    batch, _ = _validate_two_token(hidden, cfg.hidden_size)
    if start_pos != attention.next_position or start_pos <= 0:
        raise Verify2Error(
            f"ratio-4 verify-2 start_pos {start_pos} != state position "
            f"{attention.next_position} (or is a prefill position)"
        )
    if start_pos + 2 > cfg.max_seq_len:
        raise Verify2Error("ratio-4 verify-2 exceeds the state capacity")
    frequencies = attention.freqs_cis[start_pos : start_pos + 2]

    query_lora = rms_norm(
        F.linear(hidden, weights.wq_a), weights.q_norm, eps=cfg.norm_eps
    )
    query = F.linear(query_lora, weights.wq_b).reshape(
        batch, 2, cfg.num_heads, cfg.head_dim
    )
    query *= torch.rsqrt(query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps)
    query[..., -cfg.rope_dim :] = apply_rotary_emb(
        query[..., -cfg.rope_dim :], frequencies
    )

    raw_latent = rms_norm(
        F.linear(hidden, weights.wkv), weights.kv_norm, eps=cfg.norm_eps
    )
    raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
        raw_latent[..., -cfg.rope_dim :], frequencies
    )
    # NoPE QAT simulation exactly as the fullpos step (fp8 qdq group 64).
    raw_latent[..., : -cfg.rope_dim] = fp8_quant_dequant(
        raw_latent[..., : -cfg.rope_dim], group_size=64
    )

    main_projected = F.linear(hidden.float(), weights.compressor_wkv)
    main_score = F.linear(hidden.float(), weights.compressor_wgate)
    index_projected = F.linear(hidden.float(), weights.index_compressor_wkv)
    index_score = F.linear(hidden.float(), weights.index_compressor_wgate)

    index_query = F.linear(query_lora, weights.index_wq_b).reshape(
        batch, 2, cfg.index_n_heads, cfg.index_head_dim
    )
    index_query[..., -cfg.rope_dim :] = apply_rotary_emb(
        index_query[..., -cfg.rope_dim :], frequencies
    )
    index_query = fp4_quant_dequant(hadamard_transform(index_query))
    index_weights = F.linear(hidden, weights.index_weights_proj) * (
        cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5
    )

    outputs = []
    for token in range(2):
        position = start_pos + token
        phase = position % COMPRESS_RATIO
        boundary = phase == COMPRESS_RATIO - 1
        overlap_slot = COMPRESS_RATIO + phase
        compressed_row = position // COMPRESS_RATIO
        group_start_frequencies = attention.freqs_cis[
            position + 1 - COMPRESS_RATIO : position + 2 - COMPRESS_RATIO
        ]
        attention.raw[:, position % WINDOW_SIZE].copy_(
            attention._quantize_rows(raw_latent[:, token])
        )
        if attention.raw_rope is not None:
            attention.raw_rope[:, position % WINDOW_SIZE].copy_(
                raw_latent[:, token, -LATENT_ROPE_DIM:]
            )
        attention._decode_compress(
            main_projected[:, token : token + 1],
            main_score[:, token] + weights.compressor_ape[phase],
            kv_state=attention.main_kv_state,
            score_state=attention.main_score_state,
            output_dim=LATENT_DIM,
            finalizer=attention._finalize_main,
            output_cache=attention.compressed,
            output_cache_rope=attention.compressed_rope,
            overlap_slot=overlap_slot,
            boundary=boundary,
            compressed_row=compressed_row,
            group_start_frequencies=group_start_frequencies,
        )
        attention._decode_compress(
            index_projected[:, token : token + 1],
            index_score[:, token] + weights.index_compressor_ape[phase],
            kv_state=attention.index_kv_state,
            score_state=attention.index_score_state,
            output_dim=INDEX_DIM,
            finalizer=attention._finalize_index,
            output_cache=attention.indexer_kv,
            overlap_slot=overlap_slot,
            boundary=boundary,
            compressed_row=compressed_row,
            group_start_frequencies=group_start_frequencies,
        )
        if token == 0 and snapshot_out is not None:
            # Post-first-token state with the cursors the restore must land
            # on (the function-level cursor update happens only at the end).
            partial = snapshot_decode_state(attention)
            partial["scalars"] = {
                "next_position": start_pos + 1,
                "compressed_count": (start_pos + 1) // COMPRESS_RATIO,
            }
            snapshot_out.append((attention, partial))
        compressed_count = (position + 1) // COMPRESS_RATIO
        attention_kv = torch.cat(
            (
                attention._dequantized(attention.raw, attention.raw_rope),
                attention._dequantized(
                    attention.compressed[:, :compressed_count],
                    None
                    if attention.compressed_rope is None
                    else attention.compressed_rope[:, :compressed_count],
                ),
            ),
            dim=1,
        )
        if compressed_count > 0:
            index_kv = attention.indexer_kv[:, :compressed_count]
            scores = torch.einsum(
                "bshd,btd->bsht",
                index_query[:, token : token + 1].float(),
                index_kv.float(),
            )
            scores = (
                scores.relu()
                * index_weights[:, token : token + 1].float().unsqueeze(-1)
            ).sum(dim=2)
            topk_count = min(cfg.index_topk, compressed_count)
            compressed_indices = (
                scores.topk(topk_count, dim=-1).indices + WINDOW_SIZE
            ).to(torch.int32)
        else:
            compressed_indices = torch.empty(
                (batch, 1, 0), dtype=torch.int32, device=hidden.device
            )
        window = window_topk_indices(
            batch_size=batch, seqlen=1, start_pos=position, device=hidden.device
        )
        topk = torch.cat((window, compressed_indices), dim=-1).contiguous()
        outputs.append(
            torch_sparse_attention(
                query[:, token : token + 1],
                attention_kv,
                weights.attn_sink,
                topk,
                cfg.head_dim**-0.5,
            )
        )
    attention.next_position = start_pos + 2
    attention.compressed_count = (start_pos + 2) // COMPRESS_RATIO
    return _fused_output_projection(
        outputs, frequencies, cfg=cfg, wo_a=weights.wo_a, wo_b=weights.wo_b
    )


__all__ = [
    "Verify2Error",
    "ratio128_decode2",
    "ratio4_decode2",
    "restore_decode_state",
    "snapshot_decode_state",
    "window_decode2",
]
