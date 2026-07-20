"""Full-position eager ratio-4 attention (prefill at 0 + decode at any pos).

Why this exists: the plan-driven ``Ratio4TorchAttention`` paths
(``prepare_decode_plan`` / ``prepare_stateful_decode_plan``) are frozen to
saturated decode positions ``start_pos >= 128`` -- the E0ff/E0sf/E0qf
verticals only ever exercised them from position 8192 with seeded state.
The E2E golden gate decodes real prompts from position 0, where the window
top-k runs its padded branch and the overlap compressor starts from the
empty ``(0, -inf)`` state, so the ratio-4 layer needs a prefill entry and an
unrestricted-position decode step.  Window and ratio-128 layers already have
this surface (their oracle-verified ``__call__``); this module supplies the
ratio-4 counterpart.

Implementation contract:

- **Decode step** is the operator-for-operator mirror of the E0ff-verified
  ``Ratio4TorchAttention.forward_decode_tensor`` (ratio4_attention.py:1186)
  with plan-derived constants computed inline and the masked
  ``torch_sparse_attention`` core instead of the maskless gather (identical
  math when no ``-1`` padding exists, i.e. for ``start_pos >= 127``; below
  that the mask handles the reference padded-window branch,
  model.py:255-265).
- **Prefill** follows the reference exactly (model.py:484-528 attention,
  :316-342 overlap compressor start_pos==0 branch, :402-433 indexer
  prefill) using the same verified operator helpers as the decode mirror
  (``rms_norm``, ``apply_rotary_emb``, ``fp8_quant_dequant``,
  ``hadamard_transform``, ``fp4_quant_dequant``, ``window_topk_indices``,
  ``torch_sparse_attention``).  Prefill is restricted to
  ``seqlen <= window`` (128) -- the E2E prompts are <= 22 tokens; the ring
  wrap-on-prefill branch (model.py:522-523) is out of scope and rejected.
- State is held in plain reference-shaped tensors (raw ring, compressed
  rows, indexer rows, 2x4-slot overlap kv/score states initialized to
  ``(0, -inf)`` exactly like the reference ``register_buffer`` init,
  model.py:303-304), because the ``StaticRatio4KV`` seeding contract only
  admits saturated phase-0 positions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import (
    apply_rotary_emb,
    fp8_quant_dequant,
    precompute_freqs_cis,
    rms_norm,
    torch_sparse_attention,
    window_topk_indices,
)
from .ratio4_attention import (
    PreparedRatio4AttentionWeights,
    Ratio4AttentionConfig,
    fp4_quant_dequant,
    hadamard_transform,
    overlap_pool,
)
from .static_ratio4_kv import (
    COMPRESS_RATIO,
    INDEX_DIM,
    LATENT_DIM,
    WINDOW_SIZE,
)


class Ratio4FullPositionError(ValueError):
    """Raised when the full-position ratio-4 contract is violated."""


class Ratio4FullPositionAttention:
    """Stateful eager ratio-4 attention covering positions [0, max_seq_len)."""

    def __init__(
        self,
        config: Ratio4AttentionConfig,
        weights: PreparedRatio4AttentionWeights,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        config.validate()
        if weights.layer_id != config.layer_id:
            raise Ratio4FullPositionError(
                f"config layer {config.layer_id} != weights layer {weights.layer_id}"
            )
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise Ratio4FullPositionError("batch_size must be a positive integer")
        self.config = config
        self.weights = weights
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.freqs_cis = precompute_freqs_cis(
            dim=config.rope_dim,
            seqlen=config.max_seq_len,
            original_seq_len=config.original_seq_len,
            base=config.rope_theta,
            factor=config.rope_factor,
            beta_fast=config.beta_fast,
            beta_slow=config.beta_slow,
            device=self.device,
        )
        capacity = config.max_seq_len // COMPRESS_RATIO
        # Reference-shaped state (model.py:300-305, :399, :473-474).
        self.raw = torch.zeros(
            batch_size, WINDOW_SIZE, LATENT_DIM, dtype=torch.bfloat16, device=device
        )
        self.compressed = torch.zeros(
            batch_size, capacity, LATENT_DIM, dtype=torch.bfloat16, device=device
        )
        self.indexer_kv = torch.zeros(
            batch_size, capacity, INDEX_DIM, dtype=torch.bfloat16, device=device
        )
        self.main_kv_state = torch.zeros(
            batch_size, 2 * COMPRESS_RATIO, 2 * LATENT_DIM,
            dtype=torch.float32, device=device,
        )
        self.main_score_state = torch.full_like(self.main_kv_state, float("-inf"))
        self.index_kv_state = torch.zeros(
            batch_size, 2 * COMPRESS_RATIO, 2 * INDEX_DIM,
            dtype=torch.float32, device=device,
        )
        self.index_score_state = torch.full_like(self.index_kv_state, float("-inf"))
        self.next_position = 0
        self.compressed_count = 0

    # ------------------------------------------------------------------
    # compression finalizers (mirror of Ratio4TorchAttention:889-918,
    # generalized from one row to ``rows`` rows)

    def _finalize_main(self, pooled: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        value = rms_norm(
            pooled.to(torch.bfloat16), self.weights.compressor_norm, eps=cfg.norm_eps
        )
        value[..., -cfg.rope_dim:] = apply_rotary_emb(
            value[..., -cfg.rope_dim:], frequencies
        )
        value[..., : -cfg.rope_dim] = fp8_quant_dequant(
            value[..., : -cfg.rope_dim], group_size=64
        )
        return value.contiguous()

    def _finalize_index(self, pooled: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        value = rms_norm(
            pooled.to(torch.bfloat16),
            self.weights.index_compressor_norm,
            eps=cfg.norm_eps,
        )
        value[..., -cfg.rope_dim:] = apply_rotary_emb(
            value[..., -cfg.rope_dim:], frequencies
        )
        return fp4_quant_dequant(hadamard_transform(value)).contiguous()

    # ------------------------------------------------------------------
    # compressor entries

    def _prefill_compress(
        self,
        projected_kv: torch.Tensor,
        projected_score: torch.Tensor,
        ape: torch.Tensor,
        *,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        output_dim: int,
        finalizer,
        output_cache: torch.Tensor,
    ) -> int:
        """Reference overlap-compressor prefill (model.py:325-342, :307-314).

        Returns the number of finalized compressed rows (``cutoff // 4``).
        """

        batch, seqlen, _ = projected_kv.shape
        cutoff = seqlen - seqlen % COMPRESS_RATIO
        remainder = seqlen - cutoff
        if cutoff >= COMPRESS_RATIO:
            kv_state[:, :COMPRESS_RATIO].copy_(
                projected_kv[:, cutoff - COMPRESS_RATIO : cutoff]
            )
            score_state[:, :COMPRESS_RATIO].copy_(
                projected_score[:, cutoff - COMPRESS_RATIO : cutoff] + ape
            )
        if remainder:
            kv_state[:, COMPRESS_RATIO : COMPRESS_RATIO + remainder].copy_(
                projected_kv[:, cutoff:]
            )
            score_state[:, COMPRESS_RATIO : COMPRESS_RATIO + remainder].copy_(
                projected_score[:, cutoff:] + ape[:remainder]
            )
        rows = cutoff // COMPRESS_RATIO
        if rows == 0:
            return 0
        grouped_kv = projected_kv[:, :cutoff].unflatten(1, (rows, COMPRESS_RATIO))
        grouped_score = (
            projected_score[:, :cutoff].unflatten(1, (rows, COMPRESS_RATIO)) + ape
        )
        # overlap_transform (model.py:307-314): first half of the width is
        # the overlapping (previous-window) stream, second half the current.
        over_kv = grouped_kv.new_zeros(
            (batch, rows, 2 * COMPRESS_RATIO, output_dim)
        )
        over_score = grouped_score.new_full(
            (batch, rows, 2 * COMPRESS_RATIO, output_dim), float("-inf")
        )
        over_kv[:, :, COMPRESS_RATIO:] = grouped_kv[..., output_dim:]
        over_kv[:, 1:, :COMPRESS_RATIO] = grouped_kv[:, :-1, :, :output_dim]
        over_score[:, :, COMPRESS_RATIO:] = grouped_score[..., output_dim:]
        over_score[:, 1:, :COMPRESS_RATIO] = grouped_score[:, :-1, :, :output_dim]
        pooled = (over_kv * over_score.softmax(dim=2)).sum(dim=2)
        finalized = finalizer(
            pooled, self.freqs_cis[0:cutoff:COMPRESS_RATIO]
        )
        output_cache[:, :rows].copy_(finalized)
        return rows

    def _decode_compress(
        self,
        projected_kv: torch.Tensor,
        adjusted_score: torch.Tensor,
        *,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        output_dim: int,
        finalizer,
        output_cache: torch.Tensor,
        overlap_slot: int,
        boundary: bool,
        compressed_row: int,
        group_start_frequencies: torch.Tensor,
    ) -> None:
        """Mirror of ``Ratio4TorchAttention._write_overlap`` (:921-965)."""

        kv_state[:, overlap_slot].copy_(projected_kv[:, 0])
        score_state[:, overlap_slot].copy_(adjusted_score)
        if not boundary:
            return
        pooled = overlap_pool(kv_state, score_state, output_dim=output_dim)
        finalized = finalizer(pooled, group_start_frequencies)
        output_cache[:, compressed_row : compressed_row + 1].copy_(finalized)
        kv_state[:, :COMPRESS_RATIO].copy_(kv_state[:, COMPRESS_RATIO:])
        score_state[:, :COMPRESS_RATIO].copy_(score_state[:, COMPRESS_RATIO:])

    # ------------------------------------------------------------------

    def __call__(self, hidden: torch.Tensor, *, start_pos: int) -> torch.Tensor:
        cfg = self.config
        weights = self.weights
        if hidden.ndim != 3 or hidden.shape[0] != self.batch_size:
            raise Ratio4FullPositionError(
                "hidden must have shape [batch, sequence, hidden_size]"
            )
        if hidden.shape[-1] != cfg.hidden_size or hidden.dtype != torch.bfloat16:
            raise Ratio4FullPositionError("hidden must be BF16 with the model width")
        if start_pos != self.next_position:
            raise Ratio4FullPositionError(
                f"start_pos {start_pos} != state next position {self.next_position}"
            )
        batch, seqlen, _ = hidden.shape
        if start_pos == 0:
            if seqlen > WINDOW_SIZE:
                raise Ratio4FullPositionError(
                    "full-position ratio-4 prefill is frozen to <= 128 tokens"
                )
        elif seqlen != 1:
            raise Ratio4FullPositionError("decode requires exactly one token")
        if start_pos + seqlen > cfg.max_seq_len:
            raise Ratio4FullPositionError("input exceeds the state capacity")

        frequencies = self.freqs_cis[start_pos : start_pos + seqlen]

        # q path (model.py:496-499; identical ops to ratio4_attention:1219-1250)
        query_lora = rms_norm(
            F.linear(hidden, weights.wq_a), weights.q_norm, eps=cfg.norm_eps
        )
        query = F.linear(query_lora, weights.wq_b).reshape(
            batch, seqlen, cfg.num_heads, cfg.head_dim
        )
        query *= torch.rsqrt(
            query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps
        )
        query[..., -cfg.rope_dim:] = apply_rotary_emb(
            query[..., -cfg.rope_dim:], frequencies
        )

        # raw kv path (model.py:502-506)
        raw_latent = rms_norm(
            F.linear(hidden, weights.wkv), weights.kv_norm, eps=cfg.norm_eps
        )
        raw_latent[..., -cfg.rope_dim:] = apply_rotary_emb(
            raw_latent[..., -cfg.rope_dim:], frequencies
        )
        raw_latent[..., : -cfg.rope_dim] = fp8_quant_dequant(
            raw_latent[..., : -cfg.rope_dim], group_size=64
        )

        # compressor projections (fp32, model.py:322-324)
        main_projected = F.linear(hidden.float(), weights.compressor_wkv)
        main_score = F.linear(hidden.float(), weights.compressor_wgate)
        index_projected = F.linear(hidden.float(), weights.index_compressor_wkv)
        index_score = F.linear(hidden.float(), weights.index_compressor_wgate)

        if start_pos == 0:
            # ring keeps the whole (<= window) prefill (model.py:520-521)
            self.raw[:, :seqlen].copy_(raw_latent)
            main_rows = self._prefill_compress(
                main_projected,
                main_score,
                weights.compressor_ape,
                kv_state=self.main_kv_state,
                score_state=self.main_score_state,
                output_dim=LATENT_DIM,
                finalizer=self._finalize_main,
                output_cache=self.compressed,
            )
            index_rows = self._prefill_compress(
                index_projected,
                index_score,
                weights.index_compressor_ape,
                kv_state=self.index_kv_state,
                score_state=self.index_score_state,
                output_dim=INDEX_DIM,
                finalizer=self._finalize_index,
                output_cache=self.indexer_kv,
            )
            if main_rows != index_rows:
                raise AssertionError("main/index compressors disagree on row count")
            compressed_count = main_rows
            offset = seqlen  # model.py:509 (prefill: raw rows precede compressed)
            attention_kv = torch.cat(
                (raw_latent, self.compressed[:, :compressed_count]), dim=1
            )
        else:
            phase = start_pos % COMPRESS_RATIO
            boundary = phase == COMPRESS_RATIO - 1
            overlap_slot = COMPRESS_RATIO + phase
            compressed_row = start_pos // COMPRESS_RATIO
            group_start_frequencies = self.freqs_cis[
                start_pos + 1 - COMPRESS_RATIO : start_pos + 2 - COMPRESS_RATIO
            ]
            self.raw[:, start_pos % WINDOW_SIZE].copy_(raw_latent[:, 0])
            self._decode_compress(
                main_projected,
                main_score[:, 0] + weights.compressor_ape[phase],
                kv_state=self.main_kv_state,
                score_state=self.main_score_state,
                output_dim=LATENT_DIM,
                finalizer=self._finalize_main,
                output_cache=self.compressed,
                overlap_slot=overlap_slot,
                boundary=boundary,
                compressed_row=compressed_row,
                group_start_frequencies=group_start_frequencies,
            )
            self._decode_compress(
                index_projected,
                index_score[:, 0] + weights.index_compressor_ape[phase],
                kv_state=self.index_kv_state,
                score_state=self.index_score_state,
                output_dim=INDEX_DIM,
                finalizer=self._finalize_index,
                output_cache=self.indexer_kv,
                overlap_slot=overlap_slot,
                boundary=boundary,
                compressed_row=compressed_row,
                group_start_frequencies=group_start_frequencies,
            )
            compressed_count = (start_pos + 1) // COMPRESS_RATIO
            offset = WINDOW_SIZE  # model.py:509 (decode: ring precedes compressed)
            attention_kv = torch.cat(
                (self.raw, self.compressed[:, :compressed_count]), dim=1
            )

        # indexer scoring (model.py:411-433; ops mirror ratio4_attention:1313-1338)
        index_query = F.linear(query_lora, weights.index_wq_b).reshape(
            batch, seqlen, cfg.index_n_heads, cfg.index_head_dim
        )
        index_query[..., -cfg.rope_dim:] = apply_rotary_emb(
            index_query[..., -cfg.rope_dim:], frequencies
        )
        index_query = fp4_quant_dequant(hadamard_transform(index_query))
        index_weights = F.linear(hidden, weights.index_weights_proj) * (
            cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5
        )
        if compressed_count > 0:
            index_kv = self.indexer_kv[:, :compressed_count]
            scores = torch.einsum(
                "bshd,btd->bsht", index_query.float(), index_kv.float()
            )
            scores = (scores.relu() * index_weights.float().unsqueeze(-1)).sum(dim=2)
            if start_pos == 0:
                # causal row visibility over compressed rows (model.py:424-426)
                visible = (
                    torch.arange(1, seqlen + 1, device=hidden.device) // COMPRESS_RATIO
                )
                future = (
                    torch.arange(compressed_count, device=hidden.device)
                    >= visible.unsqueeze(1)
                )
                scores = scores + torch.where(
                    future, float("-inf"), 0.0
                ).to(scores.dtype)
            topk_count = min(cfg.index_topk, compressed_count)
            compressed_indices = scores.topk(topk_count, dim=-1).indices
            if start_pos == 0:
                invalid = compressed_indices >= visible.view(1, seqlen, 1)
                compressed_indices = torch.where(
                    invalid, -1 - offset, compressed_indices
                )
            compressed_indices = (compressed_indices + offset).to(torch.int32)
        else:
            compressed_indices = torch.empty(
                (batch, seqlen, 0), dtype=torch.int32, device=hidden.device
            )

        window = window_topk_indices(
            batch_size=batch,
            seqlen=seqlen,
            start_pos=start_pos,
            device=hidden.device,
        )
        topk = torch.cat((window, compressed_indices), dim=-1).contiguous()
        output = torch_sparse_attention(
            query,
            attention_kv,
            weights.attn_sink,
            topk,
            cfg.head_dim**-0.5,
        )
        output[..., -cfg.rope_dim:] = apply_rotary_emb(
            output[..., -cfg.rope_dim:], frequencies, inverse=True
        )
        grouped = output.reshape(
            batch,
            seqlen,
            cfg.o_groups,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        wo_a = weights.wo_a.reshape(
            cfg.o_groups,
            cfg.o_lora_rank,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        branch = F.linear(projected.flatten(2), weights.wo_b)

        self.next_position = start_pos + seqlen
        self.compressed_count = compressed_count
        return branch


__all__ = [
    "Ratio4FullPositionAttention",
    "Ratio4FullPositionError",
]
