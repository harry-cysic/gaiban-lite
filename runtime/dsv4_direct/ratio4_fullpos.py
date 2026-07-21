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
  ``torch_sparse_attention``).  Prefill beyond one window (C2F vertical)
  implements the reference ring wrap-on-prefill branch (model.py:518-523):
  only the last ``window`` raw rows survive in the ring, at slots
  ``position % window``; prefill attention itself runs over the full raw
  sequence + compressed rows exactly as before, so wrap only affects
  subsequent decode reads.

C2F prefill additions (all default-off, decode paths untouched):

- ``index_score_mode``: ``"ref"`` (frozen materialized FP32 chain),
  ``"fused"`` (D0b fused Triton indexer score for prefill chunks with
  ``seqlen >= fuse_min_seqlen``; semantic change, gated), or
  ``"paired_gate"`` (compute both, keep ref semantics, record an A/B
  numeric + timing record per prefill call in ``index_gate_records``).
- ``sparse_row_block``: optional query-row blocking of the prefill sparse
  attention core.  Rows are independent (per-row softmax), so slicing is
  bitwise identical; it only bounds the FP32 gather workspace
  (``[b, rows, k, 512]``) that would otherwise reach ~10.7 GB at 8192.
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
from .static_kv import (
    LATENT_ROPE_DIM,
    quantize_latent_rows,
    resolve_kv_dtype,
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
        kv_dtype: str = "bf16",
        indexer_dtype: str = "bf16",
        index_score_mode: str = "ref",
        fuse_min_seqlen: int = 1024,
        sparse_row_block: int | None = None,
    ) -> None:
        config.validate()
        if index_score_mode not in ("ref", "fused", "paired_gate"):
            raise Ratio4FullPositionError(
                f"index_score_mode must be ref/fused/paired_gate, got {index_score_mode!r}"
            )
        if (
            not isinstance(fuse_min_seqlen, int)
            or isinstance(fuse_min_seqlen, bool)
            or fuse_min_seqlen < 1
        ):
            raise Ratio4FullPositionError("fuse_min_seqlen must be a positive integer")
        if sparse_row_block is not None and (
            not isinstance(sparse_row_block, int)
            or isinstance(sparse_row_block, bool)
            or sparse_row_block < 1
        ):
            raise Ratio4FullPositionError("sparse_row_block must be None or positive")
        if weights.layer_id != config.layer_id:
            raise Ratio4FullPositionError(
                f"config layer {config.layer_id} != weights layer {weights.layer_id}"
            )
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise Ratio4FullPositionError("batch_size must be a positive integer")
        latent_dtype = resolve_kv_dtype(kv_dtype)
        if indexer_dtype not in ("bf16", "fp8"):
            raise Ratio4FullPositionError(
                f"indexer_dtype must be 'bf16' or 'fp8', got {indexer_dtype!r}"
            )
        self.config = config
        self.weights = weights
        self.batch_size = batch_size
        self.kv_dtype = kv_dtype
        self.indexer_dtype = indexer_dtype
        self.index_score_mode = index_score_mode
        self.fuse_min_seqlen = fuse_min_seqlen
        self.sparse_row_block = sparse_row_block
        self.index_gate_records: list[dict] = []
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
        # Reference-shaped state (model.py:300-305, :399, :473-474).  FP8 KV
        # stores latent rows as e4m3 (A6F fp8_cast form) and reads them back
        # through a BF16 cast; "fp8_rope_bf16" keeps the rope tail BF16 in
        # parallel side tensors.
        self.raw = torch.zeros(
            batch_size, WINDOW_SIZE, LATENT_DIM, dtype=latent_dtype, device=device
        )
        self.compressed = torch.zeros(
            batch_size, capacity, LATENT_DIM, dtype=latent_dtype, device=device
        )
        rope_split = kv_dtype == "fp8_rope_bf16"
        self.raw_rope: torch.Tensor | None = (
            torch.zeros(
                batch_size,
                WINDOW_SIZE,
                LATENT_ROPE_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
            if rope_split
            else None
        )
        self.compressed_rope: torch.Tensor | None = (
            torch.zeros(
                batch_size,
                capacity,
                LATENT_ROPE_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
            if rope_split
            else None
        )
        self.indexer_kv = torch.zeros(
            batch_size,
            capacity,
            INDEX_DIM,
            dtype=(
                torch.bfloat16 if indexer_dtype == "bf16" else torch.float8_e4m3fn
            ),
            device=device,
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
    # FP8 KV storage helpers

    def _quantize_rows(self, value: torch.Tensor) -> torch.Tensor:
        return quantize_latent_rows(value, self.raw.dtype)

    def _quantize_indexer_rows(self, value: torch.Tensor) -> torch.Tensor:
        return quantize_latent_rows(value, self.indexer_kv.dtype)

    def _quantize_dequantize_rows(self, value: torch.Tensor) -> torch.Tensor:
        if self.kv_dtype == "bf16":
            return value
        result = self._quantize_rows(value).to(torch.bfloat16)
        if self.raw_rope is not None:
            result[..., -LATENT_ROPE_DIM:] = value[..., -LATENT_ROPE_DIM:]
        return result

    def _dequantized(
        self, value: torch.Tensor, rope: torch.Tensor | None
    ) -> torch.Tensor:
        if self.kv_dtype == "bf16":
            return value
        result = value.to(torch.bfloat16)
        if rope is not None:
            result[..., -LATENT_ROPE_DIM:] = rope
        return result

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
        output_cache_rope: torch.Tensor | None = None,
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
        output_cache[:, :rows].copy_(
            quantize_latent_rows(finalized, output_cache.dtype)
        )
        if output_cache_rope is not None:
            output_cache_rope[:, :rows].copy_(finalized[..., -LATENT_ROPE_DIM:])
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
        output_cache_rope: torch.Tensor | None = None,
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
        output_cache[:, compressed_row : compressed_row + 1].copy_(
            quantize_latent_rows(finalized, output_cache.dtype)
        )
        if output_cache_rope is not None:
            output_cache_rope[:, compressed_row : compressed_row + 1].copy_(
                finalized[..., -LATENT_ROPE_DIM:]
            )
        kv_state[:, :COMPRESS_RATIO].copy_(kv_state[:, COMPRESS_RATIO:])
        score_state[:, :COMPRESS_RATIO].copy_(score_state[:, COMPRESS_RATIO:])

    # ------------------------------------------------------------------
    # indexer score backends (C2F prefill vertical)

    # Test hook: force a row block in _ref_index_scores so the blocked and
    # single-shot forms can be compared on shapes where auto-blocking would
    # not trigger.  None = auto (~1 GiB temporary bound).
    _REF_SCORE_ROW_BLOCK_OVERRIDE: int | None = None

    @staticmethod
    def _ref_index_scores(
        index_query: torch.Tensor,
        index_kv: torch.Tensor,
        index_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Frozen materialized FP32 chain (model.py:411-423).

        Two memory-only transforms vs the previous form, both value-identical:
        relu_/mul_ run in place on the fresh einsum output, and the query-row
        axis is blocked so the [b, rows, h, t] FP32 temporary stays <= ~1 GiB
        (every output element depends only on its own s row, so row blocking
        is bitwise identical; required to fit the 8192-chunk baseline arm in
        24 GB).
        """

        batch, seqlen, heads, _ = index_query.shape
        t_rows = index_kv.shape[1]
        block_bytes = 1 << 30
        row_block = max(1, block_bytes // max(1, heads * t_rows * 4))
        override = Ratio4FullPositionAttention._REF_SCORE_ROW_BLOCK_OVERRIDE
        if override is not None:
            row_block = override
        if seqlen <= row_block:
            scores = torch.einsum(
                "bshd,btd->bsht", index_query.float(), index_kv.float()
            )
            return (
                scores.relu_().mul_(index_weights.float().unsqueeze(-1)).sum(dim=2)
            )
        kv_fp32 = index_kv.float()
        out = torch.empty(
            (batch, seqlen, t_rows),
            dtype=torch.float32,
            device=index_query.device,
        )
        for begin in range(0, seqlen, row_block):
            end = min(begin + row_block, seqlen)
            scores = torch.einsum(
                "bshd,btd->bsht", index_query[:, begin:end].float(), kv_fp32
            )
            torch.sum(
                scores.relu_().mul_(
                    index_weights[:, begin:end].float().unsqueeze(-1)
                ),
                dim=2,
                out=out[:, begin:end],
            )
        return out

    @staticmethod
    def _fused_index_scores(
        index_query: torch.Tensor,
        index_kv: torch.Tensor,
        index_weights: torch.Tensor,
    ) -> torch.Tensor:
        """D0b fused Triton score (semantic change vs ref: fp32 sum order)."""

        from .ops.indexer_fused import fused_index_score

        kv = (
            index_kv
            if index_kv.dtype == torch.bfloat16
            else index_kv.to(torch.bfloat16)
        )
        return fused_index_score(index_query, kv, index_weights)

    def _paired_index_score_gate(
        self,
        index_query: torch.Tensor,
        index_kv: torch.Tensor,
        index_weights: torch.Tensor,
        *,
        mask_add: torch.Tensor | None,
        topk_count: int,
        seqlen: int,
        compressed_count: int,
    ) -> torch.Tensor:
        """Run ref and fused scoring on identical inputs; keep ref semantics.

        Appends one A/B record (CUDA-event timings, score deltas, masked
        top-k agreement) to ``index_gate_records``.  Diagnostic mode: it
        synchronizes the device to read the events.
        """

        start_ref = torch.cuda.Event(enable_timing=True)
        stop_ref = torch.cuda.Event(enable_timing=True)
        start_fused = torch.cuda.Event(enable_timing=True)
        stop_fused = torch.cuda.Event(enable_timing=True)
        start_ref.record()
        scores_ref = self._ref_index_scores(index_query, index_kv, index_weights)
        stop_ref.record()
        start_fused.record()
        scores_fused = self._fused_index_scores(
            index_query, index_kv, index_weights
        )
        stop_fused.record()
        torch.cuda.synchronize(self.device)

        delta = (scores_fused - scores_ref).abs()
        masked_ref = scores_ref
        masked_fused = scores_fused
        if mask_add is not None:
            mask = mask_add.to(scores_ref.dtype)
            masked_ref = scores_ref + mask
            masked_fused = scores_fused + mask
        topk_ref = masked_ref.topk(topk_count, dim=-1).indices
        topk_fused = masked_fused.topk(topk_count, dim=-1).indices
        sorted_ref = topk_ref.sort(dim=-1).values
        sorted_fused = topk_fused.sort(dim=-1).values
        row_exact = sorted_ref.eq(sorted_fused).all(dim=-1)
        exact_rows = int(row_exact.sum().item())
        total_rows = int(row_exact.numel())
        mismatch_overlap = None
        if exact_rows != total_rows:
            rows = (~row_exact).reshape(-1).nonzero().flatten()
            flat_ref = sorted_ref.reshape(-1, topk_count)[rows]
            flat_fused = sorted_fused.reshape(-1, topk_count)[rows]
            overlaps = []
            for row_index in range(flat_ref.shape[0]):
                merged = torch.cat(
                    (flat_ref[row_index], flat_fused[row_index])
                )
                union = int(merged.unique().numel())
                overlaps.append((2 * topk_count - union) / topk_count)
            mismatch_overlap = float(sum(overlaps) / len(overlaps))
        self.index_gate_records.append(
            {
                "layer_id": int(self.config.layer_id),
                "seqlen": int(seqlen),
                "t_rows": int(compressed_count),
                "topk_count": int(topk_count),
                "ref_ms": float(start_ref.elapsed_time(stop_ref)),
                "fused_ms": float(start_fused.elapsed_time(stop_fused)),
                "score_max_abs_diff": float(delta.max().item()),
                "score_abs_max_ref": float(scores_ref.abs().max().item()),
                "topk_rows_total": total_rows,
                "topk_rows_exact": exact_rows,
                "topk_mismatch_mean_overlap": mismatch_overlap,
            }
        )
        return scores_ref

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
            pass  # prefill accepts any seqlen within capacity (ring wrap below)
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
            # ring keeps the last <= window rows at slots position % window
            # (model.py:518-523: direct write when seqlen <= window, the
            # wrap-on-prefill split otherwise; index_copy_ over absolute
            # positions % window is the same placement in one form).
            kept = min(seqlen, WINDOW_SIZE)
            ring_slots = torch.arange(
                seqlen - kept, seqlen, device=self.device
            ).remainder(WINDOW_SIZE)
            ring_rows = raw_latent[:, seqlen - kept :]
            self.raw.index_copy_(
                1, ring_slots, self._quantize_rows(ring_rows).contiguous()
            )
            if self.raw_rope is not None:
                self.raw_rope.index_copy_(
                    1,
                    ring_slots,
                    ring_rows[..., -LATENT_ROPE_DIM:].contiguous(),
                )
            main_rows = self._prefill_compress(
                main_projected,
                main_score,
                weights.compressor_ape,
                kv_state=self.main_kv_state,
                score_state=self.main_score_state,
                output_dim=LATENT_DIM,
                finalizer=self._finalize_main,
                output_cache=self.compressed,
                output_cache_rope=self.compressed_rope,
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
            compressed_rope = self.compressed_rope
            attention_kv = torch.cat(
                (
                    self._quantize_dequantize_rows(raw_latent),
                    self._dequantized(
                        self.compressed[:, :compressed_count],
                        None
                        if compressed_rope is None
                        else compressed_rope[:, :compressed_count],
                    ),
                ),
                dim=1,
            )
        else:
            phase = start_pos % COMPRESS_RATIO
            boundary = phase == COMPRESS_RATIO - 1
            overlap_slot = COMPRESS_RATIO + phase
            compressed_row = start_pos // COMPRESS_RATIO
            group_start_frequencies = self.freqs_cis[
                start_pos + 1 - COMPRESS_RATIO : start_pos + 2 - COMPRESS_RATIO
            ]
            self.raw[:, start_pos % WINDOW_SIZE].copy_(
                self._quantize_rows(raw_latent[:, 0])
            )
            if self.raw_rope is not None:
                self.raw_rope[:, start_pos % WINDOW_SIZE].copy_(
                    raw_latent[:, 0, -LATENT_ROPE_DIM:]
                )
            self._decode_compress(
                main_projected,
                main_score[:, 0] + weights.compressor_ape[phase],
                kv_state=self.main_kv_state,
                score_state=self.main_score_state,
                output_dim=LATENT_DIM,
                finalizer=self._finalize_main,
                output_cache=self.compressed,
                output_cache_rope=self.compressed_rope,
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
            compressed_rope = self.compressed_rope
            attention_kv = torch.cat(
                (
                    self._dequantized(self.raw, self.raw_rope),
                    self._dequantized(
                        self.compressed[:, :compressed_count],
                        None
                        if compressed_rope is None
                        else compressed_rope[:, :compressed_count],
                    ),
                ),
                dim=1,
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
            mask_add = None
            visible = None
            if start_pos == 0:
                # causal row visibility over compressed rows (model.py:424-426)
                visible = (
                    torch.arange(1, seqlen + 1, device=hidden.device) // COMPRESS_RATIO
                )
                future = (
                    torch.arange(compressed_count, device=hidden.device)
                    >= visible.unsqueeze(1)
                )
                mask_add = torch.where(future, float("-inf"), 0.0)
            fuse_active = (
                self.index_score_mode in ("fused", "paired_gate")
                and start_pos == 0
                and seqlen >= self.fuse_min_seqlen
            )
            topk_count = min(cfg.index_topk, compressed_count)
            if fuse_active and self.index_score_mode == "paired_gate":
                scores = self._paired_index_score_gate(
                    index_query,
                    index_kv,
                    index_weights,
                    mask_add=mask_add,
                    topk_count=topk_count,
                    seqlen=seqlen,
                    compressed_count=compressed_count,
                )
            elif fuse_active:
                scores = self._fused_index_scores(
                    index_query, index_kv, index_weights
                )
            else:
                scores = self._ref_index_scores(
                    index_query, index_kv, index_weights
                )
            if mask_add is not None:
                # in-place add of the frozen causal mask (identical values to
                # the previous out-of-place broadcast add)
                scores = scores.add_(mask_add.to(scores.dtype))
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
        row_block = self.sparse_row_block
        if row_block is not None and start_pos == 0 and seqlen > row_block:
            # Query rows are independent (per-row mask/softmax), so blocking
            # the row axis is bitwise identical to the single call; it only
            # bounds the FP32 gather workspace [b, rows, k, 512].
            output = torch.cat(
                [
                    torch_sparse_attention(
                        query[:, begin : begin + row_block],
                        attention_kv,
                        weights.attn_sink,
                        topk[:, begin : begin + row_block],
                        cfg.head_dim**-0.5,
                    )
                    for begin in range(0, seqlen, row_block)
                ],
                dim=1,
            )
        else:
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
