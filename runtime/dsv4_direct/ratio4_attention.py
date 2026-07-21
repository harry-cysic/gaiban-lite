"""Direct-owned ratio-4 decode control for real ratio-4 checkpoints.

The implementation follows the public checkpoint reference tensor math with
plain PyTorch.  It is intentionally a correctness/control path: projections
use dequantized BF16 weights, index scoring is materialized, and sparse
attention gathers selected latent rows.  Faster kernels must be compared
against immutable inputs before replacing this path.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Mapping, Protocol

import torch
import torch.nn.functional as F

from .attention import (
    AttentionProjectionBackend,
    _project_hidden_with_backend,
    _project_output_with_backend,
    _project_query_with_backend,
    _validate_attention_projection_backend,
    apply_rotary_emb,
    fp8_quant_dequant,
    kv_fp8_qat_prefix,
    resolve_kv_qat_mode,
    precompute_freqs_cis,
    rms_norm,
    window_topk_indices,
)
from .block_weights import ResidentAttentionWeights
from .model_contract import validate_model_layer_config
from .moe_forward import dequant_fp8_block
from .static_kv import LATENT_ROPE_DIM, quantize_latent_rows
from .static_ratio4_kv import (
    COMPRESS_RATIO,
    INDEX_DIM,
    INDEX_PROJECTED_DIM,
    LATENT_DIM,
    MAIN_PROJECTED_DIM,
    SUPPORTED_RATIO4_LAYER_IDS,
    StaticRatio4KV,
    WINDOW_SIZE,
)


@dataclass(frozen=True)
class Ratio4AttentionConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    norm_eps: float
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_seq_len: int
    max_seq_len: int
    layer_id: int = 2
    # E6F attention TP sharding (variant A: the o-path only).  The frozen
    # fields above stay **global** -- they describe the model, and validate()
    # still holds them to Flash's geometry.  These two describe which slice of
    # it this rank computes, so a sharding mistake shows up as a config error
    # rather than as a silent numeric one.
    tp_size: int = 1
    tp_rank: int = 0

    @property
    def local_num_heads(self) -> int:
        return self.num_heads // self.tp_size

    @property
    def local_o_groups(self) -> int:
        return self.o_groups // self.tp_size

    @property
    def group_width(self) -> int:
        """Values per o_group -- a **global** quantity: sharding takes whole
        groups, it never narrows one."""

        return self.num_heads * self.head_dim // self.o_groups

    @classmethod
    def from_model_config(
        cls,
        config: Mapping[str, Any],
        *,
        layer_id: int = 2,
        max_seq_len: int,
    ) -> "Ratio4AttentionConfig":
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id not in SUPPORTED_RATIO4_LAYER_IDS
        ):
            raise ValueError(
                "ratio-4 attention config requires an integer frozen "
                "ratio-4 layer_id, "
                f"got {layer_id!r}"
            )
        validate_model_layer_config(config, layer_id=layer_id)
        rope = config.get("rope_scaling") or {}
        result = cls(
            hidden_size=int(config["hidden_size"]),
            num_heads=int(config["num_attention_heads"]),
            head_dim=int(config["head_dim"]),
            rope_dim=int(config["qk_rope_head_dim"]),
            q_lora_rank=int(config["q_lora_rank"]),
            o_lora_rank=int(config["o_lora_rank"]),
            o_groups=int(config["o_groups"]),
            index_n_heads=int(config["index_n_heads"]),
            index_head_dim=int(config["index_head_dim"]),
            index_topk=int(config["index_topk"]),
            norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(config["compress_rope_theta"]),
            rope_factor=float(rope.get("factor", 16.0)),
            beta_fast=int(rope.get("beta_fast", 32)),
            beta_slow=int(rope.get("beta_slow", 1)),
            original_seq_len=int(
                rope.get("original_max_position_embeddings", 65536)
            ),
            max_seq_len=int(max_seq_len),
            layer_id=layer_id,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            not isinstance(self.layer_id, int)
            or isinstance(self.layer_id, bool)
            or self.layer_id not in SUPPORTED_RATIO4_LAYER_IDS
        ):
            raise ValueError(
                "ratio-4 attention config requires an integer frozen "
                "ratio-4 layer_id, "
                f"got {self.layer_id!r}"
            )
        # DeepSeek-V4-Flash geometry, frozen from the checkpoint config.json
        # (see model_contract.EXPECTED_RATIO4_CONFIG): hidden 4096, 64 heads,
        # q_lora 1024, o_groups 8, index_topk 512 (Pro used 1024).  head_dim
        # 512 (== LATENT_DIM), rope 64, o_lora 1024, and the indexer head
        # geometry (64 heads x 128 == INDEX_DIM) are unchanged from Pro.
        expected = {
            "hidden_size": (self.hidden_size, 4096),
            "num_heads": (self.num_heads, 64),
            "head_dim": (self.head_dim, LATENT_DIM),
            "rope_dim": (self.rope_dim, 64),
            "q_lora_rank": (self.q_lora_rank, 1024),
            "o_lora_rank": (self.o_lora_rank, 1024),
            "o_groups": (self.o_groups, 8),
            "index_n_heads": (self.index_n_heads, 64),
            "index_head_dim": (self.index_head_dim, INDEX_DIM),
            "index_topk": (self.index_topk, 512),
            "norm_eps": (self.norm_eps, 1e-6),
            "rope_theta": (self.rope_theta, 160000.0),
            "rope_factor": (self.rope_factor, 16.0),
            "beta_fast": (self.beta_fast, 32),
            "beta_slow": (self.beta_slow, 1),
            "original_seq_len": (self.original_seq_len, 65536),
        }
        mismatches = {
            name: {"observed": observed, "expected": wanted}
            for name, (observed, wanted) in expected.items()
            if observed != wanted
        }
        if mismatches:
            raise ValueError(
                f"unsupported layer-{self.layer_id} attention shape: {mismatches}"
            )
        if self.max_seq_len < WINDOW_SIZE or self.max_seq_len % COMPRESS_RATIO:
            raise ValueError("max_seq_len must be a multiple of four and at least 128")
        if self.num_heads % self.o_groups:
            raise ValueError("attention heads must divide output groups")
        if (
            not isinstance(self.tp_size, int)
            or isinstance(self.tp_size, bool)
            or self.tp_size < 1
            or not isinstance(self.tp_rank, int)
            or isinstance(self.tp_rank, bool)
            or self.tp_rank not in range(self.tp_size)
        ):
            raise ValueError("tp_rank must be in range(tp_size) and tp_size >= 1")
        if self.num_heads % self.tp_size or self.o_groups % self.tp_size:
            raise ValueError(
                f"attention TP split must be exact: {self.num_heads} heads and "
                f"{self.o_groups} groups over tp_size {self.tp_size}"
            )
        if self.rope_dim <= 0 or self.rope_dim > INDEX_DIM or self.rope_dim % 2:
            raise ValueError("rope_dim must be positive, even, and at most 128")
        for value in (self.norm_eps, self.rope_theta, self.rope_factor):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("attention numerical constants must be finite and positive")


@dataclass
class PreparedRatio4AttentionWeights:
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
    index_wq_b: torch.Tensor
    index_weights_proj: torch.Tensor
    index_compressor_ape: torch.Tensor
    index_compressor_wkv: torch.Tensor
    index_compressor_wgate: torch.Tensor
    index_compressor_norm: torch.Tensor
    layer_id: int
    rank: int
    world_size: int
    checkpoint_id: str

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(value.numel() * value.element_size())
            for value in self.__dict__.values()
            if isinstance(value, torch.Tensor)
        )


@dataclass(frozen=True)
class Ratio4DecodePlan:
    start_pos: int
    phase: int
    boundary: bool
    advance_overlap_state: bool
    raw_slot: int
    overlap_slot: int
    compressed_row: int
    compressed_count_after: int
    index_topk_count: int
    batch_size: int
    hidden_size: int
    owner_id: int
    state_id: int
    frequencies: torch.Tensor
    group_start_frequencies: torch.Tensor
    window_indices: torch.Tensor
    batch_indices: torch.Tensor
    main_ape: torch.Tensor
    index_ape: torch.Tensor


@dataclass(frozen=True)
class Ratio4StatefulDecodePlan:
    """Cursor-driven fixed-shape workspace for saturated ratio-4 decode."""

    start_position: int
    stop_position: int
    candidate_width: int
    index_topk_count: int
    total_topk: int
    batch_size: int
    hidden_size: int
    owner_id: int
    state_id: int
    position: torch.Tensor
    window_columns: torch.Tensor
    compressed_columns: torch.Tensor
    topk_indices: torch.Tensor
    batch_indices: torch.Tensor
    tensor_pointers: tuple[int, ...]

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(value.numel() * value.element_size())
            for value in (
                self.window_columns,
                self.compressed_columns,
                self.topk_indices,
                self.batch_indices,
            )
        )


@dataclass(frozen=True)
class Ratio4AttentionEvidence:
    """Owned eager-only snapshots for comparison with the independent oracle."""

    query_lora: torch.Tensor
    query: torch.Tensor
    raw_latent: torch.Tensor
    main_projected_kv: torch.Tensor
    main_projected_score: torch.Tensor
    main_adjusted_score: torch.Tensor
    main_overlap_values: torch.Tensor | None
    main_overlap_logits: torch.Tensor | None
    main_compression_pooled: torch.Tensor | None
    main_compression_finalized: torch.Tensor | None
    index_projected_kv: torch.Tensor
    index_projected_score: torch.Tensor
    index_adjusted_score: torch.Tensor
    index_overlap_values: torch.Tensor | None
    index_overlap_logits: torch.Tensor | None
    index_compression_pooled: torch.Tensor | None
    index_compression_finalized: torch.Tensor | None
    index_query: torch.Tensor
    index_weights: torch.Tensor
    index_scores: torch.Tensor
    compressed_indices: torch.Tensor
    topk_indices: torch.Tensor
    selected_kv: torch.Tensor
    sparse_output: torch.Tensor
    inverse_rotated: torch.Tensor
    output_lora: torch.Tensor
    branch: torch.Tensor


class Ratio4SparseAttentionBackend(Protocol):
    """Injectable sparse core for an otherwise unchanged ratio-4 attention."""

    def __call__(
        self,
        query: torch.Tensor,
        latent_kv: torch.Tensor,
        sink: torch.Tensor,
        topk_indices: torch.Tensor,
        batch_indices: torch.Tensor,
        scale: float,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class _Ratio4OverlapEvidence:
    values: torch.Tensor
    logits: torch.Tensor
    pooled: torch.Tensor
    finalized: torch.Tensor


def _evidence_snapshot(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().clone()


# 17th vertical, leverage 3 (half-precision accumulation EXPERIMENT).
# Default off: the shipped semantics stay the FP32 score/softmax chain.
# When DSV4_R4_HALF_ACCUM=1, the ratio-4 sparse-attention score/output
# einsums run on BF16 operands (BF16-rounded score storage; matmul-internal
# FP32 accumulation) and the indexer scoring chain runs in BF16.  This is a
# semantic change and must pass the single-layer oracle numeric gate (E0ff
# limits) before it may be considered further; evidence tensors are upcast
# to their contract dtypes so the oracle ABI still applies.
_HALF_ACCUM = os.environ.get("DSV4_R4_HALF_ACCUM", "0") == "1"


def hadamard_transform(value: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform over the final dimension."""

    width = value.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    original_shape = value.shape
    transformed = value.float().reshape(-1, width)
    stride = 1
    while stride < width:
        transformed = transformed.reshape(-1, width // (2 * stride), 2, stride)
        left = transformed[:, :, 0, :]
        right = transformed[:, :, 1, :]
        transformed = torch.cat((left + right, left - right), dim=-1).reshape(
            -1, width
        )
        stride *= 2
    return (
        transformed.reshape(original_shape) * (width ** -0.5)
    ).to(value.dtype)


def fp4_quant_dequant(value: torch.Tensor, *, group_size: int = 32) -> torch.Tensor:
    """Power-of-two scaled E2M1 fake quantization used by the indexer."""

    if value.dtype != torch.bfloat16:
        raise TypeError("FP4 QAT control requires BF16 input")
    if value.shape[-1] % group_size:
        raise ValueError("FP4 QAT width must divide the group size")
    grouped = value.float().reshape(*value.shape[:-1], -1, group_size)
    absolute_max = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(
        6.0 * 2.0**-126
    )
    scale = torch.exp2(torch.ceil(torch.log2(absolute_max / 6.0)))
    normalized = (grouped / scale).clamp(-6.0, 6.0)
    magnitude = normalized.abs()
    # Midpoint comparisons encode round-to-nearest-even for the E2M1 levels.
    snapped = torch.where(
        magnitude <= 0.25,
        torch.zeros_like(magnitude),
        torch.where(
            magnitude < 0.75,
            torch.full_like(magnitude, 0.5),
            torch.where(
                magnitude <= 1.25,
                torch.ones_like(magnitude),
                torch.where(
                    magnitude < 1.75,
                    torch.full_like(magnitude, 1.5),
                    torch.where(
                        magnitude <= 2.5,
                        torch.full_like(magnitude, 2.0),
                        torch.where(
                            magnitude < 3.5,
                            torch.full_like(magnitude, 3.0),
                            torch.where(
                                magnitude <= 5.0,
                                torch.full_like(magnitude, 4.0),
                                torch.full_like(magnitude, 6.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    result = torch.copysign(snapped, normalized) * scale
    return result.reshape_as(value).to(value.dtype)


def overlap_pool(
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    output_dim: int,
) -> torch.Tensor:
    """Pool previous-left and current-right halves over one eight-token window."""

    if kv_state.shape != score_state.shape or kv_state.ndim != 3:
        raise ValueError("overlap KV/score states must have equal rank-3 shapes")
    if kv_state.shape[1] != 2 * COMPRESS_RATIO:
        raise ValueError("overlap state must contain previous and current four rows")
    if kv_state.shape[2] != 2 * output_dim:
        raise ValueError("overlap projected width must be twice the output width")
    values = torch.cat(
        (
            kv_state[:, :COMPRESS_RATIO, :output_dim],
            kv_state[:, COMPRESS_RATIO:, output_dim:],
        ),
        dim=1,
    )
    scores = torch.cat(
        (
            score_state[:, :COMPRESS_RATIO, :output_dim],
            score_state[:, COMPRESS_RATIO:, output_dim:],
        ),
        dim=1,
    )
    return (values * scores.softmax(dim=1)).sum(dim=1, keepdim=True)


def shard_ratio4_attention_weights(
    prepared: PreparedRatio4AttentionWeights,
    *,
    tp_rank: int,
    tp_size: int,
    config: Ratio4AttentionConfig,
) -> PreparedRatio4AttentionWeights:
    """Slice the o-path tensors for one TP rank (E6F variant A).

    Sharded: ``wq_b`` by head rows, ``wo_a`` by o_group, ``wo_b`` by the
    matching **input columns**, ``attn_sink`` by head.  Rank ``r`` owns heads
    ``[16r, 16r+16)``, which on Flash's geometry is exactly o_groups
    ``[2r, 2r+2)`` -- 8 heads per group, verified in E6F step 1.

    Everything else is left whole on purpose: the compressor, ``wq_a``,
    ``wkv`` and the whole indexer chain produce state that **every** rank
    needs, so slicing them would require gathering it back (design note
    variant C, which E6F showed buys no latency).
    """

    if tp_size == 1:
        return prepared
    heads = config.num_heads // tp_size
    groups = config.o_groups // tp_size
    head_rows = heads * config.head_dim
    lora_cols = groups * config.o_lora_rank

    wq_b = prepared.wq_b[tp_rank * head_rows : (tp_rank + 1) * head_rows].contiguous()
    wo_a3 = prepared.wo_a.reshape(
        config.o_groups, config.o_lora_rank, config.group_width
    )
    wo_a = (
        wo_a3[tp_rank * groups : (tp_rank + 1) * groups]
        .reshape(lora_cols, config.group_width)
        .contiguous()
    )
    wo_b = prepared.wo_b[
        :, tp_rank * lora_cols : (tp_rank + 1) * lora_cols
    ].contiguous()
    attn_sink = prepared.attn_sink[
        tp_rank * heads : (tp_rank + 1) * heads
    ].contiguous()
    return replace(
        prepared, wq_b=wq_b, wo_a=wo_a, wo_b=wo_b, attn_sink=attn_sink
    )


def prepare_ratio4_attention_weights(
    weights: ResidentAttentionWeights,
    *,
    layer_id: int,
    rank: int,
    world_size: int,
    checkpoint_id: str,
) -> PreparedRatio4AttentionWeights:
    if (
        not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or layer_id not in SUPPORTED_RATIO4_LAYER_IDS
        or not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size != 4
        or not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank not in range(world_size)
    ):
        raise ValueError(
            "prepared ratio-4 attention identity must be a frozen ratio-4 "
            "layer on TP4"
        )
    if (
        not isinstance(checkpoint_id, str)
        or len(checkpoint_id) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_id)
    ):
        raise ValueError("prepared attention requires a lowercase SHA-256 checkpoint_id")
    if weights.indexer is None:
        raise ValueError("ratio-4 attention requires indexer weights")
    resident_identity = (
        weights.layer_id,
        weights.rank,
        weights.world_size,
        weights.checkpoint_id,
    )
    requested_identity = (layer_id, rank, world_size, checkpoint_id)
    resident_identity_well_typed = (
        isinstance(weights.layer_id, int)
        and not isinstance(weights.layer_id, bool)
        and isinstance(weights.rank, int)
        and not isinstance(weights.rank, bool)
        and isinstance(weights.world_size, int)
        and not isinstance(weights.world_size, bool)
        and isinstance(weights.checkpoint_id, str)
    )
    if not resident_identity_well_typed or resident_identity != requested_identity:
        raise ValueError(
            "resident ratio-4 attention identity differs from requested identity: "
            f"resident={resident_identity}, requested={requested_identity}"
        )

    def linear(value: Any) -> torch.Tensor:
        return dequant_fp8_block(value.weight, value.scale).to(torch.bfloat16)

    indexer = weights.indexer
    return PreparedRatio4AttentionWeights(
        attn_sink=weights.attn_sink.float().contiguous().clone(),
        wq_a=linear(weights.wq_a),
        q_norm=weights.q_norm.float().contiguous().clone(),
        wq_b=linear(weights.wq_b),
        wkv=linear(weights.wkv),
        kv_norm=weights.kv_norm.float().contiguous().clone(),
        wo_a=linear(weights.wo_a),
        wo_b=linear(weights.wo_b),
        compressor_ape=weights.compressor.ape.float().contiguous().clone(),
        compressor_wkv=weights.compressor.wkv.float().contiguous().clone(),
        compressor_wgate=weights.compressor.wgate.float().contiguous().clone(),
        compressor_norm=weights.compressor.norm.float().contiguous().clone(),
        index_wq_b=linear(indexer.wq_b),
        index_weights_proj=indexer.weights_proj.contiguous().clone(),
        index_compressor_ape=indexer.compressor.ape.float().contiguous().clone(),
        index_compressor_wkv=indexer.compressor.wkv.float().contiguous().clone(),
        index_compressor_wgate=indexer.compressor.wgate.float().contiguous().clone(),
        index_compressor_norm=indexer.compressor.norm.float().contiguous().clone(),
        layer_id=layer_id,
        rank=rank,
        world_size=world_size,
        checkpoint_id=checkpoint_id,
    )


class Ratio4TorchAttention:
    """Real-weight fixed-position ratio-4 attention and indexer control."""

    def __init__(
        self,
        config: Ratio4AttentionConfig,
        weights: PreparedRatio4AttentionWeights,
        state: StaticRatio4KV,
        *,
        sparse_attention_backend: Ratio4SparseAttentionBackend | None = None,
        projection_backend: AttentionProjectionBackend | None = None,
        indexer_qat_mode: str | None = None,
        kv_qat_mode: str | None = None,
    ) -> None:
        config.validate()
        resolved_qat = (
            indexer_qat_mode
            if indexer_qat_mode is not None
            else (os.environ.get("DSV4_INDEXER_QAT_DECODE", "").strip() or "fused")
        )
        if resolved_qat not in ("ref", "fused"):
            raise ValueError(
                f"indexer_qat_mode must be ref/fused, got {resolved_qat!r}"
            )
        identities = (config.layer_id, weights.layer_id, state.layer_id)
        if (
            any(
                not isinstance(layer_id, int) or isinstance(layer_id, bool)
                for layer_id in identities
            )
            or len(set(identities)) != 1
            or config.layer_id not in SUPPORTED_RATIO4_LAYER_IDS
            or state.max_seq_len != config.max_seq_len
        ):
            raise ValueError(
                "ratio-4 attention config/weights/state identity differs: "
                f"layers={identities}, max_seq_len="
                f"({config.max_seq_len}, {state.max_seq_len})"
            )
        if (
            not isinstance(weights.world_size, int)
            or isinstance(weights.world_size, bool)
            or weights.world_size != 4
            or not isinstance(weights.rank, int)
            or isinstance(weights.rank, bool)
            or weights.rank not in range(weights.world_size)
        ):
            raise ValueError("prepared ratio-4 attention identity is invalid")
        self.config = config
        self.weights = weights
        self.state = state
        # E4F: the decode indexer QAT chain is launch-floor bound at B=1, not
        # bandwidth bound (E2F).  C4F's fused kernel is **bitwise identical**
        # here at every level that was checked -- kernel self-check, decode
        # shape (max_abs_diff 0.0), 480 in-layer steps, and a D0L long gate
        # that reproduces the frozen 494/512 with max top2_gap 0.959503173828125
        # and the same first-mismatch logits on all eight prompts.  It buys
        # +2.31% single-stream (27.740 -> 28.381 tok/s), so it is the default.
        # DSV4_INDEXER_QAT_DECODE=ref restores the unfused chain.
        self.indexer_qat_mode = resolved_qat
        # E5F: the KV-latent FP8 QAT chain, default "ref" until gated.
        self.kv_qat_mode = resolve_kv_qat_mode(kv_qat_mode)
        if sparse_attention_backend is not None and not callable(
            sparse_attention_backend
        ):
            raise TypeError("ratio-4 sparse attention backend must be callable")
        if sparse_attention_backend is not None and (
            state.kv_dtype != "bf16" or state.indexer_dtype != "bf16"
        ):
            raise ValueError(
                "injected ratio-4 sparse backends require BF16 KV storage"
            )
        self._sparse_attention_backend = sparse_attention_backend
        _validate_attention_projection_backend(
            projection_backend,
            expected_compress_ratio=COMPRESS_RATIO,
        )
        self._projection_backend = projection_backend
        self._evidence_observer: list[Ratio4AttentionEvidence] | None = None
        self._topk_observer: list[torch.Tensor] | None = None
        self.freqs_cis = precompute_freqs_cis(
            dim=config.rope_dim,
            seqlen=config.max_seq_len,
            original_seq_len=config.original_seq_len,
            base=config.rope_theta,
            factor=config.rope_factor,
            beta_fast=config.beta_fast,
            beta_slow=config.beta_slow,
            device=weights.wq_a.device,
        )

    def _indexer_qat(self, index_query: torch.Tensor) -> torch.Tensor:
        """Hadamard + FP4 QAT on the indexer query (E4F).

        ``fused`` calls C4F's single-pass Triton kernel, which is bitwise
        identical to the eager pair -- verified at the decode shape by E4F's
        micro gate and re-verified in-layer by the E2F probe's bitwise arm.
        Only the ``[..., 128]`` width with ``group_size=32`` is supported by
        the kernel; anything else must stay on the eager path.
        """

        if self.indexer_qat_mode == "fused":
            from .ops.indexer_qat import FP4_GROUP, HADAMARD_WIDTH, fused_hadamard_fp4

            if index_query.shape[-1] == HADAMARD_WIDTH and not (
                HADAMARD_WIDTH % FP4_GROUP
            ):
                return fused_hadamard_fp4(index_query)
        return fp4_quant_dequant(hadamard_transform(index_query))

    @contextmanager
    def observe_evidence(self) -> Iterator[list[Ratio4AttentionEvidence]]:
        """Collect owned stage snapshots for eager diagnostic calls in this scope."""

        if getattr(self, "_sparse_attention_backend", None) is not None:
            raise RuntimeError(
                "ratio-4 evidence observation requires the default control backend"
            )
        if getattr(self, "_evidence_observer", None) is not None:
            raise RuntimeError("ratio-4 evidence observation cannot be nested")
        observed: list[Ratio4AttentionEvidence] = []
        self._evidence_observer = observed
        try:
            yield observed
        finally:
            self._evidence_observer = None

    @contextmanager
    def observe_topk_indices(self) -> Iterator[list[torch.Tensor]]:
        """Collect only owned top-k snapshots for eager backend preflight."""

        if getattr(self, "_topk_observer", None) is not None:
            raise RuntimeError("ratio-4 top-k observation cannot be nested")
        observed: list[torch.Tensor] = []
        self._topk_observer = observed
        try:
            yield observed
        finally:
            self._topk_observer = None

    def prepare_decode_plan(
        self,
        start_pos: int,
        *,
        advance_overlap_state: bool = False,
    ) -> Ratio4DecodePlan:
        cfg = self.config
        if not isinstance(advance_overlap_state, bool):
            raise TypeError("advance_overlap_state must be bool")
        if (
            not isinstance(start_pos, int)
            or isinstance(start_pos, bool)
            or start_pos < WINDOW_SIZE
            or start_pos >= cfg.max_seq_len
        ):
            raise ValueError("decode start_pos must be in [128, max_seq_len)")
        if self.state.next_position != start_pos:
            raise ValueError("decode start_pos does not match ratio-4 state")
        compressed_before = start_pos // COMPRESS_RATIO
        if not bool(
            torch.all(self.state._compressed_count == compressed_before).item()
        ):
            raise RuntimeError("ratio-4 compressed-row metadata is inconsistent")
        absolute_raw = torch.arange(
            start_pos - WINDOW_SIZE,
            start_pos,
            dtype=torch.int64,
            device=self.state.device,
        )
        raw_slots = absolute_raw.remainder(WINDOW_SIZE)
        if not bool(
            torch.all(
                self.state._raw_positions.index_select(1, raw_slots)
                == absolute_raw.unsqueeze(0)
            ).item()
        ):
            raise RuntimeError("ratio-4 raw-ring metadata is inconsistent")
        expected_starts = torch.arange(
            0,
            compressed_before * COMPRESS_RATIO,
            COMPRESS_RATIO,
            dtype=torch.int64,
            device=self.state.device,
        )
        if not bool(
            torch.all(
                self.state._compressed_group_starts[:, :compressed_before]
                == expected_starts.unsqueeze(0)
            ).item()
        ):
            raise RuntimeError("ratio-4 compressed-row starts are inconsistent")
        phase = start_pos % COMPRESS_RATIO
        expected_previous = torch.arange(
            start_pos - phase - COMPRESS_RATIO,
            start_pos - phase,
            dtype=torch.int64,
            device=self.state.device,
        )
        for positions in (
            self.state._main_state_positions,
            self.state._index_state_positions,
        ):
            if not bool(
                torch.all(
                    positions[:, :COMPRESS_RATIO]
                    == expected_previous.unsqueeze(0)
                ).item()
            ):
                raise RuntimeError("ratio-4 previous overlap metadata is inconsistent")
            if phase:
                expected_pending = torch.arange(
                    start_pos - phase,
                    start_pos,
                    dtype=torch.int64,
                    device=self.state.device,
                )
                if not bool(
                    torch.all(
                        positions[:, COMPRESS_RATIO : COMPRESS_RATIO + phase]
                        == expected_pending.unsqueeze(0)
                    ).item()
                ):
                    raise RuntimeError("ratio-4 pending overlap metadata is inconsistent")

        compressed_after = (start_pos + 1) // COMPRESS_RATIO
        index_topk_count = min(cfg.index_topk, compressed_after)
        if index_topk_count <= 0:
            raise ValueError("ratio-4 decode requires at least one compressed candidate")
        window = window_topk_indices(
            batch_size=self.state.num_local_sequences,
            seqlen=1,
            start_pos=start_pos,
            device=self.state.device,
        )
        sparse_width = WINDOW_SIZE + index_topk_count
        batch_indices = (
            torch.arange(
                self.state.num_local_sequences,
                dtype=torch.int64,
                device=self.state.device,
            )
            .view(self.state.num_local_sequences, 1, 1)
            .expand(self.state.num_local_sequences, 1, sparse_width)
        )
        group_start = start_pos + 1 - COMPRESS_RATIO
        return Ratio4DecodePlan(
            start_pos=start_pos,
            phase=phase,
            boundary=phase == COMPRESS_RATIO - 1,
            advance_overlap_state=advance_overlap_state,
            raw_slot=start_pos % WINDOW_SIZE,
            overlap_slot=COMPRESS_RATIO + phase,
            compressed_row=compressed_before,
            compressed_count_after=compressed_after,
            index_topk_count=index_topk_count,
            batch_size=self.state.num_local_sequences,
            hidden_size=cfg.hidden_size,
            owner_id=id(self),
            state_id=id(self.state),
            frequencies=self.freqs_cis[start_pos : start_pos + 1].contiguous(),
            group_start_frequencies=self.freqs_cis[
                group_start : group_start + 1
            ].contiguous(),
            window_indices=window.to(torch.int64),
            batch_indices=batch_indices,
            main_ape=self.weights.compressor_ape[phase],
            index_ape=self.weights.index_compressor_ape[phase],
        )

    def prepare_stateful_decode_plan(
        self,
        *,
        position: torch.Tensor,
        start_position: int,
        stop_position: int,
    ) -> Ratio4StatefulDecodePlan:
        """Allocate one fixed workspace for a saturated consecutive range."""

        cfg = self.config
        if getattr(self, "_evidence_observer", None) is not None or getattr(
            self, "_topk_observer", None
        ) is not None:
            raise RuntimeError("stateful ratio-4 decode forbids active observers")
        if (
            not isinstance(start_position, int)
            or isinstance(start_position, bool)
            or not isinstance(stop_position, int)
            or isinstance(stop_position, bool)
            or start_position < WINDOW_SIZE
            or stop_position <= start_position
            or stop_position > cfg.max_seq_len
        ):
            raise ValueError(
                "stateful ratio-4 range must be a non-empty interval within capacity"
            )
        if (
            not isinstance(position, torch.Tensor)
            or tuple(position.shape) != (1,)
            or position.dtype != torch.int64
            or position.device != self.state.device
            or not position.is_contiguous()
        ):
            raise ValueError("stateful ratio-4 position must be contiguous INT64 [1]")
        if int(position.item()) != start_position:
            raise ValueError("device position does not match ratio-4 range start")

        # Reuse the fixed-plan setup audit for every existing state invariant.
        self.prepare_decode_plan(start_position, advance_overlap_state=True)
        minimum_candidates = (start_position + 1) // COMPRESS_RATIO
        if minimum_candidates < cfg.index_topk:
            raise ValueError(
                "stateful ratio-4 decode requires a saturated fixed index top-k"
            )
        candidate_width = stop_position // COMPRESS_RATIO
        if candidate_width > self.state.compressed_capacity:
            raise ValueError("stateful ratio-4 candidate width exceeds state capacity")

        batch = self.state.num_local_sequences
        total_topk = WINDOW_SIZE + cfg.index_topk
        device = self.state.device
        window_columns = torch.arange(
            WINDOW_SIZE, dtype=torch.int64, device=device
        )
        compressed_columns = torch.arange(
            candidate_width, dtype=torch.int64, device=device
        )
        topk_indices = torch.empty(
            batch, 1, total_topk, dtype=torch.int64, device=device
        )
        batch_indices = (
            torch.arange(batch, dtype=torch.int64, device=device)
            .view(batch, 1, 1)
            .expand(batch, 1, total_topk)
            .contiguous()
        )
        tensors = (
            position,
            window_columns,
            compressed_columns,
            topk_indices,
            batch_indices,
        )
        tensor_pointers = tuple(
            int(value.untyped_storage().data_ptr()) for value in tensors
        )
        if len(set(tensor_pointers)) != len(tensor_pointers):
            raise RuntimeError("stateful ratio-4 plan tensors must not alias")
        return Ratio4StatefulDecodePlan(
            start_position=start_position,
            stop_position=stop_position,
            candidate_width=candidate_width,
            index_topk_count=cfg.index_topk,
            total_topk=total_topk,
            batch_size=batch,
            hidden_size=cfg.hidden_size,
            owner_id=id(self),
            state_id=id(self.state),
            position=position,
            window_columns=window_columns,
            compressed_columns=compressed_columns,
            topk_indices=topk_indices,
            batch_indices=batch_indices,
            tensor_pointers=tensor_pointers,
        )

    def _validate_stateful_decode_plan(
        self,
        hidden: torch.Tensor,
        plan: Ratio4StatefulDecodePlan,
        *,
        ratio4_boundary: bool,
    ) -> None:
        if not isinstance(plan, Ratio4StatefulDecodePlan):
            raise TypeError("plan must be a Ratio4StatefulDecodePlan")
        if plan.owner_id != id(self) or plan.state_id != id(self.state):
            raise ValueError("stateful plan belongs to another ratio-4 state")
        if not isinstance(ratio4_boundary, bool):
            raise TypeError("ratio4_boundary must be bool")
        if getattr(self, "_evidence_observer", None) is not None or getattr(
            self, "_topk_observer", None
        ) is not None:
            raise RuntimeError("stateful ratio-4 decode forbids active observers")
        if tuple(hidden.shape) != (plan.batch_size, 1, plan.hidden_size):
            raise ValueError("stateful ratio-4 hidden shape differs from its plan")
        if hidden.dtype != torch.bfloat16 or hidden.device != self.state.device:
            raise ValueError("stateful ratio-4 hidden must use state-local BF16")
        if (
            plan.index_topk_count != self.config.index_topk
            or plan.total_topk != WINDOW_SIZE + plan.index_topk_count
            or plan.candidate_width < plan.index_topk_count
        ):
            raise ValueError("stateful ratio-4 static widths are inconsistent")

        expected = (
            ("position", plan.position, (1,), torch.int64),
            (
                "window_columns",
                plan.window_columns,
                (WINDOW_SIZE,),
                torch.int64,
            ),
            (
                "compressed_columns",
                plan.compressed_columns,
                (plan.candidate_width,),
                torch.int64,
            ),
            (
                "topk_indices",
                plan.topk_indices,
                (plan.batch_size, 1, plan.total_topk),
                torch.int64,
            ),
            (
                "batch_indices",
                plan.batch_indices,
                (plan.batch_size, 1, plan.total_topk),
                torch.int64,
            ),
        )
        pointers = []
        for name, value, shape, dtype in expected:
            if tuple(value.shape) != shape:
                raise ValueError(f"stateful ratio-4 {name} shape differs")
            if value.dtype != dtype or value.device != self.state.device:
                raise ValueError(f"stateful ratio-4 {name} dtype/device differs")
            if not value.is_contiguous():
                raise ValueError(f"stateful ratio-4 {name} must be contiguous")
            pointers.append(int(value.untyped_storage().data_ptr()))
        if len(set(pointers)) != len(pointers):
            raise ValueError("stateful ratio-4 plan tensors must not alias")
        if tuple(pointers) != plan.tensor_pointers:
            raise ValueError("stateful ratio-4 tensor storage differs from setup")

    def _main_finalizer(
        self, pooled: torch.Tensor, frequencies: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        value = rms_norm(
            pooled.to(torch.bfloat16),
            self.weights.compressor_norm,
            eps=cfg.norm_eps,
        )
        value[..., -cfg.rope_dim :] = apply_rotary_emb(
            value[..., -cfg.rope_dim :], frequencies
        )
        kv_fp8_qat_prefix(
            value, value.shape[-1] - cfg.rope_dim, mode=self.kv_qat_mode
        )
        return value.contiguous()

    def _index_finalizer(
        self, pooled: torch.Tensor, frequencies: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        value = rms_norm(
            pooled.to(torch.bfloat16),
            self.weights.index_compressor_norm,
            eps=cfg.norm_eps,
        )
        value[..., -cfg.rope_dim :] = apply_rotary_emb(
            value[..., -cfg.rope_dim :], frequencies
        )
        return fp4_quant_dequant(hadamard_transform(value)).contiguous()

    @staticmethod
    def _write_overlap(
        projected: torch.Tensor,
        adjusted_score: torch.Tensor,
        *,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        plan: Ratio4DecodePlan,
        output_dim: int,
        finalizer: Any,
        output_cache: torch.Tensor,
        output_cache_rope: torch.Tensor | None = None,
        collect_evidence: bool = False,
    ) -> _Ratio4OverlapEvidence | None:
        kv_state[:, plan.overlap_slot].copy_(projected[:, 0])
        score_state[:, plan.overlap_slot].copy_(adjusted_score)
        if not plan.boundary:
            return None
        pooled = overlap_pool(kv_state, score_state, output_dim=output_dim)
        finalized = finalizer(pooled, plan.group_start_frequencies)
        output_cache[:, plan.compressed_row : plan.compressed_row + 1].copy_(
            quantize_latent_rows(finalized, output_cache.dtype)
        )
        if output_cache_rope is not None:
            output_cache_rope[
                :, plan.compressed_row : plan.compressed_row + 1
            ].copy_(finalized[..., -LATENT_ROPE_DIM:])
        evidence = None
        if collect_evidence:
            evidence = _Ratio4OverlapEvidence(
                values=torch.cat(
                    (
                        kv_state[:, :COMPRESS_RATIO, :output_dim],
                        kv_state[:, COMPRESS_RATIO:, output_dim:],
                    ),
                    dim=1,
                ),
                logits=torch.cat(
                    (
                        score_state[:, :COMPRESS_RATIO, :output_dim],
                        score_state[:, COMPRESS_RATIO:, output_dim:],
                    ),
                    dim=1,
                ),
                pooled=pooled,
                finalized=finalized,
            )
        if plan.advance_overlap_state:
            kv_state[:, :COMPRESS_RATIO].copy_(kv_state[:, COMPRESS_RATIO:])
            score_state[:, :COMPRESS_RATIO].copy_(score_state[:, COMPRESS_RATIO:])
        return evidence

    def forward_stateful_decode_tensor(
        self,
        hidden: torch.Tensor,
        *,
        plan: Ratio4StatefulDecodePlan,
        ratio4_boundary: bool,
        stage_marker: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Run one cursor-driven ratio-4 token without host value reads."""

        self._validate_stateful_decode_plan(
            hidden, plan, ratio4_boundary=ratio4_boundary
        )
        cfg = self.config
        weights = self.weights
        state = self.state
        position = plan.position
        phase = position.remainder(COMPRESS_RATIO)
        frequencies = self.freqs_cis.index_select(0, position)
        group_start_frequencies = self.freqs_cis.index_select(
            0, position + 1 - COMPRESS_RATIO
        )
        main_ape = weights.compressor_ape.index_select(0, phase)[0]
        index_ape = weights.index_compressor_ape.index_select(0, phase)[0]

        projection_backend = getattr(self, "_projection_backend", None)
        hidden_projections = None
        if projection_backend is None:
            projected_wq_a = F.linear(hidden, weights.wq_a)
        else:
            hidden_projections = _project_hidden_with_backend(
                projection_backend,
                hidden,
                wq_a=weights.wq_a,
                wkv=weights.wkv,
            )
            projected_wq_a = hidden_projections.wq_a
        query_lora = rms_norm(
            projected_wq_a, weights.q_norm, eps=cfg.norm_eps
        )
        query_projections = None
        if projection_backend is None:
            projected_wq_b = F.linear(query_lora, weights.wq_b)
        else:
            query_projections = _project_query_with_backend(
                projection_backend,
                query_lora,
                wq_b=weights.wq_b,
                index_wq_b=weights.index_wq_b,
            )
            projected_wq_b = query_projections.wq_b
        query = projected_wq_b.reshape(
            plan.batch_size, 1, cfg.local_num_heads, cfg.head_dim
        )
        query *= torch.rsqrt(
            query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps
        )
        query[..., -cfg.rope_dim :] = apply_rotary_emb(
            query[..., -cfg.rope_dim :], frequencies
        )
        if stage_marker is not None:
            stage_marker("query_done")

        projected_wkv = (
            F.linear(hidden, weights.wkv)
            if hidden_projections is None
            else hidden_projections.wkv
        )
        raw_latent = rms_norm(projected_wkv, weights.kv_norm, eps=cfg.norm_eps)
        raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
            raw_latent[..., -cfg.rope_dim :], frequencies
        )
        kv_fp8_qat_prefix(
            raw_latent, raw_latent.shape[-1] - cfg.rope_dim, mode=self.kv_qat_mode
        )
        if stage_marker is not None:
            stage_marker("raw_kv_done")

        main_projected = F.linear(hidden.float(), weights.compressor_wkv)
        main_score = F.linear(hidden.float(), weights.compressor_wgate)
        main_adjusted_score = main_score[:, 0] + main_ape
        index_projected = F.linear(hidden.float(), weights.index_compressor_wkv)
        index_score = F.linear(hidden.float(), weights.index_compressor_wgate)
        index_adjusted_score = index_score[:, 0] + index_ape
        if stage_marker is not None:
            stage_marker("compressor_projection_done")
        state._write_decode_stateful_prevalidated(
            raw_latent,
            main_projected,
            main_adjusted_score,
            index_projected,
            index_adjusted_score,
            position=position,
            boundary=ratio4_boundary,
            group_start_frequencies=group_start_frequencies,
            main_finalizer=self._main_finalizer,
            index_finalizer=self._index_finalizer,
        )
        if stage_marker is not None:
            stage_marker("state_write_done")

        projected_index_wq_b = (
            F.linear(query_lora, weights.index_wq_b)
            if query_projections is None
            else query_projections.index_wq_b
        )
        if projected_index_wq_b is None:
            raise AssertionError("validated ratio-4 index projection is missing")
        index_query = projected_index_wq_b.reshape(
            plan.batch_size, 1, cfg.index_n_heads, cfg.index_head_dim
        )
        index_query[..., -cfg.rope_dim :] = apply_rotary_emb(
            index_query[..., -cfg.rope_dim :], frequencies
        )
        index_query = self._indexer_qat(index_query)
        index_weights = F.linear(hidden, weights.index_weights_proj) * (
            cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5
        )
        if stage_marker is not None:
            stage_marker("index_query_done")
        index_kv = state.indexer_kv[:, : plan.candidate_width]
        if _HALF_ACCUM:
            index_kv_b = (
                index_kv
                if index_kv.dtype == torch.bfloat16
                else index_kv.to(torch.bfloat16)
            )
            scores = torch.einsum("bshd,btd->bsht", index_query, index_kv_b)
            scores = (
                scores.relu_()
                .mul_(index_weights.unsqueeze(-1))
                .sum(dim=2)
                .float()
            )
        else:
            scores = torch.einsum(
                "bshd,btd->bsht", index_query.float(), index_kv.float()
            )
            # 17th vertical: in-place relu/scale (bitwise-identical
            # elementwise values, two fewer (b,1,h,t) fp32 temporaries in
            # the capture pool).
            scores = (
                scores.relu_()
                .mul_(index_weights.float().unsqueeze(-1))
                .sum(dim=2)
            )
        compressed_after = torch.div(
            position + 1, COMPRESS_RATIO, rounding_mode="floor"
        )
        visible = plan.compressed_columns.lt(compressed_after)
        scores.masked_fill_(~visible.view(1, 1, plan.candidate_width), float("-inf"))
        compressed_indices = scores.topk(
            plan.index_topk_count, dim=-1
        ).indices
        window = (
            plan.window_columns + position.remainder(WINDOW_SIZE) + 1
        ).remainder(WINDOW_SIZE)
        plan.topk_indices[..., :WINDOW_SIZE].copy_(window.view(1, 1, WINDOW_SIZE))
        plan.topk_indices[..., WINDOW_SIZE:].copy_(
            compressed_indices + WINDOW_SIZE
        )
        if stage_marker is not None:
            stage_marker("index_topk_done")

        sparse_attention_backend = getattr(
            self, "_sparse_attention_backend", None
        )
        if sparse_attention_backend is None:
            # 17th vertical (workspace slimming): one FP32 materialization of
            # the gathered rows + in-place softmax steps.  fp8/bf16 -> fp32
            # conversions are exact and the in-place elementwise kernels
            # compute the same values, so the einsum inputs -- and the output
            # -- are bitwise identical to the previous
            # gather -> bf16 -> double-``.float()`` chain.  With
            # DSV4_R4_HALF_ACCUM=1 (leverage-3 experiment) the gathered rows
            # stay BF16 and the score/output einsums run on BF16 operands.
            selected = state.latent[plan.batch_indices, plan.topk_indices]
            if _HALF_ACCUM:
                if selected.dtype != torch.bfloat16:
                    selected = selected.to(torch.bfloat16)
            else:
                selected = selected.float()
            if state.latent_rope is not None:
                selected[..., -LATENT_ROPE_DIM:] = state.latent_rope[
                    plan.batch_indices, plan.topk_indices
                ]
            if _HALF_ACCUM:
                attention_scores = torch.einsum(
                    "bshd,bskd->bshk", query, selected
                ).float() * (cfg.head_dim**-0.5)
            else:
                attention_scores = torch.einsum(
                    "bshd,bskd->bshk", query.float(), selected
                ) * (cfg.head_dim**-0.5)
            sink = weights.attn_sink.float().view(1, 1, cfg.local_num_heads, 1)
            maximum = torch.maximum(
                attention_scores.amax(dim=-1, keepdim=True), sink
            )
            exponent = attention_scores.sub_(maximum).exp_()
            denominator = exponent.sum(dim=-1, keepdim=True) + torch.exp(
                sink - maximum
            )
            probabilities = exponent.div_(denominator)
            if _HALF_ACCUM:
                probabilities = probabilities.to(torch.bfloat16)
            sparse_output = torch.einsum(
                "bshk,bskd->bshd", probabilities, selected
            ).to(query.dtype)
        else:
            sparse_output = sparse_attention_backend(
                query,
                state.latent,
                weights.attn_sink,
                plan.topk_indices,
                plan.batch_indices,
                cfg.head_dim**-0.5,
            )
            if not isinstance(sparse_output, torch.Tensor):
                raise TypeError("ratio-4 sparse backend must return a tensor")
            if tuple(sparse_output.shape) != tuple(query.shape):
                raise ValueError("ratio-4 sparse backend output shape differs")
            if (
                sparse_output.dtype != query.dtype
                or sparse_output.device != query.device
            ):
                raise ValueError(
                    "ratio-4 sparse backend output dtype/device differs"
                )
            if not sparse_output.is_contiguous():
                raise ValueError("ratio-4 sparse backend output must be contiguous")
        if stage_marker is not None:
            stage_marker("sparse_done")

        sparse_output[..., -cfg.rope_dim :] = apply_rotary_emb(
            sparse_output[..., -cfg.rope_dim :], frequencies, inverse=True
        )
        grouped = sparse_output.reshape(
            plan.batch_size,
            1,
            cfg.local_o_groups,
            cfg.group_width,
        )
        if stage_marker is not None:
            stage_marker("output_transform_done")
        wo_a = weights.wo_a.reshape(
            cfg.local_o_groups,
            cfg.o_lora_rank,
            cfg.group_width,
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        output_lora = projected.flatten(2)
        if stage_marker is not None:
            stage_marker("wo_a_done")
        if projection_backend is None:
            final_output = F.linear(output_lora, weights.wo_b)
        else:
            final_output = _project_output_with_backend(
                projection_backend,
                output_lora,
                wo_b=weights.wo_b,
            )
        if stage_marker is not None:
            stage_marker("output_done")
        return final_output

    def forward_decode_tensor(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int,
        plan: Ratio4DecodePlan,
    ) -> torch.Tensor:
        if not isinstance(plan, Ratio4DecodePlan):
            raise TypeError("plan must be a Ratio4DecodePlan")
        if plan.owner_id != id(self) or plan.state_id != id(self.state):
            raise ValueError("decode plan belongs to a different ratio-4 state")
        if start_pos != plan.start_pos:
            raise ValueError("decode start_pos does not match the fixed plan")
        if tuple(hidden.shape) != (plan.batch_size, 1, plan.hidden_size):
            raise ValueError("decode hidden shape does not match the fixed plan")
        if hidden.dtype != torch.bfloat16 or hidden.device != plan.frequencies.device:
            raise ValueError("ratio-4 decode hidden must use plan-local CUDA BF16")

        cfg = self.config
        weights = self.weights
        state = self.state
        evidence_observer = getattr(self, "_evidence_observer", None)
        sparse_attention_backend = getattr(
            self, "_sparse_attention_backend", None
        )
        if evidence_observer is not None and sparse_attention_backend is not None:
            raise RuntimeError(
                "ratio-4 evidence observation requires the default control backend"
            )
        collect_evidence = evidence_observer is not None
        projection_backend = getattr(self, "_projection_backend", None)
        hidden_projections = None
        if projection_backend is None:
            projected_wq_a = F.linear(hidden, weights.wq_a)
        else:
            hidden_projections = _project_hidden_with_backend(
                projection_backend,
                hidden,
                wq_a=weights.wq_a,
                wkv=weights.wkv,
            )
            projected_wq_a = hidden_projections.wq_a
        query_lora = rms_norm(
            projected_wq_a, weights.q_norm, eps=cfg.norm_eps
        )
        query_projections = None
        if projection_backend is None:
            projected_wq_b = F.linear(query_lora, weights.wq_b)
        else:
            query_projections = _project_query_with_backend(
                projection_backend,
                query_lora,
                wq_b=weights.wq_b,
                index_wq_b=weights.index_wq_b,
            )
            projected_wq_b = query_projections.wq_b
        query = projected_wq_b.reshape(
            plan.batch_size, 1, cfg.local_num_heads, cfg.head_dim
        )
        query *= torch.rsqrt(
            query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps
        )
        query[..., -cfg.rope_dim :] = apply_rotary_emb(
            query[..., -cfg.rope_dim :], plan.frequencies
        )

        projected_wkv = (
            F.linear(hidden, weights.wkv)
            if hidden_projections is None
            else hidden_projections.wkv
        )
        raw_latent = rms_norm(projected_wkv, weights.kv_norm, eps=cfg.norm_eps)
        raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
            raw_latent[..., -cfg.rope_dim :], plan.frequencies
        )
        kv_fp8_qat_prefix(
            raw_latent, raw_latent.shape[-1] - cfg.rope_dim, mode=self.kv_qat_mode
        )
        state.raw[:, plan.raw_slot].copy_(state._quantize_rows(raw_latent[:, 0]))
        if state.raw_rope is not None:
            state.raw_rope[:, plan.raw_slot].copy_(
                raw_latent[:, 0, -LATENT_ROPE_DIM:]
            )

        main_projected = F.linear(hidden.float(), weights.compressor_wkv)
        main_score = F.linear(hidden.float(), weights.compressor_wgate)
        main_adjusted_score = main_score[:, 0] + plan.main_ape
        main_overlap = self._write_overlap(
            main_projected,
            main_adjusted_score,
            kv_state=state.main_kv_state,
            score_state=state.main_score_state,
            plan=plan,
            output_dim=LATENT_DIM,
            finalizer=self._main_finalizer,
            output_cache=state.compressed,
            output_cache_rope=state.compressed_rope,
            collect_evidence=collect_evidence,
        )

        index_projected = F.linear(hidden.float(), weights.index_compressor_wkv)
        index_score = F.linear(hidden.float(), weights.index_compressor_wgate)
        index_adjusted_score = index_score[:, 0] + plan.index_ape
        index_overlap = self._write_overlap(
            index_projected,
            index_adjusted_score,
            kv_state=state.index_kv_state,
            score_state=state.index_score_state,
            plan=plan,
            output_dim=INDEX_DIM,
            finalizer=self._index_finalizer,
            output_cache=state.indexer_kv,
            collect_evidence=collect_evidence,
        )

        state._raw_positions[:, plan.raw_slot].fill_(start_pos)
        state._main_state_positions[:, plan.overlap_slot].fill_(start_pos)
        state._index_state_positions[:, plan.overlap_slot].fill_(start_pos)
        if plan.boundary and plan.advance_overlap_state:
            state._main_state_positions[:, :COMPRESS_RATIO].copy_(
                state._main_state_positions[:, COMPRESS_RATIO:]
            )
            state._index_state_positions[:, :COMPRESS_RATIO].copy_(
                state._index_state_positions[:, COMPRESS_RATIO:]
            )
        if plan.boundary:
            state._compressed_group_starts[:, plan.compressed_row].fill_(
                start_pos + 1 - COMPRESS_RATIO
            )
            state._compressed_count.fill_(plan.compressed_count_after)
        state._next_position.fill_(start_pos + 1)

        projected_index_wq_b = (
            F.linear(query_lora, weights.index_wq_b)
            if query_projections is None
            else query_projections.index_wq_b
        )
        if projected_index_wq_b is None:
            raise AssertionError("validated ratio-4 index projection is missing")
        index_query = projected_index_wq_b.reshape(
            plan.batch_size, 1, cfg.index_n_heads, cfg.index_head_dim
        )
        index_query[..., -cfg.rope_dim :] = apply_rotary_emb(
            index_query[..., -cfg.rope_dim :], plan.frequencies
        )
        index_query = self._indexer_qat(index_query)
        index_weights = F.linear(hidden, weights.index_weights_proj) * (
            cfg.index_head_dim**-0.5 * cfg.index_n_heads**-0.5
        )
        index_kv = state.indexer_kv[:, : plan.compressed_count_after]
        if _HALF_ACCUM:
            index_kv_b = (
                index_kv
                if index_kv.dtype == torch.bfloat16
                else index_kv.to(torch.bfloat16)
            )
            scores = torch.einsum("bshd,btd->bsht", index_query, index_kv_b)
            scores = (
                scores.relu_()
                .mul_(index_weights.unsqueeze(-1))
                .sum(dim=2)
                .float()
            )
        else:
            scores = torch.einsum(
                "bshd,btd->bsht", index_query.float(), index_kv.float()
            )
            # 17th vertical: in-place relu/scale (bitwise-identical values).
            scores = (
                scores.relu_()
                .mul_(index_weights.float().unsqueeze(-1))
                .sum(dim=2)
            )
        compressed_indices = scores.topk(plan.index_topk_count, dim=-1).indices
        topk = torch.cat(
            (plan.window_indices, compressed_indices + WINDOW_SIZE), dim=-1
        )
        topk_observer = getattr(self, "_topk_observer", None)
        if topk_observer is not None:
            topk_observer.append(topk.detach().clone())

        selected = None
        if sparse_attention_backend is None:
            # 17th vertical: single FP32 materialization + in-place softmax
            # (bitwise identical; see forward_stateful_decode_tensor).  With
            # DSV4_R4_HALF_ACCUM=1 the BF16 leverage-3 experiment applies.
            selected = state.latent[plan.batch_indices, topk]
            if _HALF_ACCUM:
                if selected.dtype != torch.bfloat16:
                    selected = selected.to(torch.bfloat16)
            else:
                selected = selected.float()
            if state.latent_rope is not None:
                selected[..., -LATENT_ROPE_DIM:] = state.latent_rope[
                    plan.batch_indices, topk
                ]
            if _HALF_ACCUM:
                attention_scores = torch.einsum(
                    "bshd,bskd->bshk", query, selected
                ).float() * (cfg.head_dim**-0.5)
            else:
                attention_scores = torch.einsum(
                    "bshd,bskd->bshk", query.float(), selected
                ) * (cfg.head_dim**-0.5)
            sink = weights.attn_sink.float().view(1, 1, cfg.local_num_heads, 1)
            maximum = torch.maximum(
                attention_scores.amax(dim=-1, keepdim=True), sink
            )
            exponent = attention_scores.sub_(maximum).exp_()
            denominator = exponent.sum(dim=-1, keepdim=True) + torch.exp(
                sink - maximum
            )
            probabilities = exponent.div_(denominator)
            if _HALF_ACCUM:
                probabilities = probabilities.to(torch.bfloat16)
            sparse_output = torch.einsum(
                "bshk,bskd->bshd", probabilities, selected
            ).to(query.dtype)
        else:
            sparse_output = sparse_attention_backend(
                query,
                state.latent,
                weights.attn_sink,
                topk,
                plan.batch_indices,
                cfg.head_dim**-0.5,
            )
            if not isinstance(sparse_output, torch.Tensor):
                raise TypeError("ratio-4 sparse backend must return a tensor")
            if tuple(sparse_output.shape) != tuple(query.shape):
                raise ValueError("ratio-4 sparse backend output shape differs")
            if sparse_output.dtype != query.dtype or sparse_output.device != query.device:
                raise ValueError("ratio-4 sparse backend output dtype/device differs")
            if not sparse_output.is_contiguous():
                raise ValueError("ratio-4 sparse backend output must be contiguous")
        sparse_snapshot = (
            _evidence_snapshot(sparse_output) if collect_evidence else None
        )
        sparse_output[..., -cfg.rope_dim :] = apply_rotary_emb(
            sparse_output[..., -cfg.rope_dim :], plan.frequencies, inverse=True
        )
        inverse_rotated = sparse_output
        grouped = inverse_rotated.reshape(
            plan.batch_size,
            1,
            cfg.local_o_groups,
            cfg.group_width,
        )
        wo_a = weights.wo_a.reshape(
            cfg.local_o_groups,
            cfg.o_lora_rank,
            cfg.group_width,
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        output_lora = projected.flatten(2)
        if projection_backend is None:
            branch = F.linear(output_lora, weights.wo_b)
        else:
            branch = _project_output_with_backend(
                projection_backend,
                output_lora,
                wo_b=weights.wo_b,
            )

        if evidence_observer is not None:
            if selected is None:
                raise AssertionError("control evidence lost selected sparse KV")

            def overlap_value(
                evidence: _Ratio4OverlapEvidence | None,
                name: str,
            ) -> torch.Tensor | None:
                return None if evidence is None else getattr(evidence, name)

            evidence_observer.append(
                Ratio4AttentionEvidence(
                    query_lora=_evidence_snapshot(query_lora),
                    query=_evidence_snapshot(query),
                    raw_latent=_evidence_snapshot(raw_latent),
                    main_projected_kv=_evidence_snapshot(main_projected),
                    main_projected_score=_evidence_snapshot(main_score),
                    main_adjusted_score=_evidence_snapshot(main_adjusted_score),
                    main_overlap_values=_evidence_snapshot(
                        overlap_value(main_overlap, "values")
                    ),
                    main_overlap_logits=_evidence_snapshot(
                        overlap_value(main_overlap, "logits")
                    ),
                    main_compression_pooled=_evidence_snapshot(
                        overlap_value(main_overlap, "pooled")
                    ),
                    main_compression_finalized=_evidence_snapshot(
                        overlap_value(main_overlap, "finalized")
                    ),
                    index_projected_kv=_evidence_snapshot(index_projected),
                    index_projected_score=_evidence_snapshot(index_score),
                    index_adjusted_score=_evidence_snapshot(index_adjusted_score),
                    index_overlap_values=_evidence_snapshot(
                        overlap_value(index_overlap, "values")
                    ),
                    index_overlap_logits=_evidence_snapshot(
                        overlap_value(index_overlap, "logits")
                    ),
                    index_compression_pooled=_evidence_snapshot(
                        overlap_value(index_overlap, "pooled")
                    ),
                    index_compression_finalized=_evidence_snapshot(
                        overlap_value(index_overlap, "finalized")
                    ),
                    index_query=_evidence_snapshot(index_query),
                    index_weights=_evidence_snapshot(index_weights),
                    index_scores=_evidence_snapshot(scores),
                    compressed_indices=_evidence_snapshot(compressed_indices),
                    topk_indices=_evidence_snapshot(topk),
                    # selected is now held in FP32 (exact); evidence keeps the
                    # oracle's BF16 contract -- every value is
                    # bf16-representable, so this cast is bitwise-lossless.
                    selected_kv=_evidence_snapshot(
                        None if selected is None else selected.to(torch.bfloat16)
                    ),
                    sparse_output=sparse_snapshot,
                    inverse_rotated=_evidence_snapshot(inverse_rotated),
                    output_lora=_evidence_snapshot(projected),
                    branch=_evidence_snapshot(branch),
                )
            )
        return branch


__all__ = [
    "PreparedRatio4AttentionWeights",
    "Ratio4AttentionEvidence",
    "Ratio4AttentionConfig",
    "Ratio4DecodePlan",
    "Ratio4StatefulDecodePlan",
    "Ratio4SparseAttentionBackend",
    "Ratio4TorchAttention",
    "fp4_quant_dequant",
    "hadamard_transform",
    "overlap_pool",
    "prepare_ratio4_attention_weights",
    "shard_ratio4_attention_weights",
]
