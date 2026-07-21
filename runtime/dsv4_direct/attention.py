"""Direct-owned ratio-128 attention path for frozen physical layers.

This module makes the intended dataflow explicit with plain PyTorch. The weight
projection and sparse accumulation modes are controls, not a semantic oracle or
a performance path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, MutableMapping, Protocol

import torch
import torch.nn.functional as F

from .block_weights import ResidentAttentionWeights
from .moe_forward import dequant_fp8_block
from .model_contract import SUPPORTED_LAYER_SPECS, validate_model_layer_config
from .stateful_decode import (
    build_padded_ratio128_sparse_indices,
    ratio128_sparse_bucket_width,
)
from .static_kv import COMPRESS_RATIO, LATENT_DIM, WINDOW_SIZE, StaticLayerKV


SUPPORTED_RATIO128_LAYER_IDS = tuple(
    layer_id
    for layer_id, specification in SUPPORTED_LAYER_SPECS.items()
    if specification["compress_ratio"] == 128
)


@dataclass(frozen=True)
class Ratio128AttentionConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    norm_eps: float
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_seq_len: int
    max_seq_len: int
    layer_id: int = 3

    @classmethod
    def from_model_config(
        cls,
        config: Mapping[str, Any],
        *,
        layer_id: int = 3,
        max_seq_len: int,
    ) -> "Ratio128AttentionConfig":
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id not in SUPPORTED_RATIO128_LAYER_IDS
        ):
            raise ValueError(
                "ratio-128 attention config requires an integer frozen "
                "ratio-128 layer_id, "
                f"got {layer_id!r}"
            )
        validate_model_layer_config(config, layer_id=layer_id)
        ratios = config.get("compress_ratios")
        if not isinstance(ratios, (list, tuple)) or len(ratios) <= layer_id:
            raise ValueError(f"compress_ratios does not cover layer {layer_id}")
        if int(ratios[layer_id]) != COMPRESS_RATIO:
            raise ValueError(
                f"layer {layer_id} requires ratio {COMPRESS_RATIO}, got {ratios[layer_id]}"
            )
        rope = config.get("rope_scaling") or {}
        result = cls(
            hidden_size=int(config["hidden_size"]),
            num_heads=int(config["num_attention_heads"]),
            head_dim=int(config["head_dim"]),
            rope_dim=int(config["qk_rope_head_dim"]),
            q_lora_rank=int(config["q_lora_rank"]),
            o_lora_rank=int(config["o_lora_rank"]),
            o_groups=int(config["o_groups"]),
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
            or self.layer_id not in SUPPORTED_RATIO128_LAYER_IDS
        ):
            raise ValueError(
                "ratio-128 attention config requires an integer frozen "
                "ratio-128 layer_id, "
                f"got {self.layer_id!r}"
            )
        # DeepSeek-V4-Flash geometry, frozen from the checkpoint config.json
        # (see model_contract.EXPECTED_RATIO128_CONFIG): hidden 4096, 64 heads,
        # q_lora 1024, o_groups 8.  head_dim 512 (== LATENT_DIM), rope 64, and
        # o_lora 1024 are unchanged from Pro.
        expected = {
            "hidden_size": (self.hidden_size, 4096),
            "num_heads": (self.num_heads, 64),
            "head_dim": (self.head_dim, LATENT_DIM),
            "rope_dim": (self.rope_dim, 64),
            "q_lora_rank": (self.q_lora_rank, 1024),
            "o_lora_rank": (self.o_lora_rank, 1024),
            "o_groups": (self.o_groups, 8),
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
        if self.rope_dim <= 0 or self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError("rope_dim must be positive, even, and no larger than head_dim")
        if self.num_heads % self.o_groups:
            raise ValueError("num_heads must divide output groups")
        if not math.isfinite(self.norm_eps) or not math.isfinite(self.rope_theta):
            raise ValueError("attention numerical constants must be finite")
        if self.max_seq_len < COMPRESS_RATIO or self.max_seq_len % COMPRESS_RATIO:
            raise ValueError("max_seq_len must be a positive multiple of 128")


@dataclass
class PreparedAttentionWeights:
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
    layer_id: int
    rank: int
    world_size: int
    checkpoint_id: str

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in self.__dict__.values()
            if isinstance(tensor, torch.Tensor)
        )


@dataclass(frozen=True)
class AttentionTrace:
    start_pos: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    query_shape: tuple[int, ...]
    attention_kv_shape: tuple[int, ...]
    topk_shape: tuple[int, ...]
    valid_topk_min: int
    valid_topk_max: int
    compressed_rows_written: tuple[int, ...]
    weight_projection_mode: str
    nope_quant_mode: str
    sparse_accumulation_mode: str
    path: str = "torch_ratio128_diagnostic_control"


@dataclass(frozen=True)
class Ratio128DecodePlan:
    """Fixed-position tensors and slots for trace-free single-token decode."""

    start_pos: int
    slot: int
    batch_size: int
    hidden_size: int
    owner_id: int
    state_id: int
    frequencies: torch.Tensor
    topk_indices: torch.Tensor
    gather_indices: torch.Tensor
    batch_indices: torch.Tensor
    compressor_ape: torch.Tensor


@dataclass(frozen=True)
class Ratio128StatefulDecodePlan:
    """Cursor-driven fixed-shape workspace for a consecutive decode range."""

    start_position: int
    stop_position: int
    bucket_width: int
    batch_size: int
    hidden_size: int
    owner_id: int
    state_id: int
    position: torch.Tensor
    topk_indices: torch.Tensor
    gather_indices: torch.Tensor
    valid_mask: torch.Tensor
    batch_indices: torch.Tensor
    tensor_pointers: tuple[int, ...]

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(value.numel() * value.element_size())
            for value in (
                self.topk_indices,
                self.gather_indices,
                self.valid_mask,
                self.batch_indices,
            )
        )


class Ratio128SparseAttentionBackend(Protocol):
    """Injectable sparse core for an otherwise unchanged ratio-128 attention."""

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
class AttentionHiddenProjections:
    """Outputs sharing one hidden-activation projection preparation."""

    wq_a: torch.Tensor
    wkv: torch.Tensor


@dataclass(frozen=True)
class AttentionQueryProjections:
    """Outputs sharing one normalized query-LORA projection preparation."""

    wq_b: torch.Tensor
    index_wq_b: torch.Tensor | None


class AttentionProjectionBackend(Protocol):
    """Checkpoint-native ordinary-attention projection bundle.

    Grouping the hidden and query projections makes activation sharing explicit
    and avoids replay correctness depending on mutable call-order caches.
    """

    compress_ratio: int

    def project_hidden(
        self, hidden: torch.Tensor
    ) -> AttentionHiddenProjections: ...

    def project_query(
        self, query_lora: torch.Tensor
    ) -> AttentionQueryProjections: ...

    def project_output(self, output_lora: torch.Tensor) -> torch.Tensor: ...


def _validate_attention_projection_backend(
    backend: AttentionProjectionBackend | None,
    *,
    expected_compress_ratio: int,
) -> None:
    if backend is None:
        return
    compress_ratio = getattr(backend, "compress_ratio", None)
    if (
        not isinstance(compress_ratio, int)
        or isinstance(compress_ratio, bool)
        or compress_ratio != expected_compress_ratio
    ):
        raise ValueError(
            "attention projection backend compress ratio differs: "
            f"observed={compress_ratio!r}, expected={expected_compress_ratio}"
        )
    for method_name in ("project_hidden", "project_query", "project_output"):
        if not callable(getattr(backend, method_name, None)):
            raise TypeError(
                f"attention projection backend {method_name} must be callable"
            )


def _validate_projection_tensor(
    name: str,
    value: Any,
    *,
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"attention projection backend {name} must return a tensor")
    expected_shape = (*activation.shape[:-1], weight.shape[0])
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"attention projection backend {name} shape differs: "
            f"observed={tuple(value.shape)}, expected={expected_shape}"
        )
    if value.dtype != activation.dtype or value.device != activation.device:
        raise ValueError(
            f"attention projection backend {name} dtype/device differs"
        )
    if not value.is_contiguous():
        raise ValueError(
            f"attention projection backend {name} output must be contiguous"
        )
    return value


def _project_hidden_with_backend(
    backend: AttentionProjectionBackend,
    hidden: torch.Tensor,
    *,
    wq_a: torch.Tensor,
    wkv: torch.Tensor,
) -> AttentionHiddenProjections:
    projected = backend.project_hidden(hidden)
    if not isinstance(projected, AttentionHiddenProjections):
        raise TypeError(
            "attention projection backend project_hidden must return "
            "AttentionHiddenProjections"
        )
    return AttentionHiddenProjections(
        wq_a=_validate_projection_tensor(
            "wq_a", projected.wq_a, activation=hidden, weight=wq_a
        ),
        wkv=_validate_projection_tensor(
            "wkv", projected.wkv, activation=hidden, weight=wkv
        ),
    )


def _project_query_with_backend(
    backend: AttentionProjectionBackend,
    query_lora: torch.Tensor,
    *,
    wq_b: torch.Tensor,
    index_wq_b: torch.Tensor | None,
) -> AttentionQueryProjections:
    projected = backend.project_query(query_lora)
    if not isinstance(projected, AttentionQueryProjections):
        raise TypeError(
            "attention projection backend project_query must return "
            "AttentionQueryProjections"
        )
    main = _validate_projection_tensor(
        "wq_b", projected.wq_b, activation=query_lora, weight=wq_b
    )
    if index_wq_b is None:
        if projected.index_wq_b is not None:
            raise ValueError(
                "ratio-128 attention projection backend returned index_wq_b"
            )
        index = None
    else:
        if projected.index_wq_b is None:
            raise ValueError(
                "ratio-4 attention projection backend omitted index_wq_b"
            )
        index = _validate_projection_tensor(
            "index_wq_b",
            projected.index_wq_b,
            activation=query_lora,
            weight=index_wq_b,
        )
    return AttentionQueryProjections(wq_b=main, index_wq_b=index)


def _project_output_with_backend(
    backend: AttentionProjectionBackend,
    output_lora: torch.Tensor,
    *,
    wo_b: torch.Tensor,
) -> torch.Tensor:
    return _validate_projection_tensor(
        "wo_b",
        backend.project_output(output_lora),
        activation=output_lora,
        weight=wo_b,
    )


def rms_norm(
    value: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    if value.shape[-1] != weight.numel() or weight.ndim != 1:
        raise ValueError(
            f"RMSNorm value/weight mismatch: {tuple(value.shape)} vs {tuple(weight.shape)}"
        )
    if not value.is_floating_point() or not weight.is_floating_point():
        raise TypeError("RMSNorm inputs must be floating point")
    value_fp32 = value.float()
    inverse_rms = torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (value_fp32 * inverse_rms * weight.float()).to(value.dtype)


def precompute_freqs_cis(
    *,
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device,
) -> torch.Tensor:
    if dim <= 0 or dim % 2 or seqlen <= 0:
        raise ValueError("RoPE dim must be positive/even and seqlen must be positive")

    def correction_dim(rotations: float) -> float:
        return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (
            2 * math.log(base)
        )

    frequencies = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
        )
    )
    if original_seq_len > 0:
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
            / (high - low)
        ).clamp(0, 1)
        smooth = 1 - ramp
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth
    positions = torch.arange(seqlen, dtype=torch.float32, device=device)
    phases = torch.outer(positions, frequencies)
    return torch.polar(torch.ones_like(phases), phases)


def apply_rotary_emb(
    value: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    if value.shape[-1] != freqs_cis.shape[-1] * 2:
        raise ValueError(
            f"RoPE value/frequency mismatch: {value.shape[-1]} vs {freqs_cis.shape[-1]}"
        )
    if value.shape[1] != freqs_cis.shape[0]:
        raise ValueError(
            f"RoPE sequence mismatch: {value.shape[1]} vs {freqs_cis.shape[0]}"
        )
    complex_value = torch.view_as_complex(
        value.float().reshape(*value.shape[:-1], -1, 2)
    )
    frequencies = freqs_cis.conj() if inverse else freqs_cis
    view_shape = [1, value.shape[1]] + [1] * (value.ndim - 3) + [freqs_cis.shape[-1]]
    rotated = torch.view_as_real(complex_value * frequencies.view(*view_shape)).flatten(-2)
    return rotated.to(value.dtype)


def fp8_quant_dequant(
    value: torch.Tensor,
    *,
    group_size: int = 64,
) -> torch.Tensor:
    """Simulate checkpoint QAT: power-of-two per-group FP8 then BF16 dequant."""

    if value.shape[-1] % group_size:
        raise ValueError(
            f"last dimension {value.shape[-1]} must be divisible by group_size {group_size}"
        )
    if value.dtype != torch.bfloat16:
        raise TypeError(f"FP8 QAT simulation requires BF16 input, got {value.dtype}")
    grouped = value.float().reshape(*value.shape[:-1], -1, group_size)
    absolute_max = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absolute_max / 448.0)))
    quantized = (grouped / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (quantized.float() * scale).reshape_as(value).to(value.dtype)


def window_topk_indices(
    *,
    batch_size: int,
    seqlen: int,
    start_pos: int,
    device: torch.device,
    window_size: int = WINDOW_SIZE,
) -> torch.Tensor:
    if batch_size <= 0 or seqlen <= 0 or start_pos < 0:
        raise ValueError("batch_size/seqlen must be positive and start_pos nonnegative")
    if start_pos > 0 and seqlen != 1:
        raise ValueError("incremental window indices require seqlen=1")
    if start_pos >= window_size - 1:
        ring = start_pos % window_size
        row = torch.cat(
            (
                torch.arange(ring + 1, window_size, device=device),
                torch.arange(0, ring + 1, device=device),
            )
        )
        matrix = row.view(1, 1, -1).expand(batch_size, 1, -1)
    elif start_pos > 0:
        row = F.pad(
            torch.arange(start_pos + 1, device=device),
            (0, window_size - start_pos - 1),
            value=-1,
        )
        matrix = row.view(1, 1, -1).expand(batch_size, 1, -1)
    else:
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        columns = torch.arange(min(seqlen, window_size), device=device)
        matrix = (base - window_size + 1).clamp(0) + columns
        matrix = torch.where(matrix > base, -1, matrix)
        matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)
    return matrix.to(torch.int32).contiguous()


def compressed_topk_indices(
    *,
    batch_size: int,
    seqlen: int,
    start_pos: int,
    offset: int,
    device: torch.device,
    ratio: int = COMPRESS_RATIO,
) -> torch.Tensor:
    if batch_size <= 0 or seqlen <= 0 or start_pos < 0 or offset < 0:
        raise ValueError("invalid compressed-index dimensions")
    if start_pos > 0:
        if seqlen != 1:
            raise ValueError("incremental compressed indices require seqlen=1")
        row = torch.arange((start_pos + 1) // ratio, device=device) + offset
        matrix = row.view(1, 1, -1).expand(batch_size, 1, -1)
    else:
        count = seqlen // ratio
        matrix = torch.arange(count, device=device).repeat(seqlen, 1)
        visible = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        matrix = torch.where(matrix >= visible, -1, matrix + offset)
        matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)
    return matrix.to(torch.int32).contiguous()


def _prefill_sparse_row_block() -> int | None:
    """Optional prefill sparse-core row block (C2F), from the environment.

    ``DSV4_PREFILL_SPARSE_ROW_BLOCK`` unset/empty/non-positive = disabled.
    """

    import os

    raw = os.environ.get("DSV4_PREFILL_SPARSE_ROW_BLOCK", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _prefill_sparse_backend() -> str:
    """Prefill-only sparse core selector (21st vertical), from the environment.

    ``DSV4_PREFILL_SPARSE_BACKEND`` unset = ``"torch"`` (the shipped masked
    einsum core).  ``"tilelang"`` swaps in the reference kernel for the
    ``start_pos == 0`` call only -- decode keeps the torch core and every
    plan-driven/stateful path is untouched.
    """

    from .ops.tilelang_sparse import resolve_prefill_sparse_backend

    return resolve_prefill_sparse_backend()


def torch_sparse_attention(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Small-shape reference sparse MLA without materializing a full context score."""

    if query.ndim != 4 or latent_kv.ndim != 3 or topk_indices.ndim != 3:
        raise ValueError("query/latent_kv/topk_indices must have ranks 4/3/3")
    batch, seqlen, heads, head_dim = query.shape
    if latent_kv.shape[0] != batch or latent_kv.shape[2] != head_dim:
        raise ValueError("query and latent KV shapes are incompatible")
    if tuple(topk_indices.shape[:2]) != (batch, seqlen):
        raise ValueError("top-k batch/sequence shape mismatch")
    if tuple(attn_sink.shape) != (heads,):
        raise ValueError(f"attn_sink shape {tuple(attn_sink.shape)} != ({heads},)")
    if query.device != latent_kv.device or query.device != topk_indices.device:
        raise ValueError("attention tensors must share one device")

    valid = topk_indices >= 0
    if bool((topk_indices[valid] >= latent_kv.shape[1]).any().item()):
        raise ValueError("top-k index exceeds latent KV capacity")
    safe = topk_indices.clamp_min(0).long()
    batch_index = (
        torch.arange(batch, device=query.device)
        .view(batch, 1, 1)
        .expand_as(safe)
    )
    selected = latent_kv[batch_index, safe]
    scores = torch.einsum(
        "bshd,bskd->bshk", query.float(), selected.float()
    ) * float(softmax_scale)
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, heads, 1)
    maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    exponent = torch.exp(scores - maximum).masked_fill(~valid.unsqueeze(2), 0.0)
    denominator = exponent.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
    probabilities = exponent / denominator
    output = torch.einsum("bshk,bskd->bshd", probabilities, selected.float())
    return output.to(query.dtype)


def _torch_sparse_decode_prevalidated(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    plan: Ratio128DecodePlan,
    softmax_scale: float,
    latent_rope: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fixed-index sparse MLA used after decode-plan validation.

    FP8 KV (A6F fp8_cast form): an e4m3 cache is cast back to BF16 right
    after the gather; an optional BF16 ``latent_rope`` side tensor overwrites
    the rope tail so positional lanes stay full precision.

    17th vertical (workspace slimming): the gathered rows are materialized in
    FP32 exactly once and the mask/softmax steps run in place.  e4m3/bf16 ->
    fp32 conversions are exact and the in-place elementwise kernels compute
    the same values, so every einsum consumes bitwise-identical inputs and
    the output is bitwise identical to the previous
    gather -> bf16 -> double-``.float()`` chain.
    """

    selected = latent_kv[plan.batch_indices, plan.gather_indices].float()
    if latent_rope is not None:
        selected[..., -latent_rope.shape[-1] :] = latent_rope[
            plan.batch_indices, plan.gather_indices
        ]
    scores = torch.einsum(
        "bshd,bskd->bshk", query.float(), selected
    ) * softmax_scale
    sink = attn_sink.float().view(1, 1, query.shape[2], 1)
    maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    exponent = scores.sub_(maximum).exp_()
    denominator = exponent.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
    probabilities = exponent.div_(denominator)
    output = torch.einsum("bshk,bskd->bshd", probabilities, selected)
    return output.to(query.dtype)


def _torch_sparse_decode_padded_prevalidated(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    plan: Ratio128StatefulDecodePlan,
    softmax_scale: float,
    latent_rope: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fixed-width sparse MLA with an explicit mask for ``-1`` padding.

    FP8 KV: same read-side cast form as the fixed-index variant above.

    17th vertical (workspace slimming): one FP32 materialization of the
    gathered rows plus in-place mask/softmax -- exact conversions and
    identical elementwise values, so the output stays bitwise identical to
    the previous chain (see ``_torch_sparse_decode_prevalidated``).
    """

    selected = latent_kv[plan.batch_indices, plan.gather_indices].float()
    if latent_rope is not None:
        selected[..., -latent_rope.shape[-1] :] = latent_rope[
            plan.batch_indices, plan.gather_indices
        ]
    selected.masked_fill_(~plan.valid_mask.unsqueeze(-1), 0.0)
    scores = torch.einsum(
        "bshd,bskd->bshk", query.float(), selected
    ) * softmax_scale
    valid = plan.valid_mask.unsqueeze(2)
    scores.masked_fill_(~valid, float("-inf"))
    sink = attn_sink.float().view(1, 1, query.shape[2], 1)
    maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    exponent = scores.sub_(maximum).exp_().masked_fill_(~valid, 0.0)
    denominator = exponent.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
    probabilities = exponent.div_(denominator)
    output = torch.einsum("bshk,bskd->bshd", probabilities, selected)
    return output.to(query.dtype)


def prepare_attention_weights(
    weights: ResidentAttentionWeights,
    *,
    layer_id: int,
    rank: int,
    world_size: int,
    checkpoint_id: str,
) -> PreparedAttentionWeights:
    if (
        not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or layer_id not in SUPPORTED_RATIO128_LAYER_IDS
        or not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size != 4
        or not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank not in range(world_size)
    ):
        raise ValueError(
            "prepared attention identity must be a frozen ratio-128 layer on TP4"
        )
    if (
        not isinstance(checkpoint_id, str)
        or len(checkpoint_id) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_id)
    ):
        raise ValueError("prepared attention requires a lowercase SHA-256 checkpoint_id")
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
            "resident ratio-128 attention identity differs from requested identity: "
            f"resident={resident_identity}, requested={requested_identity}"
        )
    if weights.indexer is not None:
        raise ValueError("ratio-128 attention must not contain indexer weights")

    def linear(value: Any) -> torch.Tensor:
        return dequant_fp8_block(value.weight, value.scale).to(torch.bfloat16)

    return PreparedAttentionWeights(
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
        layer_id=layer_id,
        rank=rank,
        world_size=world_size,
        checkpoint_id=checkpoint_id,
    )


class Ratio128TorchAttention:
    """Real-weight eager ratio-128 control backed by :class:`StaticLayerKV`."""

    def __init__(
        self,
        config: Ratio128AttentionConfig,
        weights: PreparedAttentionWeights,
        state: StaticLayerKV,
        nope_quant_mode: Literal[
            "qat_intended_e4m3", "reference_executable_bf16"
        ] = "qat_intended_e4m3",
        *,
        sparse_attention_backend: Ratio128SparseAttentionBackend | None = None,
        projection_backend: AttentionProjectionBackend | None = None,
    ) -> None:
        config.validate()
        layer_identity = (config.layer_id, weights.layer_id, state.layer_id)
        if (
            any(
                not isinstance(layer_id, int) or isinstance(layer_id, bool)
                for layer_id in layer_identity
            )
            or len(set(layer_identity)) != 1
            or config.layer_id not in SUPPORTED_RATIO128_LAYER_IDS
        ):
            raise ValueError(
                "ratio-128 attention config/weights/state identity differs: "
                f"layers={layer_identity}"
            )
        if state.max_seq_len != config.max_seq_len:
            raise ValueError("attention config and static KV capacity differ")
        identity = (weights.layer_id, weights.rank, weights.world_size)
        if (
            not isinstance(weights.world_size, int)
            or isinstance(weights.world_size, bool)
            or weights.world_size != 4
            or not isinstance(weights.rank, int)
            or isinstance(weights.rank, bool)
            or weights.rank not in range(weights.world_size)
        ):
            raise ValueError(f"prepared attention identity is invalid: {identity}")
        if (
            not isinstance(weights.checkpoint_id, str)
            or len(weights.checkpoint_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in weights.checkpoint_id
            )
        ):
            raise ValueError("prepared attention checkpoint identity is invalid")
        self.config = config
        self.weights = weights
        self.state = state
        if nope_quant_mode not in (
            "qat_intended_e4m3",
            "reference_executable_bf16",
        ):
            raise ValueError(f"unsupported NoPE quant mode {nope_quant_mode}")
        self.nope_quant_mode = nope_quant_mode
        if sparse_attention_backend is not None and not callable(
            sparse_attention_backend
        ):
            raise TypeError("ratio-128 sparse attention backend must be callable")
        if sparse_attention_backend is not None and state.kv_dtype != "bf16":
            raise ValueError(
                "injected ratio-128 sparse backends require BF16 KV storage"
            )
        self._sparse_attention_backend = sparse_attention_backend
        _validate_attention_projection_backend(
            projection_backend,
            expected_compress_ratio=COMPRESS_RATIO,
        )
        self._projection_backend = projection_backend
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

    def _nope_control(self, value: torch.Tensor) -> torch.Tensor:
        if self.nope_quant_mode == "reference_executable_bf16":
            return value
        return fp8_quant_dequant(value, group_size=64)

    def _compress_finalizer(
        self, pooled: torch.Tensor, group_starts: torch.Tensor
    ) -> torch.Tensor:
        value = rms_norm(
            pooled.to(torch.bfloat16),
            self.weights.compressor_norm,
            eps=self.config.norm_eps,
        )
        rope = value[..., -self.config.rope_dim :]
        frequencies = self.freqs_cis.index_select(0, group_starts.long())
        value[..., -self.config.rope_dim :] = apply_rotary_emb(rope, frequencies)
        value[..., : -self.config.rope_dim] = self._nope_control(
            value[..., : -self.config.rope_dim]
        )
        return value.contiguous()

    def prepare_decode_plan(self, start_pos: int) -> Ratio128DecodePlan:
        """Validate state once and materialize a fixed non-boundary decode plan."""

        if (
            not isinstance(start_pos, int)
            or isinstance(start_pos, bool)
            or start_pos < WINDOW_SIZE
            or start_pos >= self.config.max_seq_len
        ):
            raise ValueError(
                "decode plan start_pos must be an integer in [128, max_seq_len)"
            )
        if (start_pos + 1) % COMPRESS_RATIO == 0:
            raise ValueError("trace-free decode plan does not support compression boundaries")
        if self.state.next_position != start_pos:
            raise ValueError(
                f"start_pos {start_pos} != static KV next position "
                f"{self.state.next_position}"
            )

        expected_compressed = start_pos // COMPRESS_RATIO
        if not bool(
            torch.all(self.state._compressed_count == expected_compressed).item()
        ):
            raise RuntimeError("static KV compressed-row metadata is inconsistent")
        pending = start_pos % COMPRESS_RATIO
        if pending:
            expected_pending = torch.arange(
                start_pos - pending,
                start_pos,
                dtype=torch.int64,
                device=self.state.device,
            )
            if not bool(
                torch.all(
                    self.state._state_positions[:, :pending]
                    == expected_pending.unsqueeze(0)
                ).item()
            ):
                raise RuntimeError("static KV compressor metadata is inconsistent")

        absolute_raw = torch.arange(
            start_pos - WINDOW_SIZE,
            start_pos,
            dtype=torch.int64,
            device=self.state.device,
        )
        raw_slots = absolute_raw.remainder(WINDOW_SIZE)
        expected_raw = absolute_raw.unsqueeze(0).expand(
            self.state.num_local_sequences, -1
        )
        if not bool(
            torch.all(
                self.state._raw_positions.index_select(1, raw_slots) == expected_raw
            ).item()
        ):
            raise RuntimeError("static KV raw-ring metadata is inconsistent")

        window = window_topk_indices(
            batch_size=self.state.num_local_sequences,
            seqlen=1,
            start_pos=start_pos,
            device=self.state.device,
        )
        compressed = compressed_topk_indices(
            batch_size=self.state.num_local_sequences,
            seqlen=1,
            start_pos=start_pos,
            offset=WINDOW_SIZE,
            device=self.state.device,
        )
        topk = torch.cat((window, compressed), dim=-1).contiguous()
        gather = topk.to(torch.int64)
        batch_indices = (
            torch.arange(
                self.state.num_local_sequences,
                dtype=torch.int64,
                device=self.state.device,
            )
            .view(self.state.num_local_sequences, 1, 1)
            .expand_as(gather)
        )
        return Ratio128DecodePlan(
            start_pos=start_pos,
            slot=start_pos % WINDOW_SIZE,
            batch_size=self.state.num_local_sequences,
            hidden_size=self.config.hidden_size,
            owner_id=id(self),
            state_id=id(self.state),
            frequencies=self.freqs_cis[start_pos : start_pos + 1].contiguous(),
            topk_indices=topk,
            gather_indices=gather,
            batch_indices=batch_indices,
            compressor_ape=self.weights.compressor_ape[
                start_pos % COMPRESS_RATIO
            ],
        )

    def prepare_stateful_decode_plan(
        self,
        *,
        position: torch.Tensor,
        start_position: int,
        stop_position: int,
    ) -> Ratio128StatefulDecodePlan:
        """Allocate one fixed workspace for ``[start_position, stop_position)``."""

        if self._sparse_attention_backend is not None:
            raise RuntimeError(
                "stateful ratio-128 decode requires the masked direct control backend"
            )
        if (
            not isinstance(start_position, int)
            or isinstance(start_position, bool)
            or not isinstance(stop_position, int)
            or isinstance(stop_position, bool)
            or start_position < WINDOW_SIZE
            or stop_position <= start_position
            or stop_position > self.config.max_seq_len
        ):
            raise ValueError(
                "stateful decode range must be a non-empty interval within capacity"
            )
        if (
            not isinstance(position, torch.Tensor)
            or tuple(position.shape) != (1,)
            or position.dtype != torch.int64
            or position.device != self.state.device
            or not position.is_contiguous()
        ):
            raise ValueError("stateful position must be contiguous INT64 [1]")
        if int(position.item()) != start_position:
            raise ValueError("device position does not match stateful range start")
        if self.state.next_position != start_position:
            raise ValueError("ratio-128 state does not match stateful range start")

        expected_compressed = start_position // COMPRESS_RATIO
        if not bool(
            torch.all(self.state._compressed_count == expected_compressed).item()
        ):
            raise RuntimeError("static KV compressed-row metadata is inconsistent")
        pending = start_position % COMPRESS_RATIO
        if pending:
            expected_pending = torch.arange(
                start_position - pending,
                start_position,
                dtype=torch.int64,
                device=self.state.device,
            )
            if not bool(
                torch.all(
                    self.state._state_positions[:, :pending]
                    == expected_pending.unsqueeze(0)
                ).item()
            ):
                raise RuntimeError("static KV compressor metadata is inconsistent")

        absolute_raw = torch.arange(
            start_position - WINDOW_SIZE,
            start_position,
            dtype=torch.int64,
            device=self.state.device,
        )
        raw_slots = absolute_raw.remainder(WINDOW_SIZE)
        expected_raw = absolute_raw.unsqueeze(0).expand(
            self.state.num_local_sequences, -1
        )
        if not bool(
            torch.all(
                self.state._raw_positions.index_select(1, raw_slots) == expected_raw
            ).item()
        ):
            raise RuntimeError("static KV raw-ring metadata is inconsistent")

        bucket_width = ratio128_sparse_bucket_width(
            start_position, stop_position - 1
        )
        batch = self.state.num_local_sequences
        shape = (batch, 1, bucket_width)
        device = self.state.device
        topk_indices = torch.full(
            shape, -1, dtype=torch.int32, device=device
        )
        gather_indices = torch.zeros(
            shape, dtype=torch.int64, device=device
        )
        valid_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        batch_indices = (
            torch.arange(batch, dtype=torch.int64, device=device)
            .view(batch, 1, 1)
            .expand(shape)
            .contiguous()
        )
        workspaces = (
            topk_indices,
            gather_indices,
            valid_mask,
            batch_indices,
        )
        tensor_pointers = tuple(
            int(value.untyped_storage().data_ptr())
            for value in (position, *workspaces)
        )
        if len(set(tensor_pointers)) != len(tensor_pointers):
            raise RuntimeError("stateful ratio-128 workspaces must not alias")
        return Ratio128StatefulDecodePlan(
            start_position=start_position,
            stop_position=stop_position,
            bucket_width=bucket_width,
            batch_size=batch,
            hidden_size=self.config.hidden_size,
            owner_id=id(self),
            state_id=id(self.state),
            position=position,
            topk_indices=topk_indices,
            gather_indices=gather_indices,
            valid_mask=valid_mask,
            batch_indices=batch_indices,
            tensor_pointers=tensor_pointers,
        )

    def _validate_stateful_decode_plan(
        self,
        hidden: torch.Tensor,
        plan: Ratio128StatefulDecodePlan,
        *,
        ratio128_boundary: bool,
    ) -> None:
        if self._sparse_attention_backend is not None:
            raise RuntimeError(
                "stateful ratio-128 decode requires the masked direct control backend"
            )
        if not isinstance(plan, Ratio128StatefulDecodePlan):
            raise TypeError("plan must be a Ratio128StatefulDecodePlan")
        if plan.owner_id != id(self) or plan.state_id != id(self.state):
            raise ValueError("stateful decode plan belongs to another attention state")
        if not isinstance(ratio128_boundary, bool):
            raise TypeError("ratio128_boundary must be bool")
        if tuple(hidden.shape) != (plan.batch_size, 1, plan.hidden_size):
            raise ValueError("stateful hidden shape does not match its plan")
        if hidden.dtype != torch.bfloat16 or hidden.device != self.state.device:
            raise ValueError("stateful hidden must use state-local BF16 storage")
        shape = (plan.batch_size, 1, plan.bucket_width)
        expected = (
            ("position", plan.position, (1,), torch.int64),
            ("topk_indices", plan.topk_indices, shape, torch.int32),
            ("gather_indices", plan.gather_indices, shape, torch.int64),
            ("valid_mask", plan.valid_mask, shape, torch.bool),
            ("batch_indices", plan.batch_indices, shape, torch.int64),
        )
        pointers = []
        for name, value, expected_shape, expected_dtype in expected:
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"stateful {name} shape {tuple(value.shape)} != {expected_shape}"
                )
            if value.dtype != expected_dtype or value.device != self.state.device:
                raise ValueError(f"stateful {name} dtype/device differs")
            if not value.is_contiguous():
                raise ValueError(f"stateful {name} must be contiguous")
            pointer = int(value.untyped_storage().data_ptr())
            pointers.append(pointer)
        if len(set(pointers)) != len(pointers):
            raise ValueError("stateful plan tensors must not alias")
        if tuple(pointers) != plan.tensor_pointers:
            raise ValueError("stateful plan tensor storage differs from setup")

    def forward_stateful_decode_tensor(
        self,
        hidden: torch.Tensor,
        *,
        plan: Ratio128StatefulDecodePlan,
        ratio128_boundary: bool,
        stage_marker: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Run one cursor-driven graph-family token without host value reads."""

        self._validate_stateful_decode_plan(
            hidden, plan, ratio128_boundary=ratio128_boundary
        )
        cfg = self.config
        position = plan.position
        slot = position.remainder(COMPRESS_RATIO)
        frequencies = self.freqs_cis.index_select(0, position)
        compressor_ape = self.weights.compressor_ape.index_select(0, slot)[0]

        projection_backend = self._projection_backend
        hidden_projections = None
        if projection_backend is None:
            projected_wq_a = F.linear(hidden, self.weights.wq_a)
        else:
            hidden_projections = _project_hidden_with_backend(
                projection_backend,
                hidden,
                wq_a=self.weights.wq_a,
                wkv=self.weights.wkv,
            )
            projected_wq_a = hidden_projections.wq_a
        query_lora = rms_norm(
            projected_wq_a,
            self.weights.q_norm,
            eps=cfg.norm_eps,
        )
        if projection_backend is None:
            projected_wq_b = F.linear(query_lora, self.weights.wq_b)
        else:
            projected_wq_b = _project_query_with_backend(
                projection_backend,
                query_lora,
                wq_b=self.weights.wq_b,
                index_wq_b=None,
            ).wq_b
        query = projected_wq_b.reshape(
            plan.batch_size, 1, cfg.num_heads, cfg.head_dim
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
            F.linear(hidden, self.weights.wkv)
            if hidden_projections is None
            else hidden_projections.wkv
        )
        raw_latent = rms_norm(
            projected_wkv,
            self.weights.kv_norm,
            eps=cfg.norm_eps,
        )
        raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
            raw_latent[..., -cfg.rope_dim :], frequencies
        )
        raw_latent[..., : -cfg.rope_dim] = self._nope_control(
            raw_latent[..., : -cfg.rope_dim]
        )
        if stage_marker is not None:
            stage_marker("raw_kv_done")
        projected_kv = F.linear(hidden.float(), self.weights.compressor_wkv)
        projected_score = F.linear(hidden.float(), self.weights.compressor_wgate)
        adjusted_score = projected_score[:, 0] + compressor_ape
        if stage_marker is not None:
            stage_marker("compressor_projection_done")
        self.state._write_decode_stateful_prevalidated(
            raw_latent,
            projected_kv,
            adjusted_score,
            position=position,
            boundary=ratio128_boundary,
            finalize_compressed=self._compress_finalizer,
        )
        if stage_marker is not None:
            stage_marker("state_write_done")

        build_padded_ratio128_sparse_indices(
            position,
            batch_size=plan.batch_size,
            bucket_width=plan.bucket_width,
            out=plan.topk_indices,
        )
        plan.valid_mask.copy_(plan.topk_indices.ge(0))
        plan.gather_indices.copy_(plan.topk_indices)
        plan.gather_indices.clamp_min_(0)
        if stage_marker is not None:
            stage_marker("index_done")
        output = _torch_sparse_decode_padded_prevalidated(
            query,
            self.state.latent,
            self.weights.attn_sink,
            plan,
            cfg.head_dim**-0.5,
            latent_rope=self.state.latent_rope,
        )
        if stage_marker is not None:
            stage_marker("sparse_done")
        output[..., -cfg.rope_dim :] = apply_rotary_emb(
            output[..., -cfg.rope_dim :], frequencies, inverse=True
        )
        grouped = output.reshape(
            plan.batch_size,
            1,
            cfg.o_groups,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        if stage_marker is not None:
            stage_marker("output_transform_done")
        wo_a = self.weights.wo_a.reshape(
            cfg.o_groups,
            cfg.o_lora_rank,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        output_lora = projected.flatten(2)
        if stage_marker is not None:
            stage_marker("wo_a_done")
        if projection_backend is None:
            final_output = F.linear(output_lora, self.weights.wo_b)
        else:
            final_output = _project_output_with_backend(
                projection_backend,
                output_lora,
                wo_b=self.weights.wo_b,
            )
        if stage_marker is not None:
            stage_marker("output_done")
        return final_output

    def forward_decode_tensor(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int,
        plan: Ratio128DecodePlan,
    ) -> torch.Tensor:
        """Run one fixed non-boundary decode token without trace or host sync."""

        if not isinstance(plan, Ratio128DecodePlan):
            raise TypeError("plan must be a Ratio128DecodePlan")
        if plan.owner_id != id(self) or plan.state_id != id(self.state):
            raise ValueError("decode plan belongs to a different attention state")
        if start_pos != plan.start_pos:
            raise ValueError("decode start_pos does not match the fixed plan")
        if tuple(hidden.shape) != (
            plan.batch_size,
            1,
            plan.hidden_size,
        ):
            raise ValueError("decode hidden shape does not match the fixed plan")
        if hidden.dtype != torch.bfloat16:
            raise TypeError("trace-free decode requires BF16 hidden input")
        if hidden.device != plan.frequencies.device:
            raise ValueError("decode hidden and fixed plan must share a device")

        cfg = self.config
        frequencies = plan.frequencies
        projection_backend = self._projection_backend
        hidden_projections = None
        if projection_backend is None:
            projected_wq_a = F.linear(hidden, self.weights.wq_a)
        else:
            hidden_projections = _project_hidden_with_backend(
                projection_backend,
                hidden,
                wq_a=self.weights.wq_a,
                wkv=self.weights.wkv,
            )
            projected_wq_a = hidden_projections.wq_a
        query_lora = rms_norm(
            projected_wq_a,
            self.weights.q_norm,
            eps=cfg.norm_eps,
        )
        if projection_backend is None:
            projected_wq_b = F.linear(query_lora, self.weights.wq_b)
        else:
            projected_wq_b = _project_query_with_backend(
                projection_backend,
                query_lora,
                wq_b=self.weights.wq_b,
                index_wq_b=None,
            ).wq_b
        query = projected_wq_b.reshape(
            plan.batch_size, 1, cfg.num_heads, cfg.head_dim
        )
        query *= torch.rsqrt(
            query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps
        )
        query[..., -cfg.rope_dim :] = apply_rotary_emb(
            query[..., -cfg.rope_dim :], frequencies
        )

        projected_wkv = (
            F.linear(hidden, self.weights.wkv)
            if hidden_projections is None
            else hidden_projections.wkv
        )
        raw_latent = rms_norm(
            projected_wkv,
            self.weights.kv_norm,
            eps=cfg.norm_eps,
        )
        raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
            raw_latent[..., -cfg.rope_dim :], frequencies
        )
        raw_latent[..., : -cfg.rope_dim] = self._nope_control(
            raw_latent[..., : -cfg.rope_dim]
        )
        projected_kv = F.linear(hidden.float(), self.weights.compressor_wkv)
        projected_score = F.linear(hidden.float(), self.weights.compressor_wgate)
        adjusted_score = projected_score[:, 0] + plan.compressor_ape
        self.state._write_decode_nonboundary_fixed(
            raw_latent,
            projected_kv,
            adjusted_score,
            position=start_pos,
            slot=plan.slot,
        )

        sparse_attention_backend = self._sparse_attention_backend
        if sparse_attention_backend is None:
            output = _torch_sparse_decode_prevalidated(
                query,
                self.state.latent,
                self.weights.attn_sink,
                plan,
                cfg.head_dim**-0.5,
                latent_rope=self.state.latent_rope,
            )
        else:
            output = sparse_attention_backend(
                query,
                self.state.latent,
                self.weights.attn_sink,
                plan.gather_indices,
                plan.batch_indices,
                cfg.head_dim**-0.5,
            )
            if not isinstance(output, torch.Tensor):
                raise TypeError("ratio-128 sparse backend must return a tensor")
            if tuple(output.shape) != tuple(query.shape):
                raise ValueError("ratio-128 sparse backend output shape differs")
            if output.dtype != query.dtype or output.device != query.device:
                raise ValueError("ratio-128 sparse backend output dtype/device differs")
            if not output.is_contiguous():
                raise ValueError("ratio-128 sparse backend output must be contiguous")
        output[..., -cfg.rope_dim :] = apply_rotary_emb(
            output[..., -cfg.rope_dim :], frequencies, inverse=True
        )
        grouped = output.reshape(
            plan.batch_size,
            1,
            cfg.o_groups,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        wo_a = self.weights.wo_a.reshape(
            cfg.o_groups,
            cfg.o_lora_rank,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        output_lora = projected.flatten(2)
        if projection_backend is None:
            return F.linear(output_lora, self.weights.wo_b)
        return _project_output_with_backend(
            projection_backend,
            output_lora,
            wo_b=self.weights.wo_b,
        )

    def __call__(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int,
        evidence: MutableMapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, AttentionTrace]:
        cfg = self.config
        if hidden.ndim != 3 or hidden.shape[0] != self.state.num_local_sequences:
            raise ValueError("hidden must have shape [local_batch, sequence, hidden_size]")
        if hidden.shape[-1] != cfg.hidden_size or hidden.dtype != torch.bfloat16:
            raise ValueError(
                "hidden size/dtype does not match "
                f"layer-{cfg.layer_id} BF16 contract"
            )
        if start_pos != self.state.next_position:
            raise ValueError(
                f"start_pos {start_pos} != static KV next position {self.state.next_position}"
            )
        if start_pos > 0 and hidden.shape[1] != 1:
            raise ValueError("decode attention requires one token")
        if start_pos + hidden.shape[1] > cfg.max_seq_len:
            raise ValueError("attention input exceeds static KV capacity")

        batch, seqlen, _ = hidden.shape
        frequencies = self.freqs_cis[start_pos : start_pos + seqlen]

        def record(name: str, value: torch.Tensor) -> None:
            if evidence is not None:
                evidence[name] = value.detach().clone()

        projection_backend = self._projection_backend
        hidden_projections = None
        if projection_backend is None:
            projected_wq_a = F.linear(hidden, self.weights.wq_a)
        else:
            hidden_projections = _project_hidden_with_backend(
                projection_backend,
                hidden,
                wq_a=self.weights.wq_a,
                wkv=self.weights.wkv,
            )
            projected_wq_a = hidden_projections.wq_a
        query_lora = rms_norm(
            projected_wq_a,
            self.weights.q_norm,
            eps=cfg.norm_eps,
        )
        record("query_lora", query_lora)
        if projection_backend is None:
            projected_wq_b = F.linear(query_lora, self.weights.wq_b)
        else:
            projected_wq_b = _project_query_with_backend(
                projection_backend,
                query_lora,
                wq_b=self.weights.wq_b,
                index_wq_b=None,
            ).wq_b
        query = projected_wq_b.reshape(
            batch, seqlen, cfg.num_heads, cfg.head_dim
        )
        query *= torch.rsqrt(
            query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps
        )
        query[..., -cfg.rope_dim :] = apply_rotary_emb(
            query[..., -cfg.rope_dim :], frequencies
        )
        record("query", query)

        projected_wkv = (
            F.linear(hidden, self.weights.wkv)
            if hidden_projections is None
            else hidden_projections.wkv
        )
        raw_latent = rms_norm(
            projected_wkv,
            self.weights.kv_norm,
            eps=cfg.norm_eps,
        )
        raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
            raw_latent[..., -cfg.rope_dim :], frequencies
        )
        raw_latent[..., : -cfg.rope_dim] = self._nope_control(
            raw_latent[..., : -cfg.rope_dim]
        )
        record("raw_latent", raw_latent)
        projected_kv = F.linear(hidden.float(), self.weights.compressor_wkv)
        projected_score = F.linear(hidden.float(), self.weights.compressor_wgate)
        record("projected_kv", projected_kv)
        record("projected_score", projected_score)

        if start_pos == 0:
            compression = self.state.prefill_write(
                raw_latent,
                projected_kv=projected_kv,
                projected_score=projected_score,
                ape=self.weights.compressor_ape,
                finalize_compressed=self._compress_finalizer,
            )
            compressed_count = seqlen // COMPRESS_RATIO
            # FP8 KV: attention reads what the cache would return (write+read
            # round trip of the fresh rows; dequantized compressed rows).
            attention_kv = torch.cat(
                (
                    self.state.quantize_dequantize_rows(raw_latent),
                    self.state.dequantized_compressed(compressed_count),
                ),
                dim=1,
            )
            compressed = compressed_topk_indices(
                batch_size=batch,
                seqlen=seqlen,
                start_pos=0,
                offset=seqlen,
                device=hidden.device,
            )
        else:
            compression = self.state.decode_write(
                raw_latent,
                projected_kv=projected_kv,
                projected_score=projected_score,
                ape=self.weights.compressor_ape,
                finalize_compressed=self._compress_finalizer,
            )
            attention_kv = self.state.dequantized_latent()
            compressed = compressed_topk_indices(
                batch_size=batch,
                seqlen=1,
                start_pos=start_pos,
                offset=WINDOW_SIZE,
                device=hidden.device,
            )
        if compression is not None:
            record("compression_pooled", compression.pooled)
            if evidence is not None:
                compressed_rows = torch.tensor(
                    compression.row_indices,
                    dtype=torch.int64,
                    device=hidden.device,
                )
                record(
                    "compression_finalized",
                    self.state.dequantized_compressed_rows(compressed_rows),
                )
        record("attention_kv", attention_kv)
        window = window_topk_indices(
            batch_size=batch,
            seqlen=seqlen,
            start_pos=start_pos,
            device=hidden.device,
        )
        topk = torch.cat((window, compressed), dim=-1).contiguous()
        record("topk", topk)
        # C2F prefill vertical: optional query-row blocking of the prefill
        # sparse core.  Rows are independent (per-row mask/softmax), so the
        # blocked form is bitwise identical to the single call; it only
        # bounds the FP32 gather workspace at long prefill chunks.  Default
        # off (env unset) -- decode and all oracle paths are unchanged.
        # 21st vertical: the prefill sparse core may be the reference tilelang
        # kernel instead (env-selected, default torch; decode is never
        # switched).  Row blocking exists only to bound the torch core's FP32
        # gather workspace, which the kernel never materializes, so the
        # tilelang arm runs the single call (rows are independent -- both
        # forms are the same math).
        row_block = _prefill_sparse_row_block()
        sparse_core = torch_sparse_attention
        if start_pos == 0:
            backend = _prefill_sparse_backend()
            if backend != "torch":
                from .ops.tilelang_sparse import prefill_sparse_core

                sparse_core = prefill_sparse_core(backend)
                row_block = None
        if row_block is not None and start_pos == 0 and seqlen > row_block:
            output = torch.cat(
                [
                    sparse_core(
                        query[:, begin : begin + row_block],
                        attention_kv,
                        self.weights.attn_sink,
                        topk[:, begin : begin + row_block],
                        cfg.head_dim**-0.5,
                    )
                    for begin in range(0, seqlen, row_block)
                ],
                dim=1,
            )
        else:
            output = sparse_core(
                query,
                attention_kv,
                self.weights.attn_sink,
                topk,
                cfg.head_dim**-0.5,
            )
        record("sparse_output", output)
        output[..., -cfg.rope_dim :] = apply_rotary_emb(
            output[..., -cfg.rope_dim :], frequencies, inverse=True
        )
        record("inverse_rope_output", output)
        grouped = output.reshape(
            batch,
            seqlen,
            cfg.o_groups,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        wo_a = self.weights.wo_a.reshape(
            cfg.o_groups,
            cfg.o_lora_rank,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        record("output_lora", projected)
        output_lora = projected.flatten(2)
        if projection_backend is None:
            branch = F.linear(output_lora, self.weights.wo_b)
        else:
            branch = _project_output_with_backend(
                projection_backend,
                output_lora,
                wo_b=self.weights.wo_b,
            )
        record("branch", branch)
        valid = topk[topk >= 0]
        trace = AttentionTrace(
            start_pos=start_pos,
            input_shape=tuple(hidden.shape),
            output_shape=tuple(branch.shape),
            query_shape=tuple(query.shape),
            attention_kv_shape=tuple(attention_kv.shape),
            topk_shape=tuple(topk.shape),
            valid_topk_min=int(valid.min().item()),
            valid_topk_max=int(valid.max().item()),
            compressed_rows_written=()
            if compression is None
            else compression.row_indices,
            weight_projection_mode=(
                "bf16_dequantized_weight_control"
                if projection_backend is None
                else "injected_attention_projection_backend"
            ),
            nope_quant_mode=self.nope_quant_mode,
            sparse_accumulation_mode="fp32_probability_value_control",
        )
        return branch, trace


__all__ = [
    "AttentionHiddenProjections",
    "AttentionProjectionBackend",
    "AttentionQueryProjections",
    "AttentionTrace",
    "PreparedAttentionWeights",
    "Ratio128AttentionConfig",
    "Ratio128DecodePlan",
    "Ratio128StatefulDecodePlan",
    "Ratio128SparseAttentionBackend",
    "Ratio128TorchAttention",
    "SUPPORTED_RATIO128_LAYER_IDS",
    "apply_rotary_emb",
    "compressed_topk_indices",
    "fp8_quant_dequant",
    "precompute_freqs_cis",
    "prepare_attention_weights",
    "rms_norm",
    "torch_sparse_attention",
    "window_topk_indices",
]
