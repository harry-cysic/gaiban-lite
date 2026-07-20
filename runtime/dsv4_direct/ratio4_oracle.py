"""Independent ratio-4 attention, compressor, indexer, and hash-gate oracle.

This module is diagnostic mathematics, not a performance implementation.  It
does not import the direct candidate path.  FP8 projection weights are decoded
through the already independent E0e primitive.  A reference-faithful BF16
profile matches the public normalization order and direct GEMM boundary, while
a separate raw-FP32 profile retains independent division-based normalization
and higher-precision accumulation attribution.  Hadamard, E2M1 QDQ,
overlap pooling, index scoring, and hash weighting use formulations independent
from the candidate implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .attention_oracle import (
    RopeTable,
    e4m3_ue8m0_qdq,
    oracle_apply_rope,
    oracle_dequant_fp8_block,
    oracle_rms_norm,
    oracle_sparse_attention,
    oracle_window_topk_indices,
    yarn_rope_table,
)


RATIO4 = 4
WINDOW_SIZE = 128
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
REFERENCE_BF16_NORMALIZATION_PROFILE = "reference_bf16_rsqrt_operator_order"
RAW_FP32_NORMALIZATION_PROFILE = "raw_fp32_div_sqrt_attribution"
# V4-Flash port note: this repo vendors the reference under reference/ (not
# references/).  The file content -- and therefore the SHA-256 -- is identical
# to the Pro reference; all geometry differences live in config.json.
REFERENCE_NORMALIZATION_SOURCE_PATH = "reference/inference/model.py"
REFERENCE_NORMALIZATION_SOURCE_SHA256 = (
    "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71"
)
BF16_CONTROL_WEIGHT_FIELDS = frozenset(
    {
        "wq_a",
        "wq_b",
        "wkv",
        "wo_a",
        "wo_b",
        "index_wq_b",
        "index_weights_proj",
    }
)


@dataclass(frozen=True)
class OracleHashRoute:
    selected_logits: torch.Tensor
    selected_scores: torch.Tensor
    routing_weights: torch.Tensor
    routing_ids: torch.Tensor


@dataclass
class OracleRatio4Weights:
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


@dataclass(frozen=True)
class OracleRatio4State:
    raw: torch.Tensor
    compressed: torch.Tensor
    indexer_kv: torch.Tensor
    main_kv: torch.Tensor
    main_score: torch.Tensor
    index_kv: torch.Tensor
    index_score: torch.Tensor
    next_position: int
    compressed_count: int
    max_seq_len: int

    def clone(self) -> "OracleRatio4State":
        return OracleRatio4State(
            raw=self.raw.clone(),
            compressed=self.compressed.clone(),
            indexer_kv=self.indexer_kv.clone(),
            main_kv=self.main_kv.clone(),
            main_score=self.main_score.clone(),
            index_kv=self.index_kv.clone(),
            index_score=self.index_score.clone(),
            next_position=self.next_position,
            compressed_count=self.compressed_count,
            max_seq_len=self.max_seq_len,
        )


@dataclass(frozen=True)
class OracleRatio4Trace:
    query_lora: torch.Tensor
    query: torch.Tensor
    raw_latent: torch.Tensor
    main_projected_kv: torch.Tensor
    main_projected_score: torch.Tensor
    main_overlap_values: torch.Tensor | None
    main_overlap_logits: torch.Tensor | None
    main_compression_pooled: torch.Tensor | None
    main_compression_finalized: torch.Tensor | None
    index_projected_kv: torch.Tensor
    index_projected_score: torch.Tensor
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


@dataclass(frozen=True)
class OracleRatio4Step:
    trace: OracleRatio4Trace
    state: OracleRatio4State


def oracle_hash_route(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    route_scale: float = 1.5,
) -> OracleHashRoute:
    """Evaluate only the six checkpoint-selected experts in FP64.

    Hash routing does not need scores for the other 250 experts because expert
    IDs come directly from ``tid2eid``.  Selected-only FP64 dot products provide
    a practical independent oracle without a prohibitively slow full FP64 GEMM.
    The default ``route_scale`` is Flash's routed_scaling_factor 1.5 (Pro used
    2.5); callers should still pass the checkpoint value explicitly.
    """

    if hidden.ndim != 2 or gate_weight.ndim != 2 or tid2eid.ndim != 2:
        raise ValueError("hash hidden/weight/table must have ranks 2/2/2")
    rows, width = hidden.shape
    experts, weight_width = gate_weight.shape
    if rows <= 0 or width != weight_width:
        raise ValueError("hash hidden and gate weight shapes are incompatible")
    if input_ids.ndim != 1 or input_ids.shape[0] != rows:
        raise ValueError("hash input_ids must contain one ID per hidden row")
    if input_ids.dtype != torch.int64 or tid2eid.dtype != torch.int64:
        raise TypeError("hash input IDs and table must be int64")
    if not hidden.is_floating_point() or not gate_weight.is_floating_point():
        raise TypeError("hash hidden and gate weight must be floating point")
    if not (
        hidden.device == gate_weight.device == tid2eid.device == input_ids.device
    ):
        raise ValueError("hash oracle tensors must share one device")
    if tid2eid.shape[1] <= 0 or tid2eid.shape[1] >= experts:
        raise ValueError("hash table top-k width is invalid")
    if not math.isfinite(route_scale) or route_scale <= 0:
        raise ValueError("hash route_scale must be finite and positive")
    if bool((input_ids < 0).any()) or bool((input_ids >= tid2eid.shape[0]).any()):
        raise ValueError("hash input ID is outside the table")
    if not bool(torch.isfinite(hidden).all() and torch.isfinite(gate_weight).all()):
        raise ValueError("hash oracle inputs must be finite")

    ids = tid2eid.index_select(0, input_ids)
    if bool((ids < 0).any()) or bool((ids >= experts).any()):
        raise ValueError("hash table selected an expert outside the gate")
    sorted_ids = ids.sort(dim=1).values
    if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
        raise ValueError("hash table rows must select unique experts")

    selected_weight = gate_weight.index_select(0, ids.reshape(-1)).reshape(
        rows, ids.shape[1], width
    )
    logits = torch.sum(
        hidden.to(torch.float64).unsqueeze(1)
        * selected_weight.to(torch.float64),
        dim=-1,
    )
    # Stable softplus written without torch.nn.functional.softplus.
    softplus = torch.clamp_min(logits, 0.0) + torch.log1p(
        torch.exp(-torch.abs(logits))
    )
    scores = torch.sqrt(softplus)
    denominator = scores.sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(scores).all()) or bool((denominator <= 0).any()):
        raise ValueError("hash selected scores are not finite and positive")
    routing = scores / denominator * float(route_scale)
    return OracleHashRoute(
        selected_logits=logits,
        selected_scores=scores,
        routing_weights=routing,
        routing_ids=ids.clone(),
    )


def oracle_hadamard(value: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform via an explicit Sylvester matrix."""

    if not value.is_floating_point() or value.ndim < 1:
        raise TypeError("Hadamard input must be a floating tensor")
    width = value.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float32, device=value.device)
    while matrix.shape[0] < width:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    transformed = torch.matmul(value.to(torch.float32), matrix.transpose(0, 1))
    transformed = transformed * (float(width) ** -0.5)
    return transformed.to(value.dtype)


def oracle_e2m1_qdq(
    value: torch.Tensor,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    """Power-of-two group-scaled E2M1 QDQ with explicit code selection."""

    if value.dtype != torch.bfloat16:
        raise TypeError("E2M1 oracle requires BF16 input")
    if value.ndim < 1 or group_size <= 0 or value.shape[-1] % group_size:
        raise ValueError("E2M1 group size must positively divide the last dimension")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("E2M1 input must be finite")

    grouped = value.to(torch.float32).reshape(*value.shape[:-1], -1, group_size)
    tiny = 6.0 * 2.0**-126
    absolute_max = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(tiny)
    scale = torch.pow(2.0, torch.ceil(torch.log2(absolute_max / 6.0)))
    normalized = torch.clamp(grouped / scale, -6.0, 6.0)
    magnitude = normalized.abs()
    levels = torch.tensor(E2M1_LEVELS, dtype=torch.float32, device=value.device)
    distances = torch.abs(magnitude.unsqueeze(-1) - levels)
    minimum = distances.amin(dim=-1, keepdim=True)
    ties = distances == minimum
    code_ids = torch.arange(levels.numel(), device=value.device)
    even_ties = ties & ((code_ids % 2) == 0)
    nearest = distances.argmin(dim=-1)
    even_nearest = torch.where(
        even_ties,
        code_ids,
        torch.full_like(code_ids, levels.numel()),
    ).amin(dim=-1)
    chosen = torch.where(even_ties.any(dim=-1), even_nearest, nearest)
    quantized = levels[chosen]
    quantized = torch.copysign(quantized, normalized)
    return (quantized * scale).reshape_as(value).to(value.dtype)


def oracle_overlap_pool(
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    output_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool previous-left and current-right halves with explicit stable exp."""

    if kv_state.shape != score_state.shape or kv_state.ndim != 3:
        raise ValueError("overlap KV and score states must have equal rank-3 shapes")
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        raise TypeError("overlap oracle states must be float32")
    if kv_state.shape[1:] != (2 * RATIO4, 2 * output_dim):
        raise ValueError("overlap state shape does not match ratio/output width")
    if not bool(torch.isfinite(kv_state).all()):
        raise ValueError("overlap KV state must be finite")
    valid_scores = torch.isfinite(score_state) | torch.isneginf(score_state)
    if not bool(valid_scores.all()):
        raise ValueError("overlap scores must be finite or negative infinity")

    values = torch.cat(
        (
            kv_state[:, :RATIO4, :output_dim],
            kv_state[:, RATIO4:, output_dim:],
        ),
        dim=1,
    )
    logits = torch.cat(
        (
            score_state[:, :RATIO4, :output_dim],
            score_state[:, RATIO4:, output_dim:],
        ),
        dim=1,
    )
    maximum = logits.amax(dim=1, keepdim=True)
    if not bool(torch.isfinite(maximum).all()):
        raise ValueError("every overlap latent dimension needs one finite score")
    exponent = torch.exp(logits - maximum)
    probabilities = exponent / exponent.sum(dim=1, keepdim=True)
    pooled = torch.sum(values * probabilities, dim=1, keepdim=True)
    return pooled, values, logits


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        if name not in config:
            raise ValueError(f"ratio-4 oracle config is missing {name}")
        return config[name]
    try:
        return getattr(config, name)
    except AttributeError as exc:
        raise ValueError(f"ratio-4 oracle config is missing {name}") from exc


def _dimensions(config: Any) -> dict[str, int | float]:
    # Every dimension is derived from the caller-supplied config; nothing here
    # freezes Pro geometry, so the V4-Flash port (hidden 4096, 64 heads,
    # q_lora 1024, o_groups 8, index_topk 512) needs no code change -- the
    # invariant checks below hold for both geometries.
    integer_names = (
        "hidden_size",
        "num_heads",
        "head_dim",
        "rope_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
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
    if any(int(values[name]) <= 0 for name in integer_names if name != "original_seq_len"):
        raise ValueError("ratio-4 oracle dimensions must be positive")
    if int(values["original_seq_len"]) < 0:
        raise ValueError("original_seq_len must be non-negative")
    head_dim = int(values["head_dim"])
    rope_dim = int(values["rope_dim"])
    index_dim = int(values["index_head_dim"])
    if rope_dim % 2 or not 0 < rope_dim < min(head_dim, index_dim):
        raise ValueError("rope_dim must be even and smaller than both latent widths")
    if (head_dim - rope_dim) % 64:
        raise ValueError("attention NoPE width must be a multiple of 64")
    if index_dim % 32 or index_dim & (index_dim - 1):
        raise ValueError("index width must be a power of two divisible by 32")
    if int(values["num_heads"]) * head_dim % int(values["o_groups"]):
        raise ValueError("attention heads must divide output groups")
    max_seq_len = int(values["max_seq_len"])
    if max_seq_len < WINDOW_SIZE or max_seq_len % RATIO4:
        raise ValueError("ratio-4 max_seq_len must be a multiple of four")
    if int(values["index_topk"]) > max_seq_len // RATIO4:
        raise ValueError("index_topk exceeds compressed state capacity")
    if any(
        not math.isfinite(float(values[name])) or float(values[name]) <= 0
        for name in ("norm_eps", "rope_theta", "rope_factor")
    ):
        raise ValueError("ratio-4 numerical constants must be finite and positive")
    return values


def _tensor_storage_identity(value: torch.Tensor) -> tuple[str, int]:
    return (str(value.device), value.untyped_storage().data_ptr())


def _projection_weight_dtype(weights: OracleRatio4Weights) -> torch.dtype:
    dtype = weights.wq_a.dtype
    if dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("ratio-4 oracle projection weights must be float32 or bfloat16")
    return dtype


def _validate_weight_storage(
    weights: OracleRatio4Weights,
    *,
    expected_projection_dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(weights, OracleRatio4Weights):
        raise TypeError("prepared ratio-4 oracle weights have the wrong type")
    projection_dtype = _projection_weight_dtype(weights)
    if (
        expected_projection_dtype is not None
        and projection_dtype != expected_projection_dtype
    ):
        raise TypeError(
            "ratio-4 oracle projection dtype "
            f"{projection_dtype} != {expected_projection_dtype}"
        )
    devices: set[torch.device] = set()
    storage_owners: dict[tuple[str, int], str] = {}
    for name in weights.__dataclass_fields__:
        value = getattr(weights, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"ratio-4 oracle weight {name} must be a tensor")
        expected_dtype = (
            projection_dtype
            if name in BF16_CONTROL_WEIGHT_FIELDS
            else torch.float32
        )
        if value.dtype != expected_dtype:
            dtype_name = str(expected_dtype).removeprefix("torch.")
            raise TypeError(
                f"ratio-4 oracle weight {name} must be {dtype_name}"
            )
        if not value.is_contiguous():
            raise ValueError(f"ratio-4 oracle weight {name} must be contiguous")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"ratio-4 oracle weight {name} must be finite")
        identity = _tensor_storage_identity(value)
        owner = storage_owners.get(identity)
        if owner is not None:
            raise ValueError(
                f"ratio-4 oracle weights {owner} and {name} alias one storage"
            )
        storage_owners[identity] = name
        devices.add(value.device)
    if len(devices) != 1:
        raise ValueError("ratio-4 oracle weights must share one device")


def _validate_prepared_ratio4_weights(
    weights: OracleRatio4Weights,
    dimensions: Mapping[str, int | float],
    *,
    storage_validated: bool = False,
) -> None:
    if not storage_validated:
        _validate_weight_storage(weights)
    hidden = int(dimensions["hidden_size"])
    heads = int(dimensions["num_heads"])
    head_dim = int(dimensions["head_dim"])
    q_rank = int(dimensions["q_lora_rank"])
    o_rank = int(dimensions["o_lora_rank"])
    groups = int(dimensions["o_groups"])
    index_heads = int(dimensions["index_n_heads"])
    index_dim = int(dimensions["index_head_dim"])
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
        "compressor_ape": (RATIO4, 2 * head_dim),
        "compressor_wkv": (2 * head_dim, hidden),
        "compressor_wgate": (2 * head_dim, hidden),
        "compressor_norm": (head_dim,),
        "index_wq_b": (index_heads * index_dim, q_rank),
        "index_weights_proj": (index_heads, hidden),
        "index_compressor_ape": (RATIO4, 2 * index_dim),
        "index_compressor_wkv": (2 * index_dim, hidden),
        "index_compressor_wgate": (2 * index_dim, hidden),
        "index_compressor_norm": (index_dim,),
    }
    for name, shape in expected.items():
        observed = getattr(weights, name)
        if tuple(observed.shape) != shape:
            raise ValueError(
                f"ratio-4 oracle weight {name} shape {tuple(observed.shape)} != {shape}"
            )


def _validate_ratio4_state(
    state: OracleRatio4State,
    *,
    batch_size: int,
    start_pos: int,
    head_dim: int,
    index_dim: int,
    max_seq_len: int,
    device: torch.device,
) -> None:
    if not isinstance(state, OracleRatio4State):
        raise TypeError("state must be an OracleRatio4State")
    capacity = max_seq_len // RATIO4
    expected = {
        "raw": ((batch_size, WINDOW_SIZE, head_dim), torch.bfloat16),
        "compressed": ((batch_size, capacity, head_dim), torch.bfloat16),
        "indexer_kv": ((batch_size, capacity, index_dim), torch.bfloat16),
        "main_kv": ((batch_size, 2 * RATIO4, 2 * head_dim), torch.float32),
        "main_score": ((batch_size, 2 * RATIO4, 2 * head_dim), torch.float32),
        "index_kv": ((batch_size, 2 * RATIO4, 2 * index_dim), torch.float32),
        "index_score": ((batch_size, 2 * RATIO4, 2 * index_dim), torch.float32),
    }
    storage_owners: dict[tuple[str, int], str] = {}
    for name, (shape, dtype) in expected.items():
        value = getattr(state, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"ratio-4 oracle state {name} must be a tensor")
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"ratio-4 oracle state {name} must have shape {shape} and dtype {dtype}"
            )
        if value.device != device:
            raise ValueError("ratio-4 oracle state and hidden must share one device")
        if not value.is_contiguous():
            raise ValueError(f"ratio-4 oracle state {name} must be contiguous")
        identity = _tensor_storage_identity(value)
        owner = storage_owners.get(identity)
        if owner is not None:
            raise ValueError(
                f"ratio-4 oracle states {owner} and {name} alias one storage"
            )
        storage_owners[identity] = name

    latent_names = ("raw", "compressed", "indexer_kv", "main_kv", "index_kv")
    if not all(
        bool(torch.isfinite(getattr(state, name)).all()) for name in latent_names
    ):
        raise ValueError("ratio-4 oracle latent and KV states must be finite")
    for name in ("main_score", "index_score"):
        score = getattr(state, name)
        valid = torch.isfinite(score) | torch.isneginf(score)
        if not bool(valid.all()):
            raise ValueError(
                f"ratio-4 oracle state {name} must be finite or negative infinity"
            )

    integer_metadata = {
        "next_position": state.next_position,
        "compressed_count": state.compressed_count,
        "max_seq_len": state.max_seq_len,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in integer_metadata.values()
    ):
        raise TypeError("ratio-4 oracle state metadata must use integer scalars")
    if state.max_seq_len != max_seq_len:
        raise ValueError("ratio-4 oracle state and config capacities differ")
    if state.next_position != start_pos:
        raise ValueError("ratio-4 oracle state does not match start_pos")
    if state.compressed_count != start_pos // RATIO4:
        raise ValueError("ratio-4 oracle compressed_count is inconsistent")

    active_overlap = RATIO4 + start_pos % RATIO4
    for name in ("main_score", "index_score"):
        value = getattr(state, name)
        if not bool(torch.isfinite(value[:, :active_overlap]).all()):
            raise ValueError(f"active ratio-4 state {name} rows must be finite")


def _prepare_ratio4_weights(
    weights: Any,
    *,
    projection_dtype: torch.dtype,
) -> OracleRatio4Weights:
    if projection_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("ratio-4 projection dtype must be float32 or bfloat16")

    if isinstance(weights, OracleRatio4Weights):
        _validate_weight_storage(
            weights, expected_projection_dtype=projection_dtype
        )
        return weights

    def dequant(owner: Any, name: str) -> torch.Tensor:
        try:
            linear = getattr(owner, name)
            return oracle_dequant_fp8_block(
                linear.weight, linear.scale
            ).to(projection_dtype)
        except AttributeError as exc:
            raise TypeError(f"missing quantized linear {name}") from exc

    try:
        compressor = weights.compressor
        indexer = weights.indexer
        if indexer is None:
            raise TypeError("ratio-4 oracle requires indexer weights")
        index_compressor = indexer.compressor
        result = OracleRatio4Weights(
            attn_sink=weights.attn_sink.float().contiguous().clone(),
            wq_a=dequant(weights, "wq_a"),
            q_norm=weights.q_norm.float().contiguous().clone(),
            wq_b=dequant(weights, "wq_b"),
            wkv=dequant(weights, "wkv"),
            kv_norm=weights.kv_norm.float().contiguous().clone(),
            wo_a=dequant(weights, "wo_a"),
            wo_b=dequant(weights, "wo_b"),
            compressor_ape=compressor.ape.float().contiguous().clone(),
            compressor_wkv=compressor.wkv.float().contiguous().clone(),
            compressor_wgate=compressor.wgate.float().contiguous().clone(),
            compressor_norm=compressor.norm.float().contiguous().clone(),
            index_wq_b=dequant(indexer, "wq_b"),
            index_weights_proj=indexer.weights_proj.to(projection_dtype)
            .contiguous()
            .clone(),
            index_compressor_ape=index_compressor.ape.float().contiguous().clone(),
            index_compressor_wkv=index_compressor.wkv.float().contiguous().clone(),
            index_compressor_wgate=index_compressor.wgate.float().contiguous().clone(),
            index_compressor_norm=index_compressor.norm.float().contiguous().clone(),
        )
    except AttributeError as exc:
        raise TypeError("resident ratio-4 weights do not satisfy the raw contract") from exc
    _validate_weight_storage(
        result, expected_projection_dtype=projection_dtype
    )
    return result


def oracle_prepare_ratio4_weights(weights: Any) -> OracleRatio4Weights:
    """Create a non-aliasing raw-FP32 view from resident checkpoint tensors."""

    return _prepare_ratio4_weights(weights, projection_dtype=torch.float32)


def oracle_prepare_ratio4_bf16_control_weights(
    weights: Any,
) -> OracleRatio4Weights:
    """Create an independently prepared BF16-operand control weight set.

    This mode matches the effective projection operands used by the direct
    BF16-dequantized control.  It is not the checkpoint-native FP8 GEMM path;
    the raw-FP32 oracle remains a separate attribution lane.
    """

    return _prepare_ratio4_weights(weights, projection_dtype=torch.bfloat16)


def _linear_bf16(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.bfloat16:
        return torch.matmul(
            value.to(torch.bfloat16), weight.transpose(0, 1)
        )
    if weight.dtype != torch.float32:
        raise TypeError("ratio-4 oracle linear weight has an invalid dtype")
    return torch.matmul(value.float(), weight.transpose(0, 1)).to(torch.bfloat16)


def oracle_reference_bf16_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Evaluate RMSNorm with the checkpoint reference's BF16 operator order.

    The reference uses ``x * rsqrt(mean(x**2) + eps) * weight``. Rewriting it
    as ``x / sqrt(...) * weight`` can change a BF16 rounding decision before
    the indexer's discontinuous E2M1 QDQ.
    """

    if value.ndim < 1 or weight.ndim != 1 or value.shape[-1] != weight.numel():
        raise ValueError("reference BF16 RMSNorm value and weight shapes differ")
    if value.dtype != torch.bfloat16 or not weight.is_floating_point():
        raise TypeError(
            "reference BF16 RMSNorm requires BF16 values and float weights"
        )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("reference BF16 RMSNorm epsilon must be positive and finite")
    value_fp32 = value.to(torch.float32)
    inverse_rms = torch.rsqrt(
        torch.mean(value_fp32 * value_fp32, dim=-1, keepdim=True) + eps
    )
    return (value_fp32 * inverse_rms * weight.to(torch.float32)).to(value.dtype)


def _lane_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    normalization_profile: str,
) -> torch.Tensor:
    if normalization_profile == REFERENCE_BF16_NORMALIZATION_PROFILE:
        return oracle_reference_bf16_rms_norm(value, weight, eps=eps)
    if normalization_profile == RAW_FP32_NORMALIZATION_PROFILE:
        return oracle_rms_norm(value, weight, eps=eps)
    raise ValueError(
        f"unknown ratio-4 normalization profile: {normalization_profile}"
    )


def _lane_query_rms_norm(
    value: torch.Tensor,
    *,
    eps: float,
    normalization_profile: str,
) -> torch.Tensor:
    if normalization_profile == REFERENCE_BF16_NORMALIZATION_PROFILE:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    if normalization_profile == RAW_FP32_NORMALIZATION_PROFILE:
        value_fp32 = value.to(torch.float32)
        return (
            value_fp32
            / torch.sqrt(
                torch.mean(value_fp32 * value_fp32, dim=-1, keepdim=True) + eps
            )
        ).to(value.dtype)
    raise ValueError(
        f"unknown ratio-4 normalization profile: {normalization_profile}"
    )


def _finalize_main(
    pooled: torch.Tensor,
    norm: torch.Tensor,
    table: RopeTable,
    position: int,
    rope_dim: int,
    eps: float,
    normalization_profile: str,
) -> torch.Tensor:
    value = _lane_rms_norm(
        pooled.to(torch.bfloat16),
        norm,
        eps=eps,
        normalization_profile=normalization_profile,
    ).clone()
    row_table = RopeTable(
        cos=table.cos[position : position + 1],
        sin=table.sin[position : position + 1],
    )
    value[..., -rope_dim:] = oracle_apply_rope(value[..., -rope_dim:], row_table)
    value[..., :-rope_dim] = e4m3_ue8m0_qdq(
        value[..., :-rope_dim], group_size=64
    ).dequantized
    return value.contiguous()


def _finalize_index(
    pooled: torch.Tensor,
    norm: torch.Tensor,
    table: RopeTable,
    position: int,
    rope_dim: int,
    eps: float,
    normalization_profile: str,
) -> torch.Tensor:
    value = _lane_rms_norm(
        pooled.to(torch.bfloat16),
        norm,
        eps=eps,
        normalization_profile=normalization_profile,
    ).clone()
    row_table = RopeTable(
        cos=table.cos[position : position + 1],
        sin=table.sin[position : position + 1],
    )
    value[..., -rope_dim:] = oracle_apply_rope(value[..., -rope_dim:], row_table)
    return oracle_e2m1_qdq(oracle_hadamard(value), group_size=32).contiguous()


def seed_nonzero_ratio4_state(
    config: Any,
    *,
    batch_size: int,
    start_pos: int,
    main_ape: torch.Tensor,
    index_ape: torch.Tensor,
    seed: int,
    device: torch.device | str = "cpu",
) -> OracleRatio4State:
    """Build a deterministic nonzero, QAT-valid decode state without candidate code."""

    dimensions = _dimensions(config)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("oracle batch_size must be positive")
    max_seq_len = int(dimensions["max_seq_len"])
    if not WINDOW_SIZE <= start_pos < max_seq_len or start_pos % RATIO4:
        raise ValueError("nonzero oracle seed requires a phase-0 decode position")
    head_dim = int(dimensions["head_dim"])
    index_dim = int(dimensions["index_head_dim"])
    rope_dim = int(dimensions["rope_dim"])
    target = torch.device(device)
    if tuple(main_ape.shape) != (RATIO4, 2 * head_dim):
        raise ValueError("main APE shape is invalid")
    if tuple(index_ape.shape) != (RATIO4, 2 * index_dim):
        raise ValueError("index APE shape is invalid")
    if main_ape.dtype != torch.float32 or index_ape.dtype != torch.float32:
        raise TypeError("ratio-4 APE tensors must be float32")
    if main_ape.device != target or index_ape.device != target:
        raise ValueError("ratio-4 APE tensors must use the target device")

    generator = torch.Generator().manual_seed(seed)

    def normal(shape: tuple[int, ...], scale: float, dtype: torch.dtype) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * scale).to(dtype).to(target)

    raw = normal((batch_size, WINDOW_SIZE, head_dim), 0.03, torch.bfloat16)
    raw_nope = e4m3_ue8m0_qdq(raw[..., :-rope_dim], group_size=64).dequantized
    raw[..., :-rope_dim] = raw_nope
    capacity = max_seq_len // RATIO4
    compressed = torch.zeros(
        batch_size, capacity, head_dim, dtype=torch.bfloat16, device=target
    )
    completed = start_pos // RATIO4
    active_compressed = normal(
        (batch_size, completed, head_dim), 0.025, torch.bfloat16
    )
    active_compressed[..., :-rope_dim] = e4m3_ue8m0_qdq(
        active_compressed[..., :-rope_dim], group_size=64
    ).dequantized
    compressed[:, :completed].copy_(active_compressed)
    indexer_kv = torch.zeros(
        batch_size, capacity, index_dim, dtype=torch.bfloat16, device=target
    )
    active_index = normal(
        (batch_size, completed, index_dim), 0.025, torch.bfloat16
    )
    indexer_kv[:, :completed].copy_(oracle_e2m1_qdq(active_index))

    main_kv = torch.zeros(
        batch_size, 2 * RATIO4, 2 * head_dim, dtype=torch.float32, device=target
    )
    main_score = torch.full_like(main_kv, float("-inf"))
    index_kv = torch.zeros(
        batch_size, 2 * RATIO4, 2 * index_dim, dtype=torch.float32, device=target
    )
    index_score = torch.full_like(index_kv, float("-inf"))
    main_kv[:, :RATIO4].copy_(
        normal((batch_size, RATIO4, 2 * head_dim), 0.02, torch.float32)
    )
    index_kv[:, :RATIO4].copy_(
        normal((batch_size, RATIO4, 2 * index_dim), 0.02, torch.float32)
    )
    main_base = normal((batch_size, RATIO4, 2 * head_dim), 0.1, torch.float32)
    index_base = normal((batch_size, RATIO4, 2 * index_dim), 0.1, torch.float32)
    main_score[:, :RATIO4].copy_(main_base + main_ape.unsqueeze(0))
    index_score[:, :RATIO4].copy_(index_base + index_ape.unsqueeze(0))
    return OracleRatio4State(
        raw=raw,
        compressed=compressed,
        indexer_kv=indexer_kv,
        main_kv=main_kv,
        main_score=main_score,
        index_kv=index_kv,
        index_score=index_score,
        next_position=start_pos,
        compressed_count=completed,
        max_seq_len=max_seq_len,
    )


def _oracle_ratio4_attention_step(
    config: Any,
    weights: Any,
    hidden: torch.Tensor,
    *,
    start_pos: int,
    state: OracleRatio4State,
    rope_table: RopeTable | None = None,
    normalization_profile: str,
) -> OracleRatio4Step:
    """Evaluate one teacher-forced ratio-4 decode step from an immutable state."""

    dimensions = _dimensions(config)
    if normalization_profile == REFERENCE_BF16_NORMALIZATION_PROFILE:
        prepared = oracle_prepare_ratio4_bf16_control_weights(weights)
    elif normalization_profile == RAW_FP32_NORMALIZATION_PROFILE:
        prepared = oracle_prepare_ratio4_weights(weights)
    else:
        raise ValueError(
            f"unknown ratio-4 normalization profile: {normalization_profile}"
        )
    if (
        hidden.ndim != 3
        or hidden.dtype != torch.bfloat16
        or hidden.shape[1] != 1
        or not hidden.is_contiguous()
        or not bool(torch.isfinite(hidden).all())
    ):
        raise ValueError("ratio-4 oracle hidden must be [batch,1,hidden] BF16")
    batch = hidden.shape[0]
    hidden_size = int(dimensions["hidden_size"])
    head_dim = int(dimensions["head_dim"])
    rope_dim = int(dimensions["rope_dim"])
    index_dim = int(dimensions["index_head_dim"])
    max_seq_len = int(dimensions["max_seq_len"])
    if (
        hidden.shape[2] != hidden_size
        or not isinstance(start_pos, int)
        or isinstance(start_pos, bool)
        or not WINDOW_SIZE <= start_pos < max_seq_len
    ):
        raise ValueError("ratio-4 hidden shape or start position is invalid")
    _validate_prepared_ratio4_weights(
        prepared, dimensions, storage_validated=True
    )
    _validate_ratio4_state(
        state,
        batch_size=batch,
        start_pos=start_pos,
        head_dim=head_dim,
        index_dim=index_dim,
        max_seq_len=max_seq_len,
        device=hidden.device,
    )
    if prepared.wq_a.device != hidden.device:
        raise ValueError("ratio-4 oracle weights and hidden must share one device")
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
    step_table = RopeTable(
        cos=rope_table.cos[start_pos : start_pos + 1],
        sin=rope_table.sin[start_pos : start_pos + 1],
    )
    working = state.clone()
    phase = start_pos % RATIO4
    boundary = phase == RATIO4 - 1
    overlap_slot = RATIO4 + phase
    compressed_row = start_pos // RATIO4
    eps = float(dimensions["norm_eps"])

    query_lora = _lane_rms_norm(
        _linear_bf16(hidden, prepared.wq_a),
        prepared.q_norm,
        eps=eps,
        normalization_profile=normalization_profile,
    )
    query = _linear_bf16(query_lora, prepared.wq_b).reshape(
        batch, 1, int(dimensions["num_heads"]), head_dim
    )
    query = _lane_query_rms_norm(
        query,
        eps=eps,
        normalization_profile=normalization_profile,
    )
    query[..., -rope_dim:] = oracle_apply_rope(
        query[..., -rope_dim:], step_table
    )

    raw_latent = _lane_rms_norm(
        _linear_bf16(hidden, prepared.wkv),
        prepared.kv_norm,
        eps=eps,
        normalization_profile=normalization_profile,
    ).clone()
    raw_latent[..., -rope_dim:] = oracle_apply_rope(
        raw_latent[..., -rope_dim:], step_table
    )
    raw_latent[..., :-rope_dim] = e4m3_ue8m0_qdq(
        raw_latent[..., :-rope_dim], group_size=64
    ).dequantized
    working.raw[:, start_pos % WINDOW_SIZE].copy_(raw_latent[:, 0])

    main_projected = torch.matmul(
        hidden.float(), prepared.compressor_wkv.transpose(0, 1)
    )
    main_score_projected = torch.matmul(
        hidden.float(), prepared.compressor_wgate.transpose(0, 1)
    )
    working.main_kv[:, overlap_slot].copy_(main_projected[:, 0])
    working.main_score[:, overlap_slot].copy_(
        main_score_projected[:, 0] + prepared.compressor_ape[phase]
    )
    main_values = main_logits = main_pooled = main_finalized = None
    if boundary:
        main_pooled, main_values, main_logits = oracle_overlap_pool(
            working.main_kv, working.main_score, output_dim=head_dim
        )
        main_finalized = _finalize_main(
            main_pooled,
            prepared.compressor_norm,
            rope_table,
            start_pos + 1 - RATIO4,
            rope_dim,
            eps,
            normalization_profile,
        )
        working.compressed[:, compressed_row : compressed_row + 1].copy_(
            main_finalized
        )
        working.main_kv[:, :RATIO4].copy_(working.main_kv[:, RATIO4:])
        working.main_score[:, :RATIO4].copy_(working.main_score[:, RATIO4:])

    index_projected = torch.matmul(
        hidden.float(), prepared.index_compressor_wkv.transpose(0, 1)
    )
    index_score_projected = torch.matmul(
        hidden.float(), prepared.index_compressor_wgate.transpose(0, 1)
    )
    working.index_kv[:, overlap_slot].copy_(index_projected[:, 0])
    working.index_score[:, overlap_slot].copy_(
        index_score_projected[:, 0] + prepared.index_compressor_ape[phase]
    )
    index_values = index_logits = index_pooled = index_finalized = None
    if boundary:
        index_pooled, index_values, index_logits = oracle_overlap_pool(
            working.index_kv, working.index_score, output_dim=index_dim
        )
        index_finalized = _finalize_index(
            index_pooled,
            prepared.index_compressor_norm,
            rope_table,
            start_pos + 1 - RATIO4,
            rope_dim,
            eps,
            normalization_profile,
        )
        working.indexer_kv[:, compressed_row : compressed_row + 1].copy_(
            index_finalized
        )
        working.index_kv[:, :RATIO4].copy_(working.index_kv[:, RATIO4:])
        working.index_score[:, :RATIO4].copy_(working.index_score[:, RATIO4:])

    compressed_count = (start_pos + 1) // RATIO4
    index_query = _linear_bf16(query_lora, prepared.index_wq_b).reshape(
        batch, 1, int(dimensions["index_n_heads"]), index_dim
    )
    index_query[..., -rope_dim:] = oracle_apply_rope(
        index_query[..., -rope_dim:], step_table
    )
    index_query = oracle_e2m1_qdq(oracle_hadamard(index_query))
    index_weights = _linear_bf16(hidden, prepared.index_weights_proj)
    index_weights = index_weights * (
        index_dim**-0.5 * int(dimensions["index_n_heads"]) ** -0.5
    )
    active_index = working.indexer_kv[:, :compressed_count]
    per_head_scores = torch.matmul(
        index_query.float(), active_index.float().transpose(1, 2).unsqueeze(1)
    )
    index_scores = torch.sum(
        torch.clamp_min(per_head_scores, 0.0)
        * index_weights.float().unsqueeze(-1),
        dim=2,
    )
    topk_count = min(int(dimensions["index_topk"]), compressed_count)
    compressed_indices = torch.topk(index_scores, topk_count, dim=-1).indices
    window = oracle_window_topk_indices(
        batch_size=batch,
        seqlen=1,
        start_pos=start_pos,
        device=hidden.device,
        window_size=WINDOW_SIZE,
    ).to(torch.int64)
    topk = torch.cat((window, compressed_indices + WINDOW_SIZE), dim=-1)
    latent = torch.cat((working.raw, working.compressed), dim=1)
    batch_index = torch.arange(batch, device=hidden.device).view(batch, 1, 1)
    batch_index = batch_index.expand_as(topk)
    selected = latent[batch_index, topk]
    sparse = oracle_sparse_attention(
        query,
        latent,
        prepared.attn_sink,
        topk,
        head_dim**-0.5,
    )
    inverse = sparse.clone()
    inverse[..., -rope_dim:] = oracle_apply_rope(
        inverse[..., -rope_dim:], step_table, inverse=True
    )
    groups = int(dimensions["o_groups"])
    o_rank = int(dimensions["o_lora_rank"])
    grouped_width = int(dimensions["num_heads"]) * head_dim // groups
    grouped = inverse.reshape(batch, 1, groups, grouped_width)
    wo_a = prepared.wo_a.reshape(groups, o_rank, grouped_width)
    if prepared.wo_a.dtype == torch.bfloat16:
        output_lora = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
    else:
        output_lora = torch.einsum(
            "bsgd,grd->bsgr", grouped.float(), wo_a
        ).to(torch.bfloat16)
    branch = _linear_bf16(output_lora.flatten(2), prepared.wo_b)

    next_state = OracleRatio4State(
        raw=working.raw,
        compressed=working.compressed,
        indexer_kv=working.indexer_kv,
        main_kv=working.main_kv,
        main_score=working.main_score,
        index_kv=working.index_kv,
        index_score=working.index_score,
        next_position=start_pos + 1,
        compressed_count=compressed_count,
        max_seq_len=max_seq_len,
    )
    return OracleRatio4Step(
        trace=OracleRatio4Trace(
            query_lora=query_lora,
            query=query,
            raw_latent=raw_latent,
            main_projected_kv=main_projected,
            main_projected_score=main_score_projected,
            main_overlap_values=main_values,
            main_overlap_logits=main_logits,
            main_compression_pooled=main_pooled,
            main_compression_finalized=main_finalized,
            index_projected_kv=index_projected,
            index_projected_score=index_score_projected,
            index_overlap_values=index_values,
            index_overlap_logits=index_logits,
            index_compression_pooled=index_pooled,
            index_compression_finalized=index_finalized,
            index_query=index_query,
            index_weights=index_weights,
            index_scores=index_scores,
            compressed_indices=compressed_indices,
            topk_indices=topk,
            selected_kv=selected,
            sparse_output=sparse,
            inverse_rotated=inverse,
            output_lora=output_lora,
            branch=branch,
        ),
        state=next_state,
    )


def oracle_ratio4_attention_step(
    config: Any,
    weights: Any,
    hidden: torch.Tensor,
    *,
    start_pos: int,
    state: OracleRatio4State,
    rope_table: RopeTable | None = None,
) -> OracleRatio4Step:
    """Evaluate the raw-FP32 attribution profile for one ratio-4 step."""

    return _oracle_ratio4_attention_step(
        config,
        weights,
        hidden,
        start_pos=start_pos,
        state=state,
        rope_table=rope_table,
        normalization_profile=RAW_FP32_NORMALIZATION_PROFILE,
    )


def oracle_ratio4_bf16_control_step(
    config: Any,
    weights: Any,
    hidden: torch.Tensor,
    *,
    start_pos: int,
    state: OracleRatio4State,
    rope_table: RopeTable | None = None,
) -> OracleRatio4Step:
    """Evaluate the reference-faithful BF16 profile from independent state/weights."""

    return _oracle_ratio4_attention_step(
        config,
        weights,
        hidden,
        start_pos=start_pos,
        state=state,
        rope_table=rope_table,
        normalization_profile=REFERENCE_BF16_NORMALIZATION_PROFILE,
    )


__all__ = [
    "BF16_CONTROL_WEIGHT_FIELDS",
    "RAW_FP32_NORMALIZATION_PROFILE",
    "REFERENCE_BF16_NORMALIZATION_PROFILE",
    "REFERENCE_NORMALIZATION_SOURCE_PATH",
    "REFERENCE_NORMALIZATION_SOURCE_SHA256",
    "OracleHashRoute",
    "OracleRatio4State",
    "OracleRatio4Step",
    "OracleRatio4Trace",
    "OracleRatio4Weights",
    "oracle_e2m1_qdq",
    "oracle_hadamard",
    "oracle_hash_route",
    "oracle_overlap_pool",
    "oracle_prepare_ratio4_bf16_control_weights",
    "oracle_prepare_ratio4_weights",
    "oracle_reference_bf16_rms_norm",
    "oracle_ratio4_bf16_control_step",
    "oracle_ratio4_attention_step",
    "seed_nonzero_ratio4_state",
]
