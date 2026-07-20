"""Replicated frozen physical block-weight contract and loader.

The direct runtime keeps attention, compressor, norms, hyper-connections, and
the MoE gate replicated on every TP4 rank. Routed and shared expert weights
remain owned by the separate intermediate-TP loader.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import (
    DTYPE_BYTES,
    CheckpointContractError,
    TensorSpec,
    layer_prefix,
    load_weight_map,
    read_safetensors_header,
)
from .model_contract import (
    MODEL_LAYER_IDS,
    MTP_LAYER_ID,
    ModelContractError,
    SUPPORTED_LAYER_SPECS,
    validate_model_layer_config,
)


SUPPORTED_LAYER_ID = 3
# Physical decode layers plus the MTP block (checkpoint prefix mtp.0).
SUPPORTED_LAYER_IDS = MODEL_LAYER_IDS + (MTP_LAYER_ID,)
SUPPORTED_TP_SIZE = 4
SUPPORTED_COMPRESS_RATIOS = {
    layer_id: int(specification["compress_ratio"])
    for layer_id, specification in SUPPORTED_LAYER_SPECS.items()
}
WEIGHT_BLOCK_SIZE = 128

@dataclass(frozen=True)
class ExpectedBlockTensor:
    shape: tuple[int, ...]
    dtypes: tuple[str, ...]

    @property
    def nbytes(self) -> int:
        elements = 1
        for dimension in self.shape:
            elements *= dimension
        return elements * DTYPE_BYTES[self.dtypes[0]]


@dataclass
class QuantizedLinearWeights:
    weight: torch.Tensor
    scale: torch.Tensor


@dataclass
class ResidentCompressorWeights:
    ape: torch.Tensor
    wkv: torch.Tensor
    wgate: torch.Tensor
    norm: torch.Tensor


ResidentRatio128CompressorWeights = ResidentCompressorWeights


@dataclass
class ResidentIndexerWeights:
    wq_b: QuantizedLinearWeights
    weights_proj: torch.Tensor
    compressor: ResidentCompressorWeights


@dataclass
class ResidentAttentionWeights:
    attn_sink: torch.Tensor
    wq_a: QuantizedLinearWeights
    q_norm: torch.Tensor
    wq_b: QuantizedLinearWeights
    wkv: QuantizedLinearWeights
    kv_norm: torch.Tensor
    wo_a: QuantizedLinearWeights
    wo_b: QuantizedLinearWeights
    # window layers (compress_ratio 0) have neither compressor nor indexer
    compressor: ResidentCompressorWeights | None = None
    indexer: ResidentIndexerWeights | None = None
    layer_id: int | None = None
    rank: int | None = None
    world_size: int | None = None
    checkpoint_id: str | None = None


@dataclass
class ResidentMTPWeights:
    """mtp.0-only tensors (reference model.py MTPBlock :738-755).

    ``e_proj``/``h_proj`` are the FP8 bridge projections; ``enorm``/``hnorm``
    normalize the embedding and the incoming HC hidden; ``norm`` is the MTP
    block's own terminal RMSNorm before the *shared* head projection; the
    ``hc_head_*`` parameters collapse the four HC streams with the sigmoid
    form of ``ParallelHead.hc_head`` (model.py:728-735) using MTP-owned
    parameters (model.py:750-752, :765).
    """

    e_proj: QuantizedLinearWeights
    h_proj: QuantizedLinearWeights
    enorm: torch.Tensor
    hnorm: torch.Tensor
    norm: torch.Tensor
    hc_head_fn: torch.Tensor
    hc_head_base: torch.Tensor
    hc_head_scale: torch.Tensor


@dataclass
class ResidentHyperConnectionWeights:
    attn_fn: torch.Tensor
    attn_base: torch.Tensor
    attn_scale: torch.Tensor
    ffn_fn: torch.Tensor
    ffn_base: torch.Tensor
    ffn_scale: torch.Tensor


@dataclass
class ResidentGateWeights:
    weight: torch.Tensor
    bias: torch.Tensor | None = None
    layer_id: int | None = None
    rank: int | None = None
    world_size: int | None = None
    checkpoint_id: str | None = None
    tid2eid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.bias is None) == (self.tid2eid is None):
            raise ValueError("gate requires exactly one of bias or tid2eid")

    @property
    def route_kind(self) -> str:
        return "hash" if self.tid2eid is not None else "learned"


@dataclass
class ResidentBlockWeights:
    layer_id: int
    rank: int
    world_size: int
    checkpoint_id: str | None
    contract_id: str
    attention: ResidentAttentionWeights
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    hyper_connection: ResidentHyperConnectionWeights
    gate: ResidentGateWeights
    load_seconds: float
    mtp: ResidentMTPWeights | None = None

    def named_tensors(self) -> dict[str, torch.Tensor]:
        layer = layer_prefix(self.layer_id)
        attn = f"{layer}.attn"
        compressor = f"{attn}.compressor"
        tensors = {
            f"{attn}.attn_sink": self.attention.attn_sink,
            f"{attn}.wq_a.weight": self.attention.wq_a.weight,
            f"{attn}.wq_a.scale": self.attention.wq_a.scale,
            f"{attn}.q_norm.weight": self.attention.q_norm,
            f"{attn}.wq_b.weight": self.attention.wq_b.weight,
            f"{attn}.wq_b.scale": self.attention.wq_b.scale,
            f"{attn}.wkv.weight": self.attention.wkv.weight,
            f"{attn}.wkv.scale": self.attention.wkv.scale,
            f"{attn}.kv_norm.weight": self.attention.kv_norm,
            f"{attn}.wo_a.weight": self.attention.wo_a.weight,
            f"{attn}.wo_a.scale": self.attention.wo_a.scale,
            f"{attn}.wo_b.weight": self.attention.wo_b.weight,
            f"{attn}.wo_b.scale": self.attention.wo_b.scale,
            f"{layer}.attn_norm.weight": self.attn_norm,
            f"{layer}.ffn_norm.weight": self.ffn_norm,
            f"{layer}.hc_attn_fn": self.hyper_connection.attn_fn,
            f"{layer}.hc_attn_base": self.hyper_connection.attn_base,
            f"{layer}.hc_attn_scale": self.hyper_connection.attn_scale,
            f"{layer}.hc_ffn_fn": self.hyper_connection.ffn_fn,
            f"{layer}.hc_ffn_base": self.hyper_connection.ffn_base,
            f"{layer}.hc_ffn_scale": self.hyper_connection.ffn_scale,
            f"{layer}.ffn.gate.weight": self.gate.weight,
        }
        if self.attention.compressor is not None:
            tensors.update(
                {
                    f"{compressor}.ape": self.attention.compressor.ape,
                    f"{compressor}.wkv.weight": self.attention.compressor.wkv,
                    f"{compressor}.wgate.weight": self.attention.compressor.wgate,
                    f"{compressor}.norm.weight": self.attention.compressor.norm,
                }
            )
        if self.attention.indexer is not None:
            indexer = f"{attn}.indexer"
            indexer_compressor = f"{indexer}.compressor"
            tensors.update(
                {
                    f"{indexer}.wq_b.weight": self.attention.indexer.wq_b.weight,
                    f"{indexer}.wq_b.scale": self.attention.indexer.wq_b.scale,
                    f"{indexer}.weights_proj.weight": self.attention.indexer.weights_proj,
                    f"{indexer_compressor}.ape": self.attention.indexer.compressor.ape,
                    f"{indexer_compressor}.wkv.weight": self.attention.indexer.compressor.wkv,
                    f"{indexer_compressor}.wgate.weight": self.attention.indexer.compressor.wgate,
                    f"{indexer_compressor}.norm.weight": self.attention.indexer.compressor.norm,
                }
            )
        if self.gate.bias is not None:
            tensors[f"{layer}.ffn.gate.bias"] = self.gate.bias
        if self.gate.tid2eid is not None:
            tensors[f"{layer}.ffn.gate.tid2eid"] = self.gate.tid2eid
        if self.mtp is not None:
            tensors.update(
                {
                    f"{layer}.e_proj.weight": self.mtp.e_proj.weight,
                    f"{layer}.e_proj.scale": self.mtp.e_proj.scale,
                    f"{layer}.h_proj.weight": self.mtp.h_proj.weight,
                    f"{layer}.h_proj.scale": self.mtp.h_proj.scale,
                    f"{layer}.enorm.weight": self.mtp.enorm,
                    f"{layer}.hnorm.weight": self.mtp.hnorm,
                    f"{layer}.norm.weight": self.mtp.norm,
                    f"{layer}.hc_head_fn": self.mtp.hc_head_fn,
                    f"{layer}.hc_head_base": self.mtp.hc_head_base,
                    f"{layer}.hc_head_scale": self.mtp.hc_head_scale,
                }
            )
        return tensors

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in self.named_tensors().values()
        )

    def summary(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "checkpoint_id": self.checkpoint_id,
            "residency": "replicated",
            "contract_id": self.contract_id,
            "route_kind": self.gate.route_kind,
            "has_indexer": self.attention.indexer is not None,
            "load_seconds": self.load_seconds,
            "resident_bytes": self.resident_bytes,
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "nbytes": int(tensor.numel() * tensor.element_size()),
                }
                for name, tensor in self.named_tensors().items()
            },
        }


def _config_int(config: Mapping[str, Any], name: str) -> int:
    try:
        value = int(config[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointContractError(f"missing or invalid config integer: {name}") from exc
    if value <= 0:
        raise CheckpointContractError(f"config {name} must be positive, got {value}")
    return value


def _config_nonnegative_int(config: Mapping[str, Any], name: str) -> int:
    try:
        value = int(config[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointContractError(f"missing or invalid config integer: {name}") from exc
    if value < 0:
        raise CheckpointContractError(f"config {name} must be nonnegative, got {value}")
    return value


def _layer_compress_ratio(config: Mapping[str, Any], layer_id: int) -> int:
    ratios = config.get("compress_ratios")
    if not isinstance(ratios, (list, tuple)) or len(ratios) <= layer_id:
        raise CheckpointContractError(
            f"config compress_ratios does not contain layer {layer_id}"
        )
    try:
        ratio = int(ratios[layer_id])
    except (TypeError, ValueError) as exc:
        raise CheckpointContractError(
            f"invalid compression ratio for layer {layer_id}: {ratios[layer_id]!r}"
        ) from exc
    if ratio < 0:
        # 0 = pure sliding-window layer (Flash L0/L1 and MTP): no compressor
        raise CheckpointContractError(
            f"compression ratio for layer {layer_id} must be nonnegative, got {ratio}"
        )
    return ratio


def _layer_route_kind(config: Mapping[str, Any], layer_id: int) -> str:
    hash_layers = _config_nonnegative_int(config, "num_hash_layers")
    return "hash" if layer_id < hash_layers else "learned"


def _fp8_linear_specs(
    *, out_features: int, in_features: int
) -> tuple[ExpectedBlockTensor, ExpectedBlockTensor]:
    if out_features % WEIGHT_BLOCK_SIZE or in_features % WEIGHT_BLOCK_SIZE:
        raise CheckpointContractError(
            f"FP8 matrix [{out_features}, {in_features}] must preserve "
            f"{WEIGHT_BLOCK_SIZE}x{WEIGHT_BLOCK_SIZE} scale blocks"
        )
    return (
        ExpectedBlockTensor(
            (out_features, in_features), ("F8_E4M3", "F8_E4M3FN")
        ),
        ExpectedBlockTensor(
            (out_features // WEIGHT_BLOCK_SIZE, in_features // WEIGHT_BLOCK_SIZE),
            ("F8_E8M0", "F8_E8M0FNU"),
        ),
    )


def expected_block_tensor_specs(
    config: Mapping[str, Any], layer_id: int = SUPPORTED_LAYER_ID
) -> dict[str, ExpectedBlockTensor]:
    """Build the exact replicated checkpoint ABI for one frozen model block."""

    if (
        not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or layer_id not in SUPPORTED_LAYER_IDS
    ):
        raise CheckpointContractError(
            f"block loader supports layers {SUPPORTED_LAYER_IDS}, got {layer_id}"
        )

    hidden = _config_int(config, "hidden_size")
    heads = _config_int(config, "num_attention_heads")
    head_dim = _config_int(config, "head_dim")
    q_rank = _config_int(config, "q_lora_rank")
    o_groups = _config_int(config, "o_groups")
    o_rank = _config_int(config, "o_lora_rank")
    hc_mult = _config_int(config, "hc_mult")
    experts = _config_int(config, "n_routed_experts")
    compress_ratio = _layer_compress_ratio(config, layer_id)
    expected_ratio = SUPPORTED_COMPRESS_RATIOS[layer_id]
    if compress_ratio != expected_ratio:
        raise CheckpointContractError(
            f"layer {layer_id} compression ratio must be {expected_ratio}, "
            f"got {compress_ratio}"
        )
    route_kind = _layer_route_kind(config, layer_id)
    expected_route_kind = str(SUPPORTED_LAYER_SPECS[layer_id]["route_kind"])
    if route_kind != expected_route_kind:
        raise CheckpointContractError(
            f"layer {layer_id} must use a {expected_route_kind} gate, "
            f"num_hash_layers={config.get('num_hash_layers')!r}"
        )

    quantization = config.get("quantization_config")
    block_size = (
        quantization.get("weight_block_size")
        if isinstance(quantization, Mapping)
        else None
    )
    if list(block_size or ()) != [WEIGHT_BLOCK_SIZE, WEIGHT_BLOCK_SIZE]:
        raise CheckpointContractError(
            "quantization_config.weight_block_size must be [128, 128]"
        )
    total_head_dim = heads * head_dim
    if total_head_dim % o_groups:
        raise CheckpointContractError(
            f"heads*head_dim={total_head_dim} must be divisible by o_groups={o_groups}"
        )
    o_features = o_groups * o_rank
    grouped_head_dim = total_head_dim // o_groups

    wq_a, wq_a_scale = _fp8_linear_specs(
        out_features=q_rank, in_features=hidden
    )
    wq_b, wq_b_scale = _fp8_linear_specs(
        out_features=total_head_dim, in_features=q_rank
    )
    wkv, wkv_scale = _fp8_linear_specs(
        out_features=head_dim, in_features=hidden
    )
    wo_a, wo_a_scale = _fp8_linear_specs(
        out_features=o_features, in_features=grouped_head_dim
    )
    wo_b, wo_b_scale = _fp8_linear_specs(
        out_features=hidden, in_features=o_features
    )

    layer = layer_prefix(layer_id)
    attn = f"{layer}.attn"
    compressor = f"{attn}.compressor"
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden
    result = {
        f"{attn}.attn_sink": ExpectedBlockTensor((heads,), ("F32",)),
        f"{attn}.wq_a.weight": wq_a,
        f"{attn}.wq_a.scale": wq_a_scale,
        f"{attn}.q_norm.weight": ExpectedBlockTensor((q_rank,), ("BF16",)),
        f"{attn}.wq_b.weight": wq_b,
        f"{attn}.wq_b.scale": wq_b_scale,
        f"{attn}.wkv.weight": wkv,
        f"{attn}.wkv.scale": wkv_scale,
        f"{attn}.kv_norm.weight": ExpectedBlockTensor((head_dim,), ("BF16",)),
        f"{attn}.wo_a.weight": wo_a,
        f"{attn}.wo_a.scale": wo_a_scale,
        f"{attn}.wo_b.weight": wo_b,
        f"{attn}.wo_b.scale": wo_b_scale,
        f"{layer}.attn_norm.weight": ExpectedBlockTensor((hidden,), ("BF16",)),
        f"{layer}.ffn_norm.weight": ExpectedBlockTensor((hidden,), ("BF16",)),
        f"{layer}.hc_attn_fn": ExpectedBlockTensor((mix_hc, hc_dim), ("F32",)),
        f"{layer}.hc_attn_base": ExpectedBlockTensor((mix_hc,), ("F32",)),
        f"{layer}.hc_attn_scale": ExpectedBlockTensor((3,), ("F32",)),
        f"{layer}.hc_ffn_fn": ExpectedBlockTensor((mix_hc, hc_dim), ("F32",)),
        f"{layer}.hc_ffn_base": ExpectedBlockTensor((mix_hc,), ("F32",)),
        f"{layer}.hc_ffn_scale": ExpectedBlockTensor((3,), ("F32",)),
        f"{layer}.ffn.gate.weight": ExpectedBlockTensor(
            (experts, hidden), ("BF16",)
        ),
    }
    if compress_ratio:
        compressor_dim = head_dim * (2 if compress_ratio == 4 else 1)
        result.update(
            {
                f"{compressor}.ape": ExpectedBlockTensor(
                    (compress_ratio, compressor_dim), ("F32",)
                ),
                f"{compressor}.wkv.weight": ExpectedBlockTensor(
                    (compressor_dim, hidden), ("BF16",)
                ),
                f"{compressor}.wgate.weight": ExpectedBlockTensor(
                    (compressor_dim, hidden), ("BF16",)
                ),
                f"{compressor}.norm.weight": ExpectedBlockTensor(
                    (head_dim,), ("BF16",)
                ),
            }
        )
    if compress_ratio == 4:
        index_heads = _config_int(config, "index_n_heads")
        index_head_dim = _config_int(config, "index_head_dim")
        indexer = f"{attn}.indexer"
        indexer_compressor = f"{indexer}.compressor"
        index_wq_b, index_wq_b_scale = _fp8_linear_specs(
            out_features=index_heads * index_head_dim,
            in_features=q_rank,
        )
        indexer_compressor_dim = 2 * index_head_dim
        result.update(
            {
                f"{indexer}.wq_b.weight": index_wq_b,
                f"{indexer}.wq_b.scale": index_wq_b_scale,
                f"{indexer}.weights_proj.weight": ExpectedBlockTensor(
                    (index_heads, hidden), ("BF16",)
                ),
                f"{indexer_compressor}.ape": ExpectedBlockTensor(
                    (compress_ratio, indexer_compressor_dim), ("F32",)
                ),
                f"{indexer_compressor}.wkv.weight": ExpectedBlockTensor(
                    (indexer_compressor_dim, hidden), ("BF16",)
                ),
                f"{indexer_compressor}.wgate.weight": ExpectedBlockTensor(
                    (indexer_compressor_dim, hidden), ("BF16",)
                ),
                f"{indexer_compressor}.norm.weight": ExpectedBlockTensor(
                    (index_head_dim,), ("BF16",)
                ),
            }
        )
    if route_kind == "learned":
        result[f"{layer}.ffn.gate.bias"] = ExpectedBlockTensor(
            (experts,), ("F32",)
        )
    else:
        vocab = _config_int(config, "vocab_size")
        topk = _config_int(config, "num_experts_per_tok")
        result[f"{layer}.ffn.gate.tid2eid"] = ExpectedBlockTensor(
            (vocab, topk), ("I64",)
        )
    if bool(SUPPORTED_LAYER_SPECS[layer_id]["is_mtp"]):
        e_proj, e_proj_scale = _fp8_linear_specs(
            out_features=hidden, in_features=hidden
        )
        h_proj, h_proj_scale = _fp8_linear_specs(
            out_features=hidden, in_features=hidden
        )
        result.update(
            {
                f"{layer}.e_proj.weight": e_proj,
                f"{layer}.e_proj.scale": e_proj_scale,
                f"{layer}.h_proj.weight": h_proj,
                f"{layer}.h_proj.scale": h_proj_scale,
                f"{layer}.enorm.weight": ExpectedBlockTensor((hidden,), ("BF16",)),
                f"{layer}.hnorm.weight": ExpectedBlockTensor((hidden,), ("BF16",)),
                f"{layer}.norm.weight": ExpectedBlockTensor((hidden,), ("BF16",)),
                f"{layer}.hc_head_fn": ExpectedBlockTensor(
                    (hc_mult, hc_dim), ("F32",)
                ),
                f"{layer}.hc_head_base": ExpectedBlockTensor((hc_mult,), ("F32",)),
                f"{layer}.hc_head_scale": ExpectedBlockTensor((1,), ("F32",)),
            }
        )
    return result


def validate_replicated_block_contract(
    tensors: Mapping[str, TensorSpec],
    config: Mapping[str, Any],
    *,
    layer_id: int = SUPPORTED_LAYER_ID,
    rank: int,
    world_size: int = SUPPORTED_TP_SIZE,
    strict_production: bool = True,
) -> dict[str, Any]:
    """Validate the header-only ABI; the contract id is identical on all ranks."""

    errors: list[str] = []
    world_size_valid = isinstance(world_size, int) and not isinstance(world_size, bool)
    rank_valid = isinstance(rank, int) and not isinstance(rank, bool)
    if not world_size_valid or world_size != SUPPORTED_TP_SIZE:
        errors.append(f"world_size must be {SUPPORTED_TP_SIZE}, got {world_size}")
    if not rank_valid:
        errors.append(f"rank must be an integer, got {rank!r}")
    elif world_size_valid and (rank < 0 or rank >= world_size):
        errors.append(f"rank {rank} is outside world_size {world_size}")
    normalized_model_contract: dict[str, Any] = {"strict_production": False}
    if strict_production:
        try:
            normalized_model_contract = validate_model_layer_config(
                config, layer_id=layer_id
            )
        except ModelContractError as exc:
            errors.append(str(exc))
    try:
        expected = expected_block_tensor_specs(config, layer_id)
    except CheckpointContractError as exc:
        expected = {}
        errors.append(str(exc))

    compress_ratio: int | None = None
    route_kind: str | None = None
    if expected:
        compress_ratio = _layer_compress_ratio(config, layer_id)
        route_kind = _layer_route_kind(config, layer_id)
    prefix = layer_prefix(layer_id)
    gate_prefix = f"{prefix}.ffn.gate"
    forbidden_gate_tensor = None
    if route_kind == "hash":
        forbidden_gate_tensor = f"{gate_prefix}.bias"
    elif route_kind == "learned":
        forbidden_gate_tensor = f"{gate_prefix}.tid2eid"
    if forbidden_gate_tensor is not None and forbidden_gate_tensor in tensors:
        errors.append(
            f"{forbidden_gate_tensor}: forbidden for layer {layer_id} routing contract"
        )
    forbidden_prefixes: list[tuple[str, str]] = []
    if compress_ratio == 128:
        forbidden_prefixes.append(
            (f"{prefix}.attn.indexer.", "ratio-128")
        )
    elif compress_ratio == 0:
        forbidden_prefixes.append(
            (f"{prefix}.attn.indexer.", "sliding-window")
        )
        forbidden_prefixes.append(
            (f"{prefix}.attn.compressor.", "sliding-window")
        )
    for prefix, kind in forbidden_prefixes:
        for name in sorted(tensors):
            if name.startswith(prefix):
                errors.append(
                    f"{name}: forbidden for layer {layer_id} {kind} contract"
                )

    tensor_results: dict[str, Any] = {}
    for name, requirement in expected.items():
        actual = tensors.get(name)
        item_errors: list[str] = []
        if actual is None:
            item_errors.append("missing")
        else:
            if actual.shape != requirement.shape:
                item_errors.append(
                    f"shape {list(actual.shape)} != {list(requirement.shape)}"
                )
            if actual.dtype not in requirement.dtypes:
                item_errors.append(
                    f"dtype {actual.dtype} not in {list(requirement.dtypes)}"
                )
        if item_errors:
            errors.extend(f"{name}: {message}" for message in item_errors)
        tensor_results[name] = {
            "shape": list(requirement.shape),
            "dtypes": list(requirement.dtypes),
            "expected_nbytes": requirement.nbytes,
            "actual_dtype": actual.dtype if actual is not None else None,
            "actual_nbytes": actual.nbytes if actual is not None else None,
            "residency": "replicated",
            "logical_slice": "full",
            "errors": item_errors,
        }

    identity = {
        "layer_id": layer_id,
        "world_size": world_size,
        "residency": "replicated",
        "model_contract": normalized_model_contract,
        "tensors": {
            name: {"shape": spec.shape, "dtypes": spec.dtypes}
            for name, spec in expected.items()
        },
    }
    contract_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "layer_id": layer_id,
        "rank": rank,
        "world_size": world_size,
        "residency": "replicated",
        "rank_invariant_contract": True,
        "strict_production": strict_production,
        "compress_ratio": compress_ratio,
        "route_kind": route_kind,
        "model_contract": normalized_model_contract,
        "contract_id": contract_id,
        "expected_tensor_count": len(expected),
        "expected_resident_bytes": sum(item.nbytes for item in expected.values()),
        "tensors": tensor_results,
        "ok": not errors,
        "errors": errors,
    }


def inspect_replicated_block_contract(
    stage_root: Path,
    *,
    layer_id: int = SUPPORTED_LAYER_ID,
    rank: int,
    world_size: int = SUPPORTED_TP_SIZE,
    strict_production: bool = True,
) -> dict[str, Any]:
    """Read config and shard header without loading any tensor payload."""

    stage_root = stage_root.expanduser().resolve()
    with (stage_root / "config.json").open("rb") as handle:
        config = json.load(handle)
    # Resolve this layer's shard set through the index weight_map (the only
    # shard-resolution mechanism in the Flash runtime); merge the headers of
    # every involved file, filtered to this layer's namespace.
    weight_map, _ = load_weight_map(stage_root)
    key_prefix = layer_prefix(layer_id) + "."
    files = sorted(
        {filename for key, filename in weight_map.items() if key.startswith(key_prefix)}
    )
    if not files:
        raise CheckpointContractError(
            f"index weight_map holds no tensors for layer {layer_id}"
        )
    tensors: dict[str, TensorSpec] = {}
    file_metadata = []
    for filename in files:
        shard_tensors, shard_metadata = read_safetensors_header(stage_root / filename)
        for name, spec in shard_tensors.items():
            if name.startswith(key_prefix):
                tensors[name] = spec
        file_metadata.append(shard_metadata)
    result = validate_replicated_block_contract(
        tensors,
        config,
        layer_id=layer_id,
        rank=rank,
        world_size=world_size,
        strict_production=strict_production,
    )
    result["stage_root"] = str(stage_root)
    result["files"] = file_metadata
    return result


def load_replicated_block_weights(
    *,
    stage_root: Path,
    rank: int,
    world_size: int = SUPPORTED_TP_SIZE,
    layer_id: int = SUPPORTED_LAYER_ID,
    device: torch.device | str = "cpu",
    checkpoint_id: str | None = None,
    strict_production: bool = True,
) -> ResidentBlockWeights:
    """Load the validated replicated block tensors without dequantization."""

    if strict_production and (
        not isinstance(checkpoint_id, str)
        or len(checkpoint_id) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_id)
    ):
        raise CheckpointContractError(
            "strict production block loads require a lowercase SHA-256 checkpoint_id"
        )

    contract = inspect_replicated_block_contract(
        stage_root,
        layer_id=layer_id,
        rank=rank,
        world_size=world_size,
        strict_production=strict_production,
    )
    if not contract["ok"]:
        raise CheckpointContractError(
            "replicated block checkpoint contract failed: "
            + "; ".join(contract["errors"])
        )

    from .ops.marlin_moe import ShardReader

    target = torch.device(device)
    weight_map, _ = load_weight_map(Path(stage_root).expanduser().resolve())
    layer = layer_prefix(layer_id)
    attn = f"{layer}.attn"
    compressor = f"{attn}.compressor"
    is_mtp = bool(SUPPORTED_LAYER_SPECS[layer_id]["is_mtp"])
    started = time.perf_counter()
    with ShardReader(Path(stage_root).expanduser().resolve(), weight_map) as reader:
        def get(name: str) -> torch.Tensor:
            return reader.get_tensor(name).to(device=target, non_blocking=False).contiguous()

        def get_compressor(prefix: str) -> ResidentCompressorWeights:
            return ResidentCompressorWeights(
                ape=get(f"{prefix}.ape"),
                wkv=get(f"{prefix}.wkv.weight"),
                wgate=get(f"{prefix}.wgate.weight"),
                norm=get(f"{prefix}.norm.weight"),
            )

        indexer = None
        if contract["compress_ratio"] == 4:
            indexer_prefix = f"{attn}.indexer"
            indexer = ResidentIndexerWeights(
                wq_b=QuantizedLinearWeights(
                    get(f"{indexer_prefix}.wq_b.weight"),
                    get(f"{indexer_prefix}.wq_b.scale"),
                ),
                weights_proj=get(f"{indexer_prefix}.weights_proj.weight"),
                compressor=get_compressor(f"{indexer_prefix}.compressor"),
            )

        attention = ResidentAttentionWeights(
            attn_sink=get(f"{attn}.attn_sink"),
            wq_a=QuantizedLinearWeights(
                get(f"{attn}.wq_a.weight"), get(f"{attn}.wq_a.scale")
            ),
            q_norm=get(f"{attn}.q_norm.weight"),
            wq_b=QuantizedLinearWeights(
                get(f"{attn}.wq_b.weight"), get(f"{attn}.wq_b.scale")
            ),
            wkv=QuantizedLinearWeights(
                get(f"{attn}.wkv.weight"), get(f"{attn}.wkv.scale")
            ),
            kv_norm=get(f"{attn}.kv_norm.weight"),
            wo_a=QuantizedLinearWeights(
                get(f"{attn}.wo_a.weight"), get(f"{attn}.wo_a.scale")
            ),
            wo_b=QuantizedLinearWeights(
                get(f"{attn}.wo_b.weight"), get(f"{attn}.wo_b.scale")
            ),
            compressor=(
                get_compressor(compressor) if contract["compress_ratio"] else None
            ),
            indexer=indexer,
            layer_id=layer_id,
            rank=rank,
            world_size=world_size,
            checkpoint_id=checkpoint_id,
        )
        result = ResidentBlockWeights(
            layer_id=layer_id,
            rank=rank,
            world_size=world_size,
            checkpoint_id=checkpoint_id,
            contract_id=contract["contract_id"],
            attention=attention,
            attn_norm=get(f"{layer}.attn_norm.weight"),
            ffn_norm=get(f"{layer}.ffn_norm.weight"),
            hyper_connection=ResidentHyperConnectionWeights(
                attn_fn=get(f"{layer}.hc_attn_fn"),
                attn_base=get(f"{layer}.hc_attn_base"),
                attn_scale=get(f"{layer}.hc_attn_scale"),
                ffn_fn=get(f"{layer}.hc_ffn_fn"),
                ffn_base=get(f"{layer}.hc_ffn_base"),
                ffn_scale=get(f"{layer}.hc_ffn_scale"),
            ),
            gate=ResidentGateWeights(
                weight=get(f"{layer}.ffn.gate.weight"),
                bias=(
                    get(f"{layer}.ffn.gate.bias")
                    if contract["route_kind"] == "learned"
                    else None
                ),
                layer_id=layer_id,
                rank=rank,
                world_size=world_size,
                checkpoint_id=checkpoint_id,
                tid2eid=(
                    get(f"{layer}.ffn.gate.tid2eid")
                    if contract["route_kind"] == "hash"
                    else None
                ),
            ),
            load_seconds=0.0,
            mtp=(
                ResidentMTPWeights(
                    e_proj=QuantizedLinearWeights(
                        get(f"{layer}.e_proj.weight"), get(f"{layer}.e_proj.scale")
                    ),
                    h_proj=QuantizedLinearWeights(
                        get(f"{layer}.h_proj.weight"), get(f"{layer}.h_proj.scale")
                    ),
                    enorm=get(f"{layer}.enorm.weight"),
                    hnorm=get(f"{layer}.hnorm.weight"),
                    norm=get(f"{layer}.norm.weight"),
                    hc_head_fn=get(f"{layer}.hc_head_fn"),
                    hc_head_base=get(f"{layer}.hc_head_base"),
                    hc_head_scale=get(f"{layer}.hc_head_scale"),
                )
                if is_mtp
                else None
            ),
        )
    result.load_seconds = time.perf_counter() - started
    if result.resident_bytes != contract["expected_resident_bytes"]:
        raise CheckpointContractError(
            f"resident bytes {result.resident_bytes} do not match contract "
            f"{contract['expected_resident_bytes']}"
        )
    return result


__all__ = [
    "ExpectedBlockTensor",
    "QuantizedLinearWeights",
    "ResidentAttentionWeights",
    "ResidentBlockWeights",
    "ResidentCompressorWeights",
    "ResidentGateWeights",
    "ResidentHyperConnectionWeights",
    "ResidentIndexerWeights",
    "ResidentMTPWeights",
    "ResidentRatio128CompressorWeights",
    "SUPPORTED_COMPRESS_RATIOS",
    "SUPPORTED_LAYER_ID",
    "SUPPORTED_LAYER_IDS",
    "SUPPORTED_TP_SIZE",
    "expected_block_tensor_specs",
    "inspect_replicated_block_contract",
    "load_replicated_block_weights",
    "validate_replicated_block_contract",
]
