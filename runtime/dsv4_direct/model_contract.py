"""Fail-closed frozen DeepSeek-V4-Flash layer contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping


class ModelContractError(ValueError):
    pass


# Base ABI shared by every Flash block, frozen from the checkpoint's
# config.json (titan064:~/Workspace/DeepSeek-V4-Flash/config.json).
EXPECTED_RATIO128_CONFIG: dict[str, Any] = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "num_attention_heads": 64,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "sliding_window": 128,
    "rms_norm_eps": 1e-6,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "num_hash_layers": 3,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.5,  # Flash: 1.5 (Pro used 2.5)
    "swiglu_limit": 10.0,
    "norm_topk_prob": True,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "expert_dtype": "fp4",
    "compress_rope_theta": 160000,
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
}


# Ratio-4 layers carry the sparse indexer, so they additionally freeze the
# indexer geometry (index_topk is 512 in Flash, not Pro's 1024).
EXPECTED_RATIO4_CONFIG: dict[str, Any] = {
    **EXPECTED_RATIO128_CONFIG,
    "vocab_size": 129280,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 512,
}


# L0/L1 are pure sliding-window blocks (ratio 0: no compressor, no indexer)
# with the hash-router table, which needs the vocab size for tid2eid.
EXPECTED_WINDOW_HASH_CONFIG: dict[str, Any] = {
    **EXPECTED_RATIO128_CONFIG,
    "vocab_size": 129280,
}


# mtp.0 is a sliding-window learned-router block with its own head, so it
# needs the vocab size for the shared embedding/head projection.
EXPECTED_MTP_CONFIG: dict[str, Any] = {
    **EXPECTED_RATIO128_CONFIG,
    "vocab_size": 129280,
}


# Freeze the 43 block identities plus the MTP block from the checkpoint
# config.json.  Flash compress_ratios rule: L0/L1 are 0 (pure sliding
# window), even layers >=2 are 4 (compressor + indexer), odd layers are 128
# (compressor only), and the trailing entry -- compress_ratios[43] -- is the
# MTP block at ratio 0.
MODEL_LAYER_COUNT = 43
MTP_LAYER_ID = MODEL_LAYER_COUNT  # mtp.0 occupies compress_ratios[43]
MODEL_LAYER_IDS = tuple(range(MODEL_LAYER_COUNT))
FROZEN_COMPRESS_RATIOS = tuple(
    0 if layer_id < 2 else (4 if layer_id % 2 == 0 else 128)
    for layer_id in MODEL_LAYER_IDS
) + (0,)


def _attn_kind(compress_ratio: int) -> str:
    if compress_ratio == 0:
        return "window"
    if compress_ratio == 4:
        return "ratio4"
    return "ratio128"


SUPPORTED_LAYER_SPECS: dict[int, dict[str, Any]] = {
    layer_id: {
        "compress_ratio": FROZEN_COMPRESS_RATIOS[layer_id],
        "route_kind": "hash" if layer_id < 3 else "learned",
        "attn_kind": _attn_kind(FROZEN_COMPRESS_RATIOS[layer_id]),
        "is_mtp": layer_id == MTP_LAYER_ID,
    }
    for layer_id in MODEL_LAYER_IDS + (MTP_LAYER_ID,)
}


EXPECTED_QUANTIZATION_CONFIG: dict[str, Any] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}


EXPECTED_ROPE_SCALING: dict[str, Any] = {
    "factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 65536,
    "type": "yarn",
}


def _validate_model_config(
    config: Mapping[str, Any],
    *,
    layer_id: int,
    expected_layer_id: int,
    expected_ratio: int,
    expected_route_kind: str,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    errors = []
    if not isinstance(layer_id, int) or isinstance(layer_id, bool):
        errors.append(f"layer_id must be an integer, got {layer_id!r}")
    elif layer_id != expected_layer_id:
        errors.append(f"layer_id={layer_id} != {expected_layer_id}")
    for name, expected in expected_config.items():
        observed = config.get(name)
        if observed != expected:
            errors.append(f"{name}={observed!r} != {expected!r}")

    ratios = config.get("compress_ratios")
    if not isinstance(layer_id, int) or isinstance(layer_id, bool):
        errors.append("compress_ratios cannot be indexed by a non-integer layer_id")
    elif not isinstance(ratios, (list, tuple)) or len(ratios) <= layer_id:
        errors.append(f"compress_ratios does not cover layer {layer_id}")
    elif ratios[layer_id] != expected_ratio:
        errors.append(
            f"compress_ratios[{layer_id}]={ratios[layer_id]!r} != {expected_ratio}"
        )

    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        errors.append("quantization_config is missing")
    else:
        for name, expected in EXPECTED_QUANTIZATION_CONFIG.items():
            if quant.get(name) != expected:
                errors.append(
                    f"quantization_config.{name}={quant.get(name)!r} != {expected!r}"
                )

    rope = config.get("rope_scaling")
    if not isinstance(rope, Mapping):
        errors.append("rope_scaling is missing")
    else:
        for name, expected in EXPECTED_ROPE_SCALING.items():
            if rope.get(name) != expected:
                errors.append(
                    f"rope_scaling.{name}={rope.get(name)!r} != {expected!r}"
                )

    try:
        hash_layers = int(config.get("num_hash_layers", -1))
    except (TypeError, ValueError):
        hash_layers = -1
    if isinstance(layer_id, int) and not isinstance(layer_id, bool):
        if expected_route_kind == "hash" and layer_id >= hash_layers:
            errors.append(f"layer {layer_id} is not a hash-router layer")
        elif expected_route_kind == "learned" and layer_id < hash_layers:
            errors.append(f"layer {layer_id} is not a learned-router layer")

    for name in ("rms_norm_eps", "hc_eps"):
        value = config.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append(f"{name} must be finite and positive")
    if errors:
        raise ModelContractError(
            f"unsupported layer-{expected_layer_id} model config: " + "; ".join(errors)
        )
    return {
        **expected_config,
        "layer_id": layer_id,
        "compress_ratio": expected_ratio,
        "quantization": {
            **EXPECTED_QUANTIZATION_CONFIG,
            "weight_block_size": list(
                EXPECTED_QUANTIZATION_CONFIG["weight_block_size"]
            ),
        },
        "rope_scaling": dict(EXPECTED_ROPE_SCALING),
    }


def model_layer_spec(layer_id: int) -> dict[str, Any]:
    """Return an owning copy of one frozen physical-layer specification."""

    if not isinstance(layer_id, int) or isinstance(layer_id, bool):
        raise ModelContractError(f"layer_id must be an integer, got {layer_id!r}")
    specification = SUPPORTED_LAYER_SPECS.get(layer_id)
    if specification is None:
        raise ModelContractError(
            f"layer_id={layer_id} is outside frozen model layers "
            f"[0, {MODEL_LAYER_COUNT}) + mtp ({MTP_LAYER_ID})"
        )
    return dict(specification)


def validate_model_layer_config(
    config: Mapping[str, Any], *, layer_id: int
) -> dict[str, Any]:
    """Validate any of the 43+MTP frozen physical blocks against its ABI."""

    specification = model_layer_spec(layer_id)
    compress_ratio = int(specification["compress_ratio"])
    route_kind = str(specification["route_kind"])
    if specification["is_mtp"]:
        expected_config = EXPECTED_MTP_CONFIG
    elif compress_ratio == 4:
        expected_config = EXPECTED_RATIO4_CONFIG
    elif route_kind == "hash":
        # L0/L1: pure sliding-window (ratio 0) + hash router.
        expected_config = EXPECTED_WINDOW_HASH_CONFIG
    else:
        expected_config = EXPECTED_RATIO128_CONFIG
    return _validate_model_config(
        config,
        layer_id=layer_id,
        expected_layer_id=layer_id,
        expected_ratio=compress_ratio,
        expected_route_kind=route_kind,
        expected_config=expected_config,
    )


__all__ = [
    "EXPECTED_MTP_CONFIG",
    "EXPECTED_QUANTIZATION_CONFIG",
    "EXPECTED_RATIO128_CONFIG",
    "EXPECTED_RATIO4_CONFIG",
    "EXPECTED_ROPE_SCALING",
    "EXPECTED_WINDOW_HASH_CONFIG",
    "FROZEN_COMPRESS_RATIOS",
    "MODEL_LAYER_COUNT",
    "MODEL_LAYER_IDS",
    "MTP_LAYER_ID",
    "ModelContractError",
    "SUPPORTED_LAYER_SPECS",
    "model_layer_spec",
    "validate_model_layer_config",
]
