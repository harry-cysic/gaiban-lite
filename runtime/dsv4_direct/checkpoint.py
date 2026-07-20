"""Checkpoint metadata and TP slicing contracts for the Flash direct runtime.

This module deliberately parses safetensors headers without importing torch.  It
lets the first checkpoint gate run on a busy cluster node without touching CUDA.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .model_contract import (
    MTP_LAYER_ID,
    ModelContractError,
    model_layer_spec,
    validate_model_layer_config,
)


DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E8M0": 1,
    "F8_E8M0FNU": 1,
    "I8": 1,
    "I64": 8,
}

FP8_DTYPES = ("F8_E4M3", "F8_E4M3FN")
SCALE_DTYPES = ("F8_E8M0", "F8_E8M0FNU")


class CheckpointContractError(ValueError):
    """Raised when a checkpoint cannot satisfy the direct-runtime ABI."""


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_safetensors_header(path: Path) -> tuple[dict[str, TensorSpec], dict[str, Any]]:
    """Read and validate one safetensors header without reading tensor payloads."""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise CheckpointContractError(f"{path}: missing 8-byte header length")
        (header_length,) = struct.unpack("<Q", raw_length)
        if header_length <= 0 or header_length > file_size - 8:
            raise CheckpointContractError(
                f"{path}: invalid header length {header_length} for {file_size}-byte file"
            )
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise CheckpointContractError(f"{path}: truncated header")

    try:
        document = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise CheckpointContractError(f"{path}: invalid header JSON: {exc}") from exc

    tensors: dict[str, TensorSpec] = {}
    max_end = 0
    for name, raw in document.items():
        if name == "__metadata__":
            continue
        try:
            dtype = str(raw["dtype"])
            shape = tuple(int(v) for v in raw["shape"])
            offsets = tuple(int(v) for v in raw["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointContractError(f"{path}: malformed tensor entry {name}") from exc
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
            raise CheckpointContractError(f"{path}: invalid offsets for {name}: {offsets}")
        if offsets[1] > file_size - 8 - header_length:
            raise CheckpointContractError(f"{path}: payload for {name} extends beyond EOF")
        item = TensorSpec(name=name, dtype=dtype, shape=shape, data_offsets=offsets)
        element_size = DTYPE_BYTES.get(dtype)
        if element_size is None:
            raise CheckpointContractError(f"{path}: unsupported dtype {dtype} for {name}")
        expected_bytes = _product(shape) * element_size
        if item.nbytes != expected_bytes:
            raise CheckpointContractError(
                f"{path}: byte count mismatch for {name}: {item.nbytes} != {expected_bytes}"
            )
        tensors[name] = item
        max_end = max(max_end, offsets[1])

    if max_end != file_size - 8 - header_length:
        raise CheckpointContractError(
            f"{path}: payload end {max_end} does not match file payload size "
            f"{file_size - 8 - header_length}"
        )
    return tensors, {
        "path": str(path),
        "file_size": file_size,
        "header_bytes": header_length,
        "header_sha256": _sha256_bytes(raw_header),
        "tensor_count": len(tensors),
    }


def tp_intermediate_slices(intermediate_size: int, tp_size: int) -> list[dict[str, Any]]:
    """Return logical and packed checkpoint slices for intermediate TP.

    Flash: moe_intermediate_size=2048, so TP4 gives 512 rows per rank, which
    preserves both the 128-element FP8 scale blocks and the fp4 pack/scale
    granularity (512/2 and 512/32 are integral).
    """

    if tp_size <= 0 or intermediate_size % tp_size:
        raise CheckpointContractError(
            f"intermediate_size={intermediate_size} must be divisible by tp_size={tp_size}"
        )
    local = intermediate_size // tp_size
    if local % 128:
        raise CheckpointContractError(
            f"local intermediate size {local} must preserve 128-element FP8 blocks"
        )
    result = []
    for rank in range(tp_size):
        start = rank * local
        end = start + local
        result.append(
            {
                "rank": rank,
                "logical_intermediate": [start, end],
                "routed": {
                    "w1_w3_rows": [start, end],
                    "w2_packed_bytes": [start // 2, end // 2],
                    "w2_scale_columns": [start // 32, end // 32],
                },
                "shared": {
                    "w1_w3_rows": [start, end],
                    "w1_w3_scale_rows": [start // 128, end // 128],
                    "w2_columns": [start, end],
                    "w2_scale_columns": [start // 128, end // 128],
                },
            }
        )
    return result


def _expect(
    tensors: dict[str, TensorSpec],
    name: str,
    shape: tuple[int, ...],
    dtypes: tuple[str, ...],
    errors: list[str],
) -> None:
    item = tensors.get(name)
    if item is None:
        errors.append(f"missing tensor: {name}")
        return
    if item.shape != shape:
        errors.append(f"bad shape: {name}: {list(item.shape)} != {list(shape)}")
    if item.dtype not in dtypes:
        errors.append(f"bad dtype: {name}: {item.dtype} not in {list(dtypes)}")


def layer_prefix(layer_id: int) -> str:
    """Checkpoint key prefix for a physical block (mtp.0 for the MTP block)."""

    return "mtp.0" if layer_id == MTP_LAYER_ID else f"layers.{layer_id}"


def _expected_layer_tensors(
    config: dict[str, Any], layer_id: int
) -> tuple[list[tuple[str, tuple[int, ...], tuple[str, ...]]], dict[str, Any]]:
    """Build the exact tensor table for one Flash block.

    Flash layer taxonomy (verified against the titan064 checkpoint):
      - window  (ratio 0): L0/L1 and mtp.0 -- no compressor, no indexer.
      - ratio4  (even >= 2): compressor + sparse indexer.
      - ratio128 (odd >= 3): compressor only.
      - route: hash (tid2eid, layer < num_hash_layers) vs learned (gate bias).
    """

    spec = model_layer_spec(layer_id)
    ratio = int(spec["compress_ratio"])
    route_kind = str(spec["route_kind"])
    attn_kind = str(spec["attn_kind"])
    is_mtp = bool(spec["is_mtp"])

    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    experts = int(config["n_routed_experts"])
    topk = int(config["num_experts_per_tok"])
    vocab = int(config["vocab_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    q_lora = int(config["q_lora_rank"])
    o_lora = int(config["o_lora_rank"])
    o_groups = int(config["o_groups"])
    hc_mult = int(config["hc_mult"])

    prefix = layer_prefix(layer_id)
    expected: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []

    def add(name: str, shape: tuple[int, ...], *dtypes: str) -> None:
        expected.append((f"{prefix}.{name}", shape, dtypes))

    def add_fp8(name: str, rows: int, cols: int) -> None:
        # FP8 E4M3 weight + E8M0 128x128-block scale; all Flash projection
        # dims are multiples of 128 so the scale grid is exact.
        add(f"{name}.weight", (rows, cols), *FP8_DTYPES)
        add(f"{name}.scale", (rows // 128, cols // 128), *SCALE_DTYPES)

    # -- attention core (identical across all Flash layer kinds) --
    add("attn.attn_sink", (heads,), "F32")
    add("attn.q_norm.weight", (q_lora,), "BF16")
    add("attn.kv_norm.weight", (head_dim,), "BF16")  # single 512-dim kv stream
    add_fp8("attn.wq_a", q_lora, hidden)  # [1024, 4096]
    add_fp8("attn.wq_b", heads * head_dim, q_lora)  # [32768, 1024]
    add_fp8("attn.wkv", head_dim, hidden)  # [512, 4096]
    add_fp8("attn.wo_a", o_groups * o_lora, hidden)  # [8192, 4096]
    add_fp8("attn.wo_b", hidden, o_groups * o_lora)  # [4096, 8192]
    add("attn_norm.weight", (hidden,), "BF16")
    add("ffn_norm.weight", (hidden,), "BF16")

    # -- compressor / indexer, by layer kind --
    if attn_kind != "window":
        # ratio-4 compressor emits a gated 2x512 kv stream (rows 1024);
        # ratio-128 emits the plain 512 kv stream.  Both normalize at 512.
        comp_dim = 2 * head_dim if ratio == 4 else head_dim
        add("attn.compressor.ape", (ratio, comp_dim), "F32")
        add("attn.compressor.norm.weight", (head_dim,), "BF16")
        add("attn.compressor.wgate.weight", (comp_dim, hidden), "BF16")
        add("attn.compressor.wkv.weight", (comp_dim, hidden), "BF16")
    if attn_kind == "ratio4":
        index_heads = int(config["index_n_heads"])
        index_head_dim = int(config["index_head_dim"])
        index_comp_dim = 2 * index_head_dim  # gated 2x128 indexer kv stream
        add_fp8("attn.indexer.wq_b", index_heads * index_head_dim, q_lora)
        add("attn.indexer.weights_proj.weight", (index_heads, hidden), "BF16")
        # Indexer compressor is frozen at ratio 4 alongside the main one.
        add("attn.indexer.compressor.ape", (4, index_comp_dim), "F32")
        add("attn.indexer.compressor.norm.weight", (index_head_dim,), "BF16")
        add("attn.indexer.compressor.wgate.weight", (index_comp_dim, hidden), "BF16")
        add("attn.indexer.compressor.wkv.weight", (index_comp_dim, hidden), "BF16")

    # -- router --
    add("ffn.gate.weight", (experts, hidden), "BF16")
    if route_kind == "hash":
        # L0-L2: static token-id -> expert table, no learned bias.
        add("ffn.gate.tid2eid", (vocab, topk), "I64")
    else:
        # L3+ and mtp.0: learned noaux_tc bias, no tid2eid.
        add("ffn.gate.bias", (experts,), "F32")

    # -- shared expert: FP8 E4M3 + E8M0 128x128-block scales --
    for projection in ("w1", "w3"):
        add_fp8(f"ffn.shared_experts.{projection}", intermediate, hidden)
    add_fp8("ffn.shared_experts.w2", hidden, intermediate)

    # -- routed experts: fp4 packed into I8 (2 nibbles/byte) + E8M0 scales
    #    on 32-element input blocks --
    for expert_id in range(experts):
        expert = f"ffn.experts.{expert_id}"
        for projection in ("w1", "w3"):
            add(f"{expert}.{projection}.weight", (intermediate, hidden // 2), "I8")
            add(
                f"{expert}.{projection}.scale",
                (intermediate, hidden // 32),
                *SCALE_DTYPES,
            )
        add(f"{expert}.w2.weight", (hidden, intermediate // 2), "I8")
        add(f"{expert}.w2.scale", (hidden, intermediate // 32), *SCALE_DTYPES)

    # -- hyper-connections: mult 4 -> 24 = hc_mult*(hc_mult+2) mixing rows
    #    over hc_mult*hidden = 16384 flattened stream features --
    hc_rows = hc_mult * (hc_mult + 2)
    hc_cols = hc_mult * hidden
    for tag in ("attn", "ffn"):
        add(f"hc_{tag}_fn", (hc_rows, hc_cols), "F32")
        add(f"hc_{tag}_base", (hc_rows,), "F32")
        add(f"hc_{tag}_scale", (3,), "F32")

    # -- MTP block extras: bridge projections plus its own norms and
    #    hc_head (stream -> single-hidden collapse before the shared head) --
    if is_mtp:
        add_fp8("e_proj", hidden, hidden)
        add_fp8("h_proj", hidden, hidden)
        add("enorm.weight", (hidden,), "BF16")
        add("hnorm.weight", (hidden,), "BF16")
        add("norm.weight", (hidden,), "BF16")
        add("hc_head_fn", (hc_mult, hc_cols), "F32")
        add("hc_head_base", (hc_mult,), "F32")
        add("hc_head_scale", (1,), "F32")

    summary = {
        "layer_id": layer_id,
        "prefix": prefix,
        "attn_kind": attn_kind,
        "route_kind": route_kind,
        "compress_ratio": ratio,
        "is_mtp": is_mtp,
    }
    return expected, summary


def validate_layer_contract(
    tensors: dict[str, TensorSpec], config: dict[str, Any], layer_id: int
) -> dict[str, Any]:
    """Validate the exact tensor layout of one Flash block for the TP4 loader.

    The expected key set is closed: any tensor under the layer prefix that is
    not in the frozen table is reported, so hash layers cannot carry a gate
    bias, window layers cannot carry a compressor, and so on.
    """

    expected, summary = _expected_layer_tensors(config, layer_id)
    errors: list[str] = []
    for name, shape, dtypes in expected:
        _expect(tensors, name, shape, dtypes, errors)

    prefix = summary["prefix"] + "."
    expected_names = {name for name, _, _ in expected}
    observed_names = {name for name in tensors if name.startswith(prefix)}
    for name in sorted(observed_names - expected_names):
        errors.append(f"unexpected tensor: {name}")

    return {
        **summary,
        "expected_tensor_count": len(expected),
        "observed_tensor_count": len(observed_names),
        "errors": errors,
        "ok": not errors,
    }


# Top-level (non-layer) tensors: shared embedding/head plus the final norm
# and the hc_head stream-collapse parameters.
def _expected_top_level_tensors(
    config: dict[str, Any],
) -> list[tuple[str, tuple[int, ...], tuple[str, ...]]]:
    hidden = int(config["hidden_size"])
    vocab = int(config["vocab_size"])
    hc_mult = int(config["hc_mult"])
    return [
        ("embed.weight", (vocab, hidden), ("BF16",)),
        ("head.weight", (vocab, hidden), ("BF16",)),
        ("norm.weight", (hidden,), ("BF16",)),
        ("hc_head_fn", (hc_mult, hc_mult * hidden), ("F32",)),
        ("hc_head_base", (hc_mult,), ("F32",)),
        ("hc_head_scale", (1,), ("F32",)),
    ]


def _config_contract(config: dict[str, Any], tp_size: int) -> dict[str, Any]:
    # Flash geometry: hidden 4096, moe intermediate 2048, 256 routed experts,
    # topk 6, 1 shared expert, 3 hash layers, 43 layers + 1 MTP block.
    required = {
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "num_hash_layers": 3,
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "vocab_size": 129280,
    }
    errors = []
    for name, expected in required.items():
        actual = config.get(name)
        if actual != expected:
            errors.append(f"config {name}: {actual!r} != {expected!r}")
    try:
        slices = tp_intermediate_slices(int(config["moe_intermediate_size"]), tp_size)
    except (KeyError, TypeError, ValueError, CheckpointContractError) as exc:
        slices = []
        errors.append(str(exc))
    return {"ok": not errors, "errors": errors, "tp_slices": slices}


def load_weight_map(stage_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Load model.safetensors.index.json and return (weight_map, meta).

    This is the only shard-resolution mechanism in the Flash runtime: tensors
    are located through the index weight_map, never through assumed
    layer -> shard numbering.
    """

    index_path = stage_root / "model.safetensors.index.json"
    with index_path.open("rb") as handle:
        raw_index = handle.read()
    document = json.loads(raw_index)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointContractError(f"{index_path}: missing or empty weight_map")
    for key, value in weight_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CheckpointContractError(
                f"{index_path}: malformed weight_map entry {key!r}: {value!r}"
            )
    return dict(weight_map), {
        "path": str(index_path),
        "index_sha256": _sha256_bytes(raw_index),
        "weight_map_keys": len(weight_map),
    }


def inspect_stage_checkpoint(
    stage_root: Path,
    layer_ids: Iterable[int] = (0, 1, 2, 3, 4, 42, MTP_LAYER_ID),
    tp_size: int = 4,
) -> dict[str, Any]:
    """Inspect a Flash checkpoint and return a JSON-serializable gate result.

    Shards are resolved through the safetensors index weight_map (no assumed
    layer -> shard numbering), and each loaded shard header is cross-checked
    against the index for exact key agreement.
    """

    stage_root = stage_root.expanduser().resolve()
    config_path = stage_root / "config.json"
    with config_path.open("rb") as handle:
        raw_config = handle.read()
    config = json.loads(raw_config)
    config_result = _config_contract(config, tp_size)
    errors = list(config_result["errors"])

    weight_map, index_result = load_weight_map(stage_root)
    file_to_keys: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        file_to_keys.setdefault(filename, set()).add(key)

    shard_cache: dict[str, dict[str, TensorSpec]] = {}
    files: list[dict[str, Any]] = []

    def load_shard(filename: str) -> dict[str, TensorSpec]:
        cached = shard_cache.get(filename)
        if cached is not None:
            return cached
        tensors, file_result = read_safetensors_header(stage_root / filename)
        indexed = file_to_keys.get(filename, set())
        if set(tensors) != indexed:
            missing = sorted(indexed - set(tensors))[:4]
            extra = sorted(set(tensors) - indexed)[:4]
            errors.append(
                f"{filename}: header/index key mismatch "
                f"(index-only sample: {missing}; header-only sample: {extra})"
            )
        shard_cache[filename] = tensors
        files.append(file_result)
        return tensors

    layers = []
    for layer_id in layer_ids:
        layer_id = int(layer_id)
        prefix = layer_prefix(layer_id) + "."
        layer_keys = [key for key in weight_map if key.startswith(prefix)]
        if not layer_keys:
            errors.append(f"index has no tensors under {prefix}*")
            layers.append(
                {"layer_id": layer_id, "prefix": prefix.rstrip("."), "ok": False,
                 "errors": [f"index has no tensors under {prefix}*"]}
            )
            continue
        tensors: dict[str, TensorSpec] = {}
        shard_names = sorted({weight_map[key] for key in layer_keys})
        for filename in shard_names:
            tensors.update(load_shard(filename))
        layer_result = validate_layer_contract(tensors, config, layer_id)
        layer_result["shards"] = shard_names
        try:
            validate_model_layer_config(config, layer_id=layer_id)
            layer_result["model_config_ok"] = True
        except ModelContractError as exc:
            layer_result["model_config_ok"] = False
            layer_result["errors"].append(str(exc))
            layer_result["ok"] = False
        layers.append(layer_result)
        errors.extend(layer_result["errors"])

    # Top-level tensors are validated unconditionally: exact set, shapes,
    # dtypes (they live outside layers.*/mtp.* and span two Flash shards).
    top_errors: list[str] = []
    top_expected = _expected_top_level_tensors(config)
    top_observed = {
        key
        for key in weight_map
        if not key.startswith("layers.") and not key.startswith("mtp.")
    }
    top_tensors: dict[str, TensorSpec] = {}
    for filename in sorted({weight_map[key] for key in top_observed}):
        top_tensors.update(load_shard(filename))
    for name, shape, dtypes in top_expected:
        _expect(top_tensors, name, shape, dtypes, top_errors)
    for name in sorted(top_observed - {name for name, _, _ in top_expected}):
        top_errors.append(f"unexpected top-level tensor: {name}")
    errors.extend(top_errors)

    checkpoint_id_material = json.dumps(
        {
            "config_sha256": _sha256_bytes(raw_config),
            "index_sha256": index_result["index_sha256"],
            "files": sorted(
                (Path(item["path"]).name, item["file_size"], item["header_sha256"])
                for item in files
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema_version": 1,
        "experiment": "e0f-flash-checkpoint-contract",
        "ok": not errors,
        "stage_root": str(stage_root),
        "checkpoint_id": _sha256_bytes(checkpoint_id_material),
        "config_sha256": _sha256_bytes(raw_config),
        "index": index_result,
        "tp_size": tp_size,
        "config_contract": config_result,
        "top_level": {"ok": not top_errors, "errors": top_errors},
        "files": files,
        "layers": layers,
        "errors": errors,
    }
