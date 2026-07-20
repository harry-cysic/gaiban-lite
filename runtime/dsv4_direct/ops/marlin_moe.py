"""MXFP4 checkpoint loading and Marlin repack for intermediate TP.

The external package used here is an operator provider only.  Model, scheduler,
configuration, and request lifecycle stay owned by the direct runtime.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open


E8M0 = torch.float8_e8m0fnu


def tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)


def tensor_sample_sha256(tensor: torch.Tensor, sample_count: int = 4096) -> str:
    """Hash deterministic samples without copying a multi-GiB tensor to the host."""

    flat = tensor.detach().flatten()
    count = min(int(flat.numel()), sample_count)
    if count == 0:
        payload = b""
    else:
        indices = torch.linspace(
            0, flat.numel() - 1, count, device=flat.device, dtype=torch.float64
        ).to(torch.long)
        payload = flat.index_select(0, indices).contiguous().view(torch.uint8).cpu().numpy().tobytes()
    metadata = f"{list(tensor.shape)}|{tensor.dtype}|{tensor.numel()}|".encode()
    return hashlib.sha256(metadata + payload).hexdigest()


@dataclass
class SharedExpertSlice:
    w1: torch.Tensor
    s1: torch.Tensor
    w3: torch.Tensor
    s3: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor

    @property
    def resident_bytes(self) -> int:
        return tensor_bytes(self.w1, self.s1, self.w3, self.s3, self.w2, self.s2)

    def summary(self) -> dict[str, Any]:
        tensors = {
            "w1": self.w1,
            "s1": self.s1,
            "w3": self.w3,
            "s3": self.s3,
            "w2": self.w2,
            "s2": self.s2,
        }
        return {
            "resident_bytes": self.resident_bytes,
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sample_sha256": tensor_sample_sha256(tensor),
                }
                for name, tensor in tensors.items()
            },
        }


@dataclass
class MarlinRoutedWeights:
    w13_q: torch.Tensor
    w13_s: torch.Tensor
    w2_q: torch.Tensor
    w2_s: torch.Tensor

    @property
    def resident_bytes(self) -> int:
        return tensor_bytes(self.w13_q, self.w13_s, self.w2_q, self.w2_s)

    def summary(self) -> dict[str, Any]:
        tensors = {
            "w13_q": self.w13_q,
            "w13_s": self.w13_s,
            "w2_q": self.w2_q,
            "w2_s": self.w2_s,
        }
        return {
            "resident_bytes": self.resident_bytes,
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sample_sha256": tensor_sample_sha256(tensor),
                }
                for name, tensor in tensors.items()
            },
        }


@dataclass
class ResidentMoEWeights:
    routed: MarlinRoutedWeights
    shared: SharedExpertSlice
    load_seconds: float
    layer_id: int | None = None
    rank: int | None = None
    world_size: int | None = None
    intermediate_start: int | None = None
    intermediate_end: int | None = None
    checkpoint_id: str | None = None

    @property
    def resident_bytes(self) -> int:
        return self.routed.resident_bytes + self.shared.resident_bytes

    def summary(self) -> dict[str, Any]:
        routed = self.routed.summary()
        shared = self.shared.summary()
        digest = hashlib.sha256(
            "|".join(
                item["sample_sha256"]
                for group in (routed, shared)
                for item in group["tensors"].values()
            ).encode()
        ).hexdigest()
        return {
            "resident_bytes": self.resident_bytes,
            "load_seconds": self.load_seconds,
            "identity": {
                "layer_id": self.layer_id,
                "rank": self.rank,
                "world_size": self.world_size,
                "intermediate_start": self.intermediate_start,
                "intermediate_end": self.intermediate_end,
                "checkpoint_id": self.checkpoint_id,
            },
            "sample_fingerprint": digest,
            "routed": routed,
            "shared": shared,
        }


def _as_packed_bytes(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = tensor.to(device=device, non_blocking=False).contiguous()
    if tensor.dtype == torch.int8:
        tensor = tensor.view(torch.uint8)
    if tensor.dtype != torch.uint8:
        raise TypeError(f"expected packed I8/U8 checkpoint tensor, got {tensor.dtype}")
    return tensor


def _as_e8m0(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = tensor.to(device=device, non_blocking=False).contiguous()
    if tensor.dtype == torch.uint8:
        tensor = tensor.view(E8M0)
    if tensor.dtype != E8M0:
        raise TypeError(f"expected E8M0 checkpoint scale, got {tensor.dtype}")
    return tensor


def _prepare_one_mxfp4(
    packed: torch.Tensor, scale: torch.Tensor, size_n: int, size_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        mxfp4_marlin_process_scales,
    )

    if size_k % 8:
        raise ValueError(f"Marlin packed K must be divisible by 8, got {size_k}")
    permutation = torch.empty(0, dtype=torch.int, device=packed.device)
    qweight = packed.view(torch.int32).T.contiguous()
    marlin_q = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=permutation,
        size_k=size_k,
        size_n=size_n,
        num_bits=4,
        is_a_8bit=False,
    )
    marlin_s = scale.to(torch.bfloat16).T.contiguous()
    marlin_s = marlin_permute_scales(
        s=marlin_s,
        size_k=size_k,
        size_n=size_n,
        group_size=32,
        is_a_8bit=False,
    )
    marlin_s = mxfp4_marlin_process_scales(marlin_s, input_dtype=None)
    return marlin_q, marlin_s


class ShardReader:
    """Resolve tensor keys through the index weight_map (the Flash runtime's
    only shard-resolution mechanism) and cache open safetensors handles."""

    def __init__(self, stage_root: Path, weight_map: dict[str, str]):
        self._stage_root = Path(stage_root)
        self._weight_map = weight_map
        self._handles: dict[str, Any] = {}

    def _handle(self, key: str) -> Any:
        try:
            filename = self._weight_map[key]
        except KeyError as error:
            raise KeyError(f"tensor key not in index weight_map: {key}") from error
        handle = self._handles.get(filename)
        if handle is None:
            handle = safe_open(self._stage_root / filename, framework="pt", device="cpu")
            self._handles[filename] = handle
        return handle

    def get_slice(self, key: str) -> Any:
        return self._handle(key).get_slice(key)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._handle(key).get_tensor(key)

    def close(self) -> None:
        for handle in self._handles.values():
            close = getattr(handle, "close", None)
            if callable(close):
                close()
        self._handles.clear()

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _copy_slice(handle: Any, key: str, index: Any) -> torch.Tensor:
    return handle.get_slice(key)[index].contiguous()


def load_resident_moe_layer(
    *,
    stage_root: Path,
    layer_id: int,
    rank: int,
    world_size: int,
    hidden_size: int,
    intermediate_size: int,
    n_experts: int,
    device: torch.device,
    progress_every: int = 32,
    progress: Callable[[str], None] | None = None,
    checkpoint_id: str | None = None,
    key_prefix: str | None = None,
) -> ResidentMoEWeights:
    """Load all experts with each rank holding one intermediate-dimension slice.

    key_prefix overrides the tensor namespace for the MTP block ("mtp.0.ffn");
    default is the decoder layer namespace.
    """

    if (
        not isinstance(checkpoint_id, str)
        or len(checkpoint_id) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_id)
    ):
        raise ValueError(
            "resident MoE loads require a lowercase SHA-256 checkpoint_id"
        )
    if intermediate_size % world_size:
        raise ValueError("intermediate size must divide the TP world size")
    local_intermediate = intermediate_size // world_size
    start = rank * local_intermediate
    end = start + local_intermediate
    if start % 128 or end % 128:
        raise ValueError("TP slice must preserve FP8 block boundaries")

    from ..checkpoint import load_weight_map

    prefix = key_prefix or f"layers.{layer_id}.ffn"
    weight_map, _ = load_weight_map(Path(stage_root))
    started = time.perf_counter()
    routed: MarlinRoutedWeights | None = None

    with ShardReader(Path(stage_root), weight_map) as handle:
        for expert_id in range(n_experts):
            expert = f"{prefix}.experts.{expert_id}"
            w1 = _as_packed_bytes(_copy_slice(handle, f"{expert}.w1.weight", slice(start, end)), device)
            s1 = _as_e8m0(_copy_slice(handle, f"{expert}.w1.scale", slice(start, end)), device)
            w3 = _as_packed_bytes(_copy_slice(handle, f"{expert}.w3.weight", slice(start, end)), device)
            s3 = _as_e8m0(_copy_slice(handle, f"{expert}.w3.scale", slice(start, end)), device)
            w2 = _as_packed_bytes(
                _copy_slice(handle, f"{expert}.w2.weight", (slice(None), slice(start // 2, end // 2))),
                device,
            )
            s2 = _as_e8m0(
                _copy_slice(handle, f"{expert}.w2.scale", (slice(None), slice(start // 32, end // 32))),
                device,
            )
            w13 = torch.cat((w1, w3), dim=0)
            s13 = torch.cat((s1, s3), dim=0)
            q13, ms13 = _prepare_one_mxfp4(w13, s13, 2 * local_intermediate, hidden_size)
            q2, ms2 = _prepare_one_mxfp4(w2, s2, hidden_size, local_intermediate)
            if routed is None:
                routed = MarlinRoutedWeights(
                    w13_q=torch.empty((n_experts,) + q13.shape, dtype=q13.dtype, device=device),
                    w13_s=torch.empty((n_experts,) + ms13.shape, dtype=ms13.dtype, device=device),
                    w2_q=torch.empty((n_experts,) + q2.shape, dtype=q2.dtype, device=device),
                    w2_s=torch.empty((n_experts,) + ms2.shape, dtype=ms2.dtype, device=device),
                )
            routed.w13_q[expert_id].copy_(q13)
            routed.w13_s[expert_id].copy_(ms13)
            routed.w2_q[expert_id].copy_(q2)
            routed.w2_s[expert_id].copy_(ms2)
            del w1, s1, w3, s3, w2, s2, w13, s13, q13, ms13, q2, ms2
            if progress and progress_every and (expert_id + 1) % progress_every == 0:
                torch.cuda.synchronize(device)
                progress(f"layer={layer_id} rank={rank} experts={expert_id + 1}/{n_experts}")

        shared_prefix = f"{prefix}.shared_experts"
        scale_start = start // 128
        scale_end = end // 128
        shared = SharedExpertSlice(
            w1=_copy_slice(handle, f"{shared_prefix}.w1.weight", slice(start, end)).to(device).contiguous(),
            s1=_copy_slice(handle, f"{shared_prefix}.w1.scale", slice(scale_start, scale_end))
            .float()
            .to(device)
            .contiguous(),
            w3=_copy_slice(handle, f"{shared_prefix}.w3.weight", slice(start, end)).to(device).contiguous(),
            s3=_copy_slice(handle, f"{shared_prefix}.w3.scale", slice(scale_start, scale_end))
            .float()
            .to(device)
            .contiguous(),
            w2=_copy_slice(handle, f"{shared_prefix}.w2.weight", (slice(None), slice(start, end))).to(device).contiguous(),
            s2=_copy_slice(
                handle,
                f"{shared_prefix}.w2.scale",
                (slice(None), slice(scale_start, scale_end)),
            )
            .float()
            .to(device)
            .contiguous(),
        )

    if routed is None:
        raise ValueError("checkpoint has no routed experts")
    torch.cuda.synchronize(device)
    return ResidentMoEWeights(
        routed=routed,
        shared=shared,
        load_seconds=time.perf_counter() - started,
        layer_id=layer_id,
        rank=rank,
        world_size=world_size,
        intermediate_start=start,
        intermediate_end=end,
        checkpoint_id=checkpoint_id,
    )
