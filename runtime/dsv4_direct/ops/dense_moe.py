"""Dequant-to-BF16 dense grouped GEMM MoE execution for prefill.

C2F attributed 48% of prefill stage time to the Marlin grouped GEMM running at
~11.5% MFU: at prefill row counts the in-kernel W4 dequantization is pure loss
because the GEMM is compute-bound, not weight-streaming-bound (the decode
regime Marlin was chosen for).

This module keeps the routed experts resident in *checkpoint* layout (packed
E2M1 nibbles + E8M0 group scales, same bytes as the Marlin resident) and, per
forward, dequantizes a chunk of experts into BF16 and runs plain dense GEMMs.
MXFP4 values (0, +-0.5 .. +-6 scaled by a power of two) are exactly
representable in BF16, so the dequantization is lossless and the arithmetic is
BF16 x BF16 with FP32 accumulation -- the same numerical class as the Marlin
W4A16 path, with the same op order as the reference Expert.forward.

Decode keeps using Marlin: there the weight stream dominates and dequantizing
3.2 GiB per layer would be the loss instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .marlin_moe import (
    ShardReader,
    SharedExpertSlice,
    _as_e8m0,
    _as_packed_bytes,
    _copy_slice,
    tensor_bytes,
)
from ..moe_forward import FP4_VALUES


def _byte_pair_table(device: torch.device) -> torch.Tensor:
    """[256, 2] BF16 table mapping a packed byte to its (low, high) nibble
    values.  The checkpoint stores the low nibble at the even logical K index
    (dsv4_direct.moe_forward.dequant_mxfp4)."""

    values = torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)
    bytes_ = torch.arange(256, device=device, dtype=torch.long)
    low = values[bytes_ & 0x0F]
    high = values[bytes_ >> 4]
    return torch.stack((low, high), dim=-1).to(torch.bfloat16).contiguous()


def dequant_mxfp4_bf16(
    packed: torch.Tensor, scale: torch.Tensor, table: torch.Tensor
) -> torch.Tensor:
    """Exact BF16 dequantization of one or more checkpoint-layout MXFP4 tensors.

    packed [..., N, K/2] uint8, scale [..., N, K/32] E8M0 -> [..., N, K] BF16.
    Exact because every E2M1 value needs at most three significand bits and the
    E8M0 scale is a power of two.
    """

    if packed.dtype == torch.int8:
        packed = packed.view(torch.uint8)
    pairs = table[packed.long()]
    values = pairs.flatten(-2)
    decoded = scale.float().to(torch.bfloat16)
    return values * decoded.repeat_interleave(32, dim=-1)


@dataclass
class DenseRoutedWeights:
    """Routed experts in checkpoint layout, stacked over the expert axis."""

    w13_packed: torch.Tensor  # [E, 2*local_inter, hidden//2] uint8
    w13_scale: torch.Tensor  # [E, 2*local_inter, hidden//32] E8M0
    w2_packed: torch.Tensor  # [E, hidden, local_inter//2] uint8
    w2_scale: torch.Tensor  # [E, hidden, local_inter//32] E8M0

    @property
    def resident_bytes(self) -> int:
        return tensor_bytes(
            self.w13_packed, self.w13_scale, self.w2_packed, self.w2_scale
        )


@dataclass
class ResidentDenseMoEWeights:
    routed: DenseRoutedWeights
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


def load_dense_moe_layer(
    *,
    stage_root: Path,
    layer_id: int,
    rank: int,
    world_size: int,
    hidden_size: int,
    intermediate_size: int,
    n_experts: int,
    device: torch.device,
    progress_every: int = 64,
    progress: Callable[[str], None] | None = None,
    checkpoint_id: str | None = None,
    key_prefix: str | None = None,
) -> ResidentDenseMoEWeights:
    """Load the intermediate-TP slice of every expert without Marlin repack.

    Slicing is byte-identical to ops.marlin_moe.load_resident_moe_layer; only
    the post-processing differs (stack raw instead of repack).
    """

    if (
        not isinstance(checkpoint_id, str)
        or len(checkpoint_id) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_id)
    ):
        raise ValueError("resident MoE loads require a lowercase SHA-256 checkpoint_id")
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
    routed: DenseRoutedWeights | None = None

    with ShardReader(Path(stage_root), weight_map) as handle:
        for expert_id in range(n_experts):
            expert = f"{prefix}.experts.{expert_id}"
            w1 = _as_packed_bytes(
                _copy_slice(handle, f"{expert}.w1.weight", slice(start, end)), device
            )
            s1 = _as_e8m0(
                _copy_slice(handle, f"{expert}.w1.scale", slice(start, end)), device
            )
            w3 = _as_packed_bytes(
                _copy_slice(handle, f"{expert}.w3.weight", slice(start, end)), device
            )
            s3 = _as_e8m0(
                _copy_slice(handle, f"{expert}.w3.scale", slice(start, end)), device
            )
            w2 = _as_packed_bytes(
                _copy_slice(
                    handle,
                    f"{expert}.w2.weight",
                    (slice(None), slice(start // 2, end // 2)),
                ),
                device,
            )
            s2 = _as_e8m0(
                _copy_slice(
                    handle,
                    f"{expert}.w2.scale",
                    (slice(None), slice(start // 32, end // 32)),
                ),
                device,
            )
            w13 = torch.cat((w1, w3), dim=0)
            s13 = torch.cat((s1, s3), dim=0)
            if routed is None:
                routed = DenseRoutedWeights(
                    w13_packed=torch.empty(
                        (n_experts,) + w13.shape, dtype=w13.dtype, device=device
                    ),
                    w13_scale=torch.empty(
                        (n_experts,) + s13.shape, dtype=s13.dtype, device=device
                    ),
                    w2_packed=torch.empty(
                        (n_experts,) + w2.shape, dtype=w2.dtype, device=device
                    ),
                    w2_scale=torch.empty(
                        (n_experts,) + s2.shape, dtype=s2.dtype, device=device
                    ),
                )
            routed.w13_packed[expert_id].copy_(w13)
            routed.w13_scale[expert_id].copy_(s13)
            routed.w2_packed[expert_id].copy_(w2)
            routed.w2_scale[expert_id].copy_(s2)
            del w1, s1, w3, s3, w2, s2, w13, s13
            if progress and progress_every and (expert_id + 1) % progress_every == 0:
                torch.cuda.synchronize(device)
                progress(f"layer={layer_id} rank={rank} experts={expert_id + 1}/{n_experts}")

        shared_prefix = f"{prefix}.shared_experts"
        scale_start = start // 128
        scale_end = end // 128
        shared = SharedExpertSlice(
            w1=_copy_slice(handle, f"{shared_prefix}.w1.weight", slice(start, end))
            .to(device)
            .contiguous(),
            s1=_copy_slice(
                handle, f"{shared_prefix}.w1.scale", slice(scale_start, scale_end)
            )
            .float()
            .to(device)
            .contiguous(),
            w3=_copy_slice(handle, f"{shared_prefix}.w3.weight", slice(start, end))
            .to(device)
            .contiguous(),
            s3=_copy_slice(
                handle, f"{shared_prefix}.w3.scale", slice(scale_start, scale_end)
            )
            .float()
            .to(device)
            .contiguous(),
            w2=_copy_slice(
                handle, f"{shared_prefix}.w2.weight", (slice(None), slice(start, end))
            )
            .to(device)
            .contiguous(),
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
    return ResidentDenseMoEWeights(
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


class DenseRoutedExecutor:
    """Sorted per-expert dense GEMM execution of the routed half."""

    def __init__(
        self,
        routed: DenseRoutedWeights,
        *,
        n_experts: int,
        hidden_size: int,
        local_intermediate: int,
        topk: int,
        clamp_limit: float,
        expert_chunk: int = 8,
    ) -> None:
        self.routed = routed
        self.n_experts = n_experts
        self.hidden_size = hidden_size
        self.local_intermediate = local_intermediate
        self.topk = topk
        self.clamp_limit = clamp_limit
        self.expert_chunk = max(1, int(expert_chunk))
        self.device = routed.w13_packed.device
        self._table = _byte_pair_table(self.device)
        self._scratch: torch.Tensor | None = None

    def _assignment_scratch(self, assignments: int) -> torch.Tensor:
        if self._scratch is not None and self._scratch.shape[0] >= assignments:
            return self._scratch[:assignments]
        self._scratch = torch.empty(
            (assignments, self.hidden_size), dtype=torch.bfloat16, device=self.device
        )
        return self._scratch

    def __call__(
        self,
        gathered: torch.Tensor,
        route_weights: torch.Tensor,
        route_ids: torch.Tensor,
    ) -> torch.Tensor:
        """gathered [M, hidden] BF16 -> routed partial [M, hidden] BF16.

        Op order mirrors the reference Expert.forward: FP32 clamp/SiLU between
        the two GEMMs, BF16 activations into each GEMM.  The router weight is
        applied on the second GEMM's output (the vLLM/Marlin convention; a
        per-row scalar, so algebraically identical to the reference's w2-input
        placement).
        """

        rows = gathered.shape[0]
        topk = self.topk
        limit = self.clamp_limit
        flat_ids = route_ids.reshape(-1)
        flat_weights = route_weights.reshape(-1).float()
        order = torch.argsort(flat_ids.to(torch.int32), stable=True)
        sorted_rows = torch.div(order, topk, rounding_mode="floor")
        counts = torch.bincount(flat_ids.long(), minlength=self.n_experts)
        boundaries = torch.zeros(
            self.n_experts + 1, dtype=torch.long, device=self.device
        )
        torch.cumsum(counts, dim=0, out=boundaries[1:])
        host_bounds = boundaries.tolist()

        assignments = rows * topk
        scratch = self._assignment_scratch(assignments)
        inter = self.local_intermediate

        for chunk_start in range(0, self.n_experts, self.expert_chunk):
            chunk_end = min(chunk_start + self.expert_chunk, self.n_experts)
            if host_bounds[chunk_end] == host_bounds[chunk_start]:
                continue
            w13 = dequant_mxfp4_bf16(
                self.routed.w13_packed[chunk_start:chunk_end],
                self.routed.w13_scale[chunk_start:chunk_end],
                self._table,
            )
            w2 = dequant_mxfp4_bf16(
                self.routed.w2_packed[chunk_start:chunk_end],
                self.routed.w2_scale[chunk_start:chunk_end],
                self._table,
            )
            for expert in range(chunk_start, chunk_end):
                begin, stop = host_bounds[expert], host_bounds[expert + 1]
                if begin == stop:
                    continue
                index = sorted_rows[begin:stop]
                activations = gathered.index_select(0, index)
                projected = activations @ w13[expert - chunk_start].t()
                gate = projected[:, :inter].float().clamp(max=limit)
                up = projected[:, inter:].float().clamp(min=-limit, max=limit)
                hidden = (F.silu(gate) * up).to(torch.bfloat16)
                scratch[begin:stop] = hidden @ w2[expert - chunk_start].t()
            del w13, w2

        weighted = scratch.float() * flat_weights.index_select(0, order).unsqueeze(1)
        # Undo the expert-major sort, then reduce the topk axis in row order so
        # the accumulation order is independent of the routing permutation.
        restored = torch.empty_like(weighted)
        restored.index_copy_(0, order, weighted)
        return restored.view(rows, topk, self.hidden_size).sum(dim=1).to(torch.bfloat16)


__all__ = [
    "DenseRoutedExecutor",
    "DenseRoutedWeights",
    "ResidentDenseMoEWeights",
    "dequant_mxfp4_bf16",
    "load_dense_moe_layer",
]
