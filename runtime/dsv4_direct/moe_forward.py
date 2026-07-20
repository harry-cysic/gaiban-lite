"""Checkpoint-native FP8 dequantization and hash-gate helpers (Flash slice).

This is the minimal subset of gaiban's ``moe_forward.py`` required by the
attention forward ports: :func:`dequant_fp8_block` (FP8 block-scaled
projection dequant) and :func:`hash_gate_forward` (the pure token-ID router
used by the first three Flash layers, gated by the E0ff ratio-4 harness).
The learned-gate and expert forward functions are intentionally NOT ported
yet; later phases extend this module rather than renaming the import path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


E8M0 = torch.float8_e8m0fnu


def _decoded_e8m0(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.contiguous()
    if scale.dtype == torch.uint8:
        scale = scale.view(E8M0)
    if scale.dtype == E8M0:
        return scale.float()
    if scale.is_floating_point():
        return scale.float()
    raise TypeError(f"expected E8M0, uint8, or decoded floating scale, got {scale.dtype}")


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a 128x128 block-scaled FP8 checkpoint weight to float32."""

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 block weight and scale must both be 2-D")
    rows, columns = weight.shape
    row_blocks = (rows + 127) // 128
    column_blocks = (columns + 127) // 128
    expected_scale_shape = (row_blocks, column_blocks)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"FP8 block scale shape {tuple(scale.shape)} != {expected_scale_shape}"
        )
    expanded_scale = (
        _decoded_e8m0(scale)
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)[:rows, :columns]
    )
    return weight.float() * expanded_scale


@dataclass(frozen=True)
class HashGateForwardTensors:
    routing_weights: torch.Tensor
    routing_ids: torch.Tensor
    selected_scores: torch.Tensor


def hash_gate_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    route_scale: float = 1.5,
) -> HashGateForwardTensors:
    """Run the first-three-layer token-ID router without a learned top-k.

    Hash layers still compute sqrt-softplus expert scores for route weights.
    Only the selected expert IDs come from the checkpoint ``tid2eid`` table.
    Keeping the table lookup and score gather explicit prevents synthetic expert
    IDs from silently entering the real layer-2 path.  Ported unchanged from
    gaiban except the default ``route_scale``: Flash's routed_scaling_factor is
    1.5 (Pro used 2.5); callers should pass the checkpoint value explicitly.
    """

    if x.ndim != 2 or weight.ndim != 2 or tid2eid.ndim != 2:
        raise ValueError("x/weight/tid2eid must have ranks 2/2/2")
    experts, hidden = weight.shape
    if x.shape[1] != hidden:
        raise ValueError(f"x hidden size {x.shape[1]} != gate hidden size {hidden}")
    if input_ids.ndim != 1 or input_ids.shape[0] != x.shape[0]:
        raise ValueError("input_ids must contain one token ID per gate row")
    if input_ids.dtype != torch.int64 or tid2eid.dtype != torch.int64:
        raise TypeError("hash input IDs and tid2eid must be int64")
    if tid2eid.shape[1] <= 0 or tid2eid.shape[1] >= experts:
        raise ValueError("tid2eid top-k width must be positive and smaller than experts")
    if x.device != weight.device or x.device != tid2eid.device:
        raise ValueError("hash gate tensors must share one device")
    if input_ids.device != x.device:
        raise ValueError("hash gate input IDs must share the gate device")
    if not math.isfinite(route_scale) or route_scale <= 0:
        raise ValueError("route_scale must be finite and positive")

    scores = F.softplus(F.linear(x.float(), weight.float())).sqrt()
    ids = tid2eid.index_select(0, input_ids)
    selected_scores = scores.gather(1, ids)
    routing_weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights * float(route_scale)
    return HashGateForwardTensors(
        routing_weights=routing_weights.float(),
        routing_ids=ids.to(torch.int32),
        selected_scores=selected_scores.float(),
    )


__all__ = ["HashGateForwardTensors", "dequant_fp8_block", "hash_gate_forward"]
