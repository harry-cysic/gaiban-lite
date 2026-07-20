"""Pure correctness helpers for the TP4 MoE forward gate (Flash port).

These functions operate on checkpoint-native tensors and do not depend on a
model or serving runtime. The distributed runner owns collectives and invokes
vLLM only as the Marlin operator provider.

Ported from gaiban's ``moe_forward.py`` unchanged except the default
``route_scale``: Flash's routed_scaling_factor is 1.5 (Pro used 2.5).
Callers should pass the checkpoint value explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


E8M0 = torch.float8_e8m0fnu
FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class GateForwardTensors:
    routing_weights: torch.Tensor
    routing_ids: torch.Tensor
    margin: torch.Tensor
    selection_ids: torch.Tensor
    selection_scores: torch.Tensor


def gate_forward_with_boundary(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int = 6,
    route_scale: float = 1.5,
) -> GateForwardTensors:
    """Run the learned sqrt-softplus gate used from layer 3 onward.

    Bias changes expert selection only. Routing weights come from the unbiased
    scores, are normalized per row, and are then multiplied by ``route_scale``.
    The returned margin is the biased k-th score minus the (k+1)-th score for
    every row; a small margin identifies routing decisions sensitive to noise.
    """

    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("x/weight/bias must have ranks 2/2/1")
    experts, hidden = weight.shape
    if x.shape[1] != hidden:
        raise ValueError(f"x hidden size {x.shape[1]} != gate hidden size {hidden}")
    if bias.numel() != experts:
        raise ValueError(f"gate bias size {bias.numel()} != experts {experts}")
    if not isinstance(topk, int) or isinstance(topk, bool) or not 0 < topk < experts:
        raise ValueError("topk must be a positive integer smaller than expert count")
    if not math.isfinite(route_scale) or route_scale <= 0:
        raise ValueError("route_scale must be finite and positive")

    scores = F.softplus(F.linear(x.float(), weight.float())).sqrt()
    selection_scores = scores + bias.float()
    selected_values, selected_ids = selection_scores.topk(topk + 1, dim=-1)
    ids = selected_ids[:, :topk]
    margin = selected_values[:, topk - 1] - selected_values[:, topk]
    routing_weights = scores.gather(1, ids)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights * float(route_scale)
    return GateForwardTensors(
        routing_weights=routing_weights.float(),
        routing_ids=ids.to(torch.int32),
        margin=margin.float(),
        selection_ids=selected_ids,
        selection_scores=selected_values.float(),
    )


def gate_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int = 6,
    route_scale: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = gate_forward_with_boundary(
        x,
        weight,
        bias,
        topk=topk,
        route_scale=route_scale,
    )
    return result.routing_weights, result.routing_ids, result.margin


def _decoded_e8m0(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.contiguous()
    if scale.dtype == torch.uint8:
        scale = scale.view(E8M0)
    if scale.dtype == E8M0:
        return scale.float()
    if scale.is_floating_point():
        return scale.float()
    raise TypeError(f"expected E8M0, uint8, or decoded floating scale, got {scale.dtype}")


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize checkpoint-native packed E2M1 values to float32.

    The checkpoint stores the low nibble before the high nibble and one E8M0
    scale for each group of 32 logical K values.
    """

    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError("packed MXFP4 weight and scale must both be 2-D")
    if packed.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"expected int8/uint8 packed MXFP4 weight, got {packed.dtype}")
    rows = packed.shape[0]
    logical_k = packed.shape[1] * 2
    if logical_k % 32:
        raise ValueError(f"logical MXFP4 K={logical_k} must be divisible by 32")
    expected_scale_shape = (rows, logical_k // 32)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"MXFP4 scale shape {tuple(scale.shape)} != {expected_scale_shape}"
        )

    packed_u8 = packed.contiguous()
    if packed_u8.dtype == torch.int8:
        packed_u8 = packed_u8.view(torch.uint8)
    table = torch.tensor(FP4_VALUES, dtype=torch.float32, device=packed.device)
    low = table[(packed_u8 & 0x0F).long()]
    high = table[(packed_u8 >> 4).long()]
    values = torch.stack((low, high), dim=-1).flatten(1)
    expanded_scale = _decoded_e8m0(scale).repeat_interleave(32, dim=1)
    return values * expanded_scale


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


def error_metrics(observed: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Return JSON-safe max-absolute and relative-RMS error metrics."""

    if observed.shape != reference.shape:
        raise ValueError(
            f"observed shape {tuple(observed.shape)} != reference shape {tuple(reference.shape)}"
        )
    if observed.numel() == 0:
        raise ValueError("cannot compare empty tensors")

    observed_fp32 = observed.detach().float()
    reference_fp32 = reference.detach().float()
    finite = bool(
        torch.isfinite(observed_fp32).all().item()
        and torch.isfinite(reference_fp32).all().item()
    )
    if not finite:
        return {
            "finite": False,
            "max_abs": None,
            "rms_rel": None,
            "reference_rms": None,
        }

    difference = observed_fp32 - reference_fp32
    max_abs = difference.abs().max().item()
    difference_rms = difference.square().mean().sqrt().item()
    reference_rms = reference_fp32.square().mean().sqrt().item()
    denominator = max(reference_rms, torch.finfo(torch.float32).eps)
    return {
        "finite": True,
        "max_abs": float(max_abs),
        "rms_rel": float(difference_rms / denominator),
        "reference_rms": float(reference_rms),
    }


__all__ = [
    "FP4_VALUES",
    "GateForwardTensors",
    "HashGateForwardTensors",
    "dequant_fp8_block",
    "dequant_mxfp4",
    "error_metrics",
    "gate_forward",
    "gate_forward_with_boundary",
    "hash_gate_forward",
]
