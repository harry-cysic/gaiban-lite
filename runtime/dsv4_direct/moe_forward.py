"""Checkpoint-native FP8 dequantization helpers (minimal Flash port slice).

This is the minimal subset of gaiban's ``moe_forward.py`` required by the
ratio-128 attention forward port (``attention.py`` dequantizes the FP8
block-scaled attention projections through :func:`dequant_fp8_block`).  The
MoE gate/expert forward functions are intentionally NOT ported yet; later
phases extend this module rather than renaming the import path.
"""

from __future__ import annotations

import torch


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


__all__ = ["dequant_fp8_block"]
