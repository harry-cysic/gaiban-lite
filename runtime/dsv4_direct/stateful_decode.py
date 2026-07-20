"""Fixed-shape ratio-128 sparse-index primitives (minimal Flash port slice).

This is the minimal subset of gaiban's ``stateful_decode.py`` required by the
ratio-128 attention forward port: ``attention.py`` imports
:func:`ratio128_sparse_bucket_width` and
:func:`build_padded_ratio128_sparse_indices` for its cursor-driven stateful
decode plan.  The graph-family scheduling machinery (DecodeGraphFamily,
StatefulDecodeCursor, ...) is intentionally NOT ported yet; later phases
extend this module rather than renaming the import path.

Flash keeps the ratio-128 layer geometry that these helpers encode: window
128, compress ratio 128, so all constants are unchanged from gaiban.
"""

from __future__ import annotations

import torch


RATIO128 = 128
RATIO128_WINDOW_SIZE = 128
SPARSE_BUCKET_ALIGNMENT = 32

_INT64_MAX = torch.iinfo(torch.int64).max


def _require_position(name: str, position: int) -> int:
    if (
        not isinstance(position, int)
        or isinstance(position, bool)
        or position < 0
        or position > _INT64_MAX
    ):
        raise ValueError(f"{name} must be a non-negative int64 position")
    return position


def _require_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_position_tensor(position: torch.Tensor) -> None:
    if not isinstance(position, torch.Tensor):
        raise TypeError("position must be a tensor")
    if tuple(position.shape) != (1,):
        raise ValueError("position tensor must have shape [1]")
    if position.dtype != torch.int64:
        raise TypeError("position tensor must use int64")
    if not position.is_contiguous():
        raise ValueError("position tensor must be contiguous")


def ratio128_sparse_bucket_width(
    first_position: int, last_position: int
) -> int:
    """Return one 32-aligned sparse width for an inclusive decode range.

    Positions name the token being appended.  The resulting width therefore
    covers the raw ring and every compressed row visible after ``last_position``
    has been written.
    """

    first_position = _require_position("first_position", first_position)
    last_position = _require_position("last_position", last_position)
    if last_position < first_position:
        raise ValueError("last_position must not precede first_position")
    completed_rows = last_position // RATIO128 + int(
        last_position % RATIO128 == RATIO128 - 1
    )
    visible_width = RATIO128_WINDOW_SIZE + completed_rows
    return (
        (visible_width + SPARSE_BUCKET_ALIGNMENT - 1)
        // SPARSE_BUCKET_ALIGNMENT
        * SPARSE_BUCKET_ALIGNMENT
    )


def build_padded_ratio128_sparse_indices(
    position: torch.Tensor,
    *,
    batch_size: int,
    bucket_width: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build fixed-shape ring/compressed indices with an exact ``-1`` suffix.

    ``position`` remains device-resident and no tensor value is read by Python.
    It names the token being appended, so the indices include that token and a
    compressed row completed by it at a ratio-128 boundary.
    Passing ``out`` gives callers stable output storage for graph capture. The
    caller is responsible for selecting a bucket that covers its host schedule;
    :func:`ratio128_sparse_bucket_width` computes that static capacity.

    Consumers must implement ``-1`` masking explicitly.
    """

    _validate_position_tensor(position)
    batch_size = _require_positive_int("batch_size", batch_size)
    bucket_width = _require_positive_int("bucket_width", bucket_width)
    if bucket_width < RATIO128_WINDOW_SIZE:
        raise ValueError("bucket_width must cover the 128-row raw window")
    if bucket_width % SPARSE_BUCKET_ALIGNMENT:
        raise ValueError("bucket_width must be a multiple of 32")

    columns = torch.arange(
        bucket_width, dtype=torch.int64, device=position.device
    )
    position_mod = position.remainder(RATIO128)
    full_raw = (columns + position_mod + 1).remainder(RATIO128)
    partial_raw = torch.where(
        columns.le(position), columns, torch.full_like(columns, -1)
    )
    raw = torch.where(position.ge(RATIO128_WINDOW_SIZE - 1), full_raw, partial_raw)

    completed_rows = torch.div(position, RATIO128, rounding_mode="floor")
    completed_rows = completed_rows + position_mod.eq(RATIO128 - 1).to(torch.int64)
    compressed_column = columns - RATIO128_WINDOW_SIZE
    compressed = torch.where(
        compressed_column.lt(completed_rows),
        columns,
        torch.full_like(columns, -1),
    )
    row = torch.where(columns.lt(RATIO128_WINDOW_SIZE), raw, compressed)
    row = torch.where(position.ge(0), row, torch.full_like(row, -1))
    expanded = row.to(torch.int32).view(1, 1, bucket_width).expand(
        batch_size, 1, bucket_width
    )

    expected_shape = (batch_size, 1, bucket_width)
    if out is None:
        return expanded.contiguous()
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a tensor")
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"out shape {tuple(out.shape)} != {expected_shape}")
    if out.dtype != torch.int32:
        raise TypeError("out must use int32")
    if out.device != position.device:
        raise ValueError("out and position must share a device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if out.untyped_storage().data_ptr() == position.untyped_storage().data_ptr():
        raise ValueError("out must not alias the position tensor")
    out.copy_(expanded)
    return out


__all__ = [
    "RATIO128",
    "RATIO128_WINDOW_SIZE",
    "SPARSE_BUCKET_ALIGNMENT",
    "build_padded_ratio128_sparse_indices",
    "ratio128_sparse_bucket_width",
]
