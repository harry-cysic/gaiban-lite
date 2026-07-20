"""Fixed-shape stateful decode scheduling primitives (V4-Flash port).

Ported from gaiban ``dsv4_direct/stateful_decode.py``.  This module owns the
position/dispatch metadata needed to capture fixed-shape CUDA-graph families;
it does not launch graphs or mutate attention KV state itself.

**Flash graph-family derivation.**  A graph family names the set of boundary
writes every layer of a stage performs at one decode position.  From the
reference ``model.py`` decode branch (:530-533): every layer writes its raw
ring slot ``start_pos % window`` on every step; a layer with
``compress_ratio == R`` additionally completes one compressed row exactly when
its group fills, i.e. at positions ``p`` with ``p % R == R - 1``; pure
sliding-window layers (``compress_ratio == 0``, Flash L0/L1, model.py:466-481)
never enter the compressor branch (``if self.compress_ratio:`` guards
model.py:508-514/531-532), so they contribute **no** boundary write at any
position.  With Flash ratios {0, 4, 128} and 4 | 128, the per-position write
sets are therefore exactly the same three as Pro:

- ``NORMAL``                    (``p % 4 != 3``): ring writes only, everywhere.
- ``RATIO4_BOUNDARY``           (``p % 4 == 3 and p % 128 != 127``): ratio-4
  layers close one compressed row; window and ratio-128 layers ring-write only.
- ``RATIO4_RATIO128_BOUNDARY``  (``p % 128 == 127``): ratio-4 and ratio-128
  layers each close one compressed row.

No window-specific family is needed: adding one would duplicate an existing
write set.  ``classify_decode_position`` is thus unchanged from gaiban; only
its interpretation gains the "window layers ignore both flags" clause, which
``block.DirectDecodeBlock`` implements.

Flash keeps the ratio-128 layer geometry the sparse-index helpers encode:
window 128, compress ratio 128, so those constants are unchanged from gaiban.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch


RATIO4 = 4
RATIO128 = 128
RATIO128_WINDOW_SIZE = 128
SPARSE_BUCKET_ALIGNMENT = 32

DISPATCH_ERROR_NONE = 0
DISPATCH_ERROR_WRONG_FAMILY = 1
DISPATCH_ERROR_POSITION_OVERFLOW = 2
DISPATCH_ERROR_INVALID_POSITION = 4
DISPATCH_ERROR_POSITION_MISMATCH = 8
DISPATCH_ERROR_RANGE_EXHAUSTED = 16
DISPATCH_ERROR_STOP_POSITION_MISMATCH = 32
DISPATCH_ERROR_STATE_POSITION_MISMATCH = 64
DISPATCH_ERROR_STATE_ADVANCE_MISMATCH = 128

_INT64_MAX = torch.iinfo(torch.int64).max


class DecodeGraphFamily(str, Enum):
    """The three fixed-shape graphs required by any Flash decode schedule."""

    NORMAL = "normal"
    RATIO4_BOUNDARY = "ratio4_boundary"
    RATIO4_RATIO128_BOUNDARY = "ratio4_ratio128_boundary"


_FAMILY_CODES = {
    DecodeGraphFamily.NORMAL: 0,
    DecodeGraphFamily.RATIO4_BOUNDARY: 1,
    DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY: 2,
}


@dataclass(frozen=True)
class DecodeScheduleStep:
    position: int
    family: DecodeGraphFamily


def _require_family(family: DecodeGraphFamily) -> DecodeGraphFamily:
    if not isinstance(family, DecodeGraphFamily):
        raise TypeError("family must be a DecodeGraphFamily")
    return family


def classify_decode_position(position: int) -> DecodeGraphFamily:
    """Classify one token position without accepting coercible input values."""

    position = _require_position("position", position)
    if position % RATIO128 == RATIO128 - 1:
        return DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY
    if position % RATIO4 == RATIO4 - 1:
        return DecodeGraphFamily.RATIO4_BOUNDARY
    return DecodeGraphFamily.NORMAL


def family_boundary_flags(family: DecodeGraphFamily) -> tuple[bool, bool]:
    """Return ratio-4 and ratio-128 boundary flags for one graph family.

    Window layers (Flash compress_ratio == 0) consume neither flag; the
    reference decode branch performs only the ring write for them.
    """

    family = _require_family(family)
    if family is DecodeGraphFamily.NORMAL:
        return False, False
    if family is DecodeGraphFamily.RATIO4_BOUNDARY:
        return True, False
    if family is DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY:
        return True, True
    raise AssertionError(f"unhandled decode graph family {family}")


def classify_decode_position_tensor(position: torch.Tensor) -> torch.Tensor:
    """Return the graph-family code for a device-resident scalar position.

    The result is an int32 tensor with shape ``[1]``. Classification contains
    no value-dependent host read, so it may be recorded inside a CUDA graph.
    Callers must separately reject negative positions through dispatch_error.
    """

    _validate_position_tensor(position)
    ratio4_boundary = position.remainder(RATIO4).eq(RATIO4 - 1)
    ratio128_boundary = position.remainder(RATIO128).eq(RATIO128 - 1)
    return torch.where(
        ratio128_boundary,
        torch.full_like(
            position,
            _FAMILY_CODES[DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY],
        ),
        torch.where(
            ratio4_boundary,
            torch.full_like(position, _FAMILY_CODES[DecodeGraphFamily.RATIO4_BOUNDARY]),
            torch.full_like(position, _FAMILY_CODES[DecodeGraphFamily.NORMAL]),
        ),
    ).to(torch.int32)


def build_decode_schedule(
    start_position: int, step_count: int
) -> tuple[DecodeScheduleStep, ...]:
    """Build an immutable consecutive host dispatch schedule."""

    start_position = _require_position("start_position", start_position)
    step_count = _require_positive_int("step_count", step_count)
    if start_position + step_count > _INT64_MAX:
        raise ValueError("decode schedule terminal cursor exceeds int64")
    return tuple(
        DecodeScheduleStep(position, classify_decode_position(position))
        for position in range(start_position, start_position + step_count)
    )


def schedule_family_counts(
    schedule: tuple[DecodeScheduleStep, ...],
) -> dict[DecodeGraphFamily, int]:
    """Count graph launches while validating a consecutive immutable schedule."""

    if not isinstance(schedule, tuple) or not schedule:
        raise TypeError("schedule must be a non-empty tuple")
    counts = {family: 0 for family in DecodeGraphFamily}
    previous: int | None = None
    for step in schedule:
        if not isinstance(step, DecodeScheduleStep):
            raise TypeError("schedule entries must be DecodeScheduleStep values")
        if previous is not None and step.position != previous + 1:
            raise ValueError("decode schedule positions must be consecutive")
        expected = classify_decode_position(step.position)
        if step.family is not expected:
            raise ValueError("decode schedule family differs from its position")
        counts[step.family] += 1
        previous = step.position
    return counts


class StatefulDecodeCursor:
    """Own a device cursor, an independent error scalar, and a host shadow.

    ``advance_device`` belongs at the tail of each captured graph. The host
    dispatcher calls ``advance_host`` exactly once after each successful graph
    launch. Keeping those operations separate is required because Python side
    effects execute during capture, not during graph replay.
    """

    def __init__(
        self,
        *,
        start_position: int,
        device: torch.device | str = "cpu",
    ) -> None:
        start_position = _require_position("start_position", start_position)
        self._device = torch.device(device)
        self.device_position = torch.full(
            (1,), start_position, dtype=torch.int64, device=self._device
        )
        self.dispatch_error = torch.zeros(
            (1,), dtype=torch.int32, device=self._device
        )
        self._host_position = start_position
        self.validate_contract()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def cursor(self) -> torch.Tensor:
        """Alias exposing the device cursor without adding another owner."""

        return self.device_position

    @property
    def host_position(self) -> int:
        return self._host_position

    @property
    def host_family(self) -> DecodeGraphFamily:
        return classify_decode_position(self._host_position)

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for _, tensor in self._owned_tensor_items()
        )

    def _owned_tensor_items(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("device_position", self.device_position),
            ("dispatch_error", self.dispatch_error),
        )

    def _owned_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(tensor for _, tensor in self._owned_tensor_items())

    def validate_contract(self) -> tuple[tuple[str, torch.Tensor], ...]:
        """Fail closed if cursor ownership or static tensor metadata drifted."""

        _require_position("host_position", self._host_position)
        expected = {
            "device_position": ((1,), torch.int64),
            "dispatch_error": ((1,), torch.int32),
        }
        items = self._owned_tensor_items()
        tensor_attributes = {
            name
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        if tensor_attributes != set(expected):
            raise RuntimeError("cursor tensor ownership contract differs")
        pointers: set[tuple[torch.device, int]] = set()
        for name, tensor in items:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            shape, dtype = expected[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
            if tensor.dtype != dtype:
                raise TypeError(f"{name} dtype {tensor.dtype} != {dtype}")
            if tensor.device != self._device:
                raise ValueError(
                    f"{name} device {tensor.device} != cursor device {self._device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            owner = (tensor.device, int(tensor.untyped_storage().data_ptr()))
            if owner in pointers:
                raise ValueError("cursor tensors must own independent storage")
            pointers.add(owner)
        return items

    def _validate_device_guards(
        self,
        *,
        expected_position: torch.Tensor | None,
        stop_position: torch.Tensor | None,
        stop_position_constant: int | None,
        state_positions: Sequence[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, ...]:
        self.validate_contract()
        guards = []
        for name, value in (
            ("expected_position", expected_position),
            ("stop_position", stop_position),
        ):
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tuple(value.shape) != (1,):
                raise ValueError(f"{name} must have shape [1]")
            if value.dtype != torch.int64:
                raise TypeError(f"{name} must use int64")
            if value.device != self._device:
                raise ValueError(f"{name} must use the cursor device")
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            guards.append(value)
        if stop_position_constant is not None:
            _require_position("stop_position_constant", stop_position_constant)
            if stop_position is None:
                raise ValueError(
                    "stop_position_constant requires a stop_position tensor"
                )

        if state_positions is None:
            validated_states: tuple[torch.Tensor, ...] = ()
        else:
            if not isinstance(state_positions, Sequence):
                raise TypeError("device state_positions must be a sequence")
            if not state_positions:
                raise ValueError("device state_positions must be non-empty")
            validated = []
            expected_shape: tuple[int, ...] | None = None
            for index, value in enumerate(state_positions):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"state_positions[{index}] must be a tensor")
                if value.ndim != 1 or value.numel() <= 0:
                    raise ValueError(
                        f"state_positions[{index}] must be a non-empty vector"
                    )
                if value.dtype != torch.int64:
                    raise TypeError(f"state_positions[{index}] must use int64")
                if value.device != self._device:
                    raise ValueError(
                        f"state_positions[{index}] must use the cursor device"
                    )
                if not value.is_contiguous():
                    raise ValueError(
                        f"state_positions[{index}] must be contiguous"
                    )
                if expected_shape is None:
                    expected_shape = tuple(value.shape)
                elif tuple(value.shape) != expected_shape:
                    raise ValueError("device state position shapes must agree")
                validated.append(value)
            validated_states = tuple(validated)

        all_guards = (*guards, *validated_states)
        owners = {
            int(self.device_position.untyped_storage().data_ptr()),
            int(self.dispatch_error.untyped_storage().data_ptr()),
            *(int(value.untyped_storage().data_ptr()) for value in all_guards),
        }
        if len(owners) != 2 + len(all_guards):
            raise ValueError("device cursor guards must own independent storage")
        return validated_states

    def _device_preflight_error(
        self,
        family: DecodeGraphFamily,
        *,
        expected_position: torch.Tensor | None,
        stop_position: torch.Tensor | None,
        stop_position_constant: int | None,
        state_positions: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        invalid = self.device_position.lt(0)
        actual_code = classify_decode_position_tensor(self.device_position)
        wrong_family = torch.logical_and(
            actual_code.ne(_FAMILY_CODES[family]), ~invalid
        )
        overflow = self.device_position.eq(_INT64_MAX)
        position_mismatch = (
            torch.zeros_like(invalid)
            if expected_position is None
            else self.device_position.ne(expected_position)
        )
        range_exhausted = (
            torch.zeros_like(invalid)
            if stop_position is None
            else self.device_position.ge(stop_position)
        )
        stop_mismatch = (
            torch.zeros_like(invalid)
            if stop_position_constant is None
            else stop_position.ne(stop_position_constant)
        )
        state_mismatch = torch.zeros_like(invalid)
        for state_position in state_positions:
            state_mismatch = torch.logical_or(
                state_mismatch,
                state_position.ne(self.device_position).any().view(1),
            )

        error = wrong_family.to(torch.int32).mul(DISPATCH_ERROR_WRONG_FAMILY)
        for condition, error_bit in (
            (overflow, DISPATCH_ERROR_POSITION_OVERFLOW),
            (invalid, DISPATCH_ERROR_INVALID_POSITION),
            (position_mismatch, DISPATCH_ERROR_POSITION_MISMATCH),
            (range_exhausted, DISPATCH_ERROR_RANGE_EXHAUSTED),
            (stop_mismatch, DISPATCH_ERROR_STOP_POSITION_MISMATCH),
            (state_mismatch, DISPATCH_ERROR_STATE_POSITION_MISMATCH),
        ):
            error.bitwise_or_(condition.to(torch.int32).mul(error_bit))
        return error

    def guard_device_preflight(
        self,
        family: DecodeGraphFamily,
        *,
        expected_position: torch.Tensor,
        stop_position: torch.Tensor,
        stop_position_constant: int,
        state_positions: Sequence[torch.Tensor],
    ) -> None:
        """Record graph-entry errors without branching around graph work.

        A non-zero sticky result makes the whole replay disposable. It does not
        prevent later captured kernels from writing state; the caller must
        discard or restore every graph-owned state after observing an error.
        """

        family = _require_family(family)
        validated_states = self._validate_device_guards(
            expected_position=expected_position,
            stop_position=stop_position,
            stop_position_constant=stop_position_constant,
            state_positions=state_positions,
        )
        error = self._device_preflight_error(
            family,
            expected_position=expected_position,
            stop_position=stop_position,
            stop_position_constant=stop_position_constant,
            state_positions=validated_states,
        )
        self.dispatch_error.bitwise_or_(error)

    def advance_device(
        self,
        family: DecodeGraphFamily,
        *,
        expected_position: torch.Tensor | None = None,
        stop_position: torch.Tensor | None = None,
        stop_position_constant: int | None = None,
        state_positions_after: Sequence[torch.Tensor] | None = None,
    ) -> None:
        """Advance the cursor and expected-position shadow only with zero error."""

        family = _require_family(family)
        validated_states = self._validate_device_guards(
            expected_position=expected_position,
            stop_position=stop_position,
            stop_position_constant=stop_position_constant,
            state_positions=state_positions_after,
        )
        error = self._device_preflight_error(
            family,
            expected_position=expected_position,
            stop_position=stop_position,
            stop_position_constant=stop_position_constant,
            state_positions=(),
        )
        if validated_states:
            expected_after = self.device_position + 1
            state_advance_mismatch = torch.zeros_like(
                self.device_position, dtype=torch.bool
            )
            for state_position in validated_states:
                state_advance_mismatch = torch.logical_or(
                    state_advance_mismatch,
                    state_position.ne(expected_after).any().view(1),
                )
            error.bitwise_or_(
                state_advance_mismatch.to(torch.int32).mul(
                    DISPATCH_ERROR_STATE_ADVANCE_MISMATCH
                )
            )
        self.dispatch_error.bitwise_or_(error)
        can_advance = self.dispatch_error.eq(DISPATCH_ERROR_NONE)
        advance = can_advance.to(torch.int64)
        self.device_position.add_(advance)
        if expected_position is not None:
            expected_position.add_(advance)

    def advance_host(self, family: DecodeGraphFamily) -> None:
        """Advance only the host shadow after one graph replay returns."""

        family = _require_family(family)
        expected = classify_decode_position(self._host_position)
        if family is not expected:
            raise ValueError(
                f"host position {self._host_position} requires {expected.value}, "
                f"not {family.value}"
            )
        if self._host_position == _INT64_MAX:
            raise OverflowError("host cursor cannot advance beyond int64")
        self._host_position += 1

    def reset(self, position: int) -> None:
        """Reset both cursors and clear every sticky dispatch error."""

        position = _require_position("position", position)
        self.validate_contract()
        self.device_position.fill_(position)
        self.dispatch_error.zero_()
        self._host_position = position


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
    "DISPATCH_ERROR_INVALID_POSITION",
    "DISPATCH_ERROR_NONE",
    "DISPATCH_ERROR_POSITION_MISMATCH",
    "DISPATCH_ERROR_POSITION_OVERFLOW",
    "DISPATCH_ERROR_RANGE_EXHAUSTED",
    "DISPATCH_ERROR_STATE_ADVANCE_MISMATCH",
    "DISPATCH_ERROR_STATE_POSITION_MISMATCH",
    "DISPATCH_ERROR_STOP_POSITION_MISMATCH",
    "DISPATCH_ERROR_WRONG_FAMILY",
    "DecodeGraphFamily",
    "DecodeScheduleStep",
    "RATIO4",
    "RATIO128",
    "RATIO128_WINDOW_SIZE",
    "SPARSE_BUCKET_ALIGNMENT",
    "StatefulDecodeCursor",
    "build_decode_schedule",
    "build_padded_ratio128_sparse_indices",
    "classify_decode_position",
    "classify_decode_position_tensor",
    "family_boundary_flags",
    "ratio128_sparse_bucket_width",
    "schedule_family_counts",
]
