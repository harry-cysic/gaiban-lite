"""Fixed-capacity ratio-128 KV state for the direct correctness runtime.

The state owns cache layout and the incremental pooling state, but it does not
own compressor weights. Callers provide the FP32 ``wkv``/``wgate`` projections
and an APE tensor. Whenever a group completes, ``finalize_compressed`` must
apply the checkpoint RMSNorm, RoPE, and NoPE quant-simulation semantics and
return the final BF16 latent row. Keeping that boundary explicit prevents an
identity pool from being mistaken for the model's complete compressor path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import torch


WINDOW_SIZE = 128
COMPRESS_RATIO = 128
LATENT_DIM = 512

CompressionFinalizer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CompressionWrite:
    """Completed compression rows produced by one cache write."""

    row_indices: tuple[int, ...]
    group_start_positions: tuple[int, ...]
    pooled: torch.Tensor


class StaticLayerKV:
    """Batch-local, non-paged KV state for one ratio-128 attention layer.

    The cache is deliberately static: every local sequence has one raw ring and
    one compressed tail sized for ``max_seq_len``. All sequences advance at the
    same position, matching the fixed-shape E0 block benchmark.
    """

    def __init__(
        self,
        *,
        num_local_sequences: int,
        max_seq_len: int,
        layer_id: int = 3,
        device: torch.device | str = "cpu",
    ) -> None:
        if (
            not isinstance(num_local_sequences, int)
            or isinstance(num_local_sequences, bool)
            or num_local_sequences <= 0
        ):
            raise ValueError("num_local_sequences must be a positive integer")
        if (
            not isinstance(max_seq_len, int)
            or isinstance(max_seq_len, bool)
            or max_seq_len < COMPRESS_RATIO
            or max_seq_len % COMPRESS_RATIO
        ):
            raise ValueError("max_seq_len must be a positive multiple of 128")
        if not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0:
            raise ValueError("layer_id must be a non-negative integer")

        self.num_local_sequences = num_local_sequences
        self.max_seq_len = max_seq_len
        self.layer_id = layer_id
        self.device = torch.device(device)
        self.compressed_capacity = max_seq_len // COMPRESS_RATIO

        self.latent = torch.zeros(
            num_local_sequences,
            WINDOW_SIZE + self.compressed_capacity,
            LATENT_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.kv_state = torch.zeros(
            num_local_sequences,
            COMPRESS_RATIO,
            LATENT_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        self.score_state = torch.full_like(self.kv_state, float("-inf"))

        self._next_position = torch.zeros(
            num_local_sequences, dtype=torch.int64, device=self.device
        )
        self._compressed_count = torch.zeros_like(self._next_position)
        self._raw_positions = torch.full(
            (num_local_sequences, WINDOW_SIZE),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self._compressed_group_starts = torch.full(
            (num_local_sequences, self.compressed_capacity),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self._state_positions = torch.full_like(self._raw_positions, -1)

    @property
    def raw(self) -> torch.Tensor:
        return self.latent[:, :WINDOW_SIZE]

    @property
    def compressed(self) -> torch.Tensor:
        return self.latent[:, WINDOW_SIZE:]

    @property
    def next_position(self) -> int:
        positions = self._next_position
        if not bool(torch.all(positions == positions[0]).item()):
            raise RuntimeError("local sequences no longer share one next position")
        return int(positions[0].item())

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in self._owned_tensors()
        )

    def _owned_tensor_items(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("latent", self.latent),
            ("kv_state", self.kv_state),
            ("score_state", self.score_state),
            ("next_position", self._next_position),
            ("compressed_count", self._compressed_count),
            ("raw_positions", self._raw_positions),
            ("compressed_group_starts", self._compressed_group_starts),
            ("state_positions", self._state_positions),
        )

    def _owned_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(tensor for _, tensor in self._owned_tensor_items())

    def _validate_owned_tensor_contract(
        self, *, label: str
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        batch = self.num_local_sequences
        expected = {
            "latent": (
                (batch, WINDOW_SIZE + self.compressed_capacity, LATENT_DIM),
                torch.bfloat16,
            ),
            "kv_state": ((batch, COMPRESS_RATIO, LATENT_DIM), torch.float32),
            "score_state": ((batch, COMPRESS_RATIO, LATENT_DIM), torch.float32),
            "next_position": ((batch,), torch.int64),
            "compressed_count": ((batch,), torch.int64),
            "raw_positions": ((batch, WINDOW_SIZE), torch.int64),
            "compressed_group_starts": (
                (batch, self.compressed_capacity),
                torch.int64,
            ),
            "state_positions": ((batch, COMPRESS_RATIO), torch.int64),
        }
        items = self._owned_tensor_items()
        if tuple(name for name, _ in items) != tuple(expected):
            raise RuntimeError(f"{label} ratio-128 state ownership contract differs")

        storage_owners: dict[tuple[torch.device, int], str] = {}
        for name, tensor in items:
            shape, dtype = expected[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{label} ratio-128 state {name} shape "
                    f"{tuple(tensor.shape)} != {shape}"
                )
            if tensor.dtype != dtype:
                raise TypeError(
                    f"{label} ratio-128 state {name} dtype {tensor.dtype} != {dtype}"
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"{label} ratio-128 state {name} device "
                    f"{tensor.device} != {self.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(
                    f"{label} ratio-128 state {name} must be contiguous"
                )
            owner = (tensor.device, int(tensor.untyped_storage().data_ptr()))
            previous = storage_owners.setdefault(owner, name)
            if previous != name:
                raise ValueError(
                    f"{label} ratio-128 state tensors {previous} and {name} "
                    "alias one storage"
                )
        return items

    def copy_from(self, source: "StaticLayerKV") -> None:
        """Copy one complete pre-step state after validating both owners."""

        if not isinstance(source, StaticLayerKV):
            raise TypeError("ratio-128 state source must be StaticLayerKV")
        identity = (
            self.num_local_sequences,
            self.max_seq_len,
            self.layer_id,
            self.compressed_capacity,
            self.device,
        )
        source_identity = (
            source.num_local_sequences,
            source.max_seq_len,
            source.layer_id,
            source.compressed_capacity,
            source.device,
        )
        if identity != source_identity:
            raise ValueError(
                f"ratio-128 state identity {source_identity} cannot copy into {identity}"
            )

        destination_items = self._validate_owned_tensor_contract(label="destination")
        source_items = source._validate_owned_tensor_contract(label="source")
        destination_owners = {
            (tensor.device, int(tensor.untyped_storage().data_ptr()))
            for _, tensor in destination_items
        }
        source_owners = {
            (tensor.device, int(tensor.untyped_storage().data_ptr()))
            for _, tensor in source_items
        }
        if destination_owners & source_owners:
            raise ValueError("ratio-128 source and destination alias storage")

        for (destination_name, destination), (source_name, value) in zip(
            destination_items, source_items, strict=True
        ):
            if destination_name != source_name:
                raise RuntimeError("ratio-128 state copy order differs")
            destination.copy_(value)

    def reset(self) -> None:
        """Clear all payload and metadata so the state can serve a new batch."""

        self.latent.zero_()
        self.kv_state.zero_()
        self.score_state.fill_(float("-inf"))
        self._next_position.zero_()
        self._compressed_count.zero_()
        self._raw_positions.fill_(-1)
        self._compressed_group_starts.fill_(-1)
        self._state_positions.fill_(-1)

    def seed_decode_residency(
        self,
        *,
        start_pos: int,
        raw: torch.Tensor,
        compressed: torch.Tensor,
    ) -> None:
        """Seed an aligned synthetic BF16 residency for fixed-position decode.

        ``raw`` is the physical 128-slot ring payload. ``compressed`` contains
        exactly one row for every completed ratio-128 group. An aligned seed has
        no pending compressor group, so the FP32 pooling state is reset.
        """

        if (
            not isinstance(start_pos, int)
            or isinstance(start_pos, bool)
            or start_pos < COMPRESS_RATIO
            or start_pos >= self.max_seq_len
        ):
            raise ValueError(
                "seed start_pos must be an integer in [128, max_seq_len)"
            )
        if start_pos % COMPRESS_RATIO:
            raise ValueError("seed start_pos must be aligned to the ratio-128 boundary")

        completed = start_pos // COMPRESS_RATIO
        self._require_tensor(
            "seed raw",
            raw,
            (self.num_local_sequences, WINDOW_SIZE, LATENT_DIM),
            torch.bfloat16,
        )
        self._require_tensor(
            "seed compressed",
            compressed,
            (self.num_local_sequences, completed, LATENT_DIM),
            torch.bfloat16,
        )

        # Build the complete replacement before touching resident state. This
        # also makes seeding safe when a caller passes views of this state.
        latent = torch.zeros_like(self.latent)
        latent[:, :WINDOW_SIZE].copy_(raw)
        latent[:, WINDOW_SIZE : WINDOW_SIZE + completed].copy_(compressed)
        kv_state = torch.zeros_like(self.kv_state)
        score_state = torch.full_like(self.score_state, float("-inf"))

        absolute_raw = torch.arange(
            start_pos - WINDOW_SIZE,
            start_pos,
            dtype=torch.int64,
            device=self.device,
        )
        raw_slots = absolute_raw.remainder(WINDOW_SIZE)
        raw_positions = torch.full_like(self._raw_positions, -1)
        raw_positions.index_copy_(
            1,
            raw_slots,
            absolute_raw.unsqueeze(0).expand(self.num_local_sequences, -1),
        )
        compressed_starts = torch.full_like(self._compressed_group_starts, -1)
        starts = torch.arange(
            0,
            start_pos,
            COMPRESS_RATIO,
            dtype=torch.int64,
            device=self.device,
        )
        compressed_starts[:, :completed].copy_(
            starts.unsqueeze(0).expand(self.num_local_sequences, -1)
        )
        state_positions = torch.full_like(self._state_positions, -1)
        next_position = torch.full_like(self._next_position, start_pos)
        compressed_count = torch.full_like(self._compressed_count, completed)

        self.latent.copy_(latent)
        self.kv_state.copy_(kv_state)
        self.score_state.copy_(score_state)
        self._raw_positions.copy_(raw_positions)
        self._compressed_group_starts.copy_(compressed_starts)
        self._state_positions.copy_(state_positions)
        self._next_position.copy_(next_position)
        self._compressed_count.copy_(compressed_count)

    def _write_decode_nonboundary_fixed(
        self,
        raw_latent: torch.Tensor,
        projected_kv: torch.Tensor,
        adjusted_score: torch.Tensor,
        *,
        position: int,
        slot: int,
    ) -> None:
        """Commit one prevalidated non-boundary decode token without host sync."""

        self.raw[:, slot].copy_(raw_latent[:, 0])
        self.kv_state[:, slot].copy_(projected_kv[:, 0])
        self.score_state[:, slot].copy_(adjusted_score)
        self._raw_positions[:, slot].fill_(position)
        self._state_positions[:, slot].fill_(position)
        self._next_position.fill_(position + 1)

    def _write_decode_stateful_prevalidated(
        self,
        raw_latent: torch.Tensor,
        projected_kv: torch.Tensor,
        adjusted_score: torch.Tensor,
        *,
        position: torch.Tensor,
        boundary: bool,
        finalize_compressed: CompressionFinalizer,
    ) -> None:
        """Commit one prevalidated cursor-driven token without device value reads.

        ``boundary`` is fixed by the captured graph family. All position-derived
        slots remain device tensors so one graph can replay across many tokens.
        The caller validates the host schedule and advances the shared cursor
        only after every layer in the stage has consumed the current position.
        The fixed finalizer contract must be validated before capture; this hot
        path is intentionally not transactional if that finalizer raises.
        """

        batch = self.num_local_sequences
        shape = (batch, 1, LATENT_DIM)
        self._require_tensor("raw_latent", raw_latent, shape, torch.bfloat16)
        self._require_tensor("projected_kv", projected_kv, shape, torch.float32)
        self._require_tensor(
            "adjusted_score",
            adjusted_score,
            (batch, LATENT_DIM),
            torch.float32,
        )
        if (
            not isinstance(position, torch.Tensor)
            or tuple(position.shape) != (1,)
            or position.dtype != torch.int64
            or position.device != self.device
            or not position.is_contiguous()
        ):
            raise ValueError("stateful decode position must be contiguous INT64 [1]")
        if not isinstance(boundary, bool):
            raise TypeError("stateful ratio-128 boundary flag must be bool")
        if not callable(finalize_compressed):
            raise TypeError("stateful compression finalizer must be callable")

        slot = position.remainder(COMPRESS_RATIO)
        expanded_position = position.view(1, 1).expand(batch, 1)
        self.raw.index_copy_(1, slot, raw_latent)
        self.kv_state.index_copy_(1, slot, projected_kv)
        self.score_state.index_copy_(1, slot, adjusted_score.unsqueeze(1))
        self._raw_positions.index_copy_(1, slot, expanded_position)
        self._state_positions.index_copy_(1, slot, expanded_position)

        compressed_row = torch.div(position, COMPRESS_RATIO, rounding_mode="floor")
        if boundary:
            pooled = (
                self.kv_state * self.score_state.softmax(dim=1)
            ).sum(dim=1, keepdim=True)
            group_start = position + 1 - COMPRESS_RATIO
            finalized = finalize_compressed(pooled, group_start)
            self._require_tensor(
                "stateful finalized compressed latent",
                finalized,
                shape,
                torch.bfloat16,
            )
            self.compressed.index_copy_(1, compressed_row, finalized)
            self._compressed_group_starts.index_copy_(
                1,
                compressed_row,
                group_start.view(1, 1).expand(batch, 1),
            )
            compressed_row = compressed_row + 1

        self._compressed_count.copy_(compressed_row.expand(batch))
        self._next_position.copy_((position + 1).expand(batch))

    def metadata(self) -> dict[str, object]:
        """Return a JSON-safe snapshot suitable for correctness artifacts."""

        position = self.next_position
        snapshot: dict[str, object] = {
            "layer_id": self.layer_id,
            "num_local_sequences": self.num_local_sequences,
            "max_seq_len": self.max_seq_len,
            "window_size": WINDOW_SIZE,
            "compress_ratio": COMPRESS_RATIO,
            "latent_dim": LATENT_DIM,
            "latent_shape": list(self.latent.shape),
            "latent_dtype": str(self.latent.dtype),
            "compressor_state_dtype": str(self.kv_state.dtype),
            "next_position": self._next_position.cpu().tolist(),
            "compressed_count": self._compressed_count.cpu().tolist(),
            "compressor_pending_tokens": [position % COMPRESS_RATIO]
            * self.num_local_sequences,
            "raw_positions": self._raw_positions.cpu().tolist(),
            "compressed_group_starts": self._compressed_group_starts.cpu().tolist(),
            "compressor_state_positions": self._state_positions.cpu().tolist(),
        }
        json.dumps(snapshot, allow_nan=False)
        return snapshot

    def chronological_raw_positions(self) -> torch.Tensor:
        """Return raw-ring metadata ordered from oldest to newest token."""

        end = self.next_position
        start = max(0, end - WINDOW_SIZE)
        positions = torch.arange(start, end, dtype=torch.int64, device=self.device)
        slots = positions.remainder(WINDOW_SIZE)
        return self._raw_positions.index_select(1, slots)

    def prefill_write(
        self,
        raw_latent: torch.Tensor,
        *,
        projected_kv: torch.Tensor,
        projected_score: torch.Tensor,
        ape: torch.Tensor,
        finalize_compressed: CompressionFinalizer | None,
    ) -> CompressionWrite | None:
        """Write a position-zero prefill and preserve its incomplete group."""

        if self.next_position != 0:
            raise RuntimeError("prefill_write requires reset state at position zero")
        seqlen = self._validate_prefill_inputs(
            raw_latent, projected_kv, projected_score, ape
        )
        completed = seqlen // COMPRESS_RATIO
        cutoff = completed * COMPRESS_RATIO

        pooled = self._pool_complete_groups(
            projected_kv[:, :cutoff], projected_score[:, :cutoff], ape
        )
        finalized, result = self._run_finalizer(
            pooled,
            row_offset=0,
            group_start_positions=tuple(
                range(0, cutoff, COMPRESS_RATIO)
            ),
            finalize_compressed=finalize_compressed,
        )

        kept = min(seqlen, WINDOW_SIZE)
        absolute_positions = torch.arange(
            seqlen - kept, seqlen, dtype=torch.int64, device=self.device
        )
        raw_slots = absolute_positions.remainder(WINDOW_SIZE)
        self.raw.index_copy_(1, raw_slots, raw_latent[:, -kept:])
        self._raw_positions.index_copy_(
            1,
            raw_slots,
            absolute_positions.unsqueeze(0).expand(self.num_local_sequences, -1),
        )

        if finalized is not None:
            self.compressed[:, :completed].copy_(finalized)
            starts = torch.arange(
                0, cutoff, COMPRESS_RATIO, dtype=torch.int64, device=self.device
            )
            self._compressed_group_starts[:, :completed].copy_(
                starts.unsqueeze(0).expand(self.num_local_sequences, -1)
            )

        remainder = seqlen - cutoff
        if remainder:
            self.kv_state[:, :remainder].copy_(projected_kv[:, cutoff:])
            self.score_state[:, :remainder].copy_(
                projected_score[:, cutoff:] + ape[:remainder]
            )
            state_positions = torch.arange(
                cutoff, seqlen, dtype=torch.int64, device=self.device
            )
            self._state_positions[:, :remainder].copy_(
                state_positions.unsqueeze(0).expand(self.num_local_sequences, -1)
            )

        self._next_position.fill_(seqlen)
        self._compressed_count.fill_(completed)
        return result

    def decode_write(
        self,
        raw_latent: torch.Tensor,
        *,
        projected_kv: torch.Tensor,
        projected_score: torch.Tensor,
        ape: torch.Tensor,
        finalize_compressed: CompressionFinalizer | None,
    ) -> CompressionWrite | None:
        """Append one decode token, advancing the ring and ratio-128 state."""

        position = self.next_position
        if position == 0:
            raise RuntimeError("decode_write requires a preceding prefill_write")
        if position >= self.max_seq_len:
            raise RuntimeError("static KV capacity is exhausted")
        self._validate_decode_inputs(raw_latent, projected_kv, projected_score, ape)

        expected_compressed = position // COMPRESS_RATIO
        if not bool(torch.all(self._compressed_count == expected_compressed).item()):
            raise RuntimeError("compressed-row metadata is inconsistent")
        pending = position % COMPRESS_RATIO
        if pending:
            expected_positions = torch.arange(
                position - pending,
                position,
                dtype=torch.int64,
                device=self.device,
            )
            if not bool(
                torch.all(
                    self._state_positions[:, :pending]
                    == expected_positions.unsqueeze(0)
                ).item()
            ):
                raise RuntimeError("compressor-state metadata is inconsistent")

        raw_slot = position % WINDOW_SIZE
        state_slot = position % COMPRESS_RATIO
        adjusted_score = projected_score[:, 0] + ape[state_slot]
        should_compress = (position + 1) % COMPRESS_RATIO == 0
        finalized: torch.Tensor | None = None
        result: CompressionWrite | None = None
        if should_compress:
            candidate_kv = self.kv_state.clone()
            candidate_score = self.score_state.clone()
            candidate_kv[:, state_slot].copy_(projected_kv[:, 0])
            candidate_score[:, state_slot].copy_(adjusted_score)
            pooled = (
                candidate_kv * candidate_score.softmax(dim=1)
            ).sum(dim=1, keepdim=True)
            group_start = position + 1 - COMPRESS_RATIO
            finalized, result = self._run_finalizer(
                pooled,
                row_offset=expected_compressed,
                group_start_positions=(group_start,),
                finalize_compressed=finalize_compressed,
            )

        self.raw[:, raw_slot].copy_(raw_latent[:, 0])
        self._raw_positions[:, raw_slot].fill_(position)
        self.kv_state[:, state_slot].copy_(projected_kv[:, 0])
        self.score_state[:, state_slot].copy_(adjusted_score)
        self._state_positions[:, state_slot].fill_(position)

        if finalized is not None:
            row = expected_compressed
            self.compressed[:, row : row + 1].copy_(finalized)
            self._compressed_group_starts[:, row].fill_(
                position + 1 - COMPRESS_RATIO
            )
            self._compressed_count.fill_(expected_compressed + 1)

        self._next_position.fill_(position + 1)
        return result

    def _validate_prefill_inputs(
        self,
        raw_latent: torch.Tensor,
        projected_kv: torch.Tensor,
        projected_score: torch.Tensor,
        ape: torch.Tensor,
    ) -> int:
        if raw_latent.ndim != 3:
            raise ValueError("raw_latent must have shape [batch, seqlen, 512]")
        seqlen = raw_latent.shape[1]
        if not 0 < seqlen <= self.max_seq_len:
            raise ValueError("prefill seqlen must fit the non-empty static capacity")
        shape = (self.num_local_sequences, seqlen, LATENT_DIM)
        self._require_tensor("raw_latent", raw_latent, shape, torch.bfloat16)
        self._require_tensor("projected_kv", projected_kv, shape, torch.float32)
        self._require_tensor("projected_score", projected_score, shape, torch.float32)
        self._require_ape(ape)
        return seqlen

    def _validate_decode_inputs(
        self,
        raw_latent: torch.Tensor,
        projected_kv: torch.Tensor,
        projected_score: torch.Tensor,
        ape: torch.Tensor,
    ) -> None:
        shape = (self.num_local_sequences, 1, LATENT_DIM)
        self._require_tensor("raw_latent", raw_latent, shape, torch.bfloat16)
        self._require_tensor("projected_kv", projected_kv, shape, torch.float32)
        self._require_tensor("projected_score", projected_score, shape, torch.float32)
        self._require_ape(ape)

    def _require_ape(self, ape: torch.Tensor) -> None:
        self._require_tensor(
            "ape", ape, (COMPRESS_RATIO, LATENT_DIM), torch.float32
        )

    def _require_tensor(
        self,
        name: str,
        value: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
        if value.dtype != dtype:
            raise TypeError(f"{name} dtype {value.dtype} != {dtype}")
        if value.device != self.device:
            raise ValueError(f"{name} device {value.device} != {self.device}")

    def _pool_complete_groups(
        self,
        projected_kv: torch.Tensor,
        projected_score: torch.Tensor,
        ape: torch.Tensor,
    ) -> torch.Tensor:
        completed = projected_kv.shape[1] // COMPRESS_RATIO
        if completed == 0:
            return projected_kv.new_empty(
                self.num_local_sequences, 0, LATENT_DIM
            )
        grouped_kv = projected_kv.reshape(
            self.num_local_sequences, completed, COMPRESS_RATIO, LATENT_DIM
        )
        grouped_score = projected_score.reshape_as(grouped_kv)
        grouped_score = grouped_score + ape.view(1, 1, COMPRESS_RATIO, LATENT_DIM)
        return (grouped_kv * grouped_score.softmax(dim=2)).sum(dim=2)

    def _run_finalizer(
        self,
        pooled: torch.Tensor,
        *,
        row_offset: int,
        group_start_positions: tuple[int, ...],
        finalize_compressed: CompressionFinalizer | None,
    ) -> tuple[torch.Tensor | None, CompressionWrite | None]:
        completed = pooled.shape[1]
        if completed == 0:
            return None, None
        if finalize_compressed is None:
            raise RuntimeError(
                "completed compression rows require RMSNorm/RoPE/quant finalization"
            )
        starts = torch.tensor(
            group_start_positions, dtype=torch.int64, device=self.device
        )
        finalized = finalize_compressed(pooled, starts)
        self._require_tensor(
            "finalized compressed latent",
            finalized,
            (self.num_local_sequences, completed, LATENT_DIM),
            torch.bfloat16,
        )
        row_indices = tuple(range(row_offset, row_offset + completed))
        return finalized, CompressionWrite(
            row_indices=row_indices,
            group_start_positions=group_start_positions,
            pooled=pooled,
        )


__all__ = [
    "COMPRESS_RATIO",
    "LATENT_DIM",
    "WINDOW_SIZE",
    "CompressionFinalizer",
    "CompressionWrite",
    "StaticLayerKV",
]
