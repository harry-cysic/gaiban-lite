"""Static ratio-4 decode state for the direct runtime.

Ratio-4 layers keep two independent overlapping compressors: the main 512-wide
latent compressor and the 128-wide indexer compressor.  Each compressor stores
the previous four-token window beside the current four-token window, matching
the checkpoint reference layout.  This module owns only storage and metadata;
the attention implementation owns the weighted pooling and finalization math.

DeepSeek-V4-Flash port note: every layout constant below is unchanged from Pro
(window 128, ratio 4, latent 512 == head_dim, index 128 == index_head_dim per
model_contract.EXPECTED_RATIO4_CONFIG); the indexer top-k (512 in Flash) never
appears here -- it lives in the attention config.  SUPPORTED_RATIO4_LAYER_IDS
derives from this repo's frozen Flash layer specs (even layers 2..42).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import torch

from .model_contract import SUPPORTED_LAYER_SPECS


WINDOW_SIZE = 128
COMPRESS_RATIO = 4
LATENT_DIM = 512
INDEX_DIM = 128
MAIN_PROJECTED_DIM = 2 * LATENT_DIM
INDEX_PROJECTED_DIM = 2 * INDEX_DIM
OVERLAP_STATE_ROWS = 2 * COMPRESS_RATIO
SUPPORTED_RATIO4_LAYER_IDS = tuple(
    layer_id
    for layer_id, specification in SUPPORTED_LAYER_SPECS.items()
    if specification["compress_ratio"] == 4
)

CompressionFinalizer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class StaticRatio4KV:
    """Batch-local, non-paged KV and overlap state for one ratio-4 layer."""

    def __init__(
        self,
        *,
        num_local_sequences: int,
        max_seq_len: int,
        layer_id: int = 2,
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
            or max_seq_len < WINDOW_SIZE
            or max_seq_len % COMPRESS_RATIO
        ):
            raise ValueError("max_seq_len must be a multiple of four and at least 128")
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id not in SUPPORTED_RATIO4_LAYER_IDS
        ):
            raise ValueError(
                "ratio-4 direct state requires an integer frozen ratio-4 layer_id, "
                f"got {layer_id!r}"
            )

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
        self.indexer_kv = torch.zeros(
            num_local_sequences,
            self.compressed_capacity,
            INDEX_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.main_kv_state = torch.zeros(
            num_local_sequences,
            OVERLAP_STATE_ROWS,
            MAIN_PROJECTED_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        self.main_score_state = torch.full_like(
            self.main_kv_state, float("-inf")
        )
        self.index_kv_state = torch.zeros(
            num_local_sequences,
            OVERLAP_STATE_ROWS,
            INDEX_PROJECTED_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        self.index_score_state = torch.full_like(
            self.index_kv_state, float("-inf")
        )

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
        self._main_state_positions = torch.full(
            (num_local_sequences, OVERLAP_STATE_ROWS),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self._index_state_positions = torch.full_like(
            self._main_state_positions, -1
        )

    @property
    def raw(self) -> torch.Tensor:
        return self.latent[:, :WINDOW_SIZE]

    @property
    def compressed(self) -> torch.Tensor:
        return self.latent[:, WINDOW_SIZE:]

    @property
    def next_position(self) -> int:
        if not bool(torch.all(self._next_position == self._next_position[0]).item()):
            raise RuntimeError("local sequences no longer share one next position")
        return int(self._next_position[0].item())

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in self._owned_tensors()
        )

    def _owned_tensor_items(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("latent", self.latent),
            ("indexer_kv", self.indexer_kv),
            ("main_kv_state", self.main_kv_state),
            ("main_score_state", self.main_score_state),
            ("index_kv_state", self.index_kv_state),
            ("index_score_state", self.index_score_state),
            ("next_position", self._next_position),
            ("compressed_count", self._compressed_count),
            ("raw_positions", self._raw_positions),
            ("compressed_group_starts", self._compressed_group_starts),
            ("main_state_positions", self._main_state_positions),
            ("index_state_positions", self._index_state_positions),
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
            "indexer_kv": (
                (batch, self.compressed_capacity, INDEX_DIM),
                torch.bfloat16,
            ),
            "main_kv_state": (
                (batch, OVERLAP_STATE_ROWS, MAIN_PROJECTED_DIM),
                torch.float32,
            ),
            "main_score_state": (
                (batch, OVERLAP_STATE_ROWS, MAIN_PROJECTED_DIM),
                torch.float32,
            ),
            "index_kv_state": (
                (batch, OVERLAP_STATE_ROWS, INDEX_PROJECTED_DIM),
                torch.float32,
            ),
            "index_score_state": (
                (batch, OVERLAP_STATE_ROWS, INDEX_PROJECTED_DIM),
                torch.float32,
            ),
            "next_position": ((batch,), torch.int64),
            "compressed_count": ((batch,), torch.int64),
            "raw_positions": ((batch, WINDOW_SIZE), torch.int64),
            "compressed_group_starts": (
                (batch, self.compressed_capacity),
                torch.int64,
            ),
            "main_state_positions": (
                (batch, OVERLAP_STATE_ROWS),
                torch.int64,
            ),
            "index_state_positions": (
                (batch, OVERLAP_STATE_ROWS),
                torch.int64,
            ),
        }
        items = self._owned_tensor_items()
        if tuple(name for name, _ in items) != tuple(expected):
            raise RuntimeError(f"{label} ratio-4 state ownership contract differs")

        storage_owners: dict[tuple[torch.device, int], str] = {}
        for name, tensor in items:
            shape, dtype = expected[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{label} ratio-4 state {name} must be a tensor")
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{label} ratio-4 state {name} shape "
                    f"{tuple(tensor.shape)} != {shape}"
                )
            if tensor.dtype != dtype:
                raise TypeError(
                    f"{label} ratio-4 state {name} dtype {tensor.dtype} != {dtype}"
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"{label} ratio-4 state {name} device "
                    f"{tensor.device} != {self.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{label} ratio-4 state {name} must be contiguous")
            owner = (tensor.device, int(tensor.untyped_storage().data_ptr()))
            previous = storage_owners.setdefault(owner, name)
            if previous != name:
                raise ValueError(
                    f"{label} ratio-4 state tensors {previous} and {name} "
                    "alias one storage"
                )
        return items

    def copy_from(self, source: "StaticRatio4KV") -> None:
        """Copy a complete pre-step state after validating both owners."""

        if not isinstance(source, StaticRatio4KV):
            raise TypeError("ratio-4 state source must be StaticRatio4KV")
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
                f"ratio-4 state identity {source_identity} cannot copy into {identity}"
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
            raise ValueError("ratio-4 source and destination alias storage")

        for (destination_name, destination), (source_name, value) in zip(
            destination_items, source_items, strict=True
        ):
            if destination_name != source_name:
                raise RuntimeError("ratio-4 state copy order differs")
            destination.copy_(value)

    def reset(self) -> None:
        self.latent.zero_()
        self.indexer_kv.zero_()
        self.main_kv_state.zero_()
        self.main_score_state.fill_(float("-inf"))
        self.index_kv_state.zero_()
        self.index_score_state.fill_(float("-inf"))
        self._next_position.zero_()
        self._compressed_count.zero_()
        self._raw_positions.fill_(-1)
        self._compressed_group_starts.fill_(-1)
        self._main_state_positions.fill_(-1)
        self._index_state_positions.fill_(-1)

    def seed_zero_decode(
        self,
        start_pos: int,
        *,
        main_ape: torch.Tensor,
        index_ape: torch.Tensor,
    ) -> None:
        """Seed an all-zero residency for one fixed decode position.

        The payload represents a valid zero-equivalent history.  Projected KV
        is zero, while live score slots retain the checkpoint APE for their
        phase so a phase-3 replay executes the real eight-row softmax.
        """

        if (
            not isinstance(start_pos, int)
            or isinstance(start_pos, bool)
            or start_pos < WINDOW_SIZE
            or start_pos >= self.max_seq_len
        ):
            raise ValueError("start_pos must be an integer in [128, max_seq_len)")
        self._require_ape("main", main_ape, (COMPRESS_RATIO, MAIN_PROJECTED_DIM))
        self._require_ape(
            "index", index_ape, (COMPRESS_RATIO, INDEX_PROJECTED_DIM)
        )

        self.reset()
        self._seed_decode_metadata(start_pos)
        phase = start_pos % COMPRESS_RATIO
        self.main_score_state[:, :COMPRESS_RATIO].copy_(main_ape.unsqueeze(0))
        self.index_score_state[:, :COMPRESS_RATIO].copy_(index_ape.unsqueeze(0))
        if phase:
            state_slice = slice(COMPRESS_RATIO, COMPRESS_RATIO + phase)
            self.main_score_state[:, state_slice].copy_(
                main_ape[:phase].unsqueeze(0)
            )
            self.index_score_state[:, state_slice].copy_(
                index_ape[:phase].unsqueeze(0)
            )

    def seed_decode_payload(
        self,
        start_pos: int,
        *,
        raw: torch.Tensor,
        compressed: torch.Tensor,
        indexer_kv: torch.Tensor,
        main_kv_state: torch.Tensor,
        main_score_state: torch.Tensor,
        index_kv_state: torch.Tensor,
        index_score_state: torch.Tensor,
    ) -> None:
        """Atomically install one complete, independently owned decode state."""

        if (
            not isinstance(start_pos, int)
            or isinstance(start_pos, bool)
            or start_pos < WINDOW_SIZE
            or start_pos >= self.max_seq_len
        ):
            raise ValueError("start_pos must be an integer in [128, max_seq_len)")
        payload = {
            "raw": (raw, self.raw),
            "compressed": (compressed, self.compressed),
            "indexer_kv": (indexer_kv, self.indexer_kv),
            "main_kv_state": (main_kv_state, self.main_kv_state),
            "main_score_state": (main_score_state, self.main_score_state),
            "index_kv_state": (index_kv_state, self.index_kv_state),
            "index_score_state": (index_score_state, self.index_score_state),
        }
        owned_storage = {
            tensor.untyped_storage().data_ptr() for tensor in self._owned_tensors()
        }
        for name, (value, destination) in payload.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} payload must be a tensor")
            if tuple(value.shape) != tuple(destination.shape):
                raise ValueError(
                    f"{name} payload shape {tuple(value.shape)} != "
                    f"{tuple(destination.shape)}"
                )
            if value.dtype != destination.dtype or value.device != self.device:
                raise ValueError(
                    f"{name} payload must use {destination.dtype} on {self.device}"
                )
            if value.untyped_storage().data_ptr() in owned_storage:
                raise ValueError(f"{name} payload must not alias destination state")

        finite_names = (
            "raw",
            "compressed",
            "indexer_kv",
            "main_kv_state",
            "index_kv_state",
        )
        for name in finite_names:
            if not bool(torch.isfinite(payload[name][0]).all().item()):
                raise ValueError(f"{name} payload must be finite")

        active_overlap = COMPRESS_RATIO + start_pos % COMPRESS_RATIO
        for name in ("main_score_state", "index_score_state"):
            value = payload[name][0]
            if not bool(torch.isfinite(value[:, :active_overlap]).all().item()):
                raise ValueError(f"active {name} rows must be finite")
            valid = torch.isfinite(value) | torch.isneginf(value)
            if not bool(valid.all().item()):
                raise ValueError(
                    f"inactive {name} rows may be finite or negative infinity"
                )

        self.reset()
        self._seed_decode_metadata(start_pos)
        for value, destination in payload.values():
            destination.copy_(value)

    def _seed_decode_metadata(self, start_pos: int) -> None:
        """Populate decode metadata after payload storage has been reset."""

        completed = start_pos // COMPRESS_RATIO
        phase = start_pos % COMPRESS_RATIO
        group_start = start_pos - phase

        absolute_raw = torch.arange(
            start_pos - WINDOW_SIZE,
            start_pos,
            dtype=torch.int64,
            device=self.device,
        )
        raw_slots = absolute_raw.remainder(WINDOW_SIZE)
        self._raw_positions.index_copy_(
            1,
            raw_slots,
            absolute_raw.unsqueeze(0).expand(self.num_local_sequences, -1),
        )

        starts = torch.arange(
            0,
            completed * COMPRESS_RATIO,
            COMPRESS_RATIO,
            dtype=torch.int64,
            device=self.device,
        )
        self._compressed_group_starts[:, :completed].copy_(
            starts.unsqueeze(0).expand(self.num_local_sequences, -1)
        )

        previous = torch.arange(
            group_start - COMPRESS_RATIO,
            group_start,
            dtype=torch.int64,
            device=self.device,
        )
        self._main_state_positions[:, :COMPRESS_RATIO].copy_(
            previous.unsqueeze(0).expand(self.num_local_sequences, -1)
        )
        self._index_state_positions[:, :COMPRESS_RATIO].copy_(
            previous.unsqueeze(0).expand(self.num_local_sequences, -1)
        )

        if phase:
            pending = torch.arange(
                group_start,
                start_pos,
                dtype=torch.int64,
                device=self.device,
            )
            state_slice = slice(COMPRESS_RATIO, COMPRESS_RATIO + phase)
            expanded = pending.unsqueeze(0).expand(self.num_local_sequences, -1)
            self._main_state_positions[:, state_slice].copy_(expanded)
            self._index_state_positions[:, state_slice].copy_(expanded)

        self._next_position.fill_(start_pos)
        self._compressed_count.fill_(completed)

    def _write_decode_stateful_prevalidated(
        self,
        raw_latent: torch.Tensor,
        main_projected: torch.Tensor,
        main_adjusted_score: torch.Tensor,
        index_projected: torch.Tensor,
        index_adjusted_score: torch.Tensor,
        *,
        position: torch.Tensor,
        boundary: bool,
        group_start_frequencies: torch.Tensor,
        main_finalizer: CompressionFinalizer,
        index_finalizer: CompressionFinalizer,
    ) -> None:
        """Commit one cursor-driven ratio-4 token without device value reads.

        ``boundary`` is fixed by the captured graph family. A boundary always
        advances both overlap compressors because this entry point serves a
        continuous state chain, not an idempotent fixed-position graph replay.
        The caller validates the position/family relationship and finalizer
        contracts before capture; a finalizer failure is not transactional.
        """

        batch = self.num_local_sequences
        self._require_tensor(
            "raw_latent",
            raw_latent,
            (batch, 1, LATENT_DIM),
            torch.bfloat16,
        )
        self._require_tensor(
            "main_projected",
            main_projected,
            (batch, 1, MAIN_PROJECTED_DIM),
            torch.float32,
        )
        self._require_tensor(
            "main_adjusted_score",
            main_adjusted_score,
            (batch, MAIN_PROJECTED_DIM),
            torch.float32,
        )
        self._require_tensor(
            "index_projected",
            index_projected,
            (batch, 1, INDEX_PROJECTED_DIM),
            torch.float32,
        )
        self._require_tensor(
            "index_adjusted_score",
            index_adjusted_score,
            (batch, INDEX_PROJECTED_DIM),
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
            raise TypeError("stateful ratio-4 boundary flag must be bool")
        if (
            not isinstance(group_start_frequencies, torch.Tensor)
            or group_start_frequencies.ndim == 0
            or group_start_frequencies.shape[0] != 1
            or group_start_frequencies.device != self.device
            or not group_start_frequencies.is_contiguous()
        ):
            raise ValueError(
                "stateful group-start frequencies must be contiguous device-local "
                "with leading extent one"
            )
        if not callable(main_finalizer) or not callable(index_finalizer):
            raise TypeError("stateful ratio-4 finalizers must be callable")

        phase = position.remainder(COMPRESS_RATIO)
        raw_slot = position.remainder(WINDOW_SIZE)
        overlap_slot = phase + COMPRESS_RATIO
        expanded_position = position.view(1, 1).expand(batch, 1)

        self.raw.index_copy_(1, raw_slot, raw_latent)
        self.main_kv_state.index_copy_(1, overlap_slot, main_projected)
        self.main_score_state.index_copy_(
            1, overlap_slot, main_adjusted_score.unsqueeze(1)
        )
        self.index_kv_state.index_copy_(1, overlap_slot, index_projected)
        self.index_score_state.index_copy_(
            1, overlap_slot, index_adjusted_score.unsqueeze(1)
        )
        self._raw_positions.index_copy_(1, raw_slot, expanded_position)
        self._main_state_positions.index_copy_(
            1, overlap_slot, expanded_position
        )
        self._index_state_positions.index_copy_(
            1, overlap_slot, expanded_position
        )

        compressed_row = torch.div(
            position, COMPRESS_RATIO, rounding_mode="floor"
        )
        if boundary:
            main_values = torch.cat(
                (
                    self.main_kv_state[:, :COMPRESS_RATIO, :LATENT_DIM],
                    self.main_kv_state[:, COMPRESS_RATIO:, LATENT_DIM:],
                ),
                dim=1,
            )
            main_scores = torch.cat(
                (
                    self.main_score_state[:, :COMPRESS_RATIO, :LATENT_DIM],
                    self.main_score_state[:, COMPRESS_RATIO:, LATENT_DIM:],
                ),
                dim=1,
            )
            main_pooled = (
                main_values * main_scores.softmax(dim=1)
            ).sum(dim=1, keepdim=True)
            finalized_main = main_finalizer(
                main_pooled, group_start_frequencies
            )
            self._require_tensor(
                "stateful finalized main latent",
                finalized_main,
                (batch, 1, LATENT_DIM),
                torch.bfloat16,
            )

            index_values = torch.cat(
                (
                    self.index_kv_state[:, :COMPRESS_RATIO, :INDEX_DIM],
                    self.index_kv_state[:, COMPRESS_RATIO:, INDEX_DIM:],
                ),
                dim=1,
            )
            index_scores = torch.cat(
                (
                    self.index_score_state[:, :COMPRESS_RATIO, :INDEX_DIM],
                    self.index_score_state[:, COMPRESS_RATIO:, INDEX_DIM:],
                ),
                dim=1,
            )
            index_pooled = (
                index_values * index_scores.softmax(dim=1)
            ).sum(dim=1, keepdim=True)
            finalized_index = index_finalizer(
                index_pooled, group_start_frequencies
            )
            self._require_tensor(
                "stateful finalized index latent",
                finalized_index,
                (batch, 1, INDEX_DIM),
                torch.bfloat16,
            )

            self.compressed.index_copy_(1, compressed_row, finalized_main)
            self.indexer_kv.index_copy_(1, compressed_row, finalized_index)
            group_start = position + 1 - COMPRESS_RATIO
            self._compressed_group_starts.index_copy_(
                1,
                compressed_row,
                group_start.view(1, 1).expand(batch, 1),
            )

            self.main_kv_state[:, :COMPRESS_RATIO].copy_(
                self.main_kv_state[:, COMPRESS_RATIO:]
            )
            self.main_score_state[:, :COMPRESS_RATIO].copy_(
                self.main_score_state[:, COMPRESS_RATIO:]
            )
            self.index_kv_state[:, :COMPRESS_RATIO].copy_(
                self.index_kv_state[:, COMPRESS_RATIO:]
            )
            self.index_score_state[:, :COMPRESS_RATIO].copy_(
                self.index_score_state[:, COMPRESS_RATIO:]
            )
            self._main_state_positions[:, :COMPRESS_RATIO].copy_(
                self._main_state_positions[:, COMPRESS_RATIO:]
            )
            self._index_state_positions[:, :COMPRESS_RATIO].copy_(
                self._index_state_positions[:, COMPRESS_RATIO:]
            )

        compressed_after = torch.div(
            position + 1, COMPRESS_RATIO, rounding_mode="floor"
        )
        self._compressed_count.copy_(compressed_after.expand(batch))
        self._next_position.copy_((position + 1).expand(batch))

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

    def _require_ape(
        self,
        name: str,
        value: torch.Tensor,
        shape: tuple[int, int],
    ) -> None:
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} APE shape {tuple(value.shape)} != {shape}")
        if value.dtype != torch.float32 or value.device != self.device:
            raise ValueError(f"{name} APE must be FP32 on {self.device}")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} APE must be finite")

    def metadata(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "layer_id": self.layer_id,
            "num_local_sequences": self.num_local_sequences,
            "max_seq_len": self.max_seq_len,
            "window_size": WINDOW_SIZE,
            "compress_ratio": COMPRESS_RATIO,
            "latent_shape": list(self.latent.shape),
            "indexer_kv_shape": list(self.indexer_kv.shape),
            "main_overlap_shape": list(self.main_kv_state.shape),
            "index_overlap_shape": list(self.index_kv_state.shape),
            "next_position": self._next_position.cpu().tolist(),
            "compressed_count": self._compressed_count.cpu().tolist(),
            "raw_positions": self._raw_positions.cpu().tolist(),
            "compressed_group_starts": self._compressed_group_starts.cpu().tolist(),
            "main_state_positions": self._main_state_positions.cpu().tolist(),
            "index_state_positions": self._index_state_positions.cpu().tolist(),
        }
        json.dumps(value, allow_nan=False)
        return value

    def chronological_raw_positions(self) -> torch.Tensor:
        """Return raw-ring metadata from the oldest through newest token."""

        end = self.next_position
        positions = torch.arange(
            max(0, end - WINDOW_SIZE),
            end,
            dtype=torch.int64,
            device=self.device,
        )
        return self._raw_positions.index_select(1, positions.remainder(WINDOW_SIZE))


__all__ = [
    "COMPRESS_RATIO",
    "CompressionFinalizer",
    "INDEX_DIM",
    "INDEX_PROJECTED_DIM",
    "LATENT_DIM",
    "MAIN_PROJECTED_DIM",
    "OVERLAP_STATE_ROWS",
    "SUPPORTED_RATIO4_LAYER_IDS",
    "StaticRatio4KV",
    "WINDOW_SIZE",
]
