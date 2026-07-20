"""Fixed-capacity pure sliding-window KV state for the direct runtime.

Flash L0/L1 (``compress_ratio == 0``) attention keeps only a 128-slot raw
latent ring -- reference ``model.py:473`` sizes the cache as
``kv_cache_size = window_size + (max_seq_len // ratio if ratio else 0)``,
which is exactly ``window_size`` when the ratio is zero.  There is no
compressor and no compressed tail, so this state is deliberately much
smaller than :class:`dsv4_direct.static_kv.StaticLayerKV`.

Ring semantics copied from the reference forward:
- prefill (``model.py:518-523``): position ``p`` lands in slot ``p % window``;
  when ``seqlen > window`` only the last ``window`` tokens are retained
  (``cutoff = seqlen % win`` split is equivalent to the ``p % win`` mapping).
- decode (``model.py:530``): ``kv_cache[:, start_pos % win] = kv``.
"""

from __future__ import annotations

import json

import torch

from .static_kv import LATENT_DIM, WINDOW_SIZE


class StaticWindowKV:
    """Batch-local, non-paged raw-ring KV state for one window-only layer.

    Every local sequence shares one advancing position, matching the fixed
    shape E0-style correctness harnesses.
    """

    def __init__(
        self,
        *,
        num_local_sequences: int,
        max_seq_len: int,
        layer_id: int = 0,
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
        ):
            raise ValueError(
                f"max_seq_len must be an integer >= window size {WINDOW_SIZE}"
            )
        if not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0:
            raise ValueError("layer_id must be a non-negative integer")

        self.num_local_sequences = num_local_sequences
        self.max_seq_len = max_seq_len
        self.layer_id = layer_id
        self.device = torch.device(device)

        # The whole latent is the raw ring: reference model.py:473-474 with
        # compress_ratio == 0 registers kv_cache of window_size rows only.
        self.latent = torch.zeros(
            num_local_sequences,
            WINDOW_SIZE,
            LATENT_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._next_position = torch.zeros(
            num_local_sequences, dtype=torch.int64, device=self.device
        )
        self._raw_positions = torch.full(
            (num_local_sequences, WINDOW_SIZE),
            -1,
            dtype=torch.int64,
            device=self.device,
        )

    @property
    def raw(self) -> torch.Tensor:
        return self.latent

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
            for tensor in (self.latent, self._next_position, self._raw_positions)
        )

    def _owned_tensor_items(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("latent", self.latent),
            ("next_position", self._next_position),
            ("raw_positions", self._raw_positions),
        )

    def _owned_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(tensor for _, tensor in self._owned_tensor_items())

    def _validate_owned_tensor_contract(
        self, *, label: str
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        batch = self.num_local_sequences
        expected = {
            "latent": ((batch, WINDOW_SIZE, LATENT_DIM), torch.bfloat16),
            "next_position": ((batch,), torch.int64),
            "raw_positions": ((batch, WINDOW_SIZE), torch.int64),
        }
        items = self._owned_tensor_items()
        if tuple(name for name, _ in items) != tuple(expected):
            raise RuntimeError(f"{label} window state ownership contract differs")
        storage_owners: dict[tuple[torch.device, int], str] = {}
        for name, tensor in items:
            shape, dtype = expected[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{label} window state {name} shape "
                    f"{tuple(tensor.shape)} != {shape}"
                )
            if tensor.dtype != dtype:
                raise TypeError(
                    f"{label} window state {name} dtype {tensor.dtype} != {dtype}"
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"{label} window state {name} device "
                    f"{tensor.device} != {self.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{label} window state {name} must be contiguous")
            owner = (tensor.device, int(tensor.untyped_storage().data_ptr()))
            previous = storage_owners.setdefault(owner, name)
            if previous != name:
                raise ValueError(
                    f"{label} window state tensors {previous} and {name} "
                    "alias one storage"
                )
        return items

    def copy_from(self, source: "StaticWindowKV") -> None:
        """Copy one complete pre-step state after validating both owners."""

        if not isinstance(source, StaticWindowKV):
            raise TypeError("window state source must be StaticWindowKV")
        identity = (self.num_local_sequences, self.max_seq_len, self.layer_id, self.device)
        source_identity = (
            source.num_local_sequences,
            source.max_seq_len,
            source.layer_id,
            source.device,
        )
        if identity != source_identity:
            raise ValueError(
                f"window state identity {source_identity} cannot copy into {identity}"
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
            raise ValueError("window source and destination alias storage")
        for (destination_name, destination), (source_name, value) in zip(
            destination_items, source_items, strict=True
        ):
            if destination_name != source_name:
                raise RuntimeError("window state copy order differs")
            destination.copy_(value)

    def seed_decode_residency(
        self,
        *,
        start_pos: int,
        raw: torch.Tensor,
    ) -> None:
        """Seed a synthetic BF16 full-ring residency for fixed-position decode.

        Window counterpart of ``StaticLayerKV.seed_decode_residency``: with
        compress_ratio == 0 the state is the raw ring alone, so the seed is
        the physical 128-slot ring payload for positions
        ``[start_pos - 128, start_pos)`` (reference model.py:530 ring
        mapping ``position % window``).
        """

        if (
            not isinstance(start_pos, int)
            or isinstance(start_pos, bool)
            or start_pos < WINDOW_SIZE
            or start_pos >= self.max_seq_len
        ):
            raise ValueError(
                "seed start_pos must be an integer in [128, max_seq_len)"
            )
        self._require_tensor(
            "seed raw",
            raw,
            (self.num_local_sequences, WINDOW_SIZE, LATENT_DIM),
            torch.bfloat16,
        )
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
        latent = torch.zeros_like(self.latent)
        latent.index_copy_(1, raw_slots, raw)

        self.latent.copy_(latent)
        self._raw_positions.copy_(raw_positions)
        self._next_position.fill_(start_pos)

    def _write_decode_stateful_prevalidated(
        self,
        raw_latent: torch.Tensor,
        *,
        position: torch.Tensor,
    ) -> None:
        """Commit one prevalidated cursor-driven token without device value reads.

        Window counterpart of ``StaticLayerKV._write_decode_stateful_prevalidated``
        minus every compressor write: the reference decode branch for
        compress_ratio == 0 performs only ``kv_cache[:, start_pos % win] = kv``
        (model.py:530; the compressor call at :531-532 is guarded by
        ``if self.compress_ratio:``).  All position-derived slots remain device
        tensors so one captured graph can replay across many tokens.
        """

        batch = self.num_local_sequences
        self._require_tensor(
            "raw_latent", raw_latent, (batch, 1, LATENT_DIM), torch.bfloat16
        )
        if (
            not isinstance(position, torch.Tensor)
            or tuple(position.shape) != (1,)
            or position.dtype != torch.int64
            or position.device != self.device
            or not position.is_contiguous()
        ):
            raise ValueError("stateful decode position must be contiguous INT64 [1]")
        slot = position.remainder(WINDOW_SIZE)
        expanded_position = position.view(1, 1).expand(batch, 1)
        self.latent.index_copy_(1, slot, raw_latent)
        self._raw_positions.index_copy_(1, slot, expanded_position)
        self._next_position.copy_((position + 1).expand(batch))

    def reset(self) -> None:
        """Clear payload and metadata so the state can serve a new batch."""

        self.latent.zero_()
        self._next_position.zero_()
        self._raw_positions.fill_(-1)

    def chronological_raw_positions(self) -> torch.Tensor:
        """Return raw-ring metadata ordered from oldest to newest token."""

        end = self.next_position
        start = max(0, end - WINDOW_SIZE)
        positions = torch.arange(start, end, dtype=torch.int64, device=self.device)
        slots = positions.remainder(WINDOW_SIZE)
        return self._raw_positions.index_select(1, slots)

    def metadata(self) -> dict[str, object]:
        """Return a JSON-safe snapshot suitable for correctness artifacts."""

        snapshot: dict[str, object] = {
            "layer_id": self.layer_id,
            "num_local_sequences": self.num_local_sequences,
            "max_seq_len": self.max_seq_len,
            "window_size": WINDOW_SIZE,
            "compress_ratio": 0,
            "latent_dim": LATENT_DIM,
            "latent_shape": list(self.latent.shape),
            "latent_dtype": str(self.latent.dtype),
            "next_position": self._next_position.cpu().tolist(),
            "raw_positions": self._raw_positions.cpu().tolist(),
        }
        json.dumps(snapshot, allow_nan=False)
        return snapshot

    def prefill_write(self, raw_latent: torch.Tensor) -> None:
        """Write a position-zero prefill of any length up to capacity.

        Reference model.py:518-523: only the last ``min(seqlen, window)``
        tokens survive, each at ring slot ``position % window``.
        """

        if self.next_position != 0:
            raise RuntimeError("prefill_write requires reset state at position zero")
        if raw_latent.ndim != 3:
            raise ValueError("raw_latent must have shape [batch, seqlen, 512]")
        seqlen = raw_latent.shape[1]
        if not 0 < seqlen <= self.max_seq_len:
            raise ValueError("prefill seqlen must fit the non-empty static capacity")
        self._require_tensor(
            "raw_latent",
            raw_latent,
            (self.num_local_sequences, seqlen, LATENT_DIM),
            torch.bfloat16,
        )

        kept = min(seqlen, WINDOW_SIZE)
        absolute_positions = torch.arange(
            seqlen - kept, seqlen, dtype=torch.int64, device=self.device
        )
        raw_slots = absolute_positions.remainder(WINDOW_SIZE)
        self.latent.index_copy_(1, raw_slots, raw_latent[:, -kept:])
        self._raw_positions.index_copy_(
            1,
            raw_slots,
            absolute_positions.unsqueeze(0).expand(self.num_local_sequences, -1),
        )
        self._next_position.fill_(seqlen)

    def decode_write(self, raw_latent: torch.Tensor) -> None:
        """Append one decode token at ring slot ``position % window``.

        Reference model.py:530.
        """

        position = self.next_position
        if position == 0:
            raise RuntimeError("decode_write requires a preceding prefill_write")
        if position >= self.max_seq_len:
            raise RuntimeError("static window KV capacity is exhausted")
        self._require_tensor(
            "raw_latent",
            raw_latent,
            (self.num_local_sequences, 1, LATENT_DIM),
            torch.bfloat16,
        )
        slot = position % WINDOW_SIZE
        self.latent[:, slot].copy_(raw_latent[:, 0])
        self._raw_positions[:, slot].fill_(position)
        self._next_position.fill_(position + 1)

    def _write_decode_fixed(
        self,
        raw_latent: torch.Tensor,
        *,
        position: int,
        slot: int,
    ) -> None:
        """Commit one prevalidated decode token without host sync.

        Mirror of ``StaticLayerKV._write_decode_nonboundary_fixed`` for the
        compressor-free window ring (reference model.py:530): the caller
        (``WindowTorchAttention.prepare_decode_plan``) has already validated
        ``position == next_position`` and the ring metadata, so this hot path
        performs only the fixed-slot writes.
        """

        self._require_tensor(
            "raw_latent",
            raw_latent,
            (self.num_local_sequences, 1, LATENT_DIM),
            torch.bfloat16,
        )
        self.latent[:, slot].copy_(raw_latent[:, 0])
        self._raw_positions[:, slot].fill_(position)
        self._next_position.fill_(position + 1)

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


__all__ = [
    "LATENT_DIM",
    "WINDOW_SIZE",
    "StaticWindowKV",
]
