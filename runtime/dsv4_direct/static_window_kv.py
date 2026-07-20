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
