"""DP-attention (sequence-split) caliber for the TP4 direct runtime.

**Finding that defines this module (E0dpf vertical, 2026-07-20).**  The lite
runtime never head-shards attention: ``block_weights`` loads the full
replicated attention tensors on every TP rank, all three attention classes
(window / ratio-4 / ratio-128) compute the full 64 heads over
``state.num_local_sequences`` rows, and ``TP4MoE.__call__`` already runs the
exact DP collective order the DP-attention placement requires::

    all_gather(local rows -> global rows)
      -> gate + Marlin routed (itp: all experts, inter/4 rows) + shared,
         partial over the intermediate slice, on the *gathered* rows
      -> reduce_scatter(sum over ranks -> back to this rank's own rows)

Under the E1F "full replication" caliber every rank feeds the *same* B
sequences, so the gather produces 4 identical copies (4x redundant MoE row
work) and reduce_scatter returns this rank's copy.  Under the DP caliber each
rank feeds its **own** ``B_global / tp_size`` sequences; the gather then
produces ``B_global`` distinct rows and reduce_scatter returns exactly this
rank's sequences.  Both calibers are the same math per sequence; no module in
``dsv4_direct`` needs a code change to switch between them.

A "DP mode" construction therefore is: build every layer state with
``num_local_sequences = B_global // tp_size`` and feed rank-distinct
sequence rows (inputs, KV seeds, token loop).  This module owns the two
helpers that make that slicing explicit and uniform across the gate and the
bench, plus the caliber constants.

Head-loop note: the sm89 h=64 smem wall (A4F) applies to the TileLang
``sparse_attn`` kernel only.  The runtime's sparse attention is the masked
torch control backend (the stateful path asserts no injected backend), which
evaluates all 64 heads in one einsum in both calibers; no 4x16 sub-launch
exists or is needed on this path.
"""

from __future__ import annotations

from typing import Any

import torch


TP_SIZE = 4


class DPCaliberError(ValueError):
    """Raised when a DP sequence split would be malformed."""


def dp_local_batch(global_batch: int, *, tp_size: int = TP_SIZE) -> int:
    """Per-rank sequence count for one DP-split global batch."""

    if (
        not isinstance(global_batch, int)
        or isinstance(global_batch, bool)
        or global_batch <= 0
    ):
        raise DPCaliberError("global_batch must be a positive integer")
    if global_batch % tp_size:
        raise DPCaliberError(
            f"DP caliber requires global_batch divisible by tp_size "
            f"{tp_size}, got {global_batch}"
        )
    return global_batch // tp_size


def dp_row_bounds(
    tp_rank: int, local_batch: int, *, tp_size: int = TP_SIZE
) -> tuple[int, int]:
    """This rank's [lo, hi) global sequence rows under the DP split."""

    if (
        not isinstance(tp_rank, int)
        or isinstance(tp_rank, bool)
        or tp_rank not in range(tp_size)
    ):
        raise DPCaliberError(f"tp_rank must be in [0, {tp_size}), got {tp_rank!r}")
    if (
        not isinstance(local_batch, int)
        or isinstance(local_batch, bool)
        or local_batch <= 0
    ):
        raise DPCaliberError("local_batch must be a positive integer")
    lo = tp_rank * local_batch
    return lo, lo + local_batch

def dp_row_slice(
    value: torch.Tensor,
    tp_rank: int,
    local_batch: int,
    *,
    tp_size: int = TP_SIZE,
) -> torch.Tensor:
    """Contiguous copy of this rank's rows of one batch-first global tensor."""

    lo, hi = dp_row_bounds(tp_rank, local_batch, tp_size=tp_size)
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise DPCaliberError("dp_row_slice requires a batch-first tensor")
    if value.shape[0] != local_batch * tp_size:
        raise DPCaliberError(
            f"global tensor batch {value.shape[0]} != "
            f"{tp_size} * {local_batch}"
        )
    return value[lo:hi].contiguous()


def dp_slice_ratio4_oracle_state(
    state: Any, tp_rank: int, local_batch: int, *, tp_size: int = TP_SIZE
) -> Any:
    """Row-slice one ``OracleRatio4State`` built at the global batch.

    All payload tensors of the oracle state are batch-first; the scalar
    cursor fields are shared.  Slicing a global-batch payload keeps the DP
    seed rows byte-identical to the corresponding replicated-caliber rows.
    """

    from .ratio4_oracle import OracleRatio4State

    if not isinstance(state, OracleRatio4State):
        raise DPCaliberError("expected an OracleRatio4State")
    return OracleRatio4State(
        raw=dp_row_slice(state.raw, tp_rank, local_batch, tp_size=tp_size),
        compressed=dp_row_slice(
            state.compressed, tp_rank, local_batch, tp_size=tp_size
        ),
        indexer_kv=dp_row_slice(
            state.indexer_kv, tp_rank, local_batch, tp_size=tp_size
        ),
        main_kv=dp_row_slice(state.main_kv, tp_rank, local_batch, tp_size=tp_size),
        main_score=dp_row_slice(
            state.main_score, tp_rank, local_batch, tp_size=tp_size
        ),
        index_kv=dp_row_slice(state.index_kv, tp_rank, local_batch, tp_size=tp_size),
        index_score=dp_row_slice(
            state.index_score, tp_rank, local_batch, tp_size=tp_size
        ),
        next_position=state.next_position,
        compressed_count=state.compressed_count,
        max_seq_len=state.max_seq_len,
    )


def oracle_state_to_device(state: Any, device: torch.device | str) -> Any:
    """Move an ``OracleRatio4State``'s payload tensors to ``device``.

    Lets DP callers build the global-batch oracle on the CPU (its qdq
    temporaries at large global batches exceed GPU headroom), slice this
    rank's rows, and move only the slice; the generator is CPU-side either
    way, so the values are bitwise identical to a GPU-side build.
    """

    from .ratio4_oracle import OracleRatio4State

    if not isinstance(state, OracleRatio4State):
        raise DPCaliberError("expected an OracleRatio4State")
    target = torch.device(device)
    return OracleRatio4State(
        raw=state.raw.to(target),
        compressed=state.compressed.to(target),
        indexer_kv=state.indexer_kv.to(target),
        main_kv=state.main_kv.to(target),
        main_score=state.main_score.to(target),
        index_kv=state.index_kv.to(target),
        index_score=state.index_score.to(target),
        next_position=state.next_position,
        compressed_count=state.compressed_count,
        max_seq_len=state.max_seq_len,
    )


__all__ = [
    "DPCaliberError",
    "TP_SIZE",
    "dp_local_batch",
    "dp_row_bounds",
    "dp_row_slice",
    "dp_slice_ratio4_oracle_state",
    "oracle_state_to_device",
]
