"""Incremental (chunked) prefill primitives: multi-token input at ``start_pos > 0``.

Why this exists
---------------
The reference (``reference/inference/model.py``) has exactly two branches
everywhere: ``start_pos == 0`` (whole-sequence prefill) and ``start_pos > 0``
(one decode token).  The 24th vertical
(``experiments/D0L-long-prompt-oracle/README.md`` section 3.2) recorded that the
direct runtime inherits that split -- every ``__call__`` rejects
``seqlen > 1`` when ``start_pos > 0`` -- so "chunk" in the C2F series only ever
meant *the row count of one whole-sequence prefill*, never a segmented one.
Phase 4 (chunked prefill interleaved with decode) needs the real thing.

There is therefore **no reference branch to copy** for multi-token
``start_pos > 0``.  The semantics below are *derived*, and the derivation is
pinned to the reference on both sides: every chunked step must leave exactly
the state that the reference's own two branches agree on, so that

    prefill(0, S)                        (reference start_pos == 0 branch)
    == prefill(0, C) ; prefill(C, S-C)   (this module)
    == prefill(0, C) ; decode ; decode ; ...

The load-bearing observation is that the reference's prefill terminal state and
its decode running state are **the same invariant**, which is what makes
segmentation well defined at all:

  ratio-4 overlap compressor (``model.py:279-377``)
    ``kv_state[:, 0:4]`` / ``score_state[:, 0:4]``  = full-width projections
      (score with ``+ape``) of the last *completed* group.
      - prefill writes it at ``model.py:331-332`` (``kv[:, cutoff-4:cutoff]``)
      - decode rolls it at ``model.py:353-354`` (``state[:, :4] = state[:, 4:]``)
    ``kv_state[:, 4:4+r]`` / ``score_state[:, 4:4+r]``, ``r = pos % 4`` = the
      still-open group's tokens.
      - prefill writes it at ``model.py:333-335`` (the ``remainder`` split,
        ``offset = ratio`` when overlapping)
      - decode writes it at ``model.py:347-348`` (slot ``4 + pos % 4``)
    Slots ``[4+r, 8)`` are stale in both branches and are never read before
    being overwritten, because the pool at ``model.py:350-352`` only fires on
    ``(pos+1) % 4 == 0``, by which point all four are fresh.

  ratio-128 compressor (same class, ``overlap = compress_ratio == 4`` is False)
    ``kv_state[:, 0:r]``, ``r = pos % 128`` = the open group's tokens
    (prefill ``model.py:333-335`` with ``offset = 0``; decode
    ``model.py:356-357``).  No previous-group slots exist.

  raw/window ring (``model.py:518-523`` prefill, ``:530`` decode)
    slot ``pos % 128`` holds absolute position ``pos``, for the last <= 128
    positions written.  Both branches are the same placement rule.

Because those invariants coincide, a chunk boundary is just "stop mid-stream",
and the only real work is (a) seeding a chunk's *first* group from the state
instead of from the ``(0, -inf)`` fill that the reference uses for group 0, and
(b) re-indexing attention, because a chunk's queries need raw rows that the
128-slot ring no longer holds contiguously.

Attention layout for a chunk ``[P, P+L)``
-----------------------------------------
Whole-sequence prefill attends over ``cat(raw_latent[0:S], compressed)`` with
``offset = S`` (``model.py:509``, ``:526``); decode attends over
``cat(ring[0:128], compressed)`` with ``offset = 128`` (``model.py:509``,
``:533``).  Neither works for a chunk: the chunk's queries need absolute raw
positions ``[P-127, P+L)``, which spans the previous chunk (only in the ring)
*and* the current one (not in the ring yet, and the ring cannot hold both when
``L > 1``).  This module uses the union layout

    [ ring snapshot (128) | chunk raw (L) | compressed (C) ]      offset = 128 + L

built **before** the ring is advanced, and maps absolute raw position ``q`` to

    q >= P  ->  128 + (q - P)          (this chunk's rows, in order)
    q <  P  ->  q % 128                (ring slot, model.py:530 placement)

which is well defined because the window branch only ever asks for
``q >= p - 127 >= P - 127 > P - 128``, i.e. strictly inside the 128 positions
the ring still holds.  The gathered rows -- and critically their *order*, which
fixes the floating-point summation order in the sparse core -- are then
identical to the whole-sequence path, so the chunked result is bitwise equal
wherever the surrounding GEMMs are.

Visibility is unchanged from the reference, just written in absolute positions:
a query at absolute ``p`` sees raw ``[max(0, p-127), p]`` (``model.py:262-264``)
and compressed rows ``[0, (p+1) // ratio)`` (``model.py:274``, and
``model.py:271`` for the decode form, which is the same expression at
``seqlen == 1``).  Compressed row ``g`` covers positions up to ``4g+3``, so
``(p+1)//4 > g`` iff ``p >= 4g+3`` -- causal, and independent of chunking.
"""

from __future__ import annotations

import torch

__all__ = [
    "chunk_group_span",
    "chunk_raw_index_map",
    "chunk_window_topk_indices",
    "chunk_compressed_topk_indices",
    "overlap_chunk_compress",
    "plain_chunk_compress",
    "ChunkCompression",
]


class ChunkedPrefillError(ValueError):
    """Raised when an incremental-prefill contract is violated."""


def chunk_group_span(start_pos: int, seqlen: int, ratio: int) -> tuple[int, int, int, int]:
    """Group bookkeeping for the chunk ``[start_pos, start_pos + seqlen)``.

    Returns ``(pending, head, num_groups, tail)``:

    - ``pending``  = ``start_pos % ratio``, tokens already parked in the state
      (``model.py:333-335`` prefill remainder / ``model.py:347`` decode slot).
    - ``head``     = tokens of this chunk that close that open group.
    - ``num_groups`` = groups completed by this chunk, i.e.
      ``(start_pos + seqlen) // ratio - start_pos // ratio``.  This is the
      reference's ``cutoff // ratio`` (``model.py:328``, ``:337``) restricted to
      the chunk.
    - ``tail``     = ``(start_pos + seqlen) % ratio``, the new open group's size.
    """

    if ratio <= 0:
        raise ChunkedPrefillError("ratio must be positive")
    if start_pos < 0 or seqlen <= 0:
        raise ChunkedPrefillError("start_pos must be >= 0 and seqlen > 0")
    pending = start_pos % ratio
    head = (ratio - pending) % ratio
    num_groups = (start_pos + seqlen) // ratio - start_pos // ratio
    tail = (start_pos + seqlen) % ratio
    if num_groups == 0:
        # Chunk does not close the open group; it only extends it.
        head = seqlen
    return pending, head, num_groups, tail


def chunk_raw_index_map(
    absolute: torch.Tensor, *, start_pos: int, window_size: int
) -> torch.Tensor:
    """Map absolute raw positions to the ``[ring | chunk]`` attention layout.

    ``q >= start_pos`` are this chunk's rows at ``window_size + (q - start_pos)``;
    earlier ones are ring slots ``q % window_size`` (``model.py:530`` placement).
    """

    return torch.where(
        absolute >= start_pos,
        absolute - start_pos + window_size,
        absolute.remainder(window_size),
    )


def chunk_window_topk_indices(
    *,
    batch_size: int,
    seqlen: int,
    start_pos: int,
    device: torch.device,
    window_size: int = 128,
) -> torch.Tensor:
    """Sliding-window top-k for a chunk, in the ``[ring | chunk | compressed]`` layout.

    Generalizes ``model.py:262-264`` (the ``start_pos == 0`` branch, which emits
    absolute positions ``max(0, p-127) + j`` masked at ``> p``) from ``p = row``
    to ``p = start_pos + row``, then re-indexes through
    :func:`chunk_raw_index_map`.  Columns stay in ascending absolute-position
    order, exactly as the reference emits them, so the sparse core's reduction
    order over ``k`` is unchanged.
    """

    if batch_size <= 0 or seqlen <= 0 or start_pos <= 0:
        raise ChunkedPrefillError("chunk window indices need start_pos > 0")
    width = min(start_pos + seqlen, window_size)
    positions = torch.arange(
        start_pos, start_pos + seqlen, device=device
    ).unsqueeze(1)
    columns = torch.arange(width, device=device)
    absolute = (positions - window_size + 1).clamp_min(0) + columns
    matrix = chunk_raw_index_map(
        absolute, start_pos=start_pos, window_size=window_size
    )
    matrix = torch.where(absolute > positions, -1, matrix)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1).to(torch.int32).contiguous()


def chunk_compressed_topk_indices(
    *,
    batch_size: int,
    seqlen: int,
    start_pos: int,
    offset: int,
    device: torch.device,
    ratio: int,
) -> torch.Tensor:
    """Dense compressed-row top-k for a chunk (ratio-128 layers, no indexer).

    Generalizes ``model.py:273-275``: the candidate set is every compressed row
    that exists after the chunk, ``(start_pos + seqlen) // ratio``, masked per
    row by the reference's causal bound ``(p + 1) // ratio`` written in absolute
    positions.
    """

    if start_pos <= 0 or seqlen <= 0 or offset < 0:
        raise ChunkedPrefillError("chunk compressed indices need start_pos > 0")
    count = (start_pos + seqlen) // ratio
    if count == 0:
        return torch.empty((batch_size, seqlen, 0), dtype=torch.int32, device=device)
    matrix = torch.arange(count, device=device).repeat(seqlen, 1)
    visible = (
        torch.arange(start_pos + 1, start_pos + seqlen + 1, device=device) // ratio
    ).unsqueeze(1)
    matrix = torch.where(matrix >= visible, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1).to(torch.int32).contiguous()


class ChunkCompression:
    """Result of one chunked compressor step."""

    __slots__ = ("num_rows", "row_offset", "pooled", "group_starts")

    def __init__(
        self,
        *,
        num_rows: int,
        row_offset: int,
        pooled: torch.Tensor | None,
        group_starts: tuple[int, ...],
    ) -> None:
        self.num_rows = num_rows
        self.row_offset = row_offset
        self.pooled = pooled
        self.group_starts = group_starts


def _stack_chunk_groups(
    values: torch.Tensor,
    state_rows: torch.Tensor,
    *,
    pending: int,
    head: int,
    num_groups: int,
    ratio: int,
) -> torch.Tensor:
    """Group a chunk's tokens into ``[batch, num_groups, ratio, width]``.

    Group 0 is stitched from the state's open-group rows (``model.py:333-335``
    parked them there) plus this chunk's ``head`` tokens; the remainder of the
    chunk is 4-aligned and reshapes without a copy, exactly like the reference's
    ``kv.unflatten(1, (-1, ratio))`` (``model.py:337``).
    """

    aligned_groups = num_groups - (1 if pending else 0)
    aligned = values[:, head : head + aligned_groups * ratio].unflatten(
        1, (aligned_groups, ratio)
    )
    if not pending:
        return aligned
    first = torch.cat((state_rows, values[:, :head]), dim=1).unsqueeze(1)
    if aligned_groups == 0:
        return first
    return torch.cat((first, aligned), dim=1)


def overlap_chunk_compress(
    projected_kv: torch.Tensor,
    projected_score: torch.Tensor,
    ape: torch.Tensor,
    *,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    start_pos: int,
    output_dim: int,
    ratio: int = 4,
) -> tuple[torch.Tensor | None, int, tuple[int, ...]]:
    """Incremental prefill for the ratio-4 **overlap** compressor.

    Reference anchors:

    - pooling shape and the ``prev-left || cur-right`` split:
      ``model.py:339-342`` via ``overlap_transform`` (``model.py:307-314``).
      ``overlap_transform`` fills ``[:, :, ratio:]`` from the current group's
      *second* half-width and ``[:, 1:, :ratio]`` from the *previous* group's
      *first* half-width, leaving group 0's previous slots at the fill value
      (``0`` for kv, ``-inf`` for score).  The decode boundary builds the same
      eight rows explicitly at ``model.py:350-351``.
    - what "previous group" means across a chunk boundary: ``kv_state[:, :4]``
      (``model.py:331-332`` prefill / ``model.py:353-354`` decode roll).  At
      ``start_pos == 0`` those slots still hold the constructor's ``(0, -inf)``
      (``model.py:303-304``), which is *precisely* the fill ``overlap_transform``
      uses for group 0 -- so seeding from the state is a strict generalization,
      not a special case.
    - the open-group carry: ``model.py:333-335`` (prefill remainder at
      ``offset = ratio``) / ``model.py:347-348`` (decode slot ``4 + pos % 4``).

    Returns ``(pooled | None, row_offset, group_start_positions)``.  ``pooled``
    is un-normalized/un-RoPEd; the caller applies the layer's finalizer, whose
    RoPE index is the group start (``model.py:364`` ``freqs_cis[0:cutoff:ratio]``
    and ``model.py:366`` ``freqs_cis[start_pos+1-ratio]`` are the same rule).
    """

    batch, seqlen, width = projected_kv.shape
    if width != 2 * output_dim:
        raise ChunkedPrefillError("overlap projections must be twice the output width")
    pending, head, num_groups, tail = chunk_group_span(start_pos, seqlen, ratio)
    row_offset = start_pos // ratio

    if num_groups == 0:
        # No group closes: extend the open group in place (model.py:347-348
        # repeated for `seqlen` tokens).
        kv_state[:, ratio + pending : ratio + pending + seqlen].copy_(projected_kv)
        score_state[:, ratio + pending : ratio + pending + seqlen].copy_(
            projected_score + ape[pending : pending + seqlen]
        )
        return None, row_offset, ()

    current = _stack_chunk_groups(
        projected_kv,
        kv_state[:, ratio : ratio + pending],
        pending=pending,
        head=head,
        num_groups=num_groups,
        ratio=ratio,
    )
    # The parked score rows already carry `+ape` (model.py:335, :348); the
    # fresh ones get theirs here, with the phase offset the reference uses.
    aligned_groups = num_groups - (1 if pending else 0)
    aligned_score = (
        projected_score[:, head : head + aligned_groups * ratio].unflatten(
            1, (aligned_groups, ratio)
        )
        + ape
    )
    if pending:
        first_score = torch.cat(
            (
                score_state[:, ratio : ratio + pending],
                projected_score[:, :head] + ape[pending:ratio],
            ),
            dim=1,
        ).unsqueeze(1)
        current_score = (
            first_score
            if aligned_groups == 0
            else torch.cat((first_score, aligned_score), dim=1)
        )
    else:
        current_score = aligned_score

    # previous[g] = group g-1's full-width rows; previous[0] comes from the
    # state (model.py:331-332 / :353-354), which is the (0, -inf) fill at
    # start_pos == 0 -- identical to overlap_transform's group-0 behaviour.
    previous = torch.cat(
        (kv_state[:, :ratio].unsqueeze(1), current[:, :-1]), dim=1
    )
    previous_score = torch.cat(
        (score_state[:, :ratio].unsqueeze(1), current_score[:, :-1]), dim=1
    )

    over_kv = torch.cat(
        (previous[..., :output_dim], current[..., output_dim:]), dim=2
    )
    over_score = torch.cat(
        (previous_score[..., :output_dim], current_score[..., output_dim:]), dim=2
    )
    pooled = (over_kv * over_score.softmax(dim=2)).sum(dim=2)

    # Roll the state: last completed group becomes "previous" (model.py:353-354),
    # then park the new open group (model.py:333-335 / :347-348).
    kv_state[:, :ratio].copy_(current[:, -1])
    score_state[:, :ratio].copy_(current_score[:, -1])
    if tail:
        kv_state[:, ratio : ratio + tail].copy_(projected_kv[:, seqlen - tail :])
        score_state[:, ratio : ratio + tail].copy_(
            projected_score[:, seqlen - tail :] + ape[:tail]
        )
    starts = tuple(
        ratio * (row_offset + index) for index in range(num_groups)
    )
    return pooled, row_offset, starts


def plain_chunk_compress(
    projected_kv: torch.Tensor,
    projected_score: torch.Tensor,
    ape: torch.Tensor,
    *,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    start_pos: int,
    ratio: int = 128,
) -> tuple[torch.Tensor | None, int, tuple[int, ...]]:
    """Incremental prefill for the non-overlap (ratio-128) compressor.

    Reference anchors: ``overlap`` is False for ``compress_ratio != 4``
    (``model.py:290``), so the prefill branch skips ``overlap_transform`` and
    pools each group of ``ratio`` rows directly (``model.py:337-338``, ``:342``),
    the open-group carry sits at ``offset = 0`` (``model.py:329``, ``:334-335``),
    and the decode boundary pools all ``ratio`` slots at once
    (``model.py:356-359``).  There is no previous-group term at all, so a chunk
    boundary only has to stitch the open group.
    """

    batch, seqlen, width = projected_kv.shape
    pending, head, num_groups, tail = chunk_group_span(start_pos, seqlen, ratio)
    row_offset = start_pos // ratio

    if num_groups == 0:
        kv_state[:, pending : pending + seqlen].copy_(projected_kv)
        score_state[:, pending : pending + seqlen].copy_(
            projected_score + ape[pending : pending + seqlen]
        )
        return None, row_offset, ()

    current = _stack_chunk_groups(
        projected_kv,
        kv_state[:, :pending],
        pending=pending,
        head=head,
        num_groups=num_groups,
        ratio=ratio,
    )
    aligned_groups = num_groups - (1 if pending else 0)
    aligned_score = (
        projected_score[:, head : head + aligned_groups * ratio].unflatten(
            1, (aligned_groups, ratio)
        )
        + ape
    )
    if pending:
        first_score = torch.cat(
            (
                score_state[:, :pending],
                projected_score[:, :head] + ape[pending:ratio],
            ),
            dim=1,
        ).unsqueeze(1)
        current_score = (
            first_score
            if aligned_groups == 0
            else torch.cat((first_score, aligned_score), dim=1)
        )
    else:
        current_score = aligned_score

    pooled = (current * current_score.softmax(dim=2)).sum(dim=2)

    if tail:
        kv_state[:, :tail].copy_(projected_kv[:, seqlen - tail :])
        score_state[:, :tail].copy_(projected_score[:, seqlen - tail :] + ape[:tail])
    starts = tuple(ratio * (row_offset + index) for index in range(num_groups))
    return pooled, row_offset, starts
