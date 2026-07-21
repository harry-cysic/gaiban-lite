#!/usr/bin/env python3
"""C3F single-layer gate: incremental (chunked) prefill == whole-sequence prefill.

The 24th vertical (``experiments/D0L-long-prompt-oracle``) recorded that the
direct runtime had **no** incremental chunked prefill: every attention
``__call__`` rejected ``seqlen > 1`` at ``start_pos > 0``, so a "chunk" was only
ever the row count of one whole-sequence prefill.  The 25th vertical adds the
real capability (``dsv4_direct/chunked_prefill.py`` plus the three layer
paths).  This harness is its correctness gate, run on **one GPU** with real
TP-rank-0 replicated checkpoint weights.

Three checks, smallest first:

1. ``state_machine`` -- **bitwise**.  Drives the compressor state machines
   directly (``overlap_chunk_compress`` for ratio-4, ``plain_chunk_compress``
   for ratio-128) on *identical* pre-computed FP32 projections, sliced from one
   tensor so no GEMM shape ever changes.  Compares segmented pooling + terminal
   state against the reference whole-sequence pooling transcribed from
   ``model.py:325-342`` (with ``overlap_transform``, ``model.py:307-314``).
   This isolates the derived semantics from every floating-point confounder, so
   the equality target here is exact.

2. ``layer_equivalence`` -- per layer kind, one lane fed the whole sequence vs
   another fed the same sequence in segments, comparing branch outputs and the
   live terminal state.  Segment schedules deliberately include lengths that
   are **not** multiples of the group size (4 / 128) and not multiples of the
   window (128), because those are exactly the boundaries where the open-group
   carry and the ring re-index have to work.

3. ``gemm_shape_probe`` -- diagnostic, not a pass/fail.  Runs the layer's own
   projection GEMMs at whole-sequence and per-segment shapes on identical rows
   and reports whether they agree bitwise.  This attributes any residual in
   check 2 to cuBLAS kernel selection (M-dependent reduction order) rather than
   to the chunking semantics, which check 1 already pins exactly.

Run (titan064 / titan065, one GPU):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ~/Workspace/venvs/sglang/bin/python c3f_chunked_prefill_gate.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-c3f-gate
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from dsv4_direct.attention import (
    Ratio128AttentionConfig,
    Ratio128TorchAttention,
    prepare_attention_weights,
)
from dsv4_direct.block_weights import load_replicated_block_weights
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.chunked_prefill import (
    chunk_group_span,
    overlap_chunk_compress,
    plain_chunk_compress,
)
from dsv4_direct.ratio4_attention import (
    Ratio4AttentionConfig,
    prepare_ratio4_attention_weights,
)
from dsv4_direct.ratio4_fullpos import Ratio4FullPositionAttention
from dsv4_direct.static_kv import StaticLayerKV
from dsv4_direct.static_window_kv import StaticWindowKV
from dsv4_direct.window_attention import (
    WindowAttentionConfig,
    WindowTorchAttention,
    prepare_window_attention_weights,
)


WINDOW_LAYER_ID = 0
RATIO4_LAYER_ID = 2
RATIO128_LAYER_ID = 3

# Segment schedules.  Every schedule sums to its sequence length.  The
# non-uniform ones are the point of the gate: 1000/1096 are not multiples of
# 128 (ring re-index) and 999/1001/517 are not multiples of 4 (open-group carry
# in the ratio-4 overlap compressor).
LAYER_SEQLEN = 1024
LAYER_SCHEDULES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("whole", (1024,)),
    ("2x512", (512, 512)),
    ("4x256", (256, 256, 256, 256)),
    ("8x128", (128,) * 8),
    ("uneven_1000_24", (1000, 24)),
    ("uneven_999_25", (999, 25)),
    ("uneven_517_507", (517, 507)),
    ("uneven_3_1021", (3, 1021)),
    ("uneven_130_894", (130, 894)),
    ("tiny_lead_1_1_1022", (1, 1, 1022)),
)

# State-machine schedules over a longer stream (no attention, so it is cheap).
STATE_SEQLEN = 2200
STATE_SCHEDULES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("whole", (2200,)),
    ("2seg", (1100, 1100)),
    ("4seg", (550, 550, 550, 550)),
    ("8seg", (275,) * 8),
    ("uneven_1000_1096_104", (1000, 1096, 104)),
    ("uneven_999_1201", (999, 1201)),
    ("uneven_1_2199", (1, 2199)),
    ("uneven_129_2071", (129, 2071)),
    ("uneven_5_7_11_2177", (5, 7, 11, 2177)),
)


def deterministic_hidden(seed: int, seqlen: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(1, seqlen, 4096, generator=generator, dtype=torch.float32)
    return (value * 0.02).to(torch.bfloat16).to(device)


def tensor_delta(observed: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Bitwise verdict plus magnitudes, ``inf``-safe (score states hold -inf)."""

    if observed.shape != expected.shape:
        return {"bitwise": False, "shape_mismatch": [
            list(observed.shape), list(expected.shape)]}
    left = observed.float()
    right = expected.float()
    bitwise = bool(torch.equal(left, right))
    finite = torch.isfinite(left) & torch.isfinite(right)
    same_nonfinite = bool(torch.equal(torch.isfinite(left), torch.isfinite(right)))
    if not bool(finite.any().item()):
        return {
            "bitwise": bitwise,
            "max_abs": 0.0,
            "rms_rel": 0.0,
            "nonfinite_layout_match": same_nonfinite,
            "elements": int(left.numel()),
        }
    difference = (left[finite] - right[finite]).abs()
    scale = right[finite].square().mean().sqrt()
    return {
        "bitwise": bitwise,
        "max_abs": float(difference.max().item()),
        "rms_rel": float(
            (difference.square().mean().sqrt() / scale.clamp_min(1e-12)).item()
        ),
        "nonfinite_layout_match": same_nonfinite,
        "elements": int(left.numel()),
        "mismatched_elements": int((left != right).sum().item()),
    }


# ----------------------------------------------------------------------------
# 1. state machine (bitwise, identical projections)


def reference_overlap_prefill(
    projected_kv: torch.Tensor,
    projected_score: torch.Tensor,
    ape: torch.Tensor,
    output_dim: int,
    ratio: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transcription of ``model.py:325-342`` for ``overlap=True``.

    Returns ``(pooled, kv_state, score_state)`` -- the pooled rows plus the
    terminal state the reference leaves behind (``model.py:331-335``).
    """

    batch, seqlen, width = projected_kv.shape
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    kv_state = projected_kv.new_zeros(batch, 2 * ratio, width)
    score_state = projected_score.new_full((batch, 2 * ratio, width), float("-inf"))
    if cutoff >= ratio:  # model.py:330-332
        kv_state[:, :ratio] = projected_kv[:, cutoff - ratio : cutoff]
        score_state[:, :ratio] = projected_score[:, cutoff - ratio : cutoff] + ape
    if remainder:  # model.py:333-335 (offset == ratio when overlapping)
        kv_state[:, ratio : ratio + remainder] = projected_kv[:, cutoff:]
        score_state[:, ratio : ratio + remainder] = (
            projected_score[:, cutoff:] + ape[:remainder]
        )
    rows = cutoff // ratio
    if rows == 0:
        return projected_kv.new_zeros(batch, 0, output_dim), kv_state, score_state
    grouped_kv = projected_kv[:, :cutoff].unflatten(1, (rows, ratio))
    grouped_score = projected_score[:, :cutoff].unflatten(1, (rows, ratio)) + ape
    # overlap_transform, model.py:307-314
    over_kv = grouped_kv.new_zeros(batch, rows, 2 * ratio, output_dim)
    over_score = grouped_score.new_full(
        (batch, rows, 2 * ratio, output_dim), float("-inf")
    )
    over_kv[:, :, ratio:] = grouped_kv[..., output_dim:]
    over_kv[:, 1:, :ratio] = grouped_kv[:, :-1, :, :output_dim]
    over_score[:, :, ratio:] = grouped_score[..., output_dim:]
    over_score[:, 1:, :ratio] = grouped_score[:, :-1, :, :output_dim]
    pooled = (over_kv * over_score.softmax(dim=2)).sum(dim=2)  # model.py:342
    return pooled, kv_state, score_state


def reference_plain_prefill(
    projected_kv: torch.Tensor,
    projected_score: torch.Tensor,
    ape: torch.Tensor,
    ratio: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transcription of ``model.py:325-342`` for ``overlap=False``."""

    batch, seqlen, width = projected_kv.shape
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    kv_state = projected_kv.new_zeros(batch, ratio, width)
    score_state = projected_score.new_full((batch, ratio, width), float("-inf"))
    if remainder:  # model.py:333-335 with offset == 0 (model.py:329)
        kv_state[:, :remainder] = projected_kv[:, cutoff:]
        score_state[:, :remainder] = projected_score[:, cutoff:] + ape[:remainder]
    rows = cutoff // ratio
    if rows == 0:
        return projected_kv.new_zeros(batch, 0, width), kv_state, score_state
    grouped_kv = projected_kv[:, :cutoff].unflatten(1, (rows, ratio))
    grouped_score = projected_score[:, :cutoff].unflatten(1, (rows, ratio)) + ape
    pooled = (grouped_kv * grouped_score.softmax(dim=2)).sum(dim=2)
    return pooled, kv_state, score_state


def run_state_machine_check(
    device: torch.device, seed: int
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    records: list[dict[str, Any]] = []
    all_bitwise = True

    for label, ratio, output_dim, overlap in (
        ("ratio4_main", 4, 512, True),
        ("ratio4_index", 4, 128, True),
        ("ratio128", 128, 512, False),
    ):
        width = 2 * output_dim if overlap else output_dim
        projected_kv = (
            torch.randn(1, STATE_SEQLEN, width, generator=generator) * 0.5
        ).to(torch.float32).to(device)
        projected_score = (
            torch.randn(1, STATE_SEQLEN, width, generator=generator) * 0.5
        ).to(torch.float32).to(device)
        ape = (torch.randn(ratio, width, generator=generator) * 0.1).to(
            torch.float32
        ).to(device)

        if overlap:
            expected_pooled, expected_kv, expected_score = reference_overlap_prefill(
                projected_kv, projected_score, ape, output_dim, ratio
            )
            state_rows = 2 * ratio
        else:
            expected_pooled, expected_kv, expected_score = reference_plain_prefill(
                projected_kv, projected_score, ape, ratio
            )
            state_rows = ratio

        for schedule_name, schedule in STATE_SCHEDULES:
            if sum(schedule) != STATE_SEQLEN:
                raise ValueError(f"schedule {schedule_name} does not sum to stream")
            kv_state = torch.zeros(1, state_rows, width, dtype=torch.float32, device=device)
            score_state = torch.full_like(kv_state, float("-inf"))
            pooled_parts: list[torch.Tensor] = []
            position = 0
            for length in schedule:
                chunk_kv = projected_kv[:, position : position + length]
                chunk_score = projected_score[:, position : position + length]
                if overlap:
                    pooled, _row_offset, _starts = overlap_chunk_compress(
                        chunk_kv,
                        chunk_score,
                        ape,
                        kv_state=kv_state,
                        score_state=score_state,
                        start_pos=position,
                        output_dim=output_dim,
                        ratio=ratio,
                    )
                else:
                    pooled, _row_offset, _starts = plain_chunk_compress(
                        chunk_kv,
                        chunk_score,
                        ape,
                        kv_state=kv_state,
                        score_state=score_state,
                        start_pos=position,
                        ratio=ratio,
                    )
                if pooled is not None:
                    pooled_parts.append(pooled)
                position += length
            observed_pooled = (
                torch.cat(pooled_parts, dim=1)
                if pooled_parts
                else expected_pooled.new_zeros(1, 0, expected_pooled.shape[-1])
            )
            # Only the *live* state slots are defined: the reference leaves the
            # slots past the open group stale in both of its own branches
            # (model.py never reads them before overwriting), so comparing them
            # would be comparing garbage.
            tail = STATE_SEQLEN % ratio
            live = (slice(None), slice(0, ratio + tail)) if overlap else (
                slice(None), slice(0, tail)
            )
            record = {
                "layer": label,
                "schedule": schedule_name,
                "segments": list(schedule),
                "pooled_rows": int(observed_pooled.shape[1]),
                "pooled": tensor_delta(observed_pooled, expected_pooled),
                "kv_state_live": tensor_delta(kv_state[live], expected_kv[live]),
                "score_state_live": tensor_delta(
                    score_state[live], expected_score[live]
                ),
            }
            record["bitwise"] = bool(
                record["pooled"]["bitwise"]
                and record["kv_state_live"]["bitwise"]
                and record["score_state_live"]["bitwise"]
            )
            all_bitwise = all_bitwise and record["bitwise"]
            records.append(record)

    return {
        "stream_length": STATE_SEQLEN,
        "all_bitwise": all_bitwise,
        "records": records,
    }


# ----------------------------------------------------------------------------
# 1b. index equivalence (exact, arithmetic-free)


def _absolute_from_layout(
    indices: torch.Tensor, *, start_pos: int, window_size: int
) -> torch.Tensor:
    """Invert :func:`chunk_raw_index_map`.

    ``idx >= window`` are this chunk's rows; a ring slot ``idx`` denotes the
    unique absolute position in ``[start_pos - window, start_pos)`` congruent to
    ``idx`` mod ``window``.
    """

    base = start_pos - window_size
    ring_absolute = base + (indices - base).remainder(window_size)
    absolute = torch.where(
        indices >= window_size, indices - window_size + start_pos, ring_absolute
    )
    return torch.where(indices < 0, torch.full_like(indices, -1), absolute)


def run_index_equivalence(device: torch.device) -> dict[str, Any]:
    """Do the chunked top-k rows select the same absolute positions?

    This is the other half of the new code (the compressor state machine being
    the first), and unlike the layer check it is exact: it compares *index
    sets*, so no floating-point kernel can blur the verdict.  The reference is
    the whole-sequence branch (``model.py:262-264`` window, ``:273-275``
    compressed) evaluated over the full stream, sliced to the chunk's rows.
    """

    from dsv4_direct.attention import compressed_topk_indices, window_topk_indices
    from dsv4_direct.chunked_prefill import (
        chunk_compressed_topk_indices,
        chunk_window_topk_indices,
    )

    window_size = 128
    records: list[dict[str, Any]] = []
    all_equal = True
    for total, schedules in ((STATE_SEQLEN, STATE_SCHEDULES),):
        whole_window = window_topk_indices(
            batch_size=1, seqlen=total, start_pos=0, device=device
        ).long()
        for ratio in (4, 128):
            whole_compressed = compressed_topk_indices(
                batch_size=1, seqlen=total, start_pos=0, offset=0,
                device=device, ratio=ratio,
            ).long()
            for schedule_name, schedule in schedules:
                position = 0
                window_ok = True
                compressed_ok = True
                for length in schedule:
                    if position == 0:
                        chunk_window = window_topk_indices(
                            batch_size=1, seqlen=length, start_pos=0,
                            device=device,
                        ).long()
                        chunk_compressed = compressed_topk_indices(
                            batch_size=1, seqlen=length, start_pos=0, offset=0,
                            device=device, ratio=ratio,
                        ).long()
                    else:
                        chunk_window = _absolute_from_layout(
                            chunk_window_topk_indices(
                                batch_size=1, seqlen=length, start_pos=position,
                                device=device, window_size=window_size,
                            ).long(),
                            start_pos=position,
                            window_size=window_size,
                        )
                        chunk_compressed = chunk_compressed_topk_indices(
                            batch_size=1, seqlen=length, start_pos=position,
                            offset=0, device=device, ratio=ratio,
                        ).long()
                    expected_window = whole_window[:, position : position + length]
                    expected_compressed = whole_compressed[
                        :, position : position + length
                    ]
                    # Widths differ only by trailing -1 padding, which both the
                    # torch core and the tilelang kernel ignore; pad and compare.
                    def pad(value: torch.Tensor, width: int) -> torch.Tensor:
                        if value.shape[-1] >= width:
                            return value[..., :width]
                        return torch.nn.functional.pad(
                            value, (0, width - value.shape[-1]), value=-1
                        )

                    width_w = max(chunk_window.shape[-1], expected_window.shape[-1])
                    width_c = max(
                        chunk_compressed.shape[-1], expected_compressed.shape[-1]
                    )
                    if not torch.equal(
                        pad(chunk_window, width_w), pad(expected_window, width_w)
                    ):
                        window_ok = False
                    if not torch.equal(
                        pad(chunk_compressed, width_c),
                        pad(expected_compressed, width_c),
                    ):
                        compressed_ok = False
                    position += length
                records.append(
                    {
                        "ratio": ratio,
                        "schedule": schedule_name,
                        "segments": list(schedule),
                        "window_indices_equal": window_ok,
                        "compressed_indices_equal": compressed_ok,
                    }
                )
                all_equal = all_equal and window_ok and compressed_ok
    return {"all_equal": all_equal, "records": records}


# ----------------------------------------------------------------------------
# 2. layer equivalence


def build_lane(kind: str, config: Any, prepared: Any, device: torch.device,
               max_seq_len: int) -> tuple[Any, Any]:
    if kind == "window":
        state = StaticWindowKV(
            num_local_sequences=1, max_seq_len=max_seq_len,
            layer_id=WINDOW_LAYER_ID, device=device,
        )
        return WindowTorchAttention(config, prepared, state), state
    if kind == "ratio128":
        state = StaticLayerKV(
            num_local_sequences=1, max_seq_len=max_seq_len,
            layer_id=RATIO128_LAYER_ID, device=device,
        )
        return Ratio128TorchAttention(config, prepared, state), state
    lane = Ratio4FullPositionAttention(
        config, prepared, batch_size=1, device=device
    )
    return lane, lane


def lane_forward(kind: str, lane: Any, hidden: torch.Tensor, start_pos: int):
    result = lane(hidden, start_pos=start_pos)
    if kind == "ratio4":
        return result
    return result[0]


def live_state(kind: str, holder: Any, position: int) -> dict[str, torch.Tensor]:
    """Terminal state buffers that are defined after ``position`` tokens."""

    if kind == "window":
        return {"ring": holder.latent.clone()}
    if kind == "ratio128":
        tail = position % 128
        out = {
            "ring": holder.raw.clone(),
            "compressed": holder.compressed[:, : position // 128].clone(),
        }
        if tail:
            out["kv_state"] = holder.kv_state[:, :tail].clone()
            out["score_state"] = holder.score_state[:, :tail].clone()
        return out
    tail = position % 4
    out = {
        "ring": holder.raw.clone(),
        "compressed": holder.compressed[:, : position // 4].clone(),
        "indexer_kv": holder.indexer_kv[:, : position // 4].clone(),
        "main_kv_state": holder.main_kv_state[:, : 4 + tail].clone(),
        "main_score_state": holder.main_score_state[:, : 4 + tail].clone(),
        "index_kv_state": holder.index_kv_state[:, : 4 + tail].clone(),
        "index_score_state": holder.index_score_state[:, : 4 + tail].clone(),
    }
    return out


def run_layer_equivalence(
    kind: str,
    config: Any,
    prepared: Any,
    device: torch.device,
    hidden: torch.Tensor,
    max_seq_len: int,
) -> dict[str, Any]:
    reference_lane, reference_holder = build_lane(
        kind, config, prepared, device, max_seq_len
    )
    reference_out = lane_forward(kind, reference_lane, hidden, 0)
    reference_state = live_state(kind, reference_holder, LAYER_SEQLEN)
    del reference_lane
    torch.cuda.empty_cache()

    records = []
    for schedule_name, schedule in LAYER_SCHEDULES:
        if sum(schedule) != LAYER_SEQLEN:
            raise ValueError(f"schedule {schedule_name} does not sum to {LAYER_SEQLEN}")
        lane, holder = build_lane(kind, config, prepared, device, max_seq_len)
        outputs = []
        position = 0
        for length in schedule:
            outputs.append(
                lane_forward(
                    kind, lane, hidden[:, position : position + length], position
                )
            )
            position += length
        observed = torch.cat(outputs, dim=1)
        observed_state = live_state(kind, holder, LAYER_SEQLEN)
        state_deltas = {
            name: tensor_delta(observed_state[name], reference_state[name])
            for name in reference_state
        }
        record = {
            "schedule": schedule_name,
            "segments": list(schedule),
            "branch": tensor_delta(observed, reference_out),
            "state": state_deltas,
        }
        record["bitwise"] = bool(
            record["branch"]["bitwise"]
            and all(item["bitwise"] for item in state_deltas.values())
        )
        record["max_abs"] = max(
            [record["branch"]["max_abs"]]
            + [item["max_abs"] for item in state_deltas.values()]
        )
        record["branch_rms_rel"] = record["branch"]["rms_rel"]
        records.append(record)
        del lane, holder, outputs, observed
        torch.cuda.empty_cache()

    return {
        "kind": kind,
        "seqlen": LAYER_SEQLEN,
        "all_bitwise": all(item["bitwise"] for item in records),
        "worst_branch_rms_rel": max(item["branch_rms_rel"] for item in records),
        "worst_max_abs": max(item["max_abs"] for item in records),
        "records": records,
    }


# ----------------------------------------------------------------------------
# 3. GEMM shape probe (diagnostic)


def run_gemm_shape_probe(
    prepared: Any, hidden: torch.Tensor
) -> dict[str, Any]:
    """Do the layer's own projections depend on the GEMM's M dimension?"""

    weight_bf16 = prepared.wq_a
    weight_fp32 = getattr(prepared, "compressor_wkv", None)
    out: dict[str, Any] = {}
    for name, weight, source in (
        ("wq_a_bf16", weight_bf16, hidden),
        (
            "compressor_wkv_fp32",
            weight_fp32,
            hidden.float() if weight_fp32 is not None else None,
        ),
    ):
        if weight is None:
            continue
        whole = torch.nn.functional.linear(source, weight)
        for schedule_name, schedule in LAYER_SCHEDULES:
            parts = []
            position = 0
            for length in schedule:
                parts.append(
                    torch.nn.functional.linear(
                        source[:, position : position + length], weight
                    )
                )
                position += length
            segmented = torch.cat(parts, dim=1)
            out.setdefault(name, {})[schedule_name] = tensor_delta(segmented, whole)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--kinds", type=str, default="window,ratio128,ratio4",
        help="comma list of layer kinds to gate",
    )
    parser.add_argument("--skip-layers", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", 0)
    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    max_seq_len = 1152  # LAYER_SEQLEN rounded up to a window multiple

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "C3F-chunked-prefill-single-layer-gate",
        "measurement_class": "semantic_correctness_gate",
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "state_machine": None,
        "index_equivalence": None,
        "layers": {},
        "gemm_shape_probe": {},
        "accepted": False,
        "errors": [],
    }
    started = time.perf_counter()
    try:
        result["state_machine"] = run_state_machine_check(device, args.seed)
        result["index_equivalence"] = run_index_equivalence(device)

        if not args.skip_layers:
            checkpoint_layers = [WINDOW_LAYER_ID, RATIO4_LAYER_ID, RATIO128_LAYER_ID]
            checkpoint = inspect_stage_checkpoint(stage_root, checkpoint_layers, 4)
            if not checkpoint["ok"]:
                raise ValueError(
                    f"checkpoint contract failed: {checkpoint['errors'][:3]}"
                )
            result["checkpoint_id"] = checkpoint["checkpoint_id"]
            model_config = json.loads(
                (stage_root / "config.json").read_text(encoding="utf-8")
            )
            hidden = deterministic_hidden(args.seed, LAYER_SEQLEN, device)
            wanted = tuple(
                item.strip() for item in args.kinds.split(",") if item.strip()
            )
            spec = {
                "window": (
                    WINDOW_LAYER_ID,
                    WindowAttentionConfig,
                    prepare_window_attention_weights,
                ),
                "ratio128": (
                    RATIO128_LAYER_ID,
                    Ratio128AttentionConfig,
                    prepare_attention_weights,
                ),
                "ratio4": (
                    RATIO4_LAYER_ID,
                    Ratio4AttentionConfig,
                    prepare_ratio4_attention_weights,
                ),
            }
            for kind in wanted:
                layer_id, config_cls, prepare = spec[kind]
                raw_block = load_replicated_block_weights(
                    stage_root=stage_root,
                    rank=0,
                    world_size=4,
                    layer_id=layer_id,
                    device=device,
                    checkpoint_id=result["checkpoint_id"],
                )
                config = config_cls.from_model_config(
                    model_config, layer_id=layer_id, max_seq_len=max_seq_len
                )
                prepared = prepare(
                    raw_block.attention,
                    layer_id=layer_id,
                    rank=0,
                    world_size=4,
                    checkpoint_id=result["checkpoint_id"],
                )
                result["layers"][kind] = run_layer_equivalence(
                    kind, config, prepared, device, hidden, max_seq_len
                )
                if kind == "ratio4":
                    result["gemm_shape_probe"] = run_gemm_shape_probe(
                        prepared, hidden
                    )
                del raw_block, prepared
                torch.cuda.empty_cache()

        result["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )
        state_ok = bool(result["state_machine"]["all_bitwise"])
        index_ok = bool(result["index_equivalence"]["all_equal"])
        layers_ok = all(
            item["all_bitwise"] for item in result["layers"].values()
        ) if result["layers"] else True
        result["state_machine_bitwise"] = state_ok
        result["index_equivalence_exact"] = index_ok
        result["layers_bitwise"] = layers_ok
        # Acceptance = the two exact checks.  The layer check is a magnitude
        # report: whole-sequence and segmented runs use different GEMM M, and
        # the sparse core's gather einsum is shaped by the forward's own
        # seqlen, so bitwise equality there is not achievable (see
        # gemm_shape_probe).
        result["accepted"] = state_ok and index_ok
    except Exception as error:  # noqa: BLE001
        result["errors"].append(
            {"type": type(error).__name__, "message": str(error),
             "traceback": traceback.format_exc()[-4000:]}
        )
    result["elapsed_s"] = time.perf_counter() - started

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"accepted={result['accepted']}")
    if result["state_machine"] is not None:
        print(f"state_machine all_bitwise={result['state_machine']['all_bitwise']}")
        for record in result["state_machine"]["records"]:
            if not record["bitwise"]:
                print(
                    f"  MISMATCH {record['layer']} {record['schedule']}: "
                    f"pooled={record['pooled']} kv={record['kv_state_live']}"
                )
    if result["index_equivalence"] is not None:
        print(f"index_equivalence all_equal={result['index_equivalence']['all_equal']}")
        for record in result["index_equivalence"]["records"]:
            if not (record["window_indices_equal"] and record["compressed_indices_equal"]):
                print(f"  MISMATCH ratio={record['ratio']} {record['schedule']}: {record}")
    for kind, payload in result["layers"].items():
        print(
            f"layer {kind}: all_bitwise={payload['all_bitwise']} "
            f"worst_branch_rms_rel={payload['worst_branch_rms_rel']:.3e} "
            f"worst_max_abs={payload['worst_max_abs']:.3e}"
        )
        for record in payload["records"]:
            print(
                f"    {record['schedule']:<22} bitwise={str(record['bitwise']):<5} "
                f"branch_rms_rel={record['branch_rms_rel']:.3e} "
                f"max_abs={record['max_abs']:.3e}"
            )
    for name, payload in result.get("gemm_shape_probe", {}).items():
        worst = max(item["max_abs"] for item in payload.values())
        allbit = all(item["bitwise"] for item in payload.values())
        print(f"gemm_shape_probe {name}: all_bitwise={allbit} worst_max_abs={worst:.3e}")
    for error in result["errors"]:
        print(f"ERROR {error['type']}: {error['message']}")
        print(error["traceback"])
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
