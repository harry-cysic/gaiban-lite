#!/usr/bin/env python3
"""C2F: single-stage (TP4, 11 layers) chunked-prefill throughput bench.

Nineteenth vertical: establish the direct-runtime prefill baseline and wire
the two prefill-specific levers (W4A8 Marlin MoE, D0b fused Triton indexer).

Measurement口径 (stated once, used everywhere):

- ``--chunk`` is the *total* prefill length L (B=1 per rank).  With the
  default ``--prefill-chunk 0`` it is served by a single start_pos=0 forward,
  which equals the cost of the last chunk of an L-long chunked prefill (O(s^2)
  indexer/attention terms are at their end-of-sequence size), matching gaiban
  D0b's "chunk" definition -- this is the frozen C2F arm.
- 26th vertical: ``--prefill-chunk S`` instead splits L into consecutive
  S-long forwards using the 25th vertical's incremental (start_pos > 0)
  capability, and times the *whole* prefill.  Throughput stays 4*L / wall, so
  the two modes are directly comparable at equal delivered tokens.
- DP form: each TP lane feeds its own distinct B=1 sequence (per-rank seed),
  so one stage pass processes 4*L input tokens; attention runs per lane with
  full heads, MoE all-gathers 4*L rows (the runtime's native DP-attention +
  intermediate-TP MoE collective order, unchanged).
- input tok/s/stage = 4*L / stage_pass_wall.  Full PP4 pipeline projection
  P(16 cards) ~= 4*L / t_stage assumes perfect chunk interleaving across 4
  stages and stage-1-like layer mix (stages are timed on L11-L21; stage 0 is
  cheaper -- window layers -- and stage 3 has 10 layers + head).
- Timing: host walls around torch.cuda.synchronize + dist.barrier fencing
  (E1F convention); eager, no CUDA graphs; open-loop single stage -- no PP
  handoff, no decode interleaving.

Modes:
  --moe-mode w4a16|w4a8      Marlin activation dtype (w4a8 repacks at load
                             with is_a_8bit + FP8 scale processing; semantic
                             change, gated by --gate-w4a8 + E2E regression).
  --indexer ref|fused        ratio-4 prefill index score backend
                             (fused = D0b Triton kernel, active at
                             seqlen >= --fuse-min-seqlen; semantic change,
                             gated by --gate-indexer + E2E regression).
  --gate-indexer             extra paired pass (ref vs fused on identical
                             tensors: timings, score delta, top-k agreement).
  --gate-w4a8                extra layer-level A/B: load first stage layer's
                             MoE in both repacks, same hidden, compare.

Run (single node titan064, GPUs 0-3):
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=4 \
    c2f_prefill_stage_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash \
    --chunk 4096 --moe-mode w4a16 --indexer ref --out-dir out-c2f-...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import traceback
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from dsv4_direct.attention import rms_norm, torch_sparse_attention
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.hyper_connections import hc_post, hc_pre
from dsv4_direct.moe_runtime import TP4MoE, TP4MoEConfig
from dsv4_direct.ops.marlin_moe import load_resident_moe_layer
from dsv4_direct.physical_stage import (
    EXPECTED_TP_SIZE,
    PhysicalLayerMaterial,
    build_physical_stage,
)
from dsv4_direct.ratio4_fullpos import Ratio4FullPositionAttention


LOCAL_BATCH = 1
HIDDEN = 4096
HC_MULT = 4
FP8 = torch.float8_e4m3fn


# --------------------------------------------------------------------------
# allocator probe (22nd vertical): direct evidence for the prefill MoE
# bimodality.  Monotone counters are reported as per-call deltas; gauges are
# reported as absolute values before/after each MoE call.

ALLOC_COUNTERS = (
    "num_alloc_retries",
    "num_ooms",
    "num_device_alloc",
    "num_device_free",
    "num_sync_all_streams",
)
ALLOC_GAUGES = (
    "allocated_bytes.all.current",
    "reserved_bytes.all.current",
    "active_bytes.all.current",
    "inactive_split_bytes.all.current",
    "requested_bytes.all.current",
    "segment.all.current",
    "allocation.all.current",
)


def alloc_snapshot(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    snapshot = {key: int(stats.get(key, -1)) for key in ALLOC_COUNTERS}
    snapshot.update({key: int(stats.get(key, -1)) for key in ALLOC_GAUGES})
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    snapshot["driver_free_bytes"] = int(free_bytes)
    snapshot["driver_total_bytes"] = int(total_bytes)
    return snapshot


class MoEAllocProbe:
    """stage_marker hook: per-phase wall + allocator counters for one call."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.marks: list[dict[str, Any]] = []
        self.previous = time.perf_counter()

    def __call__(self, name: str) -> None:
        torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        stats = torch.cuda.memory_stats(self.device)
        self.marks.append(
            {
                "phase": name,
                "wall_ms": (now - self.previous) * 1e3,
                "num_alloc_retries": int(stats.get("num_alloc_retries", -1)),
                "num_device_alloc": int(stats.get("num_device_alloc", -1)),
                "num_device_free": int(stats.get("num_device_free", -1)),
                "reserved_bytes": int(stats.get("reserved_bytes.all.current", -1)),
                "allocated_bytes": int(stats.get("allocated_bytes.all.current", -1)),
            }
        )
        self.previous = now


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# stage lane (prefill-only variant of the e0ef2e StageLane, eager composition)


class PrefillLane:
    def __init__(
        self,
        materials: list[PhysicalLayerMaterial],
        *,
        device: torch.device,
        index_score_mode: str,
        fuse_min_seqlen: int,
        sparse_row_block: int | None,
        prefill_sparse_backend: str = "torch",
        hc_boundary_backend: Any | None = None,
    ) -> None:
        self.device = device
        self.hc_boundary_backend = hc_boundary_backend
        self.layers: list[tuple[PhysicalLayerMaterial, Any]] = []
        for material in materials:
            if material.kind == "ratio4":
                attention = Ratio4FullPositionAttention(
                    material.attention_config,
                    material.prepared,
                    batch_size=LOCAL_BATCH,
                    device=device,
                    kv_dtype=material.kv_dtype,
                    indexer_dtype=material.indexer_kv_dtype,
                    index_score_mode=index_score_mode,
                    fuse_min_seqlen=fuse_min_seqlen,
                    sparse_row_block=sparse_row_block,
                    prefill_sparse_backend=prefill_sparse_backend,
                )
            else:
                state = material.new_state(num_local_sequences=LOCAL_BATCH)
                attention = material.new_attention(state)
            self.layers.append((material, attention))

    @staticmethod
    def _attention_branch(
        material: PhysicalLayerMaterial,
        attention: Any,
        hidden: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        if material.kind == "ratio4":
            return attention(hidden, start_pos=start_pos)
        branch, _trace = attention(hidden, start_pos=start_pos)
        return branch

    def forward(
        self,
        residual: torch.Tensor,
        *,
        start_pos: int = 0,
        component_walls: dict[str, float] | None = None,
        device: torch.device | None = None,
        alloc_records: list[dict[str, Any]] | None = None,
        alloc_phase_layers: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """Eager 11-layer chain (== e0ef2e StageLane._layer_eager order)."""

        def timed(name: str, fn):
            if component_walls is None:
                return fn()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            value = fn()
            torch.cuda.synchronize(device)
            component_walls[name] = component_walls.get(name, 0.0) + (
                time.perf_counter() - started
            )
            return value

        def call_moe(
            layer_index: int,
            material: PhysicalLayerMaterial,
            hidden: torch.Tensor,
        ) -> torch.Tensor:
            if alloc_records is None:
                return material.moe.forward_tensor(hidden, slot=0)
            torch.cuda.synchronize(device)
            before = alloc_snapshot(device)
            probe = (
                MoEAllocProbe(device)
                if layer_index in alloc_phase_layers
                else None
            )
            started = time.perf_counter()
            value = material.moe.forward_tensor(
                hidden, slot=0, stage_marker=probe
            )
            torch.cuda.synchronize(device)
            call_wall = time.perf_counter() - started
            after = alloc_snapshot(device)
            record: dict[str, Any] = {
                "layer_index": layer_index,
                "kind": material.kind,
                "call_wall_ms": call_wall * 1e3,
            }
            for key in ALLOC_COUNTERS:
                record[f"delta_{key}"] = after[key] - before[key]
                record[f"total_{key}"] = after[key]
            for key in ALLOC_GAUGES:
                record[f"before_{key}"] = before[key]
                record[f"after_{key}"] = after[key]
            record["before_driver_free_bytes"] = before["driver_free_bytes"]
            record["after_driver_free_bytes"] = after["driver_free_bytes"]
            if probe is not None:
                record["phases"] = probe.marks
            alloc_records.append(record)
            return value

        if self.hc_boundary_backend is not None:
            return self._forward_fused_boundary(
                residual, start_pos=start_pos, timed=timed, call_moe=call_moe
            )

        for layer_index, (material, attention) in enumerate(self.layers):
            hc = material.raw_block.hyper_connection
            hidden, post, comb = timed(
                "hc",
                lambda: hc_pre(
                    residual,
                    hc.attn_fn,
                    hc.attn_scale,
                    hc.attn_base,
                    norm_eps=material.norm_eps,
                    sinkhorn_iters=material.sinkhorn_iters,
                    hc_eps=material.hc_eps,
                ),
            )
            hidden = timed(
                "norm",
                lambda: rms_norm(
                    hidden, material.raw_block.attn_norm, eps=material.norm_eps
                ),
            )
            branch = timed(
                f"attention_{material.kind}",
                lambda: self._attention_branch(
                    material, attention, hidden, start_pos
                ),
            )
            residual = timed("hc", lambda: hc_post(branch, residual, post, comb))
            hidden, post, comb = timed(
                "hc",
                lambda: hc_pre(
                    residual,
                    hc.ffn_fn,
                    hc.ffn_scale,
                    hc.ffn_base,
                    norm_eps=material.norm_eps,
                    sinkhorn_iters=material.sinkhorn_iters,
                    hc_eps=material.hc_eps,
                ),
            )
            hidden = timed(
                "norm",
                lambda: rms_norm(
                    hidden, material.raw_block.ffn_norm, eps=material.norm_eps
                ),
            )
            moe_output = timed(
                "moe",
                lambda: call_moe(layer_index, material, hidden),
            )
            residual = timed(
                "hc", lambda: hc_post(moe_output, residual, post, comb)
            )
        return residual

    def _forward_fused_boundary(
        self,
        residual: torch.Tensor,
        *,
        start_pos: int,
        timed: Any,
        call_moe: Any,
    ) -> torch.Tensor:
        """Boundary-fused 11-layer chain (23rd vertical, lever A).

        Same math, restructured: every ``hc_post`` is fused with the
        ``hc_pre`` + RMSNorm that immediately follows it.  An 11-layer stage
        has 21 such boundaries (11 intra-layer attn->ffn, 10 inter-layer
        ffn->next attn); the stage-entry ``hc_pre``+norm and the stage-exit
        ``hc_post`` have no fusion partner and stay eager.

        The ``hc`` bucket therefore *absorbs* the fused boundaries' RMSNorm --
        compare ``hc + norm`` across arms, not ``hc`` alone.
        """

        backend = self.hc_boundary_backend
        first_material = self.layers[0][0]
        first_hc = first_material.raw_block.hyper_connection
        hidden, post, comb = timed(
            "hc",
            lambda: hc_pre(
                residual,
                first_hc.attn_fn,
                first_hc.attn_scale,
                first_hc.attn_base,
                norm_eps=first_material.norm_eps,
                sinkhorn_iters=first_material.sinkhorn_iters,
                hc_eps=first_material.hc_eps,
            ),
        )
        hidden = timed(
            "norm",
            lambda: rms_norm(
                hidden,
                first_material.raw_block.attn_norm,
                eps=first_material.norm_eps,
            ),
        )
        for layer_index, (material, attention) in enumerate(self.layers):
            branch = timed(
                f"attention_{material.kind}",
                lambda: self._attention_branch(
                    material, attention, hidden, start_pos
                ),
            )
            hc = material.raw_block.hyper_connection
            residual, hidden, post, comb = timed(
                "hc",
                lambda: backend.post_pre_norm(
                    branch,
                    residual,
                    post,
                    comb,
                    hc_fn=hc.ffn_fn,
                    hc_scale=hc.ffn_scale,
                    hc_base=hc.ffn_base,
                    norm_weight=material.raw_block.ffn_norm,
                    norm_eps=material.norm_eps,
                    sinkhorn_iters=material.sinkhorn_iters,
                    hc_eps=material.hc_eps,
                ),
            )
            moe_output = timed(
                "moe",
                lambda: call_moe(layer_index, material, hidden),
            )
            if layer_index + 1 < len(self.layers):
                next_material = self.layers[layer_index + 1][0]
                next_hc = next_material.raw_block.hyper_connection
                residual, hidden, post, comb = timed(
                    "hc",
                    lambda: backend.post_pre_norm(
                        moe_output,
                        residual,
                        post,
                        comb,
                        hc_fn=next_hc.attn_fn,
                        hc_scale=next_hc.attn_scale,
                        hc_base=next_hc.attn_base,
                        norm_weight=next_material.raw_block.attn_norm,
                        norm_eps=next_material.norm_eps,
                        sinkhorn_iters=next_material.sinkhorn_iters,
                        hc_eps=next_material.hc_eps,
                    ),
                )
            else:
                residual = timed(
                    "hc",
                    lambda: hc_post(moe_output, residual, post, comb),
                )
        return residual

    def collect_index_gate_records(self) -> list[dict]:
        records: list[dict] = []
        for material, attention in self.layers:
            if material.kind == "ratio4":
                records.extend(attention.index_gate_records)
        return records


# --------------------------------------------------------------------------
# standalone correctness spot-checks (cheap, run once per bench)


def ring_wrap_unit_check(device: torch.device) -> dict[str, Any]:
    """Our index_copy_ ring placement == reference model.py:518-523 split."""

    results = {}
    window = 128
    for seqlen in (96, 128, 130, 1024 + 7, 2048):
        kv = torch.randn(1, seqlen, 512, dtype=torch.bfloat16, device=device)
        # reference form
        ref = torch.zeros(1, window, 512, dtype=torch.bfloat16, device=device)
        if seqlen <= window:
            ref[:, :seqlen] = kv
        else:
            cutoff = seqlen % window
            tail, head = kv[:, -window:].split([window - cutoff, cutoff], dim=1)
            ref[:, cutoff:window] = tail
            ref[:, :cutoff] = head
        # runtime form (ratio4_fullpos prefill ring write)
        ours = torch.zeros_like(ref)
        kept = min(seqlen, window)
        slots = torch.arange(seqlen - kept, seqlen, device=device).remainder(
            window
        )
        ours.index_copy_(1, slots, kv[:, seqlen - kept :].contiguous())
        results[str(seqlen)] = bool(torch.equal(ref, ours))
    results["pass"] = all(bool(v) for v in results.values())
    return results


def ref_score_row_block_unit_check(device: torch.device) -> dict[str, Any]:
    """Row-blocked ref index scoring is bitwise identical to single-shot."""

    generator = torch.Generator(device=device)
    generator.manual_seed(20260722)
    q = torch.randn(
        1, 2048, 64, 128, dtype=torch.bfloat16, device=device, generator=generator
    )
    kv = torch.randn(
        1, 512, 128, dtype=torch.bfloat16, device=device, generator=generator
    )
    w = torch.randn(
        1, 2048, 64, dtype=torch.bfloat16, device=device, generator=generator
    )
    whole = Ratio4FullPositionAttention._ref_index_scores(q, kv, w)
    Ratio4FullPositionAttention._REF_SCORE_ROW_BLOCK_OVERRIDE = 640
    try:
        blocked = Ratio4FullPositionAttention._ref_index_scores(q, kv, w)
    finally:
        Ratio4FullPositionAttention._REF_SCORE_ROW_BLOCK_OVERRIDE = None
    return {"bitwise_equal": bool(torch.equal(whole, blocked))}


def sliced_sparse_unit_check(device: torch.device) -> dict[str, Any]:
    """Row-blocked torch_sparse_attention is bitwise identical to one call."""

    generator = torch.Generator(device=device)
    generator.manual_seed(20260721)
    s, heads, dim, window, t_rows = 1024, 64, 512, 128, 256
    query = torch.randn(
        1, s, heads, dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    kv = torch.randn(
        1, s + t_rows, dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    sink = torch.randn(
        heads, dtype=torch.float32, device=device, generator=generator
    )
    base = torch.arange(s, device=device).unsqueeze(1)
    columns = torch.arange(window, device=device)
    win = ((base - window + 1).clamp(0) + columns)
    win = torch.where(win > base, -1, win)
    comp = torch.arange(t_rows, device=device).repeat(s, 1)
    visible = torch.arange(1, s + 1, device=device).unsqueeze(1) // 4
    comp = torch.where(comp >= visible, -1, comp + s)
    topk = (
        torch.cat((win, comp), dim=-1).unsqueeze(0).to(torch.int32).contiguous()
    )
    whole = torch_sparse_attention(query, kv, sink, topk, dim**-0.5)
    sliced = torch.cat(
        [
            torch_sparse_attention(
                query[:, begin : begin + 256],
                kv,
                sink,
                topk[:, begin : begin + 256],
                dim**-0.5,
            )
            for begin in range(0, s, 256)
        ],
        dim=1,
    )
    return {"bitwise_equal": bool(torch.equal(whole, sliced))}


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="11-21", help="first-last inclusive")
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument(
        "--prefill-chunk", type=int, default=0,
        help="26th vertical: segment length for a *real* chunked prefill.  0 "
        "(default) keeps the historical single whole-sequence forward at "
        "start_pos=0, so every frozen arm is reproduced verbatim.  With a "
        "positive value the --chunk-long prefill is split into consecutive "
        "start_pos>0 multi-token forwards (the 25th vertical's incremental "
        "capability); the MoE per-shape buffers are then registered on the "
        "*segment* row counts, not the total",
    )
    parser.add_argument("--moe-mode", choices=["w4a16", "w4a8"], default="w4a16")
    parser.add_argument("--indexer", choices=["ref", "fused"], default="ref")
    parser.add_argument("--fuse-min-seqlen", type=int, default=1024)
    parser.add_argument("--row-block", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument(
        "--sparse-backend", choices=["torch", "tilelang"], default="torch",
        help="prefill sparse attention core (21st vertical); tilelang uses "
        "the reference kernel with an sm89 head loop for every prefill call "
        "(ratio-4, ratio-128, window) and ignores --row-block, which exists "
        "only to bound the torch core's FP32 gather workspace",
    )
    parser.add_argument(
        "--sparse-head-chunk", type=int, default=16,
        help="tilelang head-loop width (16 is the sm89 maximum)",
    )
    parser.add_argument(
        "--gate-sparse", action="store_true",
        help="layer-level A/B of the prefill sparse core on real weights "
        "(torch vs tilelang, identical residual) -- the ratio-4 prefill gate, "
        "which has no single-layer oracle",
    )
    parser.add_argument(
        "--alloc-probe", action="store_true",
        help="22nd vertical: extra passes recording per-MoE-call allocator "
        "counters (num_alloc_retries / num_device_alloc / num_device_free) "
        "plus per-phase marks, to attribute the prefill MoE bimodality",
    )
    parser.add_argument(
        "--alloc-probe-passes", type=int, default=2,
        help="how many probe passes to record (the first is cold)",
    )
    parser.add_argument(
        "--hc-backend", choices=["default", "fused", "fused-nosplit"],
        default="default",
        help="23rd vertical lever A: 'fused' restructures the prefill chain "
        "around the TileLang hc_post+hc_pre boundary kernel (row-blocked to "
        "stay off the broken >=1024-row branch); 'default' is the eager "
        "per-op chain.  'fused-nosplit' is a diagnostic arm only",
    )
    parser.add_argument(
        "--gate-hc", action="store_true",
        help="23rd vertical lever A gate: run the eager and fused chains on "
        "the same residual and report per-boundary plus compounded "
        "stage-output deltas",
    )
    parser.add_argument(
        "--moe-overlap", choices=["off", "on"], default="off",
        help="23rd vertical lever B: overlap the TP4 MoE all-gather / "
        "reduce-scatter with the routed-expert compute by row-blocked "
        "pipelining (prefill only; no CUDA graph)",
    )
    parser.add_argument(
        "--moe-overlap-blocks", type=int, default=4,
        help="lever B row-block count for the collective/compute pipeline",
    )
    parser.add_argument(
        "--gate-moe-overlap", action="store_true",
        help="23rd vertical lever B gate: same hidden through the sequential "
        "and overlapped MoE, reported bitwise plus rel_fro",
    )
    parser.add_argument("--gate-indexer", action="store_true")
    parser.add_argument("--gate-w4a8", action="store_true")
    parser.add_argument("--progress-every", type=int, default=128)
    args = parser.parse_args()

    # ratio-128 prefill sparse-core row blocking (bitwise identical; bounds
    # the FP32 gather workspace) -- must be set before any prefill call.
    if args.row_block > 0:
        os.environ["DSV4_PREFILL_SPARSE_ROW_BLOCK"] = str(args.row_block)
    # ratio-128 / window prefill sparse core (the ratio-4 layer takes it as a
    # constructor argument); must also be set before any prefill call.
    os.environ["DSV4_PREFILL_SPARSE_BACKEND"] = args.sparse_backend
    os.environ["DSV4_PREFILL_SPARSE_HEAD_CHUNK"] = str(args.sparse_head_chunk)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(device)

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    first_layer, last_layer = (int(v) for v in args.layers.split("-"))
    layer_ids = tuple(range(first_layer, last_layer + 1))
    chunk = int(args.chunk)
    if chunk <= 0 or chunk % 128:
        raise ValueError("chunk must be a positive multiple of 128")
    max_seq_len = chunk
    global_rows = LOCAL_BATCH * chunk * EXPECTED_TP_SIZE
    marlin_input_dtype = FP8 if args.moe_mode == "w4a8" else None

    # ---- 26th vertical: segmented (真分段) prefill plan -------------------
    # ``--chunk`` is the *total* prefill length L (unchanged).  ``--prefill-
    # chunk S`` splits it into consecutive forwards; S == 0 or S >= L keeps the
    # single whole-sequence forward, i.e. the frozen C2F arm bit-for-bit.
    prefill_chunk = int(args.prefill_chunk)
    if prefill_chunk < 0 or (prefill_chunk and prefill_chunk % 128):
        raise ValueError("--prefill-chunk must be 0 or a positive multiple of 128")
    prefill_plan: list[tuple[int, int]] = []
    if prefill_chunk and prefill_chunk < chunk:
        position = 0
        while position < chunk:
            length = min(prefill_chunk, chunk - position)
            prefill_plan.append((position, length))
            position += length
    else:
        prefill_plan.append((0, chunk))
    segment_lengths = [length for _position, length in prefill_plan]
    # MoE per-shape buffer registration follows the *forward* row count.  With
    # a single whole-sequence forward this is exactly ``(4*L,)`` -- the frozen
    # registration -- so the control arm is unchanged.
    global_row_shapes = tuple(
        sorted({LOCAL_BATCH * length * EXPECTED_TP_SIZE for length in segment_lengths})
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "C2F-prefill-stage-bench",
        "measurement_class": "open_loop_single_stage_prefill",
        "koujing": (
            "prefill of a --chunk-long sequence, B=1/lane, 4 DP lanes with "
            "distinct sequences; --prefill-chunk 0 does it in one start_pos=0 "
            "forward (the frozen arm), >0 splits it into consecutive "
            "start_pos>0 forwards; input tok/s/stage = 4*chunk / wall of the "
            "*whole* prefill either way; host walls around "
            "torch.cuda.synchronize; eager, no CUDA graph, no PP handoff"
        ),
        "rank": rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": package_version("vllm"),
        "triton": package_version("triton"),
        "layers": list(layer_ids),
        "chunk": chunk,
        "global_rows": global_rows,
        # 26th vertical: segmentation evidence.  ``prefill_forwards == 1`` and
        # ``segment_lengths == [chunk]`` is the whole-sequence control arm.
        "prefill_chunk": prefill_chunk,
        "prefill_forwards": len(prefill_plan),
        "segment_lengths": list(segment_lengths),
        "segment_start_positions": [position for position, _ in prefill_plan],
        "global_row_shapes": list(global_row_shapes),
        "moe_mode": args.moe_mode,
        "indexer": args.indexer,
        "fuse_min_seqlen": args.fuse_min_seqlen,
        "row_block": args.row_block,
        "sparse_backend": args.sparse_backend,
        "sparse_head_chunk": args.sparse_head_chunk,
        "hc_backend": args.hc_backend,
        "moe_overlap": args.moe_overlap,
        "moe_overlap_blocks": args.moe_overlap_blocks,
        "iters": args.iters,
        "warmup": args.warmup,
        "seed": args.seed,
        "input_scale": args.input_scale,
        # 22nd vertical: the fast/slow MoE runs were indistinguishable in the
        # recorded metadata, so the allocator configuration is now always
        # recorded (it is the first thing to check when the bimodality moves).
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "allocator_backend": torch.cuda.get_allocator_backend(),
        # The MoE bimodality was NCCL transport selection, not the allocator:
        # without NCCL_P2P_LEVEL=SYS the TP4 collectives fall back to SHM.
        "nccl_p2p_level": os.environ.get("NCCL_P2P_LEVEL", ""),
        "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "errors": [],
    }

    try:
        if world != EXPECTED_TP_SIZE:
            raise ValueError("bench requires exactly 4 ranks (single TP4 stage)")
        tp_group = dist.new_group(ranks=list(range(EXPECTED_TP_SIZE)), backend="nccl")

        # ---- MoE collective self-check (22nd vertical) ------------------
        # The prefill MoE bucket is ~40% NCCL even on the fast path, and the
        # SHM fallback is silent.  Measure the two MoE collectives at the real
        # shapes so every result carries its own transport evidence.
        def collective_selfcheck(rows_global: int) -> dict[str, Any]:
            probe_local = torch.zeros(
                rows_global // EXPECTED_TP_SIZE,
                HIDDEN,
                dtype=torch.bfloat16,
                device=device,
            )
            probe_gathered = torch.zeros(
                rows_global, HIDDEN, dtype=torch.bfloat16, device=device
            )
            probe_reduced = torch.zeros_like(probe_local)
            collective_walls: dict[str, list[float]] = {
                "all_gather": [], "reduce_scatter": []
            }
            for probe_iteration in range(5):
                torch.cuda.synchronize(device)
                dist.barrier()
                started = time.perf_counter()
                dist.all_gather_into_tensor(
                    probe_gathered, probe_local, group=tp_group
                )
                torch.cuda.synchronize(device)
                if probe_iteration >= 2:
                    collective_walls["all_gather"].append(
                        time.perf_counter() - started
                    )
                dist.barrier()
                started = time.perf_counter()
                dist.reduce_scatter_tensor(
                    probe_reduced, probe_gathered, op=dist.ReduceOp.SUM,
                    group=tp_group,
                )
                torch.cuda.synchronize(device)
                if probe_iteration >= 2:
                    collective_walls["reduce_scatter"].append(
                        time.perf_counter() - started
                    )
            payload = probe_gathered.numel() * probe_gathered.element_size()
            bus_bytes = (EXPECTED_TP_SIZE - 1) / EXPECTED_TP_SIZE * payload
            del probe_local, probe_gathered, probe_reduced
            return {
                "rows_global": int(rows_global),
                "payload_bytes": int(payload),
                "all_gather_p50_ms": statistics.median(
                    collective_walls["all_gather"]
                ) * 1e3,
                "all_gather_bus_gbps": bus_bytes
                / statistics.median(collective_walls["all_gather"])
                / 1e9,
                "reduce_scatter_p50_ms": statistics.median(
                    collective_walls["reduce_scatter"]
                )
                * 1e3,
                "reduce_scatter_bus_gbps": bus_bytes
                / statistics.median(collective_walls["reduce_scatter"])
                / 1e9,
                "note": "SHM fallback lands near 4 GB/s; direct P2P near 24 GB/s",
            }

        # Headline self-check keeps the *total* payload so its bus numbers stay
        # directly comparable across every arm and with the frozen runs.
        result["moe_collective_selfcheck"] = collective_selfcheck(global_rows)
        # 26th vertical: the segmented arm's MoE actually moves one segment per
        # forward, so the per-segment transport efficiency is what its MoE
        # bucket sees.  Recorded separately; never replaces the headline.
        if len(global_row_shapes) == 1 and global_row_shapes[0] != global_rows:
            result["moe_collective_selfcheck_segment"] = collective_selfcheck(
                global_row_shapes[0]
            )

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                contract = inspect_stage_checkpoint(stage_root, tp_size=world)
                if not contract["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {contract['errors'][:3]}"
                    )
                envelope_holder[0] = {
                    "ok": True,
                    "config": config,
                    "checkpoint_id": contract["checkpoint_id"],
                }
            except Exception:
                envelope_holder[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        if not envelope["ok"]:
            raise ValueError(f"rank-0 preflight failed:\n{envelope['error']}")
        config = envelope["config"]
        result["checkpoint_id"] = envelope["checkpoint_id"]

        # ---------------- correctness spot-checks (cheap) ----------------
        result["ring_wrap_unit_check"] = ring_wrap_unit_check(device)
        result["sliced_sparse_unit_check"] = sliced_sparse_unit_check(device)
        result["ref_score_row_block_unit_check"] = ref_score_row_block_unit_check(
            device
        )
        if not result["ring_wrap_unit_check"]["pass"]:
            raise RuntimeError("ring wrap unit check failed")
        if not result["sliced_sparse_unit_check"]["bitwise_equal"]:
            raise RuntimeError("sliced sparse unit check failed")
        if not result["ref_score_row_block_unit_check"]["bitwise_equal"]:
            raise RuntimeError("ref score row-block unit check failed")

        # ---------------- stage load ----------------
        load_started = time.perf_counter()
        stage = build_physical_stage(
            stage_id=1,
            layer_ids=layer_ids,
            model_config=config,
            stage_root=stage_root,
            tp_rank=rank,
            tp_group=tp_group,
            tp_global_ranks=tuple(range(EXPECTED_TP_SIZE)),
            device=device,
            checkpoint_id=envelope["checkpoint_id"],
            max_seq_len=max_seq_len,
            global_row_shapes=global_row_shapes,
            slots_per_shape=1,
            progress_every=args.progress_every,
            progress=(lambda m: print(m, flush=True)) if rank == 0 else None,
            moe_marlin_input_dtype=marlin_input_dtype,
            share_moe_buffers=True,
        )
        torch.cuda.synchronize(device)
        dist.barrier()
        result["load_seconds"] = time.perf_counter() - load_started
        result["moe_resident_bytes_layer0"] = int(
            stage.materials[0].moe.resident.resident_bytes
        )
        # 26th vertical: measure the MoE per-shape (Marlin) buffer bytes that
        # the registration actually costs, so the segmented arm's memory win
        # can be split into (b) buffer registration and (a) the forward's own
        # activations rather than being asserted.  ``share_moe_buffers=True``
        # means one set backs all 11 layers.
        moe_buffer_bytes = 0
        seen_storage: set[int] = set()

        def account(value: Any) -> None:
            nonlocal moe_buffer_bytes
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                key = storage.data_ptr()
                if key in seen_storage:
                    return
                seen_storage.add(key)
                moe_buffer_bytes += storage.nbytes()
            elif isinstance(value, (list, tuple)):
                for member in value:
                    account(member)
            elif hasattr(value, "__dataclass_fields__"):
                for field_name in value.__dataclass_fields__:
                    account(getattr(value, field_name))

        for buffers in stage.materials[0].moe._buffers.values():
            account(buffers)
        result["moe_shape_buffer_bytes"] = int(moe_buffer_bytes)
        result["moe_registered_global_rows"] = list(
            stage.materials[0].moe.registered_global_rows
        )
        result["memory_allocated_bytes_after_load"] = int(
            torch.cuda.memory_allocated(device)
        )
        for material in stage.materials:
            if material.route_kind != "learned":
                raise ValueError(
                    "bench stage must be all-learned routing (no input_ids fed)"
                )

        materials = list(stage.materials)
        row_block = args.row_block if args.row_block > 0 else None
        if args.moe_overlap == "on":
            for material in materials:
                material.moe.enable_collective_overlap(args.moe_overlap_blocks)
            result["moe_overlap_active"] = [
                material.moe.collective_overlap_blocks for material in materials
            ]
        hc_boundary_backend = resolve_hc_boundary_backend(
            None if args.hc_backend == "default" else args.hc_backend
        )
        if hc_boundary_backend is not None:
            result["hc_backend_max_rows"] = hc_boundary_backend.max_rows
            # The 21 fused boundaries must all share the chain's HC
            # hyper-parameters: the backend takes one norm_eps/sinkhorn/hc_eps
            # per call, and the inter-layer boundary mixes layer i's residual
            # with layer i+1's hc_pre parameters.
            for field in ("norm_eps", "sinkhorn_iters", "hc_eps"):
                values = {getattr(m, field) for m in materials}
                if len(values) != 1:
                    raise ValueError(
                        f"fused HC boundary requires a uniform {field} across "
                        f"the stage, got {sorted(values)}"
                    )

        def make_lane(
            index_score_mode: str,
            sparse_backend: str | None = None,
            hc_backend: Any | None = "default",
        ) -> PrefillLane:
            return PrefillLane(
                materials,
                device=device,
                index_score_mode=index_score_mode,
                fuse_min_seqlen=args.fuse_min_seqlen,
                sparse_row_block=row_block,
                prefill_sparse_backend=sparse_backend or args.sparse_backend,
                hc_boundary_backend=(
                    hc_boundary_backend if hc_backend == "default" else hc_backend
                ),
            )

        def make_residual(iteration: int) -> torch.Tensor:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + 1000 * rank + iteration)
            return (
                torch.randn(
                    LOCAL_BATCH,
                    chunk,
                    HC_MULT,
                    HIDDEN,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                * args.input_scale
            ).contiguous()

        def run_prefill(
            lane: PrefillLane,
            residual: torch.Tensor,
            *,
            segment_walls: list[float] | None = None,
            **forward_kwargs: Any,
        ) -> torch.Tensor:
            """Drive the whole prefill, one forward per planned segment.

            With the default single-element plan this is exactly the frozen
            ``lane.forward(residual, start_pos=0)`` call.  Slicing dim 1 of the
            contiguous ``[1, L, 4, H]`` residual yields a contiguous view (dim 0
            is size 1), so the segmented arm pays no extra copy.  Only the last
            segment's output is retained -- the earlier ones would have been
            handed to the next PP stage and freed.
            """

            output: torch.Tensor | None = None
            for position, length in prefill_plan:
                if segment_walls is not None:
                    torch.cuda.synchronize(device)
                    segment_started = time.perf_counter()
                output = lane.forward(
                    residual[:, position : position + length],
                    start_pos=position,
                    **forward_kwargs,
                )
                if segment_walls is not None:
                    torch.cuda.synchronize(device)
                    segment_walls.append(time.perf_counter() - segment_started)
            assert output is not None
            return output

        # ---------------- timed prefill passes ----------------
        # 26th vertical: isolate the *forward's own* peak from the load peak.
        # The headline ``max_memory_allocated_bytes`` still covers the whole
        # process (max() is restored below), so the frozen number's meaning is
        # unchanged; this window additionally answers "what does one prefill
        # cost on top of the resident weights", which is what segmenting bounds.
        peak_allocated_before_timed = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_before_timed = int(torch.cuda.max_memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        walls: list[float] = []
        for iteration in range(args.warmup + args.iters):
            lane = make_lane(args.indexer)
            residual = make_residual(iteration)
            torch.cuda.synchronize(device)
            dist.barrier()
            started = time.perf_counter()
            output = run_prefill(lane, residual)
            torch.cuda.synchronize(device)
            dist.barrier()
            wall = time.perf_counter() - started
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError(f"non-finite stage output at iter {iteration}")
            if iteration >= args.warmup:
                walls.append(wall)
            if rank == 0:
                print(
                    f"iter {iteration} ({'warmup' if iteration < args.warmup else 'timed'}): "
                    f"{wall * 1e3:.1f} ms",
                    flush=True,
                )
            del lane, residual, output

        result["prefill_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        result["prefill_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
        result["prefill_activation_peak_bytes"] = (
            result["prefill_peak_allocated_bytes"]
            - result["memory_allocated_bytes_after_load"]
        )

        p50 = statistics.median(walls)
        result["stage_pass_walls_s"] = walls
        result["stage_pass_wall_p50_s"] = p50
        result["stage_pass_wall_mean_s"] = statistics.fmean(walls)
        result["input_tok_s_per_stage_dp4"] = 4 * chunk / p50
        result["input_tok_s_per_lane"] = chunk / p50
        result["pipeline16_projection_tok_s"] = 4 * chunk / p50

        # ---------------- one instrumented pass (component walls) --------
        lane = make_lane(args.indexer)
        residual = make_residual(10_000)
        component_walls: dict[str, float] = {}
        segment_walls: list[float] = []
        torch.cuda.synchronize(device)
        dist.barrier()
        started = time.perf_counter()
        run_prefill(
            lane,
            residual,
            segment_walls=segment_walls,
            component_walls=component_walls,
            device=device,
        )
        torch.cuda.synchronize(device)
        component_walls["total_instrumented"] = time.perf_counter() - started
        result["component_walls_s"] = component_walls
        result["segment_walls_s"] = segment_walls
        del lane, residual

        # ---------------- allocator probe pass (optional) ----------------
        if args.alloc_probe:
            probe_passes = []
            for probe_index in range(max(1, args.alloc_probe_passes)):
                lane = make_lane(args.indexer)
                residual = make_residual(40_000 + probe_index)
                alloc_records: list[dict[str, Any]] = []
                probe_walls: dict[str, float] = {}
                torch.cuda.synchronize(device)
                dist.barrier()
                started = time.perf_counter()
                run_prefill(
                    lane,
                    residual,
                    component_walls=probe_walls,
                    device=device,
                    alloc_records=alloc_records,
                    # phase marks on one ratio-4 and one ratio-128 layer
                    alloc_phase_layers=(0, 1),
                )
                torch.cuda.synchronize(device)
                probe_walls["total_instrumented"] = time.perf_counter() - started
                probe_passes.append(
                    {
                        "pass_index": probe_index,
                        "component_walls_s": probe_walls,
                        "moe_calls": alloc_records,
                    }
                )
                del lane, residual
            result["alloc_probe"] = {
                "form": "per-MoE-call allocator counters; monotone counters "
                "are per-call deltas, gauges are absolute before/after",
                "passes": probe_passes,
            }

        # ---------------- prefill sparse-core A/B gate (optional) --------
        # The ratio-4 prefill path has no single-layer oracle (e0ff is frozen
        # to saturated decode positions), so this is its layer-level gate:
        # identical residual through two freshly built lanes, per-layer
        # attention branch plus the compounded 11-layer stage output.
        if args.gate_sparse:

            def delta(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
                reference = reference.float()
                candidate = candidate.float()
                difference = candidate - reference
                return {
                    "rel_fro": float(
                        torch.linalg.norm(difference)
                        / torch.linalg.norm(reference)
                    ),
                    "max_abs": float(difference.abs().max()),
                    "reference_max_abs": float(reference.abs().max()),
                    "nonfinite": int((~torch.isfinite(candidate)).sum()),
                }

            # Two lanes driven in lockstep: each layer's attention is called
            # once per backend on the *same* hidden, so every delta is the
            # local kernel error rather than the compounded chain divergence.
            # The chain then advances on the torch branch for both, keeping
            # their states on one trajectory.  ratio-128 layers read the
            # backend from the environment, so it is flipped per call.
            control = make_lane(args.indexer, sparse_backend="torch")
            candidate_lane = make_lane(args.indexer, sparse_backend="tilelang")
            residual = make_residual(30_000)
            per_layer = []
            for index, (material, control_attention) in enumerate(control.layers):
                _, candidate_attention = candidate_lane.layers[index]
                hc = material.raw_block.hyper_connection
                hidden, post, comb = hc_pre(
                    residual,
                    hc.attn_fn,
                    hc.attn_scale,
                    hc.attn_base,
                    norm_eps=material.norm_eps,
                    sinkhorn_iters=material.sinkhorn_iters,
                    hc_eps=material.hc_eps,
                )
                hidden = rms_norm(
                    hidden, material.raw_block.attn_norm, eps=material.norm_eps
                )
                os.environ["DSV4_PREFILL_SPARSE_BACKEND"] = "torch"
                control_branch = PrefillLane._attention_branch(
                    material, control_attention, hidden, 0
                )
                os.environ["DSV4_PREFILL_SPARSE_BACKEND"] = "tilelang"
                candidate_branch = PrefillLane._attention_branch(
                    material, candidate_attention, hidden, 0
                )
                torch.cuda.synchronize(device)
                entry = {"layer_index": index, "kind": material.kind}
                entry.update(delta(control_branch, candidate_branch))
                per_layer.append(entry)
                del candidate_branch

                residual = hc_post(control_branch, residual, post, comb)
                hidden, post, comb = hc_pre(
                    residual,
                    hc.ffn_fn,
                    hc.ffn_scale,
                    hc.ffn_base,
                    norm_eps=material.norm_eps,
                    sinkhorn_iters=material.sinkhorn_iters,
                    hc_eps=material.hc_eps,
                )
                hidden = rms_norm(
                    hidden, material.raw_block.ffn_norm, eps=material.norm_eps
                )
                residual = hc_post(
                    material.moe.forward_tensor(hidden, slot=0),
                    residual,
                    post,
                    comb,
                )
                del control_branch, hidden
            os.environ["DSV4_PREFILL_SPARSE_BACKEND"] = args.sparse_backend
            result["sparse_gate"] = {
                "form": "layer-locked: both backends see the same hidden; the "
                "chain advances on the torch branch",
                "per_layer_branch": per_layer,
                "worst_branch_rel_fro": max(e["rel_fro"] for e in per_layer),
                "worst_by_kind": {
                    kind: max(
                        e["rel_fro"] for e in per_layer if e["kind"] == kind
                    )
                    for kind in sorted({e["kind"] for e in per_layer})
                },
            }
            del control, candidate_lane, residual
            torch.cuda.empty_cache()

        # ---------------- HC boundary chain gate (optional) --------------
        # 23rd vertical lever A.  The prefill chain has no single-layer HC
        # oracle, so this is its gate: one residual through both chains, the
        # per-boundary delta measured *locked* (both backends see the same
        # inputs, so each delta is the local kernel error), plus the freely
        # compounded 11-layer stage output, which is what the E2E golden gate
        # actually consumes.
        if args.gate_hc:

            def hc_delta(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
                reference = reference.float()
                candidate = candidate.float()
                difference = candidate - reference
                return {
                    "rel_fro": float(
                        difference.norm()
                        / torch.linalg.norm(reference).clamp_min(1e-30)
                    ),
                    "max_abs": float(difference.abs().max()),
                    "reference_max_abs": float(reference.abs().max()),
                    "bitwise_equal": bool(torch.equal(reference, candidate)),
                    "nonfinite": int((~torch.isfinite(candidate)).sum()),
                }

            gate_backend = resolve_hc_boundary_backend("fused")
            eager_backend = resolve_hc_boundary_backend("eager")
            gate_lane = make_lane(args.indexer, hc_backend=None)
            residual = make_residual(50_000)

            # (a) locked per-boundary A/B on the real chain's tensors.
            locked: list[dict[str, Any]] = []
            probe_residual = residual
            first_material = materials[0]
            first_hc = first_material.raw_block.hyper_connection
            hidden, post, comb = hc_pre(
                probe_residual,
                first_hc.attn_fn,
                first_hc.attn_scale,
                first_hc.attn_base,
                norm_eps=first_material.norm_eps,
                sinkhorn_iters=first_material.sinkhorn_iters,
                hc_eps=first_material.hc_eps,
            )
            hidden = rms_norm(
                hidden,
                first_material.raw_block.attn_norm,
                eps=first_material.norm_eps,
            )
            for index, (material, attention) in enumerate(gate_lane.layers):
                branch = PrefillLane._attention_branch(
                    material, attention, hidden, 0
                )
                hcw = material.raw_block.hyper_connection
                boundary_kwargs = dict(
                    hc_fn=hcw.ffn_fn,
                    hc_scale=hcw.ffn_scale,
                    hc_base=hcw.ffn_base,
                    norm_weight=material.raw_block.ffn_norm,
                    norm_eps=material.norm_eps,
                    sinkhorn_iters=material.sinkhorn_iters,
                    hc_eps=material.hc_eps,
                )
                reference = eager_backend.post_pre_norm(
                    branch, probe_residual, post, comb, **boundary_kwargs
                )
                candidate = gate_backend.post_pre_norm(
                    branch, probe_residual, post, comb, **boundary_kwargs
                )
                torch.cuda.synchronize(device)
                entry = {
                    "boundary": f"L{index}-intra-attn2ffn",
                    "kind": material.kind,
                }
                for name, position in (
                    ("residual", 0), ("hidden", 1), ("post", 2), ("comb", 3),
                ):
                    entry[name] = hc_delta(reference[position], candidate[position])
                locked.append(entry)
                del candidate
                # advance the chain on the eager branch so both stay on one
                # trajectory
                probe_residual, hidden, post, comb = reference
                moe_output = material.moe.forward_tensor(hidden, slot=0)
                if index + 1 < len(gate_lane.layers):
                    next_material = materials[index + 1]
                    nhc = next_material.raw_block.hyper_connection
                    inter_kwargs = dict(
                        hc_fn=nhc.attn_fn,
                        hc_scale=nhc.attn_scale,
                        hc_base=nhc.attn_base,
                        norm_weight=next_material.raw_block.attn_norm,
                        norm_eps=next_material.norm_eps,
                        sinkhorn_iters=next_material.sinkhorn_iters,
                        hc_eps=next_material.hc_eps,
                    )
                    reference = eager_backend.post_pre_norm(
                        moe_output, probe_residual, post, comb, **inter_kwargs
                    )
                    candidate = gate_backend.post_pre_norm(
                        moe_output, probe_residual, post, comb, **inter_kwargs
                    )
                    torch.cuda.synchronize(device)
                    entry = {
                        "boundary": f"L{index}-inter-ffn2attn",
                        "kind": material.kind,
                    }
                    for name, position in (
                        ("residual", 0), ("hidden", 1), ("post", 2), ("comb", 3),
                    ):
                        entry[name] = hc_delta(
                            reference[position], candidate[position]
                        )
                    locked.append(entry)
                    del candidate
                    probe_residual, hidden, post, comb = reference
                else:
                    probe_residual = hc_post(
                        moe_output, probe_residual, post, comb
                    )
                del branch, moe_output
            del gate_lane
            torch.cuda.empty_cache()

            # (b) compounded stage output: two independent full chains.
            eager_lane = make_lane(args.indexer, hc_backend=None)
            eager_out = eager_lane.forward(residual, start_pos=0)
            del eager_lane
            torch.cuda.empty_cache()
            fused_lane = make_lane(args.indexer, hc_backend=gate_backend)
            fused_out = fused_lane.forward(residual, start_pos=0)
            del fused_lane
            torch.cuda.synchronize(device)
            result["hc_gate"] = {
                "form": "locked per-boundary A/B (both backends on identical "
                "inputs, chain advanced on eager) + compounded 11-layer stage "
                "output from two independent chains",
                "boundaries": len(locked),
                "per_boundary": locked,
                "worst_by_tensor": {
                    name: max(entry[name]["rel_fro"] for entry in locked)
                    for name in ("residual", "hidden", "post", "comb")
                },
                "stage_output": hc_delta(eager_out, fused_out),
            }
            del eager_out, fused_out, residual
            torch.cuda.empty_cache()

        # ---------------- MoE overlap gate (optional) --------------------
        # 23rd vertical lever B.  The pipelined path permutes the gathered row
        # order (block-major instead of rank-major), which changes which
        # Marlin M-block a row lands in.  Everything between the collectives is
        # row-local, so per-row bitwise identity is expected; this measures it
        # on the real layer rather than assuming it.
        if args.gate_moe_overlap:
            gate_layer = materials[0]
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + 4242 + rank)
            gate_hidden = (
                torch.randn(
                    LOCAL_BATCH, chunk, HIDDEN,
                    dtype=torch.bfloat16, device=device, generator=generator,
                )
                * args.input_scale
            ).contiguous()
            gate_hidden = rms_norm(
                gate_hidden,
                gate_layer.raw_block.ffn_norm,
                eps=gate_layer.norm_eps,
            )
            before = gate_layer.moe.collective_overlap_blocks
            gate_layer.moe.enable_collective_overlap(0)
            sequential = gate_layer.moe.forward_tensor(gate_hidden, slot=0)
            per_blocks = {}
            for blocks in (2, 4, 8):
                if (LOCAL_BATCH * chunk) % blocks:
                    continue
                gate_layer.moe.enable_collective_overlap(blocks)
                overlapped = gate_layer.moe.forward_tensor(gate_hidden, slot=0)
                torch.cuda.synchronize(device)
                difference = (overlapped.float() - sequential.float())
                per_blocks[str(blocks)] = {
                    "bitwise_equal": bool(torch.equal(sequential, overlapped)),
                    "rel_fro": float(
                        difference.norm()
                        / sequential.float().norm().clamp_min(1e-30)
                    ),
                    "max_abs": float(difference.abs().max()),
                    "mismatched_elements": int(
                        (sequential != overlapped).sum()
                    ),
                    "finite": bool(torch.isfinite(overlapped).all()),
                }
                del overlapped
            gate_layer.moe.enable_collective_overlap(before)
            result["moe_overlap_gate"] = {
                "form": "same hidden through the sequential and the "
                "row-blocked pipelined MoE on the first stage layer",
                "layer_id": gate_layer.layer_id,
                "rows_local": LOCAL_BATCH * chunk,
                "sequential_rms": float(
                    sequential.float().square().mean().sqrt()
                ),
                "per_blocks": per_blocks,
            }
            del sequential, gate_hidden

        # ---------------- paired indexer gate (optional) -----------------
        if args.gate_indexer:
            lane = make_lane("paired_gate")
            residual = make_residual(20_000)
            lane.forward(residual, start_pos=0)
            torch.cuda.synchronize(device)
            result["index_gate_records"] = lane.collect_index_gate_records()
            del lane, residual

        # ---------------- W4A8 layer-level A/B gate (optional) -----------
        if args.gate_w4a8:
            gate_layer = materials[0]
            other_dtype = None if marlin_input_dtype is not None else FP8
            other_resident = load_resident_moe_layer(
                stage_root=stage_root,
                layer_id=gate_layer.layer_id,
                rank=rank,
                world_size=EXPECTED_TP_SIZE,
                hidden_size=int(config["hidden_size"]),
                intermediate_size=int(config["moe_intermediate_size"]),
                n_experts=int(config["n_routed_experts"]),
                device=device,
                progress_every=args.progress_every,
                checkpoint_id=envelope["checkpoint_id"],
                marlin_input_dtype=other_dtype,
            )
            other_moe = TP4MoE(
                config=TP4MoEConfig(
                    hidden_size=int(config["hidden_size"]),
                    intermediate_size=int(config["moe_intermediate_size"]),
                    experts=int(config["n_routed_experts"]),
                    topk=int(config["num_experts_per_tok"]),
                    route_scale=float(config["routed_scaling_factor"]),
                    clamp_limit=float(config["swiglu_limit"]),
                    world_size=EXPECTED_TP_SIZE,
                ),
                resident=other_resident,
                gate=gate_layer.raw_block.gate,
                rank=rank,
                device=device,
                global_row_shapes=(global_rows,),
                group=tp_group,
                slots_per_shape=1,
                marlin_input_dtype=other_dtype,
                buffer_donor=gate_layer.moe,
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + 777 + rank)
            gate_hidden = (
                torch.randn(
                    LOCAL_BATCH,
                    chunk,
                    HIDDEN,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                * args.input_scale
            ).contiguous()
            gate_hidden = rms_norm(
                gate_hidden,
                gate_layer.raw_block.ffn_norm,
                eps=gate_layer.norm_eps,
            )
            out_this = gate_layer.moe.forward_tensor(gate_hidden, slot=0)
            out_other = other_moe.forward_tensor(gate_hidden, slot=0)
            this_is_a16 = marlin_input_dtype is None
            out_a16 = out_this if this_is_a16 else out_other
            out_a8 = out_other if this_is_a16 else out_this
            diff = (out_a8.float() - out_a16.float())
            rel_fro = float(
                diff.norm() / out_a16.float().norm().clamp_min(1e-30)
            )
            result["w4a8_gate"] = {
                "layer_id": gate_layer.layer_id,
                "rows_local": LOCAL_BATCH * chunk,
                "rel_fro_w4a8_vs_w4a16": rel_fro,
                "max_abs_diff": float(diff.abs().max().item()),
                "w4a16_rms": float(
                    out_a16.float().square().mean().sqrt().item()
                ),
                "finite": bool(
                    torch.isfinite(out_a8).all().item()
                    and torch.isfinite(out_a16).all().item()
                ),
                "a3f_reference": (
                    "A3F numeric gate: marlin W4A8 rel_fro vs fp32 oracle "
                    "4.6e-2/8.5e-2 (== reference tilelang magnitude); "
                    "W4A16 4.2e-3/3.5e-3"
                ),
            }
            del other_moe, other_resident, gate_hidden, out_this, out_other

        # Whole-process peaks: the timed loop reset the counters, so restore the
        # pre-reset maxima to keep this field's frozen meaning.
        result["max_memory_allocated_bytes"] = max(
            peak_allocated_before_timed, int(torch.cuda.max_memory_allocated(device))
        )
        result["max_memory_reserved_bytes"] = max(
            peak_reserved_before_timed, int(torch.cuda.max_memory_reserved(device))
        )
        # 26th vertical: end-of-run occupancy (the C3F "收尾 free" counterpart).
        end_free, end_total = torch.cuda.mem_get_info(device)
        result["memory_allocated_bytes_end"] = int(
            torch.cuda.memory_allocated(device)
        )
        result["memory_reserved_bytes_end"] = int(torch.cuda.memory_reserved(device))
        result["driver_free_bytes_end"] = int(end_free)
        result["driver_total_bytes"] = int(end_total)
        result["ok"] = True
    except Exception:
        result["ok"] = False
        result["errors"].append(traceback.format_exc())

    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, result)
    if rank == 0:
        summary = dict(gathered[0])
        summary["per_rank_max_memory_allocated_bytes"] = [
            entry.get("max_memory_allocated_bytes") for entry in gathered
        ]
        summary["per_rank_ok"] = [entry.get("ok") for entry in gathered]
        summary["per_rank_errors"] = [entry.get("errors") for entry in gathered]
        summary["per_rank_prefill_peak_allocated_bytes"] = [
            entry.get("prefill_peak_allocated_bytes") for entry in gathered
        ]
        summary["per_rank_driver_free_bytes_end"] = [
            entry.get("driver_free_bytes_end") for entry in gathered
        ]
        tag = f"chunk{chunk}-{args.moe_mode}-{args.indexer}"
        if prefill_chunk:
            tag = f"{tag}-seg{prefill_chunk}"
        write_json(out_dir / f"c2f-{tag}.json", summary)
        print(json.dumps(
            {
                "chunk": chunk,
                "prefill_chunk": prefill_chunk,
                "prefill_forwards": len(prefill_plan),
                "segment_lengths": list(segment_lengths),
                "moe_mode": args.moe_mode,
                "indexer": args.indexer,
                "wall_p50_s": summary.get("stage_pass_wall_p50_s"),
                "input_tok_s_per_stage_dp4": summary.get(
                    "input_tok_s_per_stage_dp4"
                ),
                "ok": summary.get("per_rank_ok"),
            },
            indent=2,
        ))
    dist.barrier()
    dist.destroy_process_group()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
