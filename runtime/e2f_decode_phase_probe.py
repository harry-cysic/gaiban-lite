#!/usr/bin/env python3
"""E2F: phase-resolved profile of one TP4 decode super-stage at B=1.

Twenty-eighth vertical, first step of the M4 (latency mode) line.  E1F's
frozen bl=1 closed loop measures 36.3 ms/token on 16 cards and decomposes it,
per rank JSON, into 4 x 8.3 ms stage replay + 2.62 ms head + ~0.1 ms
embed/handoff -- i.e. **93% of the step is inside the four graph replays and
has no sub-attribution at all**.  The single-user bandwidth floor is 1.48
ms/stage (TARGET Section 6.1), so the replay carries a ~5.6x unexplained
factor.  This probe splits one replay into phases before anything is tuned.

Why a single-node 4-GPU probe: one PP stage is TP4-local -- its only
collectives are the MoE all_gather/reduce_scatter inside the TP group, and
the PP handoff is outside the graph.  Four GPUs of one machine therefore
reproduce one E1F stage exactly, at 1/4 the load time and with no IB
dependency.  **Platform validity is not assumed: round A measures the
uninstrumented replay and it must land on E1F's frozen per-family p50, or the
phase table below it means nothing.**

Timing: ``dsv4_direct.phase_timer.GraphPhaseRecorder``.  Phase marks are
issued once, during capture, so they become external CUDA event-record nodes
inside the graph and every replay re-records them; spans are read back after
the replay's synchronize.  Default (non-external) events capture fine but
make ``elapsed_time`` fail with ``cudaErrorInvalidValue`` -- measured, not
assumed.  Two witnesses are reported with every phase table:

  * ``instrumentation_overhead`` -- instrumented replay p50 vs round A's
    uninstrumented p50, same process, same weights, same cursor lineage;
  * ``phase_coverage`` -- sum of spans over the instrumented replay wall.

Rounds A and B are separated by a full ``teardown_stateful_graphs`` +
re-capture, since the marks are baked into the graph body.

Run (titan065, 4 GPUs, from ``run_e2f_probe.sh``):
  torchrun --standalone --nproc_per_node=4 e2f_decode_phase_probe.py \
      --stage-root ~/Workspace/DeepSeek-V4-Flash --layers 0-10 \
      --local-batch 1 --out-dir out-e2f-stage0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import traceback
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.phase_timer import GraphPhaseRecorder
from dsv4_direct.physical_stage import (
    EXPECTED_TP_SIZE,
    build_physical_stage,
    validate_live_tp_group,
)
from dsv4_direct.stateful_decode import (
    DecodeGraphFamily,
    StatefulDecodeCursor,
    build_decode_schedule,
    schedule_family_counts,
)
from dsv4_direct.stateful_graph import (
    capture_stateful_graph,
    replay_stateful_graph,
    teardown_stateful_graphs,
)

from dsv4_direct.superstage import TP4DecodeStage

from e1f_full_decode_bench import (
    EAGER_MOE_SLOT,
    full_state_sha256,
    seed_state,
    GRAPH_MOE_SLOTS,
    GRAPH_MOE_SLOT_TUPLE,
    HC_MULT,
    HIDDEN,
    StageLane,
    build_seed_payload,
    clone_state,
    copy_stage_states,
    deterministic_tensor,
    forward_eager_prevalidated,
    summarize_ms,
    synchronized_local_step,
    write_json,
)


EXPECTED_VOCAB = 129280

# Variant-lane graph MoE slots for --mode ab.  Slot 0 is the eager slot and
# 1-3 belong to the base lane (GRAPH_MOE_SLOT_TUPLE); the MoE objects are
# shared across lanes, so the variant lane needs a disjoint range.
VARIANT_MOE_SLOT_TUPLE = (5, 6, 7)
VARIANT_MOE_SLOTS: dict[Any, int] = {}  # filled in main(), keyed by family

# Marks emitted by the super-stage chain itself (one set per layer) plus the
# four stage-level ones.  Everything else -- attention internals in
# ratio4_attention/attention/window_attention, MoE internals in moe_runtime --
# is fine-grained.  Each mark is an event-record node inside the graph and
# costs real device time, so the level is a measured trade-off, not a taste:
# see ``mark_cost_us`` in the results.
COARSE_MARKS = frozenset(
    {
        "graph_start",
        "guard_done",
        "output_copy_done",
        "graph_done",
        "block_start",
        "attention_done",
        "ffn_prepare_done",
        "block_done",
    }
)


def device_telemetry(index: int) -> dict[str, Any]:
    """SM clock / temperature / power for one GPU (thermal-drift witness).

    TARGET Section 9.1: serial A/B on 4090 is untrustworthy without knowing
    the clock state, and this probe is compared against a number measured with
    eight GPUs of the machine busy rather than four.
    """

    query = "clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit"
    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    fields = [part.strip() for part in raw.split(",")]
    keys = query.split(",")
    out: dict[str, Any] = {}
    for key, value in zip(keys, fields, strict=False):
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def parse_layers(text: str) -> tuple[int, ...]:
    if "-" in text:
        first, last = text.split("-", 1)
        return tuple(range(int(first), int(last) + 1))
    return tuple(int(part) for part in text.split(",") if part)


def phase_table(
    recorder: GraphPhaseRecorder, *, replay_walls_ms: list[float]
) -> dict[str, Any]:
    """Aggregate collected passes into per-phase and per-layer tables."""

    by_phase: dict[str, list[float]] = defaultdict(list)
    by_layer: dict[str, list[float]] = defaultdict(list)
    by_layer_phase: dict[str, list[float]] = defaultdict(list)
    pass_totals: list[float] = []
    for spans in recorder.passes:
        phase_sum: dict[str, float] = defaultdict(float)
        layer_sum: dict[str, float] = defaultdict(float)
        layer_phase_sum: dict[str, float] = defaultdict(float)
        total = 0.0
        for name, value in spans:
            total += value
            layer, _, phase = name.partition("|")
            phase_sum[phase or layer] += value
            layer_sum[layer] += value
            layer_phase_sum[name] += value
        pass_totals.append(total)
        for key, value in phase_sum.items():
            by_phase[key].append(value)
        for key, value in layer_sum.items():
            by_layer[key].append(value)
        for key, value in layer_phase_sum.items():
            by_layer_phase[key].append(value)

    span_total_p50 = statistics.median(pass_totals) if pass_totals else 0.0
    wall_p50 = statistics.median(replay_walls_ms) if replay_walls_ms else 0.0

    def summarize(source: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        rows: dict[str, dict[str, float]] = {}
        for key, values in source.items():
            p50 = statistics.median(values)
            rows[key] = {
                "p50_ms": p50,
                "mean_ms": statistics.fmean(values),
                "share_of_spans": (p50 / span_total_p50) if span_total_p50 else 0.0,
                "calls_per_replay": len(
                    [1 for name, _ in recorder.passes[0] if name.split("|")[-1] == key]
                )
                if source is by_phase
                else 0,
                "samples": len(values),
            }
        return dict(sorted(rows.items(), key=lambda item: -item[1]["p50_ms"]))

    return {
        "spans_per_replay": len(recorder.passes[0]) if recorder.passes else 0,
        "span_total_p50_ms": span_total_p50,
        "instrumented_replay_p50_ms": wall_p50,
        "phase_coverage": (span_total_p50 / wall_p50) if wall_p50 else 0.0,
        "by_phase": summarize(by_phase),
        "by_layer": summarize(by_layer),
        "by_layer_phase": summarize(by_layer_phase),
    }



def apply_ab_variant(lane: Any, variant: str) -> list[int]:
    """Mutate lane B in place; return the layer ids the treatment reached."""

    applied: list[int] = []
    if variant == "none":
        # Control arm: lane B is left identical to lane A.  It exists because
        # the two lanes are not perfectly symmetric -- lane B is built second,
        # so its tensors sit later in the allocator, and it runs on a different
        # MoE slot range.  If that asymmetry alone moved the number, every
        # treatment measured this way would be wrong by the same amount.
        return applied
    for layer_id, block in zip(lane.stage.layer_ids, lane.stage.blocks, strict=True):
        attention = getattr(block, "attention", None)
        if variant == "qat_fused":
            # only ratio-4 layers carry the indexer chain
            if getattr(attention, "indexer_qat_mode", None) is not None:
                attention.indexer_qat_mode = "fused"
                applied.append(int(layer_id))
        elif variant == "kv_fp8_fused":
            # every layer type runs the KV-latent FP8 QAT chain
            if getattr(attention, "kv_qat_mode", None) is not None:
                attention.kv_qat_mode = "fused"
                applied.append(int(layer_id))
        else:
            raise ValueError(f"unknown A/B variant {variant!r}")
    return applied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", type=str, default="0-10")
    parser.add_argument("--stage-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--local-batch", type=int, default=1)
    parser.add_argument("--start-position", type=int, default=2048)
    parser.add_argument("--settle-steps", type=int, default=132)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--hc-backend", type=str, default="fused", choices=("fused", "eager", "default")
    )
    parser.add_argument("--kv-dtype", type=str, default="bf16", choices=("bf16", "fp8"))
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help=(
            "pin the KV geometry instead of deriving it from the step budget; "
            "use 3328 to match the frozen E1F bl=1 run exactly"
        ),
    )
    parser.add_argument(
        "--mark-level",
        type=str,
        default="fine",
        choices=("coarse", "fine"),
        help=(
            "coarse: stage + per-block marks only (~48 nodes); fine: every "
            "attention/MoE sub-phase (~244 nodes, ~15% overhead)"
        ),
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="wrap the timed segments in cudaProfilerStart/Stop for nsys",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("plain", "both", "ab"),
        help=(
            "plain: round A only (no marks); both: A then instrumented B; "
            "ab: two-lane paired alternating A/B of a variant (--ab-variant)"
        ),
    )
    parser.add_argument(
        "--ab-variant",
        type=str,
        default="qat_fused",
        choices=("qat_fused", "kv_fp8_fused", "none"),
        help=(
            "the treatment applied to lane B in --mode ab; 'none' is the "
            "control arm -- two identical lanes, which must read equal"
        ),
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    VARIANT_MOE_SLOTS.update(
        dict(zip(DecodeGraphFamily, VARIANT_MOE_SLOT_TUPLE, strict=True))
    )
    layer_ids = parse_layers(args.layers)
    local_batch = int(args.local_batch)
    start_position = int(args.start_position)
    settle_steps = int(args.settle_steps)
    rounds = int(args.rounds)
    steps_per_round = int(args.steps)
    measured_segments = 1 if args.mode == "plain" else 2
    # Each segment is preceded by a capture walk that consumes schedule steps
    # until all three families have occurred; the ratio-128 boundary is the
    # rare one, so budget one settle window per capture walk plus slack.
    capture_budget = measured_segments * (settle_steps + 8)
    total_steps = capture_budget + measured_segments * rounds * steps_per_round
    stop_position = start_position + total_steps
    max_seq_len = int(args.max_seq_len) or ((stop_position + 127) // 128 + 1) * 128
    if max_seq_len <= stop_position:
        raise SystemExit(
            f"max_seq_len {max_seq_len} must exceed stop_position {stop_position}"
        )
    if start_position < 2047 or start_position % 128:
        raise SystemExit("start position must be 128-aligned and >= 2047")
    if settle_steps < 132:
        raise SystemExit("settle segment must cover a ratio-128 boundary (>= 132)")

    schedule = build_decode_schedule(start_position, total_steps)
    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E2F-decode-latency-profile",
        "measurement_class": "stage_local_graph_replay_profile",
        "caliber": {
            "topology": (
                f"single node, TP4 only, one super-stage over layers "
                f"{layer_ids[0]}-{layer_ids[-1]}; no PP handoff, no embed/head; "
                "the graph replay is the whole measured object"
            ),
            "b_semantics": (
                "full replication (E1F replicated caliber): identical "
                f"{local_batch} sequences on all 4 TP ranks; MoE gathers "
                f"{local_batch * EXPECTED_TP_SIZE} rows"
            ),
            "kv": (
                f"seeded decode residency at position {start_position} "
                "(deterministic-seeded-KV-not-real-prefix, E1a27); "
                f"max_seq_len={max_seq_len}; kv_dtype={args.kv_dtype}"
            ),
            "input": (
                "the stage input residual buffer is filled once with a "
                "deterministic bf16 tensor and held constant; no transport runs "
                "inside the timed window"
            ),
            "timing": (
                "round A: host wall around replay + torch.cuda.synchronize. "
                "round B: same wall, plus in-graph external CUDA event nodes "
                "read back after the synchronize (outside the timed window)"
            ),
            "hc_backend": args.hc_backend,
        },
        "rank": rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "layer_ids": list(layer_ids),
        "local_batch": local_batch,
        "start_position": start_position,
        "stop_position": stop_position,
        "max_seq_len": max_seq_len,
        "settle_steps": settle_steps,
        "rounds": rounds,
        "steps_per_round": steps_per_round,
        "mode": args.mode,
        "mark_level": args.mark_level,
        "nccl_env": {
            key: os.environ.get(key)
            for key in ("NCCL_P2P_LEVEL", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME")
        },
        "checkpoint_id": None,
        "memory": {},
        "round_a": None,
        "round_b": None,
        "phases": None,
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }
    started = time.perf_counter()

    def memory_snapshot(label: str) -> None:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory"][label] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    try:
        if world != EXPECTED_TP_SIZE:
            raise ValueError(f"E2F requires world={EXPECTED_TP_SIZE}, got {world}")
        tp_global_ranks = tuple(range(EXPECTED_TP_SIZE))
        tp_group = dist.new_group(ranks=list(tp_global_ranks))
        warm = torch.ones(1, device=device)
        dist.all_reduce(warm, group=tp_group)
        torch.cuda.synchronize(device)
        result["tp_group_binding"] = validate_live_tp_group(
            tp_group,
            expected_local_rank=rank,
            expected_global_ranks=tp_global_ranks,
        )

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root, list(layer_ids), EXPECTED_TP_SIZE
                )
                if not checkpoint["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {checkpoint['errors'][:4]}"
                    )
                envelope_holder[0] = {
                    "ok": True,
                    "config": config_payload,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                }
            except Exception:
                envelope_holder[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        if not envelope["ok"]:
            raise ValueError(f"rank-0 preflight failed:\n{envelope['error']}")
        result["checkpoint_id"] = envelope["checkpoint_id"]
        model_config = envelope["config"]

        # ------------------------------------------------------------------
        phase_started = time.perf_counter()
        global_rows = local_batch * EXPECTED_TP_SIZE
        stage_material = synchronized_local_step(
            "load materials",
            lambda: build_physical_stage(
                stage_id=args.stage_id,
                layer_ids=layer_ids,
                model_config=model_config,
                stage_root=stage_root,
                tp_rank=rank,
                tp_group=tp_group,
                tp_global_ranks=tp_global_ranks,
                device=device,
                checkpoint_id=result["checkpoint_id"],
                max_seq_len=max_seq_len,
                global_row_shapes=(global_rows,),
                # mode ab runs two graph lanes over shared MoE objects, so the
                # slot budget has to cover both families sets plus the eager slot
                slots_per_shape=(8 if args.mode == "ab" else 4),
                kv_dtype=args.kv_dtype,
                indexer_kv_dtype=args.kv_dtype,
                progress=(
                    (lambda message: print(f"[E2F] {message}", flush=True))
                    if rank == 0
                    else None
                ),
            ),
            device=device,
            world=world,
        )
        result["diagnostic_seconds"]["load"] = time.perf_counter() - phase_started
        memory_snapshot("after_load")

        backend = resolve_hc_boundary_backend(
            None if args.hc_backend == "default" else args.hc_backend
        )
        phase_started = time.perf_counter()
        seed_payloads = {
            material.layer_id: build_seed_payload(
                material,
                seed=args.seed,
                local_batch=local_batch,
                start_position=start_position,
                device=device,
                dp_tp_rank=None,
            )
            for material in stage_material.materials
        }

        def new_lane(label: str, *, graph_moe_slots: tuple[int, int, int]) -> Any:
            """Build one lane over the shared materials.

            ``StageLane`` hardwires the graph MoE slot tuple, but the MoE
            objects are **shared across lanes** (physical_stage shares weight
            and MoE material and gives each lane only its own state/attention).
            Two graph lanes therefore cannot both capture into slots 1-3 --
            the second capture fails ``_validate_family_slot_clean``.  So the
            variant lane gets its own slot range and the stage is built with
            enough slots for both.
            """

            blocks = []
            for material in stage_material.materials:
                state = material.new_state(num_local_sequences=local_batch)
                seed_state(
                    material,
                    state,
                    seed_payloads[material.layer_id],
                    start_position=start_position,
                )
                blocks.append(material.new_block(state))
            stage = TP4DecodeStage(blocks, hc_boundary_backend=backend)
            cursor = StatefulDecodeCursor(
                start_position=start_position, device=device
            )
            lane_obj = SimpleNamespace(label=label, stage=stage, cursor=cursor)
            lane_obj.plan = stage.prepare_stateful_decode_plan(
                cursor,
                start_position=start_position,
                stop_position=stop_position,
                graph_moe_slots=graph_moe_slots,
            )
            lane_obj.state_digests = lambda: {
                str(layer_id): full_state_sha256(state)
                for layer_id, state in zip(
                    stage.layer_ids, stage.states, strict=True
                )
            }
            lane_obj.terminal = lambda expected: StageLane.terminal(lane_obj, expected)
            return lane_obj

        lane = synchronized_local_step(
            "build lane",
            lambda: new_lane("graph", graph_moe_slots=GRAPH_MOE_SLOT_TUPLE),
            device=device,
            world=world,
        )
        plan = lane.plan
        result["diagnostic_seconds"]["build"] = time.perf_counter() - phase_started
        memory_snapshot("after_build")

        # constant stage input: no transport inside the timed window.
        plan.input_residual_buffer.copy_(
            deterministic_tensor(
                seed=args.seed * 1_000_003 + 17,
                shape=(local_batch, 1, HC_MULT, HIDDEN),
                device=device,
            )
        )
        plan.input_ids_buffer.copy_(
            torch.full(
                (local_batch, 1),
                (args.seed * 2654435761) % EXPECTED_VOCAB,
                dtype=torch.int64,
                device=device,
            )
        )

        # ------------------------------------------------------------------
        # warmup (E0hf pattern), then restore to the seeded start.
        warm_schedule = schedule[:settle_steps]
        snapshots = [clone_state(state) for state in lane.stage.states]
        capture_stream = torch.cuda.Stream(device=device)
        graph_pools = {
            family: torch.cuda.graph_pool_handle() for family in DecodeGraphFamily
        }

        def restore_cycle() -> None:
            copy_stage_states(lane.stage.states, snapshots)
            lane.cursor.reset(start_position)
            plan.expected_position.fill_(start_position)
            plan.stop_position_tensor.fill_(plan.stop_position)

        def run_warm_cycle(*, graph_slots: bool) -> None:
            for step in warm_schedule:
                forward_eager_prevalidated(
                    lane.stage,
                    plan,
                    graph_family=step.family,
                    moe_slot=(
                        GRAPH_MOE_SLOTS[step.family] if graph_slots else EAGER_MOE_SLOT
                    ),
                )
                lane.cursor.advance_host(step.family)
            torch.cuda.synchronize(device)

        def warmup_all() -> None:
            run_warm_cycle(graph_slots=False)
            restore_cycle()
            with torch.cuda.stream(capture_stream):
                run_warm_cycle(graph_slots=True)
            torch.cuda.synchronize(device)
            restore_cycle()
            for slot in GRAPH_MOE_SLOT_TUPLE:
                for moe in lane.stage.moes:
                    moe.reset_free_slot_completion_event(global_rows, slot)
            evidence = lane.terminal(start_position)
            if not evidence["accepted"]:
                raise RuntimeError(f"warmup restore drifted: {evidence}")

        phase_started = time.perf_counter()
        synchronized_local_step("warmups", warmup_all, device=device, world=world)
        result["diagnostic_seconds"]["warmup"] = time.perf_counter() - phase_started
        del snapshots
        torch.cuda.empty_cache()
        memory_snapshot("after_warmup")
        if rank == 0:
            print(
                f"[E2F] warmup done ({result['diagnostic_seconds']['warmup']:.0f}s, "
                f"free {result['memory']['after_warmup']['free_bytes'] / 2**30:.2f} "
                "GiB)",
                flush=True,
            )

        # ------------------------------------------------------------------
        cursor_index = 0

        def capture_families(
            recorder: GraphPhaseRecorder | None,
            *,
            marker_family: DecodeGraphFamily | None = None,
            pools: dict[DecodeGraphFamily, Any],
            target: Any = None,
            start_index: int | None = None,
        ) -> dict[DecodeGraphFamily, Any]:
            """Walk the schedule until every family is captured; replay each.

            ``target`` selects the lane.  Every lane walks the same schedule
            from the same start, so all lanes finish the capture walk at the
            same position -- which is what makes the A/B pairing legitimate.
            """

            nonlocal cursor_index
            target = lane if target is None else target
            local_index = cursor_index if start_index is None else start_index
            graphs: dict[DecodeGraphFamily, Any] = {}
            coarse_only = args.mark_level == "coarse"
            marker = None
            if recorder is not None:
                recorder.begin()

                def marker(layer_id: int | None, name: str) -> None:  # noqa: F811
                    if coarse_only and name not in COARSE_MARKS:
                        return
                    recorder.mark(
                        f"stage|{name}" if layer_id is None else f"L{layer_id}|{name}"
                    )

            while len(graphs) < len(DecodeGraphFamily):
                step = schedule[local_index]
                if step.family not in graphs:
                    marked = marker is not None and step.family == marker_family
                    graphs[step.family] = synchronized_local_step(
                        f"capture {step.family.value}",
                        lambda step=step, marked=marked: capture_stateful_graph(
                            target.stage,
                            target.plan,
                            graph_family=step.family,
                            capture_stream=capture_stream,
                            pool=pools[step.family],
                            # Marks live in one family only: they are baked into
                            # the captured body and one recorder cannot own three
                            # interleaved mark lists.  NORMAL is the family that
                            # carries 75% of E1F's steps.
                            stage_marker=(marker if marked else None),
                        ),
                        device=device,
                        world=world,
                        group=tp_group,
                    )
                    if marked:
                        recorder.seal()
                replay_stateful_graph(
                    graphs[step.family], target.plan, graph_family=step.family
                )
                torch.cuda.synchronize(device)
                target.cursor.advance_host(step.family)
                local_index += 1
            if target is lane:
                cursor_index = local_index
            return graphs

        def timed_segment(
            graphs: dict[DecodeGraphFamily, Any],
            *,
            recorder: GraphPhaseRecorder | None,
            instrumented_family: DecodeGraphFamily | None,
        ) -> dict[str, Any]:
            nonlocal cursor_index
            rounds_out: list[dict[str, Any]] = []
            walls_instrumented: list[float] = []
            if args.cuda_profiler_range:
                torch.cuda.synchronize(device)
                torch.cuda.cudart().cudaProfilerStart()
            for round_index in range(rounds):
                by_family: dict[str, list[float]] = defaultdict(list)
                for _ in range(steps_per_round):
                    step = schedule[cursor_index]
                    cursor_index += 1
                    collect = (
                        recorder is not None and step.family == instrumented_family
                    )
                    start = time.perf_counter()
                    replay_stateful_graph(
                        graphs[step.family], plan, graph_family=step.family
                    )
                    torch.cuda.synchronize(device)
                    wall = (time.perf_counter() - start) * 1e3
                    by_family[step.family.value].append(wall)
                    if collect:
                        recorder.collect()
                        walls_instrumented.append(wall)
                    lane.cursor.advance_host(step.family)
                rounds_out.append(
                    {
                        "round": round_index,
                        "telemetry": device_telemetry(local_rank),
                        "by_family": {
                            key: summarize_ms(values) for key, values in by_family.items()
                        },
                    }
                )
            if args.cuda_profiler_range:
                torch.cuda.synchronize(device)
                torch.cuda.cudart().cudaProfilerStop()
            merged: dict[str, list[float]] = defaultdict(list)
            for entry in rounds_out:
                for key, stats in entry["by_family"].items():
                    merged[key].append(stats["p50_ms"])
            return {
                "rounds": rounds_out,
                "family_p50_ms": {
                    key: statistics.median(values) for key, values in merged.items()
                },
                "round_spread_pct": {
                    key: (
                        100.0 * (max(values) - min(values)) / statistics.median(values)
                        if statistics.median(values)
                        else 0.0
                    )
                    for key, values in merged.items()
                },
                "instrumented_walls_ms": walls_instrumented,
            }

        # round A: uninstrumented ------------------------------------------
        if args.mode != "ab":
            graphs = capture_families(None, pools=graph_pools)
            memory_snapshot("after_capture_a")
            phase_started = time.perf_counter()
            result["round_a"] = timed_segment(
                graphs, recorder=None, instrumented_family=None
            )
            result["diagnostic_seconds"]["round_a"] = (
                time.perf_counter() - phase_started
            )
            if rank == 0:
                print(
                    f"[E2F] round A p50 {result['round_a']['family_p50_ms']}",
                    flush=True,
                )
            result["teardown_a"] = teardown_stateful_graphs(
                lane.stage, plan, graphs, pool_handles=dict(graph_pools)
            )
            for slot in GRAPH_MOE_SLOT_TUPLE:
                for moe in lane.stage.moes:
                    moe.reset_free_slot_completion_event(global_rows, slot)
            torch.cuda.synchronize(device)

        # round B: instrumented --------------------------------------------
        if args.mode == "both":
            recorder = GraphPhaseRecorder(device, capacity=512)
            first_family = DecodeGraphFamily.NORMAL
            pools_b = {
                family: torch.cuda.graph_pool_handle() for family in DecodeGraphFamily
            }
            graphs_b = capture_families(
                recorder, marker_family=first_family, pools=pools_b
            )
            memory_snapshot("after_capture_b")
            phase_started = time.perf_counter()
            result["round_b"] = timed_segment(
                graphs_b, recorder=recorder, instrumented_family=first_family
            )
            result["diagnostic_seconds"]["round_b"] = (
                time.perf_counter() - phase_started
            )
            result["instrumented_family"] = first_family.value
            table = phase_table(
                recorder, replay_walls_ms=result["round_b"]["instrumented_walls_ms"]
            )
            plain_p50 = result["round_a"]["family_p50_ms"].get(first_family.value)
            table["uninstrumented_p50_ms"] = plain_p50
            table["mark_level"] = args.mark_level
            table["mark_cost_us"] = (
                1e3
                * (table["instrumented_replay_p50_ms"] - plain_p50)
                / table["spans_per_replay"]
                if plain_p50 and table["spans_per_replay"]
                else None
            )
            table["instrumentation_overhead"] = (
                (table["instrumented_replay_p50_ms"] / plain_p50 - 1.0)
                if plain_p50
                else None
            )
            result["phases"] = table
            result["round_b"]["instrumented_walls_ms"] = summarize_ms(
                result["round_b"]["instrumented_walls_ms"]
            )
            teardown_b = teardown_stateful_graphs(
                lane.stage, plan, graphs_b, pool_handles=dict(pools_b)
            )
            result["teardown_b"] = teardown_b
            if rank == 0:
                print(
                    f"[E2F] coverage {table['phase_coverage']:.4f} overhead "
                    f"{table['instrumentation_overhead']:.4%} spans "
                    f"{table['spans_per_replay']}",
                    flush=True,
                )
        elif args.mode == "plain":
            result["round_a"]["instrumented_walls_ms"] = []

        def paired_alternating(
            lane_a: Any,
            graphs_a: dict[DecodeGraphFamily, Any],
            variant_lane: Any,
            graphs_v: dict[DecodeGraphFamily, Any],
        ) -> dict[str, Any]:
            """Replay both lanes back-to-back, alternating order every step.

            TARGET 9.1: a serial A/B on a 4090 can report 51/219/318 us for the
            same configuration across three rounds, which is the same order as
            the effects being chased here.  Two resident lanes replayed
            back-to-back with the order swapped each step cancels clock and
            thermal drift to first order.

            The bitwise comparison is free and is the in-layer numeric gate:
            a variant that is bitwise identical in a micro gate still has to
            prove it in the layer, on real weights and real state.
            """

            nonlocal cursor_index
            rounds_out: list[dict[str, Any]] = []
            bitwise_steps = 0
            total_steps = 0
            first_mismatch: int | None = None
            max_abs = 0.0
            for round_index in range(rounds):
                base_ms: list[float] = []
                variant_ms: list[float] = []
                for _ in range(steps_per_round):
                    step = schedule[cursor_index]
                    cursor_index += 1
                    order = (
                        (("base", lane_a, graphs_a), ("variant", variant_lane, graphs_v))
                        if total_steps % 2 == 0
                        else (
                            ("variant", variant_lane, graphs_v),
                            ("base", lane_a, graphs_a),
                        )
                    )
                    for label, target, graphs in order:
                        start = time.perf_counter()
                        replay_stateful_graph(
                            graphs[step.family], target.plan, graph_family=step.family
                        )
                        torch.cuda.synchronize(device)
                        wall = (time.perf_counter() - start) * 1e3
                        (base_ms if label == "base" else variant_ms).append(wall)
                    if torch.equal(lane_a.plan.output_buffer, variant_lane.plan.output_buffer):
                        bitwise_steps += 1
                    else:
                        if first_mismatch is None:
                            first_mismatch = step.position
                        max_abs = max(
                            max_abs,
                            float(
                                (
                                    lane_a.plan.output_buffer.float()
                                    - variant_lane.plan.output_buffer.float()
                                )
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                    total_steps += 1
                    lane_a.cursor.advance_host(step.family)
                    variant_lane.cursor.advance_host(step.family)
                rounds_out.append(
                    {
                        "round": round_index,
                        "base": summarize_ms(base_ms),
                        "variant": summarize_ms(variant_ms),
                        "telemetry": device_telemetry(local_rank),
                    }
                )
            base_p50 = statistics.median(r["base"]["p50_ms"] for r in rounds_out)
            variant_p50 = statistics.median(r["variant"]["p50_ms"] for r in rounds_out)

            def spread(key: str) -> float:
                values = [r[key]["p50_ms"] for r in rounds_out]
                middle = statistics.median(values)
                return 100.0 * (max(values) - min(values)) / middle if middle else 0.0

            return {
                "caliber": (
                    "two resident lanes, back-to-back replay, order swapped "
                    "every step; host wall around replay + synchronize"
                ),
                "rounds": rounds_out,
                "steps": total_steps,
                "base_p50_ms": base_p50,
                "variant_p50_ms": variant_p50,
                "delta_ms": variant_p50 - base_p50,
                "delta_pct": 100.0 * (variant_p50 - base_p50) / base_p50,
                "round_spread_pct": {
                    "base": spread("base"),
                    "variant": spread("variant"),
                },
                "bitwise_steps": bitwise_steps,
                "first_mismatch_position": first_mismatch,
                "max_abs_diff": max_abs,
            }

        # ------------------------------------------------------------------
        # mode ab: two-lane paired alternating A/B (TARGET 9.1)
        if args.mode == "ab":
            lane_b = synchronized_local_step(
                "build lane b",
                lambda: new_lane("variant", graph_moe_slots=VARIANT_MOE_SLOT_TUPLE),
                device=device,
                world=world,
            )
            if lane.state_digests() != lane_b.state_digests():
                raise RuntimeError("A/B lanes were not seeded identically")
            applied = apply_ab_variant(lane_b, args.ab_variant)
            result["ab_variant"] = {"name": args.ab_variant, "applied_to": applied}
            if not applied and args.ab_variant != "none":
                raise RuntimeError(f"variant {args.ab_variant} applied to no layer")

            snapshots_b = [clone_state(state) for state in lane_b.stage.states]

            def warmup_lane_b() -> None:
                for step in warm_schedule:
                    forward_eager_prevalidated(
                        lane_b.stage, lane_b.plan, graph_family=step.family
                    )
                    lane_b.cursor.advance_host(step.family)
                torch.cuda.synchronize(device)
                copy_stage_states(lane_b.stage.states, snapshots_b)
                lane_b.cursor.reset(start_position)
                lane_b.plan.expected_position.fill_(start_position)
                lane_b.plan.stop_position_tensor.fill_(lane_b.plan.stop_position)
                with torch.cuda.stream(capture_stream):
                    for step in warm_schedule:
                        forward_eager_prevalidated(
                            lane_b.stage,
                            lane_b.plan,
                            graph_family=step.family,
                            # the variant lane's own slots: MoE objects are
                            # shared, so warming on the base slots would leave
                            # them dirty and fail lane A's capture
                            moe_slot=VARIANT_MOE_SLOTS[step.family],
                        )
                        lane_b.cursor.advance_host(step.family)
                torch.cuda.synchronize(device)
                copy_stage_states(lane_b.stage.states, snapshots_b)
                lane_b.cursor.reset(start_position)
                lane_b.plan.expected_position.fill_(start_position)
                lane_b.plan.stop_position_tensor.fill_(lane_b.plan.stop_position)
                # reset every slot either lane may have touched
                for slot in GRAPH_MOE_SLOT_TUPLE + VARIANT_MOE_SLOT_TUPLE:
                    for moe in lane_b.stage.moes:
                        moe.reset_free_slot_completion_event(global_rows, slot)

            synchronized_local_step(
                "warmup lane b", warmup_lane_b, device=device, world=world
            )
            del snapshots_b
            lane_b.plan.input_residual_buffer.copy_(plan.input_residual_buffer)
            lane_b.plan.input_ids_buffer.copy_(plan.input_ids_buffer)

            # Both lanes are still at start_position (mode ab skips round A),
            # and both walk the same schedule from index 0, so both capture
            # walks end at the same position -- that identity is what makes the
            # step-by-step pairing legitimate, and it is asserted below.
            pools_b = {
                family: torch.cuda.graph_pool_handle()
                for family in DecodeGraphFamily
            }
            graphs_a = capture_families(None, pools=graph_pools, start_index=0)
            graphs_b = capture_families(
                None, pools=pools_b, target=lane_b, start_index=0
            )
            if lane.cursor.host_position != lane_b.cursor.host_position:
                raise RuntimeError(
                    "A/B capture walks ended at different positions: "
                    f"{lane.cursor.host_position} vs {lane_b.cursor.host_position}"
                )
            memory_snapshot("after_capture_ab")

            phase_started = time.perf_counter()
            result["ab"] = paired_alternating(lane, graphs_a, lane_b, graphs_b)
            result["diagnostic_seconds"]["ab"] = time.perf_counter() - phase_started
            result["teardown_ab_a"] = teardown_stateful_graphs(
                lane.stage, plan, graphs_a, pool_handles=dict(graph_pools)
            )
            result["teardown_ab_b"] = teardown_stateful_graphs(
                lane_b.stage, lane_b.plan, graphs_b, pool_handles=dict(pools_b)
            )
            if rank == 0:
                ab = result["ab"]
                print(
                    f"[E2F] A/B {args.ab_variant}: base {ab['base_p50_ms']:.4f} "
                    f"variant {ab['variant_p50_ms']:.4f} delta "
                    f"{ab['delta_pct']:+.2f}%  bitwise {ab['bitwise_steps']}/"
                    f"{ab['steps']}",
                    flush=True,
                )

        memory_snapshot("at_end")
        result["terminal"] = lane.terminal(schedule[cursor_index].position)
        if args.mode == "ab":
            result["accepted"] = bool(
                result["ab"]["bitwise_steps"] == result["ab"]["steps"]
                and result["teardown_ab_a"].get("accepted", False)
                and result["teardown_ab_b"].get("accepted", False)
            )
        else:
            result["accepted"] = bool(
                result["terminal"]["accepted"]
                and result["teardown_a"].get("accepted", False)
                and (
                    args.mode == "plain"
                    or result.get("teardown_b", {}).get("accepted")
                )
            )
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False

    result["diagnostic_seconds"]["process"] = time.perf_counter() - started
    write_json(out_dir / f"rank{rank}.json", result)
    if result["errors"]:
        print(f"[E2F] rank {rank} FAILED\n{result['errors'][0]}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
