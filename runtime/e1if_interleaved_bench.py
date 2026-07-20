#!/usr/bin/env python3
"""E1IF: >=4-microbatch interleaved PP4 decode on the DP-caliber full config.

Thirteenth vertical (throughput realization).  Extends
``e1f_full_decode_bench.py`` (imported for all shared mechanisms) from the
serial one-batch-in-flight closed loop to a rotating microbatch pipeline:

- **Microbatch lanes.**  ``mb_count`` independent sequence groups, each a full
  DP-caliber lane: per-lane KV states, cursor, stateful plan, and CUDA-graph
  set (3 families).  Lane ``m`` owns MoE graph slots ``(1+3m, 2+3m, 3+3m)``
  (slot 0 stays the shared eager slot) because ``capture_stateful_graph``
  requires its family slot clean at capture and binds a completion event to
  it -- graph slots are lane-owned, never shared.  Graph pools are per lane
  per family (12 pools at mb=4); MoE runtimes/weights are shared across lanes
  (the E0sf lane pattern over ``PhysicalLayerMaterial``).
- **Rotation schedule.**  Iteration ``j`` of every stage serves lane
  ``j % L`` at cycle ``k = j // L``; all lanes advance the same position
  ``start + k`` in cycle ``k``.  Stage ``s`` runs iteration ``j`` at global
  tick ``j + s`` (enforced naturally by the blocking serial handoffs, E0qf
  form -- one payload per pair group per iteration, both sides in identical
  lane order).  With ``L == 4 == PP depth`` the steady state has one
  microbatch resident per stage per tick.
- **Token loop with 4-step pipeline latency.**  Stage 3 sends lane ``m``'s
  argmax token at its iteration ``4k+m``; stage 0 receives it at the start of
  its iteration ``4(k+1)+m`` (the embed for lane ``m``'s next position).
  Decode semantics per lane are exactly the serial closed loop -- each lane
  only ever consumes its own history.  Every phase ends with ``L`` drain
  receives on stage 0 so no token send is left unmatched.
- **Serial handoff, no compute/transport overlap** (out of scope for this
  vertical): every component wall is host time around
  ``torch.cuda.synchronize`` (E0qf serial decomposition caliber), so
  per-iteration transport (~1-4 ms) is paid serially; the expected loss vs a
  perfectly overlapped pipeline is the transport share of the iteration.

**Cross-talk gate** (``--check-mode gate``, mb=2 recommended): after warmup,
each lane first runs **solo** through the pipeline for ``gate_cycles`` steps
(1 microbatch in flight; lazy graph capture; per-step bitwise graph-vs-eager
against a twin lane, E1a27/E0hf caliber), recording its stage-3 token digest
trajectory and final KV digests; the lane is then restored to the seeded
start state.  Then all lanes run **interleaved** over the same positions
(replay only) and every lane's token trajectory, final KV digests, and
cursor terminal must be bitwise identical to its solo run -- interleaving
must not perturb any lane.

**Timed mode** (``--check-mode off``): interleaved settle segment (lazy
capture, >=132 cycles for all three families) then ``rounds x steps`` timed
interleaved cycles.  Aggregate throughput = (L x B_mb_global x cycles) /
round wall; per-iteration p50 gives 128 / iter_wall.  B semantics are E1F
``dp`` (measured, no conversion): lane ``m`` serves its own distinct
``B_mb_global = 4 x local_batch`` sequences, KV ``local_batch`` rows per GPU
per lane.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.dp_caliber import dp_row_slice
from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.head_stage import (
    EmbedHeadMaterial,
    embed_hc_residual,
    head_logits,
    load_embed_head_material,
)
from dsv4_direct.physical_stage import (
    EXPECTED_TP_SIZE,
    PhysicalLayerMaterial,
    build_physical_stage,
    validate_live_tp_group,
)
from dsv4_direct.pipeline_transport import SerialPairHandoff, validate_handoff_endpoint
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
    EXPECTED_VOCAB,
    HC_MULT,
    HIDDEN,
    MODEL_LAYERS,
    STAGE_COUNT,
    STAGE_LAYERS,
    WORLD,
    DirectState,
    build_seed_payload,
    clone_state,
    copy_stage_states,
    create_e1f_topology,
    deterministic_tensor,
    forward_eager_prevalidated,
    full_state_sha256,
    pair_transfer,
    run_placement_check,
    seed_state,
    summarize_ms,
    synchronized_local_step,
    tensor_sha256,
    write_json,
)


CANONICAL_CAPTURE_ORDER = [
    "normal",
    "ratio4_boundary",
    "ratio4_ratio128_boundary",
]

# Reference points from the accepted DP serial run (results/dp/
# out-e1f-dp-bg512-ctx2048): stage replay p50 ms and the derived conversion
# ceiling this vertical is asked to realize (B_global / max stage replay).
DP_BG512_CTX2048_REPLAY_P50_MS = {0: 36.6, 1: 35.5, 2: 37.2, 3: 33.1}
DP_BG512_CTX2048_SERIAL_TOK_S = 3238.0
CONVERSION_CEILING_TOK_S = 512 / 0.0372  # ~13.8k, B_global=512 / max stage


def lane_moe_slots(lane_index: int) -> dict[DecodeGraphFamily, int]:
    base = 1 + 3 * lane_index
    return {
        DecodeGraphFamily.NORMAL: base,
        DecodeGraphFamily.RATIO4_BOUNDARY: base + 1,
        DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY: base + 2,
    }


def lane_seed(seed: int, lane_index: int) -> int:
    return (seed + 7_919 * (lane_index + 1)) & ((1 << 62) - 1)


class MicrobatchLane:
    """One microbatch lane: seeded KV blocks, super-stage, cursor, plan,
    lane-owned MoE graph slots, lane-owned graph/pool registries."""

    def __init__(
        self,
        *,
        label: str,
        lane_index: int,
        materials: Sequence[PhysicalLayerMaterial],
        payloads: Mapping[int, dict[str, Any]],
        backend: Any | None,
        local_batch: int,
        start_position: int,
        stop_position: int,
        device: torch.device,
        moe_slots: Mapping[DecodeGraphFamily, int],
    ) -> None:
        self.label = label
        self.lane_index = lane_index
        self.moe_slots = dict(moe_slots)
        self.moe_slot_tuple = tuple(
            self.moe_slots[family] for family in DecodeGraphFamily
        )
        blocks = []
        for material in materials:
            state = material.new_state(num_local_sequences=local_batch)
            seed_state(
                material,
                state,
                payloads[material.layer_id],
                start_position=start_position,
            )
            blocks.append(material.new_block(state))
        self.stage = TP4DecodeStage(blocks, hc_boundary_backend=backend)
        self.cursor = StatefulDecodeCursor(start_position=start_position, device=device)
        self.plan = self.stage.prepare_stateful_decode_plan(
            self.cursor,
            start_position=start_position,
            stop_position=stop_position,
            graph_moe_slots=self.moe_slot_tuple,
        )
        self.graphs: dict[DecodeGraphFamily, torch.cuda.CUDAGraph] = {}
        self.pools = {
            family: torch.cuda.graph_pool_handle() for family in DecodeGraphFamily
        }
        self.capture_order: list[str] = []

    def state_digests(self) -> dict[str, str]:
        return {
            str(layer_id): full_state_sha256(state)
            for layer_id, state in zip(
                self.stage.layer_ids, self.stage.states, strict=True
            )
        }

    def terminal(self, expected_position: int) -> dict[str, Any]:
        plan = self.plan
        state_positions = [
            [int(value) for value in tensor.detach().cpu().tolist()]
            for tensor in plan.state_position_tensors
        ]
        evidence = {
            "host_position": plan.cursor.host_position,
            "device_position": int(plan.cursor.device_position.item()),
            "expected_position": int(plan.expected_position.item()),
            "stop_position": int(plan.stop_position_tensor.item()),
            "dispatch_error": int(plan.cursor.dispatch_error.item()),
        }
        evidence["accepted"] = bool(
            evidence["host_position"] == expected_position
            and evidence["device_position"] == expected_position
            and evidence["expected_position"] == expected_position
            and evidence["stop_position"] == plan.stop_position
            and evidence["dispatch_error"] == 0
            and all(
                all(value == expected_position for value in positions)
                for positions in state_positions
            )
        )
        return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--local-batch", type=int, default=32, help="bl per GPU per microbatch")
    parser.add_argument("--mb-count", type=int, default=4)
    parser.add_argument("--start-position", type=int, default=2048)
    parser.add_argument("--settle-cycles", type=int, default=132)
    parser.add_argument("--gate-cycles", type=int, default=132)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300, help="cycles per timed round")
    parser.add_argument(
        "--check-mode", type=str, default="off", choices=("off", "gate")
    )
    parser.add_argument(
        "--hc-backend", type=str, default="fused", choices=("fused", "eager", "default")
    )
    parser.add_argument(
        "--kv-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp8", "fp8_rope_bf16"),
        help="latent KV storage dtype (A6F fp8_cast capacity option)",
    )
    parser.add_argument(
        "--indexer-kv-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp8"),
        help="ratio-4 indexer_kv storage dtype",
    )
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--config-tag", type=str, default="nogdr-dp-interleaved")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    local_batch = int(args.local_batch)
    mb_count = int(args.mb_count)
    gate = args.check_mode == "gate"
    mb_global = local_batch * EXPECTED_TP_SIZE  # distinct sequences per microbatch
    total_global = mb_global * mb_count
    start_position = int(args.start_position)
    gate_cycles = int(args.gate_cycles)
    settle_cycles = int(args.settle_cycles)
    rounds = 0 if gate else int(args.rounds)
    steps_per_round = int(args.steps)
    if gate:
        total_cycles = gate_cycles
    else:
        total_cycles = settle_cycles + rounds * steps_per_round
    stop_position = start_position + total_cycles
    max_seq_len = ((stop_position + 127) // 128 + 1) * 128
    if start_position < 2047 or start_position % 128:
        raise SystemExit("start position must be 128-aligned and >= 2047")
    if mb_count < 2 or mb_count > 4:
        raise SystemExit("mb_count must be in [2, 4]")
    if gate and gate_cycles < 132:
        raise SystemExit("gate segment must be >= 132 cycles (all three families)")
    if not gate and settle_cycles < 132:
        raise SystemExit("settle segment must be >= 132 cycles (all three families)")

    schedule = build_decode_schedule(start_position, total_cycles)
    family_counts = schedule_family_counts(schedule)
    warm_schedule = schedule[:132]

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    slots_per_shape = 1 + 3 * mb_count

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E1F-full-decode-throughput/interleaved",
        "measurement_class": "closed_loop_interleaved_decode_throughput",
        "caliber": {
            "b_semantics": (
                "dp (E1F dp caliber, E0dpf-gated): each microbatch lane m "
                f"serves its own distinct B_mb_global={mb_global} sequences "
                f"(TP rank r holds rows [r*bl,(r+1)*bl) at bl={local_batch} "
                f"per GPU per lane); {mb_count} lanes in flight -> "
                f"B_total={total_global} distinct sequences, "
                f"{local_batch * mb_count} KV rows per GPU; throughput is "
                "measured, no conversion"
            ),
            "microbatches": mb_count,
            "pipeline_form": (
                f"PP4 rotating interleave, {mb_count} microbatches in flight; "
                "iteration j of every stage serves lane j%L at position "
                "start+j//L; per-lane token loop closes with L-iteration "
                "pipeline latency (head -> argmax -> loopback pair -> embed); "
                "serial per-iteration handoff, no compute/transport overlap "
                "(out of scope this vertical)"
            ),
            "kv": (
                f"seeded decode residency at position {start_position} "
                "(deterministic-seeded-KV-not-real-prefix, E1a27); "
                f"max_seq_len={max_seq_len}; per-lane seeds "
                "lane_seed = seed + 7919*(m+1)"
            ),
            "graphs": (
                "per lane x per family stateful CUDA graphs (E0sf), lane-owned "
                "MoE graph slots (1+3m,2+3m,3+3m), per-lane per-family pools; "
                f"slots_per_shape={slots_per_shape}, eager slot 0 shared"
            ),
            "timing": (
                "host walls around torch.cuda.synchronize per component "
                "(E0qf serial decomposition caliber)"
            ),
            "hc_backend": args.hc_backend,
            "kv_dtype": args.kv_dtype,
            "indexer_kv_dtype": args.indexer_kv_dtype,
            "transport": "NCCL P2P fixed endpoints per lane, no-GDR",
        },
        "config_tag": args.config_tag,
        "nccl_env": {
            key: os.environ.get(key)
            for key in (
                "NCCL_SOCKET_IFNAME",
                "NCCL_IB_DISABLE",
                "NCCL_P2P_LEVEL",
                "NCCL_NET_GDR_LEVEL",
            )
        },
        "rank": rank,
        "local_rank": local_rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "seed": args.seed,
        "kv_dtype": args.kv_dtype,
        "indexer_kv_dtype": args.indexer_kv_dtype,
        "local_batch": local_batch,
        "mb_count": mb_count,
        "mb_global": mb_global,
        "global_batch": total_global,
        "start_position": start_position,
        "stop_position": stop_position,
        "max_seq_len": max_seq_len,
        "gate_cycles": gate_cycles if gate else None,
        "settle_cycles": None if gate else settle_cycles,
        "rounds": rounds,
        "steps_per_round": steps_per_round if not gate else None,
        "family_counts": {
            family.value: count for family, count in family_counts.items()
        },
        "check_mode": args.check_mode,
        "stage_layer_ids": {str(s): list(v) for s, v in STAGE_LAYERS.items()},
        "reference": {
            "dp_bg512_ctx2048_replay_p50_ms": DP_BG512_CTX2048_REPLAY_P50_MS,
            "dp_bg512_ctx2048_serial_tok_s": DP_BG512_CTX2048_SERIAL_TOK_S,
            "conversion_ceiling_tok_s": CONVERSION_CEILING_TOK_S,
        },
        "checkpoint_id": None,
        "placement": None,
        "memory": {},
        "gate_record": None,
        "settle": None,
        "round_results": [],
        "terminals": None,
        "handoff_records": {},
        "teardowns": {},
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
        if world != WORLD:
            raise ValueError(f"E1IF requires world=16 (2 nodes x 8), got {world}")
        topo = create_e1f_topology(rank)
        stage = topo["stage"]
        tp_rank = topo["tp_rank"]
        result["stage"] = stage
        result["tp_rank"] = tp_rank
        stage_layer_ids = STAGE_LAYERS[stage]

        warm = torch.ones(1, device=device)
        dist.all_reduce(warm, group=topo["tp_group"])
        if topo["next_pair"] is not None:
            pair_transfer(warm, send=True, group=topo["next_pair"], peer=1)
        if topo["prev_pair"] is not None:
            pair_transfer(warm, send=False, group=topo["prev_pair"], peer=0)
        if topo["loop_pair"] is not None:
            if stage == STAGE_COUNT - 1:
                pair_transfer(warm, send=True, group=topo["loop_pair"], peer=0)
            else:
                pair_transfer(warm, send=False, group=topo["loop_pair"], peer=1)
        torch.cuda.synchronize(device)
        result["tp_group_binding"] = validate_live_tp_group(
            topo["tp_group"],
            expected_local_rank=tp_rank,
            expected_global_ranks=topo["tp_global_ranks"],
        )
        result["placement"] = run_placement_check(stage=stage, world=world)
        if not result["placement"]["accepted"]:
            raise ValueError(f"PP4 placement violated: {result['placement']}")
        if rank == 0:
            print(f"[E1IF] placement {result['placement']['stage_hosts']}", flush=True)

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root, list(range(MODEL_LAYERS)), EXPECTED_TP_SIZE
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
        # load stage materials + embed/head
        phase_started = time.perf_counter()
        global_rows = local_batch * EXPECTED_TP_SIZE

        def load_materials() -> Any:
            return build_physical_stage(
                stage_id=stage,
                layer_ids=stage_layer_ids,
                model_config=model_config,
                stage_root=stage_root,
                tp_rank=tp_rank,
                tp_group=topo["tp_group"],
                tp_global_ranks=topo["tp_global_ranks"],
                device=device,
                checkpoint_id=result["checkpoint_id"],
                max_seq_len=max_seq_len,
                global_row_shapes=(global_rows,),
                slots_per_shape=slots_per_shape,
                progress_every=args.progress_every,
                progress=(
                    (lambda message: print(f"[E1IF] {message}", flush=True))
                    if rank in (0, 4, 8, 12)
                    else None
                ),
                kv_dtype=args.kv_dtype,
                indexer_kv_dtype=args.indexer_kv_dtype,
            )

        stage_material = synchronized_local_step(
            "load materials", load_materials, device=device, world=world
        )
        embed_material: EmbedHeadMaterial | None = None
        head_material: EmbedHeadMaterial | None = None
        if stage == 0:
            embed_material = load_embed_head_material(
                stage_root=stage_root,
                device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=True,
                load_head=False,
            )
        elif stage == STAGE_COUNT - 1:
            head_material = load_embed_head_material(
                stage_root=stage_root,
                device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=False,
                load_head=True,
            )
        result["diagnostic_seconds"]["load"] = time.perf_counter() - phase_started
        memory_snapshot("after_load")
        if rank in (0, 4, 8, 12):
            print(
                f"[E1IF] stage {stage} loaded ({len(stage_layer_ids)} layers, "
                f"free {result['memory']['after_load']['free_bytes'] / 2**30:.2f} "
                f"GiB, {result['diagnostic_seconds']['load']:.0f}s)",
                flush=True,
            )

        # ------------------------------------------------------------------
        # lanes (+ eager twins in gate mode)
        backend = resolve_hc_boundary_backend(
            None if args.hc_backend == "default" else args.hc_backend
        )
        phase_started = time.perf_counter()

        def build_all_lanes() -> tuple[list[MicrobatchLane], dict[int, MicrobatchLane]]:
            lanes: list[MicrobatchLane] = []
            twins: dict[int, MicrobatchLane] = {}
            for m in range(mb_count):
                payloads = {
                    material.layer_id: build_seed_payload(
                        material,
                        seed=lane_seed(args.seed, m),
                        local_batch=local_batch,
                        start_position=start_position,
                        device=device,
                        dp_tp_rank=tp_rank,
                    )
                    for material in stage_material.materials
                }
                lanes.append(
                    MicrobatchLane(
                        label=f"mb{m}",
                        lane_index=m,
                        materials=stage_material.materials,
                        payloads=payloads,
                        backend=backend,
                        local_batch=local_batch,
                        start_position=start_position,
                        stop_position=stop_position,
                        device=device,
                        moe_slots=lane_moe_slots(m),
                    )
                )
                if gate:
                    twins[m] = MicrobatchLane(
                        label=f"mb{m}-eager",
                        lane_index=m,
                        materials=stage_material.materials,
                        payloads=payloads,
                        backend=backend,
                        local_batch=local_batch,
                        start_position=start_position,
                        stop_position=stop_position,
                        device=device,
                        moe_slots=lane_moe_slots(m),
                    )
                if rank in (0, 12):
                    print(f"[E1IF] stage {stage} lane mb{m} built", flush=True)
            return lanes, twins

        lanes, twins = synchronized_local_step(
            "build lanes", build_all_lanes, device=device, world=world
        )
        result["plan_resident_bytes_per_lane"] = int(lanes[0].plan.resident_bytes)
        result["diagnostic_seconds"]["build"] = time.perf_counter() - phase_started
        memory_snapshot("after_build")
        if rank in (0, 4, 8, 12):
            print(
                f"[E1IF] stage {stage} {mb_count} lanes built"
                f"{' (+twins)' if gate else ''}, free "
                f"{result['memory']['after_build']['free_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
        if gate:
            for m in range(mb_count):
                if lanes[m].state_digests() != twins[m].state_digests():
                    raise RuntimeError(f"lane mb{m} twin was not seeded identically")

        # ------------------------------------------------------------------
        # warmup (E0hf pattern per lane) + snapshots
        capture_stream = torch.cuda.Stream(device=device)

        def snapshot_all() -> list[list[DirectState]]:
            return [
                [clone_state(state) for state in lane.stage.states] for lane in lanes
            ]

        snapshots = synchronized_local_step(
            "snapshot states", snapshot_all, device=device, world=world
        )

        def warm_inputs(m: int, position: int) -> tuple[torch.Tensor, torch.Tensor]:
            seed_m = lane_seed(args.seed, m)
            residual = deterministic_tensor(
                seed=(seed_m * 1_000_003 + position * 7_919) & ((1 << 62) - 1),
                shape=(mb_global, 1, HC_MULT, HIDDEN),
                device=device,
            )
            residual = dp_row_slice(residual, tp_rank, local_batch)
            mixed = (seed_m * 2654435761 + position * 7919) & ((1 << 63) - 1)
            ids = torch.full(
                (local_batch, 1),
                mixed % EXPECTED_VOCAB,
                dtype=torch.int64,
                device=device,
            )
            return residual, ids

        def run_warm_cycle(
            lane: MicrobatchLane,
            *,
            moe_slots: Mapping[DecodeGraphFamily, int] | None = None,
        ) -> None:
            for step in warm_schedule:
                residual, ids = warm_inputs(lane.lane_index, step.position)
                lane.plan.input_residual_buffer.copy_(residual)
                lane.plan.input_ids_buffer.copy_(ids)
                forward_eager_prevalidated(
                    lane.stage,
                    lane.plan,
                    graph_family=step.family,
                    moe_slot=(
                        0 if moe_slots is None else moe_slots[step.family]
                    ),
                )
                lane.cursor.advance_host(step.family)
            torch.cuda.synchronize(device)

        def restore_cycle(lane: MicrobatchLane, snapshot: list[DirectState]) -> None:
            copy_stage_states(lane.stage.states, snapshot)
            lane.cursor.reset(start_position)
            lane.plan.expected_position.fill_(start_position)
            lane.plan.stop_position_tensor.fill_(lane.plan.stop_position)

        def warmup_all() -> None:
            for m, lane in enumerate(lanes):
                run_warm_cycle(lane)
                restore_cycle(lane, snapshots[m])
                with torch.cuda.stream(capture_stream):
                    run_warm_cycle(lane, moe_slots=lane.moe_slots)
                torch.cuda.synchronize(device)
                restore_cycle(lane, snapshots[m])
                for slot in lane.moe_slot_tuple:
                    for moe in lane.stage.moes:
                        moe.reset_free_slot_completion_event(global_rows, slot)
                if rank in (0, 12):
                    print(f"[E1IF] stage {stage} mb{m} warm done", flush=True)
            for m, twin in twins.items():
                run_warm_cycle(twin)
                restore_cycle(twin, snapshots[m])
            for lane in list(lanes) + list(twins.values()):
                evidence = lane.terminal(start_position)
                if not evidence["accepted"]:
                    raise RuntimeError(
                        f"{lane.label} warmup restore drifted: {evidence}"
                    )

        phase_started = time.perf_counter()
        synchronized_local_step("warmups", warmup_all, device=device, world=world)
        result["diagnostic_seconds"]["warmup"] = time.perf_counter() - phase_started
        if not gate:
            del snapshots
            snapshots = None
            torch.cuda.empty_cache()
        memory_snapshot("after_warmup")
        if rank in (0, 4, 8, 12):
            print(
                f"[E1IF] stage {stage} warmup done "
                f"({result['diagnostic_seconds']['warmup']:.0f}s, free "
                f"{result['memory']['after_warmup']['free_bytes'] / 2**30:.2f} GiB)",
                flush=True,
            )

        # ------------------------------------------------------------------
        # fixed per-lane pipeline endpoints
        stagings: dict[int, torch.Tensor] = {}
        handoffs_in: dict[int, SerialPairHandoff] = {}
        handoffs_out: dict[int, SerialPairHandoff] = {}
        handoff_counters: dict[int, int] = {}
        token_buffers: dict[int, torch.Tensor] = {}
        initial_tokens: dict[int, torch.Tensor] = {}
        for m, lane in enumerate(lanes):
            if stage > 0:
                stagings[m] = torch.empty_like(lane.plan.input_residual_buffer)
                validate_handoff_endpoint(stagings[m], local_batch=local_batch)
                handoffs_in[m] = SerialPairHandoff(
                    stage_id=1,
                    pair_group=topo["prev_pair"],
                    endpoint=stagings[m],
                    local_batch=local_batch,
                )
                handoff_counters[id(handoffs_in[m])] = 0
            if stage < STAGE_COUNT - 1:
                handoffs_out[m] = SerialPairHandoff(
                    stage_id=0,
                    pair_group=topo["next_pair"],
                    endpoint=lane.plan.output_buffer,
                    local_batch=local_batch,
                )
                handoff_counters[id(handoffs_out[m])] = 0
            token_buffers[m] = torch.zeros(
                (local_batch, 1), dtype=torch.int64, device=device
            )
            if stage == 0:
                generator = torch.Generator(device="cpu").manual_seed(
                    lane_seed(args.seed, m) + 77
                )
                initial = torch.randint(
                    0, EXPECTED_VOCAB, (mb_global, 1), generator=generator
                ).to(device)
                initial_tokens[m] = dp_row_slice(initial, tp_rank, local_batch)
                token_buffers[m].copy_(initial_tokens[m])
        zero_ids = torch.zeros((local_batch, 1), dtype=torch.int64, device=device)
        if stage != 0:
            for lane in lanes:
                lane.plan.input_ids_buffer.copy_(zero_ids)
            for twin in twins.values():
                twin.plan.input_ids_buffer.copy_(zero_ids)
        if stage == 0:
            result["initial_tokens_first8"] = {
                str(m): [int(v) for v in token_buffers[m].view(-1)[:8].cpu().tolist()]
                for m in range(mb_count)
            }

        def handoff_step(handoff: SerialPairHandoff) -> None:
            counter = handoff_counters[id(handoff)]
            handoff.transfer_step(counter)
            handoff_counters[id(handoff)] = counter + 1

        def validate_lane(lane: MicrobatchLane, family: DecodeGraphFamily) -> None:
            if stage == 0:
                external_residual = embed_hc_residual(
                    embed_material, token_buffers[lane.lane_index]
                )
                external_ids = token_buffers[lane.lane_index]
            else:
                external_residual = stagings[lane.lane_index]
                external_ids = zero_ids
            lane.stage.validate_stateful_decode_call(
                external_residual,
                input_ids_local=external_ids,
                plan=lane.plan,
                graph_family=family,
            )

        def stage0_feed(lane: MicrobatchLane) -> None:
            token_buffer = token_buffers[lane.lane_index]
            hidden = torch.nn.functional.embedding(
                token_buffer, embed_material.embed_weight
            )
            lane.plan.input_residual_buffer.copy_(
                hidden.unsqueeze(2).expand(-1, -1, HC_MULT, -1)
            )
            lane.plan.input_ids_buffer.copy_(token_buffer)

        # ------------------------------------------------------------------
        # rotating interleaved phase runner
        def run_phase(
            *,
            label: str,
            lane_indices: Sequence[int],
            base_cycle: int,
            cycles: int,
            allow_capture: bool,
            timing: dict[str, list[float]] | None = None,
            record_tokens: bool = False,
            twin_check: MicrobatchLane | None = None,
            twin_stats: dict[str, Any] | None = None,
        ) -> dict[int, list[str]]:
            lane_list = list(lane_indices)
            token_records: dict[int, list[str]] = {m: [] for m in lane_list}
            nonfinite_positions: list[int] = []
            for k in range(cycles):
                step = schedule[base_cycle + k]
                family = step.family
                for m in lane_list:
                    lane = lanes[m]
                    t0 = time.perf_counter()
                    if stage == 0:
                        if k > 0:
                            pair_transfer(
                                token_buffers[m],
                                send=False,
                                group=topo["loop_pair"],
                                peer=1,
                            )
                            torch.cuda.synchronize(device)
                        t1 = time.perf_counter()
                        stage0_feed(lane)
                        torch.cuda.synchronize(device)
                        t2 = time.perf_counter()
                        if timing is not None:
                            timing["token_wait"].append((t1 - t0) * 1e3)
                            timing["embed"].append((t2 - t1) * 1e3)
                    else:
                        handoff_step(handoffs_in[m])
                        torch.cuda.synchronize(device)
                        t2 = time.perf_counter()
                        lane.plan.input_residual_buffer.copy_(stagings[m])
                        if timing is not None:
                            timing["recv"].append((t2 - t0) * 1e3)
                    if family not in lane.graphs:
                        if not allow_capture:
                            raise RuntimeError(
                                f"{label}: lane {lane.label} family "
                                f"{family.value} not captured"
                            )
                        t_cap = time.perf_counter()

                        def capture() -> torch.cuda.CUDAGraph:
                            return capture_stateful_graph(
                                lane.stage,
                                lane.plan,
                                graph_family=family,
                                capture_stream=capture_stream,
                                pool=lane.pools[family],
                            )

                        lane.graphs[family] = synchronized_local_step(
                            f"capture {lane.label} {family.value}",
                            capture,
                            device=device,
                            world=EXPECTED_TP_SIZE,
                            group=topo["tp_group"],
                        )
                        lane.capture_order.append(family.value)
                        replay_stateful_graph(
                            lane.graphs[family], lane.plan, graph_family=family
                        )
                        torch.cuda.synchronize(device)
                        if rank in (0, 12):
                            print(
                                f"[E1IF] stage {stage} {label} captured "
                                f"{lane.label}/{family.value} at position "
                                f"{step.position} "
                                f"({time.perf_counter() - t_cap:.1f}s)",
                                flush=True,
                            )
                    else:
                        replay_stateful_graph(
                            lane.graphs[family], lane.plan, graph_family=family
                        )
                        torch.cuda.synchronize(device)
                    t3 = time.perf_counter()
                    if timing is not None:
                        timing["replay"].append((t3 - t2) * 1e3)
                    if stage < STAGE_COUNT - 1:
                        handoff_step(handoffs_out[m])
                        torch.cuda.synchronize(device)
                        t4 = time.perf_counter()
                        if timing is not None:
                            timing["send"].append((t4 - t3) * 1e3)
                    else:
                        logits = head_logits(head_material, lane.plan.output_buffer)
                        token_buffers[m].copy_(logits.argmax(dim=-1, keepdim=True))
                        torch.cuda.synchronize(device)
                        t4 = time.perf_counter()
                        pair_transfer(
                            token_buffers[m],
                            send=True,
                            group=topo["loop_pair"],
                            peer=0,
                        )
                        torch.cuda.synchronize(device)
                        t5 = time.perf_counter()
                        if timing is not None:
                            timing["head"].append((t4 - t3) * 1e3)
                            timing["token_send"].append((t5 - t4) * 1e3)
                        if record_tokens:
                            token_records[m].append(tensor_sha256(token_buffers[m]))
                            if not bool(torch.isfinite(logits).all().item()):
                                nonfinite_positions.append(step.position)
                    if timing is not None:
                        timing["iter_wall"].append(
                            (time.perf_counter() - t0) * 1e3
                        )
                    # gate-mode graph-vs-eager twin (solo phases only)
                    if twin_check is not None:
                        twin_plan = twin_check.plan
                        twin_plan.input_residual_buffer.copy_(
                            lane.plan.input_residual_buffer
                        )
                        twin_plan.input_ids_buffer.copy_(lane.plan.input_ids_buffer)
                        forward_eager_prevalidated(
                            twin_check.stage, twin_plan, graph_family=family
                        )
                        torch.cuda.synchronize(device)
                        twin_check.cursor.advance_host(family)
                        if bool(
                            torch.equal(
                                lane.plan.output_buffer, twin_plan.output_buffer
                            )
                        ):
                            twin_stats["bitwise_steps"] += 1
                        else:
                            difference = (
                                lane.plan.output_buffer.float()
                                - twin_plan.output_buffer.float()
                            )
                            twin_stats["mismatched_positions"].append(step.position)
                            twin_stats["max_abs"] = max(
                                twin_stats["max_abs"],
                                float(difference.abs().max().item()),
                            )
                    lane.cursor.advance_host(family)
                if rank == 0 and k % 64 == 0:
                    print(
                        f"[E1IF] {label} cycle {k}/{cycles} pos {step.position} "
                        f"family {family.value}",
                        flush=True,
                    )
            # drain the final cycle's tokens so no send is unmatched
            if stage == 0:
                for m in lane_list:
                    pair_transfer(
                        token_buffers[m], send=False, group=topo["loop_pair"], peer=1
                    )
                torch.cuda.synchronize(device)
                if record_tokens:
                    for m in lane_list:
                        token_records[m].append(
                            "drain:" + tensor_sha256(token_buffers[m])
                        )
            torch.cuda.synchronize(device)
            if record_tokens and stage == STAGE_COUNT - 1 and nonfinite_positions:
                token_records["nonfinite_positions"] = nonfinite_positions  # type: ignore[index]
            return token_records

        # ------------------------------------------------------------------
        if gate:
            gate_record: dict[str, Any] = {
                "cycles": gate_cycles,
                "judgment": (
                    "per lane: solo serial run (graph, lazy capture, per-step "
                    "bitwise graph-vs-eager twin) -> restore -> all lanes "
                    "interleaved over the same positions; token digest "
                    "trajectory, final KV digests, and cursor terminal must "
                    "be bitwise identical solo vs interleaved (no cross-talk)"
                ),
                "lanes": {},
            }
            solo_tokens: dict[int, list[str]] = {}
            solo_kv: dict[int, dict[str, str]] = {}
            phase_started = time.perf_counter()
            for m in range(mb_count):
                lane = lanes[m]
                synchronized_local_step(
                    f"solo mb{m} validation",
                    lambda lane=lane: validate_lane(lane, schedule[0].family),
                    device=device,
                    world=world,
                )
                twin_stats = {
                    "bitwise_steps": 0,
                    "mismatched_positions": [],
                    "max_abs": 0.0,
                }
                records = run_phase(
                    label=f"solo-mb{m}",
                    lane_indices=[m],
                    base_cycle=0,
                    cycles=gate_cycles,
                    allow_capture=True,
                    record_tokens=True,
                    twin_check=twins[m],
                    twin_stats=twin_stats,
                )
                solo_tokens[m] = records[m]
                solo_kv[m] = lane.state_digests()
                twin_final_equal = bool(solo_kv[m] == twins[m].state_digests())
                lane_record = {
                    "capture_order": list(lane.capture_order),
                    "solo_terminal": lane.terminal(start_position + gate_cycles),
                    "solo_final_output_digest": tensor_sha256(
                        lane.plan.output_buffer
                    ),
                    "graph_vs_eager": dict(twin_stats),
                    "twin_final_state_digests_equal": twin_final_equal,
                    "twin_terminal": twins[m].terminal(
                        start_position + gate_cycles
                    ),
                }
                gate_record["lanes"][str(m)] = lane_record
                # free the twin, restore the lane to the seeded start state
                del twins[m]
                torch.cuda.empty_cache()

                def restore_after_solo(lane=lane, m=m) -> None:
                    restore_cycle(lane, snapshots[m])
                    if stage == 0:
                        token_buffers[m].copy_(initial_tokens[m])
                    evidence = lane.terminal(start_position)
                    if not evidence["accepted"]:
                        raise RuntimeError(
                            f"mb{m} post-solo restore drifted: {evidence}"
                        )

                synchronized_local_step(
                    f"restore mb{m}", restore_after_solo, device=device, world=world
                )
                if rank in (0, 12):
                    print(
                        f"[E1IF] stage {stage} solo-mb{m} done: bitwise "
                        f"{twin_stats['bitwise_steps']}/{gate_cycles}",
                        flush=True,
                    )
            result["diagnostic_seconds"]["solo"] = time.perf_counter() - phase_started
            memory_snapshot("after_solo")

            # interleaved re-run over the same positions (replay only)
            phase_started = time.perf_counter()
            synchronized_local_step(
                "interleaved entry", lambda: None, device=device, world=world
            )
            inter_records = run_phase(
                label="interleaved",
                lane_indices=list(range(mb_count)),
                base_cycle=0,
                cycles=gate_cycles,
                allow_capture=False,
                record_tokens=True,
            )
            result["diagnostic_seconds"]["interleaved"] = (
                time.perf_counter() - phase_started
            )
            all_lane_ok = True
            for m in range(mb_count):
                lane = lanes[m]
                lane_record = gate_record["lanes"][str(m)]
                inter_kv = lane.state_digests()
                lane_record["interleaved_terminal"] = lane.terminal(
                    start_position + gate_cycles
                )
                lane_record["interleaved_final_output_digest"] = tensor_sha256(
                    lane.plan.output_buffer
                )
                lane_record["kv_digests_equal_solo_vs_interleaved"] = bool(
                    inter_kv == solo_kv[m]
                )
                if stage in (0, STAGE_COUNT - 1):
                    lane_record["token_trajectory_equal"] = bool(
                        inter_records[m] == solo_tokens[m]
                    )
                    lane_record["token_trajectory_len"] = len(inter_records[m])
                else:
                    lane_record["token_trajectory_equal"] = None
                lane_record["output_digest_equal"] = bool(
                    lane_record["interleaved_final_output_digest"]
                    == lane_record["solo_final_output_digest"]
                )
                lane_ok = bool(
                    lane_record["capture_order"] == CANONICAL_CAPTURE_ORDER
                    and lane_record["solo_terminal"]["accepted"]
                    and lane_record["interleaved_terminal"]["accepted"]
                    and lane_record["graph_vs_eager"]["bitwise_steps"] == gate_cycles
                    and not lane_record["graph_vs_eager"]["mismatched_positions"]
                    and lane_record["twin_final_state_digests_equal"]
                    and lane_record["kv_digests_equal_solo_vs_interleaved"]
                    and lane_record["output_digest_equal"]
                    and lane_record["token_trajectory_equal"] in (True, None)
                )
                lane_record["accepted"] = lane_ok
                all_lane_ok = all_lane_ok and lane_ok
            gate_record["accepted"] = all_lane_ok
            result["gate_record"] = gate_record
            memory_snapshot("after_interleaved")
            if rank == 0:
                print(
                    f"[E1IF] gate: {'PASS' if all_lane_ok else 'FAIL'} on this rank",
                    flush=True,
                )
        else:
            # --------------------------------------------------------------
            # timed mode: interleaved settle (lazy capture) + timed rounds
            for m in range(mb_count):
                lane = lanes[m]
                synchronized_local_step(
                    f"settle mb{m} validation",
                    lambda lane=lane: validate_lane(lane, schedule[0].family),
                    device=device,
                    world=world,
                )
            phase_started = time.perf_counter()
            run_phase(
                label="settle",
                lane_indices=list(range(mb_count)),
                base_cycle=0,
                cycles=settle_cycles,
                allow_capture=True,
            )
            settle_record = {
                "cycles": settle_cycles,
                "capture_orders": {
                    str(m): list(lanes[m].capture_order) for m in range(mb_count)
                },
                "judgment": (
                    "no per-step compare (timed mode); cross-talk gate is the "
                    "separate --check-mode gate run"
                ),
            }
            settle_record["accepted"] = all(
                lanes[m].capture_order == CANONICAL_CAPTURE_ORDER
                for m in range(mb_count)
            )
            result["settle"] = settle_record
            result["diagnostic_seconds"]["settle"] = (
                time.perf_counter() - phase_started
            )
            memory_snapshot("after_settle")
            if rank == 0:
                print(
                    f"[E1IF] settle done ({result['diagnostic_seconds']['settle']:.0f}s)",
                    flush=True,
                )

            for round_index in range(rounds):
                synchronized_local_step(
                    f"round {round_index} entry",
                    lambda: None,
                    device=device,
                    world=world,
                )
                timing: dict[str, list[float]] = {
                    key: []
                    for key in (
                        "token_wait",
                        "embed",
                        "replay",
                        "send",
                        "recv",
                        "head",
                        "token_send",
                        "iter_wall",
                    )
                }
                round_started = time.perf_counter()
                run_phase(
                    label=f"round{round_index}",
                    lane_indices=list(range(mb_count)),
                    base_cycle=settle_cycles + round_index * steps_per_round,
                    cycles=steps_per_round,
                    allow_capture=False,
                    timing=timing,
                )
                round_wall = time.perf_counter() - round_started
                for m in range(mb_count):
                    dispatch_error = int(lanes[m].cursor.dispatch_error.item())
                    if dispatch_error:
                        raise RuntimeError(
                            f"round {round_index} lane mb{m} sticky dispatch "
                            f"error {dispatch_error}"
                        )
                iter_walls = timing["iter_wall"]
                tokens_total = mb_count * mb_global * steps_per_round
                component_ms = {
                    key: float(sum(values))
                    for key, values in timing.items()
                    if values
                }
                record: dict[str, Any] = {
                    "round": round_index,
                    "cycles": steps_per_round,
                    "iterations": len(iter_walls),
                    "round_wall_s": round_wall,
                    "tokens_total": tokens_total,
                    "aggregate_tok_s_wall": tokens_total / round_wall,
                    "aggregate_tok_s_iter_p50": mb_global
                    / (sorted(iter_walls)[len(iter_walls) // 2] / 1e3),
                    "timing_ms": {
                        key: summarize_ms(values)
                        for key, values in timing.items()
                        if values
                    },
                    "component_share_of_wall": {
                        key: value / (round_wall * 1e3)
                        for key, value in component_ms.items()
                        if key != "iter_wall"
                    },
                    "replay_busy_fraction": component_ms.get("replay", 0.0)
                    / (round_wall * 1e3),
                    "host_gap_fraction": 1.0
                    - component_ms.get("iter_wall", 0.0) / (round_wall * 1e3),
                    "iter_wall_raw_ms": [round(v, 4) for v in iter_walls],
                    "replay_raw_ms": [round(v, 4) for v in timing["replay"]],
                }
                # cross-lane diagnostics + last tokens
                lane_digests = {}
                for m in range(mb_count):
                    digest = tensor_sha256(lanes[m].plan.output_buffer)
                    gathered_digests: list[Any] = [None] * EXPECTED_TP_SIZE
                    dist.all_gather_object(
                        gathered_digests, digest, group=topo["tp_group"]
                    )
                    lane_digests[str(m)] = {
                        "output_lanes_bitwise": len(set(gathered_digests)) == 1,
                        "expected_bitwise": False,  # dp semantics: distinct rows
                    }
                record["cross_tp_output_digests"] = lane_digests
                if stage == STAGE_COUNT - 1:
                    record["tokens_first8"] = {
                        str(m): [
                            int(v)
                            for v in token_buffers[m].view(-1)[:8].cpu().tolist()
                        ]
                        for m in range(mb_count)
                    }
                    logits = head_logits(head_material, lanes[0].plan.output_buffer)
                    record["logits_finite"] = bool(
                        torch.isfinite(logits).all().item()
                    )
                result["round_results"].append(record)
                if rank in (0, 12):
                    summary = record["timing_ms"].get("iter_wall", {})
                    print(
                        f"[E1IF] stage {stage} round {round_index}: iter_wall "
                        f"p50 {summary.get('p50_ms', 0):.2f} ms, aggregate "
                        f"{record['aggregate_tok_s_wall']:.0f} tok/s (wall), "
                        f"{record['aggregate_tok_s_iter_p50']:.0f} tok/s "
                        f"(iter p50), replay busy "
                        f"{record['replay_busy_fraction'] * 100:.1f}%",
                        flush=True,
                    )
                memory_snapshot(f"after_round_{round_index}")

        # ------------------------------------------------------------------
        # terminals + handoff close + per-lane teardown
        final_position = start_position + total_cycles
        result["terminals"] = {
            str(m): lanes[m].terminal(final_position) for m in range(mb_count)
        }
        per_lane_transfers = (2 * gate_cycles) if gate else total_cycles
        for m in range(mb_count):
            records = {}
            if m in handoffs_in:
                records["in"] = handoffs_in[m].close(
                    expected_steps=per_lane_transfers
                )
            if m in handoffs_out:
                records["out"] = handoffs_out[m].close(
                    expected_steps=per_lane_transfers
                )
            result["handoff_records"][str(m)] = records

        teardowns_ok = True
        for m in range(mb_count):
            lane = lanes[m]
            teardown = synchronized_local_step(
                f"teardown mb{m}",
                lambda lane=lane: teardown_stateful_graphs(
                    lane.stage, lane.plan, lane.graphs, pool_handles=lane.pools
                ),
                device=device,
                world=world,
            )
            result["teardowns"][str(m)] = teardown
            teardowns_ok = teardowns_ok and bool(teardown["accepted"])
        memory_snapshot("at_end")

        phase_accepted = (
            bool(result["gate_record"] and result["gate_record"]["accepted"])
            if gate
            else bool(
                result["settle"]
                and result["settle"]["accepted"]
                and len(result["round_results"]) == rounds
            )
        )
        result["accepted"] = bool(
            result["placement"]["accepted"]
            and phase_accepted
            and all(
                lanes[m].capture_order == CANONICAL_CAPTURE_ORDER
                for m in range(mb_count)
            )
            and all(
                result["terminals"][str(m)]["accepted"] for m in range(mb_count)
            )
            and teardowns_ok
            and not any(lanes[m].stage.poisoned for m in range(mb_count))
        )
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
    result["diagnostic_seconds"]["process"] = time.perf_counter() - started

    try:
        gathered: list[Any] = [None for _ in range(world)]
        dist.all_gather_object(gathered, result)
    except Exception:
        gathered = [result]
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False

    accepted_all = bool(
        len(gathered) == world
        and all(
            isinstance(record, dict) and record.get("accepted") for record in gathered
        )
    )
    write_json(out_dir / f"rank{rank}.json", result)
    if rank == 0:
        merged = []
        for record in gathered:
            if not isinstance(record, dict) or record.get("tp_rank") != 0:
                continue
            trimmed = dict(record)
            trimmed["round_results"] = [
                {
                    key: value
                    for key, value in round_record.items()
                    if key not in ("iter_wall_raw_ms", "replay_raw_ms")
                }
                for round_record in record.get("round_results", [])
            ]
            merged.append(trimmed)
        write_json(
            out_dir / "result.json",
            {
                "experiment": "E1F-full-decode-throughput/interleaved",
                "accepted": accepted_all,
                "kv_dtype": args.kv_dtype,
                "indexer_kv_dtype": args.indexer_kv_dtype,
                "mb_count": mb_count,
                "local_batch": local_batch,
                "global_batch": total_global,
                "check_mode": args.check_mode,
                "stage_representatives": merged,
            },
        )
        print(f"[E1IF] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
