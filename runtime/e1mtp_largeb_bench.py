#!/usr/bin/env python3
"""E1MTPLB: large-B chained MTP speculative decoding on the interleaved
PP4xTP4 pipeline (18th vertical).

Form: **chained dual-pass** (DESIGN-largeB-graph.md alternative chosen for
this vertical; see results/mtp-largeb/ for the trade-off note).  Every lane
runs one round per pipeline rotation: pass A verifies the pending token at
per-row ``positions``, pass B speculatively feeds the round's draft at
``positions + 1``.  Both passes are single-token row-position CUDA graphs
(``dsv4_direct.specdec``); accept/reject only flips per-row position advance
and the ratio-4 shadow restore -- shapes never change.  ``mb`` lanes x 2
passes fill the PP4 pipeline (mb=2 gives 4 slots in flight, the E1IF steady
state).  The MTP block (mtp.0, tail stage with head/embed) runs as two
additional per-lane graphs: MTP-1 ingests the pass-A committed pair, MTP-2
the pass-B pair (masked by accept via the refeed-healing ring), producing
the next round's draft on-device.

Gate mode (``--check-mode gate``, small bl): per lane
  1. OFF solo run (eager family machinery, the production baseline path) for
     2R steps recording per-row token streams; restore.
  2. ON solo run (row-position graphs, lazy capture) for R rounds with a
     per-pass bitwise graph-vs-eager twin; restore.
  3. ON interleaved run over all lanes; per-row streams, final state
     digests, and positions must equal the solo run (no cross-talk).
Acceptance: every row's ON emitted stream is an exact prefix of its OFF
stream (protocol losslessness at batch scale, riding the measured
row-composition independence of the Marlin MoE), twin bitwise, no
cross-talk, clean teardown.

Timed mode: settle rounds (capture) + ``--rounds x --steps`` timed chained
rounds; reports round wall, measured batch acceptance, and effective
tok/s = (B_total*cycles + accepts) / wall.
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
    head_logits,
    load_embed_head_material,
)
from dsv4_direct.model_contract import MTP_LAYER_ID
from dsv4_direct.mtp_block import MTPLane, build_mtp_layer_material
from dsv4_direct.physical_stage import (
    EXPECTED_TP_SIZE,
    build_physical_stage,
    validate_live_tp_group,
)
from dsv4_direct.specdec import (
    forward_mtp_spec,
    forward_spec_stage,
    prepare_mtp_spec_plan,
    prepare_spec_stage_plan,
)
from dsv4_direct.stateful_decode import build_decode_schedule
from dsv4_direct.superstage import TP4DecodeStage

from e1f_full_decode_bench import (
    EXPECTED_VOCAB,
    HC_MULT,
    HIDDEN,
    MODEL_LAYERS,
    STAGE_COUNT,
    STAGE_LAYERS,
    WORLD,
    StageLane,
    build_seed_payload,
    clone_state,
    copy_stage_states,
    create_e1f_topology,
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


def lane_seed(seed: int, lane_index: int) -> int:
    return (seed + 7_919 * (lane_index + 1)) & ((1 << 62) - 1)


def initial_tokens(seed: int, lane_index: int, *, offset: int, mb_global: int):
    generator = torch.Generator(device="cpu").manual_seed(
        lane_seed(seed, lane_index) + offset
    )
    return torch.randint(0, EXPECTED_VOCAB, (mb_global, 1), generator=generator)


class SpecLane:
    """One microbatch lane: seeded stage, spec plan, graphs, s3 extras."""

    def __init__(
        self,
        *,
        label: str,
        lane_index: int,
        materials: Sequence[Any],
        payload_factory: Any,
        backend: Any | None,
        local_batch: int,
        start_position: int,
        stop_position: int,
        device: torch.device,
        moe_slot_a: int,
        moe_slot_b: int,
    ) -> None:
        self.label = label
        self.lane_index = lane_index
        blocks = []
        for material in materials:
            state = material.new_state(num_local_sequences=local_batch)
            payload = payload_factory(material)
            seed_state(
                material, state, payload, start_position=start_position,
            )
            del payload
            torch.cuda.empty_cache()
            blocks.append(material.new_block(state))
        self.stage = TP4DecodeStage(blocks, hc_boundary_backend=backend)
        self.sp = prepare_spec_stage_plan(
            self.stage,
            batch_size=local_batch,
            start_position=start_position,
            stop_position=stop_position,
            moe_slot_a=moe_slot_a,
            moe_slot_b=moe_slot_b,
            device=device,
        )
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self.mtp_lane: MTPLane | None = None
        self.mtp_plan = None
        self.mtp_graphs: dict[str, torch.cuda.CUDAGraph] = {}

    def state_digests(self) -> dict[str, str]:
        digests = {
            str(layer_id): full_state_sha256(state)
            for layer_id, state in zip(
                self.stage.layer_ids, self.stage.states, strict=True
            )
        }
        if self.mtp_lane is not None:
            digests["mtp"] = full_state_sha256(self.mtp_lane.state)
        return digests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--local-batch", type=int, default=8)
    parser.add_argument("--mb-count", type=int, default=2)
    parser.add_argument("--start-position", type=int, default=2048)
    parser.add_argument("--settle-rounds", type=int, default=16)
    parser.add_argument("--gate-rounds", type=int, default=132)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300, help="chained rounds per timed segment")
    parser.add_argument("--check-mode", type=str, default="off", choices=("off", "gate"))
    parser.add_argument("--hc-backend", type=str, default="fused", choices=("fused", "eager"))
    parser.add_argument("--kv-dtype", type=str, default="fp8",
                        choices=("bf16", "fp8", "fp8_rope_bf16"))
    parser.add_argument("--indexer-kv-dtype", type=str, default="fp8",
                        choices=("bf16", "fp8"))
    parser.add_argument("--accept-mode", type=str, default="normal",
                        choices=("normal", "force_reject", "split"),
                        help="debug: override accept decisions (split = force-reject rows < bl/2)")
    parser.add_argument("--trace-rows", type=int, default=0,
                        help="debug: per-stage digest tracing of the first N rows (gate+split only)")
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--config-tag", type=str, default="mtp-largeb-chained")
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
    mb_global = local_batch * EXPECTED_TP_SIZE
    total_global = mb_global * mb_count
    start_position = int(args.start_position)
    gate_rounds = int(args.gate_rounds)
    settle_rounds = int(args.settle_rounds)
    segments = 0 if gate else int(args.rounds)
    rounds_per_segment = int(args.steps)
    if gate:
        total_rounds = gate_rounds
    else:
        total_rounds = settle_rounds + segments * rounds_per_segment
    stop_position = start_position + 2 * total_rounds + 8
    max_seq_len = ((stop_position + 127) // 128 + 1) * 128
    if start_position < 2047 or start_position % 128:
        raise SystemExit("start position must be 128-aligned and >= 2047")
    if mb_count < 2:
        raise SystemExit("chained interleave requires mb_count >= 2 (2mb slots)")
    off_steps = 2 * gate_rounds

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    # gate keeps the shared eager slot 0 (twins) + family slots 1-3 for the
    # OFF StageLane; timed drops the eager slot entirely (graphs only).
    slots_per_shape = max(1 + 2 * mb_count, 4) if gate else 2 * mb_count

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E1F-full-decode-throughput/mtp-largeb",
        "measurement_class": "chained_mtp_interleaved_decode",
        "caliber": {
            "form": (
                "chained dual-pass MTP rounds: per lane per round two "
                "single-token row-position graphs (pass A verify at "
                "positions[B], pass B draft at positions+1); accept/reject "
                "changes only per-row advance + ratio-4 shadow restore; "
                f"{mb_count} lanes x 2 passes = {2 * mb_count} pipeline "
                "slots; MTP block (mtp.0) as two per-lane tail-stage graphs"
            ),
            "b_semantics": (
                f"dp caliber: lane m serves B_mb_global={mb_global} distinct "
                f"sequences (bl={local_batch}/GPU/lane), B_total={total_global}"
            ),
            "kv": (
                f"seeded decode residency at {start_position}; "
                f"max_seq_len={max_seq_len} (covers worst-case 2 pos/round)"
            ),
            "hc_backend": args.hc_backend,
            "kv_dtype": args.kv_dtype,
            "indexer_kv_dtype": args.indexer_kv_dtype,
            "graph_pool": "single global pool (17th-vertical scope)",
        },
        "config_tag": args.config_tag,
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
        "check_mode": args.check_mode,
        "gate_rounds": gate_rounds if gate else None,
        "settle_rounds": None if gate else settle_rounds,
        "segments": segments,
        "rounds_per_segment": rounds_per_segment if not gate else None,
        "checkpoint_id": None,
        "placement": None,
        "memory": {},
        "gate_record": None,
        "round_results": [],
        "positions_final": None,
        "teardown": None,
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
            raise ValueError(f"requires world=16, got {world}")
        topo = create_e1f_topology(rank)
        stage_id = topo["stage"]
        tp_rank = topo["tp_rank"]
        result["stage"] = stage_id
        result["tp_rank"] = tp_rank
        stage_layer_ids = STAGE_LAYERS[stage_id]

        warm = torch.ones(1, device=device)
        dist.all_reduce(warm, group=topo["tp_group"])
        if topo["next_pair"] is not None:
            pair_transfer(warm, send=True, group=topo["next_pair"], peer=1)
        if topo["prev_pair"] is not None:
            pair_transfer(warm, send=False, group=topo["prev_pair"], peer=0)
        if topo["loop_pair"] is not None:
            if stage_id == STAGE_COUNT - 1:
                pair_transfer(warm, send=True, group=topo["loop_pair"], peer=0)
            else:
                pair_transfer(warm, send=False, group=topo["loop_pair"], peer=1)
        torch.cuda.synchronize(device)
        result["tp_group_binding"] = validate_live_tp_group(
            topo["tp_group"],
            expected_local_rank=tp_rank,
            expected_global_ranks=topo["tp_global_ranks"],
        )
        result["placement"] = run_placement_check(stage=stage_id, world=world)
        if not result["placement"]["accepted"]:
            raise ValueError(f"PP4 placement violated: {result['placement']}")

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root,
                    list(range(MODEL_LAYERS)) + [MTP_LAYER_ID],
                    EXPECTED_TP_SIZE,
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
        # load
        phase_started = time.perf_counter()
        global_rows = local_batch * EXPECTED_TP_SIZE

        def load_materials() -> Any:
            return build_physical_stage(
                stage_id=stage_id,
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
                    (lambda message: print(f"[E1MTPLB] {message}", flush=True))
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
        mtp_material = None
        if stage_id == 0:
            embed_material = load_embed_head_material(
                stage_root=stage_root, device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=True, load_head=False,
            )
        elif stage_id == STAGE_COUNT - 1:
            head_material = load_embed_head_material(
                stage_root=stage_root, device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=True, load_head=True,
            )
            mtp_material = synchronized_local_step(
                "load mtp material",
                lambda: build_mtp_layer_material(
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
                    kv_dtype=args.kv_dtype,
                ),
                device=device,
                world=EXPECTED_TP_SIZE,
                group=topo["tp_group"],
            )
        result["diagnostic_seconds"]["load"] = time.perf_counter() - phase_started
        memory_snapshot("after_load")
        if rank in (0, 4, 8, 12):
            print(
                f"[E1MTPLB] stage {stage_id} loaded, free "
                f"{result['memory']['after_load']['free_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )

        # ------------------------------------------------------------------
        # lanes
        backend = resolve_hc_boundary_backend(args.hc_backend)
        phase_started = time.perf_counter()
        def lane_payload_factory(m: int):
            def factory(material):
                return build_seed_payload(
                    material,
                    seed=lane_seed(args.seed, m),
                    local_batch=local_batch,
                    start_position=start_position,
                    device=device,
                    dp_tp_rank=tp_rank,
                )

            return factory

        def build_lanes():
            lanes: list[SpecLane] = []
            twins: dict[int, SpecLane] = {}
            off_lanes: dict[int, StageLane] = {}
            for m in range(mb_count):
                # payloads are per-lane transients (the resident copy is the
                # seeded state); holding every lane's payload was the bl112
                # build OOM.
                factory = lane_payload_factory(m)
                lanes.append(
                    SpecLane(
                        label=f"mb{m}",
                        lane_index=m,
                        materials=stage_material.materials,
                        payload_factory=factory,
                        backend=backend,
                        local_batch=local_batch,
                        start_position=start_position,
                        stop_position=stop_position,
                        device=device,
                        moe_slot_a=(1 + 2 * m) if gate else 2 * m,
                        moe_slot_b=(2 + 2 * m) if gate else 2 * m + 1,
                    )
                )
                if gate:
                    twins[m] = SpecLane(
                        label=f"mb{m}-eager",
                        lane_index=m,
                        materials=stage_material.materials,
                        payload_factory=factory,
                        backend=backend,
                        local_batch=local_batch,
                        start_position=start_position,
                        stop_position=stop_position,
                        device=device,
                        moe_slot_a=0,
                        moe_slot_b=0,
                    )
                    payload_dict = {
                        material.layer_id: factory(material)
                        for material in stage_material.materials
                    }
                    off_lanes[m] = StageLane(
                        label=f"mb{m}-off",
                        materials=stage_material.materials,
                        payloads=payload_dict,
                        backend=backend,
                        local_batch=local_batch,
                        start_position=start_position,
                        stop_position=stop_position,
                        device=device,
                    )
                if gate:
                    del payload_dict
                torch.cuda.empty_cache()
                if rank in (0, 12):
                    print(f"[E1MTPLB] stage {stage_id} lane mb{m} built", flush=True)
            return lanes, twins, off_lanes

        lanes, twins, off_lanes = synchronized_local_step(
            "build lanes", build_lanes, device=device, world=world
        )

        # stage-3 extras: MTP lanes + plans, per main lane (and twins)
        if stage_id == STAGE_COUNT - 1:
            for collection, slot_base in ((lanes, 1), (list(twins.values()), 0)):
                for lane in collection:
                    m = lane.lane_index
                    lane.mtp_lane = MTPLane(
                        mtp_material,
                        embed_weight=head_material.embed_weight,
                        head_weight=head_material.head_weight,
                        batch_size=local_batch,
                        device=device,
                    )
                    mtp_seed = lane_seed(args.seed, m) + 999_331
                    raw = dp_row_slice(
                        deterministic_bf16(
                            mtp_seed, (mb_global, 128, 512), scale=0.03
                        ),
                        tp_rank,
                        local_batch,
                    ).to(device)
                    lane.mtp_lane.state.seed_decode_residency(
                        start_pos=start_position, raw=raw
                    )
                    if slot_base:
                        mtp_slot_a = (1 + 2 * m) if gate else 2 * m
                        mtp_slot_b = (2 + 2 * m) if gate else 2 * m + 1
                    else:
                        mtp_slot_a = mtp_slot_b = 0
                    lane.mtp_plan = prepare_mtp_spec_plan(
                        lane.mtp_lane,
                        batch_size=local_batch,
                        moe_slot_a=mtp_slot_a,
                        moe_slot_b=mtp_slot_b,
                        device=device,
                    )
        result["diagnostic_seconds"]["build"] = time.perf_counter() - phase_started
        memory_snapshot("after_build")
        if rank in (0, 4, 8, 12):
            print(
                f"[E1MTPLB] stage {stage_id} lanes built, free "
                f"{result['memory']['after_build']['free_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )

        # ------------------------------------------------------------------
        # per-lane snapshots + restore helper
        def snapshot_lane(lane: SpecLane):
            states = [clone_state(state) for state in lane.stage.states]
            mtp = None
            if lane.mtp_lane is not None:
                mtp = clone_state(lane.mtp_lane.state)
            return states, mtp

        def restore_lane(lane: SpecLane, snapshot) -> None:
            states, mtp = snapshot
            copy_stage_states(lane.stage.states, states)
            if mtp is not None:
                lane.mtp_lane.state.copy_from(mtp)
            lane.sp.positions.fill_(start_position)
            lane.sp.advance.zero_()
            lane.sp.accept.fill_(1)

        # Gate mode restores lanes between phases long after warmup, so it
        # snapshots everything up front.  Timed mode takes one transient
        # snapshot per lane inside the warmup loop (the all-lanes clone was
        # the 17th-vertical bl=72 OOM wall, reproduced here at bl=112).
        snapshots = None
        twin_snapshots = None
        if gate:
            snapshots = synchronized_local_step(
                "snapshots",
                lambda: [snapshot_lane(lane) for lane in lanes],
                device=device,
                world=world,
            )
            twin_snapshots = [snapshot_lane(twins[m]) for m in range(mb_count)]

        # ------------------------------------------------------------------
        # loopback/chain buffers, initial tokens
        init_pending: dict[int, torch.Tensor] = {}
        init_draft: dict[int, torch.Tensor] = {}
        for m in range(mb_count):
            init_pending[m] = dp_row_slice(
                initial_tokens(args.seed, m, offset=77, mb_global=mb_global),
                tp_rank, local_batch,
            ).to(device)
            init_draft[m] = dp_row_slice(
                initial_tokens(args.seed, m, offset=177, mb_global=mb_global),
                tp_rank, local_batch,
            ).to(device)
        meta = {
            m: torch.zeros((local_batch, 3), dtype=torch.int64, device=device)
            for m in range(mb_count)
        }
        accept_wire = {
            m: torch.ones((local_batch, 1), dtype=torch.int64, device=device)
            for m in range(mb_count)
        }
        stagings = {
            m: torch.empty(
                (local_batch, 1, HC_MULT, HIDDEN),
                dtype=torch.bfloat16, device=device,
            )
            for m in range(mb_count)
        }
        # stage-3 round state
        t1_buf = {
            m: torch.zeros((local_batch, 1), dtype=torch.int64, device=device)
            for m in range(mb_count)
        }
        d1_buf = {
            m: torch.zeros((local_batch,), dtype=torch.int64, device=device)
            for m in range(mb_count)
        }
        draft_stored = {m: init_draft[m].clone() for m in range(mb_count)}
        accept_new = {
            m: torch.ones((local_batch, 1), dtype=torch.int64, device=device)
            for m in range(mb_count)
        }
        accept_sum = {
            m: torch.zeros((), dtype=torch.int64, device=device)
            for m in range(mb_count)
        }
        row_accepts = {
            m: torch.zeros((local_batch,), dtype=torch.int64, device=device)
            for m in range(mb_count)
        }

        trace_rows = int(args.trace_rows)
        trace: dict[str, Any] = {"off_in": {}, "off_out": {}, "on_in": {}, "on_out": {}}

        def trace_record(kind: str, m: int, index: int, tensor: torch.Tensor) -> None:
            if trace_rows <= 0:
                return
            store = trace[kind].setdefault(m, {})
            store[index] = [
                tensor_sha256(tensor[b]) for b in range(min(trace_rows, tensor.shape[0]))
            ]

        def reset_round_state() -> None:
            for m in range(mb_count):
                meta[m][:, 0:1] = init_pending[m]
                meta[m][:, 1:2] = init_draft[m]
                meta[m][:, 2] = 1
                draft_stored[m].copy_(init_draft[m])
                accept_sum[m].zero_()
                row_accepts[m].zero_()
                accept_new[m].fill_(1)

        reset_round_state()

        def stage0_embed(lane: SpecLane, ids: torch.Tensor) -> None:
            hidden = torch.nn.functional.embedding(
                ids, embed_material.embed_weight
            )
            lane.sp.input_residual_buffer.copy_(
                hidden.unsqueeze(2).expand(-1, -1, HC_MULT, -1)
            )
            lane.sp.input_ids_buffer.copy_(ids)

        # ------------------------------------------------------------------
        # warmup + capture machinery
        capture_stream = torch.cuda.Stream(device=device)

        def run_warm_round(lane: SpecLane, *, use_graph_slots: bool) -> None:
            override = None if (use_graph_slots or not gate) else 0
            lane.sp.input_ids_buffer.copy_(init_pending[m_holder[0]])
            lane.sp.input_residual_buffer.normal_(0.0, 0.01)
            forward_spec_stage(
                lane.stage, lane.sp, pass_b=False, moe_slot_override=override
            )
            lane.sp.input_residual_buffer.normal_(0.0, 0.01)
            forward_spec_stage(
                lane.stage, lane.sp, pass_b=True, moe_slot_override=override
            )
            if lane.mtp_lane is not None:
                lane.mtp_plan.input_residual_buffer.normal_(0.0, 0.01)
                lane.mtp_plan.input_ids_buffer.copy_(init_pending[m_holder[0]])
                forward_mtp_spec(
                    lane.mtp_lane, lane.mtp_plan, lane.sp.positions,
                    second=False, moe_slot_override=override,
                )
                forward_mtp_spec(
                    lane.mtp_lane, lane.mtp_plan, lane.sp.positions,
                    second=True, moe_slot_override=override,
                )

        m_holder = [0]

        def warmup_all() -> None:
            for m, lane in enumerate(lanes):
                m_holder[0] = m
                lane_snapshot = snapshots[m] if gate else snapshot_lane(lane)
                # eager warm (slot 0), two rounds
                lane.sp.advance.zero_()
                lane.sp.accept.fill_(1)
                run_warm_round(lane, use_graph_slots=False)
                lane.sp.advance.fill_(2)
                run_warm_round(lane, use_graph_slots=False)
                torch.cuda.synchronize(device)
                restore_lane(lane, lane_snapshot)
                # capture-stream warm (lane graph slots), two rounds
                with torch.cuda.stream(capture_stream):
                    run_warm_round(lane, use_graph_slots=True)
                    lane.sp.advance.fill_(2)
                    run_warm_round(lane, use_graph_slots=True)
                torch.cuda.synchronize(device)
                restore_lane(lane, lane_snapshot)
                for slot in (lane.sp.moe_slot_a, lane.sp.moe_slot_b):
                    for moe in lane.stage.moes:
                        moe.reset_free_slot_completion_event(global_rows, slot)
                    if lane.mtp_lane is not None:
                        lane.mtp_lane.material.moe.reset_free_slot_completion_event(
                            global_rows, slot
                        )
                if not gate:
                    del lane_snapshot
                torch.cuda.empty_cache()
                if rank in (0, 12):
                    print(f"[E1MTPLB] stage {stage_id} mb{m} warm done", flush=True)
            if gate:
                for m, twin in twins.items():
                    m_holder[0] = m
                    twin.sp.advance.zero_()
                    twin.sp.accept.fill_(1)
                    run_warm_round(twin, use_graph_slots=False)
                    torch.cuda.synchronize(device)
                    restore_lane(twin, twin_snapshots[m])

        phase_started = time.perf_counter()
        synchronized_local_step("warmup", warmup_all, device=device, world=world)
        reset_round_state()
        result["diagnostic_seconds"]["warmup"] = time.perf_counter() - phase_started
        memory_snapshot("after_warmup")
        if rank in (0, 4, 8, 12):
            print(
                f"[E1MTPLB] stage {stage_id} warmup done, free "
                f"{result['memory']['after_warmup']['free_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )

        graph_pool = torch.cuda.graph_pool_handle()

        def capture_graph(fn) -> torch.cuda.CUDAGraph:
            current = torch.cuda.current_stream(device)
            graph = torch.cuda.CUDAGraph()
            capture_stream.wait_stream(current)
            with torch.cuda.graph(graph, stream=capture_stream, pool=graph_pool):
                fn()
            current.wait_stream(capture_stream)
            torch.cuda.synchronize(device)
            return graph

        def ensure_graph(lane: SpecLane, key: str, fn, label: str):
            registry = lane.mtp_graphs if key.startswith("mtp") else lane.graphs
            short = key.split("_")[-1]
            if short in registry:
                return registry[short], False
            t_cap = time.perf_counter()
            graph = synchronized_local_step(
                f"capture {lane.label} {key}",
                lambda: capture_graph(fn),
                device=device,
                world=EXPECTED_TP_SIZE,
                group=topo["tp_group"],
            )
            registry[short] = graph
            if rank in (0, 12):
                print(
                    f"[E1MTPLB] stage {stage_id} {label} captured {lane.label} "
                    f"{key} ({time.perf_counter() - t_cap:.1f}s)",
                    flush=True,
                )
            return graph, True

        # ------------------------------------------------------------------
        # OFF phase (gate only): eager family closed loop, serial per lane
        def run_off_phase(m: int) -> torch.Tensor | None:
            lane = off_lanes[m]
            schedule = build_decode_schedule(start_position, off_steps)
            token_buffer = init_pending[m].clone()
            records = None
            if stage_id == STAGE_COUNT - 1:
                records = torch.zeros(
                    (off_steps, local_batch), dtype=torch.int64, device=device
                )
            for step_index, step in enumerate(schedule):
                if stage_id == 0:
                    if step_index > 0:
                        pair_transfer(
                            token_buffer, send=False,
                            group=topo["loop_pair"], peer=1,
                        )
                    hidden = torch.nn.functional.embedding(
                        token_buffer, embed_material.embed_weight
                    )
                    lane.plan.input_residual_buffer.copy_(
                        hidden.unsqueeze(2).expand(-1, -1, HC_MULT, -1)
                    )
                    lane.plan.input_ids_buffer.copy_(token_buffer)
                else:
                    pair_transfer(
                        stagings[m], send=False, group=topo["prev_pair"], peer=0
                    )
                    lane.plan.input_residual_buffer.copy_(stagings[m])
                    lane.plan.input_ids_buffer.zero_()
                trace_record("off_in", m, step_index, lane.plan.input_residual_buffer)
                forward_eager_prevalidated(
                    lane.stage, lane.plan, graph_family=step.family
                )
                trace_record("off_out", m, step_index, lane.plan.output_buffer)
                lane.cursor.advance_host(step.family)
                if stage_id < STAGE_COUNT - 1:
                    pair_transfer(
                        lane.plan.output_buffer, send=True,
                        group=topo["next_pair"], peer=1,
                    )
                else:
                    logits = head_logits(head_material, lane.plan.output_buffer)
                    token_buffer.copy_(logits.argmax(dim=-1, keepdim=True))
                    records[step_index].copy_(token_buffer.view(-1))
                    pair_transfer(
                        token_buffer, send=True, group=topo["loop_pair"], peer=0
                    )
                if rank == 0 and step_index % 64 == 0:
                    print(
                        f"[E1MTPLB] off-mb{m} step {step_index}/{off_steps}",
                        flush=True,
                    )
            if stage_id == 0:
                pair_transfer(
                    token_buffer, send=False, group=topo["loop_pair"], peer=1
                )
            torch.cuda.synchronize(device)
            return records

        # ------------------------------------------------------------------
        # ON slot runner
        def run_slot(
            m: int,
            pass_b: bool,
            *,
            local_round: int,
            global_round: int,
            allow_capture: bool,
            label: str,
            timing: dict[str, list[float]] | None,
            twin_stats: dict[str, Any] | None,
            record: dict[str, torch.Tensor] | None,
        ) -> None:
            lane = lanes[m]
            sp = lane.sp
            t0 = time.perf_counter()
            if stage_id == 0:
                if not pass_b:
                    if local_round > 0:
                        pair_transfer(
                            meta[m], send=False, group=topo["loop_pair"], peer=1
                        )
                        torch.cuda.synchronize(device)
                    t1 = time.perf_counter()
                    if timing is not None:
                        timing["token_wait"].append((t1 - t0) * 1e3)
                    if global_round > 0:
                        sp.accept.copy_(meta[m][:, 2])
                        sp.advance.copy_(1 + meta[m][:, 2])
                    else:
                        sp.accept.fill_(1)
                        sp.advance.zero_()
                    accept_wire[m][:, 0].copy_(sp.accept)
                    stage0_embed(lane, meta[m][:, 0:1])
                else:
                    stage0_embed(lane, meta[m][:, 1:2])
                torch.cuda.synchronize(device)
                t2 = time.perf_counter()
                if timing is not None:
                    timing["embed"].append((t2 - t0) * 1e3)
            else:
                pair_transfer(stagings[m], send=False, group=topo["prev_pair"], peer=0)
                if not pass_b:
                    pair_transfer(
                        accept_wire[m], send=False, group=topo["prev_pair"], peer=0
                    )
                torch.cuda.synchronize(device)
                t2 = time.perf_counter()
                if timing is not None:
                    timing["recv"].append((t2 - t0) * 1e3)
                if not pass_b:
                    if global_round > 0:
                        sp.accept.copy_(accept_wire[m][:, 0])
                        sp.advance.copy_(1 + accept_wire[m][:, 0])
                    else:
                        sp.accept.fill_(1)
                        sp.advance.zero_()
                sp.input_residual_buffer.copy_(stagings[m])
                sp.input_ids_buffer.zero_()

            key = "b" if pass_b else "a"
            if not pass_b:
                trace_record("on_in", m, local_round, sp.input_residual_buffer)
            if key not in lane.graphs:
                if not allow_capture:
                    raise RuntimeError(f"{label}: lane mb{m} graph {key} missing")
                ensure_graph(
                    lane, key,
                    lambda: forward_spec_stage(lane.stage, sp, pass_b=pass_b),
                    label,
                )
            lane.graphs[key].replay()
            torch.cuda.synchronize(device)
            if not pass_b:
                trace_record("on_out", m, local_round, sp.output_buffer)
            t3 = time.perf_counter()
            if timing is not None:
                timing["replay"].append((t3 - t2) * 1e3)

            # graph-vs-eager twin (gate solo phases)
            if twin_stats is not None:
                twin = twins[m]
                twin.sp.accept.copy_(sp.accept)
                twin.sp.advance.copy_(sp.advance)
                twin.sp.input_residual_buffer.copy_(sp.input_residual_buffer)
                twin.sp.input_ids_buffer.copy_(sp.input_ids_buffer)
                forward_spec_stage(twin.stage, twin.sp, pass_b=pass_b)
                torch.cuda.synchronize(device)
                if torch.equal(sp.output_buffer, twin.sp.output_buffer):
                    twin_stats["bitwise_slots"] += 1
                else:
                    twin_stats["mismatched"].append(
                        {"round": local_round, "pass": key}
                    )
                if not pass_b:
                    # keep the twin advance idempotent: advance applied inside
                    # forward_spec_stage for both arms already
                    pass

            if stage_id < STAGE_COUNT - 1:
                pair_transfer(sp.output_buffer, send=True, group=topo["next_pair"], peer=1)
                if not pass_b:
                    pair_transfer(
                        accept_wire[m], send=True, group=topo["next_pair"], peer=1
                    )
                torch.cuda.synchronize(device)
                if timing is not None:
                    timing["send"].append((time.perf_counter() - t3) * 1e3)
            else:
                logits = head_logits(head_material, sp.output_buffer)
                tok = logits.argmax(dim=-1, keepdim=True)
                t4 = time.perf_counter()
                if timing is not None:
                    timing["head"].append((t4 - t3) * 1e3)
                mtp_plan = lane.mtp_plan
                if not pass_b:
                    t1_buf[m].copy_(tok)
                    accept_new[m].copy_(tok.eq(draft_stored[m]).to(torch.int64))
                    if args.accept_mode == "force_reject":
                        accept_new[m].zero_()
                    elif args.accept_mode == "split":
                        accept_new[m][: local_batch // 2].zero_()
                    mtp_plan.input_residual_buffer.copy_(sp.output_buffer)
                    mtp_plan.input_ids_buffer.copy_(tok)
                    if "a" not in lane.mtp_graphs:
                        if not allow_capture:
                            raise RuntimeError("mtp graph a missing")
                        ensure_graph(
                            lane, "mtp_a",
                            lambda: forward_mtp_spec(
                                lane.mtp_lane, mtp_plan, sp.positions, second=False
                            ),
                            label,
                        )
                    lane.mtp_graphs["a"].replay()
                    d1_buf[m].copy_(mtp_plan.draft_buffer)
                    torch.cuda.synchronize(device)
                    if timing is not None:
                        timing["mtp"].append((time.perf_counter() - t4) * 1e3)
                    if twin_stats is not None:
                        twin = twins[m]
                        twin.mtp_plan.input_residual_buffer.copy_(
                            mtp_plan.input_residual_buffer
                        )
                        twin.mtp_plan.input_ids_buffer.copy_(
                            mtp_plan.input_ids_buffer
                        )
                        forward_mtp_spec(
                            twin.mtp_lane, twin.mtp_plan, twin.sp.positions,
                            second=False,
                        )
                        torch.cuda.synchronize(device)
                        if not torch.equal(
                            mtp_plan.draft_buffer, twin.mtp_plan.draft_buffer
                        ):
                            twin_stats["mismatched"].append(
                                {"round": local_round, "pass": "mtp_a"}
                            )
                else:
                    mtp_plan.input_residual_buffer.copy_(sp.output_buffer)
                    mtp_plan.input_ids_buffer.copy_(tok)
                    if "b" not in lane.mtp_graphs:
                        if not allow_capture:
                            raise RuntimeError("mtp graph b missing")
                        ensure_graph(
                            lane, "mtp_b",
                            lambda: forward_mtp_spec(
                                lane.mtp_lane, mtp_plan, sp.positions, second=True
                            ),
                            label,
                        )
                    lane.mtp_graphs["b"].replay()
                    torch.cuda.synchronize(device)
                    t5 = time.perf_counter()
                    if timing is not None:
                        timing["mtp"].append((t5 - t4) * 1e3)
                    if twin_stats is not None:
                        twin = twins[m]
                        twin.mtp_plan.input_residual_buffer.copy_(
                            mtp_plan.input_residual_buffer
                        )
                        twin.mtp_plan.input_ids_buffer.copy_(
                            mtp_plan.input_ids_buffer
                        )
                        forward_mtp_spec(
                            twin.mtp_lane, twin.mtp_plan, twin.sp.positions,
                            second=True,
                        )
                        torch.cuda.synchronize(device)
                        if not torch.equal(
                            mtp_plan.draft_buffer, twin.mtp_plan.draft_buffer
                        ):
                            twin_stats["mismatched"].append(
                                {"round": local_round, "pass": "mtp_b"}
                            )
                    acc = accept_new[m]
                    pending_next = torch.where(acc.bool(), tok, t1_buf[m])
                    draft_next = torch.where(
                        acc.view(-1).bool(), mtp_plan.draft_buffer, d1_buf[m]
                    )
                    meta[m][:, 0:1] = pending_next
                    meta[m][:, 1] = draft_next
                    meta[m][:, 2:3] = acc
                    draft_stored[m].copy_(draft_next.view(-1, 1))
                    accept_sum[m].add_(acc.sum())
                    row_accepts[m].add_(acc.view(-1))
                    if record is not None:
                        record["t1"][local_round].copy_(t1_buf[m].view(-1))
                        record["t2"][local_round].copy_(tok.view(-1))
                        record["acc"][local_round].copy_(acc.view(-1))
                    pair_transfer(meta[m], send=True, group=topo["loop_pair"], peer=0)
                    torch.cuda.synchronize(device)
                    if timing is not None:
                        timing["meta_send"].append(
                            (time.perf_counter() - t5) * 1e3
                        )
            if timing is not None:
                timing["slot_wall"].append((time.perf_counter() - t0) * 1e3)

        def run_on_phase(
            *,
            label: str,
            lane_indices: Sequence[int],
            rounds: int,
            allow_capture: bool,
            timing: dict[str, list[float]] | None = None,
            twin_stats_map: dict[int, dict[str, Any]] | None = None,
            records: dict[int, dict[str, torch.Tensor]] | None = None,
            round_offset: int = 0,
        ) -> None:
            for local_round in range(rounds):
                for m in lane_indices:
                    for pass_b in (False, True):
                        run_slot(
                            m, pass_b,
                            local_round=local_round,
                            global_round=round_offset + local_round,
                            allow_capture=allow_capture,
                            label=label,
                            timing=timing,
                            twin_stats=(
                                None
                                if twin_stats_map is None
                                else twin_stats_map.get(m)
                            ),
                            record=None if records is None else records.get(m),
                        )
                if rank == 0 and local_round % 32 == 0:
                    print(
                        f"[E1MTPLB] {label} round {local_round}/{rounds}",
                        flush=True,
                    )
            # drain the final metas on stage 0
            if stage_id == 0:
                for m in lane_indices:
                    pair_transfer(meta[m], send=False, group=topo["loop_pair"], peer=1)
            torch.cuda.synchronize(device)

        # ------------------------------------------------------------------
        if gate:
            gate_record: dict[str, Any] = {
                "rounds": gate_rounds,
                "off_steps": off_steps,
                "judgment": (
                    "per lane: OFF eager-family solo stream (2R steps) -> "
                    "ON solo chained rounds (graphs, per-pass bitwise "
                    "graph-vs-eager twin incl. MTP) -> restore -> ON "
                    "interleaved; per-row: ON emitted stream must be an "
                    "exact prefix of the OFF stream; interleaved streams/"
                    "states must equal solo (no cross-talk)"
                ),
                "lanes": {},
            }
            solo_records: dict[int, dict[str, torch.Tensor]] = {}
            stage_traces: dict[int, Any] = {}
            solo_digests: dict[int, dict[str, str]] = {}
            solo_positions: dict[int, list[int]] = {}
            off_records: dict[int, torch.Tensor] = {}

            # OFF phase
            phase_started = time.perf_counter()
            for m in range(mb_count):
                synchronized_local_step(
                    f"off mb{m} entry", lambda: None, device=device, world=world
                )
                off_records[m] = run_off_phase(m)
                if rank in (0, 12):
                    print(f"[E1MTPLB] off-mb{m} done", flush=True)
            result["diagnostic_seconds"]["off_phase"] = (
                time.perf_counter() - phase_started
            )

            # ON solo phases
            phase_started = time.perf_counter()
            for m in range(mb_count):
                twin_stats = {"bitwise_slots": 0, "mismatched": []}
                record = None
                if stage_id == STAGE_COUNT - 1:
                    record = {
                        "t1": torch.zeros(
                            (gate_rounds, local_batch), dtype=torch.int64,
                            device=device,
                        ),
                        "t2": torch.zeros(
                            (gate_rounds, local_batch), dtype=torch.int64,
                            device=device,
                        ),
                        "acc": torch.zeros(
                            (gate_rounds, local_batch), dtype=torch.int64,
                            device=device,
                        ),
                    }
                synchronized_local_step(
                    f"on-solo mb{m} entry", lambda: None, device=device, world=world
                )
                run_on_phase(
                    label=f"on-solo-mb{m}",
                    lane_indices=[m],
                    rounds=gate_rounds,
                    allow_capture=True,
                    twin_stats_map={m: twin_stats},
                    records=None if record is None else {m: record},
                )
                if trace_rows > 0:
                    compare = {"first_input_mismatch": None, "first_output_mismatch": None}
                    on_in = trace["on_in"].get(m, {})
                    on_out = trace["on_out"].get(m, {})
                    off_in = trace["off_in"].get(m, {})
                    off_out = trace["off_out"].get(m, {})
                    for r_i in sorted(on_out):
                        if r_i not in off_out:
                            break
                        for b in range(trace_rows):
                            if (
                                compare["first_input_mismatch"] is None
                                and on_in[r_i][b] != off_in[r_i][b]
                            ):
                                compare["first_input_mismatch"] = {
                                    "round": r_i, "row": b,
                                }
                            if (
                                compare["first_output_mismatch"] is None
                                and on_out[r_i][b] != off_out[r_i][b]
                            ):
                                compare["first_output_mismatch"] = {
                                    "round": r_i, "row": b,
                                }
                        if (
                            compare["first_input_mismatch"] is not None
                            and compare["first_output_mismatch"] is not None
                        ):
                            break
                    stage_traces[m] = compare
                    trace["on_in"].pop(m, None)
                    trace["on_out"].pop(m, None)
                solo_records[m] = record
                solo_digests[m] = lanes[m].state_digests()
                solo_positions[m] = [
                    int(v) for v in lanes[m].sp.positions.cpu().tolist()
                ]
                lane_record: dict[str, Any] = {
                    "stage_trace": stage_traces.get(m),
                    "twin_bitwise_slots": twin_stats["bitwise_slots"],
                    "twin_mismatched": twin_stats["mismatched"][:16],
                    "twin_ok": not twin_stats["mismatched"],
                    "solo_positions": solo_positions[m],
                }
                # twin final-state check
                lane_record["twin_final_state_equal"] = bool(
                    solo_digests[m] == twins[m].state_digests()
                )
                gate_record["lanes"][str(m)] = lane_record
                # restore lane + twin + round buffers
                restore_lane(lanes[m], snapshots[m])
                restore_lane(twins[m], twin_snapshots[m])
                reset_round_state()
                if rank in (0, 12):
                    print(
                        f"[E1MTPLB] on-solo-mb{m} done: twin bitwise "
                        f"{twin_stats['bitwise_slots']}, mismatches "
                        f"{len(twin_stats['mismatched'])}",
                        flush=True,
                    )
            result["diagnostic_seconds"]["on_solo"] = (
                time.perf_counter() - phase_started
            )
            memory_snapshot("after_solo")

            # ON interleaved
            phase_started = time.perf_counter()
            inter_records: dict[int, dict[str, torch.Tensor]] = {}
            if stage_id == STAGE_COUNT - 1:
                for m in range(mb_count):
                    inter_records[m] = {
                        "t1": torch.zeros(
                            (gate_rounds, local_batch), dtype=torch.int64,
                            device=device,
                        ),
                        "t2": torch.zeros(
                            (gate_rounds, local_batch), dtype=torch.int64,
                            device=device,
                        ),
                        "acc": torch.zeros(
                            (gate_rounds, local_batch), dtype=torch.int64,
                            device=device,
                        ),
                    }
            synchronized_local_step(
                "interleaved entry", lambda: None, device=device, world=world
            )
            run_on_phase(
                label="on-interleaved",
                lane_indices=list(range(mb_count)),
                rounds=gate_rounds,
                allow_capture=False,
                records=inter_records if inter_records else None,
            )
            result["diagnostic_seconds"]["on_interleaved"] = (
                time.perf_counter() - phase_started
            )

            all_ok = True
            for m in range(mb_count):
                lane_record = gate_record["lanes"][str(m)]
                inter_positions = [
                    int(v) for v in lanes[m].sp.positions.cpu().tolist()
                ]
                lane_record["interleaved_positions"] = inter_positions
                lane_record["positions_equal_solo_vs_interleaved"] = bool(
                    inter_positions == solo_positions[m]
                )
                lane_record["state_equal_solo_vs_interleaved"] = bool(
                    lanes[m].state_digests() == solo_digests[m]
                )
                if stage_id == STAGE_COUNT - 1:
                    solo = solo_records[m]
                    inter = inter_records[m]
                    lane_record["records_equal_solo_vs_interleaved"] = bool(
                        torch.equal(solo["t1"], inter["t1"])
                        and torch.equal(solo["t2"], inter["t2"])
                        and torch.equal(solo["acc"], inter["acc"])
                    )
                    # per-row prefix check vs OFF
                    off = off_records[m].cpu()
                    t1s = solo["t1"].cpu()
                    t2s = solo["t2"].cpu()
                    accs = solo["acc"].cpu()
                    prefix_rows_ok = 0
                    first_mismatch = None
                    accepts_per_row = []
                    for b in range(local_batch):
                        stream = []
                        for r in range(gate_rounds):
                            stream.append(int(t1s[r, b]))
                            if int(accs[r, b]):
                                stream.append(int(t2s[r, b]))
                        accepts_per_row.append(int(accs[:, b].sum()))
                        reference = [int(v) for v in off[: len(stream), b]]
                        if stream == reference:
                            prefix_rows_ok += 1
                        elif first_mismatch is None:
                            diverge = next(
                                (
                                    i
                                    for i, (x, y) in enumerate(
                                        zip(stream, reference)
                                    )
                                    if x != y
                                ),
                                -1,
                            )
                            first_mismatch = {
                                "row": b,
                                "first_diverging_index": diverge,
                                "stream_len": len(stream),
                            }
                    lane_record["prefix_rows_ok"] = prefix_rows_ok
                    lane_record["rows"] = local_batch
                    lane_record["first_mismatch"] = first_mismatch
                    head_rows = min(local_batch, 4)
                    head_rounds = min(gate_rounds, 24)
                    lane_record["debug_off_head"] = [
                        [int(v) for v in off[: 2 * head_rounds, b]]
                        for b in range(head_rows)
                    ]
                    lane_record["debug_on_head"] = [
                        {
                            "t1": [int(t1s[r, b]) for r in range(head_rounds)],
                            "t2": [int(t2s[r, b]) for r in range(head_rounds)],
                            "acc": [int(accs[r, b]) for r in range(head_rounds)],
                        }
                        for b in range(head_rows)
                    ]
                    lane_record["accepts_per_row"] = accepts_per_row
                    lane_record["alpha_measured"] = float(
                        sum(accepts_per_row) / (gate_rounds * local_batch)
                    )
                    lane_record["prefix_ok"] = bool(
                        prefix_rows_ok == local_batch
                    )
                else:
                    lane_record["records_equal_solo_vs_interleaved"] = None
                    lane_record["prefix_ok"] = None
                lane_ok = bool(
                    lane_record["twin_ok"]
                    and lane_record["twin_final_state_equal"]
                    and lane_record["positions_equal_solo_vs_interleaved"]
                    and lane_record["state_equal_solo_vs_interleaved"]
                    and lane_record["records_equal_solo_vs_interleaved"]
                    in (True, None)
                    and lane_record["prefix_ok"] in (True, None)
                )
                lane_record["accepted"] = lane_ok
                all_ok = all_ok and lane_ok
            gate_record["accepted"] = all_ok
            result["gate_record"] = gate_record
            memory_snapshot("after_interleaved")
            if rank == 0:
                print(
                    f"[E1MTPLB] gate: {'PASS' if all_ok else 'FAIL'} on this rank",
                    flush=True,
                )
        else:
            # --------------------------------------------------------------
            # timed: settle (capture) + timed segments
            phase_started = time.perf_counter()
            synchronized_local_step(
                "settle entry", lambda: None, device=device, world=world
            )
            run_on_phase(
                label="settle",
                lane_indices=list(range(mb_count)),
                rounds=settle_rounds,
                allow_capture=True,
            )
            result["diagnostic_seconds"]["settle"] = (
                time.perf_counter() - phase_started
            )
            memory_snapshot("after_settle")
            if rank == 0:
                print(
                    f"[E1MTPLB] settle done "
                    f"({result['diagnostic_seconds']['settle']:.0f}s)",
                    flush=True,
                )
            round_offset = settle_rounds
            accept_base = {
                m: int(accept_sum[m].item()) for m in range(mb_count)
            }
            for segment in range(segments):
                synchronized_local_step(
                    f"segment {segment} entry", lambda: None,
                    device=device, world=world,
                )
                timing: dict[str, list[float]] = {
                    key: []
                    for key in (
                        "token_wait", "embed", "recv", "replay", "send",
                        "head", "mtp", "meta_send", "slot_wall",
                    )
                }
                segment_started = time.perf_counter()
                run_on_phase(
                    label=f"segment{segment}",
                    lane_indices=list(range(mb_count)),
                    rounds=rounds_per_segment,
                    allow_capture=False,
                    timing=timing,
                    round_offset=round_offset,
                )
                wall = time.perf_counter() - segment_started
                round_offset += rounds_per_segment
                accepts_segment = 0
                if stage_id == STAGE_COUNT - 1:
                    totals = {
                        m: int(accept_sum[m].item()) for m in range(mb_count)
                    }
                    accepts_segment = sum(
                        totals[m] - accept_base[m] for m in range(mb_count)
                    )
                    accept_base = totals
                base_tokens = local_batch * mb_count * rounds_per_segment
                record = {
                    "segment": segment,
                    "rounds": rounds_per_segment,
                    "wall_s": wall,
                    "round_wall_ms": wall * 1e3 / rounds_per_segment,
                    "accepts_local": accepts_segment,
                    "alpha_local": (
                        accepts_segment / base_tokens
                        if stage_id == STAGE_COUNT - 1
                        else None
                    ),
                    "timing_ms": {
                        key: summarize_ms(values)
                        for key, values in timing.items()
                        if values
                    },
                    "slot_wall_raw_ms": [
                        round(v, 4) for v in timing["slot_wall"]
                    ],
                }
                result["round_results"].append(record)
                if rank in (0, 12):
                    print(
                        f"[E1MTPLB] stage {stage_id} segment {segment}: wall "
                        f"{wall:.2f}s, round {record['round_wall_ms']:.1f} ms, "
                        f"alpha_local {record['alpha_local']}",
                        flush=True,
                    )
                memory_snapshot(f"after_segment_{segment}")

        # ------------------------------------------------------------------
        # terminals + teardown
        result["positions_final"] = {
            str(m): [int(v) for v in lanes[m].sp.positions.cpu().tolist()]
            for m in range(mb_count)
        }
        result["row_accepts"] = (
            {
                str(m): [int(v) for v in row_accepts[m].cpu().tolist()]
                for m in range(mb_count)
            }
            if stage_id == STAGE_COUNT - 1
            else None
        )
        teardown_record: dict[str, Any] = {"errors": []}
        try:
            torch.cuda.synchronize(device)
            for lane in lanes:
                for graph in list(lane.graphs.values()) + list(
                    lane.mtp_graphs.values()
                ):
                    graph.reset()
                lane.graphs.clear()
                lane.mtp_graphs.clear()
            torch.cuda.synchronize(device)
            for lane in lanes:
                for slot in (lane.sp.moe_slot_a, lane.sp.moe_slot_b):
                    for moe in lane.stage.moes:
                        moe.reset_free_slot_completion_event(global_rows, slot)
                    if lane.mtp_lane is not None:
                        lane.mtp_lane.material.moe.reset_free_slot_completion_event(
                            global_rows, slot
                        )
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            teardown_record["accepted"] = True
        except Exception:
            teardown_record["errors"].append(traceback.format_exc())
            teardown_record["accepted"] = False
        result["teardown"] = teardown_record
        memory_snapshot("at_end")

        phase_accepted = (
            bool(result["gate_record"] and result["gate_record"]["accepted"])
            if gate
            else bool(len(result["round_results"]) == segments)
        )
        result["accepted"] = bool(
            result["placement"]["accepted"]
            and phase_accepted
            and teardown_record["accepted"]
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
            isinstance(record, dict) and record.get("accepted")
            for record in gathered
        )
    )
    # cross-stage positions consistency (rank 0 arbitration)
    positions_consistent = None
    if len(gathered) == world and all(
        isinstance(record, dict) and record.get("positions_final")
        for record in gathered
    ):
        positions_consistent = True
        for tp in range(EXPECTED_TP_SIZE):
            reference = None
            for record in gathered:
                if record.get("tp_rank") != tp:
                    continue
                value = record["positions_final"]
                if reference is None:
                    reference = value
                elif value != reference:
                    positions_consistent = False
    accepted_all = bool(accepted_all and positions_consistent in (True, None))

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
                    if key != "slot_wall_raw_ms"
                }
                for round_record in record.get("round_results", [])
            ]
            merged.append(trimmed)
        write_json(
            out_dir / "result.json",
            {
                "experiment": "E1F-full-decode-throughput/mtp-largeb",
                "accepted": accepted_all,
                "positions_consistent_across_stages": positions_consistent,
                "kv_dtype": args.kv_dtype,
                "indexer_kv_dtype": args.indexer_kv_dtype,
                "mb_count": mb_count,
                "local_batch": local_batch,
                "global_batch": total_global,
                "check_mode": args.check_mode,
                "stage_representatives": merged,
            },
        )
        print(f"[E1MTPLB] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


def deterministic_bf16(seed: int, shape: tuple[int, ...], *, scale: float):
    generator = torch.Generator(device="cpu").manual_seed(seed & ((1 << 62) - 1))
    value = torch.randn(*shape, generator=generator, dtype=torch.float32) * scale
    return value.to(torch.bfloat16)


if __name__ == "__main__":
    raise SystemExit(main())
