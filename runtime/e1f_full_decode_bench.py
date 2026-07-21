#!/usr/bin/env python3
"""E1F: full-config TP4xPP4 stateful-CUDA-graph closed-loop decode bench.

First performance vertical.  Composition of three verified mechanisms:

- **E0sf/E0hf stateful CUDA graphs**: each of the four PP stages runs its
  11 (stage 3: 10) layers as one stateful super-stage; the three decode
  graph families (normal / ratio4_boundary / ratio4_ratio128_boundary) are
  lazily captured per stage (E0sf capture-at-first-occurrence, E1a27
  warmup/restore lifecycle) and replayed on the hot path.  The fused
  TileLang HC boundary backend (E0hf) is on by default.
- **E0qf serial cross-machine handoff**: fixed-endpoint NCCL P2P
  (``SerialPairHandoff``) on per-TP-rank boundary pair groups; boundary
  1->2 crosses the IB fabric (no-GDR config).
- **e0ef2e full-model topology**: stage 0 = titan064 GPU0-3 (L0-L10 +
  embedding), stage 1 = titan064 GPU4-7 (L11-L21), stage 2 = titan065
  GPU0-3 (L22-L32), stage 3 = titan065 GPU4-7 (L33-L42 + hc_head collapse
  + final norm + head).

**Closed loop**: every decode step embeds the previous step's argmax token
(stage 3 head -> argmax -> NCCL P2P loopback pair (tp, 12+tp) -> stage 0
embedding), so tokens really flow around the ring.  Embedding and head stay
eager (outside the graphs).

**B semantics** (``--b-semantics``):

- ``replicated`` (original E1F caliber -- must not be compared with C1F or
  DP-attention numbers): the same B sequences are replicated on all four TP
  ranks of every stage.  Each GPU holds B sequences of KV (bl = B); the MoE
  gathers 4B global rows that are four identical copies, so the *distinct*
  global batch is B and throughput = B / step_wall.  Per-rank compute at
  bl=B is identical to a DP-attention deployment at global batch 4B, so a
  DP-caliber throughput estimate is 4x the replicated number (model-derived,
  reported separately, never mixed).
- ``dp`` (twelfth vertical, E0dpf-gated): true DP-attention sequence split.
  TP rank r of every stage serves its **own** ``local_batch`` sequences
  (global rows ``[r*bl, (r+1)*bl)`` of ``B_global = 4*bl``), full 64 heads,
  KV bl per GPU.  The MoE path is unchanged -- the in-package ``TP4MoE``
  all_gathers the four distinct row blocks into ``B_global`` rows, computes
  the itp partial, and reduce_scatters each rank's own rows back -- so the
  gathered rows are now all distinct and throughput = B_global / step_wall
  is a *measured* DP number (no 4x conversion).  KV seeds, initial tokens,
  and warm inputs are global-batch tensors sliced per rank, so the served
  sequence set equals the replicated caliber's at local_batch = B_global.
  Cross-lane output digests are expected to differ (distinct sequences).

**KV caliber**: seeded decode residency (E1a27
deterministic-seeded-KV-not-real-prefix) at ``--start-position`` (default
2048; ratio-4 index saturation needs >= 2047, the E0hf perf-mode memory
anchor).  ``MAX_SEQ_LEN = start + settle + rounds*steps`` rounded up to a
multiple of 128; state clearly in any report.

**Phases** per process:
1. load + build lanes + seeded residency + warmup/restore (E0hf pattern).
2. settle segment (default 132 steps, covers all three families): pipeline
   steps with lazy graph capture; with ``--check-mode bitwise`` an eager
   twin lane (same fused chain run eagerly, E0hf ``forward_eager_
   prevalidated``) consumes identical inputs and every step is compared
   bitwise (graph-vs-eager, per rank), plus end-of-segment full KV digest
   parity.
3. timed rounds (default 3 x 300 steps): replay-only hot path with the
   E0qf serial timing decomposition (per-stage replay wall, sender-observed
   handoff wall, receiver recv wall, embed/head/token-loop wall, per-step
   wall).  Every wall is host time around ``torch.cuda.synchronize`` --
   the serial-decomposition caliber, stated in results.
4. terminals, cross-lane digests, graph teardown, memory audit.

Run (from ``run_e1f_dual.sh``): torchrun --nnodes 2 --nproc-per-node 8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.dp_caliber import (
    dp_row_slice,
    dp_slice_ratio4_oracle_state,
    oracle_state_to_device,
)
from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.mode_witness import collect_attention_modes
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
from dsv4_direct.ratio4_oracle import seed_nonzero_ratio4_state
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
from dsv4_direct.static_kv import StaticLayerKV
from dsv4_direct.static_ratio4_kv import StaticRatio4KV
from dsv4_direct.static_window_kv import StaticWindowKV
from dsv4_direct.superstage import (
    TP4DecodeStage,
    TP4StatefulDecodeSuperStagePlan,
)


WORLD = 16
STAGE_COUNT = 4
MODEL_LAYERS = 43
STAGE_LAYERS: dict[int, tuple[int, ...]] = {
    0: tuple(range(0, 11)),
    1: tuple(range(11, 22)),
    2: tuple(range(22, 33)),
    3: tuple(range(33, 43)),
}
HIDDEN = 4096
HC_MULT = 4
EXPECTED_VOCAB = 129280

EAGER_MOE_SLOT = 0
GRAPH_MOE_SLOTS: dict[DecodeGraphFamily, int] = {
    DecodeGraphFamily.NORMAL: 1,
    DecodeGraphFamily.RATIO4_BOUNDARY: 2,
    DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY: 3,
}
GRAPH_MOE_SLOT_TUPLE = tuple(GRAPH_MOE_SLOTS[family] for family in DecodeGraphFamily)


# --------------------------------------------------------------------------
# generic helpers (E0sf/E0qf process forms)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    metadata = f"{list(tensor.shape)}|{tensor.dtype}|".encode("utf-8")
    return hashlib.sha256(metadata + value).hexdigest()


def deterministic_tensor(
    *, seed: int, shape: tuple[int, ...], device: torch.device, scale: float = 0.02
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32)
    return (value * scale).to(torch.bfloat16).to(device)


def summarize_ms(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": float(count),
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[count // 2],
        "p95_ms": ordered[min(count - 1, int(round(0.95 * (count - 1))))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def synchronized_local_step(
    name: str,
    fn: Any,
    *,
    device: torch.device,
    world: int,
    group: Any = None,
) -> Any:
    """Run ``fn`` locally, then reach consensus on ``group`` (default world).

    Graph capture must pass the TP stage group: in the serial pipeline the
    four stages reach their capture points at different wall times, so a
    world-wide consensus there would deadlock against the boundary P2P.
    """

    value: Any = None
    local_error: str | None = None
    try:
        value = fn()
    except Exception:
        local_error = traceback.format_exc()
        print(
            f"[E1F][rank {dist.get_rank()}] {name} raised:\n{local_error}",
            flush=True,
        )
    failed = torch.tensor(int(local_error is not None), device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    if failed.item():
        errors: list[str | None] = [None for _ in range(world)]
        dist.all_gather_object(errors, local_error, group=group)
        details = "\n".join(
            f"rank {rank}:\n{error}" for rank, error in enumerate(errors) if error
        )
        raise RuntimeError(f"{name} failed before the next collective:\n{details}")
    return value


DirectState = StaticWindowKV | StaticRatio4KV | StaticLayerKV


def clone_state(source: DirectState) -> DirectState:
    if isinstance(source, StaticWindowKV):
        result: DirectState = StaticWindowKV(
            num_local_sequences=source.num_local_sequences,
            max_seq_len=source.max_seq_len,
            layer_id=source.layer_id,
            device=source.device,
            kv_dtype=source.kv_dtype,
        )
    elif isinstance(source, StaticRatio4KV):
        result = StaticRatio4KV(
            num_local_sequences=source.num_local_sequences,
            max_seq_len=source.max_seq_len,
            layer_id=source.layer_id,
            device=source.device,
            kv_dtype=source.kv_dtype,
            indexer_dtype=source.indexer_dtype,
        )
    elif isinstance(source, StaticLayerKV):
        result = StaticLayerKV(
            num_local_sequences=source.num_local_sequences,
            max_seq_len=source.max_seq_len,
            layer_id=source.layer_id,
            device=source.device,
            kv_dtype=source.kv_dtype,
        )
    else:
        raise TypeError("unsupported direct state type")
    result.copy_from(source)  # type: ignore[arg-type]
    return result


def copy_stage_states(
    destination: Sequence[DirectState], source: Sequence[DirectState]
) -> None:
    if len(destination) != len(source):
        raise ValueError("state sets differ in length")
    for destination_state, source_state in zip(destination, source, strict=True):
        if type(destination_state) is not type(source_state):
            raise TypeError("state type differs during restore")
        destination_state.copy_from(source_state)  # type: ignore[arg-type]


def full_state_sha256(state: DirectState) -> str:
    digest = hashlib.sha256()
    for name, tensor in state._owned_tensor_items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# topology (e0ef2e PP4 groups + token loopback pairs)


def create_e1f_topology(rank: int) -> dict[str, Any]:
    specs: list[tuple[int, ...]] = [
        tuple(range(stage * 4, stage * 4 + 4)) for stage in range(STAGE_COUNT)
    ]
    for boundary in range(STAGE_COUNT - 1):
        for tp in range(EXPECTED_TP_SIZE):
            specs.append((boundary * 4 + tp, (boundary + 1) * 4 + tp))
    for tp in range(EXPECTED_TP_SIZE):
        specs.append((tp, 3 * 4 + tp))  # token loopback stage3 -> stage0
    groups = [
        dist.new_group(ranks=list(spec), backend="nccl", timeout=timedelta(minutes=60))
        for spec in specs
    ]
    stage = rank // EXPECTED_TP_SIZE
    tp_rank = rank % EXPECTED_TP_SIZE
    prev_pair = groups[STAGE_COUNT + (stage - 1) * 4 + tp_rank] if stage > 0 else None
    next_pair = (
        groups[STAGE_COUNT + stage * 4 + tp_rank] if stage < STAGE_COUNT - 1 else None
    )
    loop_pair = (
        groups[STAGE_COUNT + 12 + tp_rank] if stage in (0, STAGE_COUNT - 1) else None
    )
    return {
        "stage": stage,
        "tp_rank": tp_rank,
        "tp_group": groups[stage],
        "tp_global_ranks": specs[stage],
        "prev_pair": prev_pair,
        "next_pair": next_pair,
        "loop_pair": loop_pair,
        "all_groups": groups,
    }


def pair_transfer(tensor: torch.Tensor, *, send: bool, group: Any, peer: int) -> None:
    """One fixed-shape send/recv on a 2-rank group (e0ef2e P2P form)."""

    if not tensor.is_contiguous():
        raise ValueError("pair transfer requires a contiguous tensor")
    operation = dist.isend if send else dist.irecv
    works = dist.batch_isend_irecv(
        [dist.P2POp(operation, tensor, group=group, group_peer=peer)]
    )
    works[0].wait()


def run_placement_check(*, stage: int, world: int) -> dict[str, Any]:
    record = {
        "rank": dist.get_rank(),
        "stage": stage,
        "host": platform.node(),
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
    }
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, record)
    stage_hosts: dict[int, set[str]] = {s: set() for s in range(STAGE_COUNT)}
    for entry in gathered:
        stage_hosts[entry["stage"]].add(entry["host"])
    hosts = {s: sorted(stage_hosts[s]) for s in range(STAGE_COUNT)}
    accepted = (
        all(len(stage_hosts[s]) == 1 for s in range(STAGE_COUNT))
        and stage_hosts[0] == stage_hosts[1]
        and stage_hosts[2] == stage_hosts[3]
        and stage_hosts[0] != stage_hosts[2]
    )
    return {"stage_hosts": hosts, "accepted": bool(accepted)}


# --------------------------------------------------------------------------
# seeding (E0qf layer-type seeders, keyed by layer only).  Replicated
# caliber: the payload batch is local_batch and identical across TP ranks
# and lanes.  DP caliber: the payload is generated at the *global* batch and
# each rank keeps its own rows, so rank r's sequences are byte-identical to
# rows [r*bl, (r+1)*bl) of a replicated run at local_batch = 4*bl.


def build_seed_payload(
    material: PhysicalLayerMaterial,
    *,
    seed: int,
    local_batch: int,
    start_position: int,
    device: torch.device,
    dp_tp_rank: int | None = None,
) -> dict[str, Any]:
    layer_seed = (seed * 9_176_501 + material.layer_id * 15_485_863) & ((1 << 62) - 1)
    seed_batch = (
        local_batch if dp_tp_rank is None else local_batch * EXPECTED_TP_SIZE
    )
    # DP: generate at the global batch on the **CPU** (the generator lives
    # there anyway, so values are bitwise identical), slice this rank's
    # rows, and move only the slice -- the global ratio-4 oracle's qdq
    # temporaries at B_global do not fit next to the resident weights.
    build_device = torch.device("cpu") if dp_tp_rank is not None else device

    def rows(value: torch.Tensor) -> torch.Tensor:
        if dp_tp_rank is None:
            return value
        return dp_row_slice(value, dp_tp_rank, local_batch).to(device)

    if material.kind == "window":
        return {
            "raw": rows(
                deterministic_tensor(
                    seed=layer_seed,
                    shape=(seed_batch, 128, 512),
                    device=build_device,
                    scale=0.03,
                )
            )
        }
    if material.kind == "ratio128":
        return {
            "raw": rows(
                deterministic_tensor(
                    seed=layer_seed,
                    shape=(seed_batch, 128, 512),
                    device=build_device,
                    scale=0.03,
                )
            ),
            "compressed": rows(
                deterministic_tensor(
                    seed=layer_seed + 1,
                    shape=(seed_batch, start_position // 128, 512),
                    device=build_device,
                    scale=0.025,
                )
            ),
        }
    oracle_state = seed_nonzero_ratio4_state(
        material.attention_config,
        batch_size=seed_batch,
        start_pos=start_position,
        main_ape=material.prepared.compressor_ape.to(build_device),
        index_ape=material.prepared.index_compressor_ape.to(build_device),
        seed=layer_seed,
        device=build_device,
    )
    if dp_tp_rank is not None:
        oracle_state = oracle_state_to_device(
            dp_slice_ratio4_oracle_state(oracle_state, dp_tp_rank, local_batch),
            device,
        )
    return {"oracle": oracle_state}


def seed_state(
    material: PhysicalLayerMaterial,
    state: DirectState,
    payload: dict[str, Any],
    *,
    start_position: int,
) -> None:
    if material.kind == "window":
        assert isinstance(state, StaticWindowKV)
        state.seed_decode_residency(start_pos=start_position, raw=payload["raw"].clone())
    elif material.kind == "ratio128":
        assert isinstance(state, StaticLayerKV)
        state.seed_decode_residency(
            start_pos=start_position,
            raw=payload["raw"].clone(),
            compressed=payload["compressed"].clone(),
        )
    else:
        assert isinstance(state, StaticRatio4KV)
        oracle = payload["oracle"]
        state.seed_decode_payload(
            oracle.next_position,
            raw=oracle.raw.clone(),
            compressed=oracle.compressed.clone(),
            indexer_kv=oracle.indexer_kv.clone(),
            main_kv_state=oracle.main_kv.clone(),
            main_score_state=oracle.main_score.clone(),
            index_kv_state=oracle.index_kv.clone(),
            index_score_state=oracle.index_score.clone(),
        )


class StageLane:
    """One lane: seeded blocks, super-stage, cursor, stateful plan."""

    def __init__(
        self,
        *,
        label: str,
        materials: Sequence[PhysicalLayerMaterial],
        payloads: Mapping[int, dict[str, Any]],
        backend: Any | None,
        local_batch: int,
        start_position: int,
        stop_position: int,
        device: torch.device,
    ) -> None:
        self.label = label
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
            graph_moe_slots=GRAPH_MOE_SLOT_TUPLE,
        )

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


# --------------------------------------------------------------------------
# eager stateful body (E0hf backend-aware form; eager MoE slot 0)


def forward_eager_prevalidated(
    stage: TP4DecodeStage,
    plan: TP4StatefulDecodeSuperStagePlan,
    *,
    graph_family: DecodeGraphFamily,
    moe_slot: int = EAGER_MOE_SLOT,
) -> torch.Tensor:
    plan.cursor.guard_device_preflight(
        graph_family,
        expected_position=plan.expected_position,
        stop_position=plan.stop_position_tensor,
        stop_position_constant=plan.stop_position,
        state_positions=plan.state_position_tensors,
    )
    if stage.hc_boundary_backend is not None:
        output = stage._forward_stateful_fused_chain(
            plan.input_residual_buffer,
            input_ids_local=plan.input_ids_buffer,
            plan=plan,
            graph_family=graph_family,
            moe_slot=moe_slot,
        )
    else:
        output = plan.input_residual_buffer
        for block, layer_plan in zip(stage.blocks, plan.layer_plans, strict=True):
            output = block.forward_stateful_decode_tensor(
                output,
                input_ids_local=(
                    plan.input_ids_buffer if block.route_kind == "hash" else None
                ),
                attention_plan=layer_plan,
                graph_family=graph_family,
                moe_slot=moe_slot,
            )
    plan.output_buffer.copy_(output)
    plan.cursor.advance_device(
        graph_family,
        expected_position=plan.expected_position,
        stop_position=plan.stop_position_tensor,
        stop_position_constant=plan.stop_position,
        state_positions_after=plan.state_position_tensors,
    )
    return plan.output_buffer


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--local-batch", type=int, required=True)
    parser.add_argument("--start-position", type=int, default=2048)
    parser.add_argument("--settle-steps", type=int, default=132)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--check-mode", type=str, default="off", choices=("off", "bitwise")
    )
    parser.add_argument(
        "--b-semantics",
        type=str,
        default="replicated",
        choices=("replicated", "dp"),
        help=(
            "replicated: same B sequences on all 4 TP ranks (original E1F). "
            "dp: true DP-attention split, rank r serves its own local_batch "
            "sequences of B_global = 4*local_batch (E0dpf-gated)."
        ),
    )
    parser.add_argument(
        "--hc-backend", type=str, default="fused", choices=("fused", "eager", "default")
    )
    parser.add_argument(
        "--attention-tp-shard",
        action="store_true",
        help=(
            "E6F variant A: shard the attention o-path across TP4. NOT bitwise "
            "(changed summation order, TARGET 9.6) -- release goes through the "
            "D0L soft gate, so --check-mode bitwise still applies only to the "
            "graph-vs-eager comparison within one arm."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--config-tag", type=str, default="nogdr")
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
    dp = args.b_semantics == "dp"
    # distinct sequences served by the whole pipeline per step
    distinct_batch = local_batch * EXPECTED_TP_SIZE if dp else local_batch
    start_position = int(args.start_position)
    settle_steps = int(args.settle_steps)
    rounds = int(args.rounds)
    steps_per_round = int(args.steps)
    total_steps = settle_steps + rounds * steps_per_round
    stop_position = start_position + total_steps
    max_seq_len = ((stop_position + 127) // 128 + 1) * 128
    if start_position < 2047 or start_position % 128:
        raise SystemExit("start position must be 128-aligned and >= 2047")
    if settle_steps < 132:
        raise SystemExit(
            "settle segment must be >= 132 steps to cover a ratio-128 boundary"
        )

    schedule = build_decode_schedule(start_position, total_steps)
    family_counts = schedule_family_counts(schedule)

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E1F-full-decode-throughput",
        "measurement_class": "closed_loop_decode_throughput",
        "caliber": {
            "b_semantics": (
                "dp: true DP-attention sequence split; TP rank r serves its own "
                f"{local_batch} sequences (global rows [r*bl,(r+1)*bl) of "
                f"B_global={distinct_batch}); full 64 heads and bl KV per GPU; "
                "MoE all_gathers 4 distinct row blocks and reduce_scatters each "
                "rank's own rows; throughput = B_global / step_wall (measured, "
                "no conversion)"
                if dp
                else "full replication: identical B sequences on all 4 TP ranks "
                "per stage; bl per GPU = B; distinct global batch = B; MoE runs "
                "4B gathered rows (4 identical copies); NOT comparable with C1F "
                "or DP-attention numbers"
            ),
            "dp_equivalent_note": (
                "n/a in dp semantics: throughput is directly measured at "
                "B_global; no model-derived conversion is emitted"
                if dp
                else "per-rank compute at replicated bl=B equals a DP-attention "
                "deployment at global batch 4B; the dp_equivalent_tok_s field "
                "is 4x the measured replicated throughput (model-derived)"
            ),
            "kv": (
                f"seeded decode residency at position {start_position} "
                "(deterministic-seeded-KV-not-real-prefix, E1a27); "
                f"max_seq_len={max_seq_len}"
            ),
            "microbatches": 1,
            "pipeline_form": "serial closed-loop, one token batch in flight",
            "timing": (
                "host walls around torch.cuda.synchronize per component "
                "(E0qf serial decomposition caliber)"
            ),
            "hc_backend": args.hc_backend,
            "transport": "NCCL P2P fixed endpoints, no-GDR",
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
        "local_batch": local_batch,
        "b_semantics": args.b_semantics,
        "global_batch": distinct_batch,
        "start_position": start_position,
        "stop_position": stop_position,
        "max_seq_len": max_seq_len,
        "settle_steps": settle_steps,
        "rounds": rounds,
        "steps_per_round": steps_per_round,
        "family_counts": {
            family.value: count for family, count in family_counts.items()
        },
        "check_mode": args.check_mode,
        "stage_layer_ids": {str(s): list(v) for s, v in STAGE_LAYERS.items()},
        "checkpoint_id": None,
        "placement": None,
        "memory": {},
        "settle": None,
        "round_results": [],
        "terminals": None,
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
            raise ValueError(f"E1F requires world=16 (2 nodes x 8), got {world}")
        topo = create_e1f_topology(rank)
        stage = topo["stage"]
        tp_rank = topo["tp_rank"]
        result["stage"] = stage
        result["tp_rank"] = tp_rank
        stage_layer_ids = STAGE_LAYERS[stage]

        # warmups: one collective per TP group, one P2P per pair group.
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
            print(f"[E1F] placement {result['placement']['stage_hosts']}", flush=True)

        # rank-0 preflight: config + full 43-layer checkpoint contract.
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
                slots_per_shape=4,
                attention_tp_shard=args.attention_tp_shard,
                progress_every=args.progress_every,
                progress=(
                    (lambda message: print(f"[E1F] {message}", flush=True))
                    if rank in (0, 4, 8, 12)
                    else None
                ),
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
                f"[E1F] stage {stage} loaded ({len(stage_layer_ids)} layers, "
                f"free {result['memory']['after_load']['free_bytes'] / 2**30:.2f} "
                f"GiB, {result['diagnostic_seconds']['load']:.0f}s)",
                flush=True,
            )

        # ------------------------------------------------------------------
        # lanes
        backend = resolve_hc_boundary_backend(
            None if args.hc_backend == "default" else args.hc_backend
        )
        phase_started = time.perf_counter()

        def build_lanes() -> dict[str, StageLane]:
            payloads = {
                material.layer_id: build_seed_payload(
                    material,
                    seed=args.seed,
                    local_batch=local_batch,
                    start_position=start_position,
                    device=device,
                    dp_tp_rank=(tp_rank if dp else None),
                )
                for material in stage_material.materials
            }
            lanes: dict[str, StageLane] = {
                "graph": StageLane(
                    label="graph",
                    materials=stage_material.materials,
                    payloads=payloads,
                    backend=backend,
                    local_batch=local_batch,
                    start_position=start_position,
                    stop_position=stop_position,
                    device=device,
                )
            }
            if args.check_mode == "bitwise":
                lanes["eager"] = StageLane(
                    label="eager",
                    materials=stage_material.materials,
                    payloads=payloads,
                    backend=backend,
                    local_batch=local_batch,
                    start_position=start_position,
                    stop_position=stop_position,
                    device=device,
                )
            return lanes

        lanes = synchronized_local_step(
            "build lanes", build_lanes, device=device, world=world
        )
        graph_lane = lanes["graph"]
        eager_lane = lanes.get("eager")
        result["plan_resident_bytes"] = int(graph_lane.plan.resident_bytes)
        # Treatment witness: an env-gated variant that fails to reach the
        # process looks exactly like a variant that does not work.  Record what
        # the built blocks actually resolved to, so a no-op flag is visible in
        # the artifact instead of being read as a negative result.
        # Witness, take three.  Discovering *_mode/*_backend attributes was
        # not enough: the o-path sharding state lives in tp_size/tp_rank, which
        # match no naming convention, so an E6F run whose flag had been dropped
        # by the launcher looked identical to one where sharding did nothing.
        # Record what was *asked for* (argv) beside what was *resolved*, so the
        # two can disagree visibly.
        result["argv"] = list(sys.argv)
        result["attention_tp"] = {
            str(layer_id): {
                "tp_size": getattr(block.attention.config, "tp_size", None),
                "tp_rank": getattr(block.attention.config, "tp_rank", None),
                "local_num_heads": getattr(
                    block.attention.config, "local_num_heads", None
                ),
                "wo_b_shape": list(block.attention.weights.wo_b.shape),
            }
            for layer_id, block in zip(
                graph_lane.stage.layer_ids, graph_lane.stage.blocks, strict=True
            )
        }
        result["attention_modes"] = collect_attention_modes(
            layer_ids=graph_lane.stage.layer_ids,
            attentions=[block.attention for block in graph_lane.stage.blocks],
        )
        result["diagnostic_seconds"]["build"] = time.perf_counter() - phase_started
        memory_snapshot("after_build")
        if rank in (0, 4, 8, 12):
            print(
                f"[E1F] stage {stage} lanes built, free "
                f"{result['memory']['after_build']['free_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )

        if eager_lane is not None:
            graph_digests = graph_lane.state_digests()
            if graph_digests != eager_lane.state_digests():
                raise RuntimeError("graph/eager lanes were not seeded identically")

        # ------------------------------------------------------------------
        # warmup (E0hf pattern): warm cycles over the first settle window
        # with deterministic stage-local inputs, then restore.
        warm_schedule = schedule[:settle_steps]
        snapshots = synchronized_local_step(
            "snapshot states",
            lambda: [clone_state(state) for state in graph_lane.stage.states],
            device=device,
            world=world,
        )
        capture_stream = torch.cuda.Stream(device=device)
        graph_pools = {
            family: torch.cuda.graph_pool_handle() for family in DecodeGraphFamily
        }

        def warm_inputs(position: int) -> tuple[torch.Tensor, torch.Tensor]:
            residual = deterministic_tensor(
                seed=(args.seed * 1_000_003 + position * 7_919) & ((1 << 62) - 1),
                shape=(distinct_batch, 1, HC_MULT, HIDDEN),
                device=device,
            )
            if dp:
                residual = dp_row_slice(residual, tp_rank, local_batch)
            mixed = (args.seed * 2654435761 + position * 7919) & ((1 << 63) - 1)
            ids = torch.full(
                (local_batch, 1),
                mixed % EXPECTED_VOCAB,
                dtype=torch.int64,
                device=device,
            )
            return residual, ids

        def run_warm_cycle(
            lane: StageLane,
            *,
            moe_slots: Mapping[DecodeGraphFamily, int] | None = None,
        ) -> None:
            for step in warm_schedule:
                residual, ids = warm_inputs(step.position)
                lane.plan.input_residual_buffer.copy_(residual)
                lane.plan.input_ids_buffer.copy_(ids)
                forward_eager_prevalidated(
                    lane.stage,
                    lane.plan,
                    graph_family=step.family,
                    moe_slot=(
                        EAGER_MOE_SLOT
                        if moe_slots is None
                        else moe_slots[step.family]
                    ),
                )
                lane.cursor.advance_host(step.family)
            torch.cuda.synchronize(device)

        def restore_cycle(lane: StageLane) -> None:
            copy_stage_states(lane.stage.states, snapshots)
            lane.cursor.reset(start_position)
            lane.plan.expected_position.fill_(start_position)
            lane.plan.stop_position_tensor.fill_(lane.plan.stop_position)

        def warmup_all() -> None:
            run_warm_cycle(graph_lane)
            restore_cycle(graph_lane)
            with torch.cuda.stream(capture_stream):
                run_warm_cycle(graph_lane, moe_slots=GRAPH_MOE_SLOTS)
            torch.cuda.synchronize(device)
            restore_cycle(graph_lane)
            for slot in GRAPH_MOE_SLOT_TUPLE:
                for moe in graph_lane.stage.moes:
                    moe.reset_free_slot_completion_event(global_rows, slot)
            if eager_lane is not None:
                run_warm_cycle(eager_lane)
                restore_cycle(eager_lane)
            for lane in lanes.values():
                evidence = lane.terminal(start_position)
                if not evidence["accepted"]:
                    raise RuntimeError(
                        f"{lane.label} lane warmup restore drifted: {evidence}"
                    )

        phase_started = time.perf_counter()
        synchronized_local_step("warmups", warmup_all, device=device, world=world)
        result["diagnostic_seconds"]["warmup"] = time.perf_counter() - phase_started
        del snapshots
        torch.cuda.empty_cache()
        memory_snapshot("after_warmup")
        if rank in (0, 4, 8, 12):
            print(
                f"[E1F] stage {stage} warmup done "
                f"({result['diagnostic_seconds']['warmup']:.0f}s, free "
                f"{result['memory']['after_warmup']['free_bytes'] / 2**30:.2f} GiB)",
                flush=True,
            )

        # ------------------------------------------------------------------
        # fixed pipeline endpoints
        plan = graph_lane.plan
        staging: torch.Tensor | None = None
        handoff_in: SerialPairHandoff | None = None
        handoff_out: SerialPairHandoff | None = None
        if stage > 0:
            staging = torch.empty_like(plan.input_residual_buffer)
            validate_handoff_endpoint(staging, local_batch=local_batch)
            handoff_in = SerialPairHandoff(
                stage_id=1,
                pair_group=topo["prev_pair"],
                endpoint=staging,
                local_batch=local_batch,
            )
        if stage < STAGE_COUNT - 1:
            handoff_out = SerialPairHandoff(
                stage_id=0,
                pair_group=topo["next_pair"],
                endpoint=plan.output_buffer,
                local_batch=local_batch,
            )
        token_buffer = torch.zeros((local_batch, 1), dtype=torch.int64, device=device)
        if stage == 0:
            generator = torch.Generator(device="cpu").manual_seed(args.seed + 77)
            initial = torch.randint(
                0, EXPECTED_VOCAB, (distinct_batch, 1), generator=generator
            ).to(device)
            if dp:
                initial = dp_row_slice(initial, tp_rank, local_batch)
            token_buffer.copy_(initial)
            result["initial_tokens_first8"] = [
                int(v) for v in token_buffer.view(-1)[:8].cpu().tolist()
            ]
        zero_ids = torch.zeros((local_batch, 1), dtype=torch.int64, device=device)
        if stage != 0:
            plan.input_ids_buffer.copy_(zero_ids)
            if eager_lane is not None:
                eager_lane.plan.input_ids_buffer.copy_(zero_ids)

        graphs: dict[DecodeGraphFamily, torch.cuda.CUDAGraph] = {}
        capture_order: list[str] = []

        def validate_lane(lane: StageLane, family: DecodeGraphFamily) -> None:
            """Full stateful validation with external witnesses (contract)."""

            if stage == 0:
                external_residual = embed_hc_residual(embed_material, token_buffer)
                external_ids = token_buffer
            else:
                external_residual = (
                    staging
                    if lane is graph_lane
                    else staging.clone()  # eager lane needs a distinct external
                )
                external_ids = zero_ids
            lane.stage.validate_stateful_decode_call(
                external_residual,
                input_ids_local=external_ids,
                plan=lane.plan,
                graph_family=family,
            )

        def stage0_feed() -> None:
            """Embed current tokens into the plan input buffer (bitwise ==
            embed_hc_residual: same lookup values copied into 4 streams)."""

            hidden = torch.nn.functional.embedding(
                token_buffer, embed_material.embed_weight
            )
            plan.input_residual_buffer.copy_(
                hidden.unsqueeze(2).expand(-1, -1, HC_MULT, -1)
            )
            plan.input_ids_buffer.copy_(token_buffer)

        def pipeline_step(
            step_index: int,
            family: DecodeGraphFamily,
            timing: dict[str, list[float]] | None,
        ) -> None:
            t0 = time.perf_counter()
            if stage == 0:
                stage0_feed()
                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                replay_stateful_graph(graphs[family], plan, graph_family=family)
                torch.cuda.synchronize(device)
                t2 = time.perf_counter()
                handoff_out.transfer_step(step_index)
                torch.cuda.synchronize(device)
                t3 = time.perf_counter()
                pair_transfer(
                    token_buffer, send=False, group=topo["loop_pair"], peer=1
                )
                torch.cuda.synchronize(device)
                t4 = time.perf_counter()
                if timing is not None:
                    timing["embed"].append((t1 - t0) * 1e3)
                    timing["replay"].append((t2 - t1) * 1e3)
                    timing["send"].append((t3 - t2) * 1e3)
                    timing["token_wait"].append((t4 - t3) * 1e3)
                    timing["step_wall"].append((t4 - t0) * 1e3)
            else:
                handoff_in.transfer_step(step_index)
                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                plan.input_residual_buffer.copy_(staging)
                replay_stateful_graph(graphs[family], plan, graph_family=family)
                torch.cuda.synchronize(device)
                t2 = time.perf_counter()
                if stage < STAGE_COUNT - 1:
                    handoff_out.transfer_step(step_index)
                    torch.cuda.synchronize(device)
                    t3 = time.perf_counter()
                    if timing is not None:
                        timing["send"].append((t3 - t2) * 1e3)
                else:
                    logits = head_logits(head_material, plan.output_buffer)
                    token_buffer.copy_(logits.argmax(dim=-1, keepdim=True))
                    torch.cuda.synchronize(device)
                    t3 = time.perf_counter()
                    pair_transfer(
                        token_buffer, send=True, group=topo["loop_pair"], peer=0
                    )
                    torch.cuda.synchronize(device)
                    t4 = time.perf_counter()
                    if timing is not None:
                        timing["head"].append((t3 - t2) * 1e3)
                        timing["token_send"].append((t4 - t3) * 1e3)
                if timing is not None:
                    timing["recv"].append((t1 - t0) * 1e3)
                    timing["replay"].append((t2 - t1) * 1e3)
                    timing["step_wall"].append(
                        ((t4 if stage == STAGE_COUNT - 1 else t3) - t0) * 1e3
                    )
            graph_lane.cursor.advance_host(family)

        # ------------------------------------------------------------------
        # settle segment: lazy capture + optional bitwise graph-vs-eager
        phase_started = time.perf_counter()
        settle_record: dict[str, Any] = {
            "steps": settle_steps,
            "bitwise_steps": 0,
            "mismatched_positions": [],
            "max_abs_over_steps": 0.0,
            "judgment": (
                "graph_vs_eager_output bitwise (E1a27/E0sf), fused-vs-fused, "
                "pipeline-fed inputs"
                if args.check_mode == "bitwise"
                else "no per-step compare (check off)"
            ),
        }
        for step_index in range(settle_steps):
            step = schedule[step_index]
            if step_index == 0:
                # world-wide: all ranks are here before any transport starts.
                synchronized_local_step(
                    "settle initial validation",
                    lambda: validate_lane(graph_lane, step.family),
                    device=device,
                    world=world,
                )
            captured_here = False
            if step.family not in graphs:
                # inputs must be resident before capture+replay executes the
                # step: run the transport part manually, then capture.
                t_start = time.perf_counter()
                if stage == 0:
                    stage0_feed()
                else:
                    handoff_in.transfer_step(step_index)
                    plan.input_residual_buffer.copy_(staging)
                torch.cuda.synchronize(device)

                def capture() -> torch.cuda.CUDAGraph:
                    return capture_stateful_graph(
                        graph_lane.stage,
                        plan,
                        graph_family=step.family,
                        capture_stream=capture_stream,
                        pool=graph_pools[step.family],
                    )

                # consensus on the TP stage group only: the four stages hit
                # this point at different wall times in the serial pipeline.
                graphs[step.family] = synchronized_local_step(
                    f"capture {step.family.value}",
                    capture,
                    device=device,
                    world=EXPECTED_TP_SIZE,
                    group=topo["tp_group"],
                )
                capture_order.append(step.family.value)
                captured_here = True
                # execute the step via the fresh graph.
                replay_stateful_graph(graphs[step.family], plan, graph_family=step.family)
                torch.cuda.synchronize(device)
                if stage < STAGE_COUNT - 1:
                    handoff_out.transfer_step(step_index)
                else:
                    logits = head_logits(head_material, plan.output_buffer)
                    if not bool(torch.isfinite(logits).all().item()):
                        settle_record.setdefault("nonfinite_logit_steps", []).append(
                            step.position
                        )
                    token_buffer.copy_(logits.argmax(dim=-1, keepdim=True))
                    pair_transfer(
                        token_buffer, send=True, group=topo["loop_pair"], peer=0
                    )
                if stage == 0:
                    pair_transfer(
                        token_buffer, send=False, group=topo["loop_pair"], peer=1
                    )
                torch.cuda.synchronize(device)
                graph_lane.cursor.advance_host(step.family)
                if rank in (0, 12):
                    print(
                        f"[E1F] stage {stage} captured {step.family.value} at "
                        f"position {step.position} "
                        f"({time.perf_counter() - t_start:.1f}s)",
                        flush=True,
                    )
            else:
                pipeline_step(step_index, step.family, None)

            if eager_lane is not None:
                eager_plan = eager_lane.plan
                eager_plan.input_residual_buffer.copy_(plan.input_residual_buffer)
                eager_plan.input_ids_buffer.copy_(plan.input_ids_buffer)
                forward_eager_prevalidated(
                    eager_lane.stage, eager_plan, graph_family=step.family
                )
                torch.cuda.synchronize(device)
                eager_lane.cursor.advance_host(step.family)
                exact = bool(torch.equal(plan.output_buffer, eager_plan.output_buffer))
                if exact:
                    settle_record["bitwise_steps"] += 1
                else:
                    difference = (
                        plan.output_buffer.float() - eager_plan.output_buffer.float()
                    )
                    settle_record["mismatched_positions"].append(step.position)
                    settle_record["max_abs_over_steps"] = max(
                        settle_record["max_abs_over_steps"],
                        float(difference.abs().max().item()),
                    )
            if rank == 0 and (step_index % 32 == 0 or captured_here):
                print(
                    f"[E1F] settle step {step_index} pos {step.position} "
                    f"family {step.family.value} captured={captured_here}",
                    flush=True,
                )
        settle_record["capture_order"] = capture_order
        if eager_lane is not None:
            settle_record["final_state_digests_equal"] = bool(
                graph_lane.state_digests() == eager_lane.state_digests()
            )
            settle_record["eager_terminal"] = eager_lane.terminal(
                start_position + settle_steps
            )
            settle_record["accepted"] = bool(
                settle_record["bitwise_steps"] == settle_steps
                and settle_record["final_state_digests_equal"]
                and settle_record["eager_terminal"]["accepted"]
            )
        else:
            settle_record["accepted"] = bool(
                capture_order
                == ["normal", "ratio4_boundary", "ratio4_ratio128_boundary"]
            )
        # settle-end lane parity diagnostic (output digest across TP lanes)
        digest = tensor_sha256(plan.output_buffer)
        digests: list[Any] = [None] * EXPECTED_TP_SIZE
        dist.all_gather_object(digests, digest, group=topo["tp_group"])
        settle_record["output_lanes_bitwise"] = len(set(digests)) == 1
        # dp semantics: lanes hold distinct sequences, so this is expected
        # False and stays a diagnostic, never a gate.
        settle_record["output_lanes_bitwise_expected"] = not dp
        result["settle"] = settle_record
        result["diagnostic_seconds"]["settle"] = time.perf_counter() - phase_started
        if rank == 0:
            print(
                f"[E1F] settle done: capture_order={capture_order} "
                f"bitwise={settle_record['bitwise_steps']}/{settle_steps} "
                f"(check={args.check_mode})",
                flush=True,
            )

        # free the eager lane before timing (memory relief)
        if eager_lane is not None:
            settle_record["eager_kv_digests"] = "checked_then_freed"
            del lanes["eager"]
            eager_lane = None
            torch.cuda.empty_cache()
        memory_snapshot("after_settle")

        # ------------------------------------------------------------------
        # timed rounds
        for round_index in range(rounds):
            synchronized_local_step(
                f"round {round_index} entry validation",
                lambda: None,  # barrier-equivalent consensus point
                device=device,
                world=world,
            )
            timing: dict[str, list[float]] = {
                key: []
                for key in (
                    "embed",
                    "replay",
                    "send",
                    "recv",
                    "head",
                    "token_send",
                    "token_wait",
                    "step_wall",
                )
            }
            round_started = time.perf_counter()
            base = settle_steps + round_index * steps_per_round
            for offset in range(steps_per_round):
                step = schedule[base + offset]
                pipeline_step(base + offset, step.family, timing)
            torch.cuda.synchronize(device)
            round_wall = time.perf_counter() - round_started
            dispatch_error = int(graph_lane.cursor.dispatch_error.item())
            if dispatch_error:
                raise RuntimeError(
                    f"round {round_index} sticky dispatch error {dispatch_error}"
                )
            # cross-lane digest + logits finiteness diagnostics
            digest = tensor_sha256(plan.output_buffer)
            digests = [None] * EXPECTED_TP_SIZE
            dist.all_gather_object(digests, digest, group=topo["tp_group"])
            finite = None
            tokens_first8 = None
            if stage == STAGE_COUNT - 1:
                logits = head_logits(head_material, plan.output_buffer)
                finite = bool(torch.isfinite(logits).all().item())
                tokens_first8 = [
                    int(v) for v in token_buffer.view(-1)[:8].cpu().tolist()
                ]
            step_walls = timing["step_wall"]
            record = {
                "round": round_index,
                "steps": steps_per_round,
                "round_wall_s": round_wall,
                "timing_ms": {
                    key: summarize_ms(values)
                    for key, values in timing.items()
                    if values
                },
                "throughput_tok_s_mean": distinct_batch
                / (statistics.fmean(step_walls) / 1e3),
                "throughput_tok_s_p50": distinct_batch
                / (sorted(step_walls)[len(step_walls) // 2] / 1e3),
                "dp_equivalent_tok_s_mean": (
                    None
                    if dp
                    else 4 * local_batch / (statistics.fmean(step_walls) / 1e3)
                ),
                "output_lanes_bitwise": len(set(digests)) == 1,
                "logits_finite": finite,
                "tokens_first8": tokens_first8,
                "step_wall_raw_ms": [round(v, 4) for v in step_walls],
                "replay_raw_ms": [round(v, 4) for v in timing["replay"]],
            }
            result["round_results"].append(record)
            if rank in (0, 12):
                summary = record["timing_ms"].get("step_wall", {})
                print(
                    f"[E1F] stage {stage} round {round_index}: step_wall "
                    f"p50 {summary.get('p50_ms', 0):.2f} ms p95 "
                    f"{summary.get('p95_ms', 0):.2f} ms -> "
                    f"{record['throughput_tok_s_p50']:.1f} tok/s "
                    f"({'DP B_global=' + str(distinct_batch) if dp else 'replicated B'})",
                    flush=True,
                )
            memory_snapshot(f"after_round_{round_index}")

        # ------------------------------------------------------------------
        # terminals + teardown
        result["terminals"] = graph_lane.terminal(stop_position)
        result["handoff_records"] = {}
        if handoff_in is not None:
            result["handoff_records"]["in"] = handoff_in.close(
                expected_steps=total_steps
            )
        if handoff_out is not None:
            result["handoff_records"]["out"] = handoff_out.close(
                expected_steps=total_steps
            )
        teardown = synchronized_local_step(
            "teardown",
            lambda: teardown_stateful_graphs(
                graph_lane.stage, plan, graphs, pool_handles=graph_pools
            ),
            device=device,
            world=world,
        )
        result["teardown"] = teardown
        memory_snapshot("at_end")

        result["accepted"] = bool(
            result["placement"]["accepted"]
            and result["settle"]["accepted"]
            and capture_order
            == ["normal", "ratio4_boundary", "ratio4_ratio128_boundary"]
            and len(result["round_results"]) == rounds
            and result["terminals"]["accepted"]
            and teardown["accepted"]
            and not graph_lane.stage.poisoned
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
        # compact merged artifact: stage representatives, raw arrays dropped.
        merged = []
        for record in gathered:
            if not isinstance(record, dict):
                continue
            if record.get("tp_rank") != 0:
                continue
            trimmed = dict(record)
            trimmed["round_results"] = [
                {
                    key: value
                    for key, value in round_record.items()
                    if key not in ("step_wall_raw_ms", "replay_raw_ms")
                }
                for round_record in record.get("round_results", [])
            ]
            merged.append(trimmed)
        write_json(
            out_dir / "result.json",
            {
                "experiment": "E1F-full-decode-throughput",
                "accepted": accepted_all,
                "local_batch": local_batch,
                "check_mode": args.check_mode,
                "stage_representatives": merged,
            },
        )
        print(f"[E1F] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
