#!/usr/bin/env python3
"""E0qf: dual-node TP4 x PP2 pipeline decode gate (V4-Flash, titan064+065).

Ninth port vertical: the E0pf single-machine TP4xPP2 pipeline stretched
across two machines -- stage 0 = layers 0-5 on titan064 GPUs 0-3, stage 1 =
layers 6-11 on titan065 GPUs 0-3 -- with real weights and the identical
serial fixed-endpoint NCCL P2P handoff (``dsv4_direct.pipeline_transport``:
pair groups (r, r+4), ``SerialPairHandoff``, staged D2D unpack).  The pair
groups now cross the IB fabric (B2-recal environment: bootstrap over
``NCCL_SOCKET_IFNAME=enp33s0f0``, data over IB verbs, ``NCCL_P2P_LEVEL=SYS``;
no-GDR default, GDR via the ``libcuda-onebyte-patch`` LD_LIBRARY_PATH opt-in
plus ``NCCL_NET_GDR_LEVEL=SYS``).  Stateful eager decode over the E0sf
schedule [8192, 8324): 132 steps (99 NORMAL, 32 RATIO4_BOUNDARY, the
RATIO4_RATIO128_BOUNDARY step at 8319); serial handoff, ``--microbatches``
(default 2) interleaved serially, exactly as E0pf.

Cross-machine determinism is *measured, not assumed* (task contract): the
reference 12-layer chain runs on the stage-0 TP group (titan064) while the
pipeline runs layers 6-11 on titan065, so bitwise chain-vs-stage1 equality
requires cross-machine reproducibility of the whole TP4 block path.  Three
instruments separate "pipeline correctness" from "cross-machine numerics":

- **Cross-group collective probe** (E0pf form, now cross-machine):
  reduce_scatter + all_gather on identical inputs on both TP groups;
  diagnostic evidence for NCCL-collective determinism across the machines.
- **Same-path self-consistency (replica run).**  The whole pipeline phase
  runs twice on independently seeded-identical lanes; stage-1 outputs, KV
  digests and payload SHAs must match bitwise between the two runs.  This
  is a hard gate: it certifies the cross-machine pipeline path is
  self-deterministic regardless of how it compares against the chain.
- **Chain comparison with a mode split.**  Stage-0 exit vs chain after-L5
  is judged bitwise (same machine, E0pf precedent).  Stage-1 exit vs chain
  after-L11 is judged bitwise when it holds
  (``stage1_judgment_mode=bitwise``); if a cross-machine bf16-level
  difference is observed instead, the gate closes in
  ``numeric_gate_with_self_consistency`` mode: outputs finite, per-step
  rms_rel <= ``--stage1-rms-rel-tol``, self-consistency and the probe
  recorded as attribution evidence, and stage-1 KV digest divergence
  reported (not force-fitted).

Unconditional (mode-independent) gates: stage-0 exit bitwise + stage-0 KV
digest parity vs chain; per-step payload SHA sender==receiver (both runs;
the transport is byte-exact by contract); handoff endpoint pointer
stability; cursor terminals; and stage placement (all stage-0 ranks on one
host, all stage-1 ranks on a different host).  Pre-MoE fragment parity
(E0pf part d) is a single-machine ABI property already gated by E0pf and is
not re-run here.

Timing (report-only): per-step stage-0 compute, sender-observed handoff
wall (32 KiB/step payload; compare B2 small-packet ~150 us scale),
receiver recv wall, stage-1 compute, and per-step stage wall.

Run (driven from the workstation by ``run_e0qf_dual.sh``; manually):
  # titan064 (node 0) and titan065 (node 1), same command except --node-rank
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export NCCL_SOCKET_IFNAME=enp33s0f0 NCCL_IB_DISABLE=0 NCCL_P2P_LEVEL=SYS
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  # GDR config additionally:
  #   export LD_LIBRARY_PATH=/home/cysic/libcuda-onebyte-patch:$LD_LIBRARY_PATH
  #   export NCCL_NET_GDR_LEVEL=SYS
  CUDA_VISIBLE_DEVICES=0,1,2,3 ~/Workspace/venvs/sglang/bin/torchrun \
    --nnodes 2 --node-rank {0|1} --nproc-per-node 4 \
    --master-addr 10.234.1.64 --master-port 29631 \
    e0qf_pp2_dual_gate.py --config-tag {nogdr|gdr} \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0qf-{nogdr|gdr}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from dsv4_direct.block import DirectDecodeBlock
from dsv4_direct.block_weights import inspect_replicated_block_contract
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.physical_stage import (
    EXPECTED_TP_SIZE,
    PP2_STAGE_LAYER_IDS,
    PhysicalLayerMaterial,
    PhysicalStageMaterial,
    build_physical_stage,
    validate_live_tp_group,
)
from dsv4_direct.pipeline_transport import (
    PP2_WORLD,
    PP2GroupBundle,
    SerialPairHandoff,
    create_pp2_groups,
    validate_handoff_endpoint,
)
from dsv4_direct.ratio4_oracle import seed_nonzero_ratio4_state
from dsv4_direct.stateful_decode import (
    DecodeGraphFamily,
    StatefulDecodeCursor,
    build_decode_schedule,
    schedule_family_counts,
)
from dsv4_direct.static_kv import StaticLayerKV
from dsv4_direct.static_ratio4_kv import StaticRatio4KV
from dsv4_direct.static_window_kv import StaticWindowKV
from dsv4_direct.superstage import (
    TP4DecodeStage,
    TP4StatefulDecodeSuperStagePlan,
)


EXPECTED_VOCAB = 129280
LOCAL_BATCH = 1
GLOBAL_BATCH = LOCAL_BATCH * EXPECTED_TP_SIZE
MAX_SEQ_LEN = 8448

STAGE0_LAYER_IDS = PP2_STAGE_LAYER_IDS[0]
STAGE1_LAYER_IDS = PP2_STAGE_LAYER_IDS[1]
CHAIN_LAYER_IDS = STAGE0_LAYER_IDS + STAGE1_LAYER_IDS
STAGE0_EXIT_INDEX = len(STAGE0_LAYER_IDS) - 1

START_POSITION = 8192
STEP_COUNT = 132
STOP_POSITION = START_POSITION + STEP_COUNT
SCHEDULE = build_decode_schedule(START_POSITION, STEP_COUNT)
FAMILY_COUNTS = schedule_family_counts(SCHEDULE)

EAGER_MOE_SLOT = 0
GRAPH_MOE_SLOT_TUPLE = (1, 2, 3)  # plan default; unused (no graph capture here)


# --------------------------------------------------------------------------
# generic helpers (E0sf process form)


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


def stream_seed(*, seed: int, tp_rank: int, microbatch: int) -> int:
    """One deterministic sub-seed per (rank, microbatch) stream."""

    return (seed * 1_000_003 + tp_rank * 100_003 + microbatch * 611_953) & (
        (1 << 62) - 1
    )


def deterministic_residual(
    *, seed: int, tp_rank: int, microbatch: int, position: int, device: torch.device
) -> torch.Tensor:
    return deterministic_tensor(
        seed=(stream_seed(seed=seed, tp_rank=tp_rank, microbatch=microbatch)
              + position * 7_919) & ((1 << 62) - 1),
        shape=(LOCAL_BATCH, 1, 4, 4096),
        device=device,
    )


def deterministic_input_ids(
    *, seed: int, tp_rank: int, microbatch: int, position: int, device: torch.device
) -> torch.Tensor:
    mixed = (
        stream_seed(seed=seed, tp_rank=tp_rank, microbatch=microbatch) * 2654435761
        + position * 7919
    ) & ((1 << 63) - 1)
    return torch.full(
        (LOCAL_BATCH, 1), mixed % EXPECTED_VOCAB, dtype=torch.int64, device=device
    )


def error_metrics(observed: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if observed.shape != expected.shape:
        raise ValueError("comparison shapes differ")
    left = observed.detach().float()
    right = expected.detach().float()
    difference = left - right
    finite = bool(
        torch.isfinite(left).all().item()
        and torch.isfinite(right).all().item()
        and torch.isfinite(difference).all().item()
    )
    if not finite:
        raise ValueError("comparison is non-finite")
    rms_abs = float(difference.square().mean().sqrt().item())
    reference_rms = float(right.square().mean().sqrt().item())
    return {
        "finite": True,
        "bitwise_exact": bool(torch.equal(observed, expected)),
        "max_abs": float(difference.abs().max().item()),
        "rms_abs": rms_abs,
        "reference_rms": reference_rms,
        "rms_rel": rms_abs / max(reference_rms, 1e-12),
    }


def synchronized_local_step(
    name: str, fn: Any, *, device: torch.device, world: int
) -> Any:
    value: Any = None
    local_error: str | None = None
    try:
        value = fn()
    except Exception:
        local_error = traceback.format_exc()
        # Surface immediately: if peer ranks are wedged inside collectives,
        # the consensus below can never complete and only this print exists.
        print(
            f"[E0qf][rank {dist.get_rank()}] {name} raised:\n{local_error}",
            flush=True,
        )
    failed = torch.tensor(int(local_error is not None), device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        errors: list[str | None] = [None for _ in range(world)]
        dist.all_gather_object(errors, local_error)
        details = "\n".join(
            f"rank {rank}:\n{error}" for rank, error in enumerate(errors) if error
        )
        raise RuntimeError(f"{name} failed before the next collective:\n{details}")
    return value


DirectState = StaticWindowKV | StaticRatio4KV | StaticLayerKV


def full_state_sha256(state: DirectState) -> str:
    digest = hashlib.sha256()
    for name, tensor in state._owned_tensor_items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def summarize_ms(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": float(len(values)),
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


# --------------------------------------------------------------------------
# seeding (E0sf layer-type seeders, keyed per microbatch stream)


def build_seed_payload(
    material: PhysicalLayerMaterial,
    *,
    seed: int,
    tp_rank: int,
    microbatch: int,
    device: torch.device,
) -> dict[str, Any]:
    layer_seed = (
        stream_seed(seed=seed, tp_rank=tp_rank, microbatch=microbatch) * 9_176_501
        + material.layer_id * 15_485_863
    ) & ((1 << 62) - 1)
    if material.kind == "window":
        return {
            "raw": deterministic_tensor(
                seed=layer_seed,
                shape=(LOCAL_BATCH, 128, 512),
                device=device,
                scale=0.03,
            )
        }
    if material.kind == "ratio128":
        return {
            "raw": deterministic_tensor(
                seed=layer_seed,
                shape=(LOCAL_BATCH, 128, 512),
                device=device,
                scale=0.03,
            ),
            "compressed": deterministic_tensor(
                seed=layer_seed + 1,
                shape=(LOCAL_BATCH, START_POSITION // 128, 512),
                device=device,
                scale=0.025,
            ),
        }
    oracle_state = seed_nonzero_ratio4_state(
        material.attention_config,
        batch_size=LOCAL_BATCH,
        start_pos=START_POSITION,
        main_ape=material.prepared.compressor_ape,
        index_ape=material.prepared.index_compressor_ape,
        seed=layer_seed,
        device=device,
    )
    return {"oracle": oracle_state}


def seed_state(
    material: PhysicalLayerMaterial, state: DirectState, payload: dict[str, Any]
) -> None:
    if material.kind == "window":
        assert isinstance(state, StaticWindowKV)
        state.seed_decode_residency(
            start_pos=START_POSITION, raw=payload["raw"].clone()
        )
    elif material.kind == "ratio128":
        assert isinstance(state, StaticLayerKV)
        state.seed_decode_residency(
            start_pos=START_POSITION,
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


def new_seeded_block(
    material: PhysicalLayerMaterial, payload: dict[str, Any]
) -> DirectDecodeBlock:
    state = material.new_state(num_local_sequences=LOCAL_BATCH)
    seed_state(material, state, payload)
    return material.new_block(state)


# --------------------------------------------------------------------------
# stateful eager execution (E0sf ``forward_eager_prevalidated`` with an
# optional stage-exit capture index for the chain lane)


def forward_eager_prevalidated(
    stage: TP4DecodeStage,
    plan: TP4StatefulDecodeSuperStagePlan,
    *,
    graph_family: DecodeGraphFamily,
    moe_slot: int = EAGER_MOE_SLOT,
    capture_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    plan.cursor.guard_device_preflight(
        graph_family,
        expected_position=plan.expected_position,
        stop_position=plan.stop_position_tensor,
        stop_position_constant=plan.stop_position,
        state_positions=plan.state_position_tensors,
    )
    output = plan.input_residual_buffer
    captured: torch.Tensor | None = None
    for index, (block, layer_plan) in enumerate(
        zip(stage.blocks, plan.layer_plans, strict=True)
    ):
        output = block.forward_stateful_decode_tensor(
            output,
            input_ids_local=(
                plan.input_ids_buffer if block.route_kind == "hash" else None
            ),
            attention_plan=layer_plan,
            graph_family=graph_family,
            moe_slot=moe_slot,
        )
        if capture_index is not None and index == capture_index:
            captured = output
    plan.output_buffer.copy_(output)
    plan.cursor.advance_device(
        graph_family,
        expected_position=plan.expected_position,
        stop_position=plan.stop_position_tensor,
        stop_position_constant=plan.stop_position,
        state_positions_after=plan.state_position_tensors,
    )
    return plan.output_buffer, captured


def cursor_terminal_evidence(
    plan: TP4StatefulDecodeSuperStagePlan,
    *,
    expected_position: int,
) -> dict[str, Any]:
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
        "state_positions": state_positions,
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


class StageLane:
    """One microbatch lane: seeded blocks, stage, cursor, stateful plan."""

    def __init__(
        self,
        *,
        label: str,
        materials: Sequence[PhysicalLayerMaterial],
        payloads: Sequence[dict[str, Any]],
        device: torch.device,
    ) -> None:
        self.label = label
        self.blocks = [
            new_seeded_block(material, payload)
            for material, payload in zip(materials, payloads, strict=True)
        ]
        self.stage = TP4DecodeStage(self.blocks)
        self.cursor = StatefulDecodeCursor(
            start_position=START_POSITION, device=device
        )
        self.plan = self.stage.prepare_stateful_decode_plan(
            self.cursor,
            start_position=START_POSITION,
            stop_position=STOP_POSITION,
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
        return cursor_terminal_evidence(
            self.plan, expected_position=expected_position
        )


# --------------------------------------------------------------------------
# diagnostic probe: cross-TP-group NCCL collective bitwise determinism


def run_group_determinism_probe(
    *,
    groups: PP2GroupBundle,
    seed: int,
    device: torch.device,
    world: int,
) -> dict[str, Any]:
    """reduce_scatter+all_gather identical inputs on both TP groups; compare.

    all_gather is pure concatenation (bitwise trivially); reduce_scatter SUM
    order depends on the NCCL ring over each group's physical GPUs.  In the
    dual-node form the two TP groups live on different machines (titan064
    GPUs 0-3 vs titan065 GPUs 0-3), so this probe is the direct evidence
    for whether chain layers 6-11 on titan064 can be compared bitwise
    against pipeline stage 1 on titan065.
    """

    rows = GLOBAL_BATCH
    hidden = 4096
    local = deterministic_tensor(
        seed=(seed * 7_368_787 + groups.tp_rank * 104_729) & ((1 << 62) - 1),
        shape=(rows, hidden),
        device=device,
        scale=1.0,
    )
    scattered = torch.empty(
        rows // EXPECTED_TP_SIZE, hidden, dtype=local.dtype, device=device
    )
    dist.reduce_scatter_tensor(
        scattered, local, op=dist.ReduceOp.SUM, group=groups.tp_group
    )
    gathered = torch.empty(rows, hidden, dtype=local.dtype, device=device)
    dist.all_gather_into_tensor(gathered, scattered, group=groups.tp_group)
    torch.cuda.synchronize(device)
    record = {
        "tp_rank": groups.tp_rank,
        "stage_id": groups.stage_id,
        "reduce_scatter_sha256": tensor_sha256(scattered),
        "all_gather_sha256": tensor_sha256(gathered),
    }
    gathered_records: list[Any] = [None] * world
    dist.all_gather_object(gathered_records, record)
    pairs = []
    for tp_rank in range(EXPECTED_TP_SIZE):
        left = gathered_records[tp_rank]
        right = gathered_records[tp_rank + EXPECTED_TP_SIZE]
        pairs.append(
            {
                "tp_rank": tp_rank,
                "reduce_scatter_equal": left["reduce_scatter_sha256"]
                == right["reduce_scatter_sha256"],
                "all_gather_equal": left["all_gather_sha256"]
                == right["all_gather_sha256"],
            }
        )
    return {
        "judgment": "diagnostic_evidence_only_not_an_acceptance_gate",
        "pairs": pairs,
        "cross_group_bitwise": all(
            pair["reduce_scatter_equal"] and pair["all_gather_equal"]
            for pair in pairs
        ),
    }


# --------------------------------------------------------------------------
# dual-node placement evidence (all stage-0 ranks on one host, all stage-1
# ranks on a different host)


def run_placement_check(
    *, groups: PP2GroupBundle, world: int
) -> dict[str, Any]:
    record = {
        "rank": dist.get_rank(),
        "stage_id": groups.stage_id,
        "host": platform.node(),
    }
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, record)
    stage_hosts: dict[int, set[str]] = {0: set(), 1: set()}
    for entry in gathered:
        stage_hosts[entry["stage_id"]].add(entry["host"])
    result = {
        "stage0_hosts": sorted(stage_hosts[0]),
        "stage1_hosts": sorted(stage_hosts[1]),
    }
    result["accepted"] = bool(
        len(stage_hosts[0]) == 1
        and len(stage_hosts[1]) == 1
        and stage_hosts[0] != stage_hosts[1]
    )
    return result


# --------------------------------------------------------------------------
# chain phase (stage-0 TP group only)


def run_chain_phase(
    *,
    chain_lanes: Sequence[StageLane],
    seed: int,
    tp_rank: int,
    device: torch.device,
) -> dict[str, Any]:
    microbatches = len(chain_lanes)
    mid_records: list[list[torch.Tensor]] = [[] for _ in range(microbatches)]
    final_records: list[list[torch.Tensor]] = [[] for _ in range(microbatches)]
    for step_index, step in enumerate(SCHEDULE):
        for microbatch, lane in enumerate(chain_lanes):
            residual = deterministic_residual(
                seed=seed,
                tp_rank=tp_rank,
                microbatch=microbatch,
                position=step.position,
                device=device,
            )
            ids = deterministic_input_ids(
                seed=seed,
                tp_rank=tp_rank,
                microbatch=microbatch,
                position=step.position,
                device=device,
            )
            lane.stage.validate_stateful_decode_call(
                residual,
                input_ids_local=ids,
                plan=lane.plan,
                graph_family=step.family,
            )
            if lane.cursor.host_position != step.position:
                raise RuntimeError(
                    f"chain mb{microbatch} host cursor drifted at step {step_index}"
                )
            lane.plan.input_residual_buffer.copy_(residual)
            lane.plan.input_ids_buffer.copy_(ids)
            output, mid = forward_eager_prevalidated(
                lane.stage,
                lane.plan,
                graph_family=step.family,
                capture_index=STAGE0_EXIT_INDEX,
            )
            if mid is None:
                raise RuntimeError("chain lane did not capture the L5 exit hidden")
            mid_records[microbatch].append(mid.clone())
            final_records[microbatch].append(output.clone())
            lane.cursor.advance_host(step.family)
        if step_index % 32 == 0:
            torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)
    terminals = [lane.terminal(STOP_POSITION) for lane in chain_lanes]
    return {
        "mid_records": mid_records,
        "final_records": final_records,
        "terminals": terminals,
        "state_digests": [lane.state_digests() for lane in chain_lanes],
    }


# --------------------------------------------------------------------------
# pipeline phase (all eight ranks)


def run_pipeline_phase(
    *,
    label: str,
    stage_lanes: Sequence[StageLane],
    groups: PP2GroupBundle,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    stage_id = groups.stage_id
    tp_rank = groups.tp_rank
    microbatches = len(stage_lanes)
    # Stage 0 sends its fixed plan output buffer.  Stage 1 receives into a
    # dedicated fixed staging endpoint and D2D-unpacks it into the plan input
    # buffer (the gaiban E1b2z "stage1_d2d_unpack" form): the stage-level
    # stateful validation contract requires the validated input to be
    # external to the plan workspaces, so a direct receive into the plan
    # input buffer is off-contract by design.
    receive_buffers: list[torch.Tensor] = []
    handoffs = []
    for lane in stage_lanes:
        if stage_id == 0:
            endpoint = lane.plan.output_buffer
        else:
            endpoint = torch.empty_like(lane.plan.input_residual_buffer)
            receive_buffers.append(endpoint)
        validate_handoff_endpoint(endpoint, local_batch=LOCAL_BATCH)
        handoffs.append(
            SerialPairHandoff(
                stage_id=stage_id,
                pair_group=groups.pair_group,
                endpoint=endpoint,
                local_batch=LOCAL_BATCH,
            )
        )

    payload_shas: list[list[str]] = [[] for _ in range(microbatches)]
    stage1_outputs: list[torch.Tensor] = [
        torch.empty(
            (STEP_COUNT, LOCAL_BATCH, 1, 4, 4096),
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(microbatches)
    ]
    timing: dict[str, list[float]] = {
        "stage0_compute": [],
        "stage0_send_wall": [],
        "stage1_recv_wall": [],
        "stage1_compute": [],
        "step_wall": [],
    }

    for step_index, step in enumerate(SCHEDULE):
        step_started = time.perf_counter()
        if stage_id == 0:
            for microbatch, (lane, handoff) in enumerate(
                zip(stage_lanes, handoffs, strict=True)
            ):
                residual = deterministic_residual(
                    seed=seed,
                    tp_rank=tp_rank,
                    microbatch=microbatch,
                    position=step.position,
                    device=device,
                )
                ids = deterministic_input_ids(
                    seed=seed,
                    tp_rank=tp_rank,
                    microbatch=microbatch,
                    position=step.position,
                    device=device,
                )
                lane.stage.validate_stateful_decode_call(
                    residual,
                    input_ids_local=ids,
                    plan=lane.plan,
                    graph_family=step.family,
                )
                lane.plan.input_residual_buffer.copy_(residual)
                lane.plan.input_ids_buffer.copy_(ids)
                started = time.perf_counter()
                forward_eager_prevalidated(
                    lane.stage, lane.plan, graph_family=step.family
                )
                torch.cuda.synchronize(device)
                computed = time.perf_counter()
                payload_shas[microbatch].append(
                    tensor_sha256(lane.plan.output_buffer)
                )
                handoff.transfer_step(step_index)
                torch.cuda.synchronize(device)
                sent = time.perf_counter()
                timing["stage0_compute"].append((computed - started) * 1e3)
                timing["stage0_send_wall"].append((sent - computed) * 1e3)
                lane.cursor.advance_host(step.family)
        else:
            # Post both receives up front so the sender-side send wall is a
            # clean transfer estimate (receiver already matched).
            for microbatch, (lane, handoff) in enumerate(
                zip(stage_lanes, handoffs, strict=True)
            ):
                ids = deterministic_input_ids(
                    seed=seed,
                    tp_rank=tp_rank,
                    microbatch=microbatch,
                    position=step.position,
                    device=device,
                )
                lane.stage.validate_stateful_decode_call(
                    receive_buffers[microbatch],
                    input_ids_local=ids,
                    plan=lane.plan,
                    graph_family=step.family,
                )
                lane.plan.input_ids_buffer.copy_(ids)
            posted = time.perf_counter()
            for handoff in handoffs:
                handoff.transfer_step(step_index)
            for microbatch, lane in enumerate(stage_lanes):
                torch.cuda.synchronize(device)
                received = time.perf_counter()
                if microbatch == 0:
                    timing["stage1_recv_wall"].append((received - posted) * 1e3)
                payload_shas[microbatch].append(
                    tensor_sha256(receive_buffers[microbatch])
                )
                lane.plan.input_residual_buffer.copy_(
                    receive_buffers[microbatch]
                )
                started = time.perf_counter()
                forward_eager_prevalidated(
                    lane.stage, lane.plan, graph_family=step.family
                )
                torch.cuda.synchronize(device)
                computed = time.perf_counter()
                timing["stage1_compute"].append((computed - started) * 1e3)
                stage1_outputs[microbatch][step_index].copy_(
                    lane.plan.output_buffer
                )
                lane.cursor.advance_host(step.family)
        timing["step_wall"].append((time.perf_counter() - step_started) * 1e3)
        if tp_rank == 0 and step_index % 32 == 0:
            print(
                f"[E0qf][{label}] stage{stage_id} pipeline step {step_index} "
                f"pos {step.position} family {step.family.value}",
                flush=True,
            )
    torch.cuda.synchronize(device)

    handoff_records = [
        handoff.close(expected_steps=STEP_COUNT) for handoff in handoffs
    ]
    terminals = [lane.terminal(STOP_POSITION) for lane in stage_lanes]
    return {
        "payload_shas": payload_shas,
        "stage1_outputs": stage1_outputs if stage_id == 1 else None,
        "handoff_records": handoff_records,
        "terminals": terminals,
        "state_digests": [lane.state_digests() for lane in stage_lanes],
        "timing": {key: summarize_ms(values) for key, values in timing.items()},
        "timing_first8": {
            key: [round(value, 4) for value in values[: 8 * microbatches]]
            for key, values in timing.items()
        },
    }


def exchange_stage1_outputs(
    *,
    pipeline: Mapping[str, Any],
    groups: PP2GroupBundle,
    microbatches: int,
    device: torch.device,
) -> list[torch.Tensor] | None:
    """Bulk-return stage-1 outputs to the paired stage-0 rank for comparison."""

    if groups.stage_id == 1:
        for microbatch in range(microbatches):
            outputs = pipeline["stage1_outputs"][microbatch]
            works = dist.batch_isend_irecv(
                [
                    dist.P2POp(
                        dist.isend, outputs, group=groups.pair_group, group_peer=0
                    )
                ]
            )
            works[0].wait()
        torch.cuda.synchronize(device)
        return None
    received = []
    for _ in range(microbatches):
        buffer = torch.empty(
            (STEP_COUNT, LOCAL_BATCH, 1, 4, 4096),
            dtype=torch.bfloat16,
            device=device,
        )
        works = dist.batch_isend_irecv(
            [dist.P2POp(dist.irecv, buffer, group=groups.pair_group, group_peer=1)]
        )
        works[0].wait()
        received.append(buffer)
    torch.cuda.synchronize(device)
    return received


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--microbatches", type=int, default=2, choices=(1, 2))
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument(
        "--config-tag",
        type=str,
        required=True,
        help="transport configuration label recorded in results (nogdr | gdr)",
    )
    parser.add_argument(
        "--stage1-rms-rel-tol",
        type=float,
        default=0.02,
        help=(
            "per-step rms_rel tolerance for chain-vs-stage1 exits, applied "
            "only in numeric_gate_with_self_consistency mode"
        ),
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    microbatches = int(args.microbatches)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0qf-tp4-pp2-dual-node-serial-pipeline",
        "measurement_class": "semantic_correctness_gate_plus_reportonly_timing",
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
        "microbatches": microbatches,
        "stage1_rms_rel_tol": args.stage1_rms_rel_tol,
        "pipeline_form": (
            "stateful_eager_serial_handoff "
            "(overlap deferred; one payload in flight; microbatches interleave "
            "serially)"
        ),
        "schedule": {
            "start_position": START_POSITION,
            "stop_position": STOP_POSITION,
            "step_count": STEP_COUNT,
            "family_counts": {
                family.value: count for family, count in FAMILY_COUNTS.items()
            },
        },
        "stage_layer_ids": {
            "stage0": list(STAGE0_LAYER_IDS),
            "stage1": list(STAGE1_LAYER_IDS),
        },
        "checkpoint_id": None,
        "placement": None,
        "group_probe": None,
        "pipeline": None,
        "pipeline_replica": None,
        "self_consistency": None,
        "comparison": None,
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }

    started = time.perf_counter()
    try:
        if world != PP2_WORLD:
            raise ValueError(
                f"E0qf requires world=8 (2 nodes x TP4), got {world}"
            )
        groups = create_pp2_groups(rank)
        stage_id = groups.stage_id
        tp_rank = groups.tp_rank
        result["stage_id"] = stage_id
        result["tp_rank"] = tp_rank

        # NCCL warmups: one collective per TP group, one P2P per pair group.
        warm = torch.ones(1, device=device)
        dist.all_reduce(warm, group=groups.tp_group)
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(
                    dist.isend if stage_id == 0 else dist.irecv,
                    warm,
                    group=groups.pair_group,
                    group_peer=1 - stage_id,
                )
            ]
        )
        works[0].wait()
        torch.cuda.synchronize(device)
        result["tp_group_binding"] = validate_live_tp_group(
            groups.tp_group,
            expected_local_rank=tp_rank,
            expected_global_ranks=groups.tp_global_ranks,
        )

        # dual-node placement: stage 0 and stage 1 must live on two hosts.
        result["placement"] = run_placement_check(groups=groups, world=world)
        if not result["placement"]["accepted"]:
            raise ValueError(
                f"dual-node placement violated: {result['placement']}"
            )
        if rank == 0:
            print(
                f"[E0qf] placement stage0={result['placement']['stage0_hosts']} "
                f"stage1={result['placement']['stage1_hosts']}",
                flush=True,
            )

        # rank-0 preflight: config + checkpoint/block contracts for L0-L11.
        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root, list(CHAIN_LAYER_IDS), EXPECTED_TP_SIZE
                )
                if not checkpoint["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {checkpoint['errors'][:3]}"
                    )
                for layer_id in CHAIN_LAYER_IDS:
                    block_contract = inspect_replicated_block_contract(
                        stage_root,
                        layer_id=layer_id,
                        rank=0,
                        world_size=EXPECTED_TP_SIZE,
                    )
                    if not block_contract["ok"]:
                        raise ValueError(
                            f"layer-{layer_id} block contract failed: "
                            f"{block_contract['errors'][:3]}"
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
        # load: stage-0 ranks own L0-L11 (pipeline stage 0 + reference
        # chain); stage-1 ranks own L6-L11.
        phase_started = time.perf_counter()

        def load_materials() -> PhysicalStageMaterial:
            own_layers = CHAIN_LAYER_IDS if stage_id == 0 else STAGE1_LAYER_IDS
            return build_physical_stage(
                stage_id=stage_id,
                layer_ids=own_layers,
                model_config=model_config,
                stage_root=stage_root,
                tp_rank=tp_rank,
                tp_group=groups.tp_group,
                tp_global_ranks=groups.tp_global_ranks,
                device=device,
                checkpoint_id=result["checkpoint_id"],
                max_seq_len=MAX_SEQ_LEN,
                global_row_shapes=(GLOBAL_BATCH,),
                slots_per_shape=4,
                progress_every=args.progress_every,
                progress=(
                    (lambda message: print(f"[E0qf] {message}", flush=True))
                    if rank in (0, 4)
                    else None
                ),
            )

        stage_material = synchronized_local_step(
            "load materials", load_materials, device=device, world=world
        )
        result["diagnostic_seconds"]["load"] = time.perf_counter() - phase_started
        material_by_layer = {
            material.layer_id: material for material in stage_material.materials
        }
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_after_load"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        if rank in (0, 4):
            print(
                f"[E0qf] stage {stage_id} materials loaded, "
                f"free {free_bytes / 2**30:.2f} GiB",
                flush=True,
            )

        # ------------------------------------------------------------------
        # diagnostic probe (evidence only)
        result["group_probe"] = synchronized_local_step(
            "group determinism probe",
            lambda: run_group_determinism_probe(
                groups=groups, seed=args.seed, device=device, world=world
            ),
            device=device,
            world=world,
        )
        if rank == 0:
            print(
                "[E0qf] cross-machine collective probe bitwise="
                f"{result['group_probe']['cross_group_bitwise']}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # build lanes.  Seed payloads are shared between the chain lane, the
        # pipeline lane, and the self-consistency replica lane of the same
        # (tp_rank, microbatch, layer) stream, and identical across the pair
        # (both sides derive from tp_rank).
        phase_started = time.perf_counter()

        def build_lanes() -> dict[str, list[StageLane]]:
            payloads = {
                microbatch: {
                    layer_id: build_seed_payload(
                        material_by_layer[layer_id],
                        seed=args.seed,
                        tp_rank=tp_rank,
                        microbatch=microbatch,
                        device=device,
                    )
                    for layer_id in stage_material.layer_ids
                }
                for microbatch in range(microbatches)
            }
            lanes: dict[str, list[StageLane]] = {
                "pipeline": [],
                "replica": [],
                "chain": [],
            }
            own_stage_layers = (
                STAGE0_LAYER_IDS if stage_id == 0 else STAGE1_LAYER_IDS
            )
            for kind in ("pipeline", "replica"):
                for microbatch in range(microbatches):
                    lanes[kind].append(
                        StageLane(
                            label=f"{kind}-stage{stage_id}-mb{microbatch}",
                            materials=[
                                material_by_layer[layer_id]
                                for layer_id in own_stage_layers
                            ],
                            payloads=[
                                payloads[microbatch][layer_id]
                                for layer_id in own_stage_layers
                            ],
                            device=device,
                        )
                    )
            if stage_id == 0:
                for microbatch in range(microbatches):
                    lanes["chain"].append(
                        StageLane(
                            label=f"chain-mb{microbatch}",
                            materials=[
                                material_by_layer[layer_id]
                                for layer_id in CHAIN_LAYER_IDS
                            ],
                            payloads=[
                                payloads[microbatch][layer_id]
                                for layer_id in CHAIN_LAYER_IDS
                            ],
                            device=device,
                        )
                    )
            return lanes

        lanes = synchronized_local_step(
            "build lanes", build_lanes, device=device, world=world
        )
        result["diagnostic_seconds"]["build"] = time.perf_counter() - phase_started
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_after_build"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        if rank in (0, 4):
            print(
                f"[E0qf] lanes built, free {free_bytes / 2**30:.2f} GiB",
                flush=True,
            )

        # Pipeline and replica lanes of one stream must start from identical
        # KV bytes on every rank; on stage 0 the chain lane must match too
        # (locally verifiable for L0-L5).
        for microbatch in range(microbatches):
            pipeline_digests = lanes["pipeline"][microbatch].state_digests()
            replica_digests = lanes["replica"][microbatch].state_digests()
            if pipeline_digests != replica_digests:
                raise RuntimeError(
                    f"mb{microbatch} pipeline/replica lanes were not seeded "
                    "identically"
                )
            if stage_id == 0:
                chain_digests = lanes["chain"][microbatch].state_digests()
                for layer_id in STAGE0_LAYER_IDS:
                    if chain_digests[str(layer_id)] != pipeline_digests[
                        str(layer_id)
                    ]:
                        raise RuntimeError(
                            f"mb{microbatch} layer {layer_id} chain/pipeline "
                            "lanes were not seeded identically"
                        )

        # ------------------------------------------------------------------
        # chain phase (reference; stage-0 group only)
        phase_started = time.perf_counter()
        chain = synchronized_local_step(
            "chain phase",
            lambda: (
                run_chain_phase(
                    chain_lanes=lanes["chain"],
                    seed=args.seed,
                    tp_rank=tp_rank,
                    device=device,
                )
                if stage_id == 0
                else None
            ),
            device=device,
            world=world,
        )
        result["diagnostic_seconds"]["chain"] = time.perf_counter() - phase_started
        if rank == 0:
            print("[E0qf] chain phase complete", flush=True)

        # ------------------------------------------------------------------
        # pipeline phase (all ranks): primary run, then the same-path
        # self-consistency replica run on independently seeded-identical
        # lanes (cross-machine determinism evidence).
        runs: dict[str, dict[str, Any]] = {}
        for run_label in ("primary", "replica"):
            phase_started = time.perf_counter()
            lane_key = "pipeline" if run_label == "primary" else "replica"
            runs[run_label] = synchronized_local_step(
                f"pipeline phase ({run_label})",
                lambda lane_key=lane_key, run_label=run_label: run_pipeline_phase(
                    label=run_label,
                    stage_lanes=lanes[lane_key],
                    groups=groups,
                    seed=args.seed,
                    device=device,
                ),
                device=device,
                world=world,
            )
            result["diagnostic_seconds"][f"pipeline_{run_label}"] = (
                time.perf_counter() - phase_started
            )
            result_key = "pipeline" if run_label == "primary" else "pipeline_replica"
            result[result_key] = {
                "handoff_records": runs[run_label]["handoff_records"],
                "terminals": runs[run_label]["terminals"],
                "timing": runs[run_label]["timing"],
                "timing_first8": runs[run_label]["timing_first8"],
            }
            if rank == 0:
                print(f"[E0qf] pipeline phase ({run_label}) complete", flush=True)
        pipeline = runs["primary"]
        replica = runs["replica"]

        # Same-path self-consistency, judged locally on each rank: payload
        # SHAs (both stages), and on stage 1 additionally the exit tensors
        # and terminal KV digests, must match bitwise between the two runs.
        def self_consistency() -> dict[str, Any]:
            record: dict[str, Any] = {
                "judgment": (
                    "bitwise primary-vs-replica on the identical "
                    "cross-machine path"
                ),
                "payload_shas_equal": pipeline["payload_shas"]
                == replica["payload_shas"],
                "kv_digests_equal": pipeline["state_digests"]
                == replica["state_digests"],
            }
            if stage_id == 1:
                record["stage1_outputs_equal"] = all(
                    bool(
                        torch.equal(
                            pipeline["stage1_outputs"][microbatch],
                            replica["stage1_outputs"][microbatch],
                        )
                    )
                    for microbatch in range(microbatches)
                )
            record["accepted"] = all(
                value
                for key, value in record.items()
                if key != "judgment"
            )
            return record

        result["self_consistency"] = synchronized_local_step(
            "self consistency", self_consistency, device=device, world=world
        )
        if rank in (0, 4):
            print(
                f"[E0qf] stage{stage_id} self-consistency accepted="
                f"{result['self_consistency']['accepted']}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # comparisons (primary run vs chain)
        stage1_outputs_on_stage0 = synchronized_local_step(
            "exchange stage1 outputs",
            lambda: exchange_stage1_outputs(
                pipeline=pipeline,
                groups=groups,
                microbatches=microbatches,
                device=device,
            ),
            device=device,
            world=world,
        )

        def compare() -> dict[str, Any]:
            comparison: dict[str, Any] = {
                "judgment": (
                    "stage0 bitwise; stage1 bitwise when it holds, else "
                    "numeric gate + self-consistency (cross-machine mode "
                    "split, task contract)"
                ),
            }
            # payload sha parity sender vs receiver, exchanged world-wide.
            record = {
                "rank": rank,
                "stage_id": stage_id,
                "payload_shas": pipeline["payload_shas"],
                "replica_payload_shas": replica["payload_shas"],
                "state_digests": pipeline["state_digests"],
                "self_consistency": result["self_consistency"],
                "chain_state_digests": (
                    chain["state_digests"] if stage_id == 0 else None
                ),
            }
            gathered: list[Any] = [None] * world
            dist.all_gather_object(gathered, record)
            comparison["self_consistency_all"] = all(
                entry["self_consistency"]["accepted"] for entry in gathered
            )
            if stage_id == 0:
                peer = gathered[rank + EXPECTED_TP_SIZE]
                comparison["payload_sha_parity"] = (
                    pipeline["payload_shas"] == peer["payload_shas"]
                )
                comparison["replica_payload_sha_parity"] = (
                    replica["payload_shas"] == peer["replica_payload_shas"]
                )
                # Stage-0 exit vs chain after-L5: the sent payload sha *is*
                # the stage-0 output sha, so bitwise equality holds iff the
                # chain record's sha matches it.  Stage-1 exit compares the
                # bulk-returned device tensors directly.
                mid_exact: list[bool] = []
                final_exact: list[bool] = []
                final_finite: list[bool] = []
                worst_final: dict[str, Any] | None = None
                max_abs_final = 0.0
                max_rms_rel_final = 0.0
                for microbatch in range(microbatches):
                    sent_shas = pipeline["payload_shas"][microbatch]
                    for step_index in range(STEP_COUNT):
                        chain_mid = chain["mid_records"][microbatch][step_index]
                        mid_ok = tensor_sha256(chain_mid) == sent_shas[step_index]
                        mid_exact.append(mid_ok)
                        chain_final = chain["final_records"][microbatch][
                            step_index
                        ]
                        observed_final = stage1_outputs_on_stage0[microbatch][
                            step_index
                        ]
                        metrics = error_metrics(observed_final, chain_final)
                        final_exact.append(metrics["bitwise_exact"])
                        final_finite.append(metrics["finite"])
                        max_abs_final = max(max_abs_final, metrics["max_abs"])
                        max_rms_rel_final = max(
                            max_rms_rel_final, metrics["rms_rel"]
                        )
                        if not metrics["bitwise_exact"] and worst_final is None:
                            worst_final = {
                                "microbatch": microbatch,
                                "step_index": step_index,
                                "position": SCHEDULE[step_index].position,
                                "metrics": metrics,
                            }
                comparison["stage0_exit_bitwise_steps"] = sum(mid_exact)
                comparison["stage0_exit_all_bitwise"] = all(mid_exact)
                comparison["stage1_exit_bitwise_steps"] = sum(final_exact)
                comparison["stage1_exit_all_bitwise"] = all(final_exact)
                comparison["stage1_exit_all_finite"] = all(final_finite)
                comparison["stage1_exit_max_abs"] = max_abs_final
                comparison["stage1_exit_max_rms_rel"] = max_rms_rel_final
                comparison["first_stage1_mismatch"] = worst_final
                # KV digest parity per microbatch.
                kv = []
                for microbatch in range(microbatches):
                    chain_digests = chain["state_digests"][microbatch]
                    stage0_digests = pipeline["state_digests"][microbatch]
                    peer_digests = peer["state_digests"][microbatch]
                    stage0_ok = all(
                        chain_digests[str(layer_id)]
                        == stage0_digests[str(layer_id)]
                        for layer_id in STAGE0_LAYER_IDS
                    )
                    stage1_ok = all(
                        chain_digests[str(layer_id)]
                        == peer_digests[str(layer_id)]
                        for layer_id in STAGE1_LAYER_IDS
                    )
                    kv.append(
                        {
                            "microbatch": microbatch,
                            "stage0_kv_parity": stage0_ok,
                            "stage1_kv_parity": stage1_ok,
                        }
                    )
                comparison["kv_terminal_parity"] = kv
                comparison["stage0_kv_all_parity"] = all(
                    entry["stage0_kv_parity"] for entry in kv
                )
                comparison["stage1_kv_all_parity"] = all(
                    entry["stage1_kv_parity"] for entry in kv
                )
                comparison["chain_terminals_accepted"] = all(
                    terminal["accepted"] for terminal in chain["terminals"]
                )
                # Cross-machine mode split (task contract): bitwise when it
                # holds; otherwise close via the numeric gate, backed by the
                # replica self-consistency gate and the collective probe as
                # attribution evidence.
                if (
                    comparison["stage1_exit_all_bitwise"]
                    and comparison["stage1_kv_all_parity"]
                ):
                    comparison["stage1_judgment_mode"] = "bitwise"
                    comparison["stage1_accepted"] = True
                else:
                    comparison["stage1_judgment_mode"] = (
                        "numeric_gate_with_self_consistency"
                    )
                    comparison["stage1_accepted"] = bool(
                        comparison["stage1_exit_all_finite"]
                        and comparison["stage1_exit_max_rms_rel"]
                        <= args.stage1_rms_rel_tol
                        and comparison["self_consistency_all"]
                    )
            comparison["pipeline_terminals_accepted"] = all(
                terminal["accepted"]
                for run in (pipeline, replica)
                for terminal in run["terminals"]
            )
            comparison["handoff_accepted"] = all(
                entry["accepted"]
                for run in (pipeline, replica)
                for entry in run["handoff_records"]
            )
            return comparison

        result["comparison"] = synchronized_local_step(
            "comparison", compare, device=device, world=world
        )

        if stage_id == 0:
            comparison = result["comparison"]
            result["accepted"] = bool(
                result["placement"]["accepted"]
                and comparison["stage0_exit_all_bitwise"]
                and comparison["stage0_kv_all_parity"]
                and comparison["stage1_accepted"]
                and comparison["payload_sha_parity"]
                and comparison["replica_payload_sha_parity"]
                and comparison["self_consistency_all"]
                and comparison["chain_terminals_accepted"]
                and comparison["pipeline_terminals_accepted"]
                and comparison["handoff_accepted"]
            )
            if rank == 0:
                print(
                    "[E0qf] stage1 judgment mode: "
                    f"{comparison['stage1_judgment_mode']} "
                    f"(bitwise steps {comparison['stage1_exit_bitwise_steps']}"
                    f"/{microbatches * STEP_COUNT}, "
                    f"max_abs {comparison['stage1_exit_max_abs']:.3e}, "
                    f"max_rms_rel {comparison['stage1_exit_max_rms_rel']:.3e})",
                    flush=True,
                )
        else:
            result["accepted"] = bool(
                result["placement"]["accepted"]
                and result["comparison"]["self_consistency_all"]
                and result["comparison"]["pipeline_terminals_accepted"]
                and result["comparison"]["handoff_accepted"]
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
    write_json(out_dir / f"rank{rank}.json", result)
    if rank == 0:
        write_json(
            out_dir / "result.json",
            {
                "experiment": "E0qf-tp4-pp2-dual-node-serial-pipeline",
                "config_tag": args.config_tag,
                "accepted": accepted_all,
                "ranks": gathered,
            },
        )
        print(f"[E0qf] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
