#!/usr/bin/env python3
"""E0pf: scaled TP4 x PP2 pipeline decode gate (V4-Flash, titan064 8-GPU).

Eighth port vertical: gaiban's PP pipeline surface ported to Flash and
exercised on one machine as two whole-layer TP4 stages -- stage 0 = layers
0-5 on GPUs 0-3, stage 1 = layers 6-11 on GPUs 4-7 -- with real weights and
a serial fixed-endpoint NCCL P2P handoff (``dsv4_direct.pipeline_transport``,
E1b2a/E1b2z default-NCCL lineage; the rejected gaiban IPC/timeline
transports are documented there).  Stateful eager decode (gaiban E1b2b
class) over the E0sf schedule [8192, 8324): 132 steps covering 99 NORMAL,
32 RATIO4_BOUNDARY, and the RATIO4_RATIO128_BOUNDARY step at 8319.

Serial handoff, microbatches: the schedule runs ``--microbatches`` (default
2) independent single-sequence streams per step through the same stage
materials on isolated KV/cursor lanes.  With a serial handoff exactly one
payload is in flight; pipelined overlap (double-buffered lanes, E1b2d) is a
deliberately deferred performance treatment, so the two microbatches
interleave serially.  This is stated per the task contract.

Acceptance gates (judged bitwise -- both sides are the deterministic decode
path; E0df/E0sf precedent):

(a) **Pipeline vs sequential chain.**  A 12-layer sequential chain
    (E0sf chain mode: same eager stateful per-block composition) runs on
    the stage-0 TP4 group with identically seeded lanes.  Per step and per
    microbatch, the pipeline stage-0 exit hidden must equal the chain
    after-L5 hidden, and the pipeline stage-1 exit hidden must equal the
    chain after-L11 hidden, bitwise.  Note the chain runs its layers 6-11
    on GPUs 0-3 while the pipeline runs them on GPUs 4-7; a cross-group
    NCCL determinism probe (reduce_scatter + all_gather on identical
    inputs on both TP groups) is recorded as diagnostic evidence for this
    comparison.
(b) **KV terminal digests.**  After 132 steps the full per-layer KV state
    SHA-256 digests must match between the chain lane (layers 0-11) and
    the pipeline lanes (stage 0: 0-5 locally; stage 1: 6-11 across the
    pair), per microbatch.
(c) **Handoff contract.**  The boundary tensor is frozen at
    ``[local_batch, 1, 4, 4096]`` BF16; endpoints must stay
    pointer-stable for the whole cycle and every step's payload SHA-256
    must match sender-to-receiver.  Stage 1 receives into a fixed staging
    endpoint and D2D-unpacks into the plan input buffer (gaiban E1b2z
    ``stage1_d2d_unpack`` form; the stateful validation contract forbids
    validating the plan's own input buffer as the external input).
(d) **Pre-MoE fragment parity.**  The ported PP fragment-split surface
    (``DirectPreMoEBlockFragment`` -> portable ``StatefulPreMoEBundle`` ->
    transport-storage roundtrip -> a *different* full block's
    ``finish_stateful_decode_from_pre_moe``) must reproduce the full-block
    stateful forward bitwise on one layer of each kind (0 window/hash,
    6 ratio-4/learned, 7 ratio-128/learned).

Timing (report-only, for the t_stage model): per-step stage-0 compute,
sender-observed handoff wall, receiver recv wall, stage-1 compute.

Run (titan064):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=8 \
    e0pf_pp2_gate.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0pf
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

from dsv4_direct.block import (
    DirectDecodeBlock,
    StatefulPreMoEBundle,
)
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
    classify_decode_position,
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

FRAGMENT_LAYER_IDS = (0, 6, 7)  # window/hash, ratio-4/learned, ratio-128/learned


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
            f"[E0pf][rank {dist.get_rank()}] {name} raised:\n{local_error}",
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
    order depends on the NCCL ring over each group's physical GPUs.  This
    probe is the direct evidence for whether chain layers 6-11 on GPUs 0-3
    can be compared bitwise against pipeline stage 1 on GPUs 4-7.
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
# part (d): pre-MoE fragment parity (ported PP fragment-split surface)


def run_fragment_parity(
    *,
    chain_material_by_layer: Mapping[int, PhysicalLayerMaterial],
    seed: int,
    tp_rank: int,
    device: torch.device,
) -> dict[str, Any]:
    family = classify_decode_position(START_POSITION)
    result: dict[str, Any] = {
        "position": START_POSITION,
        "family": family.value,
        "judgment": "bitwise_fragment_vs_full_block (ported PP fragment ABI)",
        "layers": {},
        "accepted": False,
    }
    all_exact = True
    for layer_id in FRAGMENT_LAYER_IDS:
        material = chain_material_by_layer[layer_id]
        payload = build_seed_payload(
            material, seed=seed, tp_rank=tp_rank, microbatch=0, device=device
        )
        # Lane X: full block, the bitwise reference (and the consumer of the
        # fragment bundle -- a distinct instance from the producer).
        full_block = new_seeded_block(material, payload)
        # Lane Y: producer-only fragment over identically seeded state.
        fragment_state = material.new_state(num_local_sequences=LOCAL_BATCH)
        seed_state(material, fragment_state, payload)
        fragment = material.new_fragment(fragment_state)

        residual = deterministic_residual(
            seed=seed + 17,
            tp_rank=tp_rank,
            microbatch=0,
            position=START_POSITION + layer_id,
            device=device,
        )
        ids = deterministic_input_ids(
            seed=seed + 17,
            tp_rank=tp_rank,
            microbatch=0,
            position=START_POSITION + layer_id,
            device=device,
        )
        ids_argument = ids if material.route_kind == "hash" else None

        cursor_x = StatefulDecodeCursor(
            start_position=START_POSITION, device=device
        )
        plan_x = full_block.attention.prepare_stateful_decode_plan(
            position=cursor_x.device_position,
            start_position=START_POSITION,
            stop_position=START_POSITION + 1,
        )
        output_x = full_block.forward_stateful_decode_tensor(
            residual,
            input_ids_local=ids_argument,
            attention_plan=plan_x,
            graph_family=family,
            moe_slot=EAGER_MOE_SLOT,
        )
        torch.cuda.synchronize(device)

        cursor_y = StatefulDecodeCursor(
            start_position=START_POSITION, device=device
        )
        plan_y = fragment.attention.prepare_stateful_decode_plan(
            position=cursor_y.device_position,
            start_position=START_POSITION,
            stop_position=START_POSITION + 1,
        )
        bundle = fragment.prepare_stateful_decode_pre_moe(
            residual,
            input_ids_local=ids_argument,
            attention_plan=plan_y,
            graph_family=family,
        )
        # Transport-ABI roundtrip: pack the producer bundle into one owning
        # byte buffer (the PP boundary layout), then finish from the packed
        # views on a different consumer instance.
        transport = StatefulPreMoEBundle.allocate_transport(
            local_batch=LOCAL_BATCH,
            sequence=1,
            device=device,
            layer_id=material.layer_id,
            rank=tp_rank,
            world_size=EXPECTED_TP_SIZE,
            checkpoint_id=material.checkpoint_id,
            producer_owner_id=id(fragment),
        )
        transport.after_attention.copy_(bundle.after_attention)
        transport.ffn_hidden.copy_(bundle.ffn_hidden)
        transport.ffn_post.copy_(bundle.ffn_post)
        transport.ffn_comb.copy_(bundle.ffn_comb)
        output_y = full_block.finish_stateful_decode_from_pre_moe(
            transport,
            input_ids_local=ids_argument,
            moe_slot=EAGER_MOE_SLOT,
        )
        torch.cuda.synchronize(device)

        metrics = error_metrics(output_y, output_x)
        state_parity = full_state_sha256(fragment_state) == full_state_sha256(
            full_block.attention.state
        )
        transport_nbytes = StatefulPreMoEBundle.required_transport_nbytes(
            LOCAL_BATCH, 1
        )
        record = {
            "kind": material.kind,
            "route_kind": material.route_kind,
            "output": metrics,
            "state_digest_parity": state_parity,
            "producer_id_differs_from_consumer": id(fragment) != id(full_block),
            "transport_nbytes": transport_nbytes,
        }
        result["layers"][str(layer_id)] = record
        all_exact = all_exact and metrics["bitwise_exact"] and state_parity
    result["accepted"] = bool(all_exact)
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
    }

    for step_index, step in enumerate(SCHEDULE):
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
        if tp_rank == 0 and step_index % 32 == 0:
            print(
                f"[E0pf] stage{stage_id} pipeline step {step_index} "
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
        "experiment": "E0pf-tp4-pp2-serial-pipeline",
        "measurement_class": "semantic_correctness_gate_plus_reportonly_timing",
        "rank": rank,
        "local_rank": local_rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "seed": args.seed,
        "microbatches": microbatches,
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
        "group_probe": None,
        "fragment_parity": None,
        "pipeline": None,
        "comparison": None,
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }

    started = time.perf_counter()
    try:
        if world != PP2_WORLD:
            raise ValueError(f"E0pf requires world=8 (TP4 x PP2), got {world}")
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
                    (lambda message: print(f"[E0pf] {message}", flush=True))
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
                f"[E0pf] stage {stage_id} materials loaded, "
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
                "[E0pf] cross-group collective probe bitwise="
                f"{result['group_probe']['cross_group_bitwise']}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # part (d): fragment parity (stage-0 group only)
        phase_started = time.perf_counter()
        result["fragment_parity"] = synchronized_local_step(
            "fragment parity",
            lambda: (
                run_fragment_parity(
                    chain_material_by_layer=material_by_layer,
                    seed=args.seed,
                    tp_rank=tp_rank,
                    device=device,
                )
                if stage_id == 0
                else {"skipped_on_stage1": True, "accepted": True}
            ),
            device=device,
            world=world,
        )
        result["diagnostic_seconds"]["fragment"] = (
            time.perf_counter() - phase_started
        )
        if rank == 0:
            status = "PASS" if result["fragment_parity"]["accepted"] else "FAIL"
            print(f"[E0pf] part (d) fragment parity: {status}", flush=True)

        # ------------------------------------------------------------------
        # build lanes.  Seed payloads are shared between the chain lane and
        # the pipeline lane of the same (tp_rank, microbatch, layer) stream,
        # and identical across the pair (both sides derive from tp_rank).
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
            lanes: dict[str, list[StageLane]] = {"pipeline": [], "chain": []}
            own_stage_layers = (
                STAGE0_LAYER_IDS if stage_id == 0 else STAGE1_LAYER_IDS
            )
            for microbatch in range(microbatches):
                lanes["pipeline"].append(
                    StageLane(
                        label=f"stage{stage_id}-mb{microbatch}",
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
                f"[E0pf] lanes built, free {free_bytes / 2**30:.2f} GiB",
                flush=True,
            )

        # Chain and pipeline lanes of one stream must start from identical
        # KV bytes (stage-0 side can verify locally for L0-L5).
        if stage_id == 0:
            for microbatch in range(microbatches):
                chain_digests = lanes["chain"][microbatch].state_digests()
                stage0_digests = lanes["pipeline"][microbatch].state_digests()
                for layer_id in STAGE0_LAYER_IDS:
                    if chain_digests[str(layer_id)] != stage0_digests[str(layer_id)]:
                        raise RuntimeError(
                            f"mb{microbatch} layer {layer_id} chain/pipeline lanes "
                            "were not seeded identically"
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
            print("[E0pf] chain phase complete", flush=True)

        # ------------------------------------------------------------------
        # pipeline phase (all ranks)
        phase_started = time.perf_counter()
        pipeline = synchronized_local_step(
            "pipeline phase",
            lambda: run_pipeline_phase(
                stage_lanes=lanes["pipeline"],
                groups=groups,
                seed=args.seed,
                device=device,
            ),
            device=device,
            world=world,
        )
        result["diagnostic_seconds"]["pipeline"] = (
            time.perf_counter() - phase_started
        )
        result["pipeline"] = {
            "handoff_records": pipeline["handoff_records"],
            "terminals": pipeline["terminals"],
            "timing": pipeline["timing"],
            "timing_first8": pipeline["timing_first8"],
        }
        if rank == 0:
            print("[E0pf] pipeline phase complete", flush=True)

        # ------------------------------------------------------------------
        # comparisons
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
                "judgment": "bitwise (deterministic decode path both sides)",
            }
            # payload sha parity sender vs receiver, exchanged world-wide.
            record = {
                "rank": rank,
                "stage_id": stage_id,
                "payload_shas": pipeline["payload_shas"],
                "state_digests": pipeline["state_digests"],
                "chain_state_digests": (
                    chain["state_digests"] if stage_id == 0 else None
                ),
            }
            gathered: list[Any] = [None] * world
            dist.all_gather_object(gathered, record)
            if stage_id == 0:
                peer = gathered[rank + EXPECTED_TP_SIZE]
                comparison["payload_sha_parity"] = (
                    pipeline["payload_shas"] == peer["payload_shas"]
                )
                # Stage-0 exit vs chain after-L5: the sent payload sha *is*
                # the stage-0 output sha, so bitwise equality holds iff the
                # chain record's sha matches it.  Stage-1 exit compares the
                # bulk-returned device tensors directly.
                mid_exact: list[bool] = []
                final_exact: list[bool] = []
                worst_final: dict[str, Any] | None = None
                max_abs_final = 0.0
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
                        max_abs_final = max(max_abs_final, metrics["max_abs"])
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
                comparison["stage1_exit_max_abs"] = max_abs_final
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
                comparison["kv_all_parity"] = all(
                    entry["stage0_kv_parity"] and entry["stage1_kv_parity"]
                    for entry in kv
                )
                comparison["chain_terminals_accepted"] = all(
                    terminal["accepted"] for terminal in chain["terminals"]
                )
            comparison["pipeline_terminals_accepted"] = all(
                terminal["accepted"] for terminal in pipeline["terminals"]
            )
            comparison["handoff_accepted"] = all(
                entry["accepted"] for entry in pipeline["handoff_records"]
            )
            return comparison

        result["comparison"] = synchronized_local_step(
            "comparison", compare, device=device, world=world
        )

        if stage_id == 0:
            comparison = result["comparison"]
            result["accepted"] = bool(
                result["fragment_parity"]["accepted"]
                and comparison["stage0_exit_all_bitwise"]
                and comparison["stage1_exit_all_bitwise"]
                and comparison["payload_sha_parity"]
                and comparison["kv_all_parity"]
                and comparison["chain_terminals_accepted"]
                and comparison["pipeline_terminals_accepted"]
                and comparison["handoff_accepted"]
            )
        else:
            result["accepted"] = bool(
                result["comparison"]["pipeline_terminals_accepted"]
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
                "experiment": "E0pf-tp4-pp2-serial-pipeline",
                "accepted": accepted_all,
                "ranks": gathered,
            },
        )
        print(f"[E0pf] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
