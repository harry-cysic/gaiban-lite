#!/usr/bin/env python3
"""E0sf: multi-layer decode super-stage + stateful CUDA-graph gate (V4-Flash).

Sixth port vertical: gaiban's ``TP4DecodeStage`` multi-layer decode
(superstage.py), graph-family scheduling (stateful_decode.py), and CUDA-graph
lifecycle (stateful_graph.py) ported to Flash, driven with real weights over
the Flash canonical L0-L5 slice (window x2 / ratio-4 x2 / ratio-128 x2 --
every layer type and every graph-family boundary combination).

Two acceptance gates, judged per gaiban precedent:

(a) **Eager super-stage composition** (gaiban E0d / lite E0df class):
    ``TP4DecodeStage.forward_decode_tensors`` over L0-L5 must be **bitwise
    equal**, per layer and per position, to a manual chain of
    ``DirectDecodeBlock.forward_decode_tensor`` calls on an independent,
    identically seeded block set.  (E0df already established the decode path
    is deterministic and used bitwise equality for assembly parity; the
    superstage adds no arithmetic, so its gate is also bitwise.)

(b) **Stateful graph replay vs eager stateful** (gaiban E1a27 judgment,
    ``graph_vs_eager_output: bitwise_primary_plus_numeric_witness``):
    after capturing the three graph families, each of the 132 schedule steps
    replays the family graph on one lane and runs the eager stateful body on
    a second identically seeded lane; per-step outputs must be **bitwise
    equal** (``torch.equal`` primary) with finite numeric witnesses
    (max_abs / rms_rel recorded).  Cursor/state lifecycle checks follow
    E1a27: per-step device-cursor and state next-position advancement,
    end-of-run full-state digest parity between the lanes, and an accepted
    ``teardown_stateful_graphs`` evidence record.

Schedule: positions [8192, 8324), the E1a27 window re-validated for Flash:
ratio-4 saturation needs (start+1)//4 >= index_topk = 512 (=> start >= 2047,
8192 kept for continuity), and the range covers 99 NORMAL steps, 32
RATIO4_BOUNDARY steps, and the RATIO4_RATIO128_BOUNDARY step at 8319
(8319 % 128 == 127).  Family set derivation for Flash (why no window family
exists) is documented in ``dsv4_direct/stateful_decode.py``.

Initial KV state is deterministic seeded residency, not a real prefix
(E1a27: "deterministic_zero_seeded_kv_not_real_prefix"), via the layer-type
seeders: ``StaticWindowKV.seed_decode_residency`` (new, ring-only),
``StaticLayerKV.seed_decode_residency`` (E0-verified), and
``ratio4_oracle.seed_nonzero_ratio4_state`` + ``seed_decode_payload``
(E0ff/E0df-verified).

Run (titan064):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=4 \
    e0sf_superstage_gate.py \
    --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0sf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from dsv4_direct.attention import (
    Ratio128AttentionConfig,
    Ratio128TorchAttention,
    prepare_attention_weights,
)
from dsv4_direct.block import DirectDecodeBlock
from dsv4_direct.block_weights import (
    inspect_replicated_block_contract,
    load_replicated_block_weights,
)
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.moe_runtime import TP4MoE, TP4MoEConfig
from dsv4_direct.ops.marlin_moe import load_resident_moe_layer
from dsv4_direct.ratio4_attention import (
    Ratio4AttentionConfig,
    Ratio4TorchAttention,
    prepare_ratio4_attention_weights,
)
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
    SUPERSTAGE_LAYER_IDS,
    TP4DecodeStage,
    TP4StatefulDecodeSuperStagePlan,
)
from dsv4_direct.window_attention import (
    WindowAttentionConfig,
    WindowTorchAttention,
    prepare_window_attention_weights,
)


EXPECTED_WORLD = 4
EXPECTED_VOCAB = 129280
LOCAL_BATCH = 1
GLOBAL_BATCH = LOCAL_BATCH * EXPECTED_WORLD
MAX_SEQ_LEN = 8448

LAYER_IDS = SUPERSTAGE_LAYER_IDS  # (0, 1, 2, 3, 4, 5)

START_POSITION = 8192
STEP_COUNT = 132
STOP_POSITION = START_POSITION + STEP_COUNT
SCHEDULE = build_decode_schedule(START_POSITION, STEP_COUNT)
FAMILY_COUNTS = schedule_family_counts(SCHEDULE)

EAGER_POSITIONS = (8192, 8193, 8194)

GRAPH_MOE_SLOTS: dict[DecodeGraphFamily, int] = {
    DecodeGraphFamily.NORMAL: 1,
    DecodeGraphFamily.RATIO4_BOUNDARY: 2,
    DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY: 3,
}
GRAPH_MOE_SLOT_TUPLE = tuple(
    GRAPH_MOE_SLOTS[family] for family in DecodeGraphFamily
)
EAGER_MOE_SLOT = 0

# Same per-layer resident budget frozen by the E0df titan064 runs.
EXPECTED_MOE_RESIDENT_BYTES = 861_931_008


# --------------------------------------------------------------------------
# generic helpers (E0df process form)


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


def deterministic_residual(
    *, seed: int, rank: int, position: int, device: torch.device
) -> torch.Tensor:
    return deterministic_tensor(
        seed=(seed * 1_000_003 + rank * 100_003 + position * 7_919)
        & ((1 << 62) - 1),
        shape=(LOCAL_BATCH, 1, 4, 4096),
        device=device,
    )


def deterministic_input_ids(
    *, seed: int, rank: int, position: int, device: torch.device
) -> torch.Tensor:
    mixed = (seed * 2654435761 + rank * 1000003 + position * 7919) & ((1 << 63) - 1)
    return torch.full(
        (LOCAL_BATCH, 1), mixed % EXPECTED_VOCAB, dtype=torch.int64, device=device
    )


def error_metrics(observed: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """E1a27 ``error_metrics``: bitwise primary plus numeric witness."""

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


def clone_state(source: DirectState) -> DirectState:
    if isinstance(source, StaticWindowKV):
        result: DirectState = StaticWindowKV(
            num_local_sequences=source.num_local_sequences,
            max_seq_len=source.max_seq_len,
            layer_id=source.layer_id,
            device=source.device,
        )
    elif isinstance(source, StaticRatio4KV):
        result = StaticRatio4KV(
            num_local_sequences=source.num_local_sequences,
            max_seq_len=source.max_seq_len,
            layer_id=source.layer_id,
            device=source.device,
        )
    elif isinstance(source, StaticLayerKV):
        result = StaticLayerKV(
            num_local_sequences=source.num_local_sequences,
            max_seq_len=source.max_seq_len,
            layer_id=source.layer_id,
            device=source.device,
        )
    else:
        raise TypeError("unsupported direct state type")
    result.copy_from(source)  # type: ignore[arg-type]
    return result


def copy_stage_states(
    destination: Sequence[DirectState], source: Sequence[DirectState]
) -> None:
    if len(destination) != len(LAYER_IDS) or len(source) != len(LAYER_IDS):
        raise ValueError("state sets must cover L0-L5")
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


def state_next_positions_equal(
    left: Sequence[DirectState], right: Sequence[DirectState]
) -> bool:
    return all(
        torch.equal(a._next_position, b._next_position)
        for a, b in zip(left, right, strict=True)
    )


# --------------------------------------------------------------------------
# layer build + seeding


class LayerAssets:
    """Per-layer shared material: raw block weights, prepared attention, MoE."""

    def __init__(
        self,
        *,
        layer_id: int,
        model_config: Mapping[str, Any],
        stage_root: Path,
        rank: int,
        world: int,
        checkpoint_id: str,
        device: torch.device,
        progress_every: int,
    ) -> None:
        self.layer_id = layer_id
        self.device = device
        self.raw_block = load_replicated_block_weights(
            stage_root=stage_root,
            rank=rank,
            world_size=world,
            layer_id=layer_id,
            device=device,
            checkpoint_id=checkpoint_id,
        )
        moe_resident = load_resident_moe_layer(
            stage_root=stage_root,
            layer_id=layer_id,
            rank=rank,
            world_size=world,
            hidden_size=int(model_config["hidden_size"]),
            intermediate_size=int(model_config["moe_intermediate_size"]),
            n_experts=int(model_config["n_routed_experts"]),
            device=device,
            progress_every=progress_every,
            progress=(
                (lambda message: print(message, flush=True)) if rank == 0 else None
            ),
            checkpoint_id=checkpoint_id,
        )
        if moe_resident.resident_bytes != EXPECTED_MOE_RESIDENT_BYTES:
            raise ValueError(
                f"layer-{layer_id} MoE resident bytes {moe_resident.resident_bytes} "
                f"!= {EXPECTED_MOE_RESIDENT_BYTES}"
            )
        self.moe = TP4MoE(
            config=TP4MoEConfig(
                hidden_size=int(model_config["hidden_size"]),
                intermediate_size=int(model_config["moe_intermediate_size"]),
                experts=int(model_config["n_routed_experts"]),
                topk=int(model_config["num_experts_per_tok"]),
                route_scale=float(model_config["routed_scaling_factor"]),
                clamp_limit=float(model_config["swiglu_limit"]),
                world_size=world,
            ),
            resident=moe_resident,
            gate=self.raw_block.gate,
            rank=rank,
            device=device,
            global_row_shapes=(GLOBAL_BATCH,),
            slots_per_shape=4,
        )
        identity = {
            "layer_id": layer_id,
            "rank": rank,
            "world_size": world,
            "checkpoint_id": checkpoint_id,
        }
        kind = (
            "window" if layer_id < 2 else ("ratio4" if layer_id % 2 == 0 else "ratio128")
        )
        self.kind = kind
        if kind == "window":
            self.config: Any = WindowAttentionConfig.from_model_config(
                model_config, layer_id=layer_id, max_seq_len=MAX_SEQ_LEN
            )
            self.prepared = prepare_window_attention_weights(
                self.raw_block.attention, **identity
            )
        elif kind == "ratio4":
            self.config = Ratio4AttentionConfig.from_model_config(
                model_config, layer_id=layer_id, max_seq_len=MAX_SEQ_LEN
            )
            self.prepared = prepare_ratio4_attention_weights(
                self.raw_block.attention, **identity
            )
        else:
            self.config = Ratio128AttentionConfig.from_model_config(
                model_config, layer_id=layer_id, max_seq_len=MAX_SEQ_LEN
            )
            self.prepared = prepare_attention_weights(
                self.raw_block.attention, **identity
            )
        self._seed_payload: dict[str, Any] | None = None

    def build_seed_payload(self, *, seed: int, rank: int) -> None:
        """Deterministic per-rank residency payload shared by every lane."""

        layer_seed = (
            seed * 9_176_501 + rank * 104_729 + self.layer_id * 15_485_863
        ) & ((1 << 62) - 1)
        if self.kind == "window":
            self._seed_payload = {
                "raw": deterministic_tensor(
                    seed=layer_seed,
                    shape=(LOCAL_BATCH, 128, 512),
                    device=self.device,
                    scale=0.03,
                )
            }
        elif self.kind == "ratio128":
            self._seed_payload = {
                "raw": deterministic_tensor(
                    seed=layer_seed,
                    shape=(LOCAL_BATCH, 128, 512),
                    device=self.device,
                    scale=0.03,
                ),
                "compressed": deterministic_tensor(
                    seed=layer_seed + 1,
                    shape=(LOCAL_BATCH, START_POSITION // 128, 512),
                    device=self.device,
                    scale=0.025,
                ),
            }
        else:
            oracle_state = seed_nonzero_ratio4_state(
                self.config,
                batch_size=LOCAL_BATCH,
                start_pos=START_POSITION,
                main_ape=self.prepared.compressor_ape,
                index_ape=self.prepared.index_compressor_ape,
                seed=layer_seed,
                device=self.device,
            )
            self._seed_payload = {"oracle": oracle_state}

    def new_state(self) -> DirectState:
        if self.kind == "window":
            return StaticWindowKV(
                num_local_sequences=LOCAL_BATCH,
                max_seq_len=MAX_SEQ_LEN,
                layer_id=self.layer_id,
                device=self.device,
            )
        if self.kind == "ratio4":
            return StaticRatio4KV(
                num_local_sequences=LOCAL_BATCH,
                max_seq_len=MAX_SEQ_LEN,
                layer_id=self.layer_id,
                device=self.device,
            )
        return StaticLayerKV(
            num_local_sequences=LOCAL_BATCH,
            max_seq_len=MAX_SEQ_LEN,
            layer_id=self.layer_id,
            device=self.device,
        )

    def seed_state(self, state: DirectState) -> None:
        payload = self._seed_payload
        if payload is None:
            raise RuntimeError("seed payload was not built")
        if self.kind == "window":
            assert isinstance(state, StaticWindowKV)
            state.seed_decode_residency(
                start_pos=START_POSITION, raw=payload["raw"].clone()
            )
        elif self.kind == "ratio128":
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

    def new_block(self, *, model_config: Mapping[str, Any]) -> DirectDecodeBlock:
        state = self.new_state()
        self.seed_state(state)
        if self.kind == "window":
            attention: Any = WindowTorchAttention(
                self.config,
                self.prepared,
                state,
                nope_quant_mode="qat_intended_e4m3",
            )
        elif self.kind == "ratio4":
            attention = Ratio4TorchAttention(self.config, self.prepared, state)
        else:
            attention = Ratio128TorchAttention(
                self.config,
                self.prepared,
                state,
                nope_quant_mode="qat_intended_e4m3",
            )
        return DirectDecodeBlock(
            weights=self.raw_block,
            attention=attention,
            moe=self.moe,
            norm_eps=float(model_config["rms_norm_eps"]),
            sinkhorn_iters=int(model_config["hc_sinkhorn_iters"]),
            hc_eps=float(model_config["hc_eps"]),
        )


# --------------------------------------------------------------------------
# stateful helpers (E1a27 forms)


def forward_eager_prevalidated(
    stage: TP4DecodeStage,
    plan: TP4StatefulDecodeSuperStagePlan,
    *,
    graph_family: DecodeGraphFamily,
    moe_slot: int = EAGER_MOE_SLOT,
) -> torch.Tensor:
    """Execute the stateful graph body eagerly on an explicit MoE slot."""

    plan.cursor.guard_device_preflight(
        graph_family,
        expected_position=plan.expected_position,
        stop_position=plan.stop_position_tensor,
        stop_position_constant=plan.stop_position,
        state_positions=plan.state_position_tensors,
    )
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


def run_warm_cycle(
    stage: TP4DecodeStage,
    plan: TP4StatefulDecodeSuperStagePlan,
    *,
    seed: int,
    rank: int,
    device: torch.device,
    moe_slots: Mapping[DecodeGraphFamily, int] | None = None,
) -> None:
    for step in SCHEDULE:
        plan.input_residual_buffer.copy_(
            deterministic_residual(
                seed=seed, rank=rank, position=step.position, device=device
            )
        )
        plan.input_ids_buffer.copy_(
            deterministic_input_ids(
                seed=seed, rank=rank, position=step.position, device=device
            )
        )
        forward_eager_prevalidated(
            stage,
            plan,
            graph_family=step.family,
            moe_slot=(
                EAGER_MOE_SLOT if moe_slots is None else moe_slots[step.family]
            ),
        )
        plan.cursor.advance_host(step.family)
    torch.cuda.synchronize(device)


def restore_cycle(
    stage: TP4DecodeStage,
    snapshots: Sequence[DirectState],
    plan: TP4StatefulDecodeSuperStagePlan,
) -> None:
    copy_stage_states(stage.states, snapshots)
    plan.cursor.reset(START_POSITION)
    plan.expected_position.fill_(START_POSITION)
    plan.stop_position_tensor.fill_(plan.stop_position)


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


# --------------------------------------------------------------------------
# part (a): eager super-stage vs manual per-layer chain


def run_part_a(
    *,
    stage: TP4DecodeStage,
    chain_blocks: Sequence[DirectDecodeBlock],
    seed: int,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "positions": list(EAGER_POSITIONS),
        "judgment": "bitwise_per_layer_per_position (E0df assembly-parity class)",
        "steps": [],
        "accepted": False,
    }
    all_exact = True
    for position in EAGER_POSITIONS:
        residual = deterministic_residual(
            seed=seed, rank=rank, position=position, device=device
        )
        ids = deterministic_input_ids(
            seed=seed, rank=rank, position=position, device=device
        )

        def stage_step() -> tuple[torch.Tensor, ...]:
            plan = stage.prepare_decode_plan(position)
            return stage.forward_decode_tensors(
                residual,
                input_ids_local=ids,
                start_pos=position,
                plan=plan,
                moe_slot=EAGER_MOE_SLOT,
            )

        stage_outputs = synchronized_local_step(
            f"part-a stage decode @{position}", stage_step, device=device, world=world
        )

        def chain_step() -> list[torch.Tensor]:
            outputs = []
            output = residual
            for block in chain_blocks:
                if block.compression_ratio == 4:
                    layer_plan = block.attention.prepare_decode_plan(
                        position, advance_overlap_state=True
                    )
                else:
                    layer_plan = block.attention.prepare_decode_plan(position)
                output = block.forward_decode_tensor(
                    output,
                    input_ids_local=(ids if block.route_kind == "hash" else None),
                    start_pos=position,
                    attention_plan=layer_plan,
                    moe_slot=EAGER_MOE_SLOT,
                )
                outputs.append(output)
            return outputs

        chain_outputs = synchronized_local_step(
            f"part-a chain decode @{position}", chain_step, device=device, world=world
        )
        torch.cuda.synchronize(device)

        layers = {}
        for layer_id, stage_output, chain_output in zip(
            LAYER_IDS, stage_outputs, chain_outputs, strict=True
        ):
            metrics = error_metrics(stage_output, chain_output)
            layers[str(layer_id)] = metrics
            all_exact = all_exact and metrics["bitwise_exact"]
        result["steps"].append({"position": position, "layers": layers})
    result["accepted"] = bool(all_exact)
    return result


# --------------------------------------------------------------------------
# part (b): stateful graph replay vs eager stateful


def run_part_b(
    *,
    graph_stage: TP4DecodeStage,
    eager_stage: TP4DecodeStage,
    seed: int,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schedule": {
            "start_position": START_POSITION,
            "stop_position": STOP_POSITION,
            "step_count": STEP_COUNT,
            "family_counts": {
                family.value: count for family, count in FAMILY_COUNTS.items()
            },
        },
        "judgment": (
            "graph_vs_eager_output bitwise_primary_plus_numeric_witness "
            "(gaiban E1a27)"
        ),
        "initial_state": "deterministic_seeded_kv_not_real_prefix",
        "graph_moe_slots": list(GRAPH_MOE_SLOT_TUPLE),
        "eager_moe_slot": EAGER_MOE_SLOT,
        "steps": [],
        "mismatched_step_positions": [],
        "accepted": False,
    }

    cursor_graph = StatefulDecodeCursor(start_position=START_POSITION, device=device)
    cursor_eager = StatefulDecodeCursor(start_position=START_POSITION, device=device)

    def prepare_plans() -> tuple[
        TP4StatefulDecodeSuperStagePlan, TP4StatefulDecodeSuperStagePlan
    ]:
        graph_plan = graph_stage.prepare_stateful_decode_plan(
            cursor_graph,
            start_position=START_POSITION,
            stop_position=STOP_POSITION,
            graph_moe_slots=GRAPH_MOE_SLOT_TUPLE,
        )
        eager_plan = eager_stage.prepare_stateful_decode_plan(
            cursor_eager,
            start_position=START_POSITION,
            stop_position=STOP_POSITION,
            graph_moe_slots=GRAPH_MOE_SLOT_TUPLE,
        )
        return graph_plan, eager_plan

    graph_plan, eager_plan = synchronized_local_step(
        "part-b prepare stateful plans", prepare_plans, device=device, world=world
    )
    result["plan_resident_bytes"] = {
        "graph": int(graph_plan.resident_bytes),
        "eager": int(eager_plan.resident_bytes),
    }

    snapshots = synchronized_local_step(
        "part-b snapshot states",
        lambda: [clone_state(state) for state in graph_stage.states],
        device=device,
        world=world,
    )
    for snapshot, state in zip(snapshots, eager_stage.states, strict=True):
        if full_state_sha256(snapshot) != full_state_sha256(state):
            raise RuntimeError("graph/eager lanes were not seeded identically")

    capture_stream = torch.cuda.Stream(device=device)
    graph_pools = {family: torch.cuda.graph_pool_handle() for family in DecodeGraphFamily}

    def warmup_all() -> None:
        # 1. graph-lane warmup (eager slot) on the default stream.
        run_warm_cycle(
            graph_stage, graph_plan, seed=seed, rank=rank, device=device
        )
        restore_cycle(graph_stage, snapshots, graph_plan)
        # 2. graph-lane warmup on the capture stream with the family slots,
        #    so every captured kernel and slot buffer is warm (E1a27 pattern).
        with torch.cuda.stream(capture_stream):
            run_warm_cycle(
                graph_stage,
                graph_plan,
                seed=seed,
                rank=rank,
                device=device,
                moe_slots=GRAPH_MOE_SLOTS,
            )
        torch.cuda.synchronize(device)
        restore_cycle(graph_stage, snapshots, graph_plan)
        for slot in GRAPH_MOE_SLOT_TUPLE:
            for moe in graph_stage.moes:
                moe.reset_free_slot_completion_event(GLOBAL_BATCH, slot)
        # 3. eager-lane warmup (eager slot).
        run_warm_cycle(
            eager_stage, eager_plan, seed=seed, rank=rank, device=device
        )
        restore_cycle(eager_stage, snapshots, eager_plan)
        # Warmed cycles must have left zero sticky errors before restore
        # cleared them; verify both lanes restart clean at 8192.
        for label, plan in (("graph", graph_plan), ("eager", eager_plan)):
            evidence = cursor_terminal_evidence(
                plan, expected_position=START_POSITION
            )
            if not evidence["accepted"]:
                raise RuntimeError(f"{label} lane warmup restore drifted: {evidence}")

    synchronized_local_step("part-b warmups", warmup_all, device=device, world=world)

    graphs: dict[DecodeGraphFamily, torch.cuda.CUDAGraph] = {}
    capture_order: list[str] = []
    all_exact = True
    state_parity_all = True

    for step_index, step in enumerate(SCHEDULE):
        residual = deterministic_residual(
            seed=seed, rank=rank, position=step.position, device=device
        )
        ids = deterministic_input_ids(
            seed=seed, rank=rank, position=step.position, device=device
        )

        def preflight() -> None:
            graph_stage.validate_stateful_decode_call(
                residual,
                input_ids_local=ids,
                plan=graph_plan,
                graph_family=step.family,
            )
            eager_stage.validate_stateful_decode_call(
                residual,
                input_ids_local=ids,
                plan=eager_plan,
                graph_family=step.family,
            )
            if cursor_graph.host_position != step.position:
                raise RuntimeError("graph host cursor drifted")
            if cursor_eager.host_position != step.position:
                raise RuntimeError("eager host cursor drifted")

        synchronized_local_step(
            f"part-b step {step_index} preflight", preflight, device=device, world=world
        )
        graph_plan.input_residual_buffer.copy_(residual)
        graph_plan.input_ids_buffer.copy_(ids)
        eager_plan.input_residual_buffer.copy_(residual)
        eager_plan.input_ids_buffer.copy_(ids)

        captured = False
        if step.family not in graphs:
            def capture() -> torch.cuda.CUDAGraph:
                return capture_stateful_graph(
                    graph_stage,
                    graph_plan,
                    graph_family=step.family,
                    capture_stream=capture_stream,
                    pool=graph_pools[step.family],
                )

            graphs[step.family] = synchronized_local_step(
                f"part-b capture {step.family.value}", capture, device=device, world=world
            )
            capture_order.append(step.family.value)
            captured = True

        def replay() -> torch.Tensor:
            output = replay_stateful_graph(
                graphs[step.family], graph_plan, graph_family=step.family
            )
            torch.cuda.synchronize(device)
            return output

        graph_output = synchronized_local_step(
            f"part-b step {step_index} replay", replay, device=device, world=world
        )

        def eager() -> torch.Tensor:
            output = forward_eager_prevalidated(
                eager_stage, eager_plan, graph_family=step.family
            )
            torch.cuda.synchronize(device)
            return output

        eager_output = synchronized_local_step(
            f"part-b step {step_index} eager", eager, device=device, world=world
        )

        metrics = error_metrics(graph_output, eager_output)
        graph_terminal = cursor_terminal_evidence(
            graph_plan, expected_position=step.position + 1
        )
        eager_terminal = cursor_terminal_evidence(
            eager_plan, expected_position=step.position + 1
        )
        # Host cursors have not advanced yet at this point.
        cursors_ok = bool(
            graph_terminal["device_position"] == step.position + 1
            and eager_terminal["device_position"] == step.position + 1
            and graph_terminal["dispatch_error"] == 0
            and eager_terminal["dispatch_error"] == 0
        )
        if not cursors_ok:
            raise RuntimeError(
                f"step {step_index} cursor drift: graph={graph_terminal} "
                f"eager={eager_terminal}"
            )
        parity = state_next_positions_equal(graph_stage.states, eager_stage.states)
        state_parity_all = state_parity_all and parity
        all_exact = all_exact and metrics["bitwise_exact"]
        if not metrics["bitwise_exact"]:
            result["mismatched_step_positions"].append(step.position)
        result["steps"].append(
            {
                "index": step_index,
                "position": step.position,
                "family": step.family.value,
                "captured_here": captured,
                "metrics": metrics,
                "state_next_position_parity": parity,
            }
        )
        cursor_graph.advance_host(step.family)
        cursor_eager.advance_host(step.family)
        if rank == 0 and (step_index % 16 == 0 or captured):
            print(
                f"[E0sf] step {step_index} pos {step.position} "
                f"family {step.family.value} captured={captured} "
                f"bitwise={metrics['bitwise_exact']} max_abs={metrics['max_abs']:.3e}",
                flush=True,
            )

    result["capture_order"] = capture_order
    result["terminal"] = {
        "graph": cursor_terminal_evidence(
            graph_plan, expected_position=STOP_POSITION
        ),
        "eager": cursor_terminal_evidence(
            eager_plan, expected_position=STOP_POSITION
        ),
    }
    final_state_digests = {
        str(layer_id): {
            "graph": full_state_sha256(graph_state),
            "eager": full_state_sha256(eager_state),
        }
        for layer_id, graph_state, eager_state in zip(
            LAYER_IDS, graph_stage.states, eager_stage.states, strict=True
        )
    }
    result["final_state_digests"] = final_state_digests
    final_states_equal = all(
        record["graph"] == record["eager"]
        for record in final_state_digests.values()
    )

    teardown = synchronized_local_step(
        "part-b teardown",
        lambda: teardown_stateful_graphs(
            graph_stage, graph_plan, graphs, pool_handles=graph_pools
        ),
        device=device,
        world=world,
    )
    result["teardown"] = teardown

    result["accepted"] = bool(
        all_exact
        and state_parity_all
        and final_states_equal
        and result["terminal"]["graph"]["accepted"]
        and result["terminal"]["eager"]["accepted"]
        and capture_order
        == ["normal", "ratio4_boundary", "ratio4_ratio128_boundary"]
        and teardown["accepted"]
        and not graph_stage.poisoned
        and not eager_stage.poisoned
    )
    result["summary"] = {
        "steps_total": STEP_COUNT,
        "steps_bitwise_exact": sum(
            1 for record in result["steps"] if record["metrics"]["bitwise_exact"]
        ),
        "max_abs_over_steps": max(
            record["metrics"]["max_abs"] for record in result["steps"]
        ),
        "state_parity_all_steps": state_parity_all,
        "final_states_equal": final_states_equal,
        "teardown_accepted": bool(teardown["accepted"]),
    }
    return result


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
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
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0sf-superstage-stateful-graph",
        "measurement_class": "semantic_correctness_gate",
        "rank": rank,
        "local_rank": local_rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "seed": args.seed,
        "layer_ids": list(LAYER_IDS),
        "checkpoint_id": None,
        "part_a": None,
        "part_b": None,
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }

    started = time.perf_counter()
    try:
        if world != EXPECTED_WORLD:
            raise ValueError(f"E0sf requires TP4, got world={world}")
        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root, list(LAYER_IDS), world
                )
                if not checkpoint["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {checkpoint['errors'][:3]}"
                    )
                for layer_id in LAYER_IDS:
                    block_contract = inspect_replicated_block_contract(
                        stage_root, layer_id=layer_id, rank=0, world_size=world
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

        phase_started = time.perf_counter()
        assets: list[LayerAssets] = []
        for layer_id in LAYER_IDS:
            assets.append(
                synchronized_local_step(
                    f"load layer-{layer_id}",
                    lambda layer_id=layer_id: LayerAssets(
                        layer_id=layer_id,
                        model_config=model_config,
                        stage_root=stage_root,
                        rank=rank,
                        world=world,
                        checkpoint_id=result["checkpoint_id"],
                        device=device,
                        progress_every=args.progress_every,
                    ),
                    device=device,
                    world=world,
                )
            )
            if rank == 0:
                print(f"[E0sf] layer {layer_id} loaded", flush=True)
        result["diagnostic_seconds"]["load"] = time.perf_counter() - phase_started

        def build_lanes() -> dict[str, list[DirectDecodeBlock]]:
            for asset in assets:
                asset.build_seed_payload(seed=args.seed, rank=rank)
            return {
                lane: [asset.new_block(model_config=model_config) for asset in assets]
                for lane in ("stage_a", "chain_b", "graph", "eager")
            }

        phase_started = time.perf_counter()
        lanes = synchronized_local_step(
            "build lanes", build_lanes, device=device, world=world
        )
        stage_a = TP4DecodeStage(lanes["stage_a"])
        graph_stage = TP4DecodeStage(lanes["graph"])
        eager_stage = TP4DecodeStage(lanes["eager"])
        result["diagnostic_seconds"]["build"] = time.perf_counter() - phase_started
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_after_build"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        if rank == 0:
            print(
                f"[E0sf] lanes built, free {free_bytes / 2**30:.2f} GiB", flush=True
            )

        phase_started = time.perf_counter()
        result["part_a"] = run_part_a(
            stage=stage_a,
            chain_blocks=lanes["chain_b"],
            seed=args.seed,
            rank=rank,
            world=world,
            device=device,
        )
        result["diagnostic_seconds"]["part_a"] = time.perf_counter() - phase_started
        if rank == 0:
            status = "PASS" if result["part_a"]["accepted"] else "FAIL"
            print(f"[E0sf] part (a) eager super-stage: {status}", flush=True)

        phase_started = time.perf_counter()
        result["part_b"] = run_part_b(
            graph_stage=graph_stage,
            eager_stage=eager_stage,
            seed=args.seed,
            rank=rank,
            world=world,
            device=device,
        )
        result["diagnostic_seconds"]["part_b"] = time.perf_counter() - phase_started
        if rank == 0:
            status = "PASS" if result["part_b"]["accepted"] else "FAIL"
            print(f"[E0sf] part (b) stateful graph replay: {status}", flush=True)

        result["accepted"] = bool(
            result["part_a"]["accepted"] and result["part_b"]["accepted"]
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
        and all(isinstance(record, dict) and record.get("accepted") for record in gathered)
    )
    write_json(out_dir / f"rank{rank}.json", result)
    if rank == 0:
        # Steps are bulky; keep the merged artifact readable.
        merged = []
        for record in gathered:
            trimmed = dict(record)
            part_b = trimmed.get("part_b")
            if isinstance(part_b, dict):
                part_b = dict(part_b)
                part_b["steps"] = "see per-rank artifacts"
                trimmed["part_b"] = part_b
            merged.append(trimmed)
        write_json(
            out_dir / "result.json",
            {
                "experiment": "E0sf-superstage-stateful-graph",
                "accepted": accepted_all,
                "ranks": merged,
            },
        )
        print(f"[E0sf] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
