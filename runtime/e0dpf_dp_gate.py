#!/usr/bin/env python3
"""E0dpf: DP-attention (sequence-split) caliber gate vs full replication.

Twelfth vertical, part (a).  Establishes that the DP caliber -- each TP rank
serving its own ``B_global/4`` sequences with the full 64 heads, KV sized
``B_global/4`` per rank, MoE collectives untouched (the in-package
``TP4MoE`` already runs all_gather -> itp partial -> reduce_scatter) -- is
mathematically equivalent to the E1F full-replication caliber, on the E0sf
frame: single TP4 node, canonical L0-L5 slice (window x2 / ratio-4 x2 /
ratio-128 x2, hash and learned routing), stateful decode over the 132-step
schedule [8192, 8324) covering all three graph families.

Three lanes over identical global material:

- ``rep``   : full replication, ``num_local_sequences = B_global`` on every
              rank, identical global inputs on all ranks (E1F B semantics).
- ``dp``    : DP split, ``num_local_sequences = B_global/4``; rank r feeds
              rows ``[r*bl, (r+1)*bl)`` of the same global inputs and the
              same global KV seed payloads (eager stateful body).
- ``dp_graph``: same DP split, driven through captured stateful CUDA graphs
              (attention + MoE all_gather/reduce_scatter in-graph).

Gates (shaped by the E0dpf probe finding, see ``e0dpf_probe.py``):

(g1) **DP graph vs DP eager** must be bitwise per step (E0sf part-b class,
     re-run on the DP shapes), with clean cursor/teardown and equal final
     KV digests.  This is the "attention + all_gather + MoE +
     reduce_scatter all inside one stateful CUDA graph" evidence on the DP
     shapes.
(g2) **Isolated per-layer DP vs replication (hard gate)**: one decode step
     per layer with *aligned inputs* -- the DP block receives exactly rows
     ``[r*bl,(r+1)*bl)`` of the replicated chain value at every
     sub-boundary, so each sub-stage is compared in isolation.  Observed:
     window/ratio-128 attention paths are fully **bitwise** across the two
     batch shapes; the fp32 HC mix linear and the ratio-4 attention
     internals differ at fp reduction-order LSB (kernel selection follows
     the M dimension), and the MoE output differs by <= 1 bf16 ulp
     (Marlin expert-block layout + reduce_scatter at 4B vs B gathered
     rows).  The gate reports bitwise per sub-stage and bounds every
     sub-stage by the ``ISOLATED_LIMITS`` fp-LSB envelopes.
(g3) **132-step trajectory (diagnostic, soft gate)**: rank r's DP stage
     output vs rows of the local replicated output per step.  Per-layer
     bf16 MoE noise compounds through 6 layers x 132 KV-feedback steps
     (near-tie router flips), so the trajectories drift in the same class
     as the replicated caliber's own cross-TP-lane non-bitwise divergence
     (E1F finding #2, reduce_scatter summation order).  The per-step
     replicated cross-lane divergence (rank r vs rank 0 on identical
     sequences) is recorded as the baseline; the soft gate only requires
     all values finite and ``rms_rel <= 0.75`` (two independent same-scale
     bf16 trajectories would sit at ~sqrt(2)).  Final per-layer KV
     row-slices are reported as witnesses of the same drift class.

Run (titan064):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  export NCCL_P2P_LEVEL=SYS TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=4 \
    e0dpf_dp_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash --out-dir out-e0dpf
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
from dsv4_direct.dp_caliber import (
    dp_local_batch,
    dp_row_bounds,
    dp_row_slice,
    dp_slice_ratio4_oracle_state,
)
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
MAX_SEQ_LEN = 8448

LAYER_IDS = SUPERSTAGE_LAYER_IDS  # (0, 1, 2, 3, 4, 5)

START_POSITION = 8192
STEP_COUNT = 132
STOP_POSITION = START_POSITION + STEP_COUNT
SCHEDULE = build_decode_schedule(START_POSITION, STEP_COUNT)
FAMILY_COUNTS = schedule_family_counts(SCHEDULE)

GRAPH_MOE_SLOTS: dict[DecodeGraphFamily, int] = {
    DecodeGraphFamily.NORMAL: 1,
    DecodeGraphFamily.RATIO4_BOUNDARY: 2,
    DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY: 3,
}
GRAPH_MOE_SLOT_TUPLE = tuple(GRAPH_MOE_SLOTS[family] for family in DecodeGraphFamily)
EAGER_MOE_SLOT = 0

EXPECTED_MOE_RESIDENT_BYTES = 861_931_008

# (g2) isolated per-layer bounds.  Observed on titan064 (B_global=8, bl=2):
# window/ratio-128 attention paths are fully bitwise across the two batch
# shapes; the fp32 HC mix linear (M=8 vs 2 kernel selection) and the
# ratio-4 attention internals differ at fp reduction-order LSB
# (attention_hidden <= 1.2e-4 abs, attention_branch <= 6.1e-4 abs /
# 6.5e-3 rms_rel); the MoE residue (Marlin layout + reduce_scatter at 4B
# vs B gathered rows) is <= 1 bf16 ulp (<= 3.9e-3 abs, <= 4.6e-3
# rms_rel).  Bounds carry ~4x headroom without admitting semantic error.
ISOLATED_LIMITS: dict[str, tuple[float, float]] = {
    # name: (max_abs limit, rms_rel limit)
    "attention_hidden": (1e-3, 1e-3),
    "attention_post": (1e-6, 1e-6),
    "attention_comb": (1e-6, 1e-6),
    # branch outputs have O(1) entries; a single-element bf16 ulp there is
    # 2^-7 (observed 7.8e-3 on one rank/layer), so the abs bound is 1/64.
    "attention_branch": (1.0 / 64.0, 2e-2),
    "after_attention": (1e-3, 1e-3),
    "ffn_hidden": (1e-3, 1e-3),
    "moe_output": (1.0 / 16.0, 2e-2),
    "block_output": (1.0 / 16.0, 2e-2),
}
# KV rows written from aligned inputs: fp32 pooling/score states may carry
# the same LSB noise; -inf sentinel entries must match exactly.
ISOLATED_STATE_MAX_ABS_LIMIT = 1e-2
# (g3) trajectory soft gate: bounded drift of the same class as the
# replicated caliber's own cross-lane divergence; two *independent*
# same-scale bf16 trajectories would show rms_rel ~ sqrt(2).
TRAJECTORY_RMS_REL_LIMIT = 0.75


# --------------------------------------------------------------------------
# generic helpers (E0sf process forms)


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


def global_residual(
    *, seed: int, position: int, batch: int, device: torch.device
) -> torch.Tensor:
    """Global-sequence-keyed residual: identical on every rank (no rank term)."""

    return deterministic_tensor(
        seed=(seed * 1_000_003 + position * 7_919) & ((1 << 62) - 1),
        shape=(batch, 1, 4, 4096),
        device=device,
    )


def global_input_ids(
    *, seed: int, position: int, batch: int, device: torch.device
) -> torch.Tensor:
    """Per-global-sequence distinct token IDs (stresses the hash-route gather)."""

    generator = torch.Generator(device="cpu").manual_seed(
        (seed * 2654435761 + position * 7919) & ((1 << 62) - 1)
    )
    return torch.randint(
        0, EXPECTED_VOCAB, (batch, 1), generator=generator, dtype=torch.int64
    ).to(device)


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


# --------------------------------------------------------------------------
# layer build + global seeding (E0sf LayerAssets, batch-parameterized)


class LayerAssets:
    """Per-layer shared material with global-batch seed payloads."""

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
        global_batch: int,
        global_row_shapes: tuple[int, ...],
        progress_every: int,
    ) -> None:
        self.layer_id = layer_id
        self.device = device
        self.global_batch = global_batch
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
            global_row_shapes=global_row_shapes,
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

    def build_seed_payload(self, *, seed: int) -> None:
        """One **global** payload keyed by layer only; every lane slices it."""

        layer_seed = (seed * 9_176_501 + self.layer_id * 15_485_863) & ((1 << 62) - 1)
        if self.kind == "window":
            self._seed_payload = {
                "raw": deterministic_tensor(
                    seed=layer_seed,
                    shape=(self.global_batch, 128, 512),
                    device=self.device,
                    scale=0.03,
                )
            }
        elif self.kind == "ratio128":
            self._seed_payload = {
                "raw": deterministic_tensor(
                    seed=layer_seed,
                    shape=(self.global_batch, 128, 512),
                    device=self.device,
                    scale=0.03,
                ),
                "compressed": deterministic_tensor(
                    seed=layer_seed + 1,
                    shape=(self.global_batch, START_POSITION // 128, 512),
                    device=self.device,
                    scale=0.025,
                ),
            }
        else:
            oracle_state = seed_nonzero_ratio4_state(
                self.config,
                batch_size=self.global_batch,
                start_pos=START_POSITION,
                main_ape=self.prepared.compressor_ape,
                index_ape=self.prepared.index_compressor_ape,
                seed=layer_seed,
                device=self.device,
            )
            self._seed_payload = {"oracle": oracle_state}

    def new_state(self, *, local_batch: int) -> DirectState:
        if self.kind == "window":
            return StaticWindowKV(
                num_local_sequences=local_batch,
                max_seq_len=MAX_SEQ_LEN,
                layer_id=self.layer_id,
                device=self.device,
            )
        if self.kind == "ratio4":
            return StaticRatio4KV(
                num_local_sequences=local_batch,
                max_seq_len=MAX_SEQ_LEN,
                layer_id=self.layer_id,
                device=self.device,
            )
        return StaticLayerKV(
            num_local_sequences=local_batch,
            max_seq_len=MAX_SEQ_LEN,
            layer_id=self.layer_id,
            device=self.device,
        )

    def seed_state(self, state: DirectState, *, dp_rank: int | None) -> None:
        """Seed with the full global payload (rep) or this rank's rows (DP)."""

        payload = self._seed_payload
        if payload is None:
            raise RuntimeError("seed payload was not built")
        local_batch = state.num_local_sequences

        def rows(value: torch.Tensor) -> torch.Tensor:
            if dp_rank is None:
                if value.shape[0] != local_batch:
                    raise ValueError("replicated lane batch differs from payload")
                return value.clone()
            return dp_row_slice(value, dp_rank, local_batch)

        if self.kind == "window":
            assert isinstance(state, StaticWindowKV)
            state.seed_decode_residency(
                start_pos=START_POSITION, raw=rows(payload["raw"])
            )
        elif self.kind == "ratio128":
            assert isinstance(state, StaticLayerKV)
            state.seed_decode_residency(
                start_pos=START_POSITION,
                raw=rows(payload["raw"]),
                compressed=rows(payload["compressed"]),
            )
        else:
            assert isinstance(state, StaticRatio4KV)
            oracle = payload["oracle"]
            if dp_rank is not None:
                oracle = dp_slice_ratio4_oracle_state(oracle, dp_rank, local_batch)
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

    def new_block(
        self,
        *,
        model_config: Mapping[str, Any],
        local_batch: int,
        dp_rank: int | None,
    ) -> DirectDecodeBlock:
        state = self.new_state(local_batch=local_batch)
        self.seed_state(state, dp_rank=dp_rank)
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
# stateful helpers (E0sf forms)


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


class Lane:
    """One lane: seeded blocks + stage + cursor + stateful plan."""

    def __init__(
        self,
        *,
        label: str,
        assets: Sequence[LayerAssets],
        model_config: Mapping[str, Any],
        local_batch: int,
        dp_rank: int | None,
        device: torch.device,
    ) -> None:
        self.label = label
        self.local_batch = local_batch
        self.dp_rank = dp_rank
        blocks = [
            asset.new_block(
                model_config=model_config, local_batch=local_batch, dp_rank=dp_rank
            )
            for asset in assets
        ]
        self.stage = TP4DecodeStage(blocks)
        self.cursor = StatefulDecodeCursor(
            start_position=START_POSITION, device=device
        )
        self.plan = self.stage.prepare_stateful_decode_plan(
            self.cursor,
            start_position=START_POSITION,
            stop_position=STOP_POSITION,
            graph_moe_slots=GRAPH_MOE_SLOT_TUPLE,
        )

    def feed(self, residual_global: torch.Tensor, ids_global: torch.Tensor) -> None:
        if self.dp_rank is None:
            self.plan.input_residual_buffer.copy_(residual_global)
            self.plan.input_ids_buffer.copy_(ids_global)
        else:
            self.plan.input_residual_buffer.copy_(
                dp_row_slice(residual_global, self.dp_rank, self.local_batch)
            )
            self.plan.input_ids_buffer.copy_(
                dp_row_slice(ids_global, self.dp_rank, self.local_batch)
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


def run_warm_cycle(
    lane: Lane,
    *,
    seed: int,
    global_batch: int,
    device: torch.device,
    moe_slots: Mapping[DecodeGraphFamily, int] | None = None,
) -> None:
    for step in SCHEDULE:
        lane.feed(
            global_residual(
                seed=seed, position=step.position, batch=global_batch, device=device
            ),
            global_input_ids(
                seed=seed, position=step.position, batch=global_batch, device=device
            ),
        )
        forward_eager_prevalidated(
            lane.stage,
            lane.plan,
            graph_family=step.family,
            moe_slot=(
                EAGER_MOE_SLOT if moe_slots is None else moe_slots[step.family]
            ),
        )
        lane.cursor.advance_host(step.family)
    torch.cuda.synchronize(device)


def restore_cycle(lane: Lane, snapshots: Sequence[DirectState]) -> None:
    copy_stage_states(lane.stage.states, snapshots)
    lane.cursor.reset(START_POSITION)
    lane.plan.expected_position.fill_(START_POSITION)
    lane.plan.stop_position_tensor.fill_(lane.plan.stop_position)


def run_isolated_layer_compare(
    rep_lane: "Lane",
    dp_lane: "Lane",
    *,
    rank: int,
    seed: int,
    global_batch: int,
    device: torch.device,
) -> dict[str, Any]:
    """(g2) One aligned-input decode step per layer, compared in isolation.

    The replicated chain provides every layer input; the DP block receives
    exactly this rank's rows of the same values at every sub-boundary, so a
    difference at one sub-stage cannot leak into the next comparison.
    Mutates both lanes' KV at START_POSITION; the caller restores from
    snapshots afterwards.
    """

    from dsv4_direct.hyper_connections import hc_post

    bl = dp_lane.local_batch
    lo, hi = dp_row_bounds(rank, bl)
    residual_g = global_residual(
        seed=seed, position=START_POSITION, batch=global_batch, device=device
    )
    ids_g = global_input_ids(
        seed=seed, position=START_POSITION, batch=global_batch, device=device
    )
    ids_dp = dp_row_slice(ids_g, rank, bl)

    def rows(value: torch.Tensor) -> torch.Tensor:
        return value[lo:hi].contiguous()

    rep_x = residual_g
    layers: dict[str, Any] = {}
    all_ok = True
    for rep_block, dp_block in zip(
        rep_lane.stage.blocks, dp_lane.stage.blocks, strict=True
    ):
        record: dict[str, Any] = {}
        rep_h, rep_post, rep_comb = rep_block.prepare_attention(rep_x)
        dp_h, dp_post, dp_comb = dp_block.prepare_attention(rows(rep_x))
        record["attention_hidden"] = error_metrics(dp_h, rows(rep_h))
        record["attention_post"] = error_metrics(dp_post, rows(rep_post))
        record["attention_comb"] = error_metrics(dp_comb, rows(rep_comb))

        if rep_block.compression_ratio == 4:
            rep_plan = rep_block.attention.prepare_decode_plan(
                START_POSITION, advance_overlap_state=True
            )
            dp_plan = dp_block.attention.prepare_decode_plan(
                START_POSITION, advance_overlap_state=True
            )
        else:
            rep_plan = rep_block.attention.prepare_decode_plan(START_POSITION)
            dp_plan = dp_block.attention.prepare_decode_plan(START_POSITION)
        rep_attn = rep_block.attention.forward_decode_tensor(
            rep_h, start_pos=START_POSITION, plan=rep_plan
        )
        dp_attn = dp_block.attention.forward_decode_tensor(
            rows(rep_h), start_pos=START_POSITION, plan=dp_plan
        )
        record["attention_branch"] = error_metrics(dp_attn, rows(rep_attn))

        # KV write row-purity for this step (aligned inputs).  fp32 pooling
        # and score states may carry LSB noise; -inf sentinels must match
        # exactly (a -inf/-inf pair subtracts to NaN, so compare masked).
        rep_items = dict(rep_block.attention.state._owned_tensor_items())
        state_bitwise = True
        state_max_abs = 0.0
        state_nonfinite_mismatch = False
        for name, dp_tensor in dp_block.attention.state._owned_tensor_items():
            rep_tensor = rep_items[name]
            if rep_tensor.ndim < 1 or rep_tensor.shape[0] != rep_lane.local_batch:
                continue
            rep_rows = rep_tensor[lo:hi]
            bitwise = bool(torch.equal(rep_rows, dp_tensor))
            state_bitwise = state_bitwise and bitwise
            if bitwise or not rep_rows.is_floating_point():
                if not bitwise:
                    state_nonfinite_mismatch = True
                continue
            left = dp_tensor.float()
            right = rep_rows.float()
            finite = torch.isfinite(left) & torch.isfinite(right)
            if not bool(
                torch.equal(torch.isfinite(left), torch.isfinite(right))
            ) or not bool(torch.equal(left[~finite].cpu(), right[~finite].cpu())):
                state_nonfinite_mismatch = True
            if bool(finite.any().item()):
                state_max_abs = max(
                    state_max_abs,
                    float((left[finite] - right[finite]).abs().max().item()),
                )
        record["state_rows_bitwise_after_step"] = state_bitwise
        record["state_rows_max_abs"] = state_max_abs
        record["state_rows_nonfinite_mismatch"] = state_nonfinite_mismatch

        rep_after = hc_post(rep_attn, rep_x, rep_post, rep_comb)
        dp_after = hc_post(rows(rep_attn), rows(rep_x), rows(rep_post), rows(rep_comb))
        record["after_attention"] = error_metrics(dp_after, rows(rep_after))
        rep_ffn_h, rep_fpost, rep_fcomb = rep_block.prepare_ffn(rep_after)
        dp_ffn_h, _, _ = dp_block.prepare_ffn(rows(rep_after))
        record["ffn_hidden"] = error_metrics(dp_ffn_h, rows(rep_ffn_h))

        rep_moe_kwargs: dict[str, Any] = {"slot": EAGER_MOE_SLOT}
        dp_moe_kwargs: dict[str, Any] = {"slot": EAGER_MOE_SLOT}
        if rep_block.route_kind == "hash":
            rep_moe_kwargs["input_ids_local"] = ids_g
            dp_moe_kwargs["input_ids_local"] = ids_dp
        rep_moe = rep_block.moe.forward_tensor(rep_ffn_h, **rep_moe_kwargs)
        dp_moe = dp_block.moe.forward_tensor(rows(rep_ffn_h), **dp_moe_kwargs)
        record["moe_output"] = error_metrics(dp_moe, rows(rep_moe))

        rep_out = hc_post(rep_moe, rep_after, rep_fpost, rep_fcomb)
        dp_out = hc_post(dp_moe, rows(rep_after), rows(rep_fpost), rows(rep_fcomb))
        record["block_output"] = error_metrics(dp_out, rows(rep_out))

        record["attention_path_bitwise"] = bool(
            record["attention_hidden"]["bitwise_exact"]
            and record["attention_post"]["bitwise_exact"]
            and record["attention_comb"]["bitwise_exact"]
            and record["attention_branch"]["bitwise_exact"]
            and record["after_attention"]["bitwise_exact"]
            and record["ffn_hidden"]["bitwise_exact"]
            and state_bitwise
        )
        out_of_bounds = [
            name
            for name, (max_abs_limit, rms_rel_limit) in ISOLATED_LIMITS.items()
            if record[name]["max_abs"] > max_abs_limit
            or record[name]["rms_rel"] > rms_rel_limit
        ]
        if state_nonfinite_mismatch or state_max_abs > ISOLATED_STATE_MAX_ABS_LIMIT:
            out_of_bounds.append("state_rows")
        record["out_of_bounds"] = out_of_bounds
        record["accepted"] = not out_of_bounds
        all_ok = all_ok and record["accepted"]
        layers[str(rep_block.layer_id)] = record
        rep_x = rep_out

    return {
        "judgment": (
            "aligned-input per-layer isolation at position "
            f"{START_POSITION}: bitwise reported per sub-stage; every "
            "sub-stage bounded by the fp reduction-order limits in "
            "ISOLATED_LIMITS (kernel selection differs with the M "
            "dimension between the two calibers; the residue class is "
            "fp LSB, not semantic)"
        ),
        "limits": {name: list(value) for name, value in ISOLATED_LIMITS.items()},
        "state_max_abs_limit": ISOLATED_STATE_MAX_ABS_LIMIT,
        "layers": layers,
        "accepted": all_ok,
    }


def compare_final_state_slices(
    rep_lane: Lane, dp_lane: Lane, *, dp_rank: int
) -> dict[str, Any]:
    """Row-slice witness: rep KV rows [r*bl,(r+1)*bl) vs DP local KV rows."""

    lo, hi = dp_row_bounds(dp_rank, dp_lane.local_batch)
    layers: dict[str, Any] = {}
    all_bitwise = True
    max_abs = 0.0
    for layer_id, rep_state, dp_state in zip(
        rep_lane.stage.layer_ids,
        rep_lane.stage.states,
        dp_lane.stage.states,
        strict=True,
    ):
        record: dict[str, Any] = {}
        rep_items = dict(rep_state._owned_tensor_items())
        for name, dp_tensor in dp_state._owned_tensor_items():
            rep_tensor = rep_items[name]
            if (
                rep_tensor.ndim < 1
                or rep_tensor.shape[0] != rep_lane.local_batch
                or dp_tensor.shape[0] != dp_lane.local_batch
            ):
                record[name] = "skipped_not_batch_first"
                continue
            rep_rows = rep_tensor[lo:hi]
            bitwise = bool(torch.equal(rep_rows, dp_tensor))
            entry: dict[str, Any] = {"bitwise": bitwise}
            if not bitwise and rep_rows.is_floating_point():
                delta = (rep_rows.float() - dp_tensor.float()).abs()
                finite = bool(torch.isfinite(delta).all().item())
                entry["finite"] = finite
                entry["max_abs"] = float(delta.max().item()) if finite else None
                if finite:
                    max_abs = max(max_abs, entry["max_abs"])
            all_bitwise = all_bitwise and bitwise
            record[name] = entry
        layers[str(layer_id)] = record
    return {"layers": layers, "all_bitwise": all_bitwise, "max_abs": max_abs}


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--global-batch", type=int, default=8)
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

    global_batch = int(args.global_batch)
    dp_batch = dp_local_batch(global_batch)
    rep_global_rows = global_batch * EXPECTED_WORLD
    dp_global_rows = global_batch

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0dpf-dp-attention-caliber-gate",
        "measurement_class": "semantic_correctness_gate",
        "caliber": {
            "rep_lane": (
                f"full replication: {global_batch} identical sequences on every "
                f"rank (E1F B semantics), num_local_sequences={global_batch}"
            ),
            "dp_lanes": (
                f"DP split: rank r serves global rows [r*{dp_batch}, (r+1)*{dp_batch}) "
                f"of the same sequences, full 64 heads, num_local_sequences={dp_batch}; "
                "MoE collectives unchanged (in-package all_gather -> itp partial "
                "-> reduce_scatter)"
            ),
            "schedule": f"[{START_POSITION}, {STOP_POSITION}) all three families",
            "hc_backend": "eager (E0sf frame); fused-vs-fused DP graph parity is "
            "re-checked at full config by the E1F dp settle gate",
            "isolated_limits": {
                name: list(value) for name, value in ISOLATED_LIMITS.items()
            },
            "trajectory_soft_rms_rel_limit": TRAJECTORY_RMS_REL_LIMIT,
        },
        "rank": rank,
        "local_rank": local_rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "seed": args.seed,
        "global_batch": global_batch,
        "dp_local_batch": dp_batch,
        "layer_ids": list(LAYER_IDS),
        "family_counts": {
            family.value: count for family, count in FAMILY_COUNTS.items()
        },
        "checkpoint_id": None,
        "steps": [],
        "isolated": None,
        "dp_vs_rep": None,
        "dp_graph_vs_eager": None,
        "final_state_slices": None,
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }

    started = time.perf_counter()
    try:
        if world != EXPECTED_WORLD:
            raise ValueError(f"E0dpf requires TP4, got world={world}")
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
                        global_batch=global_batch,
                        global_row_shapes=(rep_global_rows, dp_global_rows),
                        progress_every=args.progress_every,
                    ),
                    device=device,
                    world=world,
                )
            )
            if rank == 0:
                print(f"[E0dpf] layer {layer_id} loaded", flush=True)
        result["diagnostic_seconds"]["load"] = time.perf_counter() - phase_started

        def build_lanes() -> dict[str, Lane]:
            for asset in assets:
                asset.build_seed_payload(seed=args.seed)
            return {
                "rep": Lane(
                    label="rep",
                    assets=assets,
                    model_config=model_config,
                    local_batch=global_batch,
                    dp_rank=None,
                    device=device,
                ),
                "dp": Lane(
                    label="dp",
                    assets=assets,
                    model_config=model_config,
                    local_batch=dp_batch,
                    dp_rank=rank,
                    device=device,
                ),
                "dp_graph": Lane(
                    label="dp_graph",
                    assets=assets,
                    model_config=model_config,
                    local_batch=dp_batch,
                    dp_rank=rank,
                    device=device,
                ),
            }

        phase_started = time.perf_counter()
        lanes = synchronized_local_step(
            "build lanes", build_lanes, device=device, world=world
        )
        rep_lane = lanes["rep"]
        dp_lane = lanes["dp"]
        dp_graph_lane = lanes["dp_graph"]
        result["diagnostic_seconds"]["build"] = time.perf_counter() - phase_started
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_after_build"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        if rank == 0:
            print(f"[E0dpf] lanes built, free {free_bytes / 2**30:.2f} GiB", flush=True)

        # DP lanes must be seeded identically; the rep lane rows must equal
        # the DP rows by construction of the global payload (checked at the
        # end via the final-state slice witness).
        if dp_lane.state_digests() != dp_graph_lane.state_digests():
            raise RuntimeError("dp/dp_graph lanes were not seeded identically")

        snapshots = {
            label: synchronized_local_step(
                f"snapshot {label}",
                lambda lane=lane: [clone_state(state) for state in lane.stage.states],
                device=device,
                world=world,
            )
            for label, lane in lanes.items()
        }

        capture_stream = torch.cuda.Stream(device=device)
        graph_pools = {
            family: torch.cuda.graph_pool_handle() for family in DecodeGraphFamily
        }

        def warmup_all() -> None:
            # graph lane: default-stream warm, then capture-stream warm on the
            # family slots (E1a27/E0sf pattern), then restore + event reset.
            run_warm_cycle(
                dp_graph_lane, seed=args.seed, global_batch=global_batch, device=device
            )
            restore_cycle(dp_graph_lane, snapshots["dp_graph"])
            with torch.cuda.stream(capture_stream):
                run_warm_cycle(
                    dp_graph_lane,
                    seed=args.seed,
                    global_batch=global_batch,
                    device=device,
                    moe_slots=GRAPH_MOE_SLOTS,
                )
            torch.cuda.synchronize(device)
            restore_cycle(dp_graph_lane, snapshots["dp_graph"])
            for slot in GRAPH_MOE_SLOT_TUPLE:
                for moe in dp_graph_lane.stage.moes:
                    moe.reset_free_slot_completion_event(dp_global_rows, slot)
            # eager lanes.
            run_warm_cycle(
                dp_lane, seed=args.seed, global_batch=global_batch, device=device
            )
            restore_cycle(dp_lane, snapshots["dp"])
            run_warm_cycle(
                rep_lane, seed=args.seed, global_batch=global_batch, device=device
            )
            restore_cycle(rep_lane, snapshots["rep"])
            for label, lane in lanes.items():
                evidence = lane.terminal(START_POSITION)
                if not evidence["accepted"]:
                    raise RuntimeError(
                        f"{label} lane warmup restore drifted: {evidence}"
                    )

        phase_started = time.perf_counter()
        synchronized_local_step("warmups", warmup_all, device=device, world=world)
        result["diagnostic_seconds"]["warmup"] = time.perf_counter() - phase_started

        # ------------------------------------------------------------------
        # (g2) isolated per-layer aligned-input compare, then restore.
        phase_started = time.perf_counter()
        result["isolated"] = synchronized_local_step(
            "isolated per-layer compare",
            lambda: run_isolated_layer_compare(
                rep_lane,
                dp_lane,
                rank=rank,
                seed=args.seed,
                global_batch=global_batch,
                device=device,
            ),
            device=device,
            world=world,
        )
        restore_cycle(rep_lane, snapshots["rep"])
        restore_cycle(dp_lane, snapshots["dp"])
        result["diagnostic_seconds"]["isolated"] = time.perf_counter() - phase_started
        if rank == 0:
            status = "PASS" if result["isolated"]["accepted"] else "FAIL"
            print(f"[E0dpf] (g2) isolated per-layer: {status}", flush=True)

        del snapshots
        torch.cuda.empty_cache()

        graphs: dict[DecodeGraphFamily, torch.cuda.CUDAGraph] = {}
        capture_order: list[str] = []
        lo, hi = dp_row_bounds(rank, dp_batch)

        dp_rep_bitwise_all = True
        dp_rep_max_rms_rel = 0.0
        dp_rep_max_abs = 0.0
        dp_rep_rms_rel_sum = 0.0
        dp_rep_mismatched: list[int] = []
        cross_lane_max_rms_rel = 0.0
        cross_lane_rms_rel_sum = 0.0
        graph_bitwise_all = True
        graph_mismatched: list[int] = []
        rep_gather = [
            torch.empty_like(rep_lane.plan.output_buffer)
            for _ in range(EXPECTED_WORLD)
        ]

        phase_started = time.perf_counter()
        for step_index, step in enumerate(SCHEDULE):
            residual_g = global_residual(
                seed=args.seed,
                position=step.position,
                batch=global_batch,
                device=device,
            )
            ids_g = global_input_ids(
                seed=args.seed,
                position=step.position,
                batch=global_batch,
                device=device,
            )

            def preflight() -> None:
                for lane in lanes.values():
                    external = (
                        residual_g
                        if lane.dp_rank is None
                        else dp_row_slice(residual_g, lane.dp_rank, lane.local_batch)
                    )
                    external_ids = (
                        ids_g
                        if lane.dp_rank is None
                        else dp_row_slice(ids_g, lane.dp_rank, lane.local_batch)
                    )
                    lane.stage.validate_stateful_decode_call(
                        external,
                        input_ids_local=external_ids,
                        plan=lane.plan,
                        graph_family=step.family,
                    )
                    if lane.cursor.host_position != step.position:
                        raise RuntimeError(f"{lane.label} host cursor drifted")

            synchronized_local_step(
                f"step {step_index} preflight", preflight, device=device, world=world
            )
            for lane in lanes.values():
                lane.feed(residual_g, ids_g)

            def rep_step() -> torch.Tensor:
                output = forward_eager_prevalidated(
                    rep_lane.stage, rep_lane.plan, graph_family=step.family
                )
                torch.cuda.synchronize(device)
                return output

            rep_output = synchronized_local_step(
                f"step {step_index} rep", rep_step, device=device, world=world
            )
            # replicated caliber cross-lane baseline: identical sequences on
            # every rank; divergence is reduce_scatter summation-order noise
            # (E1F finding #2).  This bounds the expected DP-vs-rep drift
            # class from within the replicated caliber itself.
            dist.all_gather(rep_gather, rep_output)
            rep_cross = error_metrics(rep_output, rep_gather[0])

            def dp_step() -> torch.Tensor:
                output = forward_eager_prevalidated(
                    dp_lane.stage, dp_lane.plan, graph_family=step.family
                )
                torch.cuda.synchronize(device)
                return output

            dp_output = synchronized_local_step(
                f"step {step_index} dp", dp_step, device=device, world=world
            )

            captured = False
            if step.family not in graphs:
                def capture() -> torch.cuda.CUDAGraph:
                    return capture_stateful_graph(
                        dp_graph_lane.stage,
                        dp_graph_lane.plan,
                        graph_family=step.family,
                        capture_stream=capture_stream,
                        pool=graph_pools[step.family],
                    )

                graphs[step.family] = synchronized_local_step(
                    f"capture {step.family.value}", capture, device=device, world=world
                )
                capture_order.append(step.family.value)
                captured = True

            def graph_step() -> torch.Tensor:
                output = replay_stateful_graph(
                    graphs[step.family], dp_graph_lane.plan, graph_family=step.family
                )
                torch.cuda.synchronize(device)
                return output

            graph_output = synchronized_local_step(
                f"step {step_index} graph replay", graph_step, device=device, world=world
            )

            dp_vs_rep = error_metrics(dp_output, rep_output[lo:hi])
            graph_vs_eager = error_metrics(graph_output, dp_output)
            dp_rep_bitwise_all = dp_rep_bitwise_all and dp_vs_rep["bitwise_exact"]
            dp_rep_max_rms_rel = max(dp_rep_max_rms_rel, dp_vs_rep["rms_rel"])
            dp_rep_max_abs = max(dp_rep_max_abs, dp_vs_rep["max_abs"])
            dp_rep_rms_rel_sum += dp_vs_rep["rms_rel"]
            cross_lane_max_rms_rel = max(cross_lane_max_rms_rel, rep_cross["rms_rel"])
            cross_lane_rms_rel_sum += rep_cross["rms_rel"]
            if not dp_vs_rep["bitwise_exact"]:
                dp_rep_mismatched.append(step.position)
            graph_bitwise_all = graph_bitwise_all and graph_vs_eager["bitwise_exact"]
            if not graph_vs_eager["bitwise_exact"]:
                graph_mismatched.append(step.position)

            for lane in lanes.values():
                lane.cursor.advance_host(step.family)
            result["steps"].append(
                {
                    "index": step_index,
                    "position": step.position,
                    "family": step.family.value,
                    "captured_here": captured,
                    "dp_vs_rep": dp_vs_rep,
                    "rep_cross_lane_vs_rank0": rep_cross,
                    "dp_graph_vs_eager": graph_vs_eager,
                }
            )
            if rank == 0 and (step_index % 16 == 0 or captured):
                print(
                    f"[E0dpf] step {step_index} pos {step.position} "
                    f"family {step.family.value} captured={captured} "
                    f"dp_vs_rep bitwise={dp_vs_rep['bitwise_exact']} "
                    f"rms_rel={dp_vs_rep['rms_rel']:.3e} "
                    f"graph bitwise={graph_vs_eager['bitwise_exact']}",
                    flush=True,
                )
        result["diagnostic_seconds"]["steps"] = time.perf_counter() - phase_started

        result["final_state_slices"] = compare_final_state_slices(
            rep_lane, dp_lane, dp_rank=rank
        )
        dp_graph_digests_equal = bool(
            dp_lane.state_digests() == dp_graph_lane.state_digests()
        )

        terminals = {
            label: lane.terminal(STOP_POSITION) for label, lane in lanes.items()
        }
        result["terminals"] = terminals
        teardown = synchronized_local_step(
            "teardown",
            lambda: teardown_stateful_graphs(
                dp_graph_lane.stage, dp_graph_lane.plan, graphs,
                pool_handles=graph_pools,
            ),
            device=device,
            world=world,
        )
        result["teardown"] = teardown

        trajectory_soft_pass = bool(
            dp_rep_max_rms_rel <= TRAJECTORY_RMS_REL_LIMIT
        )
        result["dp_vs_rep"] = {
            "judgment": (
                "(g3) diagnostic trajectory: per-layer bf16 MoE "
                "reduction-order noise compounds over 6 layers x 132 "
                "KV-feedback steps (near-tie router flips), same class as "
                "the replicated caliber's own cross-lane divergence; soft "
                f"gate rms_rel <= {TRAJECTORY_RMS_REL_LIMIT} and finite"
            ),
            "bitwise_all": dp_rep_bitwise_all,
            "mismatched_positions": dp_rep_mismatched[:16],
            "mismatched_count": len(dp_rep_mismatched),
            "max_rms_rel": dp_rep_max_rms_rel,
            "mean_rms_rel": dp_rep_rms_rel_sum / STEP_COUNT,
            "max_abs": dp_rep_max_abs,
            "rep_cross_lane_baseline": {
                "note": (
                    "replicated rank r vs rank 0 on identical sequences "
                    "(zero on rank 0 by definition)"
                ),
                "max_rms_rel": cross_lane_max_rms_rel,
                "mean_rms_rel": cross_lane_rms_rel_sum / STEP_COUNT,
            },
            "soft_gate_pass": trajectory_soft_pass,
            "accepted": trajectory_soft_pass,
            "mode": "bitwise" if dp_rep_bitwise_all else "numeric",
        }
        result["dp_graph_vs_eager"] = {
            "judgment": "bitwise per step (E0sf part-b class, DP shapes)",
            "bitwise_all": graph_bitwise_all,
            "mismatched_positions": graph_mismatched[:16],
            "capture_order": capture_order,
            "final_state_digests_equal": dp_graph_digests_equal,
            "accepted": bool(
                graph_bitwise_all
                and dp_graph_digests_equal
                and capture_order
                == ["normal", "ratio4_boundary", "ratio4_ratio128_boundary"]
            ),
        }
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_at_end"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

        result["accepted"] = bool(
            result["isolated"]["accepted"]
            and result["dp_vs_rep"]["accepted"]
            and result["dp_graph_vs_eager"]["accepted"]
            and all(terminal["accepted"] for terminal in terminals.values())
            and teardown["accepted"]
            and not any(lane.stage.poisoned for lane in lanes.values())
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
            trimmed = dict(record)
            trimmed["steps"] = "see per-rank artifacts"
            final_slices = trimmed.get("final_state_slices")
            if isinstance(final_slices, dict):
                trimmed["final_state_slices"] = {
                    "all_bitwise": final_slices.get("all_bitwise"),
                    "max_abs": final_slices.get("max_abs"),
                    "layers": "see per-rank artifacts",
                }
            merged.append(trimmed)
        write_json(
            out_dir / "result.json",
            {
                "experiment": "E0dpf-dp-attention-caliber-gate",
                "accepted": accepted_all,
                "ranks": merged,
            },
        )
        print(f"[E0dpf] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
