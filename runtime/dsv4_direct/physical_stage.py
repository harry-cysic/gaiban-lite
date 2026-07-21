"""Physical TP4 stage assembly for the scaled Flash TP4xPP2 pipeline.

Port of the necessary surface of gaiban ``dsv4_direct/physical_stage.py``
(``_validate_live_tp_collective`` :442, ``build_physical_layer_runtime`` :775,
``build_physical_stage`` :1247) for the E0pf vertical:

- **Live TP group binding** (gaiban :442-489): every physical layer build
  binds the declared TP identity (backend, world, local/global rank, ordered
  global ranks) to one initialized live NCCL subgroup before any weight
  loads.  This is the PP-critical part -- with world 8 the TP4 collectives
  must run on the stage subgroup, never the default group.
- **Per-layer material assembly**: the exact E0sf-verified construction
  (``load_replicated_block_weights`` + ``load_resident_moe_layer`` + TP4MoE
  + per-kind attention config/prepared weights), with the TP subgroup
  threaded into ``TP4MoE(group=...)``, promoted from the E0sf gate script
  into the package so both PP stages and the sequential reference chain
  assemble through one code path.
- **Scaled PP2 topology table**: stage 0 = layers 0-5, stage 1 = layers
  6-11 (whole-layer boundary).  Gaiban's Pro profiles (canonical 4+4,
  balanced 3+5, fractional L3 pre-MoE split via ``fractional_stage``) are
  placement policy for the 61-layer model and are not carried into this
  scaled vertical; the fragment-split *surface* itself is ported in
  ``block.DirectPreMoEBlockFragment``.

Deliberate reductions vs gaiban (policy, not accident): the provider-factory
dependency injection (``PhysicalProviderFactories`` /
``PhysicalLayerBuildDependencies``) existed to swap sparse/projection/MHC
backends per experiment arm; lite already fixed those decisions (torch sparse
backends, optional ``hc_boundary_backend``), so construction is direct.
State creation is exposed (``new_state``) but seeding stays experiment
policy, exactly as in the E0sf/E0df gates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist

from .attention import (
    Ratio128AttentionConfig,
    Ratio128TorchAttention,
    prepare_attention_weights,
    shard_ratio128_attention_weights,
)
from .block import DirectDecodeBlock, DirectPreMoEBlockFragment
from .block_weights import ResidentBlockWeights, load_replicated_block_weights
from .model_contract import SUPPORTED_LAYER_SPECS
from .moe_runtime import TP4MoE, TP4MoEConfig
from .ops.marlin_moe import load_resident_moe_layer
from .ratio4_attention import (
    Ratio4AttentionConfig,
    shard_ratio4_attention_weights,
    Ratio4TorchAttention,
    prepare_ratio4_attention_weights,
)
from .static_kv import StaticLayerKV
from .static_ratio4_kv import StaticRatio4KV
from .static_window_kv import StaticWindowKV
from .window_attention import (
    WindowAttentionConfig,
    WindowTorchAttention,
    prepare_window_attention_weights,
    shard_window_attention_weights,
)


EXPECTED_TP_SIZE = 4
PP2_WORLD_SIZE = 8

# Scaled Flash TP4xPP2 stage table: two whole-layer TP4 stages over the
# canonical L0-L11 window (window x2 / ratio-4 x5 / ratio-128 x5; every layer
# type and route kind appears).  Frozen for the E0pf vertical.
PP2_STAGE_LAYER_IDS: dict[int, tuple[int, ...]] = {
    0: (0, 1, 2, 3, 4, 5),
    1: (6, 7, 8, 9, 10, 11),
}

# Per-layer TP4 MoE resident budget frozen by the E0df/E0sf titan064 runs
# (identical geometry for every Flash MoE layer).
EXPECTED_MOE_RESIDENT_BYTES = 861_931_008

# Checkpoint-intended NoPE cache quantization for window/ratio-128 layers
# (E0wf/E0ef-verified; ratio-4 has its own internal handling).
NOPE_QUANT_MODE = "qat_intended_e4m3"


class PhysicalStageBuildError(ValueError):
    """Raised when a physical stage would be assembled off-contract."""


def pp2_stage_layer_ids(stage_id: int) -> tuple[int, ...]:
    if stage_id not in PP2_STAGE_LAYER_IDS:
        raise PhysicalStageBuildError(
            f"PP2 stage_id must be one of {sorted(PP2_STAGE_LAYER_IDS)}, "
            f"got {stage_id!r}"
        )
    return PP2_STAGE_LAYER_IDS[stage_id]


def validate_live_tp_group(
    tp_group: object,
    *,
    expected_local_rank: int,
    expected_global_ranks: tuple[int, ...],
) -> dict[str, Any]:
    """Bind a declared TP identity to one initialized live NCCL group.

    Port of gaiban ``_validate_live_tp_collective`` (physical_stage.py:442).
    """

    if tp_group is None:
        raise PhysicalStageBuildError(
            "real physical layer construction requires a non-null tp_group"
        )
    if not dist.is_initialized():
        raise PhysicalStageBuildError(
            "real physical layer construction requires initialized distributed state"
        )
    try:
        observed_backend = str(dist.get_backend(tp_group)).lower()
        observed_world = dist.get_world_size(tp_group)
        observed_local_rank = dist.get_rank(tp_group)
        observed_global_rank = dist.get_rank()
        observed_global_ranks = tuple(dist.get_process_group_ranks(tp_group))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PhysicalStageBuildError(
            "failed to inspect the declared live TP process group"
        ) from exc
    expected = {
        "backend": "nccl",
        "world_size": EXPECTED_TP_SIZE,
        "local_rank": expected_local_rank,
        "global_rank": expected_global_ranks[expected_local_rank]
        if 0 <= expected_local_rank < len(expected_global_ranks)
        else None,
        "global_ranks": tuple(expected_global_ranks),
    }
    observed = {
        "backend": observed_backend,
        "world_size": observed_world,
        "local_rank": observed_local_rank,
        "global_rank": observed_global_rank,
        "global_ranks": observed_global_ranks,
    }
    if observed != expected:
        raise PhysicalStageBuildError(
            f"live TP collective identity {observed} != declared {expected}"
        )
    return {
        **observed,
        "global_ranks": list(observed_global_ranks),
        "validated_live": True,
    }


PhysicalLayerState = StaticWindowKV | StaticRatio4KV | StaticLayerKV


@dataclass(frozen=True, slots=True)
class PhysicalLayerMaterial:
    """Loaded, TP-subgroup-bound material for one physical layer.

    Weight/MoE material is shared; ``new_state``/``new_attention``/
    ``new_block``/``new_fragment`` mint independent lanes over it (the E0sf
    lane pattern).
    """

    layer_id: int
    kind: str
    route_kind: str
    tp_rank: int
    tp_size: int
    checkpoint_id: str
    device: torch.device
    max_seq_len: int
    raw_block: ResidentBlockWeights
    prepared: Any
    attention_config: Any
    moe: TP4MoE
    norm_eps: float
    sinkhorn_iters: int
    hc_eps: float
    evidence: Mapping[str, Any]
    kv_dtype: str = "bf16"
    indexer_kv_dtype: str = "bf16"

    def new_state(self, *, num_local_sequences: int) -> PhysicalLayerState:
        if self.kind == "window":
            return StaticWindowKV(
                num_local_sequences=num_local_sequences,
                max_seq_len=self.max_seq_len,
                layer_id=self.layer_id,
                device=self.device,
                kv_dtype=self.kv_dtype,
            )
        if self.kind == "ratio4":
            return StaticRatio4KV(
                num_local_sequences=num_local_sequences,
                max_seq_len=self.max_seq_len,
                layer_id=self.layer_id,
                device=self.device,
                kv_dtype=self.kv_dtype,
                indexer_dtype=self.indexer_kv_dtype,
            )
        return StaticLayerKV(
            num_local_sequences=num_local_sequences,
            max_seq_len=self.max_seq_len,
            layer_id=self.layer_id,
            device=self.device,
            kv_dtype=self.kv_dtype,
        )

    def _attention_tp_group(self) -> Any:
        """The TP group for the o-path all-reduce, or None when unsharded.

        Reuses the MoE's group rather than making a second one: it is the same
        four ranks, and a duplicate communicator would cost memory and add a
        second thing that can be wired wrong.
        """

        if getattr(self.attention_config, "tp_size", 1) == 1:
            return None
        return self.moe.group

    def new_attention(self, state: PhysicalLayerState) -> Any:
        if self.kind == "window":
            if not isinstance(state, StaticWindowKV):
                raise PhysicalStageBuildError("window layer requires StaticWindowKV")
            return WindowTorchAttention(
                self.attention_config,
                self.prepared,
                state,
                nope_quant_mode=NOPE_QUANT_MODE,
                tp_group=self._attention_tp_group(),
            )
        if self.kind == "ratio4":
            if not isinstance(state, StaticRatio4KV):
                raise PhysicalStageBuildError("ratio-4 layer requires StaticRatio4KV")
            return Ratio4TorchAttention(
                self.attention_config,
                self.prepared,
                state,
                tp_group=self._attention_tp_group(),
            )
        if not isinstance(state, StaticLayerKV):
            raise PhysicalStageBuildError("ratio-128 layer requires StaticLayerKV")
        return Ratio128TorchAttention(
            self.attention_config,
            self.prepared,
            state,
            nope_quant_mode=NOPE_QUANT_MODE,
            tp_group=self._attention_tp_group(),
        )

    def new_block(
        self,
        state: PhysicalLayerState,
        *,
        hc_boundary_backend: Any | None = None,
    ) -> DirectDecodeBlock:
        return DirectDecodeBlock(
            weights=self.raw_block,
            attention=self.new_attention(state),
            moe=self.moe,
            norm_eps=self.norm_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            hc_eps=self.hc_eps,
            hc_boundary_backend=hc_boundary_backend,
        )

    def new_fragment(
        self,
        state: PhysicalLayerState,
        *,
        hc_boundary_backend: Any | None = None,
    ) -> DirectPreMoEBlockFragment:
        return DirectPreMoEBlockFragment(
            weights=self.raw_block,
            attention=self.new_attention(state),
            norm_eps=self.norm_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            hc_eps=self.hc_eps,
            hc_boundary_backend=hc_boundary_backend,
        )


def build_physical_layer_material(
    *,
    layer_id: int,
    model_config: Mapping[str, Any],
    stage_root: Path,
    tp_rank: int,
    tp_group: object,
    tp_global_ranks: tuple[int, ...],
    device: torch.device,
    checkpoint_id: str,
    max_seq_len: int,
    global_row_shapes: tuple[int, ...],
    slots_per_shape: int = 4,
    attention_tp_shard: bool = False,
    progress_every: int = 64,
    progress: Callable[[str], None] | None = None,
    kv_dtype: str = "bf16",
    indexer_kv_dtype: str = "bf16",
    moe_marlin_input_dtype: torch.dtype | None = None,
    moe_buffer_donor: TP4MoE | None = None,
) -> PhysicalLayerMaterial:
    """Load and construct one real-weight physical layer on a TP subgroup."""

    specification = SUPPORTED_LAYER_SPECS.get(layer_id)
    if specification is None or bool(specification["is_mtp"]):
        raise PhysicalStageBuildError(
            f"layer {layer_id!r} is not a physical decode layer"
        )
    kind = str(specification["attn_kind"])
    route_kind = str(specification["route_kind"])
    collective_evidence = validate_live_tp_group(
        tp_group,
        expected_local_rank=tp_rank,
        expected_global_ranks=tp_global_ranks,
    )

    raw_block = load_replicated_block_weights(
        stage_root=stage_root,
        rank=tp_rank,
        world_size=EXPECTED_TP_SIZE,
        layer_id=layer_id,
        device=device,
        checkpoint_id=checkpoint_id,
    )
    moe_resident = load_resident_moe_layer(
        stage_root=stage_root,
        layer_id=layer_id,
        rank=tp_rank,
        world_size=EXPECTED_TP_SIZE,
        hidden_size=int(model_config["hidden_size"]),
        intermediate_size=int(model_config["moe_intermediate_size"]),
        n_experts=int(model_config["n_routed_experts"]),
        device=device,
        progress_every=progress_every,
        progress=progress,
        checkpoint_id=checkpoint_id,
        marlin_input_dtype=moe_marlin_input_dtype,
    )
    if (
        moe_marlin_input_dtype is None
        and moe_resident.resident_bytes != EXPECTED_MOE_RESIDENT_BYTES
    ):
        raise PhysicalStageBuildError(
            f"layer-{layer_id} MoE resident bytes {moe_resident.resident_bytes} "
            f"!= {EXPECTED_MOE_RESIDENT_BYTES}"
        )
    moe = TP4MoE(
        config=TP4MoEConfig(
            hidden_size=int(model_config["hidden_size"]),
            intermediate_size=int(model_config["moe_intermediate_size"]),
            experts=int(model_config["n_routed_experts"]),
            topk=int(model_config["num_experts_per_tok"]),
            route_scale=float(model_config["routed_scaling_factor"]),
            clamp_limit=float(model_config["swiglu_limit"]),
            world_size=EXPECTED_TP_SIZE,
        ),
        resident=moe_resident,
        gate=raw_block.gate,
        rank=tp_rank,
        device=device,
        global_row_shapes=global_row_shapes,
        group=tp_group,
        slots_per_shape=slots_per_shape,
        marlin_input_dtype=moe_marlin_input_dtype,
        buffer_donor=(
            moe_buffer_donor
            if moe_buffer_donor is not None
            and moe_buffer_donor.route_kind == route_kind
            else None
        ),
    )
    if moe.route_kind != route_kind:
        raise PhysicalStageBuildError(
            f"layer-{layer_id} constructed MoE routing {moe.route_kind} "
            f"!= contract {route_kind}"
        )

    identity = {
        "layer_id": layer_id,
        "rank": tp_rank,
        "world_size": EXPECTED_TP_SIZE,
        "checkpoint_id": checkpoint_id,
    }
    if kind == "window":
        attention_config: Any = WindowAttentionConfig.from_model_config(
            model_config, layer_id=layer_id, max_seq_len=max_seq_len
        )
        prepared = prepare_window_attention_weights(
            raw_block.attention, **identity
        )
    elif kind == "ratio4":
        attention_config = Ratio4AttentionConfig.from_model_config(
            model_config, layer_id=layer_id, max_seq_len=max_seq_len
        )
        prepared = prepare_ratio4_attention_weights(
            raw_block.attention, **identity
        )
    else:
        attention_config = Ratio128AttentionConfig.from_model_config(
            model_config, layer_id=layer_id, max_seq_len=max_seq_len
        )
        prepared = prepare_attention_weights(raw_block.attention, **identity)

    # E6F variant A: shard the o-path here rather than at instantiation, so the
    # unsharded copy is never resident alongside the sharded one -- the whole
    # point of sharding is the byte count, and holding both would defeat it.
    # Note this makes ``material.prepared`` the *sharded* weights for this
    # build; the prefill bench builds ratio-4 through Ratio4FullPositionAttention
    # from an unsharded build, so it is unaffected.
    if attention_tp_shard:
        attention_config = replace(
            attention_config, tp_size=EXPECTED_TP_SIZE, tp_rank=tp_rank
        )
        attention_config.validate()
        shard_fn = {
            "window": shard_window_attention_weights,
            "ratio4": shard_ratio4_attention_weights,
            "ratio128": shard_ratio128_attention_weights,
        }[kind]
        prepared = shard_fn(
            prepared,
            tp_rank=tp_rank,
            tp_size=EXPECTED_TP_SIZE,
            config=replace(attention_config, tp_size=1, tp_rank=0),
        )

    evidence = {
        "layer_id": layer_id,
        "kind": kind,
        "attention_tp_shard": bool(attention_tp_shard),
        "route_kind": route_kind,
        "checkpoint_id": checkpoint_id,
        "moe_resident_bytes": int(moe_resident.resident_bytes),
        "registered_global_rows": list(global_row_shapes),
        "moe_slots_per_shape": int(slots_per_shape),
        "tp_collective": dict(collective_evidence),
        "kv_dtype": kv_dtype,
        "indexer_kv_dtype": indexer_kv_dtype,
        "moe_marlin_input_dtype": (
            None if moe_marlin_input_dtype is None else str(moe_marlin_input_dtype)
        ),
    }
    return PhysicalLayerMaterial(
        layer_id=layer_id,
        kind=kind,
        route_kind=route_kind,
        tp_rank=tp_rank,
        tp_size=EXPECTED_TP_SIZE,
        checkpoint_id=checkpoint_id,
        device=device,
        max_seq_len=max_seq_len,
        raw_block=raw_block,
        prepared=prepared,
        attention_config=attention_config,
        moe=moe,
        norm_eps=float(model_config["rms_norm_eps"]),
        sinkhorn_iters=int(model_config["hc_sinkhorn_iters"]),
        hc_eps=float(model_config["hc_eps"]),
        evidence=evidence,
        kv_dtype=kv_dtype,
        indexer_kv_dtype=indexer_kv_dtype,
    )


@dataclass(frozen=True, slots=True)
class PhysicalStageMaterial:
    """All layer materials of one PP stage plus assembly evidence."""

    stage_id: int
    layer_ids: tuple[int, ...]
    materials: tuple[PhysicalLayerMaterial, ...]
    evidence: Mapping[str, Any]

    def material(self, layer_id: int) -> PhysicalLayerMaterial:
        for candidate in self.materials:
            if candidate.layer_id == layer_id:
                return candidate
        raise PhysicalStageBuildError(
            f"stage {self.stage_id} does not own layer {layer_id}"
        )


def build_physical_stage(
    *,
    stage_id: int,
    layer_ids: tuple[int, ...] | None = None,
    model_config: Mapping[str, Any],
    stage_root: Path,
    tp_rank: int,
    tp_group: object,
    tp_global_ranks: tuple[int, ...],
    device: torch.device,
    checkpoint_id: str,
    max_seq_len: int,
    global_row_shapes: tuple[int, ...],
    slots_per_shape: int = 4,
    attention_tp_shard: bool = False,
    progress_every: int = 64,
    progress: Callable[[str], None] | None = None,
    kv_dtype: str = "bf16",
    indexer_kv_dtype: str = "bf16",
    moe_marlin_input_dtype: torch.dtype | None = None,
    share_moe_buffers: bool = False,
) -> PhysicalStageMaterial:
    """Load every layer material of one PP stage through one code path.

    ``layer_ids`` defaults to the frozen scaled PP2 table; the sequential
    reference chain passes the full L0-L11 window explicitly.
    """

    selected = pp2_stage_layer_ids(stage_id) if layer_ids is None else layer_ids
    if not selected or any(
        right != left + 1 for left, right in zip(selected, selected[1:])
    ):
        raise PhysicalStageBuildError(
            f"stage layer ids must be non-empty and consecutive, got {selected}"
        )
    materials = []
    # One donor *per route kind*.  A hash-routed MoE registers an extra
    # ``gathered_input_ids`` buffer that learned routing does not have, so the
    # two kinds cannot share a buffer set and ``build_physical_layer_material``
    # rejects a mismatched donor.  Electing a single donor from the stage's
    # first layer therefore silently disabled sharing for every learned layer
    # of stage 0 (layers 0-2 are hash, 3-10 learned; model_contract.py:101) --
    # 8 unshared layers, which at a 32768-row prefill shape is 8 x 2.5 GiB and
    # OOMs at load.  Stages 1-3 are all-learned and were unaffected, which is
    # why the C2F single-stage bench (layers 11-21) never saw this.
    moe_buffer_donors: dict[str, TP4MoE] = {}
    for layer_id in selected:
        route_kind = str(SUPPORTED_LAYER_SPECS[layer_id]["route_kind"])
        materials.append(
            build_physical_layer_material(
                layer_id=layer_id,
                model_config=model_config,
                stage_root=stage_root,
                tp_rank=tp_rank,
                tp_group=tp_group,
                tp_global_ranks=tp_global_ranks,
                device=device,
                checkpoint_id=checkpoint_id,
                max_seq_len=max_seq_len,
                global_row_shapes=global_row_shapes,
                slots_per_shape=slots_per_shape,
                attention_tp_shard=attention_tp_shard,
                progress_every=progress_every,
                progress=progress,
                kv_dtype=kv_dtype,
                indexer_kv_dtype=indexer_kv_dtype,
                moe_marlin_input_dtype=moe_marlin_input_dtype,
                moe_buffer_donor=moe_buffer_donors.get(route_kind),
            )
        )
        if share_moe_buffers:
            moe_buffer_donors.setdefault(route_kind, materials[-1].moe)
        if progress is not None:
            progress(f"stage {stage_id} layer {layer_id} material loaded")
    evidence = {
        "stage_id": stage_id,
        "layer_ids": list(selected),
        "layers": [dict(material.evidence) for material in materials],
        "tp_global_ranks": list(tp_global_ranks),
        "tp_rank": tp_rank,
        "kv_dtype": kv_dtype,
        "indexer_kv_dtype": indexer_kv_dtype,
    }
    return PhysicalStageMaterial(
        stage_id=stage_id,
        layer_ids=tuple(selected),
        materials=tuple(materials),
        evidence=evidence,
    )


__all__ = [
    "EXPECTED_MOE_RESIDENT_BYTES",
    "EXPECTED_TP_SIZE",
    "NOPE_QUANT_MODE",
    "PP2_STAGE_LAYER_IDS",
    "PP2_WORLD_SIZE",
    "PhysicalLayerMaterial",
    "PhysicalLayerState",
    "PhysicalStageBuildError",
    "PhysicalStageMaterial",
    "build_physical_layer_material",
    "build_physical_stage",
    "pp2_stage_layer_ids",
    "validate_live_tp_group",
]
