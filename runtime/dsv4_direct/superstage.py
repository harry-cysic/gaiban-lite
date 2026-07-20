"""Direct-owned consecutive TP4 decode stages (V4-Flash port).

Ported from gaiban ``dsv4_direct/superstage.py`` (``TP4DecodeStage`` :108,
``TP4DecodeSuperStage`` :1482).  Flash-specific changes:

- **Three-way layer dispatch.**  Pro stages contained only ratio-4/ratio-128
  layers; Flash adds the pure sliding-window type (L0/L1), so the per-layer
  plan typing (eager and stateful) and the stateful validator calls dispatch
  on ``compression_ratio in {0, 4, 128}``.  Window layers have no stateful
  boundary variant (reference model.py:530: unconditional ring write), so
  their validator takes no boundary flag.
- **Geometry.**  The residual contract is ``[local_batch, 1, 4, 4096]``
  via ``block.BLOCK_HC_MULT`` / ``block.BLOCK_HIDDEN_SIZE`` (Pro hard-coded
  ``4``/``7168``).
- **Canonical slice.**  ``SUPERSTAGE_LAYER_IDS`` is the Flash L0-L5 window
  (window x2, ratio-4 x2 at L2/L4, ratio-128 x2 at L3/L5), covering every
  layer type and every graph-family boundary combination; Pro's was L2-L5.
- **Graph families.**  Unchanged three-family set; see
  ``stateful_decode.DecodeGraphFamily`` for the Flash derivation (window
  layers add no boundary write at any position, so no fourth family exists).

The gaiban ``attention_pre_backend`` / ``ffn_post_pre_backend`` fused-operator
injection is superseded here by one optional ``hc_boundary_backend`` (E0hf):
when set, the stateful graph body runs a boundary-fused chain in which every
intra-layer (attention -> FFN) and inter-layer (FFN -> next attention) HC
boundary goes through the backend's ``post_pre_norm`` op; the stage-first
attention-side ``hc_pre`` and the stage-last tail ``hc_post`` have no fusion
partner and stay eager.  With the default ``None`` the body is byte-for-byte
the pre-E0hf per-block loop.  The PP fragment surfaces remain out of scope.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .attention import Ratio128StatefulDecodePlan
from .block import BLOCK_HC_MULT, BLOCK_HIDDEN_SIZE, DirectDecodeBlock
from .hyper_connections import hc_post
from .model_contract import SUPPORTED_LAYER_SPECS
from .ratio4_attention import Ratio4StatefulDecodePlan
from .stateful_decode import (
    DecodeGraphFamily,
    StatefulDecodeCursor,
    family_boundary_flags,
)
from .window_attention import WindowStatefulDecodePlan


# Flash canonical stage slice: every layer type and family boundary combo.
SUPERSTAGE_LAYER_IDS = (0, 1, 2, 3, 4, 5)

BlockStatefulPlan = (
    WindowStatefulDecodePlan
    | Ratio4StatefulDecodePlan
    | Ratio128StatefulDecodePlan
)


@dataclass(frozen=True)
class TP4DecodeSuperStagePlan:
    """Independently owned attention plans for one decode position."""

    start_pos: int
    batch_size: int
    hidden_size: int
    owner_id: int
    layer_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    attention_ids: tuple[int, ...]
    state_ids: tuple[int, ...]
    layer_plans: tuple[Any, ...]


@dataclass(frozen=True)
class TP4StatefulDecodeSuperStagePlan:
    """Immutable stage graph workspace bound to one shared decode cursor."""

    start_position: int
    stop_position: int
    batch_size: int
    hidden_size: int
    owner_id: int
    cursor_id: int
    layer_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    attention_ids: tuple[int, ...]
    state_ids: tuple[int, ...]
    layer_plans: tuple[BlockStatefulPlan, ...]
    cursor: StatefulDecodeCursor
    position: torch.Tensor
    state_position_tensors: tuple[torch.Tensor, ...]
    expected_position: torch.Tensor
    stop_position_tensor: torch.Tensor
    input_residual_buffer: torch.Tensor
    input_ids_buffer: torch.Tensor
    output_buffer: torch.Tensor
    graph_moe_slots: tuple[int, int, int]
    slot_buffer_ids: tuple[int, ...]
    tensor_pointers: tuple[int, ...]

    @property
    def resident_bytes(self) -> int:
        stage_bytes = sum(
            int(value.numel() * value.element_size())
            for value in (
                self.expected_position,
                self.stop_position_tensor,
                self.input_residual_buffer,
                self.input_ids_buffer,
                self.output_buffer,
            )
        )
        return stage_bytes + sum(
            int(layer_plan.resident_bytes) for layer_plan in self.layer_plans
        )


_STATEFUL_GRAPH_FAMILIES = (
    DecodeGraphFamily.NORMAL,
    DecodeGraphFamily.RATIO4_BOUNDARY,
    DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY,
)


def _require_unique(label: str, values: Sequence[object]) -> None:
    if len({id(value) for value in values}) != len(values):
        raise ValueError(f"super-stage {label} must not alias across layers")


def _require_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "super-stage composite_checkpoint_id must be a lowercase SHA-256"
        )
    return value


class TP4DecodeStage:
    """Own one non-empty sequence of distinct, consecutive TP4 decode blocks."""

    def __init__(
        self,
        blocks: Sequence[DirectDecodeBlock],
        *,
        expected_layer_ids: Sequence[int] | None = None,
        hc_boundary_backend: Any | None = None,
    ) -> None:
        if hc_boundary_backend is not None and not callable(
            getattr(hc_boundary_backend, "post_pre_norm", None)
        ):
            raise TypeError(
                "hc_boundary_backend must expose a post_pre_norm operator"
            )
        self.hc_boundary_backend = hc_boundary_backend
        self.blocks = tuple(blocks)
        expected = (
            None if expected_layer_ids is None else tuple(expected_layer_ids)
        )
        if expected is not None and len(self.blocks) != len(expected):
            raise ValueError(
                f"TP4 decode super-stage requires layers {expected}"
            )
        if not self.blocks:
            raise ValueError("TP4 decode stage requires at least one block")
        if any(not isinstance(block, DirectDecodeBlock) for block in self.blocks):
            raise TypeError("super-stage blocks must be DirectDecodeBlock instances")
        _require_unique("blocks", self.blocks)

        layer_ids = tuple(getattr(block, "layer_id", None) for block in self.blocks)
        if expected is not None and layer_ids != expected:
            raise ValueError(
                f"TP4 decode super-stage layer order {layer_ids} "
                f"!= {expected}"
            )
        if any(
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id not in SUPPORTED_LAYER_SPECS
            for layer_id in layer_ids
        ):
            raise ValueError(
                f"TP4 decode stage has unsupported layer order {layer_ids}"
            )
        if any(right != left + 1 for left, right in zip(layer_ids, layer_ids[1:])):
            raise ValueError(
                f"TP4 decode stage layers must be consecutive, got {layer_ids}"
            )
        self.layer_ids = layer_ids

        attentions = tuple(block.attention for block in self.blocks)
        states = tuple(attention.state for attention in attentions)
        moes = tuple(block.moe for block in self.blocks)
        block_weights = tuple(block.weights for block in self.blocks)
        attention_weights = tuple(attention.weights for attention in attentions)
        residents = tuple(moe.resident for moe in moes)
        gates = tuple(block.weights.gate for block in self.blocks)
        moe_gates = tuple(moe.gate for moe in moes)
        _require_unique("attention runtimes", attentions)
        _require_unique("attention states", states)
        _require_unique("MoE runtimes", moes)
        _require_unique("block weights", block_weights)
        _require_unique("prepared attention weights", attention_weights)
        _require_unique("MoE residents", residents)
        _require_unique("gate weights", gates)
        _require_unique("MoE gate weights", moe_gates)
        gate_owners: dict[int, int] = {}
        for layer_id, block_gate, moe_gate in zip(
            self.layer_ids, gates, moe_gates, strict=True
        ):
            for gate in (block_gate, moe_gate):
                previous_owner = gate_owners.setdefault(id(gate), layer_id)
                if previous_owner != layer_id:
                    raise ValueError(
                        "super-stage gate weights must not alias across layers"
                    )

        first_weights = self.blocks[0].weights
        rank = first_weights.rank
        world_size = first_weights.world_size
        composite_checkpoint_id = _require_sha256(first_weights.checkpoint_id)
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size != 4
            or rank < 0
            or rank >= world_size
        ):
            raise ValueError("super-stage requires one valid TP4 rank identity")

        for layer_id, block, attention, state, moe in zip(
            self.layer_ids,
            self.blocks,
            attentions,
            states,
            moes,
            strict=True,
        ):
            specification = SUPPORTED_LAYER_SPECS[layer_id]
            expected_ratio = int(specification["compress_ratio"])
            expected_route = str(specification["route_kind"])
            if (block.compression_ratio, block.route_kind) != (
                expected_ratio,
                expected_route,
            ):
                raise ValueError(
                    f"layer {layer_id} block does not match compression/router spec"
                )

            expected_identity = (layer_id, rank, world_size)
            component_identities = {
                "block weights": (
                    block.weights.layer_id,
                    block.weights.rank,
                    block.weights.world_size,
                ),
                "block gate": (
                    block.weights.gate.layer_id,
                    block.weights.gate.rank,
                    block.weights.gate.world_size,
                ),
                "attention weights": (
                    attention.weights.layer_id,
                    attention.weights.rank,
                    attention.weights.world_size,
                ),
                "MoE runtime": (moe.layer_id, moe.rank, moe.config.world_size),
                "MoE resident": (
                    moe.resident.layer_id,
                    moe.resident.rank,
                    moe.resident.world_size,
                ),
                "MoE gate": (moe.gate.layer_id, moe.gate.rank, moe.gate.world_size),
            }
            for label, observed in component_identities.items():
                if observed != expected_identity:
                    raise ValueError(
                        f"layer {layer_id} {label} identity {observed} "
                        f"!= {expected_identity}"
                    )
            if attention.config.layer_id != layer_id or state.layer_id != layer_id:
                raise ValueError(
                    f"layer {layer_id} attention config/state identity differs"
                )

            checkpoint_ids = (
                block.weights.checkpoint_id,
                block.weights.gate.checkpoint_id,
                attention.weights.checkpoint_id,
                moe.resident.checkpoint_id,
                moe.gate.checkpoint_id,
            )
            if any(value != composite_checkpoint_id for value in checkpoint_ids):
                raise ValueError(
                    f"layer {layer_id} components do not share the super-stage "
                    "composite_checkpoint_id"
                )
            if (block.weights.gate.route_kind, moe.route_kind, moe.gate.route_kind) != (
                expected_route,
                expected_route,
                expected_route,
            ):
                raise ValueError(
                    f"layer {layer_id} gate/MoE routing does not match {expected_route}"
                )

        buffer_maps = tuple(
            self._buffer_map(layer_id, moe)
            for layer_id, moe in zip(self.layer_ids, moes, strict=True)
        )
        expected_buffer_keys = set(buffer_maps[0])
        if not expected_buffer_keys:
            raise ValueError("super-stage MoE runtimes require registered slot buffers")
        if any(set(buffers) != expected_buffer_keys for buffers in buffer_maps[1:]):
            raise ValueError("super-stage MoE slot registrations differ across layers")
        slot_buffers = tuple(
            buffers[key]
            for buffers in buffer_maps
            for key in sorted(expected_buffer_keys)
        )
        _require_unique("MoE slot buffers", slot_buffers)

        self.rank = rank
        self.world_size = world_size
        self.composite_checkpoint_id = composite_checkpoint_id
        self.checkpoint_id = composite_checkpoint_id
        self.attentions = attentions
        self.states = states
        self.moes = moes
        self._buffer_keys = frozenset(expected_buffer_keys)
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @staticmethod
    def _buffer_map(layer_id: int, moe: object) -> Mapping[tuple[int, int], object]:
        buffers = getattr(moe, "_buffers", None)
        if not isinstance(buffers, Mapping):
            raise TypeError(f"layer {layer_id} MoE must expose its slot buffer mapping")
        for key, value in buffers.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(
                    isinstance(part, int) and not isinstance(part, bool)
                    for part in key
                )
                or key[0] <= 0
                or key[1] < 0
                or value is None
            ):
                raise ValueError(
                    f"layer {layer_id} has an invalid MoE slot registration"
                )
        return buffers

    @staticmethod
    def _validate_plan_fields(
        plan: object,
        *,
        layer_id: int,
        attention: object,
        state: object,
        start_pos: int,
    ) -> tuple[int, int]:
        if getattr(plan, "owner_id", None) != id(attention):
            raise ValueError(f"layer {layer_id} decode plan has the wrong owner")
        if getattr(plan, "state_id", None) != id(state):
            raise ValueError(f"layer {layer_id} decode plan has the wrong state owner")
        if getattr(plan, "start_pos", None) != start_pos:
            raise ValueError(f"layer {layer_id} decode plan has the wrong start_pos")
        batch_size = getattr(plan, "batch_size", None)
        hidden_size = getattr(plan, "hidden_size", None)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
            or not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or hidden_size != BLOCK_HIDDEN_SIZE
        ):
            raise ValueError(f"layer {layer_id} decode plan has an invalid shape")
        return batch_size, hidden_size

    def prepare_decode_plan(
        self,
        start_pos: int,
        *,
        advance_ratio4_overlap_state: bool = True,
    ) -> TP4DecodeSuperStagePlan:
        if (
            not isinstance(start_pos, int)
            or isinstance(start_pos, bool)
            or start_pos < 128
        ):
            raise ValueError("super-stage decode start_pos must be an integer >= 128")
        if not isinstance(advance_ratio4_overlap_state, bool):
            raise TypeError("advance_ratio4_overlap_state must be bool")

        layer_plans = []
        shapes = []
        for layer_id, block, attention, state in zip(
            self.layer_ids,
            self.blocks,
            self.attentions,
            self.states,
            strict=True,
        ):
            if block.compression_ratio == 4:
                plan = attention.prepare_decode_plan(
                    start_pos,
                    advance_overlap_state=advance_ratio4_overlap_state,
                )
            else:
                # Window and ratio-128 plans share the positional-only setup.
                plan = attention.prepare_decode_plan(start_pos)
            shapes.append(
                self._validate_plan_fields(
                    plan,
                    layer_id=layer_id,
                    attention=attention,
                    state=state,
                    start_pos=start_pos,
                )
            )
            layer_plans.append(plan)

        _require_unique("decode plans", layer_plans)
        if len(set(shapes)) != 1:
            raise ValueError("super-stage layer plans disagree on decode shape")
        batch_size, hidden_size = shapes[0]
        return TP4DecodeSuperStagePlan(
            start_pos=start_pos,
            batch_size=batch_size,
            hidden_size=hidden_size,
            owner_id=id(self),
            layer_ids=self.layer_ids,
            block_ids=tuple(id(block) for block in self.blocks),
            attention_ids=tuple(id(attention) for attention in self.attentions),
            state_ids=tuple(id(state) for state in self.states),
            layer_plans=tuple(layer_plans),
        )

    @staticmethod
    def _stateful_layer_tensor_items(
        plan: BlockStatefulPlan,
    ) -> tuple[torch.Tensor, ...]:
        # Order must match each plan type's own ``tensor_pointers`` manifest.
        if isinstance(plan, WindowStatefulDecodePlan):
            return (
                plan.position,
                plan.window_columns,
                plan.gather_indices,
                plan.batch_indices,
            )
        if isinstance(plan, Ratio4StatefulDecodePlan):
            return (
                plan.position,
                plan.window_columns,
                plan.compressed_columns,
                plan.topk_indices,
                plan.batch_indices,
            )
        if isinstance(plan, Ratio128StatefulDecodePlan):
            return (
                plan.position,
                plan.topk_indices,
                plan.gather_indices,
                plan.valid_mask,
                plan.batch_indices,
            )
        raise TypeError("super-stage contains an invalid stateful attention plan")

    @staticmethod
    def _stateful_plan_type(block: DirectDecodeBlock) -> type:
        if block.compression_ratio == 0:
            return WindowStatefulDecodePlan
        if block.compression_ratio == 4:
            return Ratio4StatefulDecodePlan
        return Ratio128StatefulDecodePlan

    @staticmethod
    def _validate_graph_moe_slots(
        graph_moe_slots: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        if (
            not isinstance(graph_moe_slots, tuple)
            or len(graph_moe_slots) != len(_STATEFUL_GRAPH_FAMILIES)
            or any(
                not isinstance(slot, int) or isinstance(slot, bool) or slot < 0
                for slot in graph_moe_slots
            )
        ):
            raise ValueError(
                "stateful graph_moe_slots must be three non-negative integers"
            )
        if len(set(graph_moe_slots)) != len(graph_moe_slots):
            raise ValueError("stateful graph families require distinct MoE slots")
        return graph_moe_slots

    def _stateful_slot_buffer_ids(
        self,
        *,
        batch_size: int,
        graph_moe_slots: tuple[int, int, int],
    ) -> tuple[int, ...]:
        global_rows = batch_size * self.world_size
        buffers = []
        for family, slot in zip(
            _STATEFUL_GRAPH_FAMILIES, graph_moe_slots, strict=True
        ):
            slot_key = (global_rows, slot)
            if slot_key not in self._buffer_keys:
                raise ValueError(
                    f"stateful {family.value} MoE slot {slot_key} is not registered "
                    "on every layer"
                )
            for layer_id, moe in zip(
                self.layer_ids, self.moes, strict=True
            ):
                layer_buffers = self._buffer_map(layer_id, moe)
                if slot_key not in layer_buffers:
                    raise ValueError(f"layer {layer_id} lost MoE slot {slot_key}")
                buffers.append(layer_buffers[slot_key])
        _require_unique("stateful MoE slot buffers", buffers)
        return tuple(id(buffer) for buffer in buffers)

    def _validate_stateful_slot_free(
        self,
        *,
        batch_size: int,
        graph_family: DecodeGraphFamily,
        graph_moe_slots: tuple[int, int, int],
    ) -> int:
        family_index = _STATEFUL_GRAPH_FAMILIES.index(graph_family)
        moe_slot = graph_moe_slots[family_index]
        global_rows = batch_size * self.world_size
        slot_key = (global_rows, moe_slot)
        for layer_id, moe in zip(self.layer_ids, self.moes, strict=True):
            selected_buffer = self._buffer_map(layer_id, moe)[slot_key]
            completion_status = getattr(moe, "slot_completion_status", None)
            if not callable(completion_status):
                raise TypeError(
                    f"layer {layer_id} MoE does not expose slot completion status"
                )
            status = completion_status(global_rows, moe_slot)
            if not isinstance(status, Mapping) or not isinstance(
                status.get("state"), str
            ):
                raise TypeError(
                    f"layer {layer_id} MoE returned invalid slot completion status"
                )
            if getattr(selected_buffer, "state", None) != status["state"]:
                raise RuntimeError(
                    f"layer {layer_id} MoE slot {slot_key} lifecycle disagrees"
                )
            if status["state"] != "free":
                raise RuntimeError(
                    f"layer {layer_id} MoE slot {slot_key} is {status['state']}; "
                    "worker restart is required"
                )
        return moe_slot

    @classmethod
    def _stateful_tensor_pointers(
        cls,
        *,
        layer_ids: Sequence[int],
        cursor: StatefulDecodeCursor,
        expected_position: torch.Tensor,
        stop_position_tensor: torch.Tensor,
        input_residual_buffer: torch.Tensor,
        input_ids_buffer: torch.Tensor,
        output_buffer: torch.Tensor,
        states: Sequence[object],
        layer_plans: Sequence[BlockStatefulPlan],
    ) -> tuple[int, ...]:
        tensors = [
            cursor.device_position,
            cursor.dispatch_error,
            expected_position,
            stop_position_tensor,
            input_residual_buffer,
            input_ids_buffer,
            output_buffer,
        ]
        for layer_plan in layer_plans:
            layer_tensors = cls._stateful_layer_tensor_items(layer_plan)
            if layer_tensors[0] is not cursor.device_position:
                raise ValueError(
                    "every stateful attention plan must share the cursor position"
                )
            layer_pointers = tuple(
                int(value.untyped_storage().data_ptr()) for value in layer_tensors
            )
            if layer_pointers != layer_plan.tensor_pointers:
                raise ValueError(
                    "stateful attention plan tensor storage differs from setup"
                )
            tensors.extend(layer_tensors[1:])
        for layer_id, state in zip(layer_ids, states, strict=True):
            validate_owned = getattr(state, "_validate_owned_tensor_contract", None)
            owned_tensors = getattr(state, "_owned_tensors", None)
            if not callable(validate_owned) or not callable(owned_tensors):
                raise TypeError(
                    f"layer {layer_id} state lacks its owned-tensor contract"
                )
            items = validate_owned(label=f"layer-{layer_id} stateful super-stage")
            state_tensors = owned_tensors()
            if (
                not isinstance(items, tuple)
                or not isinstance(state_tensors, tuple)
                or not state_tensors
                or len(items) != len(state_tensors)
                or any(
                    not isinstance(item, tuple)
                    or len(item) != 2
                    or item[1] is not tensor
                    for item, tensor in zip(items, state_tensors, strict=True)
                )
            ):
                raise RuntimeError(
                    f"layer {layer_id} state owned-tensor enumeration differs"
                )
            for tensor in state_tensors:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.device != cursor.device
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"layer {layer_id} state tensor metadata differs"
                    )
            tensors.extend(state_tensors)
        pointers = tuple(
            int(value.untyped_storage().data_ptr()) for value in tensors
        )
        if len(set(pointers)) != len(pointers):
            raise ValueError(
                "stateful cursor, output, and layer workspaces must not alias"
            )
        return pointers

    @staticmethod
    def _stateful_state_position_tensors(
        states: Sequence[object],
    ) -> tuple[torch.Tensor, ...]:
        positions = tuple(getattr(state, "_next_position", None) for state in states)
        if (
            not positions
            or any(not isinstance(value, torch.Tensor) for value in positions)
        ):
            raise TypeError(
                "stateful super-stage states must expose device next-position tensors"
            )
        typed_positions = tuple(positions)  # type: ignore[arg-type]
        expected_shape = tuple(typed_positions[0].shape)
        if (
            not expected_shape
            or any(
                tuple(value.shape) != expected_shape
                or value.dtype != torch.int64
                or not value.is_contiguous()
                for value in typed_positions
            )
        ):
            raise ValueError("stateful state next-position tensor metadata differs")
        if len({int(value.untyped_storage().data_ptr()) for value in typed_positions}) != len(
            typed_positions
        ):
            raise ValueError("stateful state next-position tensors must not alias")
        return typed_positions

    def prepare_stateful_decode_plan(
        self,
        cursor: StatefulDecodeCursor,
        *,
        start_position: int,
        stop_position: int,
        graph_moe_slots: tuple[int, int, int] = (1, 2, 3),
    ) -> TP4StatefulDecodeSuperStagePlan:
        """Prepare one fixed-shape stage workspace for a consecutive range."""

        if self._poisoned:
            raise RuntimeError("super-stage is poisoned; worker restart is required")
        if not isinstance(cursor, StatefulDecodeCursor):
            raise TypeError("cursor must be a StatefulDecodeCursor")
        if (
            not isinstance(start_position, int)
            or isinstance(start_position, bool)
            or not isinstance(stop_position, int)
            or isinstance(stop_position, bool)
            or start_position < 128
            or stop_position <= start_position
        ):
            raise ValueError(
                "stateful super-stage range must be a non-empty interval at >= 128"
            )
        if cursor.host_position != start_position:
            raise ValueError("cursor host position does not match stateful range start")
        cursor.validate_contract()
        if int(cursor.device_position.item()) != start_position:
            raise ValueError("cursor device position does not match stateful range start")
        if int(cursor.dispatch_error.item()) != 0:
            raise RuntimeError("stateful cursor has a sticky dispatch error")
        graph_moe_slots = self._validate_graph_moe_slots(graph_moe_slots)

        layer_plans = []
        shapes = []
        for layer_id, block, attention, state in zip(
            self.layer_ids,
            self.blocks,
            self.attentions,
            self.states,
            strict=True,
        ):
            if getattr(state, "device", None) != cursor.device:
                raise ValueError(
                    f"layer {layer_id} attention state and cursor device differ"
                )
            layer_plan = attention.prepare_stateful_decode_plan(
                position=cursor.device_position,
                start_position=start_position,
                stop_position=stop_position,
            )
            expected_type = self._stateful_plan_type(block)
            if not isinstance(layer_plan, expected_type):
                raise TypeError(
                    f"layer {layer_id} returned the wrong stateful plan type"
                )
            if (
                layer_plan.owner_id != id(attention)
                or layer_plan.state_id != id(state)
                or layer_plan.start_position != start_position
                or layer_plan.stop_position != stop_position
                or layer_plan.position is not cursor.device_position
            ):
                raise ValueError(
                    f"layer {layer_id} stateful plan ownership/range differs"
                )
            shape = (layer_plan.batch_size, layer_plan.hidden_size)
            if (
                not isinstance(shape[0], int)
                or isinstance(shape[0], bool)
                or shape[0] <= 0
                or shape[1] != BLOCK_HIDDEN_SIZE
            ):
                raise ValueError(f"layer {layer_id} stateful plan has invalid shape")
            self._stateful_layer_tensor_items(layer_plan)
            shapes.append(shape)
            layer_plans.append(layer_plan)

        _require_unique("stateful attention plans", layer_plans)
        if len(set(shapes)) != 1:
            raise ValueError("stateful layer plans disagree on decode shape")
        batch_size, hidden_size = shapes[0]
        residual_shape = (batch_size, 1, BLOCK_HC_MULT, hidden_size)
        input_residual_buffer = torch.empty(
            residual_shape,
            dtype=torch.bfloat16,
            device=cursor.device,
        )
        input_ids_buffer = torch.empty(
            (batch_size, 1), dtype=torch.int64, device=cursor.device
        )
        output_buffer = torch.empty(
            residual_shape, dtype=torch.bfloat16, device=cursor.device
        )
        expected_position = torch.full(
            (1,), start_position, dtype=torch.int64, device=cursor.device
        )
        stop_position_tensor = torch.full(
            (1,), stop_position, dtype=torch.int64, device=cursor.device
        )
        state_position_tensors = self._stateful_state_position_tensors(self.states)
        slot_buffer_ids = self._stateful_slot_buffer_ids(
            batch_size=batch_size,
            graph_moe_slots=graph_moe_slots,
        )
        for graph_family in _STATEFUL_GRAPH_FAMILIES:
            self._validate_stateful_slot_free(
                batch_size=batch_size,
                graph_family=graph_family,
                graph_moe_slots=graph_moe_slots,
            )
        tensor_pointers = self._stateful_tensor_pointers(
            layer_ids=self.layer_ids,
            cursor=cursor,
            expected_position=expected_position,
            stop_position_tensor=stop_position_tensor,
            input_residual_buffer=input_residual_buffer,
            input_ids_buffer=input_ids_buffer,
            output_buffer=output_buffer,
            states=self.states,
            layer_plans=layer_plans,
        )
        return TP4StatefulDecodeSuperStagePlan(
            start_position=start_position,
            stop_position=stop_position,
            batch_size=batch_size,
            hidden_size=hidden_size,
            owner_id=id(self),
            cursor_id=id(cursor),
            layer_ids=self.layer_ids,
            block_ids=tuple(id(block) for block in self.blocks),
            attention_ids=tuple(id(attention) for attention in self.attentions),
            state_ids=tuple(id(state) for state in self.states),
            layer_plans=tuple(layer_plans),
            cursor=cursor,
            position=cursor.device_position,
            state_position_tensors=state_position_tensors,
            expected_position=expected_position,
            stop_position_tensor=stop_position_tensor,
            input_residual_buffer=input_residual_buffer,
            input_ids_buffer=input_ids_buffer,
            output_buffer=output_buffer,
            graph_moe_slots=graph_moe_slots,
            slot_buffer_ids=slot_buffer_ids,
            tensor_pointers=tensor_pointers,
        )

    def _validate_forward(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        start_pos: int,
        plan: TP4DecodeSuperStagePlan,
        moe_slot: int,
    ) -> None:
        if self._poisoned:
            raise RuntimeError(
                "super-stage is poisoned; worker restart is required"
            )
        if not isinstance(plan, TP4DecodeSuperStagePlan):
            raise TypeError("plan must be a TP4DecodeSuperStagePlan")
        expected_owners = (
            id(self),
            self.layer_ids,
            tuple(id(block) for block in self.blocks),
            tuple(id(attention) for attention in self.attentions),
            tuple(id(state) for state in self.states),
        )
        observed_owners = (
            plan.owner_id,
            plan.layer_ids,
            plan.block_ids,
            plan.attention_ids,
            plan.state_ids,
        )
        if observed_owners != expected_owners:
            raise ValueError("decode plan belongs to a different super-stage")
        if start_pos != plan.start_pos:
            raise ValueError("decode start_pos does not match the super-stage plan")
        if len(plan.layer_plans) != len(self.layer_ids):
            raise ValueError("super-stage plan must contain one plan per layer")
        if len({id(layer_plan) for layer_plan in plan.layer_plans}) != len(
            plan.layer_plans
        ):
            raise ValueError("super-stage layer plans must not alias")

        if not isinstance(residual, torch.Tensor):
            raise TypeError("super-stage residual must be a tensor")
        expected_shape = (plan.batch_size, 1, BLOCK_HC_MULT, plan.hidden_size)
        if tuple(residual.shape) != expected_shape:
            raise ValueError(
                f"super-stage residual shape {tuple(residual.shape)} "
                f"!= {expected_shape}"
            )
        if residual.dtype != torch.bfloat16:
            raise TypeError("super-stage residual must be BF16")
        if (
            not isinstance(input_ids_local, torch.Tensor)
            or tuple(input_ids_local.shape) != expected_shape[:2]
            or input_ids_local.dtype != torch.int64
            or input_ids_local.device != residual.device
            or not input_ids_local.is_contiguous()
        ):
            raise ValueError(
                "super-stage input_ids_local must be contiguous device-local int64 "
                "with the residual batch/sequence shape"
            )
        if (
            not isinstance(moe_slot, int)
            or isinstance(moe_slot, bool)
            or moe_slot < 0
        ):
            raise ValueError("moe_slot must be a non-negative integer")

        global_rows = plan.batch_size * self.world_size
        slot_key = (global_rows, moe_slot)
        if slot_key not in self._buffer_keys:
            raise ValueError(f"MoE slot {slot_key} is not registered on every layer")
        selected_buffers = []
        for layer_id, moe in zip(self.layer_ids, self.moes, strict=True):
            buffers = self._buffer_map(layer_id, moe)
            if slot_key not in buffers:
                raise ValueError(f"layer {layer_id} lost MoE slot {slot_key}")
            selected_buffer = buffers[slot_key]
            selected_buffers.append(selected_buffer)
            completion_status = getattr(moe, "slot_completion_status", None)
            if not callable(completion_status):
                raise TypeError(
                    f"layer {layer_id} MoE does not expose slot completion status"
                )
            status = completion_status(global_rows, moe_slot)
            if not isinstance(status, Mapping) or not isinstance(
                status.get("state"), str
            ):
                raise TypeError(
                    f"layer {layer_id} MoE returned invalid slot completion status"
                )
            if getattr(selected_buffer, "state", None) != status["state"]:
                raise RuntimeError(
                    f"layer {layer_id} MoE slot {slot_key} lifecycle disagrees"
                )
            if status["state"] != "free":
                raise RuntimeError(
                    f"layer {layer_id} MoE slot {slot_key} is {status['state']}; "
                    "worker restart is required"
                )
        _require_unique("selected MoE slot buffers", selected_buffers)

        for layer_id, attention, state, layer_plan in zip(
            self.layer_ids,
            self.attentions,
            self.states,
            plan.layer_plans,
            strict=True,
        ):
            shape = self._validate_plan_fields(
                layer_plan,
                layer_id=layer_id,
                attention=attention,
                state=state,
                start_pos=start_pos,
            )
            if shape != (plan.batch_size, plan.hidden_size):
                raise ValueError(f"layer {layer_id} plan shape differs from stage plan")
            if getattr(state, "next_position", None) != start_pos:
                raise ValueError(
                    f"layer {layer_id} state is not positioned at {start_pos}"
                )

    def validate_decode_call(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        start_pos: int,
        plan: TP4DecodeSuperStagePlan,
        moe_slot: int = 0,
    ) -> None:
        """Validate every rank-local decode precondition before TP4 collectives."""

        self._validate_forward(
            residual,
            input_ids_local=input_ids_local,
            start_pos=start_pos,
            plan=plan,
            moe_slot=moe_slot,
        )

    def forward_decode_tensors(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        start_pos: int,
        plan: TP4DecodeSuperStagePlan,
        moe_slot: int = 0,
    ) -> tuple[torch.Tensor, ...]:
        """Return the output after each block while preserving exact call order."""

        self.validate_decode_call(
            residual,
            input_ids_local=input_ids_local,
            start_pos=start_pos,
            plan=plan,
            moe_slot=moe_slot,
        )
        expected_shape = tuple(residual.shape)
        output = residual
        outputs = []
        started = False
        try:
            for layer_id, block, state, layer_plan in zip(
                self.layer_ids,
                self.blocks,
                self.states,
                plan.layer_plans,
                strict=True,
            ):
                started = True
                output = block.forward_decode_tensor(
                    output,
                    input_ids_local=(
                        input_ids_local if block.route_kind == "hash" else None
                    ),
                    start_pos=start_pos,
                    attention_plan=layer_plan,
                    moe_slot=moe_slot,
                )
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"layer {layer_id} block output must be a tensor")
                if tuple(output.shape) != expected_shape:
                    raise ValueError(
                        f"layer {layer_id} block output shape {tuple(output.shape)} "
                        f"!= {expected_shape}"
                    )
                if output.dtype != residual.dtype or output.device != residual.device:
                    raise ValueError(
                        f"layer {layer_id} block output dtype/device differs from input"
                    )
                if getattr(state, "next_position", None) != start_pos + 1:
                    raise RuntimeError(
                        f"layer {layer_id} attention state did not advance exactly once"
                    )
                outputs.append(output)
        except BaseException:
            if started:
                self._poisoned = True
            raise
        return tuple(outputs)

    def forward_decode_tensor(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        start_pos: int,
        plan: TP4DecodeSuperStagePlan,
        moe_slot: int = 0,
    ) -> torch.Tensor:
        return self.forward_decode_tensors(
            residual,
            input_ids_local=input_ids_local,
            start_pos=start_pos,
            plan=plan,
            moe_slot=moe_slot,
        )[-1]

    def forward_decode_tensor_prevalidated(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        start_pos: int,
        plan: TP4DecodeSuperStagePlan,
        moe_slot: int,
    ) -> torch.Tensor:
        """Execute the graph hot path after an external ``validate_decode_call``."""

        output = residual
        for layer_id, block, layer_plan in zip(
            self.layer_ids,
            self.blocks,
            plan.layer_plans,
            strict=True,
        ):
            output = block.forward_decode_tensor(
                output,
                input_ids_local=(
                    input_ids_local if block.route_kind == "hash" else None
                ),
                start_pos=start_pos,
                attention_plan=layer_plan,
                moe_slot=moe_slot,
            )
        return output

    def _validate_stateful_forward(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        plan: TP4StatefulDecodeSuperStagePlan,
        graph_family: DecodeGraphFamily,
    ) -> int:
        if self._poisoned:
            raise RuntimeError("super-stage is poisoned; worker restart is required")
        if not isinstance(plan, TP4StatefulDecodeSuperStagePlan):
            raise TypeError("plan must be a TP4StatefulDecodeSuperStagePlan")
        if not isinstance(graph_family, DecodeGraphFamily):
            raise TypeError("graph_family must be a DecodeGraphFamily")
        expected_owners = (
            id(self),
            id(plan.cursor),
            self.layer_ids,
            tuple(id(block) for block in self.blocks),
            tuple(id(attention) for attention in self.attentions),
            tuple(id(state) for state in self.states),
        )
        observed_owners = (
            plan.owner_id,
            plan.cursor_id,
            plan.layer_ids,
            plan.block_ids,
            plan.attention_ids,
            plan.state_ids,
        )
        if observed_owners != expected_owners:
            raise ValueError("stateful plan belongs to a different super-stage")
        if not isinstance(plan.cursor, StatefulDecodeCursor):
            raise TypeError("stateful plan cursor has the wrong type")

        host_position = plan.cursor.host_position
        if not plan.start_position <= host_position < plan.stop_position:
            raise ValueError(
                f"host position {host_position} lies outside the stateful plan range"
            )
        if plan.cursor.host_family is not graph_family:
            raise ValueError(
                f"host position {host_position} requires "
                f"{plan.cursor.host_family.value}, not {graph_family.value}"
            )

        plan.cursor.validate_contract()
        if plan.position is not plan.cursor.device_position:
            raise ValueError("stateful plan lost its shared cursor position")
        if len(plan.layer_plans) != len(self.layer_ids) or len(
            {id(layer_plan) for layer_plan in plan.layer_plans}
        ) != len(plan.layer_plans):
            raise ValueError(
                "stateful super-stage requires one independent plan per layer"
            )
        if (
            not isinstance(plan.start_position, int)
            or isinstance(plan.start_position, bool)
            or not isinstance(plan.stop_position, int)
            or isinstance(plan.stop_position, bool)
            or plan.start_position < 128
            or plan.stop_position <= plan.start_position
        ):
            raise ValueError("stateful plan range is invalid")
        if (
            not isinstance(plan.batch_size, int)
            or isinstance(plan.batch_size, bool)
            or plan.batch_size <= 0
            or plan.hidden_size != BLOCK_HIDDEN_SIZE
        ):
            raise ValueError("stateful super-stage plan shape is invalid")

        if not isinstance(residual, torch.Tensor):
            raise TypeError("stateful super-stage residual must be a tensor")
        expected_shape = (plan.batch_size, 1, BLOCK_HC_MULT, plan.hidden_size)
        if tuple(residual.shape) != expected_shape:
            raise ValueError(
                f"stateful residual shape {tuple(residual.shape)} != {expected_shape}"
            )
        if residual.dtype != torch.bfloat16:
            raise TypeError("stateful super-stage residual must be BF16")
        if residual.device != plan.cursor.device or not residual.is_contiguous():
            raise ValueError(
                "stateful residual must be contiguous on the cursor device"
            )
        if (
            not isinstance(input_ids_local, torch.Tensor)
            or tuple(input_ids_local.shape) != expected_shape[:2]
            or input_ids_local.dtype != torch.int64
            or input_ids_local.device != residual.device
            or not input_ids_local.is_contiguous()
        ):
            raise ValueError(
                "stateful input_ids_local must be contiguous device-local int64"
            )
        if (
            not isinstance(plan.output_buffer, torch.Tensor)
            or tuple(plan.output_buffer.shape) != expected_shape
            or plan.output_buffer.dtype != torch.bfloat16
            or plan.output_buffer.device != plan.cursor.device
            or not plan.output_buffer.is_contiguous()
        ):
            raise ValueError("stateful output buffer metadata differs from setup")
        if (
            not isinstance(plan.input_residual_buffer, torch.Tensor)
            or tuple(plan.input_residual_buffer.shape) != expected_shape
            or plan.input_residual_buffer.dtype != torch.bfloat16
            or plan.input_residual_buffer.device != plan.cursor.device
            or not plan.input_residual_buffer.is_contiguous()
        ):
            raise ValueError(
                "stateful residual input buffer metadata differs from setup"
            )
        if (
            not isinstance(plan.input_ids_buffer, torch.Tensor)
            or tuple(plan.input_ids_buffer.shape) != expected_shape[:2]
            or plan.input_ids_buffer.dtype != torch.int64
            or plan.input_ids_buffer.device != plan.cursor.device
            or not plan.input_ids_buffer.is_contiguous()
        ):
            raise ValueError("stateful ID input buffer metadata differs from setup")
        for name, tensor in (
            ("expected_position", plan.expected_position),
            ("stop_position_tensor", plan.stop_position_tensor),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != (1,)
                or tensor.dtype != torch.int64
                or tensor.device != plan.cursor.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"stateful {name} metadata differs from setup")
        current_state_positions = self._stateful_state_position_tensors(self.states)
        if len(plan.state_position_tensors) != len(current_state_positions) or any(
            planned is not current
            for planned, current in zip(
                plan.state_position_tensors, current_state_positions, strict=True
            )
        ):
            raise ValueError("stateful next-position tensor ownership differs")

        graph_moe_slots = self._validate_graph_moe_slots(plan.graph_moe_slots)
        current_slot_buffer_ids = self._stateful_slot_buffer_ids(
            batch_size=plan.batch_size,
            graph_moe_slots=graph_moe_slots,
        )
        if current_slot_buffer_ids != plan.slot_buffer_ids:
            raise ValueError("stateful MoE slot buffer ownership differs from setup")
        current_pointers = self._stateful_tensor_pointers(
            layer_ids=self.layer_ids,
            cursor=plan.cursor,
            expected_position=plan.expected_position,
            stop_position_tensor=plan.stop_position_tensor,
            input_residual_buffer=plan.input_residual_buffer,
            input_ids_buffer=plan.input_ids_buffer,
            output_buffer=plan.output_buffer,
            states=self.states,
            layer_plans=plan.layer_plans,
        )
        if current_pointers != plan.tensor_pointers:
            raise ValueError("stateful super-stage tensor storage differs from setup")
        external_pointers = (
            int(residual.untyped_storage().data_ptr()),
            int(input_ids_local.untyped_storage().data_ptr()),
        )
        if (
            external_pointers[0] == external_pointers[1]
            or any(pointer in current_pointers for pointer in external_pointers)
        ):
            raise ValueError(
                "stateful inputs must not alias cursor, output, or plan workspaces"
            )

        ratio4_boundary, ratio128_boundary = family_boundary_flags(graph_family)
        attention_hidden = residual[:, :, 0, :]
        for layer_id, block, attention, state, layer_plan in zip(
            self.layer_ids,
            self.blocks,
            self.attentions,
            self.states,
            plan.layer_plans,
            strict=True,
        ):
            expected_type = self._stateful_plan_type(block)
            if not isinstance(layer_plan, expected_type):
                raise TypeError(
                    f"layer {layer_id} stateful plan has the wrong type"
                )
            if (
                layer_plan.owner_id != id(attention)
                or layer_plan.state_id != id(state)
                or layer_plan.start_position != plan.start_position
                or layer_plan.stop_position != plan.stop_position
                or layer_plan.batch_size != plan.batch_size
                or layer_plan.hidden_size != plan.hidden_size
                or layer_plan.position is not plan.cursor.device_position
            ):
                raise ValueError(
                    f"layer {layer_id} stateful plan ownership/range/shape differs"
                )
            validator = getattr(attention, "_validate_stateful_decode_plan", None)
            if not callable(validator):
                raise TypeError(
                    f"layer {layer_id} attention lacks stateful plan validation"
                )
            if block.compression_ratio == 0:
                # Window layers have no boundary variant (model.py:530).
                validator(attention_hidden, layer_plan)
            elif block.compression_ratio == 4:
                validator(
                    attention_hidden,
                    layer_plan,
                    ratio4_boundary=ratio4_boundary,
                )
            else:
                validator(
                    attention_hidden,
                    layer_plan,
                    ratio128_boundary=ratio128_boundary,
                )

        return self._validate_stateful_slot_free(
            batch_size=plan.batch_size,
            graph_family=graph_family,
            graph_moe_slots=graph_moe_slots,
        )

    def _validate_stateful_runtime_sync(
        self,
        plan: TP4StatefulDecodeSuperStagePlan,
    ) -> None:
        """Synchronously reject runtime drift before any block can mutate state."""

        host_position = plan.cursor.host_position
        try:
            dispatch_error = int(plan.cursor.dispatch_error.item())
            device_position = int(plan.cursor.device_position.item())
            expected_position = int(plan.expected_position.item())
            stop_position = int(plan.stop_position_tensor.item())
            states_match = tuple(
                bool(torch.all(position.eq(host_position)).item())
                for position in plan.state_position_tensors
            )
        except BaseException:
            self._poisoned = True
            raise
        failures = []
        if dispatch_error:
            failures.append(f"sticky dispatch error {dispatch_error}")
        if device_position != host_position:
            failures.append(
                f"device cursor {device_position} != host cursor {host_position}"
            )
        if expected_position != host_position:
            failures.append(
                f"device expected position {expected_position} != host cursor "
                f"{host_position}"
            )
        if stop_position != plan.stop_position:
            failures.append(
                f"device stop {stop_position} != plan stop {plan.stop_position}"
            )
        mismatched_layers = tuple(
            layer_id
            for layer_id, matches in zip(
                self.layer_ids, states_match, strict=True
            )
            if not matches
        )
        if mismatched_layers:
            failures.append(
                f"state next_position mismatch on layers {mismatched_layers}"
            )
        if failures:
            self._poisoned = True
            raise RuntimeError(
                "stateful safe-entry runtime drift: " + "; ".join(failures)
                + "; graph state must be discarded or restored"
            )

    def validate_stateful_decode_call(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        plan: TP4StatefulDecodeSuperStagePlan,
        graph_family: DecodeGraphFamily,
    ) -> None:
        """Validate host dispatch and all graph inputs before KV mutation."""

        self._validate_stateful_forward(
            residual,
            input_ids_local=input_ids_local,
            plan=plan,
            graph_family=graph_family,
        )
        self._validate_stateful_runtime_sync(plan)

    def _forward_stateful_fused_chain(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        plan: TP4StatefulDecodeSuperStagePlan,
        graph_family: DecodeGraphFamily,
        moe_slot: int,
        stage_marker: Callable[[int | None, str], None] | None = None,
    ) -> torch.Tensor:
        """Boundary-fused stage chain (E0hf, C2g/A5F lineage).

        Every HC boundary that has a fusion partner runs through the injected
        backend's ``post_pre_norm``: the intra-layer boundary (attention
        ``hc_post`` + FFN ``hc_pre`` + ``ffn_norm``) and the inter-layer
        boundary (FFN ``hc_post`` + next layer's attention ``hc_pre`` +
        ``attn_norm``).  The stage-first attention-side ``hc_pre`` and the
        stage-last tail ``hc_post`` have no partner and stay eager.  With the
        ``EagerHCBoundaryBackend`` this chain is bitwise identical to the
        default per-block loop (same ops, same order, same dtypes).
        """

        backend = self.hc_boundary_backend
        boundary_flags = family_boundary_flags(graph_family)
        blocks = self.blocks
        last_index = len(blocks) - 1
        current_residual = residual
        attention_hidden, post, comb = blocks[0].prepare_attention(residual)
        output: torch.Tensor | None = None
        for index, (layer_id, block, layer_plan) in enumerate(
            zip(self.layer_ids, blocks, plan.layer_plans, strict=True)
        ):
            layer_marker = (
                None
                if stage_marker is None
                else (lambda name, layer_id=layer_id: stage_marker(layer_id, name))
            )
            if layer_marker is not None:
                layer_marker("block_start")
            branch_output = block.run_stateful_attention(
                attention_hidden,
                attention_plan=layer_plan,
                boundary_flags=boundary_flags,
                stage_marker=layer_marker,
            )
            if layer_marker is not None:
                layer_marker("attention_done")
            after_attention, ffn_hidden, ffn_post, ffn_comb = block.ffn_boundary(
                branch_output,
                current_residual,
                post,
                comb,
                backend=backend,
            )
            if layer_marker is not None:
                layer_marker("ffn_prepare_done")
            moe_arguments: dict[str, Any] = {"slot": moe_slot}
            if block.route_kind == "hash":
                moe_arguments["input_ids_local"] = input_ids_local
            if layer_marker is not None:
                moe_arguments["stage_marker"] = layer_marker
            moe_output = block.moe.forward_tensor(ffn_hidden, **moe_arguments)
            if index == last_index:
                # Stage-tail hc_post has no fusion partner: stays eager.
                output = hc_post(moe_output, after_attention, ffn_post, ffn_comb)
            else:
                current_residual, attention_hidden, post, comb = (
                    blocks[index + 1].attention_boundary(
                        moe_output,
                        after_attention,
                        ffn_post,
                        ffn_comb,
                        backend=backend,
                    )
                )
            if layer_marker is not None:
                layer_marker("block_done")
        if output is None:
            raise AssertionError("fused stateful chain produced no output")
        return output

    def forward_stateful_decode_tensor_prevalidated(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        plan: TP4StatefulDecodeSuperStagePlan,
        graph_family: DecodeGraphFamily,
        stage_marker: Callable[[int | None, str], None] | None = None,
    ) -> torch.Tensor:
        """Run one fixed-address graph body under a detect-and-discard guard.

        A non-zero device guard does not branch around captured work. The output
        and every attention state are invalid. Every TP rank must discard the
        output and restore cursor/error/expected-position plus all stage state
        in place from the pre-replay snapshot, then revalidate the plan
        pointer manifest. The host cursor may advance only after an all-rank
        zero-error acceptance.
        """

        try:
            if stage_marker is not None:
                stage_marker(None, "graph_start")
            if residual is not plan.input_residual_buffer:
                raise ValueError(
                    "prevalidated residual must be the stable plan input buffer"
                )
            if input_ids_local is not plan.input_ids_buffer:
                raise ValueError(
                    "prevalidated input IDs must be the stable plan input buffer"
                )
            plan.cursor.guard_device_preflight(
                graph_family,
                expected_position=plan.expected_position,
                stop_position=plan.stop_position_tensor,
                stop_position_constant=plan.stop_position,
                state_positions=plan.state_position_tensors,
            )
            if stage_marker is not None:
                stage_marker(None, "guard_done")
            family_index = _STATEFUL_GRAPH_FAMILIES.index(graph_family)
            moe_slot = plan.graph_moe_slots[family_index]
            if self.hc_boundary_backend is not None:
                output = self._forward_stateful_fused_chain(
                    residual,
                    input_ids_local=input_ids_local,
                    plan=plan,
                    graph_family=graph_family,
                    moe_slot=moe_slot,
                    stage_marker=stage_marker,
                )
            else:
                output = residual
                for layer_id, block, layer_plan in zip(
                    self.layer_ids,
                    self.blocks,
                    plan.layer_plans,
                    strict=True,
                ):
                    arguments = {
                        "input_ids_local": (
                            input_ids_local if block.route_kind == "hash" else None
                        ),
                        "attention_plan": layer_plan,
                        "graph_family": graph_family,
                        "moe_slot": moe_slot,
                    }
                    if stage_marker is not None:
                        arguments["stage_marker"] = (
                            lambda name, layer_id=layer_id: stage_marker(
                                layer_id, name
                            )
                        )
                    output = block.forward_stateful_decode_tensor(
                        output, **arguments
                    )
            plan.output_buffer.copy_(output)
            if stage_marker is not None:
                stage_marker(None, "output_copy_done")
            plan.cursor.advance_device(
                graph_family,
                expected_position=plan.expected_position,
                stop_position=plan.stop_position_tensor,
                stop_position_constant=plan.stop_position,
                state_positions_after=plan.state_position_tensors,
            )
            if stage_marker is not None:
                stage_marker(None, "graph_done")
        except BaseException:
            self._poisoned = True
            raise
        return plan.output_buffer

    def forward_stateful_decode_tensor(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        plan: TP4StatefulDecodeSuperStagePlan,
        graph_family: DecodeGraphFamily,
    ) -> torch.Tensor:
        """Run one diagnostic eager step after external all-rank consensus.

        Validation here is rank-local and happens before TP collectives. A TP4
        controller must first converge ``validate_stateful_decode_call`` across
        every rank and ensure state cannot change before entering this helper.
        Formal graph runners must use the prevalidated body in a synchronized
        collective phase, then accept/restore state on every rank together.
        """

        self.validate_stateful_decode_call(
            residual,
            input_ids_local=input_ids_local,
            plan=plan,
            graph_family=graph_family,
        )
        try:
            plan.input_residual_buffer.copy_(residual)
            plan.input_ids_buffer.copy_(input_ids_local)
        except BaseException:
            self._poisoned = True
            raise
        output = self.forward_stateful_decode_tensor_prevalidated(
            plan.input_residual_buffer,
            input_ids_local=plan.input_ids_buffer,
            plan=plan,
            graph_family=graph_family,
        )
        try:
            dispatch_error = int(plan.cursor.dispatch_error.item())
        except BaseException:
            self._poisoned = True
            raise
        if dispatch_error:
            self._poisoned = True
            raise RuntimeError(
                f"stateful device dispatch guard failed with error {dispatch_error}; "
                "worker restart is required"
            )
        try:
            plan.cursor.advance_host(graph_family)
        except BaseException:
            self._poisoned = True
            raise
        return output

    def forward(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor,
        start_pos: int,
        plan: TP4DecodeSuperStagePlan,
        moe_slot: int = 0,
    ) -> torch.Tensor:
        return self.forward_decode_tensor(
            residual,
            input_ids_local=input_ids_local,
            start_pos=start_pos,
            plan=plan,
            moe_slot=moe_slot,
        )

    __call__ = forward


TP4DecodeStagePlan = TP4DecodeSuperStagePlan
TP4StatefulDecodeStagePlan = TP4StatefulDecodeSuperStagePlan


class TP4DecodeSuperStage(TP4DecodeStage):
    """Adapter that requires the canonical Flash L0-L5 slice."""

    def __init__(
        self,
        blocks: Sequence[DirectDecodeBlock],
        *,
        hc_boundary_backend: Any | None = None,
    ) -> None:
        super().__init__(
            blocks,
            expected_layer_ids=SUPERSTAGE_LAYER_IDS,
            hc_boundary_backend=hc_boundary_backend,
        )


__all__ = [
    "SUPERSTAGE_LAYER_IDS",
    "TP4DecodeStage",
    "TP4DecodeStagePlan",
    "TP4DecodeSuperStage",
    "TP4DecodeSuperStagePlan",
    "TP4StatefulDecodeStagePlan",
    "TP4StatefulDecodeSuperStagePlan",
]
