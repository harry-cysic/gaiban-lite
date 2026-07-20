"""CUDA-graph lifecycle primitives for direct stateful decode stages.

Ported from gaiban ``dsv4_direct/stateful_graph.py`` unchanged except for one
surface reduction: the ``HostSubmissionProbe`` replay instrumentation came
from ``pipeline_overlap`` (a PP diagnostic module outside this vertical), so
``replay_stateful_graph`` here is the bare fixed-address launch.  Everything
is geometry-free; the three-family registry contract follows
``stateful_decode.DecodeGraphFamily`` (see that module for the Flash
derivation of the family set).
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import torch

from .stateful_decode import DecodeGraphFamily


STATEFUL_GRAPH_FAMILIES = tuple(DecodeGraphFamily)
TEARDOWN_SCHEMA = "dsv4-stateful-graph-teardown-v1"
_BINDING_ATTRIBUTE = "_dsv4_stateful_graph_binding"


class StatefulGraphContractError(ValueError):
    """Raised before CUDA work when graph ownership is not trustworthy."""


def _error_record(label: str, error: BaseException) -> dict[str, str]:
    return {
        "phase": label,
        "type": type(error).__name__,
        "message": str(error),
    }


@dataclass(frozen=True)
class _StagePlanContract:
    device: torch.device
    global_rows: int
    layer_ids: tuple[int, ...]
    graph_slot_layer_ids: tuple[int, ...]
    graph_slot_moes: tuple[object, ...]
    slots: tuple[int, int, int]


@dataclass(frozen=True)
class _CapturedGraphBinding:
    plan: object
    graph_family: DecodeGraphFamily
    output_buffer: torch.Tensor
    stage_owner_id: int


def _positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise StatefulGraphContractError(f"{label} must be a positive integer")
    return value


def _tensor(value: object, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise StatefulGraphContractError(f"{label} must be a tensor")
    if not value.is_contiguous():
        raise StatefulGraphContractError(f"{label} must be contiguous")
    return value


def _stage_plan_contract(stage: object, plan: object) -> _StagePlanContract:
    if getattr(plan, "owner_id", None) != id(stage):
        raise StatefulGraphContractError(
            "stateful graph plan has the wrong stage owner"
        )
    forward = getattr(stage, "forward_stateful_decode_tensor_prevalidated", None)
    if not callable(forward):
        raise StatefulGraphContractError(
            "stateful graph stage lacks its prevalidated body"
        )

    stage_layer_ids = getattr(stage, "layer_ids", None)
    plan_layer_ids = getattr(plan, "layer_ids", None)
    if (
        not isinstance(stage_layer_ids, Sequence)
        or isinstance(stage_layer_ids, (str, bytes))
        or not isinstance(plan_layer_ids, tuple)
    ):
        raise StatefulGraphContractError("stateful graph layer ownership is invalid")
    layer_ids = tuple(stage_layer_ids)
    if (
        not layer_ids
        or layer_ids != plan_layer_ids
        or any(
            not isinstance(layer, int) or isinstance(layer, bool)
            for layer in layer_ids
        )
        or len(set(layer_ids)) != len(layer_ids)
    ):
        raise StatefulGraphContractError("stateful graph stage/plan layers differ")

    optional_slot_fields = (
        getattr(stage, "graph_slot_layer_ids", None),
        getattr(stage, "graph_slot_moes", None),
        getattr(plan, "graph_slot_layer_ids", None),
    )
    if all(value is None for value in optional_slot_fields):
        graph_slot_layer_ids = layer_ids
        moes_value = getattr(stage, "moes", None)
    elif any(value is None for value in optional_slot_fields):
        raise StatefulGraphContractError(
            "stateful graph slot ownership fields must be declared together"
        )
    else:
        stage_slot_layers, moes_value, plan_slot_layers = optional_slot_fields
        if (
            not isinstance(stage_slot_layers, Sequence)
            or isinstance(stage_slot_layers, (str, bytes))
            or not isinstance(plan_slot_layers, tuple)
        ):
            raise StatefulGraphContractError(
                "stateful graph slot layer ownership is invalid"
            )
        graph_slot_layer_ids = tuple(stage_slot_layers)
        if graph_slot_layer_ids != plan_slot_layers:
            raise StatefulGraphContractError(
                "stateful graph stage/plan slot layers differ"
            )
        if (
            any(
                not isinstance(layer, int) or isinstance(layer, bool)
                for layer in graph_slot_layer_ids
            )
            or len(set(graph_slot_layer_ids)) != len(graph_slot_layer_ids)
            or any(layer not in layer_ids for layer in graph_slot_layer_ids)
            or tuple(
                layer for layer in layer_ids if layer in graph_slot_layer_ids
            )
            != graph_slot_layer_ids
        ):
            raise StatefulGraphContractError(
                "stateful graph slot layers must be an ordered semantic subsequence"
            )
    if not isinstance(moes_value, Sequence) or isinstance(
        moes_value, (str, bytes)
    ):
        raise StatefulGraphContractError(
            "stateful graph stage MoE ownership is invalid"
        )
    graph_slot_moes = tuple(moes_value)
    if len(graph_slot_moes) != len(graph_slot_layer_ids) or len(
        {id(moe) for moe in graph_slot_moes}
    ) != len(graph_slot_moes):
        raise StatefulGraphContractError(
            "stateful graph stage MoEs must be layer-owned"
        )
    for layer_id, moe in zip(
        graph_slot_layer_ids, graph_slot_moes, strict=True
    ):
        if getattr(moe, "layer_id", None) != layer_id:
            raise StatefulGraphContractError(
                "stateful graph MoE layer identity differs from its slot owner"
            )
        if not callable(getattr(moe, "reset_free_slot_completion_event", None)):
            raise StatefulGraphContractError(
                "stateful graph MoE lacks slot event reset"
            )
        if not callable(getattr(moe, "slot_completion_status", None)):
            raise StatefulGraphContractError("stateful graph MoE lacks slot status")

    slots_value = getattr(plan, "graph_moe_slots", None)
    if (
        not isinstance(slots_value, tuple)
        or len(slots_value) != len(STATEFUL_GRAPH_FAMILIES)
        or any(
            not isinstance(slot, int) or isinstance(slot, bool) or slot < 0
            for slot in slots_value
        )
        or len(set(slots_value)) != len(slots_value)
    ):
        raise StatefulGraphContractError(
            "stateful graph plan requires three distinct non-negative MoE slots"
        )
    slots = (slots_value[0], slots_value[1], slots_value[2])

    batch_size = _positive_int(getattr(plan, "batch_size", None), label="batch_size")
    world_size = _positive_int(getattr(stage, "world_size", None), label="world_size")
    residual = _tensor(
        getattr(plan, "input_residual_buffer", None), label="input_residual_buffer"
    )
    input_ids = _tensor(
        getattr(plan, "input_ids_buffer", None), label="input_ids_buffer"
    )
    output = _tensor(getattr(plan, "output_buffer", None), label="output_buffer")
    if input_ids.device != residual.device or output.device != residual.device:
        raise StatefulGraphContractError(
            "stateful graph IO buffers must share one device"
        )
    pointers = tuple(
        (tensor.device, int(tensor.untyped_storage().data_ptr()))
        for tensor in (residual, input_ids, output)
    )
    if len(set(pointers)) != len(pointers):
        raise StatefulGraphContractError("stateful graph IO buffers must not alias")

    return _StagePlanContract(
        device=residual.device,
        global_rows=batch_size * world_size,
        layer_ids=layer_ids,
        graph_slot_layer_ids=graph_slot_layer_ids,
        graph_slot_moes=graph_slot_moes,
        slots=slots,
    )


def _require_family(graph_family: DecodeGraphFamily) -> DecodeGraphFamily:
    if not isinstance(graph_family, DecodeGraphFamily):
        raise TypeError("graph_family must be a DecodeGraphFamily")
    return graph_family


def _bind_captured_graph(
    graph: object,
    plan: object,
    graph_family: DecodeGraphFamily,
) -> None:
    if hasattr(graph, _BINDING_ATTRIBUTE):
        raise StatefulGraphContractError("stateful CUDA graph is already bound")
    output = _tensor(getattr(plan, "output_buffer", None), label="output_buffer")
    owner_id = getattr(plan, "owner_id", None)
    if not isinstance(owner_id, int) or isinstance(owner_id, bool):
        raise StatefulGraphContractError("stateful graph plan owner is invalid")
    binding = _CapturedGraphBinding(
        plan=plan,
        graph_family=graph_family,
        output_buffer=output,
        stage_owner_id=owner_id,
    )
    setattr(graph, _BINDING_ATTRIBUTE, binding)
    if getattr(graph, _BINDING_ATTRIBUTE, None) is not binding:
        raise StatefulGraphContractError("stateful CUDA graph binding was not retained")


def _graph_binding_matches(
    graph: object,
    plan: object,
    graph_family: DecodeGraphFamily,
) -> bool:
    binding = getattr(graph, _BINDING_ATTRIBUTE, None)
    return bool(
        isinstance(binding, _CapturedGraphBinding)
        and binding.plan is plan
        and binding.graph_family is graph_family
        and binding.output_buffer is getattr(plan, "output_buffer", None)
        and binding.stage_owner_id == getattr(plan, "owner_id", None)
    )


def _pool_handle_token(value: object) -> tuple[int, int] | None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(not isinstance(part, int) or isinstance(part, bool) for part in value)
    ):
        return None
    return value


def _validate_family_slot_clean(
    contract: _StagePlanContract, graph_family: DecodeGraphFamily
) -> int:
    slot = contract.slots[STATEFUL_GRAPH_FAMILIES.index(graph_family)]
    for layer_id, moe in zip(
        contract.graph_slot_layer_ids, contract.graph_slot_moes, strict=True
    ):
        status = moe.slot_completion_status(contract.global_rows, slot)
        if not isinstance(status, Mapping) or dict(status) != {
            "state": "free",
            "has_completion_event": False,
        }:
            raise StatefulGraphContractError(
                f"stateful graph family {graph_family.value} layer {layer_id} "
                f"slot {slot} is not clean"
            )
    return slot


def _cleanup_failed_capture(
    graph: object,
    contract: _StagePlanContract,
    graph_family: DecodeGraphFamily,
    *,
    already_synchronized: bool,
) -> dict[str, object]:
    """Destroy an unreturned graph before replacing its family-owned events."""

    errors: list[dict[str, str]] = []
    synchronization = {
        "before_reset": already_synchronized,
        "after_reset": False,
    }
    if not already_synchronized:
        try:
            torch.cuda.synchronize(contract.device)
            synchronization["before_reset"] = True
        except BaseException as error:
            errors.append(_error_record("synchronize_before_capture_reset", error))

    graph_reset = False
    if synchronization["before_reset"]:
        try:
            graph.reset()
            graph_reset = True
        except BaseException as error:
            errors.append(_error_record("reset_failed_capture_graph", error))
        try:
            torch.cuda.synchronize(contract.device)
            synchronization["after_reset"] = True
        except BaseException as error:
            errors.append(_error_record("synchronize_after_capture_reset", error))

    slot = contract.slots[STATEFUL_GRAPH_FAMILIES.index(graph_family)]
    layers: dict[str, object] = {}
    slot_cleanup_safe = bool(graph_reset and synchronization["after_reset"])
    for layer_id, moe in zip(
        contract.graph_slot_layer_ids, contract.graph_slot_moes, strict=True
    ):
        reset_ok = False
        status: object = None
        if slot_cleanup_safe:
            try:
                moe.reset_free_slot_completion_event(contract.global_rows, slot)
                reset_ok = True
                status = moe.slot_completion_status(contract.global_rows, slot)
            except BaseException as error:
                errors.append(
                    _error_record(
                        f"cleanup_failed_capture_layer_{layer_id}_slot_{slot}",
                        error,
                    )
                )
        status_exact = bool(
            isinstance(status, Mapping)
            and dict(status)
            == {"state": "free", "has_completion_event": False}
        )
        layers[str(layer_id)] = {
            "reset": reset_ok,
            "status": (
                {
                    "state": status.get("state"),
                    "has_completion_event": status.get("has_completion_event"),
                }
                if isinstance(status, Mapping)
                else None
            ),
            "accepted": bool(reset_ok and status_exact),
        }

    binding_released = False
    try:
        if hasattr(graph, _BINDING_ATTRIBUTE):
            delattr(graph, _BINDING_ATTRIBUTE)
        binding_released = not hasattr(graph, _BINDING_ATTRIBUTE)
    except BaseException as error:
        errors.append(_error_record("release_failed_capture_binding", error))

    accepted = bool(
        all(synchronization.values())
        and graph_reset
        and len(layers) == len(contract.graph_slot_layer_ids)
        and all(bool(record["accepted"]) for record in layers.values())
        and binding_released
        and not errors
    )
    return {
        "family": graph_family.value,
        "global_rows": contract.global_rows,
        "slot": slot,
        "synchronization": synchronization,
        "graph_reset": graph_reset,
        "layers": layers,
        "binding_released": binding_released,
        "errors": errors,
        "accepted": accepted,
    }


def capture_stateful_graph(
    stage: object,
    plan: object,
    *,
    graph_family: DecodeGraphFamily,
    capture_stream: object,
    pool: object,
    stage_marker: Callable[[int | None, str], None] | None = None,
) -> torch.cuda.CUDAGraph:
    """Capture one family using caller-owned stream and graph pool resources."""

    contract = _stage_plan_contract(stage, plan)
    graph_family = _require_family(graph_family)
    if stage_marker is not None and not callable(stage_marker):
        raise TypeError("stage_marker must be callable")
    if not callable(getattr(capture_stream, "wait_stream", None)):
        raise StatefulGraphContractError("capture_stream must expose wait_stream")
    capture_device = getattr(capture_stream, "device", None)
    if (
        not isinstance(capture_device, torch.device)
        or capture_device != contract.device
    ):
        raise StatefulGraphContractError(
            "capture stream and stateful graph plan must share one device"
        )
    if _pool_handle_token(pool) is None:
        raise StatefulGraphContractError(
            "graph pool handle must be an opaque two-integer token"
        )
    _validate_family_slot_clean(contract, graph_family)

    current_stream = torch.cuda.current_stream(contract.device)
    if current_stream == capture_stream:
        raise StatefulGraphContractError(
            "capture stream must differ from current stream"
        )
    if not callable(getattr(current_stream, "wait_stream", None)):
        raise RuntimeError("current CUDA stream lacks wait_stream")

    graph = torch.cuda.CUDAGraph()
    if not callable(getattr(graph, "replay", None)) or not callable(
        getattr(graph, "reset", None)
    ):
        raise RuntimeError("CUDA graph object lacks replay/reset lifecycle methods")

    capture_body_started = False
    synchronized = False
    try:
        capture_stream.wait_stream(current_stream)
        with torch.cuda.graph(graph, stream=capture_stream, pool=pool):
            capture_body_started = True
            arguments = {
                "input_ids_local": plan.input_ids_buffer,
                "plan": plan,
                "graph_family": graph_family,
            }
            if stage_marker is not None:
                arguments["stage_marker"] = stage_marker
            output = stage.forward_stateful_decode_tensor_prevalidated(
                plan.input_residual_buffer, **arguments
            )
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(contract.device)
        synchronized = True
        if output is not plan.output_buffer:
            raise RuntimeError(
                "captured stateful stage did not return its plan output buffer"
            )
        _bind_captured_graph(graph, plan, graph_family)
    except BaseException as error:
        if not capture_body_started:
            raise
        cleanup = _cleanup_failed_capture(
            graph,
            contract,
            graph_family,
            already_synchronized=synchronized,
        )
        if not cleanup["accepted"]:
            failure = RuntimeError(
                "failed CUDA graph capture could not be cleaned; "
                "worker restart is required"
            )
            failure.cleanup_evidence = cleanup  # type: ignore[attr-defined]
            raise failure from error
        raise
    return graph


def replay_stateful_graph(
    graph: object,
    plan: object,
    *,
    graph_family: DecodeGraphFamily,
) -> torch.Tensor:
    """Launch one captured body without adding synchronization to the hot path."""

    replay = getattr(graph, "replay", None)
    if not callable(replay):
        raise StatefulGraphContractError("stateful CUDA graph lacks replay")
    graph_family = _require_family(graph_family)
    output = _tensor(getattr(plan, "output_buffer", None), label="output_buffer")
    if not _graph_binding_matches(graph, plan, graph_family):
        raise StatefulGraphContractError(
            "stateful CUDA graph family/plan binding differs"
        )
    replay()
    return output


def _key_name(key: object) -> str:
    return key.value if isinstance(key, DecodeGraphFamily) else repr(key)


def _registry_evidence(
    registry: MutableMapping[object, object], *, kind: str
) -> dict[str, object]:
    keys = list(registry)
    values = list(registry.values())
    expected = set(STATEFUL_GRAPH_FAMILIES)
    families_exact = bool(
        len(keys) == len(STATEFUL_GRAPH_FAMILIES)
        and all(isinstance(key, DecodeGraphFamily) for key in keys)
        and set(keys) == expected
    )
    if kind == "graph":
        values_valid = all(value is not None for value in values)
        values_distinct = len({id(value) for value in values}) == len(values)
        reset_capable = all(
            callable(getattr(value, "reset", None)) for value in values
        )
    elif kind == "pool":
        tokens = [_pool_handle_token(value) for value in values]
        values_valid = all(token is not None for token in tokens)
        values_distinct = bool(
            values_valid and len(set(tokens)) == len(tokens)
        )
        reset_capable = True
    else:
        raise AssertionError(f"unknown graph registry kind {kind!r}")
    return {
        "observed_families": sorted(_key_name(key) for key in keys),
        "families_exact": families_exact,
        "values_present": all(value is not None for value in values),
        "values_valid": values_valid,
        "values_distinct": values_distinct,
        "reset_capable": reset_capable,
    }


def teardown_stateful_graphs(
    stage: object,
    plan: object,
    graphs: MutableMapping[DecodeGraphFamily, object],
    *,
    pool_handles: MutableMapping[DecodeGraphFamily, object],
) -> dict[str, object]:
    """Release a complete graph-family set and prove every owned slot is clean.

    Malformed registries are still cleared best-effort, but their evidence can
    never be accepted. CUDA/runtime failures are recorded while later safe
    cleanup steps continue where possible.

    Pool handles may legitimately be shared -- across families and across
    lanes -- when every capture/replay on the device is serialized (E0hf
    two-lane shared-pool precedent, extended by the 17th vertical): pool
    blocks then only hold capture-transient workspace, never live outputs.
    Pool-token distinctness is therefore recorded as evidence
    (``values_distinct``) but is not required for acceptance; graph objects
    themselves must still be distinct per family.
    """

    contract = _stage_plan_contract(stage, plan)
    if not isinstance(graphs, MutableMapping) or not isinstance(
        pool_handles, MutableMapping
    ):
        raise TypeError("graph and pool registries must be mutable mappings")

    graph_registry = _registry_evidence(graphs, kind="graph")
    pool_registry = _registry_evidence(pool_handles, kind="pool")
    graph_bindings = {
        family.value: bool(
            family in graphs
            and _graph_binding_matches(graphs[family], plan, family)
        )
        for family in STATEFUL_GRAPH_FAMILIES
    }
    graph_registry["bindings"] = graph_bindings
    graph_registry["bindings_exact"] = all(graph_bindings.values())
    registry_objects_distinct = graphs is not pool_handles
    graph_pool_objects_distinct = not (
        {id(value) for value in graphs.values()}
        & {id(value) for value in pool_handles.values()}
    )
    errors: list[dict[str, str]] = []
    synchronization = {
        "before_reset": False,
        "after_reset": False,
        "final": False,
    }
    reset_by_object: dict[int, bool] = {}
    binding_released_by_object: dict[int, bool] = {}
    graph_resets = {family.value: False for family in STATEFUL_GRAPH_FAMILIES}
    graph_bindings_released = {
        family.value: False for family in STATEFUL_GRAPH_FAMILIES
    }
    ordered_entries: list[tuple[object, object]] = []

    try:
        torch.cuda.synchronize(contract.device)
        synchronization["before_reset"] = True
    except BaseException as error:
        errors.append(_error_record("synchronize_before_reset", error))

    if synchronization["before_reset"]:
        ordered_entries.extend(
            (family, graphs[family])
            for family in STATEFUL_GRAPH_FAMILIES
            if family in graphs
        )
        ordered_entries.extend(
            (key, value)
            for key, value in graphs.items()
            if key not in STATEFUL_GRAPH_FAMILIES
        )
        for key, graph in ordered_entries:
            object_id = id(graph)
            if object_id not in reset_by_object:
                reset = getattr(graph, "reset", None)
                if not callable(reset):
                    reset_by_object[object_id] = False
                    errors.append(
                        {
                            "phase": f"reset_graph_{_key_name(key)}",
                            "type": "StatefulGraphContractError",
                            "message": "graph object lacks reset",
                        }
                    )
                else:
                    try:
                        reset()
                        reset_by_object[object_id] = True
                    except BaseException as error:
                        reset_by_object[object_id] = False
                        errors.append(
                            _error_record(f"reset_graph_{_key_name(key)}", error)
                        )
                if reset_by_object[object_id]:
                    try:
                        if hasattr(graph, _BINDING_ATTRIBUTE):
                            delattr(graph, _BINDING_ATTRIBUTE)
                        binding_released_by_object[object_id] = not hasattr(
                            graph, _BINDING_ATTRIBUTE
                        )
                    except BaseException as error:
                        binding_released_by_object[object_id] = False
                        errors.append(
                            _error_record(
                                f"release_graph_binding_{_key_name(key)}",
                                error,
                            )
                        )
            if isinstance(key, DecodeGraphFamily):
                graph_resets[key.value] = reset_by_object[object_id]
                graph_bindings_released[key.value] = (
                    binding_released_by_object.get(object_id, False)
                )

        try:
            torch.cuda.synchronize(contract.device)
            synchronization["after_reset"] = True
        except BaseException as error:
            errors.append(_error_record("synchronize_after_reset", error))

    ordered_entries.clear()
    graph = None
    reset = None

    slot_cleanup: dict[str, object] = {}
    reset_operations_safe = bool(
        synchronization["before_reset"]
        and synchronization["after_reset"]
        and graph_registry["families_exact"]
        and graph_registry["values_present"]
        and graph_registry["values_valid"]
        and graph_registry["values_distinct"]
        and graph_registry["reset_capable"]
        and graph_registry["bindings_exact"]
        and len(reset_by_object) == len(STATEFUL_GRAPH_FAMILIES)
        and all(reset_by_object.values())
    )
    for family, slot in zip(
        STATEFUL_GRAPH_FAMILIES, contract.slots, strict=True
    ):
        layers: dict[str, object] = {}
        for layer_id, moe in zip(
            contract.graph_slot_layer_ids,
            contract.graph_slot_moes,
            strict=True,
        ):
            reset_ok = False
            status_record: dict[str, object] = {
                "state": None,
                "has_completion_event": None,
            }
            if reset_operations_safe:
                try:
                    moe.reset_free_slot_completion_event(contract.global_rows, slot)
                    reset_ok = True
                    status = moe.slot_completion_status(contract.global_rows, slot)
                    if isinstance(status, Mapping):
                        status_record = {
                            "state": status.get("state"),
                            "has_completion_event": status.get(
                                "has_completion_event"
                            ),
                        }
                        status_exact = dict(status) == {
                            "state": "free",
                            "has_completion_event": False,
                        }
                    else:
                        status_exact = False
                except BaseException as error:
                    status_exact = False
                    errors.append(
                        _error_record(
                            f"cleanup_{family.value}_layer_{layer_id}_slot_{slot}",
                            error,
                        )
                    )
            else:
                status_exact = False
            accepted = bool(reset_ok and status_exact)
            layers[str(layer_id)] = {
                "reset": reset_ok,
                "status": status_record,
                "accepted": accepted,
            }
        slot_cleanup[family.value] = {
            "slot": slot,
            "layers": layers,
            "accepted": bool(
                len(layers) == len(contract.graph_slot_layer_ids)
                and all(bool(record["accepted"]) for record in layers.values())
            ),
        }

    try:
        graphs.clear()
    except BaseException as error:
        errors.append(_error_record("clear_graph_registry", error))
    try:
        pool_handles.clear()
    except BaseException as error:
        errors.append(_error_record("clear_pool_registry", error))
    graph_registry_cleared = not graphs
    pool_registry_cleared = not pool_handles

    gc_collected = False
    gc_unreachable: int | None = None
    try:
        collected = gc.collect()
        gc_collected = True
        if isinstance(collected, int) and not isinstance(collected, bool):
            gc_unreachable = collected
    except BaseException as error:
        errors.append(_error_record("gc_collect", error))

    empty_cache_called = False
    try:
        torch.cuda.empty_cache()
        empty_cache_called = True
    except BaseException as error:
        errors.append(_error_record("empty_cache", error))
    try:
        torch.cuda.synchronize(contract.device)
        synchronization["final"] = True
    except BaseException as error:
        errors.append(_error_record("synchronize_final", error))

    graph_registry["reset"] = graph_resets
    graph_registry["bindings_released"] = graph_bindings_released
    pool_registry["registry_cleared"] = pool_registry_cleared
    graph_registry["registry_cleared"] = graph_registry_cleared
    slot_cleanup_accepted = bool(
        len(slot_cleanup) == len(STATEFUL_GRAPH_FAMILIES)
        and all(bool(record["accepted"]) for record in slot_cleanup.values())
    )
    accepted = bool(
        graph_registry["families_exact"]
        and graph_registry["values_present"]
        and graph_registry["values_valid"]
        and graph_registry["values_distinct"]
        and graph_registry["reset_capable"]
        and graph_registry["bindings_exact"]
        and all(graph_resets.values())
        and all(graph_bindings_released.values())
        and pool_registry["families_exact"]
        and pool_registry["values_present"]
        and pool_registry["values_valid"]
        and registry_objects_distinct
        and graph_pool_objects_distinct
        and slot_cleanup_accepted
        and all(synchronization.values())
        and graph_registry_cleared
        and pool_registry_cleared
        and gc_collected
        and empty_cache_called
        and not errors
    )
    return {
        "schema": TEARDOWN_SCHEMA,
        "families": [family.value for family in STATEFUL_GRAPH_FAMILIES],
        "global_rows": contract.global_rows,
        "graph_moe_slots": list(contract.slots),
        "semantic_layer_ids": list(contract.layer_ids),
        "graph_slot_layer_ids": list(contract.graph_slot_layer_ids),
        "graph_registry": graph_registry,
        "pool_registry": pool_registry,
        "registry_objects_distinct": registry_objects_distinct,
        "graph_pool_objects_distinct": graph_pool_objects_distinct,
        "synchronization": synchronization,
        "slot_cleanup": slot_cleanup,
        "slot_cleanup_accepted": slot_cleanup_accepted,
        "gc_collected": gc_collected,
        "gc_unreachable": gc_unreachable,
        "empty_cache_called": empty_cache_called,
        "errors": errors,
        "accepted": accepted,
    }


__all__ = [
    "STATEFUL_GRAPH_FAMILIES",
    "TEARDOWN_SCHEMA",
    "StatefulGraphContractError",
    "capture_stateful_graph",
    "replay_stateful_graph",
    "teardown_stateful_graphs",
]
