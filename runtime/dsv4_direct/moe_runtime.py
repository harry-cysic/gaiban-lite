"""Reusable TP4 MoE component for frozen physical-layer runtimes.

Ported from gaiban unchanged except the frozen geometry contract in
``TP4MoEConfig``: DeepSeek-V4-Flash uses hidden 4096, moe_intermediate 2048,
256 routed experts, and routed_scaling_factor 1.5 (Pro: 7168/3072/384/2.5).
topk 6, swiglu clamp 10.0, and TP world 4 are unchanged.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .block_weights import ResidentGateWeights
from .deterministic_moe_align import (
    DeterministicMoEAlignment,
    allocate_deterministic_moe_alignment,
    deterministic_moe_align_block_size,
)
from .moe_forward import (
    dequant_fp8_block,
    gate_forward_with_boundary,
    hash_gate_forward,
)
from .model_contract import MODEL_LAYER_COUNT, SUPPORTED_LAYER_SPECS
from .ops.marlin_moe import ResidentMoEWeights, SharedExpertSlice, tensor_bytes


@dataclass(frozen=True)
class TP4MoEConfig:
    hidden_size: int = 4096
    intermediate_size: int = 2048
    experts: int = 256
    topk: int = 6
    route_scale: float = 1.5
    clamp_limit: float = 10.0
    world_size: int = 4

    @property
    def local_intermediate(self) -> int:
        return self.intermediate_size // self.world_size

    def validate(self) -> None:
        expected = {
            "hidden_size": (self.hidden_size, 4096),
            "intermediate_size": (self.intermediate_size, 2048),
            "experts": (self.experts, 256),
            "topk": (self.topk, 6),
            "world_size": (self.world_size, 4),
        }
        mismatches = {
            name: {"observed": observed, "expected": wanted}
            for name, (observed, wanted) in expected.items()
            if observed != wanted
        }
        if mismatches:
            raise ValueError(f"unsupported TP4 MoE config: {mismatches}")
        if self.intermediate_size % self.world_size:
            raise ValueError("intermediate size must divide TP world size")
        if self.route_scale != 1.5 or self.clamp_limit != 10.0:
            raise ValueError("MoE route scale/clamp contract must be 1.5/10.0")


@dataclass
class PreparedSharedBF16:
    w1: torch.Tensor
    w3: torch.Tensor
    w2: torch.Tensor

    @property
    def resident_bytes(self) -> int:
        return tensor_bytes(self.w1, self.w3, self.w2)


@dataclass(frozen=True)
class MoETrace:
    local_input_shape: tuple[int, ...]
    gathered_shape: tuple[int, ...]
    partial_shape: tuple[int, ...]
    local_output_shape: tuple[int, ...]
    route_ids_row_zero: tuple[int, ...]
    route_weights_row_zero: tuple[float, ...]
    route_margin_min: float | None
    route_digest: str
    stage_digests: dict[str, str]
    buffer_slot: int
    shared_path: str = "bf16_dequant_correctness_fallback"
    route_source: str = "native"


@dataclass(frozen=True)
class MoERouteOverride:
    """Explicit full-global learned route for diagnostic counterfactuals."""

    ids: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class MoERouteTensors:
    weights: torch.Tensor
    ids: torch.Tensor
    margin: torch.Tensor | None
    selection_ids: torch.Tensor
    selection_scores: torch.Tensor
    local_input: torch.Tensor | None = None
    gate_logits: torch.Tensor | None = None
    unbiased_scores: torch.Tensor | None = None
    biased_scores: torch.Tensor | None = None
    native_weights: torch.Tensor | None = None
    native_ids: torch.Tensor | None = None
    route_source: str = "native"


@dataclass(frozen=True)
class MoERouteCaptureBuffer:
    """Stable device storage populated by a captured MoE route graph."""

    global_rows: int
    slot: int
    owner_id: int
    ids: torch.Tensor
    weights: torch.Tensor
    generation: torch.Tensor
    tensor_pointers: tuple[int, int, int]

    @property
    def resident_bytes(self) -> int:
        return tensor_bytes(self.ids, self.weights, self.generation)


@dataclass(frozen=True)
class MoERouteCaptureEvidence:
    """Owning host snapshot of one registered route-capture buffer."""

    global_rows: int
    slot: int
    ids: torch.Tensor
    weights: torch.Tensor
    generation: int
    device_tensor_pointers: tuple[int, int, int]
    route_digest: str


@dataclass(frozen=True)
class _MoERouteCaptureRegistration:
    """Runtime-private identity manifest independent of public buffer metadata."""

    capture: MoERouteCaptureBuffer
    capture_object_id: int
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    tensor_object_ids: tuple[int, int, int]
    tensor_pointers: tuple[int, int, int]


@dataclass
class _MarlinBuffers:
    workspace: torch.Tensor
    cache13: torch.Tensor
    cache2: torch.Tensor
    output: torch.Tensor
    gathered: torch.Tensor
    combined: torch.Tensor
    reduced: torch.Tensor
    alignment: DeterministicMoEAlignment
    block_size_m: int
    ready_event: torch.cuda.Event
    gathered_input_ids: torch.Tensor | None = None
    has_completion_event: bool = False
    state: str = "free"


@dataclass(frozen=True)
class _TensorStorageManifest:
    tensor_object_id: int
    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    data_pointer: int
    storage_pointer: int
    storage_nbytes: int


def _tensor_storage_manifest(tensor: torch.Tensor) -> _TensorStorageManifest:
    storage = tensor.untyped_storage()
    return _TensorStorageManifest(
        tensor_object_id=id(tensor),
        device=tensor.device,
        dtype=tensor.dtype,
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        data_pointer=int(tensor.data_ptr()),
        storage_pointer=int(storage.data_ptr()),
        storage_nbytes=int(storage.nbytes()),
    )


_W4A8_QUANT_PRESET_DONE = False


def _preset_w4a8_activation_quant() -> None:
    """Install the per-token FP8 activation quant used by Marlin W4A8.

    ``marlin_quant_input`` resolves through ``QuantFP8`` which needs a vllm
    config context (A1.5 lesson, carried by A3F ``common.preset_w4a8_quant``);
    preset the module singleton with the same single-kernel per-token quant.
    Idempotent; only affects the FP8-input Marlin path (W4A16 never consults
    ``_quant_fp8_method``).
    """

    global _W4A8_QUANT_PRESET_DONE
    if _W4A8_QUANT_PRESET_DONE:
        return
    from vllm import _custom_ops as ops
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu

    def _per_token_fp8(x: torch.Tensor):
        try:
            return ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
        except TypeError:  # signature drift fallback (eager, slower)
            xf = x.float()
            s = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
            return (xf / s).clamp(-448, 448).to(torch.float8_e4m3fn), s

    mu._quant_fp8_method = _per_token_fp8
    _W4A8_QUANT_PRESET_DONE = True


def prepare_shared_bf16(shared: SharedExpertSlice) -> PreparedSharedBF16:
    return PreparedSharedBF16(
        w1=dequant_fp8_block(shared.w1, shared.s1).to(torch.bfloat16),
        w3=dequant_fp8_block(shared.w3, shared.s3).to(torch.bfloat16),
        w2=dequant_fp8_block(shared.w2, shared.s2).to(torch.bfloat16),
    )


def shared_bf16_partial(
    hidden: torch.Tensor,
    weights: PreparedSharedBF16,
    *,
    clamp_limit: float = 10.0,
) -> torch.Tensor:
    if hidden.ndim != 2 or hidden.shape[1] != weights.w1.shape[1]:
        raise ValueError("shared expert hidden/weight shape mismatch")
    if hidden.dtype != torch.bfloat16:
        raise TypeError("shared expert correctness fallback requires BF16 input")
    gate = F.linear(hidden, weights.w1).float().clamp(max=clamp_limit)
    up = (
        F.linear(hidden, weights.w3)
        .float()
        .clamp(min=-clamp_limit, max=clamp_limit)
    )
    activated = (F.silu(gate) * up).to(torch.bfloat16)
    return F.linear(activated, weights.w2)


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(f"{list(tensor.shape)}|{tensor.dtype}|".encode())
        digest.update(
            tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        )
    return digest.hexdigest()


def _marlin_block_size_m(
    *,
    rows: int,
    topk: int,
    experts: int,
    input_dtype: torch.dtype | None = None,
) -> int:
    """Match vLLM's WNA16 MoE block-size selection for this fixed ABI.

    Mirrors marlin_moe.py: the 1-byte activation paths (W4A8 INT8/FP8) have
    no kernel thread config below block 16, so vLLM raises the floor to 16
    (otherwise the kernel rejects small-M shapes with "Invalid thread
    config", e.g. decode MKN=[4,4096,1024]).
    """

    if rows <= 0 or topk <= 0 or experts <= 0:
        raise ValueError("Marlin rows, topk, and experts must be positive")
    selected = 64
    for candidate in (8, 16, 32, 48, 64):
        if rows * topk / experts / candidate < 0.9:
            selected = candidate
            break
    if input_dtype is not None and input_dtype.itemsize == 1:
        selected = max(selected, 16)
    return selected


def _validate_moe_identity(
    *,
    config: TP4MoEConfig,
    resident: ResidentMoEWeights,
    gate: ResidentGateWeights,
    rank: int,
    device: torch.device,
) -> tuple[int, str]:
    if resident.routed.w13_q.shape[0] != config.experts:
        raise ValueError("resident routed expert count mismatch")
    if tuple(gate.weight.shape) != (config.experts, config.hidden_size):
        raise ValueError("gate weight shape mismatch")
    if gate.weight.device != device:
        raise ValueError("gate weight must use the configured device")

    layer_id = resident.layer_id
    layer_spec = (
        SUPPORTED_LAYER_SPECS.get(layer_id)
        if isinstance(layer_id, int) and not isinstance(layer_id, bool)
        else None
    )
    if layer_spec is None:
        raise ValueError(
            f"TP4MoE resident layer must be in [0, {MODEL_LAYER_COUNT}), "
            f"got {layer_id!r}"
        )
    expected_start = rank * config.local_intermediate
    expected_end = expected_start + config.local_intermediate
    resident_identity = (
        resident.layer_id,
        resident.rank,
        resident.world_size,
        resident.intermediate_start,
        resident.intermediate_end,
    )
    if resident_identity != (
        layer_id,
        rank,
        config.world_size,
        expected_start,
        expected_end,
    ):
        raise ValueError(
            f"resident MoE identity {resident_identity} does not match layer/rank slice"
        )
    gate_identity = (gate.layer_id, gate.rank, gate.world_size)
    if gate_identity != (layer_id, rank, config.world_size):
        raise ValueError(
            f"gate identity {gate_identity} does not match resident layer/rank"
        )

    route_kind = str(layer_spec["route_kind"])
    if route_kind not in ("hash", "learned"):
        raise ValueError(
            f"layer {layer_id} has unsupported routing contract {route_kind!r}"
        )
    if gate.route_kind != route_kind:
        raise ValueError(
            f"layer {layer_id} requires {route_kind} routing, got {gate.route_kind}"
        )
    if route_kind == "hash":
        tid2eid = gate.tid2eid
        if gate.bias is not None or tid2eid is None:
            raise ValueError("hash gate requires tid2eid and forbids bias")
        if tid2eid.ndim != 2 or tid2eid.shape[1] != config.topk:
            raise ValueError(
                f"hash tid2eid shape must be [vocab, {config.topk}], "
                f"got {tuple(tid2eid.shape)}"
            )
        if tid2eid.dtype != torch.int64 or tid2eid.device != device:
            raise ValueError("hash tid2eid must be int64 on the configured device")
        _validate_hash_route_table(tid2eid, experts=config.experts)
    else:
        bias = gate.bias
        if gate.tid2eid is not None or bias is None:
            raise ValueError("learned gate requires bias and forbids tid2eid")
        if tuple(bias.shape) != (config.experts,):
            raise ValueError("learned gate bias shape mismatch")
        if bias.device != device:
            raise ValueError("learned gate bias must use the configured device")

    checkpoint_ids = (resident.checkpoint_id, gate.checkpoint_id)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in checkpoint_ids
    ):
        raise ValueError("gate and MoE require SHA-256 checkpoint identities")
    if checkpoint_ids[0] != checkpoint_ids[1]:
        raise ValueError("gate and MoE weights come from different checkpoints")
    return layer_id, route_kind


def _validate_hash_route_table(
    tid2eid: torch.Tensor,
    *,
    experts: int,
) -> None:
    """Validate checkpoint hash IDs once, before any graph or timed region."""

    if tid2eid.device.type == "meta":
        return
    minimum = int(tid2eid.min().item())
    maximum = int(tid2eid.max().item())
    if minimum < 0 or maximum >= experts:
        raise ValueError(
            f"hash route table expert range [{minimum}, {maximum}] is outside "
            f"[0, {experts})"
        )
    ordered = tid2eid.sort(dim=1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
        raise ValueError("hash route table rows must contain unique expert IDs")


def _snapshot_learned_route_override(
    value: MoERouteOverride,
    *,
    global_rows: int,
    config: TP4MoEConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and own an injected route before any collective or slot change."""

    if not isinstance(value, MoERouteOverride):
        raise TypeError("route_override must be a MoERouteOverride")
    ids = value.ids
    weights = value.weights
    if not isinstance(ids, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("route override IDs and weights must be tensors")
    expected_shape = (global_rows, config.topk)
    if tuple(ids.shape) != expected_shape or tuple(weights.shape) != expected_shape:
        raise ValueError(
            "route override IDs/weights must have full-global shape "
            f"{expected_shape}"
        )
    if ids.dtype != torch.int32 or weights.dtype != torch.float32:
        raise TypeError("route override requires INT32 IDs and FP32 weights")
    if ids.device != device or weights.device != device:
        raise ValueError("route override tensors must use the configured device")
    if not ids.is_contiguous() or not weights.is_contiguous():
        raise ValueError("route override tensors must be contiguous")
    if ids.requires_grad or weights.requires_grad:
        raise ValueError("route override tensors must not require gradients")
    ids = ids.detach().clone(memory_format=torch.contiguous_format)
    weights = weights.detach().clone(memory_format=torch.contiguous_format)
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("route override weights must be finite")
    if bool((weights <= 0).any().item()):
        raise ValueError("route override weights must be strictly positive")
    minimum = int(ids.min().item())
    maximum = int(ids.max().item())
    if minimum < 0 or maximum >= config.experts:
        raise ValueError(
            f"route override expert range [{minimum}, {maximum}] is outside "
            f"[0, {config.experts})"
        )
    ordered = ids.sort(dim=1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
        raise ValueError("route override rows must contain unique expert IDs")
    row_sums = weights.sum(dim=1)
    expected_sums = torch.full_like(row_sums, float(config.route_scale))
    maximum_error = float((row_sums - expected_sums).abs().max().item())
    tolerance = (
        8.0
        * torch.finfo(torch.float32).eps
        * max(1.0, abs(float(config.route_scale)))
    )
    if maximum_error > tolerance:
        raise ValueError(
            "route override weights must sum to route_scale per row; "
            f"max_abs_error={maximum_error}, tolerance={tolerance}"
        )
    return ids, weights


class TP4MoE:
    """All experts resident per rank with one intermediate slice per TP rank.

    Collective launches remain scheduler-serialized. Slots isolate storage, but
    they do not make NCCL collective ordering safe across Python threads.
    """

    def __init__(
        self,
        *,
        config: TP4MoEConfig,
        resident: ResidentMoEWeights,
        gate: ResidentGateWeights,
        rank: int,
        device: torch.device,
        global_row_shapes: Iterable[int],
        group: dist.ProcessGroup | None = None,
        slots_per_shape: int = 1,
        alignment_provider: Callable[..., object] | None = None,
        marlin_input_dtype: torch.dtype | None = None,
        buffer_donor: "TP4MoE | None" = None,
    ) -> None:
        config.validate()
        if marlin_input_dtype not in (None, torch.float8_e4m3fn):
            raise ValueError(
                f"unsupported Marlin input dtype {marlin_input_dtype!r}"
            )
        if getattr(resident, "marlin_input_dtype", None) != marlin_input_dtype:
            raise ValueError(
                "resident Marlin repack input dtype "
                f"{getattr(resident, 'marlin_input_dtype', None)!r} does not "
                f"match requested {marlin_input_dtype!r}; W4A16/W4A8 layouts "
                "are incompatible"
            )
        if not dist.is_initialized():
            raise RuntimeError("TP4MoE requires an initialized process group")
        if dist.get_world_size(group) != config.world_size or dist.get_rank(group) != rank:
            raise ValueError("process-group rank/world does not match TP4MoE")
        layer_id, route_kind = _validate_moe_identity(
            config=config,
            resident=resident,
            gate=gate,
            rank=rank,
            device=device,
        )

        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            _fused_marlin_moe,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
        )
        from vllm.scalar_type import scalar_types

        self.config = config
        self.resident = resident
        self.gate = gate
        self.layer_id = layer_id
        self.route_kind = route_kind
        self.rank = rank
        self.device = device
        self.group = group
        self.shared = prepare_shared_bf16(resident.shared)
        self._fused = _fused_marlin_moe
        self._activation = MoEActivation.SILU
        self._quant_type = scalar_types.float4_e2m1f
        self._make_workspace = marlin_make_workspace_new
        self._marlin_input_dtype = marlin_input_dtype
        if marlin_input_dtype is not None:
            _preset_w4a8_activation_quant()
        # C2F 23rd vertical lever B: row-blocked collective/compute pipeline.
        # 0 or 1 == the sequential path (unchanged).  Set through
        # `enable_collective_overlap`.
        self._overlap_blocks = 0
        # Pure observability: which path each forward actually took.
        self.overlap_stats: dict[str, int] = {
            "overlapped_calls": 0,
            "overlapped_rows": 0,
            "sequential_calls": 0,
            "sequential_rows": 0,
        }
        self._overlap_alignment: dict[tuple[int, int, int], Any] = {}
        self._route_tensor_observer: list[MoERouteTensors] | None = None
        self._route_observer_captures_local_input = False
        self._route_capture_buffers: dict[
            tuple[int, int], MoERouteCaptureBuffer
        ] = {}
        self._route_capture_registrations: dict[
            tuple[int, int], _MoERouteCaptureRegistration
        ] = {}
        if alignment_provider is not None:
            if route_kind != "learned":
                raise ValueError(
                    "an injected MoE alignment provider is only valid for learned routing"
                )
            if not callable(alignment_provider):
                raise TypeError("alignment_provider must be callable")
        self._owned_alignment_provider = alignment_provider
        self._active_alignment_provider = alignment_provider
        self._alignment_provider_binding_active = False
        self._alignment_provider_binding_keys: frozenset[tuple[int, int]] = frozenset()
        self._alignment_provider_poisoned = False
        self._alignment_provider_tensors = self._snapshot_alignment_provider_tensors(
            alignment_provider
        )
        self._alignment_provider_manifests = tuple(
            _tensor_storage_manifest(tensor)
            for tensor in self._alignment_provider_tensors
        )
        if slots_per_shape < 1:
            raise ValueError("slots_per_shape must be positive")
        if buffer_donor is not None:
            # C2F prefill vertical: per-layer TP4MoE instances of one stage
            # never run concurrently, so the multi-GB per-shape Marlin
            # buffers can be shared across layers (at prefill rows they
            # would otherwise multiply by the layer count and OOM the card).
            # The buffer state machine ("free"/"in_flight"/"poisoned") and
            # completion events already serialize reuse.
            if not isinstance(buffer_donor, TP4MoE):
                raise TypeError("buffer_donor must be a TP4MoE")
            if buffer_donor is self:
                raise ValueError("buffer_donor must be a different instance")
            if buffer_donor.config != self.config:
                raise ValueError("buffer_donor config differs")
            if (
                buffer_donor.device != self.device
                or buffer_donor.group is not self.group
            ):
                raise ValueError("buffer_donor device/group differs")
            if buffer_donor.route_kind != self.route_kind:
                raise ValueError(
                    "buffer_donor route kind differs (input-ID storage shape)"
                )
            needed = {
                (int(rows), slot)
                for rows in global_row_shapes
                for slot in range(slots_per_shape)
            }
            missing = needed - set(buffer_donor._buffers)
            if missing:
                raise ValueError(
                    f"buffer_donor lacks registered shapes {sorted(missing)}"
                )
            self._buffers = buffer_donor._buffers
        else:
            self._buffers = {}
            for rows in global_row_shapes:
                for slot in range(slots_per_shape):
                    self._register_shape(int(rows), slot)
        self._validate_alignment_provider_storage()

    def _snapshot_alignment_provider_tensors(
        self, provider: Callable[..., object] | None
    ) -> tuple[torch.Tensor, ...]:
        if provider is None:
            return ()
        storage_tensors = getattr(provider, "storage_tensors", None)
        if not callable(storage_tensors):
            raise TypeError("alignment_provider must expose callable storage_tensors")
        observed = storage_tensors()
        if not isinstance(observed, tuple):
            raise TypeError("alignment_provider.storage_tensors() must return a tuple")
        pointers: set[int] = set()
        for tensor in observed:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("alignment provider storage must contain only tensors")
            if tensor.device != self.device:
                raise ValueError("alignment provider storage device differs")
            if tensor.numel() <= 0 or not tensor.is_contiguous():
                raise ValueError(
                    "alignment provider storage must be nonempty and contiguous"
                )
            if tensor.requires_grad:
                raise ValueError("alignment provider storage must not require gradients")
            storage = tensor.untyped_storage()
            if (
                tensor.storage_offset() != 0
                or storage.nbytes() != tensor.numel() * tensor.element_size()
            ):
                raise ValueError(
                    "alignment provider tensors must own their complete storage"
                )
            pointer = int(storage.data_ptr())
            if pointer in pointers:
                raise ValueError("alignment provider storage tensors must not alias")
            pointers.add(pointer)
        return observed

    def _alignment_runtime_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors: list[torch.Tensor] = []

        def append(value: object) -> None:
            if isinstance(value, torch.Tensor):
                tensors.append(value)

        for name in ("w1", "w3", "w2"):
            append(getattr(self.shared, name, None))
        for owner, names in (
            (
                getattr(self.resident, "routed", None),
                ("w13_q", "w13_s", "w2_q", "w2_s"),
            ),
            (
                getattr(self.resident, "shared", None),
                ("w1", "s1", "w3", "s3", "w2", "s2"),
            ),
            (self.gate, ("weight", "bias", "tid2eid")),
        ):
            for name in names:
                append(getattr(owner, name, None))
        for buffers in self._buffers.values():
            for value in buffers.__dict__.values():
                append(value)
            tensors.extend(
                (
                    buffers.alignment.sorted_token_ids,
                    buffers.alignment.expert_ids,
                    buffers.alignment.num_tokens_post_padded,
                )
            )
        tensors.extend(
            tensor
            for registration in self._route_capture_registrations.values()
            for tensor in registration.tensors
        )
        return tuple(tensors)

    def _validate_alignment_provider_storage(self) -> None:
        provider = self._owned_alignment_provider
        if provider is None:
            if self._alignment_provider_tensors:
                raise AssertionError("provider-free runtime retained provider storage")
            if self._alignment_provider_manifests:
                raise AssertionError("provider-free runtime retained storage manifests")
            return
        observed = self._snapshot_alignment_provider_tensors(provider)
        expected = self._alignment_provider_tensors
        if len(observed) != len(expected) or any(
            actual is not wanted for actual, wanted in zip(observed, expected)
        ):
            raise ValueError("alignment provider storage tensor identity drifted")
        observed_manifests = tuple(
            _tensor_storage_manifest(tensor) for tensor in observed
        )
        if observed_manifests != self._alignment_provider_manifests:
            raise ValueError("alignment provider storage manifest drifted")
        provider_pointers = {
            manifest.storage_pointer for manifest in observed_manifests
        }
        runtime_pointers = {
            int(tensor.untyped_storage().data_ptr())
            for tensor in self._alignment_runtime_tensors()
        }
        if provider_pointers.intersection(runtime_pointers):
            raise ValueError("alignment provider storage aliases TP4MoE storage")

    def _require_alignment_runtime_healthy(self) -> None:
        if self._alignment_provider_poisoned:
            raise RuntimeError(
                "TP4 MoE alignment-provider lifecycle is poisoned; worker restart "
                "is required"
            )

    def _poison_alignment_runtime(self) -> None:
        self._alignment_provider_poisoned = True
        for buffers in self._buffers.values():
            buffers.state = "poisoned"

    def _register_shape(self, global_rows: int, slot: int) -> None:
        cfg = self.config
        if global_rows <= 0 or global_rows % cfg.world_size:
            raise ValueError("global MoE rows must be positive and divide TP4")
        key = (global_rows, slot)
        if key in self._buffers:
            return
        local_rows = global_rows // cfg.world_size
        block_size_m = _marlin_block_size_m(
            rows=global_rows,
            topk=cfg.topk,
            experts=cfg.experts,
            input_dtype=self._marlin_input_dtype,
        )
        alignment_template = torch.empty(
            global_rows,
            cfg.topk,
            dtype=torch.int32,
            device=self.device,
        )
        self._buffers[key] = _MarlinBuffers(
            workspace=self._make_workspace(self.device, cfg.world_size),
            cache13=torch.empty(
                global_rows
                * cfg.topk
                * max(2 * cfg.local_intermediate, cfg.hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            ),
            cache2=torch.empty(
                global_rows * cfg.topk * cfg.local_intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            output=torch.empty(
                global_rows, cfg.hidden_size, dtype=torch.bfloat16, device=self.device
            ),
            gathered=torch.empty(
                global_rows, cfg.hidden_size, dtype=torch.bfloat16, device=self.device
            ),
            combined=torch.empty(
                global_rows, cfg.hidden_size, dtype=torch.bfloat16, device=self.device
            ),
            reduced=torch.empty(
                local_rows, cfg.hidden_size, dtype=torch.bfloat16, device=self.device
            ),
            alignment=allocate_deterministic_moe_alignment(
                alignment_template,
                block_size=block_size_m,
                num_experts=cfg.experts,
            ),
            block_size_m=block_size_m,
            ready_event=torch.cuda.Event(blocking=False),
            gathered_input_ids=(
                torch.empty(global_rows, dtype=torch.int64, device=self.device)
                if self.route_kind == "hash"
                else None
            ),
        )

    @property
    def registered_global_rows(self) -> tuple[int, ...]:
        return tuple(sorted({rows for rows, _ in self._buffers}))

    @property
    def registered_slots(self) -> dict[int, tuple[int, ...]]:
        return {
            rows: tuple(sorted(slot for candidate, slot in self._buffers if candidate == rows))
            for rows in self.registered_global_rows
        }

    @property
    def alignment_provider(self) -> Callable[..., object] | None:
        """Return the physically owned alignment provider, independent of A/B mode."""

        return self._owned_alignment_provider

    @contextmanager
    def bind_alignment_provider_for_capture(
        self,
        provider: Callable[..., object] | None,
        *,
        buffer_keys: Iterable[tuple[int, int]],
    ) -> Iterator[None]:
        """Select the Python path used by one fresh CUDA-graph capture.

        Existing graphs retain the kernels captured earlier. Before reusing the
        listed keys in a different mode, callers must synchronize, destroy their
        graphs, and reset each completion event.
        """

        if provider is not None and provider is not self._owned_alignment_provider:
            raise ValueError("capture provider must be the physically owned provider")
        self._require_alignment_runtime_healthy()
        if self._alignment_provider_binding_active:
            raise RuntimeError("alignment-provider capture binding cannot be nested")
        keys = tuple(buffer_keys)
        if not keys:
            raise ValueError("alignment-provider capture requires buffer keys")
        for key in keys:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in key
                )
            ):
                raise TypeError("alignment-provider capture keys must be integer pairs")
            if key not in self._buffers:
                raise ValueError(
                    f"alignment-provider capture key {key} is not registered"
                )
        if len(set(keys)) != len(keys):
            raise ValueError("alignment-provider capture buffer keys must be unique")
        key_set = set(keys)
        nonfree = {
            key: buffers.state
            for key, buffers in self._buffers.items()
            if buffers.state != "free"
        }
        if nonfree:
            raise RuntimeError(
                "alignment-provider capture binding requires all slots free; "
                f"states={nonfree}"
            )
        unsafe = {
            key: {
                "state": buffers.state,
                "has_completion_event": buffers.has_completion_event,
            }
            for key, buffers in self._buffers.items()
            if key in key_set
            and (buffers.state != "free" or buffers.has_completion_event)
        }
        if unsafe:
            raise RuntimeError(
                "alignment-provider capture binding requires free reset slots; "
                f"unsafe={unsafe}"
            )
        other_lifecycle = {
            key: (buffers.state, buffers.has_completion_event, buffers.ready_event)
            for key, buffers in self._buffers.items()
            if key not in key_set
        }
        lifecycle_before = {
            key: (buffers.state, buffers.has_completion_event, buffers.ready_event)
            for key, buffers in self._buffers.items()
        }
        try:
            self._validate_alignment_provider_storage()
        except Exception as exc:
            self._poison_alignment_runtime()
            raise RuntimeError(
                "alignment-provider storage validation failed; worker restart "
                "is required"
            ) from exc
        previous = self._active_alignment_provider
        self._active_alignment_provider = provider
        self._alignment_provider_binding_active = True
        self._alignment_provider_binding_keys = frozenset(keys)
        completed = False
        try:
            yield
            completed = True
        finally:
            self._active_alignment_provider = previous
            self._alignment_provider_binding_active = False
            self._alignment_provider_binding_keys = frozenset()
            invalid_targets = {
                key: {
                    "state": self._buffers[key].state,
                    "has_completion_event": self._buffers[key].has_completion_event,
                }
                for key in keys
                if (
                    self._buffers[key].state != "free"
                    or not self._buffers[key].has_completion_event
                    or self._buffers[key].ready_event is not lifecycle_before[key][2]
                )
                if completed
            }
            changed_others = {
                key: {
                    "before": (before[0], before[1], id(before[2])),
                    "after": (
                        self._buffers[key].state,
                        self._buffers[key].has_completion_event,
                        id(self._buffers[key].ready_event),
                    ),
                }
                for key, before in other_lifecycle.items()
                if self._buffers[key].state != before[0]
                or self._buffers[key].has_completion_event != before[1]
                or self._buffers[key].ready_event is not before[2]
            }
            exceptional_changes = {
                key: {
                    "before": (before[0], before[1], id(before[2])),
                    "after": (
                        self._buffers[key].state,
                        self._buffers[key].has_completion_event,
                        id(self._buffers[key].ready_event),
                    ),
                }
                for key, before in lifecycle_before.items()
                if not completed
                and (
                    self._buffers[key].state != before[0]
                    or self._buffers[key].has_completion_event != before[1]
                    or self._buffers[key].ready_event is not before[2]
                )
            }
            storage_error = None
            try:
                self._validate_alignment_provider_storage()
            except Exception as exc:
                storage_error = str(exc)
            if (
                invalid_targets
                or changed_others
                or exceptional_changes
                or storage_error is not None
            ):
                self._poison_alignment_runtime()
                raise RuntimeError(
                    "alignment-provider capture lifecycle validation failed; "
                    f"targets={invalid_targets}, other_changes={changed_others}, "
                    f"exceptional_changes={exceptional_changes}, "
                    f"storage_error={storage_error}"
                )

    @property
    def extra_resident_bytes(self) -> int:
        tensors = [self.shared.w1, self.shared.w3, self.shared.w2]
        tensors.extend(self._alignment_provider_tensors)
        for buffers in self._buffers.values():
            tensors.extend(
                value
                for value in buffers.__dict__.values()
                if isinstance(value, torch.Tensor)
            )
            tensors.extend(
                (
                    buffers.alignment.sorted_token_ids,
                    buffers.alignment.expert_ids,
                    buffers.alignment.num_tokens_post_padded,
                )
            )
        tensors.extend(
            tensor
            for registration in self._route_capture_registrations.values()
            for tensor in registration.tensors
        )
        return tensor_bytes(*tensors)

    @property
    def registered_route_capture_slots(self) -> dict[int, tuple[int, ...]]:
        return {
            rows: tuple(
                sorted(
                    slot
                    for candidate, slot in self._route_capture_registrations
                    if candidate == rows
                )
            )
            for rows in sorted({rows for rows, _ in self._route_capture_registrations})
        }

    def register_route_capture(
        self, global_rows: int, slot: int = 0
    ) -> MoERouteCaptureBuffer:
        """Allocate one stable route target before CUDA graph capture."""

        self._require_alignment_runtime_healthy()
        if self._alignment_provider_binding_active:
            raise RuntimeError("route capture cannot register during graph capture")

        if (
            not isinstance(global_rows, int)
            or isinstance(global_rows, bool)
            or not isinstance(slot, int)
            or isinstance(slot, bool)
        ):
            raise TypeError("route capture global_rows and slot must be integers")
        key = (global_rows, slot)
        if key not in self._buffers:
            raise ValueError(
                f"route capture key {key} has no registered MoE buffer; "
                f"registered={self.registered_slots}"
            )
        if (
            key in self._route_capture_buffers
            or key in self._route_capture_registrations
        ):
            raise RuntimeError(f"route capture key {key} is already registered")
        shape = (global_rows, self.config.topk)
        ids = torch.full(shape, -1, dtype=torch.int32, device=self.device)
        weights = torch.zeros(shape, dtype=torch.float32, device=self.device)
        generation = torch.zeros(1, dtype=torch.int64, device=self.device)
        pointers = (
            int(ids.untyped_storage().data_ptr()),
            int(weights.untyped_storage().data_ptr()),
            int(generation.untyped_storage().data_ptr()),
        )
        if len(set(pointers)) != len(pointers):
            raise RuntimeError("route capture tensors must not alias")
        capture = MoERouteCaptureBuffer(
            global_rows=global_rows,
            slot=slot,
            owner_id=id(self),
            ids=ids,
            weights=weights,
            generation=generation,
            tensor_pointers=pointers,
        )
        registration = _MoERouteCaptureRegistration(
            capture=capture,
            capture_object_id=id(capture),
            tensors=(ids, weights, generation),
            tensor_object_ids=(id(ids), id(weights), id(generation)),
            tensor_pointers=pointers,
        )
        self._route_capture_buffers[key] = capture
        self._route_capture_registrations[key] = registration
        try:
            self._validate_route_capture_buffer(key, capture)
        except BaseException:
            del self._route_capture_buffers[key]
            del self._route_capture_registrations[key]
            raise
        try:
            self._validate_alignment_provider_storage()
        except BaseException as exc:
            del self._route_capture_buffers[key]
            del self._route_capture_registrations[key]
            self._poison_alignment_runtime()
            raise RuntimeError(
                "route capture invalidated alignment-provider storage; worker "
                "restart is required"
            ) from exc
        return capture

    def _validate_route_capture_buffer(
        self,
        key: tuple[int, int],
        capture: MoERouteCaptureBuffer,
    ) -> None:
        if not isinstance(capture, MoERouteCaptureBuffer):
            raise TypeError("registered route capture has the wrong type")
        registration = self._route_capture_registrations.get(key)
        if registration is None:
            raise RuntimeError("route capture has no private registration manifest")
        if (
            capture is not registration.capture
            or id(capture) != registration.capture_object_id
        ):
            raise ValueError("route capture object differs from registration")
        global_rows, slot = key
        if (
            capture.owner_id != id(self)
            or capture.global_rows != global_rows
            or capture.slot != slot
        ):
            raise ValueError("route capture identity differs from registration")
        expected = (
            ("ids", capture.ids, (global_rows, self.config.topk), torch.int32),
            (
                "weights",
                capture.weights,
                (global_rows, self.config.topk),
                torch.float32,
            ),
            ("generation", capture.generation, (1,), torch.int64),
        )
        pointers = []
        for index, (name, value, shape, dtype) in enumerate(expected):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"route capture {name} must be a tensor")
            if value is not registration.tensors[index] or id(value) != (
                registration.tensor_object_ids[index]
            ):
                raise ValueError(
                    f"route capture {name} tensor differs from registration"
                )
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"route capture {name} shape {tuple(value.shape)} != {shape}"
                )
            if value.dtype != dtype or value.device != self.device:
                raise ValueError(f"route capture {name} dtype/device differs")
            if not value.is_contiguous():
                raise ValueError(f"route capture {name} must be contiguous")
            if value.requires_grad:
                raise ValueError(f"route capture {name} must not require gradients")
            pointers.append(int(value.untyped_storage().data_ptr()))
        if len(set(pointers)) != len(pointers):
            raise ValueError("route capture tensors must not alias")
        if capture.tensor_pointers != registration.tensor_pointers:
            raise ValueError("route capture pointer contract differs from registration")
        if tuple(pointers) != registration.tensor_pointers:
            raise ValueError("route capture tensor storage differs from registration")

        occupied = {
            int(value.untyped_storage().data_ptr())
            for buffers in self._buffers.values()
            for value in (
                buffers.workspace,
                buffers.cache13,
                buffers.cache2,
                buffers.output,
                buffers.gathered,
                buffers.combined,
                buffers.reduced,
                buffers.alignment.sorted_token_ids,
                buffers.alignment.expert_ids,
                buffers.alignment.num_tokens_post_padded,
                buffers.gathered_input_ids,
            )
            if isinstance(value, torch.Tensor)
        }
        occupied.update(
            pointer
            for other_key, other in self._route_capture_registrations.items()
            if other_key != key
            for pointer in other.tensor_pointers
        )
        if occupied.intersection(pointers):
            raise ValueError("route capture storage aliases another resident buffer")

    def route_capture_buffer(
        self, global_rows: int, slot: int = 0
    ) -> MoERouteCaptureBuffer:
        key = (global_rows, slot)
        capture = self._route_capture_buffers.get(key)
        if capture is None:
            raise ValueError(f"route capture key {key} is not registered")
        self._validate_route_capture_buffer(key, capture)
        return capture

    def snapshot_route_capture(
        self,
        global_rows: int,
        slot: int = 0,
        *,
        expected_generation: int,
    ) -> MoERouteCaptureEvidence:
        """Synchronously own a CPU snapshot outside graph replay/timed regions."""

        if (
            not isinstance(expected_generation, int)
            or isinstance(expected_generation, bool)
            or expected_generation <= 0
        ):
            raise ValueError("expected_generation must be a positive integer")
        capture = self.route_capture_buffer(global_rows, slot)
        generation = int(capture.generation.item())
        if generation == 0:
            raise RuntimeError("route capture has not recorded an effective route")
        if generation != expected_generation:
            raise RuntimeError(
                f"route capture generation {generation} != expected "
                f"{expected_generation}"
            )
        ids = capture.ids.detach().cpu().clone().contiguous()
        weights = capture.weights.detach().cpu().clone().contiguous()
        generation_after_copy = int(capture.generation.item())
        if generation_after_copy != expected_generation:
            raise RuntimeError(
                "route capture generation changed while owning host evidence"
            )
        return MoERouteCaptureEvidence(
            global_rows=global_rows,
            slot=slot,
            ids=ids,
            weights=weights,
            generation=generation,
            device_tensor_pointers=capture.tensor_pointers,
            route_digest=_tensor_digest(ids, weights),
        )

    def reset_route_capture_generation(
        self, global_rows: int, slot: int = 0
    ) -> None:
        """Invalidate captured evidence in place after graph setup/warmup."""

        capture = self.route_capture_buffer(global_rows, slot)
        capture.generation.zero_()

    def _capture_route_for_marlin(
        self,
        key: tuple[int, int],
        route_weights: torch.Tensor,
        route_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy effective routes into stable storage without host value reads."""

        registrations = self._route_capture_registrations
        if key not in registrations:
            return route_weights, route_ids
        if set(self._route_capture_buffers) != set(registrations):
            raise RuntimeError("route capture registry differs from private manifest")
        for registered_key, registered_capture in self._route_capture_buffers.items():
            self._validate_route_capture_buffer(registered_key, registered_capture)
        capture = self._route_capture_buffers[key]
        expected_shape = (key[0], self.config.topk)
        expected = (
            ("ids", route_ids, torch.int32),
            ("weights", route_weights, torch.float32),
        )
        source_pointers = []
        for name, value, dtype in expected:
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"effective route {name} shape {tuple(value.shape)} != "
                    f"{expected_shape}"
                )
            if value.dtype != dtype or value.device != self.device:
                raise ValueError(f"effective route {name} dtype/device differs")
            if not value.is_contiguous():
                raise ValueError(f"effective route {name} must be contiguous")
            pointer = int(value.untyped_storage().data_ptr())
            source_pointers.append(pointer)
        if len(set(source_pointers)) != len(source_pointers):
            raise ValueError("effective route IDs and weights must not alias")
        capture_target_pointers = {
            pointer
            for registration in registrations.values()
            for pointer in registration.tensor_pointers
        }
        if capture_target_pointers.intersection(source_pointers):
            raise ValueError("effective route storage aliases a capture target")
        capture.ids.copy_(route_ids)
        capture.weights.copy_(route_weights)
        capture.generation.add_(1)
        return capture.weights, capture.ids

    def slot_completion_status(self, global_rows: int, slot: int) -> dict[str, Any]:
        """Return host-only lifecycle state for a pre-registered buffer slot."""

        key = (global_rows, slot)
        if key not in self._buffers:
            raise ValueError(f"global row/slot {key} was not pre-registered")
        buffers = self._buffers[key]
        return {
            "state": buffers.state,
            "has_completion_event": buffers.has_completion_event,
        }

    def reset_free_slot_completion_event(self, global_rows: int, slot: int) -> None:
        """Replace a free slot's event after its owning CUDA graph is destroyed.

        The caller must synchronize the device and reset the old ``CUDAGraph``
        before invoking this host-only lifecycle operation.
        """

        self._require_alignment_runtime_healthy()
        if self._alignment_provider_binding_active:
            raise RuntimeError(
                "MoE slot completion events cannot reset during graph capture"
            )
        key = (global_rows, slot)
        if key not in self._buffers:
            raise ValueError(f"global row/slot {key} was not pre-registered")
        buffers = self._buffers[key]
        if buffers.state != "free":
            raise RuntimeError(f"TP4 MoE buffer slot {key} is not free")
        buffers.ready_event = torch.cuda.Event(blocking=False)
        buffers.has_completion_event = False

    @contextmanager
    def observe_route_tensors(
        self, *, capture_local_input: bool = False
    ) -> Iterator[list[MoERouteTensors]]:
        """Retain eager route evidence, optionally including the local MoE input."""

        if not isinstance(capture_local_input, bool):
            raise TypeError("capture_local_input must be bool")
        if self._route_tensor_observer is not None:
            raise RuntimeError("route tensor observation cannot be nested")
        observed: list[MoERouteTensors] = []
        self._route_tensor_observer = observed
        self._route_observer_captures_local_input = capture_local_input
        try:
            yield observed
        finally:
            self._route_tensor_observer = None
            self._route_observer_captures_local_input = False

    # ------------------------------------------------------------------
    # C2F 23rd vertical lever B: overlap the two TP4 collectives with the
    # routed-expert compute.
    #
    # The sequential path runs all_gather -> gate -> marlin -> shared ->
    # combine -> reduce_scatter, so both collectives (17.7 ms/layer at
    # chunk 8192, 44% of the MoE bucket) are fully exposed.  Everything
    # between the two collectives is *per row*, so the call can be split into
    # row blocks and pipelined: block k's all_gather overlaps block k-1's
    # compute, and block k's reduce_scatter overlaps block k+1's compute.
    #
    # Layout.  `all_gather_into_tensor` writes rank-major, so a per-block
    # gather produces a **block-major** buffer:
    #
    #     gathered[k*B*W + r*B + j] == rank r's local row k*B + j
    #
    # rather than the sequential path's `gathered[r*L + i]`.  That is a row
    # permutation, and it is self-consistent across both collectives: the
    # matching per-block `reduce_scatter_tensor(reduced[k*B:(k+1)*B],
    # combined[k*B*W:(k+1)*B*W])` returns this rank's own block-k rows, so the
    # assembled local output is in the ordinary row order.  Both slices are
    # contiguous views, so no repacking copies are needed.
    #
    # Semantics.  Gate, Marlin, the shared expert and the combine are all
    # row-local, so the permutation cannot change a row's value; what it does
    # change is which Marlin M-block a row lands in (the alignment groups rows
    # by expert into `block_size_m` blocks).  Marlin's K reduction is over the
    # hidden dimension and does not depend on a row's M position, so per-row
    # bitwise identity is *expected* -- but it is expected, not proven, so
    # `--gate-moe-overlap` measures it rather than assuming it.
    # ------------------------------------------------------------------

    def enable_collective_overlap(self, blocks: int) -> None:
        """Turn on (blocks > 1) or off (blocks <= 1) the pipelined path."""

        if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 0:
            raise ValueError("overlap blocks must be a non-negative integer")
        self._overlap_blocks = blocks

    @property
    def collective_overlap_blocks(self) -> int:
        return self._overlap_blocks

    def _overlap_supported(
        self,
        *,
        local_rows: int,
        blocks: int,
        slot: int,
        global_rows: int,
        collect_trace: bool,
        collect_stage_digests: bool,
        route_override: Any,
    ) -> bool:
        """Only the plain prefill configuration takes the pipelined path."""

        return (
            blocks > 1
            and self.route_kind == "learned"
            and route_override is None
            and not collect_trace
            and not collect_stage_digests
            and self._route_tensor_observer is None
            and self._active_alignment_provider is None
            and not self._route_capture_registrations
            and local_rows % blocks == 0
            and (local_rows // blocks) > 0
        )

    def _overlap_alignment_for(
        self, global_rows: int, slot: int, blocks: int, block_global_rows: int
    ):
        key = (global_rows, slot, blocks)
        existing = self._overlap_alignment.get(key)
        if existing is not None:
            return existing
        cfg = self.config
        block_size_m = _marlin_block_size_m(
            rows=block_global_rows,
            topk=cfg.topk,
            experts=cfg.experts,
            input_dtype=self._marlin_input_dtype,
        )
        template = torch.empty(
            block_global_rows, cfg.topk, dtype=torch.int32, device=self.device
        )
        alignment = allocate_deterministic_moe_alignment(
            template, block_size=block_size_m, num_experts=cfg.experts
        )
        self._overlap_alignment[key] = (alignment, block_size_m)
        return self._overlap_alignment[key]

    def _compute_block(
        self,
        gathered_block: torch.Tensor,
        combined_block: torch.Tensor,
        buffers: _MarlinBuffers,
        alignment,
        block_size_m: int,
    ) -> None:
        """Gate + Marlin + shared + combine for one row block (in place)."""

        cfg = self.config
        rows = gathered_block.shape[0]
        bias = self.gate.bias
        if bias is None:
            raise AssertionError("validated learned gate lost bias")
        gate = gate_forward_with_boundary(
            gathered_block,
            self.gate.weight,
            bias,
            topk=cfg.topk,
            route_scale=cfg.route_scale,
        )
        route_weights = gate.routing_weights.contiguous()
        route_ids = gate.routing_ids.contiguous()
        deterministic_moe_align_block_size(
            route_ids,
            block_size=block_size_m,
            num_experts=cfg.experts,
            output=alignment,
        )
        cache13_rows = rows * cfg.topk * max(
            2 * cfg.local_intermediate, cfg.hidden_size
        )
        cache2_rows = rows * cfg.topk * cfg.local_intermediate
        contributions = self._fused(
            hidden_states=gathered_block,
            w1=self.resident.routed.w13_q,
            w2=self.resident.routed.w2_q,
            bias1=None,
            bias2=None,
            w1_scale=self.resident.routed.w13_s,
            w2_scale=self.resident.routed.w2_s,
            topk_weights=route_weights,
            num_topk=cfg.topk,
            quant_type=self._quant_type,
            apply_router_weight_on_input=False,
            expert_map=None,
            block_size_m=block_size_m,
            sorted_token_ids=alignment.sorted_token_ids,
            expert_ids=alignment.expert_ids,
            num_tokens_post_padded=alignment.num_tokens_post_padded,
            activation=self._activation,
            workspace=buffers.workspace,
            intermediate_cache13=buffers.cache13[:cache13_rows],
            intermediate_cache2=buffers.cache2[:cache2_rows],
            output=None,
            input_dtype=self._marlin_input_dtype,
            is_k_full=True,
            clamp_limit=cfg.clamp_limit,
        ).view(rows, cfg.topk, cfg.hidden_size)
        routed = torch.sum(contributions, dim=1, out=buffers.output[:rows])
        shared = shared_bf16_partial(
            gathered_block, self.shared, clamp_limit=cfg.clamp_limit
        )
        torch.add(routed, shared, out=combined_block)

    def _call_overlapped(
        self,
        local_flat: torch.Tensor,
        buffers: _MarlinBuffers,
        blocks: int,
        original_shape: tuple[int, ...],
        stage_marker: Callable[[str], None] | None,
    ) -> torch.Tensor:
        cfg = self.config
        local_rows = local_flat.shape[0]
        block_rows = local_rows // blocks
        block_global_rows = block_rows * cfg.world_size
        alignment, block_size_m = self._overlap_alignment_for(
            local_rows * cfg.world_size, 0, blocks, block_global_rows
        )

        if stage_marker is not None:
            stage_marker("moe_inputs_ready")

        # Issue every all_gather up front: they queue on the NCCL stream and
        # drain while the compute stream works through the earlier blocks.
        gather_works = []
        for index in range(blocks):
            gather_works.append(
                dist.all_gather_into_tensor(
                    buffers.gathered[
                        index * block_global_rows : (index + 1) * block_global_rows
                    ],
                    local_flat[index * block_rows : (index + 1) * block_rows],
                    group=self.group,
                    async_op=True,
                )
            )
        if stage_marker is not None:
            stage_marker("moe_all_gather_issued")

        scatter_works = []
        for index in range(blocks):
            gather_works[index].wait()
            begin = index * block_global_rows
            end = begin + block_global_rows
            combined_block = buffers.combined[begin:end]
            self._compute_block(
                buffers.gathered[begin:end],
                combined_block,
                buffers,
                alignment,
                block_size_m,
            )
            # Issued while block index+1 is still to be computed, so the
            # reduce-scatter overlaps that compute rather than trailing it.
            scatter_works.append(
                dist.reduce_scatter_tensor(
                    buffers.reduced[
                        index * block_rows : (index + 1) * block_rows
                    ],
                    combined_block,
                    op=dist.ReduceOp.SUM,
                    group=self.group,
                    async_op=True,
                )
            )
            if stage_marker is not None:
                stage_marker(f"moe_block{index}_computed")
        for work in scatter_works:
            work.wait()
        if stage_marker is not None:
            stage_marker("moe_reduce_scatter_done")
        return buffers.reduced.reshape(original_shape).clone()

    def __call__(
        self,
        hidden_local: torch.Tensor,
        *,
        input_ids_local: torch.Tensor | None = None,
        route_override: MoERouteOverride | None = None,
        local_step: Callable[[str, Callable[[], Any]], Any] | None = None,
        slot: int = 0,
        collect_trace: bool = True,
        collect_stage_digests: bool = False,
        stage_marker: Callable[[str], None] | None = None,
    ) -> tuple[torch.Tensor, MoETrace | None]:
        cfg = self.config

        def run_local(name: str, fn: Callable[[], Any]) -> Any:
            return fn() if local_step is None else local_step(name, fn)

        def prepare_input() -> tuple[
            tuple[int, ...],
            torch.Tensor,
            torch.Tensor | None,
            tuple[torch.Tensor, torch.Tensor] | None,
            _MarlinBuffers,
        ]:
            if hidden_local.ndim != 3 or hidden_local.shape[-1] != cfg.hidden_size:
                raise ValueError(
                    "MoE input must have shape [local_batch, sequence, hidden]"
                )
            if hidden_local.dtype != torch.bfloat16 or hidden_local.device != self.device:
                raise ValueError("MoE input must use the configured CUDA BF16 device")
            local_ids = None
            if self.route_kind == "learned":
                if input_ids_local is not None:
                    raise ValueError("learned routing forbids input_ids_local")
            else:
                if route_override is not None:
                    raise ValueError("hash routing forbids route_override")
                if not isinstance(input_ids_local, torch.Tensor):
                    raise ValueError("hash routing requires input_ids_local")
                if tuple(input_ids_local.shape) != tuple(hidden_local.shape[:-1]):
                    raise ValueError(
                        "input_ids_local shape must match hidden local row axes"
                    )
                if (
                    input_ids_local.dtype != torch.int64
                    or input_ids_local.device != self.device
                ):
                    raise ValueError(
                        "input_ids_local must be int64 on the configured device"
                    )
                if not input_ids_local.is_contiguous():
                    raise ValueError("input_ids_local must be contiguous")
                local_ids = input_ids_local.view(-1)
            original_shape = tuple(hidden_local.shape)
            local_flat = hidden_local.reshape(-1, cfg.hidden_size).contiguous()
            global_rows = local_flat.shape[0] * cfg.world_size
            validated_override = None
            if route_override is not None:
                if self.route_kind != "learned":
                    raise AssertionError("validated hash route accepted an override")
                validated_override = _snapshot_learned_route_override(
                    route_override,
                    global_rows=global_rows,
                    config=cfg,
                    device=self.device,
                )
            key = (global_rows, slot)
            if key not in self._buffers:
                raise ValueError(
                    f"global row/slot {key} was not pre-registered; "
                    f"registered={self.registered_slots}"
                )
            self._require_alignment_runtime_healthy()
            if (
                self._alignment_provider_binding_active
                and key not in self._alignment_provider_binding_keys
            ):
                raise RuntimeError(
                    f"MoE buffer slot {key} is outside the active graph-capture lease"
                )
            buffers = self._buffers[key]
            if buffers.state != "free":
                raise RuntimeError(
                    f"TP4 MoE buffer slot {key} is {buffers.state}; worker restart "
                    "is required after a poisoned slot"
                )
            if self.route_kind == "hash" and buffers.gathered_input_ids is None:
                raise RuntimeError("hash route buffer has no gathered input-ID storage")
            if self.route_kind == "learned" and buffers.gathered_input_ids is not None:
                raise RuntimeError("learned route buffer unexpectedly owns input-ID storage")
            return original_shape, local_flat, local_ids, validated_override, buffers

        (
            original_shape,
            local_flat,
            local_input_ids,
            validated_override,
            buffers,
        ) = run_local("prepare TP4 MoE input", prepare_input)
        buffers.state = "in_flight"
        if buffers.has_completion_event:
            torch.cuda.current_stream(self.device).wait_event(buffers.ready_event)
        try:
            if self._overlap_supported(
                local_rows=local_flat.shape[0],
                blocks=self._overlap_blocks,
                slot=slot,
                global_rows=local_flat.shape[0] * cfg.world_size,
                collect_trace=collect_trace,
                collect_stage_digests=collect_stage_digests,
                route_override=route_override,
            ):
                # Observability only: a gate needs to prove the pipelined path
                # actually engaged rather than silently falling back (the
                # fallback in _overlap_supported is deliberately quiet).
                self.overlap_stats["overlapped_calls"] += 1
                self.overlap_stats["overlapped_rows"] += int(local_flat.shape[0])
                local_output = self._call_overlapped(
                    local_flat,
                    buffers,
                    self._overlap_blocks,
                    original_shape,
                    stage_marker,
                )
                buffers.ready_event.record(torch.cuda.current_stream(self.device))
                buffers.has_completion_event = True
                buffers.state = "free"
                if stage_marker is not None:
                    stage_marker("moe_finalize_done")
                return local_output, None
            self.overlap_stats["sequential_calls"] += 1
            self.overlap_stats["sequential_rows"] += int(local_flat.shape[0])
            if stage_marker is not None:
                stage_marker("moe_inputs_ready")
            dist.all_gather_into_tensor(buffers.gathered, local_flat, group=self.group)
            if stage_marker is not None:
                stage_marker("moe_hidden_all_gather_done")
            if local_input_ids is not None:
                gathered_input_ids = buffers.gathered_input_ids
                if gathered_input_ids is None:
                    raise AssertionError("validated hash route buffer lost ID storage")
                dist.all_gather_into_tensor(
                    gathered_input_ids,
                    local_input_ids,
                    group=self.group,
                )
            if stage_marker is not None:
                stage_marker("moe_ids_all_gather_done")

            def compute_partial() -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                dict[str, str],
            ]:
                route_source = "native"
                if self.route_kind == "hash":
                    tid2eid = self.gate.tid2eid
                    gathered_input_ids = buffers.gathered_input_ids
                    if tid2eid is None or gathered_input_ids is None:
                        raise AssertionError("validated hash gate lost route tensors")
                    gate = hash_gate_forward(
                        buffers.gathered,
                        self.gate.weight,
                        tid2eid,
                        gathered_input_ids,
                        route_scale=cfg.route_scale,
                    )
                    route_weights = gate.routing_weights.contiguous()
                    route_ids = gate.routing_ids.contiguous()
                    route_margin = None
                    selection_ids = route_ids
                    selection_scores = gate.selected_scores.contiguous()
                    native_route_weights = route_weights
                    native_route_ids = route_ids
                else:
                    bias = self.gate.bias
                    if bias is None:
                        raise AssertionError("validated learned gate lost bias")
                    gate = gate_forward_with_boundary(
                        buffers.gathered,
                        self.gate.weight,
                        bias,
                        topk=cfg.topk,
                        route_scale=cfg.route_scale,
                    )
                    native_route_weights = gate.routing_weights.contiguous()
                    native_route_ids = gate.routing_ids.contiguous()
                    route_margin = gate.margin.contiguous()
                    selection_ids = gate.selection_ids.contiguous()
                    selection_scores = gate.selection_scores.contiguous()
                    if validated_override is None:
                        route_weights = native_route_weights
                        route_ids = native_route_ids
                    else:
                        route_ids, route_weights = validated_override
                        route_source = "explicit_global"
                if stage_marker is not None:
                    stage_marker("moe_route_done")
                if self._route_tensor_observer is not None:
                    gate_logits = None
                    unbiased_scores = None
                    biased_scores = None
                    if self.route_kind == "learned":
                        bias = self.gate.bias
                        if bias is None:
                            raise AssertionError("validated learned gate lost bias")
                        gate_logits = F.linear(
                            buffers.gathered.float(), self.gate.weight.float()
                        ).contiguous()
                        unbiased_scores = F.softplus(gate_logits).sqrt().contiguous()
                        biased_scores = (unbiased_scores + bias.float()).contiguous()
                    self._route_tensor_observer.append(
                        MoERouteTensors(
                            weights=route_weights,
                            ids=route_ids,
                            margin=route_margin,
                            selection_ids=selection_ids,
                            selection_scores=selection_scores,
                            local_input=(
                                local_flat.detach().clone()
                                if self._route_observer_captures_local_input
                                else None
                            ),
                            gate_logits=gate_logits,
                            unbiased_scores=unbiased_scores,
                            biased_scores=biased_scores,
                            native_weights=native_route_weights,
                            native_ids=native_route_ids,
                            route_source=route_source,
                        )
                    )
                capture_key = (buffers.gathered.shape[0], slot)
                if capture_key in self._route_capture_registrations:
                    route_weights, route_ids = self._capture_route_for_marlin(
                        capture_key,
                        route_weights,
                        route_ids,
                    )
                alignment_provider = self._active_alignment_provider
                if alignment_provider is None:
                    deterministic_moe_align_block_size(
                        route_ids,
                        block_size=buffers.block_size_m,
                        num_experts=cfg.experts,
                        output=buffers.alignment,
                    )
                else:
                    try:
                        aligned = alignment_provider(
                            route_ids,
                            block_size=buffers.block_size_m,
                            num_experts=cfg.experts,
                            output=buffers.alignment,
                            slot=slot,
                        )
                    except BaseException:
                        self._poison_alignment_runtime()
                        raise
                    if aligned is not buffers.alignment:
                        self._poison_alignment_runtime()
                        raise RuntimeError(
                            "alignment provider must return the caller-owned output"
                        )
                contributions = self._fused(
                    hidden_states=buffers.gathered,
                    w1=self.resident.routed.w13_q,
                    w2=self.resident.routed.w2_q,
                    bias1=None,
                    bias2=None,
                    w1_scale=self.resident.routed.w13_s,
                    w2_scale=self.resident.routed.w2_s,
                    topk_weights=route_weights,
                    num_topk=cfg.topk,
                    quant_type=self._quant_type,
                    apply_router_weight_on_input=False,
                    expert_map=None,
                    block_size_m=buffers.block_size_m,
                    sorted_token_ids=buffers.alignment.sorted_token_ids,
                    expert_ids=buffers.alignment.expert_ids,
                    num_tokens_post_padded=(
                        buffers.alignment.num_tokens_post_padded
                    ),
                    activation=self._activation,
                    workspace=buffers.workspace,
                    intermediate_cache13=buffers.cache13,
                    intermediate_cache2=buffers.cache2,
                    output=None,
                    input_dtype=self._marlin_input_dtype,
                    is_k_full=True,
                    clamp_limit=cfg.clamp_limit,
                ).view(buffers.gathered.shape[0], cfg.topk, cfg.hidden_size)
                routed = torch.sum(contributions, dim=1, out=buffers.output)
                if stage_marker is not None:
                    stage_marker("moe_routed_done")
                shared = shared_bf16_partial(
                    buffers.gathered, self.shared, clamp_limit=cfg.clamp_limit
                )
                if stage_marker is not None:
                    stage_marker("moe_shared_done")
                # Elementwise BF16 add straight into the reduce-scatter input.
                # ATen's CUDA add promotes BF16 to opmath_t = float, adds in
                # FP32 and rounds once on store, so this is *bitwise* identical
                # to the earlier `(routed.float() + shared.float()).to(bf16)`
                # form -- verified exhaustively over all 2**32 ordered BF16
                # pairs by `c2f_moe_combine_gate.py` (0 mismatches) -- while
                # dropping the three FP32 [global_rows, hidden] temporaries
                # (1.61 GiB of transient allocation per call at chunk 8192).
                torch.add(routed, shared, out=buffers.combined)
                if stage_marker is not None:
                    stage_marker("moe_combine_done")
                stage_digests = {}
                if collect_stage_digests:
                    stage_digests = {
                        "gathered_input": _tensor_digest(buffers.gathered),
                        "routed_partial": _tensor_digest(routed),
                        "shared_partial": _tensor_digest(shared),
                        "combined_partial": _tensor_digest(buffers.combined),
                    }
                    if buffers.gathered_input_ids is not None:
                        stage_digests["gathered_input_ids"] = _tensor_digest(
                            buffers.gathered_input_ids
                        )
                return route_weights, route_ids, route_margin, stage_digests

            route_weights, route_ids, route_margin, stage_digests = run_local(
                "compute TP4 MoE partial", compute_partial
            )
            dist.reduce_scatter_tensor(
                buffers.reduced,
                buffers.combined,
                op=dist.ReduceOp.SUM,
                group=self.group,
            )
            if stage_marker is not None:
                stage_marker("moe_reduce_scatter_done")

            def finish() -> tuple[torch.Tensor, MoETrace | None]:
                local_view = buffers.reduced.reshape(original_shape)
                # The returned tensor owns its storage; the slot can be reused after
                # this clone without racing a downstream consumer on another stream.
                local_output = local_view.clone()
                trace = None
                if collect_trace:
                    if collect_stage_digests:
                        stage_digests["reduce_scatter_output"] = _tensor_digest(
                            buffers.reduced
                        )
                    trace = MoETrace(
                        local_input_shape=original_shape,
                        gathered_shape=tuple(buffers.gathered.shape),
                        partial_shape=tuple(buffers.combined.shape),
                        local_output_shape=tuple(local_output.shape),
                        route_ids_row_zero=tuple(int(value) for value in route_ids[0].cpu()),
                        route_weights_row_zero=tuple(
                            float(value) for value in route_weights[0].cpu()
                        ),
                        route_margin_min=(
                            None
                            if route_margin is None
                            else float(route_margin.min().item())
                        ),
                        route_digest=_tensor_digest(route_ids, route_weights),
                        stage_digests=stage_digests,
                        buffer_slot=slot,
                        route_source=(
                            "explicit_global"
                            if validated_override is not None
                            else "native"
                        ),
                    )
                buffers.ready_event.record(torch.cuda.current_stream(self.device))
                buffers.has_completion_event = True
                buffers.state = "free"
                return local_output, trace

            output_and_trace = run_local("finalize TP4 MoE trace", finish)
            if stage_marker is not None:
                stage_marker("moe_finalize_done")
            return output_and_trace
        except BaseException:
            buffers.state = "poisoned"
            raise

    def forward_tensor(
        self,
        hidden_local: torch.Tensor,
        *,
        input_ids_local: torch.Tensor | None = None,
        route_override: MoERouteOverride | None = None,
        slot: int = 0,
        stage_marker: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Return an owning output tensor without host-side trace extraction."""

        output, trace = self(
            hidden_local,
            input_ids_local=input_ids_local,
            route_override=route_override,
            slot=slot,
            collect_trace=False,
            stage_marker=stage_marker,
        )
        if trace is not None:
            raise AssertionError("trace-free TP4 MoE path unexpectedly produced a trace")
        return output


__all__ = [
    "MoETrace",
    "MoERouteCaptureBuffer",
    "MoERouteCaptureEvidence",
    "MoERouteTensors",
    "MoERouteOverride",
    "PreparedSharedBF16",
    "TP4MoE",
    "TP4MoEConfig",
    "prepare_shared_bf16",
    "shared_bf16_partial",
]
