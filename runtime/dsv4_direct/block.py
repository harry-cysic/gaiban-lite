"""Composition of direct-owned HC, attention, and TP4 MoE blocks (V4-Flash).

Ported from gaiban ``dsv4_direct/block.py`` (``DirectDecodeBlock``, :704).
Flash-specific changes:

- **Three-way attention dispatch.**  Pro dispatched on
  ``compress_ratio == 4 ? Ratio4 : Ratio128``; Flash adds the pure
  sliding-window layer type (L0/L1, ``compress_ratio == 0``,
  reference model.py:466-481), so the dispatch is
  ``0 -> WindowTorchAttention / WindowDecodePlan``,
  ``4 -> Ratio4TorchAttention / Ratio4DecodePlan``,
  ``128 -> Ratio128TorchAttention / Ratio128DecodePlan``.
- **Geometry.**  The residual contract is
  ``[local_batch, sequence, 4, 4096]`` (Pro: ``[.., 4, 7168]``); both values
  come from ``model_contract.EXPECTED_RATIO128_CONFIG`` instead of literals.
- **Routing.**  ``route_kind`` follows ``model_contract.SUPPORTED_LAYER_SPECS``
  (hash for layers < num_hash_layers == 3, learned from layer 3 on), matching
  the gaiban gate/MoE identity cross-checks unchanged.

Deliberate scope reductions for this port vertical (deferred, not dropped by
accident):

- ``forward_stateful_decode_tensor`` (cursor-driven graph-family forward) is
  ported for the CUDA-graph vertical.  Since the E0pf PP vertical it is the
  gaiban composition ``prepare_stateful_decode_pre_moe`` ->
  ``finish_stateful_decode_from_pre_moe`` over a validated
  ``StatefulPreMoEBundle`` (gaiban block.py:1146/:1232/:1365); the composed
  math is identical to the pre-E0pf direct composition, the bundle adds only
  host-side ABI validation.  ``DirectPreMoEBlockFragment`` (gaiban :1392,
  producer-only pre-MoE PP fragment) is ported with the Flash three-way
  attention dispatch; the bundle geometry is the Flash residual contract
  (``[b, s, 4, 4096]`` bf16 + ``[b, s, 4096]`` bf16 + fp32 HC post/comb)
  instead of Pro's 7168.
- The gaiban ``attention_pre_backend`` / ``ffn_post_pre_backend`` injection
  hooks are replaced by one optional ``hc_boundary_backend`` (E0hf, C2g/A5F
  lineage): when set, the intra-layer boundary (attention ``hc_post`` + FFN
  ``hc_pre`` + ``ffn_norm``) runs through the backend's fused
  ``post_pre_norm`` op; the inter-layer boundary is composed by the
  superstage chain via :meth:`DirectDecodeBlock.attention_boundary`.  With
  the default ``None`` the block uses the verified eager
  ``hc_pre``/``hc_post``/``rms_norm`` composition, byte-for-byte the
  pre-E0hf behavior.
- ``Layer3DirectBlock`` / ``Layer2DirectBlock`` were Pro test-only wrappers;
  the single ``DirectDecodeBlock`` covers all three Flash layer types, so
  they are not carried over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .attention import (
    Ratio128DecodePlan,
    Ratio128StatefulDecodePlan,
    Ratio128TorchAttention,
    rms_norm,
)
from .block_weights import ResidentBlockWeights
from .hyper_connections import hc_post, hc_pre
from .model_contract import EXPECTED_RATIO128_CONFIG, SUPPORTED_LAYER_SPECS
from .moe_runtime import TP4MoE
from .ratio4_attention import (
    Ratio4DecodePlan,
    Ratio4StatefulDecodePlan,
    Ratio4TorchAttention,
)
from .stateful_decode import DecodeGraphFamily, family_boundary_flags
from .window_attention import (
    WindowDecodePlan,
    WindowStatefulDecodePlan,
    WindowTorchAttention,
)


# Frozen Flash residual geometry (checkpoint config.json via model_contract).
BLOCK_HC_MULT = int(EXPECTED_RATIO128_CONFIG["hc_mult"])
BLOCK_HIDDEN_SIZE = int(EXPECTED_RATIO128_CONFIG["hidden_size"])

BlockAttention = (
    WindowTorchAttention | Ratio4TorchAttention | Ratio128TorchAttention
)
BlockDecodePlan = WindowDecodePlan | Ratio4DecodePlan | Ratio128DecodePlan
BlockStatefulDecodePlan = (
    WindowStatefulDecodePlan
    | Ratio4StatefulDecodePlan
    | Ratio128StatefulDecodePlan
)

# compress_ratio -> (attention type, fixed decode-plan type, stateful
# cursor-driven plan type).
_ATTENTION_DISPATCH: dict[int, tuple[type, type, type]] = {
    0: (WindowTorchAttention, WindowDecodePlan, WindowStatefulDecodePlan),
    4: (Ratio4TorchAttention, Ratio4DecodePlan, Ratio4StatefulDecodePlan),
    128: (
        Ratio128TorchAttention,
        Ratio128DecodePlan,
        Ratio128StatefulDecodePlan,
    ),
}


@dataclass(frozen=True, slots=True, eq=False)
class StatefulPreMoEBundle:
    """Owned tensor ABI between a stateful block's prepare and MoE tail.

    Port of gaiban ``StatefulPreMoEBundle`` (block.py:44) with the Flash
    geometry (``hc_mult`` 4, ``hidden_size`` 4096 via the frozen block
    constants).  The material identity is portable across distinct producer
    and consumer block instances; ``producer_owner_id`` is evidence only,
    compatibility is decided from layer/rank/world/checkpoint identity and
    the tensor ABI.
    """

    after_attention: torch.Tensor
    ffn_hidden: torch.Tensor
    ffn_post: torch.Tensor
    ffn_comb: torch.Tensor
    layer_id: int
    rank: int
    world_size: int
    checkpoint_id: str
    producer_owner_id: int
    _transport_storage: torch.Tensor | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _tensor_snapshot: tuple[tuple[object, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _transport_snapshot: tuple[object, ...] | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._validate_identity()
        snapshot = self._validate_and_snapshot_tensors()
        object.__setattr__(self, "_tensor_snapshot", snapshot)
        transport_snapshot = self._validate_transport_storage()
        object.__setattr__(self, "_transport_snapshot", transport_snapshot)

    @classmethod
    def create(
        cls,
        *,
        after_attention: torch.Tensor,
        ffn_hidden: torch.Tensor,
        ffn_post: torch.Tensor,
        ffn_comb: torch.Tensor,
        layer_id: int,
        rank: int,
        world_size: int,
        checkpoint_id: str,
        producer_owner_id: int,
    ) -> StatefulPreMoEBundle:
        """Construct and validate a producer or receiver-owned bundle."""

        return cls(
            after_attention=after_attention,
            ffn_hidden=ffn_hidden,
            ffn_post=ffn_post,
            ffn_comb=ffn_comb,
            layer_id=layer_id,
            rank=rank,
            world_size=world_size,
            checkpoint_id=checkpoint_id,
            producer_owner_id=producer_owner_id,
        )

    @property
    def material_identity(self) -> tuple[int, int, int, str]:
        return (self.layer_id, self.rank, self.world_size, self.checkpoint_id)

    @property
    def transport_storage(self) -> torch.Tensor | None:
        return self._transport_storage

    @property
    def transport_offsets(self) -> tuple[int, int, int, int] | None:
        if self._transport_storage is None:
            return None
        batch, sequence = self.after_attention.shape[:2]
        offsets, _ = self._transport_layout(batch, sequence)
        return offsets

    @staticmethod
    def required_transport_nbytes(local_batch: int, sequence: int) -> int:
        for name, value in (("local_batch", local_batch), ("sequence", sequence)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _, total_nbytes = StatefulPreMoEBundle._transport_layout(
            local_batch, sequence
        )
        return total_nbytes

    @staticmethod
    def _transport_layout(
        local_batch: int,
        sequence: int,
    ) -> tuple[tuple[int, int, int, int], int]:
        sizes = (
            local_batch * sequence * BLOCK_HC_MULT * BLOCK_HIDDEN_SIZE * 2,
            local_batch * sequence * BLOCK_HIDDEN_SIZE * 2,
            local_batch * sequence * BLOCK_HC_MULT * 4,
            local_batch * sequence * BLOCK_HC_MULT * BLOCK_HC_MULT * 4,
        )
        offsets = (0, sizes[0], sizes[0] + sizes[1], sum(sizes[:3]))
        return offsets, sum(sizes)

    @classmethod
    def allocate_transport(
        cls,
        *,
        local_batch: int,
        sequence: int,
        device: torch.device | str | int | None,
        layer_id: int,
        rank: int,
        world_size: int,
        checkpoint_id: str,
        producer_owner_id: int,
    ) -> StatefulPreMoEBundle:
        """Allocate the exact owning byte buffer used by the transport ABI."""

        total_nbytes = cls.required_transport_nbytes(local_batch, sequence)
        storage = torch.empty(total_nbytes, dtype=torch.uint8, device=device)
        return cls.create_from_storage(
            storage,
            local_batch=local_batch,
            sequence=sequence,
            layer_id=layer_id,
            rank=rank,
            world_size=world_size,
            checkpoint_id=checkpoint_id,
            producer_owner_id=producer_owner_id,
        )

    @classmethod
    def create_from_storage(
        cls,
        storage: torch.Tensor,
        *,
        local_batch: int,
        sequence: int,
        layer_id: int,
        rank: int,
        world_size: int,
        checkpoint_id: str,
        producer_owner_id: int,
    ) -> StatefulPreMoEBundle:
        """Create four exact typed views over one preallocated byte buffer."""

        total_nbytes = cls.required_transport_nbytes(local_batch, sequence)
        if not isinstance(storage, torch.Tensor):
            raise TypeError("stateful pre-MoE transport storage must be a tensor")
        if (
            storage.dtype != torch.uint8
            or storage.ndim != 1
            or not storage.is_contiguous()
        ):
            raise ValueError(
                "stateful pre-MoE transport storage must be contiguous 1D uint8"
            )
        if (
            storage.numel() != total_nbytes
            or storage.storage_offset() != 0
            or storage.data_ptr() != storage.untyped_storage().data_ptr()
            or storage.untyped_storage().nbytes() != total_nbytes
        ):
            raise ValueError(
                "stateful pre-MoE transport storage must own and exactly cover "
                f"{total_nbytes} bytes"
            )
        offsets, _ = cls._transport_layout(local_batch, sequence)

        def typed_view(
            index: int,
            end: int,
            dtype: torch.dtype,
            shape: tuple[int, ...],
        ) -> torch.Tensor:
            byte_view = storage.narrow(0, offsets[index], end - offsets[index])
            return byte_view.view(dtype).view(shape)

        after_attention = typed_view(
            0,
            offsets[1],
            torch.bfloat16,
            (local_batch, sequence, BLOCK_HC_MULT, BLOCK_HIDDEN_SIZE),
        )
        ffn_hidden = typed_view(
            1,
            offsets[2],
            torch.bfloat16,
            (local_batch, sequence, BLOCK_HIDDEN_SIZE),
        )
        ffn_post = typed_view(
            2,
            offsets[3],
            torch.float32,
            (local_batch, sequence, BLOCK_HC_MULT),
        )
        ffn_comb = typed_view(
            3,
            total_nbytes,
            torch.float32,
            (local_batch, sequence, BLOCK_HC_MULT, BLOCK_HC_MULT),
        )
        return cls(
            after_attention=after_attention,
            ffn_hidden=ffn_hidden,
            ffn_post=ffn_post,
            ffn_comb=ffn_comb,
            layer_id=layer_id,
            rank=rank,
            world_size=world_size,
            checkpoint_id=checkpoint_id,
            producer_owner_id=producer_owner_id,
            _transport_storage=storage,
        )

    def validate(self) -> None:
        """Fail if identity or any tensor binding changed after construction."""

        self._validate_identity()
        if self._validate_and_snapshot_tensors() != self._tensor_snapshot:
            raise ValueError("stateful pre-MoE bundle tensor bindings are not stable")
        if self._validate_transport_storage() != self._transport_snapshot:
            raise ValueError("stateful pre-MoE transport storage is not stable")

    def _validate_identity(self) -> None:
        if (
            not isinstance(self.layer_id, int)
            or isinstance(self.layer_id, bool)
            or self.layer_id not in SUPPORTED_LAYER_SPECS
        ):
            raise ValueError("stateful pre-MoE bundle has an invalid layer_id")
        for name, value in (("rank", self.rank), ("world_size", self.world_size)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"stateful pre-MoE bundle {name} must be an integer")
        if self.world_size <= 0 or self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("stateful pre-MoE bundle rank/world_size are invalid")
        if (
            not isinstance(self.checkpoint_id, str)
            or len(self.checkpoint_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.checkpoint_id
            )
        ):
            raise ValueError(
                "stateful pre-MoE bundle checkpoint_id must be lowercase SHA-256"
            )
        if (
            not isinstance(self.producer_owner_id, int)
            or isinstance(self.producer_owner_id, bool)
            or self.producer_owner_id <= 0
        ):
            raise ValueError(
                "stateful pre-MoE bundle producer_owner_id must be a positive integer"
            )

    def _validate_and_snapshot_tensors(
        self,
    ) -> tuple[tuple[object, ...], ...]:
        tensors = (
            ("after_attention", self.after_attention),
            ("ffn_hidden", self.ffn_hidden),
            ("ffn_post", self.ffn_post),
            ("ffn_comb", self.ffn_comb),
        )
        if any(not isinstance(value, torch.Tensor) for _, value in tensors):
            raise TypeError("stateful pre-MoE bundle values must all be tensors")
        if self.after_attention.ndim != 4 or self.after_attention.shape[2:] != (
            BLOCK_HC_MULT,
            BLOCK_HIDDEN_SIZE,
        ):
            raise ValueError(
                "stateful pre-MoE bundle after_attention must have shape "
                f"[local_batch, sequence, {BLOCK_HC_MULT}, {BLOCK_HIDDEN_SIZE}]"
            )
        batch, sequence = self.after_attention.shape[:2]
        expected = {
            "after_attention": (
                (batch, sequence, BLOCK_HC_MULT, BLOCK_HIDDEN_SIZE),
                torch.bfloat16,
            ),
            "ffn_hidden": ((batch, sequence, BLOCK_HIDDEN_SIZE), torch.bfloat16),
            "ffn_post": ((batch, sequence, BLOCK_HC_MULT), torch.float32),
            "ffn_comb": (
                (batch, sequence, BLOCK_HC_MULT, BLOCK_HC_MULT),
                torch.float32,
            ),
        }
        device = self.after_attention.device
        snapshot: list[tuple[object, ...]] = []
        byte_ranges: list[tuple[str, torch.device, int, int]] = []
        for name, value in tensors:
            expected_shape, expected_dtype = expected[name]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"stateful pre-MoE bundle {name} shape {tuple(value.shape)} "
                    f"!= {expected_shape}"
                )
            if value.dtype != expected_dtype:
                raise TypeError(
                    f"stateful pre-MoE bundle {name} dtype {value.dtype} "
                    f"!= {expected_dtype}"
                )
            if value.device != device:
                raise ValueError(
                    "stateful pre-MoE bundle tensors must share one device"
                )
            if not value.is_contiguous():
                raise ValueError(
                    f"stateful pre-MoE bundle {name} must be contiguous"
                )
            storage = value.untyped_storage()
            byte_start = (
                storage.data_ptr()
                + value.storage_offset() * value.element_size()
            )
            byte_end = byte_start + value.numel() * value.element_size()
            byte_ranges.append((name, value.device, byte_start, byte_end))
            snapshot.append(
                (
                    id(value),
                    tuple(value.shape),
                    value.dtype,
                    value.device,
                    tuple(value.stride()),
                    value.storage_offset(),
                    storage.data_ptr(),
                    storage.nbytes(),
                )
            )
        for index, (left_name, left_device, left_start, left_end) in enumerate(
            byte_ranges
        ):
            for right_name, right_device, right_start, right_end in byte_ranges[
                index + 1 :
            ]:
                if left_device == right_device and max(left_start, right_start) < min(
                    left_end, right_end
                ):
                    raise ValueError(
                        "stateful pre-MoE bundle tensor byte ranges must not overlap: "
                        f"{left_name}/{right_name}"
                    )
        return tuple(snapshot)

    def _validate_transport_storage(self) -> tuple[object, ...] | None:
        storage = self._transport_storage
        if storage is None:
            return None
        batch, sequence = self.after_attention.shape[:2]
        offsets, total_nbytes = self._transport_layout(batch, sequence)
        if (
            not isinstance(storage, torch.Tensor)
            or storage.dtype != torch.uint8
            or storage.ndim != 1
            or not storage.is_contiguous()
            or storage.device != self.after_attention.device
            or storage.numel() != total_nbytes
            or storage.storage_offset() != 0
            or storage.data_ptr() != storage.untyped_storage().data_ptr()
            or storage.untyped_storage().nbytes() != total_nbytes
        ):
            raise ValueError(
                "stateful pre-MoE transport storage no longer exactly owns its ABI"
            )
        base_pointer = storage.data_ptr()
        tensors = (
            self.after_attention,
            self.ffn_hidden,
            self.ffn_post,
            self.ffn_comb,
        )
        observed_offsets = tuple(value.data_ptr() - base_pointer for value in tensors)
        if observed_offsets != offsets or any(
            value.untyped_storage().data_ptr() != base_pointer for value in tensors
        ):
            raise ValueError(
                "stateful pre-MoE transport views do not exactly cover their offsets"
            )
        final_end = (
            observed_offsets[-1]
            + self.ffn_comb.numel() * self.ffn_comb.element_size()
        )
        if offsets[0] != 0 or final_end != total_nbytes:
            raise ValueError(
                "stateful pre-MoE transport views do not fully cover storage"
            )
        return (
            id(storage),
            storage.device,
            storage.data_ptr(),
            storage.untyped_storage().nbytes(),
            offsets,
            total_nbytes,
        )


class DirectDecodeBlock:
    """Trace-free block with independent compression and routing contracts.

    Flash layers exercise three combinations of the two checkpoint axes that
    matter to this composition: window (ratio 0) with hash routing (L0/L1),
    ratio-4 with hash or learned routing (even layers), and ratio-128 with
    learned routing (odd layers >= 3).  The composition is identical across
    those combinations; only the attention plan type and whether token IDs
    enter the MoE router differ.
    """

    def __init__(
        self,
        *,
        weights: ResidentBlockWeights,
        attention: BlockAttention,
        moe: TP4MoE,
        norm_eps: float = 1e-6,
        sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        hc_boundary_backend: Any | None = None,
    ) -> None:
        layer_id = weights.layer_id
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id not in SUPPORTED_LAYER_SPECS
        ):
            raise ValueError(
                f"DirectDecodeBlock supports layers {tuple(SUPPORTED_LAYER_SPECS)}, "
                f"got {layer_id!r}"
            )
        specification = SUPPORTED_LAYER_SPECS[layer_id]
        if bool(specification["is_mtp"]):
            raise ValueError(
                "DirectDecodeBlock does not compose the MTP block "
                "(embedding/head surface is out of scope)"
            )
        compression_ratio = int(specification["compress_ratio"])
        route_kind = str(specification["route_kind"])
        if compression_ratio not in _ATTENTION_DISPATCH:
            raise ValueError(
                f"layer-{layer_id} compression ratio {compression_ratio} has no "
                "attention dispatch"
            )
        (
            expected_attention_type,
            expected_plan_type,
            expected_stateful_plan_type,
        ) = _ATTENTION_DISPATCH[compression_ratio]
        if not isinstance(attention, expected_attention_type):
            raise TypeError(
                f"layer-{layer_id} compression ratio {compression_ratio} requires "
                f"{expected_attention_type.__name__}"
            )
        identity = (weights.layer_id, weights.rank, weights.world_size)
        expected_identity = (layer_id, moe.rank, moe.config.world_size)
        if identity != expected_identity or getattr(moe, "layer_id", None) != layer_id:
            raise ValueError(
                f"block weight identity {identity} does not match MoE {expected_identity}"
            )
        attention_identity = (
            attention.weights.layer_id,
            attention.weights.rank,
            attention.weights.world_size,
        )
        if attention_identity != identity or attention.state.layer_id != layer_id:
            raise ValueError("attention and block layer/rank/world identities differ")
        checkpoint_ids = {
            weights.checkpoint_id,
            weights.gate.checkpoint_id,
            attention.weights.checkpoint_id,
            moe.resident.checkpoint_id,
        }
        if None in checkpoint_ids or len(checkpoint_ids) != 1:
            raise ValueError("block components require one non-null checkpoint identity")
        observed_routes = (weights.gate.route_kind, getattr(moe, "route_kind", None))
        if observed_routes != (route_kind, route_kind):
            raise ValueError(
                f"layer-{layer_id} block requires checkpoint {route_kind} routing, "
                f"got gate/MoE {observed_routes}"
            )
        if norm_eps != 1e-6 or sinkhorn_iters != 20 or hc_eps != 1e-6:
            raise ValueError("direct block numerical contract must be 1e-6/20/1e-6")
        if hc_boundary_backend is not None and not callable(
            getattr(hc_boundary_backend, "post_pre_norm", None)
        ):
            raise TypeError(
                "hc_boundary_backend must expose a post_pre_norm operator"
            )
        self.layer_id = layer_id
        self.compression_ratio = compression_ratio
        self.route_kind = route_kind
        self._attention_plan_type = expected_plan_type
        self._stateful_attention_plan_type = expected_stateful_plan_type
        self.weights = weights
        self.attention = attention
        self.moe = moe
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        self.hc_boundary_backend = hc_boundary_backend

    @staticmethod
    def validate_residual(residual: torch.Tensor) -> tuple[int, ...]:
        input_shape = tuple(residual.shape)
        if residual.ndim != 4 or residual.shape[2:] != (
            BLOCK_HC_MULT,
            BLOCK_HIDDEN_SIZE,
        ):
            raise ValueError(
                "block residual must have shape [local_batch, sequence, "
                f"{BLOCK_HC_MULT}, {BLOCK_HIDDEN_SIZE}]"
            )
        if residual.dtype != torch.bfloat16:
            raise TypeError("block residual must be BF16")
        return input_shape

    @staticmethod
    def validate_input_ids(
        input_ids_local: torch.Tensor, residual: torch.Tensor
    ) -> None:
        if tuple(input_ids_local.shape) != tuple(residual.shape[:2]):
            raise ValueError("hash-route input IDs must match local batch/sequence axes")
        if input_ids_local.dtype != torch.int64:
            raise TypeError("hash-route input IDs must be int64")
        if input_ids_local.device != residual.device:
            raise ValueError("hash-route input IDs and residual must share one device")
        if not input_ids_local.is_contiguous():
            raise ValueError("hash-route input IDs must be contiguous")

    def _hc_pre(
        self,
        residual: torch.Tensor,
        *,
        branch: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.weights.hyper_connection
        if branch == "attn":
            params = (hc.attn_fn, hc.attn_scale, hc.attn_base)
        elif branch == "ffn":
            params = (hc.ffn_fn, hc.ffn_scale, hc.ffn_base)
        else:
            raise ValueError(f"unknown HC branch {branch}")
        return hc_pre(
            residual,
            *params,
            norm_eps=self.norm_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            hc_eps=self.hc_eps,
        )

    def prepare_attention(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """HC attention-side pre-reduction + attention RMSNorm.

        Mirror of :meth:`prepare_ffn` for the attention branch; the op order
        is exactly the pre-E0hf inline sequence, so callers composing it are
        bitwise identical to the original composition.
        """

        self.validate_residual(residual)
        attention_hidden, attention_post, attention_comb = self._hc_pre(
            residual, branch="attn"
        )
        attention_hidden = rms_norm(
            attention_hidden,
            self.weights.attn_norm,
            eps=self.norm_eps,
        )
        return attention_hidden, attention_post, attention_comb

    def _attention_branch_decode(
        self,
        residual: torch.Tensor,
        *,
        start_pos: int,
        attention_plan: BlockDecodePlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_hidden, attention_post, attention_comb = (
            self.prepare_attention(residual)
        )
        attention_output = self.attention.forward_decode_tensor(
            attention_hidden,
            start_pos=start_pos,
            plan=attention_plan,
        )
        return attention_output, attention_post, attention_comb

    def attention_half_decode(
        self,
        residual: torch.Tensor,
        *,
        start_pos: int,
        attention_plan: BlockDecodePlan,
    ) -> torch.Tensor:
        attention_output, attention_post, attention_comb = (
            self._attention_branch_decode(
                residual,
                start_pos=start_pos,
                attention_plan=attention_plan,
            )
        )
        return hc_post(
            attention_output,
            residual,
            attention_post,
            attention_comb,
        )

    def _attention_branch_stateful_decode(
        self,
        residual: torch.Tensor,
        *,
        attention_plan: BlockStatefulDecodePlan,
        boundary_flags: tuple[bool, bool],
        stage_marker: Callable[[str], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cursor-driven attention branch (gaiban block.py:879, Flash 3-way).

        The graph family's boundary flags map onto the layer types per the
        reference decode branch: ratio-4 layers consume ``ratio4_boundary``,
        ratio-128 layers consume ``ratio128_boundary``, and window layers
        (compress_ratio == 0, Flash-only) consume neither -- their decode
        step is the unconditional ring write (model.py:530 with the
        compressor branch at :531-532 skipped).
        """

        attention_hidden, attention_post, attention_comb = (
            self.prepare_attention(residual)
        )
        if stage_marker is not None:
            stage_marker("attention_input_ready")
        attention_output = self.run_stateful_attention(
            attention_hidden,
            attention_plan=attention_plan,
            boundary_flags=boundary_flags,
            stage_marker=stage_marker,
        )
        return attention_output, attention_post, attention_comb

    def run_stateful_attention(
        self,
        attention_hidden: torch.Tensor,
        *,
        attention_plan: BlockStatefulDecodePlan,
        boundary_flags: tuple[bool, bool],
        stage_marker: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Dispatch one cursor-driven attention step on a prepared hidden.

        Extracted from the stateful attention branch so the fused-boundary
        chain (which produces ``attention_hidden`` inside the boundary op)
        can reuse the exact three-way plan/type dispatch.
        """

        ratio4_boundary, ratio128_boundary = boundary_flags
        arguments: dict[str, Any] = {"plan": attention_plan}
        if stage_marker is not None:
            arguments["stage_marker"] = stage_marker
        if self.compression_ratio == 0:
            if not isinstance(attention_plan, WindowStatefulDecodePlan):
                raise AssertionError("validated window stateful plan changed type")
            if not isinstance(self.attention, WindowTorchAttention):
                raise AssertionError("validated window attention changed type")
        elif self.compression_ratio == 4:
            if not isinstance(attention_plan, Ratio4StatefulDecodePlan):
                raise AssertionError("validated ratio-4 stateful plan changed type")
            if not isinstance(self.attention, Ratio4TorchAttention):
                raise AssertionError("validated ratio-4 attention changed type")
            arguments["ratio4_boundary"] = ratio4_boundary
        else:
            if not isinstance(attention_plan, Ratio128StatefulDecodePlan):
                raise AssertionError("validated ratio-128 stateful plan changed type")
            if not isinstance(self.attention, Ratio128TorchAttention):
                raise AssertionError("validated ratio-128 attention changed type")
            arguments["ratio128_boundary"] = ratio128_boundary
        return self.attention.forward_stateful_decode_tensor(
            attention_hidden, **arguments
        )

    def attention_half_stateful_decode(
        self,
        residual: torch.Tensor,
        *,
        attention_plan: BlockStatefulDecodePlan,
        boundary_flags: tuple[bool, bool],
        stage_marker: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        attention_output, attention_post, attention_comb = (
            self._attention_branch_stateful_decode(
                residual,
                attention_plan=attention_plan,
                boundary_flags=boundary_flags,
                stage_marker=stage_marker,
            )
        )
        return hc_post(
            attention_output,
            residual,
            attention_post,
            attention_comb,
        )

    def prepare_ffn(
        self, after_attention: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_residual(after_attention)
        ffn_hidden, ffn_post, ffn_comb = self._hc_pre(
            after_attention, branch="ffn"
        )
        ffn_hidden = rms_norm(
            ffn_hidden,
            self.weights.ffn_norm,
            eps=self.norm_eps,
        )
        return ffn_hidden, ffn_post, ffn_comb

    def ffn_boundary(
        self,
        branch_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        *,
        backend: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Intra-layer HC boundary: attention ``hc_post`` + FFN ``hc_pre`` + norm.

        Returns ``(after_attention, ffn_hidden, ffn_post, ffn_comb)``.  With
        ``backend=None`` (and no injected block backend) this is the eager
        composition in the original op order; otherwise the backend's fused
        ``post_pre_norm`` runs the boundary with this layer's FFN HC
        parameters and ``ffn_norm``.
        """

        backend = self.hc_boundary_backend if backend is None else backend
        if backend is None:
            after_attention = hc_post(branch_output, residual, post, comb)
            ffn_hidden, ffn_post, ffn_comb = self.prepare_ffn(after_attention)
            return after_attention, ffn_hidden, ffn_post, ffn_comb
        hc = self.weights.hyper_connection
        return backend.post_pre_norm(
            branch_output,
            residual,
            post,
            comb,
            hc_fn=hc.ffn_fn,
            hc_scale=hc.ffn_scale,
            hc_base=hc.ffn_base,
            norm_weight=self.weights.ffn_norm,
            norm_eps=self.norm_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            hc_eps=self.hc_eps,
        )

    def attention_boundary(
        self,
        branch_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        *,
        backend: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inter-layer HC boundary into **this** layer's attention branch.

        Fuses the previous layer's FFN ``hc_post`` with this layer's
        attention-side ``hc_pre`` + ``attn_norm``.  Returns
        ``(block_input_residual, attention_hidden, attention_post,
        attention_comb)``.  Only the superstage chain composes this surface;
        a standalone block's tail ``hc_post`` has no fusion partner and stays
        eager.
        """

        if backend is None:
            block_residual = hc_post(branch_output, residual, post, comb)
            attention_hidden, attention_post, attention_comb = (
                self.prepare_attention(block_residual)
            )
            return block_residual, attention_hidden, attention_post, attention_comb
        hc = self.weights.hyper_connection
        return backend.post_pre_norm(
            branch_output,
            residual,
            post,
            comb,
            hc_fn=hc.attn_fn,
            hc_scale=hc.attn_scale,
            hc_base=hc.attn_base,
            norm_weight=self.weights.attn_norm,
            norm_eps=self.norm_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            hc_eps=self.hc_eps,
        )

    def _validate_route_input(
        self,
        input_ids_local: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> None:
        if self.route_kind == "hash":
            if not isinstance(input_ids_local, torch.Tensor):
                raise ValueError("hash routing requires input_ids_local")
            self.validate_input_ids(input_ids_local, reference)
        elif input_ids_local is not None:
            raise ValueError("learned routing forbids input_ids_local")

    def validate_stateful_pre_moe_bundle(
        self,
        bundle: StatefulPreMoEBundle,
    ) -> StatefulPreMoEBundle:
        """Validate one portable bundle against this block's material."""

        if type(bundle) is not StatefulPreMoEBundle:
            raise TypeError("stateful pre-MoE bundle has the wrong type")
        bundle.validate()
        expected_identity = (
            self.layer_id,
            self.weights.rank,
            self.weights.world_size,
            self.weights.checkpoint_id,
        )
        if bundle.material_identity != expected_identity:
            raise ValueError(
                "stateful pre-MoE bundle material identity "
                f"{bundle.material_identity} != {expected_identity}"
            )
        return bundle

    @staticmethod
    def _validate_bundle_owns_producer_outputs(
        bundle: StatefulPreMoEBundle,
        residual: torch.Tensor,
    ) -> None:
        residual_start = residual.data_ptr()
        residual_end = residual_start + residual.numel() * residual.element_size()
        for name in ("after_attention", "ffn_hidden", "ffn_post", "ffn_comb"):
            value = getattr(bundle, name)
            value_start = value.data_ptr()
            value_end = value_start + value.numel() * value.element_size()
            if value.device == residual.device and max(
                value_start, residual_start
            ) < min(value_end, residual_end):
                raise ValueError(
                    f"stateful pre-MoE bundle {name} must not alias its residual input"
                )

    def prepare_stateful_decode_pre_moe(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor | None = None,
        attention_plan: BlockStatefulDecodePlan,
        graph_family: DecodeGraphFamily,
        stage_marker: Callable[[str], None] | None = None,
    ) -> StatefulPreMoEBundle:
        """Run stateful attention and FFN preparation without the MoE tail.

        Port of gaiban block.py:1146.  The op sequence is byte-identical to
        the pre-E0pf ``forward_stateful_decode_tensor`` prefix; the bundle
        adds only host-side ABI validation on the produced tensors.
        """

        self.validate_residual(residual)
        self._validate_route_input(input_ids_local, residual)
        if not isinstance(attention_plan, self._stateful_attention_plan_type):
            raise TypeError(
                f"layer-{self.layer_id} compression ratio {self.compression_ratio} "
                f"requires {self._stateful_attention_plan_type.__name__}"
            )
        boundary_flags = family_boundary_flags(graph_family)
        if stage_marker is not None:
            stage_marker("block_start")
        if self.hc_boundary_backend is None:
            after_attention = self.attention_half_stateful_decode(
                residual,
                attention_plan=attention_plan,
                boundary_flags=boundary_flags,
                stage_marker=stage_marker,
            )
            if stage_marker is not None:
                stage_marker("attention_done")
            ffn_hidden, ffn_post, ffn_comb = self.prepare_ffn(after_attention)
        else:
            attention_output, attention_post, attention_comb = (
                self._attention_branch_stateful_decode(
                    residual,
                    attention_plan=attention_plan,
                    boundary_flags=boundary_flags,
                    stage_marker=stage_marker,
                )
            )
            if stage_marker is not None:
                stage_marker("attention_done")
            after_attention, ffn_hidden, ffn_post, ffn_comb = self.ffn_boundary(
                attention_output, residual, attention_post, attention_comb
            )
        bundle = StatefulPreMoEBundle.create(
            after_attention=after_attention,
            ffn_hidden=ffn_hidden,
            ffn_post=ffn_post,
            ffn_comb=ffn_comb,
            layer_id=self.layer_id,
            rank=self.weights.rank,
            world_size=self.weights.world_size,
            checkpoint_id=self.weights.checkpoint_id,
            producer_owner_id=id(self),
        )
        self._validate_bundle_owns_producer_outputs(bundle, residual)
        if stage_marker is not None:
            stage_marker("ffn_prepare_done")
        return bundle

    def finish_stateful_decode_from_pre_moe(
        self,
        bundle: StatefulPreMoEBundle,
        *,
        input_ids_local: torch.Tensor | None = None,
        moe_slot: int = 0,
        stage_marker: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Run the stateful MoE tail from a validated portable bundle."""

        bundle = self.validate_stateful_pre_moe_bundle(bundle)
        self._validate_route_input(input_ids_local, bundle.after_attention)
        moe_arguments: dict[str, Any] = {"slot": moe_slot}
        if self.route_kind == "hash":
            if input_ids_local is None:
                raise AssertionError("validated hash routing lost input IDs")
            moe_arguments["input_ids_local"] = input_ids_local
        if stage_marker is not None:
            moe_arguments["stage_marker"] = stage_marker
        moe_output = self.moe.forward_tensor(bundle.ffn_hidden, **moe_arguments)
        output = hc_post(
            moe_output,
            bundle.after_attention,
            bundle.ffn_post,
            bundle.ffn_comb,
        )
        if stage_marker is not None:
            stage_marker("block_done")
        return output

    def forward_decode_tensor(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor | None = None,
        start_pos: int,
        attention_plan: BlockDecodePlan,
        moe_slot: int = 0,
        stage_events: tuple[torch.cuda.Event, ...] | None = None,
        stage_marker: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Run one fixed-position block without host trace work."""

        self.validate_residual(residual)
        self._validate_route_input(input_ids_local, residual)
        if not isinstance(attention_plan, self._attention_plan_type):
            raise TypeError(
                f"layer-{self.layer_id} compression ratio {self.compression_ratio} "
                f"requires {self._attention_plan_type.__name__}"
            )
        if stage_events is not None and stage_marker is not None:
            raise ValueError("coarse stage events and named stage marker are exclusive")
        if stage_events is not None:
            if len(stage_events) != 5:
                raise ValueError("direct block stage timing requires five boundary events")
            if len({id(event) for event in stage_events}) != 5:
                raise ValueError("direct block stage timing events must be distinct")
            stage_events[0].record()
        if stage_marker is not None:
            stage_marker("block_start")
        if self.hc_boundary_backend is None:
            after_attention = self.attention_half_decode(
                residual,
                start_pos=start_pos,
                attention_plan=attention_plan,
            )
            if stage_events is not None:
                stage_events[1].record()
            if stage_marker is not None:
                stage_marker("attention_done")
            ffn_hidden, ffn_post, ffn_comb = self.prepare_ffn(after_attention)
        else:
            # Fused intra-layer boundary: ``attention_done`` here marks the
            # raw attention branch (pre-hc_post) because hc_post is fused
            # into the boundary op with the FFN hc_pre + norm.
            attention_output, attention_post, attention_comb = (
                self._attention_branch_decode(
                    residual,
                    start_pos=start_pos,
                    attention_plan=attention_plan,
                )
            )
            if stage_events is not None:
                stage_events[1].record()
            if stage_marker is not None:
                stage_marker("attention_done")
            after_attention, ffn_hidden, ffn_post, ffn_comb = self.ffn_boundary(
                attention_output, residual, attention_post, attention_comb
            )
        if stage_events is not None:
            stage_events[2].record()
        if stage_marker is not None:
            stage_marker("ffn_prepare_done")
        moe_arguments: dict[str, Any] = {"slot": moe_slot}
        if self.route_kind == "hash":
            if input_ids_local is None:
                raise AssertionError("validated hash routing lost input IDs")
            moe_arguments["input_ids_local"] = input_ids_local
        if stage_marker is not None:
            moe_arguments["stage_marker"] = stage_marker
        moe_output = self.moe.forward_tensor(ffn_hidden, **moe_arguments)
        if stage_events is not None:
            stage_events[3].record()
        output = hc_post(
            moe_output,
            after_attention,
            ffn_post,
            ffn_comb,
        )
        if stage_events is not None:
            stage_events[4].record()
        if stage_marker is not None:
            stage_marker("block_done")
        return output

    def forward_stateful_decode_tensor(
        self,
        residual: torch.Tensor,
        *,
        input_ids_local: torch.Tensor | None = None,
        attention_plan: BlockStatefulDecodePlan,
        graph_family: DecodeGraphFamily,
        moe_slot: int = 0,
        stage_marker: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Run one cursor-driven block without advancing the shared cursor.

        Gaiban composition (block.py:1365): ``prepare_stateful_decode_pre_moe``
        followed by ``finish_stateful_decode_from_pre_moe`` over a validated
        portable bundle.  The composed math is byte-identical to the pre-E0pf
        direct composition (E0sf-verified); the bundle indirection is the PP
        fragment-split surface exercised by the E0pf vertical.
        """

        bundle = self.prepare_stateful_decode_pre_moe(
            residual,
            input_ids_local=input_ids_local,
            attention_plan=attention_plan,
            graph_family=graph_family,
            stage_marker=stage_marker,
        )
        return self.finish_stateful_decode_from_pre_moe(
            bundle,
            input_ids_local=input_ids_local,
            moe_slot=moe_slot,
            stage_marker=stage_marker,
        )


class DirectPreMoEBlockFragment:
    """Producer-only stateful block fragment with no MoE runtime surface.

    Port of gaiban ``DirectPreMoEBlockFragment`` (block.py:1392) with the
    Flash three-way attention dispatch (window/ratio-4/ratio-128) instead of
    Pro's two-way dispatch, and the single optional ``hc_boundary_backend``
    instead of the Pro ``attention_pre_backend``/``ffn_post_pre_backend``
    pair (the same substitution the E0hf vertical made on the full block).
    A fragment produces a portable ``StatefulPreMoEBundle`` that any full
    ``DirectDecodeBlock`` with the same layer/rank/world/checkpoint material
    can finish; this is the PP fragment-split (pre-MoE stage boundary)
    producer surface.
    """

    __slots__ = (
        "layer_id",
        "compression_ratio",
        "route_kind",
        "_attention_plan_type",
        "_stateful_attention_plan_type",
        "weights",
        "attention",
        "hc_boundary_backend",
        "norm_eps",
        "sinkhorn_iters",
        "hc_eps",
    )

    def __init__(
        self,
        *,
        weights: ResidentBlockWeights,
        attention: BlockAttention,
        norm_eps: float = 1e-6,
        sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        hc_boundary_backend: Any | None = None,
    ) -> None:
        layer_id = weights.layer_id
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id not in SUPPORTED_LAYER_SPECS
        ):
            raise ValueError(
                f"DirectPreMoEBlockFragment supports layers "
                f"{tuple(SUPPORTED_LAYER_SPECS)}, got {layer_id!r}"
            )
        specification = SUPPORTED_LAYER_SPECS[layer_id]
        if bool(specification["is_mtp"]):
            raise ValueError(
                "DirectPreMoEBlockFragment does not compose the MTP block"
            )
        compression_ratio = int(specification["compress_ratio"])
        route_kind = str(specification["route_kind"])
        if compression_ratio not in _ATTENTION_DISPATCH:
            raise ValueError(
                f"layer-{layer_id} compression ratio {compression_ratio} has no "
                "attention dispatch"
            )
        (
            expected_attention_type,
            expected_plan_type,
            expected_stateful_plan_type,
        ) = _ATTENTION_DISPATCH[compression_ratio]
        if not isinstance(attention, expected_attention_type):
            raise TypeError(
                f"layer-{layer_id} compression ratio {compression_ratio} requires "
                f"{expected_attention_type.__name__}"
            )
        identity = (weights.layer_id, weights.rank, weights.world_size)
        attention_identity = (
            attention.weights.layer_id,
            attention.weights.rank,
            attention.weights.world_size,
        )
        if attention_identity != identity or attention.state.layer_id != layer_id:
            raise ValueError(
                "attention and fragment layer/rank/world identities differ"
            )
        checkpoint_ids = {
            weights.checkpoint_id,
            weights.gate.checkpoint_id,
            attention.weights.checkpoint_id,
        }
        if None in checkpoint_ids or len(checkpoint_ids) != 1:
            raise ValueError(
                "pre-MoE fragment components require one non-null checkpoint identity"
            )
        if weights.gate.route_kind != route_kind:
            raise ValueError(
                f"layer-{layer_id} fragment requires checkpoint {route_kind} "
                f"routing, got gate {weights.gate.route_kind}"
            )
        if norm_eps != 1e-6 or sinkhorn_iters != 20 or hc_eps != 1e-6:
            raise ValueError("direct fragment numerical contract must be 1e-6/20/1e-6")
        if hc_boundary_backend is not None and not callable(
            getattr(hc_boundary_backend, "post_pre_norm", None)
        ):
            raise TypeError(
                "hc_boundary_backend must expose a post_pre_norm operator"
            )
        self.layer_id = layer_id
        self.compression_ratio = compression_ratio
        self.route_kind = route_kind
        self._attention_plan_type = expected_plan_type
        self._stateful_attention_plan_type = expected_stateful_plan_type
        self.weights = weights
        self.attention = attention
        self.hc_boundary_backend = hc_boundary_backend
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps

    validate_residual = staticmethod(DirectDecodeBlock.validate_residual)
    validate_input_ids = staticmethod(DirectDecodeBlock.validate_input_ids)
    _hc_pre = DirectDecodeBlock._hc_pre
    prepare_attention = DirectDecodeBlock.prepare_attention
    _attention_branch_stateful_decode = (
        DirectDecodeBlock._attention_branch_stateful_decode
    )
    run_stateful_attention = DirectDecodeBlock.run_stateful_attention
    attention_half_stateful_decode = DirectDecodeBlock.attention_half_stateful_decode
    prepare_ffn = DirectDecodeBlock.prepare_ffn
    ffn_boundary = DirectDecodeBlock.ffn_boundary
    _validate_route_input = DirectDecodeBlock._validate_route_input
    _validate_bundle_owns_producer_outputs = staticmethod(
        DirectDecodeBlock._validate_bundle_owns_producer_outputs
    )
    prepare_stateful_decode_pre_moe = (
        DirectDecodeBlock.prepare_stateful_decode_pre_moe
    )


__all__ = [
    "BLOCK_HC_MULT",
    "BLOCK_HIDDEN_SIZE",
    "BlockDecodePlan",
    "BlockStatefulDecodePlan",
    "DirectDecodeBlock",
    "DirectPreMoEBlockFragment",
    "StatefulPreMoEBundle",
]
