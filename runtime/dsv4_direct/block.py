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
  ported for the CUDA-graph vertical as a direct composition of the stateful
  attention half + ``prepare_ffn`` + MoE tail.  The gaiban
  ``StatefulPreMoEBundle`` / ``prepare_stateful_decode_pre_moe`` /
  ``DirectPreMoEBlockFragment`` indirection is a PP fragment-split surface
  and stays deferred with the rest of PP.
- The ``attention_pre_backend`` / ``ffn_post_pre_backend`` fused-operator
  injection hooks are superstage performance surfaces and are omitted; this
  block always uses the verified eager ``hc_pre``/``hc_post``/``rms_norm``
  composition.
- ``Layer3DirectBlock`` / ``Layer2DirectBlock`` were Pro test-only wrappers;
  the single ``DirectDecodeBlock`` covers all three Flash layer types, so
  they are not carried over.
"""

from __future__ import annotations

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

    def _attention_branch_decode(
        self,
        residual: torch.Tensor,
        *,
        start_pos: int,
        attention_plan: BlockDecodePlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_residual(residual)
        attention_hidden, attention_post, attention_comb = self._hc_pre(
            residual, branch="attn"
        )
        attention_hidden = rms_norm(
            attention_hidden,
            self.weights.attn_norm,
            eps=self.norm_eps,
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

        self.validate_residual(residual)
        attention_hidden, attention_post, attention_comb = self._hc_pre(
            residual, branch="attn"
        )
        attention_hidden = rms_norm(
            attention_hidden,
            self.weights.attn_norm,
            eps=self.norm_eps,
        )
        if stage_marker is not None:
            stage_marker("attention_input_ready")
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
        attention_output = self.attention.forward_stateful_decode_tensor(
            attention_hidden, **arguments
        )
        return attention_output, attention_post, attention_comb

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

        Direct composition of the stateful attention half, FFN preparation,
        and MoE tail (gaiban block.py:1365 via :1146/:1232).  The gaiban
        ``StatefulPreMoEBundle`` / ``DirectPreMoEBlockFragment`` indirection
        exists for the PP fragment split and is deliberately not carried into
        this vertical; the composed math is identical.
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
        after_attention = self.attention_half_stateful_decode(
            residual,
            attention_plan=attention_plan,
            boundary_flags=boundary_flags,
            stage_marker=stage_marker,
        )
        if stage_marker is not None:
            stage_marker("attention_done")
        ffn_hidden, ffn_post, ffn_comb = self.prepare_ffn(after_attention)
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
        output = hc_post(
            moe_output,
            after_attention,
            ffn_post,
            ffn_comb,
        )
        if stage_marker is not None:
            stage_marker("block_done")
        return output


__all__ = [
    "BLOCK_HC_MULT",
    "BLOCK_HIDDEN_SIZE",
    "BlockDecodePlan",
    "BlockStatefulDecodePlan",
    "DirectDecodeBlock",
]
