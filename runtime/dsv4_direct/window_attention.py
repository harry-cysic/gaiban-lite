"""Direct-owned pure sliding-window attention path (Flash L0/L1).

Flash layers with ``compress_ratio == 0`` are a layer type that does not
exist in Pro: reference ``model.py`` ``Attention.__init__`` (:466-471) builds
neither a compressor nor an indexer, sizes ``kv_cache`` to exactly
``window_size`` rows (:473), and -- critically -- disables YaRN for RoPE
(:477-479):

    original_seq_len, rope_theta = 0, args.rope_theta

so the frequency table is the plain base-10000 RoPE table.  Everything else
(low-rank q projection + q_norm + weightless per-head query RMS, wkv +
kv_norm + NoPE FP8 QAT simulation, attn_sink softmax, inverse RoPE on the
output, grouped ``wo_a`` einsum, ``wo_b``) is shared with the ratio-128 path
verified in E0ef, so this module reuses those primitives from
``dsv4_direct.attention`` and only removes the compressor/compressed-KV
machinery.

This is a semantic control path, not a performance path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, MutableMapping

import torch
import torch.nn.functional as F

from .attention import (
    apply_rotary_emb,
    fp8_quant_dequant,
    precompute_freqs_cis,
    rms_norm,
    torch_sparse_attention,
    window_topk_indices,
)
from .block_weights import ResidentAttentionWeights
from .model_contract import SUPPORTED_LAYER_SPECS, validate_model_layer_config
from .moe_forward import dequant_fp8_block
from .static_kv import LATENT_DIM, WINDOW_SIZE
from .static_window_kv import StaticWindowKV


# Flash pure sliding-window layers: compress_ratio == 0
# (model_contract.FROZEN_COMPRESS_RATIOS => layers 0 and 1) plus the MTP
# block (mtp.0, layer id 43), whose Attention is the same ratio-0
# sliding-window type (reference model.py MTPBlock -> Block -> Attention with
# compress_ratios[43] == 0: no compressor, no indexer, no-YaRN base RoPE).
SUPPORTED_WINDOW_LAYER_IDS = tuple(
    layer_id
    for layer_id, specification in SUPPORTED_LAYER_SPECS.items()
    if specification["compress_ratio"] == 0
)


@dataclass(frozen=True)
class WindowAttentionConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    norm_eps: float
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_seq_len: int
    max_seq_len: int
    layer_id: int = 0

    @classmethod
    def from_model_config(
        cls,
        config: Mapping[str, Any],
        *,
        layer_id: int = 0,
        max_seq_len: int,
    ) -> "WindowAttentionConfig":
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id not in SUPPORTED_WINDOW_LAYER_IDS
        ):
            raise ValueError(
                "window attention config requires an integer frozen "
                f"sliding-window layer_id, got {layer_id!r}"
            )
        validate_model_layer_config(config, layer_id=layer_id)
        ratios = config.get("compress_ratios")
        if not isinstance(ratios, (list, tuple)) or len(ratios) <= layer_id:
            raise ValueError(f"compress_ratios does not cover layer {layer_id}")
        if int(ratios[layer_id]) != 0:
            raise ValueError(
                f"layer {layer_id} requires compress_ratio 0, got {ratios[layer_id]}"
            )
        rope = config.get("rope_scaling") or {}
        result = cls(
            hidden_size=int(config["hidden_size"]),
            num_heads=int(config["num_attention_heads"]),
            head_dim=int(config["head_dim"]),
            rope_dim=int(config["qk_rope_head_dim"]),
            q_lora_rank=int(config["q_lora_rank"]),
            o_lora_rank=int(config["o_lora_rank"]),
            o_groups=int(config["o_groups"]),
            norm_eps=float(config["rms_norm_eps"]),
            # Reference model.py:477-479: pure sliding-window layers use the
            # base rope_theta (config "rope_theta" == 10000), not
            # compress_rope_theta, and force original_seq_len = 0 which
            # disables the whole YaRN branch of precompute_freqs_cis
            # (model.py:221 `if original_seq_len > 0:`).
            rope_theta=float(config["rope_theta"]),
            # factor/beta_* are carried for provenance only; they are
            # mathematically inert when original_seq_len == 0 because both
            # the candidate table builder (attention.precompute_freqs_cis)
            # and the reference guard the YaRN correction behind
            # original_seq_len > 0.
            rope_factor=float(rope.get("factor", 16.0)),
            beta_fast=int(rope.get("beta_fast", 32)),
            beta_slow=int(rope.get("beta_slow", 1)),
            original_seq_len=0,
            max_seq_len=int(max_seq_len),
            layer_id=layer_id,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if (
            not isinstance(self.layer_id, int)
            or isinstance(self.layer_id, bool)
            or self.layer_id not in SUPPORTED_WINDOW_LAYER_IDS
        ):
            raise ValueError(
                "window attention config requires an integer frozen "
                f"sliding-window layer_id, got {self.layer_id!r}"
            )
        # DeepSeek-V4-Flash geometry, frozen from the checkpoint config.json
        # (model_contract.EXPECTED_WINDOW_HASH_CONFIG): hidden 4096, 64 heads,
        # head_dim 512 (== LATENT_DIM), rope 64, q_lora 1024, o_lora 1024,
        # o_groups 8.  RoPE is frozen to the no-YaRN reference semantics
        # (model.py:477-479): base theta 10000 and original_seq_len 0.
        expected = {
            "hidden_size": (self.hidden_size, 4096),
            "num_heads": (self.num_heads, 64),
            "head_dim": (self.head_dim, LATENT_DIM),
            "rope_dim": (self.rope_dim, 64),
            "q_lora_rank": (self.q_lora_rank, 1024),
            "o_lora_rank": (self.o_lora_rank, 1024),
            "o_groups": (self.o_groups, 8),
            "norm_eps": (self.norm_eps, 1e-6),
            "rope_theta": (self.rope_theta, 10000.0),
            "rope_factor": (self.rope_factor, 16.0),
            "beta_fast": (self.beta_fast, 32),
            "beta_slow": (self.beta_slow, 1),
            "original_seq_len": (self.original_seq_len, 0),
        }
        mismatches = {
            name: {"observed": observed, "expected": wanted}
            for name, (observed, wanted) in expected.items()
            if observed != wanted
        }
        if mismatches:
            raise ValueError(
                f"unsupported layer-{self.layer_id} window attention shape: {mismatches}"
            )
        if self.rope_dim <= 0 or self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError(
                "rope_dim must be positive, even, and no larger than head_dim"
            )
        if self.num_heads % self.o_groups:
            raise ValueError("num_heads must divide output groups")
        if not math.isfinite(self.norm_eps) or not math.isfinite(self.rope_theta):
            raise ValueError("attention numerical constants must be finite")
        if (
            not isinstance(self.max_seq_len, int)
            or isinstance(self.max_seq_len, bool)
            or self.max_seq_len < WINDOW_SIZE
        ):
            raise ValueError(
                f"max_seq_len must be an integer >= window size {WINDOW_SIZE}"
            )


@dataclass
class PreparedWindowAttentionWeights:
    """BF16 dequantized-weight control view of one window-layer attention."""

    attn_sink: torch.Tensor
    wq_a: torch.Tensor
    q_norm: torch.Tensor
    wq_b: torch.Tensor
    wkv: torch.Tensor
    kv_norm: torch.Tensor
    wo_a: torch.Tensor
    wo_b: torch.Tensor
    layer_id: int
    rank: int
    world_size: int
    checkpoint_id: str

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in self.__dict__.values()
            if isinstance(tensor, torch.Tensor)
        )


@dataclass(frozen=True)
class WindowDecodePlan:
    """Fixed-position tensors and slots for trace-free single-token decode.

    Window counterpart of ``attention.Ratio128DecodePlan``: with
    ``compress_ratio == 0`` there is no compressor state, so the plan carries
    only the RoPE slice, the full-ring top-k gather, and the ring write slot.
    ``start_pos >= WINDOW_SIZE`` is required so every ring row is valid and
    the maskless fixed-index sparse core applies (positions below one full
    window keep using the traced ``__call__`` path, exactly like the
    ratio-128 plan's ``[128, max_seq_len)`` contract).
    """

    start_pos: int
    slot: int
    batch_size: int
    hidden_size: int
    owner_id: int
    state_id: int
    frequencies: torch.Tensor
    topk_indices: torch.Tensor
    gather_indices: torch.Tensor
    batch_indices: torch.Tensor


@dataclass(frozen=True)
class WindowStatefulDecodePlan:
    """Cursor-driven fixed-shape workspace for a consecutive decode range.

    Window counterpart of ``attention.Ratio128StatefulDecodePlan``.  A window
    layer has no compressed rows, so the visible KV at every position
    ``p >= 128`` is exactly the full 128-slot ring; the workspace is one fixed
    ``[batch, 1, 128]`` gather rebuilt from the shared device position each
    step.  There is no boundary variant: the reference decode branch for
    compress_ratio == 0 is the same ring write at every position
    (model.py:530), so one plan serves every graph family unchanged.
    """

    start_position: int
    stop_position: int
    batch_size: int
    hidden_size: int
    owner_id: int
    state_id: int
    position: torch.Tensor
    window_columns: torch.Tensor
    gather_indices: torch.Tensor
    batch_indices: torch.Tensor
    tensor_pointers: tuple[int, ...]

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(value.numel() * value.element_size())
            for value in (
                self.window_columns,
                self.gather_indices,
                self.batch_indices,
            )
        )


def _window_sparse_decode_prevalidated(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    plan: "WindowDecodePlan | WindowStatefulDecodePlan",
    softmax_scale: float,
    latent_rope: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fixed-index sparse MLA over the full window ring (no padding mask).

    Same math as ``attention._torch_sparse_decode_prevalidated``; duplicated
    here so the window module does not depend on a ratio-128 private helper.

    17th vertical (workspace slimming): one FP32 materialization of the
    gathered rows plus in-place softmax -- exact conversions and identical
    elementwise values keep the output bitwise identical to the previous
    gather -> bf16 -> double-``.float()`` chain.
    """

    selected = latent_kv[plan.batch_indices, plan.gather_indices].float()
    if latent_rope is not None:
        selected[..., -latent_rope.shape[-1] :] = latent_rope[
            plan.batch_indices, plan.gather_indices
        ]
    scores = torch.einsum(
        "bshd,bskd->bshk", query.float(), selected
    ) * softmax_scale
    sink = attn_sink.float().view(1, 1, query.shape[2], 1)
    maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
    exponent = scores.sub_(maximum).exp_()
    denominator = exponent.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
    probabilities = exponent.div_(denominator)
    output = torch.einsum("bshk,bskd->bshd", probabilities, selected)
    return output.to(query.dtype)


@dataclass(frozen=True)
class WindowAttentionTrace:
    start_pos: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    query_shape: tuple[int, ...]
    attention_kv_shape: tuple[int, ...]
    topk_shape: tuple[int, ...]
    valid_topk_min: int
    valid_topk_max: int
    weight_projection_mode: str
    nope_quant_mode: str
    sparse_accumulation_mode: str
    path: str = "torch_window_diagnostic_control"


def prepare_window_attention_weights(
    weights: ResidentAttentionWeights,
    *,
    layer_id: int,
    rank: int,
    world_size: int,
    checkpoint_id: str,
) -> PreparedWindowAttentionWeights:
    if (
        not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or layer_id not in SUPPORTED_WINDOW_LAYER_IDS
        or not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size != 4
        or not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank not in range(world_size)
    ):
        raise ValueError(
            "prepared window attention identity must be a frozen "
            "sliding-window layer on TP4"
        )
    if (
        not isinstance(checkpoint_id, str)
        or len(checkpoint_id) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_id)
    ):
        raise ValueError(
            "prepared window attention requires a lowercase SHA-256 checkpoint_id"
        )
    resident_identity = (
        weights.layer_id,
        weights.rank,
        weights.world_size,
        weights.checkpoint_id,
    )
    requested_identity = (layer_id, rank, world_size, checkpoint_id)
    resident_identity_well_typed = (
        isinstance(weights.layer_id, int)
        and not isinstance(weights.layer_id, bool)
        and isinstance(weights.rank, int)
        and not isinstance(weights.rank, bool)
        and isinstance(weights.world_size, int)
        and not isinstance(weights.world_size, bool)
        and isinstance(weights.checkpoint_id, str)
    )
    if not resident_identity_well_typed or resident_identity != requested_identity:
        raise ValueError(
            "resident window attention identity differs from requested identity: "
            f"resident={resident_identity}, requested={requested_identity}"
        )
    # Reference model.py:466-471: compress_ratio == 0 layers instantiate
    # neither compressor nor indexer.
    if weights.compressor is not None:
        raise ValueError("window attention must not contain compressor weights")
    if weights.indexer is not None:
        raise ValueError("window attention must not contain indexer weights")

    def linear(value: Any) -> torch.Tensor:
        return dequant_fp8_block(value.weight, value.scale).to(torch.bfloat16)

    return PreparedWindowAttentionWeights(
        attn_sink=weights.attn_sink.float().contiguous().clone(),
        wq_a=linear(weights.wq_a),
        q_norm=weights.q_norm.float().contiguous().clone(),
        wq_b=linear(weights.wq_b),
        wkv=linear(weights.wkv),
        kv_norm=weights.kv_norm.float().contiguous().clone(),
        wo_a=linear(weights.wo_a),
        wo_b=linear(weights.wo_b),
        layer_id=layer_id,
        rank=rank,
        world_size=world_size,
        checkpoint_id=checkpoint_id,
    )


class WindowTorchAttention:
    """Real-weight eager window-only control backed by :class:`StaticWindowKV`."""

    def __init__(
        self,
        config: WindowAttentionConfig,
        weights: PreparedWindowAttentionWeights,
        state: StaticWindowKV,
        nope_quant_mode: Literal[
            "qat_intended_e4m3", "reference_executable_bf16"
        ] = "qat_intended_e4m3",
    ) -> None:
        config.validate()
        layer_identity = (config.layer_id, weights.layer_id, state.layer_id)
        if (
            any(
                not isinstance(layer_id, int) or isinstance(layer_id, bool)
                for layer_id in layer_identity
            )
            or len(set(layer_identity)) != 1
            or config.layer_id not in SUPPORTED_WINDOW_LAYER_IDS
        ):
            raise ValueError(
                "window attention config/weights/state identity differs: "
                f"layers={layer_identity}"
            )
        if state.max_seq_len != config.max_seq_len:
            raise ValueError("attention config and static KV capacity differ")
        identity = (weights.layer_id, weights.rank, weights.world_size)
        if (
            not isinstance(weights.world_size, int)
            or isinstance(weights.world_size, bool)
            or weights.world_size != 4
            or not isinstance(weights.rank, int)
            or isinstance(weights.rank, bool)
            or weights.rank not in range(weights.world_size)
        ):
            raise ValueError(f"prepared attention identity is invalid: {identity}")
        if (
            not isinstance(weights.checkpoint_id, str)
            or len(weights.checkpoint_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in weights.checkpoint_id
            )
        ):
            raise ValueError("prepared attention checkpoint identity is invalid")
        self.config = config
        self.weights = weights
        self.state = state
        if nope_quant_mode not in (
            "qat_intended_e4m3",
            "reference_executable_bf16",
        ):
            raise ValueError(f"unsupported NoPE quant mode {nope_quant_mode}")
        self.nope_quant_mode = nope_quant_mode
        # No-YaRN table: with original_seq_len == 0 the correction branch of
        # precompute_freqs_cis is skipped (attention.py:455, mirroring the
        # reference model.py:221 guard), so this is the plain base-10000
        # table demanded by model.py:477-481 for compress_ratio == 0 layers.
        self.freqs_cis = precompute_freqs_cis(
            dim=config.rope_dim,
            seqlen=config.max_seq_len,
            original_seq_len=config.original_seq_len,
            base=config.rope_theta,
            factor=config.rope_factor,
            beta_fast=config.beta_fast,
            beta_slow=config.beta_slow,
            device=weights.wq_a.device,
        )

    def _nope_control(self, value: torch.Tensor) -> torch.Tensor:
        if self.nope_quant_mode == "reference_executable_bf16":
            return value
        return fp8_quant_dequant(value, group_size=64)

    def prepare_decode_plan(self, start_pos: int) -> WindowDecodePlan:
        """Validate state once and materialize a fixed full-ring decode plan.

        Mirrors ``Ratio128TorchAttention.prepare_decode_plan`` minus every
        compressor check: the window layer has no compressed rows and no
        pending compressor state, so only the raw-ring metadata is validated.
        """

        if (
            not isinstance(start_pos, int)
            or isinstance(start_pos, bool)
            or start_pos < WINDOW_SIZE
            or start_pos >= self.config.max_seq_len
        ):
            raise ValueError(
                "window decode plan start_pos must be an integer in "
                f"[{WINDOW_SIZE}, max_seq_len)"
            )
        if self.state.next_position != start_pos:
            raise ValueError(
                f"start_pos {start_pos} != static KV next position "
                f"{self.state.next_position}"
            )
        absolute_raw = torch.arange(
            start_pos - WINDOW_SIZE,
            start_pos,
            dtype=torch.int64,
            device=self.state.device,
        )
        raw_slots = absolute_raw.remainder(WINDOW_SIZE)
        expected_raw = absolute_raw.unsqueeze(0).expand(
            self.state.num_local_sequences, -1
        )
        if not bool(
            torch.all(
                self.state._raw_positions.index_select(1, raw_slots) == expected_raw
            ).item()
        ):
            raise RuntimeError("static window KV raw-ring metadata is inconsistent")

        # start_pos >= WINDOW_SIZE guarantees the full-ring branch of
        # window_topk_indices (all indices valid, so the maskless fixed-index
        # sparse core is legal).
        topk = window_topk_indices(
            batch_size=self.state.num_local_sequences,
            seqlen=1,
            start_pos=start_pos,
            device=self.state.device,
        )
        if bool((topk < 0).any().item()):
            raise AssertionError("full-ring window plan produced padded indices")
        gather = topk.to(torch.int64)
        batch_indices = (
            torch.arange(
                self.state.num_local_sequences,
                dtype=torch.int64,
                device=self.state.device,
            )
            .view(self.state.num_local_sequences, 1, 1)
            .expand_as(gather)
        )
        return WindowDecodePlan(
            start_pos=start_pos,
            slot=start_pos % WINDOW_SIZE,
            batch_size=self.state.num_local_sequences,
            hidden_size=self.config.hidden_size,
            owner_id=id(self),
            state_id=id(self.state),
            frequencies=self.freqs_cis[start_pos : start_pos + 1].contiguous(),
            topk_indices=topk,
            gather_indices=gather,
            batch_indices=batch_indices,
        )

    def forward_decode_tensor(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int,
        plan: WindowDecodePlan,
    ) -> torch.Tensor:
        """Run one fixed full-ring decode token without trace or host sync.

        Identical operator chain to the decode branch of ``__call__``
        (verified against the raw-FP32 window oracle in E0wf), with the plan
        supplying frequencies/top-k and the ring write going through the
        prevalidated fixed-slot path.
        """

        if not isinstance(plan, WindowDecodePlan):
            raise TypeError("plan must be a WindowDecodePlan")
        if plan.owner_id != id(self) or plan.state_id != id(self.state):
            raise ValueError("decode plan belongs to a different attention state")
        if start_pos != plan.start_pos:
            raise ValueError("decode start_pos does not match the fixed plan")
        if tuple(hidden.shape) != (plan.batch_size, 1, plan.hidden_size):
            raise ValueError("decode hidden shape does not match the fixed plan")
        if hidden.dtype != torch.bfloat16:
            raise TypeError("trace-free decode requires BF16 hidden input")
        if hidden.device != plan.frequencies.device:
            raise ValueError("decode hidden and fixed plan must share a device")

        cfg = self.config
        frequencies = plan.frequencies
        query_lora = rms_norm(
            F.linear(hidden, self.weights.wq_a),
            self.weights.q_norm,
            eps=cfg.norm_eps,
        )
        query = F.linear(query_lora, self.weights.wq_b).reshape(
            plan.batch_size, 1, cfg.num_heads, cfg.head_dim
        )
        query *= torch.rsqrt(
            query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps
        )
        query[..., -cfg.rope_dim :] = apply_rotary_emb(
            query[..., -cfg.rope_dim :], frequencies
        )

        raw_latent = rms_norm(
            F.linear(hidden, self.weights.wkv),
            self.weights.kv_norm,
            eps=cfg.norm_eps,
        )
        raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
            raw_latent[..., -cfg.rope_dim :], frequencies
        )
        raw_latent[..., : -cfg.rope_dim] = self._nope_control(
            raw_latent[..., : -cfg.rope_dim]
        )
        self.state._write_decode_fixed(
            raw_latent, position=start_pos, slot=plan.slot
        )

        output = _window_sparse_decode_prevalidated(
            query,
            self.state.latent,
            self.weights.attn_sink,
            plan,
            cfg.head_dim**-0.5,
            latent_rope=self.state.latent_rope,
        )
        output[..., -cfg.rope_dim :] = apply_rotary_emb(
            output[..., -cfg.rope_dim :], frequencies, inverse=True
        )
        grouped = output.reshape(
            plan.batch_size,
            1,
            cfg.o_groups,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        wo_a = self.weights.wo_a.reshape(
            cfg.o_groups,
            cfg.o_lora_rank,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        return F.linear(projected.flatten(2), self.weights.wo_b)

    def prepare_stateful_decode_plan(
        self,
        *,
        position: torch.Tensor,
        start_position: int,
        stop_position: int,
    ) -> WindowStatefulDecodePlan:
        """Allocate one fixed workspace for ``[start_position, stop_position)``.

        Mirrors ``Ratio128TorchAttention.prepare_stateful_decode_plan`` minus
        every compressor/bucket check: the window layer's visible KV width is
        the constant 128-row ring, so no position-dependent bucket sizing is
        needed and every index is valid at every covered position.
        """

        if (
            not isinstance(start_position, int)
            or isinstance(start_position, bool)
            or not isinstance(stop_position, int)
            or isinstance(stop_position, bool)
            or start_position < WINDOW_SIZE
            or stop_position <= start_position
            or stop_position > self.config.max_seq_len
        ):
            raise ValueError(
                "stateful window decode range must be a non-empty interval "
                "within capacity"
            )
        if (
            not isinstance(position, torch.Tensor)
            or tuple(position.shape) != (1,)
            or position.dtype != torch.int64
            or position.device != self.state.device
            or not position.is_contiguous()
        ):
            raise ValueError("stateful position must be contiguous INT64 [1]")
        if int(position.item()) != start_position:
            raise ValueError("device position does not match stateful range start")
        if self.state.next_position != start_position:
            raise ValueError("window state does not match stateful range start")

        absolute_raw = torch.arange(
            start_position - WINDOW_SIZE,
            start_position,
            dtype=torch.int64,
            device=self.state.device,
        )
        raw_slots = absolute_raw.remainder(WINDOW_SIZE)
        expected_raw = absolute_raw.unsqueeze(0).expand(
            self.state.num_local_sequences, -1
        )
        if not bool(
            torch.all(
                self.state._raw_positions.index_select(1, raw_slots) == expected_raw
            ).item()
        ):
            raise RuntimeError("static window KV raw-ring metadata is inconsistent")

        batch = self.state.num_local_sequences
        device = self.state.device
        shape = (batch, 1, WINDOW_SIZE)
        window_columns = torch.arange(
            WINDOW_SIZE, dtype=torch.int64, device=device
        )
        gather_indices = torch.zeros(shape, dtype=torch.int64, device=device)
        batch_indices = (
            torch.arange(batch, dtype=torch.int64, device=device)
            .view(batch, 1, 1)
            .expand(shape)
            .contiguous()
        )
        tensors = (position, window_columns, gather_indices, batch_indices)
        tensor_pointers = tuple(
            int(value.untyped_storage().data_ptr()) for value in tensors
        )
        if len(set(tensor_pointers)) != len(tensor_pointers):
            raise RuntimeError("stateful window workspaces must not alias")
        return WindowStatefulDecodePlan(
            start_position=start_position,
            stop_position=stop_position,
            batch_size=batch,
            hidden_size=self.config.hidden_size,
            owner_id=id(self),
            state_id=id(self.state),
            position=position,
            window_columns=window_columns,
            gather_indices=gather_indices,
            batch_indices=batch_indices,
            tensor_pointers=tensor_pointers,
        )

    def _validate_stateful_decode_plan(
        self,
        hidden: torch.Tensor,
        plan: WindowStatefulDecodePlan,
    ) -> None:
        if not isinstance(plan, WindowStatefulDecodePlan):
            raise TypeError("plan must be a WindowStatefulDecodePlan")
        if plan.owner_id != id(self) or plan.state_id != id(self.state):
            raise ValueError("stateful decode plan belongs to another attention state")
        if tuple(hidden.shape) != (plan.batch_size, 1, plan.hidden_size):
            raise ValueError("stateful hidden shape does not match its plan")
        if hidden.dtype != torch.bfloat16 or hidden.device != self.state.device:
            raise ValueError("stateful hidden must use state-local BF16 storage")
        shape = (plan.batch_size, 1, WINDOW_SIZE)
        expected = (
            ("position", plan.position, (1,), torch.int64),
            ("window_columns", plan.window_columns, (WINDOW_SIZE,), torch.int64),
            ("gather_indices", plan.gather_indices, shape, torch.int64),
            ("batch_indices", plan.batch_indices, shape, torch.int64),
        )
        pointers = []
        for name, value, expected_shape, expected_dtype in expected:
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"stateful {name} shape {tuple(value.shape)} != {expected_shape}"
                )
            if value.dtype != expected_dtype or value.device != self.state.device:
                raise ValueError(f"stateful {name} dtype/device differs")
            if not value.is_contiguous():
                raise ValueError(f"stateful {name} must be contiguous")
            pointers.append(int(value.untyped_storage().data_ptr()))
        if len(set(pointers)) != len(pointers):
            raise ValueError("stateful plan tensors must not alias")
        if tuple(pointers) != plan.tensor_pointers:
            raise ValueError("stateful plan tensor storage differs from setup")

    def forward_stateful_decode_tensor(
        self,
        hidden: torch.Tensor,
        *,
        plan: WindowStatefulDecodePlan,
        stage_marker: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        """Run one cursor-driven graph-family token without host value reads.

        Identical operator chain to ``forward_decode_tensor`` with every
        position-derived quantity computed from the shared device cursor: the
        RoPE slice via ``index_select``, the ring write slot via
        ``position % 128``, and the full-ring gather rebuilt in the eager
        chronological order ``(column + position % 128 + 1) % 128`` (the
        closed form of ``window_topk_indices``'s full-ring concatenation), so
        the gather order -- and therefore the accumulation order -- is bitwise
        identical to the fixed eager plan at the same position.
        """

        self._validate_stateful_decode_plan(hidden, plan)
        cfg = self.config
        position = plan.position
        frequencies = self.freqs_cis.index_select(0, position)

        query_lora = rms_norm(
            F.linear(hidden, self.weights.wq_a),
            self.weights.q_norm,
            eps=cfg.norm_eps,
        )
        query = F.linear(query_lora, self.weights.wq_b).reshape(
            plan.batch_size, 1, cfg.num_heads, cfg.head_dim
        )
        query *= torch.rsqrt(
            query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps
        )
        query[..., -cfg.rope_dim :] = apply_rotary_emb(
            query[..., -cfg.rope_dim :], frequencies
        )
        if stage_marker is not None:
            stage_marker("query_done")

        raw_latent = rms_norm(
            F.linear(hidden, self.weights.wkv),
            self.weights.kv_norm,
            eps=cfg.norm_eps,
        )
        raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
            raw_latent[..., -cfg.rope_dim :], frequencies
        )
        raw_latent[..., : -cfg.rope_dim] = self._nope_control(
            raw_latent[..., : -cfg.rope_dim]
        )
        if stage_marker is not None:
            stage_marker("raw_kv_done")
        self.state._write_decode_stateful_prevalidated(
            raw_latent, position=position
        )
        if stage_marker is not None:
            stage_marker("state_write_done")

        ring = position.remainder(WINDOW_SIZE)
        chronological = (plan.window_columns + ring + 1).remainder(WINDOW_SIZE)
        plan.gather_indices.copy_(
            chronological.view(1, 1, WINDOW_SIZE).expand(
                plan.batch_size, 1, WINDOW_SIZE
            )
        )
        if stage_marker is not None:
            stage_marker("index_done")
        output = _window_sparse_decode_prevalidated(
            query,
            self.state.latent,
            self.weights.attn_sink,
            plan,
            cfg.head_dim**-0.5,
            latent_rope=self.state.latent_rope,
        )
        if stage_marker is not None:
            stage_marker("sparse_done")
        output[..., -cfg.rope_dim :] = apply_rotary_emb(
            output[..., -cfg.rope_dim :], frequencies, inverse=True
        )
        grouped = output.reshape(
            plan.batch_size,
            1,
            cfg.o_groups,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        wo_a = self.weights.wo_a.reshape(
            cfg.o_groups,
            cfg.o_lora_rank,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        final_output = F.linear(projected.flatten(2), self.weights.wo_b)
        if stage_marker is not None:
            stage_marker("output_done")
        return final_output

    def __call__(
        self,
        hidden: torch.Tensor,
        *,
        start_pos: int,
        evidence: MutableMapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, WindowAttentionTrace]:
        cfg = self.config
        if hidden.ndim != 3 or hidden.shape[0] != self.state.num_local_sequences:
            raise ValueError(
                "hidden must have shape [local_batch, sequence, hidden_size]"
            )
        if hidden.shape[-1] != cfg.hidden_size or hidden.dtype != torch.bfloat16:
            raise ValueError(
                "hidden size/dtype does not match "
                f"layer-{cfg.layer_id} BF16 contract"
            )
        if start_pos != self.state.next_position:
            raise ValueError(
                f"start_pos {start_pos} != static KV next position "
                f"{self.state.next_position}"
            )
        if start_pos > 0 and hidden.shape[1] != 1:
            raise ValueError("decode attention requires one token")
        if start_pos + hidden.shape[1] > cfg.max_seq_len:
            raise ValueError("attention input exceeds static KV capacity")

        batch, seqlen, _ = hidden.shape
        frequencies = self.freqs_cis[start_pos : start_pos + seqlen]

        def record(name: str, value: torch.Tensor) -> None:
            if evidence is not None:
                evidence[name] = value.detach().clone()

        # q path: model.py:496-499 (wq_a -> q_norm -> wq_b -> weightless
        # per-head RMS -> RoPE on the trailing rope_dim lanes).
        query_lora = rms_norm(
            F.linear(hidden, self.weights.wq_a),
            self.weights.q_norm,
            eps=cfg.norm_eps,
        )
        record("query_lora", query_lora)
        query = F.linear(query_lora, self.weights.wq_b).reshape(
            batch, seqlen, cfg.num_heads, cfg.head_dim
        )
        query *= torch.rsqrt(
            query.square().mean(dim=-1, keepdim=True) + cfg.norm_eps
        )
        query[..., -cfg.rope_dim :] = apply_rotary_emb(
            query[..., -cfg.rope_dim :], frequencies
        )
        record("query", query)

        # kv path: model.py:502-506 (wkv -> kv_norm -> RoPE -> NoPE FP8 QAT
        # simulation with group size 64, same as the E0ef-verified control).
        raw_latent = rms_norm(
            F.linear(hidden, self.weights.wkv),
            self.weights.kv_norm,
            eps=cfg.norm_eps,
        )
        raw_latent[..., -cfg.rope_dim :] = apply_rotary_emb(
            raw_latent[..., -cfg.rope_dim :], frequencies
        )
        raw_latent[..., : -cfg.rope_dim] = self._nope_control(
            raw_latent[..., : -cfg.rope_dim]
        )
        record("raw_latent", raw_latent)

        # Cache write + attention KV selection.  Prefill (model.py:518-523,
        # 528): the ring keeps the last min(seqlen, window) tokens at slots
        # position % window, while attention runs over the *full* prefill
        # latent with absolute-position window indices.  Decode (model.py:530,
        # 533): write slot start_pos % window, attend over the 128-row ring
        # with ring-slot window indices.  No compressed rows exist for
        # compress_ratio == 0 (model.py:508-514 skipped), so the top-k is the
        # window part alone (model.py:507, 515).
        if start_pos == 0:
            self.state.prefill_write(raw_latent)
            # FP8 KV: attention reads what the cache write+read returns.
            attention_kv = self.state.quantize_dequantize_rows(raw_latent)
        else:
            self.state.decode_write(raw_latent)
            attention_kv = self.state.dequantized_latent()
        record("attention_kv", attention_kv)
        topk = window_topk_indices(
            batch_size=batch,
            seqlen=seqlen,
            start_pos=start_pos,
            device=hidden.device,
        )
        record("topk", topk)
        output = torch_sparse_attention(
            query,
            attention_kv,
            self.weights.attn_sink,
            topk,
            cfg.head_dim**-0.5,
        )
        record("sparse_output", output)
        # model.py:534: inverse RoPE on the attended rope lanes.
        output[..., -cfg.rope_dim :] = apply_rotary_emb(
            output[..., -cfg.rope_dim :], frequencies, inverse=True
        )
        record("inverse_rope_output", output)
        # model.py:537-542: grouped o_lora einsum then wo_b.
        grouped = output.reshape(
            batch,
            seqlen,
            cfg.o_groups,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        wo_a = self.weights.wo_a.reshape(
            cfg.o_groups,
            cfg.o_lora_rank,
            cfg.num_heads * cfg.head_dim // cfg.o_groups,
        )
        projected = torch.einsum("bsgd,grd->bsgr", grouped, wo_a)
        record("output_lora", projected)
        branch = F.linear(projected.flatten(2), self.weights.wo_b)
        record("branch", branch)
        valid = topk[topk >= 0]
        trace = WindowAttentionTrace(
            start_pos=start_pos,
            input_shape=tuple(hidden.shape),
            output_shape=tuple(branch.shape),
            query_shape=tuple(query.shape),
            attention_kv_shape=tuple(attention_kv.shape),
            topk_shape=tuple(topk.shape),
            valid_topk_min=int(valid.min().item()),
            valid_topk_max=int(valid.max().item()),
            weight_projection_mode="bf16_dequantized_weight_control",
            nope_quant_mode=self.nope_quant_mode,
            sparse_accumulation_mode="fp32_probability_value_control",
        )
        return branch, trace


__all__ = [
    "PreparedWindowAttentionWeights",
    "SUPPORTED_WINDOW_LAYER_IDS",
    "WindowAttentionConfig",
    "WindowAttentionTrace",
    "WindowDecodePlan",
    "WindowStatefulDecodePlan",
    "WindowTorchAttention",
    "prepare_window_attention_weights",
]
