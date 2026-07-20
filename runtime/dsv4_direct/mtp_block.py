"""MTP (multi-token prediction) block material and semantic forward lane.

Reference semantics (``reference/inference/model.py`` ``MTPBlock`` :738-766,
wired by ``Transformer.__init__`` :789-793):

- **Inputs**: the MTP block consumes the *pre-head* HC residual streams
  ``x [b, s, hc, d]`` of the main model at position ``p`` together with the
  **already-determined next token** ``input_ids`` (the token that is/will be
  fed to the main model at position ``p+1``); its logits predict the token
  *after* that (the draft for position ``p+2``'s input).
- **Bridge** (:757-763): ``e = enorm(embed(input_ids))`` (shared embedding,
  :792), ``x = hnorm(x)`` per-stream, ``x = e_proj(e).unsqueeze(2) +
  h_proj(x)`` -- the embedding projection broadcasts over the ``hc`` streams.
- **Block core** (:764 via ``Block.forward`` :688-700): a normal HC block with
  layer_id 43 => ``compress_ratios[43] == 0``: pure sliding-window attention
  (no compressor/indexer, no-YaRN base-10000 RoPE, model.py:477-479) and a
  learned-router MoE (43 >= num_hash_layers).
- **Head** (:765): ``logits = head(x, hc_head_fn, hc_head_scale,
  hc_head_base, norm)`` -- the **shared** main ``head.weight`` projection
  (:793) applied through the MTP block's **own** sigmoid hc_head collapse
  parameters (:750-752) and its **own** terminal RMSNorm (:746).  Logits are
  last-position fp32 (``ParallelHead.get_logits`` :716).

The reference ``generate.py`` never invokes MTP; the draft-verify decode
protocol built on this lane lives in the gate scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from .attention import rms_norm
from .block_weights import ResidentBlockWeights, load_replicated_block_weights
from .checkpoint import layer_prefix
from .head_stage import EMBED_DIM, EMBED_VOCAB, HC_MULT, hc_head_collapse_tensors
from .hyper_connections import hc_post, hc_pre
from .model_contract import MTP_LAYER_ID, SUPPORTED_LAYER_SPECS
from .moe_forward import dequant_fp8_block
from .moe_runtime import TP4MoE, TP4MoEConfig
from .ops.marlin_moe import load_resident_moe_layer
from .physical_stage import (
    EXPECTED_MOE_RESIDENT_BYTES,
    EXPECTED_TP_SIZE,
    NOPE_QUANT_MODE,
    validate_live_tp_group,
)
from .static_window_kv import StaticWindowKV
from .window_attention import (
    WindowAttentionConfig,
    WindowTorchAttention,
    prepare_window_attention_weights,
)


class MTPBlockError(ValueError):
    """Raised when the MTP block would be assembled or run off-contract."""


@dataclass
class PreparedMTPBridgeWeights:
    """BF16 dequantized-weight control view of the mtp.0-only tensors."""

    e_proj: torch.Tensor  # [4096, 4096] BF16 (FP8 block dequant)
    h_proj: torch.Tensor  # [4096, 4096] BF16 (FP8 block dequant)
    enorm: torch.Tensor  # [4096] BF16
    hnorm: torch.Tensor  # [4096] BF16
    norm: torch.Tensor  # [4096] fp32 (reference widens, model.py:189)
    hc_head_fn: torch.Tensor  # [4, 16384] fp32
    hc_head_base: torch.Tensor  # [4] fp32
    hc_head_scale: torch.Tensor  # [1] fp32


@dataclass(frozen=True, slots=True)
class MTPLayerMaterial:
    """Loaded, TP-subgroup-bound material for the MTP block (mtp.0)."""

    layer_id: int
    tp_rank: int
    tp_size: int
    checkpoint_id: str
    device: torch.device
    max_seq_len: int
    raw_block: ResidentBlockWeights
    prepared_attention: Any
    attention_config: WindowAttentionConfig
    bridge: PreparedMTPBridgeWeights
    moe: TP4MoE
    norm_eps: float
    sinkhorn_iters: int
    hc_eps: float
    hc_mult: int
    kv_dtype: str
    evidence: Mapping[str, Any]


def prepare_mtp_bridge_weights(raw_block: ResidentBlockWeights) -> PreparedMTPBridgeWeights:
    if raw_block.mtp is None:
        raise MTPBlockError("resident block does not carry mtp.0 extra tensors")
    mtp = raw_block.mtp

    def linear(value: Any) -> torch.Tensor:
        return dequant_fp8_block(value.weight, value.scale).to(torch.bfloat16)

    return PreparedMTPBridgeWeights(
        e_proj=linear(mtp.e_proj),
        h_proj=linear(mtp.h_proj),
        enorm=mtp.enorm.contiguous().clone(),
        hnorm=mtp.hnorm.contiguous().clone(),
        norm=mtp.norm.float().contiguous().clone(),
        hc_head_fn=mtp.hc_head_fn.contiguous().clone(),
        hc_head_base=mtp.hc_head_base.contiguous().clone(),
        hc_head_scale=mtp.hc_head_scale.contiguous().clone(),
    )


def build_mtp_layer_material(
    *,
    model_config: Mapping[str, Any],
    stage_root: Path,
    tp_rank: int,
    tp_group: object,
    tp_global_ranks: tuple[int, ...],
    device: torch.device,
    checkpoint_id: str,
    max_seq_len: int,
    global_row_shapes: tuple[int, ...],
    slots_per_shape: int = 1,
    progress_every: int = 64,
    progress: Callable[[str], None] | None = None,
    kv_dtype: str = "bf16",
) -> MTPLayerMaterial:
    """Load and construct the mtp.0 block on a TP subgroup.

    Mirror of ``physical_stage.build_physical_layer_material`` with the MTP
    checkpoint namespace (``mtp.0.*`` block tensors, ``mtp.0.ffn`` experts).
    """

    specification = SUPPORTED_LAYER_SPECS[MTP_LAYER_ID]
    if not bool(specification["is_mtp"]) or str(specification["attn_kind"]) != "window":
        raise MTPBlockError("frozen MTP specification is not a window block")
    collective_evidence = validate_live_tp_group(
        tp_group,
        expected_local_rank=tp_rank,
        expected_global_ranks=tp_global_ranks,
    )

    raw_block = load_replicated_block_weights(
        stage_root=stage_root,
        rank=tp_rank,
        world_size=EXPECTED_TP_SIZE,
        layer_id=MTP_LAYER_ID,
        device=device,
        checkpoint_id=checkpoint_id,
    )
    if raw_block.mtp is None:
        raise MTPBlockError("mtp.0 load produced no MTP extra tensors")
    if raw_block.gate.route_kind != "learned":
        raise MTPBlockError("mtp.0 must use the learned noaux_tc router")
    moe_resident = load_resident_moe_layer(
        stage_root=stage_root,
        layer_id=MTP_LAYER_ID,
        rank=tp_rank,
        world_size=EXPECTED_TP_SIZE,
        hidden_size=int(model_config["hidden_size"]),
        intermediate_size=int(model_config["moe_intermediate_size"]),
        n_experts=int(model_config["n_routed_experts"]),
        device=device,
        progress_every=progress_every,
        progress=progress,
        checkpoint_id=checkpoint_id,
        key_prefix=f"{layer_prefix(MTP_LAYER_ID)}.ffn",
    )
    if moe_resident.resident_bytes != EXPECTED_MOE_RESIDENT_BYTES:
        raise MTPBlockError(
            f"mtp.0 MoE resident bytes {moe_resident.resident_bytes} "
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
    )
    if moe.route_kind != "learned":
        raise MTPBlockError("constructed mtp.0 MoE is not learned-routed")

    attention_config = WindowAttentionConfig.from_model_config(
        model_config, layer_id=MTP_LAYER_ID, max_seq_len=max_seq_len
    )
    prepared_attention = prepare_window_attention_weights(
        raw_block.attention,
        layer_id=MTP_LAYER_ID,
        rank=tp_rank,
        world_size=EXPECTED_TP_SIZE,
        checkpoint_id=checkpoint_id,
    )
    bridge = prepare_mtp_bridge_weights(raw_block)

    evidence = {
        "layer_id": MTP_LAYER_ID,
        "kind": "mtp_window",
        "route_kind": "learned",
        "checkpoint_id": checkpoint_id,
        "moe_resident_bytes": int(moe_resident.resident_bytes),
        "registered_global_rows": list(global_row_shapes),
        "moe_slots_per_shape": int(slots_per_shape),
        "tp_collective": dict(collective_evidence),
        "kv_dtype": kv_dtype,
    }
    return MTPLayerMaterial(
        layer_id=MTP_LAYER_ID,
        tp_rank=tp_rank,
        tp_size=EXPECTED_TP_SIZE,
        checkpoint_id=checkpoint_id,
        device=device,
        max_seq_len=max_seq_len,
        raw_block=raw_block,
        prepared_attention=prepared_attention,
        attention_config=attention_config,
        bridge=bridge,
        moe=moe,
        norm_eps=float(model_config["rms_norm_eps"]),
        sinkhorn_iters=int(model_config["hc_sinkhorn_iters"]),
        hc_eps=float(model_config["hc_eps"]),
        hc_mult=int(model_config["hc_mult"]),
        kv_dtype=kv_dtype,
        evidence=evidence,
    )


class MTPLane:
    """One per-prompt MTP decode lane over shared MTP material.

    ``forward`` runs the full reference MTPBlock dataflow (bridge -> HC block
    core -> own hc_head/norm -> shared head) for one prefill (``start_pos ==
    0``, ``seqlen >= 1``) or one decode token (``seqlen == 1``), advancing the
    lane's private sliding-window KV state.  The MTP lane only ever ingests
    committed ``(hidden, next_token)`` pairs, so it needs no rollback.
    """

    def __init__(
        self,
        material: MTPLayerMaterial,
        *,
        embed_weight: torch.Tensor,
        head_weight: torch.Tensor,
        batch_size: int,
        device: torch.device,
        evidence_sink: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if tuple(embed_weight.shape) != (EMBED_VOCAB, EMBED_DIM) or (
            embed_weight.dtype != torch.bfloat16
        ):
            raise MTPBlockError("MTP lane requires the BF16 shared embedding table")
        if tuple(head_weight.shape) != (EMBED_VOCAB, EMBED_DIM) or (
            head_weight.dtype != torch.float32
        ):
            raise MTPBlockError("MTP lane requires the fp32 shared head projection")
        self.material = material
        self.embed_weight = embed_weight
        self.head_weight = head_weight
        self.device = torch.device(device)
        self.state = StaticWindowKV(
            num_local_sequences=batch_size,
            max_seq_len=material.max_seq_len,
            layer_id=material.layer_id,
            device=device,
            kv_dtype=material.kv_dtype,
        )
        self.attention = WindowTorchAttention(
            material.attention_config,
            material.prepared_attention,
            self.state,
            nope_quant_mode=NOPE_QUANT_MODE,
        )
        self.evidence_sink = evidence_sink

    def _record(self, name: str, value: torch.Tensor) -> None:
        if self.evidence_sink is not None:
            self.evidence_sink[name] = value.detach().clone()

    def bridge(
        self, hidden_hc: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """model.py:760-763: enorm/embed + hnorm + e_proj/h_proj sum."""

        material = self.material
        if hidden_hc.ndim != 4 or hidden_hc.shape[2:] != (HC_MULT, EMBED_DIM):
            raise MTPBlockError(
                f"MTP hidden must be [b, s, {HC_MULT}, {EMBED_DIM}]"
            )
        if hidden_hc.dtype != torch.bfloat16:
            raise MTPBlockError("MTP hidden must be BF16")
        if (
            input_ids.ndim != 2
            or input_ids.dtype != torch.int64
            or tuple(input_ids.shape) != tuple(hidden_hc.shape[:2])
        ):
            raise MTPBlockError("MTP input_ids must be int64 [b, s] matching hidden")
        embedded = F.embedding(input_ids, self.embed_weight)
        embedded = rms_norm(embedded, material.bridge.enorm, eps=material.norm_eps)
        self._record("enorm_embedded", embedded)
        normed_hidden = rms_norm(
            hidden_hc, material.bridge.hnorm, eps=material.norm_eps
        )
        self._record("hnorm_hidden", normed_hidden)
        bridged = F.linear(embedded, material.bridge.e_proj).unsqueeze(2) + F.linear(
            normed_hidden, material.bridge.h_proj
        )
        self._record("bridged", bridged)
        return bridged

    def block_core(
        self, residual: torch.Tensor, *, start_pos: int, moe_slot: int = 0
    ) -> torch.Tensor:
        """One HC block (model.py Block.forward :688-700) with window attention."""

        material = self.material
        hc = material.raw_block.hyper_connection
        hidden, post, comb = hc_pre(
            residual,
            hc.attn_fn,
            hc.attn_scale,
            hc.attn_base,
            norm_eps=material.norm_eps,
            sinkhorn_iters=material.sinkhorn_iters,
            hc_eps=material.hc_eps,
        )
        hidden = rms_norm(
            hidden, material.raw_block.attn_norm, eps=material.norm_eps
        )
        self._record("attn_hidden", hidden)
        self._record("attn_post", post)
        self._record("attn_comb", comb)
        branch, _trace = self.attention(hidden, start_pos=start_pos)
        self._record("attn_branch", branch)
        residual = hc_post(branch, residual, post, comb)
        self._record("after_attention", residual)
        hidden, post, comb = hc_pre(
            residual,
            hc.ffn_fn,
            hc.ffn_scale,
            hc.ffn_base,
            norm_eps=material.norm_eps,
            sinkhorn_iters=material.sinkhorn_iters,
            hc_eps=material.hc_eps,
        )
        hidden = rms_norm(hidden, material.raw_block.ffn_norm, eps=material.norm_eps)
        self._record("ffn_hidden", hidden)
        self._record("ffn_post", post)
        self._record("ffn_comb", comb)
        moe_output = material.moe.forward_tensor(
            hidden, input_ids_local=None, slot=moe_slot
        )
        self._record("moe_output", moe_output)
        residual = hc_post(moe_output, residual, post, comb)
        self._record("block_output", residual)
        return residual

    def head_logits(self, residual: torch.Tensor) -> torch.Tensor:
        """MTP-owned hc_head + norm, shared head; last position, fp32.

        model.py:765 -> ParallelHead.forward :718-726 with the MTP block's
        own ``hc_head_*``/``norm`` and the shared ``head.weight``.
        """

        material = self.material
        collapsed = hc_head_collapse_tensors(
            residual,
            hc_head_fn=material.bridge.hc_head_fn,
            hc_head_base=material.bridge.hc_head_base,
            hc_head_scale=material.bridge.hc_head_scale,
            norm_eps=material.norm_eps,
            hc_eps=material.hc_eps,
        )
        self._record("hc_head_collapsed", collapsed)
        value = collapsed.float()
        value = value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + material.norm_eps
        )
        normed = (material.bridge.norm * value).to(collapsed.dtype)
        self._record("final_norm", normed)
        logits = F.linear(normed[:, -1].float(), self.head_weight)
        self._record("logits", logits)
        return logits

    def forward(
        self,
        hidden_hc: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        start_pos: int,
        moe_slot: int = 0,
    ) -> torch.Tensor:
        """Full MTPBlock forward (model.py:757-766); returns [b, vocab] fp32."""

        if start_pos != self.state.next_position:
            raise MTPBlockError(
                f"MTP start_pos {start_pos} != lane position "
                f"{self.state.next_position}"
            )
        if start_pos > 0 and hidden_hc.shape[1] != 1:
            raise MTPBlockError("MTP decode steps take exactly one position")
        bridged = self.bridge(hidden_hc, input_ids)
        residual = self.block_core(bridged, start_pos=start_pos, moe_slot=moe_slot)
        return self.head_logits(residual)


__all__ = [
    "MTPBlockError",
    "MTPLane",
    "MTPLayerMaterial",
    "PreparedMTPBridgeWeights",
    "build_mtp_layer_material",
    "prepare_mtp_bridge_weights",
]
