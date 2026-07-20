"""Embedding entry and norm/head exit for the full-model E2E vertical.

The lite runtime so far owned only the 43 physical blocks; this module adds
the model entry/exit surfaces, written directly against the reference
implementation (``reference/inference/model.py``):

- **Embedding** (``Transformer.forward`` :803-805 + ``ParallelEmbedding``
  :83-105): a BF16 row lookup in ``embed.weight`` ``[129280, 4096]`` followed
  by the Hyper-Connections stream expansion ``unsqueeze(2).repeat(1, 1,
  hc_mult, 1)``.  The reference shards the vocab and combines partial
  lookups with an ``all_reduce``; since exactly one rank contributes nonzero
  rows per token, a fully replicated table produces value-identical BF16
  results, so the TP4 lite runtime replicates the table on every stage-entry
  rank (no collective).
- **hc_head collapse** (``ParallelHead.hc_head`` :728-735): flatten the four
  residual streams, fp32 RMS over the flattened ``hc_mult * dim`` features,
  ``pre = sigmoid(mixes * hc_scale + hc_base) + hc_eps`` (``hc_head_scale``
  is ``[1]`` and broadcasts over the ``hc_mult`` mixes), stream-weighted fp32
  sum, cast back to BF16.  Unlike the per-block ``hc_pre`` there is **no**
  Sinkhorn split here and no post/comb state -- the head collapse is
  terminal.
- **Final norm** (``RMSNorm`` :183-196 via ``ParallelHead.forward`` :721):
  the checkpoint stores ``norm.weight`` in BF16 but the reference holds the
  parameter in fp32; the math is ``(weight_fp32 * (x_fp32 * rsqrt(mean(x^2)
  + eps))).to(bf16)``.
- **Logits** (``ParallelHead.get_logits`` :715-716): the checkpoint stores
  ``head.weight`` in BF16 but the reference loads it into an fp32 parameter
  (exact widening cast), then computes ``F.linear(x[:, -1].float(),
  weight)`` -- last position only, fp32 output.  The reference all-gathers
  vocab shards (:722-726); the lite runtime replicates the full head on
  every tail rank, which yields the identical concatenated value, so no
  collective is needed (per-rank full logits; take the tp_rank-0 lane as
  the canonical output).

MTP is deliberately out of scope (mtp-off run; the D0 oracle's
``generate.py`` never invokes the MTP block either).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import load_weight_map
from .model_contract import EXPECTED_RATIO128_CONFIG


EMBED_VOCAB = 129280
EMBED_DIM = int(EXPECTED_RATIO128_CONFIG["hidden_size"])
HC_MULT = int(EXPECTED_RATIO128_CONFIG["hc_mult"])


class HeadStageError(ValueError):
    """Raised when embed/head material would leave the frozen contract."""


@dataclass
class EmbedHeadMaterial:
    """Replicated model entry/exit tensors for one rank.

    ``embed_weight`` is present on stage-entry ranks, the head tensors on
    tail-stage ranks; either side may be ``None`` for the other role.
    """

    checkpoint_id: str
    device: torch.device
    embed_weight: torch.Tensor | None
    head_weight: torch.Tensor | None  # fp32 (reference model.py:713)
    norm_weight: torch.Tensor | None  # fp32 (reference model.py:189)
    hc_head_fn: torch.Tensor | None  # fp32 [hc_mult, hc_mult*dim]
    hc_head_base: torch.Tensor | None  # fp32 [hc_mult]
    hc_head_scale: torch.Tensor | None  # fp32 [1]
    norm_eps: float = 1e-6
    hc_eps: float = 1e-6

    @property
    def resident_bytes(self) -> int:
        return sum(
            int(value.numel() * value.element_size())
            for value in (
                self.embed_weight,
                self.head_weight,
                self.norm_weight,
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
            )
            if value is not None
        )


def load_embed_head_material(
    *,
    stage_root: Path,
    device: torch.device | str,
    checkpoint_id: str,
    load_embed: bool,
    load_head: bool,
) -> EmbedHeadMaterial:
    """Load the top-level checkpoint tensors through the index weight_map."""

    if not load_embed and not load_head:
        raise HeadStageError("embed/head material requires at least one role")
    if (
        not isinstance(checkpoint_id, str)
        or len(checkpoint_id) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_id)
    ):
        raise HeadStageError("embed/head material requires a SHA-256 checkpoint_id")
    from .ops.marlin_moe import ShardReader

    target = torch.device(device)
    root = Path(stage_root).expanduser().resolve()
    weight_map, _ = load_weight_map(root)

    def check(name: str, tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise HeadStageError(
                f"top-level tensor {name}: shape/dtype "
                f"{tuple(tensor.shape)}/{tensor.dtype} != {shape}/{dtype}"
            )
        return tensor

    embed_weight = head_weight = norm_weight = None
    hc_head_fn = hc_head_base = hc_head_scale = None
    with ShardReader(root, weight_map) as reader:
        def get(name: str) -> torch.Tensor:
            return reader.get_tensor(name).to(device=target).contiguous()

        if load_embed:
            embed_weight = check(
                "embed.weight", get("embed.weight"), (EMBED_VOCAB, EMBED_DIM), torch.bfloat16
            )
        if load_head:
            # Checkpoint stores head/norm in BF16; the reference widens both
            # to fp32 parameters (model.py:189, :713).  The cast is exact.
            head_weight = check(
                "head.weight", get("head.weight"), (EMBED_VOCAB, EMBED_DIM), torch.bfloat16
            ).float()
            norm_weight = check(
                "norm.weight", get("norm.weight"), (EMBED_DIM,), torch.bfloat16
            ).float()
            hc_head_fn = check(
                "hc_head_fn", get("hc_head_fn"), (HC_MULT, HC_MULT * EMBED_DIM), torch.float32
            )
            hc_head_base = check(
                "hc_head_base", get("hc_head_base"), (HC_MULT,), torch.float32
            )
            hc_head_scale = check(
                "hc_head_scale", get("hc_head_scale"), (1,), torch.float32
            )
    return EmbedHeadMaterial(
        checkpoint_id=checkpoint_id,
        device=target,
        embed_weight=embed_weight,
        head_weight=head_weight,
        norm_weight=norm_weight,
        hc_head_fn=hc_head_fn,
        hc_head_base=hc_head_base,
        hc_head_scale=hc_head_scale,
    )


def embed_hc_residual(
    material: EmbedHeadMaterial, input_ids: torch.Tensor
) -> torch.Tensor:
    """Token IDs -> HC residual streams (reference model.py:803-805).

    ``input_ids`` is ``[batch, sequence]`` int64; returns
    ``[batch, sequence, hc_mult, dim]`` BF16 (four copies of the embedding).
    """

    if material.embed_weight is None:
        raise HeadStageError("this rank did not load the embedding table")
    if input_ids.ndim != 2 or input_ids.dtype != torch.int64:
        raise HeadStageError("embedding input_ids must be [batch, sequence] int64")
    if int(input_ids.min()) < 0 or int(input_ids.max()) >= EMBED_VOCAB:
        raise HeadStageError("embedding input_ids outside the vocabulary")
    hidden = F.embedding(input_ids, material.embed_weight)
    return hidden.unsqueeze(2).repeat(1, 1, HC_MULT, 1).contiguous()


def hc_head_collapse(material: EmbedHeadMaterial, residual: torch.Tensor) -> torch.Tensor:
    """Collapse the four HC streams (reference ParallelHead.hc_head :728-735)."""

    if (
        material.hc_head_fn is None
        or material.hc_head_base is None
        or material.hc_head_scale is None
    ):
        raise HeadStageError("this rank did not load the hc_head parameters")
    if residual.ndim != 4 or residual.shape[2:] != (HC_MULT, EMBED_DIM):
        raise HeadStageError(
            f"hc_head residual must be [b, s, {HC_MULT}, {EMBED_DIM}]"
        )
    dtype = residual.dtype
    flattened = residual.flatten(2).float()
    inverse_rms = torch.rsqrt(
        flattened.square().mean(dim=-1, keepdim=True) + material.norm_eps
    )
    mixes = F.linear(flattened, material.hc_head_fn) * inverse_rms
    pre = (
        torch.sigmoid(mixes * material.hc_head_scale + material.hc_head_base)
        + material.hc_eps
    )
    collapsed = torch.sum(
        pre.unsqueeze(-1) * flattened.view(residual.shape), dim=2
    )
    return collapsed.to(dtype)


def final_norm(material: EmbedHeadMaterial, hidden: torch.Tensor) -> torch.Tensor:
    """Terminal RMSNorm with the fp32-held weight (reference model.py:191-196)."""

    if material.norm_weight is None:
        raise HeadStageError("this rank did not load the final norm weight")
    dtype = hidden.dtype
    value = hidden.float()
    value = value * torch.rsqrt(
        value.square().mean(dim=-1, keepdim=True) + material.norm_eps
    )
    return (material.norm_weight * value).to(dtype)


def head_logits(material: EmbedHeadMaterial, residual: torch.Tensor) -> torch.Tensor:
    """HC collapse -> final norm -> last-position fp32 logits.

    Mirrors ``ParallelHead.forward`` (reference model.py:718-726) with a
    fully replicated head: returns ``[batch, vocab]`` fp32 for the **last**
    sequence position only (:716).
    """

    if material.head_weight is None:
        raise HeadStageError("this rank did not load the head projection")
    collapsed = hc_head_collapse(material, residual)
    normed = final_norm(material, collapsed)
    return F.linear(normed[:, -1].float(), material.head_weight)


__all__ = [
    "EMBED_DIM",
    "EMBED_VOCAB",
    "HC_MULT",
    "EmbedHeadMaterial",
    "HeadStageError",
    "embed_hc_residual",
    "final_norm",
    "hc_head_collapse",
    "head_logits",
    "load_embed_head_material",
]
