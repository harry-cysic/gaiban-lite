"""Pure PyTorch Hyper-Connections correctness helpers.

The direct runtime keeps the checkpoint's Hyper-Connections ABI without
depending on a model or serving runtime.  For DeepSeek-V4-Pro, a layer residual
has shape ``[batch, sequence, 4, 7168]``.  ``hc_pre`` reduces the four residual
streams to one hidden state, and ``hc_post`` expands a branch output back to
four streams while mixing the previous residual streams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


HCBranch = Literal["attn", "ffn"]


@dataclass(frozen=True)
class HyperConnectionParameterNames:
    """Checkpoint keys for one layer's attention or FFN HC transform."""

    fn: str
    scale: str
    base: str


def layer_hc_parameter_names(
    layer_id: int, branch: HCBranch
) -> HyperConnectionParameterNames:
    """Return the unmodified checkpoint keys consumed by :func:`hc_pre`."""

    if not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0:
        raise ValueError("layer_id must be a non-negative integer")
    if branch not in ("attn", "ffn"):
        raise ValueError("branch must be 'attn' or 'ffn'")
    prefix = f"layers.{layer_id}.hc_{branch}"
    return HyperConnectionParameterNames(
        fn=f"{prefix}_fn",
        scale=f"{prefix}_scale",
        base=f"{prefix}_base",
    )


def _require_floating(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}")


def _validate_sinkhorn(sinkhorn_iters: int, eps: float) -> None:
    if (
        not isinstance(sinkhorn_iters, int)
        or isinstance(sinkhorn_iters, bool)
        or sinkhorn_iters < 1
    ):
        raise ValueError("sinkhorn_iters must be a positive integer")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("HC epsilon must be finite and positive")


def _validate_hyperparameters(
    *, norm_eps: float, sinkhorn_iters: int, hc_eps: float
) -> None:
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    _validate_sinkhorn(sinkhorn_iters, hc_eps)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split checkpoint-produced logits into pre, post, and residual mixes.

    ``mixes`` is ``[batch, sequence, (2 + hc_mult) * hc_mult]``.  The first
    ``hc_mult`` values produce the pre-reduction weights, the next ``hc_mult``
    values produce the branch post weights, and the remaining values produce a
    Sinkhorn-normalized ``[input_stream, output_stream]`` residual matrix.

    All outputs are float32.  The result shapes are ``[b, s, hc]``,
    ``[b, s, hc]``, and ``[b, s, hc, hc]`` respectively.
    """

    if mixes.ndim != 3:
        raise ValueError(f"mixes must have shape [b, s, mix], got {tuple(mixes.shape)}")
    if (
        not isinstance(hc_mult, int)
        or isinstance(hc_mult, bool)
        or hc_mult < 1
    ):
        raise ValueError("hc_mult must be a positive integer")
    _validate_sinkhorn(sinkhorn_iters, eps)
    _require_floating("mixes", mixes)
    _require_floating("hc_scale", hc_scale)
    _require_floating("hc_base", hc_base)

    mix_features = (2 + hc_mult) * hc_mult
    if mixes.shape[-1] != mix_features:
        raise ValueError(
            f"mixes last dimension {mixes.shape[-1]} != {mix_features} for hc_mult={hc_mult}"
        )
    if tuple(hc_scale.shape) != (3,):
        raise ValueError(f"hc_scale shape {tuple(hc_scale.shape)} != (3,)")
    if tuple(hc_base.shape) != (mix_features,):
        raise ValueError(
            f"hc_base shape {tuple(hc_base.shape)} != ({mix_features},)"
        )
    if mixes.device != hc_scale.device or mixes.device != hc_base.device:
        raise ValueError("mixes, hc_scale, and hc_base must be on the same device")

    logits = mixes.float()
    scale = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(
        logits[..., :hc_mult] * scale[0] + base[:hc_mult]
    ) + eps
    post = 2.0 * torch.sigmoid(
        logits[..., hc_mult : 2 * hc_mult] * scale[1]
        + base[hc_mult : 2 * hc_mult]
    )
    comb = (
        logits[..., 2 * hc_mult :].reshape(*logits.shape[:-1], hc_mult, hc_mult)
        * scale[2]
        + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    )

    # Match the checkpoint reference exactly: softmax supplies the first row
    # normalization, followed by a column normalization, then alternating
    # row/column normalizations for the remaining iterations.
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce HC residual streams and create the matching post-update state.

    Contract for the production checkpoint::

        residual [b, s, 4, 7168]
        hc_fn   [24, 28672]
        hc_scale [3]
        hc_base [24]
          -> hidden [b, s, 7168]       (same dtype as ``residual``)
          -> post   [b, s, 4]          (float32)
          -> comb   [b, s, 4, 4]       (float32, input HC first)

    ``hc_fn`` consumes the RMS-normalized flattened residual but the resulting
    ``pre`` weights reduce the original, unnormalized residual streams.
    """

    if residual.ndim != 4:
        raise ValueError(
            f"residual must have shape [b, s, hc, hidden], got {tuple(residual.shape)}"
        )
    _require_floating("residual", residual)
    _require_floating("hc_fn", hc_fn)
    _validate_hyperparameters(
        norm_eps=norm_eps, sinkhorn_iters=sinkhorn_iters, hc_eps=hc_eps
    )
    batch, sequence, hc_mult, hidden_size = residual.shape
    if batch < 1 or sequence < 1 or hc_mult < 1 or hidden_size < 1:
        raise ValueError("residual dimensions must all be positive")

    mix_features = (2 + hc_mult) * hc_mult
    flattened_hidden = hc_mult * hidden_size
    if tuple(hc_fn.shape) != (mix_features, flattened_hidden):
        raise ValueError(
            f"hc_fn shape {tuple(hc_fn.shape)} != "
            f"({mix_features}, {flattened_hidden})"
        )
    if residual.device != hc_fn.device:
        raise ValueError("residual and hc_fn must be on the same device")

    flattened = residual.flatten(2).float()
    inverse_rms = torch.rsqrt(
        flattened.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    mixes = F.linear(flattened, hc_fn.float()) * inverse_rms
    pre, post, comb = hc_split_sinkhorn(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=hc_eps,
    )
    hidden = torch.einsum("bsh,bshd->bsd", pre, residual.float())
    return hidden.to(residual.dtype), post, comb


def hc_post(
    branch_output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Expand one branch output and update the HC residual streams.

    ``branch_output`` is ``[b, s, hidden]`` and ``residual`` is
    ``[b, s, hc, hidden]``.  ``comb[k, j]`` moves input stream ``k`` into output
    stream ``j``; therefore the efficient contraction is ``comb.T @ residual``.
    The returned ``[b, s, hc, hidden]`` tensor has ``branch_output.dtype``.
    """

    if branch_output.ndim != 3:
        raise ValueError(
            "branch_output must have shape [b, s, hidden], got "
            f"{tuple(branch_output.shape)}"
        )
    if residual.ndim != 4:
        raise ValueError(
            f"residual must have shape [b, s, hc, hidden], got {tuple(residual.shape)}"
        )
    _require_floating("branch_output", branch_output)
    _require_floating("residual", residual)
    _require_floating("post", post)
    _require_floating("comb", comb)

    batch, sequence, hc_mult, hidden_size = residual.shape
    if tuple(branch_output.shape) != (batch, sequence, hidden_size):
        raise ValueError(
            f"branch_output shape {tuple(branch_output.shape)} != "
            f"({batch}, {sequence}, {hidden_size})"
        )
    if tuple(post.shape) != (batch, sequence, hc_mult):
        raise ValueError(
            f"post shape {tuple(post.shape)} != ({batch}, {sequence}, {hc_mult})"
        )
    if tuple(comb.shape) != (batch, sequence, hc_mult, hc_mult):
        raise ValueError(
            f"comb shape {tuple(comb.shape)} != "
            f"({batch}, {sequence}, {hc_mult}, {hc_mult})"
        )
    devices = {branch_output.device, residual.device, post.device, comb.device}
    if len(devices) != 1:
        raise ValueError("branch_output, residual, post, and comb must share a device")

    residual_mix = torch.matmul(comb.float().transpose(-1, -2), residual.float())
    branch_mix = post.float().unsqueeze(-1) * branch_output.float().unsqueeze(-2)
    return (branch_mix + residual_mix).to(branch_output.dtype)
