"""Injectable Hyper-Connections boundary backends (V4-Flash direct runtime).

A "boundary" is the op pair that sits between two half-layers:
the previous half-layer's ``hc_post`` (branch output expanded back into the
four residual streams) immediately followed by the next half-layer's
``hc_pre`` + RMSNorm (streams reduced to the next branch input).  Both the
intra-layer boundary (attention branch -> FFN branch) and the inter-layer
boundary (FFN/MoE branch -> next layer's attention branch) have this exact
shape.

Two backends implement one shared contract:

``post_pre_norm(branch_output, residual, post, comb, *, hc_fn, hc_scale,
hc_base, norm_weight, norm_eps, sinkhorn_iters, hc_eps)``
    -> ``(residual_new, hidden_norm, post_new, comb_new)``

- ``EagerHCBoundaryBackend`` composes the verified fp32
  ``hc_post`` / ``hc_pre`` / ``rms_norm`` helpers in the exact op order the
  default per-block path uses, so a chain restructured around this backend is
  **bitwise identical** to the unmodified runtime (used as the restructuring
  self-check in E0hf).
- ``FusedTilelangHCBoundaryBackend`` wraps vLLM's
  ``mhc_fused_post_pre_tilelang`` (gaiban C2g path, quantified for Flash
  decode shapes in ``experiments/A5F-hc-boundary-fusion``: 2.92x at B=512,
  post/comb <= ~1e-5, hidden/residual at bf16-1-ulp).  Two A5F/C2 findings
  are baked in:

  * ``norm_weight=None``: the installed vLLM ``with_norm`` kernel branch is
    **not numerically equivalent** for >= 128 tokens on sm_89
    (gaiban ``c2f_fused_hc.py:76-81``), so the RMSNorm stays a separate
    verified ``rms_norm`` call on the fused pre-norm activation.
  * the kernel returns ``post`` as ``[..., hc, 1]``; it is squeezed back to
    the ``[..., hc]`` reference shape so eager ``hc_post`` (the stage-tail
    boundary, which has no fusion partner) can consume it directly.

Backend selection is by construction only (block/stage constructor argument);
``hc_boundary_backend_from_env`` maps the ``DSV4_HC_BOUNDARY_BACKEND``
environment switch (``eager`` -> None, ``fused`` -> fused backend) for
callers that want an environment toggle without changing call sites.
"""

from __future__ import annotations

import os

import torch

from .attention import rms_norm
from .hyper_connections import hc_post, hc_pre


# Reference model.py post weights are 2 * sigmoid(...); the fused kernel takes
# the multiplier as an argument.
HC_POST_MULT = 2.0


class EagerHCBoundaryBackend:
    """Verified fp32 eager composition, in the default path's exact op order."""

    name = "eager"

    def post_pre_norm(
        self,
        branch_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        *,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        sinkhorn_iters: int,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_new = hc_post(branch_output, residual, post, comb)
        hidden, post_new, comb_new = hc_pre(
            residual_new,
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps=norm_eps,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=hc_eps,
        )
        hidden = rms_norm(hidden, norm_weight, eps=norm_eps)
        return residual_new, hidden, post_new, comb_new


class FusedTilelangHCBoundaryBackend:
    """vLLM TileLang fused hc_post+hc_pre boundary with a separate RMSNorm.

    Row ceiling (C2F 23rd vertical, lever A).  vLLM's
    ``_tilelang_hc_prenorm_gemm`` dispatches to a *different* kernel,
    ``hc_prenorm_gemm_block_m_tilelang``, at exactly ``num_tokens >= 1024``::

        if n_splits == 1 and use_default_config and x.shape[0] >= 1024:
            hc_prenorm_gemm_block_m_tilelang(...)

    and ``mhc_fused_post_pre_tilelang`` calls it without forwarding ``tile_n``
    or ``n_splits``, so ``use_default_config`` is always True and the branch
    cannot be avoided through the public arguments.  On this sm_89 stack that
    kernel is **numerically wrong**, not merely lower precision: measured
    against the verified eager composition on real layer-11 HC weights,
    ``rel_fro`` jumps from ~2.3e-4 at <= 896 rows to ~1.1e-1 at >= 1024 rows
    (``post`` and ``comb`` effectively decorrelate).  This is a *second*,
    higher threshold than the already-recorded ``with_norm`` >= 128 token
    problem, and it lands on every prefill chunk (1024/2048/4096/8192).

    The boundary is row-independent -- ``mhc_post``, the pre-norm GEMM (a
    per-row 16384->24 contraction plus that row's own sum of squares) and the
    per-row sigmoid/sinkhorn all touch exactly one token -- so the call is
    split into row blocks that stay on the validated branch.  Decode shapes
    (largest recorded ``local_batch`` is 192) never reach the ceiling, so they
    take the single-call path and are bitwise unchanged.
    """

    name = "fused"

    # Largest row count verified equivalent to the eager composition.
    MAX_ROWS = 896

    def __init__(self, max_rows: int | None = MAX_ROWS) -> None:
        # Import at construction so eager-only processes never touch vLLM.
        from vllm.model_executor.kernels.mhc.tilelang import (
            mhc_fused_post_pre_tilelang,
        )

        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows must be positive or None")
        self._kernel = mhc_fused_post_pre_tilelang
        self.max_rows = max_rows

    def _call_kernel(self, branch_output, residual, post, comb, packed):
        *kernel_args, kernel_kwargs = packed
        return self._kernel(
            branch_output.contiguous(),
            residual.contiguous(),
            post.contiguous(),
            comb.contiguous(),
            *kernel_args,
            **kernel_kwargs,
        )

    def _call_split(self, branch_output, residual, post, comb, packed, rows):
        """Row-blocked kernel call; outputs are written into one buffer set."""

        hc_mult = residual.shape[-2]
        hidden_size = residual.shape[-1]
        outer_shape = residual.shape[:-2]
        branch_flat = branch_output.reshape(rows, hidden_size)
        residual_flat = residual.reshape(rows, hc_mult, hidden_size)
        post_flat = post.reshape(rows, hc_mult)
        comb_flat = comb.reshape(rows, hc_mult, hc_mult)

        # Even blocks: fewer, larger launches than a fixed-stride split, with
        # no short tail.
        blocks = -(-rows // self.max_rows)
        block = -(-rows // blocks)

        residual_new = torch.empty_like(residual_flat)
        hidden = torch.empty(
            rows, hidden_size, dtype=branch_output.dtype, device=residual.device
        )
        post_new = torch.empty(
            rows, hc_mult, 1, dtype=post.dtype, device=residual.device
        )
        comb_new = torch.empty(
            rows, hc_mult, hc_mult, dtype=comb.dtype, device=residual.device
        )
        for begin in range(0, rows, block):
            end = min(begin + block, rows)
            part = self._call_kernel(
                branch_flat[begin:end],
                residual_flat[begin:end],
                post_flat[begin:end],
                comb_flat[begin:end],
                packed,
            )
            residual_new[begin:end].copy_(part[0])
            post_new[begin:end].copy_(part[1])
            comb_new[begin:end].copy_(part[2])
            hidden[begin:end].copy_(part[3])
        return (
            residual_new.view(*outer_shape, hc_mult, hidden_size),
            post_new.view(*outer_shape, hc_mult, 1),
            comb_new.view(*outer_shape, hc_mult, hc_mult),
            hidden.view(*outer_shape, hidden_size),
        )

    def post_pre_norm(
        self,
        branch_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        *,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        sinkhorn_iters: int,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if branch_output.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
            raise TypeError("fused HC boundary requires BF16 branch/residual")
        if post.dtype != torch.float32 or comb.dtype != torch.float32:
            raise TypeError("fused HC boundary requires FP32 post/comb")
        kernel_args = (
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps,  # rms_eps inside hc_pre's fn-input normalization
            hc_eps,  # hc_pre_eps
            hc_eps,  # hc_sinkhorn_eps
            HC_POST_MULT,
            sinkhorn_iters,
        )
        kernel_kwargs = dict(
            n_splits=1,
            tile_n=1,
            # C2/A5F: the installed with_norm kernel branch is not
            # numerically equivalent for >=128 tokens on sm_89; keep the
            # verified rms_norm as a separate kernel.
            norm_weight=None,
            norm_eps=norm_eps,
        )
        packed = kernel_args + (kernel_kwargs,)

        rows = 1
        for dim in branch_output.shape[:-1]:
            rows *= int(dim)
        if self.max_rows is None or rows <= self.max_rows:
            residual_new, post_new, comb_new, hidden = self._kernel(
                branch_output.contiguous(),
                residual.contiguous(),
                post.contiguous(),
                comb.contiguous(),
                *kernel_args,
                **kernel_kwargs,
            )
        else:
            residual_new, post_new, comb_new, hidden = self._call_split(
                branch_output, residual, post, comb, packed, rows
            )
        hidden = rms_norm(hidden, norm_weight, eps=norm_eps)
        return residual_new, hidden, post_new.squeeze(-1), comb_new


def resolve_hc_boundary_backend(
    name: str | None,
) -> EagerHCBoundaryBackend | FusedTilelangHCBoundaryBackend | None:
    """Map a backend name to an instance.

    ``None``/``"default"`` -> ``None`` (the unmodified per-block eager path);
    ``"eager"`` -> restructured chain with eager math (bitwise self-check);
    ``"fused"`` -> TileLang fused boundary.
    """

    if name is None or name == "default":
        return None
    if name == "eager":
        return EagerHCBoundaryBackend()
    if name == "fused":
        return FusedTilelangHCBoundaryBackend()
    if name == "fused-nosplit":
        # Diagnostic only: the raw kernel with no row ceiling, i.e. the
        # >= 1024-row `hc_prenorm_gemm_block_m_tilelang` branch that the C2F
        # 23rd vertical measured as numerically wrong.  Never ship this.
        return FusedTilelangHCBoundaryBackend(max_rows=None)
    raise ValueError(f"unknown HC boundary backend {name!r}")


def hc_boundary_backend_from_env(
    variable: str = "DSV4_HC_BOUNDARY_BACKEND",
) -> EagerHCBoundaryBackend | FusedTilelangHCBoundaryBackend | None:
    return resolve_hc_boundary_backend(os.environ.get(variable) or None)


__all__ = [
    "HC_POST_MULT",
    "EagerHCBoundaryBackend",
    "FusedTilelangHCBoundaryBackend",
    "hc_boundary_backend_from_env",
    "resolve_hc_boundary_backend",
]
