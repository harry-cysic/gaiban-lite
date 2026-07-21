"""Fused lightning-indexer score kernel (sm_89, Triton) for prefill.

Ported unchanged from gaiban ``experiments/D0b-prefill-roofline/indexer_fused.py``
(kernel + reference only; the reference-model monkeypatch is not carried over --
the lite runtime integrates through ``Ratio4FullPositionAttention``'s
``index_score_mode`` instead).

The reference indexer prefill score (reference/inference/model.py Indexer.forward,
mirrored by ``ratio4_fullpos.py``) computes:

    index_score = einsum("bshd,btd->bsht", q, kv)            # [b,s,h,t]
    index_score = (index_score.relu() * weights[...,None]).sum(dim=2)  # [b,s,t]

The [b,s,h,t] tensor is materialized and immediately reduced over h.  At
chunk=8192 (s=8192, h=64, t=2048) that is a multi-GB FP32 temporary whose
relu/mul/sum passes are pure DRAM traffic -- the worst-scaling O(s^2) prefill
bucket (gaiban D0b measured ~53 ms of the ratio-4 attention half at 8192 on
Pro geometry).

This kernel fuses the per-head GEMM + relu + weight-scale + h-reduction so the
h axis never touches DRAM: score[b,s,t] = sum_h relu(q[b,s,h,:] . kv[b,t,:]) * w[b,s,h].
The kv tile is shared across all heads (L2-resident), so DRAM traffic collapses
to ~q-read + out-write.  bf16 tl.dot -> fp32 accumulate; relu/weight/reduction
in fp32.  Flash indexer geometry (64 heads x 128 dim) is identical to Pro; only
topk changed (1024 -> 512), which lives outside this kernel (mask + topk stay
in torch, in the caller).

Numeric class vs the runtime reference path: the reference dots BF16 values
upcast to FP32 (exact products, FP32 accumulation in cuBLAS order); the kernel
dots BF16 on tensor cores with FP32 accumulation in tile order.  Products of
the FP4-quantized indexer values are exact in both, so any output delta is
FP32 summation-order roundoff alone -- gated per integration (topk agreement).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception as _e:  # pragma: no cover
    _HAVE_TRITON = False
    _TRITON_ERR = _e


if _HAVE_TRITON:

    @triton.jit
    def _index_score_kernel(
        q_ptr, kv_ptr, w_ptr, out_ptr,
        S, T, H,
        stride_qb, stride_qs, stride_qh, stride_qd,
        stride_kb, stride_kt, stride_kd,
        stride_wb, stride_ws, stride_wh,
        stride_ob, stride_os, stride_ot,
        BLOCK_S: tl.constexpr, BLOCK_T: tl.constexpr, D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_t = tl.program_id(2)

        s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        d_off = tl.arange(0, D)
        s_mask = s_off < S
        t_mask = t_off < T

        # kv tile loaded transposed: [D, BLOCK_T], shared across all heads.
        kv = tl.load(
            kv_ptr + pid_b * stride_kb + d_off[:, None] * stride_kd
            + t_off[None, :] * stride_kt,
            mask=t_mask[None, :], other=0.0,
        )  # bf16 [D, BLOCK_T]

        acc = tl.zeros((BLOCK_S, BLOCK_T), dtype=tl.float32)
        for h in range(0, H):
            q = tl.load(
                q_ptr + pid_b * stride_qb + s_off[:, None] * stride_qs
                + h * stride_qh + d_off[None, :] * stride_qd,
                mask=s_mask[:, None], other=0.0,
            )  # bf16 [BLOCK_S, D]
            qk = tl.dot(q, kv)                       # fp32 accumulate
            qk = tl.maximum(qk, 0.0)                 # relu
            w = tl.load(
                w_ptr + pid_b * stride_wb + s_off * stride_ws + h * stride_wh,
                mask=s_mask, other=0.0,
            ).to(tl.float32)                         # [BLOCK_S]
            acc += qk * w[:, None]

        tl.store(
            out_ptr + pid_b * stride_ob + s_off[:, None] * stride_os
            + t_off[None, :] * stride_ot,
            acc, mask=s_mask[:, None] & t_mask[None, :],
        )


def fused_index_score(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    block_s: int = 64,
    block_t: int = 128,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Fused reduced index score.

    q:       [b, s, h, d] (bf16; fp4-simulated values ok)
    kv:      [b, t, d]    (bf16)
    weights: [b, s, h]    (bf16/fp32)
    returns: [b, s, t]    (fp32) == sum_h relu(q_h @ kv^T) * w_h  (pre-mask, pre-topk)
    """

    if not _HAVE_TRITON:
        raise RuntimeError(f"triton unavailable: {_TRITON_ERR}")
    if q.ndim != 4 or kv.ndim != 3 or weights.ndim != 3:
        raise ValueError("fused index score requires q[b,s,h,d], kv[b,t,d], w[b,s,h]")
    b, s, h, d = q.shape
    t = kv.shape[1]
    if kv.shape != (b, t, d) or weights.shape != (b, s, h):
        raise ValueError(
            f"fused index score shape mismatch: q={tuple(q.shape)}, "
            f"kv={tuple(kv.shape)}, w={tuple(weights.shape)}"
        )
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("fused index score requires BF16 q/kv")
    out = torch.empty((b, s, t), device=q.device, dtype=torch.float32)
    grid = (b, triton.cdiv(s, block_s), triton.cdiv(t, block_t))
    _index_score_kernel[grid](
        q, kv, weights, out,
        s, t, h,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv.stride(0), kv.stride(1), kv.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_S=block_s, BLOCK_T=block_t, D=d,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def ref_index_score(
    q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Materialized reference: FP32 einsum + in-place relu/scale + h-reduction.

    Identical values to the runtime ratio-4 scoring chain
    (``ratio4_fullpos.Ratio4FullPositionAttention.__call__`` ref arm).
    """

    scores = torch.einsum("bshd,btd->bsht", q.float(), kv.float())
    return scores.relu_().mul_(weights.float().unsqueeze(-1)).sum(dim=2)


__all__ = ["fused_index_score", "ref_index_score"]
