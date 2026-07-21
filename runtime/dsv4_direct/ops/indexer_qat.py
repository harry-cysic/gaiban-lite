"""Fused indexer-query QAT control: Hadamard + FP4 quant/dequant (C4F).

Twenty-seventh vertical.  The ratio-4 prefill phase profile
(``experiments/C4F-ratio4-attention/``) put ``fp4_quant_dequant(
hadamard_transform(index_query))`` at **29.75 ms of a 72.07 ms layer (41.3%)**
-- larger than the tilelang sparse core (15.78 ms).  Neither half is a GEMM:
every step materializes another 268 MB FP32 temporary over the
``[1, 8192, 64, 128]`` indexer query.  At the 816 GB/s this kernel goes on to
measure, 29.75 ms is ~24 GB of round trips against 0.268 GB of actual
information (134 MB BF16 in, 134 MB BF16 out) -- ~90x redundant traffic, which
is exactly the 90.5x the fusion recovers.  The whole chain is one row-local
function of 128 contiguous values, so it fuses into a single load/store.

The kernel is written to be **bitwise identical** to the eager pair, not
merely close:

- The Hadamard reshape/cat form in ``attention.hadamard_transform`` is exactly
  the standard in-place FWHT butterfly.  Writing ``i = a*2s + b*s + c`` for
  the ``(width//(2s), 2, s)`` view, ``cat((left+right, left-right), -1)``
  lands ``new[a*2s + c] = t[a*2s + c] + t[a*2s + s + c]`` and
  ``new[a*2s + s + c] = t[a*2s + c] - t[a*2s + s + c]``.  The kernel performs
  the same seven FP32 butterfly stages in the same pairing order, so every
  output is produced by an identical sequence of FP32 adds.
- The eager chain rounds to BF16 between the two halves
  (``hadamard_transform`` returns ``value.dtype``, ``fp4_quant_dequant``
  re-widens with ``.float()``).  The kernel keeps that round trip -- dropping
  it would be a real numeric change.
- ``copysign`` is done on the sign bit rather than with a comparison, so a
  ``-0.0`` normalized value yields ``-0.0`` exactly as ``torch.copysign``
  does.
- ``exp2(ceil(log2(amax / 6)))`` uses the libdevice functions that ATen's
  eager ops lower to.  This is the one step that is matched by construction
  rather than by algebra, so ``bitwise_selfcheck`` re-verifies it against the
  eager chain on real tensors at every gate.

Only the ``[..., 128]`` indexer-query width with ``group_size=32`` is
supported; anything else must keep the eager path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


HADAMARD_WIDTH = 128
FP4_GROUP = 32
_STAGES = (1, 2, 4, 8, 16, 32, 64)


@triton.jit
def _butterfly(
    value,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    SPAN: tl.constexpr,
):
    """One FWHT stage: ``new[i] = t[i] +/- t[i | SPAN]`` in reference order."""

    blocks: tl.constexpr = WIDTH // (2 * SPAN)
    viewed = tl.permute(tl.reshape(value, (ROWS, blocks, 2, SPAN)), (0, 1, 3, 2))
    left, right = tl.split(viewed)
    joined = tl.permute(tl.join(left + right, left - right), (0, 1, 3, 2))
    return tl.reshape(joined, (ROWS, WIDTH))


@triton.jit
def _hadamard_fp4_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    hadamard_scale,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS
    rows = row_start + tl.arange(0, ROWS)
    row_mask = rows < n_rows
    columns = tl.arange(0, WIDTH)
    offsets = rows[:, None] * WIDTH + columns[None, :]
    mask = row_mask[:, None]

    value = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # ---- Hadamard: seven FP32 butterfly stages, reference pairing order ----
    value = _butterfly(value, ROWS, WIDTH, 1)
    value = _butterfly(value, ROWS, WIDTH, 2)
    value = _butterfly(value, ROWS, WIDTH, 4)
    value = _butterfly(value, ROWS, WIDTH, 8)
    value = _butterfly(value, ROWS, WIDTH, 16)
    value = _butterfly(value, ROWS, WIDTH, 32)
    value = _butterfly(value, ROWS, WIDTH, 64)

    value = value * hadamard_scale
    # BF16 round trip between the two eager ops (semantically load-bearing).
    value = value.to(tl.bfloat16).to(tl.float32)

    # ---- FP4 E2M1 fake quantization, per group of 32 ----
    grouped = tl.reshape(value, (ROWS, GROUPS, GROUP))
    absolute_max = tl.max(tl.abs(grouped), axis=2, keep_dims=True)
    # clamp_min(6.0 * 2.0**-126); 1.5 * 2**-124 is exact in FP32
    absolute_max = tl.maximum(absolute_max, 7.052966104933725e-38)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(absolute_max / 6.0)))
    normalized = tl.minimum(tl.maximum(grouped / scale, -6.0), 6.0)
    magnitude = tl.abs(normalized)
    snapped = tl.where(
        magnitude <= 0.25,
        0.0,
        tl.where(
            magnitude < 0.75,
            0.5,
            tl.where(
                magnitude <= 1.25,
                1.0,
                tl.where(
                    magnitude < 1.75,
                    1.5,
                    tl.where(
                        magnitude <= 2.5,
                        2.0,
                        tl.where(
                            magnitude < 3.5,
                            3.0,
                            tl.where(magnitude <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    # copysign on the sign bit: -0.0 must survive, which a comparison loses.
    sign = normalized.to(tl.int32, bitcast=True) & -2147483648
    signed = (snapped.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    result = tl.reshape(signed * scale, (ROWS, WIDTH))

    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=mask)


def fused_hadamard_fp4(value: torch.Tensor, *, rows_per_block: int = 8) -> torch.Tensor:
    """``fp4_quant_dequant(hadamard_transform(value))`` in one pass.

    ``value`` must be BF16, CUDA, contiguous, and end in a 128-wide axis.
    """

    if value.dtype != torch.bfloat16:
        raise TypeError("fused indexer QAT requires BF16 input")
    if value.shape[-1] != HADAMARD_WIDTH:
        raise ValueError(
            f"fused indexer QAT is specialized to width {HADAMARD_WIDTH}, "
            f"got {value.shape[-1]}"
        )
    if not value.is_cuda:
        raise ValueError("fused indexer QAT requires a CUDA tensor")
    source = value if value.is_contiguous() else value.contiguous()
    flat = source.reshape(-1, HADAMARD_WIDTH)
    n_rows = flat.shape[0]
    out = torch.empty_like(flat)
    grid = (triton.cdiv(n_rows, rows_per_block),)
    _hadamard_fp4_kernel[grid](
        flat,
        out,
        n_rows,
        HADAMARD_WIDTH**-0.5,
        ROWS=rows_per_block,
        WIDTH=HADAMARD_WIDTH,
        GROUPS=HADAMARD_WIDTH // FP4_GROUP,
        GROUP=FP4_GROUP,
        num_warps=4,
    )
    return out.reshape(value.shape)


def bitwise_selfcheck(
    *,
    device: torch.device,
    shapes: tuple[tuple[int, ...], ...] = ((1, 1024, 64, 128), (1, 97, 3, 128)),
    seed: int = 20260727,
) -> dict:
    """Compare the fused kernel with the eager pair on random real-range data."""

    from ..ratio4_attention import fp4_quant_dequant, hadamard_transform

    records = []
    for shape in shapes:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        sample = torch.randn(
            *shape, dtype=torch.float32, device=device, generator=generator
        ).to(torch.bfloat16)
        reference = fp4_quant_dequant(hadamard_transform(sample))
        candidate = fused_hadamard_fp4(sample)
        equal = bool(torch.equal(reference, candidate))
        difference = (candidate.float() - reference.float()).abs()
        records.append(
            {
                "shape": list(shape),
                "bitwise_equal": equal,
                "mismatched_elements": int(
                    (candidate != reference).sum().item()
                ),
                "elements": int(reference.numel()),
                "max_abs_diff": float(difference.max().item()),
                "reference_abs_max": float(reference.abs().max().item()),
            }
        )
    return {
        "records": records,
        "bitwise_equal": all(record["bitwise_equal"] for record in records),
    }


__all__ = ["FP4_GROUP", "HADAMARD_WIDTH", "bitwise_selfcheck", "fused_hadamard_fp4"]
