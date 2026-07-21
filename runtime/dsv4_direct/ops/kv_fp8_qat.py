"""Fused KV-latent FP8 QAT: per-group scale + E4M3 round trip (E5F).

E2F put the decode ``raw_kv_done`` span at ~30 us/layer of non-GEMV time on
**every** layer type (ratio-4, ratio-128, window alike -- they all run the same
``rms_norm -> rope -> fp8_quant_dequant`` chain on the KV latent), against only
5.2 us of ``wkv`` GEMV bytes.  At B=1 that span is ~25 unfused eager kernels
sitting on the 4090 minimum kernel duration, so it is the same
launch-floor-bound shape E4F removed from the indexer, and it is worth more in
total because it is not confined to ratio-4 layers.

This kernel fuses the ``fp8_quant_dequant`` part of that chain.  The
``rms_norm``/rope part is left alone for now: fusing it needs the norm weight
and the rope tables and would be much harder to keep bitwise, so it is a
separate question.

**Per group, not per row.**  The decode call site quantizes
``raw_latent[..., :-rope_dim]``, which is 448 wide -- 7 groups of 64, and 7 is
not a power of two, so a ``(ROWS, GROUPS, GROUP)`` block shape does not exist
in Triton.  Flattening to ``(-1, GROUP)`` makes every Triton row exactly one
quantization group, which removes the constraint entirely and makes the kernel
independent of how many groups a row carries.

Bitwise discipline follows C4F's ``indexer_qat``: the goal is bitwise equality
with the eager chain, not closeness.

- ``clamp_min(1e-4)``, ``/448.0``, ``clamp(-448, 448)`` use the same literals,
  which are the same FP32 values on both paths.
- ``exp2(ceil(log2(amax / 448)))`` uses the libdevice functions ATen lowers to.
- The E4M3 round trip is the one step matched **by construction rather than by
  algebra** -- Triton's ``float8e4nv`` cast and ATen's
  ``.to(torch.float8_e4m3fn)`` must agree on round-to-nearest-even.  That is
  exactly why ``bitwise_selfcheck`` re-verifies it on real-range data at every
  gate instead of trusting it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


FP8_GROUP = 64
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4


@triton.jit
def _kv_fp8_qat_kernel(
    in_ptr,
    out_ptr,
    n_groups,
    # Passed in rather than read from module globals: a @triton.jit body may
    # only reach globals built as ``tl.constexpr(...)``, and the annotation
    # form is explicitly not supported.
    FP8_MAX: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    ROWS: tl.constexpr,
    GROUP: tl.constexpr,
):
    group_start = tl.program_id(0) * ROWS
    groups = group_start + tl.arange(0, ROWS)
    group_mask = groups < n_groups
    columns = tl.arange(0, GROUP)
    offsets = groups[:, None] * GROUP + columns[None, :]
    mask = group_mask[:, None]

    value = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    absolute_max = tl.max(tl.abs(value), axis=1, keep_dims=True)
    absolute_max = tl.maximum(absolute_max, AMAX_FLOOR)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(absolute_max / FP8_MAX)))
    normalized = tl.minimum(tl.maximum(value / scale, -FP8_MAX), FP8_MAX)
    # E4M3 round trip: the eager path materializes float8_e4m3fn and reads it
    # back as FP32, so the kernel must do the same cast, not an approximation.
    quantized = normalized.to(tl.float8e4nv).to(tl.float32)

    tl.store(out_ptr + offsets, (quantized * scale).to(tl.bfloat16), mask=mask)


@triton.jit
def _kv_fp8_qat_prefix_kernel(
    ptr,
    n_pairs,
    GROUPS_PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    ROWS: tl.constexpr,
    GROUP: tl.constexpr,
):
    """In-place variant over the first ``GROUPS_PER_ROW * GROUP`` of each row.

    The call sites read and write a strided prefix
    (``latent[..., :448] = qat(latent[..., :448])`` on a 512-wide row), so an
    out-of-place kernel would need a gather into contiguous memory and a
    scatter back -- two extra elementwise kernels, which is most of what this
    fusion is trying to remove in the first place.  Reading and writing the
    prefix directly keeps the whole chain at one kernel, and matches the eager
    idiom, which is also in place.
    """

    pairs = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    pair_mask = pairs < n_pairs
    rows = pairs // GROUPS_PER_ROW
    group_in_row = pairs % GROUPS_PER_ROW
    columns = tl.arange(0, GROUP)
    offsets = (
        rows[:, None] * ROW_STRIDE + group_in_row[:, None] * GROUP + columns[None, :]
    )
    mask = pair_mask[:, None]

    value = tl.load(ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    absolute_max = tl.max(tl.abs(value), axis=1, keep_dims=True)
    absolute_max = tl.maximum(absolute_max, AMAX_FLOOR)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(absolute_max / FP8_MAX)))
    normalized = tl.minimum(tl.maximum(value / scale, -FP8_MAX), FP8_MAX)
    quantized = normalized.to(tl.float8e4nv).to(tl.float32)
    tl.store(ptr + offsets, (quantized * scale).to(tl.bfloat16), mask=mask)


def fused_kv_fp8_qat_prefix_(
    tensor: torch.Tensor, width: int, *, pairs_per_block: int = 8
) -> torch.Tensor:
    """In place: ``tensor[..., :width] = fp8_quant_dequant(tensor[..., :width])``.

    ``tensor`` must be BF16, CUDA, contiguous, and ``width`` divisible by 64.
    """

    if tensor.dtype != torch.bfloat16:
        raise TypeError("fused KV FP8 QAT requires BF16 input")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("fused KV FP8 QAT prefix requires a contiguous CUDA tensor")
    if width % FP8_GROUP or width <= 0 or width > tensor.shape[-1]:
        raise ValueError(
            f"width {width} must be a positive multiple of {FP8_GROUP} "
            f"and at most {tensor.shape[-1]}"
        )
    row_stride = tensor.shape[-1]
    n_rows = tensor.numel() // row_stride
    groups_per_row = width // FP8_GROUP
    n_pairs = n_rows * groups_per_row
    grid = (triton.cdiv(n_pairs, pairs_per_block),)
    _kv_fp8_qat_prefix_kernel[grid](
        tensor,
        n_pairs,
        GROUPS_PER_ROW=groups_per_row,
        ROW_STRIDE=row_stride,
        FP8_MAX=FP8_MAX,
        AMAX_FLOOR=AMAX_FLOOR,
        ROWS=pairs_per_block,
        GROUP=FP8_GROUP,
        num_warps=4,
    )
    return tensor


def fused_kv_fp8_qat(value: torch.Tensor, *, groups_per_block: int = 8) -> torch.Tensor:
    """``fp8_quant_dequant(value, group_size=64)`` in one pass.

    ``value`` must be BF16, CUDA, and have a last dimension divisible by 64.
    """

    if value.dtype != torch.bfloat16:
        raise TypeError("fused KV FP8 QAT requires BF16 input")
    if value.shape[-1] % FP8_GROUP:
        raise ValueError(
            f"last dimension {value.shape[-1]} must be divisible by {FP8_GROUP}"
        )
    if not value.is_cuda:
        raise ValueError("fused KV FP8 QAT requires a CUDA tensor")
    source = value if value.is_contiguous() else value.contiguous()
    flat = source.reshape(-1, FP8_GROUP)
    n_groups = flat.shape[0]
    out = torch.empty_like(flat)
    grid = (triton.cdiv(n_groups, groups_per_block),)
    _kv_fp8_qat_kernel[grid](
        flat,
        out,
        n_groups,
        FP8_MAX=FP8_MAX,
        AMAX_FLOOR=AMAX_FLOOR,
        ROWS=groups_per_block,
        GROUP=FP8_GROUP,
        num_warps=4,
    )
    return out.reshape(value.shape)


def bitwise_selfcheck(
    *,
    device: torch.device,
    shapes: tuple[tuple[int, ...], ...] = ((1, 1, 448), (1, 8192, 448), (3, 17, 64)),
    seed: int = 20260728,
) -> dict:
    """Compare the fused kernel with the eager chain on real-range data.

    Real range matters: the E4M3 cast is only exercised meaningfully when the
    per-group magnitudes span several exponents, so the samples are drawn at a
    few scales rather than from one unit normal.
    """

    from ..attention import fp8_quant_dequant

    records = []
    for index, shape in enumerate(shapes):
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        for scale in (1.0, 0.02, 50.0):
            sample = (
                (torch.randn(*shape, generator=generator, dtype=torch.float32) * scale)
                .to(torch.bfloat16)
                .to(device)
            )
            reference = fp8_quant_dequant(sample, group_size=FP8_GROUP)
            candidate = fused_kv_fp8_qat(sample)
            records.append(
                {
                    "shape": list(shape),
                    "input_scale": scale,
                    "variant": "out_of_place",
                    "bitwise_equal": bool(torch.equal(reference, candidate)),
                    "max_abs_diff": float(
                        (reference.float() - candidate.float()).abs().max().item()
                    ),
                }
            )

            # prefix/in-place variant against the eager in-place idiom, on a
            # wider row so the strided prefix is actually exercised
            width = shape[-1]
            padded_shape = (*shape[:-1], width + FP8_GROUP)
            wide = torch.zeros(padded_shape, dtype=torch.bfloat16, device=device)
            wide[..., :width] = sample
            wide[..., width:] = 1.5  # tail must survive untouched
            expected = wide.clone()
            expected[..., :width] = fp8_quant_dequant(
                expected[..., :width], group_size=FP8_GROUP
            )
            actual = fused_kv_fp8_qat_prefix_(wide.clone(), width)
            records.append(
                {
                    "shape": list(padded_shape),
                    "input_scale": scale,
                    "variant": "prefix_in_place",
                    "bitwise_equal": bool(torch.equal(expected, actual)),
                    "tail_preserved": bool(
                        torch.equal(actual[..., width:], expected[..., width:])
                    ),
                    "max_abs_diff": float(
                        (expected.float() - actual.float()).abs().max().item()
                    ),
                }
            )
    return {
        "accepted": all(record["bitwise_equal"] for record in records),
        "records": records,
    }


__all__ = [
    "FP8_GROUP",
    "bitwise_selfcheck",
    "fused_kv_fp8_qat",
    "fused_kv_fp8_qat_prefix_",
]
