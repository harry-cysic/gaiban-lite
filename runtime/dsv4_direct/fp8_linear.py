"""Direct-owned checkpoint-native block-FP8 2D GEMM primitives.

The checkpoint linear layout is kept resident as FP8 ``weight[N, K]`` plus
one power-of-two scale per 128x128 weight block.  Activations use one scale per
row and 128-wide K block.  All output and scratch storage belongs to the
caller, so the launch wrappers contain no device allocation or synchronization
and can be captured by a CUDA graph.

This module deliberately covers only ordinary 2D reference-linear projections.
The grouped ``wo_a`` projection needs a separate provider and is not flattened
into this tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the development host has no Triton
    triton = None
    tl = None


SCHEMA: Final = "dsv4.e1b2g.block_fp8_primitives.v1"
FP8_DTYPE: Final = torch.float8_e4m3fn
E8M0_DTYPE: Final = torch.float8_e8m0fnu
OUTPUT_DTYPE: Final = torch.bfloat16
WEIGHT_BLOCK_N: Final = 128
BLOCK_K: Final = 128
FP8_MAX: Final = 448.0
MIN_ACTIVATION_AMAX: Final = 1.0e-4
SUPPORTED_SPLIT_K: Final = (1, 2, 4, 8)

LAYOUT_CONTRACT: Final = {
    "activation": "row_major[M,K]",
    "activation_quantized": "row_major[M,K]",
    "activation_scale": "row_major[M,K/128]",
    "weight": "checkpoint_row_major[N,K]",
    "weight_scale": "checkpoint_row_major[ceil(N/128),K/128]",
    "output": "row_major[M,N]",
    "split_k_workspace": "row_major[split_k,M,N]",
    "workspace_dtype": "torch.float32",
    "output_dtype": "torch.bfloat16",
    "grouped_wo_a_supported": False,
}


@dataclass(frozen=True)
class QuantizationPlan:
    """Static launch and output layout for activation block quantization."""

    m: int
    k: int
    block_m: int = 16
    block_k: int = BLOCK_K

    @property
    def quantized_shape(self) -> tuple[int, int]:
        return (self.m, self.k)

    @property
    def scale_shape(self) -> tuple[int, int]:
        return (self.m, self.k // self.block_k)

    @property
    def grid(self) -> tuple[int, int]:
        return (_ceil_div(self.m, self.block_m), self.k // self.block_k)


@dataclass(frozen=True)
class GemmTuning:
    """Compile-time tile parameters shared by full-K and split-K launches."""

    block_m: int = 32
    block_n: int = 64
    num_warps: int = 4
    num_stages: int = 3


DEFAULT_GEMM_TUNING: Final = GemmTuning()


@dataclass(frozen=True)
class GemmPlan:
    """Static dispatch decision and exact caller-owned workspace contract."""

    m: int
    n: int
    k: int
    split_k: int
    tuning: GemmTuning = DEFAULT_GEMM_TUNING

    @property
    def activation_shape(self) -> tuple[int, int]:
        return (self.m, self.k)

    @property
    def activation_scale_shape(self) -> tuple[int, int]:
        return (self.m, self.k // BLOCK_K)

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.n, self.k)

    @property
    def weight_scale_shape(self) -> tuple[int, int]:
        return (_ceil_div(self.n, WEIGHT_BLOCK_N), self.k // BLOCK_K)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.m, self.n)

    @property
    def output_grid(self) -> tuple[int, int]:
        return (
            _ceil_div(self.m, self.tuning.block_m),
            _ceil_div(self.n, self.tuning.block_n),
        )

    @property
    def gemm_grid(self) -> tuple[int, ...]:
        if self.split_k == 1:
            return self.output_grid
        return (*self.output_grid, self.split_k)

    @property
    def k_blocks_per_split(self) -> int:
        return _ceil_div(self.k // BLOCK_K, self.split_k)

    @property
    def workspace_shape(self) -> tuple[int, ...]:
        if self.split_k == 1:
            return (0,)
        return (self.split_k, self.m, self.n)

    @property
    def workspace_elements(self) -> int:
        if self.split_k == 1:
            return 0
        return self.split_k * self.m * self.n

    @property
    def workspace_bytes(self) -> int:
        return self.workspace_elements * 4

    @property
    def dispatch(self) -> str:
        return "full_k" if self.split_k == 1 else "split_k_reduce"


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _positive_dimension(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def make_quantization_plan(
    m: int,
    k: int,
    *,
    block_m: int = 16,
) -> QuantizationPlan:
    """Build the fixed per-128 activation quantization launch."""

    m = _positive_dimension("m", m)
    k = _positive_dimension("k", k)
    block_m = _positive_dimension("block_m", block_m)
    if k % BLOCK_K:
        raise ValueError(f"k must be divisible by {BLOCK_K}, got {k}")
    if block_m not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"unsupported quantization block_m={block_m}")
    return QuantizationPlan(m=m, k=k, block_m=block_m)


def _validate_tuning(tuning: GemmTuning) -> None:
    if tuning.block_m not in (16, 32, 64):
        raise ValueError(f"unsupported GEMM block_m={tuning.block_m}")
    if tuning.block_n not in (32, 64, 128):
        raise ValueError(f"unsupported GEMM block_n={tuning.block_n}")
    if tuning.num_warps not in (4, 8):
        raise ValueError(f"unsupported GEMM num_warps={tuning.num_warps}")
    if tuning.num_stages not in (2, 3, 4):
        raise ValueError(f"unsupported GEMM num_stages={tuning.num_stages}")


def _automatic_split_k(m: int, n: int, k: int, tuning: GemmTuning) -> int:
    """Increase low-M launch width without splitting already-wide projections."""

    output_tiles = _ceil_div(m, tuning.block_m) * _ceil_div(n, tuning.block_n)
    k_blocks = k // BLOCK_K
    if output_tiles < 32 and k_blocks >= 32:
        return 4
    if output_tiles < 96 and k_blocks >= 16:
        return 2
    return 1


def make_gemm_plan(
    m: int,
    n: int,
    k: int,
    *,
    split_k: int | None = None,
    tuning: GemmTuning = DEFAULT_GEMM_TUNING,
) -> GemmPlan:
    """Freeze GEMM dispatch and the exact workspace shape before graph capture."""

    m = _positive_dimension("m", m)
    n = _positive_dimension("n", n)
    k = _positive_dimension("k", k)
    _validate_tuning(tuning)
    if k % BLOCK_K:
        raise ValueError(f"k must be divisible by {BLOCK_K}, got {k}")
    selected_split = (
        _automatic_split_k(m, n, k, tuning) if split_k is None else split_k
    )
    if isinstance(selected_split, bool) or selected_split not in SUPPORTED_SPLIT_K:
        raise ValueError(
            f"split_k must be one of {SUPPORTED_SPLIT_K}, got {selected_split!r}"
        )
    if selected_split > k // BLOCK_K:
        raise ValueError(
            f"split_k={selected_split} exceeds K-block count {k // BLOCK_K}"
        )
    return GemmPlan(m=m, n=n, k=k, split_k=selected_split, tuning=tuning)


def primitive_contract() -> dict[str, object]:
    """Return a JSON-friendly identity for benchmark source manifests."""

    return {
        "schema": SCHEMA,
        "block_k": BLOCK_K,
        "weight_block_n": WEIGHT_BLOCK_N,
        "fp8_dtype": str(FP8_DTYPE),
        "scale_dtypes": [str(E8M0_DTYPE)],
        "supported_split_k": list(SUPPORTED_SPLIT_K),
        "default_tuning": {
            "block_m": DEFAULT_GEMM_TUNING.block_m,
            "block_n": DEFAULT_GEMM_TUNING.block_n,
            "num_warps": DEFAULT_GEMM_TUNING.num_warps,
            "num_stages": DEFAULT_GEMM_TUNING.num_stages,
        },
        "layout": dict(LAYOUT_CONTRACT),
        "caller_preallocated": [
            "activation_quantized",
            "activation_scale",
            "output",
            "workspace",
        ],
        "cuda_graph_friendly": True,
    }


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for E1b2g block-FP8 primitives")


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtypes: tuple[torch.dtype, ...],
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype not in dtypes:
        expected = ", ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} dtype must be one of ({expected}), got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _scale_pointer(scale: torch.Tensor) -> torch.Tensor:
    # E8M0 is exponent-only.  Passing its byte view avoids relying on a Triton
    # frontend dtype mapping and preserves checkpoint storage without a copy.
    if scale.dtype != E8M0_DTYPE:
        raise TypeError(f"scale must use {E8M0_DTYPE}, got {scale.dtype}")
    return scale.view(torch.uint8)


def validate_e8m0_scale_values(name: str, scale: torch.Tensor) -> None:
    """Reject the reserved E8M0 NaN encoding once, outside captured hot paths."""

    if not isinstance(name, str) or not name:
        raise ValueError("scale preflight requires a non-empty name")
    if scale.dtype != E8M0_DTYPE:
        raise TypeError(f"{name} must use {E8M0_DTYPE}, got {scale.dtype}")
    if bool((scale.view(torch.uint8) == 255).any().item()):
        raise ValueError(f"{name} contains reserved E8M0 byte 255")


if triton is not None:

    @triton.jit
    def _load_block_scale(pointer, offsets, mask):
        value = tl.load(pointer + offsets, mask=mask, other=0).to(tl.int32)
        bits = tl.where(value == 0, 1 << 22, value << 23)
        return bits.to(tl.float32, bitcast=True)


    @triton.jit
    def _activation_quant_kernel(
        X,
        Q,
        S,
        M: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K_SIZE: tl.constexpr,
        FP8_MAX_VALUE: tl.constexpr,
        MIN_AMAX: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_k = pid_k * BLOCK_K_SIZE + tl.arange(0, BLOCK_K_SIZE)
        mask = (offsets_m[:, None] < M) & (offsets_k[None, :] < K)
        values = tl.load(
            X + offsets_m[:, None] * K + offsets_k[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        amax = tl.max(tl.abs(values), axis=1)
        unrounded_scale = tl.maximum(amax, MIN_AMAX) * (1.0 / FP8_MAX_VALUE)
        scale_bits = unrounded_scale.to(tl.int32, bitcast=True)
        biased_exponent = (scale_bits >> 23) & 0xFF
        biased_exponent += (scale_bits & 0x7FFFFF) != 0
        biased_exponent = tl.maximum(1, tl.minimum(254, biased_exponent))
        scale = (biased_exponent << 23).to(tl.float32, bitcast=True)
        quantized = tl.maximum(
            -FP8_MAX_VALUE,
            tl.minimum(FP8_MAX_VALUE, values / scale[:, None]),
        )
        tl.store(
            Q + offsets_m[:, None] * K + offsets_k[None, :],
            quantized,
            mask=mask,
        )
        scale_offsets = offsets_m * (K // BLOCK_K_SIZE) + pid_k
        scale_mask = offsets_m < M
        tl.store(
            S + scale_offsets,
            biased_exponent.to(tl.uint8),
            mask=scale_mask,
        )


    @triton.jit
    def _block_fp8_gemm_kernel(
        A,
        AS,
        B,
        BS,
        C,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_SIZE: tl.constexpr,
        WEIGHT_BLOCK_N_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K_SIZE)
        k_blocks: tl.constexpr = K // BLOCK_K_SIZE
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for kb in range(0, k_blocks):
            a = tl.load(
                A
                + offsets_m[:, None] * K
                + kb * BLOCK_K_SIZE
                + offsets_k[None, :],
                mask=offsets_m[:, None] < M,
                other=0.0,
            )
            b = tl.load(
                B
                + offsets_n[None, :] * K
                + kb * BLOCK_K_SIZE
                + offsets_k[:, None],
                mask=offsets_n[None, :] < N,
                other=0.0,
            )
            a_scale = _load_block_scale(
                AS,
                offsets_m * k_blocks + kb,
                offsets_m < M,
            )
            b_scale = _load_block_scale(
                BS,
                (offsets_n // WEIGHT_BLOCK_N_SIZE) * k_blocks + kb,
                offsets_n < N,
            )
            accumulator += (
                tl.dot(a, b, out_dtype=tl.float32)
                * a_scale[:, None]
                * b_scale[None, :]
            )

        tl.store(
            C + offsets_m[:, None] * N + offsets_n[None, :],
            accumulator.to(tl.bfloat16),
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


    @triton.jit
    def _block_fp8_gemm_split_k_kernel(
        A,
        AS,
        B,
        BS,
        PARTIAL,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
        K_BLOCKS_PER_SPLIT: tl.constexpr,
        BLOCK_K_SIZE: tl.constexpr,
        WEIGHT_BLOCK_N_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_split = tl.program_id(2)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K_SIZE)
        k_blocks: tl.constexpr = K // BLOCK_K_SIZE
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for local_kb in range(0, K_BLOCKS_PER_SPLIT):
            kb = pid_split * K_BLOCKS_PER_SPLIT + local_kb
            valid_kb = kb < k_blocks
            a = tl.load(
                A
                + offsets_m[:, None] * K
                + kb * BLOCK_K_SIZE
                + offsets_k[None, :],
                mask=(offsets_m[:, None] < M) & valid_kb,
                other=0.0,
            )
            b = tl.load(
                B
                + offsets_n[None, :] * K
                + kb * BLOCK_K_SIZE
                + offsets_k[:, None],
                mask=(offsets_n[None, :] < N) & valid_kb,
                other=0.0,
            )
            a_scale = _load_block_scale(
                AS,
                offsets_m * k_blocks + kb,
                (offsets_m < M) & valid_kb,
            )
            b_scale = _load_block_scale(
                BS,
                (offsets_n // WEIGHT_BLOCK_N_SIZE) * k_blocks + kb,
                (offsets_n < N) & valid_kb,
            )
            accumulator += (
                tl.dot(a, b, out_dtype=tl.float32)
                * a_scale[:, None]
                * b_scale[None, :]
            )

        partial_offsets = (
            pid_split * M * N + offsets_m[:, None] * N + offsets_n[None, :]
        )
        tl.store(
            PARTIAL + partial_offsets,
            accumulator,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


    @triton.jit
    def _split_k_reduce_kernel(
        PARTIAL,
        C,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for split in range(0, SPLIT_K):
            offsets = split * M * N + offsets_m[:, None] * N + offsets_n[None, :]
            accumulator += tl.load(PARTIAL + offsets, mask=mask, other=0.0)
        tl.store(
            C + offsets_m[:, None] * N + offsets_n[None, :],
            accumulator.to(tl.bfloat16),
            mask=mask,
        )


def activation_quant_into(
    x: torch.Tensor,
    quantized_out: torch.Tensor,
    scale_out: torch.Tensor,
    *,
    plan: QuantizationPlan,
) -> None:
    """Quantize ``x`` into caller-owned FP8 and power-of-two scale buffers."""

    _require_triton()
    if x.device.type != "cuda":
        raise ValueError(f"x must be a CUDA tensor, got {x.device}")
    _check_tensor(
        "x",
        x,
        shape=plan.quantized_shape,
        dtypes=(torch.bfloat16, torch.float16, torch.float32),
        device=x.device,
    )
    _check_tensor(
        "quantized_out",
        quantized_out,
        shape=plan.quantized_shape,
        dtypes=(FP8_DTYPE,),
        device=x.device,
    )
    _check_tensor(
        "scale_out",
        scale_out,
        shape=plan.scale_shape,
        dtypes=(E8M0_DTYPE,),
        device=x.device,
    )
    scale_pointer = _scale_pointer(scale_out)
    _activation_quant_kernel[plan.grid](
        x,
        quantized_out,
        scale_pointer,
        M=plan.m,
        K=plan.k,
        BLOCK_M=plan.block_m,
        BLOCK_K_SIZE=BLOCK_K,
        FP8_MAX_VALUE=FP8_MAX,
        MIN_AMAX=MIN_ACTIVATION_AMAX,
        num_warps=4,
        num_stages=1,
    )


def block_fp8_gemm_into(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    plan: GemmPlan,
) -> None:
    """Launch a full-K or deterministic split-K block-FP8 GEMM into ``out``."""

    _require_triton()
    if activation.device.type != "cuda":
        raise ValueError(
            f"activation must be a CUDA tensor, got {activation.device}"
        )
    device = activation.device
    _check_tensor(
        "activation",
        activation,
        shape=plan.activation_shape,
        dtypes=(FP8_DTYPE,),
        device=device,
    )
    _check_tensor(
        "activation_scale",
        activation_scale,
        shape=plan.activation_scale_shape,
        dtypes=(E8M0_DTYPE,),
        device=device,
    )
    _check_tensor(
        "weight",
        weight,
        shape=plan.weight_shape,
        dtypes=(FP8_DTYPE,),
        device=device,
    )
    _check_tensor(
        "weight_scale",
        weight_scale,
        shape=plan.weight_scale_shape,
        dtypes=(E8M0_DTYPE,),
        device=device,
    )
    _check_tensor(
        "out",
        out,
        shape=plan.output_shape,
        dtypes=(OUTPUT_DTYPE,),
        device=device,
    )
    _check_tensor(
        "workspace",
        workspace,
        shape=plan.workspace_shape,
        dtypes=(torch.float32,),
        device=device,
    )

    tuning = plan.tuning
    activation_scale_pointer = _scale_pointer(activation_scale)
    weight_scale_pointer = _scale_pointer(weight_scale)
    if plan.split_k == 1:
        _block_fp8_gemm_kernel[plan.gemm_grid](
            activation,
            activation_scale_pointer,
            weight,
            weight_scale_pointer,
            out,
            M=plan.m,
            N=plan.n,
            K=plan.k,
            BLOCK_M=tuning.block_m,
            BLOCK_N=tuning.block_n,
            BLOCK_K_SIZE=BLOCK_K,
            WEIGHT_BLOCK_N_SIZE=WEIGHT_BLOCK_N,
            num_warps=tuning.num_warps,
            num_stages=tuning.num_stages,
        )
        return

    _block_fp8_gemm_split_k_kernel[plan.gemm_grid](
        activation,
        activation_scale_pointer,
        weight,
        weight_scale_pointer,
        workspace,
        M=plan.m,
        N=plan.n,
        K=plan.k,
        BLOCK_M=tuning.block_m,
        BLOCK_N=tuning.block_n,
        SPLIT_K=plan.split_k,
        K_BLOCKS_PER_SPLIT=plan.k_blocks_per_split,
        BLOCK_K_SIZE=BLOCK_K,
        WEIGHT_BLOCK_N_SIZE=WEIGHT_BLOCK_N,
        num_warps=tuning.num_warps,
        num_stages=tuning.num_stages,
    )
    _split_k_reduce_kernel[plan.output_grid](
        workspace,
        out,
        M=plan.m,
        N=plan.n,
        BLOCK_M=tuning.block_m,
        BLOCK_N=tuning.block_n,
        SPLIT_K=plan.split_k,
        num_warps=tuning.num_warps,
        num_stages=1,
    )


__all__ = [
    "BLOCK_K",
    "DEFAULT_GEMM_TUNING",
    "E8M0_DTYPE",
    "FP8_DTYPE",
    "GemmPlan",
    "GemmTuning",
    "LAYOUT_CONTRACT",
    "OUTPUT_DTYPE",
    "QuantizationPlan",
    "SCHEMA",
    "SUPPORTED_SPLIT_K",
    "WEIGHT_BLOCK_N",
    "activation_quant_into",
    "block_fp8_gemm_into",
    "make_gemm_plan",
    "make_quantization_plan",
    "primitive_contract",
    "validate_e8m0_scale_values",
]
