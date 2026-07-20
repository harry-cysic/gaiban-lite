"""Direct-owned checkpoint-FP8 weight-only linear primitives.

Both variants consume an unchanged BF16 activation and execute the same
BF16-by-BF16 MMA, split-K partial layout, reduction order, and BF16 output
cast.  The anchor reads a prepared BF16 weight.  The candidate instead reads
the checkpoint E4M3 weight and E8M0 block scale, reconstructs the BF16 weight
tile inside the Triton program, and never materializes a full BF16 weight.

Output and the optional split-K workspace belong to the caller.  The hot
wrappers perform no device or storage allocation and no device
synchronization, so they are suitable for CUDA graph capture.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the development host has no Triton
    triton = None
    tl = None


SCHEMA: Final = "dsv4.e1b2q.weight_only_linear.v1"
ACTIVATION_DTYPE: Final = torch.bfloat16
ANCHOR_WEIGHT_DTYPE: Final = torch.bfloat16
FP8_DTYPE: Final = torch.float8_e4m3fn
E8M0_DTYPE: Final = torch.float8_e8m0fnu
OUTPUT_DTYPE: Final = torch.bfloat16
WORKSPACE_DTYPE: Final = torch.float32
WEIGHT_BLOCK_N: Final = 128
BLOCK_K: Final = 128
SUPPORTED_SPLIT_K: Final = (1, 2, 4, 8)

ANCHOR_VARIANT: Final = "prepared_bf16_weight_bf16_mma"
CANDIDATE_VARIANT: Final = "checkpoint_fp8_e8m0_weight_bf16_mma"

LAYOUT_CONTRACT: Final = {
    "activation": "row_major_bf16[M,K]",
    "anchor_weight": "row_major_bf16[N,K]",
    "candidate_weight": "checkpoint_row_major_e4m3[N,K]",
    "candidate_weight_scale": (
        "checkpoint_row_major_e8m0[ceil(N/128),K/128]"
    ),
    "candidate_tile_weight": "ephemeral_bf16[BLOCK_K,BLOCK_N]",
    "output": "row_major_bf16[M,N]",
    "split_k_workspace": "row_major_fp32[split_k,M,N]",
    "activation_quantization": None,
    "materialized_candidate_bf16_weight": None,
    "weight_sized_scratch": None,
}


@dataclass(frozen=True)
class GemmTuning:
    """Compile-time geometry shared by anchor and weight-only candidate."""

    block_m: int = 32
    block_n: int = 64
    num_warps: int = 4
    num_stages: int = 3


DEFAULT_GEMM_TUNING: Final = GemmTuning()


@dataclass(frozen=True)
class GemmPlan:
    """Static launch and caller-owned storage contract for both variants."""

    m: int
    n: int
    k: int
    split_k: int
    tuning: GemmTuning = DEFAULT_GEMM_TUNING

    @property
    def activation_shape(self) -> tuple[int, int]:
        return (self.m, self.k)

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


def _validate_tuning(tuning: GemmTuning) -> None:
    if not isinstance(tuning, GemmTuning):
        raise TypeError("tuning must be GemmTuning")
    for name in ("block_m", "block_n", "num_warps", "num_stages"):
        value = getattr(tuning, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"GEMM {name} must be an integer, got {value!r}")
    if tuning.block_m not in (16, 32, 64):
        raise ValueError(f"unsupported GEMM block_m={tuning.block_m}")
    if tuning.block_n not in (32, 64, 128):
        raise ValueError(f"unsupported GEMM block_n={tuning.block_n}")
    if WEIGHT_BLOCK_N % tuning.block_n:
        raise ValueError(
            "GEMM block_n must divide the checkpoint weight N block"
        )
    if tuning.num_warps not in (4, 8):
        raise ValueError(f"unsupported GEMM num_warps={tuning.num_warps}")
    if tuning.num_stages not in (2, 3, 4):
        raise ValueError(f"unsupported GEMM num_stages={tuning.num_stages}")


def _automatic_split_k(m: int, n: int, k: int, tuning: GemmTuning) -> int:
    """Match the accepted ordinary-attention low-M launch-width policy."""

    output_tiles = _ceil_div(m, tuning.block_m) * _ceil_div(
        n, tuning.block_n
    )
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
    """Freeze one geometry used without change by both linear variants."""

    m = _positive_dimension("m", m)
    n = _positive_dimension("n", n)
    k = _positive_dimension("k", k)
    _validate_tuning(tuning)
    if k % BLOCK_K:
        raise ValueError(f"k must be divisible by {BLOCK_K}, got {k}")
    selected_split = (
        _automatic_split_k(m, n, k, tuning) if split_k is None else split_k
    )
    if (
        isinstance(selected_split, bool)
        or not isinstance(selected_split, int)
        or selected_split not in SUPPORTED_SPLIT_K
    ):
        raise ValueError(
            f"split_k must be one of {SUPPORTED_SPLIT_K}, "
            f"got {selected_split!r}"
        )
    if selected_split > k // BLOCK_K:
        raise ValueError(
            f"split_k={selected_split} exceeds K-block count "
            f"{k // BLOCK_K}"
        )
    return GemmPlan(
        m=m,
        n=n,
        k=k,
        split_k=selected_split,
        tuning=tuning,
    )


def launch_geometry(plan: GemmPlan) -> dict[str, object]:
    """Return the variant-independent launch identity for evidence records."""

    if not isinstance(plan, GemmPlan):
        raise TypeError("plan must be GemmPlan")
    tuning = plan.tuning
    return {
        "gemm_grid": plan.gemm_grid,
        "reduce_grid": plan.output_grid if plan.split_k != 1 else None,
        "block_m": tuning.block_m,
        "block_n": tuning.block_n,
        "block_k": BLOCK_K,
        "split_k": plan.split_k,
        "k_blocks_per_split": plan.k_blocks_per_split,
        "num_warps": tuning.num_warps,
        "num_stages": tuning.num_stages,
        "accumulator_dtype": str(torch.float32),
        "output_cast_dtype": str(OUTPUT_DTYPE),
        "workspace_shape": plan.workspace_shape,
    }


def variant_geometry_contract(plan: GemmPlan) -> dict[str, dict[str, object]]:
    """Bind both public variants to an identical geometry document."""

    geometry = launch_geometry(plan)
    return {
        ANCHOR_VARIANT: dict(geometry),
        CANDIDATE_VARIANT: dict(geometry),
    }


def primitive_contract() -> dict[str, object]:
    """Return a JSON-friendly identity for source and benchmark manifests."""

    return {
        "schema": SCHEMA,
        "variants": [ANCHOR_VARIANT, CANDIDATE_VARIANT],
        "activation_dtype": str(ACTIVATION_DTYPE),
        "anchor_weight_dtype": str(ANCHOR_WEIGHT_DTYPE),
        "candidate_weight_dtype": str(FP8_DTYPE),
        "candidate_scale_dtype": str(E8M0_DTYPE),
        "output_dtype": str(OUTPUT_DTYPE),
        "workspace_dtype": str(WORKSPACE_DTYPE),
        "block_k": BLOCK_K,
        "weight_block_n": WEIGHT_BLOCK_N,
        "supported_split_k": list(SUPPORTED_SPLIT_K),
        "default_tuning": {
            "block_m": DEFAULT_GEMM_TUNING.block_m,
            "block_n": DEFAULT_GEMM_TUNING.block_n,
            "num_warps": DEFAULT_GEMM_TUNING.num_warps,
            "num_stages": DEFAULT_GEMM_TUNING.num_stages,
        },
        "layout": dict(LAYOUT_CONTRACT),
        "same_geometry": True,
        "same_bf16_mma": True,
        "same_split_k_reduction": True,
        "activation_quantization": False,
        "candidate_dequantization": (
            "tile_local_e4m3_times_e8m0_to_bf16_before_mma"
        ),
        "candidate_materializes_bf16_weight": False,
        "caller_preallocated": ["output", "split_k_workspace"],
        "reserved_e8m0_cold_preflight_required": True,
        "reserved_e8m0_preflight_validator": (
            "validate_e8m0_scale_values"
        ),
        "hot_wrapper_assumes_prevalidated_e8m0": True,
        "non_alias_cold_preflight_required": True,
        "non_alias_preflight_validator": "validate_non_alias_tensors",
        "hot_wrapper_device_or_storage_allocation": False,
        "hot_wrapper_device_synchronization": False,
        "cuda_graph_friendly": True,
    }


def kernel_variant_contract(plan: GemmPlan) -> dict[str, object]:
    """Describe the exact JIT variants a runner must compile and warm up."""

    geometry = launch_geometry(plan)
    return {
        "schema": "dsv4.e1b2q.weight_only_linear.kernel_variants.v1",
        "gemm_kernel": (
            "_weight_only_linear_kernel"
            if plan.split_k == 1
            else "_weight_only_linear_split_k_kernel"
        ),
        "reduce_kernel": (
            None if plan.split_k == 1 else "_split_k_reduce_kernel"
        ),
        "variant_constexpr": "CHECKPOINT_FP8_WEIGHT",
        "variants": {
            ANCHOR_VARIANT: False,
            CANDIDATE_VARIANT: True,
        },
        "geometry": geometry,
        "warmup_policy": (
            "warmup_by_invoking_bf16_gemm_into_then_w8a16_gemm_into_once_per_plan_"
            "before_capture"
        ),
    }


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} dtype must be {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_common_tensors(
    activation: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    plan: GemmPlan,
) -> torch.device:
    if not isinstance(plan, GemmPlan):
        raise TypeError("plan must be GemmPlan")
    if not isinstance(activation, torch.Tensor):
        raise TypeError("activation must be a tensor")
    device = activation.device
    _check_tensor(
        "activation",
        activation,
        shape=plan.activation_shape,
        dtype=ACTIVATION_DTYPE,
        device=device,
    )
    _check_tensor(
        "out",
        out,
        shape=plan.output_shape,
        dtype=OUTPUT_DTYPE,
        device=device,
    )
    _check_tensor(
        "workspace",
        workspace,
        shape=plan.workspace_shape,
        dtype=WORKSPACE_DTYPE,
        device=device,
    )
    return device


def validate_bf16_anchor_tensors(
    activation: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    plan: GemmPlan,
) -> torch.device:
    """Validate anchor metadata without launching or synchronizing."""

    device = _validate_common_tensors(
        activation, out, workspace, plan=plan
    )
    _check_tensor(
        "weight",
        weight,
        shape=plan.weight_shape,
        dtype=ANCHOR_WEIGHT_DTYPE,
        device=device,
    )
    return device


def validate_fp8_weight_only_tensors(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    plan: GemmPlan,
) -> torch.device:
    """Validate candidate metadata without launching or synchronizing."""

    device = _validate_common_tensors(
        activation, out, workspace, plan=plan
    )
    _check_tensor(
        "weight",
        weight,
        shape=plan.weight_shape,
        dtype=FP8_DTYPE,
        device=device,
    )
    _check_tensor(
        "weight_scale",
        weight_scale,
        shape=plan.weight_scale_shape,
        dtype=E8M0_DTYPE,
        device=device,
    )
    return device


def validate_non_alias_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> None:
    """Cold preflight for one contiguous, single-device storage set.

    Empty tensors, including the split-1 workspace, have no storage span and
    cannot alias.  They still participate in the tensor, contiguity, and
    common-device checks.
    """

    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("non-alias tensor mapping must be non-empty")

    device: torch.device | None = None
    spans: list[tuple[str, int, int]] = []
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("non-alias tensor names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError(
                "non-alias tensors must share one device: "
                f"{name} is on {tensor.device}, expected {device}"
            )
        if tensor.numel() == 0:
            continue

        start = int(tensor.data_ptr())
        stop = start + int(tensor.numel() * tensor.element_size())
        for other_name, other_start, other_stop in spans:
            if start < other_stop and other_start < stop:
                raise ValueError(
                    "linear tensor storage must not alias: "
                    f"{other_name} overlaps {name}"
                )
        spans.append((name, start, stop))


def _scale_pointer(scale: torch.Tensor) -> torch.Tensor:
    # E8M0 is exponent-only.  Its byte view preserves checkpoint storage and
    # avoids relying on a Triton frontend mapping for the E8M0 torch dtype.
    if scale.dtype != E8M0_DTYPE:
        raise TypeError(f"scale must use {E8M0_DTYPE}, got {scale.dtype}")
    return scale.view(torch.uint8)


def validate_e8m0_scale_values(name: str, scale: torch.Tensor) -> None:
    """Cold preflight rejecting the reserved E8M0 NaN encoding."""

    if not isinstance(name, str) or not name:
        raise ValueError("scale preflight requires a non-empty name")
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if scale.dtype != E8M0_DTYPE:
        raise TypeError(f"{name} must use {E8M0_DTYPE}, got {scale.dtype}")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if bool((scale.view(torch.uint8) == 255).any().item()):
        raise ValueError(f"{name} contains reserved E8M0 byte 255")


def decode_e8m0_scale_for_oracle(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 bytes to FP32 for CPU tests and cold numerical oracles."""

    validate_e8m0_scale_values("scale", scale)
    encoded = scale.view(torch.uint8).to(torch.int32)
    bits = torch.where(
        encoded == 0,
        torch.full_like(encoded, 1 << 22),
        encoded << 23,
    )
    return bits.view(torch.float32)


def _require_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError(f"linear tensors must be CUDA tensors, got {device}")


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for E1b2q weight-only linear")


if triton is not None:

    @triton.jit
    def _load_e8m0_scale(pointer, offsets, mask):
        encoded = tl.load(pointer + offsets, mask=mask, other=0).to(tl.int32)
        # Byte zero represents 2^-127, an FP32 subnormal.  Other bytes map
        # directly to the biased FP32 exponent field.
        bits = tl.where(encoded == 0, 1 << 22, encoded << 23)
        return bits.to(tl.float32, bitcast=True)


    @triton.jit
    def _weight_only_linear_kernel(
        A,
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
        CHECKPOINT_FP8_WEIGHT: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K_SIZE)
        k_blocks: tl.constexpr = K // BLOCK_K_SIZE
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for kb in range(0, k_blocks):
            activation = tl.load(
                A
                + offsets_m[:, None] * K
                + kb * BLOCK_K_SIZE
                + offsets_k[None, :],
                mask=offsets_m[:, None] < M,
                other=0.0,
            ).to(tl.bfloat16)
            weight = tl.load(
                B
                + offsets_n[None, :] * K
                + kb * BLOCK_K_SIZE
                + offsets_k[:, None],
                mask=offsets_n[None, :] < N,
                other=0.0,
            )
            if CHECKPOINT_FP8_WEIGHT:
                weight_scale = _load_e8m0_scale(
                    BS,
                    (offsets_n // WEIGHT_BLOCK_N_SIZE) * k_blocks + kb,
                    offsets_n < N,
                )
                weight = (
                    weight.to(tl.float32) * weight_scale[None, :]
                ).to(tl.bfloat16)
            else:
                weight = weight.to(tl.bfloat16)
            accumulator += tl.dot(
                activation,
                weight,
                out_dtype=tl.float32,
            )

        tl.store(
            C + offsets_m[:, None] * N + offsets_n[None, :],
            accumulator.to(tl.bfloat16),
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


    @triton.jit
    def _weight_only_linear_split_k_kernel(
        A,
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
        CHECKPOINT_FP8_WEIGHT: tl.constexpr,
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
            activation = tl.load(
                A
                + offsets_m[:, None] * K
                + kb * BLOCK_K_SIZE
                + offsets_k[None, :],
                mask=(offsets_m[:, None] < M) & valid_kb,
                other=0.0,
            ).to(tl.bfloat16)
            weight = tl.load(
                B
                + offsets_n[None, :] * K
                + kb * BLOCK_K_SIZE
                + offsets_k[:, None],
                mask=(offsets_n[None, :] < N) & valid_kb,
                other=0.0,
            )
            if CHECKPOINT_FP8_WEIGHT:
                weight_scale = _load_e8m0_scale(
                    BS,
                    (offsets_n // WEIGHT_BLOCK_N_SIZE) * k_blocks + kb,
                    (offsets_n < N) & valid_kb,
                )
                weight = (
                    weight.to(tl.float32) * weight_scale[None, :]
                ).to(tl.bfloat16)
            else:
                weight = weight.to(tl.bfloat16)
            accumulator += tl.dot(
                activation,
                weight,
                out_dtype=tl.float32,
            )

        partial_offsets = (
            pid_split * M * N
            + offsets_m[:, None] * N
            + offsets_n[None, :]
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
            offsets = (
                split * M * N
                + offsets_m[:, None] * N
                + offsets_n[None, :]
            )
            accumulator += tl.load(PARTIAL + offsets, mask=mask, other=0.0)
        tl.store(
            C + offsets_m[:, None] * N + offsets_n[None, :],
            accumulator.to(tl.bfloat16),
            mask=mask,
        )


def _launch_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_pointer: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    plan: GemmPlan,
    checkpoint_fp8_weight: bool,
) -> None:
    """Shared device/storage-allocation-free path for both variants."""

    tuning = plan.tuning
    if plan.split_k == 1:
        _weight_only_linear_kernel[plan.gemm_grid](
            activation,
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
            CHECKPOINT_FP8_WEIGHT=checkpoint_fp8_weight,
            num_warps=tuning.num_warps,
            num_stages=tuning.num_stages,
        )
        return

    _weight_only_linear_split_k_kernel[plan.gemm_grid](
        activation,
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
        CHECKPOINT_FP8_WEIGHT=checkpoint_fp8_weight,
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


def bf16_gemm_into(
    activation: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    plan: GemmPlan,
) -> None:
    """Run the prepared-BF16 anchor into caller-owned storage."""

    device = validate_bf16_anchor_tensors(
        activation, weight, out, workspace, plan=plan
    )
    _require_cuda_device(device)
    _require_triton()
    _launch_linear(
        activation,
        weight,
        weight,
        out,
        workspace,
        plan=plan,
        checkpoint_fp8_weight=False,
    )


def w8a16_gemm_into(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    plan: GemmPlan,
) -> None:
    """Run checkpoint-FP8 weight-only BF16 MMA into caller-owned storage."""

    device = validate_fp8_weight_only_tensors(
        activation,
        weight,
        weight_scale,
        out,
        workspace,
        plan=plan,
    )
    _require_cuda_device(device)
    _require_triton()
    _launch_linear(
        activation,
        weight,
        _scale_pointer(weight_scale),
        out,
        workspace,
        plan=plan,
        checkpoint_fp8_weight=True,
    )


# Descriptive aliases retained for callers that prefer semantic names over the
# compact benchmark ABI.  They reference the same hot wrapper objects.
bf16_anchor_linear_into = bf16_gemm_into
fp8_weight_only_linear_into = w8a16_gemm_into


__all__ = [
    "ACTIVATION_DTYPE",
    "ANCHOR_VARIANT",
    "ANCHOR_WEIGHT_DTYPE",
    "BLOCK_K",
    "CANDIDATE_VARIANT",
    "DEFAULT_GEMM_TUNING",
    "E8M0_DTYPE",
    "FP8_DTYPE",
    "GemmPlan",
    "GemmTuning",
    "LAYOUT_CONTRACT",
    "OUTPUT_DTYPE",
    "SCHEMA",
    "SUPPORTED_SPLIT_K",
    "WEIGHT_BLOCK_N",
    "WORKSPACE_DTYPE",
    "bf16_anchor_linear_into",
    "bf16_gemm_into",
    "decode_e8m0_scale_for_oracle",
    "fp8_weight_only_linear_into",
    "kernel_variant_contract",
    "launch_geometry",
    "make_gemm_plan",
    "primitive_contract",
    "validate_bf16_anchor_tensors",
    "validate_e8m0_scale_values",
    "validate_fp8_weight_only_tensors",
    "validate_non_alias_tensors",
    "variant_geometry_contract",
    "w8a16_gemm_into",
]
