"""Deterministic fixed-capacity MoE alignment for direct-runtime experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeterministicMoEAlignment:
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor


def max_padded_tokens(
    num_assignments: int, num_experts: int, block_size: int
) -> int:
    """Match the fixed-capacity allocation used by vLLM's alignment wrapper."""

    for name, value in (
        ("num_assignments", num_assignments),
        ("num_experts", num_experts),
        ("block_size", block_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    maximum = num_assignments + num_experts * (block_size - 1)
    if num_assignments < num_experts:
        maximum = min(num_assignments * block_size, maximum)
    return maximum


def allocate_deterministic_moe_alignment(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
) -> DeterministicMoEAlignment:
    _validate_metadata(topk_ids, block_size=block_size, num_experts=num_experts)
    maximum = max_padded_tokens(topk_ids.numel(), num_experts, block_size)
    return DeterministicMoEAlignment(
        sorted_token_ids=torch.empty(
            maximum, dtype=torch.int32, device=topk_ids.device
        ),
        expert_ids=torch.empty(
            (maximum + block_size - 1) // block_size,
            dtype=torch.int32,
            device=topk_ids.device,
        ),
        num_tokens_post_padded=torch.empty(
            1, dtype=torch.int32, device=topk_ids.device
        ),
    )


def deterministic_moe_align_block_size(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    output: DeterministicMoEAlignment | None = None,
) -> DeterministicMoEAlignment:
    """Build expert-major alignment with ascending flat token/top-k indices.

    The returned layout matches the public ``moe_align_block_size`` ABI:
    ``sorted_token_ids`` stores row-major indices into ``topk_ids.flatten()``;
    unused rows contain the sentinel ``topk_ids.numel()``; and every active
    block in ``expert_ids`` names one expert. Every route ID must satisfy
    ``0 <= id < num_experts``; expert maps and ignored invalid experts are
    deliberately outside this direct-runtime contract. Composite sort keys are
    unique, so the valid token order is independent of CUDA block scheduling.

    Output tensors have fixed maximum capacity. The active prefix length remains
    a device scalar and this function performs no host read of route-dependent
    data. Intermediate Torch operators still need a CUDA-graph and latency gate
    before this can be considered a production backend.
    """

    _validate_metadata(topk_ids, block_size=block_size, num_experts=num_experts)
    if output is None:
        output = allocate_deterministic_moe_alignment(
            topk_ids, block_size=block_size, num_experts=num_experts
        )
    _validate_output(
        output,
        topk_ids=topk_ids,
        block_size=block_size,
        num_experts=num_experts,
    )

    num_assignments = topk_ids.numel()
    flat_experts = topk_ids.reshape(-1).to(torch.int64)
    flat_indices = torch.arange(
        num_assignments, dtype=torch.int64, device=topk_ids.device
    )
    radix = num_assignments + 1
    composite = flat_experts * radix + flat_indices
    sorted_composite = torch.sort(composite).values
    sorted_experts = torch.div(sorted_composite, radix, rounding_mode="floor")
    sorted_flat_indices = torch.remainder(sorted_composite, radix)

    counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    counts.scatter_add_(0, flat_experts, torch.ones_like(flat_experts))
    dense_offsets = torch.cumsum(counts, dim=0) - counts
    padded_counts = torch.div(
        counts + block_size - 1, block_size, rounding_mode="floor"
    ) * block_size
    padded_ends = torch.cumsum(padded_counts, dim=0)
    padded_offsets = padded_ends - padded_counts

    ranks_within_expert = flat_indices - dense_offsets[sorted_experts]
    padded_positions = padded_offsets[sorted_experts] + ranks_within_expert
    output.sorted_token_ids.fill_(num_assignments)
    output.sorted_token_ids.scatter_(
        0, padded_positions, sorted_flat_indices.to(torch.int32)
    )

    block_starts = torch.arange(
        output.expert_ids.numel(), dtype=torch.int64, device=topk_ids.device
    ) * block_size
    block_experts = torch.searchsorted(padded_ends, block_starts, right=True)
    total_padded = padded_ends[-1]
    output.expert_ids.copy_(
        torch.where(
            block_starts < total_padded,
            block_experts,
            torch.full_like(block_experts, -1),
        ).to(torch.int32)
    )
    output.num_tokens_post_padded.copy_(total_padded.to(torch.int32).reshape(1))
    return output


def _validate_metadata(
    topk_ids: torch.Tensor, *, block_size: int, num_experts: int
) -> None:
    if not isinstance(topk_ids, torch.Tensor):
        raise TypeError("topk_ids must be a tensor")
    if topk_ids.ndim != 2 or topk_ids.numel() == 0:
        raise ValueError("topk_ids must have non-empty shape [tokens, topk]")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must use int32 or int64")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    for name, value in (("block_size", block_size), ("num_experts", num_experts)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    maximum = max_padded_tokens(topk_ids.numel(), num_experts, block_size)
    int32_max = torch.iinfo(torch.int32).max
    if (
        topk_ids.numel() > int32_max
        or num_experts - 1 > int32_max
        or maximum > int32_max
    ):
        raise ValueError("alignment metadata exceeds the int32 Marlin ABI")
    if num_experts * (topk_ids.numel() + 1) > torch.iinfo(torch.int64).max:
        raise ValueError("alignment composite key would overflow int64")


def _validate_output(
    output: DeterministicMoEAlignment,
    *,
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> None:
    if not isinstance(output, DeterministicMoEAlignment):
        raise TypeError("output must be DeterministicMoEAlignment")
    maximum = max_padded_tokens(topk_ids.numel(), num_experts, block_size)
    expected = (
        ("sorted_token_ids", output.sorted_token_ids, (maximum,)),
        (
            "expert_ids",
            output.expert_ids,
            ((maximum + block_size - 1) // block_size,),
        ),
        ("num_tokens_post_padded", output.num_tokens_post_padded, (1,)),
    )
    pointers = set()
    for name, tensor, shape in expected:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"output {name} must be a tensor")
        if tuple(tensor.shape) != shape or tensor.dtype != torch.int32:
            raise ValueError(f"output {name} shape/dtype differs from the ABI")
        if tensor.device != topk_ids.device or not tensor.is_contiguous():
            raise ValueError(f"output {name} device/layout differs from topk_ids")
        pointers.add(tensor.untyped_storage().data_ptr())
    if len(pointers) != len(expected):
        raise ValueError("alignment output tensors must not alias storage")


__all__ = [
    "DeterministicMoEAlignment",
    "allocate_deterministic_moe_alignment",
    "deterministic_moe_align_block_size",
    "max_padded_tokens",
]
