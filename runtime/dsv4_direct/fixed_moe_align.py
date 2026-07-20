"""Direct-owned exact MoE alignment for the fixed B240 decode geometry.

The hot wrapper in this module is intentionally narrow: ``topk_ids`` has shape
``[240, 6]``, there are 256 experts (Flash: n_routed_experts 256, Pro used
384; the only geometry constant this module hard-codes), and Marlin consumes
blocks of 8 routes.
All output and scratch storage is caller-owned.  Once Triton has been warmed up,
the four launches are CUDA-graph capturable and perform no route-dependent host
read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .deterministic_moe_align import DeterministicMoEAlignment

try:
    import triton
    import triton.language as tl
except ImportError:  # Development and CPU-oracle hosts do not need Triton.
    triton = None
    tl = None


FIXED_ROWS = 240
FIXED_TOPK = 6
FIXED_NUM_ASSIGNMENTS = FIXED_ROWS * FIXED_TOPK
FIXED_NUM_EXPERTS = 256
FIXED_BLOCK_SIZE = 8
FIXED_SENTINEL = FIXED_NUM_ASSIGNMENTS
FIXED_MAX_PADDED_TOKENS = (
    FIXED_NUM_ASSIGNMENTS + FIXED_NUM_EXPERTS * (FIXED_BLOCK_SIZE - 1)
)
FIXED_MAX_BLOCKS = FIXED_MAX_PADDED_TOKENS // FIXED_BLOCK_SIZE
WORKSPACE_CAPACITY = 512

_INITIALIZE_BLOCK = 256
_SCAN_BLOCK = 256
_MAX_BLOCKS_PER_EXPERT = 256


@dataclass(frozen=True)
class FixedB240AlignmentPlan:
    route_shape: tuple[int, int] = (FIXED_ROWS, FIXED_TOPK)
    num_assignments: int = FIXED_NUM_ASSIGNMENTS
    num_experts: int = FIXED_NUM_EXPERTS
    block_size: int = FIXED_BLOCK_SIZE
    sentinel: int = FIXED_SENTINEL
    max_padded_tokens: int = FIXED_MAX_PADDED_TOKENS
    max_blocks: int = FIXED_MAX_BLOCKS
    workspace_capacity: int = WORKSPACE_CAPACITY
    launch_count: int = 4


FIXED_B240_ALIGNMENT_PLAN = FixedB240AlignmentPlan()


@dataclass(frozen=True)
class FixedB240AlignmentWorkspace:
    """Caller-owned scratch; the first 385 values become padded offsets/total."""

    counts_and_offsets: torch.Tensor


class FixedB240MoEAlignmentProvider:
    """Own one fixed scratch workspace per physical MoE graph slot."""

    def __init__(
        self,
        *,
        device: torch.device,
        global_row_shapes: Iterable[int],
        slots_per_shape: int,
        route_kind: str,
    ) -> None:
        device = torch.device(device)
        shapes = tuple(global_row_shapes)
        if shapes != (FIXED_ROWS,):
            raise ValueError("fixed B240 alignment requires global_row_shapes=(240,)")
        if (
            not isinstance(slots_per_shape, int)
            or isinstance(slots_per_shape, bool)
            or slots_per_shape < 1
        ):
            raise ValueError("fixed B240 alignment slots_per_shape must be positive")
        if route_kind != "learned":
            raise ValueError("fixed B240 alignment is only valid for learned routing")
        self.device = device
        self.global_row_shapes = shapes
        self.slots_per_shape = slots_per_shape
        self.route_kind = route_kind
        self._workspaces = {
            (FIXED_ROWS, slot): FixedB240AlignmentWorkspace(
                counts_and_offsets=torch.empty(
                    WORKSPACE_CAPACITY,
                    dtype=torch.int32,
                    device=device,
                )
            )
            for slot in range(slots_per_shape)
        }

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            self._workspaces[(FIXED_ROWS, slot)].counts_and_offsets
            for slot in range(self.slots_per_shape)
        )

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.storage_tensors()
        )

    def provider_identity(self) -> dict[str, object]:
        return {
            "provider": "direct_fixed_b240_moe_alignment",
            "route_kind": self.route_kind,
            "global_row_shapes": list(self.global_row_shapes),
            "slots_per_shape": self.slots_per_shape,
            "route_shape": list(FIXED_B240_ALIGNMENT_PLAN.route_shape),
            "num_experts": FIXED_NUM_EXPERTS,
            "block_size": FIXED_BLOCK_SIZE,
            "workspace_capacity": WORKSPACE_CAPACITY,
            "launch_count": FIXED_B240_ALIGNMENT_PLAN.launch_count,
        }

    def __call__(
        self,
        topk_ids: torch.Tensor,
        *,
        block_size: int,
        num_experts: int,
        output: DeterministicMoEAlignment,
        slot: int,
    ) -> DeterministicMoEAlignment:
        if block_size != FIXED_BLOCK_SIZE or num_experts != FIXED_NUM_EXPERTS:
            raise ValueError("fixed B240 alignment Marlin geometry differs")
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise TypeError("fixed B240 alignment slot must be an integer")
        key = (FIXED_ROWS, slot)
        if key not in self._workspaces:
            raise ValueError(f"fixed B240 alignment slot {slot} is not registered")
        if topk_ids.device != self.device:
            raise ValueError("fixed B240 alignment route device differs")
        return fixed_b240_moe_align_block8(
            topk_ids,
            output=output,
            workspace=self._workspaces[key],
        )


def fixed_b240_max_padded_tokens() -> int:
    """Return the fixed Marlin ABI capacity, including worst-case padding."""

    return FIXED_MAX_PADDED_TOKENS


def _validate_route_metadata(topk_ids: torch.Tensor) -> None:
    if not isinstance(topk_ids, torch.Tensor):
        raise TypeError("topk_ids must be a tensor")
    if tuple(topk_ids.shape) != FIXED_B240_ALIGNMENT_PLAN.route_shape:
        raise ValueError("topk_ids must have the fixed shape [240, 6]")
    if topk_ids.dtype != torch.int32:
        raise TypeError("topk_ids must use the fixed int32 ABI")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")


def allocate_fixed_b240_alignment(
    topk_ids: torch.Tensor,
) -> tuple[DeterministicMoEAlignment, FixedB240AlignmentWorkspace]:
    """Allocate the fixed output/workspace on a cold setup path."""

    _validate_route_metadata(topk_ids)
    device = topk_ids.device
    output = DeterministicMoEAlignment(
        sorted_token_ids=torch.empty(
            FIXED_MAX_PADDED_TOKENS, dtype=torch.int32, device=device
        ),
        expert_ids=torch.empty(FIXED_MAX_BLOCKS, dtype=torch.int32, device=device),
        num_tokens_post_padded=torch.empty(1, dtype=torch.int32, device=device),
    )
    workspace = FixedB240AlignmentWorkspace(
        counts_and_offsets=torch.empty(
            WORKSPACE_CAPACITY, dtype=torch.int32, device=device
        )
    )
    return output, workspace


def validate_fixed_b240_alignment_tensors(
    topk_ids: torch.Tensor,
    *,
    output: DeterministicMoEAlignment,
    workspace: FixedB240AlignmentWorkspace,
) -> torch.device:
    """Validate the fixed metadata and non-alias contract without reading routes."""

    _validate_route_metadata(topk_ids)
    if not isinstance(output, DeterministicMoEAlignment):
        raise TypeError("output must be DeterministicMoEAlignment")
    if not isinstance(workspace, FixedB240AlignmentWorkspace):
        raise TypeError("workspace must be FixedB240AlignmentWorkspace")

    tensors = (
        ("topk_ids", topk_ids, (FIXED_ROWS, FIXED_TOPK)),
        (
            "output.sorted_token_ids",
            output.sorted_token_ids,
            (FIXED_MAX_PADDED_TOKENS,),
        ),
        ("output.expert_ids", output.expert_ids, (FIXED_MAX_BLOCKS,)),
        (
            "output.num_tokens_post_padded",
            output.num_tokens_post_padded,
            (1,),
        ),
        (
            "workspace.counts_and_offsets",
            workspace.counts_and_offsets,
            (WORKSPACE_CAPACITY,),
        ),
    )
    pointers: set[int] = set()
    for name, tensor, shape in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(tensor.shape) != shape or tensor.dtype != torch.int32:
            raise ValueError(f"{name} shape/dtype differs from the fixed ABI")
        if tensor.device != topk_ids.device or not tensor.is_contiguous():
            raise ValueError(f"{name} device/layout differs from topk_ids")
        pointers.add(tensor.untyped_storage().data_ptr())
    if len(pointers) != len(tensors):
        raise ValueError("route, output, and workspace tensors must not alias storage")
    return topk_ids.device


def _require_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("fixed B240 alignment requires CUDA tensors")


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError("fixed B240 alignment requires Triton")


if triton is not None:

    @triton.jit
    def _initialize_outputs_kernel(
        sorted_token_ids,
        expert_ids,
        NUM_SORTED: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        SENTINEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            sorted_token_ids + offsets,
            SENTINEL,
            mask=offsets < NUM_SORTED,
        )
        tl.store(expert_ids + offsets, -1, mask=offsets < NUM_BLOCKS)


    @triton.jit
    def _count_experts_kernel(
        topk_ids,
        counts_and_offsets,
        NUM_ASSIGNMENTS: tl.constexpr,
        SCAN_BLOCK: tl.constexpr,
    ):
        expert = tl.program_id(0)
        count = 0
        for start in range(0, NUM_ASSIGNMENTS, SCAN_BLOCK):
            positions = start + tl.arange(0, SCAN_BLOCK)
            routed = tl.load(
                topk_ids + positions,
                mask=positions < NUM_ASSIGNMENTS,
                other=-1,
            )
            matches = (positions < NUM_ASSIGNMENTS) & (routed == expert)
            count += tl.sum(matches.to(tl.int32), axis=0)
        tl.store(counts_and_offsets + expert, count)


    @triton.jit
    def _padded_prefix_kernel(
        counts_and_offsets,
        num_tokens_post_padded,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        PREFIX_BLOCK: tl.constexpr,
    ):
        experts = tl.arange(0, PREFIX_BLOCK)
        counts = tl.load(
            counts_and_offsets + experts,
            mask=experts < NUM_EXPERTS,
            other=0,
        )
        padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        exclusive = tl.cumsum(padded, axis=0) - padded
        total = tl.sum(padded, axis=0)
        tl.store(
            counts_and_offsets + experts,
            exclusive,
            mask=experts < NUM_EXPERTS,
        )
        tl.store(counts_and_offsets + NUM_EXPERTS, total)
        tl.store(num_tokens_post_padded, total)


    @triton.jit
    def _stable_scatter_kernel(
        topk_ids,
        counts_and_offsets,
        sorted_token_ids,
        expert_ids,
        NUM_ASSIGNMENTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        SCAN_BLOCK: tl.constexpr,
        MAX_BLOCKS_PER_EXPERT: tl.constexpr,
    ):
        expert = tl.program_id(0)
        output_start = tl.load(counts_and_offsets + expert)
        output_end = tl.load(counts_and_offsets + expert + 1)

        block_offsets = tl.arange(0, MAX_BLOCKS_PER_EXPERT)
        block_count = (output_end - output_start) // BLOCK_SIZE
        tl.store(
            expert_ids + output_start // BLOCK_SIZE + block_offsets,
            expert,
            mask=block_offsets < block_count,
        )

        running_count = 0
        for start in range(0, NUM_ASSIGNMENTS, SCAN_BLOCK):
            positions = start + tl.arange(0, SCAN_BLOCK)
            routed = tl.load(
                topk_ids + positions,
                mask=positions < NUM_ASSIGNMENTS,
                other=-1,
            )
            matches = (positions < NUM_ASSIGNMENTS) & (routed == expert)
            match_values = matches.to(tl.int32)
            ranks = tl.cumsum(match_values, axis=0) - match_values
            tl.store(
                sorted_token_ids + output_start + running_count + ranks,
                positions,
                mask=matches,
            )
            running_count += tl.sum(match_values, axis=0)


def _launch_fixed_b240_alignment(
    topk_ids: torch.Tensor,
    output: DeterministicMoEAlignment,
    workspace: FixedB240AlignmentWorkspace,
) -> None:
    _initialize_outputs_kernel[
        (triton.cdiv(FIXED_MAX_PADDED_TOKENS, _INITIALIZE_BLOCK),)
    ](
        output.sorted_token_ids,
        output.expert_ids,
        NUM_SORTED=FIXED_MAX_PADDED_TOKENS,
        NUM_BLOCKS=FIXED_MAX_BLOCKS,
        SENTINEL=FIXED_SENTINEL,
        BLOCK=_INITIALIZE_BLOCK,
        num_warps=4,
    )
    _count_experts_kernel[(FIXED_NUM_EXPERTS,)](
        topk_ids,
        workspace.counts_and_offsets,
        NUM_ASSIGNMENTS=FIXED_NUM_ASSIGNMENTS,
        SCAN_BLOCK=_SCAN_BLOCK,
        num_warps=4,
    )
    _padded_prefix_kernel[(1,)](
        workspace.counts_and_offsets,
        output.num_tokens_post_padded,
        NUM_EXPERTS=FIXED_NUM_EXPERTS,
        BLOCK_SIZE=FIXED_BLOCK_SIZE,
        PREFIX_BLOCK=WORKSPACE_CAPACITY,
        num_warps=8,
    )
    _stable_scatter_kernel[(FIXED_NUM_EXPERTS,)](
        topk_ids,
        workspace.counts_and_offsets,
        output.sorted_token_ids,
        output.expert_ids,
        NUM_ASSIGNMENTS=FIXED_NUM_ASSIGNMENTS,
        BLOCK_SIZE=FIXED_BLOCK_SIZE,
        SCAN_BLOCK=_SCAN_BLOCK,
        MAX_BLOCKS_PER_EXPERT=_MAX_BLOCKS_PER_EXPERT,
        num_warps=4,
    )


def fixed_b240_moe_align_block8(
    topk_ids: torch.Tensor,
    *,
    output: DeterministicMoEAlignment,
    workspace: FixedB240AlignmentWorkspace,
) -> DeterministicMoEAlignment:
    """Write exact expert-major alignment into fixed caller-owned storage."""

    device = validate_fixed_b240_alignment_tensors(
        topk_ids, output=output, workspace=workspace
    )
    _require_cuda_device(device)
    _require_triton()
    _launch_fixed_b240_alignment(topk_ids, output, workspace)
    return output


def fixed_b240_alignment_oracle(topk_ids: torch.Tensor) -> DeterministicMoEAlignment:
    """Build the exact fixed layout on CPU for cold-path semantic gates."""

    _validate_route_metadata(topk_ids)
    if topk_ids.device.type != "cpu":
        raise ValueError("the fixed B240 oracle accepts CPU tensors only")
    flat = topk_ids.reshape(-1).tolist()
    if any(route < 0 or route >= FIXED_NUM_EXPERTS for route in flat):
        raise ValueError("every route ID must be in [0, 256)")

    sorted_values: list[int] = []
    expert_values: list[int] = []
    for expert in range(FIXED_NUM_EXPERTS):
        matches = [index for index, route in enumerate(flat) if route == expert]
        if not matches:
            continue
        padded = (
            (len(matches) + FIXED_BLOCK_SIZE - 1) // FIXED_BLOCK_SIZE
        ) * FIXED_BLOCK_SIZE
        sorted_values.extend(matches)
        sorted_values.extend([FIXED_SENTINEL] * (padded - len(matches)))
        expert_values.extend([expert] * (padded // FIXED_BLOCK_SIZE))

    total = len(sorted_values)
    sorted_values.extend(
        [FIXED_SENTINEL] * (FIXED_MAX_PADDED_TOKENS - len(sorted_values))
    )
    expert_values.extend([-1] * (FIXED_MAX_BLOCKS - len(expert_values)))
    return DeterministicMoEAlignment(
        sorted_token_ids=torch.tensor(sorted_values, dtype=torch.int32),
        expert_ids=torch.tensor(expert_values, dtype=torch.int32),
        num_tokens_post_padded=torch.tensor([total], dtype=torch.int32),
    )


__all__ = [
    "FIXED_BLOCK_SIZE",
    "FIXED_B240_ALIGNMENT_PLAN",
    "FIXED_MAX_BLOCKS",
    "FIXED_MAX_PADDED_TOKENS",
    "FIXED_NUM_ASSIGNMENTS",
    "FIXED_NUM_EXPERTS",
    "FIXED_ROWS",
    "FIXED_SENTINEL",
    "FIXED_TOPK",
    "WORKSPACE_CAPACITY",
    "FixedB240AlignmentPlan",
    "FixedB240AlignmentWorkspace",
    "FixedB240MoEAlignmentProvider",
    "allocate_fixed_b240_alignment",
    "fixed_b240_alignment_oracle",
    "fixed_b240_max_padded_tokens",
    "fixed_b240_moe_align_block8",
    "validate_fixed_b240_alignment_tensors",
]
