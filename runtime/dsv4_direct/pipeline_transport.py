"""Serial NCCL P2P handoff for the scaled Flash TP4xPP2 pipeline.

Port lineage and selection rationale (E0pf vertical):

- Gaiban's PP2 transport went through E1b2a (single-position NCCL
  ``batch_isend_irecv`` bring-up) -> E1b2b/c/d (stateful eager / graph /
  overlap) -> the in-package ``pipeline_overlap.py``
  (``DoubleBufferedP2PTransport`` staged-lane overlap and
  ``DirectRendezvousP2PTransport`` fixed-endpoint form) -> the E1b2f/E1b2z
  handoff-attribution runtimes built on it.  The alternative transports were
  all *rejected* in gaiban: E1b3a CUDA-IPC host-generation, E1b3c phase
  separation, and the E1b3i GPU-timeline CUDA-IPC architecture each failed
  their direction/absolute gates; the retained mechanics are E1b2z's
  default-NCCL send/recv, and the last open fix direction (E1b3j) changes
  only physical rank placement, not the transport.
- This module therefore ports the **fixed-endpoint NCCL P2P form**
  (``DirectRendezvousP2PTransport`` semantics: one pair group per TP rank,
  one ``P2POp`` per step on a pointer-stable endpoint) reduced to a serial
  cycle: no staging lanes, no double buffering, no probe machinery.  Those
  are overlap/attribution treatments, not correctness surface; with a serial
  schedule there is exactly one payload in flight and stream-order ``wait()``
  gives the full ordering.  Overlap (and with it >1 in-flight microbatch)
  is deliberately deferred to the two-machine / performance vertical.

Handoff tensor contract (frozen for E0pf): ``[local_batch, 1, 4, 4096]``
BF16, contiguous, CUDA-resident -- the Flash block residual at the stage
boundary (Pro: ``[b, 1, 4, 7168]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from .block import BLOCK_HC_MULT, BLOCK_HIDDEN_SIZE


TP_SIZE = 4
PP2_WORLD = 8
HANDOFF_DTYPE = torch.bfloat16


class PipelineTransportError(RuntimeError):
    """Raised when the PP handoff would leave its frozen contract."""


def pp2_group_rank_specs() -> tuple[tuple[int, ...], ...]:
    """The E1b2a group table: two TP4 stages plus one pair group per TP rank."""

    return (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )


@dataclass(frozen=True, slots=True)
class PP2GroupBundle:
    tp_group: object
    pair_group: object
    all_groups: tuple[object, ...]
    stage_id: int
    tp_rank: int
    tp_global_ranks: tuple[int, ...]
    pair_global_ranks: tuple[int, int]


def create_pp2_groups(
    rank: int,
    *,
    timeout: timedelta = timedelta(minutes=3),
) -> PP2GroupBundle:
    """Create the frozen PP2 subgroups on every rank (E1b2a ``create_groups``)."""

    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < PP2_WORLD:
        raise PipelineTransportError(f"rank must be in [0, {PP2_WORLD})")
    if not dist.is_initialized() or dist.get_world_size() != PP2_WORLD:
        raise PipelineTransportError("PP2 groups require an initialized world of 8")
    specs = pp2_group_rank_specs()
    groups = tuple(
        dist.new_group(ranks=list(ranks), backend="nccl", timeout=timeout)
        for ranks in specs
    )
    stage_id = rank // TP_SIZE
    tp_rank = rank % TP_SIZE
    return PP2GroupBundle(
        tp_group=groups[stage_id],
        pair_group=groups[2 + tp_rank],
        all_groups=groups,
        stage_id=stage_id,
        tp_rank=tp_rank,
        tp_global_ranks=specs[stage_id],
        pair_global_ranks=(tp_rank, tp_rank + TP_SIZE),
    )


def validate_handoff_endpoint(
    endpoint: torch.Tensor, *, local_batch: int
) -> tuple[int, ...]:
    """Enforce the frozen ``[b, 1, 4, 4096]`` BF16 boundary contract."""

    expected_shape = (local_batch, 1, BLOCK_HC_MULT, BLOCK_HIDDEN_SIZE)
    if not isinstance(endpoint, torch.Tensor):
        raise PipelineTransportError("handoff endpoint must be a tensor")
    if tuple(endpoint.shape) != expected_shape:
        raise PipelineTransportError(
            f"handoff endpoint shape {tuple(endpoint.shape)} != {expected_shape}"
        )
    if endpoint.dtype != HANDOFF_DTYPE:
        raise PipelineTransportError(
            f"handoff endpoint dtype {endpoint.dtype} != {HANDOFF_DTYPE}"
        )
    if endpoint.device.type != "cuda":
        raise PipelineTransportError("handoff endpoint must be CUDA-resident")
    if not endpoint.is_contiguous():
        raise PipelineTransportError("handoff endpoint must be contiguous")
    return expected_shape


class SerialPairHandoff:
    """Serial fixed-endpoint NCCL P2P handoff on one PP pair group.

    Stage 0 sends its stage-exit residual from the fixed stage output
    buffer; stage 1 receives into a fixed staging endpoint that the caller
    D2D-unpacks into the stage input (the gaiban E1b2z staged form --
    ``stage1_d2d_unpack``; the stateful stage validation contract requires
    the validated input to be external to the plan workspaces).  The
    endpoint binding is validated to be pointer-stable across the whole
    cycle, per the gaiban fixed-endpoint transport contract.
    """

    def __init__(
        self,
        *,
        stage_id: int,
        pair_group: object,
        endpoint: torch.Tensor,
        local_batch: int,
    ) -> None:
        if stage_id not in (0, 1):
            raise PipelineTransportError("stage_id must be zero or one")
        validate_handoff_endpoint(endpoint, local_batch=local_batch)
        if pair_group is None or dist.get_world_size(pair_group) != 2:
            raise PipelineTransportError("pair group must be a live 2-rank group")
        self.stage_id = stage_id
        self.pair_group = pair_group
        self.endpoint = endpoint
        self.local_batch = local_batch
        self._binding = self._endpoint_binding()
        self._steps_transferred = 0

    def _endpoint_binding(self) -> tuple[object, ...]:
        return (
            id(self.endpoint),
            int(self.endpoint.data_ptr()),
            int(self.endpoint.untyped_storage().data_ptr()),
            tuple(self.endpoint.shape),
            tuple(self.endpoint.stride()),
            self.endpoint.dtype,
            self.endpoint.device,
        )

    @property
    def role(self) -> str:
        return "send" if self.stage_id == 0 else "receive"

    @property
    def payload_nbytes(self) -> int:
        return int(self.endpoint.numel() * self.endpoint.element_size())

    def transfer_step(self, step: int) -> None:
        """Post one send/recv of the fixed endpoint and order it on the
        current stream (``Work.wait``)."""

        if step != self._steps_transferred:
            raise PipelineTransportError(
                f"handoff steps must be ordered: got {step}, "
                f"expected {self._steps_transferred}"
            )
        if self._endpoint_binding() != self._binding:
            raise PipelineTransportError("handoff endpoint binding drifted")
        operation = dist.isend if self.stage_id == 0 else dist.irecv
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(
                    operation,
                    self.endpoint,
                    group=self.pair_group,
                    group_peer=1 - self.stage_id,
                )
            ]
        )
        if len(works) != 1 or works[0] is None:
            raise PipelineTransportError("pair P2P returned an invalid Work list")
        works[0].wait()
        self._steps_transferred += 1

    def close(self, *, expected_steps: int) -> dict[str, object]:
        if self._endpoint_binding() != self._binding:
            raise PipelineTransportError("handoff endpoint binding drifted at close")
        record = {
            "role": self.role,
            "stage_id": self.stage_id,
            "steps_transferred": self._steps_transferred,
            "expected_steps": expected_steps,
            "payload_nbytes": self.payload_nbytes,
            "endpoint_shape": list(self.endpoint.shape),
            "endpoint_dtype": str(self.endpoint.dtype),
            "endpoint_pointer_stable": True,
            "accepted": self._steps_transferred == expected_steps,
        }
        if not record["accepted"]:
            raise PipelineTransportError(
                f"handoff cycle transferred {self._steps_transferred} steps, "
                f"expected {expected_steps}"
            )
        return record


__all__ = [
    "HANDOFF_DTYPE",
    "PP2_WORLD",
    "PP2GroupBundle",
    "PipelineTransportError",
    "SerialPairHandoff",
    "TP_SIZE",
    "create_pp2_groups",
    "pp2_group_rank_specs",
    "validate_handoff_endpoint",
]
