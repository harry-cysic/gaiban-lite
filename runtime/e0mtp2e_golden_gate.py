#!/usr/bin/env python3
"""E0mtp2e: MTP draft-verify E2E golden gate (full model, TP4xPP4, B=1).

Sixteenth vertical: MTP (multi-token prediction) speculative decoding over the
complete DeepSeek-V4-Flash decode path.  The mtp.0 block (sliding-window
attention + learned-router MoE + HC, with e_proj/h_proj/enorm/hnorm bridge and
its own hc_head/norm over the *shared* embedding/head, reference model.py
:738-766) lives on the tail stage (titan065 socket 1, with the head).

Draft-verify protocol (standard 1-draft speculative decoding; the reference
generate.py never invokes MTP, so the protocol is defined here):
  state: main KV committed through position P; pending input x_{P+1}; MTP fed
  pairs (h_j, x_{j+1}) through j == P, draft z ~ x_{P+2}.
  round: verify-step feeds [x_{P+1}, z] as ONE two-position decode step ->
  y_{P+1}, y_{P+2}.  Greedy accept iff y_{P+1} == z (draft == verify argmax):
  emit y_{P+1}, y_{P+2}, commit P+2.  Reject: emit y_{P+1} only, roll the
  second position's state mutations back (post-first-token snapshot), commit
  P+1.  MTP then ingests the newly committed (hidden, next-token) pairs
  (1 on reject, 2 on accept) and the last ingest's argmax is the next draft.
  Greedy outputs are therefore protocol-identical to MTP-off decoding as long
  as the verify step's per-position logits argmax-match the single-token path.

Arms (all B=1, fused HC boundary + FP8 KV production baseline by default):
  off_teacher  -- exact E0e2e teacher-forced golden gate (baseline
                  reproduction; fused+fp8 arm previously measured 467/482).
  mtp_teacher  -- identical teacher-forced main path (bitwise: MTP only adds
                  a parallel mtp.0 lane) + per-step draft hit rate against the
                  model's own predictions (golden-trajectory acceptance).
  off_free     -- free-running greedy decode (closed loop), the MTP-off
                  reference stream + per-step wall time.
  mtp_free     -- draft-verify with the verify step run as two chained
                  single-token passes (bitwise-identical operator chain to
                  off_free by construction) + snapshot/rollback on reject.
                  Hard gate: emitted tokens must equal off_free exactly.
  mtp_fused    -- draft-verify with the real fused two-position verify step
                  (dsv4_direct.verify2; hidden-side GEMMs fused over both
                  tokens, per-position state/sparse cores, in-step post-t1
                  snapshots) + per-round wall time.  Token equality vs
                  off_free is measured (near-tie argmax flips from the [2, d]
                  GEMM shapes are the known risk) and reported.

Run (driven by ``run_e0mtp2e_dual.sh``): torchrun 2x8, same topology/env as
the E0e2e gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from dsv4_direct.attention import rms_norm
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.head_stage import (
    EmbedHeadMaterial,
    embed_hc_residual,
    head_logits,
    head_logits_all,
    load_embed_head_material,
)
from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.hyper_connections import hc_post, hc_pre
from dsv4_direct.model_contract import MTP_LAYER_ID
from dsv4_direct.mtp_block import MTPLane, build_mtp_layer_material
from dsv4_direct.physical_stage import (
    EXPECTED_TP_SIZE,
    PhysicalLayerMaterial,
    build_physical_stage,
    validate_live_tp_group,
)
from dsv4_direct.ratio4_fullpos import Ratio4FullPositionAttention
from dsv4_direct.verify2 import (
    ratio128_decode2,
    ratio4_decode2,
    restore_decode_state,
    snapshot_decode_state,
    window_decode2,
)


WORLD = 16
STAGE_COUNT = 4
MODEL_LAYERS = 43
STAGE_LAYERS: dict[int, tuple[int, ...]] = {
    0: tuple(range(0, 11)),
    1: tuple(range(11, 22)),
    2: tuple(range(22, 33)),
    3: tuple(range(33, 43)),
}
MAX_SEQ_LEN = 256
MAX_COMPARE_STEPS = 128
LOCAL_BATCH = 1
HIDDEN = 4096
HC_MULT = 4
CANONICAL_RANK = 12  # stage 3, tp_rank 0
ARMS = ("off_teacher", "mtp_teacher", "off_free", "mtp_free", "mtp_fused")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    metadata = f"{list(tensor.shape)}|{tensor.dtype}|".encode("utf-8")
    return hashlib.sha256(metadata + value).hexdigest()


# --------------------------------------------------------------------------
# topology (E0e2e form)


def create_pp4_groups(rank: int) -> dict[str, Any]:
    specs: list[tuple[int, ...]] = [
        tuple(range(stage * 4, stage * 4 + 4)) for stage in range(STAGE_COUNT)
    ]
    for boundary in range(STAGE_COUNT - 1):
        for tp in range(EXPECTED_TP_SIZE):
            specs.append((boundary * 4 + tp, (boundary + 1) * 4 + tp))
    groups = [
        dist.new_group(ranks=list(spec), backend="nccl", timeout=timedelta(minutes=60))
        for spec in specs
    ]
    stage = rank // EXPECTED_TP_SIZE
    tp_rank = rank % EXPECTED_TP_SIZE
    prev_pair = groups[STAGE_COUNT + (stage - 1) * 4 + tp_rank] if stage > 0 else None
    next_pair = (
        groups[STAGE_COUNT + stage * 4 + tp_rank] if stage < STAGE_COUNT - 1 else None
    )
    return {
        "stage": stage,
        "tp_rank": tp_rank,
        "tp_group": groups[stage],
        "tp_global_ranks": specs[stage],
        "prev_pair": prev_pair,
        "next_pair": next_pair,
        "all_groups": groups,
    }


def pair_transfer(tensor: torch.Tensor, *, send: bool, group: Any) -> None:
    if not tensor.is_contiguous():
        raise ValueError("pair transfer requires a contiguous tensor")
    operation = dist.isend if send else dist.irecv
    works = dist.batch_isend_irecv(
        [dist.P2POp(operation, tensor, group=group, group_peer=1 if send else 0)]
    )
    works[0].wait()


def run_placement_check(*, stage: int, world: int) -> dict[str, Any]:
    record = {
        "rank": dist.get_rank(),
        "stage": stage,
        "host": platform.node(),
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
    }
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, record)
    stage_hosts: dict[int, set[str]] = {s: set() for s in range(STAGE_COUNT)}
    for entry in gathered:
        stage_hosts[entry["stage"]].add(entry["host"])
    hosts = {s: sorted(stage_hosts[s]) for s in range(STAGE_COUNT)}
    accepted = (
        all(len(stage_hosts[s]) == 1 for s in range(STAGE_COUNT))
        and stage_hosts[0] == stage_hosts[1]
        and stage_hosts[2] == stage_hosts[3]
        and stage_hosts[0] != stage_hosts[2]
    )
    return {"stage_hosts": hosts, "accepted": bool(accepted), "ranks": gathered}


def broadcast_payload(payload: Any, *, src: int = CANONICAL_RANK) -> Any:
    holder: list[Any] = [payload if dist.get_rank() == src else None]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


# --------------------------------------------------------------------------
# per-prompt stage lane (E0e2e form + verify-2 + snapshot/rollback)


class StageLane:
    def __init__(
        self,
        materials: Sequence[PhysicalLayerMaterial],
        *,
        backend: Any | None,
        device: torch.device,
    ) -> None:
        self.device = device
        self.backend = backend
        self.layers: list[tuple[PhysicalLayerMaterial, Any]] = []
        for material in materials:
            if material.kind == "ratio4":
                attention = Ratio4FullPositionAttention(
                    material.attention_config,
                    material.prepared,
                    batch_size=LOCAL_BATCH,
                    device=device,
                    kv_dtype=material.kv_dtype,
                    indexer_dtype=material.indexer_kv_dtype,
                )
            else:
                state = material.new_state(num_local_sequences=LOCAL_BATCH)
                attention = material.new_attention(state)
            self.layers.append((material, attention))

    # -- state snapshot / rollback (chained-verify form) --

    def _state_object(self, material: PhysicalLayerMaterial, attention: Any) -> Any:
        return attention if material.kind == "ratio4" else attention.state

    def snapshot_states(self) -> list[tuple[Any, dict[str, Any]]]:
        return [
            (target, snapshot_decode_state(target))
            for target in (
                self._state_object(material, attention)
                for material, attention in self.layers
            )
        ]

    @staticmethod
    def restore_states(snapshots: list[tuple[Any, dict[str, Any]]]) -> None:
        for target, snapshot in snapshots:
            restore_decode_state(target, snapshot)

    # -- single-token / prefill path (bitwise-identical to the E0e2e gate) --

    @staticmethod
    def _attention_branch(
        material: PhysicalLayerMaterial, attention: Any, hidden: torch.Tensor, start_pos: int
    ) -> torch.Tensor:
        if material.kind == "ratio4":
            return attention(hidden, start_pos=start_pos)
        branch, _trace = attention(hidden, start_pos=start_pos)
        return branch

    def _layer_eager(
        self,
        material: PhysicalLayerMaterial,
        attention: Any,
        residual: torch.Tensor,
        *,
        start_pos: int,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        hc = material.raw_block.hyper_connection
        hidden, post, comb = hc_pre(
            residual,
            hc.attn_fn,
            hc.attn_scale,
            hc.attn_base,
            norm_eps=material.norm_eps,
            sinkhorn_iters=material.sinkhorn_iters,
            hc_eps=material.hc_eps,
        )
        hidden = rms_norm(hidden, material.raw_block.attn_norm, eps=material.norm_eps)
        branch = self._attention_branch(material, attention, hidden, start_pos)
        residual = hc_post(branch, residual, post, comb)
        hidden, post, comb = hc_pre(
            residual,
            hc.ffn_fn,
            hc.ffn_scale,
            hc.ffn_base,
            norm_eps=material.norm_eps,
            sinkhorn_iters=material.sinkhorn_iters,
            hc_eps=material.hc_eps,
        )
        hidden = rms_norm(hidden, material.raw_block.ffn_norm, eps=material.norm_eps)
        moe_output = material.moe.forward_tensor(
            hidden,
            input_ids_local=input_ids if material.route_kind == "hash" else None,
            slot=0,
        )
        return hc_post(moe_output, residual, post, comb)

    def _forward_fused_chain(
        self,
        residual: torch.Tensor,
        *,
        start_pos: int,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        first_material, _ = self.layers[0]
        hc = first_material.raw_block.hyper_connection
        hidden, post, comb = hc_pre(
            residual,
            hc.attn_fn,
            hc.attn_scale,
            hc.attn_base,
            norm_eps=first_material.norm_eps,
            sinkhorn_iters=first_material.sinkhorn_iters,
            hc_eps=first_material.hc_eps,
        )
        hidden = rms_norm(
            hidden, first_material.raw_block.attn_norm, eps=first_material.norm_eps
        )
        for index, (material, attention) in enumerate(self.layers):
            branch = self._attention_branch(material, attention, hidden, start_pos)
            hcw = material.raw_block.hyper_connection
            residual, hidden, post, comb = self.backend.post_pre_norm(
                branch,
                residual,
                post,
                comb,
                hc_fn=hcw.ffn_fn,
                hc_scale=hcw.ffn_scale,
                hc_base=hcw.ffn_base,
                norm_weight=material.raw_block.ffn_norm,
                norm_eps=material.norm_eps,
                sinkhorn_iters=material.sinkhorn_iters,
                hc_eps=material.hc_eps,
            )
            moe_output = material.moe.forward_tensor(
                hidden,
                input_ids_local=input_ids if material.route_kind == "hash" else None,
                slot=0,
            )
            if index + 1 < len(self.layers):
                next_material = self.layers[index + 1][0]
                nhc = next_material.raw_block.hyper_connection
                residual, hidden, post, comb = self.backend.post_pre_norm(
                    moe_output,
                    residual,
                    post,
                    comb,
                    hc_fn=nhc.attn_fn,
                    hc_scale=nhc.attn_scale,
                    hc_base=nhc.attn_base,
                    norm_weight=next_material.raw_block.attn_norm,
                    norm_eps=next_material.norm_eps,
                    sinkhorn_iters=next_material.sinkhorn_iters,
                    hc_eps=next_material.hc_eps,
                )
            else:
                residual = hc_post(moe_output, residual, post, comb)
        return residual

    def forward(
        self,
        residual: torch.Tensor,
        *,
        start_pos: int,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.backend is not None and residual.shape[1] == 1:
            return self._forward_fused_chain(
                residual, start_pos=start_pos, input_ids=input_ids
            )
        for material, attention in self.layers:
            residual = self._layer_eager(
                material,
                attention,
                residual,
                start_pos=start_pos,
                input_ids=input_ids,
            )
        return residual

    # -- fused two-position verify step --

    def _attention_verify2(
        self,
        material: PhysicalLayerMaterial,
        attention: Any,
        hidden2: torch.Tensor,
        start_pos: int,
        snapshot_out: list[tuple[Any, dict[str, Any]]],
    ) -> torch.Tensor:
        if material.kind == "window":
            return window_decode2(
                attention, hidden2, start_pos=start_pos, snapshot_out=snapshot_out
            )
        if material.kind == "ratio4":
            return ratio4_decode2(
                attention, hidden2, start_pos=start_pos, snapshot_out=snapshot_out
            )
        return ratio128_decode2(
            attention, hidden2, start_pos=start_pos, snapshot_out=snapshot_out
        )

    def _boundary_per_token(
        self,
        branch_token: torch.Tensor,
        residual_token: torch.Tensor,
        post_token: torch.Tensor,
        comb_token: torch.Tensor,
        *,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        sinkhorn_iters: int,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One hc_post -> hc_pre -> norm boundary at seqlen 1 (per token).

        Uses the fused TileLang boundary when active (its quantified decode
        shape), the eager composition otherwise -- per token, so the boundary
        math matches the baseline single-token steps exactly.
        """

        if self.backend is not None:
            return self.backend.post_pre_norm(
                branch_token,
                residual_token,
                post_token,
                comb_token,
                hc_fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
                sinkhorn_iters=sinkhorn_iters,
                hc_eps=hc_eps,
            )
        residual_token = hc_post(branch_token, residual_token, post_token, comb_token)
        hidden_token, post_token, comb_token = hc_pre(
            residual_token,
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps=norm_eps,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=hc_eps,
        )
        hidden_token = rms_norm(hidden_token, norm_weight, eps=norm_eps)
        return residual_token, hidden_token, post_token, comb_token

    def forward_verify2(
        self,
        residual2: torch.Tensor,
        *,
        start_pos: int,
        input_ids2: torch.Tensor | None,
        snapshot_out: list[tuple[Any, dict[str, Any]]],
    ) -> torch.Tensor:
        """Fused two-position decode: [b, 2, hc, d] -> [b, 2, hc, d].

        Attention/MoE GEMMs run fused over both positions; HC boundaries run
        per position at the baseline seqlen-1 shape.
        """

        if residual2.shape[1] != 2:
            raise ValueError("verify-2 residual must carry exactly two positions")
        residual_t = [residual2[:, 0:1].contiguous(), residual2[:, 1:2].contiguous()]
        hidden_t: list[torch.Tensor] = [None, None]  # type: ignore[list-item]
        post_t: list[torch.Tensor] = [None, None]  # type: ignore[list-item]
        comb_t: list[torch.Tensor] = [None, None]  # type: ignore[list-item]
        first_material, _ = self.layers[0]
        hc = first_material.raw_block.hyper_connection
        for token in range(2):
            hidden, post, comb = hc_pre(
                residual_t[token],
                hc.attn_fn,
                hc.attn_scale,
                hc.attn_base,
                norm_eps=first_material.norm_eps,
                sinkhorn_iters=first_material.sinkhorn_iters,
                hc_eps=first_material.hc_eps,
            )
            hidden_t[token] = rms_norm(
                hidden, first_material.raw_block.attn_norm, eps=first_material.norm_eps
            )
            post_t[token] = post
            comb_t[token] = comb
        for index, (material, attention) in enumerate(self.layers):
            hidden2 = torch.cat(hidden_t, dim=1)
            branch2 = self._attention_verify2(
                material, attention, hidden2, start_pos, snapshot_out
            )
            hcw = material.raw_block.hyper_connection
            for token in range(2):
                (
                    residual_t[token],
                    hidden_t[token],
                    post_t[token],
                    comb_t[token],
                ) = self._boundary_per_token(
                    branch2[:, token : token + 1],
                    residual_t[token],
                    post_t[token],
                    comb_t[token],
                    hc_fn=hcw.ffn_fn,
                    hc_scale=hcw.ffn_scale,
                    hc_base=hcw.ffn_base,
                    norm_weight=material.raw_block.ffn_norm,
                    norm_eps=material.norm_eps,
                    sinkhorn_iters=material.sinkhorn_iters,
                    hc_eps=material.hc_eps,
                )
            moe2 = material.moe.forward_tensor(
                torch.cat(hidden_t, dim=1),
                input_ids_local=(
                    input_ids2 if material.route_kind == "hash" else None
                ),
                slot=0,
            )
            if index + 1 < len(self.layers):
                next_material = self.layers[index + 1][0]
                nhc = next_material.raw_block.hyper_connection
                for token in range(2):
                    (
                        residual_t[token],
                        hidden_t[token],
                        post_t[token],
                        comb_t[token],
                    ) = self._boundary_per_token(
                        moe2[:, token : token + 1],
                        residual_t[token],
                        post_t[token],
                        comb_t[token],
                        hc_fn=nhc.attn_fn,
                        hc_scale=nhc.attn_scale,
                        hc_base=nhc.attn_base,
                        norm_weight=next_material.raw_block.attn_norm,
                        norm_eps=next_material.norm_eps,
                        sinkhorn_iters=next_material.sinkhorn_iters,
                        hc_eps=next_material.hc_eps,
                    )
            else:
                for token in range(2):
                    residual_t[token] = hc_post(
                        moe2[:, token : token + 1],
                        residual_t[token],
                        post_t[token],
                        comb_t[token],
                    )
        return torch.cat(residual_t, dim=1)


# --------------------------------------------------------------------------
# pipeline pass helpers (one full 43-layer traversal)


def pipeline_pass(
    *,
    step_tokens: list[int],
    position: int,
    lane: StageLane,
    topo: dict[str, Any],
    embed_material: EmbedHeadMaterial | None,
    device: torch.device,
    verify2: bool = False,
    snapshot_out: list[tuple[Any, dict[str, Any]]] | None = None,
) -> torch.Tensor:
    """Run one pipeline traversal; returns this stage's output residual."""

    stage = topo["stage"]
    seqlen = len(step_tokens)
    input_ids = torch.tensor([step_tokens], dtype=torch.int64, device=device)
    if stage == 0:
        residual = embed_hc_residual(embed_material, input_ids)
    else:
        residual = torch.empty(
            (LOCAL_BATCH, seqlen, HC_MULT, HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        pair_transfer(residual, send=False, group=topo["prev_pair"])
    if verify2:
        if seqlen != 2 or snapshot_out is None:
            raise ValueError("verify-2 pass requires two tokens and a snapshot sink")
        residual = lane.forward_verify2(
            residual,
            start_pos=position,
            input_ids2=input_ids,
            snapshot_out=snapshot_out,
        )
    else:
        residual = lane.forward(residual, start_pos=position, input_ids=input_ids)
    if stage < STAGE_COUNT - 1:
        pair_transfer(residual.contiguous(), send=True, group=topo["next_pair"])
    return residual


def logits_record(logits: torch.Tensor, golden: int | None) -> dict[str, Any]:
    top2 = torch.topk(logits, 2)
    record = {
        "predicted": int(top2.indices[0].item()),
        "top1_logit": float(top2.values[0].item()),
        "top2_gap": float((top2.values[0] - top2.values[1]).item()),
    }
    if golden is not None:
        record["golden"] = int(golden)
        record["match"] = record["predicted"] == int(golden)
        record["golden_deficit"] = record["top1_logit"] - float(
            logits[int(golden)].item()
        )
    return record


# --------------------------------------------------------------------------
# arms


class MTPDriver:
    """Stage-3 MTP lane driver; inert on stages 0-2."""

    def __init__(
        self,
        *,
        mtp_material: Any | None,
        embed_head: EmbedHeadMaterial | None,
        device: torch.device,
        active: bool,
    ) -> None:
        self.active = active
        self.lane: MTPLane | None = None
        self.device = device
        if active and mtp_material is not None:
            if embed_head is None or embed_head.embed_weight is None:
                raise ValueError("MTP driver requires tail embed material")
            self.lane = MTPLane(
                mtp_material,
                embed_weight=embed_head.embed_weight,
                head_weight=embed_head.head_weight,
                batch_size=LOCAL_BATCH,
                device=device,
            )

    def prefill(self, residual: torch.Tensor, prompt_tokens: list[int]) -> None:
        if self.lane is None:
            return
        length = len(prompt_tokens)
        if length < 2:
            return
        ids = torch.tensor([prompt_tokens[1:]], dtype=torch.int64, device=self.device)
        self.lane.forward(residual[:, : length - 1], ids, start_pos=0)

    def step(
        self, residual_token: torch.Tensor, next_token: int, start_pos: int
    ) -> int:
        """Ingest one committed (hidden, next-token) pair; return the draft."""

        if self.lane is None:
            raise RuntimeError("MTP step on an inactive driver")
        ids = torch.tensor([[next_token]], dtype=torch.int64, device=self.device)
        logits = self.lane.forward(residual_token, ids, start_pos=start_pos)
        return int(torch.argmax(logits[0]).item())


def run_prompt_teacher(
    *,
    prompt_tokens: list[int],
    golden_tokens: list[int],
    compare_steps: int,
    lane: StageLane,
    mtp: MTPDriver,
    topo: dict[str, Any],
    embed_material: EmbedHeadMaterial | None,
    head_material: EmbedHeadMaterial | None,
    device: torch.device,
) -> dict[str, Any]:
    """Teacher-forced golden comparison; MTP (if active) rides along."""

    stage = topo["stage"]
    prompt_len = len(prompt_tokens)
    steps: list[dict[str, Any]] = []
    decode_ms: list[float] = []
    draft_hits: list[dict[str, Any]] = []
    pending_draft: int | None = None
    for step in range(compare_steps):
        if step == 0:
            position = 0
            step_tokens = prompt_tokens
        else:
            position = prompt_len + step - 1
            step_tokens = [golden_tokens[step - 1]]
        started = time.perf_counter()
        residual = pipeline_pass(
            step_tokens=step_tokens,
            position=position,
            lane=lane,
            topo=topo,
            embed_material=embed_material,
            device=device,
        )
        if stage == STAGE_COUNT - 1:
            logits = head_logits(head_material, residual)
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError(f"non-finite logits at teacher step {step}")
            record = logits_record(logits[0], golden_tokens[step])
            record["step"] = step
            steps.append(record)
            if mtp.active and step >= 1 and pending_draft is not None:
                draft_hits.append(
                    {
                        "step": step,
                        "draft": pending_draft,
                        "predicted": record["predicted"],
                        "accepted": pending_draft == record["predicted"],
                        "draft_matches_golden": pending_draft
                        == int(golden_tokens[step]),
                    }
                )
            if mtp.active:
                if step == 0:
                    mtp.prefill(residual, prompt_tokens)
                    pending_draft = mtp.step(
                        residual[:, -1:], int(golden_tokens[0]), prompt_len - 1
                    )
                elif step + 1 < compare_steps:
                    pending_draft = mtp.step(
                        residual, int(golden_tokens[step]), position
                    )
                else:
                    pending_draft = None
        torch.cuda.synchronize(device)
        if step > 0:
            decode_ms.append((time.perf_counter() - started) * 1e3)

    result: dict[str, Any] = {
        "prompt_len": prompt_len,
        "compare_steps": compare_steps,
        "decode_ms_mean": statistics.fmean(decode_ms) if decode_ms else None,
    }
    if stage == STAGE_COUNT - 1:
        mismatches = [record for record in steps if not record["match"]]
        accepted_count = sum(1 for hit in draft_hits if hit["accepted"])
        result.update(
            {
                "steps": steps,
                "matched": sum(1 for record in steps if record["match"]),
                "mismatches": mismatches,
                "first_mismatch": mismatches[0] if mismatches else None,
                "predicted_tokens": [record["predicted"] for record in steps],
                "draft_events": len(draft_hits),
                "draft_accepted": accepted_count,
                "draft_acceptance_rate": (
                    accepted_count / len(draft_hits) if draft_hits else None
                ),
                "draft_golden_hits": sum(
                    1 for hit in draft_hits if hit["draft_matches_golden"]
                ),
            }
        )
    return result


def run_prompt_free(
    *,
    prompt_tokens: list[int],
    n_tokens: int,
    mode: str,  # "off" | "chained" | "fused"
    lane: StageLane,
    mtp: MTPDriver,
    topo: dict[str, Any],
    embed_material: EmbedHeadMaterial | None,
    head_material: EmbedHeadMaterial | None,
    device: torch.device,
) -> dict[str, Any]:
    """Free-running greedy decode, optionally with MTP draft-verify."""

    stage = topo["stage"]
    tail = stage == STAGE_COUNT - 1
    prompt_len = len(prompt_tokens)

    # ---- prefill ----
    residual = pipeline_pass(
        step_tokens=prompt_tokens,
        position=0,
        lane=lane,
        topo=topo,
        embed_material=embed_material,
        device=device,
    )
    payload = None
    if tail:
        logits = head_logits(head_material, residual)
        first_token = int(torch.argmax(logits[0]).item())
        draft = None
        if mode != "off" and mtp.active:
            mtp.prefill(residual, prompt_tokens)
            draft = mtp.step(residual[:, -1:], first_token, prompt_len - 1)
        payload = {"token": first_token, "draft": draft}
    payload = broadcast_payload(payload)
    emitted: list[int] = [payload["token"]]
    draft = payload["draft"]
    committed = prompt_len - 1  # last position whose KV is committed
    pending = payload["token"]  # input for position committed + 1

    step_ms: list[float] = []
    round_records: list[dict[str, Any]] = []

    if mode == "off":
        while len(emitted) < n_tokens:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            residual = pipeline_pass(
                step_tokens=[pending],
                position=committed + 1,
                lane=lane,
                topo=topo,
                embed_material=embed_material,
                device=device,
            )
            token_payload = None
            if tail:
                logits = head_logits(head_material, residual)
                token_payload = {"token": int(torch.argmax(logits[0]).item())}
            token_payload = broadcast_payload(token_payload)
            torch.cuda.synchronize(device)
            step_ms.append((time.perf_counter() - started) * 1e3)
            emitted.append(token_payload["token"])
            committed += 1
            pending = token_payload["token"]
    else:
        while len(emitted) < n_tokens:
            torch.cuda.synchronize(device)
            round_started = time.perf_counter()
            verify_position = committed + 1
            snapshots: list[tuple[Any, dict[str, Any]]] = []
            if mode == "chained":
                residual_first = pipeline_pass(
                    step_tokens=[pending],
                    position=verify_position,
                    lane=lane,
                    topo=topo,
                    embed_material=embed_material,
                    device=device,
                )
                snapshots = lane.snapshot_states()
                residual_second = pipeline_pass(
                    step_tokens=[draft],
                    position=verify_position + 1,
                    lane=lane,
                    topo=topo,
                    embed_material=embed_material,
                    device=device,
                )
            else:
                residual_pair = pipeline_pass(
                    step_tokens=[pending, draft],
                    position=verify_position,
                    lane=lane,
                    topo=topo,
                    embed_material=embed_material,
                    device=device,
                    verify2=True,
                    snapshot_out=snapshots,
                )
                residual_first = residual_pair[:, 0:1].contiguous()
                residual_second = residual_pair[:, 1:2].contiguous()
            torch.cuda.synchronize(device)
            verify_done = time.perf_counter()
            decision = None
            if tail:
                if mode == "chained":
                    logits_first = head_logits(head_material, residual_first)[0]
                    logits_second = head_logits(head_material, residual_second)[0]
                else:
                    both = head_logits_all(
                        head_material, torch.cat([residual_first, residual_second], dim=1)
                    )
                    logits_first, logits_second = both[0, 0], both[0, 1]
                if not bool(torch.isfinite(logits_first).all().item()):
                    raise RuntimeError("non-finite verify logits (first position)")
                first_pred = int(torch.argmax(logits_first).item())
                second_pred = int(torch.argmax(logits_second).item())
                decision = {
                    "accept": first_pred == draft,
                    "first": first_pred,
                    "second": second_pred,
                }
            decision = broadcast_payload(decision)
            accept = bool(decision["accept"])
            if not accept:
                # roll back the second position's state mutations
                StageLane.restore_states(snapshots)
            new_draft = None
            if tail and mtp.active:
                if accept:
                    mtp.step(residual_first, decision["first"], verify_position)
                    new_draft = mtp.step(
                        residual_second, decision["second"], verify_position + 1
                    )
                else:
                    new_draft = mtp.step(
                        residual_first, decision["first"], verify_position
                    )
            new_draft = broadcast_payload({"draft": new_draft})["draft"]
            torch.cuda.synchronize(device)
            round_ms = (time.perf_counter() - round_started) * 1e3
            if accept:
                emitted.extend([decision["first"], decision["second"]])
                committed = verify_position + 1
                pending = decision["second"]
            else:
                emitted.append(decision["first"])
                committed = verify_position
                pending = decision["first"]
            draft = new_draft
            round_records.append(
                {
                    "accepted": accept,
                    "round_ms": round_ms,
                    "verify_ms": (verify_done - round_started) * 1e3,
                    "tokens": 2 if accept else 1,
                }
            )
    emitted = emitted[:n_tokens]

    result: dict[str, Any] = {
        "mode": mode,
        "prompt_len": prompt_len,
        "n_tokens": n_tokens,
        "emitted_tokens": emitted,
        "step_ms_mean": statistics.fmean(step_ms) if step_ms else None,
        "step_ms_p50": statistics.median(step_ms) if step_ms else None,
    }
    if round_records:
        accepted_rounds = sum(1 for r in round_records if r["accepted"])
        result.update(
            {
                "rounds": len(round_records),
                "accepted_rounds": accepted_rounds,
                "acceptance_rate": accepted_rounds / len(round_records),
                "round_ms_mean": statistics.fmean(
                    r["round_ms"] for r in round_records
                ),
                "round_ms_p50": statistics.median(
                    [r["round_ms"] for r in round_records]
                ),
                "verify_ms_mean": statistics.fmean(
                    r["verify_ms"] for r in round_records
                ),
                "tokens_generated": sum(r["tokens"] for r in round_records) + 1,
                "effective_ms_per_token": (
                    sum(r["round_ms"] for r in round_records)
                    / max(1, sum(r["tokens"] for r in round_records))
                ),
                "round_records": round_records,
            }
        )
    return result


# --------------------------------------------------------------------------


def tokenizer_preflight(
    stage_root: Path, golden_prompts: list[dict[str, Any]]
) -> dict[str, Any]:
    sys.path.insert(0, str(stage_root / "encoding"))
    from encoding_dsv4 import encode_messages  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(str(stage_root))
    checks = []
    for entry in golden_prompts:
        encoded = tokenizer.encode(
            encode_messages(
                [{"role": "user", "content": entry["prompt"]}],
                thinking_mode="chat",
            )
        )
        checks.append(
            {
                "prompt": entry["prompt"][:48],
                "prompt_len": len(entry["prompt_tokens"]),
                "retokenized_equal": encoded == entry["prompt_tokens"],
            }
        )
    return {
        "eos_token_id": int(tokenizer.eos_token_id),
        "checks": checks,
        "accepted": all(check["retokenized_equal"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--oracle-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--arms", type=str, default=",".join(ARMS),
        help="comma list of arms to run",
    )
    parser.add_argument(
        "--hc-backend", type=str, default="fused", choices=("eager", "fused")
    )
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=MAX_COMPARE_STEPS)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument(
        "--kv-dtype", type=str, default="fp8",
        choices=("bf16", "fp8", "fp8_rope_bf16"),
    )
    parser.add_argument(
        "--indexer-kv-dtype", type=str, default="bf16", choices=("bf16", "fp8")
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0mtp2e-mtp-draft-verify-golden-gate",
        "measurement_class": "semantic_e2e_golden_token_gate_plus_eager_timing",
        "rank": rank,
        "local_rank": local_rank,
        "world": world,
        "host": platform.node(),
        "kv_dtype": args.kv_dtype,
        "indexer_kv_dtype": args.indexer_kv_dtype,
        "hc_backend": args.hc_backend,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "arms": arms,
        "stage_layer_ids": {str(s): list(v) for s, v in STAGE_LAYERS.items()},
        "mtp_layer_id": MTP_LAYER_ID,
        "checkpoint_id": None,
        "placement": None,
        "tokenizer_preflight": None,
        "arm_results": {},
        "summary": None,
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }
    started = time.perf_counter()

    try:
        if world != WORLD:
            raise ValueError(f"E0mtp2e requires world=16, got {world}")
        for arm in arms:
            if arm not in ARMS:
                raise ValueError(f"unknown arm {arm!r}")
        topo = create_pp4_groups(rank)
        stage = topo["stage"]
        tp_rank = topo["tp_rank"]
        result["stage"] = stage
        result["tp_rank"] = tp_rank

        warm = torch.ones(1, device=device)
        dist.all_reduce(warm, group=topo["tp_group"])
        if topo["next_pair"] is not None:
            pair_transfer(warm, send=True, group=topo["next_pair"])
        if topo["prev_pair"] is not None:
            pair_transfer(warm, send=False, group=topo["prev_pair"])
        torch.cuda.synchronize(device)
        result["tp_group_binding"] = validate_live_tp_group(
            topo["tp_group"],
            expected_local_rank=tp_rank,
            expected_global_ranks=topo["tp_global_ranks"],
        )
        result["placement"] = run_placement_check(stage=stage, world=world)
        if not result["placement"]["accepted"]:
            raise ValueError(f"PP4 placement violated: {result['placement']}")
        if rank == 0:
            print(f"[E0mtp2e] placement {result['placement']['stage_hosts']}", flush=True)

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root,
                    list(range(MODEL_LAYERS)) + [MTP_LAYER_ID],
                    EXPECTED_TP_SIZE,
                )
                if not checkpoint["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {checkpoint['errors'][:4]}"
                    )
                oracle = json.loads(
                    args.oracle_json.expanduser().read_text(encoding="utf-8")
                )
                golden_prompts = oracle["prompts"]
                if args.max_prompts > 0:
                    golden_prompts = golden_prompts[: args.max_prompts]
                tokenizer_check = tokenizer_preflight(stage_root, golden_prompts)
                if not tokenizer_check["accepted"]:
                    raise ValueError("tokenizer parity failed")
                envelope_holder[0] = {
                    "ok": True,
                    "config": config_payload,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "tokenizer_preflight": tokenizer_check,
                    "prompts": [
                        {
                            "prompt": entry["prompt"],
                            "prompt_tokens": entry["prompt_tokens"],
                            "completion_tokens": entry["completion_tokens"],
                        }
                        for entry in golden_prompts
                    ],
                }
            except Exception:
                envelope_holder[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        if not envelope["ok"]:
            raise ValueError(f"rank-0 preflight failed:\n{envelope['error']}")
        result["checkpoint_id"] = envelope["checkpoint_id"]
        result["tokenizer_preflight"] = {
            "accepted": envelope["tokenizer_preflight"]["accepted"],
            "eos_token_id": envelope["tokenizer_preflight"]["eos_token_id"],
        }
        model_config = envelope["config"]
        prompts = envelope["prompts"]

        # MoE row registration: decode rows (4), fused verify rows (8), one
        # prefill shape per distinct prompt length; MTP additionally needs the
        # (length - 1) prefill shapes.
        prompt_lengths = sorted({len(entry["prompt_tokens"]) for entry in prompts})
        main_rows = sorted(
            {EXPECTED_TP_SIZE, 2 * EXPECTED_TP_SIZE}
            | {EXPECTED_TP_SIZE * length for length in prompt_lengths}
        )
        mtp_rows = sorted(
            {EXPECTED_TP_SIZE}
            | {
                EXPECTED_TP_SIZE * (length - 1)
                for length in prompt_lengths
                if length >= 2
            }
        )
        result["global_row_shapes"] = {"main": main_rows, "mtp": mtp_rows}

        phase_started = time.perf_counter()
        stage_material = build_physical_stage(
            stage_id=stage,
            layer_ids=STAGE_LAYERS[stage],
            model_config=model_config,
            stage_root=stage_root,
            tp_rank=tp_rank,
            tp_group=topo["tp_group"],
            tp_global_ranks=topo["tp_global_ranks"],
            device=device,
            checkpoint_id=result["checkpoint_id"],
            max_seq_len=MAX_SEQ_LEN,
            global_row_shapes=tuple(main_rows),
            slots_per_shape=1,
            kv_dtype=args.kv_dtype,
            indexer_kv_dtype=args.indexer_kv_dtype,
            progress_every=args.progress_every,
            progress=(
                (lambda message: print(f"[E0mtp2e] {message}", flush=True))
                if rank in (0, 4, 8, 12)
                else None
            ),
        )
        embed_material = None
        head_material = None
        mtp_material = None
        if stage == 0:
            embed_material = load_embed_head_material(
                stage_root=stage_root,
                device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=True,
                load_head=False,
            )
        elif stage == STAGE_COUNT - 1:
            # tail loads BOTH: the head for logits and the embedding for the
            # MTP bridge (shared tables, reference model.py:792-793).
            head_material = load_embed_head_material(
                stage_root=stage_root,
                device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=True,
                load_head=True,
            )
            needs_mtp = any(arm.startswith("mtp") for arm in arms)
            if needs_mtp:
                mtp_material = build_mtp_layer_material(
                    model_config=model_config,
                    stage_root=stage_root,
                    tp_rank=tp_rank,
                    tp_group=topo["tp_group"],
                    tp_global_ranks=topo["tp_global_ranks"],
                    device=device,
                    checkpoint_id=result["checkpoint_id"],
                    max_seq_len=MAX_SEQ_LEN,
                    global_row_shapes=tuple(mtp_rows),
                    slots_per_shape=1,
                    kv_dtype=args.kv_dtype,
                    progress=(
                        (lambda message: print(f"[E0mtp2e] mtp {message}", flush=True))
                        if rank == 12
                        else None
                    ),
                )
        result["diagnostic_seconds"]["load"] = time.perf_counter() - phase_started
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_after_load"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        if rank in (0, 4, 8, 12):
            print(
                f"[E0mtp2e] stage {stage} loaded "
                f"(free {free_bytes / 2**30:.2f} GiB, "
                f"{result['diagnostic_seconds']['load']:.0f}s)",
                flush=True,
            )
        dist.barrier()

        backend = resolve_hc_boundary_backend(
            None if args.hc_backend == "eager" else "fused"
        )

        # ------------------------------------------------------------------
        for arm in arms:
            arm_started = time.perf_counter()
            arm_rows: list[dict[str, Any]] = []
            for prompt_index, entry in enumerate(prompts):
                prompt_tokens = [int(t) for t in entry["prompt_tokens"]]
                golden_tokens = [int(t) for t in entry["completion_tokens"]]
                compare_steps = min(
                    len(golden_tokens), MAX_COMPARE_STEPS, args.max_steps
                )
                lane = StageLane(
                    stage_material.materials, backend=backend, device=device
                )
                mtp_driver = MTPDriver(
                    mtp_material=mtp_material,
                    embed_head=head_material,
                    device=device,
                    active=arm.startswith("mtp") and stage == STAGE_COUNT - 1,
                )
                if arm in ("off_teacher", "mtp_teacher"):
                    record = run_prompt_teacher(
                        prompt_tokens=prompt_tokens,
                        golden_tokens=golden_tokens,
                        compare_steps=compare_steps,
                        lane=lane,
                        mtp=mtp_driver,
                        topo=topo,
                        embed_material=embed_material,
                        head_material=head_material,
                        device=device,
                    )
                else:
                    mode = {
                        "off_free": "off",
                        "mtp_free": "chained",
                        "mtp_fused": "fused",
                    }[arm]
                    record = run_prompt_free(
                        prompt_tokens=prompt_tokens,
                        n_tokens=compare_steps,
                        mode=mode,
                        lane=lane,
                        mtp=mtp_driver,
                        topo=topo,
                        embed_material=embed_material,
                        head_material=head_material,
                        device=device,
                    )
                del lane, mtp_driver
                record["prompt_index"] = prompt_index
                record["arm"] = arm
                holder: list[Any] = [record if rank == CANONICAL_RANK else None]
                dist.broadcast_object_list(holder, src=CANONICAL_RANK)
                canonical = holder[0]
                arm_rows.append(canonical)
                if rank == 0:
                    brief = {
                        key: canonical.get(key)
                        for key in (
                            "matched",
                            "compare_steps",
                            "draft_acceptance_rate",
                            "acceptance_rate",
                            "rounds",
                            "step_ms_mean",
                            "round_ms_mean",
                            "effective_ms_per_token",
                        )
                        if canonical.get(key) is not None
                    }
                    print(
                        f"[E0mtp2e][{arm}] prompt {prompt_index}: {brief}",
                        flush=True,
                    )
            result["arm_results"][arm] = arm_rows
            result["diagnostic_seconds"][f"arm_{arm}"] = (
                time.perf_counter() - arm_started
            )

        # ------------------------------------------------------------------
        # summary + acceptance
        summary: dict[str, Any] = {}
        arm_results = result["arm_results"]

        def total_matched(arm: str) -> tuple[int, int]:
            rows = arm_results.get(arm, [])
            return (
                sum(int(row.get("matched", 0)) for row in rows),
                sum(int(row.get("compare_steps", 0)) for row in rows),
            )

        if "off_teacher" in arm_results:
            matched, total = total_matched("off_teacher")
            summary["off_teacher_golden"] = {"matched": matched, "total": total}
        if "mtp_teacher" in arm_results:
            matched, total = total_matched("mtp_teacher")
            rows = arm_results["mtp_teacher"]
            events = sum(int(row.get("draft_events", 0)) for row in rows)
            accepted = sum(int(row.get("draft_accepted", 0)) for row in rows)
            summary["mtp_teacher_golden"] = {"matched": matched, "total": total}
            summary["mtp_teacher_acceptance"] = {
                "events": events,
                "accepted": accepted,
                "rate": accepted / events if events else None,
                "per_prompt": [
                    {
                        "prompt_index": row["prompt_index"],
                        "events": row.get("draft_events"),
                        "accepted": row.get("draft_accepted"),
                        "rate": row.get("draft_acceptance_rate"),
                    }
                    for row in rows
                ],
            }
            if "off_teacher" in arm_results:
                identical = all(
                    off.get("predicted_tokens") == mtp.get("predicted_tokens")
                    for off, mtp in zip(
                        arm_results["off_teacher"], rows, strict=True
                    )
                )
                summary["teacher_streams_identical"] = bool(identical)
        for arm in ("mtp_free", "mtp_fused"):
            if arm in arm_results and "off_free" in arm_results:
                diffs = []
                for off, mtp in zip(
                    arm_results["off_free"], arm_results[arm], strict=True
                ):
                    off_tokens = off["emitted_tokens"]
                    mtp_tokens = mtp["emitted_tokens"]
                    diffs.append(
                        {
                            "prompt_index": mtp["prompt_index"],
                            "equal": off_tokens == mtp_tokens,
                            "diverged_positions": sum(
                                1
                                for a, b in zip(off_tokens, mtp_tokens, strict=True)
                                if a != b
                            ),
                            "acceptance_rate": mtp.get("acceptance_rate"),
                        }
                    )
                rounds = sum(int(row.get("rounds", 0)) for row in arm_results[arm])
                accepted_rounds = sum(
                    int(row.get("accepted_rounds", 0)) for row in arm_results[arm]
                )
                summary[f"{arm}_vs_off_free"] = {
                    "all_equal": all(d["equal"] for d in diffs),
                    "per_prompt": diffs,
                    "rounds": rounds,
                    "accepted_rounds": accepted_rounds,
                    "acceptance_rate": (
                        accepted_rounds / rounds if rounds else None
                    ),
                }
        if "off_free" in arm_results:
            step_means = [
                row["step_ms_mean"]
                for row in arm_results["off_free"]
                if row.get("step_ms_mean")
            ]
            summary["off_free_step_ms_mean"] = (
                statistics.fmean(step_means) if step_means else None
            )
        if "mtp_fused" in arm_results:
            round_means = [
                row["round_ms_mean"]
                for row in arm_results["mtp_fused"]
                if row.get("round_ms_mean")
            ]
            eff = [
                row["effective_ms_per_token"]
                for row in arm_results["mtp_fused"]
                if row.get("effective_ms_per_token")
            ]
            summary["mtp_fused_round_ms_mean"] = (
                statistics.fmean(round_means) if round_means else None
            )
            summary["mtp_fused_effective_ms_per_token"] = (
                statistics.fmean(eff) if eff else None
            )
        result["summary"] = summary

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_at_end"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

        accepted = bool(
            result["placement"]["accepted"]
            and result["tokenizer_preflight"]["accepted"]
            and all(
                len(arm_results.get(arm, [])) == len(prompts) for arm in arms
            )
        )
        if "mtp_teacher" in arms and "off_teacher" in arms:
            accepted = accepted and bool(summary.get("teacher_streams_identical"))
        if "mtp_free" in arms and "off_free" in arms:
            accepted = accepted and bool(
                summary.get("mtp_free_vs_off_free", {}).get("all_equal")
            )
        result["accepted"] = accepted
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
    result["diagnostic_seconds"]["process"] = time.perf_counter() - started

    try:
        gathered: list[Any] = [None for _ in range(world)]
        dist.all_gather_object(gathered, result["accepted"])
        accepted_all = bool(
            len(gathered) == world and all(bool(value) for value in gathered)
        )
    except Exception:
        result["errors"].append(traceback.format_exc())
        accepted_all = False
    write_json(out_dir / f"rank{rank}.json", result)
    if rank == 0:
        write_json(
            out_dir / "result.json",
            {
                "experiment": "E0mtp2e-mtp-draft-verify-golden-gate",
                "accepted": accepted_all,
                "checkpoint_id": result["checkpoint_id"],
                "kv_dtype": result["kv_dtype"],
                "hc_backend": result["hc_backend"],
                "summary": result.get("summary"),
                "placement": result["placement"],
                "tokenizer_preflight": result["tokenizer_preflight"],
                "errors": result["errors"],
            },
        )
        print(f"[E0mtp2e] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
