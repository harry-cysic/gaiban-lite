#!/usr/bin/env python3
"""E0e2e: full-model 43-layer TP4xPP4 dual-node golden-token E2E gate.

Tenth port vertical (E2E close-out): the complete DeepSeek-V4-Flash decode
path -- embedding entry, all 43 physical blocks, final HC collapse + norm +
head -- runs as a 16-rank TP4xPP4 pipeline over titan064+titan065 and is
judged against the D0 reference oracle's golden greedy tokens
(``experiments/D0-reference-oracle/results/oracle-mp8.json``).

Topology (feasibility section 3 + B.1):
  stage 0 = titan064 GPU0-3 (socket 0)  L0-L10   + embedding
  stage 1 = titan064 GPU4-7 (socket 1)  L11-L21  (IB-boundary sender, NIC
                                                  affinity on socket 1)
  stage 2 = titan065 GPU0-3 (socket 0)  L22-L32
  stage 3 = titan065 GPU4-7 (socket 1)  L33-L42  + final norm/head

Execution form (documented trade-off): the E0wf/E0ef-verified ``__call__``
attention paths (real full-sequence prefill at position 0, then per-token
decode at arbitrary positions -- the padded window branch below position
127 was verified in E0wf/E0ef) drive every layer; ratio-4 layers use the
new ``Ratio4FullPositionAttention`` (see ``dsv4_direct/ratio4_fullpos.py``;
its decode step is the operator mirror of the E0ff-verified candidate,
certified bitwise by ``e0e2e_ratio4_selfcheck.py``).  The plan/stateful
decode paths are frozen to saturated positions >= 128 and are not usable
for real prompts from position 0, so this gate does not use them.  The
prompt is prefetched in one full-sequence forward (the reference oracle's
own prefill form) rather than teacher-forced token-by-token from zero.

Decode is teacher-forced on the golden tokens: at every completion position
the runtime argmax is compared against the golden token, mismatches are
recorded (position, both tokens, top-2 logit gap, golden-token logit
deficit), and the *golden* token is fed regardless, so every position is
compared and the total mismatch count is exact.  For 128-token completions
the trailing synthetic EOS appended by ``generate.py`` is excluded
(compare_steps = min(len(completion_tokens), 128)).

The HC boundary backend is selectable (``--hc-backends eager,fused``); the
fused TileLang boundary (E0hf) applies to decode steps (sequence length 1,
its quantified shape); prefill always runs the eager composition in both
modes, so fused-vs-eager token differences are attributable to decode-step
boundary math alone.

MTP is off (the D0 oracle's generate path never invokes MTP).  B=1, eager,
no performance claims.

Run (driven by ``run_e0e2e_dual.sh``):
  torchrun --nnodes 2 --node-rank {0|1} --nproc-per-node 8 \
    --master-addr 10.234.1.64 --master-port 29641 \
    e0ef2e_golden_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash \
    --oracle-json oracle-mp8.json --out-dir out-e0e2e
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
    load_embed_head_material,
)
from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.hyper_connections import hc_post, hc_pre
from dsv4_direct.physical_stage import (
    EXPECTED_TP_SIZE,
    PhysicalLayerMaterial,
    build_physical_stage,
    validate_live_tp_group,
)
from dsv4_direct.ratio4_fullpos import Ratio4FullPositionAttention


WORLD = 16
STAGE_COUNT = 4
MODEL_LAYERS = 43
STAGE_LAYERS: dict[int, tuple[int, ...]] = {
    0: tuple(range(0, 11)),
    1: tuple(range(11, 22)),
    2: tuple(range(22, 33)),
    3: tuple(range(33, 43)),
}
MAX_SEQ_LEN = 256  # D0 default: prompts <= 22 + 128 decode steps; multiple of 128
MAX_COMPARE_STEPS = 128
LOCAL_BATCH = 1
HIDDEN = 4096
HC_MULT = 4


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
# topology


def create_pp4_groups(rank: int) -> dict[str, Any]:
    """Four TP4 stage groups + one boundary pair group per (boundary, tp)."""

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
    prev_pair = (
        groups[STAGE_COUNT + (stage - 1) * 4 + tp_rank] if stage > 0 else None
    )
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
    """One fixed-shape send/recv on a 2-rank boundary group (E0qf P2P form)."""

    if not tensor.is_contiguous():
        raise ValueError("pair transfer requires a contiguous tensor")
    # Boundary groups are (lower_stage_rank, higher_stage_rank): the sender
    # is always group rank 0 and the receiver group rank 1.
    operation = dist.isend if send else dist.irecv
    works = dist.batch_isend_irecv(
        [
            dist.P2POp(
                operation, tensor, group=group, group_peer=1 if send else 0
            )
        ]
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


# --------------------------------------------------------------------------
# per-prompt stage lane (fresh KV state over shared weight materials)


class StageLane:
    def __init__(
        self,
        materials: Sequence[PhysicalLayerMaterial],
        *,
        backend: Any | None,
        device: torch.device,
        ratio4_index_mode: str = "ref",
        fuse_min_seqlen: int = 1024,
        fused_scope: str = "decode",
    ) -> None:
        self.device = device
        self.backend = backend
        # C2F 23rd vertical lever A: the fused boundary was previously gated to
        # decode shapes (A5F's quantified regime).  With the >=1024-row kernel
        # branch avoided (see FusedTilelangHCBoundaryBackend.MAX_ROWS) the same
        # chain is valid at prefill shapes.  The scope is explicit so a prefill
        # arm can be compared against the frozen eager-everywhere golden with
        # decode held fixed.
        if fused_scope not in ("decode", "prefill", "both"):
            raise ValueError(f"unknown fused scope {fused_scope!r}")
        self.fused_scope = fused_scope
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
                    index_score_mode=ratio4_index_mode,
                    fuse_min_seqlen=fuse_min_seqlen,
                    tp_group=(
                        material.moe.group
                        if getattr(material.attention_config, "tp_size", 1) > 1
                        else None
                    ),
                )
            else:
                state = material.new_state(num_local_sequences=LOCAL_BATCH)
                attention = material.new_attention(state)
            self.layers.append((material, attention))

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
        """One block in the default eager op order (== DirectDecodeBlock)."""

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
        """E0hf fused-boundary chain: stage entry/exit eager, boundaries fused."""

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
        # Fused boundary is applied on decode shapes by default (A5F's
        # quantified regime); --fused-scope extends or moves it (C2F 23rd
        # vertical lever A).
        is_decode = residual.shape[1] == 1
        scope = self.fused_scope
        if self.backend is not None and (
            (is_decode and scope in ("decode", "both"))
            or (not is_decode and scope in ("prefill", "both"))
        ):
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


# --------------------------------------------------------------------------
# prefill-path evidence
#
# D0L (24th vertical): with 10-22 token prompts the prefill row count never
# reaches any of the thresholds that prefill-only code paths switch on, so the
# gate could not tell "the lever ran and was harmless" from "the lever never
# ran".  These helpers read the pure-observability counters on the HC boundary
# backend and on each layer's TP4MoE so a run can *prove* which path executed.


def reset_path_evidence(lane: StageLane) -> None:
    backend = lane.backend
    if backend is not None and hasattr(backend, "reset_stats"):
        backend.reset_stats()
    for material, _ in lane.layers:
        for key in material.moe.overlap_stats:
            material.moe.overlap_stats[key] = 0


def collect_path_evidence(lane: StageLane) -> dict[str, Any]:
    backend = lane.backend
    hc: dict[str, Any] | None = None
    if backend is not None and hasattr(backend, "call_stats"):
        stats = backend.call_stats
        hc = {
            "calls": int(stats["calls"]),
            "split_calls": int(stats["split_calls"]),
            "kernel_launches": int(stats["kernel_launches"]),
            "max_rows": backend.max_rows,
            "row_histogram": {
                str(rows): int(count)
                for rows, count in sorted(stats["row_histogram"].items())
            },
        }
    moe = {
        "overlapped_calls": 0,
        "overlapped_rows": 0,
        "sequential_calls": 0,
        "sequential_rows": 0,
    }
    for material, _ in lane.layers:
        for key in moe:
            moe[key] += int(material.moe.overlap_stats[key])
    return {"hc_boundary": hc, "moe": moe}


# --------------------------------------------------------------------------
# per-prompt pipeline run


def run_prompt(
    *,
    prompt_index: int,
    prompt_tokens: list[int],
    golden_tokens: list[int],
    compare_steps: int,
    lane: StageLane,
    topo: dict[str, Any],
    embed_material: EmbedHeadMaterial | None,
    head_material: EmbedHeadMaterial | None,
    device: torch.device,
    mode: str,
    prefill_chunk: int = 0,
) -> dict[str, Any]:
    stage = topo["stage"]
    prompt_len = len(prompt_tokens)
    steps: list[dict[str, Any]] = []
    decode_ms: list[float] = []
    lane_agreement = True

    prefill_evidence: dict[str, Any] | None = None
    reset_path_evidence(lane)
    # Per-prompt activation peak.  Resident weights are already allocated, so
    # what this measures on top of them is the forward's own workspace -- the
    # quantity chunked prefill is supposed to bound.
    torch.cuda.reset_peak_memory_stats(device)
    resident_gib = torch.cuda.memory_allocated(device) / (1024 ** 3)

    # 25th vertical: the prompt may be prefilled incrementally.  ``--prefill-chunk 0``
    # (default) keeps the single whole-sequence forward the gate always ran, so
    # every previously frozen arm is byte-identical.  With a chunk size the
    # prompt is split into consecutive ``start_pos > 0`` multi-token forwards;
    # only the last one produces the step-0 comparison logits, because
    # ``head_logits`` reads the final position (head_stage.py:262).
    plan: list[tuple[int, list[int]]] = []
    if prefill_chunk and prefill_chunk < prompt_len:
        position = 0
        while position < prompt_len:
            length = min(prefill_chunk, prompt_len - position)
            plan.append((position, prompt_tokens[position : position + length]))
            position += length
    else:
        plan.append((0, prompt_tokens))
    prefill_forwards = len(plan)
    prefill_chunk_lengths = [len(item[1]) for item in plan]
    for step in range(1, compare_steps):
        plan.append((prompt_len + step - 1, [golden_tokens[step - 1]]))

    prefill_wall_ms = 0.0
    for plan_index, (position, step_tokens) in enumerate(plan):
        # Comparison steps start at the *last* prefill forward.
        step = plan_index - (prefill_forwards - 1)
        seqlen = len(step_tokens)
        started = time.perf_counter()

        input_ids = None
        if stage == 0:
            input_ids = torch.tensor(
                [step_tokens], dtype=torch.int64, device=device
            )
            residual = embed_hc_residual(embed_material, input_ids)
        else:
            residual = torch.empty(
                (LOCAL_BATCH, seqlen, HC_MULT, HIDDEN),
                dtype=torch.bfloat16,
                device=device,
            )
            pair_transfer(residual, send=False, group=topo["prev_pair"])
        residual = lane.forward(
            residual, start_pos=position, input_ids=input_ids
        )
        if stage < STAGE_COUNT - 1:
            pair_transfer(
                residual.contiguous(), send=True, group=topo["next_pair"]
            )
        elif step < 0:
            # Intermediate prefill chunk: nothing to compare, because
            # ``head_logits`` reads only the final position (head_stage.py:262).
            # Every tail lane skips the same collectives, so no rank diverges.
            pass
        else:
            logits = head_logits(head_material, residual)
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError(
                    f"non-finite logits at prompt {prompt_index} step {step}"
                )
            top2 = torch.topk(logits[0], 2)
            predicted = int(top2.indices[0].item())
            golden = int(golden_tokens[step])
            record = {
                "step": step,
                "position": int(position + seqlen - 1),
                "predicted": predicted,
                "golden": golden,
                "match": predicted == golden,
                "top1_logit": float(top2.values[0].item()),
                "top2_logit": float(top2.values[1].item()),
                "top2_gap": float((top2.values[0] - top2.values[1]).item()),
                "golden_logit": float(logits[0, golden].item()),
            }
            record["golden_deficit"] = record["top1_logit"] - record["golden_logit"]
            steps.append(record)
            # TP-lane agreement: all four tail lanes should argmax alike;
            # divergence would flag cross-lane bf16 drift at the token level.
            lane_predictions: list[Any] = [None] * EXPECTED_TP_SIZE
            dist.all_gather_object(
                lane_predictions, predicted, group=topo["tp_group"]
            )
            if len(set(lane_predictions)) != 1:
                lane_agreement = False
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        if step <= 0:
            prefill_wall_ms += elapsed_ms
            if step == 0:
                # Snapshot before any decode step can pollute the counters.
                # The counters cover every prefill forward, because
                # ``reset_path_evidence`` ran before the first one.
                prefill_evidence = collect_path_evidence(lane)
                prefill_evidence["seqlen"] = prompt_len
                prefill_evidence["rows_per_lane"] = LOCAL_BATCH * prompt_len
                prefill_evidence["moe_global_rows"] = (
                    LOCAL_BATCH * prompt_len * EXPECTED_TP_SIZE
                )
                prefill_evidence["prefill_forwards"] = prefill_forwards
                prefill_evidence["chunk_lengths"] = prefill_chunk_lengths
                prefill_evidence["max_chunk_rows_per_lane"] = (
                    LOCAL_BATCH * max(prefill_chunk_lengths)
                )
                prefill_evidence["wall_ms"] = prefill_wall_ms
                prefill_evidence["peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                )
        else:
            decode_ms.append(elapsed_ms)

    # cross-lane residual digest at the final step (diagnostic only)
    digest = tensor_sha256(residual)
    digests: list[Any] = [None] * EXPECTED_TP_SIZE
    dist.all_gather_object(digests, digest, group=topo["tp_group"])

    result: dict[str, Any] = {
        "prompt_index": prompt_index,
        "mode": mode,
        "stage": stage,
        "prompt_len": prompt_len,
        "compare_steps": compare_steps,
        "final_residual_lanes_bitwise": len(set(digests)) == 1,
        "decode_ms_mean": (
            statistics.fmean(decode_ms) if decode_ms else None
        ),
        "prefill_evidence": prefill_evidence,
        "run_evidence": collect_path_evidence(lane),
    }
    if stage == STAGE_COUNT - 1:
        mismatches = [record for record in steps if not record["match"]]
        result.update(
            {
                "steps": steps,
                "matched": sum(1 for record in steps if record["match"]),
                "mismatches": mismatches,
                "first_mismatch": mismatches[0] if mismatches else None,
                "lane_argmax_agreement": lane_agreement,
                "predicted_tokens": [record["predicted"] for record in steps],
            }
        )
    return result


# --------------------------------------------------------------------------


def tokenizer_preflight(
    stage_root: Path, golden_prompts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Re-tokenize every prompt on the checkpoint tokenizer + reference
    encoding and demand exact agreement with the golden prompt_tokens."""

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
        "--hc-backends",
        type=str,
        default="eager,fused",
        help="comma list of HC boundary modes to run (eager|fused)",
    )
    parser.add_argument(
        "--moe-overlap-blocks", type=int, default=0,
        help="C2F 23rd vertical lever B: row-block count for the pipelined "
        "TP4 MoE collectives (0/1 = the sequential path)",
    )
    parser.add_argument(
        "--fused-scope", choices=["decode", "prefill", "both"], default="decode",
        help="C2F 23rd vertical lever A: which shapes the fused HC boundary "
        "chain covers.  'decode' is the frozen behaviour; 'prefill' is the "
        "lever-A arm (decode held eager so the only delta vs the frozen "
        "golden is prefill HC)",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=0, help="0 = all golden prompts"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=MAX_SEQ_LEN,
        help="attention state capacity (multiple of 128).  The D0 default 256 "
        "covers 22-token prompts + 128 decode steps; the D0L long-prompt oracle "
        "needs prompt_len + decode steps",
    )
    parser.add_argument(
        "--prompt-min-tokens", type=int, default=0,
        help="D0L: keep only golden prompts with at least this many prompt "
        "tokens",
    )
    parser.add_argument(
        "--prompt-max-tokens", type=int, default=0,
        help="D0L: keep only golden prompts with at most this many prompt "
        "tokens (0 = no cap).  Prompt lengths are bucketed across runs because "
        "every distinct length registers its own Marlin per-shape buffer set "
        "(~80 KiB per global row: a 8192-token prompt alone is 2.5 GiB)",
    )
    parser.add_argument(
        "--share-moe-buffers", action="store_true",
        help="D0L: share one Marlin per-shape buffer set across the stage's "
        "layers of the same route kind.  Buffers are scratch and layers run "
        "serially, so this is output-identical; without it an 11-layer stage "
        "at a 32768-row prefill shape needs ~29 GB and OOMs at load",
    )
    parser.add_argument(
        "--prefill-chunk", type=int, default=0,
        help="C3F: split the prompt prefill into consecutive chunks of this "
        "many tokens (0 = one whole-sequence prefill, the frozen behaviour).  "
        "Chunks after the first enter the runtime at start_pos > 0 with "
        "seqlen > 1, i.e. the incremental path added in the 25th vertical",
    )
    parser.add_argument(
        "--max-steps", type=int, default=MAX_COMPARE_STEPS,
        help="cap on compared completion steps per prompt",
    )
    parser.add_argument(
        "--attention-tp-shard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="E6F variant A: shard the attention o-path across TP4 (not bitwise)",
    )
    parser.add_argument(
        "--embed-device", default=None,
        help=(
            "device for the 1010 MiB embedding table (e.g. 'cpu').  Stage 0 "
            "OOMs loading it at --max-seq-len 8320; the lookup is a pure "
            "gather so an off-device table is bitwise identical."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument(
        "--kv-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp8", "fp8_rope_bf16"),
        help="latent KV storage dtype for every layer kind (FP8 KV E2E arm)",
    )
    parser.add_argument(
        "--indexer-kv-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp8"),
        help="ratio-4 indexer_kv storage dtype",
    )
    parser.add_argument(
        "--ratio4-index-mode",
        type=str,
        default="ref",
        choices=("ref", "fused"),
        help="C2F arm: ratio-4 prefill index score backend (fused = D0b Triton)",
    )
    parser.add_argument(
        "--fuse-min-seqlen",
        type=int,
        default=1024,
        help="C2F arm: minimum prefill seqlen for the fused index score "
        "(lower it below prompt lengths to exercise fused prefill in E2E)",
    )
    parser.add_argument(
        "--moe-input-dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp8"),
        help="C2F arm: Marlin MoE activation dtype (fp8 = W4A8 repack; "
        "applies to every MoE call in this gate)",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # 30 min, not 120: no single collective here legitimately takes even a
    # minute (stage load is ~12s, the 8192 prefill a few minutes), so a long
    # wait always means a rank-mismatch deadlock.  The timeout is the backstop
    # that turns that into a crash instead of an indefinite hang.
    dist.init_process_group(
        "nccl", device_id=device, timeout=timedelta(minutes=30)
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    modes = [mode.strip() for mode in args.hc_backends.split(",") if mode.strip()]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E0e2e-tp4-pp4-golden-token-gate",
        "measurement_class": "semantic_e2e_golden_token_gate",
        "rank": rank,
        "local_rank": local_rank,
        "world": world,
        "host": platform.node(),
        "kv_dtype": args.kv_dtype,
        "indexer_kv_dtype": args.indexer_kv_dtype,
        "ratio4_index_mode": args.ratio4_index_mode,
        "fuse_min_seqlen": args.fuse_min_seqlen,
        "moe_input_dtype": args.moe_input_dtype,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "modes": modes,
        "stage_layer_ids": {str(s): list(v) for s, v in STAGE_LAYERS.items()},
        "nccl_env": {
            key: os.environ.get(key)
            for key in (
                "NCCL_SOCKET_IFNAME",
                "NCCL_IB_DISABLE",
                "NCCL_P2P_LEVEL",
                "NCCL_NET_GDR_LEVEL",
            )
        },
        "checkpoint_id": None,
        "placement": None,
        "tokenizer_preflight": None,
        "prompt_results": {},
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }
    started = time.perf_counter()
    summary_rows: list[dict[str, Any]] = []

    try:
        if world != WORLD:
            raise ValueError(f"E0e2e requires world=16 (2 nodes x 8), got {world}")
        for mode in modes:
            if mode not in ("eager", "fused"):
                raise ValueError(f"unknown HC backend mode {mode!r}")
        topo = create_pp4_groups(rank)
        stage = topo["stage"]
        tp_rank = topo["tp_rank"]
        result["stage"] = stage
        result["tp_rank"] = tp_rank

        # warmups: one collective per TP group, one P2P per boundary group.
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
            print(
                f"[E0e2e] placement {result['placement']['stage_hosts']}",
                flush=True,
            )

        # rank-0 preflight: full 43-layer + top-level checkpoint contract,
        # golden tokens, tokenizer parity.
        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root, list(range(MODEL_LAYERS)), EXPECTED_TP_SIZE
                )
                if not checkpoint["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {checkpoint['errors'][:4]}"
                    )
                oracle = json.loads(
                    args.oracle_json.expanduser().read_text(encoding="utf-8")
                )
                golden_prompts = oracle["prompts"]
                if args.prompt_min_tokens > 0:
                    golden_prompts = [
                        entry
                        for entry in golden_prompts
                        if len(entry["prompt_tokens"]) >= args.prompt_min_tokens
                    ]
                if args.prompt_max_tokens > 0:
                    golden_prompts = [
                        entry
                        for entry in golden_prompts
                        if len(entry["prompt_tokens"]) <= args.prompt_max_tokens
                    ]
                if not golden_prompts:
                    raise ValueError(
                        "prompt length filter selected no golden prompts"
                    )
                if args.max_prompts > 0:
                    golden_prompts = golden_prompts[: args.max_prompts]
                tokenizer_check = tokenizer_preflight(stage_root, golden_prompts)
                if not tokenizer_check["accepted"]:
                    raise ValueError(
                        f"tokenizer parity failed: {tokenizer_check['checks']}"
                    )
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

        # MoE global-row registration: 4 rows per decode token plus one
        # prefill shape per distinct forward length.  With --prefill-chunk the
        # forwards are the chunks, not the whole prompt, so the registered
        # shapes have to be the *chunk* lengths (a missed shape makes the
        # Marlin per-shape buffer lookup miss at run time).
        prefill_lengths: set[int] = set()
        for entry in prompts:
            prompt_len = len(entry["prompt_tokens"])
            chunk = args.prefill_chunk
            if chunk and chunk < prompt_len:
                position = 0
                while position < prompt_len:
                    prefill_lengths.add(min(chunk, prompt_len - position))
                    position += chunk
            else:
                prefill_lengths.add(prompt_len)
        prefill_rows = sorted(
            {EXPECTED_TP_SIZE * length for length in prefill_lengths}
        )
        global_row_shapes = tuple([EXPECTED_TP_SIZE] + prefill_rows)
        result["global_row_shapes"] = list(global_row_shapes)
        result["prompt_lengths"] = [
            len(entry["prompt_tokens"]) for entry in prompts
        ]
        result["max_seq_len"] = args.max_seq_len
        result["share_moe_buffers"] = bool(args.share_moe_buffers)
        # 9.11: record the RESOLVED value, not just what was asked for.  This
        # one has no *_mode naming convention, so mode_witness cannot find it,
        # and it is exactly the field whose absence let a dropped
        # --attention-tp-shard read as "the lever does nothing" (E6F step 8).
        result["attention_tp_shard"] = bool(args.attention_tp_shard)
        result["argv"] = list(sys.argv[1:])
        result["prefill_chunk"] = int(args.prefill_chunk)

        # The last decode step reads position prompt_len + compare_steps - 2 and
        # writes one row, so the state must admit prompt_len + compare_steps - 1.
        needed = max(
            len(entry["prompt_tokens"])
            + min(
                len(entry["completion_tokens"]), MAX_COMPARE_STEPS, args.max_steps
            )
            - 1
            for entry in prompts
        )
        if needed > args.max_seq_len:
            raise ValueError(
                f"--max-seq-len {args.max_seq_len} too small: longest prompt + "
                f"compared decode steps needs {needed}"
            )

        # ------------------------------------------------------------------
        # load stage materials + embed/head material
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
            max_seq_len=args.max_seq_len,
            global_row_shapes=global_row_shapes,
            slots_per_shape=1,
            attention_tp_shard=args.attention_tp_shard,
            share_moe_buffers=args.share_moe_buffers,
            kv_dtype=args.kv_dtype,
            indexer_kv_dtype=args.indexer_kv_dtype,
            moe_marlin_input_dtype=(
                torch.float8_e4m3fn if args.moe_input_dtype == "fp8" else None
            ),
            progress_every=args.progress_every,
            progress=(
                (lambda message: print(f"[E0e2e] {message}", flush=True))
                if rank in (0, 4, 8, 12)
                else None
            ),
        )
        # C2F 23rd vertical lever B: row-blocked MoE collective/compute
        # overlap.  The pipelined path only engages on prefill row counts --
        # the decode call has local_rows == local_batch, which is not divisible
        # by the block count in this gate's B=1 form, so decode falls back to
        # the sequential path automatically.
        if args.moe_overlap_blocks > 1:
            for material in stage_material.materials:
                material.moe.enable_collective_overlap(args.moe_overlap_blocks)
            result["moe_overlap_blocks"] = args.moe_overlap_blocks

        embed_material = None
        head_material = None
        if stage == 0:
            embed_material = load_embed_head_material(
                stage_root=stage_root,
                device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=True,
                load_head=False,
                embed_device=args.embed_device,
            )
        elif stage == STAGE_COUNT - 1:
            head_material = load_embed_head_material(
                stage_root=stage_root,
                device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=False,
                load_head=True,
            )
        result["diagnostic_seconds"]["load"] = time.perf_counter() - phase_started
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_after_load"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        if rank in (0, 4, 8, 12):
            print(
                f"[E0e2e] stage {stage} loaded "
                f"({len(STAGE_LAYERS[stage])} layers, "
                f"free {free_bytes / 2**30:.2f} GiB, "
                f"{result['diagnostic_seconds']['load']:.0f}s)",
                flush=True,
            )
        dist.barrier()

        # ------------------------------------------------------------------
        # golden-token runs
        for mode in modes:
            backend = resolve_hc_boundary_backend(
                None if mode == "eager" else "fused"
            )
            mode_started = time.perf_counter()
            for prompt_index, entry in enumerate(prompts):
                prompt_tokens = [int(t) for t in entry["prompt_tokens"]]
                golden_tokens = [int(t) for t in entry["completion_tokens"]]
                compare_steps = min(
                    len(golden_tokens), MAX_COMPARE_STEPS, args.max_steps
                )
                lane = StageLane(
                    stage_material.materials,
                    backend=backend,
                    device=device,
                    ratio4_index_mode=args.ratio4_index_mode,
                    fuse_min_seqlen=args.fuse_min_seqlen,
                    fused_scope=args.fused_scope,
                )
                record = run_prompt(
                    prompt_index=prompt_index,
                    prompt_tokens=prompt_tokens,
                    golden_tokens=golden_tokens,
                    compare_steps=compare_steps,
                    lane=lane,
                    topo=topo,
                    embed_material=embed_material,
                    head_material=head_material,
                    device=device,
                    mode=mode,
                    prefill_chunk=args.prefill_chunk,
                )
                del lane
                # tail-stage record travels to every rank for logging and the
                # rank-0 result file (rank 12 = stage3 tp_rank0 is canonical).
                holder: list[Any] = [record if rank == 12 else None]
                dist.broadcast_object_list(holder, src=12)
                canonical = holder[0]
                summary_rows.append(canonical)
                result["prompt_results"].setdefault(mode, []).append(
                    canonical if rank in (0, 12) else {
                        "prompt_index": prompt_index,
                        "matched": canonical["matched"],
                        "compare_steps": canonical["compare_steps"],
                    }
                )
                if rank == 0:
                    first = canonical["first_mismatch"]
                    print(
                        f"[E0e2e][{mode}] prompt {prompt_index}: "
                        f"{canonical['matched']}/{canonical['compare_steps']} "
                        "golden tokens matched"
                        + (
                            ""
                            if first is None
                            else (
                                f"; first mismatch step {first['step']} "
                                f"pred {first['predicted']} vs golden "
                                f"{first['golden']} top2_gap "
                                f"{first['top2_gap']:.4f} golden_deficit "
                                f"{first['golden_deficit']:.4f}"
                            )
                        )
                        + f" (decode {canonical['decode_ms_mean'] or 0:.0f} ms/step)",
                        flush=True,
                    )
            result["diagnostic_seconds"][f"mode_{mode}"] = (
                time.perf_counter() - mode_started
            )

        # ------------------------------------------------------------------
        # mode summaries + acceptance
        mode_summaries: dict[str, Any] = {}
        for mode in modes:
            rows = [row for row in summary_rows if row["mode"] == mode]
            total = sum(row["compare_steps"] for row in rows)
            matched = sum(row["matched"] for row in rows)
            all_mismatches = [
                mismatch for row in rows for mismatch in row["mismatches"]
            ]
            gaps = sorted(m["top2_gap"] for m in all_mismatches)
            deficits = sorted(m["golden_deficit"] for m in all_mismatches)
            mode_summaries[mode] = {
                "prompts": len(rows),
                "total_tokens": total,
                "matched_tokens": matched,
                "match_rate": matched / total if total else None,
                "mismatch_count": len(all_mismatches),
                "mismatch_top2_gap": {
                    "min": gaps[0] if gaps else None,
                    "median": gaps[len(gaps) // 2] if gaps else None,
                    "max": gaps[-1] if gaps else None,
                },
                "mismatch_golden_deficit": {
                    "min": deficits[0] if deficits else None,
                    "median": deficits[len(deficits) // 2] if deficits else None,
                    "max": deficits[-1] if deficits else None,
                },
                "lane_argmax_agreement": all(
                    row["lane_argmax_agreement"] for row in rows
                ),
                "per_prompt": [
                    {
                        "prompt_index": row["prompt_index"],
                        "prompt_len": row["prompt_len"],
                        "matched": row["matched"],
                        "compare_steps": row["compare_steps"],
                        "first_mismatch": row["first_mismatch"],
                        "prefill_evidence": row.get("prefill_evidence"),
                        # C3F: the argmax stream itself, so two arms can be
                        # compared position by position instead of only on
                        # their match counts against golden (two arms can
                        # score alike while predicting differently).
                        "predicted_tokens": row.get("predicted_tokens"),
                    }
                    for row in rows
                ],
                # D0L: aggregate proof that the prefill chunk regime was
                # actually entered on this arm.
                "prefill_coverage": {
                    "prompt_lengths": [row["prompt_len"] for row in rows],
                    "hc_split_calls": sum(
                        (row.get("prefill_evidence") or {}).get("hc_boundary", {})
                        .get("split_calls", 0)
                        if (row.get("prefill_evidence") or {}).get("hc_boundary")
                        else 0
                        for row in rows
                    ),
                    "hc_calls": sum(
                        (row.get("prefill_evidence") or {}).get("hc_boundary", {})
                        .get("calls", 0)
                        if (row.get("prefill_evidence") or {}).get("hc_boundary")
                        else 0
                        for row in rows
                    ),
                    "moe_overlapped_calls": sum(
                        (row.get("prefill_evidence") or {})
                        .get("moe", {})
                        .get("overlapped_calls", 0)
                        for row in rows
                    ),
                    "moe_sequential_calls": sum(
                        (row.get("prefill_evidence") or {})
                        .get("moe", {})
                        .get("sequential_calls", 0)
                        for row in rows
                    ),
                },
            }
        if len(modes) == 2 and "eager" in modes and "fused" in modes:
            eager_tokens = {
                row["prompt_index"]: row["predicted_tokens"]
                for row in summary_rows
                if row["mode"] == "eager"
            }
            fused_tokens = {
                row["prompt_index"]: row["predicted_tokens"]
                for row in summary_rows
                if row["mode"] == "fused"
            }
            diverged = {
                str(index): sum(
                    1
                    for left, right in zip(
                        eager_tokens[index], fused_tokens[index], strict=True
                    )
                    if left != right
                )
                for index in eager_tokens
            }
            mode_summaries["eager_vs_fused"] = {
                "prompts_with_divergence": sum(
                    1 for value in diverged.values() if value
                ),
                "diverged_predictions_per_prompt": diverged,
                "total_diverged_predictions": sum(diverged.values()),
            }
        result["mode_summaries"] = mode_summaries

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_at_end"] = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }

        # Hard gates: topology, tokenizer parity, contract, finite logits
        # (raised on violation above), complete runs.  Token match rates are
        # reported as measured -- bf16 runtime vs the fp8-kernel MP8
        # reference legitimately admits near-tie flips (task contract).
        result["accepted"] = bool(
            result["placement"]["accepted"]
            and result["tokenizer_preflight"]["accepted"]
            and all(
                len(result["prompt_results"].get(mode, [])) == len(prompts)
                for mode in modes
            )
        )
    except Exception:
        # Announce immediately.  A rank that fails here skips every collective
        # the healthy ranks are still queued on, so the run does not crash --
        # it deadlocks, and the only symptom is that stdout stops.  Diagnosing
        # that from the outside costs ~20 minutes and a py-spy dump; printing
        # the traceback the moment it is caught costs nothing.
        print(
            f"[E0e2e][rank {rank}] FAILED (run will now deadlock on the next "
            f"collective -- kill it):\n{traceback.format_exc()}",
            flush=True,
        )
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
                "experiment": "E0e2e-tp4-pp4-golden-token-gate",
                "accepted": accepted_all,
                "checkpoint_id": result["checkpoint_id"],
                "mode_summaries": result.get("mode_summaries"),
                "max_seq_len": result.get("max_seq_len"),
                "prompt_lengths": result.get("prompt_lengths"),
                "global_row_shapes": result.get("global_row_shapes"),
                "share_moe_buffers": result.get("share_moe_buffers"),
                "prefill_chunk": result.get("prefill_chunk", 0),
                "moe_overlap_blocks": result.get("moe_overlap_blocks", 0),
                "fused_scope": args.fused_scope,
                "kv_dtype": args.kv_dtype,
                "memory_at_end": result.get("memory_at_end"),
                "placement": result["placement"],
                "tokenizer_preflight": result["tokenizer_preflight"],
                "errors": result["errors"],
            },
        )
        print(f"[E0e2e] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
