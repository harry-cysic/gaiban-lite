#!/usr/bin/env python3
"""E8F: minimal single-path serving loop (TARGET section 10 Phase 1).

The engine already exists -- this wires two proven halves into a resident
16-rank request loop and measures the real single-path serving discount
(the 39.2 tok/s bare-engine number vs a framework-caliber per-request number,
which section 1.2 currently only *infers* at 20%).  See
docs/design-serving-shell.md.

  * decode engine: e1f's free-running closed loop -- stage-3 head -> argmax ->
    NCCL loopback -> stage-0 re-embed -> graph replay, one graph family per
    fixed shape, position-agnostic so one capture serves every request.
  * prefill + handoff: e0ef2e's StageLane real prefill (chunked, eager) and the
    E7F handoff (snapshot the prefill state, install it into a decode stage
    whose decode-only MoEs share the prefill resident weights -- Blocker B).

New here: real prefill per request (not e1f's synthetic seed), free-running
until EOS or max_new_tokens (not a fixed step count), a serial request loop, and
per-request framework timing.  No scheduler / slot recycling / batch admission
(section 10: not needed for single-path); the decode stage's state is
re-installed per request via seed_decode_payload, which is the slot-recycling
seam Phase 2 extends rather than rewrites.

Prompts come from a JSONL/oracle file (no HTTP yet -- HTTP is deferred, the
discount is measurable without it).  Greedy (argmax) sampling, matching e1f's
closed loop.

Run (dual node, 16 ranks) via run_e8f_serving.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import e0ef2e_golden_gate as gate
from e1f_full_decode_bench import (
    EXPECTED_TP_SIZE,
    STAGE_COUNT,
    create_e1f_topology,
    pair_transfer,
)
from e1f_full_decode_bench import pair_transfer as loop_transfer  # noqa: F401
from e7f_handoff_gate import (
    clone_static_state,
    install_ratio4,
    restore_static_state,
    snapshot_ratio4,
)
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.head_stage import (
    embed_hc_residual,
    head_logits,
    load_embed_head_material,
)
from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.physical_stage import build_physical_stage
from dsv4_direct.stateful_decode import (
    DecodeGraphFamily,
    StatefulDecodeCursor,
    classify_decode_position,
)
from dsv4_direct.stateful_graph import (
    capture_stateful_graph,
    replay_stateful_graph,
)

HC_MULT = gate.HC_MULT
HIDDEN = gate.HIDDEN
LOCAL_BATCH = 1
STAGE_LAYERS = gate.STAGE_LAYERS
GRAPH_MOE_SLOT_TUPLE = (1, 2, 3)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# decode stage built once for capture reuse; state re-installed per request


def build_decode_stage(materials, *, backend, device):
    """One decode stage with decode-only MoEs sharing the prefill resident
    weights (E7F Blocker B), built once so its graphs capture once and its
    state is re-installed per request."""

    from dsv4_direct.block import DirectDecodeBlock
    from dsv4_direct.moe_runtime import TP4MoE
    from dsv4_direct.superstage import TP4DecodeStage

    decode_rows = EXPECTED_TP_SIZE * LOCAL_BATCH
    blocks = []
    states = []
    for material in materials:
        state = material.new_state(num_local_sequences=LOCAL_BATCH)
        pm = material.moe
        decode_moe = TP4MoE(
            config=pm.config,
            resident=pm.resident,
            gate=pm.gate,
            rank=material.tp_rank,
            device=material.device,
            global_row_shapes=(decode_rows,),
            group=pm.group,
            slots_per_shape=4,
            marlin_input_dtype=pm._marlin_input_dtype,
        )
        block = DirectDecodeBlock(
            weights=material.raw_block,
            attention=material.new_attention(state),
            moe=decode_moe,
            norm_eps=material.norm_eps,
            sinkhorn_iters=material.sinkhorn_iters,
            hc_eps=material.hc_eps,
            hc_boundary_backend=backend,
        )
        blocks.append(block)
        states.append(state)
    stage = TP4DecodeStage(blocks, hc_boundary_backend=backend)
    return stage, states


def install_prefill_state(materials, lane, states) -> None:
    """Re-install one prefill's end state into the decode stage (in place, so
    the captured graph -- which reads the state tensors by address -- stays
    valid)."""

    for material, (mat2, attention), state in zip(
        materials, lane.layers, states, strict=True
    ):
        if material.kind == "ratio4":
            install_ratio4(state, snapshot_ratio4(attention))
        else:
            restore_static_state(state, clone_static_state(attention.state))


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--oracle-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=8320)
    parser.add_argument("--prefill-chunk", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--prompt-min-tokens", type=int, default=128)
    parser.add_argument("--prompt-max-tokens", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--hc-backend", type=str, default="fused")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if world != STAGE_COUNT * EXPECTED_TP_SIZE:
        raise SystemExit(f"expected 16 ranks, got {world}")

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = Path(args.out_dir)
    topo = create_e1f_topology(rank)
    stage = topo["stage"]
    tp_rank = topo["tp_rank"]
    max_seq_len = int(args.max_seq_len)

    result: dict[str, Any] = {
        "experiment": "E8F-single-path-serving-loop",
        "schema_version": 1,
        "argv": [str(v) for v in sys.argv],
        "rank": rank,
        "stage": stage,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "max_seq_len": max_seq_len,
        "prefill_chunk": int(args.prefill_chunk),
        "max_new_tokens": int(args.max_new_tokens),
        "rounds": int(args.rounds),
        "bare_engine_tok_s": 39.2,
        "requests": [],
        "errors": [],
        "accepted": False,
    }

    try:
        # ------------------------------------------------------------------
        # rank-0 preflight: config, checkpoint id, tokenizer, prompts, eos
        envelope_holder: list[Any] = [None]
        if rank == 0:
            model_config = json.loads(
                (stage_root / "config.json").read_text(encoding="utf-8")
            )
            checkpoint = inspect_stage_checkpoint(
                stage_root, list(range(gate.MODEL_LAYERS)), EXPECTED_TP_SIZE
            )
            if not checkpoint["ok"]:
                raise ValueError(f"checkpoint failed: {checkpoint['errors'][:3]}")
            oracle = json.loads(
                args.oracle_json.expanduser().read_text(encoding="utf-8")
            )
            prompts = [
                {"prompt": e.get("prompt", ""), "prompt_tokens": [int(t) for t in e["prompt_tokens"]]}
                for e in oracle["prompts"]
                if args.prompt_min_tokens <= len(e["prompt_tokens"]) <= args.prompt_max_tokens
            ][: args.max_prompts]
            preflight = gate.tokenizer_preflight(stage_root, oracle["prompts"][:1])
            envelope_holder[0] = {
                "ok": True,
                "config": model_config,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "eos_token_id": preflight["eos_token_id"],
                "prompts": prompts,
            }
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        model_config = envelope["config"]
        result["checkpoint_id"] = envelope["checkpoint_id"]
        eos_token_id = int(envelope["eos_token_id"])
        prompts = envelope["prompts"]
        vocab = int(model_config["vocab_size"])
        result["num_prompts"] = len(prompts)
        result["eos_token_id"] = eos_token_id

        # ------------------------------------------------------------------
        # load materials (share_moe_buffers for prefill) + embed/head
        prefill_rows = sorted(
            {EXPECTED_TP_SIZE * min(args.prefill_chunk, len(p["prompt_tokens"]))
             if args.prefill_chunk else EXPECTED_TP_SIZE * len(p["prompt_tokens"])
             for p in prompts}
        ) if prompts else []
        # also register the per-chunk full size
        chunk_rows = {EXPECTED_TP_SIZE * min(args.prefill_chunk, len(p["prompt_tokens"]))
                      for p in prompts}
        global_row_shapes = tuple(sorted({EXPECTED_TP_SIZE} | chunk_rows
                                         | {EXPECTED_TP_SIZE * len(p["prompt_tokens"]) for p in prompts
                                            if len(p["prompt_tokens"]) <= (args.prefill_chunk or 1 << 30)}))
        result["global_row_shapes"] = list(global_row_shapes)

        started = time.perf_counter()
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
            max_seq_len=max_seq_len,
            global_row_shapes=global_row_shapes,
            slots_per_shape=1,
            attention_tp_shard=True,
            share_moe_buffers=True,
            kv_dtype="bf16",
            indexer_kv_dtype="bf16",
            progress_every=args.progress_every,
            progress=(lambda m: print(f"[E8F] {m}", flush=True)) if rank in (0, 4, 8, 12) else None,
        )
        materials = list(stage_material.materials)
        embed_material = head_material = None
        if stage == 0:
            embed_material = load_embed_head_material(
                stage_root=stage_root, device=device,
                checkpoint_id=result["checkpoint_id"], load_embed=True, load_head=False,
            )
        elif stage == STAGE_COUNT - 1:
            head_material = load_embed_head_material(
                stage_root=stage_root, device=device,
                checkpoint_id=result["checkpoint_id"], load_embed=False, load_head=True,
            )
        result["load_seconds"] = time.perf_counter() - started
        backend = resolve_hc_boundary_backend(
            None if args.hc_backend == "default" else args.hc_backend
        )

        # ------------------------------------------------------------------
        # build the decode stage once (state re-installed per request)
        decode_stage, decode_states = build_decode_stage(
            materials, backend=backend, device=device
        )
        result["decode_stage_built"] = True
        dist.barrier()
        if rank == 0:
            print(f"[E8F] loaded + decode stage built, {len(prompts)} prompts", flush=True)

        # graphs / plan / cursor are created lazily on the first request's
        # prefill state (capture reuse across requests).
        serving_state: dict[str, Any] = {"graphs": None}

        def prefill_lane_for(prompt_tokens: list[int]):
            """A fresh prefill lane (e0ef2e StageLane) prefilled over the prompt.
            Rebuilt per request, matching the golden gate's per-prompt lane."""
            lane = gate.StageLane(
                materials, backend=backend, device=device,
                ratio4_index_mode="ref", fuse_min_seqlen=1024, fused_scope="decode",
            )
            chunk = args.prefill_chunk
            plen = len(prompt_tokens)
            position = 0
            plan_chunks = []
            if chunk and chunk < plen:
                while position < plen:
                    length = min(chunk, plen - position)
                    plan_chunks.append((position, prompt_tokens[position:position + length]))
                    position += length
            else:
                plan_chunks.append((0, prompt_tokens))
            residual_out = None
            for (pos, toks) in plan_chunks:
                seqlen = len(toks)
                if stage == 0:
                    ids = torch.tensor([toks], dtype=torch.int64, device=device)
                    residual = embed_hc_residual(embed_material, ids)
                else:
                    residual = torch.empty(
                        (LOCAL_BATCH, seqlen, HC_MULT, HIDDEN),
                        dtype=torch.bfloat16, device=device,
                    )
                    pair_transfer(residual, send=False, group=topo["prev_pair"], peer=0)
                residual = lane.forward(
                    residual, start_pos=pos,
                    input_ids=(ids if stage == 0 else None),
                )
                if stage < STAGE_COUNT - 1:
                    pair_transfer(residual.contiguous(), send=True, group=topo["next_pair"], peer=1)
                else:
                    residual_out = residual
            return lane, residual_out

        def serve_request(req_index: int, prompt_tokens: list[int]) -> dict[str, Any]:
            plen = len(prompt_tokens)
            stop = plen + args.max_new_tokens
            torch.cuda.synchronize(device)
            t_prefill0 = time.perf_counter()
            lane, prefill_residual = prefill_lane_for(prompt_tokens)
            # install prefill state -> decode stage (in place, so the captured
            # graph -- which reads the state tensors by address -- stays valid)
            install_prefill_state(materials, lane, decode_states)
            del lane
            torch.cuda.synchronize(device)
            t_prefill1 = time.perf_counter()

            # ONE plan + cursor + loop buffers, built once (fixed plen) and
            # reset per request (e1f restore_cycle: cursor.reset + expected /
            # stop-position fill).  All prompts in a run share plen, so the plan
            # shapes (candidate_width etc.) and the captured graphs are reusable;
            # a differing plen is a hard error, not a silent re-capture.
            if serving_state.get("plan") is None:
                serving_state["plen"] = plen
                cursor = StatefulDecodeCursor(start_position=plen, device=device)
                plan = decode_stage.prepare_stateful_decode_plan(
                    cursor, start_position=plen, stop_position=stop,
                    graph_moe_slots=GRAPH_MOE_SLOT_TUPLE,
                )
                token_buffer = torch.zeros((LOCAL_BATCH, 1), dtype=torch.int64, device=device)
                staging = torch.empty_like(plan.input_residual_buffer) if stage > 0 else None
                serving_state.update(
                    plan=plan, cursor=cursor, token_buffer=token_buffer, staging=staging
                )
            elif serving_state["plen"] != plen:
                raise ValueError(
                    f"E8F run requires a fixed prompt length; got {plen} after "
                    f"{serving_state['plen']} (bucket by length, one run each)"
                )
            plan = serving_state["plan"]
            cursor = serving_state["cursor"]
            token_buffer = serving_state["token_buffer"]
            staging = serving_state["staging"]
            # reset the plan/cursor to plen for this request (state already
            # re-installed at plen above)
            cursor.reset(plen)
            plan.expected_position.fill_(plen)
            plan.stop_position_tensor.fill_(plan.stop_position)

            # Tight closed loop (e1f pipeline_step form): no per-step host sync
            # or object broadcast -- those inflated the first run's 36.5 ms/tok.
            # Fixed length (max_new_tokens) for a clean discount measurement; EOS
            # is a serving feature with its own small cost, deferred.  Stage 3
            # accumulates the argmax tokens in a device tensor and they are read
            # once at the end.
            graphs = serving_state["graphs"]
            capture_here = graphs is None
            if capture_here:
                graphs = {}
            capture_stream = torch.cuda.Stream(device=device)
            n_gen = int(args.max_new_tokens)
            gen_tokens = (
                torch.zeros(n_gen, dtype=torch.int64, device=device)
                if stage == STAGE_COUNT - 1 else None
            )

            def ensure_capture(family):
                if capture_here and family not in graphs:
                    torch.cuda.synchronize(device)
                    graphs[family] = capture_stateful_graph(
                        decode_stage, plan, graph_family=family,
                        capture_stream=capture_stream, pool=torch.cuda.graph_pool_handle(),
                    )

            # first token: stage 3 heads the prefill residual, loops it to stage 0.
            gen_idx = 0
            if stage == STAGE_COUNT - 1:
                logits = head_logits(head_material, prefill_residual)
                token_buffer.copy_(logits.argmax(dim=-1, keepdim=True))
                gen_tokens[gen_idx].copy_(token_buffer.view(-1)[0])
                pair_transfer(token_buffer, send=True, group=topo["loop_pair"], peer=0)
            if stage == 0:
                pair_transfer(token_buffer, send=False, group=topo["loop_pair"], peer=1)
            gen_idx = 1

            torch.cuda.synchronize(device)
            t_decode0 = time.perf_counter()
            for step in range(n_gen - 1):
                family = classify_decode_position(plen + step)
                if stage == 0:
                    hidden = torch.nn.functional.embedding(token_buffer, embed_material.embed_weight)
                    plan.input_residual_buffer.copy_(hidden.unsqueeze(2).expand(-1, -1, HC_MULT, -1))
                    plan.input_ids_buffer.copy_(token_buffer)
                    ensure_capture(family)
                    replay_stateful_graph(graphs[family], plan, graph_family=family)
                    pair_transfer(plan.output_buffer.contiguous(), send=True, group=topo["next_pair"], peer=1)
                    pair_transfer(token_buffer, send=False, group=topo["loop_pair"], peer=1)
                else:
                    pair_transfer(staging, send=False, group=topo["prev_pair"], peer=0)
                    plan.input_residual_buffer.copy_(staging)
                    ensure_capture(family)
                    replay_stateful_graph(graphs[family], plan, graph_family=family)
                    if stage < STAGE_COUNT - 1:
                        pair_transfer(plan.output_buffer.contiguous(), send=True, group=topo["next_pair"], peer=1)
                    else:
                        logits = head_logits(head_material, plan.output_buffer)
                        token_buffer.copy_(logits.argmax(dim=-1, keepdim=True))
                        gen_tokens[gen_idx].copy_(token_buffer.view(-1)[0])
                        pair_transfer(token_buffer, send=True, group=topo["loop_pair"], peer=0)
                cursor.advance_host(family)
                gen_idx += 1
            torch.cuda.synchronize(device)
            t_decode1 = time.perf_counter()

            if capture_here:
                serving_state["graphs"] = graphs
            # read tokens once (stage 3), broadcast to rank 0 for the record
            tok_holder = [gen_tokens.cpu().tolist() if stage == STAGE_COUNT - 1 else None]
            dist.broadcast_object_list(tok_holder, src=12)
            new_tokens = tok_holder[0]
            decode_ms = []  # per-step timing removed; use decode_wall / n

            n_new = len(new_tokens)
            n_loop = max(n_gen - 1, 1)
            prefill_ms = (t_prefill1 - t_prefill0) * 1e3          # prefill lane + install
            first_token_ms = (t_decode0 - t_prefill0) * 1e3       # prefill + plan + first-token head
            decode_wall_ms = (t_decode1 - t_decode0) * 1e3        # the n_gen-1 loop steps
            decode_ms_per_token = decode_wall_ms / n_loop
            framework_ms = (t_decode1 - t_prefill0) * 1e3         # whole request
            rec = {
                "request": req_index,
                "prompt_len": plen,
                "new_tokens": n_new,
                "prefill_ms": prefill_ms,
                "first_token_ms": first_token_ms,
                "decode_wall_ms": decode_wall_ms,
                "decode_ms_per_token": decode_ms_per_token,
                "framework_ms": framework_ms,
                "framework_tok_s": (n_new / framework_ms * 1e3) if framework_ms > 0 else None,
                "decode_only_tok_s": (1e3 / decode_ms_per_token) if decode_ms_per_token > 0 else None,
                "captured": capture_here,
                "first_tokens": new_tokens[:8],
            }
            return rec

        # ------------------------------------------------------------------
        # serve each prompt, `rounds` times (round 0 also captures)
        for rnd in range(args.rounds):
            for i, p in enumerate(prompts):
                rec = serve_request(i, p["prompt_tokens"])
                rec["round"] = rnd
                result["requests"].append(rec)
                if rank == 0:
                    print(
                        f"[E8F] r{rnd} req{i} len{rec['prompt_len']} "
                        f"+{rec['new_tokens']}tok framework {rec['framework_tok_s']:.1f} tok/s "
                        f"(prefill {rec['prefill_ms']:.0f}ms, decode {rec['decode_ms_per_token']:.1f}ms/tok)",
                        flush=True,
                    )

        result["accepted"] = bool(result["requests"])

    except Exception as error:  # noqa: BLE001
        result["errors"].append(
            {"type": type(error).__name__, "message": str(error),
             "traceback": traceback.format_exc()}
        )
        print(f"[E8F] rank {rank} FAILED: {error}", flush=True)
        traceback.print_exc()

    write_json(out_dir / f"rank{rank}.json", result)
    # The resident-loop teardown hangs on destroy_process_group with this
    # topology's ~19 custom groups (GPUs idle, not spinning), which would stall
    # the launcher's done sentinel for the full deadline.  The data is already
    # on disk, so force-exit instead of blocking: torchrun collects each
    # worker's exit code and the launcher's `echo $? > done` fires promptly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
