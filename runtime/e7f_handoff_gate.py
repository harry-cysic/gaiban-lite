#!/usr/bin/env python3
"""E7F step 2: is a *prefilled* state a valid decode state?

Motivation (see experiments/E7F-single-path-serving/README.md).  This repo has
two paths that have never been connected:

  * the real-prompt path (``e0ef2e_golden_gate.py``) prefills a prompt and then
    decodes **eagerly** -- it never captures a graph, and it carries all the
    quality evidence (D0L 614/640);
  * the fast path (``e1f_full_decode_bench.py``) decodes from a **synthetically
    seeded** state -- it carries all the speed evidence (39.2 tok/s).

Single-path serving is the first thing that needs both at once.  The join is a
state handoff, and its only non-trivial part is ratio-4: window / ratio-128
layers already prefill straight into the ``Static*KV`` objects that decode uses
(``e0ef2e_golden_gate.py`` StageLane), while ratio-4 prefills into a separate
``Ratio4FullPositionAttention`` whose state must be installed via
``StaticRatio4KV.seed_decode_payload``.

``e0e2e_ratio4_selfcheck.py`` already proved that install bitwise -- but at
``tp_size=1`` (its config never sets tp_size; the dataclass default is 1), on
one layer, on the attention branch alone, without a graph.  E6F has since made
sharding the default.  So the frozen precedent does not cover today's default,
and its artifact has no sharding witness at all.

**What this gate asks** (deliberately narrower than "does serving work"):
after a real multi-token prefill, does the *whole block chain* decoding from
the handed-off state agree with the prefill lane decoding forward from its own
state?

  arm R (reference) -- the e0ef2e lane continues decoding: ratio-4 stays on
                       ``Ratio4FullPositionAttention``, other kinds stay on
                       their ``Static*KV``.  This is the form every frozen
                       golden number was produced in.
  arm C (candidate) -- states handed off into the decode-side objects, then
                       ``TP4DecodeStage.forward_decode_tensors``.

Bitwise agreement means the prefilled state *is* a valid decode state.  Graph
correctness then composes from E0sf, which already showed graph == eager for a
valid state; this gate deliberately does **not** capture a graph, so that a
failure here is unambiguous about which join broke.

Run (titan065, 4 GPUs, one TP4 stage):
  export CUDA_HOME=/usr/local/cuda-13.2
  export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=4 \
    e7f_handoff_gate.py --stage-root ~/Workspace/DeepSeek-V4-Flash \
    --out-dir out-e7f-handoff
"""

from __future__ import annotations

import argparse
import json
import platform
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
    EAGER_MOE_SLOT,
    GRAPH_MOE_SLOTS,
    GRAPH_MOE_SLOT_TUPLE,
    forward_eager_prevalidated,
)
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.mode_witness import collect_attention_modes
from dsv4_direct.physical_stage import build_physical_stage
from dsv4_direct.stateful_decode import (
    DecodeGraphFamily,
    StatefulDecodeCursor,
    build_decode_schedule,
)
from dsv4_direct.stateful_graph import capture_stateful_graph, replay_stateful_graph
from dsv4_direct.static_ratio4_kv import StaticRatio4KV
from dsv4_direct.superstage import TP4DecodeStage

EXPECTED_TP_SIZE = 4
HC_MULT = gate.HC_MULT
HIDDEN = gate.HIDDEN
LOCAL_BATCH = 1


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def deterministic_residual(
    *, seed: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed & ((1 << 62) - 1))
    value = torch.randn(
        (LOCAL_BATCH, seqlen, HC_MULT, HIDDEN), generator=generator, dtype=torch.float32
    )
    return (value * 0.02).to(device=device, dtype=torch.bfloat16)


def deterministic_ids(
    *, seed: int, seqlen: int, device: torch.device, vocab: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed & ((1 << 62) - 1))
    value = torch.randint(0, vocab, (LOCAL_BATCH, seqlen), generator=generator)
    return value.to(device=device, dtype=torch.int64)


def error_metrics(observed: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = (observed.float() - expected.float()).abs()
    return {
        "bitwise": bool(torch.equal(observed, expected)),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
    }


# ---------------------------------------------------------------------------
# the handoff itself


RATIO4_PAYLOAD_FIELDS = (
    "raw",
    "compressed",
    "indexer_kv",
    "main_kv_state",
    "main_score_state",
    "index_kv_state",
    "index_score_state",
)


def snapshot_ratio4(attention: Any) -> dict[str, Any]:
    """Copy out the full-position ratio-4 state (the payload + its metadata)."""

    payload = {name: getattr(attention, name).clone() for name in RATIO4_PAYLOAD_FIELDS}
    payload["next_position"] = int(attention.next_position)
    payload["compressed_count"] = int(attention.compressed_count)
    return payload


def install_ratio4(state: StaticRatio4KV, snapshot: dict[str, Any]) -> None:
    """Install a full-position snapshot into the decode-side static state.

    ``seed_decode_payload`` demands BF16 latent/indexer rows regardless of the
    storage dtype and refuses payloads that alias the destination, so every
    tensor here is an independent clone in the payload contract's dtype.
    """

    state.seed_decode_payload(
        snapshot["next_position"],
        raw=snapshot["raw"].to(torch.bfloat16).clone(),
        compressed=snapshot["compressed"].to(torch.bfloat16).clone(),
        indexer_kv=snapshot["indexer_kv"].to(torch.bfloat16).clone(),
        main_kv_state=snapshot["main_kv_state"].clone(),
        main_score_state=snapshot["main_score_state"].clone(),
        index_kv_state=snapshot["index_kv_state"].clone(),
        index_score_state=snapshot["index_score_state"].clone(),
    )


def clone_static_state(state: Any) -> dict[str, torch.Tensor]:
    return {
        name: tensor.clone()
        for name, tensor in state.__dict__.items()
        if isinstance(tensor, torch.Tensor)
    }


def restore_static_state(state: Any, snapshot: dict[str, torch.Tensor]) -> None:
    for name, tensor in snapshot.items():
        destination = getattr(state, name)
        if not isinstance(destination, torch.Tensor):
            raise TypeError(f"{name} is no longer a tensor on {type(state).__name__}")
        destination.copy_(tensor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", type=str, default="0,1,2,3")
    parser.add_argument("--prefill-len", type=int, default=256)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument("--kv-dtype", type=str, default="bf16")
    parser.add_argument("--indexer-kv-dtype", type=str, default="bf16")
    parser.add_argument(
        "--attention-tp-shard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="E6F variant A; default matches the released runtime default",
    )
    parser.add_argument("--hc-backend", type=str, default="default")
    parser.add_argument(
        "--graph-arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also decode the handed-off state through captured graphs",
    )
    parser.add_argument(
        "--break-handoff",
        type=str,
        default="none",
        choices=("none", "ratio4-skip", "static-skip", "perturb-compressed"),
        help=(
            "negative control: deliberately omit half the handoff.  A gate that "
            "cannot be made to fail proves nothing (TARGET 9.12), and this one "
            "compares two arms that would agree trivially if arm C silently ran "
            "on the same state object as arm R."
        ),
    )
    args = parser.parse_args()

    layer_ids = [int(item) for item in args.layers.split(",") if item.strip()]
    prefill_len = int(args.prefill_len)
    decode_steps = int(args.decode_steps)

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != EXPECTED_TP_SIZE:
        raise RuntimeError(f"expected a {EXPECTED_TP_SIZE}-rank TP group, got {world}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    stage_root = Path(args.stage_root)

    result: dict[str, Any] = {
        "experiment": "E7F-prefill-decode-state-handoff",
        "schema_version": 1,
        # 9.11: record BOTH sides -- what was asked for and what was resolved.
        "argv": [str(item) for item in sys.argv],
        "requested": {
            "attention_tp_shard": bool(args.attention_tp_shard),
            "layers": layer_ids,
            "prefill_len": prefill_len,
            "decode_steps": decode_steps,
            "max_seq_len": int(args.max_seq_len),
            "kv_dtype": args.kv_dtype,
            "break_handoff": args.break_handoff,
            "graph_arm": bool(args.graph_arm),
            "indexer_kv_dtype": args.indexer_kv_dtype,
        },
        "rank": rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "accepted": False,
        "errors": [],
        "steps": [],
    }

    try:
        model_config = json.loads(
            (stage_root / "config.json").read_text(encoding="utf-8")
        )
        checkpoint = inspect_stage_checkpoint(
            stage_root, layer_ids, EXPECTED_TP_SIZE
        )
        if not checkpoint["ok"]:
            raise ValueError(
                f"checkpoint contract failed: {checkpoint['errors'][:4]}"
            )
        result["checkpoint_id"] = checkpoint["checkpoint_id"]
        vocab = int(model_config["vocab_size"])

        tp_global_ranks = tuple(range(EXPECTED_TP_SIZE))
        tp_group = dist.new_group(ranks=list(tp_global_ranks))

        # Decode rows are TP_SIZE (B=1 per lane); the prefill forward needs its
        # own registered row shape (TARGET 3.8: these buffers are the real
        # memory driver, and an unregistered shape is a hard failure).
        global_row_shapes = (EXPECTED_TP_SIZE, EXPECTED_TP_SIZE * prefill_len)
        result["global_row_shapes"] = list(global_row_shapes)

        started = time.perf_counter()
        stage_material = build_physical_stage(
            stage_id=0,
            layer_ids=layer_ids,
            model_config=model_config,
            stage_root=stage_root,
            tp_rank=rank,
            tp_group=tp_group,
            tp_global_ranks=tp_global_ranks,
            device=device,
            checkpoint_id=checkpoint["checkpoint_id"],
            max_seq_len=int(args.max_seq_len),
            global_row_shapes=global_row_shapes,
            slots_per_shape=4,
            attention_tp_shard=bool(args.attention_tp_shard),
            kv_dtype=args.kv_dtype,
            indexer_kv_dtype=args.indexer_kv_dtype,
            progress_every=args.progress_every,
            progress=(
                (lambda message: print(f"[E7F] {message}", flush=True))
                if rank == 0
                else None
            ),
        )
        result["load_seconds"] = time.perf_counter() - started
        materials = list(stage_material.materials)
        result["layer_kinds"] = {
            str(material.layer_id): material.kind for material in materials
        }

        backend = resolve_hc_boundary_backend(
            None if args.hc_backend == "default" else args.hc_backend
        )

        # --------------------------------------------------------------
        # prefill, in exactly the form the golden gate uses
        lane = gate.StageLane(materials, backend=backend, device=device)

        # 9.11 resolved side: sharding lives in tp_size/tp_rank, which does not
        # match the *_mode naming convention that auto-discovery keys on -- the
        # exact gap that let E6F read a dropped flag as "no effect".
        attention_tp: dict[str, Any] = {}
        for material, attention in lane.layers:
            config = material.attention_config
            entry = {
                "tp_size": int(getattr(config, "tp_size", 1)),
                "tp_rank": int(getattr(config, "tp_rank", 0)),
            }
            for name in ("local_num_heads", "local_o_groups"):
                if hasattr(config, name):
                    entry[name] = int(getattr(config, name))
            wo_b = getattr(material.prepared, "wo_b", None)
            if isinstance(wo_b, torch.Tensor):
                entry["wo_b_shape"] = list(wo_b.shape)
            attention_tp[str(material.layer_id)] = entry
        result["attention_tp"] = attention_tp
        result["attention_modes"] = collect_attention_modes(lane.layers)

        prefill_residual = deterministic_residual(
            seed=args.seed * 31 + 7, seqlen=prefill_len, device=device
        )
        prefill_ids = deterministic_ids(
            seed=args.seed * 17 + 3, seqlen=prefill_len, device=device, vocab=vocab
        )
        started = time.perf_counter()
        lane.forward(prefill_residual, start_pos=0, input_ids=prefill_ids)
        torch.cuda.synchronize(device)
        result["prefill_seconds"] = time.perf_counter() - started

        # --------------------------------------------------------------
        # snapshot the prefill-end state BEFORE arm R mutates it
        ratio4_snapshots: dict[int, dict[str, Any]] = {}
        static_snapshots: dict[int, dict[str, torch.Tensor]] = {}
        for material, attention in lane.layers:
            if material.kind == "ratio4":
                ratio4_snapshots[material.layer_id] = snapshot_ratio4(attention)
            else:
                static_snapshots[material.layer_id] = clone_static_state(
                    attention.state
                )
        result["prefill_end_positions"] = {
            str(layer_id): snapshot["next_position"]
            for layer_id, snapshot in ratio4_snapshots.items()
        }

        # per-step decode inputs, shared by both arms
        step_inputs = []
        for index in range(decode_steps):
            step_inputs.append(
                (
                    deterministic_residual(
                        seed=args.seed * 101 + index, seqlen=1, device=device
                    ),
                    deterministic_ids(
                        seed=args.seed * 211 + index,
                        seqlen=1,
                        device=device,
                        vocab=vocab,
                    ),
                )
            )

        # --------------------------------------------------------------
        # arm R: the reference lane keeps decoding from its own state
        reference_outputs: list[torch.Tensor] = []
        for index, (residual, ids) in enumerate(step_inputs):
            output = lane.forward(
                residual, start_pos=prefill_len + index, input_ids=ids
            )
            torch.cuda.synchronize(device)
            reference_outputs.append(output.clone())

        # --------------------------------------------------------------
        # arm C: hand the prefill-end state off to the decode-side objects
        states = []
        for material, attention in lane.layers:
            state = material.new_state(num_local_sequences=LOCAL_BATCH)
            if material.kind == "ratio4":
                if args.break_handoff != "ratio4-skip":
                    snapshot = ratio4_snapshots[material.layer_id]
                    if args.break_handoff == "perturb-compressed":
                        # A *valid but wrong* state: positions, shapes and
                        # finiteness all still satisfy seed_decode_payload, so
                        # the validators cannot catch it and only the arm-R
                        # comparison can.  ratio4-skip / static-skip trip the
                        # position validators instead, which tests those, not
                        # the comparison.
                        snapshot = dict(snapshot)
                        perturbed = snapshot["compressed"].clone()
                        perturbed[0, 0, 0] += 0.5
                        snapshot["compressed"] = perturbed
                    install_ratio4(state, snapshot)
            elif args.break_handoff != "static-skip":
                restore_static_state(state, static_snapshots[material.layer_id])
            states.append(state)
        blocks = [
            material.new_block(state, hc_boundary_backend=backend)
            for (material, _), state in zip(lane.layers, states, strict=True)
        ]
        candidate_stage = TP4DecodeStage(blocks, hc_boundary_backend=backend)

        mismatched: list[int] = []
        for index, (residual, ids) in enumerate(step_inputs):
            position = prefill_len + index
            plan = candidate_stage.prepare_decode_plan(position)
            outputs = candidate_stage.forward_decode_tensors(
                residual,
                input_ids_local=ids,
                start_pos=position,
                plan=plan,
            )
            torch.cuda.synchronize(device)
            metrics = error_metrics(outputs[-1], reference_outputs[index])
            metrics["step"] = index
            metrics["position"] = position
            result["steps"].append(metrics)
            if not metrics["bitwise"]:
                mismatched.append(position)

        result["mismatched_positions"] = mismatched

        # --------------------------------------------------------------
        # arm G: the same handed-off state, decoded through captured graphs.
        #
        # E0sf already showed graph == eager for a *seeded* state, so this arm
        # is about provenance only: a prefilled state has to survive warmup,
        # restore and capture the same way a manufactured one does.  It runs
        # against arm R, not against arm C, so a break shows up as
        # "graph disagrees with the reference lane" rather than being masked by
        # comparing two things that share the same handoff.
        graph_mismatched: list[int] = []
        if args.graph_arm:
            def restore_handoff() -> None:
                for (material, _), state in zip(lane.layers, states, strict=True):
                    if material.kind == "ratio4":
                        install_ratio4(state, ratio4_snapshots[material.layer_id])
                    else:
                        restore_static_state(
                            state, static_snapshots[material.layer_id]
                        )

            restore_handoff()
            stop_position = prefill_len + decode_steps
            cursor = StatefulDecodeCursor(
                start_position=prefill_len, device=device
            )
            stateful_plan = candidate_stage.prepare_stateful_decode_plan(
                cursor,
                start_position=prefill_len,
                stop_position=stop_position,
                graph_moe_slots=GRAPH_MOE_SLOT_TUPLE,
            )
            schedule = build_decode_schedule(prefill_len, decode_steps)
            result["graph_families"] = [step.family.value for step in schedule]

            def reset_cycle() -> None:
                restore_handoff()
                cursor.reset(prefill_len)
                stateful_plan.expected_position.fill_(prefill_len)
                stateful_plan.stop_position_tensor.fill_(stop_position)

            def warm_cycle(*, graph_slots: bool) -> None:
                for index, step in enumerate(schedule):
                    residual, ids = step_inputs[index]
                    stateful_plan.input_residual_buffer.copy_(residual)
                    stateful_plan.input_ids_buffer.copy_(ids)
                    forward_eager_prevalidated(
                        candidate_stage,
                        stateful_plan,
                        graph_family=step.family,
                        moe_slot=(
                            GRAPH_MOE_SLOTS[step.family]
                            if graph_slots
                            else EAGER_MOE_SLOT
                        ),
                    )
                    cursor.advance_host(step.family)
                torch.cuda.synchronize(device)

            # E1a27/E1F warmup: once on the default stream with the eager slot,
            # once on the capture stream with the family slots, so every
            # captured kernel and slot buffer is already warm.
            capture_stream = torch.cuda.Stream(device=device)
            warm_cycle(graph_slots=False)
            reset_cycle()
            with torch.cuda.stream(capture_stream):
                warm_cycle(graph_slots=True)
            torch.cuda.synchronize(device)
            reset_cycle()
            global_rows = EXPECTED_TP_SIZE * LOCAL_BATCH
            for slot in GRAPH_MOE_SLOT_TUPLE:
                for moe in candidate_stage.moes:
                    moe.reset_free_slot_completion_event(global_rows, slot)

            # Arm S: the *stateful eager* path on the same handed-off state.
            #
            # Without this arm a graph-vs-arm-R mismatch is unattributable: arm
            # R and arm C both run the non-stateful decode implementation,
            # while the graph runs the stateful one, so a disagreement could be
            # capture OR a legitimate sum-order difference between two decode
            # implementations (9.6).  Arm S isolates that: graph vs S is the
            # E0sf comparison (same implementation, must be bitwise), S vs R is
            # the implementation difference.
            stateful_eager_outputs: list[torch.Tensor] = []
            for index, step in enumerate(schedule):
                residual, ids = step_inputs[index]
                stateful_plan.input_residual_buffer.copy_(residual)
                stateful_plan.input_ids_buffer.copy_(ids)
                output = forward_eager_prevalidated(
                    candidate_stage,
                    stateful_plan,
                    graph_family=step.family,
                    moe_slot=EAGER_MOE_SLOT,
                )
                torch.cuda.synchronize(device)
                stateful_eager_outputs.append(output.clone())
            result["stateful_eager_steps"] = [
                dict(
                    error_metrics(stateful_eager_outputs[index], reference_outputs[index]),
                    step=index,
                    position=step.position,
                    family=step.family.value,
                )
                for index, step in enumerate(schedule)
            ]
            reset_cycle()
            for slot in GRAPH_MOE_SLOT_TUPLE:
                for moe in candidate_stage.moes:
                    moe.reset_free_slot_completion_event(global_rows, slot)

            graphs: dict[DecodeGraphFamily, Any] = {}
            pools = {
                family: torch.cuda.graph_pool_handle()
                for family in DecodeGraphFamily
            }
            capture_order: list[str] = []
            graph_steps: list[dict[str, Any]] = []
            for index, step in enumerate(schedule):
                residual, ids = step_inputs[index]
                stateful_plan.input_residual_buffer.copy_(residual)
                stateful_plan.input_ids_buffer.copy_(ids)
                if step.family not in graphs:
                    graphs[step.family] = capture_stateful_graph(
                        candidate_stage,
                        stateful_plan,
                        graph_family=step.family,
                        capture_stream=capture_stream,
                        pool=pools[step.family],
                    )
                    capture_order.append(step.family.value)
                output = replay_stateful_graph(
                    graphs[step.family],
                    stateful_plan,
                    graph_family=step.family,
                )
                torch.cuda.synchronize(device)
                metrics = error_metrics(output, reference_outputs[index])
                metrics["step"] = index
                metrics["position"] = step.position
                metrics["family"] = step.family.value
                # The gate's real graph criterion: same implementation, so this
                # one must be bitwise.  vs_reference is reported for context but
                # spans two decode implementations.
                metrics["vs_stateful_eager"] = error_metrics(
                    output, stateful_eager_outputs[index]
                )
                graph_steps.append(metrics)
                if not metrics["vs_stateful_eager"]["bitwise"]:
                    graph_mismatched.append(step.position)
            result["graph_steps"] = graph_steps
            result["graph_capture_order"] = capture_order
            result["graph_mismatched_positions"] = graph_mismatched

        result["accepted"] = (
            not mismatched
            and not graph_mismatched
            and bool(result["steps"])
            and (not args.graph_arm or bool(result.get("graph_steps")))
        )

    except Exception as error:  # noqa: BLE001 - a rank-local failure must be loud
        result["errors"].append(
            {"type": type(error).__name__, "message": str(error),
             "traceback": traceback.format_exc()}
        )
        print(f"[E7F] rank {rank} FAILED: {error}", flush=True)
        traceback.print_exc()

    write_json(out_dir / f"rank{rank}.json", result)
    if rank == 0:
        print(
            f"[E7F] handoff accepted={result['accepted']} "
            f"eager_mismatched={result.get('mismatched_positions')} "
            f"graph_mismatched={result.get('graph_mismatched_positions')}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
