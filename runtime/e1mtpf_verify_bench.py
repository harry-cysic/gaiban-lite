#!/usr/bin/env python3
"""E1MTPF: MTP verify-round wall-time bench at production ctx (B=1, eager).

Gate (c) timing counterpart: the golden gate (e0mtp2e) measures acceptance and
token equality on real prompts (ctx ~15-30); this bench measures the
draft-verify **round cost** at production context 2048 with synthetically
seeded eager full-position states (E0ff/E1F seeding practice: values are
deterministic small-scale noise; timing-only, no semantic claims).

Measured, per pipeline form (16 ranks, eager full-position lanes -- the same
execution form whose B=1 single-token step was measured against the graph
path in E1F: eager fused-HC 55-57 ms vs graph 36.3 ms):

  baseline   -- single-token closed-loop steps (token feedback broadcast).
  accept     -- verify-2 fused pass + decision broadcast + 2 MTP steps +
                draft broadcast (positions advance +2).
  reject     -- verify-2 fused pass + decision broadcast + state rollback +
                1 MTP step + draft broadcast (positions advance +1).

Effective ms/token at acceptance rate a:
  (a * t_accept + (1-a) * t_reject) / (1 + a)
with a taken from the golden gate's measured acceptance.  The projection onto
the graph regime scales by the measured verify2/step ratio (both regimes are
weight-read-bound at B=1, where a two-row GEMM costs the same weight traffic
as one row); the graph-capture of the two-token verify step itself is the
deferred large-B design item.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import e0mtp2e_golden_gate as gate
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.head_stage import head_logits, head_logits_all, load_embed_head_material
from dsv4_direct.hc_boundary_backend import resolve_hc_boundary_backend
from dsv4_direct.model_contract import MTP_LAYER_ID
from dsv4_direct.mtp_block import build_mtp_layer_material
from dsv4_direct.physical_stage import EXPECTED_TP_SIZE, build_physical_stage
from dsv4_direct.ratio4_fullpos import Ratio4FullPositionAttention
from dsv4_direct.static_kv import StaticLayerKV
from dsv4_direct.static_window_kv import StaticWindowKV


CTX = 2048
MAX_SEQ_LEN = 4096
VOCAB = 129280
WINDOW = 128
RATIO128 = 128
RATIO4 = 4


def deterministic(seed: int, shape: tuple[int, ...], device, scale=0.02, dtype=torch.bfloat16):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32) * scale
    return value.to(dtype).to(device)


def seed_window_state(state: StaticWindowKV, *, position: int, seed: int) -> None:
    device = state.device
    payload = deterministic(seed, tuple(state.latent.shape), device)
    state.latent.copy_(payload)
    if state.latent_rope is not None:
        state.latent_rope.copy_(payload[..., -state.latent_rope.shape[-1] :])
    absolute = torch.arange(position - WINDOW, position, dtype=torch.int64, device=device)
    slots = absolute.remainder(WINDOW)
    state._raw_positions.index_copy_(
        1, slots, absolute.unsqueeze(0).expand(state.num_local_sequences, -1)
    )
    state._next_position.fill_(position)


def seed_ratio128_state(state: StaticLayerKV, *, position: int, seed: int) -> None:
    if position % RATIO128:
        raise ValueError("ratio-128 seed position must be group-aligned")
    device = state.device
    completed = position // RATIO128
    state.seed_decode_residency(
        start_pos=position,
        raw=deterministic(seed, (state.num_local_sequences, WINDOW, 512), device),
        compressed=deterministic(
            seed + 1, (state.num_local_sequences, completed, 512), device
        ),
    )


def seed_ratio4_state(
    attention: Ratio4FullPositionAttention, *, position: int, seed: int
) -> None:
    if position % RATIO4:
        raise ValueError("ratio-4 seed position must be group-aligned")
    device = attention.device
    count = position // RATIO4
    attention.raw.copy_(deterministic(seed, tuple(attention.raw.shape), device))
    attention.compressed[:, :count].copy_(
        deterministic(seed + 1, (attention.raw.shape[0], count, 512), device)
    )
    attention.indexer_kv[:, :count].copy_(
        deterministic(seed + 2, (attention.raw.shape[0], count, 128), device)
    )
    if attention.raw_rope is not None:
        attention.raw_rope.copy_(
            deterministic(seed + 3, tuple(attention.raw_rope.shape), device)
        )
    if attention.compressed_rope is not None:
        attention.compressed_rope[:, :count].copy_(
            deterministic(seed + 4, (attention.raw.shape[0], count, 64), device)
        )
    for offset, tensor in enumerate(
        (
            attention.main_kv_state,
            attention.main_score_state,
            attention.index_kv_state,
            attention.index_score_state,
        )
    ):
        tensor.copy_(
            deterministic(
                seed + 10 + offset, tuple(tensor.shape), device, scale=0.05,
                dtype=torch.float32,
            )
        )
    attention.next_position = position
    attention.compressed_count = count


def seed_lane(lane: gate.StageLane, *, position: int, seed: int) -> None:
    for index, (material, attention) in enumerate(lane.layers):
        layer_seed = seed + 1000 * material.layer_id + index
        if material.kind == "window":
            seed_window_state(attention.state, position=position, seed=layer_seed)
        elif material.kind == "ratio4":
            seed_ratio4_state(attention, position=position, seed=layer_seed)
        else:
            seed_ratio128_state(attention.state, position=position, seed=layer_seed)


def token_sequence(seed: int, count: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, VOCAB, (count,), generator=generator).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kv-dtype", type=str, default="fp8",
                        choices=("bf16", "fp8", "fp8_rope_bf16"))
    parser.add_argument("--indexer-kv-dtype", type=str, default="bf16",
                        choices=("bf16", "fp8"))
    parser.add_argument("--hc-backend", type=str, default="fused",
                        choices=("eager", "fused"))
    parser.add_argument("--settle", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)
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
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E1MTPF-mtp-verify-round-bench",
        "measurement_class": "eager_fullpos_wall_time_b1_ctx2048",
        "rank": rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "kv_dtype": args.kv_dtype,
        "indexer_kv_dtype": args.indexer_kv_dtype,
        "hc_backend": args.hc_backend,
        "ctx": CTX,
        "settle": args.settle,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "phases": {},
        "checkpoint_id": None,
        "accepted": False,
        "errors": [],
        "diagnostic_seconds": {},
    }
    started = time.perf_counter()

    try:
        if world != gate.WORLD:
            raise ValueError(f"E1MTPF requires world=16, got {world}")
        topo = gate.create_pp4_groups(rank)
        stage = topo["stage"]
        tp_rank = topo["tp_rank"]
        result["stage"] = stage
        warm = torch.ones(1, device=device)
        dist.all_reduce(warm, group=topo["tp_group"])
        if topo["next_pair"] is not None:
            gate.pair_transfer(warm, send=True, group=topo["next_pair"])
        if topo["prev_pair"] is not None:
            gate.pair_transfer(warm, send=False, group=topo["prev_pair"])
        torch.cuda.synchronize(device)
        result["placement"] = gate.run_placement_check(stage=stage, world=world)
        if not result["placement"]["accepted"]:
            raise ValueError("PP4 placement violated")

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config_payload = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                checkpoint = inspect_stage_checkpoint(
                    stage_root,
                    list(range(gate.MODEL_LAYERS)) + [MTP_LAYER_ID],
                    EXPECTED_TP_SIZE,
                )
                if not checkpoint["ok"]:
                    raise ValueError(f"checkpoint contract failed: {checkpoint['errors'][:4]}")
                envelope_holder[0] = {
                    "ok": True,
                    "config": config_payload,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                }
            except Exception:
                envelope_holder[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        if not envelope["ok"]:
            raise ValueError(f"rank-0 preflight failed:\n{envelope['error']}")
        result["checkpoint_id"] = envelope["checkpoint_id"]
        model_config = envelope["config"]

        load_started = time.perf_counter()
        stage_material = build_physical_stage(
            stage_id=stage,
            layer_ids=gate.STAGE_LAYERS[stage],
            model_config=model_config,
            stage_root=stage_root,
            tp_rank=tp_rank,
            tp_group=topo["tp_group"],
            tp_global_ranks=topo["tp_global_ranks"],
            device=device,
            checkpoint_id=result["checkpoint_id"],
            max_seq_len=MAX_SEQ_LEN,
            global_row_shapes=(EXPECTED_TP_SIZE, 2 * EXPECTED_TP_SIZE),
            slots_per_shape=1,
            kv_dtype=args.kv_dtype,
            indexer_kv_dtype=args.indexer_kv_dtype,
            progress=(
                (lambda message: print(f"[E1MTPF] {message}", flush=True))
                if rank in (0, 12)
                else None
            ),
        )
        embed_material = None
        head_material = None
        mtp_material = None
        if stage == 0:
            embed_material = load_embed_head_material(
                stage_root=stage_root, device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=True, load_head=False,
            )
        elif stage == gate.STAGE_COUNT - 1:
            head_material = load_embed_head_material(
                stage_root=stage_root, device=device,
                checkpoint_id=result["checkpoint_id"],
                load_embed=True, load_head=True,
            )
            mtp_material = build_mtp_layer_material(
                model_config=model_config,
                stage_root=stage_root,
                tp_rank=tp_rank,
                tp_group=topo["tp_group"],
                tp_global_ranks=topo["tp_global_ranks"],
                device=device,
                checkpoint_id=result["checkpoint_id"],
                max_seq_len=MAX_SEQ_LEN,
                global_row_shapes=(EXPECTED_TP_SIZE,),
                slots_per_shape=1,
                kv_dtype=args.kv_dtype,
            )
        result["diagnostic_seconds"]["load"] = time.perf_counter() - load_started
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_after_load"] = {
            "free_bytes": int(free_bytes), "total_bytes": int(total_bytes)
        }
        dist.barrier()
        backend = resolve_hc_boundary_backend(
            None if args.hc_backend == "eager" else "fused"
        )
        tokens = token_sequence(args.seed, 16 * (args.settle + args.rounds) + 64)

        def fresh_lane(position: int) -> tuple[gate.StageLane, gate.MTPDriver]:
            lane = gate.StageLane(
                stage_material.materials, backend=backend, device=device
            )
            seed_lane(lane, position=position, seed=args.seed)
            driver = gate.MTPDriver(
                mtp_material=mtp_material,
                embed_head=head_material,
                device=device,
                active=stage == gate.STAGE_COUNT - 1,
            )
            if driver.lane is not None:
                seed_window_state(
                    driver.lane.state, position=position, seed=args.seed + 999_331
                )
            return lane, driver

        def run_baseline(repeat: int) -> dict[str, Any]:
            lane, _driver = fresh_lane(CTX)
            cursor = CTX
            token_iter = iter(tokens)
            samples: list[float] = []
            for step in range(args.settle + args.rounds):
                torch.cuda.synchronize(device)
                step_started = time.perf_counter()
                residual = gate.pipeline_pass(
                    step_tokens=[next(token_iter)],
                    position=cursor,
                    lane=lane,
                    topo=topo,
                    embed_material=embed_material,
                    device=device,
                )
                payload = None
                if stage == gate.STAGE_COUNT - 1:
                    logits = head_logits(head_material, residual)
                    payload = {"token": int(torch.argmax(logits[0]).item())}
                gate.broadcast_payload(payload)
                torch.cuda.synchronize(device)
                if step >= args.settle:
                    samples.append((time.perf_counter() - step_started) * 1e3)
                cursor += 1
            del lane
            return {"repeat": repeat, "samples_ms": samples}

        def run_rounds(
            repeat: int, *, forced_accept: bool, chained: bool = False
        ) -> dict[str, Any]:
            lane, driver = fresh_lane(CTX)
            cursor = CTX  # verify first position
            token_iter = iter(tokens)
            samples: list[float] = []
            verify_samples: list[float] = []
            for step in range(args.settle + args.rounds):
                pending = next(token_iter)
                draft = next(token_iter)
                torch.cuda.synchronize(device)
                round_started = time.perf_counter()
                snapshots: list[tuple[Any, dict[str, Any]]] = []
                if chained:
                    residual_first = gate.pipeline_pass(
                        step_tokens=[pending],
                        position=cursor,
                        lane=lane,
                        topo=topo,
                        embed_material=embed_material,
                        device=device,
                    )
                    snapshots = lane.snapshot_states()
                    residual_second = gate.pipeline_pass(
                        step_tokens=[draft],
                        position=cursor + 1,
                        lane=lane,
                        topo=topo,
                        embed_material=embed_material,
                        device=device,
                    )
                    residual_pair = torch.cat(
                        [residual_first, residual_second], dim=1
                    )
                else:
                    residual_pair = gate.pipeline_pass(
                        step_tokens=[pending, draft],
                        position=cursor,
                        lane=lane,
                        topo=topo,
                        embed_material=embed_material,
                        device=device,
                        verify2=True,
                        snapshot_out=snapshots,
                    )
                torch.cuda.synchronize(device)
                verify_done = time.perf_counter()
                decision = None
                if stage == gate.STAGE_COUNT - 1:
                    both = head_logits_all(head_material, residual_pair)
                    decision = {
                        "first": int(torch.argmax(both[0, 0]).item()),
                        "second": int(torch.argmax(both[0, 1]).item()),
                    }
                decision = gate.broadcast_payload(decision)
                if not forced_accept:
                    gate.StageLane.restore_states(snapshots)
                new_draft = None
                if driver.lane is not None:
                    if forced_accept:
                        driver.step(
                            residual_pair[:, 0:1].contiguous(),
                            decision["first"],
                            cursor,
                        )
                        new_draft = driver.step(
                            residual_pair[:, 1:2].contiguous(),
                            decision["second"],
                            cursor + 1,
                        )
                    else:
                        new_draft = driver.step(
                            residual_pair[:, 0:1].contiguous(),
                            decision["first"],
                            cursor,
                        )
                gate.broadcast_payload({"draft": new_draft})
                torch.cuda.synchronize(device)
                if step >= args.settle:
                    samples.append((time.perf_counter() - round_started) * 1e3)
                    verify_samples.append((verify_done - round_started) * 1e3)
                cursor += 2 if forced_accept else 1
            del lane, driver
            return {
                "repeat": repeat,
                "samples_ms": samples,
                "verify_ms": verify_samples,
            }

        def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
            merged = [value for record in records for value in record["samples_ms"]]
            summary = {
                "mean_ms": statistics.fmean(merged),
                "p50_ms": statistics.median(merged),
                "p95_ms": sorted(merged)[int(0.95 * len(merged))],
                "repeat_p50_ms": [
                    statistics.median(record["samples_ms"]) for record in records
                ],
            }
            verify = [
                value
                for record in records
                for value in record.get("verify_ms", [])
            ]
            if verify:
                summary["verify_p50_ms"] = statistics.median(verify)
            return summary

        for phase, runner in (
            ("baseline", lambda repeat: run_baseline(repeat)),
            (
                "fused_accept",
                lambda repeat: run_rounds(repeat, forced_accept=True),
            ),
            (
                "fused_reject",
                lambda repeat: run_rounds(repeat, forced_accept=False),
            ),
            (
                "chained_accept",
                lambda repeat: run_rounds(repeat, forced_accept=True, chained=True),
            ),
            (
                "chained_reject",
                lambda repeat: run_rounds(repeat, forced_accept=False, chained=True),
            ),
        ):
            phase_started = time.perf_counter()
            records = [runner(repeat) for repeat in range(args.repeats)]
            result["phases"][phase] = summarize(records)
            result["phases"][phase]["records"] = records
            result["diagnostic_seconds"][phase] = time.perf_counter() - phase_started
            if rank == 0:
                print(
                    f"[E1MTPF] {phase}: p50 "
                    f"{result['phases'][phase]['p50_ms']:.1f} ms "
                    f"(repeats {result['phases'][phase]['repeat_p50_ms']})",
                    flush=True,
                )

        # effective ms/token model at measured golden-gate acceptance rates
        base = result["phases"]["baseline"]["p50_ms"]
        result["effective_model"] = {"baseline_step_p50_ms": base}
        for form in ("fused", "chained"):
            t_acc = result["phases"][f"{form}_accept"]["p50_ms"]
            t_rej = result["phases"][f"{form}_reject"]["p50_ms"]
            result["effective_model"][form] = {
                "accept_round_p50_ms": t_acc,
                "reject_round_p50_ms": t_rej,
                "per_alpha": {
                    f"{alpha:.2f}": {
                        "effective_ms_per_token": (
                            (alpha * t_acc + (1 - alpha) * t_rej) / (1 + alpha)
                        ),
                        "speedup_vs_baseline": base
                        / ((alpha * t_acc + (1 - alpha) * t_rej) / (1 + alpha)),
                    }
                    for alpha in (0.5, 0.6, 0.66, 0.7, 0.8, 0.86, 0.9, 1.0)
                },
            }
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["memory_at_end"] = {
            "free_bytes": int(free_bytes), "total_bytes": int(total_bytes)
        }
        result["accepted"] = True
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["accepted"] = False
    result["diagnostic_seconds"]["process"] = time.perf_counter() - started

    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, result["accepted"])
    accepted_all = all(bool(value) for value in gathered)
    gate.write_json(out_dir / f"rank{rank}.json", result)
    if rank == 0:
        gate.write_json(
            out_dir / "result.json",
            {
                "experiment": "E1MTPF-mtp-verify-round-bench",
                "accepted": accepted_all,
                "checkpoint_id": result["checkpoint_id"],
                "kv_dtype": args.kv_dtype,
                "hc_backend": args.hc_backend,
                "ctx": CTX,
                "phases": {
                    name: {k: v for k, v in phase.items() if k != "records"}
                    for name, phase in result["phases"].items()
                },
                "effective_model": result.get("effective_model"),
                "errors": result["errors"],
            },
        )
        print(f"[E1MTPF] overall: {'PASS' if accepted_all else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if accepted_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
