#!/usr/bin/env python3
"""C2F: single-stage (TP4, 11 layers) chunked-prefill throughput bench.

Nineteenth vertical: establish the direct-runtime prefill baseline and wire
the two prefill-specific levers (W4A8 Marlin MoE, D0b fused Triton indexer).

Measurement口径 (stated once, used everywhere):

- One "chunk" is a single-shot prefill call at start_pos=0 with sequence
  length L (B=1 per rank).  This equals the cost of the last chunk of an
  L-long chunked prefill (O(s^2) indexer/attention terms are at their
  end-of-sequence size), matching gaiban D0b's "chunk" definition.
- DP form: each TP lane feeds its own distinct B=1 sequence (per-rank seed),
  so one stage pass processes 4*L input tokens; attention runs per lane with
  full heads, MoE all-gathers 4*L rows (the runtime's native DP-attention +
  intermediate-TP MoE collective order, unchanged).
- input tok/s/stage = 4*L / stage_pass_wall.  Full PP4 pipeline projection
  P(16 cards) ~= 4*L / t_stage assumes perfect chunk interleaving across 4
  stages and stage-1-like layer mix (stages are timed on L11-L21; stage 0 is
  cheaper -- window layers -- and stage 3 has 10 layers + head).
- Timing: host walls around torch.cuda.synchronize + dist.barrier fencing
  (E1F convention); eager, no CUDA graphs; open-loop single stage -- no PP
  handoff, no decode interleaving.

Modes:
  --moe-mode w4a16|w4a8      Marlin activation dtype (w4a8 repacks at load
                             with is_a_8bit + FP8 scale processing; semantic
                             change, gated by --gate-w4a8 + E2E regression).
  --indexer ref|fused        ratio-4 prefill index score backend
                             (fused = D0b Triton kernel, active at
                             seqlen >= --fuse-min-seqlen; semantic change,
                             gated by --gate-indexer + E2E regression).
  --gate-indexer             extra paired pass (ref vs fused on identical
                             tensors: timings, score delta, top-k agreement).
  --gate-w4a8                extra layer-level A/B: load first stage layer's
                             MoE in both repacks, same hidden, compare.

Run (single node titan064, GPUs 0-3):
  ~/Workspace/venvs/sglang/bin/torchrun --standalone --nproc_per_node=4 \
    c2f_prefill_stage_bench.py --stage-root ~/Workspace/DeepSeek-V4-Flash \
    --chunk 4096 --moe-mode w4a16 --indexer ref --out-dir out-c2f-...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import traceback
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from dsv4_direct.attention import rms_norm, torch_sparse_attention
from dsv4_direct.checkpoint import inspect_stage_checkpoint
from dsv4_direct.hyper_connections import hc_post, hc_pre
from dsv4_direct.moe_runtime import TP4MoE, TP4MoEConfig
from dsv4_direct.ops.marlin_moe import load_resident_moe_layer
from dsv4_direct.physical_stage import (
    EXPECTED_TP_SIZE,
    PhysicalLayerMaterial,
    build_physical_stage,
)
from dsv4_direct.ratio4_fullpos import Ratio4FullPositionAttention


LOCAL_BATCH = 1
HIDDEN = 4096
HC_MULT = 4
FP8 = torch.float8_e4m3fn


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# stage lane (prefill-only variant of the e0ef2e StageLane, eager composition)


class PrefillLane:
    def __init__(
        self,
        materials: list[PhysicalLayerMaterial],
        *,
        device: torch.device,
        index_score_mode: str,
        fuse_min_seqlen: int,
        sparse_row_block: int | None,
    ) -> None:
        self.device = device
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
                    index_score_mode=index_score_mode,
                    fuse_min_seqlen=fuse_min_seqlen,
                    sparse_row_block=sparse_row_block,
                )
            else:
                state = material.new_state(num_local_sequences=LOCAL_BATCH)
                attention = material.new_attention(state)
            self.layers.append((material, attention))

    @staticmethod
    def _attention_branch(
        material: PhysicalLayerMaterial,
        attention: Any,
        hidden: torch.Tensor,
        start_pos: int,
    ) -> torch.Tensor:
        if material.kind == "ratio4":
            return attention(hidden, start_pos=start_pos)
        branch, _trace = attention(hidden, start_pos=start_pos)
        return branch

    def forward(
        self,
        residual: torch.Tensor,
        *,
        start_pos: int = 0,
        component_walls: dict[str, float] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Eager 11-layer chain (== e0ef2e StageLane._layer_eager order)."""

        def timed(name: str, fn):
            if component_walls is None:
                return fn()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            value = fn()
            torch.cuda.synchronize(device)
            component_walls[name] = component_walls.get(name, 0.0) + (
                time.perf_counter() - started
            )
            return value

        for material, attention in self.layers:
            hc = material.raw_block.hyper_connection
            hidden, post, comb = timed(
                "hc",
                lambda: hc_pre(
                    residual,
                    hc.attn_fn,
                    hc.attn_scale,
                    hc.attn_base,
                    norm_eps=material.norm_eps,
                    sinkhorn_iters=material.sinkhorn_iters,
                    hc_eps=material.hc_eps,
                ),
            )
            hidden = timed(
                "norm",
                lambda: rms_norm(
                    hidden, material.raw_block.attn_norm, eps=material.norm_eps
                ),
            )
            branch = timed(
                f"attention_{material.kind}",
                lambda: self._attention_branch(
                    material, attention, hidden, start_pos
                ),
            )
            residual = timed("hc", lambda: hc_post(branch, residual, post, comb))
            hidden, post, comb = timed(
                "hc",
                lambda: hc_pre(
                    residual,
                    hc.ffn_fn,
                    hc.ffn_scale,
                    hc.ffn_base,
                    norm_eps=material.norm_eps,
                    sinkhorn_iters=material.sinkhorn_iters,
                    hc_eps=material.hc_eps,
                ),
            )
            hidden = timed(
                "norm",
                lambda: rms_norm(
                    hidden, material.raw_block.ffn_norm, eps=material.norm_eps
                ),
            )
            moe_output = timed(
                "moe",
                lambda: material.moe.forward_tensor(hidden, slot=0),
            )
            residual = timed(
                "hc", lambda: hc_post(moe_output, residual, post, comb)
            )
        return residual

    def collect_index_gate_records(self) -> list[dict]:
        records: list[dict] = []
        for material, attention in self.layers:
            if material.kind == "ratio4":
                records.extend(attention.index_gate_records)
        return records


# --------------------------------------------------------------------------
# standalone correctness spot-checks (cheap, run once per bench)


def ring_wrap_unit_check(device: torch.device) -> dict[str, Any]:
    """Our index_copy_ ring placement == reference model.py:518-523 split."""

    results = {}
    window = 128
    for seqlen in (96, 128, 130, 1024 + 7, 2048):
        kv = torch.randn(1, seqlen, 512, dtype=torch.bfloat16, device=device)
        # reference form
        ref = torch.zeros(1, window, 512, dtype=torch.bfloat16, device=device)
        if seqlen <= window:
            ref[:, :seqlen] = kv
        else:
            cutoff = seqlen % window
            tail, head = kv[:, -window:].split([window - cutoff, cutoff], dim=1)
            ref[:, cutoff:window] = tail
            ref[:, :cutoff] = head
        # runtime form (ratio4_fullpos prefill ring write)
        ours = torch.zeros_like(ref)
        kept = min(seqlen, window)
        slots = torch.arange(seqlen - kept, seqlen, device=device).remainder(
            window
        )
        ours.index_copy_(1, slots, kv[:, seqlen - kept :].contiguous())
        results[str(seqlen)] = bool(torch.equal(ref, ours))
    results["pass"] = all(bool(v) for v in results.values())
    return results


def ref_score_row_block_unit_check(device: torch.device) -> dict[str, Any]:
    """Row-blocked ref index scoring is bitwise identical to single-shot."""

    generator = torch.Generator(device=device)
    generator.manual_seed(20260722)
    q = torch.randn(
        1, 2048, 64, 128, dtype=torch.bfloat16, device=device, generator=generator
    )
    kv = torch.randn(
        1, 512, 128, dtype=torch.bfloat16, device=device, generator=generator
    )
    w = torch.randn(
        1, 2048, 64, dtype=torch.bfloat16, device=device, generator=generator
    )
    whole = Ratio4FullPositionAttention._ref_index_scores(q, kv, w)
    Ratio4FullPositionAttention._REF_SCORE_ROW_BLOCK_OVERRIDE = 640
    try:
        blocked = Ratio4FullPositionAttention._ref_index_scores(q, kv, w)
    finally:
        Ratio4FullPositionAttention._REF_SCORE_ROW_BLOCK_OVERRIDE = None
    return {"bitwise_equal": bool(torch.equal(whole, blocked))}


def sliced_sparse_unit_check(device: torch.device) -> dict[str, Any]:
    """Row-blocked torch_sparse_attention is bitwise identical to one call."""

    generator = torch.Generator(device=device)
    generator.manual_seed(20260721)
    s, heads, dim, window, t_rows = 1024, 64, 512, 128, 256
    query = torch.randn(
        1, s, heads, dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    kv = torch.randn(
        1, s + t_rows, dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    sink = torch.randn(
        heads, dtype=torch.float32, device=device, generator=generator
    )
    base = torch.arange(s, device=device).unsqueeze(1)
    columns = torch.arange(window, device=device)
    win = ((base - window + 1).clamp(0) + columns)
    win = torch.where(win > base, -1, win)
    comp = torch.arange(t_rows, device=device).repeat(s, 1)
    visible = torch.arange(1, s + 1, device=device).unsqueeze(1) // 4
    comp = torch.where(comp >= visible, -1, comp + s)
    topk = (
        torch.cat((win, comp), dim=-1).unsqueeze(0).to(torch.int32).contiguous()
    )
    whole = torch_sparse_attention(query, kv, sink, topk, dim**-0.5)
    sliced = torch.cat(
        [
            torch_sparse_attention(
                query[:, begin : begin + 256],
                kv,
                sink,
                topk[:, begin : begin + 256],
                dim**-0.5,
            )
            for begin in range(0, s, 256)
        ],
        dim=1,
    )
    return {"bitwise_equal": bool(torch.equal(whole, sliced))}


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="11-21", help="first-last inclusive")
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--moe-mode", choices=["w4a16", "w4a8"], default="w4a16")
    parser.add_argument("--indexer", choices=["ref", "fused"], default="ref")
    parser.add_argument("--fuse-min-seqlen", type=int, default=1024)
    parser.add_argument("--row-block", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--gate-indexer", action="store_true")
    parser.add_argument("--gate-w4a8", action="store_true")
    parser.add_argument("--progress-every", type=int, default=128)
    args = parser.parse_args()

    # ratio-128 prefill sparse-core row blocking (bitwise identical; bounds
    # the FP32 gather workspace) -- must be set before any prefill call.
    if args.row_block > 0:
        os.environ["DSV4_PREFILL_SPARSE_ROW_BLOCK"] = str(args.row_block)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(device)

    stage_root = args.stage_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    first_layer, last_layer = (int(v) for v in args.layers.split("-"))
    layer_ids = tuple(range(first_layer, last_layer + 1))
    chunk = int(args.chunk)
    if chunk <= 0 or chunk % 128:
        raise ValueError("chunk must be a positive multiple of 128")
    max_seq_len = chunk
    global_rows = LOCAL_BATCH * chunk * EXPECTED_TP_SIZE
    marlin_input_dtype = FP8 if args.moe_mode == "w4a8" else None

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "C2F-prefill-stage-bench",
        "measurement_class": "open_loop_single_stage_prefill",
        "koujing": (
            "single-shot prefill at start_pos=0, B=1/lane, 4 DP lanes with "
            "distinct sequences; input tok/s/stage = 4*chunk / stage wall; "
            "host walls around torch.cuda.synchronize; eager, no CUDA graph, "
            "no PP handoff"
        ),
        "rank": rank,
        "world": world,
        "host": platform.node(),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": package_version("vllm"),
        "triton": package_version("triton"),
        "layers": list(layer_ids),
        "chunk": chunk,
        "global_rows": global_rows,
        "moe_mode": args.moe_mode,
        "indexer": args.indexer,
        "fuse_min_seqlen": args.fuse_min_seqlen,
        "row_block": args.row_block,
        "iters": args.iters,
        "warmup": args.warmup,
        "seed": args.seed,
        "input_scale": args.input_scale,
        "errors": [],
    }

    try:
        if world != EXPECTED_TP_SIZE:
            raise ValueError("bench requires exactly 4 ranks (single TP4 stage)")
        tp_group = dist.new_group(ranks=list(range(EXPECTED_TP_SIZE)), backend="nccl")

        envelope_holder: list[Any] = [None]
        if rank == 0:
            try:
                config = json.loads(
                    (stage_root / "config.json").read_text(encoding="utf-8")
                )
                contract = inspect_stage_checkpoint(stage_root, tp_size=world)
                if not contract["ok"]:
                    raise ValueError(
                        f"checkpoint contract failed: {contract['errors'][:3]}"
                    )
                envelope_holder[0] = {
                    "ok": True,
                    "config": config,
                    "checkpoint_id": contract["checkpoint_id"],
                }
            except Exception:
                envelope_holder[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(envelope_holder, src=0)
        envelope = envelope_holder[0]
        if not envelope["ok"]:
            raise ValueError(f"rank-0 preflight failed:\n{envelope['error']}")
        config = envelope["config"]
        result["checkpoint_id"] = envelope["checkpoint_id"]

        # ---------------- correctness spot-checks (cheap) ----------------
        result["ring_wrap_unit_check"] = ring_wrap_unit_check(device)
        result["sliced_sparse_unit_check"] = sliced_sparse_unit_check(device)
        result["ref_score_row_block_unit_check"] = ref_score_row_block_unit_check(
            device
        )
        if not result["ring_wrap_unit_check"]["pass"]:
            raise RuntimeError("ring wrap unit check failed")
        if not result["sliced_sparse_unit_check"]["bitwise_equal"]:
            raise RuntimeError("sliced sparse unit check failed")
        if not result["ref_score_row_block_unit_check"]["bitwise_equal"]:
            raise RuntimeError("ref score row-block unit check failed")

        # ---------------- stage load ----------------
        load_started = time.perf_counter()
        stage = build_physical_stage(
            stage_id=1,
            layer_ids=layer_ids,
            model_config=config,
            stage_root=stage_root,
            tp_rank=rank,
            tp_group=tp_group,
            tp_global_ranks=tuple(range(EXPECTED_TP_SIZE)),
            device=device,
            checkpoint_id=envelope["checkpoint_id"],
            max_seq_len=max_seq_len,
            global_row_shapes=(global_rows,),
            slots_per_shape=1,
            progress_every=args.progress_every,
            progress=(lambda m: print(m, flush=True)) if rank == 0 else None,
            moe_marlin_input_dtype=marlin_input_dtype,
            share_moe_buffers=True,
        )
        torch.cuda.synchronize(device)
        dist.barrier()
        result["load_seconds"] = time.perf_counter() - load_started
        result["moe_resident_bytes_layer0"] = int(
            stage.materials[0].moe.resident.resident_bytes
        )
        for material in stage.materials:
            if material.route_kind != "learned":
                raise ValueError(
                    "bench stage must be all-learned routing (no input_ids fed)"
                )

        materials = list(stage.materials)
        row_block = args.row_block if args.row_block > 0 else None

        def make_lane(index_score_mode: str) -> PrefillLane:
            return PrefillLane(
                materials,
                device=device,
                index_score_mode=index_score_mode,
                fuse_min_seqlen=args.fuse_min_seqlen,
                sparse_row_block=row_block,
            )

        def make_residual(iteration: int) -> torch.Tensor:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + 1000 * rank + iteration)
            return (
                torch.randn(
                    LOCAL_BATCH,
                    chunk,
                    HC_MULT,
                    HIDDEN,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                * args.input_scale
            ).contiguous()

        # ---------------- timed prefill passes ----------------
        walls: list[float] = []
        for iteration in range(args.warmup + args.iters):
            lane = make_lane(args.indexer)
            residual = make_residual(iteration)
            torch.cuda.synchronize(device)
            dist.barrier()
            started = time.perf_counter()
            output = lane.forward(residual, start_pos=0)
            torch.cuda.synchronize(device)
            dist.barrier()
            wall = time.perf_counter() - started
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError(f"non-finite stage output at iter {iteration}")
            if iteration >= args.warmup:
                walls.append(wall)
            if rank == 0:
                print(
                    f"iter {iteration} ({'warmup' if iteration < args.warmup else 'timed'}): "
                    f"{wall * 1e3:.1f} ms",
                    flush=True,
                )
            del lane, residual, output

        p50 = statistics.median(walls)
        result["stage_pass_walls_s"] = walls
        result["stage_pass_wall_p50_s"] = p50
        result["stage_pass_wall_mean_s"] = statistics.fmean(walls)
        result["input_tok_s_per_stage_dp4"] = 4 * chunk / p50
        result["input_tok_s_per_lane"] = chunk / p50
        result["pipeline16_projection_tok_s"] = 4 * chunk / p50

        # ---------------- one instrumented pass (component walls) --------
        lane = make_lane(args.indexer)
        residual = make_residual(10_000)
        component_walls: dict[str, float] = {}
        torch.cuda.synchronize(device)
        dist.barrier()
        started = time.perf_counter()
        lane.forward(
            residual,
            start_pos=0,
            component_walls=component_walls,
            device=device,
        )
        torch.cuda.synchronize(device)
        component_walls["total_instrumented"] = time.perf_counter() - started
        result["component_walls_s"] = component_walls
        del lane, residual

        # ---------------- paired indexer gate (optional) -----------------
        if args.gate_indexer:
            lane = make_lane("paired_gate")
            residual = make_residual(20_000)
            lane.forward(residual, start_pos=0)
            torch.cuda.synchronize(device)
            result["index_gate_records"] = lane.collect_index_gate_records()
            del lane, residual

        # ---------------- W4A8 layer-level A/B gate (optional) -----------
        if args.gate_w4a8:
            gate_layer = materials[0]
            other_dtype = None if marlin_input_dtype is not None else FP8
            other_resident = load_resident_moe_layer(
                stage_root=stage_root,
                layer_id=gate_layer.layer_id,
                rank=rank,
                world_size=EXPECTED_TP_SIZE,
                hidden_size=int(config["hidden_size"]),
                intermediate_size=int(config["moe_intermediate_size"]),
                n_experts=int(config["n_routed_experts"]),
                device=device,
                progress_every=args.progress_every,
                checkpoint_id=envelope["checkpoint_id"],
                marlin_input_dtype=other_dtype,
            )
            other_moe = TP4MoE(
                config=TP4MoEConfig(
                    hidden_size=int(config["hidden_size"]),
                    intermediate_size=int(config["moe_intermediate_size"]),
                    experts=int(config["n_routed_experts"]),
                    topk=int(config["num_experts_per_tok"]),
                    route_scale=float(config["routed_scaling_factor"]),
                    clamp_limit=float(config["swiglu_limit"]),
                    world_size=EXPECTED_TP_SIZE,
                ),
                resident=other_resident,
                gate=gate_layer.raw_block.gate,
                rank=rank,
                device=device,
                global_row_shapes=(global_rows,),
                group=tp_group,
                slots_per_shape=1,
                marlin_input_dtype=other_dtype,
                buffer_donor=gate_layer.moe,
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + 777 + rank)
            gate_hidden = (
                torch.randn(
                    LOCAL_BATCH,
                    chunk,
                    HIDDEN,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                * args.input_scale
            ).contiguous()
            gate_hidden = rms_norm(
                gate_hidden,
                gate_layer.raw_block.ffn_norm,
                eps=gate_layer.norm_eps,
            )
            out_this = gate_layer.moe.forward_tensor(gate_hidden, slot=0)
            out_other = other_moe.forward_tensor(gate_hidden, slot=0)
            this_is_a16 = marlin_input_dtype is None
            out_a16 = out_this if this_is_a16 else out_other
            out_a8 = out_other if this_is_a16 else out_this
            diff = (out_a8.float() - out_a16.float())
            rel_fro = float(
                diff.norm() / out_a16.float().norm().clamp_min(1e-30)
            )
            result["w4a8_gate"] = {
                "layer_id": gate_layer.layer_id,
                "rows_local": LOCAL_BATCH * chunk,
                "rel_fro_w4a8_vs_w4a16": rel_fro,
                "max_abs_diff": float(diff.abs().max().item()),
                "w4a16_rms": float(
                    out_a16.float().square().mean().sqrt().item()
                ),
                "finite": bool(
                    torch.isfinite(out_a8).all().item()
                    and torch.isfinite(out_a16).all().item()
                ),
                "a3f_reference": (
                    "A3F numeric gate: marlin W4A8 rel_fro vs fp32 oracle "
                    "4.6e-2/8.5e-2 (== reference tilelang magnitude); "
                    "W4A16 4.2e-3/3.5e-3"
                ),
            }
            del other_moe, other_resident, gate_hidden, out_this, out_other

        result["max_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        result["max_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
        result["ok"] = True
    except Exception:
        result["ok"] = False
        result["errors"].append(traceback.format_exc())

    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, result)
    if rank == 0:
        summary = dict(gathered[0])
        summary["per_rank_max_memory_allocated_bytes"] = [
            entry.get("max_memory_allocated_bytes") for entry in gathered
        ]
        summary["per_rank_ok"] = [entry.get("ok") for entry in gathered]
        summary["per_rank_errors"] = [entry.get("errors") for entry in gathered]
        tag = f"chunk{chunk}-{args.moe_mode}-{args.indexer}"
        write_json(out_dir / f"c2f-{tag}.json", summary)
        print(json.dumps(
            {
                "chunk": chunk,
                "moe_mode": args.moe_mode,
                "indexer": args.indexer,
                "wall_p50_s": summary.get("stage_pass_wall_p50_s"),
                "input_tok_s_per_stage_dp4": summary.get(
                    "input_tok_s_per_stage_dp4"
                ),
                "ok": summary.get("per_rank_ok"),
            },
            indent=2,
        ))
    dist.barrier()
    dist.destroy_process_group()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
