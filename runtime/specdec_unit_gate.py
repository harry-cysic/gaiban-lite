#!/usr/bin/env python3
"""Unit gate for dsv4_direct.specdec (18th vertical, single GPU).

Fabricated-weight, layer-level proofs that the row-position machinery is
bitwise faithful to the verified family path:

T1  uniform-position equivalence: B=4 rows all at the same position, K
    consecutive steps (covers ratio-4 and ratio-128 boundaries).  Row-position
    forward output and every owned state tensor must be bitwise identical to
    the family ``forward_stateful_decode_tensor`` at each step.
T2  chained-round protocol equivalence (the desync case): B=4 rows run
    pass-A/pass-B rounds with a forced per-row accept pattern (shadow +
    masked restore for ratio-4).  Every pass-A output and every accepted
    pass-B output must be bitwise identical to an independent B=1 family
    lane fed only the committed hidden sequence of that row.
T3  graph arm: the same protocol driven through captured CUDA graphs
    (pass-A graph with round head/snapshot, pass-B graph) must match an
    eager row-position twin bitwise every round.

Run: python specdec_unit_gate.py --out out-specdec-unit [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from dsv4_direct.attention import (
    PreparedAttentionWeights,
    Ratio128AttentionConfig,
    Ratio128TorchAttention,
)
from dsv4_direct.ratio4_attention import (
    PreparedRatio4AttentionWeights,
    Ratio4AttentionConfig,
    Ratio4TorchAttention,
)
from dsv4_direct.specdec import (
    Ratio4RowWS,
    _RATIO4_SHADOW_NAMES,
    build_layer_row_ws,
    ratio128_rowpos_forward,
    ratio4_rowpos_forward,
    window_rowpos_forward,
)
from dsv4_direct.static_kv import StaticLayerKV
from dsv4_direct.static_ratio4_kv import StaticRatio4KV
from dsv4_direct.static_window_kv import StaticWindowKV
from dsv4_direct.stateful_decode import (
    classify_decode_position,
    DecodeGraphFamily,
    ratio128_sparse_bucket_width,
)
from dsv4_direct.window_attention import (
    PreparedWindowAttentionWeights,
    WindowAttentionConfig,
    WindowTorchAttention,
)


CKPT = "0" * 64
START = 2048
MAX_SEQ = 4096
BATCH = 4


def det(seed: int, shape: tuple[int, ...], device, dtype=torch.bfloat16, scale=0.02):
    generator = torch.Generator(device="cpu").manual_seed(seed & ((1 << 62) - 1))
    value = torch.randn(*shape, generator=generator, dtype=torch.float32) * scale
    return value.to(dtype).to(device)


def build_window(device, kv_dtype: str, seed: int):
    cfg = WindowAttentionConfig(
        hidden_size=4096, num_heads=64, head_dim=512, rope_dim=64,
        q_lora_rank=1024, o_lora_rank=1024, o_groups=8, norm_eps=1e-6,
        rope_theta=10000.0, rope_factor=16.0, beta_fast=32, beta_slow=1,
        original_seq_len=0, max_seq_len=MAX_SEQ, layer_id=0,
    )
    weights = PreparedWindowAttentionWeights(
        attn_sink=det(seed + 1, (64,), device, torch.float32, 0.1),
        wq_a=det(seed + 2, (1024, 4096), device),
        q_norm=1.0 + det(seed + 3, (1024,), device, torch.float32, 0.05),
        wq_b=det(seed + 4, (64 * 512, 1024), device),
        wkv=det(seed + 5, (512, 4096), device),
        kv_norm=1.0 + det(seed + 6, (512,), device, torch.float32, 0.05),
        wo_a=det(seed + 7, (8 * 1024, 4096), device),
        wo_b=det(seed + 8, (4096, 8 * 1024), device),
        layer_id=0, rank=0, world_size=4, checkpoint_id=CKPT,
    )

    def new_state():
        state = StaticWindowKV(
            num_local_sequences=BATCH, max_seq_len=MAX_SEQ, layer_id=0,
            device=device, kv_dtype=kv_dtype,
        )
        state.seed_decode_residency(
            start_pos=START, raw=det(seed + 9, (BATCH, 128, 512), device, scale=0.03)
        )
        return state

    def new_attention(state):
        return WindowTorchAttention(cfg, weights, state)

    return new_state, new_attention, "window"


def build_ratio128(device, kv_dtype: str, seed: int):
    cfg = Ratio128AttentionConfig(
        hidden_size=4096, num_heads=64, head_dim=512, rope_dim=64,
        q_lora_rank=1024, o_lora_rank=1024, o_groups=8, norm_eps=1e-6,
        rope_theta=160000.0, rope_factor=16.0, beta_fast=32, beta_slow=1,
        original_seq_len=65536, max_seq_len=MAX_SEQ, layer_id=3,
    )
    weights = PreparedAttentionWeights(
        attn_sink=det(seed + 1, (64,), device, torch.float32, 0.1),
        wq_a=det(seed + 2, (1024, 4096), device),
        q_norm=1.0 + det(seed + 3, (1024,), device, torch.float32, 0.05),
        wq_b=det(seed + 4, (64 * 512, 1024), device),
        wkv=det(seed + 5, (512, 4096), device),
        kv_norm=1.0 + det(seed + 6, (512,), device, torch.float32, 0.05),
        wo_a=det(seed + 7, (8 * 1024, 4096), device),
        wo_b=det(seed + 8, (4096, 8 * 1024), device),
        compressor_ape=det(seed + 9, (128, 512), device, torch.float32, 0.1),
        compressor_wkv=det(seed + 10, (512, 4096), device, torch.float32),
        compressor_wgate=det(seed + 11, (512, 4096), device, torch.float32),
        compressor_norm=1.0 + det(seed + 12, (512,), device, torch.float32, 0.05),
        layer_id=3, rank=0, world_size=4, checkpoint_id=CKPT,
    )

    def new_state():
        state = StaticLayerKV(
            num_local_sequences=BATCH, max_seq_len=MAX_SEQ, layer_id=3,
            device=device, kv_dtype=kv_dtype,
        )
        state.seed_decode_residency(
            start_pos=START,
            raw=det(seed + 13, (BATCH, 128, 512), device, scale=0.03),
            compressed=det(seed + 14, (BATCH, START // 128, 512), device, scale=0.025),
        )
        return state

    def new_attention(state):
        return Ratio128TorchAttention(cfg, weights, state)

    return new_state, new_attention, "ratio128"


def build_ratio4(device, kv_dtype: str, indexer_dtype: str, seed: int):
    cfg = Ratio4AttentionConfig(
        hidden_size=4096, num_heads=64, head_dim=512, rope_dim=64,
        q_lora_rank=1024, o_lora_rank=1024, o_groups=8,
        index_n_heads=64, index_head_dim=128, index_topk=512, norm_eps=1e-6,
        rope_theta=160000.0, rope_factor=16.0, beta_fast=32, beta_slow=1,
        original_seq_len=65536, max_seq_len=MAX_SEQ, layer_id=2,
    )
    weights = PreparedRatio4AttentionWeights(
        attn_sink=det(seed + 1, (64,), device, torch.float32, 0.1),
        wq_a=det(seed + 2, (1024, 4096), device),
        q_norm=1.0 + det(seed + 3, (1024,), device, torch.float32, 0.05),
        wq_b=det(seed + 4, (64 * 512, 1024), device),
        wkv=det(seed + 5, (512, 4096), device),
        kv_norm=1.0 + det(seed + 6, (512,), device, torch.float32, 0.05),
        wo_a=det(seed + 7, (8 * 1024, 4096), device),
        wo_b=det(seed + 8, (4096, 8 * 1024), device),
        compressor_ape=det(seed + 9, (4, 1024), device, torch.float32, 0.1),
        compressor_wkv=det(seed + 10, (1024, 4096), device, torch.float32),
        compressor_wgate=det(seed + 11, (1024, 4096), device, torch.float32),
        compressor_norm=1.0 + det(seed + 12, (512,), device, torch.float32, 0.05),
        index_wq_b=det(seed + 13, (64 * 128, 1024), device),
        index_weights_proj=det(seed + 14, (64, 4096), device),
        index_compressor_ape=det(seed + 15, (4, 256), device, torch.float32, 0.1),
        index_compressor_wkv=det(seed + 16, (256, 4096), device, torch.float32),
        index_compressor_wgate=det(seed + 17, (256, 4096), device, torch.float32),
        index_compressor_norm=1.0 + det(seed + 18, (128,), device, torch.float32, 0.05),
        layer_id=2, rank=0, world_size=4, checkpoint_id=CKPT,
    )
    capacity = MAX_SEQ // 4

    def new_state():
        state = StaticRatio4KV(
            num_local_sequences=BATCH, max_seq_len=MAX_SEQ, layer_id=2,
            device=device, kv_dtype=kv_dtype, indexer_dtype=indexer_dtype,
        )
        state.seed_decode_payload(
            START,
            raw=det(seed + 19, (BATCH, 128, 512), device, scale=0.03),
            compressed=det(seed + 20, (BATCH, capacity, 512), device, scale=0.025),
            indexer_kv=det(seed + 21, (BATCH, capacity, 128), device, scale=0.05),
            main_kv_state=det(seed + 22, (BATCH, 8, 1024), device, torch.float32, 0.05),
            main_score_state=det(seed + 23, (BATCH, 8, 1024), device, torch.float32, 0.2),
            index_kv_state=det(seed + 24, (BATCH, 8, 256), device, torch.float32, 0.05),
            index_score_state=det(seed + 25, (BATCH, 8, 256), device, torch.float32, 0.2),
        )
        return state

    def new_attention(state):
        return Ratio4TorchAttention(cfg, weights, state)

    return new_state, new_attention, "ratio4"


def slice_state(source, row: int):
    """Independent B=1 copy of one batch row of a state."""

    if isinstance(source, StaticWindowKV):
        result = StaticWindowKV(
            num_local_sequences=1, max_seq_len=source.max_seq_len,
            layer_id=source.layer_id, device=source.device, kv_dtype=source.kv_dtype,
        )
    elif isinstance(source, StaticLayerKV):
        result = StaticLayerKV(
            num_local_sequences=1, max_seq_len=source.max_seq_len,
            layer_id=source.layer_id, device=source.device, kv_dtype=source.kv_dtype,
        )
    else:
        result = StaticRatio4KV(
            num_local_sequences=1, max_seq_len=source.max_seq_len,
            layer_id=source.layer_id, device=source.device,
            kv_dtype=source.kv_dtype, indexer_dtype=source.indexer_dtype,
        )
    for (name, dst), (_, src) in zip(
        result._owned_tensor_items(), source._owned_tensor_items(), strict=True
    ):
        dst.copy_(src[row : row + 1])
    return result


def family_forward(attention, hidden, plan, position: int):
    family = classify_decode_position(position)
    if isinstance(attention, WindowTorchAttention):
        return attention.forward_stateful_decode_tensor(hidden, plan=plan)
    if isinstance(attention, Ratio4TorchAttention):
        return attention.forward_stateful_decode_tensor(
            hidden, plan=plan,
            ratio4_boundary=family is not DecodeGraphFamily.NORMAL,
        )
    return attention.forward_stateful_decode_tensor(
        hidden, plan=plan,
        ratio128_boundary=family is DecodeGraphFamily.RATIO4_RATIO128_BOUNDARY,
    )


def rowpos_forward(attention, hidden, positions, ws):
    if isinstance(attention, WindowTorchAttention):
        return window_rowpos_forward(attention, hidden, positions, ws)
    if isinstance(attention, Ratio4TorchAttention):
        return ratio4_rowpos_forward(attention, hidden, positions, ws)
    return ratio128_rowpos_forward(attention, hidden, positions, ws)


def state_digests(state) -> dict[str, str]:
    import hashlib

    out = {}
    for name, tensor in state._owned_tensor_items():
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        out[name] = hashlib.sha256(raw).hexdigest()
    return out


def t1_uniform(builder, device, steps: int, seed: int) -> dict[str, Any]:
    new_state, new_attention, label = builder
    fam_att = new_attention(new_state())
    row_att = new_attention(new_state())
    stop = START + steps + 2
    position_tensor = torch.full((1,), START, dtype=torch.int64, device=device)
    fam_plan = fam_att.prepare_stateful_decode_plan(
        position=position_tensor, start_position=START, stop_position=stop
    )
    ws = build_layer_row_ws(
        row_att, batch=BATCH, stop_position=stop,
        ratio128_bucket_width=ratio128_sparse_bucket_width(START, stop - 1),
    )
    positions = torch.full((BATCH,), START, dtype=torch.int64, device=device)
    mismatches = []
    for step in range(steps):
        position = START + step
        hidden = det(seed + 31 * step, (BATCH, 1, 4096), device, scale=0.02)
        fam_out = family_forward(fam_att, hidden, fam_plan, position)
        row_out = rowpos_forward(row_att, hidden, positions, ws)
        if not torch.equal(fam_out, row_out):
            mismatches.append({"step": step, "kind": "output"})
        if step % 16 == 15 or step == steps - 1:
            if state_digests(fam_att.state) != state_digests(row_att.state):
                mismatches.append({"step": step, "kind": "state"})
        position_tensor.fill_(position + 1)
        positions.fill_(position + 1)
        if mismatches:
            break
    return {
        "test": "t1_uniform", "layer": label, "steps": steps,
        "mismatches": mismatches, "accepted": not mismatches,
    }


def accept_pattern(round_index: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + 977 * round_index)
    return (torch.rand(BATCH, generator=generator) < 0.6)


def t2_protocol(builder, device, rounds: int, seed: int) -> dict[str, Any]:
    """Pollution invariance: two chained-round runs share every committed
    input and the accept pattern but feed different pass-B hiddens on
    rejected rows.  All pass-A outputs, all accepted pass-B outputs, and the
    final state (after two trailing forced-accept flush rounds) must be
    bitwise identical -- proving rejected speculative work leaves no trace.
    Batch shapes are identical between the runs, so bitwise is exact."""

    new_state, new_attention, label = builder

    def run(vary_rejected: bool):
        att = new_attention(new_state())
        stop = START + 2 * (rounds + 2) + 4
        bucket = ratio128_sparse_bucket_width(START, stop - 1)
        ws = build_layer_row_ws(
            att, batch=BATCH, stop_position=stop, ratio128_bucket_width=bucket
        )
        positions = torch.full((BATCH,), START, dtype=torch.int64, device=device)
        advance = torch.zeros(BATCH, dtype=torch.int64, device=device)
        accept = torch.ones(BATCH, dtype=torch.int64, device=device)
        committed_a = []
        committed_b = []
        accepts_total = 0
        for round_index in range(rounds + 2):
            if round_index < rounds:
                pattern = accept_pattern(round_index, seed).to(device)
            else:
                pattern = torch.ones(BATCH, dtype=torch.bool, device=device)
            hidden_a = det(
                seed + 101 * round_index + 1, (BATCH, 1, 4096), device, scale=0.02
            )
            hidden_b = det(
                seed + 101 * round_index + 2, (BATCH, 1, 4096), device, scale=0.02
            )
            if vary_rejected:
                hidden_junk = det(
                    seed + 101 * round_index + 3, (BATCH, 1, 4096), device,
                    scale=0.02,
                )
                hidden_b = torch.where(
                    pattern.view(-1, 1, 1), hidden_b, hidden_junk
                )
            # round head
            positions.add_(advance)
            if isinstance(ws, Ratio4RowWS):
                keep = accept.ne(0)
                for name in _RATIO4_SHADOW_NAMES:
                    live = getattr(att.state, name)
                    mask = (
                        keep.view(-1, 1) if live.ndim == 2 else keep.view(-1, 1, 1)
                    )
                    live.copy_(torch.where(mask, live, ws.shadow[name]))
            out_a = rowpos_forward(att, hidden_a, positions, ws).clone()
            if isinstance(ws, Ratio4RowWS):
                for name in _RATIO4_SHADOW_NAMES:
                    ws.shadow[name].copy_(getattr(att.state, name))
            out_b = rowpos_forward(att, hidden_b, positions + 1, ws).clone()
            committed_a.append(out_a)
            committed_b.append((pattern.clone(), out_b))
            accepts_total += int(pattern.sum())
            accept.copy_(pattern.to(torch.int64))
            advance.copy_(1 + accept)
        return att, committed_a, committed_b, accepts_total, positions.clone()

    att1, a1, b1, accepts, pos1 = run(vary_rejected=False)
    att2, a2, b2, _, pos2 = run(vary_rejected=True)
    mismatches = []
    for round_index, (out1, out2) in enumerate(zip(a1, a2, strict=True)):
        if not torch.equal(out1, out2):
            mismatches.append({"round": round_index, "pass": "a"})
    for round_index, ((pattern, out1), (_, out2)) in enumerate(
        zip(b1, b2, strict=True)
    ):
        rows = pattern.view(-1, 1, 1)
        if not torch.equal(
            torch.where(rows, out1, torch.zeros_like(out1)),
            torch.where(rows, out2, torch.zeros_like(out2)),
        ):
            mismatches.append({"round": round_index, "pass": "b_accepted"})
    positions_equal = bool(torch.equal(pos1, pos2))
    state_equal = state_digests(att1.state) == state_digests(att2.state)
    return {
        "test": "t2_protocol_pollution_invariance", "layer": label,
        "rounds": rounds, "accepts_total": accepts,
        "positions_equal": positions_equal,
        "final_state_equal": bool(state_equal),
        "mismatches": mismatches[:16],
        "accepted": bool(not mismatches and positions_equal and state_equal),
    }


def t3_graph(builder, device, rounds: int, seed: int) -> dict[str, Any]:
    new_state, new_attention, label = builder
    graph_att = new_attention(new_state())
    eager_att = new_attention(new_state())
    stop = START + 2 * rounds + 4
    bucket = ratio128_sparse_bucket_width(START, stop - 1)
    ws_g = build_layer_row_ws(
        graph_att, batch=BATCH, stop_position=stop, ratio128_bucket_width=bucket
    )
    ws_e = build_layer_row_ws(
        eager_att, batch=BATCH, stop_position=stop, ratio128_bucket_width=bucket
    )
    positions_g = torch.full((BATCH,), START, dtype=torch.int64, device=device)
    positions_e = torch.full((BATCH,), START, dtype=torch.int64, device=device)
    advance = torch.zeros(BATCH, dtype=torch.int64, device=device)
    accept = torch.ones(BATCH, dtype=torch.int64, device=device)
    hidden_a = torch.zeros((BATCH, 1, 4096), dtype=torch.bfloat16, device=device)
    hidden_b = torch.zeros((BATCH, 1, 4096), dtype=torch.bfloat16, device=device)
    out_a = torch.zeros((BATCH, 1, 4096), dtype=torch.bfloat16, device=device)
    out_b = torch.zeros((BATCH, 1, 4096), dtype=torch.bfloat16, device=device)

    def head(att, ws, positions):
        positions.add_(advance)
        if isinstance(ws, Ratio4RowWS):
            keep = accept.ne(0)
            for name in _RATIO4_SHADOW_NAMES:
                live = getattr(att.state, name)
                mask = keep.view(-1, 1) if live.ndim == 2 else keep.view(-1, 1, 1)
                live.copy_(torch.where(mask, live, ws.shadow[name]))

    def snap(att, ws):
        if isinstance(ws, Ratio4RowWS):
            for name in _RATIO4_SHADOW_NAMES:
                ws.shadow[name].copy_(getattr(att.state, name))

    def body_a():
        head(graph_att, ws_g, positions_g)
        out_a.copy_(rowpos_forward(graph_att, hidden_a, positions_g, ws_g))
        snap(graph_att, ws_g)

    def body_b():
        out_b.copy_(rowpos_forward(graph_att, hidden_b, positions_g + 1, ws_g))

    # kernel warmup on a side stream, then restore graph-arm state
    capture_stream = torch.cuda.Stream(device=device)
    saved = [
        tensor.clone() for _, tensor in graph_att.state._owned_tensor_items()
    ]
    with torch.cuda.stream(capture_stream):
        body_a()
        body_b()
    torch.cuda.synchronize(device)
    for (name, tensor), snapshot_value in zip(
        graph_att.state._owned_tensor_items(), saved, strict=True
    ):
        tensor.copy_(snapshot_value)
    positions_g.fill_(START)
    if isinstance(ws_g, Ratio4RowWS):
        for name in _RATIO4_SHADOW_NAMES:
            ws_g.shadow[name].zero_()

    pool = torch.cuda.graph_pool_handle()
    current = torch.cuda.current_stream(device)
    graphs = []
    for body in (body_a, body_b):
        graph = torch.cuda.CUDAGraph()
        capture_stream.wait_stream(current)
        with torch.cuda.graph(graph, stream=capture_stream, pool=pool):
            body()
        current.wait_stream(capture_stream)
        torch.cuda.synchronize(device)
        graphs.append(graph)

    mismatches = []
    for round_index in range(rounds):
        pattern = accept_pattern(round_index, seed).to(device)
        hidden_a.copy_(det(seed + 71 * round_index + 1, (BATCH, 1, 4096), device))
        hidden_b.copy_(det(seed + 71 * round_index + 2, (BATCH, 1, 4096), device))
        graphs[0].replay()
        graph_out_a = out_a.clone()
        graphs[1].replay()
        graph_out_b = out_b.clone()
        # eager twin
        head(eager_att, ws_e, positions_e)
        eager_out_a = rowpos_forward(eager_att, hidden_a, positions_e, ws_e).clone()
        snap(eager_att, ws_e)
        eager_out_b = rowpos_forward(
            eager_att, hidden_b, positions_e + 1, ws_e
        ).clone()
        if not torch.equal(graph_out_a, eager_out_a):
            mismatches.append({"round": round_index, "pass": "a"})
        if not torch.equal(graph_out_b, eager_out_b):
            mismatches.append({"round": round_index, "pass": "b"})
        accept.copy_(pattern.to(torch.int64))
        advance.copy_(1 + accept)
        if len(mismatches) > 8:
            break
    torch.cuda.synchronize(device)
    if not torch.equal(positions_g, positions_e):
        mismatches.append({"kind": "positions_diverged"})
    state_equal = state_digests(graph_att.state) == state_digests(eager_att.state)
    for graph in graphs:
        graph.reset()
    return {
        "test": "t3_graph", "layer": label, "rounds": rounds,
        "final_state_equal": bool(state_equal),
        "mismatches": mismatches[:16],
        "accepted": bool(not mismatches and state_equal),
    }


def splice_rows(dst_state, src_state, rows):
    for (_, d), (_, s) in zip(
        dst_state._owned_tensor_items(), src_state._owned_tensor_items(),
        strict=True,
    ):
        d[rows] = s[rows]


def t4_desync(builder, device, steps: int, seed: int) -> dict[str, Any]:
    """Mixed per-row positions vs uniform references.

    Rows sit at offsets [0, 1, 2, 3] relative to base position p.  The mixed
    batch must produce, per row, bitwise the same output as a uniform batch
    at that row's position with identical row content (stock ops are
    row-independent, so any mismatch is a per-row indexing bug in the
    row-position machinery)."""

    new_state, new_attention, label = builder
    offsets = [0, 1, 2, 3]
    base = new_state()
    # states advanced by k uniform row-position steps, k = 0..3
    advanced = [base]
    atts_tmp = new_attention(base)
    for k in range(1, 4):
        prev = advanced[k - 1]
        nxt = new_state()
        for (_, d), (_, s) in zip(
            nxt._owned_tensor_items(), prev._owned_tensor_items(), strict=True
        ):
            d.copy_(s)
        att = new_attention(nxt)
        ws = build_layer_row_ws(
            att, batch=BATCH, stop_position=START + steps + 8,
            ratio128_bucket_width=ratio128_sparse_bucket_width(
                START, START + steps + 7
            ),
        )
        positions = torch.full(
            (BATCH,), START + k - 1, dtype=torch.int64, device=device
        )
        hidden = det(seed + 55_001 * k, (BATCH, 1, 4096), device, scale=0.02)
        rowpos_forward(att, hidden, positions, ws)
        advanced.append(nxt)
    # mixed state: row b from advanced[offsets[b]]
    mixed_state = new_state()
    for b, k in enumerate(offsets):
        for (_, d), (_, s) in zip(
            mixed_state._owned_tensor_items(),
            advanced[k]._owned_tensor_items(),
            strict=True,
        ):
            d[b : b + 1] = s[b : b + 1]
    att_mixed = new_attention(mixed_state)
    stop = START + steps + 16
    bucket = ratio128_sparse_bucket_width(START, stop - 1)
    ws_mixed = build_layer_row_ws(
        att_mixed, batch=BATCH, stop_position=stop, ratio128_bucket_width=bucket
    )
    ref_atts = []
    ref_ws = []
    for k in range(4):
        state = new_state()
        for (_, d), (_, s) in zip(
            state._owned_tensor_items(), advanced[k]._owned_tensor_items(),
            strict=True,
        ):
            d.copy_(s)
        att = new_attention(state)
        ref_atts.append(att)
        ref_ws.append(
            build_layer_row_ws(
                att, batch=BATCH, stop_position=stop,
                ratio128_bucket_width=bucket,
            )
        )
    positions_mixed = torch.tensor(
        [START + k for k in offsets], dtype=torch.int64, device=device
    )
    mismatches = []
    for step in range(steps):
        hidden = det(seed + 77_003 * step, (BATCH, 1, 4096), device, scale=0.02)
        out_mixed = rowpos_forward(
            att_mixed, hidden, positions_mixed, ws_mixed
        ).clone()
        for k in range(4):
            positions_ref = torch.full(
                (BATCH,), START + k + step, dtype=torch.int64, device=device
            )
            out_ref = rowpos_forward(
                ref_atts[k], hidden, positions_ref, ref_ws[k]
            )
            b = offsets.index(k)
            if not torch.equal(out_mixed[b : b + 1], out_ref[b : b + 1]):
                mismatches.append(
                    {"step": step, "row": b, "position": START + k + step}
                )
        positions_mixed += 1
        if len(mismatches) > 8:
            break
    return {
        "test": "t4_desync", "layer": label, "steps": steps,
        "mismatches": mismatches[:16], "accepted": not mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--t1-steps", type=int, default=132)
    parser.add_argument("--t2-rounds", type=int, default=136)
    parser.add_argument("--t3-rounds", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    accepted = True
    try:
        for kv_dtype, indexer_dtype in (("bf16", "bf16"), ("fp8", "fp8")):
            builders = [
                build_window(device, kv_dtype, args.seed + 10),
                build_ratio128(device, kv_dtype, args.seed + 20),
                build_ratio4(device, kv_dtype, indexer_dtype, args.seed + 30),
            ]
            for builder in builders:
                for test, count in (
                    (t1_uniform, args.t1_steps),
                    (t2_protocol, args.t2_rounds),
                    (t3_graph, args.t3_rounds),
                    (t4_desync, args.t1_steps),
                ):
                    record = test(builder, device, count, args.seed)
                    record["kv_dtype"] = kv_dtype
                    record["indexer_dtype"] = indexer_dtype
                    results.append(record)
                    accepted = accepted and record["accepted"]
                    print(
                        f"[unit] {record['test']} {record['layer']} kv={kv_dtype} "
                        f"-> {'PASS' if record['accepted'] else 'FAIL'} "
                        f"({record.get('mismatches')!r:.120s})",
                        flush=True,
                    )
    except Exception:
        accepted = False
        results.append({"error": traceback.format_exc()})
        print(traceback.format_exc(), flush=True)

    payload = {
        "experiment": "specdec-unit-gate",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "batch": BATCH,
        "start": START,
        "results": results,
        "accepted": bool(accepted),
        "seconds": time.perf_counter() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"[unit] overall: {'PASS' if accepted else 'FAIL'}", flush=True)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
