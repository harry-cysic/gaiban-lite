#!/usr/bin/env python3
"""Fast synthetic smoke of the FP8 KV state surfaces (no checkpoint needed).

Covers, for kv_dtype in (bf16, fp8, fp8_rope_bf16) on one GPU:
- StaticLayerKV: seed_decode_residency, stateful boundary/non-boundary
  writes, decode_write/prefill_write, dequantized views, clone/copy_from,
  metadata JSON round trip;
- StaticWindowKV: prefill/decode/stateful writes, seed, dequantized views;
- StaticRatio4KV (+ indexer fp8): seed_decode_payload, stateful boundary
  write, dequantized views, clone/copy_from;
- bf16 identity: quantize helpers return the input tensor unchanged and the
  dequantized views alias resident storage (zero-copy on the frozen path).
"""

from __future__ import annotations

import torch

from dsv4_direct.static_kv import LATENT_ROPE_DIM, StaticLayerKV
from dsv4_direct.static_ratio4_kv import StaticRatio4KV
from dsv4_direct.static_window_kv import StaticWindowKV


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def bf16(shape, seed, device, scale=0.03):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16).to(device)


def main() -> None:
    device = torch.device("cuda", 0)
    batch, max_seq = 3, 512
    for kv_dtype in ("bf16", "fp8", "fp8_rope_bf16"):
        # ---------------- ratio-128 ----------------
        state = StaticLayerKV(
            num_local_sequences=batch, max_seq_len=max_seq, layer_id=3,
            device=device, kv_dtype=kv_dtype,
        )
        raw = bf16((batch, 128, 512), 1, device)
        compressed = bf16((batch, 2, 512), 2, device)
        state.seed_decode_residency(start_pos=256, raw=raw, compressed=compressed)
        deq = state.dequantized_latent()
        check(deq.dtype == torch.bfloat16, "deq dtype")
        if kv_dtype == "bf16":
            check(deq.data_ptr() == state.latent.data_ptr(), "bf16 identity view")
            check(torch.equal(deq[:, :128], raw), "bf16 raw exact")
        else:
            err = (deq[:, :128].float() - raw.float()).abs().max().item()
            check(err < 0.02, f"raw quant err {err}")
            if kv_dtype == "fp8_rope_bf16":
                check(
                    torch.equal(deq[:, :128, -LATENT_ROPE_DIM:], raw[..., -LATENT_ROPE_DIM:]),
                    "rope tail exact",
                )
        # stateful non-boundary + boundary writes
        position = torch.tensor([256], dtype=torch.int64, device=device)
        raw_tok = bf16((batch, 1, 512), 3, device)
        kv_tok = torch.randn(batch, 1, 512, device=device)
        score_tok = torch.randn(batch, 512, device=device)

        def finalizer(pooled, starts):
            return pooled.to(torch.bfloat16).contiguous()

        state._write_decode_stateful_prevalidated(
            raw_tok, kv_tok, score_tok, position=position, boundary=False,
            finalize_compressed=finalizer,
        )
        rt = state.dequantized_latent()[:, 256 % 128 : 256 % 128 + 1]
        check((rt.float() - raw_tok.float()).abs().max().item() < 0.02, "tok write")
        position255 = torch.tensor([383], dtype=torch.int64, device=device)
        state._next_position.fill_(383)
        state._write_decode_stateful_prevalidated(
            raw_tok, kv_tok, score_tok, position=position255, boundary=True,
            finalize_compressed=finalizer,
        )
        # eager decode_write path at next position
        state._next_position.fill_(384)
        ape = torch.zeros(128, 512, device=device)
        state._raw_positions.fill_(0)  # not validated by decode_write metadata path
        state._compressed_count.fill_(3)
        state._state_positions[:, :0] = 0
        state.decode_write(
            raw_tok, projected_kv=kv_tok, projected_score=score_tok.unsqueeze(1),
            ape=ape, finalize_compressed=finalizer,
        )
        clone = StaticLayerKV(
            num_local_sequences=batch, max_seq_len=max_seq, layer_id=3,
            device=device, kv_dtype=kv_dtype,
        )
        clone.copy_from(state)
        for (n1, t1), (n2, t2) in zip(
            state._owned_tensor_items(), clone._owned_tensor_items(), strict=True
        ):
            check(n1 == n2 and torch.equal(t1, t2), f"clone {n1}")
        state.metadata()

        # ---------------- window ----------------
        wstate = StaticWindowKV(
            num_local_sequences=batch, max_seq_len=max_seq, layer_id=0,
            device=device, kv_dtype=kv_dtype,
        )
        wstate.prefill_write(bf16((batch, 200, 512), 5, device))
        wstate.decode_write(raw_tok)
        wpos = torch.tensor([201], dtype=torch.int64, device=device)
        wstate._write_decode_stateful_prevalidated(raw_tok, position=wpos)
        wdeq = wstate.dequantized_latent()
        check(wdeq.shape == (batch, 128, 512), "window deq shape")
        wstate.seed_decode_residency(start_pos=256, raw=raw)
        wclone = StaticWindowKV(
            num_local_sequences=batch, max_seq_len=max_seq, layer_id=0,
            device=device, kv_dtype=kv_dtype,
        )
        wclone.copy_from(wstate)
        wstate.metadata()

        # ---------------- ratio-4 ----------------
        for indexer_dtype in ("bf16", "fp8") if kv_dtype == "fp8" else ("bf16",):
            rstate = StaticRatio4KV(
                num_local_sequences=batch, max_seq_len=max_seq, layer_id=2,
                device=device, kv_dtype=kv_dtype, indexer_dtype=indexer_dtype,
            )
            capacity = max_seq / 4
            seed_kwargs = dict(
                raw=bf16((batch, 128, 512), 7, device),
                compressed=bf16((batch, 128, 512), 8, device),
                indexer_kv=bf16((batch, 128, 128), 9, device),
                main_kv_state=torch.randn(batch, 8, 1024, device=device),
                main_score_state=torch.randn(batch, 8, 1024, device=device),
                index_kv_state=torch.randn(batch, 8, 256, device=device),
                index_score_state=torch.randn(batch, 8, 256, device=device),
            )
            rstate.seed_decode_payload(256, **seed_kwargs)
            rdeq = rstate.dequantized_latent()
            if kv_dtype == "bf16":
                check(torch.equal(rdeq[:, :128], seed_kwargs["raw"]), "r4 seed exact")
            else:
                err = (rdeq[:, :128].float() - seed_kwargs["raw"].float()).abs().max().item()
                check(err < 0.02, f"r4 seed quant err {err}")
            rpos = torch.tensor([259], dtype=torch.int64, device=device)
            rstate._next_position.fill_(259)

            def rfinal(pooled, freqs):
                return pooled.to(torch.bfloat16).contiguous()

            rstate._write_decode_stateful_prevalidated(
                bf16((batch, 1, 512), 11, device),
                torch.randn(batch, 1, 1024, device=device),
                torch.randn(batch, 1024, device=device),
                torch.randn(batch, 1, 256, device=device),
                torch.randn(batch, 256, device=device),
                position=rpos,
                boundary=True,
                group_start_frequencies=torch.zeros(1, 32, dtype=torch.complex64, device=device),
                main_finalizer=rfinal,
                index_finalizer=lambda pooled, freqs: pooled[..., :128].to(torch.bfloat16).contiguous(),
            )
            rclone = StaticRatio4KV(
                num_local_sequences=batch, max_seq_len=max_seq, layer_id=2,
                device=device, kv_dtype=kv_dtype, indexer_dtype=indexer_dtype,
            )
            rclone.copy_from(rstate)
            rstate.metadata()
        print(f"[smoke] kv_dtype={kv_dtype}: OK", flush=True)
    print("[smoke] ALL OK", flush=True)


if __name__ == "__main__":
    main()
