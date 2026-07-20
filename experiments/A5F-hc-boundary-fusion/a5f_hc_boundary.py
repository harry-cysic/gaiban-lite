"""A5F: C2g tilelang HC boundary fusion at Flash decode shapes.

The C1F integrated bench shows HC ~700 us/layer at B=512 (fp32 sinkhorn +
eager op chain) — the largest single non-MoE item. This microbench measures
the vLLM TileLang fused boundary (hc_post + hc_pre + RMSNorm in one kernel,
gaiban C2g path) against the reference eager chain at decode shapes (s=1),
plus numerical equivalence.

Boundary op = attn-side hc_post(attn_out, residual) -> ffn-side hc_pre ->
ffn_norm. Run on one GPU:
  <venv>/bin/python a5f_hc_boundary.py
Needs model.py/kernel.py/config.json importable (same dir or
../../reference/inference).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (HERE, os.path.join(HERE, "..", "..", "reference", "inference")):
    if os.path.exists(os.path.join(_c, "model.py")):
        sys.path.insert(0, _c)
        CFG_DIR = _c
        break
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + os.pathsep + os.environ.get("PATH", "")

import torch
import model as M

with open(os.path.join(CFG_DIR, "config.json")) as f:
    CFG = json.load(f)
DIM = CFG["dim"]


def build_block():
    args = M.ModelArgs(**{**CFG, **dict(max_batch_size=8, max_seq_len=1024,
                                        n_routed_experts=8, n_hash_layers=0)})
    with torch.device("cuda"):
        blk = M.Block(3, args)
    g = torch.Generator("cuda"); g.manual_seed(7)
    for nm in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
               "hc_attn_scale", "hc_ffn_scale"):
        getattr(blk, nm).data.normal_(0, 0.02, generator=g)
    blk.ffn_norm.weight.data.fill_(1.0)
    return blk


def boundary_ref(blk, attn_out, residual, post_prev, comb_prev):
    r = M.Block.hc_post(blk, attn_out, residual, post_prev, comb_prev)
    h, post, comb = M.Block.hc_pre(blk, r, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
    h = blk.ffn_norm(h)
    return r, post, comb, h


def boundary_fused(blk, attn_out, residual, post_prev, comb_prev):
    from vllm.model_executor.kernels.mhc.tilelang import mhc_fused_post_pre_tilelang
    residual_cur, post_cur, comb_cur, layer_input = mhc_fused_post_pre_tilelang(
        attn_out, residual, post_prev, comb_prev,
        blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base,
        blk.norm_eps, blk.hc_eps, blk.hc_eps, 2.0, blk.hc_sinkhorn_iters,
        n_splits=1, tile_n=1,
        # C2 finding: with_norm branch not numerically equivalent for >=128
        # tokens on sm89 — keep reference RMSNorm as separate kernel
        norm_weight=None, norm_eps=blk.ffn_norm.eps)
    layer_input = blk.ffn_norm(layer_input)
    return residual_cur, post_cur.squeeze(-1), comb_cur, layer_input


def stats(ref, got):
    e = (ref.float() - got.float()).abs()
    return (f"max={e.max().item():.2e}",
            f"allclose(2e-3)={torch.allclose(ref.float(), got.float(), rtol=2e-3, atol=2e-3)}")


def bench_graph(fn, iters=50, warmup=10):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3


def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    blk = build_block()

    for B in (128, 256, 512):
        g = torch.Generator("cuda"); g.manual_seed(100 + B)
        residual = torch.randn(B, 1, blk.hc_mult, DIM, dtype=torch.bfloat16, generator=g)
        attn_out = torch.randn(B, 1, DIM, dtype=torch.bfloat16, generator=g)
        _, post_prev, comb_prev = M.Block.hc_pre(
            blk, residual, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)

        r0, p0, c0, h0 = boundary_ref(blk, attn_out, residual, post_prev, comb_prev)
        r1, p1, c1, h1 = boundary_fused(blk, attn_out, residual, post_prev, comb_prev)
        print(f"\n== B={B} (s=1) numerical ==")
        print("  residual:", *stats(r0, r1))
        print("  post    :", *stats(p0, p1))
        print("  comb    :", *stats(c0, c1))
        print("  h_norm  :", *stats(h0, h1))

        t_ref = bench_graph(lambda: boundary_ref(blk, attn_out, residual, post_prev, comb_prev))
        t_fus = bench_graph(lambda: boundary_fused(blk, attn_out, residual, post_prev, comb_prev))
        print(f"  graph us: ref={t_ref:.1f}  fused={t_fus:.1f}  speedup={t_ref/t_fus:.2f}x")
    print("\nDONE")


if __name__ == "__main__":
    main()
