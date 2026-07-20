"""A4F: V4-Flash per-layer non-MoE time (attention + indexer + compressor + HC).

Flash regear of gaiban A4 a4_attn_timing.py. Differences vs Pro:
  - geometry from reference config.json (dim 4096, 64 heads, q_lora 1024,
    o_groups 8, index_topk 512, window 128);
  - three layer types: layer 0 (compress_ratio 0, pure sliding window — new in
    Flash), layer 2 (ratio 4, indexer), layer 3 (ratio 128, no indexer);
  - default --world 4 (TP4 -> n_local_heads = 64/4 = 16, fits sm89 smem
    natively per gaiban A4 findings).

Method unchanged: real model.py Block with a TINY never-called MoE (8 experts);
context via start_pos; time attention half + HC around FFN, eager and CUDA-graph.

Run: <venv>/bin/python a4f_attn_timing.py [--smoke] [--world 4]
     (model.py/kernel.py/config.json must be importable: same dir or
      ../../reference/inference)
"""
import os, sys, argparse, json
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

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu

with open(os.path.join(CFG_DIR, "config.json")) as _f:
    CFG = json.load(_f)
DIM = CFG["dim"]
N_HEADS = CFG["n_heads"]


def quant_fp8_weight(w):
    n, k = w.shape
    wb = w.float().reshape(n // 128, 128, k // 128, 128)
    amax = wb.abs().amax(dim=(1, 3)).clamp_min(1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    wq = (wb / scale[:, None, :, None]).clamp(-448, 448).to(FP8)
    return wq.reshape(n, k).contiguous(), scale.to(E8M0).contiguous()


def fill(block, gen):
    """Random-fill attention + HC params (timing only). MoE experts untouched."""
    done = set()
    for m in block.attn.modules():
        if isinstance(m, M.Linear) and m.weight is not None:
            if m.weight.dtype == FP8:
                wq, ws = quant_fp8_weight(torch.randn(m.weight.shape[0], m.weight.shape[1],
                                                      device="cuda", generator=gen) * 0.02)
                m.weight.data.copy_(wq); m.scale.data.copy_(ws)
                done.add(id(m.weight)); done.add(id(m.scale))
            elif m.weight.dtype in (torch.bfloat16, torch.float32):
                m.weight.data.normal_(0, 0.02, generator=gen)
                done.add(id(m.weight))
        elif isinstance(m, M.RMSNorm):
            m.weight.data.fill_(1.0)
            done.add(id(m.weight))
    # remaining float params (ape, attn_sink, etc.) — torch.empty garbage can be
    # inf/NaN; give them finite values
    for p in block.attn.parameters():
        if id(p) not in done and p.dtype in (torch.bfloat16, torch.float32):
            p.data.normal_(0, 0.5, generator=gen)
    for nm in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
               "hc_attn_scale", "hc_ffn_scale"):
        getattr(block, nm).data.normal_(0, 0.02, generator=gen)


def build_block(layer_id, max_seq, max_b, gen):
    overrides = dict(max_batch_size=max_b, max_seq_len=max_seq,
                     n_routed_experts=8, n_hash_layers=0)
    args = M.ModelArgs(**{**CFG, **overrides})
    with torch.device("cuda"):
        blk = M.Block(layer_id, args)
    fill(blk, gen)
    return blk


def attn_half(blk, x, start_pos):
    r = x
    h, post, comb = blk.hc_pre(x, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
    h = blk.attn_norm(h)
    h = blk.attn(h, start_pos)
    return blk.hc_post(h, r, post, comb)


def ffn_hc(blk, x):  # HC+norm overhead around the MoE (MoE itself = A3F, TP-scaled)
    r = x
    h, post, comb = blk.hc_pre(x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
    h = blk.ffn_norm(h)
    return blk.hc_post(h, r, post, comb)


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3  # us


def bench_graph(fn, iters=50, warmup=10):
    try:
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
    except Exception as e:
        print(f"    [graph capture failed: {type(e).__name__}: {str(e)[:80]}]")
        return float("nan")


def configure_rotate(mode):
    if mode == "stub":
        M.rotate_activation = lambda x: x * (x.size(-1) ** -0.5)
        return "stub"

    from fast_hadamard_transform import hadamard_transform

    def rotate_activation(x):
        assert x.dtype == torch.bfloat16
        return hadamard_transform(x, scale=x.size(-1) ** -0.5)

    M.rotate_activation = rotate_activation
    return "fht"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--world", type=int, default=4,
                    help="attention TP width: n_local_heads = 64/world. "
                         "TP4 -> h=16 fits sm89 smem natively.")
    ap.add_argument("--rotate", choices=["fht", "stub"], default="fht")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    M.default_dtype = FP8; M.scale_fmt = CFG["scale_fmt"]; M.scale_dtype = E8M0
    M.world_size = args.world
    M.dist.all_reduce = lambda *a, **k: None
    rotate_label = configure_rotate(args.rotate)
    print(f"world_size={args.world} -> n_local_heads={N_HEADS // args.world} (rotate={rotate_label})")
    gen = torch.Generator("cuda"); gen.manual_seed(0)
    print("GPU:", torch.cuda.get_device_name(0))

    CTXS = [4096, 16384] if args.smoke else [4096, 8192, 16384, 65536]
    BS = [16, 64] if args.smoke else [1, 16, 64, 128, 256, 512]
    MAXSEQ = max(CTXS) + 256
    MAXB = max(BS)

    # layer 0: compress_ratio 0 (pure sliding window); layer 2: ratio 4 (indexer);
    # layer 3: ratio 128 (compressor, no indexer)
    for lid, ratio in ((0, 0), (2, 4), (3, 128)):
        blk = build_block(lid, MAXSEQ, MAXB, gen)
        print(f"\n=== layer_id={lid} (compress_ratio={ratio}) ===")
        print(f"  eager = un-graphed (launch-bound); graph = CUDA-graph replay (compute)")
        print(f"  {'ctx':>6} {'B':>4} | {'attn_eager':>10} {'attn_graph':>10} | {'hc_graph':>9} | {'fixed_graph':>11}")
        for ctx in CTXS:
            for B in BS:
                x = torch.randn(B, 1, blk.hc_mult, DIM, device="cuda", dtype=torch.bfloat16, generator=gen)
                sp = ctx
                a_e = bench(lambda: attn_half(blk, x, sp))
                a_g = bench_graph(lambda: attn_half(blk, x, sp))
                h_g = bench_graph(lambda: ffn_hc(blk, x))
                fixed_g = a_g + h_g
                print(f"  {ctx:>6} {B:>4} | {a_e:>10.1f} {a_g:>10.1f} | {h_g:>9.1f} | {fixed_g:>11.1f}")
        del blk
        torch.cuda.empty_cache()
    print("\nDONE")


if __name__ == "__main__":
    main()
