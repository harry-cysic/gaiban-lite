"""A3' main benchmark: full MoE-layer decode step on grouped Marlin MXFP4.

Timed unit per iteration (op-for-op the engine decode path):
  gate (fp32 linear + sqrtsoftplus + bias topk + gather/renorm/route_scale)
  -> fused_marlin_moe (384 experts, clamp_limit=10 swiglu, router weight fused
     in gemm2, topk-sum)
  -> shared expert FP8 (tilelang fp8_gemm, reference op sequence)
  -> fp32 add + bf16 cast

L2 hygiene: topk_ids re-randomized every iteration from a pool (the full
weight set 13.5 GB >> 72 MB L2, but at low B a fixed routing would re-read the
same few experts from L2; the pool also matches real decode where routing
changes every step).

W4A16 and W4A8 need differently-prepared weights (~13.5 GB each) -> separate
process runs:  --mode w4a16 | w4a8

Run: <venv>/bin/python a3_moe_bench.py --mode w4a16 [--E 384] [--smoke]
"""
import argparse, csv, os

import torch
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["w4a16", "w4a8"], default="w4a16")
    ap.add_argument("--E", type=int, default=C.N_EXPERTS)
    ap.add_argument("--B", default="1,4,8,16,32,64,96,128,192,256,384,512")
    ap.add_argument("--dists", default="gate,uniform,zipf,degen")
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clamp", type=float, default=C.SWIGLU_LIMIT)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny sweep for smoke test")
    args = ap.parse_args()
    if args.smoke:
        args.B, args.dists, args.pool = "8,64", "uniform", 8

    C.setup(args.seed)
    E, topk = args.E, C.TOPK
    Bs = [int(b) for b in args.B.split(",")]
    dists = args.dists.split(",")
    csv_path = args.csv or f"a3_results_{args.mode}_E{E}.csv"

    from vllm import __version__ as vllm_ver
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.scalar_type import scalar_types
    print("vllm", vllm_ver, "| mode", args.mode, "| E", E)

    input_dtype = C.FP8 if args.mode == "w4a8" else None
    if input_dtype is not None:
        C.preset_w4a8_quant()

    print("preparing %d experts (chunked repack)..." % E)
    W = C.prep_marlin_moe_experts(E, C.MOE_INTER, C.DIM, input_dtype, args.seed)
    print("  marlin weights resident: %.2f GiB (%.1f MB/expert)"
          % (W.bytes_per_expert * E / 2**30, W.bytes_per_expert / 2**20))
    gen = torch.Generator(C.DEV); gen.manual_seed(args.seed + 1)
    gw, gb = C.make_gate_params(E, gen)
    shared = C.SharedExpertFP8(gen)
    ws = marlin_make_workspace_new(torch.device(C.DEV), 4)
    qid = scalar_types.float4_e2m1f.id

    fields = ["mode", "dist", "B", "t_us", "distinct", "wbytes_MB", "eff_GBs",
              "tok_s_stage", "tokps_gpu_ub63", "host_async", "iters"]
    rows = []
    print("  %-7s %5s | %9s %8s %9s %9s %11s %7s" %
          ("dist", "B", "t_us", "distinct", "wread_MB", "eff_GBs", "tok/s_stage", "async"))

    for B in Bs:
        nx = 32 if B <= 128 else 8
        x_pool = [torch.randn(B, C.DIM, device=C.DEV, dtype=torch.bfloat16, generator=gen) * 0.1
                  for _ in range(nx)]
        ic13 = torch.empty(B * topk * max(2 * C.MOE_INTER, C.DIM),
                           device=C.DEV, dtype=torch.bfloat16)
        ic2 = torch.empty(B * topk * C.MOE_INTER, device=C.DEV, dtype=torch.bfloat16)
        out = torch.empty(B, C.DIM, device=C.DEV, dtype=torch.bfloat16)

        for dist in dists:
            if dist == "gate":
                idx_pool = None
            else:
                idx_pool = C.make_idx_pool(dist, args.pool, B, E, topk, gen)

            def step(i):
                x = x_pool[i % nx]
                ov = None if idx_pool is None else idx_pool[i % args.pool]
                w, idx = C.gate_forward(x, gw, gb, indices_override=ov)
                routed = fused_marlin_moe(
                    x, W.w13_q, W.w2_q, None, None, W.w13_s, W.w2_s,
                    topk_weights=w, topk_ids=idx, quant_type_id=qid,
                    activation=MoEActivation.SILU, workspace=ws,
                    intermediate_cache13=ic13, intermediate_cache2=ic2,
                    output=out, input_dtype=input_dtype,
                    clamp_limit=args.clamp if args.clamp > 0 else None)
                sh = shared(x)
                return (routed.float() + sh.float()).to(torch.bfloat16)

            iters = 100 if B <= 64 else (50 if B <= 256 else 30)
            t = C.bench(step, iters=iters, warmup=15)
            har = C.host_async_ratio(step, t)

            if idx_pool is None:  # measure gate's own routing spread
                d = C.distinct_stats(torch.stack(
                    [C.gate_forward(x_pool[i % nx], gw, gb)[1].long() for i in range(16)]))
            else:
                d = C.distinct_stats(idx_pool)
            wread = d * W.bytes_per_expert
            gbs = wread / (t * 1e-6) / 1e9
            tok_s = B / (t * 1e-6)
            rows.append(dict(zip(fields, [args.mode, dist, B, round(t, 1), round(d, 1),
                                          round(wread / 2**20, 1), round(gbs, 1),
                                          round(tok_s, 1), round(tok_s / 63, 1),
                                          round(har, 3), iters])))
            print("  %-7s %5d | %9.1f %8.1f %9.1f %9.1f %11.1f %7.3f" %
                  (dist, B, t, d, wread / 2**20, gbs, tok_s, har))
        del x_pool, ic13, ic2, out

    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader(); wtr.writerows(rows)
    print("wrote", os.path.abspath(csv_path))


if __name__ == "__main__":
    main()
