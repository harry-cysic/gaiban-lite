"""A3' numerics anchor: marlin MoE layer vs reference model.MoE vs fp32 oracle.

Same weights (ckpt format), same gate outputs, E=32 so the oracle is cheap.
Known-legal differences (documented in README):
  - reference multiplies router weight on the w2 *input* (model.py:604), vllm
    on the w2 *output* (fused in gemm2) -- mathematically equivalent;
  - reference routed path act-quants per-128 fp8 (act_quant) + tilelang
    fp4_gemm; marlin W4A16 keeps acts bf16 (so it should be *closer* to
    oracle than the reference is); W4A8 quants per-token fp8.
Pass criteria: rel_fro(marlin16, oracle) <= rel_fro(reference, oracle), and
W4A8 at the A1.5-measured ~3e-2 level.

Also runs a clamp-ACTIVE config (x scaled up) to confirm swiglu_limit_func
clamping matches the reference under saturation.

Run: <venv>/bin/python a3_numerics.py
"""
import torch
import torch.nn.functional as F
import common as C

E_SMALL = 32


def build_reference_moe(ckpt, gw, gb, shared_w, args_extra=None):
    """model.MoE with production quant globals, params filled from ckpt tensors."""
    import model as M
    M.default_dtype = C.FP8          # shared expert FP8 (Transformer.__init__ behavior)
    M.scale_fmt = "ue8m0"
    M.scale_dtype = C.E8M0
    args = M.ModelArgs(
        dim=C.DIM, moe_inter_dim=C.MOE_INTER, n_routed_experts=E_SMALL,
        n_shared_experts=1, n_activated_experts=C.TOPK,
        score_func="sqrtsoftplus", route_scale=C.ROUTE_SCALE,
        swiglu_limit=C.SWIGLU_LIMIT, expert_dtype="fp4", n_hash_layers=0)
    with torch.device(C.DEV):
        moe = M.MoE(3, args)
    moe.gate.weight.data.copy_(gw)
    moe.gate.bias.data.copy_(gb)
    for e, (p13, s13, p2, s2) in enumerate(ckpt):
        ex = moe.experts[e]
        n = C.MOE_INTER
        ex.w1.weight.data.copy_(p13[:n].view(torch.float4_e2m1fn_x2))
        ex.w1.scale.data.copy_(s13[:n].view(C.E8M0))
        ex.w3.weight.data.copy_(p13[n:].view(torch.float4_e2m1fn_x2))
        ex.w3.scale.data.copy_(s13[n:].view(C.E8M0))
        ex.w2.weight.data.copy_(p2.view(torch.float4_e2m1fn_x2))
        ex.w2.scale.data.copy_(s2.view(C.E8M0))
    for name, (wq, ws) in shared_w.items():
        lin = getattr(moe.shared_experts, name)
        lin.weight.data.copy_(wq)
        lin.scale.data.copy_(ws)
    return moe


def oracle_moe(x2d, ckpt, weights, indices, shared_w, limit=C.SWIGLU_LIMIT):
    """fp32 dequant oracle of the full MoE layer, expert-at-a-time."""
    xf = x2d.float()
    y = torch.zeros_like(xf)
    stats = {"max_gate": 0.0, "max_up": 0.0}

    def ffn(xe, w1d, w3d, w2d, wt=None):
        g = xe @ w1d.T
        u = xe @ w3d.T
        stats["max_gate"] = max(stats["max_gate"], g.abs().max().item())
        stats["max_up"] = max(stats["max_up"], u.abs().max().item())
        u = torch.clamp(u, min=-limit, max=limit)
        g = torch.clamp(g, max=limit)
        h = F.silu(g) * u
        if wt is not None:
            h = h * wt
        return h @ w2d.T

    n = C.MOE_INTER
    for e in range(len(ckpt)):
        rows, kth = torch.where(indices == e)
        if rows.numel() == 0:
            continue
        p13, s13, p2, s2 = ckpt[e]
        w1d = C.dequant_fp4(p13[:n], s13[:n])
        w3d = C.dequant_fp4(p13[n:], s13[n:])
        w2d = C.dequant_fp4(p2, s2)
        y[rows] += ffn(xf[rows], w1d, w3d, w2d, weights[rows, kth, None].float())
        del w1d, w3d, w2d

    def dq8(wq, ws):
        n_, k_ = wq.shape
        s = ws.float().repeat_interleave(128, 0)[:n_].repeat_interleave(128, 1)[:, :k_]
        return wq.float() * s
    y += ffn(xf, dq8(*shared_w["w1"]), dq8(*shared_w["w3"]), dq8(*shared_w["w2"]))
    return y, stats


def rel(a, b):
    d = a.float() - b.float()
    return (d.norm() / b.float().norm()).item()


def main():
    C.setup(0)
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from vllm.scalar_type import scalar_types
    C.preset_w4a8_quant()
    ws = marlin_make_workspace_new(torch.device(C.DEV), 4)
    qid = scalar_types.float4_e2m1f.id

    gen = torch.Generator(C.DEV); gen.manual_seed(7)
    W16 = C.prep_marlin_moe_experts(E_SMALL, C.MOE_INTER, C.DIM, None, seed=7, keep_ckpt=True)
    W8 = C.prep_marlin_moe_experts(E_SMALL, C.MOE_INTER, C.DIM, C.FP8, seed=7)
    gw, gb = C.make_gate_params(E_SMALL, gen)
    shared_w = {n: C.quant_fp8_weight(
        torch.randn(*s, device=C.DEV, dtype=torch.float32, generator=gen) * 0.05)
        for n, s in {"w1": (C.MOE_INTER, C.DIM), "w3": (C.MOE_INTER, C.DIM),
                     "w2": (C.DIM, C.MOE_INTER)}.items()}
    moe_ref = build_reference_moe(W16.ckpt, gw, gb, shared_w)

    def marlin_layer(x2d, W, input_dtype, weights, idx32):
        routed = fused_marlin_moe(
            x2d, W.w13_q, W.w2_q, None, None, W.w13_s, W.w2_s,
            topk_weights=weights, topk_ids=idx32, quant_type_id=qid,
            activation=MoEActivation.SILU, workspace=ws,
            input_dtype=input_dtype, clamp_limit=C.SWIGLU_LIMIT)
        sh = moe_ref.shared_experts(x2d)   # same module both paths
        return (routed.float() + sh.float()).to(torch.bfloat16)

    # random-ckpt routed weights have rms ~4.8 -> |gate| std ~ 405*xscale;
    # 0.004 keeps max|gate|,|up| < 10 (clamp provably inactive), 0.2 saturates.
    for tag, xscale in (("clamp-inactive", 0.004), ("clamp-ACTIVE", 0.2)):
        print(f"\n--- {tag} (x*{xscale}) ---")
        for B in (16, 64):
            x = torch.randn(B, C.DIM, device=C.DEV, dtype=torch.bfloat16, generator=gen) * xscale
            weights, idx32 = C.gate_forward(x, gw, gb)
            o_ref = moe_ref(x.view(1, B, C.DIM),
                            torch.zeros(1, B, dtype=torch.long, device=C.DEV)).view(B, C.DIM)
            o_m16 = marlin_layer(x, W16, None, weights, idx32)
            o_m8 = marlin_layer(x, W8, C.FP8, weights, idx32)
            o_orc, st = oracle_moe(x, W16.ckpt, weights, idx32.long(), shared_w)
            print("  B=%-3d max|gate|=%.2f max|up|=%.2f" % (B, st["max_gate"], st["max_up"]))
            print("    rel_fro vs oracle:  ref=%.3e  marlin16=%.3e  marlin8=%.3e"
                  % (rel(o_ref, o_orc), rel(o_m16, o_orc), rel(o_m8, o_orc)))
            print("    marlin16 vs ref: %.3e" % rel(o_m16, o_ref))


if __name__ == "__main__":
    main()
