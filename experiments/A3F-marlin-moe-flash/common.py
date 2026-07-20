"""Shared helpers for A3' (grouped MoE Marlin MXFP4 benchmark on sm_89).

Conventions match the DSV4 checkpoint / reference implementation:
- expert weight: packed e2m1 nibbles along K (low nibble = even K index),
  per-32 e8m0 scales -> exactly what `make_ckpt_weight` generates and what
  A1.5 validated byte-level against vllm's prepare chain.
- w13 = cat([w1, w3]) along N: first half gate, second half up. This matches
  vllm `swiglu_limit_func` ("first half is gate") and V4 `Expert.forward`.
- gate: sqrtsoftplus + bias-shifted topk + renorm + route_scale (model.py Gate).
- shared expert: FP8 per-128 (default_dtype), tilelang fp8_gemm path, incl. the
  reference's double act_quant of x (w1(x) and w3(x) each quantize).

vllm imports are kept inside functions so vllm-free scripts (a3x, ref-loop)
can import this module too.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (HERE, os.path.join(HERE, "..", "..", "references", "inference")):
    if os.path.exists(os.path.join(_c, "kernel.py")):
        sys.path.insert(0, _c); break
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + os.pathsep + os.environ.get("PATH", "")

from types import SimpleNamespace

import torch
import torch.nn.functional as F

DEV = "cuda"
FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu

# V4 MoE dims (config.json)
# V4-Flash MoE dims (reference/config.json); env-overridable so the same
# bench can measure the TP4-local shard (A3F_MOE_INTER=512).
DIM = int(os.environ.get("A3F_DIM", "4096"))
MOE_INTER = int(os.environ.get("A3F_MOE_INTER", "2048"))
N_EXPERTS = int(os.environ.get("A3F_N_EXPERTS", "256"))
TOPK = int(os.environ.get("A3F_TOPK", "6"))
ROUTE_SCALE, SWIGLU_LIMIT = 1.5, 10.0

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def setup(seed=0):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.bfloat16)
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    return DEV


def make_ckpt_weight(N, K, gen):
    """Random weight in checkpoint format: packed nibbles + per-32 e8m0 scales.
    Scale exponents 122..129 (2^-5..2^2) keep dequant values sane."""
    packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=DEV, generator=gen)
    sexp = torch.randint(122, 130, (N, K // 32), dtype=torch.uint8, device=DEV, generator=gen)
    return packed, sexp


def dequant_fp4(packed, sexp):
    """fp32 [N, K] oracle dequant of ckpt-format fp4."""
    tab = FP4_TABLE.to(packed.device)
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    v = torch.stack([tab[lo], tab[hi]], dim=-1).flatten(1)
    scale = sexp.view(E8M0).float().repeat_interleave(32, dim=1)
    return v * scale


# --------------------------------------------------------------- marlin prep
def _prep_one(packed, sexp, size_n, size_k, is8):
    """Mirror one-expert body of vllm prepare_moe_mxfp4_layer_for_marlin."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import mxfp4_marlin_process_scales
    perm = torch.empty(0, dtype=torch.int, device=packed.device)
    qw = packed.view(torch.int32).T.contiguous()
    mq = ops.gptq_marlin_repack(b_q_weight=qw, perm=perm, size_k=size_k,
                                size_n=size_n, num_bits=4, is_a_8bit=is8)
    ws = sexp.view(E8M0).to(torch.bfloat16).T.contiguous()  # (K//32, N)
    ws = marlin_permute_scales(s=ws, size_k=size_k, size_n=size_n,
                               group_size=32, is_a_8bit=is8)
    ws = mxfp4_marlin_process_scales(ws, input_dtype=FP8 if is8 else None)
    return mq, ws


def prep_marlin_moe_experts(E, n_inter, k_hidden, input_dtype=None, seed=0, keep_ckpt=False):
    """Generate E experts (ckpt format) and repack to marlin moe layout,
    expert-by-expert to bound peak memory (vllm's own prepare would cat-copy
    the full 8.5 GB w13 and OOM a 24 GB card at E=384).

    Returns ns with stacked marlin tensors:
      w13_q (E, k/16, 4n), w13_s, w2_q (E, n/16, 2k), w2_s
    and, if keep_ckpt, per-expert ckpt tensors for oracle/reference use.
    """
    gen = torch.Generator(DEV); gen.manual_seed(seed)
    is8 = input_dtype is not None
    out = SimpleNamespace(ckpt=[] if keep_ckpt else None, input_dtype=input_dtype)
    for e in range(E):
        p13, s13 = make_ckpt_weight(2 * n_inter, k_hidden, gen)
        p2, s2 = make_ckpt_weight(k_hidden, n_inter, gen)
        mq13, ms13 = _prep_one(p13, s13, 2 * n_inter, k_hidden, is8)
        mq2, ms2 = _prep_one(p2, s2, k_hidden, n_inter, is8)
        if e == 0:
            out.w13_q = torch.empty((E,) + mq13.shape, dtype=mq13.dtype, device=DEV)
            out.w13_s = torch.empty((E,) + ms13.shape, dtype=ms13.dtype, device=DEV)
            out.w2_q = torch.empty((E,) + mq2.shape, dtype=mq2.dtype, device=DEV)
            out.w2_s = torch.empty((E,) + ms2.shape, dtype=ms2.dtype, device=DEV)
        out.w13_q[e], out.w13_s[e] = mq13, ms13
        out.w2_q[e], out.w2_s[e] = mq2, ms2
        if keep_ckpt:
            out.ckpt.append((p13, s13, p2, s2))
    wbytes = sum(t.numel() * t.element_size() for t in (out.w13_q, out.w13_s, out.w2_q, out.w2_s))
    out.bytes_per_expert = wbytes // E
    return out


def preset_w4a8_quant():
    """marlin_quant_input -> QuantFP8 needs a vllm config context (A1.5 lesson);
    preset the module singleton with the same single-kernel per-token quant."""
    from vllm import _custom_ops as ops
    import vllm.model_executor.layers.quantization.utils.marlin_utils as mu

    def _per_token_fp8(x):
        try:
            return ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
        except TypeError:  # signature drift fallback (eager, slower: note in results)
            xf = x.float()
            s = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
            return (xf / s).clamp(-448, 448).to(FP8), s

    mu._quant_fp8_method = _per_token_fp8
    return _per_token_fp8


# --------------------------------------------------------------------- gate
def make_gate_params(E, gen):
    gw = torch.randn(E, DIM, device=DEV, dtype=torch.bfloat16, generator=gen) * 0.02
    gb = torch.randn(E, device=DEV, dtype=torch.float32, generator=gen) * 0.01
    return gw, gb


def gate_forward(x2d, gw, gb, topk=TOPK, route_scale=ROUTE_SCALE, indices_override=None):
    """Replicates model.py Gate.forward (score_func='sqrtsoftplus', non-hash)."""
    scores = F.linear(x2d.float(), gw.float())
    scores = F.softplus(scores).sqrt()
    original = scores
    if indices_override is None:
        idx = (scores + gb).topk(topk, dim=-1)[1]
    else:
        idx = indices_override
    w = original.gather(1, idx)
    w = w / w.sum(dim=-1, keepdim=True)
    w = w * route_scale
    return w.float(), idx.to(torch.int32)


# ------------------------------------------------------------ shared expert
def quant_fp8_weight(w):
    """Per-128x128 block fp8 quant with ue8m0 scale (matches convert.py / Linear fp8)."""
    n, k = w.shape
    wb = w.float().reshape(n // 128, 128, k // 128, 128)
    amax = wb.abs().amax(dim=(1, 3)).clamp_min(1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    wq = (wb / scale[:, None, :, None]).clamp(-448, 448).to(FP8)
    return wq.reshape(n, k).contiguous(), scale.to(E8M0).contiguous()


class SharedExpertFP8:
    """model.py Expert.forward with FP8 per-128 weights on the tilelang kernels,
    op-for-op: w1(x), w3(x) each act_quant x; fp32 upcast; clamped swiglu;
    cast to bf16 before w2."""

    def __init__(self, gen, dim=DIM, inter=MOE_INTER, limit=SWIGLU_LIMIT):
        mk = lambda n, k: quant_fp8_weight(
            torch.randn(n, k, device=DEV, dtype=torch.float32, generator=gen) * 0.05)
        self.w1 = mk(inter, dim)
        self.w3 = mk(inter, dim)
        self.w2 = mk(dim, inter)
        self.limit = limit

    def __call__(self, x):
        from kernel import act_quant, fp8_gemm
        a, s = act_quant(x, 128, "ue8m0", E8M0)
        gate = fp8_gemm(a, s, *self.w1, E8M0).float()
        a, s = act_quant(x, 128, "ue8m0", E8M0)        # reference double-quants x
        up = fp8_gemm(a, s, *self.w3, E8M0).float()
        up = torch.clamp(up, min=-self.limit, max=self.limit)
        gate = torch.clamp(gate, max=self.limit)
        h = (F.silu(gate) * up).to(torch.bfloat16)
        a, s = act_quant(h, 128, "ue8m0", E8M0)
        return fp8_gemm(a, s, *self.w2, E8M0)


# ------------------------------------------------------- routing index pools
def make_idx_pool(dist, npool, B, E, topk, gen, zipf_alpha=1.1):
    """(npool, B, topk) int64 expert indices, sampled without replacement."""
    if dist == "uniform":
        logits = torch.rand(npool * B, E, device=DEV, generator=gen)
    elif dist == "zipf":
        p = torch.arange(1, E + 1, device=DEV, dtype=torch.float32) ** (-zipf_alpha)
        gumbel = -torch.log(-torch.log(
            torch.rand(npool * B, E, device=DEV, generator=gen).clamp_min(1e-20)))
        logits = p.log() + gumbel  # Gumbel-top-k = Plackett-Luce w/o replacement
    elif dist == "degen":
        return torch.arange(topk, device=DEV).expand(npool, B, topk).contiguous()
    else:
        raise ValueError(dist)
    return logits.topk(topk, dim=-1)[1].view(npool, B, topk)


def distinct_stats(idx_pool):
    counts = [idx_pool[i].unique().numel() for i in range(idx_pool.size(0))]
    return sum(counts) / len(counts)


# ------------------------------------------------------------------- timing
def bench(fn, iters=50, warmup=10, repeats=3):
    """fn(i) -> per-iteration callable (i for pool cycling). Returns best-mean us."""
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()
    best = float("inf")
    it = warmup
    for _ in range(repeats):
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            fn(it); it += 1
        e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters)
    return best * 1e3  # us


def host_async_ratio(fn, t_gpu_us, n=20):
    """Host wall time for n un-synced launches / GPU time. << 1 => no hidden sync."""
    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        fn(10_000 + i)
    t_host = (time.perf_counter() - t0) * 1e6 / n
    torch.cuda.synchronize()
    return t_host / t_gpu_us
