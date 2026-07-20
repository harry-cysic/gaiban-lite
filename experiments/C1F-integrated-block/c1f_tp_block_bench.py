"""C1F: V4-Flash integrated TP block decode benchmark (gaiban C1' regear).

Flash geometry from reference/config.json (dim 4096, 64 heads, 256 experts,
inter 2048, topk 512). Differences vs gaiban C1':
  - --moe-mode itp (default): per-expert intermediate-TP — every rank holds all
    256 experts sliced to inter/world rows; routed+shared partial sums are
    all_reduced together (the project's declared placement).
    --moe-mode ep: contiguous expert-parallel shard (gaiban C1' behavior).
  - sparse_attn base = reference kernel.py (block=64); h=16 per launch fits
    sm89 natively (A4F). DP-attention head-loops 64 heads as 4x16 sub-launches.
  - per-GPU model-level throughput divisor uses 44 layer-equivalents (43+MTP).

Run with torchrun on one titan node, e.g.:
  CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=SYS torchrun --standalone \
    --nnodes=1 --nproc_per_node=4 c1f_tp_block_bench.py --ctx 8192 \
    --B 128,256,512 --attn-mode dp [--breakdown]
Requires alongside (or on sys.path): common.py (A3F Flash version), model.py,
kernel.py, config.json (reference).
"""

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + os.pathsep + os.environ.get("PATH", "")

HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    HERE,
    os.path.join(HERE, "..", "A3F-marlin-moe-flash"),
    os.path.join(HERE, "..", "..", "reference", "inference"),
):
    if os.path.exists(os.path.join(candidate, "model.py")) or os.path.exists(os.path.join(candidate, "common.py")):
        sys.path.insert(0, candidate)

import torch
import torch.distributed as dist

import common as C
import model as M
from kernel import sparse_attn as sparse_attn_base

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu

_cfg_path = next(os.path.join(d, "config.json") for d in (HERE,
               os.path.join(HERE, "..", "..", "reference", "inference"))
               if os.path.exists(os.path.join(d, "config.json")))
with open(_cfg_path) as _f:
    CFG = json.load(_f)
DIM = CFG["dim"]
N_HEADS = CFG["n_heads"]
LAYER_EQUIV = 44  # 43 layers + MTP block


def quant_fp8_weight(w):
    n, k = w.shape
    wb = w.float().reshape(n // 128, 128, k // 128, 128)
    amax = wb.abs().amax(dim=(1, 3)).clamp_min(1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    wq = (wb / scale[:, None, :, None]).clamp(-448, 448).to(FP8)
    return wq.reshape(n, k).contiguous(), scale.to(E8M0).contiguous()


def fill_attention_and_hc(block, gen):
    done = set()
    for mod in block.attn.modules():
        if isinstance(mod, M.Linear) and mod.weight is not None:
            if mod.weight.dtype == FP8:
                wq, ws = quant_fp8_weight(torch.randn(
                    mod.weight.shape[0], mod.weight.shape[1],
                    device="cuda", generator=gen) * 0.02)
                mod.weight.data.copy_(wq)
                mod.scale.data.copy_(ws)
                done.add(id(mod.weight)); done.add(id(mod.scale))
            elif mod.weight.dtype in (torch.bfloat16, torch.float32):
                mod.weight.data.normal_(0, 0.02, generator=gen)
                done.add(id(mod.weight))
        elif isinstance(mod, M.RMSNorm):
            mod.weight.data.fill_(1.0)
            done.add(id(mod.weight))
    for p in block.attn.parameters():
        if id(p) not in done and p.dtype in (torch.bfloat16, torch.float32):
            p.data.normal_(0, 0.5, generator=gen)
    for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
                 "hc_attn_scale", "hc_ffn_scale"):
        getattr(block, name).data.normal_(0, 0.02, generator=gen)


def model_args(max_seq, max_b):
    overrides = dict(max_batch_size=max_b, max_seq_len=max_seq,
                     n_routed_experts=8, n_hash_layers=0)
    return M.ModelArgs(**{**CFG, **overrides})


class MarlinMoE(torch.nn.Module):
    """Marlin MXFP4 routed + FP8 shared MoE under two placements.

    itp: all E experts per rank, inter sliced to inter/world (row-parallel);
         routed+shared partial sums all_reduced together.
    ep : E/world contiguous experts per rank, full inter; routed all_reduced,
         shared replicated (gaiban C1' behavior).
    """

    def __init__(self, world, rank, mode="itp", gen_seed=0, allreduce_dtype="bf16", input_dtype=None):
        super().__init__()
        self.world = world
        self.rank = rank
        self.mode = mode
        self.e_global = C.N_EXPERTS
        self.allreduce_dtype = allreduce_dtype
        self.input_dtype = input_dtype

        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
        from vllm.scalar_type import scalar_types

        self.fused_marlin_moe = fused_marlin_moe
        self.moe_activation = MoEActivation.SILU
        self.qid = scalar_types.float4_e2m1f.id
        self.workspace = marlin_make_workspace_new(torch.device("cuda"), 4)
        if input_dtype is not None:
            C.preset_w4a8_quant()

        gen = torch.Generator("cuda")
        gen.manual_seed(gen_seed + 1000)
        self.gw, self.gb = C.make_gate_params(self.e_global, gen)

        if mode == "itp":
            assert C.MOE_INTER % world == 0
            self.e_local = self.e_global
            self.n_inter = C.MOE_INTER // world
            self.expert_map = None
            self.shared = C.SharedExpertFP8(gen, inter=self.n_inter)
        else:
            assert self.e_global % world == 0
            self.e_local = self.e_global // world
            self.n_inter = C.MOE_INTER
            start = rank * self.e_local
            expert_map = torch.full((self.e_global,), -1, device="cuda", dtype=torch.int32)
            expert_map[start:start + self.e_local] = torch.arange(self.e_local, device="cuda", dtype=torch.int32)
            self.expert_map = expert_map
            self.shared = C.SharedExpertFP8(gen)

        self.W = C.prep_marlin_moe_experts(self.e_local, self.n_inter, C.DIM, input_dtype, seed=gen_seed + rank)
        self.ic13 = None
        self.ic2 = None
        self.out = None

    def _ensure_workspace(self, rows):
        topk = C.TOPK
        need = rows * topk * max(2 * self.n_inter, C.DIM)
        if self.ic13 is not None and self.ic13.numel() >= need:
            return
        self.ic13 = torch.empty(need, device="cuda", dtype=torch.bfloat16)
        self.ic2 = torch.empty(rows * topk * self.n_inter, device="cuda", dtype=torch.bfloat16)
        self.out = torch.empty(rows, C.DIM, device="cuda", dtype=torch.bfloat16)

    def routed_local(self, x2d):
        rows = x2d.size(0)
        self._ensure_workspace(rows)
        weights, idx = C.gate_forward(x2d, self.gw, self.gb)
        return self.fused_marlin_moe(
            x2d, self.W.w13_q, self.W.w2_q, None, None, self.W.w13_s, self.W.w2_s,
            topk_weights=weights, topk_ids=idx, quant_type_id=self.qid,
            activation=self.moe_activation, workspace=self.workspace,
            intermediate_cache13=self.ic13, intermediate_cache2=self.ic2,
            output=self.out, input_dtype=self.input_dtype, clamp_limit=C.SWIGLU_LIMIT,
            global_num_experts=self.e_global, expert_map=self.expert_map)

    def shared_only(self, x2d):
        return self.shared(x2d)

    def local_no_ar(self, x):
        shape = x.size()
        x2d = x.view(-1, C.DIM).contiguous()
        routed = self.routed_local(x2d)
        shared = self.shared_only(x2d)
        return (routed.float() + shared.float()).to(torch.bfloat16).view(shape)

    def forward(self, x, input_ids=None):
        shape = x.size()
        x2d = x.view(-1, C.DIM).contiguous()
        if self.mode == "itp":
            # routed and shared are both partial over inter slices: one AR total
            out = (self.routed_local(x2d).float() + self.shared_only(x2d).float()).to(torch.bfloat16)
            if self.world > 1:
                if self.allreduce_dtype == "fp32":
                    out = out.float()
                    dist.all_reduce(out)
                    out = out.to(torch.bfloat16)
                else:
                    dist.all_reduce(out)
            return out.view(shape)
        routed = self.routed_local(x2d)
        if self.world > 1:
            if self.allreduce_dtype == "fp32":
                routed = routed.float()
                dist.all_reduce(routed)
                routed = routed.to(torch.bfloat16)
            else:
                dist.all_reduce(routed)
        shared = self.shared_only(x2d)
        return (routed.float() + shared.float()).to(torch.bfloat16).view(shape)


def bench(fn, iters=20, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return e0.elapsed_time(e1) / iters * 1e3


def bench_graph(fn, iters=30, warmup=8):
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        if dist.is_initialized():
            dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        e0 = torch.cuda.Event(True)
        e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            graph.replay()
        e1.record()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        return e0.elapsed_time(e1) / iters * 1e3
    except Exception as exc:
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"    graph_capture_failed={type(exc).__name__}: {str(exc)[:180]}")
        return float("nan")


def bench_graph_no_model_allreduce(fn, iters=30, warmup=8):
    original = M.dist.all_reduce
    M.dist.all_reduce = lambda *args, **kwargs: None
    try:
        return bench_graph(fn, iters=iters, warmup=warmup)
    finally:
        M.dist.all_reduce = original


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


def attn_half(block, x, start_pos):
    residual = x
    h, post, comb = block.hc_pre(x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    h = block.attn_norm(h)
    h = block.attn(h, start_pos)
    return block.hc_post(h, residual, post, comb)


def ffn_hc_only(block, x):
    residual = x
    h, post, comb = block.hc_pre(x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    h = block.ffn_norm(h)
    return block.hc_post(h, residual, post, comb)


def ffn_norm_input(block, x):
    h, _, _ = block.hc_pre(x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    return block.ffn_norm(h)


def ffn_half(block, x, input_ids):
    residual = x
    h, post, comb = block.hc_pre(x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    h = block.ffn_norm(h)
    h = block.ffn(h, input_ids)
    return block.hc_post(h, residual, post, comb)


def make_headloop_sparse_attn(base, max_heads):
    """Run >max_heads heads as multiple sub-launches (sm89 smem: h=16 fits;
    h=64 single-launch needs 141312 B > 101376 optin, A4F). topk_idxs shared
    across heads; attn_sink per-head."""
    def sparse_attn_headloop(q, kv, attn_sink, topk_idxs, softmax_scale):
        h = q.size(2)
        if h <= max_heads:
            return base(q, kv, attn_sink, topk_idxs, softmax_scale)
        outs = []
        for lo in range(0, h, max_heads):
            hi = min(lo + max_heads, h)
            outs.append(base(q[:, :, lo:hi].contiguous(), kv,
                             attn_sink[lo:hi].contiguous(), topk_idxs, softmax_scale))
        return torch.cat(outs, dim=2)
    return sparse_attn_headloop


def dp_block_forward(block, x, start_pos, input_ids, world, rank):
    """DP-attention block step: full-head attention on B/world sequences,
    all_gather, then MoE over the full microbatch."""
    B = x.size(0)
    bl = B // world
    lo = rank * bl
    a_local = attn_half(block, x[lo:lo + bl], start_pos)
    a_full = torch.empty(B, *a_local.shape[1:], dtype=a_local.dtype, device=a_local.device)
    dist.all_gather_into_tensor(a_full, a_local.contiguous())
    return ffn_half(block, a_full, input_ids)


def build_block(layer_id, max_seq, attn_max_b, world, rank, moe_mode, allreduce_dtype):
    args = model_args(max_seq, attn_max_b)
    gen = torch.Generator("cuda")
    gen.manual_seed(1234 + layer_id)
    with torch.device("cuda"):
        block = M.Block(layer_id, args)
    fill_attention_and_hc(block, gen)
    block.ffn = MarlinMoE(world, rank, mode=moe_mode, gen_seed=4321 + layer_id,
                          allreduce_dtype=allreduce_dtype)
    block.requires_grad_(False)
    block.eval()
    return block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--B", default="128,256,512")
    parser.add_argument("--layer-ids", default="2,3,0",
                        help="2=ratio-4 with indexer, 3=ratio-128, 0=pure sliding window")
    parser.add_argument("--moe-mode", choices=["itp", "ep"], default="itp",
                        help="itp: all 256 experts, inter/world rows per rank (project "
                             "placement). ep: 256/world whole experts per rank (C1' style).")
    parser.add_argument("--allreduce-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attn-mode", choices=["head-shard", "dp"], default="dp",
                        help="head-shard: 64 heads split across N ranks. "
                             "dp: full 64 heads on B/N sequences (KV/N lever).")
    parser.add_argument("--sa-max-heads", type=int, default=16,
                        help="heads per sparse_attn launch (sm89 smem limit)")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--rotate", choices=["fht", "stub"], default="fht")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    dp = args.attn_mode == "dp"
    M.world_size = 1 if dp else world
    M.rank = rank
    M.default_dtype = FP8
    M.scale_fmt = CFG["scale_fmt"]
    M.scale_dtype = E8M0
    heads_rank = N_HEADS if dp else N_HEADS // world
    M.sparse_attn = (make_headloop_sparse_attn(sparse_attn_base, args.sa_max_heads)
                     if heads_rank > args.sa_max_heads else sparse_attn_base)
    rotate_label = configure_rotate(args.rotate)
    n_attn_launch = -(-heads_rank // args.sa_max_heads)

    if rank == 0:
        print(f"C1F integrated TP block | world={world} | attn={args.attn_mode} | "
              f"moe={args.moe_mode} | gpu={torch.cuda.get_device_name(0)}")
        print("heads/rank=%d | attn_launches=%d | rotate=%s | allreduce_dtype=%s | "
              "inter/rank=%d | experts/rank=%d" % (
                  heads_rank, n_attn_launch, rotate_label, args.allreduce_dtype,
                  C.MOE_INTER // world if args.moe_mode == "itp" else C.MOE_INTER,
                  C.N_EXPERTS if args.moe_mode == "itp" else C.N_EXPERTS // world))
        print(f"{'layer':>5} {'ratio':>5} {'B':>5} {'bl':>4} {'eager_us':>10} {'graph_us':>10} "
              f"{'tok/s_layer':>12} {'tok/s/GPU':>10} {'mem_GiB':>8}")

    Bs = [int(x) for x in args.B.split(",") if x]
    if dp:
        dropped = [B for B in Bs if B % world != 0]
        Bs = [B for B in Bs if B % world == 0]
        if dropped and rank == 0:
            print(f"# dp: dropping B not divisible by world={world}: {dropped}")
    layer_ids = [int(x) for x in args.layer_ids.split(",") if x]
    max_b = max(Bs)
    max_seq = args.ctx + 256
    attn_max_b = max_b // world if dp else max_b

    for layer_id in layer_ids:
        ratio = CFG["compress_ratios"][layer_id]
        block = build_block(layer_id, max_seq, attn_max_b, world, rank,
                            args.moe_mode, args.allreduce_dtype)
        for B in Bs:
            bl = B // world if dp else B
            gen = torch.Generator("cuda")
            gen.manual_seed(9000 + B + layer_id)
            x = torch.randn(B, 1, block.hc_mult, DIM, device="cuda", dtype=torch.bfloat16, generator=gen)
            ids = torch.zeros(B, 1, dtype=torch.long, device="cuda")

            def step():
                with torch.inference_mode():
                    if dp:
                        return dp_block_forward(block, x, args.ctx, ids, world, rank)
                    return block(x, args.ctx, ids)

            eager = bench(step, iters=args.iters, warmup=8)
            graph = float("nan") if args.no_graph else bench_graph(step, iters=max(args.iters, 30), warmup=8)
            mem = torch.cuda.max_memory_allocated() / 2**30
            if rank == 0:
                best = graph if graph == graph else eager
                tok_layer = B / (best * 1e-6)
                tok_gpu = tok_layer / (world * LAYER_EQUIV)
                print(f"{layer_id:5d} {ratio:5d} {B:5d} {bl:4d} {eager:10.1f} {graph:10.1f} "
                      f"{tok_layer:12.1f} {tok_gpu:10.1f} {mem:8.2f}")

            if args.breakdown and dp:
                xl = x[rank * bl:rank * bl + bl].contiguous()
                with torch.inference_mode():
                    a_local = attn_half(block, xl, args.ctx)
                a_full = torch.empty(B, *a_local.shape[1:], dtype=a_local.dtype, device="cuda")

                def attn_local_step():
                    with torch.inference_mode():
                        return attn_half(block, xl, args.ctx)

                def ag_step():
                    dist.all_gather_into_tensor(a_full, a_local.contiguous())
                    return a_full

                def moe_step():
                    with torch.inference_mode():
                        return ffn_half(block, a_full, ids)

                with torch.inference_mode():
                    moe_x2d = ffn_norm_input(block, a_full).view(-1, C.DIM).contiguous()
                    block.ffn._ensure_workspace(moe_x2d.size(0))

                def routed_step():
                    with torch.inference_mode():
                        return block.ffn.routed_local(moe_x2d)

                def shared_step():
                    with torch.inference_mode():
                        return block.ffn.shared_only(moe_x2d)

                g_iters = max(args.iters, 30)
                attn_local_g = bench_graph(attn_local_step, iters=g_iters, warmup=8)
                ag_g = bench_graph(ag_step, iters=g_iters, warmup=8)
                moe_g = bench_graph(moe_step, iters=g_iters, warmup=8)
                routed_g = bench_graph(routed_step, iters=g_iters, warmup=8)
                shared_g = bench_graph(shared_step, iters=g_iters, warmup=8)
                if rank == 0:
                    print(f"  DP-BD layer={layer_id:d} B={B:d} bl={bl:d} | "
                          f"attn_local={attn_local_g:8.1f} allgather={ag_g:7.1f} "
                          f"moe_total={moe_g:8.1f} routed={routed_g:8.1f} "
                          f"shared={shared_g:8.1f} (us)")

            if args.breakdown and not dp:
                with torch.inference_mode():
                    moe_x = ffn_norm_input(block, x).contiguous()
                    moe_x2d = moe_x.view(-1, C.DIM).contiguous()
                    block.ffn._ensure_workspace(moe_x2d.size(0))
                    ar_buf = torch.empty(moe_x2d.size(0), C.DIM, device="cuda", dtype=torch.bfloat16)
                    ar_buf.fill_(rank + 1)

                def attn_step():
                    with torch.inference_mode():
                        return attn_half(block, x, args.ctx)

                def ffn_hc_step():
                    with torch.inference_mode():
                        return ffn_hc_only(block, x)

                def moe_total_step():
                    with torch.inference_mode():
                        return block.ffn(moe_x)

                def moe_local_step():
                    with torch.inference_mode():
                        return block.ffn.local_no_ar(moe_x)

                def routed_step():
                    with torch.inference_mode():
                        return block.ffn.routed_local(moe_x2d)

                def shared_step():
                    with torch.inference_mode():
                        return block.ffn.shared_only(moe_x2d)

                def moe_ar_step():
                    with torch.inference_mode():
                        dist.all_reduce(ar_buf)
                        return ar_buf

                g_iters = max(args.iters, 30)
                attn_g = bench_graph(attn_step, iters=g_iters, warmup=8)
                attn_noar_g = bench_graph_no_model_allreduce(attn_step, iters=g_iters, warmup=8)
                ffn_hc_g = bench_graph(ffn_hc_step, iters=g_iters, warmup=8)
                moe_total_g = bench_graph(moe_total_step, iters=g_iters, warmup=8)
                moe_local_g = bench_graph(moe_local_step, iters=g_iters, warmup=8)
                routed_g = bench_graph(routed_step, iters=g_iters, warmup=8)
                shared_g = bench_graph(shared_step, iters=g_iters, warmup=8)
                moe_ar_g = bench_graph(moe_ar_step, iters=g_iters, warmup=8)
                if rank == 0:
                    attn_ar = attn_g - attn_noar_g
                    print(f"  BD layer={layer_id:d} B={B:d} | attn={attn_g:8.1f} "
                          f"attn_noar={attn_noar_g:8.1f} attn_ar={attn_ar:7.1f} "
                          f"ffn_hc={ffn_hc_g:7.1f} moe_total={moe_total_g:8.1f} "
                          f"moe_local={moe_local_g:8.1f} moe_ar={moe_ar_g:7.1f} "
                          f"routed={routed_g:8.1f} shared={shared_g:8.1f} (us)")
        del block
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
