# A3F-marlin-moe-flash:grouped Marlin MXFP4 MoE 换 Flash 几何

日期 2026-07-20。Phase 1 kernel regear 第一件套。单卡 titan065(GPU0),
venv `~/Workspace/venvs/sglang`(vllm 0.22.1 kernel 库),CUDA_HOME=/usr/local/cuda-13.2。

## 动机与方法

复用 gaiban A3 bench(`common.py`/`a3_moe_bench.py`/`a3_numerics.py` 原样拷贝,
几何常量改为 env 可覆盖),Flash 几何:E=256、K=hidden=4096、inter=2048、topk=6、
route_scale=1.5。两个形状口径:

- **inter2048**:整 expert(对照 Pro A3 锚点口径 384×(3072,7168) 的直接换算);
- **inter512**:per-expert intermediate-TP 在 TP4 下的**每卡真实 decode 形状**
  (每卡持每 expert 的 1/4 inter 切片)。

计时单元 = 完整 MoE decode 步(gate → fused_marlin_moe → shared FP8 → add),
路由每步换池,L2 卫生同 gaiban。dist=gate 为真实 gate 分布口径。

## 数值 gate(a3_numerics,E=32,通过)

| 口径 | rel_fro vs fp32 oracle |
|---|---|
| reference(model.py, act-quant fp8 + tilelang fp4_gemm) | 4.69e-2(clamp-inactive)/ 8.7e-2(clamp-ACTIVE) |
| **marlin W4A16** | **4.2e-3 / 3.5e-3**(比 reference 更贴 oracle,预期内:act 保持 bf16) |
| marlin W4A8 | 4.6e-2 / 8.5e-2(≈ reference 同量级) |

通过判据(marlin16 ≤ ref;W4A8 在 A1.5 量级)成立,Flash shape 下 Marlin 链路数值正确。

## Bench 结果(dist=gate,完整 CSV 见 results/)

**inter2048(整 expert 口径)**:

| B | t/层 (µs) | distinct experts | 读量 (MB) | eff GB/s |
|---:|---:|---:|---:|---:|
| 64 | 2740 | 183 | 2336 | 894 |
| 96 | 3165 | 213 | 2713 | **899(峰值)** |
| 128 | 3412 | 228 | 2913 | 895 |
| 512 | 4279 | 255 | 3256 | 798 |

峰值 899 GB/s = Pro 锚点(916–929)的 ~97%,**kernel 族换 Flash 几何成立**。

**inter512(TP4 每卡 decode 真实形状)**:

| B | t/层 (µs) | 读量 (MB) | eff GB/s |
|---:|---:|---:|---:|
| ≤64 | ~840–908(平坦) | ≤582 | ≤725 |
| 128 | 998 | 723 | **760(峰值)** |
| 256 | 1126 | 793 | 739 |
| 512 | 1241 | 813 | 687 |

w4a8 与 w4a16 在两个口径都基本重合(±2%),decode 用 W4A16 的分工不变。

## 关键结论

1. **窄 N 切片有实效税**:inter512 峰值 760 GB/s vs inter2048 的 899(−15%),
   B=512 时 687 GB/s。GEMM tile 在 N=512 下效率下降。
2. **对 t_stage 模型的影响**:DP-attention 下每卡 marlin 每层见全 B=512,
   实测 1.241 ms/层 × 11 层 = **13.7 ms/stage MoE**,高于可行性 §5.2 的
   ~10.8 ms(那是按 ~900 GB/s 摊的)。其余分项不变时 t_stage ≈ 27–30 ms,
   decode 预估从 19–21k 收敛到 **~17–19k tok/s**——仍在 15–25k 目标区间内,
   §5.2 的分解表应在 Phase 2 实测后修订。
3. **grouped marlin 有 ~840 µs 调用地板**(B≤64 时间平坦,两口径同现),
   低并发 decode 由该地板主导;与 Pro「decode 是 launch-bound」结论同族,
   全 stage CUDA graph 的必要性再次确认。
4. **形态 A/B 候选(Phase 2)**:EP(每卡 64 整 expert)与 intermediate-TP
   字节量相同但 GEMM 保持 N=2048(~894 GB/s 口径),理论上省 ~27% MoE 时间;
   代价是 token 路由不均衡与 all2all/gather 结构变化。按长期原则,置换前须
   causal A/B 证伪 intermediate-TP,先记录不动。

Artifacts:`results/a3f_{w4a16,w4a8}_inter{2048,512}.csv`、`results/log_*.txt`
(含 numerics 全文)。运行现场 titan065:`~/a3f-marlin/`。
