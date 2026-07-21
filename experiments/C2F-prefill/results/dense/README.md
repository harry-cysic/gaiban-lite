# C2F-dense:prefill MoE 执行形态与归因复核(2026-07-21)

第二十竖条。**假设被证伪,且顺带纠正了 C2F 主实验的一处归因错误。**
全部数字为本目录 JSON,titan064,3–10 次重复,event 计时。

## 动机(来自 C2F 归因)

C2F 把 prefill 的 48% 归给"Marlin 大 M MFU 仅 ~11.5%",并提出用
dequant→BF16 dense GEMM 替换(潜在 ~5×)。本竖条实现并实测该路线。

## 结论 1:dense BF16 路线证伪(慢 4–17×)

实现:expert 常驻 checkpoint 布局(packed E2M1 + E8M0,**字节数与 Marlin 常驻
完全相同 861,931,008**,且免 repack 加载快 7×:0.8 s vs 5.8 s),逐 chunk
dequant 成 BF16 + 排序分组 dense GEMM(`dsv4_direct/ops/dense_moe.py`)。

数值前提成立:

| 口径 | rel_fro vs FP32 oracle |
|---|---|
| dense(BF16 dequant + BF16 GEMM/FP32 累加) | **4.284e-3** |
| marlin W4A16(冻结路径) | 4.331e-3 |

且 `dequant_bf16_exact = true`——MXFP4 值(≤3 位有效 + 2 幂 scale)在 BF16 中
精确可表示,dequant 无损。

速度(routed 半层,单卡,gathered 行数):

| rows | marlin (ms) | dense (ms) | dense/marlin |
|---:|---:|---:|---:|
| 4096 | 3.39 | 57.9 | 0.06× |
| 8192 | 5.58 | 62.6 | 0.09× |
| 16384 | 10.55 | 69.0 | 0.15× |
| 32768 | 20.45 | 84.4 | **0.24×** |

归因:dense 路径的 GEMM 只跑到 ~29 TFLOPS,且 epilogue(fp32 clamp/silu 与
[assignments, hidden] fp32 加权/反排序)产生 ~13 GB 额外流量;Marlin 把整条
swiglu + 路由加权 + 反排序全部融进 kernel。**结论:prefill 继续用 Marlin。**

## 结论 2:Marlin 在 prefill 已达 ~82% 峰值,C2F 的 "MFU 11.5%" 归因有误

32768 行 routed GEMM = 2.47 TFLOP / 18.3 ms = **135 TFLOPS**,即 RTX 4090
BF16 dense 峰值(~165 TFLOPS)的 **~82%**。C2F 的 11.5% 来自把**整个 MoE 调用
的 131 ms** 当成 GEMM 时间与 15 ms dense roofline 相比;实际 GEMM 只占其中
18–21 ms。

runtime 各组件(单卡,`c2f-moe-probe.json`):

| rows | block_size_m | 确定性对齐 | 私有 Marlin | topk 归约 | 合计 | 公有 API |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 64 | 0.55 | 5.42 | 0.50 | 6.46 | 5.71 |
| 32768 | 64 | 0.52 | 18.29 | 2.00 | 20.81 | 20.43 |

自研确定性对齐与 block-size 选择都不是问题(合计与 vLLM 公有 API 等价)。

## 结论 3:MoE 调用的真实分相(4 卡,真实权重,chunk 8192 → 32768 gathered 行)

`c2f-moe-phase-8192.json`,**整调用 49.8 ms/层**:

| 相 | ms | 占比 |
|---|---:|---:|
| all_gather(268 MB) | 8.48 | 17% |
| gate(fp32 [32768,4096]×[4096,256] + topk) | 3.03 | 6% |
| **routed Marlin** | **21.00** | **42%** |
| shared expert(FP8 tilelang) | 3.43 | 7% |
| combine(fp32 加 + cast) | 4.97 | 10% |
| reduce_scatter(268 MB) | 8.73 | 18% |
| finalize | 0.12 | 0% |

DP 集合通信合计 17.2 ms/层(35%),其有效带宽 ~31 GB/s,已在 B1 标定的机内
上限附近——**集合通信本身无优化空间,只能靠改并行形态减少通信量**。

## 结论 4:C2F 的 "MoE = prefill 50%" 不可复现,需重做 prefill 归因

同一调用、同口径(含 C2F 的 per-rank sync / 无 barrier 形态)实测均为
**49.8 ms/层**(`-nobarrier.json`:host wall 49.80 vs event 49.78;
`collect_trace=True` 仅 +2.5 ms)。C2F 报的 131 ms/层无法在隔离条件下复现,
差额 ~80 ms/层 只能来自 11 层链式上下文(MoE 是链上唯一含集合通信处,会吸收
attention 侧累积的跨 rank 偏斜;以及链中每层临时张量的分配器压力)。

**含义**:若 MoE 真实计算是 0.55 s/pass 而非 1.44 s,则 prefill 的大头是
attention 侧(C2F 自己也测到 eager torch attention 占 40%)——**prefill 的
优化目标应是 attention 路径(runtime 至今用的是 torch masked-einsum 正确性
实现,tilelang sparse kernel 从未接入),而不是 MoE**。该重新归因是下一竖条
的前置,本竖条不据此下最终结论。

## 产物

`c2f-dense-gate.json`(数值门 + 行扫描)、`c2f-moe-probe.json`(组件)、
`c2f-moe-phase-8192{,-trace,-nobarrier}.json`(4 卡分相)。
代码:`runtime/dsv4_direct/ops/dense_moe.py`(保留,作为 dense 路线的可复现
证据与 checkpoint 布局常驻加载器)、`runtime/c2f_dense_moe_gate.py`、
`runtime/c2f_moe_component_probe.py`、`runtime/c2f_moe_phase_probe.py`。
默认执行路径未改动(Marlin)。
