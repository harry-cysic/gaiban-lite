# C2F 重归因(2026-07-21):prefill 由 attention 主导,基线上修 41–45%

第二十竖条的副产物。起因:dense-MoE 竖条测得 MoE 整调用 49.8 ms/层,而 C2F
主实验的 MoE 桶是 131 ms/层。用 C2F **自己的脚本、参数与协议**复跑定位。

## 复现结果(titan064,4 卡,11 层 L11–L21,chunk 8192,iters 5/warmup 2)

| 臂 | 轮 | input tok/s/stage | moe (s) | attn ratio-4 | attn ratio-128 | hc | total_instr |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline w4a16/ref | r1 | **15,060** | 0.548 | 0.958 | 0.329 | 0.313 | 2.181 |
| baseline | r2 | 15,044 | 0.551 | 0.959 | 0.330 | 0.313 | 2.186 |
| all-on w4a8+fused | r1 | **16,602** | 0.485 | 0.820 | 0.329 | 0.313 | 1.981 |
| all-on | r2 | 16,608 | 0.482 | 0.821 | 0.329 | 0.313 | 1.980 |

轮间离散 <0.2%。对照 C2F 报告值:baseline 10,688、all-on 11,417。

## 差额的确切归属:全部在 MoE 桶

C2F 与本次的 **attention / HC 分量逐项吻合**(ratio-4 0.955 vs 0.958、
ratio-128 0.328 vs 0.329、hc 0.313 vs 0.313),唯独 MoE 1.439 vs 0.549 s。
吞吐差 (32768/10688 − 32768/15060) = 0.89 s 与 MoE 桶差 0.89 s **精确相等**。

MoE 桶 49.9 ms/层 与两个独立测量一致:2 层链式 50.1 ms/层、隔离 4 卡分相
49.8 ms/层(见 `../dense/`)。故 C2F 那次运行的 MoE 确实慢 2.6×,但**不是
kernel/几何性质**,在当前环境下不复现。最可能成因是分配器状态:MoE 每层每次
分配数 GB 的 fp32 combine 临时量,若进程内已有碎片(C2F 的矩阵/gate 臂同进程
串跑)会退化为 cudaMalloc/cudaFree 循环。**记录为鲁棒性缺陷:prefill MoE 的
fp32 临时量应预分配复用**(潜在收益即这 0.89 s ≈ +32% 吞吐)。

## 修正后的 prefill 归因(all-on,每 pass 1.981 s)

| 分项 | s | 占比 |
|---|---:|---:|
| **ratio-4 attention** | 0.820 | **41%** |
| MoE(整调用) | 0.483 | 24% |
| ratio-128 attention | 0.329 | 17% |
| HC | 0.313 | 16% |
| norm | 0.032 | 2% |
| attention 合计 | **1.149** | **58%** |

**prefill 由 attention 主导,不是 MoE。** runtime 的 attention 至今是 torch
masked-einsum 正确性实现(reference 的 tilelang `sparse_attn` 从未接入
direct runtime——DP 竖条已记录该事实),这是最大且未动过的杠杆。

## 对 §5.3 与单池目标的更新

- prefill P(16 卡投影 = 单 stage 口径)**16.6k input tok/s**(原记 11.4k)。
- 单池 T = 1/(1/8733 + 8/16608) = **1,677 tok/s**(原记 ~1.2k),目标带
  3.2–4.2k 仍需 P ≈ 40k+。
- 路径:attention 换真 kernel(58% 的大头;若 3× 则 pass 1.98→1.21 s ≈ 27k)
  + prefill HC 融合(16%)+ MoE 临时量预分配 → §5.3 的 30–40k 带可及。

## 产物

`out-c2f-v2-{base,allon}-{1,2}/`(本次 4 轮)、`out-c2f-attr11/`(首次 11 层
复现)、`out-c2f-attr2/`(2 层对照,MoE 50.1 ms/层)。
