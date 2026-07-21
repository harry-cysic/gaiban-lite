# C4F — ratio-4 attention 分相定位 + 融合 indexer QAT 核

第二十七竖条(2026-07-21,titan064 / titan065 实测)。C2F 冻结的 prefill 配置
(tilelang sparse core + 整段 prefill)在 **25,307 input tok/s** 上,ratio-4
attention 是最大的非 MoE 分量。本竖条先做**层内分相**,再按占比选杠杆。

## 结论速览

| 项 | 结果 |
|---|---|
| **分相发现** | ratio-4 层 72.07 ms 里,`fp4_quant_dequant(hadamard_transform(index_query))` 占 **29.75 ms = 41.3%** —— **比 tilelang 稀疏核(15.77 ms)还大**。它一个 GEMM 都没有,是一串纯 elementwise |
| **杠杆 A(融合 indexer QAT 核)** | **放行**。单 Triton kernel,**逐位相等**(5 组形状 0 个不等元素);29.767 → 0.329 ms = **90.5×** |
| **端到端** | C2F 同口径 3 轮 **28,609 / 28,622 / 28,657**,中位 **28,622 input tok/s**,对 25,307 为 **1.131×(+13.1%)**;轮间散布 0.167% |
| **长门** | **494/512,逐 prompt 62/62/63/63/61/61/59/63 与冻结基线完全相同**;8 条 prompt 的 `first_mismatch` 全部字段(step / predicted / golden / 三个 logit)逐位一致 |
| **显存** | 每卡峰值 **20.409 → 17.845 GiB(−2.564 GiB,−12.6%)** |
| **杠杆 B(compressor FP32 cast 提取)** | 逐位相等但只值 0.166 ms/层 = 0.83 ms/pass = **0.07%,低于 0.167% 的轮间噪声 → 不放行**,代码保留默认关 |
| **杠杆 C(wo_a 的 o_groups=8 分组 einsum)** | **判死**:einsum 3.243 ms vs 显式 bmm 3.233 ms(差 0.3%)。**没有退化**,einsum 已经降到高效批量 GEMM;块对角稠密形式反而 8.1× 更慢(26.4 ms) |
| **杠杆 D(投影 GEMM 形状/dtype)** | **判死**:wq_b / wo_a / wo_b 实测 **158–165 TFLOPS ≈ 4090 BF16 稠密张量峰值的 96–100%**,算力饱和、无带宽余量可回收 —— 与 decode 侧 W8A16 判中性同结论、不同原因 |
| **单池 T** | P=25,307 → 28,622,**T 2,322 → 2,538(+9.3%)**;带下沿 3.2k 需 **P ≈ 40,406** |
| **下一杠杆(已定界,不在本竖条)** | tilelang 稀疏核 **51.6 TFLOPS = 31% MFU**,外加包装层 **16.1% 的搬运税** |

**一句话**:上一轮把稀疏核换成 tilelang 之后,ratio-4 的头号成本已经**不是
attention 本身**,而是 indexer query 的 QAT 仿真链;把它融成一个逐位相等的
kernel 就拿到 +13.1%,而且是**零语义变更**。

---

## 1. 分相 profile(先做的事)

### 1.1 仪器

- [`runtime/dsv4_direct/phase_timer.py`](../../runtime/dsv4_direct/phase_timer.py):
  `PhaseRecorder` 在流上记录 CUDA event、**pass 结束后只同步一次**,不像 C2F
  的 per-component `synchronize`(短 forward 下可达 +14.7%)。
- [`runtime/dsv4_direct/ratio4_fullpos.py`](../../runtime/dsv4_direct/ratio4_fullpos.py)
  的 `__call__` 里插了 30 个 `self._mark(...)` 点。`phase_recorder` 是**类属性,
  默认 None**,所以未挂 recorder 时每个点只付一次属性读 + 一次分支;**没有任何
  张量算子、dtype 或顺序改动**。
- [`runtime/c4f_ratio4_phase_probe.py`](../../runtime/c4f_ratio4_phase_probe.py):
  单进程单卡探针。**为什么单卡够**:本 runtime 的 attention 是 DP 形态 ——
  每条 TP lane 用**完全复制**的权重(`load_replicated_block_weights`)在自己的
  B=1 序列上算**全部 64 头**,ratio-4 层里没有任何集合、也不依赖 TP。所以
  一张卡就是 C2F bench 的一条 lane。

**保真度**(chunk 8192、layer 12、真实权重):

| 量 | 值 |
|---|---|
| 未插桩层 p50 | **72.073 ms** |
| 插桩层 p50 | 72.164 ms(**开销 +0.13%**) |
| 相位事件求和 | 72.085 ms(**覆盖 100.02%**) |
| ×5 层折算 | **0.3604 s** |
| C2F 冻结三轮的 `attention_ratio4` 桶 | 0.3611 / 0.3613 / 0.3611 s |
| 差 | **−0.2%** |

> 任务书记的是 "0.3635 s / 34.0%"。**秒数**与实测 0.3611–0.3613 差 0.7%,可视作
> 同口径不同轮;但 **34.0% 这个占比在冻结 JSON 里对任何分母都复现不出来**
> (对 `total_instrumented` 1.3005 s 是 **27.8%**,对未插桩 wall 1.2948 s 是
> 27.9%)。本文一律用实测秒数,占比另注分母。

### 1.2 分相表(chunk 8192,layer 12,真实权重,10 iters p50)

冻结配置 = W4A8 Marlin MoE + D0b fused indexer + tilelang 稀疏核 +
`sparse_row_block=1024`。

| # | 相位 | 冻结 (ms) | 占层 | 杠杆 A 后 (ms) | 占层 |
|---:|---|---:|---:|---:|---:|
| 1 | **`idx_hadamard_fp4`** | **29.746** | **41.27%** | **0.284** | 0.67% |
| 2 | `sparse_core`(tilelang) | 15.775 | 21.89% | 15.771 | 37.00% |
| 3 | `wo_b` | 3.480 | 4.83% | 3.478 | 8.16% |
| 4 | `q_wq_b` | 3.335 | 4.63% | 3.336 | 7.83% |
| 5 | `wo_a_einsum` | 3.214 | 4.46% | 3.214 | 7.54% |
| 6 | `q_head_norm` | 2.951 | 4.09% | 2.951 | 6.92% |
| 7 | `idx_score`(D0b fused) | 2.507 | 3.48% | 2.495 | 5.85% |
| 8 | `comp_main_wkv_fp32` | 1.755 | 2.44% | 1.757 | 4.12% |
| 9 | `comp_main_wgate_fp32` | 1.736 | 2.41% | 1.740 | 4.08% |
| 10 | `out_rope` | 0.905 | 1.26% | 0.903 | 2.12% |
| 11 | `q_rope` | 0.892 | 1.24% | 0.892 | 2.09% |
| 12 | `idx_rope` | 0.882 | 1.22% | 0.882 | 2.07% |
| 13 | `idx_wq_b` | 0.821 | 1.14% | 0.821 | 1.93% |
| 14 | `comp_index_wkv_fp32` | 0.636 | 0.88% | 0.637 | 1.50% |
| 15 | `comp_index_wgate_fp32` | 0.616 | 0.86% | 0.617 | 1.45% |
| 16 | `idx_topk` | 0.606 | 0.84% | 0.607 | 1.42% |
| 17 | `q_wq_a_norm` | 0.572 | 0.79% | 0.574 | 1.35% |
| 18 | `compress_main` | 0.496 | 0.69% | 0.495 | 1.16% |
| 19 | `kv_wkv_norm` | 0.268 | 0.37% | 0.268 | 0.63% |
| 20 | `compress_index` | 0.241 | 0.33% | 0.239 | 0.56% |
| 21 | `idx_mask_add` | 0.177 | 0.25% | 0.177 | 0.42% |
| 22 | `idx_weights_proj` | 0.112 | 0.15% | 0.112 | 0.26% |
| 23 | `kv_fp8_qdq` | 0.100 | 0.14% | 0.099 | 0.23% |
| 24 | `idx_mask_build` | 0.086 | 0.12% | 0.086 | 0.20% |
| 25 | `idx_index_fixup` | 0.078 | 0.11% | 0.080 | 0.19% |
| 26 | `window_index` | 0.032 | 0.04% | 0.037 | 0.09% |
| 27 | `kv_cat` | 0.020 | 0.03% | 0.020 | 0.05% |
| 28 | `kv_rope` | 0.018 | 0.02% | 0.018 | 0.04% |
| 29 | `topk_cat` | 0.014 | 0.02% | 0.025 | 0.06% |
| 30 | `ring_write` | 0.009 | 0.01% | 0.008 | 0.02% |
| | **层合计** | **72.073** | | **42.627** | |

相位命名对应 `ratio4_fullpos.py::__call__` 的语句顺序:q 投影链
(`q_wq_a_norm` → `q_wq_b` → `q_head_norm` → `q_rope`)、raw KV 链
(`kv_wkv_norm` → `kv_rope` → `kv_fp8_qdq`)、四个 FP32 compressor 投影、
overlap 压缩器(`compress_main` / `compress_index`)、indexer
(`idx_wq_b` → `idx_rope` → `idx_hadamard_fp4` → `idx_weights_proj` →
`idx_mask_build` → `idx_score` → `idx_mask_add` → `idx_topk` →
`idx_index_fixup`)、稀疏核、出口(`out_rope` → `wo_a_einsum` → `wo_b`)。

### 1.3 从分相表读出的三件事

1. **头号项不是 attention,是 QAT 仿真。** `idx_hadamard_fp4` 是
   `fp4_quant_dequant(hadamard_transform(index_query))` 这一对函数。输入
   `[1, 8192, 64, 128]` BF16 = 134 MB;`hadamard_transform` 有 7 级
   `cat((left+right, left-right))`、`fp4_quant_dequant` 有 amax / log2 / ceil /
   exp2 / 除 / clamp / 7 层嵌套 `where` / copysign —— 每一步都物化一个 268 MB
   的 FP32 中间量,一次 GEMM 都没有。按融合核实测的 816 GB/s 反推,这 29.75 ms
   对应约 **24 GB 的显存往返**,而这条链的信息量只有 **0.268 GB**(读 134 MB
   BF16 + 写 134 MB BF16)—— 即 **约 90× 的冗余搬运**,与融合后实测的 90.5×
   加速自洽。微基准拆开:hadamard **13.277 ms**、fp4 **16.506 ms**,合
   29.783 ms,与相位表的 29.746 ms 差 0.1%。
2. **投影 GEMM 已经顶到硬件峰值,没有杠杆。** 见 §4。
3. **稀疏核已经是新的头号项**(换算后占层 37%),但它是 reference kernel,
   本竖条只做定界(§5)。

---

## 2. 杠杆 A:融合 indexer QAT 核(**放行**)

### 2.1 实现

[`runtime/dsv4_direct/ops/indexer_qat.py`](../../runtime/dsv4_direct/ops/indexer_qat.py),
一个 Triton kernel 吃掉整条链。目标不是"接近",是**逐位相等**:

- **Hadamard 是标准 FWHT。** 把 `hadamard_transform` 的 reshape/cat 形式按
  `i = a·2s + b·s + c` 展开(`(width//(2s), 2, s)` 视图),
  `cat((left+right, left-right), -1)` 恰好落成
  `new[a·2s+c] = t[a·2s+c] + t[a·2s+s+c]`、
  `new[a·2s+s+c] = t[a·2s+c] − t[a·2s+s+c]` —— 就是原地蝶形。kernel 用
  `tl.reshape`/`tl.permute`/`tl.split`/`tl.join` 做**同样配对顺序的 7 级 FP32
  蝶形**,所以每个输出都由完全相同的一串 FP32 加减产生。
- **保留中间 BF16 舍入。** eager 链里 `hadamard_transform` 返回
  `value.dtype`(BF16)、`fp4_quant_dequant` 再 `.float()` 回来。这个往返是
  有语义的,kernel 照做(`.to(tl.bfloat16).to(tl.float32)`)——省掉它才是真的
  数值改动。
- **copysign 走符号位而不是比较。** `torch.copysign(0.0, -0.0) = -0.0`;用
  `where(n < 0, ...)` 会把它变成 `+0.0`。kernel 用
  `(snapped_bits | (n_bits & 0x80000000))`,`-0.0` 原样保留。
- **`exp2(ceil(log2(amax/6)))` 用 libdevice 同款函数。** 这是唯一"按构造匹配"
  而非"按代数等价"的一步,所以门里在真实数据上逐位复验(下表)。

只支持 `[..., 128]` × `group_size=32` 这一种形状(indexer query 的宽度);
其他宽度一律走 eager。**decode(seqlen == 1)无条件走 eager**,与 D0b fused
indexer 的 `fuse_min_seqlen` 约定一致,decode 侧头条数字不受影响。

### 2.2 kernel 级门([`results/out-c4f-qat/c4f-kernel-qat.json`](results/out-c4f-qat/c4f-kernel-qat.json))

| 形状 | 元素数 | 逐位相等 | 不等元素 | max&#124;Δ&#124; |
|---|---:|---|---:|---:|
| `[1, 8192, 64, 128]`(部署形状) | 67,108,864 | **是** | **0** | 0.0 |
| `[1, 4096, 64, 128]` | 33,554,432 | **是** | **0** | 0.0 |
| `[1, 1024, 64, 128]` | 8,388,608 | **是** | **0** | 0.0 |
| `[1, 97, 3, 128]`(非对齐) | 37,248 | **是** | **0** | 0.0 |
| `[1, 1, 64, 128]`(单 token) | 8,192 | **是** | **0** | 0.0 |

速度与显存(chunk 8192):

| 量 | eager | fused | 比 |
|---|---:|---:|---:|
| p50 | 29.767 ms | **0.329 ms** | **90.5×** |
| 峰值分配 | 3,976 MB | **268 MB** | −93.3% |
| 有效带宽 | — | **816 GB/s** | 4090 规格 1008 GB/s 的 81% |

**0.329 ms 已经是搬运下界**:读 134 MB + 写 134 MB = 268 MB,816 GB/s 下
0.329 ms —— 这条链剩不下什么了。

### 2.3 层级门(真实权重,chunk 8192,[`results/out-c4f-qatfused/c4f-ab-qatfused.json`](results/out-c4f-qatfused/c4f-ab-qatfused.json))

同一 residual 过两条独立建的 layer-12 lane:

| 量 | 结果 |
|---|---|
| 输出 `bitwise_equal` | **True**(`rel_fro` 0.0,`max_abs_diff` 0.0) |
| 状态 `compressed` / `indexer_kv` / `raw` | **全部逐位相等** |
| 有限性 | OK |

### 2.4 长门(D0L,16 卡跨机,[`results/long-gate/e2e-long-qatfused.json`](results/long-gate/e2e-long-qatfused.json))

用与冻结基线**完全同一个仪器**(`run_c4f_long_gate.sh` 与
`run_e0l2e_long_arm.sh` 除 `DSV4_INDEXER_QAT` 外逐行相同:tilelang prefill
sparse + eager HC + 顺序 MoE,`--max-seq-len 4224 --max-steps 64`):

| 量 | 基线 | 杠杆 A |
|---|---|---|
| **总分** | **494/512 = 0.96484375** | **494/512 = 0.96484375** |
| 逐 prompt | 62/62/63/63/61/61/59/63 | **62/62/63/63/61/61/59/63** |
| mismatch 数 | 18 | 18 |
| max `top2_gap` | 0.6356 | **0.6356**(基线包络 0.9595) |

8 条 prompt 的 `first_mismatch` 记录 —— step、predicted、golden、
`top1_logit`、`top2_logit`、`golden_deficit` —— **全部字段逐位一致**。这是预期
结果:层级已经逐位相等,长门只是把这一点在 43 层满配模型上复验了一遍。
**未放宽任何容差。**

长门里 prefill 的 wall 也独立佐证了增益(1024/2048/4096 行三档):
prompt 3 (2048) 1477.5 → 1330.0 ms(1.111×)、prompt 6 (4096)
2965.2 → 2653.5 ms(1.117×)、prompt 7 (4096) 2961.4 → 2649.8 ms(1.118×)。

### 2.5 端到端吞吐(C2F 同口径,3 轮)

口径完全沿用 C2F:11 层 L11–21、chunk 8192、B=1/lane 的 DP4、iters 5 /
warmup 2、headline 取**未插桩** p50、`NCCL_P2P_LEVEL=SYS`。

| 轮 | 基线 tok/s | 杠杆 A tok/s |
|---|---:|---:|
| 1 | 25,312.5 | **28,656.8** |
| 2 | 25,304.8 | **28,622.0** |
| 3 | 25,306.7 | **28,608.9** |
| **中位** | **25,306.7** | **28,622.0** |
| 轮间散布 | 0.030% | 0.167% |

**1.131×(+13.1%)**。

分量墙钟(instrumented pass 中位,秒):

| 分量 | 基线 | 杠杆 A | Δ |
|---|---:|---:|---:|
| MoE(11 层) | 0.4340 | 0.4320 | −0.0020 |
| HC(×22) | 0.3134 | 0.3132 | −0.0002 |
| **attention ratio-4(5 层)** | **0.3611** | **0.2138** | **−0.1473** |
| attention ratio-128(6 层) | 0.1588 | 0.1584 | −0.0004 |
| rms_norm | 0.0322 | 0.0324 | +0.0002 |
| 未插桩 wall p50 | 1.2948 | 1.1449 | **−0.1500** |

**ratio-4 桶的 −0.1473 s 解释了整 pass −0.1500 s 的 98.2%**,其余三桶在
±0.6% 内不动 —— 没有副作用,也没有把成本挪到别处。单卡探针预测的
−0.1473 s(0.3604 → 0.2131)与实测 −0.1473 s **完全吻合**。

P2P 自检(每轮 JSON 的 `moe_collective_selfcheck`):all-gather 总线
**23.16 / 23.54 / 23.61 GB/s** —— 落在直连 P2P 的 ~24 GB/s,不是 SHM 回退的
~4 GB/s。

**收尾复核**(同一份最终代码上再各跑一轮,
[`results/c2f-finalcode/`](results/c2f-finalcode/)):

| 臂 | tok/s | wall p50 | ratio-4 桶 | 每卡峰值 | AG 总线 |
|---|---:|---:|---:|---:|---:|
| `DSV4_INDEXER_QAT=ref` | 25,359.9 | 1.2921 s | 0.3605 s | **20.409 GiB** | 23.87 GB/s |
| `DSV4_INDEXER_QAT=fused` | 28,610.7 | 1.1453 s | 0.2137 s | **17.845 GiB** | 23.24 GB/s |

`ref` 臂对冻结基线 25,304.8–25,312.5 差 **+0.21%(噪声内)**、每卡峰值
**20.409 GiB 与冻结值完全一致** —— 证明 30 个相位 mark 点与两个 mode hook
**对默认路径零代价、零改动**。`fused` 臂落在前述三轮的 28,609–28,657 带内。

### 2.6 显存

每卡 prefill 峰值分配 **20.409 → 17.845 GiB(−2.564 GiB,−12.6%)**,来自不再
物化 3.98 GB 的 FP32 中间量。这直接抬高 C3F 记的"整段 prefill 上限
≈11.4K token"(该上限是显存限,不是算力限),但本竖条未重测该上限。

---

## 3. 杠杆 B:compressor FP32 cast 提取(**不放行**)

`ratio4_fullpos.py` 里四个 FP32 compressor 投影各自写了 `hidden.float()`,
即同一个 BF16→FP32 转换算了 4 遍(每遍 134 MB 读 / 268 MB 写)。
`Tensor.float()` 是纯确定性 elementwise 转换,提取成一次是**公共子表达式消除,
不是数值改动**。

| 量 | 结果 |
|---|---|
| 层级 A/B(真实权重,chunk 8192) | **逐位相等**,输出与三个状态张量全等 |
| 微基准预期 | `comp_main` 1.741 → 1.524 ms、`comp_index` 0.634 → 0.466 ms(cast 单趟 0.233 ms) |
| **层内实测** | 42.627 → **42.461 ms/层(−0.166 ms)** |
| 折算 | −0.83 ms/pass = **0.07%** |

**判定:不放行。** 0.07% 低于 3 轮的 0.167% 散布,拿它去换一次长门的风险预算
不划算。代码保留(`compressor_cast_mode`,默认 `"ref"`),留给以后真正需要时
一并翻开。预期 0.65 ms/层没有兑现成 0.166 ms,说明这几个 cast 已被分配器
复用 / 与 GEMM 部分重叠 —— 这本身是"微基准不等于层内"的一个提醒。

---

## 4. 杠杆 C / D:按分相表判死

### 4.1 wo_a 的 `o_groups=8` 分组 einsum —— 没有退化

任务书怀疑 `torch.einsum("bsgd,grd->bsgr", grouped, wo_a)` 会掉进低效 kernel。
在部署形状(`grouped [1, 8192, 8, 4096]`、`wo_a [8, 1024, 4096]`)上实测:

| 形式 | p50 (ms) |
|---|---:|
| `torch.einsum("bsgd,grd->bsgr")`(现役) | **3.243** |
| 显式 `bmm`(permute → bmm → permute) | 3.233 |
| 块对角稠密 `F.linear`(8192×8192 权重) | 26.399 |

einsum 与手写 bmm 差 **0.3%** —— 它已经降到了高效批量 GEMM;块对角形式反而
**8.1× 更慢**(算了 8× 的零)。**判死。**

### 4.2 投影 GEMM 的形状 / dtype —— 算力已饱和

| GEMM | FLOPs | p50 (ms) | 实测 TFLOPS | 占 4090 BF16 稠密峰值(165.2) |
|---|---:|---:|---:|---:|
| `q_wq_b` (1024→32768) | 550.0 G | 3.336 | 164.9 | **99.8%** |
| `wo_b` (8192→4096) | 550.0 G | 3.478 | 158.1 | **95.7%** |
| `wo_a` (分组 8×4096→1024) | 550.0 G | 3.214 | 171.1 | ~100%(含时钟 boost) |

**判死。** prefill 侧这三个投影是**算力受限、且已经跑在张量核峰值上**,W8A16
之类的权重量化在这里没有可回收的带宽 —— 与 decode 侧"W8A16 判中性"结论相同,
原因相反(decode 是权重带宽受限但收益被别处吃掉,prefill 是压根不带宽受限)。

### 4.3 compressor 的 FP32 段 —— 可做但主动不做

reference 语义要求这四个投影在 FP32(`model.py:322-324`),实测代价
**4.743 ms/层(占层 6.6%)**。开 TF32 的收益已量化:

| 形式 | main (1024 宽) | index (256 宽) |
|---|---:|---:|
| FP32(现役,含 cast) | 1.741 ms | 0.634 ms |
| FP32(cast 已提取) | 1.524 ms | 0.466 ms |
| **TF32** | **0.933 ms** | **0.244 ms** |
| BF16 | 0.460 ms | 0.132 ms |

TF32 可省约 1.6 ms/层 = 8 ms/pass ≈ **0.7%**。**不做**:这是真语义变更
(FP32 → 10 位尾数),要占一次长门的风险预算,而 0.7% 不值。如实记录,不改。

---

## 5. 下一杠杆:tilelang 稀疏核(本竖条只定界)

杠杆 A 之后 `sparse_core` 占 ratio-4 层的 **37%**(15.77 ms/层 = 78.9 ms/pass
= 整 pass 的 6.9%)。拆开量化(chunk 8192,`[1,8192,64,512]` query、
`[1,10240,512]` KV、640 个 top-k 索引):

| 段 | p50 (ms) | 占比 |
|---|---:|---:|
| 包装层总计(`tilelang_sparse_attention`) | 15.872 | 100% |
| — reference kernel 本体(4 次 head-chunk 调用) | 13.312 | 83.9% |
| — head 切片 `.contiguous()`(4 × 134 MB) | 1.196 | 7.5% |
| — 其余包装(输出 `copy_` 回写、校验归约 + `.tolist()` 同步、空行修补) | 1.364 | 8.6% |

- **kernel 本体 51.6 TFLOPS = 4090 BF16 峰值的 31%**(687.2 GFLOP / 13.312 ms)。
- **包装层税 16.1%**,来自 sm89 共享内存上限强制的 `head_chunk=16` head 循环
  (32 亦超限,A4F 已记):每个 head chunk 都要把 query 切片 `.contiguous()`
  一次、把结果 `copy_` 回去一次,合计约 2.15 GB 的额外搬运。

两条路都有:(a) 重写 kernel 提 MFU,(b) 把 head-chunk 布局一路贯通到
`wo_a`(64 头 / o_groups=8 = 每组 8 头,head_chunk 16 恰好是 2 个 o_group,
所以出口侧天然可以按 head chunk 分块而不必拼回),把 16.1% 的搬运税去掉。
**均不在本竖条范围。**

---

## 6. 产物

| 文件 | 内容 |
|---|---|
| [`runtime/dsv4_direct/phase_timer.py`](../../runtime/dsv4_direct/phase_timer.py) | 无同步 CUDA-event 相位记录器 |
| [`runtime/dsv4_direct/ops/indexer_qat.py`](../../runtime/dsv4_direct/ops/indexer_qat.py) | 融合 Hadamard + FP4 QAT kernel + `bitwise_selfcheck` |
| [`runtime/c4f_ratio4_phase_probe.py`](../../runtime/c4f_ratio4_phase_probe.py) | 单卡探针:`--mode profile / micro / kernel / ab` |
| [`runtime/run_c4f_probe.sh`](../../runtime/run_c4f_probe.sh) | 探针 launcher(titan064 单卡) |
| [`runtime/run_c4f_c2f_bench.sh`](../../runtime/run_c4f_c2f_bench.sh) | C2F 同口径吞吐 launcher(带 `DSV4_INDEXER_QAT`,`NCCL_P2P_LEVEL=SYS`) |
| [`runtime/run_c4f_long_gate.sh`](../../runtime/run_c4f_long_gate.sh) | D0L 长门 launcher(16 卡跨机) |
| [`results/`](results/) | 全部 JSON:分相 × 3、微基准 × 2、kernel 门、层级 A/B × 2、C2F 三轮、长门 |

### 开关

| 环境变量 / 参数 | 默认 | 说明 |
|---|---|---|
| `DSV4_INDEXER_QAT` / `indexer_qat_mode` | `ref` | `fused` = 杠杆 A。**已放行**,shipped launcher 里设 `fused`(与 `DSV4_PREFILL_SPARSE_BACKEND=tilelang` 同一先例:runtime 默认保守,launcher 打开) |
| `DSV4_COMPRESSOR_CAST` / `compressor_cast_mode` | `ref` | `hoist` = 杠杆 B。逐位相等但增益低于噪声,**不放行** |

### 收尾

两机 16 卡显存全部回到 **1 MiB / 24564 MiB**,无残留 compute 进程。

---

## 7. 对上层数字的影响

| 量 | 之前 | 现在 |
|---|---:|---:|
| prefill(C2F 同口径,单 stage = PP4 投影) | 25,307 | **28,622 input tok/s** |
| ratio-4 attention 桶 | 0.3611 s(27.8%) | **0.2138 s(18.6%)** |
| prefill 每卡峰值 | 20.409 GiB | **17.845 GiB** |
| 单池 T(D = 8,733) | 2,322 | **2,538** |
| 带下沿 3.2k 所需 P | 40,406 | 40,406(未变,现已走到 70.8%) |

新的 prefill 归因(instrumented total 1.151 s 为分母):**MoE 37.5%、
HC 27.2%、ratio-4 18.6%、ratio-128 13.8%、norm 2.8%**。
**MoE 重新成为头号桶**,其中 17.7 ms/层是 P2P 集合下界(C2F 已量化);
HC 升到第二(27.2%),而 prefill HC 融合的 +19.9% 仍卡在 vLLM kernel
≥1024 行数值错误上 —— 那条路一旦有正确实现,现在的价值比之前更高。
