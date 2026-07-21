# prefill MoE 双模的真因:NCCL 走了 SHM,不是分配器(2026-07-21)

第二十二竖条。第二十/二十一竖条记录了 prefill MoE 桶的**双模**(同一份代码、
同一脚本参数:快 0.483–0.551 s/pass ≈ 44–50 ms/层,慢 1.36–1.44 s/pass ≈
124–131 ms/层),线索是 `max_memory_reserved` 20.50 vs 23.98 GB,推测"分配器
被迫走 cudaMalloc/cudaFree 循环"。

**结论:分配器假说被证伪。** 双模是 **NCCL 传输选择**:GPU0–3 在
`nvidia-smi topo -m` 里是 **NODE** 距离(同 NUMA、跨 PCIe host bridge),
NCCL 默认 P2P 级别不覆盖该距离,于是 TP4 MoE 的两个集合通信退到
**SHM/direct(经主机内存中转)**,带宽 4.1 GB/s 而非 P2P 的 23.8 GB/s。
本 runtime **其余所有 launcher 都 export `NCCL_P2P_LEVEL=SYS`,唯独 4 个 C2F
launcher 漏了**——这就是快慢两支的分水岭。`max_memory_reserved` 那条线索是
**另一个未记录旋钮(`PYTORCH_CUDA_ALLOC_CONF` 有无)的巧合**,与快慢无因果。

修完后同口径实测:**torch 臂 17,022 / tilelang 臂 25,308 input tok/s/stage**,
三轮离散 0.26% / **0.03%**;第二十一竖条那个"折算 24.3k"**变成实测 25.3k**。

## 1. 直接证据

### 1.1 分配器计数器:全 0(证伪分配器假说)

`c2f_prefill_stage_bench.py --alloc-probe` 在**每次 MoE 调用**前后记录
`torch.cuda.memory_stats()`(产物 `c2f-alloc-probe-expandable.json`)。慢分支
2 个 pass × 11 层 = 22 次调用,**逐次**:

| 计数器 | 每次调用的增量 |
|---|---:|
| `num_alloc_retries` | **0** |
| `num_device_alloc` | **0** |
| `num_device_free` | **0** |
| `num_ooms` | **0** |
| `num_sync_all_streams` | **0** |

`reserved_bytes.all.current` 恒为 20.50 GiB(不动),
`torch.cuda.mem_get_info` 的驱动侧空闲恒为 2.37 GiB(不动)。
**没有任何 cudaMalloc/cudaFree 循环、没有 retry、没有 empty_cache 式回收。**
分配器在慢分支下完全静止——它不可能是 2.8× 的成因。

### 1.2 相位分解:80% 在两个集合通信

同一探针用 `stage_marker` 逐相位打点。三列分别是慢分支、只修 NCCL、再叠
combine 重写(§4.3),**同一个 L0 调用,同一台机**:

| 相位(均取各自 probe pass 0 的 L0) | 慢分支(SHM) | 只修 NCCL | + combine 重写 | |
|---|---:|---:|---:|---|
| `moe_inputs_ready` | 0.06 | 0.04 | 0.04 | |
| **`moe_hidden_all_gather_done`** | **52.62** | **8.92** | **8.74** | all_gather_into_tensor |
| `moe_ids_all_gather_done` | 0.14 | 0.10 | 0.10 | |
| `moe_route_done` | 3.15 | 3.15 | 3.14 | |
| `moe_routed_done` | 15.06 | 15.09 | 15.17 | marlin ×2 + topk sum |
| `moe_shared_done` | 3.57 | 3.58 | 3.61 | |
| **`moe_combine_done`** | 5.09 | 5.09 | **0.98** | §4.3 |
| **`moe_reduce_scatter_done`** | **47.34** | **8.78** | **9.06** | reduce_scatter_tensor |
| `moe_finalize_done` | 0.26 | 0.25 | 0.25 | |
| **L0 合计** | **127.4** | **45.2** | **41.2** | ms |
| **11 层均值** | **125.02** | **43.95** | **40.12** | ms/层 |

两个集合通信从 99.96 ms 降到 17.70 ms;**其余每一相位逐项不动**(≤0.03 ms 差)。
即双模之差 **100% 落在 NCCL 上**。第三列只有 `moe_combine_done` 再动
(5.09 → 0.98,**−4.11 ms/层**,与 §4.3 微基准预测的 −4.10 ms/层一致到 0.2%)。

### 1.3 NCCL 传输:`SHM/direct`,`isAllDirectP2p 0`

`c2f_moe_collective_probe.py`(4 rank,MoE 真实形状 [8192,4096]→[32768,4096]
bf16)+ `NCCL_DEBUG=INFO`:

(下表取 `tp_subgroup` 通信组,即 MoE 实际使用的那个;`world` 组同值。)

| 环境 | NCCL 判定 | 通道 | all_gather | reduce_scatter |
|---|---|---|---:|---:|
| 默认(C2F launcher 原样) | `isAllDirectP2p 0` | `0[0] -> 1[1] via **SHM/direct/direct**` | 48.84 ms / **4.12 GB/s** | 46.99 ms / **4.28 GB/s** |
| `LD_LIBRARY_PATH` 加 `~/libcuda-onebyte-patch` | `isAllDirectP2p 0` | 同上 SHM | 48.88 ms / 4.12 GB/s | 47.40 ms / 4.25 GB/s |
| **`NCCL_P2P_LEVEL=SYS`** | **`isAllDirectP2p 1`** | P2P | **8.46 ms / 23.79 GB/s** | **8.65 ms / 23.27 GB/s** |

**5.77× / 5.43×**。注意两点:

- `torch.cuda.can_device_access_peer` 在**三种情况下都是 True**(全 8 卡两两
  皆 True),`nvidia-smi topo -p2p r` 也全 OK——**这两个检查不具诊断力**,
  唯一有判别力的是 NCCL 自己的 `isAllDirectP2p` 与实测带宽。
- 单独把打过补丁的 libcuda 放进 `LD_LIBRARY_PATH` **无效**;起作用的是
  `NCCL_P2P_LEVEL`。

### 1.4 算术闭合

慢分支 81 ms/层的差额可完全由集合通信解释(相位取 §1.2 的 L0):
非通信相位合计 **27.33 ms** + P2P 通信 17.70 = **45.03 ms/层**,
×11 = 0.495 s ≈ 冻结快分支的 **0.483–0.485 s**(差 2%);
非通信 27.33 + SHM 通信 99.96 = **127.29 ms/层**,×11 = 1.400 s ≈ 冻结慢分支的
**1.362–1.378 s**(差 2%)。两支都对上,无残留。

## 2. `max_memory_reserved` 是红鲱鱼(反例实测)

第二十一竖条把 reserved 20.50 vs 23.98 GB 当作分配器线索。三个整机 A/B 直接
拆开这两个旋钮:

| 运行 | `PYTORCH_CUDA_ALLOC_CONF` | `NCCL_P2P_LEVEL` | tok/s | moe (s) | peak alloc | peak reserved |
|---|---|---|---:|---:|---:|---:|
| `c2f-alloc-probe-expandable.json` | `expandable_segments:True` | 未设 | 11,490 | 1.3739 | 20.41 GiB | **20.50 GiB** |
| `c2f-alloc-probe-default.json` | **未设** | 未设 | 11,438 | 1.3763 | 20.42 GiB | **22.33 GiB** |
| `c2f-alloc-p2pfix-expandable.json` | `expandable_segments:True` | **SYS** | **16,657** | **0.4822** | 20.41 GiB | **20.50 GiB** |

- 第 2 行有**快分支的显存指纹**(22.33 GiB,即 21 竖条记的 23.98 GB)却是**慢**的;
- 第 3 行有**慢分支的显存指纹**(20.50 GiB)却是**快**的。

即 reserved 的高低完全由 `PYTORCH_CUDA_ALLOC_CONF` 是否设置决定
(expandable segments 几乎不留余量 → reserved ≈ allocated + 95 MiB;
默认缓存分配器留 ~1.9 GiB 圆整/碎片余量),**与 MoE 快慢正交**。两个旋钮在
20/21 竖条的会话里恰好同向变化,才造出"相关"的假象。

## 3. 为什么会漏:4 个 C2F launcher 是仅有的例外

```
$ for f in runtime/*.sh; do grep -q ENV_BASE $f && ! grep -q NCCL_P2P_LEVEL $f && echo $f; done
runtime/run_c2f_prefill_titan.sh
runtime/run_c2f_tilelang_bench.sh
runtime/run_c2f_tilelang_gate.sh
runtime/run_c2f_tilelang_oracles.sh
runtime/run_e0mf_titan.sh
```

其余 launcher(全部双机 E2E / decode / MTP 系列)的 ENV_BASE 都带
`NCCL_P2P_LEVEL=SYS`。重归因那 4 轮"快"的运行(`out-c2f-v2-*`)**没有留下
launcher**(README 只记了产物目录),是手工 ssh 跑的——显然沿用了带
`NCCL_P2P_LEVEL=SYS` 的标准 ENV_BASE,于是同一份代码分裂成两支。

> `run_e0mf_titan.sh`(MTP block oracle,单机)同样漏了,**本竖条未改**:
> 它是正确性 oracle,其冻结产物是在无 P2P 环境下取得的,改环境需重跑它自己的
> 门。**作为发现记录在此,由其 owner 决定。**

## 4. 修法

### 4.1 环境层(真因,必须)

4 个 C2F launcher 的 ENV_BASE 补 `export NCCL_P2P_LEVEL=SYS`。

### 4.2 防复发(让它不能再静默)

bench 现在**无条件**记录 `nccl_p2p_level` / `pytorch_cuda_alloc_conf` /
`allocator_backend` / `nccl_version`(20、21 竖条的结果 JSON 里这些字段全无,
这正是当时无法归因的原因),并在 stage load 之前跑一次
**`moe_collective_selfcheck`**——按 MoE 真实形状实测两个集合通信,把
`all_gather_bus_gbps` / `reduce_scatter_bus_gbps` 写进每一份结果 JSON。
SHM 落在 ~4 GB/s、P2P 落在 ~23–24 GB/s,**一眼可判**。

### 4.3 代码层(与真因无关,但独立成立):combine 重写

`TP4MoE.__call__` 原来的合并式:

```python
buffers.combined.copy_((routed.float() + shared.float()).to(torch.bfloat16))
```

对两个 BF16 张量做加法,却物化 3 个 FP32 `[32768, 4096]` 临时量 + 1 个 BF16
临时量(chunk 8192 下 **1.61 GiB/次调用**)。改为:

```python
torch.add(routed, shared, out=buffers.combined)
```

**数值语义零变化,且是逐位的**:ATen 的 CUDA elementwise add 把 BF16 提升到
`opmath_t = float`,在 FP32 里加、存回时只舍入一次,与原式代数与舍入均同。
这不是抽样验证——BF16 只有 2¹⁶ 个值,`c2f_moe_combine_gate.py`
**穷举了全部 2³² 个有序 BF16 对**:

| 项 | 值 |
|---|---:|
| 检查对数 | **4,294,967,296**(= 2³²,全域) |
| 位不一致 | **0** |
| 有限输入下位不一致 | **0** |
| 真实形状 [32768,4096] 上逐位一致 | **true** |

真实形状上的代价:**4.967 ms → 0.869 ms(5.71×)**,瞬时分配
**1.61 GiB → 0**,即预测 **−4.10 ms/层**。整机相位实测
**5.09 → 0.98 ms = −4.11 ms/层**(§1.2 第三列),**预测命中到 0.2%**;
MoE 桶口径 0.4822 → 0.4374 s(−4.07 ms/层)。

> 这条正是重归因 README 里"prefill MoE 的 fp32 临时量应预分配复用"那条待办——
> 只是最优解不是"预分配复用",而是**根本不需要那些临时量**。它值 +2.5%
> 吞吐,不是那条待办估的 +32%(+32% 全部是 NCCL 的)。

## 5. 验收

### 5.1 正确性 — E2E golden(e0ef2e,16 rank 双机,eager HC)

两臂各跑一次,对照冻结基线:

| 臂 | 冻结 | 本竖条 | 逐 prompt(本竖条) | `accepted` |
|---|---:|---:|---|---|
| torch-prefill(基线形态) | 468/482 | **468/482** | 2, 27, 127, 124, 11, 22, 32, 123 | true |
| tilelang-prefill | 472/482 | **472/482** | 2, 29, 127, 124, 12, 22, 32, 124 | true |

**两臂逐 prompt 与冻结值完全一致**(torch 臂对 `2, 27, 127, 124, 11, 22, 32,
123`,tilelang 臂对 `2, 29, 127, 124, 12, 22, 32, 124`),连分歧的
`mismatch_top2_gap` 分布都对上(tilelang 臂 max 0.5621 / median 0.3254 /
min 0.0145,冻结 0.56206 / 0.32544 / 0.01446)。这与 §4.3 的穷举位级门一致:
combine 重写**不是"误差足够小"**,而是**逐位相同**,所以 E2E 只可能逐 token 相同。

不需要额外的单层数值门:数值语义未变(2³² 对穷举 0 位差),NCCL 传输切换也
不改变 ring reduce 的逐元素归约序,且两个冻结 golden 本来就是在
`NCCL_P2P_LEVEL=SYS` 的双机 launcher 下取得的。

### 5.2 吞吐 — C2F 同口径各 3 轮(chunk 8192,11 层 L11–21,iters 5/warmup 2,all-on)

| 臂 | 轮 | input tok/s/stage | moe (s) | attn 合计 (s) | total_instr | ag GB/s | rs GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| torch | m1 | 17,051 | 0.4356 | 1.1454 | 1.9276 | 23.69 | 22.46 |
| torch | m2 | 17,008 | 0.4380 | 1.1490 | 1.9334 | 23.93 | 22.50 |
| torch | m3 | 17,006 | 0.4401 | 1.1494 | 1.9359 | 23.80 | 23.14 |
| **torch 均值** | | **17,022** | **0.4379** | 1.1479 | 1.9323 | | |
| tilelang | m1 | 25,313 | 0.4322 | 0.5197 | 1.2984 | 23.61 | 22.35 |
| tilelang | m2 | 25,305 | 0.4349 | 0.5202 | 1.3017 | 23.64 | 22.91 |
| tilelang | m3 | 25,307 | 0.4340 | 0.5199 | 1.3005 | 23.54 | 22.15 |
| **tilelang 均值** | | **25,308** | **0.4337** | 0.5199 | 1.3002 | | |

**轮间稳定性(双模已消除)**:

| 臂 | tok/s 离散 | moe 桶离散 | attn 桶离散 | 对比:20/21 竖条的双模 |
|---|---:|---:|---:|---|
| torch | **0.26%** | **1.04%** | 0.35% | 0.483 vs 1.372 s = **2.84×** |
| tilelang | **0.03%** | **0.64%** | 0.08% | 同上 |

MoE 桶 6 轮全部落在 **0.4322–0.4401 s**(39.3–40.0 ms/层),再无第二支。

**对照冻结值**:

| 口径 | 冻结 | 本竖条实测 | |
|---|---:|---:|---|
| torch 臂(16.6k 基线) | 16,602 | **17,022** | +2.5%(= combine 重写) |
| tilelang 臂 | 24,268(**折算**) | **25,308**(**实测**) | 折算兑现,+4.3% |
| MoE 桶 | 0.4825–0.4849(快支) | **0.4322–0.4401**(6 轮) | −9.9%(= combine 重写) |
| attention 合计(torch) | 1.1468 | 1.1479 | +0.1%(未动,如期) |
| attention 合计(tilelang) | 0.5201 | 0.5199 | −0.0%(未动,如期) |

attention 分量与冻结值逐项吻合到 0.1%,说明本竖条的 A/B 干净——变的只有 MoE。

**单池投影**(`1/T = 1/D + 8/P`,D = 8733):
torch P=17,022 → T = **1,711**;tilelang P=25,308 → T = **2,322**(原折算 2,251)。

### 5.3 显存

| | 冻结(21 竖条 tilelang 臂) | 本竖条 6 轮 |
|---|---:|---:|
| peak allocated | 20.409 GiB | **20.409 GiB** |
| peak reserved | 20.502 GiB | **20.502 GiB** |

**高水位未变**,如实记录:被删掉的 1.61 GiB 是 MoE 调用内的**瞬时**分配,而
进程峰值并不由 MoE 相位决定——探针记录的 MoE 各相位边界 allocated 最高只有
**16.35 GiB**,峰值 20.41 GiB 由 stage load 与 attention 工作区撑起。所以这项
收益体现在**每次调用的分配抖动**(1.61 GiB → 0)与随之而来的 4.11 ms/层,
不体现在 peak 上。

## 6. 意外发现(原样记录)

1. **`can_device_access_peer` / `nvidia-smi topo -p2p r` 对本机不具诊断力**:
   两者在 SHM 回退时也全报 OK(8×4090 两两皆 True,含跨 socket)。唯一有判别力
   的是 NCCL 的 `isAllDirectP2p` 与实测带宽。凡是靠前者判断"P2P 已生效"的结论
   都需要重验。
2. **`~/libcuda-onebyte-patch` 单独放进 `LD_LIBRARY_PATH` 不改变任何东西**
   (实测仍 SHM / 4.13 GB/s)。它与系统 libcuda 确实只差 1 个字节
   (offset 4380316,`0x60` vs `0x40`),但决定 NCCL 走不走 P2P 的是
   `NCCL_P2P_LEVEL`。集群备忘里"P2P 补丁已生效"应理解为**驱动层可用**,
   不等于 **NCCL 会用**。
3. **`run_e0mf_titan.sh` 同样漏了 `NCCL_P2P_LEVEL`**(§3),未改,待其 owner 决定。
4. **20/21 竖条的结果 JSON 没有记录任何环境旋钮**,这是当时无法归因的直接原因;
   现已补齐(§4.2)。**建议:任何新 bench 的结果 JSON 都应自带环境指纹 +
   一个能区分快慢路径的自检量。**
5. MoE 桶修完后仍是 prefill 第一大项(tilelang 臂 0.434/1.300 = **33.4%**),
   其中 **17.7 ms/层(44%)是 NCCL 的 P2P 下限**,15.2 ms 是 marlin。
   下一步若继续压 MoE,应从 all-gather/reduce-scatter 的**重叠**(与 attention
   计算 overlap)而不是从分配器入手。

## 7. 产物

| 文件 | 内容 |
|---|---|
| `c2f-alloc-probe-expandable.json` | 慢分支 alloc 探针(计数器全 0 + 逐相位) |
| `c2f-alloc-probe-default.json` | 慢分支 + 默认分配器(快分支显存指纹,仍慢) |
| `c2f-alloc-p2pfix-expandable.json` | `NCCL_P2P_LEVEL=SYS` 后(16,657,慢分支显存指纹) |
| `c2f-alloc-final-p2p-and-combine.json` | 最终形态 alloc 探针(17,011;combine 相位 0.98 ms) |
| `moe-collective-default.json` | 集合通信探针,默认(SHM,4.1 GB/s) |
| `moe-collective-p2p-patch.json` | 加 patched libcuda(无变化) |
| `moe-collective-p2plevel-sys.json` | `NCCL_P2P_LEVEL=SYS`(P2P,23.8 GB/s) |
| `moe-combine-gate.json` | combine 重写穷举门(2³² 对,0 位差) |
| `c2f-chunk8192-{torch,tilelang}-m{1,2,3}.json` | 吞吐各 3 轮 |
| `e2e-moealloc-{torch,tilelang}-prefill.json` | E2E golden 两臂 |

代码:`runtime/dsv4_direct/moe_runtime.py`(combine)、
`runtime/c2f_prefill_stage_bench.py`(`--alloc-probe`、环境指纹、
`moe_collective_selfcheck`)、`runtime/c2f_moe_collective_probe.py`、
`runtime/c2f_moe_combine_gate.py`、launcher `run_c2f_moe_alloc.sh` +
4 个 C2F launcher 的 `NCCL_P2P_LEVEL=SYS`。

## 附:全仓 launcher 审计——decode 侧结论不受影响(主会话核实)

`NCCL_P2P_LEVEL=SYS` 的缺失是否波及既有 decode 数字?逐个 launcher 审计
(`runtime/*.sh`,23 个):

- **19 个显式设置**(含所有 decode/E2E 双机 launcher:`run_e1f_dual.sh`、
  `run_e1f_dp_dual.sh`、`run_e1if_dual.sh`、`run_e1if_kv_dual.sh`、
  `run_e1if_ws_dual.sh`、`run_e0qf_dual.sh`、`run_e0e2e*`、`run_e1mtp*` 等)。
- **2 个前沿扫描脚本**(`run_e1if_fp8_frontier.sh`、`run_e1if_ws_frontier.sh`)
  自身未设,但**均委托**给上面带 SYS 的 dual launcher(第 18 行分别调用
  `run_e1if_kv_dual.sh` / `run_e1if_ws_dual.sh`)→ **实际带 P2P**。
- **2 个 gate 脚本**(`run_ws_gates.sh`、`run_e0mf_titan.sh`)未设:gate 是
  同一次运行内两臂互比,传输选择只影响速度不影响判定;`run_e0mf_titan.sh`
  已被本竖条显式标记(其冻结产物即在无 P2P 下产生)。

**结论:decode 侧的全部头条数字(8K 前沿 6392→7523→8733、2K 8570/9656、
E1F B 扫描、MTP 与 DP 竖条)均在 P2P 生效下测得,不需重估**;本次 SHM 回退
的影响域仅限 C2F prefill 的 20/21 竖条测量,已在本目录修正。
