# A0-flash-checkpoint-contract:V4-Flash 原始分片张量契约核实

日期 2026-07-20。Phase 0 前置项(原"Flash 版 convert + Marlin repack"的落地形态)。

## 动机与路线修正

摸清 gaiban 的 checkpoint 链路后确认:Pro 的 `dsv4-checkpoint-stages/` 只是**整分片
rsync 分组**,没有离线张量变换;Marlin MXFP4 repack(`gptq_marlin_repack` +
`marlin_permute_scales` + `mxfp4_marlin_process_scales`)与 per-expert
intermediate-TP 切片都发生在**运行时加载侧**(`dsv4_direct/ops/marlin_moe.py::
load_resident_moe_layer`,gaiban)。两台 titan 已各有完整权重
(`~/Workspace/DeepSeek-V4-Flash/`),双机形态下无需 stage 目录分发——加载器带层过滤
直读完整 checkpoint 即可。因此 Phase 0 的离线工作收敛为:核实 Flash 原始分片契约,
供 Phase 1/2 适配 `checkpoint.py::validate_layer_contract/_config_contract`。

## 方法

`inspect_flash_contract.py` 在 titan064 上仅读 safetensors header(不触 payload),
枚举代表层(L0/L1 滑窗、L2 ratio-4+hash、L3 ratio-128、L4/L42 ratio-4、mtp.0)与
顶层张量的 key/dtype/shape。完整输出:`flash-contract.txt`。

## 契约要点(与 Flash 几何逐项吻合)

- **routed experts**(256/层):`w1/w3.weight` I8 `[2048, 2048]`(inter 2048 ×
  hidden 4096/2 nibble-packed)+ `scale` E8M0 `[2048, 128]`(4096/32);
  `w2.weight` I8 `[4096, 1024]` + `scale` `[4096, 64]`。即 Marlin repack 输入
  w13: K=4096/N=2048×2,w2: K=2048/N=4096,group_size=32 —— 与 gaiban 加载路径
  的输入约定同构,仅换 shape。
- **shared expert**:FP8 E4M3 + E8M0 128×128 block scale(`[16,32]`/`[32,16]`),
  与 Pro 完全同格式(A3c-v4 kernel 可直接换 shape)。
- **attention 投影**:全部 FP8 E4M3 + E8M0 block scale:`wq_a [1024,4096]`、
  `wq_b [32768,1024]`(64 heads × 512)、`wkv [512,4096]`、`wo_a [8192,4096]`
  (o_groups 8 × o_lora 1024)、`wo_b [4096,8192]`;E1b2q W8A16 路径可复用。
- **层型分布**:L0/L1 无 compressor/indexer(纯滑窗,compress_ratio 0);
  L0–L2 gate 用 `tid2eid` I64 `[129280,6]`(hash 路由,无 bias);L3+ gate 用
  `bias` F32 `[256]`(noaux_tc);indexer(`indexer.compressor.* / wq_b /
  weights_proj`)只在 ratio-4 层(L2,4,6,…,42);ratio-128 层(L3,5,…,41)只有
  compressor(`ape [128,512]`)。
- **HC**:每层 `hc_{attn,ffn}_{base [24], fn [24,16384], scale [3]}` F32
  (mix_hc=(2+4)×4=24),顶层与 MTP 各有 `hc_head_{base [4], fn [4,16384],
  scale [1]}`。
- **MTP**(mtp.0):自带完整 256-expert FFN、e_proj/h_proj `[4096,4096]` FP8、
  enorm/hnorm/norm、attn(无 compressor,滑窗)与 hc_head——体量≈一个完整层
  (~3.2 GiB experts),尾 stage 摆放需计入。
- 顶层:`embed/head [129280,4096]` BF16 分离权重;`norm [4096]`。

## 结论

1. Flash 原始分片与 gaiban 加载侧约定**同格式、仅换几何**;Phase 1/2 的适配点是
   `_config_contract` 常量表、`validate_layer_contract` 形状表与层型分支
   (滑窗层/无 indexer 层/hash gate 层),不需要任何离线 repack 工具。
2. mp8 reference 转换产物(oracle 用)已在 titan064
   `~/Workspace/DeepSeek-V4-Flash-mp8/`(8×20.6 GiB,含 tokenizer)。

Artifacts:`flash-contract.txt`(完整 header 契约)、`inspect_flash_contract.py`。
