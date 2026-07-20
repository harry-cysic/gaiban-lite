I have a comprehensive picture. Here is the structured change-surface report.

---

# DSV4-Direct decode 前向:模块结构与 V4-Flash 移植改动面

所有路径相对 `/home/harry/gaiban/experiments/E0-direct-runtime/dsv4_direct/`,行号为该文件内行号。参考模型在 `/home/harry/gaiban/references/inference/model.py` + `references/config.json`。

**重要前提**:dsv4_direct 里**没有 tilelang / sparse_attn32**。稀疏 attention 是纯 torch 参考实现,通过 Protocol 注入;真正的 tilelang `sparse_attn32` 内核只出现在 E0 顶层实验脚本(`e1a26_sparse_attn32_bf16x2_tilelang.py`、`e1a6_tilelang_sparse_backend.py` 等),经 `Ratio128SparseAttentionBackend` / `Ratio4SparseAttentionBackend` 注入。

---

## 1. 单层 decode 前向:模块与依赖图

**入口链(自顶向下)**
- `physical_stage.py` — `TP4PhysicalDecodeStage`(:1061,继承 `TP4DecodeStage`),`build_physical_stage`(:1247)、`build_physical_layer_runtime`(:775)。装配一层的 attention/moe/kv/block providers。
- `superstage.py` — `TP4DecodeStage`(:108),多层 decode 循环:`forward_decode_tensors`(:918)→逐 block `block.forward_decode_tensor`(:949);stateful 版 `forward_stateful_decode_tensor_prevalidated`(:1326→:1386)。`TP4DecodeSuperStage`(:1482)。
- `block.py` — 单层前向核心:
  - `DirectDecodeBlock`(:704),`__init__`(:713)按 `compress_ratio` 选 attention 类型(:738-750),`forward_decode_tensor`(:1262)、`forward_stateful_decode_tensor`(:1365)。
  - 前向拆两半:`_attention_branch_decode`(:841)/`_attention_branch_stateful_decode`(:879)→ `prepare_ffn`(:1080)→ `prepare_stateful_decode_pre_moe`(:1146)/`finish_stateful_decode_from_pre_moe`(:1232)。
  - `Layer3DirectBlock`(:483,ratio-128 单元测试用)、`Layer2DirectBlock`(:1518,ratio-4)、`DirectPreMoEBlockFragment`(:1392,PP 切分)。
  - 调用图:`block → attention.{Ratio128,Ratio4}TorchAttention` + `moe.TP4MoE` + `hyper_connections.{hc_pre,hc_post}`(import :19)。
- `attention.py` — `Ratio128TorchAttention`(:738);`ratio4_attention.py` — `Ratio4TorchAttention`(:497)。
- `moe_runtime.py` — `TP4MoE`(:426),`__call__`(:1219)/`forward_tensor`(:1568)。

**static KV / stateful graph / physical stage 的文件与核心类**
- static KV:`static_kv.py` — `StaticLayerKV`(:36,ratio-128);`static_ratio4_kv.py` — `StaticRatio4KV`(ratio-4,含 indexer/overlap 状态)。
- stateful:`stateful_decode.py` — `DecodeGraphFamily`(:39,枚举 NORMAL/RATIO4_BOUNDARY/RATIO4_RATIO128_BOUNDARY)、`StatefulDecodeCursor`(:169)、`classify_decode_position`(:83)、`family_boundary_flags`(:94)。
- CUDA graph:`stateful_graph.py` — `capture_stateful_graph`(:382)、`replay_stateful_graph`(:471)、`teardown_stateful_graphs`(:570)。
- physical:`physical_stage.py` — `PhysicalLayerRuntime`(:621)、`TP4PhysicalDecodeStage`(:1061)、`PhysicalStageBuildRequest`(:136)。
- 权重:`block_weights.py` — `ResidentBlockWeights` / `ResidentAttentionWeights` / `ResidentGateWeights`,shape 断言集中于此。

---

## 2. Attention 前向

**稀疏 kernel 来源**:注入式。`attention.py:239` `Ratio128SparseAttentionBackend(Protocol)`,`ratio4_attention.py:285` `Ratio4SparseAttentionBackend(Protocol)`。默认实现是纯 torch:`torch_sparse_attention`(attention.py:572)、`_torch_sparse_decode_prevalidated`(:616)、`_torch_sparse_decode_padded_prevalidated`(:638)。tilelang `sparse_attn32` 由外部脚本注入,dsv4_direct 内部无硬编码。

**几何硬编码位置(文件:行号:常量)**
- `model_contract.py:16-23` — hidden_size 7168、num_attention_heads 128、head_dim 512、qk_rope_head_dim 64、q_lora_rank 1536、o_lora_rank 1024、o_groups 16、sliding_window 128。
- `attention.py:107-134` — `Ratio128AttentionConfig.validate` 逐字段断言(hidden 7168、num_heads 128、head_dim=LATENT_DIM 512、rope_dim 64、q_lora 1536、o_lora 1024、o_groups 16)。**这是 ratio-128 层的死墙,必须改。**
- `static_kv.py:20-22` — `WINDOW_SIZE=128`、`COMPRESS_RATIO=128`、`LATENT_DIM=512`(attention.py:24 import)。
- `window_topk_indices`(attention.py:510,window 128)、`compressed_topk_indices`(:547,ratio 128)。
- softmax scale `cfg.head_dim**-0.5`:attention.py:~1214/1348/1356(head_dim 512 不变,无需改)。

**compressor / indexer 前向 & ratio-4/128 分支**
- compressor 权重解包:`prepare_attention_weights`(attention.py:663,`compressor_ape/wkv/wgate/norm`)。ratio-128 compressor 无 indexer(:711 断言 `weights.indexer is not None → raise`)。
- ratio-4 分支在 `ratio4_attention.py`:`prepare_ratio4_attention_weights`(:414),含 `index_wq_b`、`index_weights_proj`、`index_compressor_*`(:484-489);`Ratio4TorchAttention._index_finalizer`(:901)、`_main_finalizer`(:884)、`_write_overlap`(:916)。indexer 常量 `index_n_heads 64`/`index_head_dim 128`/`index_topk 1024` 断言在 `ratio4_attention.py:132-134`。
- ratio 分支的 compressor 维度差异:`block_weights.py:363` `compressor_dim = head_dim * (2 if compress_ratio == 4 else 1)`;indexer compressor `2 * index_head_dim`(:409)。`block_weights.py:493` `if compress_ratio == 128:` 分支。
- `static_ratio4_kv.py:20-26` — WINDOW 128、COMPRESS_RATIO 4、LATENT_DIM 512、INDEX_DIM 128、INDEX_PROJECTED_DIM `2*INDEX_DIM`、OVERLAP_STATE_ROWS `2*COMPRESS_RATIO`。

**wo_a 的 o_groups 分组 einsum**:`attention.py`,三处(stateful / decode / prefill-`__call__`):
- :1224-1233(`forward_stateful_decode_tensor`)、:1373-1380(`forward_decode_tensor`)、:1553-1560(`__call__`)。模式均为 `wo_a.reshape(o_groups, o_rank, num_heads*head_dim//o_groups)` + `einsum("bsgd,grd->bsgr", grouped, wo_a)`。这三处用 `cfg.o_groups`,是**参数化的**,只要 config.validate 放行即可,无需改 einsum 本体。

---

## 3. moe_runtime.py

- **fused_marlin_moe 调用点**:import `_fused_marlin_moe` / `marlin_make_workspace_new` 在 `moe_runtime.py:460-464`(来自 vLLM),存 `self._fused`(:477);实际调用在 `__call__` 内 `self._fused(...)`(约 :1454,`__call__` 起 :1219 + 相对 236)。底层 marlin ABI 在 `ops/marlin_moe.py`(`ResidentMoEWeights`/`SharedExpertSlice`)。
- **gate 实现**:`moe_forward.py` — `gate_forward_with_boundary`(:55,sqrt-softplus 在 :83 `F.softplus(F.linear(...)).sqrt()`,bias 只影响选择、权重用 unbiased,route_scale 乘在 :90),`hash_gate_forward`(:118,score 同样 sqrtsoftplus :152,expert id 走 `tid2eid` 表 :153)。调用点:hash `moe_runtime.py:~1352`、learned `~1370`;in-graph 变体 `F.softplus(gate_logits).sqrt()` 在 `moe_runtime.py:1400`。
- **deterministic MoE align**:`deterministic_moe_align.py` — `deterministic_moe_align_block_size`(:58)、`allocate_deterministic_moe_alignment`(:35)、`max_padded_tokens`(:17)。作用:复现 vLLM `moe_align_block_size` 的固定容量、expert-major、升序 flat-index 布局,保证 Marlin 输入确定性;`moe_runtime.py:212-236` 选用注入 provider 或该默认实现。另见 `fixed_moe_align.py`(备用固定容量实现)。
- **shared expert FP8 前向**:`prepare_shared_bf16`(:201,把 FP8 block dequant 成 BF16)、`shared_bf16_partial`(:209,clamp_limit 10.0);构造于 `TP4MoE.__init__`(:476)。
- **几何常量硬编码(文件:行号)**:`moe_runtime.py:32-36`(`TP4MoEConfig`:intermediate 3072、experts 384、topk 6、route_scale 2.5、clamp 10.0),`validate` 断言在 :46-61(`intermediate==3072`、`experts==384`、`topk==6`、`route_scale==2.5 and clamp==10.0`)。route_scale 默认 2.5 也在 `moe_forward.py:61/106/124`。expert 数/几何权威定义在 `model_contract.py:25-31`(moe_intermediate_size 3072、n_routed_experts 384、num_experts_per_tok 6、routed_scaling_factor 2.5、swiglu_limit 10.0)。

---

## 4. HC(Hyper-Connections)

- 独立模块 `hyper_connections.py`(纯 PyTorch,不依赖 reference/serving runtime):`hc_split_sinkhorn`(:74)、`hc_pre`(:145)、`hc_post`(:220)、`layer_hc_parameter_names`(:33)。
- block 通过 `from .hyper_connections import hc_post, hc_pre`(block.py:19)使用;`DirectDecodeBlock._hc_pre`(:820)传 `hc_mult`/`sinkhorn_iters`/`hc_eps`。
- `hc_split_sinkhorn` **定义在本模块内**(:74),`hc_pre` 内部调用它(:196)。语义与 reference `model.py` 对齐(注释明示复刻 checkpoint 参考)。
- **维度无关**:`hc_mult`、`hidden_size` 全部从 `residual.shape` 推导(hc_pre :180 `batch, sequence, hc_mult, hidden_size = residual.shape`),`mix_features=(2+hc_mult)*hc_mult`。docstring 里写死 7168/28672/4 仅注释,非断言。→ **HC 换 Flash 几何无需改代码**,只要 checkpoint 权重 shape 自洽。

---

## 5. 单层前向正确性验证(oracle 对拍)

**oracle 参考实现(纯数学,自带 RoPE/compressor/稀疏 attention)**
- `attention_oracle.py` — ratio-128:`oracle_ratio128_attention_step`(:855)、`oracle_sparse_attention`(:386)、`yarn_rope_table`(:147)、`oracle_ratio128_compress`(:348)、`init_ratio128_oracle_state`(:696)。
- `ratio4_oracle.py` — ratio-4:`oracle_ratio4_attention_step`(:1135)、`oracle_ratio4_bf16_control_step`(:1157)、`oracle_hash_route`(:148)、`oracle_overlap_pool`(:280)。

**对拍测试文件(E0d/E0e/E0f,可复用模板)** — 均在 `experiments/E0-direct-runtime/`(上一级):
- `e0e_tp4_attention_oracle.py` — 拿 `Ratio128TorchAttention` 逐步输出对 `oracle_ratio128_attention_step`。关键函数:`run_case`(:353,注入 deterministic hidden)、`compare_phase`(:252)、`tensor_metric`(:177,指标 rms_rel / row_rms_rel / max_abs)。输入=`deterministic_hidden`(:163)。
- `e0f_tp4_layer2_ratio4_semantic.py` — ratio-4 + hash gate 对 `oracle_ratio4_attention_step`/`oracle_ratio4_bf16_control_step`。容差:`HASH_RMS_REL_LIMIT=0.00002`(:196)、`HASH_ROW_RMS_REL_LIMIT=0.00008`(:197)、row_sum ≤ 1e-5(:1340)。`compare_attention_step`(:1084)、`run_hash_gate`(:1268)。provenance 指向 `references/inference/model.py` + `config.json`(:578-583)。
- `e0d_tp4_layer3_block.py` — 整层 `Layer3DirectBlock`(实例化 :685)端到端。容差 `NUMERIC_SIGNATURE_ATOL=2e-6`(:44),`numeric_signature_max_abs_delta`(:143)判定 `delta <= NUMERIC_SIGNATURE_ATOL`(:875)。
- 单元测试:`test_attention_oracle.py`、`test_model_contract.py`、`test_deterministic_moe_align.py`、`test_physical_stage.py`。

**对拍模式**:oracle 是 in-repo 纯数学参考(而非直接 import reference model.py 前向);reference `model.py`/`config.json` 作为 provenance/几何真值来源。容差是相对 RMS(2e-5~8e-5)+ 绝对 max_abs(2e-6)。移植 Flash 时复用 `e0e/e0f/e0d` 三件套即可,只需替换 oracle 里的几何维度推导函数(`attention_oracle.py:_oracle_dimensions` :599、`ratio4_oracle.py:_dimensions` :334)。

---

## 6. 换 Flash 几何:必改 vs 可直接复用

### 必须改(文件:行号 清单)

**几何真值 / 断言**
- `model_contract.py:13-42` `EXPECTED_LAYER3_CONFIG`:hidden 7168→4096(:16)、heads 128→64(:17)、o_groups 16→8(:22)、q_lora_rank 1536→1024(:20)、num_hidden_layers 61→44(:42)。head_dim 512(:18)、sliding_window 128(:23)、o_lora_rank 1024(:21)不变。
- `model_contract.py:45-51` `EXPECTED_LAYER2_CONFIG`:index_topk 1024→512(:51);index_n_heads 64(:49)、index_head_dim 128(:50)按 Flash 规格确认。
- `model_contract.py:67` `MODEL_LAYER_COUNT=61→44`;`:69-71` `FROZEN_COMPRESS_RATIOS`(L0/L1 目前是 128,Flash 要改成"纯滑窗层"新类型——见下);`:73-79` `SUPPORTED_LAYER_SPECS`(route_kind hash if <3 保留)。
- `attention.py:107-134` `Ratio128AttentionConfig.validate` 全部期望值(7168/128/512/64/1536/1024/16)。
- `ratio4_attention.py:132-134` index_n_heads/index_head_dim/index_topk 期望值(64/128/1024→按 Flash)。
- `block_weights.py:334-352,363,402-427,493` linear out/in features、`compressor_dim`、indexer shape、`compress_ratio==128` 分支。
- `moe_runtime.py:32-36,46-61` `TP4MoEConfig` 与 validate(experts 384、intermediate 3072、topk 6、route_scale 2.5、clamp 10.0——Flash 若改专家数/几何需同步)。
- `moe_forward.py:61/106/124` route_scale 默认。

**层型分支(新增 L0/L1 纯滑窗层型)** — 现状 L0/L1 是 ratio-128,Flash 要求"纯滑窗"新层型:
- `model_contract.py:69-71,73-79` — FROZEN_COMPRESS_RATIOS / SUPPORTED_LAYER_SPECS 需新增 window-only 层类别。
- `block.py:738-750` — `DirectDecodeBlock.__init__` attention 类型分派(现仅 `compress_ratio==4 ? Ratio4 : Ratio128`),需加纯滑窗分支。
- `block.py:1094-1129,841-943` — attention branch 分派与 plan 类型校验。
- `attention.py:27-31` `SUPPORTED_RATIO128_LAYER_IDS`(从 spec 派生,L0/L1 若不再是 ratio-128 会连锁影响 `Ratio128AttentionConfig`/`prepare_attention_weights` 的 layer_id 白名单:attention.py:63/105/674/761)。
- `stateful_decode.py:39-107` — `DecodeGraphFamily` 边界分类(ratio4/ratio128 boundary)需新增滑窗层边界处理。

**KV 布局常量**
- `static_kv.py:20-22`、`static_ratio4_kv.py:20-26`(window/ratio/latent/index dim;head_dim=512、window=128 不变,index 相关随 topk 512 调整)。

**硬编码 TP=4 假设**(若 Flash 改并行度):`attention.py:668` `world_size != 4`、`moe_runtime.py` `TP4MoEConfig.world_size`、`superstage.py`/`physical_stage.py` TP4 命名与断言。Flash 若仍 TP4 则不动。

### 无需改动可直接复用

- `hyper_connections.py`(全模块,维度无关;`hc_mult` 从 shape 推导)。
- `deterministic_moe_align.py` / `fixed_moe_align.py`(与 experts/block_size 参数化,几何无关)。
- `moe_forward.py` 的 `gate_forward_with_boundary` / `hash_gate_forward` **算法本体**(:55/:118,sqrtsoftplus+bias+route_scale 逻辑参数化,只随 config 默认值动)。
- `attention.py` 的 wo_a o_groups einsum(:1224/1373/1553)、`torch_sparse_attention`(:572)、`window_topk_indices`/`compressed_topk_indices`(:510/:547)——均以 `cfg.*` 参数化,放行 config 后自适应。
- 稀疏 backend Protocol(`attention.py:239`、`ratio4_attention.py:285`)——注入点不变,替换注入的 kernel 即可。
- `stateful_graph.py`(capture/replay/teardown,与几何无关)。
- oracle 对拍框架 `e0d/e0e/e0f` + `attention_oracle.py`/`ratio4_oracle.py`(仅需改各自 `_dimensions`/`_oracle_dimensions` 维度推导:`attention_oracle.py:599`、`ratio4_oracle.py:334`,及 config 常量)。
- `checkpoint.py`、`fp8_linear.py`(FP8 block dequant,块 128×128 与几何解耦)、`moe_forward.py:dequant_fp8_block/dequant_mxfp4`(:207/:175)。