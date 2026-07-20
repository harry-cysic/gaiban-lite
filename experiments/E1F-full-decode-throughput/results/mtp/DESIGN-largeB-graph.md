# MTP verify-step 大 B / CUDA-graph 化可行性说明(设计,不在本竖条实现)

日期 2026-07-21。背景:本竖条(第十六竖条)在 B=1 eager 全位置路径上实现并验收了
MTP draft-verify(见 `README.md`);本文回答"批内逐序列接受/拒绝 + stateful graph
框架(E1F/E1IF)如何容纳 2-token verify step"的设计问题。

## 1. 问题:接受/拒绝不同步破坏 graph 均匀性

E1F/E1IF 的 stateful graph 按"每步每序列恰好前进 1 个位置"捕获(三个
DecodeGraphFamily 由共享 device cursor 驱动)。MTP 下每轮每序列前进 1 或 2 个
位置(拒绝/接受),B>1 时批内位置立即失步:

- window ring 写 slot、compressor phase、ratio-4 boundary family、compressed
  行可见数全部变成 per-sequence 量,现有"共享单 cursor + 全批同 family"的
  capture 前提失效;
- MoE 的 global-row 形状仍均匀(每轮恒为 2 token/序列),不是阻塞点;
- 阻塞点集中在 attention 状态机(cursor/family/topk 几何)。

## 2. 推荐设计:定长 2-token verify graph + per-sequence 位置向量

核心决策:**graph 形状永远均匀 —— 每轮每序列固定处理 2 个输入 token
(pending, draft),接受/拒绝只改变"提交语义",不改变计算形状。**

1. **定长 verify graph**:捕获一个 seqlen-2 的 decode graph(本竖条的
   `dsv4_direct/verify2.py` 已给出 eager 语义参照:hidden 侧 GEMM 两 token 融合,
   cache 写/稀疏核逐位置)。每轮 replay 一次,无论上一轮各序列接受与否。
2. **per-sequence position 向量**:把共享标量 cursor 换成 `positions[B]`
   (int64, device)。RoPE 频率、ring slot、compressor phase、compressed 行数
   均以 gather/remainder 从 `positions` 逐序列导出 —— window 层已经是
   `position.remainder(128)` 的封闭形式,推广到向量即可;ratio-128/4 的
   boundary 分支改为**掩码执行**(两条路径都算,按 `phase == ratio-1` 掩码
   提交),消除 per-family graph 分裂。掩码化的代价是每步恒做一次 compressor
   pooling(与现 boundary family 的代价相同量级,B=1 下 <1% step)。
3. **拒绝回滚 = 掩码化影子提交**:本竖条 B=1 的 post-first-token snapshot/restore
   不能进 graph(host 端 clone/copy)。graph 版把第二 token 的状态提交改为
   **写影子缓冲**(ring 行、compressor state、compressed 行各留一个 per-sequence
   影子槽),下一轮 replay 开头用 `accept_mask[B]`(上一轮 verify 的 argmax 比较,
   device 端算出,不回 host)选择"影子提交"或"丢弃"。所有分支均为固定形状的
   `torch.where`/`index_copy`,graph 友好。
4. **token 供给**:verify 输入 `[pending, draft]` 由上一轮 graph 内的
   argmax/gather 直接产出(head 已在尾 stage;draft 来自 MTP block 的
   argmax),配合 E1F 已有的 closed-loop 回环即可全程不回 host。
5. **MTP block 摆放**:尾 stage(titan065 s1,与 head/embed 同处,D6.4 预留);
   MTP 自身是 window 层型,已有 WindowStatefulDecodePlan 可直接复用
   (`SUPPORTED_WINDOW_LAYER_IDS` 本竖条已含 layer 43),每轮 1-2 步:
   graph 化时同样定长 2 步 + 掩码提交第二步。

## 3. 吞吐语义(大 B 口径)

- 每轮每序列产出 `1 + accepted` 个 token,批内независимо;轮时间 ~= 恒定
  (定长 graph),aggregate tok/s = `B * (1 + α_mean) / t_round`。
- t_round 相对单 token 步的增量:B=1 时 hidden 侧 GEMM 权重读主导,2-row GEMM
  与 1-row 同成本(本竖条 E1MTPF 实测 verify2/step 比值见 README);大 B 时
  GEMM 行数翻倍开始收敛为 ~2x 计算,但 4090 decode 在 bl<=192 区间仍强权重带宽
  主导(E1F replay 次线性),预计 t_round/t_step 在 bl=128 时 ~1.2-1.4,
  aggregate 收益 (1+α)/(t_round/t_step) 仍 >1.2x;确切值须实测。
- 序列间位置失步 → ctx 长度失步:KV 容量按 max-position 预留,回收(重排/
  compaction)与 serving 侧 batching 策略是后续 serving 竖条的事。

## 4. 不采用的替代方案

- **拒绝即重放整轮(全量 restore + refeed)**:每拒绝多付一整轮,α=0.66 时
  实效收益从 ~1.5x 掉到 ~1.2x,且仍需 per-sequence 失步处理,无简化收益。
- **批内同步接受(全批一致才接受)**:B=8 时 α_batch = α^8 ≈ 0.036,收益消失。
- **chained 2x 单 token graph**:无 GEMM 融合,B=1 下轮时 ~2x 步时,负收益
  (本竖条实测 chained round ≈ 2x 单步,见 README mtp_free 臂)。

## 5. 与本竖条产物的衔接

- `dsv4_direct/verify2.py` 的三种层型 seqlen-2 语义(GEMM 融合边界、逐位置
  状态提交顺序、post-first-token 快照点)就是 graph 版的语义 oracle;
  E0mtp2e `mtp_free`(chained,逐位与 off 相同)与 `mtp_fused`(融合)两臂
  给出 token 级等价性证据链。
- 掩码化 boundary 的正确性 gate 可复用 E0sf 的 graph-vs-eager 逐位框架,
  加 per-sequence 失步位置的 family 交叉用例。
