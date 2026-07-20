# D0-reference-oracle：官方 reference 实现 golden-token oracle（sm89）

日期 2026-07-20。Phase 0 前置项完成件。

动机：为后续 direct runtime 的语义对拍建立独立 oracle——用官方 reference 实现（慢但数值对）在固定 prompt 集上生成 golden tokens。

方法与过程：
1. **convert**：reference `convert.py` 在 titan064 上把 `~/Workspace/DeepSeek-V4-Flash/`（46 分片原始 checkpoint）转成 MP=8 布局 → `~/Workspace/DeepSeek-V4-Flash-mp8/`（8×20.6 GiB + tokenizer，experts 保持 packed FP4），转换全程约 3 分钟（1TB RAM + page cache）。
2. **单机冒烟**：`torchrun --standalone --nproc-per-node 8 generate.py`，贪心 16 token，输出 "Paris" 正确。每卡显存 22,635 MiB / 24,564 MiB（权重 20.6 GiB + 运行时），余量约 1.4 GB——**MP=8 单机可跑成立**。
3. **坑（已记录）**：tilelang JIT 必须 `export CUDA_HOME=/usr/local/cuda-13.2` 并把其 bin/lib64 置于 PATH/LD_LIBRARY_PATH 前；否则用 venv 内 pip 的 `nvidia/cu13/bin/nvcc` 编译报 "CUDA compiler and CUDA toolkit headers are incompatible"（与 gaiban `*_titan.sh` 的做法一致）。
4. **oracle 冻结**：`oracle_generate.py`（fork 自 generate.py 的 batch 贪心路径，temperature=0/argmax，max_batch_size=prompt 数）对 `oracle_prompts.txt` 固定 8 条 prompt（中英文事实/代码/算术/翻译/列举/诗歌）生成 max 128 token，把 prompt/completion 的 token ID、解码文本与环境指纹（hostname、world_size=8、torch 2.11.0+cu130、config md5 aba1b3578f5554013ff20a422c81c9b7、checkpoint 文件清单）写入 JSON。

结论：
1. reference 实现在 sm89 单机 8×4090 上端到端跑通 V4-Flash（FP4 experts + tilelang kernels），输出质量正常（17×23=391 过程正确、素数列举正确、绝句成篇）。
2. golden artifact：`results/oracle-mp8.json`（8 条 prompt 全部记录 token ID）。后续 runtime 对拍以该 JSON 的 completion_tokens 为准；重新生成用本目录脚本（贪心确定性）。
3. 语义注意：oracle 是 batch=8 left-pad 联跑的 reference 语义；逐层 canary 对拍（D5 式）后续在 Phase 2 建。

Artifacts：`oracle_prompts.txt`、`oracle_generate.py`、`results/oracle-mp8.json`；titan064:`~/flash-oracle/`（运行现场）、`~/Workspace/DeepSeek-V4-Flash-mp8/`（mp8 checkpoint，不进 Git）。
