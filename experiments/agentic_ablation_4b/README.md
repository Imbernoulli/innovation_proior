# Agentic 消融 @ Qwen3.5-4B（2026-08-22，gpublaze）

**问题**：innovation 语料里的 473 条 agentic 数据该不该留。
r3 定稿（07 月）把它剔了（「实测偏负」：空 think 稀释 → 训后 think 中位 27,071→1,782；后续 bloat
审计又落实 4×重复训练 / 428 条零-think boilerplate / think 错位）。但 8 月所有配方
（innnew / ctl / gated_v2 / `innovation_final_timeonly`）都是全量重建、**默认把 agentic 带回来了**
——包括 rlv12 的 soupNEW10 / loraIM 起点。两边都没有在最新语料上的受控证据；当年判负还是
40 题 shard 弱统计口径。本实验在本机（gpublaze，8×H100）用 Qwen3.5-4B 一次性裁决。

**设计**：两臂唯一差异 = agentic 473 行（按非空 `tools` 字段过滤，r3 同款筛法）。

| 臂 | innovation | maintain（共用） |
|---|---|---|
| `withag` | timeonly 重建 2,622 行（含 agentic） | wave2(750) + wave3(5,291) = 6,041 行，单遍 |
| `noag` | 同上剔 agentic = 2,149 行 | 同上 |

- innovation = `sft/build_sft.py` HEAD 重建（trajectories.json 六轨迹年份已修，0 条 year-None）
  + timeonly 变换（08-18 裁定，实现照抄 `training/FrontierSmith/scripts/build_training_final_innovation.py`：
  剥 persona 句 + delivery 条款，保留任务设定；338 行 v4 的竞程 system 按原实现保留）。
- maintain = **不做 repeat**（用户裁定 2026-08-22：回放是蒸馏数据稀缺时的权宜，campaign 收官后
  体量本身够了）。wave2 现行 hard-only 750 + wave3 final 5,291，按构建规则零重叠；rollout 行自带的
  题面 persona system 原样保留（两臂一致，不影响对照）。三个文件规格化为同一 schema
  （conversations/system/tools，缺省补 ""）防 pyarrow 混排地雷。
- 配方镜像 9B 线：full FT / ZeRO-3 / bf16 / lr 5e-6 / 1 epoch / cutoff 53760 / warmup 0.1 / wd 0.1 /
  eff. bsz 128；起点 = Qwen3.5-4B instruct（对齐 a100）。**2×H100 跑**（用户裁定），两臂串行。

**执行流水线（自主，用户裁定 2026-08-22）**：盯 GPU → 空出 2 卡立即开训 → 盯报错/效率 →
两臂跑完 → soup（average，α 先做 0.1/0.2）→ 本机 vLLM serve → 评测四件套：
**FrontierCS Algorithm（172 题）/ FrontierCS Research / ALE-40 / MLS CPU**，182 题全量口径不用
40 题 shard；同时记录 think 长度分布（判负机制的直接复验）。判读锚点：两臂差 < FCS 噪声底
1.0 ⇒ agentic 无关紧要；noag 显著高 ⇒ 复现判负；withag 高 ⇒ 翻案。
并行线：RL 环境（reward verification 栈）就绪后，另拨 2 卡从 Qwen3.5-4B 直接起 RL。

**产物**：`build_arms.py`（构建两臂）、`sft_withag.yaml` / `sft_noag.yaml`（训练配置）、
`launch_*.sh`（启动脚本）。数据 jsonl 不入库（可由 `python3 build_arms.py` 确定性重建）。
