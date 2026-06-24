# Innovation-Prior 实验文档（中文）

> 本文档汇总「用 Innovation-Prior SFT 数据微调模型 → 看能力（尤其是发现/创新）是否提升，RL 是否进一步放大」这一整套实验：训练流程、模型矩阵、**每个评测到底是什么 / 怎么评的**、以及完整结果。
> 深入的行为级分析（模型到底学到了什么、为什么变好/变坏）见 **[CASE_STUDY_zh.md](CASE_STUDY_zh.md)**。
> 最后更新：2026-06。所有路径均相对于 `/scratch/gpfs/CHIJ/bohan/fs`。

---

## 0. 一句话结论（先给）

- **核心假设（创新-SFT 提升能力、RL 进一步放大）在竞赛类评测上没有得到支持**：纯 SFT 在 FrontierCS / ALE 上把模型从起点打崩（instruct 起点 3.14 → 纯 SFT 0.015）。
- **但这很可能是「评测口径错配」**：在 **MLS-Bench（ML 研究类任务）** 上，被 FCS 打崩的同一个 method-SFT 模型反而**超过起点**（instruct 起点 0.0643 → method-SFT 0.0794）。
- **model-soup（小比例权重平均）能稳健地把崩掉的能力「恢复」回起点水平**；但它偶尔「超过起点」的现象，逐样本核对下来基本是**单跑噪声**，不是真正的能力放大。
- **发现类评测（ThetaEvolve / TTT）噪声主导、不可判别**：~12–40 次迭代、单 seed、大量 cell 卡在 seed floor，模型间差异被 seed 间方差淹没。
- 模型从数据里学到的是**「创新的叙事姿态（register）」而非「可提交的解」**——详见 case study。

---

## 1. 目标与假设

总目标：**证明在 Innovation-Prior SFT 数据上微调 Instruct/Base 模型能提升能力（尤其是 discovery/innovation），且 RL 能进一步放大**，在以下评测上衡量：

| 评测 | 简称 | 类型 |
|---|---|---|
| FrontierCS | FCS | 竞赛算法编程 |
| ALE-Bench | ALE | 启发式优化编程（AtCoder Heuristic 风格） |
| ThetaEvolve（circle-packing 等） | Theta | 进化式数学发现 |
| TTT 同族自相关不等式（AC3） | TTT | 数学发现 |
| MLS-Bench | MLS | ML 研究/工程任务 |

---

## 2. 数据：Innovation Prior

数据来自本仓库本身（`methods/`、`trajectories/`、`sft/`）。构造哲学（README 原文）：

> "Finished papers are written theorem-first... **Innovation Prior** runs that backwards. For each paper, the method is rewritten as a first-person *discovery* trace — re-derived from the pain points and prior art of its time, hitting walls and self-correcting until it lands on the idea and the code."

即**每条样本 = 把一篇已发表论文从第一性原理「重新发现」一遍**：从当时的痛点和现有技术出发，撞墙、自我修正，最终落到方法与代码。每个 method 产出三件 markdown：`context.md`（背景/痛点）、`reasoning.md`（第一人称长推导，主体）、`answer.md`（提炼总结 + 参考代码）。

### 用于 SFT 的数据切分（LLaMA-Factory ShareGPT 格式）

| 数据集 | 文件 | 条数 | 说明 |
|---|---|---|---|
| `innovation_full` | full innovation SFT | — | 全量（含 agentic 轨迹） |
| `innovation_method` | `LF-innov/data/innovation_method_u.jsonl` | **1201** | **只取 method 的长推理子集**（`<think>` 中位 ~5621 token，无空 think） |
| `innovation_method_traj` | `LF-innov/data/innovation_method_traj_u.jsonl` | **1879** | method(1201) + trajectory(678)，剔除了 473 条 agentic（按非空 `tools` 字段过滤） |

> **为什么要 method-only**：全量数据里 agentic 样本会用「空 think」稀释思考，导致训练后思考坍缩。只取 method 长推理子集后，思考长度恢复（method-SFT 思考中位 ~27071 vs 全量 ~1782）。

---

## 3. 训练流程

### 3.1 SFT（LLaMA-Factory innovation fork）

- Fork：`Imbernoulli/LLaMA-Factory @ feat/per-turn-loss-mask`（本地 `LF-innov/`）。
- **Reasoning folding / per-turn loss mask**：折叠的历史轮 think 置空且 `loss=False`，当前轮完整且 `loss=True`；审计确认 99.97% 的 `<think>` 内容 token 被训练。
- 全参微调；Qwen3.5-9B 全 FT 时 LLaMA-Factory 会丢掉 MTP head（`mtp.*` 共 15 个 key）——因此 soup 合并脚本必须容忍 key 集合不对称（见 `cc_model_soup_merge.py`）。

### 3.2 Model-soup（权重平均合并）

合并：`merged = α·SFT + (1−α)·START`，其中 α = SFT 占比。
- 命名：`soup_<family>_<ratio>_<data>_soupaKK`，KK ∈ {10,20,30,50,70}（即 α=0.1…0.7）。
- 脚本 `FrontierSmith/scripts/cc_model_soup_merge.py`：逐 SFT key 混合共享 key，复制 SFT-only key，丢弃 base-only key（容忍 MTP 不对称）。

### 3.3 RL（verl GRPO，thinking 口径）

- verl GRPO，混合 train 集（FrontierCS+FrontierSmith+ALE-Bench parquet），FrontierCS+ALE-Bench 作 val。
- **per-task 奖励归一化**：`FS_PERTASK_REWARD_NORM=1`，hook 在 `experimental/reward_loop/reward_manager/naive.py`。
- 评测口径与 SFT 一致（thinking on）。
- 已知坑（已修，详见仓库内 `experiments/code` 和 memory）：go-judge cgroup OOM（`DBUS_SESSION_BUS_ADDRESS=/dev/null`）、判题端口冲突、base 无 chat template、image-processor 缺失、以及**把 NPU 风格的全量 CPU offload 误抄到 4×H200 导致每步慢到 ~22 分钟**（应关掉 actor 的 param/optimizer offload）。

---

## 4. 模型矩阵

两个家族 × 两类起点 × 数据 × soup：

- **家族**：`q3` = Qwen3-8B；`q35` = Qwen3.5-9B。
- **起点（aNN）**：`a00` = 纯 base，`a100` = 纯 instruct（早期还做过 a20/a50/a80 的 instruct/base 加权平均，后续按用户要求只保留 base/instruct 两端）。
- **数据**：`method`、`methodtraj`（早期还有 `innovonly`/`innovmaint`）。
- **soup**：soup10/20/30/50/70。

每个模型在 FCS+ALE+Theta+TTT 上统一评测；选定子集再上 MLS-Bench 与 RL。

---

## 5. 评测方法（每个评测是什么、怎么评的）

### 5.1 FrontierCS（FCS）— 竞赛算法编程

- 模型在 thinking 模式下解竞赛题，产出 C++ 代码；官方评测器抽取最长 ```cpp 围栏块、编译、跑测试点，给连续部分分。
- 口径：`metrics.frontiercs.score`，报 **mean@5**（每题 5 样本取平均再对题平均）与 **best@5**（每题取 5 样本最优再平均）。
- 评分链条（关键）：闭合 `</think>` → 抽最长 ```cpp 块 → 编译通过 → 过测试点。`strip_think` 是 `rpartition("</think>")`，**没闭合 think 就原样返回全文**，于是抽不到完整代码块 → 0 分。
- **shard_0 覆盖警告**：我们手头的 `cc_eval_*` shard_0 只覆盖 **40 题**（完整 benchmark ~182 题），其中 33 题对所有模型都是 0 分，**真正承载信号的只有 ~7 题**。任何「A>B」都由 5–7 题 × 5 样本撑起，统计支撑极弱。
- **Algorithm vs Research 划分（重要）**：Frontier-CS 官方分两个 track —— **Algorithmic（172 题，写 C++，go-judge 评）** 和 **Research（68 题，写 Python `Solution.solve()`，各题官方 `evaluator.py` 评，0–100，测加速比/精度）**。**我们之前所有评测只跑了 Algorithmic 这 172 题**（`prepare_frontiercs_parquet.py` 只指向 `algorithmic/problems`）。Research 68 题 = 64 标准 + 4 个 poc_generation（需 Docker-in-Docker，跑不了）；64 题里 21 题需 GPU（Triton kernel）、43 题 CPU。Research 现已接进评测（本集群无 Docker，改为直接在 GPU 节点跑 `evaluator.py`），metrics 单独键 `metrics.frontiercs_research.score`，**与 frontiercs/alebench 分开，不硬平均**；先跑 21 题 GPU 子集，43 题 CPU 需逐题装依赖（待办）。脚本：`scripts/frontiercs_research_eval.py`、`slurm/cc_eval_research_ailab.sh`。

### 5.2 ALE-Bench（ALE）— 启发式优化编程

- AtCoder Heuristic 风格的优化题，连续打分。
- **关键口径**：`metrics.alebench.score`（=performance）有一个 ~310 的**失败地板**——当提交编译不了 / 不可评分时仍会落在 ~310。**真实信号是 `overall_absolute_score`（绝对分）**：abs=0 ⇒ 提交是坏的（"BROKEN"），310.5 那个"分"其实是失败地板。
- 本文表格里 ALE 标 `!` 表示 abs≤0（提交坏掉），此时 score 不可信。

### 5.3 ThetaEvolve（Theta）— 进化式数学发现

**它是什么**：DeepMind AlphaEvolve 的开源简化+扩展复现——**LLM 驱动的进化式程序搜索**。给一个带 `# EVOLVE-BLOCK` 可编辑区的种子程序，用 MAP-Elites + island 模型维护种群；每次迭代采样一个父程序、让 LLM 产出 diff（SEARCH/REPLACE）生成子程序、执行打分、选择。代码：`ThetaEvolve/openevolve_adapted/`。

**原 benchmark 的任务集**（`openevolve_adapted/examples/`）：

| 任务目录 | 目标 | 方向 | seed/baseline |
|---|---|---|---|
| **circle_packing_modular** | n=26 圆装进单位正方形，**最大化半径之和** | max | seed ≈ **0.9598**，AlphaEvolve 目标 2.635 |
| first_autocorr_inequality | 一阶自相关不等式常数 | — | 无本地 vLLM smoke 配置，**未跑** |
| second_autocorr_inequality | C₂ 下界 R(f)=‖f∗f‖₂²/(‖f∗f‖₁·‖f∗f‖∞) | max | target 0.96 |
| third_autocorr_inequality | C₃ 自相关常数 | min | target/SOTA 1.4557，seed ≈3.159 |
| hadamard_matrix | ±1 矩阵 |det|/理论上界 | max | — |

**我们实际跑了什么**：主体是 **circle_packing_modular（仅 n=26）**（122 个 run 目录）；少量 hadamard；自相关任务通过 TTT 包装跑（见 5.4）。harness：`FrontierSmith/slurm/cc_eval_theta_openevolve_ailab.sh`（本地 vLLM 起模型，跑 OpenEvolve 进化循环，读最优分）。

**指标 `best_combined_score`**（聚合器读这个）：对 circle-packing **就是最优合法 26 圆的「半径之和」**（不是比值、不是最小间距）；非法装箱记 0。

**seed floor 0.96 是什么**（必读）：0.9597642169962064 是**未进化的种子程序本身**的分。当 summary 里 `best_combined_score == 0.95976…` 且 `best_is_seed: true`，意味着**模型从未产出一个打败种子的合法 diff**，报的"最优"就是种子，**与模型无关、不能读成发现信号**。好的分大致在 2.0–2.64（趋近 AlphaEvolve 目标 2.635）；我们 122 个 circle run 最大只到 **2.29**。

**预算与噪声**：默认 **~12 次迭代**、**单 seed（3407）**。122 个 circle run 里 **31 个（25%）恰好卡在 seed floor**；多 seed 噪声探针（`noise_theta_*`）显示同一配置在「卡 seed」和「~1.3」之间纯靠 seed 摆动。→ **Theta circle 列基本是单跑、噪声主导、大量 cell 卡地板**。

### 5.4 TTT（AC3）— 数学发现（**重要纠正**）

**「TTT」在我们的 run 名里只是个标签，我们并没有跑 TTT-Discover 的原生方法。** TTT-Discover 的原生 RL loop 硬绑 `gpt-oss-20b/120b` + Tinker 云 API（要联网），离线集群 + 本地 Qwen3.5-9B 跑不了（`discovery.py:74` 的 assert）。而 `third_autocorr_inequality`（AC3）**在 TTT-Discover 仓库里根本不存在**——它是 ThetaEvolve 的任务。

**我们实际做的**：因为 AC3 与 TTT-Discover 的 AC 族（AC1/AC2)数学同形（同一卷积打分），我们用 **ThetaEvolve 的 OpenEvolve 引擎离线跑了 AC3**，输出重新打上 `ttt_` 标签（`cc_eval_ttt_discover_openevolve_ailab.sh` 直接 delegate 到 theta harness）。

**AC3 是什么数学问题**：构造 [-1/4,1/4] 上的非负阶梯函数 f，**最小化 C3 = 2n·max|conv(f,f)| / (Σf)²**（Host–Vinuesa 型极值自相关问题）。方向 minimize，SOTA 目标 C3=1.4557，种子 ≈3.159。

**指标**：`best_combined_score = 1/(C3+EPS)`（越大越好）；**seed floor = 0.3166**（C3≈3.159 的种子）。好的值应趋近 1/1.4557≈0.687；我们最高只到 **~0.594**（没人到 SOTA）。

**为什么 TTT 列噪声主导**：预算极小（20–40 迭代、单 seed）；113 个 AC3 run 里 **14 个恰好返回种子**（`best_is_seed`）；其余落在 0.32–0.59 的宽随机带，由「这 20–40 次里哪个随机 diff 碰巧有用」决定，而非模型能力。seed 间方差淹没模型间差异 → **整列不可判别**。

> 参考 TTT-Discover 原 benchmark 的真实任务（我们都**没跑**）：AC1、AC2、Erdős 最小重叠常数 C₅、circle packing、GPU TriMul kernel、AtCoder AHC、单细胞 RNA 去噪。

### 5.5 MLS-Bench（MLS）— ML 研究/工程任务

- 20 个 CPU 任务（causal-*、ml-*、optimization-*、mlsys-* 等），在 apptainer 容器内本地跑，模型由 vLLM serve；`mlsbench agent` 解题 + `mlsbench score` 读回归一化 [0,1] 分。
- harness：`FrontierSmith/scripts/mlsbench_run_cpu_tasks.py` + `FrontierSmith/slurm/cc_eval_mlsbench_cpu_ailab.sh`。报 20 任务**均分**。
- **修过的 bug**：`mlsbench score` 把 JSON 打到 stdout、把 UserWarning 打到 stderr；早期 `stderr=STDOUT` 合并导致 json 解析失败、静默丢分（起点模型均值从误报的 0.0011 修正到 0.0226，~21×）。
- 默认 `CONCURRENCY=20`（20 任务并行一波，墙钟由最慢任务界定）。
- **零分诊断（为什么 13/20 任务常 0）**：三类混合根因，**不是单一"模型差"**。(i) **causal-\* 簇 = harness 的 prompt 路径前缀 bug**：EDITABLE 头给的是带包前缀的正确路径（`causal-learn/bench/...`），但任务正文和 diff 头把前缀剥掉了（`bench/...`），模型跟了正文 → 编辑被包白名单拒（`Package 'bench' is not in allowed packages`）→ 一次没编辑成功 → 只跑 baseline stub → 0。这是 MLS-Bench 任务作者侧 bug，可修。(ii) **ml-\*/optimization-\* 簇 = 模型自身代码坏**（SyntaxError/IndentationError）或越界编辑空转——路径对时 harness 照常放行。(iii) **基础设施崩**（容器挂载失败、`/data/adbench/*.npz` 缺失）。测试次数 `max_tests=3` 已生效（日志实锤 `You have used 1/3 tests`），`submit(n)` 可挑任意已跑 test，非"只能一次"。

---

## 6. 结果汇总

### 6.1 method / methodtraj 矩阵（FCS mean/best · ALE score+真实abs · Theta · TTT）

> ALE 的 `!` = abs≤0（提交坏掉，score 是失败地板，不可信）。`–` = 缺测或软链 tag 不匹配。

**q3 a00（BASE）**：全 0。**注意：这一列是 infra artifact，不是「base 太弱」**——q3-base 在 harness 里因 `max_model_len` 配置 bug 被 vLLM 在生成前 400 掉每一个请求（详见 case study §base），根本没测到模型。

**q3 a100（INSTRUCT）**

| 角色 | FCS mean/best | ALE/abs | Theta | TTT |
|---|---|---|---|---|
| START | 2.525 / 6.304 | 548.7 / 607M | 1.84 | 0.38 |
| method SFT | 1.365 / 2.276 | 426.0 / 2263M | 1.27 | 0.32 |
| **method soup10** | **3.322 / 6.090** | 543 / 1404M | 1.88 | 0.40 |
| method soup30 | 1.077 / 2.576 | 548 / 10733M | 1.60 | 0.50 |
| methodtraj soup30 | **2.815 / 5.153** | 565 / 4331M | 1.84 | 0.48 |

**q35 a00（BASE）**

| 角色 | FCS mean/best | ALE/abs | Theta | TTT |
|---|---|---|---|---|
| START | 2.080 / 5.166 | 374.5 / 710M | 0.96 | 0.50 |
| method SFT | 0.350 / 1.176 | 323 / 169M | – | – |
| method soup20 | 2.082 / 3.904 | 340 / 311M | 0.96 | 0.32 |
| method soup50 | 2.203 / 5.418 | 320 / 0.2M! | – | – |

**q35 a100（INSTRUCT）**

| 角色 | FCS mean/best | ALE/abs | Theta | TTT |
|---|---|---|---|---|
| START | 3.139 / 7.877 | 359.7 / 2954M | 1.35 | 0.45 |
| method SFT | **0.015 / 0.039** | 310! / 0 | – | – |
| **method soup10** | **2.924 / 7.463** | 397 / 440M | 2.09 | 0.38 |
| method soup20 | 2.348 / 6.558 | 378 / 183M | 1.09 | 0.44 |
| method soup50 | 1.313 / 3.926 | 317 / 0.7M! | – | – |
| method soup70 | 0.010 / 0.024 | 322 / 0! | – | – |
| methodtraj soup10 | 2.836 / 6.060 | 369 / 176M | 1.65 | 0.50 |

**读法**：纯 SFT 在 instruct 起点上把 FCS 从 3.14 砸到 0.015；soup10 拉回 2.92；soup 比例越高越差（剂量-响应，约 40–50% 处阈值式崩塌）。详见 case study。

### 6.2 MLS-Bench 批量（20 任务均分；与起点对照）

（8 个模型全跑完 20/20，除 a100_innovonly_sft 19/20。基线 = q35 instruct start 0.0643）

| 模型 | MLS mean | FCS mean@5 对照 |
|---|---|---|
| q35 base start（Qwen3.5-9B-Base） | 0.0764 | 2.080 |
| q35 instruct start（基线） | 0.0643 | 3.139 |
| **q35 base + method-SFT** | **0.0943 ↑** | 0.350（崩） |
| **q35 base + method-soup20** | **0.0908 ↑** | 2.082（≈起点） |
| **q35 instruct + method-SFT** | **0.0794 ↑** | **0.015（崩到底）** |
| q35 instruct innovonly-soup50 | 0.0728 ↑ | — |
| q35 base innovonly-soup50 | 0.0597 | — |
| q35 instruct method-soup10 | 0.0538 ↓ | 2.924 |
| q35 instruct innovonly-SFT | 0.0420（19/20） | — |

**关键反转 / 双重分离**：在 FCS 上被打到 0.015 的 instruct method-SFT，在 MLS 上是 **0.0794 > 起点 0.0643**；两个臂的纯 SFT 都在 MLS 上超过各自起点。而 FCS 甜点 instruct-soup10 在 MLS 上 **0.0538 < 起点**。即 **FCS：SFT 最差、soup 最好；MLS：SFT 最好、soup 最差** —— 干净的双重分离。这是「**FCS/ALE 奖励提交简单正确代码、惩罚探索；MLS 这种 ML 研究任务恰恰是 innovation 取向能用上的地方；soup 恢复执行纪律（FCS↑）的同时稀释掉研究取向（MLS↓）**」的直接证据。**例外的折中点**：base 臂的 **method-soup20** 双轴都好（FCS 2.082≈起点 + MLS 0.0908>起点），是"既保住能力又拿到创新 proxy"的现成存在性证明。
> **注意（零分诊断后）**：MLS 的绝对分受上面"零分诊断"里的 harness/infra 问题压低（causal 簇因 prompt 路径 bug 自动 0、部分任务数据缺失）。这些 0 是评测侧问题，不是模型能力——所以 MLS 的**模型间相对排序**比绝对值更可信，且修完 causal bug 后整体分会上移。
**诚实警告**：单跑、20 个噪声任务、soup 那两行未跑完，不能过度解读；方向上与 case study 机制一致。
非零贡献集中在少数任务：optimization-evolution-strategy(0.49)、hyperparameter-search(0.30)、causal-treatment-effect(0.26)、symbolic-regression(0.13)；13/20 任务对起点是 0（编辑了不允许的包、从不提交——与 FCS 同一个"不落地"签名）。

### 6.3 RL（amplify 实验）

目标：RL 是否把 method-soup 放得比 start 更高。修了一连串 infra bug（fragile 2 卡配置 → 退回 proven 4 卡；go-judge OOM；image-processor 缺失；base-chat 的 NCCL NVLS 偶发错）后，pipeline 已能稳定训练（`rlm2_sfta00` 等已过 `global_step_2`）。诊断出**慢的根因是误抄了 NPU 风格的全量 CPU offload**，已改快配置（关 actor offload、response 降到 20k、`save_freq=5`），并在扩展更多起点（q3-inst / q35-inst / q35-base + 好的 SFT/soup）。**最终放大结论待 checkpoint 评测后给出。**

---

## 7. 关键发现（指向 case study）

1. **Discovery 论点未被支持**：TTT 纯噪声、Theta 大多噪声/卡地板、竞赛类被 SFT 打崩。
2. **SFT 学到的是「创新腔调」而非「实质」**：详见 [CASE_STUDY_zh.md](CASE_STUDY_zh.md)。
3. **思考坍缩是数据效应**：method-only 子集恢复思考长度（27071 vs 全量 1782）。
4. **soup 的稳健效应是「恢复」，不是「放大」**：小比例 soup 修回 SFT 破坏的执行纪律；偶尔「超过起点」逐样本核对是单跑噪声。
5. **评测口径错配**：MLS 上 method-SFT 反超起点，FCS/ALE 上崩——同一模型、相反结论。
6. **发现的评测 bug / 注意事项**：q3-base 的 `max_model_len` 配置 bug（整列 artifact）；ALE 的 310 失败地板；Theta/TTT 的 seed floor 与噪声;shard_0 只覆盖 40/182 题。

---

## 8. 代码

实验相关代码见 [`experiments/scripts/`](scripts/)（评测 harness、聚合器、soup 合并、矩阵编排器、RL 配置）。各文件顶部有说明。
