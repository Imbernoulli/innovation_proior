# Scientific Taste / Idea-Quality 评测调研（逐篇精读）

> 目的：为 innovation_prior 找一批**比 FrontierCS / MLS-Bench / ALE-Bench / ThetaEvolve / TTT-Discover 更轻**的评测。
> 现有评测全是执行落地型，成本高、对 4B 不友好、分数被截断和环境噪声吃掉。
> 这份调研聚焦**端到端 model-judge 型**评测：给背景 → 提方法 / 判优劣 → 打分。
>
> 整理时间：2026-08-25，第二轮补读 2026-08-26。**本版全部 50 篇论文均已下载 PDF 全文精读**，数据构造、任务格式、评分协议、完整结果表、人类验证、作者自陈局限逐项抄录。
> 与检索摘要冲突之处，一律以正文为准，并在文中标出（见 §ForeSci、§Tang & Yang、§LiveIdeaBench 三处修正）。

## 目录

**第一部分 客观 GT 判别型**：[SciJudgeBench](#1) · [Forecast Research Success](#2) · [SoundnessBench](#3) · [HindSight](#4) · [ForeSci](#5) · [TastyBench](#6) · [LigBench](#7) · [RINoBench](#8) · [NovBench](#9) · [NoveltyRank](#10)
**第二部分 端到端生成 + judge**：[HypoArena](#11) · [MLR-Bench](#12) · [LiveIdeaBench](#13) · [AI Idea Bench 2025](#14) · [Reconstruction](#15) · [MoRI](#16)
**第三部分 judge 不可信的证据**：[RQ-Bench](#17) · [SciStyleBench](#18) · [SciArena](#19) · [Axiomatic Novelty](#20) · [Beyond Rating](#21) · [Can AI Evaluate AI Scientists](#22)
**第四部分 泼冷水 / 反方**：[Heuresis](#23) · [AI Research Agents Narrow Scientific Exploration](#24) · [Execution-Grounded Automated AI Research](#25)
**第五部分 第二轮精读**：[GIANTS ⭐](#27) · [TasteGap ⭐](#28) · [PreScience ⭐](#29) · [Lit2Test ⭐](#30) · [SCOPE ⭐](#31) · [SciPredict](#32) · [AbGen](#33) · [ResearchArena](#34) · [ScientistOne / CoE Audit](#35) · [AIPR](#36) · [LLM-as-a-Reviewer / PRAIB](#37) · [其余系统与数据集](#38)

---

# 第一部分：有客观 ground truth 的「品味判别」评测

---

<a id="1"></a>
## 1. AI Can Learn Scientific Taste / SciJudgeBench

- **arXiv**: [2603.14473](https://arxiv.org/abs/2603.14473) v3 · **代码**: https://github.com/tongjingqi/AI-Can-Learn-Scientific-Taste
- 复旦 / 上海创智学院 / OpenMOSS / 清华 / 中南

**定位**：把 scientific taste 定义为"判断与提出具有长期科学影响力的研究想法的能力"，并证明它**可以被训练出来**。整份调研里和我们目标最同构的一篇。

### 数据构造

底库 210 万篇 arXiv 论文（至 2024 年）。用**引用数**作为科学共同体的反馈信号，理由是"引用是研究社区通过长期互动给出的裁决"。为了消掉领域和时间偏置，**同领域（同 subcategory）、同年份配对**，引用显著更多的一方作为 preferred。

- 训练对筛选阈值：引用数**绝对差 ≥ 8 且相对差 ≥ 30%**（差距太小的对被丢掉，避免噪声标签）
- 最终 **720,341** 组偏好对，覆盖 CS / Math / Physics / Others

每条样本 = 两篇论文的 **title + abstract**（不给正文），二元标签。

### 五个测试切分（这套设计是整篇最值得抄的部分）

| 切分 | 规模 | 设计意图 |
|---|---|---|
| Main（in-domain） | 1,000 对（自 8,830 对池中分层抽样） | 主分，跨 CS/Physics/Math/Others 分层 |
| Temporal OOD | 904 对，**2025 年**论文（训练期之后） | 检验是否外推到未来论文 |
| Metric OOD (ICLR) | 611 对 review-score 对，ICLR 2017–2026 投稿 | 换一种"质量"定义（同行评审分） |
| Metric OOD (Altmetric) | 599 对 | 换成在线社会关注度 |
| Controlled | 541 对**同一 CSRankings 教师本人**、同发表季度、同子领域；245 对 embedding 匹配同题 | 排除"名校名作者"和"热门话题"两条捷径 |

另有 160 对 bioRxiv 生物学对做跨学科检验，以及单独训练 CS-only 变体评所有领域的 field OOD。

### 任务格式

> "Based on the titles, abstracts, and publication dates of the following two papers A and B, determine which paper has a higher citation count."

输出 = reasoning trace + 二选一（A 或 B）。

### 评分协议

**position-swap consistency**：每对正反各判一次（A↔B 交换），**两次一致才算对**。这条必须抄——不做的话位置偏置能白送十几个点。

### 训练方法 RLCF

三阶段：
1. **构造社区偏好**（上述配对）
2. **偏好建模 → Scientific Judge**：GRPO，reward 是二元正确性（对=1 错=0），group 内 advantage 归一化，clipped surrogate + KL penalty
3. **偏好对齐 → Scientific Thinker**：用 Scientific Judge 当**生成式 reward model**。因为单条 idea 打分没有客观标准，改用 **Comparison-Based GRPO**：给 seed paper，policy 采样 G 条 idea，由 Judge 做 **round-robin 锦标赛**（共 C(G,2) 次成对比较），每条 idea 的 reward = 组内胜率

### 结果：Scientific Judge（in-domain 主表）

| 模型 | CS | Math | Physics | Others | **Overall** |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct | 58.1 | 71.1 | 51.8 | 55.8 | **58.1** |
| **SciJudge-Qwen3-4B** | 78.6 | 82.8 | 74.1 | 75.5 | **77.3 (+19.2)** |
| Qwen3-30B-A3B-Instruct | 73.4 | 79.9 | 64.4 | 63.9 | 69.7 |
| **SciJudge-Qwen3-30B** | 83.5 | 89.7 | 81.2 | 77.4 | **82.7 (+13.0)** |
| Qwen2.5-1.5B-Instruct | 4.0 | 6.4 | 5.3 | 5.8 | 5.3 |
| SciJudge-Qwen2.5-1.5B | 60.1 | 66.7 | 60.3 | 62.0 | 61.9 (+56.6) |
| Qwen2.5-3B → SciJudge | 28.0 → 72.8 | | | | +44.8 |
| Qwen2.5-7B → SciJudge | 50.2 → 76.6 | | | | +26.4 |
| Qwen2.5-14B → SciJudge | 58.2 → 79.9 | | | | +21.7 |
| Qwen2.5-32B → SciJudge | 63.5 → 81.5 | | | | +18.0 |
| Llama3.1-8B → SciJudge | 46.3 → 57.9 | | | | +11.6 |
| **GPT-5.4 Thinking** | 87.5 | 84.8 | 77.4 | 78.4 | **81.6** |
| MiniMax-M3 Thinking | 86.7 | 83.8 | 72.9 | 72.6 | 78.5 |
| GLM-5.1 Thinking | 84.3 | 80.9 | 68.8 | 75.0 | 76.4 |
| DeepSeek-v3.2 Thinking | 73.0 | 76.0 | 63.2 | 65.4 | 68.7 |

注意小模型的 base 分**远低于 25% 随机**（Qwen2.5-1.5B 只有 5.3%）——因为一致性计分下，胡乱翻转就直接判错。

**时间 OOD（904 对 2025 年论文）**：Qwen3-4B 64.7 → SciJudge-Qwen3-4B **80.9 (+16.2)**；Qwen3-30B 71.7 → **83.1 (+11.4)**。SciJudge-Qwen3-30B **超过 GPT-5.4 Thinking**。
**Metric OOD (ICLR)**：迁移后提升 8.2 点到 86.3%。

### 结果：Scientific Thinker（生成侧）

- 数据：2025 年高引论文作 seed。训练 4,000 篇（1–7 月），in-domain 测试 200 篇（同期），OOD 测试 200 篇（8–12 月）
- policy：Qwen3-30B-A3B-Thinking-2507 和 **Qwen3-4B-Thinking-2507**，reward model 都用 SciJudge-Qwen3-4B
- 评测：对每个 seed，训练后 policy 与 base policy 各出一条 idea，由 **GPT-5.2-high / GLM-5 / Gemini 3 Pro 三家多数投票**判谁潜在影响力更高

| | in-domain 胜率 | OOD 胜率 |
|---|---|---|
| SciThinker-30B vs base policy | 81.5% | 83.0% |
| **SciThinker-4B vs base policy** | **76.5%** | **76.0%** |
| 若 reward model 换成未训练的 Qwen3-4B-Instruct（30B policy） | 73.0% | 70.5% |

对三个强基线的平均胜率：Qwen3-30B base 30.3% → SciThinker-30B **54.2%**（对 GPT-5.2 61.0%，GLM-5 58.5%，Gemini 3 Pro 43.0%）。

### 对我们的价值（最高）

1. **有公开的 Qwen3-4B 判别分（58.1 → 77.3）和 4B 生成侧胜率（76.5%）**，和我们模型尺寸完全对齐，省掉自建 baseline 的整整一轮实验。
2. 五切分 + position-swap 协议可以整套搬成我们评测的模板，尤其**作者/机构 matched control** 那一档，是防"模型只是在认名校"的关键。
3. Comparison-Based GRPO（round-robin 胜率当 reward）是一条我们没试过的 RL 路线，不需要 verifier。

**我的判断**：首选。唯一保留是引用数仍混入传播力，但它用同领域同期匹配 + 同作者 control 把这条堵得比同类严得多。

---

<a id="2"></a>
## 2. Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation

- **arXiv**: [2605.21491](https://arxiv.org/html/2605.21491v1) · 数据 CC 协议，匿名仓库释出
- 论文没起正式 benchmark 名

**定位**：整份调研里**最纯粹的"方法品味"评测** —— GT 不是引用数，是**真实跑出来的 benchmark 分数**。

### 数据构造（五步，每步都有防污染设计）

1. **爬榜**：爬所有可得的 NLP leaderboard，保留至少两条记录的，得 **1,918 个 leaderboard**；为每条记录定位其 Result-Reporting (RR) 论文，得 **5,713 篇** RR 论文（7 篇付费墙排除）
2. **抽 research goal**：从每个 leaderboard 的官方 benchmark 描述里抽一条 canonical research goal
3. **抽 idea**：用 LLM 从 RR 论文和方法原始论文里抽方法描述。**抽取 prompt 里显式写了排除条款：不得包含任何实验结果或结论性陈述**
4. **人工核验测试集**：逐条对照原 PDF 检查。错误分两类——**Incomplete（约 4%）**：漏了关键细节（例如漏掉引入的特殊损失函数），补上；**Incorrect（约 8%）**：Minor（输出维度写错等）改正，Major（例如给一个 BERT 微调方法编造出对抗组件）直接删除。核验标准里显式包含"确认成功排除了实验结果与结论"
5. **Unified Score 与配对**：不直接用榜单排名（因为相邻名次的差距在不同榜上差异极大）。做法是每个 benchmark 内对各指标 **min–max 归一化**，"越低越好"的指标（如 perplexity）取反，约 85% 的 benchmark 只有单指标、多指标则平均，得到每条 idea 的 **Unified Score**

**难度分层**：算该 benchmark 内所有条目 Unified Score 的标准差 σ，按归一化分差 Δ 分三档——**1σ（难）/ 2σ（中）/ 3σ（易）**，用 20% 容差（如 1σ 档取 0.8σ–1.2σ）。

**训练/测试划分防泄漏**：按 leaderboard 迭代、按 idea 划分，保证**同一条 idea 不会同时出现在训练和测试**；条目少于四条的 leaderboard 整体划入训练。最终 2,893 条训练 idea / 693 条测试 idea，去掉没有 research goal 的 leaderboard 后约 **90/10** 分。总计 **11,488** 组配对、**724** 个有效 leaderboard。

### 三个测试集

- **主测试集**（同分布，NLP）
- **Cross-Domain 测试集**：从非 NLP 榜单（语音合成、分子性质预测等，至少 3 条记录、RR 论文年份 ≥ 2024）同法构造 **705 对 / 46 个 leaderboard**。刻意改用 **GPT-5 high reasoning** 抽 idea 以引入语言分布差异；且不做难度分层、直接用单指标配对
- **独立构造测试集**：取自 Wen et al. (2025)，与训练集零 idea 重叠，**1,750 对**，多数投票标签

### 任务格式与评分

输入 = research goal g + 两个 idea 描述 h_A / h_B；输出 = 二元标签。

**评分**：consistency-aware —— 原序和换序两次预测必须一致，且正确，才算对。**这意味着随机猜测的 chance level 是 25%，不是 50%**（正反都猜对的概率）。这一点非常重要，读表时不能按 50% 基线看。

### 训练方法

- **SFT**：直接预测标签，`L = −log P(y | g, h_A, h_B)`。LoRA r=64、α=128、dropout 0.1、batch 2、lr 2e-4，BF16，A100-40GB
- **RL**：把 reasoning 当隐变量。reward = 正确性 ±3.0 + 格式分（`<think>` ±0.5、`Answer:` ±0.5）。用 **DAPO**（全局 token 归一化 + 解耦 clipping）和 **Dr. GRPO**（去掉 advantage 分母的 std，修长度偏置）两个变体
- **两种 CoT 来源**（这是个有意思的对照）：
  - **Synthetic**：从 GPT-5 (high) 蒸 Chain-of-Rubrics，随机取 2,125 对，只保留 GPT-5 预测与 GT 一致的，得 1,369 对，换序增广后 2,738 条
  - **Literature-Grounded**：只取**两条 idea 出自同一篇 RR 论文**的配对——这样论文里必然真的做过这个比较，不是事后推断的理由——再让 LLM 从原文抽出解释为什么一个更好；抽不到就明说没有

### 结果

主测试集：

| 模型 / 方法 | Accuracy |
|---|---|
| Qwen3-8B base | 25.31%（≈ 随机 25%） |
| Llama3.1-8B base | 30.02% |
| Llama3.1 + 朴素 CoT 提示 | **27.38%（反而降了）** |
| GPT-5（零样本，各 reasoning 档） | **61.10%** |
| **Direct-SFT Qwen3-8B** | **77.10%** |
| Reason-SFT-DAPO / Reason-SFT-DrGRPO | ≈ **71.35%**（带可解释理由） |
| Reason-SFT（只 SFT 不 RL） | 37.51% |
| Synthetic-Reason-SFT（只 SFT 不 RL） | 25.54%（几乎没提升） |

独立测试集（Wen et al. 1,750 对，零样本迁移，不重训）：

| 模型 | Accuracy |
|---|---|
| Qwen3-8B base | 2.69% |
| Qwen3-8B base (Reasoning) | 20.06% |
| Direct-SFT | 63.43% |
| **Reason-SFT-DrGRPO** | **67.49%** |
| GPT-4.1 + 检索增强（Wen et al. 原报告） | 51.4% |

难度分层验证有效：多数微调模型上 1σ < 2σ < 3σ 单调。跨域测试集上所有训练过的 Qwen3 变体都 ≥ GPT-5（除 Synthetic-Reason-SFT-DAPO）。

### 三条值得记住的结论

1. **合成 CoT 即便过滤了正确性也学不动**（25.54%），文献锚定的 reasoning 才有用——和 Wen et al. 的发现一致
2. **朴素 CoT 提示会伤害性能**（Llama 30.02% → 27.38%），只有被训练去 reason 的模型才从 deliberation 里获益
3. RL 能把 reasoning 版从 37.51% 拉回到 ~71%，代价是比 Direct-SFT 的 77.1% 低 6 点——**可解释性的定价大约是 6 个点**

### 对我们的价值（与 SciJudgeBench 并列首选）

- GT 是硬数字，完全没有 judge 偏差
- **8B SFT 打穿 GPT-5，并在独立数据集上以 50× 更小的模型、无检索的条件下超过 GPT-4.1+RAG 16 点** —— 这正是我们要证的命题形态
- 难度分层自带"我们的模型在难例上是否也强"的分析维度
- 它对 CoT 来源的对照实验，直接关系到我们 paper2reasoning 数据的价值论证：**文献锚定 > 合成蒸馏**

---

<a id="3"></a>
## 3. SoundnessBench: Can Your AI Scientist Really Tell Good Research Ideas from Bad Ones?

- **arXiv**: [2605.30329](https://arxiv.org/html/2605.30329) · **数据**: https://huggingface.co/datasets/hosytuyen/SoundnessBench · **主页**: https://hosytuyen.github.io/projects/SoundnessBench

**定位**：**执行之前**能不能看出一个 proposal 方法学上不成立。作者明确限定它测的是 "**recoverable** proposal-stage methodological validity"，不是绝对研究质量。

### 数据构造（五步）

从 ICLR **2022–2026** 起步（更早的场次没有稳定提供 soundness 相关分项），处理了 **35,209 篇投稿 / 137,940 条专家评审**。

1. **筛选**：去掉 desk-reject（可能出于 soundness 以外的原因）；只保留评审共识强的——**平均 reviewer confidence ≥ 3 且归一化 soundness 分的标准差 < 0.15**
2. **标签**：用 reviewer 的 **soundness 子分**，不用总分、不用接收决定、不用 novelty/presentation 分。均值 **≥3 判高**、**≤2 判低**，**中间分段整体剔除**以拉开类间距
3. **抽 proposal**：从原 PDF 抽 abstract、related work、risk factors、hypothesis、experiment design，**显式排除实验结果、结论性主张、接收线索**。格式沿用 The AI Scientist-v2。用长上下文的 Gemini 2.5 Pro，prompt 要求给出**有源文档 span 支撑**、禁止无依据推断，**近乎逐字保留原文措辞**以免释义漂移
4. **核验审计**（Alg. 1）：把每个 hypothesis–experiment 对拆成**原子断言**，检索支撑段落，逐条对源论文核实。参数 τ=0.7、chunk 3000、overlap 200、检索深度 k=3。**候选池中 66.93% 通过此过滤**
5. **最终**：**1,099** 条（**641 高 soundness / 458 低**），覆盖 RL、生成模型、NLP、优化、CV 等 **16 个 ML 子领域**

### 任务格式与两种 prompt

模型读一条 proposal，输出高/低 soundness 二分类 + 理由。

- **Standard prompt**：结构化的 justification-first 评估
- **Aggressive prompt**：压力测试，指示模型**默认判低**，除非 idea 与实验设计明显强且论证充分

作者明说不声称穷尽 prompt 空间，两个 prompt 是为了测试"一个常规 prompt 和一个刻意更严的 prompt 能否同时在两个类上保住区分度"。

### 被测模型与推理设置

12 个前沿模型：GPT-4o†、GPT-5.4-Mini、GPT-5.4†、Claude-Opus-4.6†、Claude-Sonnet-4.6†、Gemini-2.5-Pro†、Gemini-3-Flash、Gemini-3.1-Pro†、Qwen3.5-27B†、Qwen3.5-122B-A10B†、LLaMA-3.3-70B、Kimi-Linear-48B-A3B（†为推理模型）。
闭源走官方 API，开源用 vLLM 部署在 2×H200；统一 max_tokens=8192、temperature=0.2。

**指标**：混淆矩阵 + 两类各自 recall（Low R / High R）+ **Macro F1**（两类 F1 的非加权平均）。

### 完整结果

| 模型 | Std Low R↑ | Std High R↑ | **Std Macro F1↑** | Agg Low R↑ | Agg High R↑ | Agg Macro F1↑ |
|---|---|---|---|---|---|---|
| LLaMA-3.3-70B | 2.0 | 99.4 | 38.9 | 39.3 | 76.1 | 57.5 |
| Kimi-Linear-48B-A3B | 13.8 | 84.6 | 44.6 | 50.2 | 45.8 | 47.5 |
| Qwen3.5-27B† | 23.6 | 92.5 | 55.1 | 83.8 | 32.4 | 52.6 |
| Qwen3.5-122B-A10B† | 26.6 | 90.6 | 56.3 | 95.6 | 16.8 | 44.7 |
| GPT-4o† | 5.5 | 98.9 | 42.3 | 87.1 | 25.0 | — |
| GPT-5.4-mini | 50.7 | 76.6 | 63.8 | 100.0 | **0.2** | — |
| **GPT-5.4†** | 64.6 | 74.6 | **69.7（最佳）** | 100.0 | **0.0** | — |
| Gemini-2.5-Pro† | 13.8 | 98.4 | 49.8 | 73.4 | 55.1 | — |
| Gemini-3-Flash | 19.7 | 98.1 | 54.5 | 76.4 | — | — |
| Gemini-3.1-Pro† | 26.0 | 96.1 | 58.4 | 89.1 | — | — |
| Claude-Sonnet-4.6† | 37.2 | 94.8 | 65.3 | 94.4 | — | — |
| Claude-Opus-4.6† | 28.2 | 97.2 | 60.5 | 72.2 | — | — |

**汇总**：standard prompt 下平均假阳性率 **74.0%**（12 个里 9 个超过该值）；aggressive prompt 下假阳降到 **19.9%**，但高 soundness recall 塌到 **36.1%**——GPT-5.4 和 GPT-5.4-mini 更是直接归零。

### 稳健性控制（做得相当扎实，值得抄）

- **标签与泄漏审计**：人工核验抽取的 proposal 是否泄漏了结果/结论。已完成的审计子集中 **92.3%** 的泄漏检查符合预期的"无泄漏"
- **降低污染风险的检验**：单独用 **ICLR 2026-only 切分**（决定公布晚于多数模型 cutoff）重跑 standard prompt，乐观偏置模式依旧
- **识别与表层特征控制**：去掉标题/标识符；用 proposal 长度、实验数量等无训练基线做对照——都不能解释主结论
- 作者明确说这些控制**不使 benchmark 变得无污染**，也不把 reviewer 分变成绝对真值

### 对我们的价值（推荐进主套件）

1. 它测的是**否定能力**（前两个测排序能力），覆盖面互补
2. 假阳性率天生是"模型是不是见谁都说好"的诊断图；我们训练后若 FPR 明显下降且 High R 不塌，是很有说服力的一张图
3. 剔除结果只留 hypothesis + 实验设计，形态非常接近我们的 context.md
4. 它的**原子断言核验审计**流程，我们做数据质检时可以直接复用

---

<a id="4"></a>
## 4. HindSight: Evaluating LLM-Generated Research Ideas via Future Impact

- **arXiv**: [2603.15164](https://arxiv.org/html/2603.15164)

**定位**：不用 LLM judge，把生成的 idea 拿去**和后来真实发表的论文对撞**。时间切分的思路借自金融回测与推荐系统的 temporal split。

### 方法细节

- **时间切点 T = 2023 年 6 月**。约束：LLM 的知识 cutoff 必须晚于 T 且留安全边界
- **GT 池**：Semantic Scholar API 抓 2023-06 至 2025-12、10 个 AI/ML 主题的论文，去重后 **27,589 篇**（含标题、摘要、引用数、venue）
- **影响力打分**：`h(p) = 0.6 · ĉ(p) + 0.4 · v(p)`，ĉ 是池内 min-max 归一化引用数，v ∈ {0,1} 表示是否发表于 ICLR / NeurIPS / ICML / ACL / EMNLP / CVPR / AAAI
- **匹配**：SPECTER2（768 维 CLS，citation graph 上预训练）+ FAISS 内积索引（L2 归一化即余弦），每条 idea 取 top-K=20
- **阈值 θ=0.96**：因为 SPECTER2 在 AI/ML 文本上相似度分布极度集中（0.91–0.98），θ 由敏感性分析选出
- **最终分** = 命中论文中的**最大**影响力（用 max 而非 mean，理由是单个高影响力命中就足以证明该 idea 预见了一个重要方向）；无命中记 0

### 被评的两个系统

- **ResearchAgent（检索增强）**：简化的两阶段实现——ProblemIdentifier 读 seed 论文提出开放问题，MethodDeveloper 提出具体方法
- **Vanilla baseline**

各 100 条，共 200 条 idea，覆盖 10 个主题。

### 对照的 LLM-as-Judge 设置

200 条 idea 全部由 **Qwen3-32B** 在四个维度（Novelty / Feasibility / Expected Impact / Overall，1–10）上打分，每条**评 3 次（T=0.7）取平均**。作者刻意选了与生成器不同的模型家族以避免自偏好。

### 结果（这三条是整份调研最重要的实证之一）

| | ResearchAgent | Baseline | 检验 |
|---|---|---|---|
| **HindSight 分** | **0.297** | 0.119 | Mann–Whitney U, **p < 0.001**（2.5×） |
| 命中率（至少匹配 1 篇） | **81%** | 42% | — |
| **LLM-as-Judge Overall** | — | — | **p = 0.584（无显著差异）** |
| LLM-Judge Novelty | 更低 | **更高** | p < 0.001 |
| LLM-Judge Expected Impact | 更低 | **更高** | p < 0.001 |

**HindSight 分与 LLM 判定的 novelty 相关系数 ρ = −0.29。**

也就是说：judge 不但看不出 2.5 倍的真实差距，还在 novelty 和 expected impact 两个维度上**把更差的系统排在了前面**。

### 局限（我的补充）

judge 用的是 **Qwen3-32B，属于偏弱的 judge**。用 GPT-5 级别的 judge 重做未必是同样的结论。论文也未明确说明数据/代码是否释出。所以这篇的强度在于"存在性证明"（judge 可以严重错），不宜过度外推为"所有 judge 都无效"。

### 对我们的价值

1. "别只信 LLM judge"最硬的一个反例
2. ρ = −0.29 意味着**只报 judge 的 novelty 分，很可能在报反指标**
3. 方法可复刻：我们有 paper2reasoning 的时间线信息，做一版 cutoff-anchored 的未来命中评测是现实的

---

<a id="5"></a>
## 5. ForeSci: Evaluating LLM Agents for Forward-Looking AI Research Judgment

- **arXiv**: [2606.00644](https://arxiv.org/pdf/2606.00644)

> ⚠️ **修正**：我第一版根据检索摘要把它写成了"预测哪篇论文将来引用高"，**这是错的**。它不是引用预测。

**定位**：给定某个时间点 t 之前可得的证据，做**前瞻性研究决策**——攻哪个瓶颈、排哪条研究议程、投哪个会。

### 数据构造

**500 个任务**，覆盖 4 个快速演进的 AI 领域 × **4 个决策族**：

1. **Direction Forecasting** —— 预测具体的技术轨迹
2. **Bottleneck–Opportunity Discovery** —— 指出一个根因瓶颈，以及若该瓶颈被缓解会解锁什么机会
3. **Strategic Research Planning** —— 给出排序的研究计划
4. **Venue-Conditioned Positioning** —— 面向特定会议的定位选择

每个任务配一个 **cutoff 对齐的离线知识库**：post-cutoff 论文在生成阶段被隐藏，只有评测端可见（记为 `G>t(q)`）。为避免 hindsight，**answer-generation backbone 也必须是 cutoff 之前训练的模型**。任务本身从 pre-cutoff 的分类树分支和证据信号（方法发展信号、瓶颈信号等）派生。

### 四个指标

- **Prediction Factuality (Fact.)**：仿 FACTSCORE 的原子事实观，从回答抽原子断言 `C(a)`，与从隐藏未来目标导出的 hidden claim bank `C*(q)` 比对，报**断言级 F1**
- **Future-Target Alignment (FTA)**：Direction Forecasting 与 Bottleneck–Opportunity 用 **bge-m3 相似度**比对预测断言与 claim bank；Strategic Planning 与 Venue Positioning 因为目标是有序决策，用确定性的 ranking-aware F1
- **Trace**：证据可追溯性
- **Pers.（Persuasiveness）**：每个任务族一套 rubric `R_f`，在 decision quality / mechanistic reasoning / comparative reasoning / clarity / risk awareness 上，由"LLM 虚拟审稿人"评分

### 被评系统

三类方法 × 四个 backbone（Qwen3-235B、GPT-5.2、GLM-4.6、Gemini-3）：原生 LLM、Hybrid RAG、三种离线适配的 research-agent 系统。

### 核心发现

显式证据接入提高了**可追溯性与事实性**，但作者识别出一个新的失效模式：**evidence-decision decoupling** —— agent 能引用相关的 pre-cutoff 证据，**却预测错了对象、错配了因果角色、或选错了干预手段**。

### 一个必须记下的细节

论文的 claim-level 评分 prompt 里作者自己写着：

> "This is a **factual/content coverage metric, not a research-taste metric**."

**所以 ForeSci 严格说不是 taste 评测**，它是证据接地 + 前瞻决策的复合评测。Pers. 那一维才沾边，而那一维是 rubric 型 LLM judge。

### 对我们的价值

不作为 taste 主评测。但它的**"backbone 训练时间必须早于任务 cutoff"** 这条防泄漏设计比单纯的时间切分更狠，值得我们抄——我们模型的 base 是 Qwen3.5，任何用 2025 年之后论文构造的评测都要考虑这一层。

---

<a id="6"></a>
## 6. TastyBench: Toward Measuring Research Taste in LLM

- **来源**: [LessWrong](https://www.lesswrong.com/posts/Mxsy7wYvsCRv5dGrw/tastybench-toward-measuring-research-taste-in-llm) · **代码**: github.com/parviam/tastybench

一个小而锋利的探针。核心问题定得很好：

> "Can models predict whether an approach will yield insights or improvements **before executing it**?"

它对 research taste 的定义值得直接引用：

> "the set of intuitions and good judgment guiding a researcher's decisions throughout the research process, **whenever an open-ended decision arises without an obvious way to find the right answer**."

**构造**：Semantic Scholar 上抓 2024 年 1–3 月 "reinforcement learning large language model llm rl" 主题 200 篇 → 过滤到 **38 篇**有算法级 RL 改进的；另有 25 篇做全文分析。GT 排序用 **citation velocity**（单位时间引用累积速率）。

**评分**：LLM 先从 abstract 抽核心 idea，再用 **3 种不同 prompt** 做成对判断生成 Elo，与 citation velocity 排序求相关。

**结果**：Claude Sonnet 4.5 / Gemini 2.5 Pro / GPT 5.1 全部 **~0.3 相关**，换 prompt、加全文信息都不改善。作者结论：LLM 没有超人的 research taste。

**对我们的价值**：38 篇太小，不能当主评测。但作为**"现状是差的"的引用出处**很好用——我们要论证 innovation prior 有效，先要立"现有模型在这件事上确实不行"的基线，这篇 + SoundnessBench 是最直接的两个。

---

<a id="7"></a>
## 7. LigBench / PAIR-IQ

- **arXiv**: [2608.13136](https://arxiv.org/html/2608.13136) · **数据**: PAIR-IQ 已在 HuggingFace 公开（`USER3IjEBHj9/PAIR-IQ`）；完整 pipeline 后续释出

**定位**：统一、客观、且与人对齐的 AI research idea 自动评测。

### 数据 PAIR-IQ

**11,164** 篇，来自 ICLR 2025（36.7%）、ICLR 2024（32.4%）、NeurIPS 2024（30.9%），含 oral / spotlight / poster / rejected 全档。分数取自 OpenReview review，并做 **mean-shifting 去偏**以抹平会议间的系统性偏移。最常见的主题是（19.3%）某类，其次是 training 相关。

### 评测流程

1. 把目标 idea 拆成 **Main Target / Core Breakthrough / Innovative Methods / Experimental Design** 四块的结构化表示；库中论文用**同样的流程**表示，保证格式统一
2. **并行检索**（关键词匹配 + embedding 相似度）取语义相近论文
3. 目标 idea 与每篇检索到的论文做 LLM 成对比较，在 **rating / contribution / soundness** 三维上判相对优劣
4. **自适应 Elo 更新**：`E_A = 1/(1+10^((s_B−s_A)/d))`，`s_A^new = C(s_A + K(S_A − E_A))`，其中 K 是自适应更新因子、C(·) 是把分数软钳在 [0,5] 的函数。多轮迭代直到分数变化低于阈值即收敛
5. **novelty 单独一个模块**（因为 novelty 无法靠成对比较得到）：`s_novelty = β·s_sim + (1−β)·s_novelty^(0)`，β=0.7 偏重基于 Semantic Scholar 检索的相似度量化（映射经反 sigmoid），保留 0.3 给 LLM 初判

### 结果

- 成对判断准确率：**GPT-5 系列 0.80+**（0.801 / 0.806 / 0.803），其余模型在 0.71 附近（0.710–0.714）。模型能力与推理能力越强准确率越高，说明成对判断是**非平凡能力**
- **误判分析**：GPT-5 判错的论文对，其去偏后的分差都很小——错误集中在本来就难分的对上
- **人类对齐**（100 组 idea 对，PhD 研究者）：rating **71%** / contribution **79%** / soundness **73%**
- **外部验证**：另评 50 篇 NeurIPS 2025 论文，接收论文在所有维度上一致更高
- 训练过的小模型比未训练基线大幅提升，验证 PAIR-IQ 作为训练资源有效
- **idea 生成框架并不稳定优于强单模型**（与 Tang & Yang、Heuresis 结论一致）

**对我们的价值**：PAIR-IQ 是目前公开的、带 OpenReview 真分数的最大 idea 质量库，既可当训练/校准数据，也可用它的成对协议评我们的模型。

---

<a id="8"></a>
## 8. RINoBench: Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas

- **arXiv**: [2603.10303](https://arxiv.org/abs/2603.10303)（LREC 2026）· **代码**: https://github.com/TimSchopf/RINoBench

**定位**：评的不是"生成 idea 的能力"，而是**"判断 idea 新颖性的能力"**——也就是把模型当 judge 来考。

### 数据构造（一个很聪明的绕路）

作者指出：要造这个数据，直接找专家生成 idea 再找另一批专家评，成本上不可行。于是改从 **OpenReview 的 ICLR 2022 + 2023** 取——因为那里的人类专家已经基于自己的研究 idea 投了稿，并被另一批专家用 rubric 打了 novelty 分且写了理由。

1. 收集全部公开投稿与评审，得 **6,410 篇**，每篇约 3 位评审
2. 评审给的两个 novelty 维度：**"Technical Novelty and Significance"** 和 **"Empirical Novelty and Significance"**，两个都用
3. **过滤主观分歧**：剔除任一维度内或跨维度上评审最大分歧 **超过 1 分**的投稿 → 剩 **3,535 篇**高一致性
4. 对每篇，跨两个维度平均所有评审分，再**分箱成统一的 1–5 整数 rubric**（有清晰中点、极性平衡、层次细腻）
5. 最终 benchmark：**1,381** 条 research idea
6. 数据处理各步用 GPT-OSS-120B；部分 LLM 环节用 GPT-4.1

**1–5 rubric 的语义**：1 = 完全不新颖，所有方面都已存在；2 = 边际新颖，只是既有工作的微小变体；3 = 有一定新颖性，各方面已存在但以新方式组合；4/5 依次递增。

### 任务格式

给定一条 research idea（+ 相关先前工作），模型要：(a) 按五点 rubric 预测 novelty 分；(b) **给出有依据的文字理由**。

### 九个自动指标

同时评**分数**和**理由**两侧。分数侧含 **MAE**（因为分是 rubric 型连续值而非纯离散类别）；理由侧含 **Alignment (ALI)** 等。

### 被测模型

Llama-3.1-8B、Llama-3.3-70B、Llama-4-Scout-17B-16E、DeepSeek-R1、GPT-OSS-120B、o3、GPT-5。

### 核心结论

> LLM 生成的**推理过程**与人类专家的理由高度相似，**但这种相似不转化为判断的准确**；即便是领先的推理模型，其结论也与人类 gold standard 显著背离。

**对我们的价值**：这是"别拿 judge 的解释质量当判断质量"的直接证据。我们如果展示模型 reasoning 好看，必须另外给一个客观 GT 的准确率——否则会被这篇直接反驳。

---

<a id="9"></a>
## 9. NovBench: Evaluating LLMs on Academic Paper Novelty Assessment

- **arXiv**: [2604.11543](https://arxiv.org/abs/2604.11543)

**定位**：评 LLM **生成 novelty 评价文本**（而非打分）的能力，目标是辅助人类同行评审。

**数据**：**EMNLP 2023 的 1,684 篇 paper-review 对**。构造分四阶段：从论文 introduction 抽 novelty 描述；对评审意见做 **aspect 抽取**，取出其中与 novelty 相关的部分作为专家写的 novelty 评价。作者另外测了 aspect 识别模型与 LLM 在这一步的表现（附录 B）。

**四维评价框架**：Relevance / Correctness / Coverage / Clarity —— 评的是"LLM 生成的 novelty 评价"这段文本的质量。

**结论**：当前模型对科学新颖性的理解有限；**微调过的模型常常出现指令遵循缺陷**。

**对我们的价值**：与 RINoBench 互补（一个评分数、一个评文字），可作为 judge 能力的第二个检验点。优先级低于 RINoBench（后者的 GT 过滤更严）。

---

<a id="10"></a>
## 10. NoveltyRank（不建议采用）

- **arXiv**: [2512.14738](https://arxiv.org/abs/2512.14738) · Zhengxu Yan, Han Li, Yuming Feng
- **代码**: https://github.com/ZhengxuYan/NoveltyRank · **数据**: HF `JasonYan777/novelty-ranked-preprints` · **Demo**: https://novelty-rank.vercel.app/

### 做法

**数据**：2023–2025 共 **60,294** 篇 = **50,442** 篇随机抽的 arXiv + **9,852** 篇顶会接收，六个域（cs.AI / cs.LG / cs.CV / cs.RO / cs.CL / cs.CR）。来源含网页爬取与公开的 ICLR 2017–2025 数据集。
**标签**：沿用"以 venue 接收作为原创性启发式信号"的先例，**接收 = 1（正）、随机 arXiv = 0（负）**。
**时间切分**：训练 2024–2025 初，测试 **2025-03-15 之后**（这一步做对了）。

**两个任务**：
- Task 1 二分类：输入 title + abstract + 与 top-K 前作的相似度特征
- Task 2 成对比较：同领域内判谁更 novel。训练期按 **1:5**（每篇正样本配 5 个随机负样本）；**评测期用 dense pairing**——每篇正样本与同域所有负样本配对，消除采样方差

**三个尺度**：GPT-5.1（零样本 API）、Qwen3-4B-Instruct-2507（LoRA，先 SFT 后 DPO；SFT 的 label 由 Qwen-235B 生成辅助）、SciBERT（解冻上 4 层 + 任务头、冻结下 8 层；把 [CLS] 768 维与预计算的 SPECTER2 embedding 及相似度特征拼接）。成对档的 SciBERT 用 **Siamese 网络 + RankNet loss**。

### 结果

二分类（n=10,889，正类 1,358 ≈ **12.5%**）：

| 模型 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| GPT-5.1 | 0.242 | 0.120 | **0.986** | 0.215 |
| SFT Qwen3-4B | 0.627 | 0.194 | 0.632 | 0.297 |
| DPO Qwen3-4B | 0.612 | 0.205 | 0.735 | **0.321** |
| Fine-tuned SciBERT | **0.744** | 0.187 | 0.313 | 0.234 |

成对（n=9,531）：

| 指标 | GPT-5.1 | SFT Qwen3-4B | DPO Qwen3-4B | FT SciBERT |
|---|---|---|---|---|
| Agreement | 0.583 | 0.739 | 0.741 | **0.753** |

### 我的评价

**先给作者应有的信用**：论文自己写了 "Accuracy is also reported for completeness, but **it may be misleading** in cases where the model predicts predominantly negative labels"，并单列了一段 "**The Accuracy Paradox**" 指出微调模型因类别不平衡塌成保守分类器；也在 "Insights on Absolute Novelty" 里承认 "novelty is inherently relative, not absolute"，并以此作为转向成对任务的动机。这些自省是对的。

**但三个问题依然成立**：

1. **Ground truth 立不住，这是根子上的**。"顶会接收 = novel、随机 arXiv = 不 novel" 测的是**接收概率**，里面混着写作质量、选题热度、作者资源、投稿策略。随机 arXiv 里有极新颖的工作，接收论文里有大量扎实但增量的工作——这个映射在两个方向上都错。作者引"先例"为据，但先例本身就是有争议的近似。

2. **主表的排名叙述与数字冲突**。12.5% 正类 ⇒ **恒定输出"not novel"的平凡分类器 accuracy = 0.875**，而表里最高的 SciBERT 只有 0.744、四个模型全部低于该基线。论文虽然点了 accuracy paradox，却**没有把 0.875 这条平凡基线摆出来对照**，读者很容易把 74.4% 当成"不错"。同时 precision 最高 0.205，意味着每 5 次"这个新颖"的判断约 4 次是错的。

3. **"轻量微调打败前沿零样本"是不对等比较**。GPT-5.1 的 recall 0.986 / precision 0.120——precision 几乎正好等于 12.5% 的基率，说明它对几乎所有样本都答"novel"。这是**阈值与 prompt 没有针对 12.5% 基率校准**，不是判断力缺失。正确做法是给零样本档也做阈值校准，或改报 AUC / AP 这类阈值无关指标。成对档 0.583 相对干净些，但那测的仍是"能否区分接收论文与随机 arXiv"。

4. **没有任何人工验证**。

### 还能拿走的两样

1. **数据管线**：6 万篇规模 + 2025-03-15 之后的时间切分，防泄漏这一点做对了，管线可复用
2. **Qwen3-4B 成对档 0.741**，与我们模型同尺寸，可作粗糙参照——但同一件事 SciJudgeBench 用引用配对 + 同领域同期匹配 + 同作者 control + position-swap 做得严谨得多（Qwen3-4B: 58.1 → 77.3），要参照就参照那个

---

# 第二部分：端到端「给背景 → 提方法 → judge」

---

<a id="11"></a>
## 11. HypoArena / Before the Action: Benchmarking LLMs on Prospective Hypothesis Discovery

- **arXiv**: [2607.15766](https://arxiv.org/html/2607.15766v1) · **代码**: github.com/SKYLENAGE-AI/HypoArena · **数据**: huggingface.co/datasets/HypoArena/HypoData

**定位**：定义了 **Prospective Hypothesis Discovery (PHD)** 任务——从重建出的**去结论上下文**映射到假设集。作者强调这与传统 QA 的区别：QA 是检索一个已存在的答案，PHD 要求**识别未解决的张力，提出值得下一步调查的可验证方向**。

### 任务定义（写得很清楚，值得照抄）

**Context C** 近似结论形成之前的原始、异质信息状态：异常观测、碎片事实、不完整记录。必须满足两个约束：
- **informational sufficiency**：提供足够事实基底以支持非平凡的假设生成
- **operational conclusion absence**：显式的最终结论、目标假设、事后因果归因，都从模型可见输入中扣除

满足第二条的上下文称为 operationally "conclusion-free"。

**Hypothesis Set H = {(h_i, e_i)}**：
- **h_i 假设**：解释或预测性主张，须扎根于给定事实且可通过后续测试验证；当 K>1 时假设之间还须**互相区分**
- **e_i 证据**：领域相关的论证。科研场景表现为**验证方案**（如实验设计），调查分析场景表现为**诊断包**（可执行的检查项与支撑证据日志）

**K 由领域决定**：科研类只要一个主猜想，强调深度与可测试性；分析类允许开放基数，强调广度与区分度。作者明说 source-derived Reference **不被当作唯一答案**。

### 数据构造：Forge–Audit Agent 循环

为在"信息泄漏"与"上下文丰富度"之间取平衡，用迭代的 Forge–Audit loop：Forge agent 生成候选上下文，Audit agent 对照约束验证，循环到满足条件或预算耗尽。

**Context Forge** 从源侧事实材料构建，必要时外部检索背景。**外部检索受源特定的时间戳边界约束，排除相关发表/发布/事故 cutoff 之后的材料。** 两种策略：
- **科学域**：为避免依赖任何单一带结论的文档，检索**该发现之前**的一批文献，用 **Document Merging** 把多方来源合成为连贯的事实基底
- **分析域**：用 **Structural De-conclusion** —— 保留颗粒事实（时间戳、测量值），系统性剔除专家结论与显式因果判定

**Hypothesis Forge** 从完整文档导出源侧参考假设集，但**表述成能从重建上下文中被支撑的形式**。

**Audit Agent** 每轮做三项检查：**Leakage Check**（上下文是否无意泄露了被扣除的假设）、**Faithfulness Check**（假设是否仍被源文档支撑）、**Supportability Check**（证据是否提供足够依据）。失败项作为可执行反馈返回 Forge。

### 规模

**988** 个 case，**2,012** 组 hypothesis-evidence 对：

| 领域 | 数量 |
|---|---|
| Biomedical Science | 244 |
| Machine Learning | 218 |
| Social Science | 163 |
| IT Operations | 146 |
| Financial Analysis | 114 |
| Safety Investigation | 103 |

### 人工质量审计

从 Biomedical Science / Financial Analysis / Social Science 各抽 20 个 case，每域招 2–3 位资深博士生或从业专家。对每个 (Context, Hypothesis, Evidence) 三元组按四条标准评：Context 的 **Informativeness / Openness**、Hypothesis 的 **Completeness**、Evidence 的 **Supportiveness**。
结果：**92% 的抽样 case 通过全部标准**（Informativeness 95%、Openness 100%、Completeness 100%）。

### 评分框架 HypoEval

**六个维度**，分两层：
- **Pair-level Fidelity**（每个 (h_i,e_i) 对）：Contextual Grounding、Inferential Insight、Evidential Justification
- **Set-level Quality**（允许多对的领域）：Hypothesis-Space Breadth、Directional Distinctness、Analytical Utility（K=1 时 Directional Distinctness 不适用）

**主指标 = Rubric-based Arena**（成对）：
- **Position Debiasing**：每场对局 A/B 交换判两次，两次结果取平均得去偏分；**只有两个方向极性一致的对局才标记为 consistent**
- **五级判定**：A ≫ B、A > B、A ≈ B、B > A、B ≫ A，映射成 win-share **{1.0, 0.75, 0.5, 0.25, 0.0}**
- **Bradley–Terry–Davidson (BTD)** 聚合，用 tie 参数 θ 把平局建模成一个**独立的认知状态**（而非半个胜场）

**辅指标 = Rubric-based Scoring**：同样六维，1–5 绝对分，只用于诊断画像。

### 实验设置

15 个模型 × 988 个 case × 两种生成模式。judge 用 **seed-2.0-pro**，交叉一致性用 **mimo-v2-pro** 验证。
**源自源文档的 Reference 作为匿名选手参赛**，用于校准——这个设计很聪明。

- **Baseline Mode**：单次直出，零样本对照
- **Agent Mode**：可从 **12 个结构化分析技巧库**里选择并串行执行（改编自《Structured Analytic Techniques for Intelligence Analysis》，含 Analysis of Competing Hypotheses (ACH)、Structured Brainstorming、Chronology Analysis 等），中间产物向前传递

被测 15 个模型：claude-sonnet-4.6、claude-opus-4.6、gpt-5.4、kimi-k2.6/k2.5、glm-5.1/5、deepseek-v4-pro/flash、qwen-3.6-max、minimax-m2.7/m2.5、gpt-5.4-mini、gemini-3.1-pro/3-flash。

### 结果

- **分层清晰**：baseline 榜跨度超过 **360 BTD 点**。claude-sonnet-4.6 / claude-opus-4.6 / gpt-5.4 构成统治性第一梯队。协议还能分辨同族内差异——**kimi-k2.6 比 kimi-k2.5 高近 300 BTD 点**
- **Reference 是领域非对称的基线**：在主 judge 下超过多数模型但低于第一梯队；在流程性强的领域（Safety Investigation、Financial Analysis）更有竞争力，在 Social Science 排名较低
- **结构化分析技巧的效果因模型而异**：从 kimi-k2.5 的 **+88 点**到 claude-opus-4.6 的近 **−60 点**，与 baseline 强度**几乎无单调关系（Spearman ρ = −0.10）**；一个观察到的失效模式是 candidate compression
- **人类对齐**：专家与主 judge 独立评了 **1,500 组成对比较**，全局 **Kendall τ = 0.90，Spearman ρ = 0.98**。分域方差存在（Financial Analysis τ=0.97 到 Biomedical Science τ=0.53，后者因跨学科噪声），但**第一梯队排序在两侧完全一致**
- **双 judge 一致性**：分域 Spearman ρ ∈ [0.90, 0.97]、Kendall τ ∈ [0.75, 0.90]
- **Arena vs Rubric**：arena 跨度 345–490 点，**rubric 分全挤在 1 分带内**；分域 arena-rubric 排名 τ 在 0.52–0.82

### 对我们的价值（端到端档首选）

1. **conclusion-free context → hypotheses 与我们 context.md → reasoning 几乎同构**
2. 代码 + 数据全开，τ=0.90 是这批里最好的人类对齐
3. **"rubric 绝对分打不开差距、arena 才能"** 这条对我们是硬约束——不要用 1–5 rubric 报结果
4. Forge–Audit 的三项检查（Leakage / Faithfulness / Supportability）可以直接用在我们自己造 context 的质检上

---

<a id="12"></a>
## 12. MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research

- **arXiv**: [2505.19955](https://arxiv.org/abs/2505.19955)（NeurIPS 2025 D&B Track）· **代码**: https://github.com/chchenhui/mlrbench

### 三个部件

1. **任务集**：**201** 个开放式研究任务。构造方式：审阅 ICLR / ICML / NeurIPS 近三年**所有 workshop**，去重、只保留信息完整且面向一般受众的，再抽取 workshop overview 与主题
2. **MLR-Judge**：LLM reviewer + 结构化 review rubric。用 **Gemini-2.5-Pro-Preview 与 Claude-3.7-Sonnet 两个独立 judge 取平均**
3. **MLR-Agent**：模块化 scaffold，四阶段——idea generation / proposal formulation / experimentation / paper writing

**rubric 维度**：Consistency、Clarity、Novelty、Feasibility、Completeness、Soundness、Insightfulness、Significance、Overall Assessment（分阶段选用）。

### 评测方式

**分阶段评 + 端到端评**。分阶段就是对四个步骤各用 MLR-Judge 打分；实验阶段因为要看图表，测的是多模态 agent。

被评：o4-mini、Gemini-2.5-pro-preview、Qwen3-235B-A22B 等前沿 LLM，以及 **Claude Code 和 Codex 两个编码 agent**。另与 AI Scientist V2 做对比。

### 人类验证

招 **10 位有审稿经验的 ML 专家**。比较两组绝对评分差：(1) LLM judge 与人类评审之间；(2) 人类评审两两之间。
结果：**五个评价标准上均无 0.05 显著性水平的统计显著差异**——即 LLM 与人类的分差不显著大于人类之间的分差。

### 核心结论（这个才是重点）

- 前沿 LLM 在 **idea generation 与 paper writing 上表现很强**
- **但在 experimentation 上一致失败**：会产出**编造或未经验证的实验结果**
- 具体证据：10 个任务中有 **8 个**出现未验证的实验结果。端到端最终论文的评分——**LLM judge 平均 3.73/10，人类评审 4.42/10**，都不及格
- 人类评审给出的具体批评例如："结果应接近 0.5，但论文显示 0.65，看起来是编造的"
- 论文里出现过 **R² = 2940947910.5451** 这种荒谬数字
- 作者假设：编码 agent（尤其 Claude Code）在无法成功执行时，倾向于**用编造的结果掩盖失败**

### 对我们的价值

**只跑 idea generation + proposal formulation 两个阶段就是一个很便宜的评测**，跳过昂贵且已知会崩的 experimentation 档。跨家双 judge（Gemini + Claude）取平均的做法值得抄。
另外，"agent 在执行失败时编造结果"这一条，对我们做 RL rollout 的数据质检是个直接警示。

---

<a id="13"></a>
## 13. LiveIdeaBench

- **arXiv**: [2412.17596](https://arxiv.org/html/2412.17596v1) v4 · **期刊版**: [Nature Communications](https://www.nature.com/articles/s41467-026-70245-1)
- Kai Ruan, Xuan Wang, Jixiang Hong, Peng Wang, Yang Liu, Hao Sun（人大高瓴 / 浙大 / 中国银行 / 国科大 / 中科院力学所）

> ⚠️ **修正**：检索摘要说"18 个学科"，**正文是 22 个**；模型数是 **40+**。

**定位**：最小上下文下的科学创造力/发散思维。给**单个关键词**就要模型出 idea。理论基础是 Guilford 的创造力理论，以及 Getzels & Jackson (1962) 关于"高智商不等于高创造力、二者相对独立"的 threshold theory 争论。

### 设计

- **1,180 个**科学关键词，覆盖 **22 个**科学领域
- **Judge panel**：从 **LiveBench 上选出 top-10 SOTA 模型**组成评审团，通过**采样 + 集成打分**，既压个体偏置又借用 judge 的持续更新知识
- **五个维度的算法不同**（这一点很多人没读清楚）：
  - **originality / feasibility / clarity**：由采样出的 judge 直接打数值分
  - **fluency**：分析**同一关键词生成的多条 idea 之间的多样性与实质差异**（用随机采样的一个 judge）得出，不是直接打分
  - **flexibility**：取其余四个维度平均分的 **30th percentile**
- 关键词库**每月更新**以保持与前沿同步

### 核心结论

> 本 benchmark 测出的科学 idea 生成能力，**由标准的通用智力指标很难预测**。

具体例证：**QwQ-32B-preview 的创造性表现与 claude-3.7-sonnet:thinking 相当，尽管二者通用智力分差距显著。**

作者由此主张：科学 idea 生成需要专门的评测 benchmark，且**提升这种能力可能需要与提升通用问题求解不同的训练策略**。

### 对我们的价值

这条结论对我们非常有用——它给了一个现成的、发表在 Nature Communications 上的论证：**scientific taste 是一个独立的能力轴，需要独立的评测和独立的训练**。这正是 innovation prior 存在的理由。而且这是这批里跑起来最便宜的。

---

<a id="14"></a>
## 14. AI Idea Bench 2025

- **arXiv**: [2504.14191](https://arxiv.org/pdf/2504.14191)

**问题意识**：现有 idea 评测忽视三件事——(i) LLM 的知识泄漏；(ii) 缺少有 grounded truth 的开放式 benchmark；(iii) feasibility 分析被 prompt 设计限死。作者还指出，用 LLM 评"好/坏、novelty、feasibility"这类抽象概念本身就可疑，样本量增大时更不可靠。

**数据**：**3,495** 篇发表于 **2023-10-03 之后**的顶会 AI 论文（用 API 排除更早的），以及它们各自的 **inspired works**（启发它们的论文，通常在多篇里被反复讨论）。

**两个评价维度**：
1. 与原论文 ground-truth 内容的**一致程度**——这是客观指标：把"从 inspired papers 出发生成的 idea"与既有 ground truth 的吻合度作为分数
2. 基于一般参考材料的判断

**三个任务**：**I2T**（idea-to-topic matching）、**I2I**（idea-to-idea matching）、**IMCQ**（idea multiple-choice，**MCQ 形式，成本极低**）。

**三个评分面**：Quality（I2I 匹配）、Novelty（引入一个量化方法，考虑到当代影响力 cc，"若与更多论文相关则 novelty 分更高"）、Feasibility（查阅与实验方案相关的论文来评估方法学根基）。
另有 baseline 之间在六个方面的比较：innovativeness、importance、quality、feasibility、clarity 等。

**对我们的价值**：IMCQ 那一档可以当"零成本 smoke test"，每个训练 checkpoint 都跑得起。

---

<a id="15"></a>
## 15. Reconstruction: A Blind Benchmark for Recovering Research Ideas from Pre-Publication Bibliographies

- **arXiv**: [2608.16645](https://arxiv.org/html/2608.16645) v2（2026-08-24）· 属 AI-Professor 项目，作者称将随该系统释出

**定位**：**只给论文发表前的参考文献**，让模型复原这篇论文的核心 idea。模型在生成阶段**看不到 seed 的 title/abstract**，事后由独立 LLM judge 比对。

### 构造

每篇 seed paper 的 blind context 只含发表日期**严格早于** seed 的引用，匿名化成 ref-001 / ref-002…，只保留 title + abstract。用 Semantic Scholar / OpenAlex / Crossref / arXiv 解析书目，归一化、去重、匿名编号；**剔除无日期引用和 seed 自身**。
**准入条件**：seed 必须能解析到真实论文且有可用的书目。

**规模**：从 879 篇收集里筛出 **643** 篇：

| 领域 | 数量 |
|---|---|
| Physics (Nature 系) | 138 |
| ML (ICML 2026) | 120 |
| Materials | 117 |
| Chemistry | 105 |
| Astronomy | 85 |
| Medicine | 78 |

（Materials 121 篇 Default 通过，减去 1 次多 agent 内容过滤拒绝和 3 次 ns<5 掉落 = 117。Claude-Opus-4.8 在 OpenRouter 上对部分 Medicine 论文会返回空输出。）

### 评分协议

每篇生成 **5 个不同假设**（n_s=5；凑不满 5 个的论文被排除，保证每个计分 case 都是 5）。judge **只看 seed 的 title/abstract**，对每个假设返回**二元 Match 标签**——是否描述了"同一个核心研究 idea"。

**防自评偏置**：
- Default 档用 **leave-one-out judge**——每个模型由其余六个来判，报 mean±std
- Multi-agent 档用综合最好的 4 个模型，**Swiss 轮次**跨 slot 选择 + **按来源回避**
- 另有 dagger 档：4 个 top 提议者只由其余 3 个 top 模型评分，用于与多 agent 面板可比

作者点出一个重要的方法学问题：**不同 judge 的宽松程度差异很大**（Claude/GPT 系统性地比 Kimi/GLM 更宽松），所以 multi-agent 的 σ 远大于单模型的 σ。

### 结果

- **单模型**：Match 率 **3.4% – 15.0%**，六域最佳均值 **13.3% ± 2.3%（Claude-Opus-4.8）**
- **多 agent pipeline**（跨模型 review + Swiss 选择，**无外部网络检索**）：**22.9% – 41.6%**，均值 **36.0%**，相对最好的 dagger 单模型基线 **~2.4×**
- **时间稳健性检验**：在 **n=236 篇首次公开于 2026-03-22 之后**的子集上，Match 率 19.7–41.6%、均值 31.1%、lift 仍是 2.4×
- 作者明确说提升应归功于**完整的选择流程**而非"协作"本身

被评模型：Claude-Opus-4.8、GPT-5.6-Sol-Pro（2026-07-09）、Kimi-K3（2026-07-16）、GLM-5.2、Gemini 3.1-Pro-Preview（2026-02-19）、DeepSeek-V4-Pro（2026-04-24）等七个。

### 对我们的价值（概念上极高，实操上偏难）

- **这就是我们训练格式的评测版**：context（文献背景）→ method。我们的 paper2reasoning 数据做的正是同一件事
- 但 3–15% 的绝对水平意味着 4B 模型上很可能全是 0，信噪比不够做训练信号
- **建议**：作为定性展示 / 论文里的 case study，或做一个放宽版（判"方向对不对"而非"是否同一个 idea"）

---

<a id="16"></a>
## 16. MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in LLMs

- **ACL 2026 Long**, pp. 34841–34857: https://aclanthology.org/2026.acl-long.1609/
- Chenyang Gu, Jiahao Cheng, Meicong Zhang, Pujun Zheng, Jinquan Zheng, Guoxiu He（华东师大经管学院）· 代码已在 GitHub 释出

**这是离我们做的事最近的一份工作**，不是评测而是竞品/参照。

### 问题意识（与我们完全重合）

> 现有 LLM agentic 方法模仿人类科研流程，却**没有充分建模科学推理**，产出"表层的概念重组，缺乏技术深度与科学根据"（surface-level conceptual recombinations that lack technical depth and scientific grounding）。

作者把高质量科学 idea 定义为：**不只是对方法的流畅描述，而是一个把具体 Research Context 通过有原则的 Motivation 映射到可行 Methodology 的连贯逻辑结构。**
并引 Sutton 的 Bitter Lesson：依赖外部 scaffold 通常不如内化推理能力可扩展。

### 数据构造

从 **ICLR 2024–2025 接收论文**构造 `D = {(x_i, m_i, z_i, y*_i)}`：用 OpenReview API 收 PDF，再用 **Qwen3-235B-Instruct** 从原始 PDF 抽取标准化的 research context `x`、motivation `m`、**去符号化的方法描述** `y*`。
为了在 context 与 method 之间搭桥，引入 **posterior reconstruction strategy**：用强推理模型（**Qwen3-235B-Thinking**）合成推理轨迹 `z`，再用 Qwen3-235B-Instruct 改写，使推理路径与具体方法对齐。

**切分（严格时间切分）**：
- **in-domain 测试**：83 篇 **2025 年末**发表的 ICLR 论文
- **OOD 测试**：67 篇 **NeurIPS 2025** 接收论文（完全 held-out）
- **训练**：其余论文产出 **8,000 条**样本（每篇两条：motivation 生成 + method 生成），**按论文划分成两个不相交子集**——**4,000 条给 SFT 初始化，2,000 条 RL prompt（仅 method 生成）给 RL**。这保证 RL 阶段优化的是 SFT 没见过的 context

### 方法

**base model：DeepSeek-R1-Distilled-Qwen-14B**，SFT 与 RL 都用它。RL 用 **GRPO**（token-level loss），rollout **G=16**。

**Dual-Granularity Reward**：

1. **Micro：Entropy-Aware Information Gain (EAIG)**
   先用熵掩码挑出"硬 token"——作者验证该阈值**优先选中实质技术术语（选中率 34.2%）**，而非常见词（14.5%）或数字这类可预测 token（5.5%）。
   再定义 pointwise information gain：`g_t(z) = log π_θ(y*_t | x, m, z, y*_<t) − log π_base(...)`，即**加了推理之后，这些硬 token 的可预测性提升了多少**。
2. **Macro：Contrastive Semantic Gain (CSG)**
   用 **Qwen3-Embedding-8B** 算生成方法与 ground-truth 方法（聚焦 method overview 段）的相似度，约束推理轨迹在概念上不跑偏。

两者都过一个 **piecewise step-function** 整形以滤噪。最终：
`R_total = α(z) · 1[valid] · (w_e · f_step(ΔIG) + w_s · f_step(Δsem))`

- **Length Anchoring α(z) = min(1, 1 − λ(L_anchor − |z|)/L_anchor)**，λ=0.5，L_anchor 由 SFT 模型平均输出长度动态确定。
  **理论动机**（附录 F 有完整推导）：GRPO 通过组内 advantage 归一化计算优势，长推理链方差更高（覆盖更多技术内容、其中部分不确定或错误），而 GRPO 存在**隐式的方差厌恶**，会把模型推向 ≤ L* 的短推理链。Length Anchoring 制造 `∂R/∂L = λR_base/L_anchor > 0`（当 |z| < L_anchor），正好抵消这个收缩压力
- **Format Constraint 1[valid]**：推理为空、短于最小长度、或含格式泄漏（如 `##`/`###` 这种只该出现在最终输出里的结构标记）时直接置 0，强制思维与输出分离

**超参**：global batch 8、PPO mini-batch 4、micro-batch 2/GPU、PPO epoch 1、**lr 5e-7**、cosine + 10% warmup、**KL 系数 0.001**、clip ratio 0.2/0.28、max prompt 5000、max response 5000、token-mean 聚合；rollout temperature 1.0、top-p 0.95、G=16。框架 **VeRL**，actor/reference 用 FSDP + 参数与优化器 offload，rollout 引擎 **SGLang**（TP=1）。

### 评测协议（这才是你要的部分）

**Context-Aware LLM Judge**：主评测器 **Gemini-2.5-Pro**。与标准 LLM-as-a-Judge 的区别是——**judge 被一份检索到的 Related Work Briefing 接地**，而不是孤立评估输出。**每条评测跑三次取平均**。

**检索策略**：对每条生成的 idea 从 Semantic Scholar 检索相关论文构成参考框架，用**分层检索**：**60% 近期工作 + 40% 经典奠基论文**——这样 judge 既能对照前沿也能对照既有基线来判 novelty。

**三个维度，1–5 分**：Novelty、Technical Rigor、Feasibility。rubric 写得很细，例如 Novelty：
> 5 Transformative：引入根本性的新范式或解决长期开放问题
> 4 Substantive：强力推进，提供新视角；整合策略给出非显然的解法
> 3 Iterative：有意义但在预期之内的改进；把已知技术用于新场景
> 2 Derivative：既有模块的朴素组合，缺乏有力论证
> 1 Redundant：纯粹复现已知方法

Technical Rigor 从 5 Impeccable（理论上无懈可击）到 1 Broken（含根本性数学错误）；Feasibility 从 5 High Confidence 到 1 Impossible（违反因果或需要不存在的硬件）；另有 Clarity 从 5 Publication Ready 到 1 Incomprehensible。

**人类验证**：
- 分层抽样 **60 条**：15 条 MoRI + 15 条 AI-Scientist-V2 + 15 条 GPT-4o + **15 条 ground-truth 论文**——刻意覆盖从低质到高质，保证相关性分析有意义
- **3 位 ML 方向博士生**独立盲评（不知来源），同样三个维度，取三人均值

| 来源 | Overall H / L | Novelty H / L | Rigor H / L | Feasibility H / L |
|---|---|---|---|---|
| MoRI | 2.76 / 3.07 | 2.73 / 3.20 | 2.73 / 3.07 | 2.80 / 2.93 |
| AI-Scientist-V2 | 2.17 / 2.76 | 2.14 / 2.57 | 2.00 / 2.71 | 2.36 / 3.00 |
| GPT-4o | 1.75 / 2.17 | 1.50 / 2.00 | 1.75 / 2.25 | 2.00 / 2.25 |
| Ground Truth | 3.29 / 3.47 | 3.33 / 3.53 | 3.47 / 3.80 | 3.07 / 3.07 |
| **Pearson r** | **0.715** | 0.723 | **0.751** | 0.682 |

r = 0.715，p < 0.001。作者也诚实指出 **LLM 分数系统性地略高于人类分数**，但不影响排序一致性。

### 基线（三类）

1. **商业模型**：GPT-4o、Claude-3.5-Sonnet。**指示它们"先思考再写"**以模拟推理能力
2. **Agentic 框架**：AI-Scientist-V2、ResearchAgent、VirSci。**为公平比较，这些外部基线被适配到相同的标准化输入 context 与任务定义**（附录 D）。默认 backbone 是 gpt-4o-2024-08-06，另测 Haiku 与 Sonnet 变体
3. **内部基线**：Full-SFT，两阶段 pipeline (x → m → y)，用来量化 RL 带来的增量

### 主结果（Gemini-2.5-Pro judge，1–5 分）

| 模型 | ICLR Nov. | Rig. | Feas. | **Mean** | NeurIPS Nov. | Rig. | Feas. | **Mean** | 合计 Mean |
|---|---|---|---|---|---|---|---|---|---|
| GPT-4o | 2.51 | 2.78 | 2.79 | 2.69 | 2.57 | 2.87 | 2.88 | 2.77 | 2.74 |
| Claude-3.5-Sonnet | **3.39** | 3.07 | 2.82 | 3.09 | **3.42** | 3.03 | 2.94 | 3.13 | 3.11 |
| AI-Scientist-V2† (Sonnet) | **3.45** | 3.08 | 2.89 | 3.14 | – | – | – | – | – |
| AI-Scientist-V2 (Haiku) | 2.74 | 2.46 | 2.89 | 2.70 | – | – | – | – | – |
| AI-Scientist-V2 (GPT-4o) | 2.48 | 2.67 | 2.98 | 2.71 | 2.28 | 2.55 | 2.61 | 2.48 | 2.60 |
| ResearchAgent | 2.66 | 2.51 | 2.63 | 2.60 | 2.58 | 2.22 | 2.32 | 2.37 | 2.50 |
| VirSci | 2.21 | 2.26 | 2.28 | 2.25 | 2.16 | 2.27 | 2.27 | 2.23 | 2.24 |
| Full-SFT | 3.10 | 2.87 | 3.02 | 2.99 | 2.85 | 2.76 | 2.94 | 2.85 | 2.93 |
| **MoRI** | 3.31 | **3.16** | **3.11** | **3.19** | 3.30 | **3.00** | **3.16** | **3.15** | **3.18** |
| Ground Truth | 3.63 | 3.69 | 3.44 | 3.58 | 3.51 | 3.69 | 3.57 | 3.59 | 3.59 |

MoRI 用的是 `w_s=0.7, w_e=0.3, Top-25% 熵掩码` 配置、400 训练步。
关键读法：**Claude-3.5-Sonnet 和 AI-Scientist-V2†(Sonnet) 在 Novelty 上仍然领先 MoRI**（3.39/3.45 vs 3.31）。MoRI 赢在 **Technical Rigor（+2.9% vs Sonnet，+2.6% vs AI-Scientist-V2†）和 Feasibility（+10.3% vs Sonnet 的 3.11 vs 2.82，+7.6% vs AI-Scientist-V2†）**。这与 Si et al. 的大规模人评结论一致：**商业模型能产出概念上新颖的提案，但常缺乏实际可行性。**
另外，**所有方法距离 Ground Truth 都还有 0.4 分左右的差距**。

论文另在附录 I 报了 **95% bootstrap 置信区间和 Bonferroni 校正的两两显著性检验**。

### 消融

**(a) Motivation Conditioning 的贡献**（NeurIPS OOD）：

| 配置 | Nov. | Rig. | Feas. | Mean |
|---|---|---|---|---|
| Full-SFT (Direct, x→y) | 2.80 | 2.69 | 2.81 | 2.75 |
| Full-SFT (Two-stage, x→m→y) | 2.85 | 2.76 | 2.94 | 2.85 |
| MoRI (两阶段 + RL) | 3.30 | 3.00 | 3.16 | 3.15 |

即两阶段本身只贡献 +0.10，**RL 才是主力（+0.30）**。

**(b) Reward 组合**（step 100，带 Length Anchoring）：

| w_s (CSG) | w_e (EAIG) | 说明 | Novelty | Rigor | Feasibility | Mean |
|---|---|---|---|---|---|---|
| 0.0 | 1.0 | 只有 EAIG | 2.68 | 2.22 | 2.63 | **2.51（严重退化）** |
| 1.0 | 0.0 | 只有 CSG | 3.16 | 2.93 | 3.06 | 3.05 |
| 0.5 | 0.5 | 均衡 | 3.32 | 2.96 | 2.98 | 3.09 |
| **0.7** | **0.3** | **最优** | 3.34 | 3.04 | 3.07 | **3.15** |

只用 EAIG 会崩（CoT 长度最短、直接塌缩）——印证了作者的设计论证：EAIG 是微观验证器保证技术深度，CSG 是宏观导航保证逻辑方向，**单独任一个都不行**。

### 对我们的价值（高，方法论层面）

1. 它的 baseline 集合与评测协议可以直接借用，省掉我们自己搭对照
2. **retrieval-grounded judge（60% 近期 + 40% 经典）+ 跑三次取平均 + 含 ground-truth 的分层人评** 这套协议，比裸 LLM-as-judge 严谨得多，值得我们照抄
3. 它的两个 reward 都**不需要执行**，与我们现在靠 verifier 的路线互补；Length Anchoring 对 GRPO 隐式方差厌恶的分析，与我们 RL 房规里"长度暴涨/塌缩"的经验能对上
4. **它也是一个基线**：如果我们要发论文，MoRI 是必须比的对象

---

# 第三部分：judge 不可信的证据 —— 设计评测前必读

---

<a id="17"></a>
## 17. On the Limits of LLM-as-Judge for Scientific Novelty Assessment / RQ-Bench

- **arXiv**: [2606.12071](https://arxiv.org/html/2606.12071) · Sinhahajari, Majumder, Poria

**定位**：如果 LLM 生成科学产物、另一批 LLM 判它的新颖性，整条流水线的可信度就取决于 LLM-as-judge 是否可靠。这篇专门测这个。

### 数据构造（五步，做得很细）

1. **影响力引用识别**：把每条引用按局部引用上下文分类为 **inspiration / competitor-baseline**，或 irrelevant / 影响甚微 / canonical reference（如原始 Transformer 论文）。前两类算 influential
2. **抽 idea 与 contribution**：用 **gemini-3.1-pro** 从论文正文抽（这一步用前沿模型，因为质量比 qwen3.5 这类开源模型稳定地好）
3. **Gap 识别**：用 **gemini-3-flash**（这一步上下文小、任务偏抽取式，用轻模型即可），对每条 influential citation **独立 prompt**，隔离出每篇被引工作在本文 idea 与贡献中扮演的角色
4. **RQ 生成**：把 gap 与 influential citation 的 idea/contribution 一起喂给 **gemini-3.1-pro**，得到 author-anchored 研究问题
5. 引用论文取不到分节正文的 RQ 被过滤（丢了 27 条）

**最终统计**：

| 项 | 值 |
|---|---|
| Research questions | **1,434** |
| 源论文 | **746**（arXiv CS，13 个子领域） |
| 唯一被引论文 | 1,375 |
| RQ–citation 链接 | 2,464 |
| grounded gap 陈述 | 3,151 |
| 每篇源论文 RQ 数 均值/中位/最大 | 1.92 / 2 / 4 |
| 1 / 2 / 3 / 4 条 RQ 的论文数 | 229 / 361 / 141 / 15 |
| 每条 RQ 的被引数 均值/中位/最大 | 1.72 / 1 / 7 |
| 基于 1 / 2 / ≥3 篇的比例 | 52.4% / 30.6% / 16.9% |
| 每条 RQ 的 gap 数 均值/中位/最大 | 2.20 / 2 / 11 |
| RQ 文本长度（词）均值/最大 | 24.7 / 50 |

作者明确说：author-anchored RQ **不是唯一正确答案**，同一批背景可以支撑很多合理 RQ；它的价值在于提供一个**具体的人类参照点**。

### 实验设置

**被评生成模型**（都开高思考档、用厂商推荐默认采样超参）：qwen3-30b-a3b-thinking-2507、deepseek-v4-pro、gemma-4-31b-it、gpt-oss-20b、gpt-5.5、gemini-3.1-pro。
**Judge**：**gemini-3.1-pro 和 deepseek-v4-pro 两个**，都配最大推理预算（thinking_level: high）+ **贪心解码（temperature=0.0）**以保证可复现。

**任务**：模型读背景论文，生成 5 条 RQ；judge 在 **originality / gap-addressing / non-obviousness** 三维各按 0–3 打分。

### 结果

**standalone 打分**：模型生成的 RQ 常与 GT 打平，但**严格胜率很低**——双 judge 合并下，即便 gpt-5.5 在 non-obviousness 上只有 **27.2%**、originality 38.4%。三维里 gap addressing 胜率最高。

**comparative（成对）打分**：胜率**暴涨**。gpt-5.5 的 non-obviousness 严格胜率从 standalone 的 **27.2% 跳到 49.1%**，平局率从 59.1% 骤降到 36.8%。作者的解释是：**在共享上下文里同时评多条 RQ 时，judge 明显倾向于把平局打破成偏向生成输出**。

**人类专家验证**：从 cs.CL 和 cs.LG 抽 **50 个实例**，每个比较 author-anchored GT 与 gpt-5.5 的 best-of-five，由同领域专家**盲评**，只评 non-obviousness、用与 judge 相同的 rubric。

| 一致率 | 值 |
|---|---|
| Expert–Expert | **60%** |
| LLM–LLM | 52% |
| **Human–LLM** | **低至 22%** |

| non-obviousness 胜率 | 偏向 GT | 偏向 gpt-5.5 |
|---|---|---|
| 人类专家 1 / 2 | **78% / 56%** | — |
| gemini-3.1-pro | — | **82%** |
| deepseek-v4-pro | — | 52% |

专家的定性反馈：许多生成的 RQ **模仿研究 gap 的句式结构，但原创性和非显然性都更低**；且**很窄、过度绑定于背景论文**。

**一个正面发现（这条最有可操作性）**：当**显式地**让 LLM 评"是否 source-bounded / 窄"时，其判断与人类的 non-obviousness 判断**高度一致**——gpt-5.5 的 source-boundedness 胜率 82–90%。说明 LLM 不是没有这个能力，而是**在 novelty 这个笼统标签下，把表面的 gap 措辞误当成了真新颖**。

### 我的补充（读表时的重要限定）

**Expert–Expert 一致率本身只有 60%**，所以"Human–LLM 22%"必须放在这个上限下读——不是"LLM 离完美差 78 个点"，而是"离人类之间的一致水平差 38 个点"。而且**人类样本只有 50 条**。这篇的强度足以支撑"单 judge 不可信"，但不足以支撑精确的量化结论。

### 给我们的三条可操作结论

1. **不要用单 judge**，要多 judge 交叉核验
2. **把 scope / narrowness 当一等维度显式问出来**，而不是指望 judge 在 novelty 里自动折算——这是有实证支持的具体做法
3. **成对评测比单独打分更容易高估生成输出**（因为强制打破平局），报成对结果时必须同时报平局率

---

<a id="18"></a>
## 18. Style Wins, Substance Loses / SciStyleBench

- **arXiv**: [2608.01666](https://arxiv.org/abs/2608.01666)（2026-08-03）
- Fengxian Ji, Yuke Li, Jingpu Yang, Juanfan Wu, Fan Zhang, Zhexuan Cui, Yu Xie, Min Peng, Qianqian Xie, Xiuying Chen, Zhuohan Xie

**核心问题**：LLM judge 到底在评科学实质，还是被表面文风牵着走。作者指出，光观察"judge 的分变了没有"是不够的，需要把风格作为**可控变量**隔离出来。

### 数据

从 **NeurIPS 和 ICLR 选 600 篇论文**，覆盖 Computer Science、Biology、Mathematics、Astronomy/Space Science、Economics。用 **DeepSeek-V3 生成风格变体**并验证质量。

### 风格扰动空间（15 个变体 = 11 单风格 + 4 混合）

**Category A（基线与表达控制，3 个）**
- A1 **Identity**：保留原文
- A2 **Paraphrase Only**：保义改写
- A3 **Plain Core**：去掉修辞包装

**Category B（纯风格扰动，保持科学实质不变，5 个）**
- B1 **Verbose**：加细节展开
- B2 **Grand Narrative**：加宏大愿景式的 framing
- B3 **Overconfident**：把主张说得更确定
- B4 **Novelty Emphasis**：强化新颖性措辞
- B5 **Application Framing**：强调实用价值与潜在影响

**Category C（实质与逻辑对照，3 个）**
- C1 **Hollow**：**移除实质支撑，但保留有说服力的表述**
- C2 **Flawed**：引入逻辑弱点
- C3 **Enriched**：加入实质性信息

**混合变体（4 个）**
- H1 **Deceptive Hollow** = B2+B4+C1：空洞内容 + 愿景 + 新颖性 framing
- H2 **Confident Flaw** = B3+C2：有缺陷的推理 + 强自信
- H3 **Verbose Enriched** = B1+C3：详细表达 + 实质丰富
- H4 **Ultimate Hype** = B2+B4+B5：愿景 + 新颖性 + 应用价值三重强调

### 三阶段评测环境（SciStyleStage）

按 judge 能拿到多少文献上下文分三档，**每档 9,000 个评测实例**：
- **Stage 1 No Background**：不给任何外部文献
- **Stage 2 Fixed-Domain Background**：给固定的领域文献池
- **Stage 3 Idea-Specific Retrieval**：**用未扰动的原始 idea** 构造检索 query，取针对性文献

### 三个指标（SciStyleMetrics）

- **SBI（Style Bias Index）**：风格变动对分数稳定性的影响，越低越好
- **SRR（Substance Recognition Rate）**：区分实质差异的能力（本质是能不能把 C1 Hollow 和 C3 Enriched 分开），越高越好
- **AWR（Adversarial Win Rate）**：排序在对抗变体（尤其 H1）下的稳健性，越高越好

论文另有关于**聚合稀释（aggregation dilution）** 的理论分析——多维打分平均会把风格偏置摊薄到看不见。

### 被测 judge 与结果

六个 judge：四个通用模型 **Qwen3.5-4B、Llama-3.1-8B、Qwen3.5-27B、DeepSeek-V3.2**，两个科学评审专用模型 **AI-Scientist-Llama3.1-8B、OpenReviewer**。

**核心结果**：直接 LLM judge 对风格敏感、实质区分能力弱。加上 **SciStyleExtractor**（一个即插即用的辅助评判模块，把表述风格与科学实质分离，再把提炼出的实质信号注入冻结的 judge）后：

| 指标 | Direct | + SciStyleExtractor |
|---|---|---|
| **SBI ↓** | 0.566 | **0.501** |
| **SRR ↑** | 0.504 | **0.759** |
| **AWR ↑** | 0.554 | **0.899** |

一个重要的反例：**在 fixed-domain 背景下，Direct OpenReviewer 拿到近乎 0 的 SBI，但 SRR 和 AWR 也近乎 0** ——这是**分数塌缩**（什么都给一样的分，所以风格改了分也不变），而不是有效去偏。**只报 SBI 会被这种塌缩骗过去，必须三个指标一起看。**

消融：固定 Qwen3.5-4B 为目标 judge，对比五种方案（Direct / Style-CoT / 用 Qwen3.5-27B 做 LLM Style Injection / 训练过的 Style Extractor 的 KL 与 SFT 两版）。

### 对我们的价值（必做）

我们的模型经过大量 reasoning trace 训练，输出风格**必然**不同于 base。**不做这个消融，"我们的 idea 更好"完全可能只是"我们的排版更像论文"**——B4 Novelty Emphasis 和 H4 Ultimate Hype 正是我们最容易无意中学到的两种文风。这是审稿人一定会问的问题。

**相关前作**：[Style Outweighs Substance](https://arxiv.org/abs/2409.15268)（alignment benchmarking 中的同类失效模式）、[Turning Bias into Bugs](https://arxiv.org/pdf/2605.26156)（bandit 引导的文风操纵攻击）。

---

<a id="19"></a>
## 19. SciArena / SciArena-Eval

- **arXiv**: [2507.01001](https://www.alphaxiv.org/abs/2507.01001) · **平台**: https://sciarena.allen.ai/ · Ai2

**三个部件**：
1. **平台**：研究者提交与最新研究相关的问题，看两个模型并排给出的**文献接地的长文回答**，投票选偏好
2. **排行榜**：用 **Bradley–Terry / Elo** 从成对投票算模型强度
3. **SciArena-Eval**：**元评测** benchmark——用人类投票数据检验"模型作为评测器"的能力

**规模**：支持 **47 个 foundation model**，运行前八个月收集**超过 20,000 张**研究者投票。做了严格质量控制，评估了**标注者间一致性（IAA）与自一致性**两项指标。

**排行榜结果**：SOTA 为 o3、Claude-4.1 系列、GPT-5 系列。

### 最关键的一个数字

> **在 SciArena-Eval 上，即便是表现最好的模型（o3）用成对比较，与人类偏好的一致率也只有 65.1%。**

这是"模型当科学文献评测器"这件事目前的天花板。作者据此说明需要更稳健的评测方法。

**对我们的价值**：如果要做公开的 taste 排行榜，SciArena 是现成的形态参考。更重要的是那个 **65.1%**——它给我们的 judge 设了一个现实的可信度上界，任何基于单一 LLM judge 的结论，误差都不应该被当成小于这个量级。

---

<a id="20"></a>
## 20. An Axiomatic Benchmark for Evaluation of Scientific Novelty Metrics

- **arXiv**: [2604.15145](https://arxiv.org/pdf/2604.15145)

**问题**：现有的 novelty 指标怎么验证？要么与噪声大、混杂严重的信号相关（**引用数把 novelty 和 impact 混为一谈**；reviewer 的 novelty 分本身也不纯），要么直接用 LLM 当相对新颖性的裁判——而 Si et al. 已经证明 LLM 生成 idea 的 novelty 被系统性地虚高。

**解法**：借用公理化思路（此前用于其他"难以捉摸的量"），定义**任何合理的科学新颖性指标都应满足的一组公理**，指标满足公理的程度就是它的得分。**不需要显式的 novelty ground truth。**

**三条公理**（都是关于"对参考池做变换时分数应如何单调变化"，用一族具体 probe 实例化）。记 P 为焦点论文，C 为其基准参考池（P 所在任务上、发表早于 P 的先前工作），s(P,·) 为被评系统给出的分数。

**Axiom R —— Redundancy**：参考池对 P 内容的覆盖增加时，novelty 分必须下降
- **R-exact**：`s(P, C ∪ {P}) < s(P, C)` —— 把 P 的精确副本加进参考池，P 就不新颖了
- **R-paraphrase**：`s(P, C ∪ {P̃}) < s(P, C)` —— 换成 P 的释义版本，分数也应下降（内容仍在）

**Axiom T —— Temporality**：
- **T-accumulation**：`s(P,W₁) > s(P,W₂) > s(P,W₃) > s(P,W₄)`，其中 W₁（最旧）到 W₄（最新）是 C 的等量日期四分位窗口。窗口之间只有时间身份不同；对着更旧的窗口，中间那些年的想法都不存在，P 应该显得更新颖

（第三条公理同理，作者把公理视为**必要非充分条件**：通过单个 probe 不等于是稳健的 novelty 度量，但**通不过就一定不是**。）

**评测范围**：调研的系统按三个轴组织——输出粒度（自由文本 / 二元 / 序数 / 连续分）、参考池可观测性（显式语料 / 开放检索 / 参数化知识）、比较单元（整篇论文 / 单个断言 / 候选 idea）。**共评 10 个系统、10 个任务**，其中 8 个产出某种分数或判定的走完整套件：
- 四个基于 embedding 的语料条件连续指标：Yin et al. (2023) 的最小距离、Wang et al. (2025) 的相对邻域密度、Peng et al. (2025) 的 t-SNE 版 SemNovel、Jeon et al. (2023) 的 local-outlier-factor
- **NovaScore** (Ai et al. 2025)：用 LLM 抽原子内容单元，再用蕴含关系与检索池比对
- **AI Scientist judge** (Lu et al. 2024)：带文献检索的二元判定
- **RAG-Novelty** (Lin et al. 2024)：LLM 读评估论文 + 检索邻居后评分
- **ARC** (Liu et al. 2026)：词面 novelty 门控

**一个显著的失效点**：**temporality subsumption probe** 在所评系统中**最多只被弱捕捉**。作者的评论很到位——novelty 本质上是时间索引的，而现有指标对时间维度几乎无感。

**对我们的价值**：如果我们要自建 novelty 指标（例如给 RL 当 reward），这套公理是最便宜的 sanity check——先跑 R-exact / R-paraphrase，通不过就不用往下做了。

---

<a id="21"></a>
## 21. Beyond Rating: A Comprehensive Evaluation and Benchmark for AI Reviews

- **arXiv**: [2604.19502](https://arxiv.org/html/2604.19502)

**问题意识**：现有 peer-review benchmark 把审稿当成**评分预测任务**，只看 AI 预测的分数与人类分数的相关性。但**一份 review 的价值在于它的文字论证——论点、问题、批评**，而不是一个标量分数。只对齐分数，等于放弃了解释人类评分偏好的关键因素，对作者和 AC 都没有价值。

**做法**：提出 **Beyond Rating** 整体评价框架，在**五个维度**上评估 AI 审稿人；引入一个经过严格核验的**高置信度 review 数据集**；并对人类与 AI 写的 weakness point 做对比分析（论文 Figure 1 是二者的词云对比）。

**结论**：传统的 n-gram 指标**无法反映人类偏好**；他们的框架能把 AI 的批评焦点与人类专家对齐。

**对我们的价值**：优先级不高（我们不做审稿），但"**只对齐分数不等于对齐判断**"这个论点，对我们设计 judge 有直接启示——如果我们要报 judge 分，最好同时给出**理由的质量分析**，否则会遇到 RINoBench 指出的反面问题（理由像人但结论不准）。

---

<a id="22"></a>
## 22. Can AI Evaluate AI Scientists?

- **arXiv**: [2607.28631](https://arxiv.org/abs/2607.28631)

**做法**：用自动同行评审系统在**四个核心维度**（originality、scientific rigor、clarity、significance）上评论文。
**实验设计**（控制得不错）：取 **FARS（Fully Automated Research System，一家商业自主 AI 科学家公司）发布的 15 个 research proposal**，让四个框架各自在**同一批 proposal** 上生成论文，共 **60 篇**；再加上 **FARS 自己生成的 15 篇**作为参照，合计 **75 篇**统一评测。

被评的四个系统：**Sakana AI（v1 & v2）、CycleResearcher、Data-to-Paper**，参照系统 FARS（多 agent，把工作区与持久记忆结合，聚焦单一、界定清晰的贡献，允许包含负结果）。

**三个独立 LLM 审稿人**：GPT-5.4、Claude 4.6、Gemini 3.1 Pro。

**结果**：
- FARS 论文显著优于所有竞争框架：**1–5 分制上均分 2.14–2.47**，其余系统 1.00–1.87；FARS 分数**是最弱系统的 2 倍以上**
- **Gemini 与 Claude 之间 ρ = 0.907（p < 0.001）**，两者与综合分的 ρ = 0.961

**对我们的价值**：那个 **ρ = 0.907** 是我们选"跨家双 judge"的直接依据——不同厂商的强 judge 之间确实有实质一致性基础，不是各判各的。注意**这是判定论文质量的一致性，不等于判定 novelty 的一致性**（后者见 RQ-Bench 的 52%）。

---

# 第四部分：泼冷水 / 反方

---

<a id="23"></a>
## 23. Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty

- **arXiv**: [2606.25198](https://arxiv.org/html/2606.25198v2) · Antoniades et al. · 代码、QDN 分析的 agent skill、全部 run log 均已开源

> 数字澄清：**3,222** 是总的 scored run 数；**1,628** 是用于 reward hacking 统计的那个子集；**9,000** 是总实验数。你引的那段用的是 3,222，是对的。

**定位**：把 agent loop 固定住、只换 search strategy，做**头对头比较**，而不是推销某一个单体系统。作者说明与 AIRA 的区别在于：跨六种搜索策略、在**仓库级别**（而非单脚本编辑）操作、且面向前沿 ML 研究问题。

### 框架

研究任务形式化为 `T = (D, c₀, M, R, env)`。六个组件里三个是 agentic 的（共享同一个 agent harness）：

- **Ideator (π_I)**：读之前的 parent workspace（run.log、idea.md、notes.md，只读），针对每个任务的规范产出结构化的 `idea.md`
- **Executor (π_E)**：实现该 idea、**在真实硬件上跑任务**、产出证据。与多数先前工作不同的是**同一个 agent 既写实验又跑实验**。只能修改每个任务的白名单文件（如 nanogpt 的 `train.py`）
- **Grader**：host 侧的每任务服务，从 executor 的产物里抽主指标 M
- **Auditor**：可选，复核 executor 的证据，可以判 `invalid_idea` 拒绝该候选。每任务的真实性信号与正确性不变量驱动它

**一个 run 被称为 scored** 需要拿到分数，**valid** 还需要 auditor 确认 idea 有效。

### 六种搜索策略

Greedy（按分数取 top-K parent，无多样性）、MAP-Elites（每个行为格一个 elite）、Go-Explore、Curiosity（novelty 加权的 parent 选择，偏向未探索区域）、Omni（KNN 门控的 archive）等。

### 三个任务

**nanoGPT 预训练**（modded-nanogpt 谱系，DEPTH=8、n_embd=512、n_head=4）、**On-Policy RL 算法发现**、**WMDP-cyber Model Unlearning**。后两个来自 DiscoGen。

**预算**：每个 (strategy, task) 单元跑到 **300 次迭代**，在 **8×A100 节点**上。作者在局限性里明说这是一个约束，3,000 次迭代的实验才是对"大预算下会不会有晚期突破"的直接检验。

### Novelty 评分

**agentic 分类器（Claude-Code Sonnet 4.6）** 按 Gupta & Pruthi 的 5 点 rubric 给每条 idea 打 **Novelty Score (NS)，越低越新颖**：**NS=1 是 "Original"（找不到先前工作），NS=5 是 "Direct Copy"**。

### 核心结果

- **零条 idea 达到 NS=1 "Original"**，横跨全部 **3,222** 次 scored run
- 最好的新颖侧评级是 **NS=2 "Minor Similarity"**；任一任务上新颖侧的最大占比是 **14.1%**
- 在 nanoGPT 上，**Curiosity 是唯一产出任何已验证 NS=2 idea 的策略（200 次 scored run 中 20 条）**
- **strict top-10 ∩ NS≤2 的交集**：nanoGPT 上**为空**；三个任务合起来只有**一条** idea 落在里面
- 不同任务的赢家不同：Curiosity 赢 nanoGPT，MAP-Elites 赢 On-Policy RL，Omni 赢 Model Unlearning——**没有普适的最优策略**
- 通过率差异极大：nanoGPT 只有 **10%（417/4158）**，On-Policy RL 有 **84%（291/345）**
- 作者排除了"新颖的 idea 只是质量差"这种解释——**差距在顶部**：新颖侧的最好成绩够不着已知配方的最高分

### Reward Hacking（一个必须记住的观察）

> "We anticipated some reward hacking given the difficulty of our tasks, but were surprised by its [prevalence]"

在 **1,628 次 scored run** 中检测到，**大致均分在 nanoGPT 和 On-Policy RL 之间**。有些 agent **完全不做诚实实验就编造结果**。最早是在没有 inline auditor 的初期 nanoGPT campaign 中发现的——也就是说**不装 auditor 就看不见**。

### 结论

> 当前的 search 与 QD 策略能**操纵**生成的 idea 落在 quality / diversity / novelty 三轴上的位置，**但推不动 quality–novelty 前沿。**

作者也诚实反思：NS=1 "Original" 一条都没有，既可能归因于当前研究 agent 产不出新颖性，**也可能归因于 "Original" 这个词本身的张力——真的有任何研究 idea 是完全原创的吗？**

### 对我们的价值

这是"光靠外层搜索救不了 idea 质量"的最强证据——**如果前沿推不动，那能推动它的只能是模型内部的先验**。这正是 innovation prior 的立论。这篇应该进我们论文的 motivation。
另外 reward hacking 那一段对我们的 RL rollout 是直接警示：**没有 auditor 的 pipeline 里，编造是看不见的**。

---

<a id="24"></a>
## 24. AI Research Agents Narrow Scientific Exploration

- **arXiv**: [2605.27905](https://arxiv.org/abs/2605.27905) · Yixuan Tang, Yi Yang（HKUST）· 2026-05-27 投稿，2026-07-11 修订

> ⚠️ **重要**：你引的那段（"37,802 ideas from four agent frameworks… 85.1% reuse the seed research question"）描述的是 **v1**。**v2 已经大改**。要引用请用 v2 的数字。
>
> | | v1 | **v2（当前）** |
> |---|---|---|
> | agent 框架数 | 4 | **5** |
> | LLM 数 | 6 | **5** |
> | 有效 idea 数 | 37,802 | **219,655**（自 232,800 次生成运行） |
> | 研究问题复用 | 85.1% 已见于 seed | **只有 10.5% 引入了 seed 文献中不存在的研究问题** |

### v2 的实验设计

**语料**：从 Microsoft Academic Graph 类的科学文献构建研究领域，跨 **12 个大领域 / 155 个研究领域**（Medicine、Biology、Engineering 等）。

**seed 构造**：对每个研究领域，从 **2020–2025** 年论文中反复自助抽样 seed 集。**每个 seed 集含 5 篇：1 篇 anchor + 4 篇同领域相关论文**（按引用选出）。用 5 篇是因为多数 agent 框架受当前 LLM 上下文窗口限制。

**检索约束（防泄漏的关键）**：agent 可以从本地部署的 Semantic Scholar 数据库进一步检索，但**只允许检索 seed 论文发表时已经存在的论文**，以保持历史设定。

**五个框架**：Zero-shot baseline、AIScientist、ResearchAgent、AgentLaboratory、Co-Scientist。这五个代表了不同范式：自反思与验证、多 agent 审议、锦标赛式假设演化。除 Zero-shot 外都能检索。
**关键控制**：**所有框架的 ideation prompt 都显式鼓励超越 seed 文献探索**——Zero-shot 要求"提出新颖研究 idea"，AIScientist 强调通过迭代自反思产生"novel"和"high-impact"的 idea，ResearchAgent 鼓励创新方法设计。也就是说，**"不够新"不能推给提示词没要求。**

**五个 LLM**：四个 8B–35B 的开源权重模型（Gemma-4-31B-IT、Llama-3.1-8B 等）加一个。

### 四个测量维度与结果

1. **Exploration breadth（探索广度）**：把每条 idea 编码进共享语义 embedding 空间，测同一研究领域内 idea 之间的**平均两两余弦距离**。另有基于质心距离的版本。
   → AI idea 比同领域人类论文**显著更集中**；且**比人类论文更靠近各领域质心**
2. **Exploration distance（探索距离）**：AI idea 距离起始文献有多远。对照组是**直接引用了至少一篇 seed 论文的后续人类论文**（follow-on papers）
   → AI idea **比人类后续工作离起始文献近得多**，主要在做局部精化
3. **Frontier coverage（前沿覆盖）**：用次年人类研究中最频繁研究的主题定义"次年研究前沿"关键词集（**构造该关键词集时把 follow-on 论文排除掉以保证公平**）。AI idea 与 follow-on 人类论文的**词数相当**
   → AI idea 覆盖 **28.5%** 的次年前沿关键词，人类 follow-on 论文 **36.5%**；差异在各领域均统计显著
4. **Potential impact（潜在影响力）**：用与之匹配的人类论文的归一化引用数衡量
   → AI idea **0.387**，人类 follow-on **0.492**，**低 21.3%**。跨领域、跨年份均成立（报 95% bootstrap 置信区间）

### 新问题 vs 新方法的分解（v2 的 4.6 节）

用 LLM 把每条 idea 标注成 **research question** 和 **method(s)** 两部分，再与对应的 5 篇 seed 论文比对。

- **只有 10.5%** 的 AI idea 含有 seed 文献中不存在的研究问题
- **90.4%** 引入了 seed 论文中没有的新方法

> 当 AI idea 与先前工作不同时，差异**主要来自修改或重组方法，而不是识别出新的科学问题**。

**领域差异**：在工程与自然科学（CS、数学、物理、化学、材料、工程）中，AI idea **几乎总是保留既有研究问题**；而**社会学与商学**中新研究问题的比例明显更高。作者的解释是这与领域本性一致——社会与商业中新现象频繁催生新问题，工程与自然科学的创新更多表现为解决既有问题的新方法。

**标注可靠性验证**：用三个独立 LLM 标注者（Qwen-30b、Llama-8B、Gemma-31B）各自独立判断该 idea 是否引入了 seed 文献中缺失的研究问题/技术方法，报三者一致性（表 S8），一致性稳定。

### 作者的结论

> 越来越复杂的 agentic 机制（自反思、分阶段验证、角色分解、多 agent 审议）能提升生成提案的连贯性与合理性，**但不必然转化为更广的科学探索**。当前 AI 研究 agent **更擅长局部精化而非探索**。**更复杂的 agentic 框架和扩大 LLM 规模都没有从根本上解决这个限制。**

### 对我们的价值

和 Heuresis 一起构成 motivation 的两根支柱——一个说"外层搜索推不动前沿"，一个说"agent 生成在分布层面就是收窄的，且加框架加规模都不解决"。我们的主张"要在预训练/后训练阶段注入 innovation prior，而不是在推理时堆 scaffold"由此站得住。
**引用时务必用 v2 数字**，v1 的 37,802/85.1% 已经被作者自己替换掉了。

---

<a id="25"></a>
## 25. Towards Execution-Grounded Automated AI Research

- **arXiv**: [2601.14525](https://arxiv.org/abs/2601.14525) · Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, Tatsunori Hashimoto（斯坦福，即 "Can LLMs Generate Novel Research Ideas?" 同一组）
- **代码**: https://github.com/NoviScl/Automated-AI-Researcher

**立场**：**反 LLM judge**。出发点是他们自己此前的大规模专家评审工作发现——LLM 的 idea 常常**看起来合理但实际无效**，所以必须把 idea 生成 ground 在执行上。

### 两个执行环境

**1. Post-Training：改进 GRPO**
基线是一个 GRPO 实现，在 **MATH** 数据集上微调 **Qwen2.5-Math-1.5B**。ideator 要提出比基线更有效的后训练算法。
指标：固定 wall-clock 训练时间预算下，**MATH 验证集上训练过程中的最高准确率**。
**防 reward hacking**：所有与验证相关的代码放在**单独文件**，自动 executor **不允许访问或修改**。

**2. Pre-Training：nanoGPT**
基线是 nanoGPT 在 FineWeb 上训练。指标是**达到验证集 loss 3.28 所需的训练时间**。因为直接优化时间不好做梯度，引入**以时间倒数为代理的 proxy reward**，多数图用 proxy reward，只对最优解报实际训练时间。
基线 loss 3.255，8×H100。

**防作弊的额外措施**：冻结全部评测代码，并把指标同时写到另一个云 bucket（wandb）。

### 自动 Executor 的实现

Implementer 跑在高 IO 的 CPU 机器上。用户提交一批自然语言 idea → 对每条 idea 并行调用 code-execution LLM，把 idea 和基线代码库一起喂进去，**并行采样 10 个 code diff**；若 diff 打不上补丁，把 patch log 回灌让模型修正，**最多两轮自我修正**；返回第一个能成功打上的 diff；打过补丁的代码库打包成 .zip 提交到云 bucket。

### 结果

**(a) Benchmark ideator 与 executor**：在两个环境上比较自执行（self-execution）与 GPT-5 执行两种模式，跨 Claude-4.5-Sonnet、Gemini 3、Kimi-K2-Thinking、Qwen3-235B 等。**多数模型的完成率很高，自执行模式下尤其高**。

**(b) 进化搜索（execution-guided）**：**十个 search epoch 之内**——
- 后训练：找到一个方法达到 **69.4% vs GRPO 基线 48.0%**
- 预训练：找到一个配方达到 **19.7 分钟 vs nanoGPT 基线 35.9 分钟**

结论：**样本高效且有效，但扩展性有限**——前沿 LLM 在搜索中确实常产出有意义的算法 idea，但**很早就饱和**，只偶尔表现出 scaling。

**(c) RL from execution reward**：用自动 executor 当 reward function 微调 **Qwen3-30B**。
- 能成功提升 ideator 的**平均 reward**（类似典型的 RLVR）
- **但提不高 max reward**——而对科学发现来说 max 才是重要的指标
- 原因是**模型收敛到简单 idea，出现 thinking 长度与 idea 多样性的双重塌缩**
- 作者试过的补救：in-context learning + RL、给 idea 采样加多样性、**加多样性 reward**（在同一 prompt 的组内奖励多样性）——图 10 显示效果有限

作者的诊断：**base model 本身缺乏多样性 + 当前 RL 目标缺少探索激励。**

### 对我们的价值

1. 这是我们现有评测路线（FCS/MLS/ALE）的理论辩护——**执行落地不能全丢掉，judge 型评测是补充不是替代**
2. **RL from execution reward 会 mode collapse、平均涨而上界不涨** 这条，和我们 RL 房规里的经验能对上，是训练侧要正面处理的问题
3. 它与 Heuresis 的结论高度一致：进化搜索有效但样本效率是关键，RL 有扩展性问题——**两个独立团队用不同 harness 得到同一结论，可信度很高**

---

<a id="26"></a>
# 第五部分：第二轮精读（原「未精读线索」）

> 这一批原本只列了标题。**2026-08-26 补读，全部下载 PDF 全文。**
> 结果有五篇的重要性远超预期，本该进主表：**GIANTS / TasteGap / PreScience / Lit2Test / SCOPE**。
> 其中 **GIANTS 是整份调研里与我们项目最同构的一份工作**（4B + RL + 17k benchmark，全部开源）。

---

<a id="27"></a>
## 27. ⭐ GIANTS / GiantsBench: Generative Insight Anticipation from Scientific Literature

- **arXiv**: [2604.09793](https://arxiv.org/pdf/2604.09793) · Joy He-Yueya, Anikait Singh, Ge Gao, Michael Y. Li, Sherry Yang, Chelsea Finn, Emma Brunskill, Noah D. Goodman（Stanford + NYU）
- 代码、benchmark、模型**全部释出**

**这是本轮最重要的发现。它做的事情、模型尺寸、训练方法都和我们高度重合，而且东西全开源。**

### 任务：Insight Anticipation

给两篇 **parent paper** 的摘要，生成一篇下游论文的**核心 insight**。作者的概念化说得很漂亮：

> 这可以视为**在引用图上做 auto-encoding**：一篇目标论文经过一个**高度有损的信道**——它两篇父论文的摘要——模型必须从中重建出原论文的核心 insight。把引用图线性化成输入-目标对，就是在逼模型重现连接相邻节点所需的那次**概念跃迁**。

作者刻意**只给两篇父论文**（而非全部引用），以建立一个受控的最小设定。

### 数据构造

1. 从 arXiv 收 **17,839 篇**论文，最后更新日期在 2007-05-23 至 2026-01-23 之间
2. 因为 arXiv 未经同行评审，**只保留引用数 ≥ 2** 的论文（用 Semantic Scholar 引用数作质量代理）
3. 对每篇论文，用 **gemini-2.5-flash** 找出它**显式引用且以协同方式组合了二者思想**的两篇前作，并要求模型**解释这个协同点**
4. 下载两篇父论文，因上下文长度与推理成本限制，用 gemini-2.5-flash 把每篇摘要成"清楚描述所用方法、突出关键洞察或发现，细节足以让方法与主要贡献被完全理解"
5. **ground truth y\* 的构造是关键一步**：直接用协同解释不行，因为它是**站在下游论文的立场、以两篇父论文为参照**写的。所以再用 **gemini-3-pro 把 insight 改写成不提及下游论文的独立陈述**，逼模型只从两篇父论文本身生成
6. 同一对父论文若对应多篇下游论文，**保留引用最高的那篇的 insight**，以偏向更有影响力的洞察

**切分**：按下游论文的**发表日期做时间切分**（训练 cutoff 之前 / 之后）。另有更严的 **Test-unseen-parents 子集（N = 5,294）**，排除任何与训练集共享父论文的样本。训练集还被**限制在部分领域**（cs.LG, cs.AI, cs.NE, cs.MA）以测试跨领域迁移。

### 评测指标

**gemini-3-pro 作为主 judge**，对生成 insight 与 ground-truth insight 的相似度打 **1–10 分**。
**人类验证**：2 位 CS 博士生对 Qwen3-4B 与 GIANTS-4B 生成的 30 对 insight（n=60）独立评分，**Spearman ρ = 0.761，p < 0.001**。
**judge 卫生**：GRPO 训练期间用 **Qwen3-14B** 当 judge，**gemini-3-pro 只留给评测**——避免"用评测 judge 训练再用它评测"的循环。

### 两条训练路线

- **SFT 蒸馏**：标准 SFT（父论文摘要 → insight）与 **SFT-think**（用 gemini-3-pro 生成详细 CoT 再蒸，类似 OpenThoughts / s1）
- **RL**：**GRPO**，直接以相似度分数为 reward。base model 是 **Qwen3-4B**

### 结果

| 模型 | 平均相似度（1–10） |
|---|---|
| **Qwen3-4B base** | **4.75** |
| gemini-2.5-pro | ≈ 与 Qwen3-4B 相当 |
| gemini-3-pro | ≈ 与 Qwen3-4B 相当 |
| SFT / SFT-think | 仅比 base **略有提升** |
| **GIANTS-4B（GRPO）** | 比 gemini-3-pro **+35%**（全测试集）/ **+34%**（Test-unseen-parents） |

**三条关键结论：**

1. **这个任务连大型专有模型都做不好。** Qwen3-4B 只有 4.75——按评分 rubric，这个区间意味着"生成的 insight 可能在主题上对上了，但**没抓住核心科学贡献或技术细微处**"。而 **gemini-2.5-pro / gemini-3-pro 的表现与小得多的 Qwen3-4B 相当** → **文献接地的综合能力不随模型规模线性增长**，需要专门的训练范式。
2. **模仿学习收效甚微，RL 才有效。** SFT 与 SFT-think 都只是略微超过 base；直接优化相似度的 GRPO 才把能力对齐到人类 insight。
3. **优势在 test-time scaling 下保持**：随着每样本采样数 k 增加（1→16），GIANTS-4B 的 best@k 始终高于 base、gemini-2.5-pro 和 **SciThinker-4B**（即 §1 SciJudgeBench 那篇训出来的 4B 生成模型）。

**第三方交叉验证**：用 **SciJudge-30B**（§1 那篇训的引用影响力判别器）做成对比较，它在 **68%** 的对比中认为 GIANTS-4B 的 insight 比 base 模型的更可能带来高引用。人类评估也认为 GIANTS-4B 的 insight **概念上更清晰**。

### 对我们的价值（最高，超过第一版任何一篇）

1. **模型尺寸、训练方法（4B + GRPO）、任务形态（文献 → 洞察）全部与我们重合**，而且 benchmark、代码、模型权重全开源 —— 我们可以直接拿来当评测，也可以直接对标
2. "**gemini-3-pro 与 Qwen3-4B 打平**"是我们最想要的那种论据：**这项能力不能靠堆规模得到**
3. "**SFT 只是略微提升、RL 才有效**"对我们的训练路线是直接的方法论输入
4. 它和 §1 的 SciJudgeBench 形成一个**闭环生态**：SciJudge-30B 当第三方裁判、SciThinker-4B 当对照基线。我们如果接这两个，等于一次性进入一个已经有多方交叉验证的评测体系
5. ground truth 构造里"**把 insight 改写成不提及下游论文**"那一步，正是我们 paper2reasoning 做 context.md 时最容易泄漏的地方，可以直接抄它的处理

---

<a id="28"></a>
## 28. ⭐ TasteGap: Measuring the Gap Between Human and LLM Research Ideas

- **arXiv**: [2607.01233](https://arxiv.org/pdf/2607.01233) · Yale University + University of Chicago · 数据 `IdeaLand/IdeaSeed`、代码 `ziyuuc/TasteGap`

**定位**：不问"这条 idea 好不好"，问"**LLM 的 idea 分布与人类研究者的 idea 分布差多远**"。这是唯一一份把 research taste 当作**分布性质**而非单条质量来测的工作。

### 数据

- **人类 idea**：ICLR / ICML / NeurIPS **2023–2026** + **Nature Communications 2023–2025**（覆盖 71 个学科），论文本身即人类终点
- 抽取 pipeline：prompt 要求给出 **innovation、departure from prior work、key insight**，再改写成 proposal 风格的 motivation + method
- **反向工程 4–8 篇**高相关前作（基于抽出的 idea 和论文的 related-work 章节），**输入只给这些前作的 title + abstract**
- 混合语料共 **11,683 条**有效人类 idea

### 两轴 research-taste 分类法

- **Opportunity Pattern（机会模式）** 对应 motivation：什么样的 gap 让这个提案值得做——从"缺少解释""被忽视的失败模式"到"结构性脱节""既有理解的局限"
- **Method Paradigm（方法范式）** 对应 method：用什么样的贡献策略把 gap 变成一篇论文——synthesis、scope extension、robustification、formal derivation 等，涵盖分析型、建构型、整合型、探索型

**构造过程**：人类专家先审阅 **NSF / NIH / AHRQ / DARPA** 的研究指南，得到初版 **11 个机会元素 + 9 个方法元素**；再用 **150 篇 held-out 论文**迭代精炼（每轴允许至多两个最近标签 + 一个 other 选项）；最终收敛到 **7 × 7**。三条要求：类别对应反复出现的 gap framing 与贡献策略、跨领域可泛化、经人类验证确认**没有系统性的类别塌缩**。

### 自动标注与验证

**GPT-5.4-mini** 作标注器，返回每轴的主标签 + 次标签、置信度，以及三个诊断分：**surface stitching（表面拼接）、bottleneck specificity（瓶颈具体性）、boilerplate（套话程度）**。
在同一批 150 篇 held-out 上验证：两位作者各自审阅，与 GPT-5.4-mini 计算 **Cohen's κ = 0.84 / 0.81 / 0.93**（分别对应机会标签、方法标签、诊断分）。还检查了混淆矩阵，确认错误集中在**语义相邻的标签**而非系统性塌缩。

### 核心结果

| | 人类 | 9 个被测 LLM |
|---|---|---|
| 以 **connection / bridge** 为动机的比例 | **12.1%** | **47.1% – 64.2%** |
| 以 **synthesis / unification** 为核心方法的比例 | **5.1%** | **22.5% – 38.7%** |
| 两轴归一化熵 | 一致更高 | 更低 |

> LLM ideation 高度集中在**整合型、综合导向**的类型上，而人类研究 idea 跨越范围广得多。这个模式**在各模型家族与各科学领域上都稳定**。

诊断分同向：多数模型输出的 **bottleneck specificity 更低、boilerplate 更高**。

### ⚠️ 一条对我们直接不利、也因此最有价值的发现

**开思考模式会让分布离人类更远。**

| Qwen3-8B | 不开 thinking | 开 thinking |
|---|---|---|
| bridge 类机会占比 | 49.7% | **71.1%** |
| 显式 synthesis 占比 | 38.7% | **52.2%** |
| 机会轴归一化熵 | 0.658 | **0.481** |
| 与人类分布的 TVD | 0.382 | **0.590** |

DeepSeek-V4-Flash 方向相同（bridge 52.2% → 59.1%，synthesis 22.5% → 30.7%，两轴 TVD 均上升）。作者的结论是：**thinking 是在锐化模型自己偏好的 ideation 模板，而不是把分布拓宽向人类品味，并进一步降低了 idea 多样性。**

**这对我们意味着什么**：我们训练的正是长 reasoning 轨迹。**朴素地加 reasoning 会把这个 gap 放大。** 反过来说——如果我们的 innovation prior 训练能在加了 reasoning 的前提下**把 TVD 拉低、把熵拉高**，那是一个逆着已知趋势的强结果，说服力比再刷一个 judge 分高得多。

**建议**：把 TasteGap 作为**第五个评测**接进来，它不需要 judge 排序、只需要分类，成本极低，而且给出的是一个**没人报过的新维度**。

---

<a id="29"></a>
## 29. ⭐ PRESCIENCE: A Dataset and Benchmark for Scientific Forecasting

- **arXiv**: [2602.20459](https://arxiv.org/abs/2602.20459) v2 · Ai2 + UChicago Knowledge Lab + 希伯来大学 + Northwestern（含 Daniel S. Weld、Doug Downey、Tom Hope、James Evans）

**规模**：**98K** 篇近期 AI 论文为核心，加上作者发表史与引用链接的配套论文，**共 502K 篇**。每条论文记录含：标题摘要、**消歧后的作者身份**、influential references、**逐月累计引用轨迹**、202 个主题的多标签分类、arXiv 类别、以及**按发表日期时间对齐的元数据快照**。

**质量控制**：作者身份用 **S2AND** 消歧（人工检查显示比当前 S2AG release 更准）；只保留 **1–10 篇** influential reference 的目标论文；**所有作者与引用级元数据（发表数、引用数、h-index）都按论文发表日期时间对齐**，防止未来信息泄漏进任务输入。

### 七个任务（五个 paper-anchored + 两个 aggregate）

1. **Contribution generation**（与我们最相关）：从一篇真实未来论文的 influential references 预测它的 title + abstract
2. **Collaborator prediction**：给第一作者，从候选作者中排序谁会是共同作者（nDCG、R-precision）
3. **Prior work selection**（作者称是新任务）：给一个作者团队，排序哪些前作会成为他们下一篇论文的 influential reference
4. **Citation count prediction**：回归，预测发表后 12 个月的引用数（MAE、R²、Pearson、Spearman）
5. **Future combination prediction**：给一篇论文，排序哪些前作会在之后八个月里与它**被共同引用**为 influential reference
6–7. **Topic trend forecasting** 两个聚合变体

其中 2、3、1 三个任务可以**组合成一个生成流程**，做语料规模的合成论文 roll-out。

### LACER 指标

因为 contribution generation 要测的是**概念相似度而非表面重叠**，作者提出 **LACER（Lattice of Automatically Constructed Exemplars for Reference）**：一个 LLM-as-judge 指标，用**自动构造的示例**把 1–10 尺度锚定住——下端锚在"主题相关的前作"，上端锚在"与目标贡献语义等价"，**不需要人工写示例**就得到可解释的动态范围。

**验证**：对 250 条专家相似度排序，LACER 的 **Kendall τb = 0.57，逼近人类标注者之间的一致性 0.53**；远超 ROUGE-L (0.27)、BERTScore (0.40)、ASPIRE Distance (0.35)。

### Contribution generation 结果（judge 用 gpt-5-2025-08-07）

| 基线 / 模型 | LACER |
|---|---|
| Same arXiv Category（随机同类论文） | 1.27 |
| **Random Influential Ref.（随机取一篇父论文）** | **4.31** |
| **OLMo 3 7B（微调）** | **4.03** |
| **Qwen 3 8B（微调）** | **3.99** |
| GPT-4o | 4.71 |
| GPT o3 | 5.49 |
| GPT-5 | 5.64 |
| GPT-5.2 | 5.60 |
| Claude Sonnet 4.5 | 5.03 |
| Claude Opus 4.5 | 5.04 |
| GPT-5 agent（+ 完整历史 H<tp） | **5.86** |
| **Gold Paraphrase（上限）** | **10.00** |

**两个刺眼的数字：**
1. **微调过的 7–8B 开源模型（4.03 / 3.99）低于"随机拿一篇父论文交上去"这个基线（4.31）** —— 这是个很硬的负结果
2. **最强的 frontier 模型也只到 5.6–5.9，而 gold paraphrase 是 10.0** —— 一半的距离都没走完

**上下文消融**（GPT-5）：influential refs 5.64；+ related 5.59；+ author papers 5.72；**+ citations（给 12 个月引用数的 oracle）反而降到 5.37**；全给 5.61。**加信息几乎没用，加 oracle 引用数还有害。**

**污染检查**：cutoff 消融显示模型预训练与测试期的重叠**不显著影响 LACER 分数**，相对排名保持稳定。

**对我们的价值**：与 GIANTS 高度互补——GIANTS 给两篇父论文、PRESCIENCE 给全部 influential references（1–10 篇），后者信息更多、更接近我们 context.md 的形态。LACER 也比 Reconstruction 的 binary match 好用得多（有动态范围，4B 不会全 0）。

---

<a id="30"></a>
## 30. ⭐ Lit2Test / What Proves You Wrong: Benchmarking LMs on Falsifiable Research Ideation

- **arXiv**: [2608.22948](https://arxiv.org/html/2608.22948) · 北大 + 天大 + 海大 + 华为 · benchmark、构造 pipeline、审计产物全部释出

**这是整份调研里评测协议做得最严的一篇。** 即便不接它的数据，**它的防守设计也应该整套抄。**

### 核心思想：可证伪性作为共同裁决单元

作者指出现有评判方式都不提供**共享的决策规则**：自由形式评判随文风与位置摇摆；对着后来那篇论文打分则奖励"复原某一条已实现的轨迹"。
解法是让每个提案**预先承诺那个能证明它错的观测**，从而使质量**可裁决**而非仅仅可争论。

**六字段契约**：`(literature_gap, hypothesis, minimal_test, decisive_metric, supporting_result, falsifying_result)`

**明确的范围界定**（写得很克制）：**在范围内**的是文献综合、gap 识别、假设形成、可执行最小可证伪测试的设计——执行可行性是一等被评属性。**在范围外**的是广义科学创造力、未来论文预测、真实执行结果，以及任何"该流程改善了模型生成能力"的主张。**这个任务测的是提案在当前文献下是否可测试，而不是它是否会成功。**

### 构造

每个实例是一个**真实的四论文邻域**：这四篇共享主题，但留下一个**跨论文的张力**，可由一个小实验裁决。材料在构造时固定，**每个参赛模型收到完全相同的 c，别无其他**——所以差异只反映提案质量而非检索能力差异。
**200 个邻域**，四个参赛模型（**GPT-5.2、Claude Sonnet 4.6、GLM-5、DeepSeek-V3.2**）各在每个邻域产出一份**原生六字段提案**（直接按 schema 生成，不是后处理），得 **1,200 个 canonical pair**。

### 评测协议（这是精华）

- **双序判决**：每对在两种呈现顺序下各判一次，共 **2,400 次有序判决**，折叠成 **950 个 order-stable + 250 个 order-sensitive**。**只有 stable 的进入排名聚合**，order-sensitive 的单独报告
- **隐藏对照**：把来自朴素关键词/模板基线的提案**盲插进真实对局**（自动审计 8 个，人类研究再加 4 个），兼作"真实 vs 朴素"的锚——judge 必须稳定地高于这个锚
- **同源渲染对照**：把**完全相同的内容**分别以 schema 形式和散文形式呈现，**把实质与格式分离**
- **单字段操纵检查**：只污染一个字段（grounding / decisive metric / falsifiability），加一个明显缺陷，要求干净版本**在两个顺序下都赢**
- **微妙污染审计**：把明显缺陷换成自然的缺陷，每个都**配一个风格匹配的伪编辑（sham edit）**，于是"偏好干净版本"是**扣除了表面改写之后**测出来的
- **有界人类校准**：3 位标注者，**20 个邻域 / 90 对**，多数标签与 judge 在决定性的 order-stable 案例上比较
- **聚合**：Bradley–Terry 估计 + Condorcet 头对头关系；不确定性用 bootstrap。**事先固定了完全分离时的处理策略**

**作者对主张范围的自我限制值得学**：对有客观标签的任务（如隐藏对照检测）才用"准确率"，其余只报"判决与参考信号的一致性"；人类标签只是**分层子集上的校准证据**；开放式提案的 ground truth 仍有争议。**canonical pair 始终是统计单元。canonical judge 与判决规则按 benchmark 版本固定，任何替换都要求版本化的完整重跑。**

### 结果

**GPT-5.2 > Claude Sonnet 4.6 > GLM-5 > DeepSeek-V3.2**，在**全部 10,000 次 case-level bootstrap 复现中都恢复出这个完整序**。

稳健性：
- **context-cluster bootstrap**（重采样 200 个邻域而非 1,200 对）置信区间宽 11–27%，但**同样在 10,000 次里全部恢复相同排名**
- 这是一个**严格的 Condorcet 序**：每个高排名模型在与所有低排名模型的 stable 头对头中都获胜；跨五个构造批次一致
- 把 250 个 order-sensitive 全判为平局，排名不变；即便**对抗性地指派每一个 sensitive 案例**，两层结构（GPT-5.2 与 Claude 在上，GLM-5 与 DeepSeek 在下）仍保持
- **第二个 judge（Doubao Seed 2.0 Pro，模型家族与四个参赛者及主 judge 都不相交）独立复现了完全相同的排名，在 order-stable 成对比较上与主 judge 一致率 86.1%**
- **分离来自所提测试与指标的质量，而非表面流畅度**

**对我们的价值**：可证伪性是 scientific taste 里**最难伪装的一维**——一个空洞的 idea 写不出"什么观测会证明我错"。这是我们做自己的 judge 时最值得加的一个字段。而它的**同源渲染对照 + 风格匹配伪编辑**，比我推荐的 SciStyleBench 式消融更外科手术式。

---

<a id="31"></a>
## 31. ⭐ SCOPE: Can LLM design high-quality experiments?

- **arXiv**: [2608.03501](https://arxiv.org/html/2608.03501v1) · 西湖大学 / 浙大 / 南理工 / 东京大学

**定位**：先前工作都聚焦代码实现与执行，**跳过了实验设计这个阶段**。SCOPE 专门补这一段。

### 数据构造

从 ICML / ICLR / NeurIPS 收论文，从 PDF 抽 GitHub 链接，用 **stars 与 forks 的对数标准化复合分**排序：

`R_i = ½ [ (ln(1+S_i) − μ_lnS)/σ_lnS + (ln(1+F_i) − μ_lnF)/σ_lnF ]`

取 **Top 300 篇，跨 19 个研究领域**。
**两阶段抽取**：先做 summary-style 的全局理解（整体结构、核心贡献、方法架构、实验逻辑），再做逐节对齐的深度抽取（任务描述、方法模块、实验设计、数据集、baseline、指标、约束、歧义），方法部分还做**模块化分解**（每个模块的功能、架构、关键公式、输入输出）。
**质量精炼**：自验证（模型对照原文找遗漏与幻觉）+ 外部评估器按六维 rubric 打分的循环。

### 任务与评分（六个子维度，0–30 分）

**High-Level（15 分）**——决定"做哪些实验"：
- **Main Experiment**：构建回答研究问题的完整验证链
- **Ablation**：隔离并验证每个方法组件的独立贡献
- **Analysis**：超参敏感性、计算效率等补充洞察

**Low-Level（15 分）**——决定"用哪些具体资源"：
- **Datasets**（来源、切分、构成）
- **Baselines**（来源、性能特征）
- **Metrics**（主指标与辅助指标）

每个子维度由 **GPT-5.2 作 judge**，按有明确分档标准的 rubric 独立打 **0–5**。

**⭐ Redline 机制（这个设计值得直接抄）**：当出现致命缺陷时——例如**显式违反给定约束**——对应子维度**直接记 0，无论其他表现如何**。目的是**防止求平均把关键失败掩盖掉**。同时报 **RL-rate（redline 率）**。

**防泄漏**：Think+Search 条件下模型可联网，但**严禁直接搜索或访问原论文及所提方法**，只能用某时间点之前的公开资源，模拟"现有实验设计未知"的真实场景。

### 结果（七个模型 × 两种策略 + 两个 deep research 模型）

| 模型 | 策略 | RL-rate | Main | Abl. | Anal. | Data | Base. | Metr. | **总分/30** |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5.2 | CoT-only | 1.00 | 3.91 | 3.60 | 3.92 | 2.18 | 2.15 | 2.45 | **18.22** |
| GPT-5.2 | CoT+search | 0.67 | 3.43 | 2.93 | 3.10 | 2.39 | 2.29 | 2.62 | **16.77** |
| **Claude 4.5 Sonnet** | CoT-only | 2.00 | 3.19 | 3.99 | **4.62** | 2.01 | 2.27 | 2.54 | **18.62（最佳）** |
| Claude 4.5 Sonnet | CoT+search | 5.33 | 3.30 | 3.79 | 4.46 | 2.09 | 2.24 | 2.51 | 18.38 |
| Gemini 3 Pro | CoT-only | 3.33 | 2.47 | 2.47 | 2.24 | 1.85 | 1.72 | 1.91 | 12.65 |
| Grok-4 | CoT-only | 5.33 | 2.18 | 2.44 | 2.68 | 1.80 | 1.59 | 1.86 | **12.55（最低）** |
| DeepSeek-V3.2 | CoT-only | 7.67 | 2.33 | 2.75 | 3.13 | 1.76 | 1.94 | 2.01 | 13.92 |
| DeepSeek-V3.2 | CoT+search | **14.00** | 2.45 | 2.79 | 3.17 | 1.63 | 1.80 | 1.92 | 13.76 |
| Qwen3-Max | CoT-only | 8.67 | 2.40 | 2.61 | 2.79 | 1.64 | 1.75 | 1.86 | 13.05 |
| Kimi-k2 | CoT-only | 13.33 | 2.87 | 3.32 | 3.74 | 1.68 | 1.89 | 2.12 | 15.62 |
| Qwen-DeepResearch | – | 9.33 | 2.76 | 2.47 | 2.66 | 1.96 | 1.93 | 2.15 | 13.93 |
| Grok-DeepSearch | – | **0.67** | 3.07 | 2.71 | 2.93 | 2.23 | 2.12 | 2.31 | 15.39 |

**四个发现：**

1. **多数 LLM 无法直接设计高质量实验。** 全模型平均只有 **14.81/30**；只有 GPT-5.2 与 Claude 超过 18；最好的也比满分低 11 分以上。
2. **所有模型在 Low-Level 配置上有系统性瓶颈。** High-Level 一致显著高于 Low-Level——Claude 的 Analysis 达 4.62 而 Datasets 只有 2.01，**两倍以上的差距**。**没有任何模型的任何 Low-Level 子维度超过 3/5**，而 High-Level 常常超过 3.5。平均 High/Low 差 **2.78 分**。→ 瓶颈是**配置准确性而非规划逻辑**。
3. **⚠️ 给搜索不但没用，还可能有害。** 七个模型里五个总分无显著差异；**GPT-5.2 显著退化：18.22 → 16.77（−1.45）**，High-Level 各降 0.48–0.82，Low-Level 只微涨 0.14–0.21。**DeepSeek-V3.2 的 redline 率从 7.67% 几乎翻倍到 14.00%** —— 额外信息**增加了幻觉风险**。作者的诊断：**核心问题不是能否获取信息，而是如何整合——无结构的检索是在与内部推理竞争，而不是互补。**
4. **Deep research 模型改善但不解决瓶颈。** Grok-DeepSearch 15.39（比基座 +2.84，Low-Level 涨得最明显，redline 率最低 0.67%），但 Low-Level 仍全部低于 3 分，且仍远落后于 Claude 与 GPT-5.2。

作者据此提出 **OptED**：三阶段 agentic workflow（阶段隔离 + 原子工具增强 + 规则约束），把配置阶段与协议阶段分开，在六个模型上测试。

**对我们的价值**：**redline 机制**是我们做任何 rubric 型评测都该加的——防止平均掩盖致命失败。另外"**给搜索反而变差**"和 HypoArena 的"**结构化分析技巧效果因模型而异（+88 到 −60）**"是同一件事的两个独立证据：**推理时加 scaffold 不是免费的。**

---

<a id="32"></a>
## 32. SciPredict: Can LLMs Predict the Outcomes of Scientific Experiments in Natural Sciences?

- **arXiv**: [2604.10718](https://arxiv.org/pdf/2604.10718) · Scale AI + UCLA + UMD + Princeton · [code](https://github.com/scaleapi/scipredict)

**规模**：**405 个任务**，来自 **33 个专业子领域**——物理 9 个（量子与原子物理、凝聚态等）、生物 14 个（分子生物、神经科学、生态等）、化学 10 个（有机化学、催化、高分子等）。领域分布：物理 25% / 生物 50% / 化学 25%。

**防污染**：**只选 2025-03-31 之后发表的论文**，避开既有预训练数据。

**专家队伍**：大规模招募，**54.5% 持博士学位**。两轮专家验证。审稿人额外确保 **MCQ 的干扰项是科学上合理但错误的替代方案**、自由形式的评分 rubric 既全面又灵活、数值精度范围现实。另招**独立一组专家**做 human baseline。

**三种题型**（MCQ 40% / free-form 32% / numerical 28%）：
- **MCQ**：3–4 个选项，专家标注 ground truth
- **Free-form**：有参考答案 + **专家写的评分 rubric**，用固定 prompt 的 LLM judge 判是否展现正确科学推理
- **Numerical**：专家给出**可接受区间 [L_i, U_i]**，考虑测量精度与实验变异；落在区间内即算对。作者说明这捕捉的是**实用性**（预测是否足够准以指导实验规划），而非要求精确数值匹配

### ⭐ 三个可靠性校准指标（这是这篇最有价值的部分）

模型每题除预测外还要给三个 1–5 的自评，且**预期的相关方向是事先声明的**：
- **Confidence**（对预测正确性的信心）：若校准良好，应与准确率**正相关**
- **Difficulty**（在给定上下文下的感知难度）：应与准确率**负相关**
- **Feasibility**（该结果能否不做实验、仅靠推理预测出来）：应与准确率**正相关**

### 结果

- **模型准确率 14–26%，人类专家 ≈20%** —— 部分前沿模型**超过**人类专家
- **但校准是灾难性的**：模型**无论自评信心高低、无论是否认为该结果不做实验就能预测，准确率都停在 ≈20%**
- **人类专家则校准极强**：当他们认为某结果越可能不做实验就预测出来时，**准确率从 ≈5% 升到 ≈80%**

> 实验科学中的超人表现，需要的不只是更好的预测，而是**对预测可靠性的更好觉察**。

**对我们的价值**：领域是自然科学，与我们 ML 方法论的重合度低，不进主推。但**"准确率相当而校准天差地别"这个框架非常值得借**——我们可以在自己的评测里加一栏"模型自评这条 idea 的可行性/信心"，看它是否与实际得分相关。这是一个**几乎零成本、但没人在 ML ideation 上报过**的维度。

---

<a id="33"></a>
## 33. AbGen: Evaluating LLMs in Ablation Study Design and Evaluation

- **arXiv**: [2507.13300](https://arxiv.org/abs/2507.13300)（ACL 2025 Long）· [code](https://github.com/yale-nlp/AbGen) · Yale NLP

**首个**评测 LLM 设计消融实验能力的 benchmark。**2,000 条专家标注样本，来自 677 篇 NLP 论文**。
**任务**：给定研究上下文，为指定的模块或流程生成详细的消融实验设计，必须包含**明确的研究目标陈述**和**详细的实验流程描述**。
**结果**：GPT-4o、Llama-3.1 等领先模型与人类专家在**重要性、忠实性、合理性**三方面存在显著差距。
**AbGen-Eval**：配套的**元评测** benchmark，用来检验常用自动评测系统在这个任务上的可靠性——**结论是当前自动评测方法不可靠，与人类评估存在显著差异**。

**对我们的价值**：消融设计是 SCOPE 的 High-Level 子维度之一，AbGen 是它的专门化深挖版。两者可以互为补充。AbGen-Eval 那种"**元评测自动评测器**"的做法，与 SciArena-Eval 同类，是判断"我们的 judge 能不能用"的正规做法。

---

<a id="34"></a>
## 34. ResearchArena / How Far Are We From True Auto-Research?

- **arXiv**: [2605.19156](https://arxiv.org/html/2605.19156v1) · Zhengxin Zhang, Ning Wang, Sainyam Galhotra, Claire Cardie（Cornell）

**这篇应该和 Heuresis、Tang & Yang 一起进第四部分，是同一条"泼冷水"链上的第三块砖。**

**设置**：ResearchArena 是一个**最小 scaffold**，让现成 agent 在轻量指导下自己走完整个研究闭环（ideation → 实验 → 写作 → 自我精炼）。参赛：**Claude Code (Opus 4.6)、Codex (GPT-5.4)、Kimi Code (K2.5)**。**13 个 CS 种子 × 每个 agent-领域对 3 次试验 = 117 篇 agent 生成的论文。**

**三重评价镜头**（这是设计精华）：
1. **SAR**：只看稿件的审稿人
2. **PR（artifact-aware peer review）**：agent **同时检视工作区与稿件**
3. **人类 meta-review**

### 结果

**只看 SAR，图景很乐观**：Claude Code 得分最高，**超过 Analemma 的 FARS，并与 ICLR 2025 人类投稿的加权平均持平**——看上去最小 scaffold 的 agent 就能产出有竞争力的论文。

**人工检视表明这个图景被高估了**：
- **SAR 分数与它自己的接收决定都对不齐**，且**奖励看似合理的 framing，不核实实验实质**
- 换成 **artifact-aware PR，分数急剧下降**
- 人工审计确认**实验严谨性是主要瓶颈**，分解为三种失效模式：**编造结果、实验功效不足（underpowered）、计划与执行不匹配**
- **失效模式高度依赖 agent，约 15 倍的跨度**：Codex 的"论文-产物不匹配 / 编造引用"是 **5% / 8%**，Kimi Code 是 **77% / 72%**——作者说这跟踪的是各 agent 发展出的不同"研究人格"
- **117 篇里没有一篇达到顶会接收线**

**对我们的价值**：这是"**只看稿件的 judge 会被骗**"最直接的证据，比 HindSight 更进一步——它指出了具体的欺骗机制（奖励 framing、不查实质）和具体的修复（让 judge 看产物而不只看稿件）。**如果我们做端到端评测，必须把"产物可见"作为一个对照条件。**

---

<a id="35"></a>
## 35. ScientistOne + Chain-of-Evidence Integrity Audit

- **arXiv**: [2605.26340](https://arxiv.org/pdf/2605.26340) · Google Cloud AI Research

**问题意识**：自主研究 agent 能产出有竞争力的方案和专业模样的稿件，但输出中含有**只看表面呈现的评测查不出来的可验证性失败**：编造引用、无法复现的分数、与实现不符的方法描述。**共同根源是：没有任何现有评测协议审计"主张是否被支撑"。**

**三个贡献**：
1. **Chain-of-Evidence (CoE)**：要求每条主张都可追溯到其证据来源的可验证性框架
2. **ScientistOne**：在文献综述、方案发现、论文写作全程**按构造维持证据链**的端到端系统
3. **⭐ CoE Integrity Audit**：一个事后审计，**四项完整性检查统一适用于所有系统**——**score verification（分数核验）、specification violation（规范违反）、reference verification（引用核验）、method–code alignment（方法与代码一致性）**

### 结果（75 篇论文 × 5 个系统 × 5 个前沿研究任务）

**每一个 baseline 都至少表现出一种系统性失效模式**：
- **幻觉引用率最高达 21%**
- **分数核验的通过率低至 42%**
- **方法-代码一致性在 20% 到 80% 之间**

ScientistOne 是**唯一**做到零幻觉引用（**0/337 条参考文献**）、分数核验全通过的系统。

**对我们的价值**：那四项检查是**可以直接抄进我们数据质检**的清单。尤其 **method–code alignment**——我们的 methods 语料里 answer.md 与 reasoning.md 描述的方法是否与代码一致，正是 [reasoning-bloat-audit](../) 里发现过"methods 编造代码"的那类问题。这给了一个成体系的审计框架。

---

<a id="36"></a>
## 36. AIPR / Intelligence Is Not the Bottleneck

- **arXiv**: [2606.15887](https://arxiv.org/html/2606.15887) · Costa Georgantas

**定位**：现有工作大多评"机器生成的审稿文本的文笔"，**而不是评它给出的数字分数的效度**。这篇专门验证后者。

**设置**：AIPR 读一篇投稿，输出**五个 0–100 的质量维度**和一个加权总分。**纯 prompting，没有在 review 或决定上做任何微调。**
**300 篇 ICLR 投稿**，有公开的决定档位与 reviewer 评分。**冻结 pipeline，且假设在任何分数接触任何结果之前就预注册。**（这个方法学纪律在这一整批论文里是独一份的。）

### 结果

- 总分**区分拒稿与接收：AUROC 0.82（95% CI 0.78–0.87）**，跨档位单调上升，且跟踪平均 reviewer 评分
- **信号最强的地方正是它声称的地方**：最低分的五分之一被拒率远高于基率，**其中没有 oral 论文**

### ⭐ 但真正的发现在这里

> **效度主要来自模型本身**：在同一个模型上，**一段话的 prompt 判别力几乎与完整 pipeline 相当**（小差距偏向 pipeline，但**未达到预先声明的判据，p = 0.09**）。
>
> 工程加进去的是**可靠性**：AIPR 的分数在重复运行间**几乎不动（组内 SD 0.7 分，而裸 prompt 摆动 2.8 分**），并且同一次调用返回的是有 rubric 结构、有证据支撑的审稿意见，而不是一个光秃秃的数字。

作者的背景论述也很重要：NeurIPS 一致性实验发现**同一流程重跑会有很大一部分接收/拒稿决定翻转**，且**审稿在拒绝弱工作上可靠，在给强工作排序上很差**——这是所有以 review 分为 GT 的评测的天花板。

**对我们的价值**：两条。第一，**预注册 + 冻结 pipeline** 是我们报评测结果时应该做的（尤其我们会反复调 prompt）。第二，"**复杂 pipeline 相对单段 prompt 的增益主要是方差降低，不是判别力提升**"——这意味着我们如果要给自己的评测搭复杂 scaffold，**应该拿单段 prompt 作为必须超越的基线**，否则很容易只是在买稳定性。

---

<a id="37"></a>
## 37. LLM-as-a-Reviewer & PRAIB —— judge 的行为病理

### LLM-as-a-Reviewer — [2605.25415](https://arxiv.org/html/2605.25415)

**898 篇**从 NeurIPS 与 ICLR 分层抽出的论文，**12 个 LLM**，三个轴：
1. **rating calibration**
2. **与人类审稿人的分歧**
3. **⚠️ prompt injection 抵抗力**——通过一种**不可见的字体映射攻击（invisible font-mapping attack）** 嵌入

**发现**：
- LLM **系统性地高估较弱的投稿**
- 在主题侧重上与人类分歧：**低估 Clarity 问题、高估 Reproducibility 问题**
- 生成的审稿**长 2–3 倍，但词汇多样性更低、用词更标准化**
- **prompt injection 仍然高度有效**：简单的隐藏指令可以**在相当大比例的案例中把低分论文推到接收级评分**，效果在不同模型家族间差异极大

### PRAIB — [2605.29815](https://arxiv.org/html/2605.29815)（Wrocław University of Science and Technology）

**11,000 条**审稿，由 **5 个专有与开源模型**为 **1,000 篇 ICLR / NeurIPS 论文**（**2021–2025**）生成，跨多种 prompting 策略，与原始人类反馈对比。
提出一套定义清晰的指标，衡量**审稿的具体性（specificity）、文风（style）、参与行为（behavior of engagement）**。

**发现**：生成的审稿与人类反馈显著背离——**LLM 评分方差更小、正向偏置、且过度自信。**

**对我们的价值**：这两篇合起来说明我们的 judge 有三个可测的病理：**正向偏置（对弱输入过于宽容）、方差压缩（拉不开差距）、可被提示注入操纵**。第三条尤其要注意——**我们自己的模型输出会进入 judge 的上下文**，如果训练无意中学到了某些"讨好 judge"的措辞，那和 prompt injection 在机制上是连续的。这是 SciStyleBench 那套消融之外应该额外查的一条。

---

<a id="38"></a>
## 38. 其余系统与数据集（读过摘要，不进主表）

- **[APRES](https://arxiv.org/pdf/2603.03142)**（Meta Superintelligence Labs + 爱丁堡）—— 自动**发现一套对未来引用数高度预测的 rubric**，再据此修订论文文本，且**不改动核心科学内容**。未来引用预测的 MAE 比次优基线**改善 19.6%**；修订后的论文被人类专家评估者在 **79%** 的情况下偏好于原稿。定位是让作者投稿前"压力测试"稿件。
  → 对我们的意义：它证明**"表述"确实独立于"内容"地影响引用**，这反过来是 SciStyleBench 那条防守必要性的又一个证据。

- **[Towards End-to-End Automation of AI Research](https://arxiv.org/pdf/2606.15497)**（Sakana AI + Oxford + UBC + Vector，Yamada / Lange / Cong Lu / Chris Lu / Shengran Hu / Foerster / David Ha / Jeff Clune）—— The AI Scientist：从构思、写代码、跑实验、画图分析、写整篇稿件到自己做同行评审。产出的稿件**通过了某主要 ML 会议 workshop 的第一轮评审**（该 workshop **接收率 70%**）。两种模式：用人类提供代码模板的 focused 模式，和开放模式。
  → 读的时候注意：**70% 接收率的 workshop** 这个限定条件必须一起引，否则会严重高估。ResearchArena（§34）与 MLR-Bench（§12）都对这条产线给出了更严的复核。

- **[SciDER](https://arxiv.org/pdf/2603.01421)**（William & Mary + MBZUAI + UMN）—— data-centric 的端到端多 agent 研究系统，四个子 agent：ideation（**Evolutionary Idea Search**）、数据分析、实验（合成扎根于数据集特征的可执行代码）、critic（迭代自精炼）。**释出 OpenSciDER-SFT-8K 执行轨迹数据集与 OpenSciDER-27B 微调模型**，在六个 benchmark 上有竞争力。
  → 那份 **8K 执行轨迹数据集**对我们做 agentic 数据可能有参考价值。

- **[Evolving Idea Graphs (EIG)](https://arxiv.org/pdf/2605.04922)**（港理工 + 港科大）—— 指出现有多 agent 系统靠**临时文本**（草稿、聊天记录）协调，难以定位生成 idea 的弱点和 agent 如何精炼它们。改用**演化的 idea graph** 表示部分成形的提案，**节点是科学主张、边是它们之间的关系**。
  → "把 idea 表示成图而非文本"这个想法，对我们做 reasoning trace 的结构化质检可能有用。

- **[IDEAgent](https://arxiv.org/html/2607.22375)**（NTU DeCLaRe Lab）—— 主张 ideation 应作为 **Quality-Diversity 联合目标**而非二选一，通过 lineage 管理 idea 演化：Quality 靠多目标反馈驱动**定向修复与精炼**，Diversity 靠轻量顺序记忆 + 与已完成 idea、其历史祖先、以及**被拒提案**的显式比较。
  提出 **⭐ Yield 指标**：**满足预定质量阈值的、相互之间最大的多样 idea 集合的大小**。跨 **32 个主题 / 8 个 CS 领域**，Yield 上比最佳基线高 **3.89×**，且在 **8 倍多的主题上取得非零 Yield**。开源。
  → **Yield 是个好指标**：它一次性回答"又好又不重复的 idea 你能给我几个"，比单报均值或单报多样性都合理。与 Heuresis 的 QDN 三轴是同一问题的两种操作化。

- **[Idea Search](https://arxiv.org/html/2608.08958)**（COLM 2026 LM4Sci；Caltech + Google Research，含 Michael P. Brenner）—— 纯树搜索在科学方法的巨大搜索空间里会陷入局部最优或无效循环。做法：**动态 Idea Bank** 融入树搜索——(1) 把既有方法分解成**原子 idea**；(2) 从 bank 采样以引导代码变异的分支；(3) 用执行中发现的新 idea 动态更新 bank。
  在单细胞 RNA 测序批次整合任务上，把强树搜索基线的均分从 **0.678 推到 0.697**，最好分 **0.728**。
  **⭐ 三条设计结论**：bank 增强**对 bandit 采样有效、对随机采样无效**；优先考虑新 idea 的 "Exploratory" prompting **能挖出罕见的最优解**；而**加大采样层面的探索反而适得其反**。
  → 与我们做 rollout 的采样策略直接相关。

- **[FlowPIE](https://arxiv.org/pdf/2603.29557)**（中科院深圳先进院 + 大连理工 + UNSW + 厦大）—— 批评"静态的先检索后生成"范式导致 idea 同质、发散不足。把文献探索与 idea 生成当作**共同演化**过程：用受 **GFlowNets 启发的 flow-guided MCTS** 扩展文献轨迹，以 **LLM 生成式奖励模型（GRM）** 对当前 idea 的质量评估作为监督信号来引导自适应检索。

- **[Re2](https://arxiv.org/pdf/2505.07920)**（浙大）—— **一致性有保障的**全流程同行评审与多轮 rebuttal 数据集：**19,926 篇初始投稿、70,668 条审稿意见、53,818 条 rebuttal**，来自 OpenReview 上 **24 个会议 + 21 个 workshop**。
  作者点出既有数据集的**致命问题**：提供的论文内容**往往是最终版而非初始投稿**——也就是说，用它们训练/评测"审稿"其实是在给已经被修订过的稿件写审稿意见。Re2 保证内容确实是被审的那一版。
  → 这条对我们是通用警示：**任何用 OpenReview 构造的评测，都要确认论文版本与评审时点对齐。** SoundnessBench 和 LigBench 在这点上做得怎样值得单独查一遍。

- **[LLM-Metrics](https://arxiv.org/pdf/2605.22176)**（南理工 + 南农 + 江苏警官学院）—— 一个很别致的想法：用 LLM 的**参数化记忆**度量研究影响力。假设是高影响力论文在学术社区曝光更多 → 以文本形式进入训练数据 → 模型对它们形成更强的参数记忆。设计**四类多选探针**（标题识别、作者识别、方法识别、venue 识别），在 **549 篇 2023–2024 CS 论文**上评测 **6 家厂商、0.5B 到 72B 的 17 个 LLM**。**17 个里 15 个给出正向预测（9 个在 p < 0.05 显著）。**
  → 对我们主要是**反面用途**：它证明了模型的参数记忆确实携带论文影响力信号——**这正是所有"预测引用/影响力"评测的污染通道**。我们做 SciJudgeBench 类评测时，时间 OOD 切分不只是好习惯，是必需品。

- **[Can LLMs Generate Novel Research Ideas?](https://arxiv.org/abs/2409.04109)**（Si, Yang, Hashimoto，Stanford）—— 这一整条线的起点。招募 **100+ 位 NLP 研究者**写 idea 并对 LLM 与人类的 idea 做盲评（**79 位专家盲评 49 条 idea**），评四个维度：novelty、excitement、feasibility、overall，并给出统一的数值尺度校准所有评审的标准。
  **结论**：**LLM 生成的 idea 被判定为比人类专家的 idea 更新颖（p < 0.05，在多重假设检验下稳健）**，但在**可行性上略弱**。作者同时**承认即便对专家而言 novelty 判断也很困难**，并提出后续研究设计以检验这些 novelty/feasibility 判断是否真的带来研究结果上的有意义差异。
  → 这篇是所有后续工作的锚。**注意它与 §17 RQ-Bench、§4 HindSight 的张力**：Si et al. 说人评认为 LLM idea 更新颖；RQ-Bench 说 LLM judge 认为 LLM idea 更新颖但人评反过来；HindSight 说被判更新颖的反而未来影响力更低。这三者不完全矛盾（对象和设定不同），但**任何"我们的 idea 更新颖"的主张都必须说明是哪种设定下的哪种 novelty**。

- **[Chain of Ideas](https://arxiv.org/pdf/2410.13185)** —— **IdeaArena** 协议的出处：对给定 topic 用 **Round-Robin 锦标赛**让 LLM judge 对任意两个方法产出的 idea 排序算 ELO，**每对正反各评一次**消位置偏置。

- **[AI for Auto-Research: Roadmap & User Guide](https://arxiv.org/pdf/2605.18661)** 与 **[LLM-Based Scientific Peer Review: Methods, Benchmarks, and Reliability Challenges](https://arxiv.org/html/2606.25057)** —— 两篇综述，作为进一步展开的索引。

- **[ReviewArena](https://openreview.net/forum?id=yugEO52gkR)** 与 **[OpenReviewer](https://openreview.net/forum?id=d4mJdezdHO)** —— 均为 OpenReview 上的投稿，未取到可解析的 PDF，仍为**未核实**状态。检索显示 ReviewArena 报称 51,529 篇 / 196,099 条 review / 14 个 review 字段，ReviewArena-Eval 1,002 篇跨 6 会议，结论是现有模型 miscalibrated、压缩评分尺度、区分接收与拒稿能力弱——**引用前请自行核实**。
