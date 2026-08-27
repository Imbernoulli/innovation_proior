# 值得看的 / 值得接的：Taste 类评测精选

> 逐篇精读详情见 [TASTE_EVAL_SURVEY_zh.md](TASTE_EVAL_SURVEY_zh.md)（50 篇全文精读）。
> 这份只讲**该看哪几篇、该接哪几个、怎么接、以及不接哪个**。
>
> 背景：现有评测（FrontierCS / MLS-Bench / ALE-Bench / ThetaEvolve / TTT-Discover）全是执行落地型，成本高、对 4B 不友好、信号被截断与环境噪声吃掉。要补一批轻量的 model-judge 端到端评测。
>
> 整理时间：2026-08-25，第二轮补读 2026-08-26
>
> **人机直接对比**（AI 的方法 vs 人类的方法，以及用 reject/poster/spotlight/oral 当刻度）单独成文：[AI_VS_HUMAN_METHODS_zh.md](AI_VS_HUMAN_METHODS_zh.md)

---

## 一句话结论

**接五个评测：GIANTS（生成，4B 同尺寸对标）+ SciJudgeBench（判别）+ PapersWithCode 方法对比（方法品味）+ SoundnessBench（否定能力）+ TasteGap（分布）。四条防守：换序一致性 / 跨家双 judge / 文风消融 / redline。**

---

## 如果只看六篇

| # | 论文 | 为什么值得看 |
|---|---|---|
| 1 | **⭐[GIANTS / GiantsBench](https://arxiv.org/pdf/2604.09793)**（Stanford） | **整份调研里与我们项目最同构的一份**：**Qwen3-4B + GRPO**，任务是从两篇父论文生成下游论文的核心 insight，17k benchmark + 代码 + 模型全开源。两个杀手级结论：**gemini-2.5-pro / gemini-3-pro 的表现与 Qwen3-4B base 相当**（能力不随规模线性增长）；**SFT 与 SFT-think 只略微提升，RL 才把它拉到超 gemini-3-pro 35%**。 |
| 2 | **[AI Can Learn Scientific Taste](https://arxiv.org/abs/2603.14473)**（SciJudgeBench） | 唯一同时给出**公开 Qwen3-4B 判别分（58.1→77.3）和 4B 生成侧胜率（76.5%）**的工作。五切分 + position-swap + 同作者 control 可整套抄。**与 GIANTS 构成闭环生态**：SciJudge-30B 是 GIANTS 的第三方裁判，SciThinker-4B 是它的对照基线。 |
| 3 | **⭐[TasteGap](https://arxiv.org/pdf/2607.01233)**（Yale + UChicago） | 唯一把 taste 当**分布性质**测的工作。人类只有 12.1% 以 connection 为动机、5.1% 用 synthesis 为核心方法；9 个 LLM 是 **47.1–64.2%** 和 **22.5–38.7%**。**且开 thinking 会让分布离人类更远**（Qwen3-8B：熵 0.658→0.481，TVD 0.382→0.590）。 |
| 4 | **[Heuresis](https://arxiv.org/html/2606.25198v2)** + **[Narrow Exploration](https://arxiv.org/abs/2605.27905)** + **[ResearchArena](https://arxiv.org/html/2605.19156v1)** | motivation 三根支柱：3,222 次 scored run 里**零条 idea 达到 "Original"**；219,655 条 AI idea 里**只有 10.5% 引入新研究问题**；117 篇 agent 论文**没有一篇达到顶会接收线**，且**只看稿件的 judge 会给出与真相相反的乐观图景**。 |
| 5 | **⭐[Lit2Test](https://arxiv.org/html/2608.22948)** | **评测协议做得最严的一篇**，即便不接数据也该抄它的防守：同源渲染对照、风格匹配伪编辑、单字段污染检查、隐藏朴素基线锚、双序折叠只用 stable、context-cluster bootstrap、跨家第二 judge（86.1% 一致）。核心思想——**每个提案必须预先承诺"什么观测会证明我错"**，把质量从可争论变成可裁决。 |
| 6 | **[MoRI](https://aclanthology.org/2026.acl-long.1609/)**（ACL 2026） | 离我们做的事最近的竞品：同样诊断"表层概念重组、缺技术深度"，同样 SFT+RL，同样从 ICLR 论文反推 motivation→method。它的 retrieval-grounded judge 协议（60% 近期 + 40% 经典、跑三次取平均）、含 ground-truth 的 60 条分层人评（r=0.715）、以及 Length Anchoring 对 GRPO 方差厌恶的推导，都可以直接拿。 |

---

## 主推：接这五个

### A. 生成档（与我们训练目标同构）

#### 1. ⭐ GIANTS / GiantsBench — [2604.09793](https://arxiv.org/pdf/2604.09793) · 代码/benchmark/模型全开源

- **任务**：给两篇 parent paper 的摘要，生成下游论文的核心 insight。作者的概念化：**在引用图上做 auto-encoding** —— 目标论文经过"父论文摘要"这个有损信道，模型重建那次概念跃迁
- **数据**：17,839 篇 arXiv（引用数 ≥2），gemini-2.5-flash 找出被显式引用且被协同组合的两篇前作并解释协同点；**再用 gemini-3-pro 把 insight 改写成不提及下游论文的独立陈述**（这一步防泄漏，我们做 context.md 时该照抄）；同对父论文保留引用最高的下游 insight
- **切分**：按下游论文发表日期时间切分；另有 **Test-unseen-parents（N=5,294）**，排除与训练集共享父论文的样本；训练集限定在 cs.LG/AI/NE/MA 以测跨域迁移
- **指标**：gemini-3-pro 判 1–10 相似度；人类验证 **Spearman ρ = 0.761**（2 位 CS 博士生，n=60）。**训练期用 Qwen3-14B 当 judge，gemini-3-pro 只留给评测** —— judge 卫生做得干净
- **参照分**：**Qwen3-4B base 4.75**；**gemini-2.5-pro / gemini-3-pro ≈ 与 Qwen3-4B 相当**；SFT / SFT-think 仅略升；**GIANTS-4B（GRPO）比 gemini-3-pro +35%（全集）/ +34%（unseen-parents）**；best@k 在 k=1..16 全程高于 base、gemini-2.5-pro 和 SciThinker-4B
- **第三方交叉验证**：SciJudge-30B 在 68% 的成对比较中认为 GIANTS-4B 的 insight 更可能高引
- **接它的理由**：模型尺寸、训练方法、任务形态三重重合，东西全开源，且已经嵌在一个有多方交叉验证的生态里。**这是我们最应该先接的一个。**

### B. 客观 GT 判别档（成本最低，没有 judge 偏差）

#### 2. SciJudgeBench — [2603.14473](https://arxiv.org/abs/2603.14473) · [code](https://github.com/tongjingqi/AI-Can-Learn-Scientific-Taste)

- **任务**：给两篇论文的 title/abstract/日期，判谁引用更高。输出 reasoning + A/B
- **规模**：720,341 训练对（同 subcategory 同年份配对，引用绝对差 ≥8 且相对差 ≥30%）
- **五个测试切分**：main 1,000 / 时间 OOD 904（2025 年）/ 指标 OOD ICLR review 611 / 指标 OOD Altmetric 599 / **同作者同季度 matched 541 + 同题 embedding matched 245**
- **评分**：position-swap consistency，正反两次一致才算对
- **参照分**：GPT-5.4 Thinking **81.6**；**Qwen3-4B 58.1 → SciJudge-4B 77.3**；Qwen3-30B 69.7 → 82.7（超过 GPT-5.4）；时间 OOD 上 4B 64.7 → **80.9**
- **生成侧**：SciThinker-4B 对 base policy 胜率 **76.5% / OOD 76.0%**

#### 3. PapersWithCode 方法对比 — [2605.21491](https://arxiv.org/html/2605.21491v1)

- **任务**：给 research goal + 两个方法描述，判哪个在该 benchmark 上更强。**GT 是真实跑出来的分数**
- **规模**：1,918 个 NLP leaderboard → 5,713 篇 RR 论文 → **724 个有效 leaderboard / 11,488 对**；按 idea 划分防泄漏；测试集**逐条对照原 PDF 人工核验**（4% incomplete 补全、8% incorrect 修正或删除），显式确认 idea 描述中不含实验结果
- **难度分层**：按 benchmark 内 Unified Score 的 σ 分 1σ/2σ/3σ 三档（20% 容差）
- **⚠️ 读表关键**：consistency 计分下**随机基线是 25%，不是 50%**
- **参照分**：Qwen3-8B base 25.31，GPT-5 **61.10**，**Direct-SFT 77.10**，Reason-SFT-DrGRPO 71.35；独立测试集上 **67.49 vs GPT-4.1+检索 51.4**（模型小 50×、无检索）
- **顺带一条对我们很重要的对照**：**GPT-5 蒸的合成 CoT 学不动（25.54%），文献锚定的 reasoning 才有用（37.51→71%）** —— 直接支持 paper2reasoning 路线

#### 4. SoundnessBench — [2605.30329](https://arxiv.org/html/2605.30329) · [HF data](https://huggingface.co/datasets/hosytuyen/SoundnessBench)

- **任务**：执行前判断 proposal 方法学是否成立。二分类
- **规模**：从 ICLR 2022–2026 的 35,209 篇投稿 / 137,940 条评审里，按 reviewer confidence ≥3 且归一化 soundness 标准差 <0.15 过滤，用 soundness 子分打标（≥3 高 / ≤2 低，中间段剔除），抽取近逐字的 hypothesis + 实验设计并**显式剔除结果与结论**，过原子断言检索核验（τ=0.7，通过率 66.93%）→ **1,099 条（641 高 / 458 低）**
- **参照分**：12 个前沿模型 standard prompt 下平均 **74.0% 假阳性率**，最佳 Macro F1 = GPT-5.4 的 **69.7**；aggressive prompt 下假阳降到 19.9% 但 High R 塌到 36.1%，**GPT-5.4 与 GPT-5.4-mini 的 High R 直接归 0**

### C. 分布档（新维度，没人报过）

#### 5. ⭐ TasteGap — [2607.01233](https://arxiv.org/pdf/2607.01233) · 数据 `IdeaLand/IdeaSeed`

- **任务**：不排序质量，只**分类**——把每条 idea 打上 opportunity pattern × method paradigm 两轴标签，与人类分布比 TVD 和熵
- **规模**：11,683 条人类 idea（ICLR/ICML/NeurIPS 2023–2026 + Nature Communications 2023–2025，71 个学科），每篇反向工程 4–8 篇前作，输入只给前作 title+abstract
- **分类法**：7×7，由人类专家审阅 **NSF/NIH/AHRQ/DARPA** 研究指南得初版（11+9 元素），再在 150 篇 held-out 上迭代精炼
- **标注器**：GPT-5.4-mini，与两位作者的 Cohen's κ = **0.84 / 0.81 / 0.93**，且确认错误集中在语义相邻标签而非类别塌缩
- **参照分**：人类 connection 动机 **12.1%** / synthesis 方法 **5.1%**；9 个 LLM 是 **47.1–64.2%** / **22.5–38.7%**；人类两轴归一化熵一致更高
- **⚠️ 为什么这个对我们最关键**：**开 thinking 会让分布离人类更远**（Qwen3-8B：bridge 49.7→71.1%、synthesis 38.7→52.2%、熵 0.658→0.481、TVD 0.382→0.590；DeepSeek-V4-Flash 同向）。我们训练的正是长 reasoning —— **朴素加 reasoning 会放大这个 gap**。反过来，**如果 innovation prior 能在加了 reasoning 的前提下把 TVD 拉低、熵拉高，那是逆着已知趋势的强结果**，比再刷一个 judge 分有说服力得多
- **成本**：只需分类不需排序，极低

### 备选（便宜，可当 smoke test）

- **[PRESCIENCE](https://arxiv.org/abs/2602.20459)**（Ai2）—— contribution generation 任务：从一篇未来论文的全部 influential references（1–10 篇，比 GIANTS 的两篇信息更多、更接近我们 context.md）预测其 title+abstract。**LACER 指标**（自动构造示例锚定 1–10 尺度，τb=0.57 逼近人类 IAA 0.53，远超 ROUGE-L/BERTScore/ASPIRE）。两个刺眼数字：**微调 7–8B（4.03/3.99）低于"随机交一篇父论文"基线（4.31）**；**最强 frontier 只到 5.6–5.9，gold paraphrase 是 10.0**。加信息几乎没用、给 oracle 引用数还有害（5.64→5.37）
- **[LiveIdeaBench](https://www.nature.com/articles/s41467-026-70245-1)**（Nature Comm）—— 单关键词 → idea，1,180 关键词 × **22 个学科** × 40+ 模型。结论：这项能力**与通用智力 benchmark 不强耦合**（QwQ-32B-preview ≈ claude-3.7-sonnet:thinking）→ 现成的、发在 Nature Comm 上的论证：**taste 是独立能力轴**
- **[MLR-Bench](https://arxiv.org/abs/2505.19955)**（NeurIPS 2025 D&B）—— **只跑 idea + proposal 两阶段就很便宜**，跳过昂贵且已知会崩的 experimentation 档（10 个任务里 8 个出现编造结果，端到端论文 LLM judge 3.73/10、人类 4.42/10）
- **[SCOPE](https://arxiv.org/html/2608.03501v1)** —— 实验设计评测，300 篇 × 19 领域，六子维度 0–30。全模型平均只有 **14.81/30**，**没有任何模型的任何 Low-Level 子维度（数据集/baseline/指标）超过 3/5**。它的 **redline 机制**必抄（见下）
- **[AI Idea Bench 2025](https://arxiv.org/pdf/2504.14191)** 的 **IMCQ** 档 —— MCQ 形式，零成本，每个 checkpoint 都跑得起
- **[Reconstruction](https://arxiv.org/html/2608.16645)** —— 只给发表前匿名参考文献复原核心 idea。概念上是我们训练格式的评测版，但单模型 Match 率只有 3.4–15.0%，**4B 上大概率全 0 → 只作定性 case study**。要这个形态请优先用 GIANTS 或 PRESCIENCE（有连续分数，不会全 0）

---

## 四条必须带上的防守

不做这些，结论会被这批论文**直接反驳**：

### 1. 双向换序 + consistency-only 计分
抄 SciJudgeBench 的 position-swap 协议：正反各判一次，两次一致才算对。
**注意随机基线随之改变** —— PapersWithCode 那个评测 chance level 是 **25% 不是 50%**；SciJudgeBench 里 Qwen2.5-1.5B base 只有 5.3%（远低于随机）也是这个原因。报表必须说明。
**再进一步（Lit2Test 的做法）**：把双序判决折叠成 order-stable 与 order-sensitive 两堆，**只有 stable 的进入排名聚合**，sensitive 的单独报告；再做 **context-cluster bootstrap**（重采样上下文而非配对）确认排名稳定。
**⚠️ 另外**：RQ-Bench 证明**成对评测本身会系统性抬高生成输出**（强迫打破平局，gpt-5.5 的 non-obviousness 胜率 27.2%→49.1%，平局率 59.1%→36.8%），报成对结果时**必须同时报平局率**。

### 2. 至少两个跨家 judge，并把 scope 当一等维度
Can AI Evaluate AI Scientists 实测 **Gemini 与 Claude 之间 ρ = 0.907**；Lit2Test 的跨家第二 judge（Doubao Seed 2.0 Pro）与主 judge 在 order-stable 对上一致 **86.1%** 且复现完全相同排名。
但注意上界：**SciArena-Eval 上最好的模型（o3）与 20,000 张人类投票的一致率只有 65.1%**；RQ-Bench 里 Human–LLM 22%（Human–Human 本身也只有 60%）。**任何单 judge 结论的误差都不该被当成小于这个量级。**
**可操作的解药**（RQ-Bench 实证）：不要笼统问 novelty，**显式问"这条 idea 是否 narrow / source-bounded"** —— 这样问时 LLM 与人类判断高度一致（82–90%）。
**judge 卫生**（GIANTS 的做法）：**训练期用的 judge 必须与评测期的 judge 不同**。

### 3. 文风与实质分离
基础版：跑 [SciStyleBench](https://arxiv.org/abs/2608.01666) 式实验，内容固定只改文风，报 **SBI / SRR / AWR 三个一起**。15 个变体里 **B4 Novelty Emphasis** 和 **H4 Ultimate Hype** 是我们最容易学到的文风，**C1 Hollow** 与 **H1 Deceptive Hollow** 是关键对抗项。
**⚠️ 只报 SBI 会被骗**：Direct OpenReviewer 在 fixed-domain 下 SBI 近 0，但 SRR/AWR 也近 0 —— 那是**分数塌缩**不是去偏。
**升级版（Lit2Test 的做法，更外科手术）**：
- **同源渲染对照**：同一份内容分别以结构化 schema 与散文呈现
- **单字段污染检查**：只污染一个字段加明显缺陷，干净版必须在两序都赢
- **微妙污染审计**：自然缺陷 vs **风格匹配的伪编辑**，测的是扣除表面改写之后的偏好
- **隐藏朴素基线锚**：把模板/关键词基线的输出盲插进真实对局，judge 必须稳定高于它
**另外要查 prompt injection**：LLM-as-a-Reviewer 证明**隐藏指令能把低分论文推到接收级**，而我们自己的模型输出会进入 judge 上下文 —— 训练若无意学到"讨好 judge"的措辞，与注入在机制上是连续的。

### 4. ⭐ Redline：不要让平均掩盖致命失败
抄 SCOPE 的做法：**出现致命缺陷（如显式违反给定约束）时，该子维度直接记 0，无论其他表现如何**，并单独报 **redline 率**。
它在 SCOPE 里立刻抓出了真东西：**DeepSeek-V3.2 开搜索后 redline 率从 7.67% 翻倍到 14.00%** —— 这个信号在总分上完全看不出来（13.92 → 13.76）。
配套的审计清单可以直接用 ScientistOne 的 **CoE Integrity Audit** 四项：**分数核验、规范违反、引用核验、method–code alignment**。它在 75 篇论文上量出：**幻觉引用率最高 21%、分数核验通过率低至 42%、方法-代码一致性 20–80%**。

---

## 一条反直觉的横向发现：推理时加 scaffold 不是免费的

四个独立来源，同一个方向：

| 来源 | 证据 |
|---|---|
| **SCOPE** | 给搜索使 GPT-5.2 从 **18.22 掉到 16.77**；DeepSeek-V3.2 的 redline 率翻倍。"核心问题不是能否获取信息，而是如何整合——无结构的检索是在与内部推理竞争" |
| **HypoArena** | 结构化分析技巧的效果 **+88 到 −60 BTD 点**，与 baseline 强度 **ρ = −0.10**（毫无规律） |
| **TasteGap** | 开 thinking 让 idea 分布**离人类更远**（熵降、TVD 升） |
| **Idea Search** | Idea Bank 增强**对 bandit 采样有效、对随机采样无效**；**加大采样层面的探索反而适得其反** |
| **Heuresis / Tang & Yang** | 六种搜索策略推不动 quality–novelty 前沿；五种 agent 框架 + 五个 LLM 都不拓宽探索 |

**这正好是 innovation prior 的立论**：能推动前沿的不是推理时的 scaffold，而是模型内部的先验。**而且 GIANTS 已经用 4B + RL 给出了正面证据**（SFT 只略升、RL 才有效、且 4B 打过 gemini-3-pro）。

---

## 不建议采用：NoveltyRank

[2512.14738](https://arxiv.org/abs/2512.14738)

**先给作者信用**：论文自己写了 accuracy "may be misleading"，单列了 "The Accuracy Paradox" 一段，也承认 novelty 是相对的。

**但三个问题依然成立**：
1. **GT 立不住**。"顶会接收 = novel、随机 arXiv = not novel" 测的是**接收概率**，混着写作质量、选题热度、作者资源。
2. **主表排名叙述与数字冲突**。测试集正类 12.5% ⇒ 平凡分类器 accuracy = **0.875**；表里最高的 SciBERT 只有 0.744 —— **四个模型全部低于平凡基线**，论文没把 0.875 摆出来对照。precision 最高 0.205。
3. **"轻量微调打败前沿零样本"是不对等比较**。GPT-5.1 的 recall 0.986 / precision 0.120 ≈ 基率 —— 它对几乎所有样本都答 novel，是阈值未校准，不是判断力缺失。
4. 没有任何人工验证。

**还能拿走**：数据管线（60,294 篇、2025-03-15 之后时间切分）；Qwen3-4B 成对档 0.741 可作粗糙参照 —— 但同一件事 SciJudgeBench 做得严谨得多。

---

## 引用勘误（写论文前必看）

### 1. Tang & Yang 的数字换过了
你手上那段引文（"37,802 ideas from four agent frameworks… 85.1% reuse the seed research question"）描述的是 **v1**：

| | v1 | **v2（当前，2026-07-11 修订）** |
|---|---|---|
| 框架 / LLM | 4 / 6 | **5 / 5** |
| 有效 idea | 37,802 | **219,655**（自 232,800 次生成运行） |
| 研究问题 | 85.1% 已见于 seed | **只有 10.5% 引入了 seed 中不存在的研究问题**（90.4% 引入新方法） |

其余 v2 数字：12 个大领域 / 155 个研究领域；前沿覆盖 **28.5% vs 人类 follow-on 36.5%**；潜在影响力 **0.387 vs 0.492（低 21.3%）**。**引用请用 v2。**

### 2. ForeSci 不是引用预测
它是 **500 个任务 × 4 个 AI 领域 × 4 个决策族**（方向预测 / 瓶颈发现 / 研究规划 / 会议定位），配 cutoff 对齐的离线知识库，指标 Fact / FTA / Trace / Pers。论文自己的评分 prompt 里写着 "This is a **factual/content coverage metric, not a research-taste metric**"。**不进主推。**
但它有一条设计值得抄：**answer-generation backbone 本身也必须早于任务 cutoff。**

### 3. AI Scientist "通过同行评审"要连限定条件一起引
Sakana 那篇（[2606.15497](https://arxiv.org/pdf/2606.15497)）说稿件通过了某主要 ML 会议 workshop 的第一轮评审 —— **该 workshop 接收率 70%**。ResearchArena 与 MLR-Bench 对同一条产线的复核结论都严厉得多。

### 4. Si et al. / RQ-Bench / HindSight 三者的 novelty 结论看似矛盾
Si et al. 说**人评**认为 LLM idea 更新颖（p<0.05）；RQ-Bench 说 **LLM judge** 认为 LLM idea 更新颖但**人评反过来**；HindSight 说被判更新颖的**未来影响力反而更低**（ρ=−0.29）。对象和设定不同，不完全矛盾，但**任何"我们的 idea 更新颖"的主张都必须说明是哪种设定下的哪种 novelty**。

（Heuresis 的数字是对的：**3,222** 是总 scored run 数，1,628 是 reward-hacking 统计子集，9,000 是总实验数。）

---

## 数据污染风险

这批评测**几乎全部建在 arXiv / OpenReview 论文上**，而我们的训练数据正是从真实论文反推 reasoning。

`decontam/eval_registry.json` 目前只覆盖 FCS / ALE / THETA / TTT / MLS 五个 benchmark。选定评测后必须：

1. 把 `sft/_sft_tags.jsonl` 里的 method slug 对新评测的题目集再过一遍，扩 `eval_registry.json`
2. **优先用时间 OOD 切分**：GIANTS 的时间切分 + Test-unseen-parents、SciJudgeBench 的 2025 split、MoRI 的 NeurIPS 2025 OOD、SciPredict 的 2025-03-31 之后、Reconstruction 的 2026-03-22 之后子集
3. 抄 SoundnessBench：单独跑一个 **ICLR 2026-only 切分**看结论是否稳定
4. 抄 ForeSci 的更严设计：确认**我们 base 模型的训练时点**早于评测题目的时间边界
5. **⚠️ 抄 Re2 的警示**：既有 OpenReview 数据集有个致命问题——**提供的论文内容往往是最终版而非初始投稿**。任何用 OpenReview 构造的评测都要确认论文版本与评审时点对齐
6. **⚠️ LLM-Metrics 证明模型的参数记忆确实携带论文影响力信号**（4 类 MCQ 探针、549 篇、17 个模型里 15 个正向）—— 这是所有"预测引用/影响力"评测的污染通道，时间 OOD 不是好习惯而是必需品

---

## 落地顺序建议

1. **GIANTS**（同尺寸、全开源、代码即插即用，最快出结论）→ `experiments/scripts/eval/`
2. **SciJudgeBench**（判别侧，有 4B 参照分，且与 GIANTS 同生态）
3. **PapersWithCode 方法对比**（GT 最干净，最贴我们的命题）
4. **SoundnessBench** 作为 FPR 诊断，成本几乎为零
5. **TasteGap** 作为分布维度 —— 这是最可能出**新结论**的一个
6. **四条防守随第 1 步一起进协议**，不要事后补

---

## 附：这批工作的整体图景

三派意见并存，我们的位置在中间：

- **执行派**（[Si et al.](https://arxiv.org/abs/2601.14525) / [ResearchArena](https://arxiv.org/html/2605.19156v1) / [ScientistOne](https://arxiv.org/pdf/2605.26340)）：judge 不可信，必须看产物。十轮进化搜索找到后训练 **69.4% vs 48.0%**、预训练 **19.7min vs 35.9min**；但 **RL from execution reward 会 mode collapse**（平均涨、max 不涨）。只看稿件的 judge 会给出与真相相反的乐观图景；幻觉引用率可达 21%、方法-代码一致性低至 20%。
  → **我们现有的 FCS/MLS/ALE 路线由此得到辩护，judge 型评测是补充不是替代。**
- **怀疑派**（[RQ-Bench](https://arxiv.org/html/2606.12071) / [SciStyleBench](https://arxiv.org/abs/2608.01666) / [RINoBench](https://arxiv.org/abs/2603.10303) / [NovBench](https://arxiv.org/abs/2604.11543) / [SciArena](https://allenai.org/blog/sciarena) / [TastyBench](https://www.lesswrong.com/posts/Mxsy7wYvsCRv5dGrw/tastybench-toward-measuring-research-taste-in-llm) / [Axiomatic](https://arxiv.org/pdf/2604.15145) / [PRAIB](https://arxiv.org/html/2605.29815) / [LLM-as-a-Reviewer](https://arxiv.org/html/2605.25415)）：judge 被文风牵着走、结论与专家背离、最好也只有 65.1% 与人类一致、正向偏置、方差压缩、可被提示注入操纵、现有 novelty 指标连时间维度都测不出来。
  → **四条防守全部来自这一派。**
- **可学派**（⭐[GIANTS](https://arxiv.org/pdf/2604.09793) / [SciJudgeBench](https://arxiv.org/abs/2603.14473) / [PapersWithCode 对比](https://arxiv.org/html/2605.21491v1) / [LigBench](https://arxiv.org/html/2608.13136) / [MoRI](https://aclanthology.org/2026.acl-long.1609/)）：用社区信号、真实性能、OpenReview 分或相似度当监督，**小模型微调能显著超过零样本前沿模型**（GIANTS-4B 超 gemini-3-pro 35%；SciJudge-4B 58.1→77.3；8B SFT 77.1 vs GPT-5 61.1；SciThinker-4B 胜率 76.5%）。
  → **这是我们的主战场，也是最有利的证据来源。**

三派其实不矛盾：**执行派证明了外层搜索、scaffold 与 RL-from-execution 都推不动上界；怀疑派证明了 judge 单独不可信；可学派证明了品味本身可以被训进模型里，而且 4B 就够。** 合起来正好是 innovation prior 的完整论证链。
