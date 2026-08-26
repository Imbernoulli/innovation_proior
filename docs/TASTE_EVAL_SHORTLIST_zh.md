# 值得看的 / 值得接的：Taste 类评测精选

> 逐篇精读详情见 [TASTE_EVAL_SURVEY_zh.md](TASTE_EVAL_SURVEY_zh.md)（25 篇全文精读 + 约 25 条未核实线索）。
> 这份只讲**该看哪几篇、该接哪几个、怎么接、以及不接哪个**。
>
> 背景：现有评测（FrontierCS / MLS-Bench / ALE-Bench / ThetaEvolve / TTT-Discover）全是执行落地型，成本高、对 4B 不友好、信号被截断与环境噪声吃掉。要补一批轻量的 model-judge 端到端评测。
>
> 整理时间：2026-08-25

---

## 一句话结论

**接三个客观 GT 的判别评测（SciJudgeBench / PapersWithCode 方法对比 / SoundnessBench）+ 一个端到端生成评测（HypoArena），并强制带三条防守（换序一致性 / 跨家双 judge / 文风消融）。**

---

## 如果只看五篇

| # | 论文 | 为什么值得看 |
|---|---|---|
| 1 | **[AI Can Learn Scientific Taste](https://arxiv.org/abs/2603.14473)**（SciJudgeBench） | 唯一同时给出**公开 Qwen3-4B 判别分（58.1→77.3）和 4B 生成侧胜率（76.5%）**的工作，尺寸完全对齐。五切分 + position-swap + 同作者 control 的评测设计可整套抄。RLCF / Comparison-Based GRPO 是一条不需要 verifier 的 RL 路线。 |
| 2 | **[Heuresis](https://arxiv.org/html/2606.25198v2)** + **[Narrow Scientific Exploration](https://arxiv.org/abs/2605.27905)** | motivation 的两根支柱：3,222 次 scored run 里**零条 idea 达到"Original"**、top-10∩NS≤2 只有一条；219,655 条 AI idea 里**只有 10.5% 引入了新研究问题**、前沿覆盖 28.5% vs 人类 36.5%、潜在影响力低 21.3%。**加框架、加规模都不解决。** |
| 3 | **[HindSight](https://arxiv.org/html/2603.15164)** | 一篇里同时给出三个数：真实未来影响力分出 2.5×（p<0.001，命中率 81% vs 42%）、**LLM judge 判无差异（p=0.584）**、**judge 的 novelty 与真实影响力负相关 ρ=−0.29 且把更差的系统排在前面**。 |
| 4 | **[RQ-Bench](https://arxiv.org/html/2606.12071)** | "novelty mirage"：成对评测会强迫 judge 打破平局（gpt-5.5 的 non-obviousness 胜率 27.2%→49.1%，平局率 59.1%→36.8%），专家却反给作者原始 RQ 78%。**但同一篇给了解药**：显式问"窄不窄/是否绑死来源"时，LLM 判断与人类高度一致（82–90%）。 |
| 5 | **[MoRI](https://aclanthology.org/2026.acl-long.1609/)**（ACL 2026） | 离我们做的事最近的竞品：同样诊断"表层概念重组、缺技术深度"，同样 SFT+RL，同样从 ICLR 论文反推 motivation→method。它的 retrieval-grounded judge 协议、含 ground-truth 的分层人评、以及 Length Anchoring 对 GRPO 方差厌恶的推导，都可以直接拿。 |

---

## 主推：接这四个

### A. 客观 GT 判别档（成本最低，没有 judge 偏差）

#### 1. SciJudgeBench — [2603.14473](https://arxiv.org/abs/2603.14473) · [code](https://github.com/tongjingqi/AI-Can-Learn-Scientific-Taste)

- **任务**：给两篇论文的 title/abstract/日期，判谁引用更高。输出 reasoning + A/B
- **规模**：720,341 训练对（自 210 万篇 arXiv，同 subcategory 同年份配对，引用绝对差 ≥8 且相对差 ≥30%）
- **五个测试切分**：main 1,000 / 时间 OOD 904（2025 年）/ 指标 OOD ICLR review 611 / 指标 OOD Altmetric 599 / **同作者同季度 matched 541 + 同题 embedding matched 245**
- **评分**：position-swap consistency，正反两次一致才算对
- **参照分**：GPT-5.4 Thinking **81.6**；**Qwen3-4B 58.1 → SciJudge-4B 77.3**；Qwen3-30B 69.7 → 82.7（超过 GPT-5.4）；时间 OOD 上 4B 64.7 → **80.9**
- **生成侧**：SciThinker-4B 对 base policy 胜率 **76.5% / OOD 76.0%**（三家 judge 多数投票）
- **接它的理由**：省掉自建 4B baseline 的整整一轮实验；那档**同作者 matched control** 是防"模型只是在认名校"的关键设计

#### 2. PapersWithCode 方法对比 — [2605.21491](https://arxiv.org/html/2605.21491v1)

- **任务**：给 research goal + 两个方法描述，判哪个在该 benchmark 上更强。**GT 是真实跑出来的分数**
- **规模**：1,918 个 NLP leaderboard → 5,713 篇 RR 论文 → **724 个有效 leaderboard / 11,488 对**；按 idea 划分防泄漏；测试集**逐条对照原 PDF 人工核验**（4% incomplete 补全、8% incorrect 修正或删除），并显式确认 idea 描述中不含实验结果
- **难度分层**：按 benchmark 内 Unified Score 的 σ 分 1σ/2σ/3σ 三档（20% 容差）
- **另有两个外部测试集**：跨域 705 对 / 46 个非 NLP 榜单；独立构造 1,750 对（Wen et al.）
- **⚠️ 读表关键**：consistency 计分下**随机基线是 25%，不是 50%**
- **参照分**：Qwen3-8B base 25.31，GPT-5 **61.10**，**Direct-SFT 77.10**，Reason-SFT-DrGRPO 71.35（带可解释理由）；独立测试集上 **67.49 vs GPT-4.1+检索 51.4**（模型小 50×、无检索）
- **接它的理由**：最纯的"方法品味"，最贴我们 context→method 的内核；而且它顺带证明了 **文献锚定的 CoT 可学（37.51→71）、GPT-5 蒸的合成 CoT 学不动（25.54）** —— 这直接支持我们 paper2reasoning 的路线

#### 3. SoundnessBench — [2605.30329](https://arxiv.org/html/2605.30329) · [HF data](https://huggingface.co/datasets/hosytuyen/SoundnessBench)

- **任务**：执行前判断 proposal 方法学是否成立。二分类
- **规模**：从 ICLR 2022–2026 的 **35,209 篇投稿 / 137,940 条评审**里，先按 **reviewer confidence ≥3 且归一化 soundness 标准差 <0.15** 过滤，再用 soundness 子分（≥3 高 / ≤2 低，中间段剔除）打标，抽取近逐字的 hypothesis + 实验设计并**显式剔除结果与结论**，最后过一遍**原子断言检索核验**（τ=0.7，通过率 66.93%）→ **1,099 条（641 高 / 458 低），16 个子领域**
- **参照分**：12 个前沿模型 standard prompt 下平均 **74.0% 假阳性率**（12 个里 9 个超过），最佳 Macro F1 = GPT-5.4 的 **69.7**；aggressive prompt 下假阳降到 19.9% 但 High R 塌到 36.1%，**GPT-5.4 和 GPT-5.4-mini 的 High R 直接归 0**
- **接它的理由**：测"否定能力"（前两个测排序能力）；FPR 天生是"是不是见谁都说好"的诊断图；它的原子断言核验流程我们做数据质检也能直接复用

### B. 端到端生成档

#### 4. HypoArena — [2607.15766](https://arxiv.org/html/2607.15766v1) · [code](https://github.com/SKYLENAGE-AI/HypoArena) · [HF data](https://huggingface.co/datasets/HypoArena/HypoData)

- **任务**：给 **conclusion-free** 上下文 → 生成假设集 {(假设, 证据/验证方案)}。**与我们 context.md → reasoning 几乎同构**
- **规模**：988 case / 2,012 对 / 6 领域（Biomedical 244、ML 218、Social 163、ITOps 146、Financial 114、Safety 103）
- **构造**：Forge–Audit 循环。科学域用**该发现之前**的文献做 Document Merging；分析域用 Structural De-conclusion 保留颗粒事实、剔除专家结论。Audit 每轮做 **Leakage / Faithfulness / Supportability** 三项检查
- **人工审计**：三域各抽 20 例、2–3 位资深博士/从业专家，**92% 全项通过**
- **评分**：Arena 成对，A/B 交换判两次取平均、**只有极性一致才算 consistent**；五级判定映射成 win-share {1.0, .75, .5, .25, 0}；**Bradley–Terry–Davidson** 聚合（平局作为独立认知状态建模）。judge = seed-2.0-pro，交叉验证 mimo-v2-pro。**源自源文档的 Reference 作为匿名选手参赛校准**
- **人类对齐**：1,500 组成对比较，**Kendall τ = 0.90 / Spearman ρ = 0.98**（这批里最好）
- **⚠️ 一条硬约束**：论文实测 arena 拉开 345–490 BTD 点，**rubric 绝对分全挤在 1 分以内**。→ 我们**不要用 1–5 rubric 报结果**
- **另一条**：结构化分析技巧的效果因模型而异（+88 到 −60 点，与 baseline 强度 ρ = −0.10）——**别默认加 scaffold 就有效**

### 备选（便宜，可当 smoke test）

- **[LiveIdeaBench](https://www.nature.com/articles/s41467-026-70245-1)**（Nature Comm）—— 单关键词 → idea，**1,180 关键词 × 22 个学科 × 40+ 模型**（注：不是检索摘要说的 18 个学科）。judge panel = LiveBench top-10 采样集成；originality/feasibility/clarity 直接打分，**fluency 由同关键词多条 idea 之间的多样性导出，flexibility 取其余四维均值的 30th percentile**。
  **它的结论对我们最有用**：这项能力**与通用智力 benchmark 不强耦合**（QwQ-32B-preview ≈ claude-3.7-sonnet:thinking，尽管通用分差距显著）→ 现成的、发在 Nature Comm 上的论证：**taste 是独立能力轴，需要独立评测和独立训练**
- **[MLR-Bench](https://arxiv.org/abs/2505.19955)**（NeurIPS 2025 D&B）—— 201 个 workshop 研究任务。**只跑 idea + proposal 两阶段就很便宜**，跳过昂贵且已知会崩的 experimentation 档（10 个任务里 8 个出现编造结果，端到端论文 LLM judge 3.73/10、人类 4.42/10）。跨家双 judge（Gemini-2.5-Pro + Claude-3.7-Sonnet）取平均值得抄；人类验证显示 LLM–人分差与人–人分差**无统计显著差异**
- **[AI Idea Bench 2025](https://arxiv.org/pdf/2504.14191)** 的 **IMCQ** 档 —— MCQ 形式，零成本，每个 checkpoint 都跑得起。3,495 篇 2023-10 之后的顶会论文 + 其 inspired works
- **[Reconstruction](https://arxiv.org/html/2608.16645)** —— 只给发表前的匿名参考文献（ref-001…，仅 title+abstract，严格早于 seed 日期）复原论文核心 idea，643 篇 × 6 领域，每篇出 5 个假设，leave-one-out judge 判 binary Match。**概念上就是我们训练格式的评测版**，但单模型 Match 率只有 3.4–15.0%（Claude-Opus-4.8 均值 13.3%），多 agent 才到 22.9–41.6%。**4B 上大概率全 0 → 只作定性 case study，或做一个放宽版（判方向对不对而非是否同一个 idea）**

---

## 三条必须带上的防守

不做这三条，结论会被这批论文**直接反驳**：

### 1. 双向换序 + consistency-only 计分
抄 SciJudgeBench 的 position-swap 协议：同一对正反各判一次，两次一致才算对。
**注意随机基线随之改变**——PapersWithCode 那个评测的 chance level 是 **25% 不是 50%**，SciJudgeBench 里 Qwen2.5-1.5B 的 base 分只有 5.3%（远低于随机）也是这个原因。报表时必须说明。
**另外**：RQ-Bench 证明**成对评测本身会系统性抬高生成输出**（强迫打破平局，胜率 27.2%→49.1%），所以报成对结果时**必须同时报平局率**。

### 2. 至少两个跨家 judge，并把 scope 当一等维度
Can AI Evaluate AI Scientists 实测 **Gemini 与 Claude 之间 ρ = 0.907**（与综合分 0.961）——跨家有一致性基础，同家会共谋。
但注意上界：**SciArena-Eval 上最好的模型（o3）与 20,000 张人类投票的一致率只有 65.1%**；RQ-Bench 里 LLM–LLM 一致率 52%、Human–LLM 22%（而 Human–Human 本身也只有 60%）。**任何单 judge 结论的误差都不该被当成小于这个量级。**
**可操作的解药**（RQ-Bench 实证）：不要笼统问 novelty，**显式问"这条 idea 是否 narrow / source-bounded"** —— 这样问的时候 LLM 与人类判断高度一致（82–90%）。

### 3. 文风受控消融
跑一版 [SciStyleBench](https://arxiv.org/abs/2608.01666) 式实验：内容固定、只改文风，报 **SBI / SRR / AWR 三个一起**。
它的 15 个变体里，**B4 Novelty Emphasis** 和 **H4 Ultimate Hype (B2+B4+B5)** 正是我们最容易无意中学到的两种文风；**C1 Hollow**（去掉实质、保留说服力表述）和 **H1 Deceptive Hollow** 是关键对抗项。
**⚠️ 只报 SBI 会被骗**：论文里 Direct OpenReviewer 在 fixed-domain 下拿到近 0 的 SBI，但 SRR/AWR 也近 0 —— 那是**分数塌缩**，不是去偏。
参考量级：直接 judge 的 SBI/SRR/AWR = 0.566/0.504/0.554，加上 SciStyleExtractor 后 = 0.501/0.759/0.899。

---

## 不建议采用：NoveltyRank

[2512.14738](https://arxiv.org/abs/2512.14738) · [code](https://github.com/ZhengxuYan/NoveltyRank) · HF `JasonYan777/novelty-ranked-preprints`

**先给作者信用**：论文自己写了 accuracy "may be misleading"，单列了 "The Accuracy Paradox" 一段，也承认 "novelty is inherently relative, not absolute" 并以此作为转向成对任务的动机。这些自省是对的。

**但三个问题依然成立**：

1. **GT 立不住，这是根子上的**。"顶会接收 = novel(1)、随机 arXiv = not novel(0)" 测的是**接收概率**，混着写作质量、选题热度、作者资源、投稿策略。随机 arXiv 里有极新颖的，接收论文里有大量增量的——两个方向都错。
2. **主表的排名叙述与数字冲突**。测试集正类 **12.5%** ⇒ 恒定输出"not novel"的平凡分类器 accuracy = **0.875**；表里最高的 SciBERT 只有 0.744、Qwen3-4B DPO 0.612、GPT-5.1 0.242 —— **四个模型全部低于平凡基线**。论文点了 accuracy paradox，却没有把 0.875 摆出来对照。precision 最高 0.205，即每 5 次"这个新颖"约 4 次是错的。
3. **"轻量微调打败前沿零样本"是不对等比较**。GPT-5.1 的 recall 0.986 / precision 0.120，precision 几乎正好等于 12.5% 基率 —— 它对几乎所有样本都答"novel"。这是阈值/prompt 没针对基率校准，不是判断力缺失。正确做法是给零样本档也做校准，或改报 AUC/AP 这类阈值无关指标。成对档 0.583 相对干净，但测的仍是"能否区分接收论文与随机 arXiv"。
4. 没有任何人工验证。

**还能拿走两样**：数据管线（60,294 篇 = 50,442 随机 arXiv + 9,852 顶会接收，2025-03-15 之后时间切分，防泄漏做对了）；Qwen3-4B 成对档 0.741 同尺寸可作粗糙参照——但同一件事 SciJudgeBench 做得严谨得多，要参照就参照那个。

---

## 两处引用勘误（重要，写论文前必看）

### 1. Tang & Yang 的数字换过了
你手上那段引文（"37,802 ideas from four agent frameworks… 85.1% reuse the seed research question"）描述的是 **v1**。**v2 已大改**：

| | v1 | **v2（当前，2026-07-11 修订）** |
|---|---|---|
| 框架 / LLM | 4 / 6 | **5 / 5** |
| 有效 idea | 37,802 | **219,655**（自 232,800 次生成运行） |
| 研究问题 | 85.1% 已见于 seed | **只有 10.5% 引入了 seed 中不存在的研究问题**（90.4% 引入新方法） |

其余 v2 数字：12 个大领域 / 155 个研究领域；前沿覆盖 **28.5% vs 人类 follow-on 36.5%**；潜在影响力 **0.387 vs 0.492（低 21.3%）**。**引用请用 v2。**

### 2. ForeSci 不是引用预测
我第一版按检索摘要写成了"预测哪篇论文将来引用高"，**是错的**。它是 **500 个任务 × 4 个 AI 领域 × 4 个决策族**（Direction Forecasting / Bottleneck–Opportunity Discovery / Strategic Research Planning / Venue-Conditioned Positioning），配 cutoff 对齐的离线知识库，指标是 Fact（断言级 F1）/ FTA / Trace / Pers。
而且论文自己的评分 prompt 里写着 "This is a **factual/content coverage metric, not a research-taste metric**"。**所以它不算 taste 评测，不进主推。**
但它有一条设计值得抄：**answer-generation backbone 本身也必须早于任务 cutoff** —— 这比单纯的数据时间切分更狠。

（Heuresis 的数字则是对的：**3,222** 是总 scored run 数，1,628 是 reward-hacking 统计的子集，9,000 是总实验数。）

---

## 一个真实风险：数据污染

这批评测**几乎全部建在 arXiv / OpenReview 论文上**，而我们的训练数据正是从真实论文反推 reasoning。

`decontam/eval_registry.json` 目前只覆盖 FCS / ALE / THETA / TTT / MLS 五个 benchmark。选定评测后必须：

1. 把 `sft/_sft_tags.jsonl` 里的 method slug 对新评测的题目集再过一遍，扩 `eval_registry.json`
2. **优先用时间 OOD 切分**：SciJudgeBench 的 2025 split、MoRI 的 NeurIPS 2025 OOD、LigBench 的 NeurIPS 2025 验证集、HypoArena、Reconstruction 的 2026-03-22 之后子集
3. 抄 SoundnessBench 的做法：单独跑一个 **ICLR 2026-only 切分**看结论是否稳定
4. 抄 ForeSci 的更严设计：确认**我们 base 模型的训练时点**早于评测题目的时间边界

---

## 落地顺序建议

1. **SciJudgeBench**（有 4B 参照分，最快出结论）→ `experiments/scripts/eval/`
2. **PapersWithCode 方法对比**（GT 最干净，最贴我们的命题）
3. **SoundnessBench** 作为 FPR 诊断，成本几乎为零
4. **HypoArena** 作为端到端展示，**用 arena 不用 rubric**
5. 三条防守随第 1 步一起进协议，不要事后补

---

## 附：这批工作的整体图景

三派意见并存，我们的位置在中间：

- **执行派**（[Si et al.](https://arxiv.org/abs/2601.14525)）：judge 不可信，必须真跑 GPU。十轮进化搜索找到后训练 **69.4% vs 48.0%**、预训练 **19.7min vs 35.9min**；但 **RL from execution reward 会 mode collapse——平均 reward 涨、max reward 不涨**，思维长度与 idea 多样性双重塌缩，加多样性 reward 也救不回来。
  → **我们现有的 FCS/MLS/ALE 路线由此得到辩护，judge 型评测是补充不是替代**；且 mode collapse 这条与我们 RL 房规经验对得上。
- **怀疑派**（[RQ-Bench](https://arxiv.org/html/2606.12071) / [SciStyleBench](https://arxiv.org/abs/2608.01666) / [RINoBench](https://arxiv.org/abs/2603.10303) / [NovBench](https://arxiv.org/abs/2604.11543) / [SciArena](https://allenai.org/blog/sciarena) / [TastyBench](https://www.lesswrong.com/posts/Mxsy7wYvsCRv5dGrw/tastybench-toward-measuring-research-taste-in-llm) / [Axiomatic](https://arxiv.org/pdf/2604.15145)）：judge 被文风牵着走、结论与专家背离、理由像人但判断不准、最好也只有 65.1% 与人类一致、现有 novelty 指标连时间维度都测不出来。
  → **三条防守全部来自这一派。**
- **可学派**（[SciJudgeBench](https://arxiv.org/abs/2603.14473) / [PapersWithCode 对比](https://arxiv.org/html/2605.21491v1) / [LigBench](https://arxiv.org/html/2608.13136) / [MoRI](https://aclanthology.org/2026.acl-long.1609/)）：用社区信号、真实性能或 OpenReview 分当监督，**小模型微调能显著超过零样本前沿模型**（4B: 58.1→77.3；8B SFT 77.1 vs GPT-5 61.1；SciThinker-4B 胜率 76.5%）。
  → **这是我们的主战场，也是最有利的证据来源。**

三派其实不矛盾：**执行派证明了外层搜索与 RL 都推不动上界，怀疑派证明了 judge 单独不可信，可学派证明了品味本身可以被训进模型里。** 合起来正好是 innovation prior 的完整论证链。
