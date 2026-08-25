# 值得看的 / 值得接的：Taste 类评测精选

> 逐篇详情见 [TASTE_EVAL_SURVEY_zh.md](TASTE_EVAL_SURVEY_zh.md)。这份只讲**该看哪几篇、该接哪几个、以及不接哪个**。
>
> 背景：现有评测（FrontierCS / MLS-Bench / ALE-Bench / ThetaEvolve / TTT-Discover）全是执行落地型，成本高、对 4B 不友好、信号被截断和环境噪声吃掉。要补一批轻量的 model-judge 端到端评测。
>
> 整理时间：2026-08-25

---

## 一句话结论

**接三个客观 GT 的判别评测（SciJudgeBench / PapersWithCode 方法对比 / SoundnessBench）+ 一个端到端生成评测（HypoArena），并强制带三条防守（换序一致性 / 跨家双 judge / 文风消融）。**

---

## 如果只看五篇

| # | 论文 | 为什么值得看 |
|---|---|---|
| 1 | **[AI Can Learn Scientific Taste](https://arxiv.org/abs/2603.14473)**（SciJudgeBench） | 唯一给出**公开 Qwen3-4B 数字**的（58.1% → 77.3%），和我们模型尺寸完全对齐。四切分 + position-swap 的评测设计可以整套抄。 |
| 2 | **[Heuresis](https://arxiv.org/html/2606.25198v2)** + **[AI Research Agents Narrow Scientific Exploration](https://arxiv.org/abs/2605.27905)** | 我们论文 motivation 的两根支柱：**外层搜索推不动 quality–novelty 前沿**（9,000 次实验）+ **agent 生成在分布上就是收窄的**（219,655 个 idea，85.1% 复用 seed RQ）。→ 能推动前沿的只能是模型内部先验。 |
| 3 | **[HindSight](https://arxiv.org/html/2603.15164)** | 一篇里同时给出三个数：真实未来影响力能分出 2.5×（p<0.001）、**LLM judge 判无差异（p=0.584）**、**judge 的 novelty 与真实影响力负相关 ρ=−0.29**。只报 judge novelty 分 = 报反指标。 |
| 4 | **[On the Limits of LLM-as-Judge](https://arxiv.org/html/2606.12071)**（RQ-Bench） | "novelty mirage"：LLM judge 给模型输出 82% 胜率，专家反给作者原始 RQ 78%，一致率低到 22%。**把 scope/narrowness 当一等维度显式问出来**是可操作的解法。 |
| 5 | **[MoRI](https://aclanthology.org/2026.acl-long.1609/)**（ACL 2026） | 离我们做的事最近的竞品：同样诊断"表层概念重组、缺技术深度"，同样 SFT+RL。它的 reward 设计（entropy-aware information gain + contrastive semantic gain）和 baseline 集合可以直接对照借用。 |

---

## 主推：接这四个

### A. 客观 GT 判别档（成本最低，没有 judge 偏差）

**1. SciJudgeBench** — [2603.14473](https://arxiv.org/abs/2603.14473) · [code](https://github.com/tongjingqi/AI-Can-Learn-Scientific-Taste)
- 任务：给两篇论文的 title/abstract/日期，判谁引用更高
- 规模：720k 训练对；测试分四档 —— main 1,000 / 时间 OOD 904（2025 年）/ 指标 OOD (ICLR review) 611 / 指标 OOD (Altmetric) 599 / 作者机构 matched 541 + 同题 embedding matched 245
- 参照分：GPT-5.4 Thinking **81.6%**，**Qwen3-4B base 58.1% → 训练后 77.3%**
- **接它的理由**：省掉自建 4B baseline 的一整轮实验

**2. PapersWithCode 方法对比** — [2605.21491](https://arxiv.org/html/2605.21491v1)
- 任务：给 research goal + 两个方法，判哪个在该 benchmark 上更强。**GT 是真实跑出来的分数**
- 规模：11,488 对 / 724 个 benchmark，带 1σ/2σ/3σ 难度分层
- 参照分：base ~25%，GPT-5 **61.1%**，**Qwen3-8B SFT 77.1%**
- **接它的理由**：最纯的"方法品味"，最贴我们 context→method 的内核；而且"8B SFT 打穿 GPT-5"证明这题可学、有区分度

**3. SoundnessBench** — [2605.30329](https://arxiv.org/html/2605.30329) · [HF data](https://huggingface.co/datasets/hosytuyen/SoundnessBench)
- 任务：执行前判断 proposal 方法学是否成立。二分类
- 规模：1,099 条 ICLR 2022–2026 提案（641 高 / 458 低，标签取 reviewer soundness 子分），**显式剔除结果与结论**
- 参照分：12 个前沿模型平均 **74.0% 假阳性率**；收紧 prompt 后 FPR 降到 19.9% 但 recall 塌到 36.1%
- **接它的理由**：测"否定能力"（前两个测"排序能力"）；FPR 天生是"是不是见谁都说好"的诊断图；数据在 HF 直接可跑

### B. 端到端生成档

**4. HypoArena** — [2607.15766](https://arxiv.org/html/2607.15766v1) · [code](https://github.com/SKYLENAGE-AI/HypoArena) · [HF data](https://huggingface.co/datasets/HypoArena/HypoData)
- 任务：给 **conclusion-free** 上下文 → 生成假设集。**和我们 context.md → reasoning 几乎同构**
- 规模：988 case / 2,012 hypothesis-evidence 对 / 6 领域
- 评分：Arena 成对 + position debias + Bradley-Terry-Davidson，六维度
- 人类对齐：**Kendall τ = 0.90**（这批里最好）
- **一条硬约束**：论文实测 arena 能拉开 345–490 点，**rubric 绝对分全挤在 1 分以内**。→ 我们**不要用 1–5 rubric 报结果**

### 备选（便宜，可当 smoke test）
- **[LiveIdeaBench](https://www.nature.com/articles/s41467-026-70245-1)** —— 单关键词 → idea，1,180 关键词 × 18 学科，跑起来最便宜。**它的结论"这项能力与通用智力 benchmark 不强耦合"，正好给了我们"taste 是独立能力轴、需要独立训练"的现成论证。**
- **[MLR-Bench](https://arxiv.org/abs/2505.19955)** —— 201 个 workshop 研究任务，**只跑 idea + proposal 两阶段**就很便宜，跳过昂贵的 experimentation 档。跨家双 judge（Gemini + Claude）做法值得抄。
- **[AI Idea Bench 2025](https://arxiv.org/pdf/2504.14191)** 的 **IMCQ** 档 —— MCQ 形式，零成本，每个 checkpoint 都跑得起。
- **[Reconstruction](https://arxiv.org/html/2608.16645)** —— 只给发表前的匿名参考文献复原论文核心 idea，643 篇 × 6 领域。**概念上就是我们训练格式的评测版**，但单模型 Match 率只有 3.4–15.0%，4B 上大概率全 0。**建议只作定性 case study，或做一个放宽版（判方向对不对而非同一个 idea）。**

---

## 三条必须带上的防守

不做这三条，结论会被上面那批论文**直接反驳**：

1. **双向换序 + consistency-only 计分**。抄 SciJudgeBench 的 position-swap 协议：同一对正反各判一次，两次一致才算对。不做的话位置偏置能白送十几个点。

2. **至少两个跨家 judge**。Can AI Evaluate AI Scientists 实测 Gemini 与 Claude 之间 ρ = 0.907、与综合分 ρ = 0.961 —— 跨家有一致性基础；同家会共谋。RQ-Bench 的建议同此：多 judge 交叉核验，且把 **scope / narrowness 当一等维度显式问**，不要指望 judge 在 novelty 里自动折算。

3. **文风受控消融**。跑一版 [SciStyleBench](https://arxiv.org/abs/2608.01666) 式的实验：内容固定、只改文风，报 SBI / SRR / AWR。**我们的模型经过大量 reasoning trace 训练，输出风格必然不同于 base——不做这个消融，"我们的 idea 更好"完全可能只是"我们的排版更像论文"。** 这是审稿人一定会问的。

---

## 不建议采用：NoveltyRank

[2512.14738](https://arxiv.org/abs/2512.14738) · [code](https://github.com/ZhengxuYan/NoveltyRank)

**三个问题**：

1. **GT 立不住**。"顶会接收 = novel(1)、随机 arXiv = not novel(0)" 测的是**接收概率**，混着写作质量、选题热度、作者资源。随机 arXiv 里有极新颖的，接收论文里有大量增量的——这个映射两个方向都错。

2. **主表自我证伪**。测试集正类占 **12.5%** ⇒ 恒定输出"not novel"的平凡分类器 accuracy = **0.875**。表里最高的 SciBERT 只有 0.744、Qwen3-4B DPO 0.612、GPT-5.1 0.242 —— **四个模型全部低于平凡基线**，论文仍按 accuracy 组织叙述。precision 最高 0.205，即每 5 次"这个新颖"约 4 次是错的。

3. **"轻量微调打败前沿零样本"是不对等比较**。GPT-5.1 的 recall 0.986 / precision 0.120，precision 几乎正好等于 12.5% 基率 —— 它对几乎所有样本都答"novel"。这是阈值/prompt 没针对基率校准，不是判断力缺失。正确做法是给零样本档也做校准，或改报 AUC 这类阈值无关指标。

**还能拿走两样**：一是数据管线（60,294 篇 = 50,442 随机 arXiv + 9,852 顶会接收，2025-03-15 之后做时间切分，防泄漏这点做对了）；二是 Qwen3-4B 在成对档 0.741，同尺寸可作粗糙参照——但同一件事 SciJudgeBench 做得严谨得多，要参照就参照那个。

---

## 一个真实风险：数据污染

这批评测**全部建在 arXiv / OpenReview 论文上**，而我们的训练数据正是从真实论文反推 reasoning。

`decontam/eval_registry.json` 目前只覆盖 FCS / ALE / THETA / TTT / MLS 五个 benchmark。选定评测后必须：

1. 把 `sft/_sft_tags.jsonl` 里的 method slug 对新评测的题目集再过一遍，扩 `eval_registry.json`
2. **优先用时间 OOD 切分**：SciJudgeBench 的 2025 split、LigBench 的 NeurIPS 2025 验证集、HypoArena
3. 参考 [Intelligence Is Not the Bottleneck](https://arxiv.org/html/2606.15887) 的做法——用决定公布时间晚于模型 cutoff 的会议（如 ICLR 2026，2026-01 公布）做无泄漏验证

---

## 落地顺序建议

1. 先接 **SciJudgeBench**（有 4B 参照分，最快出结论），加到 `experiments/scripts/eval/`
2. 再接 **PapersWithCode 方法对比**（GT 最干净，最贴我们的命题）
3. **SoundnessBench** 作为 FPR 诊断，成本几乎为零
4. **HypoArena** 作为端到端展示，用 arena 不用 rubric
5. 三条防守随第 1 步一起进协议，不要事后补

---

## 附：这批工作的整体图景

三派意见并存，我们的位置在中间：

- **执行派**（[Si et al.](https://arxiv.org/abs/2601.14525)）：judge 不可信，必须真跑 GPU。他们十轮进化搜索找到 69.4% vs 48.0% 的后训练方法、19.7min vs 35.9min 的预训练配方；但 RL from execution reward 会 mode collapse。→ **我们现有的 FCS/MLS/ALE 路线由此得到辩护，judge 型评测是补充不是替代。**
- **怀疑派**（[RQ-Bench](https://arxiv.org/html/2606.12071) / [SciStyleBench](https://arxiv.org/abs/2608.01666) / [RINoBench](https://arxiv.org/abs/2603.10303) / [NovBench](https://arxiv.org/abs/2604.11543) / [TastyBench](https://www.lesswrong.com/posts/Mxsy7wYvsCRv5dGrw/tastybench-toward-measuring-research-taste-in-llm)）：现有 judge 被文风牵着走、结论与专家背离、理由像人但判断不准。→ **我们要做的三条防守全部来自这一派。**
- **可学派**（[SciJudgeBench](https://arxiv.org/abs/2603.14473) / [2605.21491](https://arxiv.org/html/2605.21491v1) / [LigBench](https://arxiv.org/html/2608.13136)）：用社区信号或真实性能当监督，小模型微调能显著超过零样本前沿模型。→ **这是我们的主战场，也是最有利的证据来源。**
