# Scientific Taste / Idea-Quality 评测调研（逐篇）

> 目的：为 innovation_prior 找一批**比 FrontierCS / MLS-Bench / ALE-Bench / ThetaEvolve / TTT-Discover 更轻**的评测。
> 现有评测全是执行落地型（跑代码、刷记录），成本高、对 4B 模型不友好、且大量分数被截断和运行环境噪声吃掉。
> 这份调研聚焦**端到端 model-judge 型**评测：给背景 → 提方法/判优劣 → 打分。
>
> 整理时间：2026-08-25

## 核实等级说明

每篇标注了我实际读到什么程度，**不要把 ○ 级的数字当定论**：

| 标记 | 含义 |
|---|---|
| ● | 抓了全文 / HTML，表格与实验细节直接读到，数字可信 |
| ◐ | 读到摘要 + 结构化要点，主干可信、细节未逐条核对 |
| ○ | 仅检索片段级，**数字需回原文核实**，此处只作线索 |

---

# 第一部分：有客观 ground truth 的「品味判别」评测

这类评测输入是 title+abstract 或两个方法描述，输出 A/B 或 好/坏。
**优点**：4B 模型跑得动、评分是硬指标、没有 LLM judge 偏差。这是我们要的「更容易的评测」的主力。

---

## 1. ● AI Can Learn Scientific Taste / SciJudgeBench

- **arXiv**: [2603.14473](https://arxiv.org/abs/2603.14473)（v3）
- **代码**: https://github.com/tongjingqi/AI-Can-Learn-Scientific-Taste
- **单位**: OpenMOSS / 复旦

**定位**：把「scientific taste」定义为"判断与提出具有长期科学影响力的研究想法的能力"，并且证明它**可以被训练出来**。这是整份调研里和我们目标最同构的一篇。

**数据构造**
- 底库 210 万篇 arXiv 论文（至 2024）
- 构造 **720,341** 组 field- and time-matched 偏好对（v3 正文数字；早期版本报 696,758）
- 匹配规则：同 subcategory、相近发表窗口 —— 控掉领域和年份，让信号反映质量而非领域热度
- 训练对的筛选阈值：引用数绝对差 ≥ 8 **且** 相对差 ≥ 30%

**测试集切分（这个设计值得抄）**
| 切分 | 规模 | 作用 |
|---|---|---|
| Main（in-domain） | 1,000 对（自 8,830 对池） | 主分 |
| Temporal OOD | 904 对（2025 年论文） | 防训练期泄漏 |
| Metric OOD (ICLR) | 611 对 review-score 对 | 换一种"质量"定义 |
| Metric OOD (Altmetric) | 599 对 | 换成社会关注度 |
| Controlled | 541 对作者/机构匹配 + 245 对 embedding 匹配同题 | 排除"名校名作者"捷径 |

**任务格式**
> "Based on the titles, abstracts, and publication dates of the following two papers A and B, determine which paper has a higher citation count."

输出 = reasoning trace + 二选一。

**评分**：**position-swap consistency** —— 同一对正反各判一次，两次一致才算对。这条必须抄，否则位置偏置能白送十几个点。

**结果**
| 模型 | in-domain | 时间 OOD (2025) |
|---|---|---|
| GPT-5.4 Thinking | 81.6% | — |
| MiniMax-M3 Thinking | 78.5% | — |
| Qwen3-30B base | 69.7% | 71.7% |
| **SciJudge-Qwen3-30B** | **82.7%** (+13.0pp) | **83.1%** (+11.4pp) |
| **Qwen3-4B base** | **58.1%** | — |
| **SciJudge-Qwen3-4B** | **77.3%** (+19.2pp) | — |
| Qwen2.5-32B base → SciJudge | 63.5% → 81.5% (+18.0pp) | — |

**Scientific Thinker（生成侧）**：给 seed paper 的 title/abstract，产出 follow-up idea 的 reasoning + abstract。由 GPT-5.2-high / GLM-5 / Gemini 3 Pro 三家多数投票判影响力潜力。
- vs base policy 胜率 81.5%
- vs 强基线平均胜率 54.2%
- 时间 OOD 上仍保持 83.0%

**对我们的价值（最高）**
1. **有公开的 Qwen3-4B 数字**（58.1 → 77.3），和我们模型尺寸完全对齐，省掉自建 baseline 的一整轮实验。
2. RLCF（Reinforcement Learning from Community Feedback）这套"用社区信号当监督"的思路，和我们用真实论文反推 reasoning 是互补的两条腿。
3. 它的四切分 + position-swap 协议可以直接搬成我们评测的模板。

**我的判断**：首选。唯一要注意的是引用数作为 GT 仍然混入了"传播力"，但它用同领域同期匹配 + 作者/机构 matched control 把这条堵得比同类严得多。

---

## 2. ● Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation

- **arXiv**: [2605.21491](https://arxiv.org/html/2605.21491v1)
- 论文没给正式 benchmark 名，自称"comparative empirical forecasting"数据集

**定位**：整份调研里**最纯粹的"方法品味"评测** —— GT 不是引用数，是**真实跑出来的 benchmark 分数**。

**数据构造**
- 爬 PapersWithCode 排行榜
- 从 benchmark 描述抽 research goal，定位每个方法的原始论文
- 在同一 benchmark 内部配对，用归一化的性能指标定标签
- **11,488** 组标注对，覆盖 **724** 个有效 benchmark，90/10 train/test

**任务格式**：给定 research goal + 两个方法描述 → 预测哪个在该 benchmark 上表现更好。每条样本含"research goal 和两个 idea 的描述"+ 实测结果标签。

**评分**：accuracy，且带 consistency-aware 评估（换序后必须稳定）。另有按性能分离度分层的难度子集（1σ / 2σ / 3σ）。

**结果**
| 模型 | accuracy |
|---|---|
| base 模型（未微调） | ~25%（近随机以下，说明格式都跟不上） |
| GPT-5 | 61.1% |
| RL 变体（带可解释理由） | 71.35% |
| **Qwen3-8B SFT** | **77.1%** |

**可用性**：CC 协议，匿名仓库释出。论文自陈局限是范围偏 NLP benchmark。

**对我们的价值（很高）**
1. GT 是硬数字，完全没有 judge 偏差 —— 这是"更容易的评测"里最干净的一个。
2. **8B SFT 打穿 GPT-5**，说明这题**可学且有区分度**。这正好是我们要证明的命题形态：innovation prior 训练带来的是通用 benchmark 测不到的判断力。
3. 难度分层（1σ/2σ/3σ）自带一个"我们的模型在难例上是不是也强"的分析维度。

**我的判断**：和 SciJudgeBench 并列首选，且互补 —— 一个测"论文级影响力品味"，一个测"方法级有效性品味"。后者更贴近我们训练数据里 context→method 的内核。

---

## 3. ● SoundnessBench: Can Your AI Scientist Really Tell Good Research Ideas from Bad Ones?

- **arXiv**: [2605.30329](https://arxiv.org/html/2605.30329)
- **数据**: https://huggingface.co/datasets/hosytuyen/SoundnessBench
- **主页**: https://hosytuyen.github.io/projects/SoundnessBench

**定位**：**执行之前**能不能看出一个 proposal 方法学上不成立。这是"品味 = 判别力"最直接的操作化。

**数据构造**
- 来源 ICLR 2022–2026 投稿，重建为 1,099 条 research proposal
- 标签用 reviewer 的 **soundness 子分**：高 ≥3 / 低 ≤2
- 抽取时逐字保留 hypothesis 与实验设计，**显式剔除结果与结论**（这一步是关键，否则模型只是在读答案）
- 有 verification audit：把断言拆成原子命题回源核对，保证抽取保真

**规模**：1,099 条（641 高 soundness / 458 低），覆盖 16 个 ML 子领域。

**评分**：二分类。

**结果（12 个前沿模型：GPT / Claude / Gemini / Qwen / LLaMA / Kimi 各变体）**
- 常规 prompt：平均 **74.0% 假阳性率** —— 也就是绝大多数不成立的 proposal 被放行
- 收紧 prompt：假阳降到 19.9%，但高 soundness 的 recall 崩到 **36.1%**
- 结论：这不是稳定的科学判断力，是 **prompt-sensitive** 的摇摆

**对我们的价值（高）**
1. 数据在 HF 上，直接可跑，成本极低。
2. 假阳性率这个指标天生就是个"模型是不是见谁都说好"的诊断 —— 我们训练后如果 FPR 明显下降且 recall 不塌，是很有说服力的一张图。
3. 剔除结果只留 hypothesis + 实验设计，形态上非常接近我们的 context.md。

**我的判断**：推荐进主套件。它测的是"否定能力"，而上面两篇测"排序能力"，合起来覆盖面更完整。

---

## 4. ● HindSight: Evaluating LLM-Generated Research Ideas via Future Impact

- **arXiv**: [2603.15164](https://arxiv.org/html/2603.15164)

**定位**：不用 LLM judge，把生成的 idea 拿去**和后来真实发表的论文对撞**。

**方法**
- 把 idea 生成系统限制在 cutoff 之前的文献
- GT 库：Semantic Scholar 抓的 2023-06 至 2025-12 之间 **27,589** 篇论文
- 用 SPECTER2 embedding 匹配（阈值 θ=0.96）
- 影响力 h(p) = 0.6 × 归一化引用 + 0.4 × venue 声望
- 最终分 = 所有命中论文里的最大影响力

**规模**：200 条生成 idea（100 条来自检索增强的 ResearchAgent，100 条 vanilla baseline），覆盖 10 个 AI/ML 主题。

**结果（这三条是整份调研最重要的实证之一）**
- 检索增强 vs vanilla：**0.297 vs 0.119**，2.5×，p < 0.001
- 同样两组，**LLM-as-Judge 判无显著差异（p = 0.584）**
- HindSight 分数与 LLM 判定的 novelty **负相关，ρ = −0.29**

**可用性**：论文未明确说明是否释出数据/代码。

**对我们的价值**
1. 这是"别只信 LLM judge"最硬的一个反例：judge 看不出的差距，未来影响力锚定能看出 2.5 倍。
2. ρ = −0.29 意味着**judge 越觉得新颖的，越可能不是真有价值的** —— 我们如果只报 judge novelty 分，很可能在报反指标。
3. 方法本身可以复刻：我们有 paper2reasoning 的时间线信息，做一版 cutoff-anchored 的未来命中评测是可行的。

**我的判断**：不一定作为主评测（200 条太小、复现要自建 GT 库），但**它的结论必须写进我们的评测设计文档**。

---

## 5. ● ForeSci: Evaluating LLM Agents for Forward-Looking AI Research Judgment

- **arXiv**: [2606.00644](https://arxiv.org/pdf/2606.00644)
- **代码**: ResearchForesight GitHub repo

**定位**：预测哪些论文将来影响力大，即 forward-looking research judgment。

**数据**：ACL / CVPR / ECCV / EMNLP / ICCV / ICLR / ICML / KDD / NeurIPS / SIGIR 十个会议 2025 起的 call for papers，配 arXiv + Semantic Scholar 的历史引用轨迹。

**评分**：预测排序 vs 实际引用增长 / 接收结果的相关性指标。GT 是真实引用与接收数据，**不是 LLM judge**。

**结果**：对比 GPT-5.2 / Qwen3 / Gemini 3 等，性能随模型能力与推理深度显著变化；同时测了 RAG 与 agent 两种策略。

**对我们的价值**：和 SciJudgeBench 高度同类但覆盖会议论文而非 arXiv 全量，可作为第二个时间 OOD 检验点。优先级低于 SciJudgeBench（后者数据量与切分设计更完整）。

---

## 6. ● TastyBench: Toward Measuring Research Taste in LLM

- **来源**: [LessWrong](https://www.lesswrong.com/posts/Mxsy7wYvsCRv5dGrw/tastybench-toward-measuring-research-taste-in-llm)
- **代码**: github.com/parviam/tastybench

**定位**：一个小而锋利的探针。核心问题定得很好：
> "Can models predict whether an approach will yield insights or improvements **before executing it**?"

它对 research taste 的定义值得引用：
> "the set of intuitions and good judgment guiding a researcher's decisions throughout the research process, **whenever an open-ended decision arises without an obvious way to find the right answer**."

**构造**：Semantic Scholar 上抓 2024 年 1–3 月 "reinforcement learning large language model llm rl" 主题 200 篇 → 过滤到 **38 篇**有算法级 RL 改进的。另有 25 篇做全文分析。GT 排序用 **citation velocity**（单位时间引用累积速率）。

**评分**：LLM 先从 abstract 抽核心 idea，再用 3 种不同 prompt 做成对判断生成 Elo，与 citation velocity 排序求相关。

**结果**：Claude Sonnet 4.5 / Gemini 2.5 Pro / GPT 5.1 全部 **~0.3 相关**，换 prompt、加全文信息都没用。作者结论是"LLM 没有超人的 research taste"。

**对我们的价值**：38 篇太小，不能当主评测。但作为**"现状是差的"的引用出处**很好用 —— 我们要论证 innovation prior 有效，需要先立一个"现有模型在这件事上确实不行"的基线，这篇 + SoundnessBench 是最直接的两个。

---

## 7. ● LigBench / PAIR-IQ

- **arXiv**: [2608.13136](https://arxiv.org/html/2608.13136)
- **数据**: PAIR-IQ 已在 HuggingFace 公开；完整 pipeline 后续释出

**定位**：自动化、细粒度、且**与人对齐**的 AI research idea 评测。

**四个维度**：rating（整体质量）/ contribution（对后续工作的意义）/ soundness（方法学完整性）/ novelty（原创性）。
把 idea 拆成 Main Target、Core Breakthrough、Innovative Methods、Experimental Design 四块再判。

**数据 PAIR-IQ**：**11,000+** 篇，来自 ICLR 2024、ICLR 2025、NeurIPS 2024，含 oral / spotlight / poster / rejected 全档。分数取自 OpenReview review，并做 mean-shifting 去掉会议间的系统偏移。

**判定协议**：把目标 idea 与从库里检索出的语义相近论文反复成对比较，自适应 Elo（K 因子按轮次指数衰减）迭代到收敛。novelty 另有一个模块 = LLM 初始化 + Semantic Scholar 相似度量化。

**人类对齐**：PhD 研究者在 100 组 idea 对上，rating **71%** / contribution **79%** / soundness **73%** 一致。另在 50 篇 NeurIPS 2025 论文上验证：接收论文在所有维度上一致更高。

**结果**：GPT-5 系列成对判断准确率 **0.80+**；小模型经训练后大幅提升；**idea 生成框架并不稳定优于强单模型**（和 Tang & Yang、Heuresis 结论一致）。

**对我们的价值**：PAIR-IQ 是目前公开的、带 OpenReview 真分数的最大 idea 质量库，可以当训练/校准数据用，也可以拿它的成对协议评我们的模型。

---

## 8. ◐ ReviewArena

- **OpenReview**: https://openreview.net/forum?id=yugEO52gkR

**规模**：NeurIPS / ICLR / ICML / CoRL / COLM / EMNLP / TMLR 等 OpenReview 场馆，**51,529 篇论文 / 196,099 条 review**，14 个 review 字段。
**ReviewArena-Eval**：1,002 篇、跨 6 个会议，带各会议自己的评测协议。

**结论**：现有模型 miscalibrated、把评分尺度压扁、区分接收与拒稿的能力弱。

**对我们的价值**：主要作为"peer-review 类评测规模上限"的参照。直接用它评我们的 4B 有点重，但它的 miscalibration 结论提醒我们：**别用绝对分数报结果，用成对/排序**。

---

## 9. ○ NovBench / RINoBench —— 评"判断能力"本身

- **NovBench**: [2604.11543](https://arxiv.org/abs/2604.11543) —— 1,684 组 paper-review 对，取自某顶级 NLP 会议；从 introduction 抽 novelty 描述，配专家写的 novelty 评价。四维框架：Relevance / Correctness / Coverage / Clarity。
- **RINoBench / Is this Idea Novel?**: [2603.10303](https://arxiv.org/abs/2603.10303)（LREC 2026）—— 1,381 条 research idea，人类专家判定 + 9 个自动指标，同时评 rubric 分数和文字理由。

**共同结论（重要）**：LLM 写出来的**理由**和人类高度相似，但这种相似**不转化为判断的准确**；即便是强推理模型，结论也和专家 gold 显著背离。

**对我们的价值**：这是"别拿 judge 的解释质量当判断质量"的直接证据。我们如果展示模型 reasoning 好看，必须另外给一个客观 GT 的准确率，否则会被这两篇直接反驳。

---

# 第二部分：端到端「给背景 → 提方法 → judge」评测

这一档就是你说的 model judge 端到端评测。

---

## 10. ● HypoArena / Before the Action: Benchmarking LLMs on Prospective Hypothesis Discovery

- **arXiv**: [2607.15766](https://arxiv.org/html/2607.15766v1)
- **代码**: github.com/SKYLENAGE-AI/HypoArena
- **数据**: huggingface.co/datasets/HypoArena/HypoData

**定位**：给**conclusion-free**（去掉结论的）上下文，让模型生成合理的假设集 —— 模拟答案未知时的真实发现过程。

**规模**：**988** 个 case，2,012 组 hypothesis-evidence 对，六个领域：
| 领域 | 数量 |
|---|---|
| Biomedical Science | 244 |
| Machine Learning | 218 |
| Social Science | 163 |
| IT Operations | 146 |
| Financial Analysis | 114 |
| Safety Investigation | 103 |

**评分框架 HypoEval，两条腿**
1. **Arena（主）**：成对比较 + position debiasing + **Bradley-Terry-Davidson** 聚合，六个维度：Contextual Grounding / Inferential Insight / Evidential Justification / Hypothesis-Space Breadth / Directional Distinctness / Analytical Utility
2. **Rubric（辅）**：每维 1–5 直接打分，只做诊断画像

**测了 15 个模型**：claude-sonnet-4.6、claude-opus-4.6、gpt-5.4、kimi-k2.6/k2.5、glm-5.1/5、deepseek-v4-pro/flash、qwen-3.6-max、minimax-m2.7/m2.5、gpt-5.4-mini、gemini-3.1-pro/3-flash

**结果**
- 分层清晰：前三名（claude-sonnet-4.6 / claude-opus-4.6 / gpt-5.4）拉开 **360+ BTD 点**
- **Arena vs rubric 严重背离**：arena 拉开 345–490 点，rubric 全挤在 **1 分以内**
- 结构化分析技巧的效果因模型而异：**+88 到 −60 点**都有
- 与专家偏好 **Kendall τ = 0.90**

**对我们的价值（高）**
1. **conclusion-free context → hypotheses 和我们的 context.md → reasoning 几乎同构**，是端到端这一档里形态最贴的。
2. 代码 + 数据全开，τ=0.90 的人类对齐是这批里最好的。
3. "**rubric 绝对分打不开差距、arena 才能**"这条对我们的评测设计是硬约束 —— 不要用 1-5 rubric 报结果。

---

## 11. ◐ MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research

- **arXiv**: [2505.19955](https://arxiv.org/abs/2505.19955)（NeurIPS 2025 D&B Track）
- **代码**: https://github.com/chchenhui/mlrbench

**三个部件**
- **任务集**：201 个开放式研究任务，取自 NeurIPS / ICLR / ICML 的 workshop 主题
- **MLR-Judge**：LLM reviewer + 精心设计的 review rubric。用两个 LLM（Gemini-2.5-Pro-Preview 与 Claude-3.7-Sonnet）独立评分再平均
- **MLR-Agent**：模块化 scaffold，四阶段 —— idea generation / proposal formulation / experimentation / paper writing

**rubric 维度**：Consistency、Clarity、Novelty、Feasibility、Completeness、Soundness、Insightfulness、Significance、Overall。

**人类验证**：Human-Human 与 Human-LLM 的绝对评分差异**无统计显著差别**，支持 MLR-Judge 作为可扩展代理。

**对我们的价值**：**只跑 idea generation + proposal formulation 两个阶段就是一个很便宜的评测**，跳过 experimentation 那一档（那才是贵的部分）。跨家双 judge 的做法值得抄。

---

## 12. ◐ LiveIdeaBench

- **arXiv**: [2412.17596](https://arxiv.org/html/2412.17596v1)
- **期刊版**: [Nature Communications](https://www.nature.com/articles/s41467-026-70245-1)

**定位**：最小上下文下的科学创造力/发散思维。给**单个关键词**就要模型出 idea。

**规模**：1,180 个高影响力科学关键词，覆盖 18 个学科；关键词库**每月更新**保持与前沿同步。

**评分**：基于 Guilford 创造力理论的五个维度 —— originality / feasibility / fluency / flexibility / clarity。用**动态 judge panel**：从多个 SOTA 模型里随机采样若干做集成打分，既压个体偏置，又借用 judge 的持续更新知识。

**最重要的结论**：模型在这类发散思维任务上的表现，**与通用智力 benchmark 的表现不强耦合**。

**对我们的价值**：这条结论对我们非常有用 —— 它给了一个现成的论证：**scientific taste 是一个独立的能力轴，需要独立的评测和独立的训练**。这正是 innovation prior 存在的理由。而且这是这批里跑起来最便宜的。

---

## 13. ◐ AI Idea Bench 2025

- **arXiv**: [2504.14191](https://arxiv.org/pdf/2504.14191)

**问题意识**：现有 idea 评测忽视了三件事 —— LLM 的知识泄漏、缺少有 grounded truth 的开放式 benchmark、feasibility 分析被 prompt 设计限死。

**数据**：3,495 篇 AI 论文 + 它们各自的 inspired works。

**三个任务**
1. **I2T** idea-to-topic matching
2. **I2I** idea-to-idea matching
3. **IMCQ** idea multiple-choice —— **这个是 MCQ 形式，成本极低**

**三个维度**：Quality（I2I 匹配）、Novelty（到既有文献的距离）、Feasibility（基于参考文献的引用影响力评估方法学根基）。

**对我们的价值**：IMCQ 那一档可以当"零成本 smoke test"，每次训练 checkpoint 都跑得起。

---

## 14. ● Reconstruction: A Blind Benchmark for Recovering Research Ideas from Pre-Publication Bibliographies

- **arXiv**: [2608.16645](https://arxiv.org/html/2608.16645)（v2, 2026-08-24）
- 属于 AI-Professor 项目，作者称将随该系统释出数据与代码

**定位**：**只给论文发表前的参考文献**（匿名化成 ref-001, ref-002…，只留 title + abstract），让模型复原这篇论文的核心 idea。

**构造**：每篇 seed paper 的 blind context 只含发表日期**严格早于** seed 的引用；用 Semantic Scholar / OpenAlex / Crossref / arXiv 解析书目，归一化、去重、匿名编号；剔除无日期引用和 seed 自身。

**规模**：**643** 篇（从 879 篇收集里筛出），六个领域：
| 领域 | 数量 |
|---|---|
| Physics (Nature 系) | 138 |
| ML (ICML 2026) | 120 |
| Materials | 117 |
| Chemistry | 105 |
| Astronomy | 85 |
| Medicine | 78 |

**评分**：每篇生成 5 个不同假设；独立 LLM judge 判每个假设是否与 seed 论文描述"同一个核心研究 idea"（binary Match）。默认用 leave-one-out judge（每个模型由其余六个来判）；多 agent 档用综合最好的 4 个模型，Swiss 轮次选择 + 按来源回避防自评。

**结果**
- 单模型：Match 率 **3.4% – 15.0%**，Claude-Opus-4.8 均值最高 **13.3%**
- 多 agent pipeline（跨模型 review + Swiss 选择）：**22.9% – 41.6%**，相对最好单模型 **~2.4×** 提升（paper-level bootstrap 95% CI [2.3, 2.6]）

**对我们的价值（概念上极高，实操上偏难）**
- **这就是我们训练格式的评测版**：context（文献背景）→ method。我们的 paper2reasoning 数据做的正是同一件事。
- 但 3–15% 的绝对水平意味着 4B 模型上很可能全是 0，信噪比不够做训练信号。
- **建议**：作为定性展示 / 论文里的一个 case study 用，不作为主指标。或者做一个放宽版（判"方向对不对"而非"同一个 idea"）。

---

## 15. ● MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in LLMs

- **ACL 2026 Long**: https://aclanthology.org/2026.acl-long.1609/
- 作者：Chenyang Gu, Jiahao Cheng, Meicong Zhang, Pujun Zheng, Jinquan Zheng, Guoxiu He
- 代码已在 GitHub 释出

**问题意识与我们完全重合**：现有 LLM ideation 方法"产出表层的概念重组，缺乏技术深度"（surface-level conceptual recombinations that lack technical depth）。

**方法**：SFT → RL，双 reward
1. **entropy-aware information gain** —— 逼出技术细节
2. **contrastive semantic gain** —— 守住科学有效性

**评测维度**：novelty、technical rigor、feasibility。对手是商业 LLM 和复杂 agentic baseline，报称一致更优。

**对我们的价值（高，但是方法论层面）**
- 这是**离我们做的事最近的一份工作**，不是评测而是竞品/参照。
- 它的 baseline 集合和评测协议可以直接借用，省掉我们自己搭对照。
- 它的 reward 设计（信息增益 + 对比语义增益）值得和我们的 RL 房规配方对照看 —— 我们现在靠 verifier，它靠这两个无需执行的代理 reward。

---

## 16. ◐ IdeaArena（协议，非数据集）

来自 Chain-of-Ideas 一脉。做法：对给定 topic，用 **Round-Robin 锦标赛**让 LLM judge 对任意两个方法产出的 idea 排序，算 ELO；**每对正反各评一次**消位置偏置。

**对我们的价值**：一个可以套在任何自建题目上的轻协议。我们如果要拿自己的 methods/ 语料出题，直接用这套。

---

# 第三部分：judge 不可信的证据 —— 设计评测前必读

---

## 17. ● On the Limits of LLM-as-Judge for Scientific Novelty Assessment / RQ-Bench

- **arXiv**: [2606.12071](https://arxiv.org/html/2606.12071)

**构造**：从 **746** 篇近期 arXiv CS 论文里抽 research question，构造 **1,434** 条 author-anchored 参考 RQ（锚在被引工作与作者点出的 gap 上）。模型读背景论文后生成 5 条 RQ，judge 按 originality / gap-addressing / non-obviousness 各 0–3 打分。

**核心发现 ——「novelty mirage」**
- 单独打分时，LLM judge 给模型生成的 RQ 各维度都很高
- 成对比较时这种偏好**进一步放大**：gemini-3.1-pro 给模型输出 **82%** 胜率
- **专家反过来**：在 non-obviousness 上强烈偏好作者原始 RQ，胜率 **78%** 和 56%
- 专家-LLM 一致率低到 **22%**
- 专家认为很多生成的 RQ "narrow or source-bound"（窄、被来源绑死）
- 有意思的是：**当显式地让 LLM 评"窄不窄 / 是否被来源绑死"时，它的判断又和人类的 non-obviousness 判断高度一致** —— 说明 LLM 是把表面的 gap 措辞误当成了真新颖

**给我们的可操作结论**
1. **不要用单 judge**；要多 judge 交叉核验
2. 把 **scope / narrowness 当一等维度**显式问出来，而不是指望 judge 在 novelty 里自动折算
3. 只报 LLM judge 的 novelty 分 = 系统性高估

---

## 18. ◐ Style Wins, Substance Loses: A Diagnosis of LLM-as-Judge in Idea Generation / SciStyleBench

- **arXiv**: [2608.01666](https://arxiv.org/abs/2608.01666)（2026-08-03）
- 作者：Fengxian Ji, Yuke Li, Jingpu Yang, Juanfan Wu, Fan Zhang, Zhexuan Cui, Yu Xie, Min Peng, Qianqian Xie, Xiuying Chen, Zhuohan Xie

**问题**：LLM judge 到底在评科学实质，还是在被表面文风牵着走。

**三个部件**
1. **SciStyleStage** —— 三阶段评测环境，对**固定不变的科学内容**施加受控文风扰动。三种设定：无上下文 / 固定领域上下文 / 开放域检索上下文。覆盖 **600** 条科学 idea × **15** 种文风变体，每种设定 **9,000** 个评测实例
2. **SciStyleMetrics** —— Style Bias Index (SBI)、Substance Recognition Rate (SRR)、Adversarial Win Rate (AWR)
3. **SciStyleExtractor** —— 即插即用的评测模块，用于缓解文风偏置

**对我们的价值（必做）**：我们的模型经过大量 reasoning trace 训练，输出风格会明显不同于 base。**如果不做这个消融，"我们的 idea 更好"完全可能只是"我们的排版更像论文"**。这是审稿人一定会问的问题。

**相关前作**：[Style Outweighs Substance](https://arxiv.org/abs/2409.15268)（alignment benchmarking 里的同类失效模式）、[Turning Bias into Bugs](https://arxiv.org/pdf/2605.26156)（bandit 引导的文风操纵攻击）。

---

## 19. ● NoveltyRank: A Retrieval-Augmented Framework for Conceptual Novelty Estimation in AI Research

- **arXiv**: [2512.14738](https://arxiv.org/abs/2512.14738)
- 作者：Zhengxu Yan, Han Li, Yuming Feng
- **代码**: https://github.com/ZhengxuYan/NoveltyRank ／ **数据**: HF `JasonYan777/novelty-ranked-preprints` ／ **Demo**: https://novelty-rank.vercel.app/

**做法**
- 语义表示学习 + 对既有文献的检索式比对
- 两个任务：**Task 1 二分类**（novel / not，输入 title + abstract + 与 top-K 前作的相似度分数）；**Task 2 成对比较**（同领域内谁更 novel）
- 三个尺度：GPT-5.1（零样本 API）、Qwen3-4B（LoRA，先 SFT 后 DPO）、SciBERT（解冻上 4 层 + 任务头，冻结下 8 层）

**数据**：2023–2025 共 **60,294** 篇 = **50,442** 篇随机抽的 arXiv + **9,852** 篇顶会接收，六个域（AI, ML, CV, Robotics, NLP, Cryptography）。来源含网页爬取与公开的 ICLR 2017–2025 数据集。
**时间切分**：训练 2024–2025 初，测试 2025-03-15 之后。

**结果**

二分类（测试 10,889 条，正类占 **12.5%**）：
| 模型 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| GPT-5.1 | 0.242 | 0.120 | 0.986 | 0.215 |
| SFT Qwen3-4B | 0.627 | 0.194 | 0.632 | 0.297 |
| DPO Qwen3-4B | 0.612 | 0.205 | 0.735 | 0.321 |
| Fine-tuned SciBERT | 0.744 | 0.187 | 0.313 | 0.234 |

成对比较（9,531 对）：
| 模型 | Agreement |
|---|---|
| GPT-5.1 | 0.583 |
| SFT Qwen3-4B | 0.739 |
| DPO Qwen3-4B | 0.741 |
| Fine-tuned SciBERT | 0.753 |

**我的批评（这篇做得不好，不建议采用）**

1. **Ground truth 立不住**。"顶会接收 = novel(1)，随机抽的 arXiv = not novel(0)" 测的不是新颖性，是**接收概率**——里面混着写作质量、选题热度、作者资源、投稿策略。随机 arXiv 里有极新颖的工作，接收论文里有大量扎实但增量的工作，这个映射在两个方向上都错。

2. **主表的数字自我证伪**。正类占 12.5% ⇒ **恒定输出"not novel"的平凡分类器 accuracy = 0.875**。表里最高的 SciBERT 是 0.744，**四个模型全部低于平凡基线**。论文仍按 accuracy 组织叙述。同时 precision 最高 0.205，意味着每 5 次"这个新颖"的判断里约 4 次是错的。

3. **"轻量微调打败前沿零样本"是不对等比较**。GPT-5.1 的 recall 0.986 / precision 0.120——precision 几乎正好等于 12.5% 的基率，说明它对几乎所有样本都答"novel"。这是阈值与 prompt 没有针对 12.5% 基率校准，不是判断力缺失。拿校准过的微调模型和未校准的零样本模型比 accuracy，得出的结论不成立。正确做法是给零样本档也做阈值/校准（或改报 AUC 这类阈值无关指标）。

4. **成对档同样继承了 GT 的问题**。0.753 的 agreement 是"能不能区分接收论文和随机 arXiv"，不是"能不能区分更新颖和不那么新颖"。

5. **没有任何人工验证**。它自己的局限性讨论承认"novelty 本质是相对的、不是绝对的"——这恰恰说明 Task 1 从一开始就不该那么设。

**还能拿走的两样东西**
1. **数据管线**：6 万篇的规模 + 2025-03-15 之后的时间切分，防泄漏这一点做对了，管线可复用。
2. **Qwen3-4B 在成对档达到 0.741** —— 和我们模型同尺寸，可作粗糙参照。但同一件事 SciJudgeBench 用引用配对 + 同领域同期匹配 + position-swap 做得严谨得多，要参照就参照那个（Qwen3-4B: 58.1% → 77.3%）。

---

## 20. ◐ Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty

- **arXiv**: [2606.25198](https://arxiv.org/html/2606.25198v2)（Antoniades et al.）
- 代码、QDN 分析的 agent skill、全部 run log 均已开源

**框架**：可组合的 loop —— ideator agent 读之前的 parent workspace 提出改动 → executor agent 在沙箱 workspace 实现并跑任务 → grader 从产物里抽分数 → 可选的 auditor 复核证据，可以判分数无效。

**规模**：基于成熟的 Search 与 Quality-Diversity 算法搭了 **6** 个 AI Research loop，应用到 3 个任务（LLM 预训练、on-policy RL、model unlearning），共 **9,000** 次实验。
> 注：你给的段落写"3,222 scored agent runs across six search strategies"；我检索到的另一处片段写 1,628 scored runs。这个数字**以原文为准**，我没有逐页核实。

**核心结论**：当前的 search 与 QD 策略能**操纵**生成的 idea 落在 quality / diversity / novelty 三轴上的位置，但**推不动 quality–novelty 前沿**。

**附带的重要发现**：必须专门检测 reward hacking，否则搜索会偏离任务本意。

**对我们的价值**：这是"光靠外层搜索救不了 idea 质量"的最强证据——**如果前沿推不动，那能推动它的只能是模型内部的先验**。这正是 innovation prior 的立论。这篇应该进我们论文的 motivation。

---

## 21. ◐ AI Research Agents Narrow Scientific Exploration

- **arXiv**: [2605.27905](https://arxiv.org/abs/2605.27905)（Yixuan Tang, Yi Yang；2026-05-27 投稿，2026-07-11 修订）

**规模**：用 5 个 agent 框架 × 5 个 LLM，为不同科学领域生成 **219,655** 个 idea；筛出 **37,802** 个有效（有效 = 成功完成且结构化输出非空）。
框架含 Zero-shot、AI Scientist、ResearchAgent、Agent Lab（你给的段落写"four agent frameworks"，摘要另处写 five，以原文为准）；模型含 Qwen、Llama、Gemma 等。

**结论四条**
1. AI 生成的 idea 比同领域人类论文**更集中**
2. 比人类后续工作**离起始文献更近**（不敢走远）
3. 与**未来真实人类研究对不齐**
4. 落在历史科学版图中**影响力更低的区域**

**另一个数字**：**85.1%** 的 idea 只是复用了 seed research question，差异主要来自对既有技术的重组。

**对我们的价值**：和 Heuresis 一起构成 motivation 的两根支柱 —— 一个说"外层搜索推不动前沿"，一个说"agent 生成在分布层面就是收窄的"。我们的主张"要在预训练/后训练阶段注入 innovation prior，而不是在推理时堆 scaffold"由此站得住。

---

## 22. ● Towards Execution-Grounded Automated AI Research

- **arXiv**: [2601.14525](https://arxiv.org/abs/2601.14525)
- 作者：Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, Tatsunori Hashimoto（斯坦福，即"Can LLMs Generate Novel Research Ideas?"同一组）
- **代码**: https://github.com/NoviScl/Automated-AI-Researcher

**立场**：**反 LLM judge**。他们的论点是评测必须落在真实代码执行上，而不是靠语言模型打分。

**做法**：造了一个自动 executor 实现 idea 并跑大规模 GPU 实验；把两个现实问题——LLM 预训练与后训练——转成执行环境，看前沿 LLM 能不能生成可实现的 idea。

**结果**
- **进化搜索**：十个 search epoch 内找到一个后训练方法达到 **69.4% vs 基线 48.0%**；找到一个预训练配方达到 **19.7 分钟 vs nanoGPT 基线 35.9 分钟**
- **从执行 reward 做 RL**：平均 reward 提升了，但出现 **mode collapse**——模型收敛到更简单的 idea，不再探索

**对我们的价值**
1. 这是我们现有评测路线（FCS/MLS/ALE）的理论辩护，说明**执行落地不能全丢掉**——judge 型评测是补充不是替代。
2. RL from execution reward 会 mode collapse 这条，和我们 RL 房规里的经验对得上，值得在训练侧留意。
3. 结论上它和 Heuresis 一致：进化搜索有效但样本效率是关键，RL 有扩展性问题。

---

# 第四部分：其余检索到的相关工作（○ 级，仅作线索）

以下这些我**只看到检索片段**，数字未核实，列出来是为了不漏掉线索。

## 评测 / benchmark
- **[An Axiomatic Benchmark for Evaluation of Scientific Novelty Metrics](https://arxiv.org/pdf/2604.15145)** —— 用公理化的方式检验各种 novelty 指标是否自洽。设计我们自己的 novelty 指标前值得看。
- **[Can AI Evaluate AI Scientists?](https://arxiv.org/abs/2607.28631)** —— 用三个独立 LLM reviewer（GPT-5.4 / Gemini / Claude）在 originality / rigor / clarity / significance 四维评四个 AI Scientist 框架（Sakana v1&v2、CycleResearcher、Data-to-Paper），15 个提案 × 60 篇论文 + 15 篇 FARS 基准论文。**Gemini 与 Claude 之间 ρ = 0.907**，与综合分 ρ = 0.961。→ 用跨家 judge 是有一致性基础的。
- **[Beyond Rating: A Comprehensive Evaluation and Benchmark for AI Reviews](https://arxiv.org/html/2604.19502)** —— 不只评打分，评整份 review 的质量。
- **[SciArena](https://allenai.org/blog/sciarena)**（[2507.01001](https://www.alphaxiv.org/abs/2507.01001)，Ai2）—— 科学文献任务上的开放评测平台，真人研究者投票 + Bradley-Terry Elo 排行 + SciArena-Eval（评"评测系统"本身）。如果要做一个公开的 taste 排行榜，这是现成的形态参考。
- **[PreScience: A Dataset and Benchmark for Scientific Forecasting](https://arxiv.org/pdf/2602.20459)** —— 科学预测。
- **[SciPredict: Can LLMs Predict the Outcomes of Scientific Experiments in Natural Sciences?](https://arxiv.org/pdf/2604.10718)** —— 自然科学实验结果预测。
- **[Can LLM design high-quality experiments?](https://arxiv.org/html/2608.03501v1)** —— 自主实验设计的系统性 benchmark。
- **AbGen / AblationBench** —— 专门做 ablation study 任务的两个 benchmark（检索中提及，未找到直链）。
- **[Measuring the Gap Between Human and LLM Research Ideas](https://arxiv.org/pdf/2607.01233)**
- **[What Proves You Wrong: Benchmarking Language Models on Falsifiable Research Ideation](https://arxiv.org/html/2608.22948)** —— 角度很好：能不能提出**可证伪**的 idea。可证伪性是 scientific taste 里最难伪装的一维。
- **[LLM-as-a-Reviewer](https://arxiv.org/html/2605.25415)**、**[PRAIB](https://arxiv.org/html/2605.29815)**、**[Re2](https://arxiv.org/pdf/2505.07920)**、**[OpenReviewer](https://openreview.net/forum?id=d4mJdezdHO)**、**[Intelligence Is Not the Bottleneck](https://arxiv.org/html/2606.15887)** —— peer review 侧的一批。最后那篇用 ICLR 2026 投稿（决定于 2026-01 公布，晚于评分模型的 2025-08 cutoff）做无泄漏验证，切分设计值得抄。
- **[LLM-Based Scientific Peer Review: Methods, Benchmarks, and Reliability Challenges](https://arxiv.org/html/2606.25057)** —— 综述。

## 方法 / 系统（竞品或参照）
- **[ResearchStudio-Idea](https://arxiv.org/pdf/2607.04439)** —— 从 ML 会议真实结果里提炼的证据锚定 ideation skill suite。
- **[IDEAAAgent](https://arxiv.org/html/2607.22375)** —— agentic quality-diversity 搜索做 idea 生成。
- **[Idea Search](https://arxiv.org/html/2608.08958)** —— 用 idea 引导树搜索来探索多样的科学方法。
- **[FlowPIE](https://arxiv.org/pdf/2603.29557)** —— 测试时的科学 idea 演化。
- **[Evolving Idea Graphs with Learnable Edits-and-Commits](https://arxiv.org/pdf/2605.04922)** —— 多 agent ideation。
- **[ScientistOne](https://arxiv.org/pdf/2605.26340)** —— chain-of-evidence 做自主研究；评测用 ScholarPeer（gemini-3.1-pro 背书 + 文献检索）。
- **[APRES](https://arxiv.org/pdf/2603.03142)** —— agentic 论文修改与评估系统。
- **[GIANTS](https://arxiv.org/pdf/2604.09793)** —— 从科学文献做 insight 预期生成。
- **[Towards End-to-End Automation of AI Research](https://arxiv.org/pdf/2606.15497)**、**[How Far Are We From True Auto-Research?](https://arxiv.org/html/2605.19156v1)**、**[AI for Auto-Research: Roadmap & User Guide](https://arxiv.org/pdf/2605.18661)** —— 三篇路线图/综述。
- **[SciDER](https://arxiv.org/pdf/2603.01421)** —— 数据驱动的端到端 researcher。
- **[LLM-Metrics](https://arxiv.org/pdf/2605.22176)** —— 用 LLM 记忆度量研究影响力。
- **[Can LLMs Generate Novel Research Ideas?](https://arxiv.org/abs/2409.04109)** —— Si 等的前作，100+ NLP 研究者的大规模人评，这一整条线的起点。
- **[Chain of Ideas](https://arxiv.org/pdf/2410.13185)** —— IdeaArena 协议的出处。
