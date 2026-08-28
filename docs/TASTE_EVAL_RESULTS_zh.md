# Taste 评测套件：结果与数据审计

2026-08-27。评测选自 [TASTE_EVAL_SURVEY_zh.md](TASTE_EVAL_SURVEY_zh.md) /
[TASTE_EVAL_SHORTLIST_zh.md](TASTE_EVAL_SHORTLIST_zh.md)。
harness 见 `experiments/scripts/eval/taste/README.md`。

> **这套评测全部是 judge 型或客观 GT 型，没有任何一档需要我们真跑实验占卡。**
> 执行落地那一档由既有的 FCS / ALE / MLS / Research 承担，这里不重复。

## 0. 一句话

在这套「判断 / 生成洞察」的评测上，**我们训练过的模型没有一处比 base 显著更好**；
唯一扛过多重比较校正的差异是**负向**的。但这**不推翻**执行类评测上的既有结论
——那是另一个轴（见 §5）。

那条负向拆开之后：**训练没有让模型对论文变笨**（两序一致时的准确率 0.800 vs base 0.793），
它做的是**拆掉认知刹车**——「我无法核实」从 93.3% 掉到 54.4%，断言具体引用数字从
41.3% 涨到 56.5%，位置粘滞率配对上升 +5.6～+7.2pt。而「先锁答案、再编规则圆场」
这个失败模式**是 base 自带的**（反向筛的 24 个控制 case，24/24 base 也这么干），
我们只是提高了它的触发率。根因是体裁：2,901 条轨迹里落地时带保留的只有 4 条（§3.1–3.3）。

## 1. 协议

- **采样**：Qwen3.5 官方 thinking 配置，也是本仓 `EVAL_ON_JIAOLAB_zh.md` §4.2「不可动」的那套：
  `T=1.0 / top_p=0.95 / top_k=20 / min_p=0 / presence_penalty=1.5 / repetition_penalty=1.0`。
  no-thinking 对照档用 `T=0.7 / top_p=0.8`。
- **`presence_penalty=1.5` 不可省。** 第一轮漏了它，16–51% 的样本复读到 32k 上限被判错，
  base 在 SciJudge thinking 上从 **65.3% 掉到 26.2%**，整张表作废重跑。
  截断样本 100% 是字面 25-gram 死循环。
- **`repetition_penalty` 必须保持 1.0**：1.15 曾把 FrontierCS 从 7.231 打到 0.666
  （记在 9B model card 上，仓库内无底稿，属未核实声明）。
- **配对 bootstrap** + **Holm 多重比较校正**。37 次比较里未校正 p<0.05 有 5 个，
  纯运气预期 1.9 个；**校正后只剩 2 个存活**。
- **打分类评测一律用校准无关的指标**（模型的刻度不同，绝对分会混进偏移）：
  RINoBench 用成对排序一致率，SoundnessBench 用 AUC。

## 2. 主结果

九个臂（4B: base/wd01/soup_a10；9B: base/soup_a10/soup_wd03_a20/rl_base/rl_soupNEW10/rl_soupWD03）
× 11 个 benchmark。**Holm 校正后存活的只有两条，都是 4B 全量 SFT 在 SciJudgeBench 上的负向**：

| | base | ours wd01 | Δ | Holm |
|---|---:|---:|---:|---|
| SciJudge thinking（1000 对） | 65.3% | 60.3% | **−5.0pt** | ✅ 存活 |
| SciJudge no-thinking（1000 对） | 60.5% | 52.5% | **−8.2pt** | ✅ 存活 |
| GiantsBench（judge 1–10） | 4.54 | 4.13 | −0.41 (p=.0004) | — |

其余全部不显著，或在另一个家族里符号翻转。**没有任何正向显著。**

**harness 可信度**：base 在 SciJudge main 59.8–60.5（论文 Qwen3-4B-Instruct **58.1**）、
时间 OOD 63.4（论文 **64.7**）。两个独立锚点都对上。

**污染**：对 `methods.json` 的 531 个 arXiv id 查过——GiantsBench 抽样 400 条 **0** 命中、
SciJudge main 11/1000、SoundnessBench 1/1099。SciPredict 和 PAIR-IQ 结构性零污染
（题目分别晚于 2025-03、来自 ICLR/NeurIPS 2024-25 评审）。

## 2.5 9B 家族：独立复现了同一个结论

6 个 bench × 5 个对照臂，item 配对，Holm 校正（family = 30）：

| bench | 指标 | base | soup_a10 | soup_wd03_a20 | rl_base | rl_soupNEW10 | rl_soupWD03 |
|---|---|---:|---:|---:|---:|---:|---:|
| SciJudge | 换序一致率 | 0.638 | 0.640 | 0.645 | 0.638 | 0.623 | 0.650 |
| PAIR-IQ | 换序一致率 | 0.455 | 0.466 | 0.469 | 0.450 | 0.447 | 0.448 |
| Soundness | 准确率 | 0.632 | 0.644 | 0.639 | 0.638 | 0.621 | 0.604 |
| SciPredict | 准确率 | 0.242 | 0.267 | 0.280 | 0.224 | 0.248 | 0.304 |
| RINo | 负 MAE | −1.004 | −0.986 | −1.007 | −1.000 | −1.069 | −1.072 |
| GiantsBench | 洞察相似度 1-10 | 4.728 | 4.751 | 4.649 | 4.793 | 4.711 | **4.335** |

**Holm 后 1 个显著，是负的**：GiantsBench / rl_soupWD03 −0.390 [−0.617,−0.169]，p=0.0003。
分解验证过不是「无洞察被 floor 到 1 分」的假象：仅按已判分项算仍是 −0.269 [−0.485,−0.056]，
约 31% 来自格式失败（抽不出洞察 base 0.8% vs 该臂 4.8%），69% 来自判官给分真的更低。

**两家族合计 67 个对照，Holm 后 0 正向、3 负向。** 三个负向指向同一件事——
**输出可靠性下降**，不是知识变少。

> ⚠️ 报表 `experiments/scripts/eval/taste/report9b.py` 内置了判分覆盖率防呆。
> 第一次跑时 giants/rl_soupNEW10 读出 Δ=−0.803 p=0.0003、Holm 后唯一显著——
> 那是判分只跑了 79/400 项的假象，完整 400 项下是 **+0.018 p=0.89**。
> 短文件照样能算均值、照样能过 bootstrap，没有任何指标会抗议。

## 3. 那条唯一确凿的负向，拆开看不是「变笨」

把换序一致性拆成 `P(两序一致) × P(一致时答对)`：

| | base | wd01 |
|---|---:|---:|
| 两序一致率 | **0.829** | **0.759** |
| **一致时的准确率** | 0.793 | **0.800**（不显著更高） |
| 单次生成准确率 | 0.743 | 0.724 |

**缺口 100% 来自一致性。** 129 道「base 对、wd01 错」的题里，75% 是换序翻转而非判错。
「base + 5.6% 随机翻转」这个零模型能精确复现 wd01 的全部数字。
换序一致性对单次可靠性大致是平方关系，单次掉 1.9pt 报出来就是 5.0pt。

被排除的解释：**不是想得少**（按长度分箱对齐后每一档都更不一致）；
**不是难题上判错**（一致时的准确率四个难度档全部持平或更高）；
**不是学会吹新颖性**（novelty 措辞密度 base 0.25 / wd01 0.23，反而是 base 更依赖
「这看起来像综述/会议论文」这类表面线索：25.8% vs 15.7%）。

### 3.1 读了 121 个 case：这个失败模式是 base 自带的，我们只是提高了触发率

先说一个方法论坑，因为它差点让我们得出反向结论。

**第一批（97 例，有偏）**：按「base 两序都对、wd01 两序答同一个字母」筛样，
16 个独立子代理按同一份 rubric 盲读。结果 mirror 92/96、空泛规则 93/96、
base 两序锚定同一线索 96/96。

**这三个数全是选样条件蕴含的，不是发现。** 按「base 对、SFT 错」筛出来的样本，
base 当然一致、SFT 当然自相矛盾。单独看这批数据，会得出「decide-then-justify
是我们训出来的」这个错误结论。

**控制组（24 例，反向筛）**：从 58 对「wd01 两序都对、base 答同一个字母」里抽 24 例，
同一份 rubric，4 个子代理盲读。

| | 第一批（SFT 失败） | **控制组（base 失败）** |
|---|---:|---:|
| 前后矛盾地描述同一篇论文 | 92/96 | **24/24** |
| 论据是空泛规则 | 93/96 | **24/24** |
| 同一条规则在两序里被反向使用 | — | **24/24** |
| 首次表态位置（占思考比例）中位 | 0.30 | **~0.28** |

base 的原话，同一个模型、同两篇摘要、两个顺序：

> "Surveys > Specific Technique Papers in citation volume"
> "**Method Papers** beat **Survey Papers** in raw citation count"

出版日期这个顺序无关的事实被两边同样地机会主义使用——指向想要的字母时是
"an earlier publication gives Paper B a slight advantage"，指向反面时是
"recency is not a differentiating factor"。base 甚至会伪造对**评测答案本身**的记忆：
"I found a memory trace... In similar examples, the one concerning Dark Matter tends to be ranked higher"。

**结论修正：先锁字母、再编规则圆场，是 Qwen3.5 自带的行为。我们的数据没有创造它，
只是把触发率推高了。**

### 3.2 那么我们到底改变了什么（配对 + soup 对照）

soup 一列是 α_eff=0.023 的近-base 对照（见 §6），它在每一行都贴着 base，
所以下面的差异确实来自 SFT 方向而非噪声：

| SciJudge，1000 对配对 | base | **wd01** | soup |
|---|---:|---:|---:|
| 位置粘滞率 | 17.1% | **22.7%** | 18.5% |
| 配对差 vs base（think） | — | **+5.6pt [+2.6,+8.6]** | +1.4pt ns |
| 配对差 vs base（no-think） | — | **+7.2pt [+3.8,+10.7]** | +0.1pt ns |
| 说「我无法核实 / 查不到」 | **93.3%** | **54.4%** | 91.9% |
| 断言一个具体引用数字 | 41.3% | **56.5%** | 41.2% |
| 固定开场 "Here's a thinking process" | 0% | **77%** | 0% |

一句话：**训练没让模型对论文变笨（一致时准确率 0.800 vs 0.793），
它让模型不再说「我不知道」，并且更早停下。** base 本来就有那个毛病，
我们把它的刹车——认知对冲——拆掉了 93%→54%。

**一个被证伪的猜想（记录下来免得被再次引用）**：我原以为是「想得短 → 位置偏见」。
不成立：no-think 模式下两个模型几乎都不对冲（1.1% / 0.2%），wd01 仍然 +7.2pt 粘滞。
「少对冲」和「更粘位置」是两个独立后果，不是一条因果链。

### 3.3 在 v2 训练语料上复核落点统计

`innovation_v2_timeonly.jsonl`（2,901 行，wd01 实际训练用的那份）最后一个计损助手轮：

| | |
|---|---:|
| 最终答案里提到走过死路 / 失败 | **11 / 2901 = 0.4%** |
| 最终答案带 caveat 或 limitation | **4 / 2901 = 0.1%** |
| 整条 trace 出现过一次「我不确定」 | 211 = 7.3% |

**2,901 条轨迹，落地时带保留的有 4 条。** 这不是数据质量问题，是**体裁问题**：
我们写的是已知答案的发现的事后复盘，照着 answer.md 往回写，所以每条必然落地。
模型学到的是科学家**汇报成果时的语气**，不是科学家**做决定时的过程**。

## 4. 数据审计：为什么会这样

### 4.1 语料是按结果筛的，没有负类

对 2,885 条 innovation 轨迹（助手侧中位 29,935 字符）：

| | |
|---|---:|
| 提到备选方案（why not / instead of / one could） | 75.3% |
| **结尾 2000 字符里还带着一条被排除的路** | **1.2%** |
| **以「这条路走不通」结尾** | **0 / 2885** |
| 引用式点名具体前作 (Author, 20xx) | 14.0% |
| 只泛泛说 prior work / baseline | 37.0% |
| **结尾带局限 / 条件措辞** | **2.1%** |
| 结尾带绝对化措辞 | 4.1% |

**形式有，但不承重**：备选是开场的装饰，不影响落点；比较是泛化的，不是具体的；
结尾几乎一律是断言。

而这套评测的 GT 全部是**对比标签**（哪个引用高、哪个评审分高、方案成不成立——
SoundnessBench 有 458 条是「不成立」）。**只有正类的数据训不出判别器。**

### 4.2 行为上的兑现

| 每千词 | base | wd01 | |
|---|---:|---:|---|
| 断言式自信措辞 | 0.43 | 1.02 | 2.4× |
| 显式不确定 | 0.35 | 0.27 | 0.77× |
| 编造引用数（"~50-150 citations"） | 0.09 | 0.30 | 3.4× |

- **题目特有内容覆盖掉了 55–67%**（RINo/GiantsBench），即**实际做的前作比对少了三分之二**
- RINoBench 给分：base **一个 5 分都没给过**，wd01 给了 **19.2%**（人类 12.3%）
- 抽到的原句：*"It is **genuinely** absent from the list. Rating this at 5."*
  ——在只做了 base 八分之一的前作检查之后

### 4.3 训练混合：81% 的 token 不是 innovation

`sft_full_wd01.yaml` 的 `dataset: innov_v2,maintain_w2w3`：

| 切片 | 行数 | 助手侧词数 |
|---|---:|---:|
| innovation | 2,901（32%） | 13.8M（**19%**） |
| maintain_w2w3（rollout 蒸馏） | 6,041（68%） | 58.1M（**81%**） |

**现在无法把任何效果归因给 innovation prior**。需要一个只训 innovation 的臂——目前没有。

### 4.4 少数派模板被放大 4–5 倍

`"Here's a thinking process:\n\n1.  **Analyze User Input:**"` 在 maintain 里占 **20.1%** 的文档，
一个 epoch 后在 wd01 输出里占 **82.3%**（PAIR-IQ / SciPredict 上 **100%**），base 是 **0/5937**。
`✅`（11.8%→86%）、`**Self-Correction**` 头（16.8%→74%）同理。

**溯源**：来自真实 rollout，且 **100% 来自 Qwen3.6-27B**（Qwen3.8-27B 的 3,062 行里 0 条），
按域集中在 ifollow 81.5% / reasoning 32.5% / math 21.6%，code 只有 2.2%。
外部教师（deepseek/poe）0%。

**实测它不影响准确率**（带 60.8% vs 不带 59.4%）——它证明的是**这批数据里的表面模式会被高倍放大**。

**provenance 恢复用 `sft/filter_maintain.py`**（ziran, `27b42550e`）：从 **git 历史**
推断——08-13 那个 commit 的快照里已有的行是 q36，之后新增的是 q38。**远程也能用**，
因为只依赖 git 历史，不依赖 gitignored 的 tags 文件。

本文作者在 gpublaze 上用**记录下来的真值**核过它（那台机器还有 `sft/_wave3_tags.jsonl`，
逐行对齐、带组装时写入的 `source`）：两边都给判定的 5,166 条上**一致率 99.83%**，
**q38 侧完全精确**（3,062 条一条不差），唯一分歧是 **9 条 q36 被标成 q38**——正是那个
commit 自己预言的 "9 apart"，方向单一、影响可忽略。校验结果已写进该脚本的注释。

完整构成（真值）：Qwen3.8-27B 3,062 / Qwen3.6-27B 2,810 / 空 111 / deepseek-v4-pro 33 /
codex:gpt-5.5 9 / qwen3.7-max 7 / 未匹配 9。

### 4.5 对称审查：我们的推理是承重的，蒸馏版不是

§4.1–4.4 讲的是语料**缺什么**。这一节讲它**有什么**——因为我一度顺着
「只讲故事、不做推理」的说法往下推，被用户纠正，然后做了对称的对抗性审查
（同一套 rubric，10 个独立 Opus 判官，一边挑蒸馏版的错、一边挑 gold 的错）：

| | n | 带事实/算术错误的 case | 95%CI |
|---|---:|---:|---|
| **gold 手写原文** | 24 | **9 = 37.5%** | [21.2, 57.3]% |
| Inkling 蒸馏（Arm A） | 80 | **75 = 93.8%** | [86.2, 97.3]% |

置信区间完全不重叠。而且性质不同——判官对 gold 的原话是
*"none of the flagged items is a hard arithmetic or published-result error;
all four are hedged imprecision or causal overreach"*。

**gold 声称的 79 次验证动作，判官独立复算确认 69 次正确 = 87%**，
而且是真去复现的：L-SHADE 的「5000 evals 买到 179 代」从 N=90 线性递减模拟**正好得 179**；
BitNet 的「RMS error ≈ 0.61」解析上是 √(1−2/π)=0.6028，重跑得 0.603；
硬币找零的每一次手工 trace **逐位复现**。

蒸馏版则是 *"the self-verification language is present and the verification itself is not"*：
`"[6,5,4,3] gives 17 (cross-checked by an independent brute force)"`（最优是 14）、
`print(surf)  # -0.280899`（代码实际算出 ≈−0.076）、
三种量化策略报出逐字节相同的分数。

**所以问题分两层，修法完全不同：**

| | 我们的原文 | 蒸馏版 |
|---|---|---|
| 推理承重、算术正确 | ✅ 87% | ❌ |
| 展示搜索（失败分支） | ❌ 4/2901 | 有搜索的**语言**，但否掉的常是正确答案 |

**缺搜索痕迹是加法；蒸馏把已有的推理正确性换成了搜索的赝品，是净损。**

### 4.6 年份戳：19.7% 的行拿到错误的时代锚

`sft/build_sft.py` 的 docstring 写明 `trajectory first-method year`——
阶梯上**每一级都用第一级的年份**。`ml-dimensionality-reduction` 是
PCA(1901) → TriMap(2019) → PaCMAP(2021) → UMAP(2018) → t-SNE(2008)，
**五级全部标 "It is now year 1901"**，于是出现「1901 年 import sklearn」。

`tools/audit_year_stamps.py` 全量扫描：**572 / 2901 = 19.7%**，
其中 >50 年 28 行、20–50 年 55 行。只用不可能和普通英文撞车的判据
（显式引用、库名、带数字/连字符的模型名）——早前一版用 `clip` 之类做关键词，
188 次命中 143 次是梯度裁剪，误报率 76%。

修法不是一行：`meta.json` 每一级有 `slug` 但 `year` 全为 null，要先补逐级年份。
且这是 docstring 写明的**设计决策**而非笔误，改前需确认原意。

**连带影响**：早前那个「教师有没有穿越」的检测（手写 9.7% vs 蒸馏 9.5%，无差异）
**不可用**——基线本身 19.7% 是错的，模型早就学会忽略这个字段了。修好后需重做。

## 5. 和执行类评测的关系

那边的好结果是真的（`RL_DATA_REPORT.md:145`、`MLS_AGENTIC_REGRESSION_REPORT.md:455`），
但**是另一个轴**：那边测「能不能做出来」，这边测「能不能判断」。
`CASE_STUDY_zh.md:182` 早就写过这个区分。

**唯一真正冲突的一点**：把 RL 后的收益归因给「soup 是更好的起点」。
pre-RL 的 soup 和 base 在四个客观评测上分不开（p=0.19–0.96），权重上 94.7% 逐位相同；
而三个 RL 模型**彼此的距离（1.01–1.08e-3）比各自到 base 的距离（0.95e-3）还大**，
行为上也一样（Cohen's κ：RL 后彼此 < RL 前彼此，5/5 个 benchmark）。
**RL 位移是起点差异的约 12 倍。** 验证方法：同一起点换 seed 再跑一条，看 ALE 差多少。

## 6. 附带修掉的两个 bug

**bf16 存不下小 α 的 model soup。** 名义 α=0.1 实测 `α_eff=0.023`、89% 权重与 base 逐位相同；
α=0.2 实测 0.087。根因是 SFT 步长中位数只有 **1.31 个 bf16 ulp**，
bf16 里要让中位数权重动一下需要 **α ≥ 0.38**。
`cc_model_soup_merge.py` 已修（fp32 累加 + `--out-dtype float16` + 写完回读校验 α_eff，
对不上直接 SystemExit），两份拷贝已同步。**历史上那次 α 扫描真正扫的是 0 / 0.023 / 0.087 / 1.0。**
详见 memory `soup-bf16-alpha-collapse`。

**judge rubric 必须逐字。** `judge_ref.py` 曾对 GiantsBench 用 Figure 12 的改写版，
同一批生成上比逐字版低约 0.4 分——比大多数臂间差异还大。已改为转调逐字 prompt。
同理 **dtype 也要配对照**：bf16 base 与 fp16 base 在 GiantsBench 上差 0.25 分（p=0.044）。

## 7. 下一步

1. **换目标函数**。SFT 打 token 分，推理看起来合理就够；RL 打结果分，推理必须真帮上忙。
   GIANTS（§27）和 SciJudgeBench（§1）两篇独立证据同向：这类能力 SFT 推不动。
   SciJudgeBench 放出 **720,341 条训练对**，公开结果 Qwen3-4B **58.1 → 77.3**。
2. **补负类**。现成来源：PAIR-IQ 的 **4,086 篇 reject**（带 main/approach + reviewer 分，已在磁盘上）、
   ICLR 2022–2026 的 35,209 投稿 / 137,940 评审。这批同时是 RL 判别式 reward 的现成信号。
3. **innovation 语料微操**（按性价比）：只改结尾（加「什么观测会证伪」）→ 把已有备选搬到落点
   → 点名具体前作。第一条改动量只有全文的 1/15，且 RINoBench/SoundnessBench 直接读它的效果。
   §3.2 给了更锐的靶子：**把认知对冲加回去**，2,901 条里只有 4 条落地带保留是可直接干预的量。
4. **做一个纯 innovation 臂**，否则永远分不清测的是 innovation prior 还是蒸馏切片。
5. **Tinker 蒸馏臂（进行中）**：用 Tinker API 在同一份 innovation 语料上 LoRA 微调
   Inkling-Small，再让它 teacher-forced 重写每个计损轮的 `<think>`（answer / tool call
   原样保留，因为 observation 只对记录下来的 action 有效），用重写后的语料训 Qwen3.5-4B。
   对照配置与 wd01 **只差两行**（dataset / output_dir）。脚本见
   `experiments/scripts/tinker/`，配置见 `experiments/tinker_distill_4b/sft_full_distill.yaml`。
   验的是「手写 reasoning 太 off-policy」这个假设；要盯的风险是模型口吻会不会把
   语料里唯一的优点（第一人称科学家视角）一起蒸掉。

## 8. 产物

```
experiments/scripts/eval/taste/          harness（11 个 bench 的 prompt/解析/判分/打分/报表）
experiments/scripts/tinker/              Tinker 蒸馏臂（build_data / train_inkling / sample_inkling）
experiments/tinker_distill_4b/           蒸馏臂的 4B 训练配置（与 wd01 只差 dataset/output_dir）
docs/TASTE_EVAL_RESULTS_zh.md            本文
```
数据 `.cache/taste_eval/`、产物 `outputs_taste/`，均 gitignored。
