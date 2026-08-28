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
4. **做一个纯 innovation 臂**，否则永远分不清测的是 innovation prior 还是蒸馏切片。

## 8. 产物

```
experiments/scripts/eval/taste/     harness（11 个 bench 的 prompt/解析/判分/打分/报表）
tools/label_maintain_provenance.py  蒸馏语料的教师来源回接
docs/TASTE_EVAL_RESULTS_zh.md       本文
```
数据 `.cache/taste_eval/`、产物 `outputs_taste/`，均 gitignored。
