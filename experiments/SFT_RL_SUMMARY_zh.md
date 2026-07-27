# SFT + RL 实验总结（2026-07-24）

> 目标：系统找 9B 的最佳 SFT recipe + average(soup) setting，并回答「数据 fix（去污染 verbatim-code gate + maintain 组合）后，是不是比 fix 前更好」；以及 35B RL 的正确打开方式。所有分数为 strict5 口径（固定 denominator，mean@5）。

> ⚠️ **2026-07-26 口径审计**：本文早期的若干 ALE 结论**统计上站不住**（ALE 只有 **10 道题**，同一模型两半的分差中位数 41.6、最大 135.6）。审计结论与更正见 **§1.6**，那里的表是**唯一**按统一口径重算过的表；与 §1.5 及更早表格冲突时**以 §1.6 为准**。

## 0. 一句话结论

- **SFT 端**：直接 full-FT 会塌 FCS；**average(soup) 必须做**。α=0.1 附近的几个配方（`allver` / `pure` / `gated_v2_a5` / `coding`）FCS 都落在 6.1–6.8，**彼此在噪声内不可区分**（FCS 全量 172 题的 SE≈0.5）；能确定的只有「**低 α 好、α≥0.3 明显塌**」。ALE 的所有配方间比较**都在噪声内**，不要用来选 recipe。
- ⛔ **早期写的「allver_a10 FCS 7.34 是唯一超过 base」已被证伪**：base 同口径补测 = **6.816**，最好的 soup allver_a10 = 6.765，**打平但没超过**，其余全部更低。9B full-FT+soup 的真实成绩是「不掉分」，不是「涨分」。见 §1.6.1。
- **RL 端**：**synth RL 会毁模型**（过优化→coherence collapse，FCS/ALE/Research 全掉）；**research RL 是对的**（reward 有信号、不塌、research 18.4→19.57 回升到 base 水平）。

## 1. SFT + soup 矩阵（9B，全部从 Qwen3.5-9B-bf16，clean_full 超参）

### 1.1 直接 SFT（不 average）→ FCS 全塌
| 数据 | FCS | 说明 |
|---|---|---|
| base | 7.05 | 锚点 |
| clean_full_wd01（旧干净数据） | 2.42 | 塌 |
| clean_maintr3（+新maintain_r3） | 2.68 | 塌 |
| maintr3_pure（只有maintain） | 1.28 | 塌 |
| maintr3_filt（长度过滤） | 2.95 | 塌 |

→ ~~任何 full-FT 都塌 FCS；maintain 救不了直接 SFT。~~ **⚠️ 这句话是错的，见 §1.8**：`wave2 ×8` 回放的裸 SFT（无 soup）FCS = **6.954**，追平 base。塌陷取决于**数据配比**，不是 full-FT 本身。上表这几个塌了，是因为它们的对口成分剂量太低。

### 1.2 soup(average) α-sweep（soup = α·SFT + (1−α)·base）→ 部分恢复
| 模型 | α=0.1 | α=0.2 | α=0.3 |
|---|---|---|---|
| clean（旧） | 6.52 | 5.90 | 3.93 |
| pure | 6.42 | — | 5.51 |
| filt | 6.07 | 5.01 | 5.07 |
| coding（只用coding maintain） | 6.10 | 5.97 | — |
| nomath | — | — | 5.59 |
| **allver（新+旧 maintain 全加）** | **7.34** | — | 4.96 |

**关键发现**：
- **α 越小 FCS 越高**（更多 base），但创新倾向越弱 → α 是 trade-off 旋钮。
- **⭐ allver_a10（FCS 7.34）是唯一 > base 的 full-FT soup** —— 配方 B（把以前 deprecate 的旧 maintain 也加上）验证有效。
- maintain 的 domain 筛选（coding-only / 去数学物理）对 FCS 影响不大（都 ~6.1 < allver）。

**交叉验证（2026-07-25 巡检）**：用官方 leaderboard `avg_at_5` 口径（172 题全覆盖）重算，排名与 strict5 **完全一致**：allver_a10(6.765) > pure_a10(6.417) > coding_a10(6.102) > filt_a10(5.882) > clean_a10(5.323)。→ **"allver_a10 最优" 对评测口径稳健**。注意：单 shard 仅 86 题、mean@5 噪声大（coding_a10 两 shard 4.91 vs 7.30），**<0.5 分的差距在误差范围内**，absolute 值随口径变化（strict5 7.34 vs 官方 avg@5 6.765），结论只看相对排名。

### 1.3 数据 fix（去污染 verbatim-code gate）前后对比
| 指标 | fix 前（clean_a10 / allver_a10） | fix 后（gated_allver） |
|---|---|---|
| FCS | clean_a10 6.52 / **allver_a10 7.34** | a5 6.05 / a10 5.77 / a20 5.54 |
| ALE | clean_a10 366.2 / base 356.6 | **a5 367.9 / a10 377.4**（均 > base） |

**结论：数据 fix（去污染）主要帮 ALE**（ gated_a10 ALE 377.4 > 旧 clean_a10 366.2 > base 356.6）；**对 FCS 没超过 allver 单用**（去污染的 FCS 塌缩主要来自创新数据本身的过发散，不是脏代码噪声）。**最佳组合 = allver maintain（FCS 高）+ gated 干净数据（ALE 高）。**

### 1.4 研究(research) 与最强项
- soup 的 research 普遍 ~10-11.5 << base 19.7（full-FT 在 research 上也亏）。
- **最强单模型仍是 LoRA r32_s01：FCS 9.83**（远超所有 full-FT soup）。full-FT 路线的天花板明显低于 LoRA。

### 1.5 gated_v2（输入侧泄漏清理）soup 扫描 —— 强烈 FCS↔ALE 此消彼长（2026-07-26）

gated_v2 = 输入泄漏清理（218 题 statement 去掉解题剧透）+ gated 数据 + wave2 + allver maintain。soup α 扫描（官方 avg@5，172 题，每模型 2 shard）：

| 模型 | FCS | ALE |
|---|---|---|
| allver_a10（FCS 基准） | **6.765** | 340.3（原写 366.2，错） |
| gated_v2_a5 | **6.164** | 364.5 |
| **gated_v2_a10** | 4.606 | **413.0**（史上最高 ALE） |
| gated_v2_a20 | 5.553 | 385.3 |
| gated_v2_a30 | 4.229 | 332.8 |
| base | ~6.9 | 356.6 |

> ⚠️ **上表两处已证实写错，见 §1.6 更正**：(1) allver_a10 的 ALE 不是 366.2（那是 clean_a10 的数、串行了），实测 **340.3**；(2)「413.0 是史上最高 ALE」**是错的**，`pure_a10` = **429.1**（同为双 shard）。

**结论（更正后）**：
- **gated_v2 在任何 α 都没超过 allver_a10 的 FCS**（gated_v2 最高 a5=6.164 vs allver_a10=6.765）——但这 0.6 的差距**在噪声内**（FCS SE≈0.5），只能说 gated_v2 **没有改进 FCS**，不能说 allver 更好。
- 「gated_v2 换来了 ALE」**不成立**：413.0 既不是最高（pure_a10 429.1），与其它 α=0.1 配方的差距也小于 ALE 的噪声（±40）。
- α 关系**非单调、噪声大**；唯一稳健的方向是 **α≥0.3 明显变差**（FCS 3.9–5.3）。
- 总评（保守版）：输入泄漏清理（gated_v2）在现有测量精度下**看不出对 FCS 或 ALE 有净影响**。它的价值论证应回到数据侧（防泄漏本身是对的），而不是这两个指标上的分数。

### 1.6 ⭐ 口径审计 + 全量重算（2026-07-26，本文唯一权威表）

把 `outputs/cc_eval_soup_*` 下**所有** soup 的 `shard_*/summary_shard.json` 用同一口径重算（FCS = `official_leaderboard_metrics.frontiercs.avg_at_5`，按 `num_problems` 加权；ALE = `metrics.alebench.performance.mean@5`，两 shard 平均）：

**噪声先量化**（18 个双 shard 模型，同一模型两半之间的分差）：

| 指标 | 题数 | 半-半分差中位数 | 最大 | 全量均值 SE（推算） | **可区分阈值(≈2SE)** |
|---|---|---|---|---|---|
| FCS | 172 | 1.03 | 3.03 | ≈0.5 | **~1.0 分** |
| ALE | **10** | 41.6 | 135.6 | ≈18 | **~40 分** |

→ **ALE 只有 10 道题**，本文所有「ALE 涨了 10–60 分」的结论**全部在噪声内**，作废。FCS 差距 <1.0 也不可区分。

**双 shard（172 题，可信）**：

| 模型 | FCS | ALE |
|---|---|---|
| **base Qwen3.5-9B（同口径锚点，2026-07-27 补测）** | **6.816** | 347.0 |
| maintr3_allver_a10 | 6.765 | 340.3 |
| maintr3_pure_a10 | 6.417 | **429.1** |
| maintr3_pure_a20 | 6.385 | 375.1 |
| maintr3_pure_a5 | 6.208 | 339.0 |
| gated_v2_allver_a5 | 6.164 | 364.5 |
| maintr3_coding_a10 | 6.102 | 365.8 |
| gated_allver_a5 | 6.055 | 367.9 |
| maintr3_coding_a20 | 5.970 | 314.5 |
| maintr3_clean_a20 | 5.901 | 309.7 |
| maintr3_filt_a10 | 5.882 | 298.8 |
| gated_allver_a10 | 5.772 | 377.4 |
| maintr3_pure_a30 | 5.667 | 386.9 |
| maintr3_nomath_a30 | 5.567 | 371.5 |
| gated_v2_allver_a20 | 5.553 | 385.3 |
| gated_allver_a20 | 5.541 | 301.2 |
| maintr3_allver_a30 | 5.270 | 295.3 |
| gated_v2_allver_a10 | 4.606 | 413.0 |
| gated_v2_allver_a30 | 4.229 | 332.8 |
| maintr3_clean_a30 | 3.927 | 335.2 |

**单 shard（86 题，噪声翻倍，不要与上表并列排序）**：filt_a30 5.809/407.3、longthink_a30 5.630/311.0、**clean_a10 5.323/358.0**、coding_a30 4.281/364.6、longthink_a10 3.644/314.5。

**本次审计查出的 4 个问题**：
1. §1.5 表里 allver_a10 的 ALE 写成 366.2 —— 那是 clean_a10 的数，**串行了**；实测 340.3（shard 301.3 / 379.2）。
2. 「gated_v2_a10 = 史上最高 ALE 413.0」**是错的**：`pure_a10` 双 shard 429.1 更高。且两者差 16 分 << 噪声 40，本来也不该排序。
3. §1.2 的交叉验证里 **clean_a10(5.323) 只有 1 个 shard**（86 题），被当成全量数并入排名。
4. **base 9B 从来没在这条 official avg@5 管线上测过**（旧 base 数 7.05/356.6 是 strict5 + 另一次 eval）。所有「超过 base」的断言当时**没有同口径锚点**。

### 1.6.1 ⛔ base 锚点补测结果（2026-07-27）：**没有任何 full-FT soup 超过 base**

`base_q35_official`（11634479/80，同一条 2-shard official 管线）：**FCS 6.816 / ALE 347.0**（shard 5.45 & 8.19 / 314 & 380）。

| | FCS | vs base |
|---|---|---|
| **base** | **6.816** | — |
| 最好的 soup（allver_a10） | 6.765 | **−0.05（打平，在噪声内）** |
| pure_a10 | 6.417 | −0.40 |
| gated_v2_a5 | 6.164 | −0.65 |
| 其余全部 | ≤6.10 | 更低 |

→ **本文最初的核心结论「allver_a10 是唯一超过 base 的 full-FT soup」是错的。** 那句话是拿 strict5 口径的 soup（7.34）去比另一次 eval 的 base（7.05）得来的——两个数不同管线、不可比。同口径重测后：**9B full-FT + soup 这条路线，最好的结果只是「打平 base」，一个都没超过。**

这不改变别的结论，但改变整条线的意义：full-FT 的价值不在 FCS 涨分（涨不了），而在于**注入创新倾向的同时不掉分**（α=0.1 做到了打平）。真正超过 base 的仍然只有 **LoRA r32_s01（FCS 9.83）**——注意那个数也是旧口径，同样需要同口径复测才能引用。

**数据完整性**：每个 shard 的真实 error（非 null）2–6 / 457，~1%，低于 MAX_ERRORS=12 → 分数本身干净，问题全在**汇总与解读**，不在评测。少数 shard 题数不满 172（allver_a30 162、gated_v2_a20 169 / a30 170），已在表中按 num_problems 加权。

**补齐的 α 洞**（2026-07-27 已完成）：`maintr3_allver_a20` = FCS 5.469 / ALE 308.0；`maintr3_pure_a5` = FCS 6.208 / ALE 339.0。两者都低于各自的 a10 → **α=0.1 在 allver 和 pure 两条线上都是最优点**，这是目前唯一跨配方稳健的 α 结论。仍缺 eval 的：clean_a5、filt_a20、nomath_a10/a20、longthink_a20、method_minthink_a10/20/30。

### 1.7 ⭐ 07-08 之后数据做了什么 + 控制变量对照实验（2026-07-27 发起）

**问题**：FCS 最好的 `allver` 线用的是 **07-08 的数据**（`innovation_clean_decontam_traj`，2225 行，07-08 构建）。而 07-08 之后我们对数据做了 **1348 个 commit** 的精细优化。最新数据训出来的 `gated_v2_allver` 反而没更好（FCS 6.164 vs 6.765）。必须搞清楚是**数据改坏了**，还是**对比本身不干净**。

**07-08 之后到底改了什么**（1348 个 commit，只统计 methods/ trajectories/ data_v4/）：

| 类别 | commit 数 | 内容 |
|---|---|---|
| train_answer 修复 | ~558 | 恢复与 answer.md 逐字一致的代码；替换「自己编的」重实现 |
| 输入侧泄漏清理 | ~186+ | statement 里的 Background/Evaluation-settings 不再交代解法 |
| 去掉「表演式 bug 戏码」 | ~数百 | reasoning 里假装踩坑再修的戏剧化叙事被删（大量在「其他」里）|
| 批量删模板开头 | 47 traces | 规则式，无 LLM |
| 时代错置引用 / 中译英 / in-frame 违规 | ~34 | 前向引用的文献、中文残留、事后诸葛口吻 |

改动集中在 **07-21（510 commit）和 07-25（772 commit）**。文件层面：`reasoning.md` 893 个、`train_answer.md` 615 个、`context.md` 239 个被改。单元数几乎没变（methods 1303→1304，data_v4 347→358）——**是内容重写，不是加数据**。

同期 `sft/build_sft.py` 也改了 6 次（verbatim-code gate、`code/` 目录 canonical、输入泄漏清理配套）。

**为什么原来的对比不干净**：`allver` 和 `gated_v2_allver` 之间同时变了 **3 件事**——
1. 数据内容（同名题目里 **80%** 的正文被改过）；
2. 集合本身（旧 traj 独有 74 条 / 新 gated_v2 独有 170 条）；
3. **wave2 成分不同**（allver 用 `innovation_wave2_clean` 1352 行，gated_v2_allver 用 `innovation_wave2_r3` 758 行）。

**对照实验设计（已发，2 臂）**：用 **HEAD 的 build_sft.py** 分别跑 **07-08 的数据树**（`git worktree` at `8ae41b601`，builder 和 decontam gate 都拷 HEAD 的进去）和 **当前数据树**，其余全部锁死：

| 臂 | 数据内容 | builder | 其余 mix | job |
|---|---|---|---|---|
| `sft_q35_ctl_old` | **07-08** | HEAD | wave2_clean + maintain_r3 + maintain | 11658797 |
| `sft_q35_ctl_new` | **当前** | HEAD | 同上（逐字相同）| 11658798 |

两个 yaml **只差一行**（dataset 名）。构建结果：ctl_old 2702 行 / ctl_new 2590 行；`method_ta_bypass` 从 **590 → 7**（说明 558 个 train_answer 修复是真的生效了——旧数据里 590 个 method 的 train_answer 对不上 answer.md，被迫回退到 answer 通道）。

跑完各做 α=0.1 soup + FCS/ALE 同口径评测，**ctl_new − ctl_old 就是 1348 个 commit 的净效果**，不再混别的变量。另外 `ctl_old` 与已训的 `allver`（旧 builder + traj 子集）之差，单独给出 **builder 改动**的效果。

已知的一个预期风险：去掉「表演式 bug 戏码」删掉的是模型学到的**自检行为**。如果 FCS 奖励「写完检查一遍再交」的代码，这一项可能是负效果——ctl 对照能验证这个猜想。

### 1.8 ⭐ maintain 数据：来历、版本、结果（2026-07-27 梳理）

**来历**：不是我们造的数据。是从 HuggingFace 扒的 Qwen-distill 能力回放集，来源 `khazarai` / `WithinUsAI` / `armand pi (Claude-Code)` / `nvidia Open-SWE`，**903 条**。06-21 从 `innovation_distill` 改名 `innovation_maintain`，用途是**回放 base 的原有能力、抵消创新数据造成的塌陷**。

⚠️ **07-06 被从 repo 里删了**（commit `a0827ea48`「Drop HF-scraped maintenance set」，理由是"training is innovation-only now"），连构建脚本 `sft/build_maintain.py` 一起删。→ **maintain 现在无法从 repo 重建**，只剩 `LF-innov/data/` 下已构建的 jsonl。要继续用就得守好这些文件。

**版本谱系**：

| 注册名 | 文件 | 行数 | 时间 | 说明 |
|---|---|---|---|---|
| `innovation_maintain`（"旧"） | maintain_sft_u.jsonl | 903 | 06-22 | 初版 |
| `innovation_maintain_r3`（"新"） | maintain_r3.jsonl | 903 | 07-06 | fold-think 修复后重渲染 |
| `innovation_maintain_r3_filt` | | 622 | 07-21 | 长度过滤（均长 108k→35k 字符）|
| `innovation_maintain_r3_coding` | | 463 | 07-21 | 只留 coding；**think 中位数=0**（多数无思考）|
| `innovation_maintain_r3_nomath` | | 782 | 07-21 | 去数学物理 |
| `innovation_maintain_r3_longthink` | | 259 | 07-21 | 长思考；think 中位数 **8572** vs r3 的 273（过滤正确）|

⚠️ **「新旧 maintain 全加」其实是同一批数据加了两遍。** 逐条比对：旧 900 个 unique prompt 与 r3 的 900 个**完全重合（重叠 900，各自独有 0）**，但**逐字相同的 0 条**——r3 只是修了 fold-think 的重渲染版。所以 `allver` = `maintain_r3` + `maintain` **= 同样 900 条 × 2 剂量**，而且其中一份是**我们特意修掉的那个旧渲染**。原文档说的「把以前 deprecate 的旧 maintain 也加上」不是"增加多样性"，是**加倍剂量（并混入一份已知有 bug 的渲染）**。

**结果**（官方 avg@5，soup α=0.1；标 † 者仅单 shard）：

| 模型 | maintain 成分 | FCS | ALE |
|---|---|---|---|
| base | — | **6.816** | 347.0 |
| allver_a10 | r3 + 旧（2×903）| 6.765 | 340.3 |
| **pure_a10** | **只有 r3，完全没有创新数据** | **6.417** | **429.1** |
| coding_a10 | coding 463 | 6.102 | 365.8 |
| filt_a10 | filt 622 | 5.882 | 298.8 |
| nomath_a30 | nomath 782 | 5.567 | 371.5 |
| clean_maintr3_a10† | r3 903（1×）| 5.323† | 358.0 |
| longthink_a10† | longthink 259 | 3.644† | 314.5 |

**读出来的三件事**：
1. **maintain 剂量单调有效**：r3 单份（clean_maintr3 5.323†）< r3 双份（allver 6.765）。allver 之所以是 full-FT 里最好的，很可能就是**剂量**，不是配方巧思。
2. **`pure`（只有 maintain、一条创新数据都没有）就能到 6.417**，逼近 allver 的 6.765 和 base 的 6.816。→ **掉分的是创新数据，保分的是 maintain**；创新数据在 FCS 上是净负担，maintain 是解药。
3. **maintain 的领域筛选都不如不筛**（coding 6.102 / filt 5.882 / nomath 5.567 / longthink 3.644† 全部 < 双份 6.765）。尤其 `longthink` 最差——只留长思考样本反而最伤 FCS。

**⚠️ 命名陷阱**：`c2_maint2x/4x/8x` 这三个模型**回放的是 wave2，不是 maintain**（配置里是 `innovation_wave2_clean_r2..r8`，全部指向同一个 wave2 文件）。它们的**原始 SFT（未 soup）**成绩：

| | FCS（raw SFT，无 soup）| ALE |
|---|---|---|
| wave2 ×2 | 3.759 | 302.9 |
| wave2 ×4 | 6.035 | 371.3 |
| **wave2 ×8** | **6.954** | 384.6 |

→ **这推翻了 §1.1 的「任何 full-FT 都塌 FCS」**：8× wave2 回放的**裸 SFT 就有 6.954，已经追平甚至略超 base(6.816)，完全没用 soup**。所以塌陷是**数据配比问题**，soup 只是补救手段之一，不是唯一解。**下一个该试的方向是 wave2 高倍回放 + maintain 双份，而不是继续在 soup 的 α 上抠。**

## 2. 35B RL：synth 毁模型，research 才对

| | 结论 |
|---|---|
| **synth RL**（500 优化题） | **净有害**：reward 升但 FCS 9.83→5.73、ALE 447→325、Research 也掉。机制 = 过优化→coherence collapse（重复循环+过早放弃+解质量降），**不是截断、不是 ceiling**（3-subagent 取证）。held-out val reward=0.014（无信号）。 |
| **research RL**（64 题研究） | **有效**：reward 有信号（max=1.0、短输出、~3%截断），s6 模型 research **18.4→19.57**（回升到 base 19.7 水平）。 |

→ **RL 应走 research（稠密可学的 reward），不走 synth。** 新加了 wave-2b ~600 题（总 1102 题），正在用更大 batch（50×16，KL 0.01）在 base 和 LoRA-r32s01 上试 research/synth 对比。

**新 run 健康验证（2026-07-26，step-1 rollout 实测 800 条）**：50×16 + KL 0.01 + 1102 题 + clip-higher + resp 40960 的配置下，**第一步就完全不塌**：
- reward 信号稠密：mean 0.273 / max 1.0 / **67% 非零**（旧 collapse 只有 0.014）。
- 长度不塌：median ~113k 字符，0% 短输出，无放弃迹象。
- 重复循环 **0.12%**（旧 22%）。
→ 机制上是旧 synth collapse 的反面。已据此发两个完整 20 步 run：base 臂（11618968）+ r32s01 臂（11623616，含 w2/w3 自动续跑）。代价：step 慢（~2.5h/step，35B×40k 长尾），20 步需跨 3 个 24h 窗口。

## 3. 当前最佳 setting（可直接复用）

- **9B full-FT + soup**：`allver` 数据（clean_decontam_traj + wave2 + maintain_r3 + 旧 maintain），**α=0.1**（official avg@5 = 6.765；strict5 口径曾记 7.34）。注意 pure_a10 / gated_v2_a5 / coding_a10 与它**在噪声内并列**，选 allver 是因为它在两套口径下都排第一，不是因为差距显著。
- **9B 更强**：LoRA r32_s01 → **FCS 9.83**。
- **35B RL**：research 数据 + KL anchor 0.01 + clip-higher + cap 40960 + NCCL 超时 1800s。
- 数据管线：build_sft.py verbatim-code gate（train_answer 必须与 answer.md 逐字一致，否则回退到已评审 answer）。

## 4. 待办 / 下一步

- research-RL 跑满 20 步（base + LoRA 两臂，50×16，1102 题）→ 看 research 能否超过 base + FCS/ALE 有无附带收益。
- 35B：用 allver/gated 的最佳 9B recipe 复刻到 35B SFT，再接 research-RL。
- 修 symbolic_regression 的 PySR 环境（reward adapter 里 2 个题族 fail-soft 0）。
