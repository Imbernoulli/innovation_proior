# SFT + RL 实验总结（2026-07-24）

> 目标：系统找 9B 的最佳 SFT recipe + average(soup) setting，并回答「数据 fix（去污染 verbatim-code gate + maintain 组合）后，是不是比 fix 前更好」；以及 35B RL 的正确打开方式。所有分数为 strict5 口径（固定 denominator，mean@5）。

> ⚠️ **2026-07-26 口径审计**：本文早期的若干 ALE 结论**统计上站不住**（ALE 只有 **10 道题**，同一模型两半的分差中位数 41.6、最大 135.6）。审计结论与更正见 **§1.6**，那里的表是**唯一**按统一口径重算过的表；与 §1.5 及更早表格冲突时**以 §1.6 为准**。

## 0. 一句话结论

- **SFT 端**：直接 full-FT 会塌 FCS；**average(soup) 必须做**。α=0.1 附近的几个配方（`allver` / `pure` / `gated_v2_a5` / `coding`）FCS 都落在 6.1–6.8，**彼此在噪声内不可区分**（FCS 全量 172 题的 SE≈0.5）；能确定的只有「**低 α 好、α≥0.3 明显塌**」。ALE 的所有配方间比较**都在噪声内**，不要用来选 recipe。
- ⚠️ 早期写的「allver_a10 FCS 7.34 是唯一超过 base」用的是 strict5 口径、且 **base 从没在同一条 official 管线上测过**；base 锚点正在补测（job 11634479/80），补完前该断言**不成立**。
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

→ **任何 full-FT 都塌 FCS；maintain 救不了直接 SFT。**

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
| maintr3_allver_a10 | **6.765** | 340.3 |
| maintr3_pure_a10 | 6.417 | **429.1** |
| maintr3_pure_a20 | 6.385 | 375.1 |
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
4. **base 9B 从来没在这条 official avg@5 管线上测过**（旧 base 数 7.05/356.6 是 strict5 + 另一次 eval）。所有「超过 base」的断言目前**没有同口径锚点**。已提 `base_q35_official`（11634479/11634480）补测。

**数据完整性**：每个 shard 的真实 error（非 null）2–6 / 457，~1%，低于 MAX_ERRORS=12 → 分数本身干净，问题全在**汇总与解读**，不在评测。少数 shard 题数不满 172（allver_a30 162、gated_v2_a20 169 / a30 170），已在表中按 num_problems 加权。

**补齐中的 α 洞**（模型已存在，只缺 eval，已提交 pli）：`maintr3_allver_a20`（11634484/85）、`maintr3_pure_a5`（11634489/90）。仍缺 eval 的：clean_a5、filt_a20、nomath_a10/a20、longthink_a20、method_minthink_a10/20/30。

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
