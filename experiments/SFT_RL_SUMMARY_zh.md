# SFT + RL 实验总结（2026-07-24）

> 目标：系统找 9B 的最佳 SFT recipe + average(soup) setting，并回答「数据 fix（去污染 verbatim-code gate + maintain 组合）后，是不是比 fix 前更好」；以及 35B RL 的正确打开方式。所有分数为 strict5 口径（固定 denominator，mean@5）。

## 0. 一句话结论

- **SFT 端**：直接 full-FT 会塌 FCS；**average(soup) 必须做**，且**配方 B（最新 maintain + 旧 maintain 全加，`allver`）在 α=0.1 时 FCS=7.34，是唯一超过 base(7.05) 的 full-FT soup**。数据 fix（去污染）主要**帮 ALE**（ gated_a10 ALE=377.4 > base 356.6 > 旧 clean_a10 366.2），**对 FCS 没超过 allver 单用**。
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

### 1.3 数据 fix（去污染 verbatim-code gate）前后对比
| 指标 | fix 前（clean_a10 / allver_a10） | fix 后（gated_allver） |
|---|---|---|
| FCS | clean_a10 6.52 / **allver_a10 7.34** | a5 6.05 / a10 5.77 / a20 5.54 |
| ALE | clean_a10 366.2 / base 356.6 | **a5 367.9 / a10 377.4**（均 > base） |

**结论：数据 fix（去污染）主要帮 ALE**（ gated_a10 ALE 377.4 > 旧 clean_a10 366.2 > base 356.6）；**对 FCS 没超过 allver 单用**（去污染的 FCS 塌缩主要来自创新数据本身的过发散，不是脏代码噪声）。**最佳组合 = allver maintain（FCS 高）+ gated 干净数据（ALE 高）。**

### 1.4 研究(research) 与最强项
- soup 的 research 普遍 ~10-11.5 << base 19.7（full-FT 在 research 上也亏）。
- **最强单模型仍是 LoRA r32_s01：FCS 9.83**（远超所有 full-FT soup）。full-FT 路线的天花板明显低于 LoRA。

## 2. 35B RL：synth 毁模型，research 才对

| | 结论 |
|---|---|
| **synth RL**（500 优化题） | **净有害**：reward 升但 FCS 9.83→5.73、ALE 447→325、Research 也掉。机制 = 过优化→coherence collapse（重复循环+过早放弃+解质量降），**不是截断、不是 ceiling**（3-subagent 取证）。held-out val reward=0.014（无信号）。 |
| **research RL**（64 题研究） | **有效**：reward 有信号（max=1.0、短输出、~3%截断），s6 模型 research **18.4→19.57**（回升到 base 19.7 水平）。 |

→ **RL 应走 research（稠密可学的 reward），不走 synth。** 新加了 wave-2b ~600 题（总 1102 题），正在用更大 batch（50×16，KL 0.01）在 base 和 LoRA-r32s01 上试 research/synth 对比。

## 3. 当前最佳 setting（可直接复用）

- **9B full-FT + soup**：`allver` 数据（clean_decontam_traj + wave2 + maintain_r3 + 旧 maintain），α=0.1 → **FCS 7.34**。
- **9B 更强**：LoRA r32_s01 → **FCS 9.83**。
- **35B RL**：research 数据 + KL anchor 0.01 + clip-higher + cap 40960 + NCCL 超时 1800s。
- 数据管线：build_sft.py verbatim-code gate（train_answer 必须与 answer.md 逐字一致，否则回退到已评审 answer）。

## 4. 待办 / 下一步

- research-RL 跑满 20 步（base + LoRA 两臂，50×16，1102 题）→ 看 research 能否超过 base + FCS/ALE 有无附带收益。
- 35B：用 allver/gated 的最佳 9B recipe 复刻到 35B SFT，再接 research-RL。
- 修 symbolic_regression 的 PySR 环境（reward adapter 里 2 个题族 fail-soft 0）。
