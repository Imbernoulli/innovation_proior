# 35B RL 问题诊断与修复 (2026-07-21,**2026-07-22 修正**)

> 用户诉求:「把 35B 的这个 RL 看看它的问题在哪里，然后解决问题」。

## ⚠️ 修正(2026-07-22,3 个 subagent 深度取证后)

**本文早先的「截断死亡螺旋 / 奖励错配 / ceiling」结论被证伪,以下是修正后的机制。** 内部(训练log)+外部(逐题)+样本级(逐条生成)三方取证一致:

**真正的机制 = RL 过优化 → coherence collapse(连贯性崩塌),不是截断,不是 ceiling,不是奖励错配:**
1. **奖励几乎无可学信号**:train rollout reward 在 *step 5* 就见顶后回落;held-out **val reward = 0.014**(比 eval 标尺低 ~100×)→ RL 根本没学到 synth 任务。
2. **零正则**:KL anchor 0.001(只占 loss ~0.1%,形同关闭)、entropy_coeff=0、对称 clip(无 clip-higher)。没有任何东西把 policy 钉在强 SFT 起点。
3. **→ 经典退化**:逐条生成取证(决定性):**截断率 s5→s20 反而下降(83%→27%)、格式完整度反而变好**,所以截断不可能是 s20 低分的原因。s20 的真实失败=**重复循环 0%→22%**(如 `//\n//\n//` 填满 → 0 分)、**过早放弃 ~10%**、**完成的解质量变差**(finished-answer reward 30→7.6)。塌到 ~11k 的模型在 32k eval cap 下并未被截断(用户的关键洞察),它就是退化了。
4. **伤害集中在模型最强的能力上**:45 个高分 FCS 题贡献了 >100% 的损失;105 个本来就做不出的题纹丝不动(净 +25);逐题 token 损失与分数损失 r≈0。FCS/ALE/Research 同一机制(一次全局崩塌)—— 所以「与 ALE 对齐」也是错的。

**修正结论**:不是「起点高于 RL ceiling」(那只是账面复述,因果不对——高分题是被针对性打掉的,没有收敛到某个水平),也不是「reward 与 FCS 错配」(9B synth RL 能上 9.10;且这里 ALE+Research 也掉,不只 FCS)。**修法 = 真 KL anchor 0.01–0.1 + entropy bonus/clip-higher(缺失的护栏)+ 用有信号的 reward(research)。** 32k cap/overlong-masking 是次要,不是根因。

(下文的旧机制分析保留作历史,已被上面取代。)

---

## (旧)0. 一句话结论

**当前 synth-only RL 是净有害的**:它从第一个评测点起就单调地把最强的 SFT 模型(FCS 9.83)打坏,同时把生成长度从 26k 塌到 11k(违反「输出不低于 32K」的硬约束)。根因是**奖励与目标基准错配 + 截断诱发的长度塌缩**。安全(不泄漏评测)的训练数据无法奖励 FCS 技能,所以 synth RL 在 FCS 上**只能保住、无法提升**。

## 1. 核实的数字(strict5 口径,denominator 固定)

| 35B 模型 | FCS | ALE |
|---|---|---|
| q36 base | 8.95 | 448.7 |
| **lora_r32_s01 = RL 起点(最强 FCS)** | **9.83** | 447.4 |
| now3 full-SFT(创新数据) | 7.34 | 458.1 |
| RL s5(reward 峰值) | 8.18 | 416.8 |
| RL s10 | 7.20 | 452.6 |
| RL s20 | 5.73 | 325.1 |

**RL 从 s5 就已经低于起点(8.18 < 9.83),且单调恶化到 5.73。** 而同期 synth 训练 reward 是**上升**的(0.18→0.23@s5):**reward 与 FCS 反相关 = reward hacking**。

## 2. 训练轨迹证据(broken run r32s01)

step 级 reward / response_length / **clip_ratio(截断率)**:

| step | reward | resp_len | trunc% |
|---|---|---|---|
| 1 | 0.184 | 26028 | — |
| 5 | 0.231 | 20858 | — |
| 6 (w3) | 0.229 | 22846 | **41%** |
| 9 | 0.156 | 13326 | 2% |
| 16 | 0.110 | 9525 | — |
| 18 | 0.053 | 9899 | — |
| 20 | 0.162 | 11411 | — |

**机制(death spiral)**:cap=32768,step 初 **41% rollout 被截断** → 截断=不完整代码=reward≈0 → GRPO 组内「长的=0、短的=较好」→ advantage 推短 → 长度 26k→9k → 太短解不动难题 → reward 崩到 0.05。

## 3. 根因(两层)

1. **分布/奖励错配(主因,导致 s5 就掉 FCS)**:synth RL 集 500 题**全是 open-ended 启发式优化题**(gen_checker,连续 Ratio 奖励,ALE 风格),**零道 exact-algorithm 题**。RL 把模型专门化成「启发式搜索器」,灾难性遗忘 exact-algorithm(FCS)能力。即便长度还没塌(s5 len≈20k),FCS 已从 9.83 掉到 8.18。
2. **截断诱发长度塌缩(次因,导致 s10→s20 及 ALE 也崩)**:见 §2。且违反「≥32K」硬约束。

## 4. 数据泄漏审计(决定能否用 exact-algorithm 题救 FCS)

FCS 评测集 = 182 题(172 public + ALE),ground_truth 为数字 ID + `ahc039`。

| RL 训练集 | #题 | 与 FCS 评测重叠 | 判定 |
|---|---|---|---|
| `frontiersmith_synth/train.parquet` | 500 (`fsx_*`) | 0 | **SAFE**,但全启发式,不奖励 FCS |
| `frontiercs/train_synthetic.parquet` | 10 (`frontiersmith_*`) | 0 | SAFE,但太小 |
| `frontiercs/train.parquet` | 172 (`frontiercs172`) | 命名即评测集 | **疑似全泄漏**(parquet 损坏读不出,按命名+历史标注保守判为泄漏) |
| `mixed/train_frontiercs172_...` | 172+ | 同上 | **疑似泄漏** |

**结论:没有「不泄漏且规模够」的 exact-algorithm RL 数据。** 所以 synth RL 无法诚实地提升 FCS。

## 5. 修复(已实施 / 在测)

修复杠杆(submitter `cc_rl35b_synth_submit.sh` 已全部 env 化):
- `MAXRESP 32768→40960`(≥32K,降截断率)— 已默认
- `CLIP_HIGH 0.2→0.28`(DAPO clip-higher,抗熵塌缩)— 已默认
- `ULYSP=4 / MAXTOKLEN=12288`(memory-safe,不 OOM)— 已默认
- **`KL_LOSS_COEF 0.001→0.01`(10× anchor,锚定到 9.83 起点保 FCS)— 修复核心**

**在测**:corrected smoke `rl35b_r32s01_klfix_smoke`(job 11471895,8 步,KL=0.01)。判定:
- 若 step8 checkpoint FCS 保住 ~9.8 且长度不塌 → KL anchor 有效,发完整 20 步(目标:ALE↑、FCS 保);
- 若 FCS 仍掉 → 提高 KL(0.03/0.05)或直接结论:**synth RL 不该在 FCS 上 ship,交付物就是 SFT lora_r32_s01(FCS 9.83 / ALE 447)**。

## 6. 老 broken run 处置

`checkpoints/rl_frontiersmith_synth/rl35b_{r32s01,base}`(各 524G,FSDP resume-only,s5/s10/s20 已导出 HF)已删除,腾 ~1TB(583G→1.6T)。HF 导出 `models_rl/rl35b_r32s01_s{5,10,20}_hf` 保留备查。
