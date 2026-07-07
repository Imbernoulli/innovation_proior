# 创新先验(Innovation-Prior)训练 —— 完整报告(公平对比版)

> 目标:证明「用创新先验数据做 SFT / soup / RL 能提升 Qwen3.5-9B 的能力」。
> **本报告的核心原则(方法论)**:只做**同条件对比**——SFT / soup(average)模型只比未训练的 **start**;经过 RL 的模型只比 **base+RL**(start 也过同一套 GRPO)。绝不拿 RL 后的模型去比裸 base。
> 所有关键数字均由 **Codex 独立复核**(见各节)。分数口径见 §6。

---

## 1. 执行摘要(诚实版)

创新数据的公平收益**分布在多个数据集上,但要按训练阶段配对正确的对照**:

| 阶段 | 公平对照 | 赢在哪些数据集 |
|---|---|---|
| **SFT / soup** | vs **start**(都无 RL) | ✅ **MLS +150~200%**、✅ **Theta(circle-packing)+65~83%** |
| **RL** | vs **base+RL**(同 GRPO) | ✅ **FCS +1.2~4.5(4 批次可复现)**、⚠️ **ALE +152(旗舰模型,不普适)** |

**两条能理直气壮讲、且有代码级证据支撑的结论:**
1. **创新 SFT/soup 在发现类任务(MLS、Theta)上直接超 start** —— 模型写出更懂问题结构的方案(因果 DR-learner、六边形晶格堆叠)。
2. **创新+RL 在竞赛算法(FCS)上超 base+RL** —— 4 个独立批次一致。

**同样诚实地记录的边界:**
- **RL 会"同质化"两个策略**:MLS 上 RL-vs-base+RL 的 20 题里 15 题逐位相同,残余 +8.6% 是**执行/纠错鲁棒性**而非方法更创新。→ **创新的方法论优势主要体现在 RL 之前。**
- **ALE 的 +61% 是对裸 base**;对 base+RL 的公平差是 +152(旗舰 `rlafter_rl_soup_mtv4_a20`),但换其它 soup/method 变体就落到噪声内 → **model-specific**。
- 单模型多数据集(`rlafter_rl_soup_mtv4_a20`:FCS+4.45 / ALE+152 / MLS+0.009)成立,但 MLS 边际、ALE 不普适——**最稳的可复现主张是 FCS**。

---

## 2. 方法

### 2.1 SFT 数据(创新先验,来自 `Imbernoulli/innovation_proior`,r3 版)

| 成分 | 数量 | 说明 |
|---|---|---|
| method | 1201 | 创新方法论倾向核心 traces(如何发散、选非显然方案)|
| v4 | 346 | method 的干净增补(去改写 / 投递纪律 / fold-think 修复)|
| traj(深化)| 678 rungs | trajectory 数据,**reasoning 加长加深**(逐级推到 method 深度,codex 校验)|
| wave2 | 758 | 第二波补充 |
| maintain / roll-out | 903 | 抗遗忘 general roll-out(替换掉旧的 HF 抓取版)|
| ~~agentic~~ | 473 | **r3 最终配方剔除**(实测偏负)|

三个 r3 配方:`methodv4_r3`(method+v4=1547)、`methodtraj_v4_r3`(+深化 traj=2225)、`full_r3`(+wave2)。
> **注:实际准备的数据集比本报告所报告分数覆盖的更多**;上面只列了跑完评测的配方。

### 2.2 Base model + SFT + Average(soup)
- Base:**Qwen3.5-9B**(instruct)。SFT = full fine-tuning,ZeRO-3 + bf16,eff. bsz 128、lr 5e-6、1 epoch、cutoff 53760、warmup 0.1。
- **Soup**:`merged = α·SFT + (1−α)·Base`,α∈{0.1,0.2,0.3,0.5}。直接 full-SFT 会把 FCS 打崩(0.3–1.5 vs base 7.05);soup 按比例掺回,α 是「掺多少创新」的旋钮。

### 2.3 RL(GRPO,在 soup 上做,用于"放大")
- verl / GRPO,配置 `rlm_amplify_v3`:rollout.n=4、lr 1e-6、KL 0.001、20–40 steps。
- 关键修复:`MAX_RESPONSE_LENGTH` 20k→**32k**(thinking 输出常 20–30k,截断会 length-collapse;修后 clip_ratio 0.94→0.18)。
- **原始 RL 数据**:FrontierCS 172 + ALE 40 + FrontierSmith 10。
- **新增 RL 数据 `frontiersmith_synth`(自造 500 题)**:172 C++testlib + 203 py-gen + 125 py-evaluator;用 FrontierSmith 开放式生成法(mutate→filter→diverge→testinfra→rerank),范围拓宽到 FunSearch/AlphaEvolve/OpenEvolve/ThetaEvolve 等;纯确定性打分;对比「我们自造题 RL」vs「FrontierCS RL」。

---

## 3. 公平对比结果(Codex 逐个复核一致)

### 3.1 SFT / soup vs **start**(都无 RL —— 发现类任务赢)

| 数据集 | 我们(SFT/soup) | start(base) | 公平 Δ |
|---|---|---|---|
| **MLS**(devfix) | methodtraj_sft **0.114** | 0.038 | **+200%** ✅ |
| **Theta**(circle-packing) | innov_soupa70 **1.75** | 0.96 | **+65~83%** ✅ |
| FCS | soup 0.3–5.3 | 7.05 | ❌ 输(SFT 崩竞赛)|
| ALE | soup ≈286(地板) | 356.6 | ❌ 输 |

### 3.2 RL vs **base+RL**(同 GRPO —— 竞赛类任务赢)

**FCS(4 个独立批次,全部赢;Codex 确认 8 个 delta 到小数点后 4 位):**
| 批次 | 我们(RL) | base+RL | 公平 Δ |
|---|---|---|---|
| rlafter | rl_soup_mtv4_a20 **8.08** | rl_start 3.63 | **+4.45** |
| rlafter | rl_methodv4_r2_a30 7.36 | 3.63 | +3.73 |
| clean | rl_methodv4_r2 6.98 | rlstart 5.44 | +1.54 |
| peak/strip sweep | 6.3–6.9 | 4.3–4.5 | +2.0~2.4 |

**ALE(混合,旗舰赢,不普适):** rl_soup_mtv4_a20 575.6 vs base+RL 423.2 = **+152**;clean_rl_methodv4_r2 550.7 vs 389.7 = +161;但 methodtraj/其它变体为负或噪声内(ALE std@5≈134)。

**MLS(几乎持平):** 只有 rl_soup_mtv4_a20(0.1158)险胜 base+RL(0.1066),**+0.009**;其它 RL 模型均低于 base+RL → **MLS 的 RL 提升大头是 RL 本身,不是创新数据。**

---

## 4. Case Study(代码级证据)

### 4.1 ✅ 正向 showcase(公平:SFT/soup vs start,都无 RL)

**MLS · causal-treatment-effect(金标准:双方都跑通,我们 4.75×)**
`method_soup10`(我们)vs `q35_start`(base),条件平均处理效应估计:
- base:自称 X-Learner,实则**朴素 T-learner**(两回归相减,混杂下有偏)→ 0.0549。
- 我们:真正的 **doubly-robust DR-learner** —— Neyman 正交伪结局 `[T−e(X)]·[Y−m(X,T)]/e(X)`、同时 cross-fit outcome 与倾向性、倾向裁剪 (0.05,0.95) → **0.2606(+375%)**,三个混杂数据集全赢。
- 另 5 题:base 崩、我们跑通且用对结构(ANM 噪声不对称、多保真 BO EI×fidelity、离散 CI-test α=0.005 校准)。

**Theta · circle-packing(n=26)**
`innovonly` 等(我们)vs `q35_a00`(base):
- base **冻在 seed 0.96**(thinking 16 轮只吐 1 个有效 diff 还崩了)。
- 我们跑出 **1.58–1.75**:六边形 6 重环、√3 晶格 + π/(2√3) 密度论证、破对称扰动。最干净归因:`innovonly` 的 **gen-1 直接由 seed 变异出的 1.585 = 单次模型编辑 +0.63**。

### 4.2 ⚠️ ALE 启发式(frame caveat:我们-RL vs **裸 base**,非 base+RL)

`rlafter_rl_soup_mtv4_a20` vs `clean_start`(注:严格公平应对 base+RL,聚合公平 Δ=+152):
- **AHC046(滑冰)**:我们把 Slide 建模成**图的边跑 BFS**(证明最少转数),base 用脆弱的逐轴贪心。best 1119 vs 116。
- **AHC016(图时光机)**:我们写**噪声模型 MLE 解码器** `orig·(1−ε)+(MAX−orig)·ε`,base 追求过早的 N 最小化 + 代码崩。best 1191 vs 929。
- **AHC011(滑动树)**:我们**直接对目标函数贪心爬山**(合法),base 写了死代码 + 非法输出(WA −80)。mean 575 vs 92。
- 诚实反例:AHC024(是"保持合法"赢,非创新)、AHC027(top sample 就是题面给的 DFS baseline)。

### 4.3 ❗ 诚实反例(公平:RL vs base+RL,都过 RL)

**MLS · RL-vs-base+RL:RL 会同质化。** `rlafter_rl_soup_mtv4_a20` vs `rlafter_rl_start`:20 题里 **15 题逐位相同**;残余 +8.6% 来自我们**能从首稿 bug 里 `undo` 恢复并提交**(执行鲁棒性),甚至 2 个 case 是**交了更简单的代码**才赢(反创新)。→ **创新的方法论优势在 RL 之后基本被冲淡;正向证据在 RL 之前(§4.1)。**

---

## 5. 局限与未完成

- **r3(最新一代)评测未跑完**:集群 QOS 关联被禁用,新作业(GPU+CPU)全部无法提交;methodv4/full 的 RL 判据仍缺。r3 局部(methodtraj_a10 MLS=0.091 bundler 口径)已印证同方向。
- **ALE case 的 frame**:如 §4.2,cases 是对裸 base,严格公平应对 base+RL;聚合公平差(+152)只对旗舰成立。
- **ALE 方差大**(std@5≈134):单个 ±20~60 的 ALE 差在噪声内。
- **MLS `.partial2` artifact**:早前误引的 base=0.1335 是只跑了 3 题的损坏快照,已弃用;真值 base=0.038(devfix)/0.064(旧)。

---

## 6. 评测口径 & 复现

- **FCS/ALE**:`cc_eval_<TAG>_thinking_32k_both_vllm/summary.json` → `metrics.frontiercs.reward.mean@5` / `metrics.alebench.performance.mean@5`。FCS 用 strip-think 后取最长 C++ 块(公平口径,base≈7.05)。ALE 判题 cpp17。
- **MLS**:`cc_mlsbench_cpu_<TAG>[.devfix]/summary.json` → `mean_score`(20 题;弃用所有 `.partial*` 与 `_archived_404_*`)。
- **Theta/TTT**:`ThetaEvolve/outputs/cc_eval_theta_<tag>_<task>/job_*/summary.json` → `best_objective_value`。
- **Codex 复核**:MLS 数字审计、FCS/ALE 8 个公平 delta 均由 codex-cli 独立重算一致。
- **命名**:`mtv4`=methodtraj_v4、`aNN`=soup α、`rlafter/rl_soup`=在 soup 上做的 RL、`*_start`/`rlstart`=同批次 base+RL 对照。

---

## 附:关键路径
- SFT 配方 `LF-innov/examples/train_full/auto/os-q35_a100_*_r3*.yaml`;soup `FrontierSmith/scripts/cc_model_soup_merge.py`;RL `FrontierSmith/slurm/rlm_amplify_v3_ailab.sh`;synth RL `FrontierSmith/data/frontiersmith_synth/` + `verl/.../reward_score/frontiersmith_synth.py`;统一评测 `FrontierSmith/slurm/cc_eval_all_benchmarks.sh`。
- 姊妹文档:`CAMPAIGN_SUMMARY_zh.md`(方法+分数总表)、`CASE_STUDY_zh.md`(FCS 侧:奖励纪律而非发散)。
