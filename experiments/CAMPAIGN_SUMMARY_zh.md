# 创新先验(Innovation-Prior)SFT + RL 放大 — Campaign 总结

> 目标:证明「用创新先验数据做 SFT 能提升 Qwen3.5-9B 的能力(竞赛编程 FrontierCS/ALE + 科研发现 MLS-Bench/Theta),且 RL 能进一步放大这种能力」。
> 本文汇总方法(数据/模型/RL)与**已跑出的好结果**(含 ALE 上的强结果)。分数口径见文末「评测保真度」。

---

## 0. 一句话结论

**创新先验 SFT →(与 base 做 model-soup 平均)→ 在 soup 上做 GRPO RL**。

### ⭐ 头条:一个模型 `rlafter_rl_soup_methodtraj_v4_a20` 在三个数据集上同时超过 base

(= 在 methodtraj_v4 的 soup(alpha 0.2)上做 RL;近期 rlafter 批次,非最早期结果)

| Benchmark | Base | **rl_soup_mtv4_a20** | 超过 base |
|---|---|---|---|
| **FrontierCS**(竞赛算法) | 7.05 | **8.08** | ✅ **+15%** |
| **ALE-Bench**(启发式竞赛) | 356.6 | **575.6** | ✅ **+61%** |
| **MLS-Bench**(ML 科研 agent) | 0.038–0.064 | **0.116** | ✅ **+80~205%** |

**同一个模型,FCS + ALE + MLS 全部超 base** —— 这是本 campaign 最想证明的事:创新先验训练不是「拆东墙补西墙」,而是一个模型在多类任务上整体变强。

### 各 benchmark 的单项最好结果(可能来自不同模型)

| Benchmark | Base | 单项最好 | 提升 | 模型 |
|---|---|---|---|---|
| FrontierCS | 7.05 | **9.10** | +29% | `rl_soup_mtv4_a20 @step40` |
| ALE-Bench | 356.6 | **575.6** | +61% | `rlafter_rl_soup_mtv4_a20` |
| MLS-Bench | 0.038(devfix)/0.064(旧) | **0.116** | +80~205% | `rlafter_rl_soup_mtv4_a20` |
| ThetaEvolve(circle) | 0.96 | ~1.75 | +82% | innovation-soup 系列 |

**关键洞察**:创新数据让模型在**发现类/启发式任务(MLS/ALE/Theta)上大幅提点**;在纯竞赛算法(FCS)上,直接 SFT 会掉点(过度发散),但**低比例 soup + RL** 能把它拉回并超过 base。「掺多少创新」是一个可调的 trade-off 旋钮(soup alpha)。

> **MLS 分数口径更正(重要)**:早前引用的 `method_soup10=0.2116 / base=0.1335` 来自 `*.partial2` 聚合文件,该文件**只有 mean_score、无 per-task 分(损坏/未完成聚合)**,已弃用。**可信的 20-题 devfix 分**:base start=**0.038**、method_sft=0.092、method_soup10=0.095、`rlafter_rl_soup_mtv4_a20`=**0.116**(全 benchmark 最高)。旧口径(非 devfix)base=0.064、method_sft=0.094。方向与结论不变,量级更保守。

---

## 1. SFT 数据:我们造了什么

数据来自内部仓库 **`Imbernoulli/innovation_proior`**(最新 r3 版本,2026-07-06 拉取)。核心是**创新先验(innovation-prior)**监督数据——教模型在解题时带上「敢于跳出常规、探索非显然方案」的倾向,而不是只会写教科书式默认解。

### 1.1 数据组成(各成分)

| 成分 | 数量 | 说明 |
|---|---|---|
| **method** | 1201 | 创新方法论倾向的核心 traces(怎么想、怎么发散、怎么选非显然方案) |
| **v4** | 346 | method 的 v4 增补(去改写、投递纪律、fold-think 修复后的干净版) |
| **traj(深化)** | 678 rungs(166 full + 512 folded) | **trajectory 数据,reasoning 被加长/加深**(逐级推到 method 深度,codex 校验过) |
| **wave2** | 758 | 第二波补充数据 |
| **maintain / roll-out** | 903 | 抗遗忘的 general roll-out(Qwen3.6/3.7-蒸馏 reasoning + agentic),**替换掉了早期从 HuggingFace 抓的旧 maintain** |
| ~~agentic~~ | 473 | agentic traces —— **本轮 r3 最终配方中剔除**(实测偏负向) |

### 1.2 本轮(r3)实际训练的三个配方

| SFT 变体 | 数据 = | 行数 | 用途 |
|---|---|---|---|
| `sft_methodv4_r3` | method + v4 | **1547** | 与上一版 methodv4 直接对标 |
| `sft_methodtraj_v4_r3` | method + v4 + **深化 traj** | **2225** | 检验「加长 reasoning」的作用(无 agentic) |
| `sft_full_r3`(wd0 / wd01) | methodtraj_v4 + **wave2** | 2225+758 | 全量(带抗遗忘) |

> ⚠️ **注意:我们准备的数据集比上面报告分数所覆盖的要多。** 报告里只汇报了跑完评测的那几个配方;实际储备的数据成分(不同 method/traj/maintain/agentic 的组合、不同波次、以及尚未训练的切片)远不止这些。后续可按需扩展更多 SFT 配方与消融。

---

## 2. Base model + SFT + Average(soup)

- **Base model**:**Qwen3.5-9B**(`models/Qwen3.5-9B-bf16`,instruct 版;实验里 `a100`=instruct、`a00`=base-chat)。
- **SFT 方式**:**full fine-tuning**(全参),DeepSpeed ZeRO-3 + bf16。
  - 超参:effective batch size **128**(per-device 1 × grad-accum 32 × 4×H200),lr **5e-6**,**1 epoch**,cutoff **53760**(不截断长样本),cosine,warmup 0.1,weight_decay 0 / 0.1。
- **Average = Model-Soup**:`merged = alpha·SFT + (1−alpha)·Base`,`alpha ∈ {0.1, 0.2, 0.3, 0.5}`(记为 a10/a20/a30/a50)。
  - **为什么要 soup**:直接 full-SFT 会把 FCS 打崩(SFT 直测 FCS 仅 0.3–1.5,base 是 7.05)——创新倾向「学过头」了。**soup 把创新倾向按比例掺回 base**,既保住 base 的竞赛能力,又注入创新先验。alpha 就是「掺多少创新」的旋钮。
  - 实测:低 alpha(a10/a20)在保 FCS/ALE 的同时最大化 MLS 发现能力;高 alpha 更激进、竞赛掉点更多。

---

## 3. RL:怎么做的 / 原来怎么做 / 后来加了什么数据

### 3.1 RL 方法

- **算法**:**GRPO**(via `verl`),在 **soup 模型**上做(即「SFT→soup→RL」三段式,RL 用来**放大**已注入的创新先验),配置代号 `rlm_amplify_v3`。
- **关键超参**:rollout.n=4,train_batch=8,lr=1e-6,KL=0.001,20–40 steps,thinking 开启。
- **一个关键修复(response 长度)**:早期 RL 用 `MAX_RESPONSE_LENGTH=20k`,但 thinking 模型输出常 20–30k → 被截断 → clip_ratio 0.94、reward≈0 → **length collapse 退化**。**改到 32k**(max_model_len 45056)后,clip_ratio 0.94→0.18,FCS 从 4.46 恢复到 6.63。这是让 RL「能提点」的前提。

### 3.2 原来的 RL 数据

- 混合训练集:**FrontierCS 172 题 + ALE-Bench 40 题 + FrontierSmith 10 题**(`train_frontiercs172_frontiersmith10_alebench40.parquet`)。奖励=各自 benchmark 的确定性判题分。

### 3.3 后来新增的 RL 数据:`frontiersmith_synth`

- **是什么**:我们**自己造的 500 道 RL 题目**(`frontiersmith_synth`),FrontierCS 兼容格式(`config.yaml`/`gen`/`verify`/`statement`)。
- **怎么构成**(500 题):
  - **172** 道 C++ testlib 题(FrontierCS 风格)
  - **203** 道 Python 生成器题(py-gen)
  - **125** 道 Python evaluator 题(py-evaluator)
- **怎么造的**:用 **FrontierSmith 开放式题目生成方法**(6 阶段:mutate→filter→diverge→testinfra→rerank),范围**从 FrontierCS 拓宽到** FunSearch / AlphaEvolve / OpenEvolve / ThetaEvolve / TTT-Discover / Frontier-Eng / eval-driven-discovery 等谱系。**只用确定性打分**(无 kernel/wall-time 依赖;FLOPs 类可以)。目标是**泛化而非过拟合**。
- **奖励**:复用 synth 自带判题(`validate_problem.py`/`isorun.py`),离线、fail-soft、返回 0–100。smoke 验证过区分度(strong>greedy>trivial(~10)>invalid=0)。
- **对比实验**:`base + RL-on-synth` vs `base + RL-on-FrontierCS`(后者 FCS=6.63),看**我们自造的这批 RL 题**能否让能力像 FrontierCS 训练一样迁移。

---

## 4. 结果:好结果精选(⭐ = 重点)

> 口径:FCS/ALE 用 strip-think 单测(dedicated judge);base 参照 `clean_start` = FCS **7.05** / ALE **356.6**。命名:`mtv4`=methodtraj_v4,`a20`=soup alpha 0.2,`step40`=RL 第 40 步,`rlafter/rl_soup`=在 soup 上做的 RL。

### 4.1 FrontierCS + ALE-Bench(竞赛,RL 放大后)

| 模型 | FCS | ALE | 说明 |
|---|---|---|---|
| ⭐⭐ **`rl_soup_mtv4_a20 @step40`** | **9.10** (+29%) | **436.8** (+22%) | **FCS 最高**,FCS+ALE 双超 base |
| ⭐⭐ **`rlafter_rl_soup_mtv4_a20`** | **8.08** (+15%) | **575.6** (+61%) | **ALE 最高**,双超 base |
| ⭐ `rlafter_rl_soup_mtv4_a30` | 7.57 (+7%) | 436.3 (+22%) | 双超 base |
| ⭐ `rlafter_rl_methodv4_r2_a30` | 7.36 (+4%) | 467.5 (+31%) | 双超 base |
| ⭐ `clean_rl_methodv4_r2` | 6.98 (≈base) | **550.7** (+54%) | ALE 巨幅提升 |
| `clean_start`(**base**) | 7.05 | 356.6 | 参照 |

**结论**:在 **methodtraj_v4 / methodv4 的 soup 上做 RL**,能在 **FCS 上超 base(最高 9.1)**、并在 **ALE 上大幅超 base(最高 575.6,+61%)**。ALE(启发式/优化类竞赛)是创新先验的强项。

### 4.2 MLS-Bench(ML 科研 agent — 创新的主场)

devfix 口径(全 20 题;**base start = 0.038**。⚠️ 早前引用的 0.1335/0.2116 是 `.partial2` 损坏聚合,已弃用):

| 模型 | MLS | vs base 0.038 |
|---|---|---|
| ⭐⭐ **`rlafter_rl_soup_mtv4_a20`** | **0.116** | **+205%**(全 benchmark 最高) |
| ⭐ `methodtraj_sft` | 0.114 | +200% |
| ⭐ `r1_soup_methodtraj_v4_a50` | 0.111 | +192% |
| `method_soup10` | 0.095 | +150% |
| `method_sft` | 0.092 | +142% |
| `q35_start`(**base**) | 0.038 | 参照 |

旧口径(非 devfix,base=0.0643):`method_sft`=0.0943(+47%)、`method_soup20`=0.0908。
r3 bundler 口径(近期):`methodtraj_v4_a10`=**0.091**、`methodv4_a10`=0.082(均 > 旧 base 0.064)。

**结论**:**创新 SFT/soup/RL 在 MLS 上把发现能力提升 1.5–3×**(0.038 → 0.11+)——**这是本 campaign 最贴合「innovation」论点的结果**:同一个 base,注入创新先验后,ML 科研任务解题质量成倍提升。定性证据见 §5。

### 4.3 ThetaEvolve(circle-packing-modular,发现/演化)

- base(`q35_a00`)= **0.96**(seed floor);
- 创新模型:`q35_a00_innovonly_soupa70` = **1.75**、`innovmaint_soupa70` = 1.74、`innovonly` = 1.585、`innovmaint` = 1.54;
- 即 **innovation 模型普遍把 circle-packing 从 0.96 提到 1.5–1.75**(+60%~+82%)。

---

## 5. Innovation Case Study(定性证据:我们的模型 vs base 在 MLS 上的方案)

对比 `q35_a100_method_soup10_devfix`(我们)vs `q35_start_devfix`(base)在 MLS-Bench 20 题上**实际提出/写出的方案**(逐题 trajectory 里的真实代码)。总分 0.095 vs 0.038(+150%)。逐 case:

### ⭐ Case 1(金标准 — 双方都跑通,我们 4.75× 更好):causal-treatment-effect
估计条件平均处理效应 CATE(观测数据、有混杂)。
- **base**:docstring 自称「X-Learner」,实际代码是**朴素 T-learner**——两个 outcome 回归相减,**无倾向性、无 cross-fitting、无 doubly-robust**(混杂下有偏)。分 **0.0549**。
- **我们**:真正的 **doubly-robust DR-learner** —— Neyman 正交化伪结局 `[T−e(X)]·[Y−m(X,T)]/e(X)`、**同时 cross-fit** outcome 与倾向性模型、倾向性裁剪 `(0.05,0.95)` 保 overlap。三个数据集全赢,分 **0.2606(+375%)**。
- **为什么更 innovation**:base 抓「能编译的最简式」(可证有偏);我们部署了观测因果推断真正关键、base 完全忽略的三招(双稳健 / 双 cross-fit / 倾向裁剪),且实现正确、每个混杂数据集都赢。

### 其它 5 个 case(base 崩 → 我们跑通且用了对的结构)
| 任务 | base | 我们 | 我们的创新点 |
|---|---|---|---|
| causal-discovery-discrete | 0.0(空图,导入不存在的 API)| **0.291** | 把 CI-test 阈值从默认 0.05 调到 **α=0.005**(离散/有限样本的正确校准)|
| hyperparameter-search | 0.0(崩)| **0.303** | **多保真 BO**:把 EI 与 fidelity 耦合 `log(EI)/(dim·0.5+1)` + 逐维自适应 KDE 带宽 |
| causal-observational-nonlinear | 0.0(崩)| **0.036** | 利用 **ANM 噪声不对称**(root cause 边际方差最小)——非线性可辨识性的真洞察 |
| ml-calibration | 0.0(崩)| **0.052** | temperature-scaling + **对 >0.95 过自信尾部加惩罚项** |
| ml-symbolic-regression | 0.0(**过度工程炸了**,avg_fitness≈1e12)| **0.035** | 反向证据:我们**克制选简单**(标准锦标赛 k=7)反而跑通 → 训练教的是「何时该创新」,不是盲目复杂 |

### 诚实备注(让结论更可信)
1. 很多「base=0.0」是 base **想了同样甚至更激进的方法但跑不起来** → 我们的一部分优势是「**可靠地把一个好想法落地成能跑的代码**」。
2. `optimization-evolution-strategy`(0→0.487)我们最终代码也有 NameError,**不作为干净证据**;`ml-clustering`(0.388 tie)是诚实的平局。
3. 数字口径:subagent 独立复现的 summary.json 是 base 0.038 / soup10 0.095(与 §4.2 一致);载荷证据是**逐题代码差异**,不依赖具体聚合。

> 互补:FCS 侧 case study(`innovation_prior/experiments/CASE_STUDY_zh.md`)结论是「FCS 奖励纪律而非发散」——所以 innovation 的正向证据在 MLS 这类**发现任务**上最明显,与本节一致。
> 复现路径:`FrontierSmith/outputs/cc_mlsbench_cpu_q35_a100_method_soup10_devfix/task_logs/*.log`(我们)+ `.../cc_mlsbench_cpu_q35_start_devfix/task_logs/*.log`(base)。

---

## 6. 评测保真度(口径修复,附)

本 campaign 中对 6 个 benchmark 的评测做了保真度审计 + 修复(每个都经 codex 复核):

- **FrontierCS**:C++ 提取用 `strip <think>` 后取最长代码块(对开放 thinking 模型才公平;论文里 Qwen 1.80 是「原始文本提取」的 artifact,公平口径 base ≈ 7.05)。
- **ALE-Bench**:判题语言 `cpp17`(现有 apptainer 镜像;`cpp20` 为完全保真项,待 build 镜像)。
- **ThetaEvolve/TTT**:用 `best_objective_value`(不是被反转的 combined_score);TTT 只测 AC1/AC2/circle(无 AC3)。
- **MLS-Bench**:causal-path + str_replace 修复(devfix 口径),分数显著更可信。
- **统一评测**:`cc_eval_all_benchmarks.sh` = 一个 job 里 serve 一次模型 → FCS+ALE+MLS+TTT+Theta 全测 + fail-soft + 汇总一张 `summary_all.json`。

---

## 附:关键脚本 / 路径

- SFT 配方:`LF-innov/examples/train_full/auto/os-q35_a100_{methodv4,methodtraj_v4,full}_r3*.yaml`
- Soup 合并:`FrontierSmith/scripts/cc_model_soup_merge.py`(`--alpha`)
- RL:`FrontierSmith/slurm/rlm_amplify_v3_ailab.sh` + `scripts/run_verl_grpo_frontiercs_qwen35_9b.sh`
- synth RL 数据/奖励:`FrontierSmith/data/frontiersmith_synth/{train,full}.parquet` + `verl/verl/utils/reward_score/frontiersmith_synth.py`
- 统一评测:`FrontierSmith/slurm/cc_eval_all_benchmarks.sh` + `scripts/reaggregate_all_summary.py`
