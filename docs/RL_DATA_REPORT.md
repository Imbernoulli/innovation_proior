# 我们的 RL 训练数据是怎么造的、与 FrontierSmith 的对比、以及覆盖领域

> 本文档供写文章引用。所有数字以 `Imbernoulli/innovation_proior` 仓库（main，r3 批次，2026-07）为准，关键 RL 分数来自 `experiments/CAMPAIGN_SUMMARY_zh.md` 与 `FrontierSmith/outputs/` 原始 summary.json。

---

## 1. RL 数据是怎么造出来的？

我们的 RL 训练数据分两层：

1. **上层是训练/评测用的「题目包」**：每个包包含 statement、数据生成器、判分器 / checker、参考解 ladder，以及 `config.yaml`。
2. **下层是流水线**：把这些题目转成 VERL/GRPO 需要的 parquet，喂给 Qwen3.5-9B（或 Qwen3.6-35B）做在线 RL。

### 1.1 最初的 RL 数据来源

直接复用 FrontierSmith 的评测资源：

| 数据源 | 题数 | 类型 | 奖励来源 |
|---|---|---|---|
| Frontier-CS algorithmic | 172 | C++，读 stdin，官方 go-judge | 官方判题器 |
| FrontierSmith 自带 demo | 10 | 开放题 mutation | 自带 gen + chk |
| ALE-Bench | 40 | AtCoder Heuristic 风格 | ALE 判题器 |

这些题目被 `FrontierSmith/scripts/prepare_*_parquet.py` 转成统一 parquet：

```python
{
  "prompt": [{"role":"user","content":"You are a competitive programmer. Solve ... Output ONLY the C++ code wrapped in ```cpp and ```.\n\n{statement}\n\nGenerate solution code:"}],
  "reward_model": {"ground_truth": "<problem_id>"},
  "data_source": "frontiercs"   # 或 frontiercs_research / alebench 等
}
```

训练脚本 `FrontierSmith/scripts/run_verl_grpo_frontiercs_qwen35_9b.sh` 用 **VERL + GRPO**：

- rollout n = 4/8
- 奖励 = 真实 judge 返回的分数
- 在 **soup 模型** 上做 RL（SFT → soup → RL 三段式）

一个关键 infra 修复：早期 `MAX_RESPONSE_LENGTH=20k`，thinking 模型输出常 20–30k，被截断后 clip_ratio 高达 0.94、reward≈0，导致 length collapse。改成 **32k**（max_model_len 45k）后 clip_ratio 降到 0.18，RL 才能正常提点。

### 1.2 后来新增的核心 RL 数据：`frontiersmith_synth`

`innovation_prior/frontiersmith_synth/` 复现并扩展了 FrontierSmith **未公开的 orchestrator + test/checker generator**，批量生成开放题。最终得到 **500 道机器验证题**（仓库 `reports/summary.json`），构成：

| 格式 | 题数 | 形态 |
|---|---|---|
| C++ testlib（FrontierCS 兼容） | 172 | `gen.cpp` + `chk.cc` |
| Python generator（py-gen） | 203 | 构造题 + Python verifier |
| Python evaluator（py-evaluator） | 125 | evolve-a-heuristic，frozen `evaluator.py` |

#### 五种格式与对应框架

| 格式 | 奖励机制 | 覆盖的框架/来源 |
|---|---|---|
| **A** testlib C++ 组合优化 | `gen.cpp` + `chk.cc` → `Ratio:` | FrontierCS、ALE-Bench |
| **B** evolve-a-heuristic | frozen `evaluator.py` → `Ratio:` + `Vector:` | FunSearch、AlphaEvolve、OpenEvolve、ThetaEvolve、TTT-Discover、Frontier-Eng、MLS-Bench |
| **C** constructive artifact + verifier | `verify.py`（精确/几何）→ `Ratio:` | AlphaEvolve、OpenEvolve、ThetaEvolve、FunSearch |
| **D** FLOPs/op-count kernel surrogate | `counter.py` + 等价门 → `Ratio:` | AlphaEvolve |
| **E** symbolic / scientific-law discovery | held-out 误差 + 复杂度 → `Ratio:` | FrontierCS、OpenEvolve、MLS-Bench |

统一评分契约：`Ratio: <float ∈ [0,1]>`，trivial≈0.1，10× better 封顶 1.0。

#### 生成流程

```text
seed_list.jsonl (spec: format/tier/family/theme/scale/variant)
    → 1 agent / 1 spec 读对应 AGENT_BRIEF
    → 写 statement + generator + checker/scorer + config.yaml + 4-tier 参考解
    → 8-gate harness 自验
    → 不 PASS 则 Repair（最多 6 轮）
    → reports/summary.json 汇总
```

#### 8-gate 机器验证

 FrontierSmith 用 LLM agent 互相检查 checker，可能一致但错。我们改成执行层验证：

- G1 compile/import
- G2 数据生成非空
- G3 分数在 [0,1]
- G4 确定性重跑
- G5 非法解 score≈0
- G5b 对抗输入（empty/garbage/nan/inf/injected Ratio）也≈0
- **G5c isolation**：候选解跑在 bubblewrap sandbox，不能读到 judge 源码/隐藏答案
- G6 trivial 校准 ≈0.1
- G7 strong > trivial
- G8 per-test score vector 真正 diverge

#### reward-hacking 防护

用 bubblewrap 把候选解关进 fresh user/pid/net/ipc/uts/mount namespace，整个 synth 树被 `--tmpfs`-hidden，`/proc` 私有。这阻止了：

- `sys._getframe().f_back` 偷 judge 隐藏 oracle
- 读 `/proc/<judge>/mem` 或 `cmdline`
- 读同目录的 `gen.py`/labels 反推答案

### 1.3 为修落点造的专项 C++ 数据

`data_v4/` 里还有一批 FrontierCS 风格单文件 C++ 题（最终进入 SFT 的为 346 条）：

- 读 stdin、单文件、可 `g++` 编译
- 每题带 `verify/{sol.cpp, brute.py, gen.py}`，暴力对拍验证
- spine 是 debug + self-verify（针对旧数据「写完代码不验证」的问题）

这些题不是从现有 benchmark 复制，而是重新设计的，目标是让模型把「创新想法」收敛成「可提交、可判分的单文件 C++」。

---

## 2. 相比 FrontierSmith，我们的创新点和优势

### FrontierSmith 做了什么

FrontierSmith（arXiv 2605.14445）把闭式竞赛题变异成开放式优化题（改目标 / 加输出约束 / 放宽输入结构），用 LLM agent 合成并交叉验证 generator + checker，再按思路发散度筛选。它开源了 **10 个 sample problems + 训练/评测代码**，但**核心的 orchestrator 和 test/checker generator 没有公开**。

### 我们多做了什么 / 哪里做得更好

| 维度 | FrontierSmith | 我们 |
|---|---|---|
| **Orchestrator 公开性** | 10 道 demo + 训练脚本，generator  withheld | `frontiersmith_synth/` 完整复现并扩展了 withheld 部分 |
| **规模** | 10 demo problems | **500 题全部机器验证** |
| **格式** | 基本围绕 testlib C++ | **5 种格式**（A/B/C/D/E） |
| **覆盖框架** | FrontierCS、ALE-Bench | FrontierCS、ALE、FunSearch、AlphaEvolve、OpenEvolve、ThetaEvolve、TTT-Discover、Frontier-Eng、MLS-Bench、Eval-driven Scaling |
| **验证方式** | LLM agent 互相检查 | **8-gate execution-grounded harness** |
| **Reward hacking 防护** | 无 OS 级隔离 | **bubblewrap sandbox（G5c）** |
| **盲评结果** | 10 题 demo | Panel 2：ours 新颖度 7.56/总体 7.75，FrontierSmith 6.70/6.20；FrontierSmith 5 题 broken，ours 0 |
| **失败驱动迭代** | 单向「造题→训练」 | 「造题→训练→读真实输出→返工数据」多轮（v4、wave-2、de-rewrite） |
| **RL 效果** | 官方数据 RL FCS≈6.63 | **自造 synth 题 RL 在官方 FrontierCS 上 FCS≈7.61，且 soup+RL 最高达 9.10** |

### 最关键的 RL 结果

基础参照 `clean_start`：**FCS 7.05，ALE 356.6，MLS 0.038**。

| 模型 | 训练数据 | FCS | ALE | MLS |
|---|---|---:|---:|---:|
| `clean_start` | 无 RL | 7.05 | 356.6 | 0.038 |
| base + RL on 官方 FrontierCS | FrontierCS 172 + ALE 40 + FS 10 | 6.63 | — | — |
| **base + RL on `frontiersmith_synth`** | **500 道自造题** | **≈7.61** | — | — |
| `rl_soup_mtv4_a20 @step40` | methodtraj_v4 soup + RL | **9.10** (+29%) | 436.8 | — |
| **`rlafter_rl_soup_methodtraj_v4_a20`** | methodtraj_v4 soup + RL | **8.08** (+15%) | **575.6** (+61%) | **0.116** (+205%) |

结论：

1. **自造 RL 题可以反哺官方 benchmark**：用 `frontiersmith_synth` 500 题做 RL，迁移到官方 FrontierCS 评测上，效果优于直接用官方 FrontierCS 数据做 RL（7.61 vs 6.63）。
2. **创新先验 + soup + RL 能同时提升竞赛、启发式和科研发现三类任务**：`rlafter_rl_soup_methodtraj_v4_a20` 是首个在 FCS、ALE、MLS 上同时超过 base 的模型。

---

## 3. RL 数据来源涉及哪些领域？

### 3.1 现有 benchmark

- **Frontier-CS algorithmic**：172 题，组合 / 优化 / 构造 / 交互
- **FrontierSmith 10 demos**：Scorched Bridges、Farmwide Teleport、Prime Resonance Retuning 等
- **ALE-Bench**：AtCoder Heuristic 风格优化题

### 3.2 自造 `frontiersmith_synth` 题的覆盖

500 题覆盖 **136 个 task-types、134 个 domains、12 个 source frameworks**：

| 格式 | 题数 | 主题领域 |
|---|---|---|
| A testlib 组合优化 | 182 | 图算法、组合优化、竞赛算法、调度、字符串 |
| B evolve-a-heuristic | 125 | 启发式搜索、演化算法、ML 训练代理、科学发现 |
| C constructive + verifier | 119 | 圆堆积、Hadamard 矩阵、cap set、dispersion、几何构造 |
| D FLOPs/op-count | 40 | matmul tensor rank、addition chains、sorting networks、XOR/reversible circuits、QAOA SWAP count |
| E symbolic / scientific-law | 34 | 天体物理、化学、系统生物学、经济学、ML scaling laws、材料、流行病学、流体动力学、热力学、药理学、量化金融 |

按 tier（难度/定位）分布：

| Tier | 题数 | 定位 |
|---|---|---|
| S | 160 | graph & combinatorial core |
| A | 140 | math-discovery / heuristic evolution |
| G | 84 | breadth-fill（domains/tasks） |
| B | 60 | engineering + science |
| C | 40 | ML-method + exotic |
| N | 16 | bespoke high-novelty |

Source framework 分布（一题可映射多框架）：Frontier-CS 167、AlphaEvolve 104、ALE-Bench 95、Frontier-Eng 92、SimpleTES 71、MLS-Bench 67、OpenEvolve 63、ThetaEvolve 42、TTT-Discover 42、FunSearch 41、FrontierSmith 16。

### 3.3 专项修复数据 `data_v4/`

- **fcs-/***：FrontierCS 风格单文件 C++，覆盖 DP、图论、树、字符串、数论、几何、数据结构、贪心/交换论证。
- **ale-/***：ALE-Bench 风格启发式优化题。

这些不是复制现有题，而是**重新设计的新题**，全部经过 g++ 编译 + 暴力对拍验证。

---

## 4. 写文章可用的一页纸总结

> 我们的 RL 数据 pipeline 分两层：底层是 `frontiersmith_synth`，它复现并扩展了 FrontierSmith withheld 的 orchestrator，用 deterministic 8-gate harness + bubblewrap sandbox 批量产出 **500 道跨格式、跨框架题目**；上层把这些题与 Frontier-CS、ALE-Bench 混合，输入 VERL+GRPO。关键发现是：用 `frontiersmith_synth` 自造的 500 题做 RL，在官方 FrontierCS 评测上效果优于直接用官方 FrontierCS 数据做 RL（7.61 vs 6.63）；进一步把创新先验 SFT、model-soup 与 RL 结合，得到 `rlafter_rl_soup_methodtraj_v4_a20`，在 FCS、ALE、MLS 三个 benchmark 上同时超过 base。相比 FrontierSmith，我们不仅公开了生成器，还从 10 题扩展到 500 题、从 1 种格式扩展到 5 种、覆盖 10+ 个科学发现/进化搜索框架，并根据真实模型失败做了多轮数据补救。数据来源横跨组合优化、数学发现、启发式进化、工程科学优化、ML 方法设计、kernel surrogate、符号回归、天体物理、化学、系统生物学、经济学、材料等领域。

---

## 附录：关键文件路径

- 生成器说明：`innovation_prior/frontiersmith_synth/README.md`、`DESIGN.md`、`REPORT.md`
- 生成流程：`innovation_prior/frontiersmith_synth/generate_problems.workflow.js`
- 验证 harness：`innovation_prior/frontiersmith_synth/harness/validate_problem.py`、`validate_pyproblem.py`
- 汇总：`innovation_prior/frontiersmith_synth/reports/summary.json`
- RL 配置：`FrontierSmith/scripts/run_verl_grpo_frontiercs_qwen35_9b.sh`
- Campaign 完整结果：`innovation_prior/experiments/CAMPAIGN_SUMMARY_zh.md`
- 失败根因与数据补救：`innovation_prior/experiments/MODEL_FAILURE_ROOTCAUSE_zh.md`、`DATA_REMEDIATION_zh.md`
