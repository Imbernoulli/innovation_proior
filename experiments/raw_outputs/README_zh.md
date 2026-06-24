# Raw outputs：Instruct 系模型的全部原始输出 + 打分

> 给 **SFT 数据作者** 看的「逐字原始材料」。这里 dump 了 **以 Qwen 指令模型为起点（Instruct-as-Start，即 a100 系）** 的整套模型在各评测上**真实产生的原始输出**和**对应打分**，方便你直接读模型到底吐了什么、为什么会这样打分，而不必去翻评测机器上的目录。
>
> 这是「**全量 dump**」，不是精选；精选版（start/sft/average 三方对照、含完整 prompt 的少量代表例）在隔壁 [`../data_feedback/examples/`](../data_feedback/examples/)。

---

## 0. 范围（只含 Instruct 系 = a100）

只包含**以指令模型为起点**的一族（论文里的 *Instruct-as-Start*），**不含** base 系（a00）。对 q35（Qwen3.5-9B-Instruct/bf16）和 q3（Qwen3-8B-Instruct）两条线，每条最多覆盖：

| model_tag（本目录中的人类可读名） | 实际 checkpoint | 含义 |
| --- | --- | --- |
| `q35_inst_start` / `q3_inst_start` | `models/Qwen3.5-9B-bf16` / `models/Qwen3-8B` | **起点**：指令模型本身（对照） |
| `q35_a100_method_sft` / `q3_a100_method_sft` | `models_sft/sft_q{35,3}_a100_method` | **method** 全参 SFT |
| `q35_a100_method_soup10` / `q3_a100_method_soup10` | `models_sft/soup_q{35,3}_a100_method_soupa10` | method SFT 与起点的 **model-soup**（α=0.10） |
| `q35_a100_methodtraj_sft` / `q3_a100_methodtraj_sft` | `models_sft/sft_q{35,3}_a100_methodtraj` | **methodtraj** 全参 SFT |
| `q35_a100_methodtraj_soup10` / `q3_a100_methodtraj_soup10` | `models_sft/soup_q{35,3}_a100_methodtraj_soupa10` | methodtraj 的 model-soup（α=0.10） |

> `innovonly` 一族已**去范围**，不在此 dump 内。
>
> **每个 model_tag 都已逐条核对其来源 eval 目录的 `config.model` / SLURM 日志里实际 serve 的 checkpoint 路径**，确认是 a100（指令系），不是 a00（base 系）。这一步很重要——评测机器上有一批**误指**的软链接：`cc_eval_q35_a100_method_*`（FrontierCS 算法）和 `cc_eval_theta_q35_a100_method_*`（ThetaEvolve）其实软链到了 `q35m_a100_innovonly_*`（innovonly 跑），**已剔除、未纳入**（见下方各 benchmark 的覆盖说明与缺口）。

---

## 1. 目录结构

```
experiments/raw_outputs/
├── README_zh.md                     ← 本文件
├── _manifest.json                   ← 机器可读清单：每个 (benchmark, model_tag) → 源目录 + 文件数
├── frontiercs_research_gpu/         ← FrontierCS 研究赛道，GPU 子集（21 题，Triton kernel 等）
│   ├── prompts.jsonl.gz             ← 21 题的完整 prompt（problem_idx → 逐字题面），各模型共用
│   └── <model_tag>/
│       ├── samples.jsonl.gz         ← 原始输出（5 采样/题）：text=完整推理+答案, metrics=打分
│       └── summary.json            ← 逐题 + 汇总分数（明文，可直接读）
├── frontiercs_research_cpu/         ← FrontierCS 研究赛道，CPU 子集（43 题）
│   ├── prompts.jsonl.gz
│   └── <model_tag>/{samples.jsonl.gz, summary.json}
├── frontiercs_algorithm/            ← FrontierCS 算法赛道（40 题）+ ALE-Bench（8 题），同一次 "both" 跑
│   ├── prompts.jsonl.gz             ← 48 条：前 40 = frontiercs 算法题, 后 8 = alebench 题（data_source 区分）
│   └── <model_tag>/{samples.jsonl.gz, summary.json}
├── mlsbench/                        ← MLS-Bench（20 个研究任务）
│   └── <model_tag>/
│       ├── summary.json            ← 逐任务 + mean_score（明文）
│       └── task_logs/*.log.gz       ← agent 完整转录（含初始 prompt + 运行结果）
└── thetaevolve_circle_packing/      ← ThetaEvolve（circle_packing_modular 任务）
    └── <model_tag>/
        ├── best_program.py          ← 演化出的最优程序（明文）
        ├── best_program_info.json   ← 该程序的分数
        ├── summary.json            ← best_combined_score 等
        └── config_used.yaml
```

### 怎么读

```bash
# FrontierCS：解压看某模型在某题上的原始输出
zcat frontiercs_research_gpu/q35_inst_start/samples.jsonl.gz | head -1 | python -m json.tool
#   text            = 模型完整输出（含 <think> 推理 + 最终答案/代码），未截断
#   metrics.score   = 该采样得分（FrontierCS 0–100 连续部分分 / ALE 为绝对/相对分）
#   problem_idx     = 题号，join 到同目录 prompts.jsonl.gz 取完整题面
#   ground_truth    = 题目名；error 非空 = 该采样基础设施失败（超时/编译环境），非模型内容

# 取某题的完整 prompt
zcat frontiercs_research_gpu/prompts.jsonl.gz | python -c "import json,sys; [print(json.loads(l)['prompt']) for l in sys.stdin if json.loads(l)['problem_idx']==19]"

# MLS：解压看 agent 转录
zcat mlsbench/q35_a100_method_sft/task_logs/causal-discovery-discrete.log.gz | less
```

> **关于 prompt**：`samples.jsonl` 本身只存模型**输出**（`text`），不含题面。题面我从评测用的 parquet 抽出来放到每个 benchmark 目录的 `prompts.jsonl.gz`，按 `problem_idx` join。

---

## 2. 各 benchmark 覆盖 + 已核对的分数表

> **以下所有数字均现读自本目录内的 `summary.json`**（不是凭记忆）。FrontierCS 报 `score` 的 **mean@5 / best@5**（每题 5 采样）；MLS 报 `mean_score`；ThetaEvolve 报 `best_combined_score`。

### 2.1 FrontierCS — Research GPU 子集（21 题，`frontiercs_research_gpu/*/summary.json`）

| model_tag | mean@5 | best@5 | n |
| --- | ---: | ---: | ---: |
| **q35_inst_start**（起点） | **4.366** | 6.858 | 21 |
| q35_a100_method_sft | 0.456 | 1.562 | 21 |
| q35_a100_method_soup10 | 1.421 | 3.804 | 21 |
| **q3_inst_start**（起点） | 0.001 | 0.004 | 21 |
| q3_a100_method_sft | 0.000 | 0.000 | 21 |
| q3_a100_method_soup10 | 0.473 | 1.600 | 21 |

→ q35 上 **起点 ≫ SFT**，soup 部分回收但仍不及起点；q3 起点本就近 0（这批 Triton kernel 题对 8B 太难），soup 反而略升。

### 2.2 FrontierCS — Research CPU 子集（43 题，`frontiercs_research_cpu/*/summary.json`）

| model_tag | mean@5 | best@5 | n |
| --- | ---: | ---: | ---: |
| **q35_inst_start**（起点） | **13.695** | 29.266 | 43 |
| q35_a100_method_sft | 10.750 | 21.134 | 43 |
| q3_a100_methodtraj_sft | 12.459 | 27.242 | 43 |

→ 同样 **起点 > method-SFT**；methodtraj（q3）介于其间。
（注：q35 的 method-soup / methodtraj CPU 跑在评测机上是 **404 全失败**的废跑，已剔除——见 §4。）

### 2.3 FrontierCS — Algorithm 赛道 + ALE-Bench（40+8 题，`frontiercs_algorithm/*/summary.json`）

算法赛道与 ALE-Bench 是**同一次 `source=both` 的跑**：`samples.jsonl.gz` 里 `data_source=frontiercs` 是 40 道算法题，`data_source=alebench` 是 8 道 ALE 题；`summary.json` 的 `metrics` 同时含 `frontiercs` 与 `alebench` 两组分数。**ALE-Bench 的原始输出与打分即包含在此**（无独立目录）。

| model_tag | FCS mean@5 | FCS best@5 | ALE 相对分 mean@5 |
| --- | ---: | ---: | ---: |
| q35_a100_method_soup10 | 2.924 | 7.463 | 4.71e+09 |
| q35_a100_methodtraj_sft | 0.000 | 0.001 | 0 |
| q35_a100_methodtraj_soup10 | 2.836 | 6.060 | 2.12e+09 |
| q3_a100_method_sft | 1.365 | 2.276 | 7.22e+08 |
| q3_a100_method_soup10 | 3.322 | 6.090 | 1.76e+09 |
| q3_a100_methodtraj_sft | 0.100 | 0.286 | 4.61e+08 |
| q3_a100_methodtraj_soup10 | 2.260 | 4.842 | 5.38e+09 |

→ SFT（尤其 methodtraj）几乎打到 0；soup 才把算法/ALE 能力捞回来。**注意缺口**：q35 的 method-SFT 与两条线的 **inst_start 都没有算法跑**（前者只有误指 innovonly 的废软链，后者起点根本没在算法赛道上跑过），故此处无起点对照行。

### 2.4 MLS-Bench（20 研究任务，`mlsbench/*/summary.json`）

| model_tag | mean_score | n_scored/n_tasks |
| --- | ---: | ---: |
| q35_inst_start（起点） | 0.0643 | 20/20 |
| **q35_a100_method_sft** | **0.0794** | 20/20 |
| q35_a100_method_soup10 | 0.0538 | 20/20 |
| q3_inst_start（起点，**不完整**） | 0.0000 | **1/20** |

→ **关键反转**：在 MLS 研究任务上 **method-SFT（0.0794）> 起点（0.0643）> method-soup（0.0538）**——与 FrontierCS 完全相反。逐任务看（`summary.json` 的 `tasks[]`），SFT 在 `ml-clustering-algorithm`(0→0.388)、`optimization-nas`(0→0.032)、`ml-dimensionality-reduction`(0→0.013) 上从 0 涨起来；起点则在 `ml-symbolic-regression`、`optimization-evolution-strategy` 上更强。这条「FCS 跌 / MLS 涨」的双重分离是整套实验的核心现象。
（q3 起点只跑了 1 个任务就中断，**不可用作对照**；q35/q3 的 method/methodtraj MLS 跑尚未完成，故 MLS 覆盖最窄。）

### 2.5 ThetaEvolve — circle_packing_modular（`thetaevolve_circle_packing/*/summary.json`，best_combined_score）

| model_tag | best_combined_score |
| --- | ---: |
| q35_inst_start（起点） | 1.3505 |
| q35_a100_method_soup10 | 2.0943 |
| q35_a100_methodtraj_sft | 0.9598 |
| q35_a100_methodtraj_soup10 | 1.6468 |
| q3_inst_start（起点） | 1.8378 |
| q3_a100_method_sft | 1.2738 |
| q3_a100_method_soup10 | 1.8830 |
| q3_a100_methodtraj_sft | 0.9623 |
| q3_a100_methodtraj_soup10 | 1.9027 |

→ methodtraj-SFT 最差（<起点）；method-soup 反而最高。每个目录的 `best_program.py` 是演化出的最优程序，可直接读。
（注：q35 的 `method`-SF/ 起点之外的 `q35_a100_method`(无 soup) 同样是误指 innovonly 的废软链，故 q35 method 行用的是 `method_soup10`；q3 的 `method`-SFT 是真跑，已纳入。）

---

## 3. 三方对照 call-out（同一题上 Start vs SFT vs Soup 的「腔调差」）

下面是从 `frontiercs_research_gpu` 里挑的、最能看出 **SFT 学到的是「创新叙事姿态」而非可提交解** 的例子。**逐字摘自 `samples.jsonl.gz`**（取每模型该题最高分采样的开头）。

### `vector_addition/2_24`（problem_idx=19）— start 高分、SFT 崩、soup 回收

逐题 mean@5：**start 39.19 / sft 0.00 / soup 9.97**。

- **q35_inst_start**（该采样 score=50.00）—— 直奔工程要点：
  > "The problem asks me to implement a high-performance Triton kernel for vector addition on large vectors (2^24 = 16,777,216 elements). Let me break down the requirements: 1. Create a Solution class … 2. The code must define an `add` function using Triton kernels … Key considerations: Use appropriate block sizes …"

- **q35_a100_method_sft**（该采样 score=0.00）—— 套上「创新分析腔」：编号小标题、加粗、把带宽写成 $N=2^{24}$ 的数学命题，铺陈一大段「Key Challenges & Considerations」，**但最终没产出能跑的提交**：
  > "1. **Analyze the Problem:** * **Task:** … * **Vector Size:** $N = 2^{24} = 16{,}777{,}216$ … * **Goal:** Maximize bandwidth … 2. **Key Challenges & Considerations** …"

- **q35_a100_method_soup10**（该采样 score=49.83）—— soup 把腔调拉回到「先讲清需求再写对代码」，分数随之回收：
  > "The user wants me to create a high-performance Triton kernel for vector addition. Let me break down what's needed: 1. A `Solution` class … 2. An `add` function that uses Triton JIT … Key considerations for optimization: Use appropriate block sizes …"

> 读法：三者**题面完全相同**（见 `frontiercs_research_gpu/prompts.jsonl.gz` 的 `problem_idx=19`），差别全在输出风格——SFT 把「创新论文式的分析铺陈」迁移了进来，**在「奖励简单正确代码」的题上反而 backfire**；soup 中和了这种腔调。`vector_addition/2_28`(start 33.5/sft 0/soup 19.9)、`vector_addition/2_20`(start 19.0/sft 9.6/soup 0) 是同向的另外两例，可在同一对 `samples.jsonl.gz` 里按 `problem_idx` 20 / 18 复核。

---

## 4. 完整性与 caveat（请勿误读为「模型全 0」）

1. **MLS 原始输出是「按采集口径完整」而非 100% 逐字**：`task_logs/*.log.gz` 含初始 prompt + agent 转录 + 运行结果，但 harness 对部分**很长的 prompt/代码回显做了 verbose 折叠**；评测机上的 `saves/` seed 目录基本为空。本 dump 收录的是**实际存在**的转录，未做美化。请据此理解 MLS 的「完整度」。
2. **`error` 非空的采样 = 基础设施失败**（题级超时 / 编译环境 / 评测器 import 失败），**不是模型内容**，打分计 0。FrontierCS research 各跑都有少量这种采样（正常）。
3. **已剔除的废跑**（避免「假 0 分」污染）：评测机上以下 research-CPU 跑是 **404「model does not exist」全失败**（vLLM serve 名配错），**不含任何真实输出，已不纳入**：`cc_eval_q35_soup_method_soupa10_researchcpu`、`cc_eval_q35_sft_methodtraj_researchcpu`，以及若干带 `_researchcpu` 的 in-progress 重跑（无 `summary.json`）。
4. **已剔除的误指软链接**：`cc_eval_q35_a100_method_*`（FCS 算法）、`cc_eval_theta_q35_a100_method_*`（ThetaEvolve）软链到 `q35m_a100_innovonly_*`（innovonly），**不是 method**，未纳入。这造成了 §2.3 / §2.5 里 q35 method 行的缺口。
5. **q3_inst_start 的 MLS 只跑了 1/20 任务**（中断），其 `mean_score=0` **不可用作起点对照**，仅保留以示存在。
6. **一条原始行带未转义控制字符**：`frontiercs_algorithm/q3_a100_methodtraj_soup10/samples.jsonl.gz` 中有 1 行（共 241 行）因模型输出了裸控制字符而**严格 JSON 解析会失败**——这是源文件本身的 quirk，本 dump **逐字保留**未改动，其余 240 行正常。

---

## 5. 体积

gzip 后整个 `raw_outputs/` 约 **22 MB**（最大单文件 ~3.9 MB，远低于 GitHub 100 MB 硬限）；未压缩约 91 MB。按本仓库既有惯例（`sft/*.jsonl.gz`），大的 `samples.jsonl` / MLS `*.log` 均已 **gzip**（`zcat` / `gunzip -k` 可读），`summary.json` 与小 json 保持明文便于直接阅读。
