# SFT 训练数据 × 五个评测的**数据泄露审计**（语义级，超越 n-gram）

审计时间：2026-07-08。产物目录：[`../decontam/`](../decontam/)。可复现：`python3 decontam/audit_leakage.py`。

> 配套：`DATA_CONTAMINATION_AUDIT_zh.md`（上一轮 **n-gram** 审计）。本文**补上 n-gram 抓不到的那半**——
> 语义复述 / 经典问题重构 / 同任务改写。上一轮结论「FCS/ALE 没有直接题面污染」在**逐字**意义上成立，
> 但**在语义意义上严重不成立**：训练集里有整整一族样本是**照着评测任务反向构造**的（circle packing 是原型，
> 但远不止它），n-gram 天然看不见。

---

## 0. 一句话

**训练数据对五个评测里的四个存在结构性泄露**，且大多是 n-gram 检测不到的语义级：

- **MLS-Bench**：139 个 trajectory slug **就是** MLS 任务 slug（同研究问题 + 同接口 + 同 baseline ladder）；
  其中 **51 条**还额外注入了**非原生、更强的 baseline**（jSO / SOAP / UniPC / diff-transformer / gated-DeltaNet …）
  作为 finale——这正是你说的「Type-1」泄露，且可机械枚举。
- **ThetaEvolve / TTT-Discover**：训练集里有一整族 **AlphaEvolve 数学发现题**的重构——circle packing n=26、
  Erdős min-overlap C5、自相关不等式 AC1/2/3、Hadamard、Heilbronn、cap set、kissing number、fast matmul——
  很多还带 `-autoevolver-record` / `-frontier-largeN` / `-record` 后缀，**直接把评测题的已发表纪录解写进训练**。
- **ALE-Bench**：`ale-atcoder-ahc039` + `ahc039-*` = 真实 AtCoder AHC039，连 ALE-Agent 分数都抄进了 context。
- **FrontierCS**：**research track 语义泄露严重**——不止 `trimul`/`denoising`，还有 FlashAttention（5 个 method）、
  MLA（2）、SIMP 拓扑优化、IMPT 质子治疗、CS:APP malloc、以及 MAGIC×2/kNN-smoothing（**就是** denoising 基准自带的参考方法）
  等约 20 个 method 直接重构了 research 任务的意图解；algorithmic(172 数字题)track 则**基本干净**（0 命中）。
  另有 `v4`(346) / `wave2`(码/数学) 是 FCS/ALE **同分布合成**（造题非抄题，风险低但需逐条核）。

**产物**：给每条 SFT 样本打了泄露标签（`decontam/leakage_tags_*.jsonl`，与三份 jsonl 行对齐，带 `decontam_action`），
并把去污染 gate **接进了 `sft/build_sft.py`**（源头 turn 级处理），生成**去污染副本**
`decontam/clean_rebuilt/innovation_sft.jsonl(.gz)`，**原始 `sft/*` 零改动**。

> **最终处置策略（按 2026-07-08 你的口径，非一刀切删）**——规则见 `decontam/decontam_rules.json`：
> - **整条删（65 行 / 28 method + 8 trajectory）**：discovery 里**启发式/进化搜索跑出来的纪录构造** + AHC039
>   （它们是针对评测实例搜出来的具体解）。
> - **只删 finale 那一 turn（51 条 MLS 轨迹）**：把注入的非原生更强 baseline 那一轮去掉，**保留** baseline ladder。
> - **保留**：MLS 同任务的 baseline ladder、正经 paper 方法（含 39 个 finale method 的独立版）、FCS-research 方法、
>   v4/wave2 合成（n-gram 复核只撞到 boilerplate，无真实题面重合）。
> - 净效果：`innovation_sft` 2698 → **2582**（只删 116 行）；`wave2`(758)/`v4`(346) 原样保留。

---

## 1. 范围与方法

**训练数据（三份已发布 jsonl，行对齐各自的 `_*_tags.jsonl`）**：

| 文件 | 行数 | 构成（kind） |
|---|---:|---|
| `sft/innovation_sft.jsonl(.gz)` | 2698 | method 1201 · v4 346 · traj_full 166 · traj_folded 512 · agentic_full 127 · agentic_folded 346 |
| `sft/innovation_wave2_sft.jsonl(.gz)` | 758 | reasoning 397 · ifollow 141 · code 119 · math 92 · fcs_codex 9 |
| `sft/innovation_v4_sft.jsonl(.gz)` | 346 | 合成 FCS/ALE 风格单文件 C++ |

**评测任务全集（本机核实）**：

- **MLS-Bench**：`/srv/home/bohanlyu/MLS-Bench/tasks/*`（161 个带 config 的任务；已评测 20 个）。
- **ThetaEvolve**：`openevolve/examples/alphaevolve_math_problems/*` 有**可运行 evaluator** 的题：
  circle_packing、first/second/third_autocorr_ineq、hadamard、**heilbronn_triangle/convex**、**kissing_number**、
  **matmul**、erdos_min_overlap、uncertainty_ineq、minimizing_max_min_dist、sums_diffs_finite_sets、hexagon_packing。
- **TTT-Discover**：AC1/AC2、Erdős min-overlap C5、circle packing、GPU TriMul kernel、AtCoder AHC、单细胞 RNA 去噪。
- **ALE-Bench**：AtCoder Heuristic（AHC）题。项目内已知 8 题（ahc008/011/015/016/024/025/026/027，打包进 FCS shard）；
  完整 ALE-Bench 题库本机无 manifest，**AHC039 是否在其中未能本地确认**。
- **FrontierCS**：algorithmic ~172（数字 id 竞赛题，C++ stdin）+ research 68（`frontier_eval/tasks/*`：trimul、
  denoising、flash_attention、mla、predict_modality、perturbation_prediction、topology_optimization …）。

**方法**（`decontam/audit_leakage.py`，确定性）：把每条 SFT 样本的**源 slug**（tag `id`）与
(a) 精心整理的评测任务族注册表 `decontam/eval_registry.json`、(b) MLS 任务 slug 全集、
(c) MLS trajectory 的 `meta.json` finale rung 是否为原生 baseline，逐一比对，产出行级标签 + 去污染副本。

---

## 2. 发现（按评测）

### 2.1 MLS-Bench —— 同任务 + Type-1 非原生 baseline

- **同任务**：**139** 个 trajectory slug ∈ MLS 任务 slug；连同其 agentic 镜像，共 **996 行** SFT 样本是 MLS 同任务。
  内容核实（行对齐）：例如 `sft/innovation_sft.jsonl:1976` = `optimization-nas` 训练样本，human turn 原文
  「NAS-Bench-201 contains 15,625 architectures … 30 validation queries」——就是 MLS `optimization-nas` 本身。
  已评测 20 题里命中 **19**（仅 `ml-selective-deferral` 未进 SFT）。→ **MLS 上的任何提升都不是干净 held-out**。
- **Type-1（你特别关心的：轨迹自动加了非原生、比原生更强的 baseline）**：**51/139** 条 MLS trajectory 带一个
  `meta.json` 里 `finale:true` 的 endpoint rung，其方法**不在**该 MLS 任务的原生 baseline 里，且 meta 自述「明显更强」。
  完整清单见 `decontam/mls_type1_nonnative_finale.json`，例如：

  | MLS 任务 | 注入的非原生 finale | 原生 baseline |
  |---|---|---|
  | optimization-evolution-strategy | **jSO**（CEC-2017 冠军 DE） | de, ga_sbx, lshade |
  | cv-diffusion-efficiency | **UniPC**（NeurIPS-2023） | ddim, dpm2s, dpm3m_sde… |
  | llm-pretrain-attention | **diff-transformer** | qk_norm, rope… |
  | llm-pretrain-optimizer | **SOAP** | adamw_nesterov, lion, muon |
  | llm-pretrain-linear-attention | **gated-DeltaNet** | deltanet, gla, retnet |
  | rl-offpolicy-continuous | **CrossQ** | ddpg, sac, td3 |
  | …（共 51 条） | | |

  这是**比「同任务」更硬的泄露**：模型直接学到了该 MLS 任务「当前最强解」，而 MLS 推理时**并不提供**这个方法。
  可机械识别：*finale rung 的 slug 不在任务 config.json baselines 里* 即命中。
- **双重出现**：51 个 finale 方法里 **39 个**同时以独立 `method` 样本存在（`methods/soap`、`methods/diff-transformer`、
  `methods/unipc`、`methods/jso` …）。独立 method 以「年份条件下的通用发现」呈现、风险较低，但仍等于把该 MLS 任务的
  SOTA 又喂了一遍——本审计把它们标为 `mls_finale_method_standalone`（REVIEW，**保留但标注**）。

### 2.2 ThetaEvolve / TTT-Discover —— AlphaEvolve 数学发现题族（经典问题重构，n-gram 全盲）

训练集里存在一整族**照着评测题反推**的样本，很多明确以「够到已发表纪录」为终点（`-autoevolver-record` /
`-frontier-largeN` / `-record` / `-goldberg-optimal`）。逐条内容核实（如 `innovation_sft.jsonl:159` =
circle-autoevolver-record，human 原文「26 circles … maximize sum of radii … AlphaEvolve 2.63586… ShinkaEvolve
2.635983… AutoEvolver 2.635988…」）。族清单（唯一源 slug，跨 full/folded 展开为多行）：

| 族 | 评测任务 | CRITICAL 源 slug | HIGH 源 slug |
|---|---|---|---|
| circle_packing_n26 | Theta circle_packing / TTT circle packing | circle-autoevolver-record, geom-circle-packing-26 | circle-packing-in-square, circle-grid-baseline, circle-single/multistart/structured-perturb-slsqp |
| erdos_min_overlap_c5 | TTT Erdős C5 / Theta | erdos-autoevolver-record, erdos-frontier-largeN, math-erdos-min-overlap | erdos-minimum-overlap, erdos-uniform-baseline, erdos-coarse-slsqp, erdos-finer-basinhop |
| autocorr_inequalities | Theta AC1/2/3 / TTT AC1/2 | c1-autoevolver-record, c1-frontier-largeN, c2-frontier-largeN, c2-record, math-autocorrelation-c1, math-autocorrelation-inequality | autocorrelation-inequalities |
| heilbronn_triangle | Theta heilbronn（可运行） | geom-heilbronn-triangle, heilbronn-goldberg-optimal | heilbronn-triangle/grid-baseline/random-multistart/sa-positions/softmin-polish |
| cap_set | FunSearch/AlphaEvolve | capset-funsearch-evolved, combi-cap-set | cap-set, capset-greedy-lex/random-multistart/structured-priority |
| hadamard_maxdet | Theta hadamard_matrix | hadamard-orrick-record | synth-hadamard-maxdet |
| kissing_number | Theta kissing_number（可运行） | — | kissing-number |
| fast_matrix_mult | Theta matmul（可运行） | — | fast-matrix-multiplication |
| low_autocorr_binary_seq | LABS（AlphaEvolve 邻近，MEDIUM） | — | (low-autocorrelation-sequences, 保留标注) |

> **注意「同名异题」不算泄露**：`erdos-ko-rado`、`erdos-stone-simonovits`、`probabilistic-method-erdos` 虽带 Erdős，
> 但**不是** min-overlap 那道题，未被标记（见 §2.2 false-friend）。

### 2.3 ALE-Bench —— 真实 AHC039

`ale-atcoder-ahc039`（trajectory）+ `ahc039-bbox-rect/grid-greedy/grid-sa/shinka-targeted-sa`（method）=
真实 AtCoder AHC039「Purse Seine Fishing」，`00-initial-context.md` 连 ALE-Agent(2880)/2nd(3140) 的分都写进去了，
终点是 ShinkaEvolve 精炼过的退火解。**AHC039 是否在 ALE-Bench 官方题库本机无法确认**，但既是真实 AHC 题、又带竞赛
纪录解，按 **CRITICAL** 处理。（项目内已知的 8 道 ALE shard 题号均非 039，故那 8 题不泄露。）

### 2.4 FrontierCS

- **Research track 直接重构（约 20 个 method，n-gram 全盲）**：`methods/trimul` = FCS research `trimul`；
  `ttt-discover-denoise`/`raw-baseline-denoise`/`bio-scrna-denoise` **+ `magic-imputation`/`magic-diffusion`/`knn-smoothing`**
  = `denoising`（后三个**就是**该基准自带的参考法）；FlashAttention `flashattention`/`flash-attention-2`/`flash-v1/2/3` = `flash_attention`；
  `mla`/`mla-attention` = `mla`；`simp` = `topology_optimization`；`proton-therapy-impt` = `proton_therapy_planning`；
  `malloc-allocator` = `malloclab`。→ CRITICAL/HIGH（已删）。另有约 13 个技术相关但有 gap 的（`mma-svanberg`、`aes-rijndael`、
  `sha256-merkle-damgard`、`keccak`、`muon-tomography`、`quadruped-gait`、`gappy-pod-sensors` …）→ MEDIUM（保留 + 标注，逐条核）。
- **同分布合成（造题非抄题，风险低、保留但标注）**：`v4`(346, `ale-*`/`cp-*`，如 ale-01=「Drone Survey Sweep」TSP)
  与 `wave2` 的 `code`(119, 公开 `hardtests` 竞赛库) / `math`(92) / `fcs_codex`(9, Codex gpt-5.5 解合成 FCS 风格题)。
  它们与 FCS/ALE **同输入分布**（这正是补救文档要的落地能力），但需**逐条**确认没有意外撞到真实题。
- **Algorithmic track（172 数字题）语义扫描（见 §3）**：**基本干净**——40 道 shard 题 0 命中；竞赛经典
  `dsu-on-tree` / `gale-shapley` / `karmarkar-karp-binpacking` / `balkanoi-2011-time-is-money` / `schonhage-strassen`
  + `tammes-problem` / `thomson-problem` 标为 REVIEW（本机只有 40/172 题面，剩余 ~132 题无法核；这些经典**可能**正好是
  某道 FCS 题的意图解，n-gram 抓不到改写题面）。

---

## 3. 语义长尾（subagent 复核）

> 本节汇总三个并行 subagent 的复核结论（FCS 语义扫描、发现题族完整性、MLS 非原生 baseline 确认）。

- **MLS Type-1 确认**（已并入 §2.1）：抽查 8 条，2 条命中（jSO / UniPC），且**精确对应** finale-非原生-baseline
  这一机械信号；据此全量枚举得 51 条。

- **FrontierCS 语义扫描**（读了全部 40 道 shard algorithmic 题面 + `frontier_eval/tasks/*` research 任务定义 + 命中
  method 的 `context.md`）：
  - **algorithmic track 基本干净**：40 道题里 **0** 个 method 重构了意图解（这些题是新造的交互/构造/优化题）；
    只有两个弱/可能（`sidon-sets`→#111 XOR-distinct、`tammes/thomson`→#112 SphereSpread），且都有真实技术 gap。
    之前标的竞赛经典（dsu-on-tree 等）不对应这 40 道，须映射到另外 ~132 道（本机无题面，留 REVIEW）。
  - **research track 泄露多**（约 20 个 method，均已并入 §2.4 / 注册表）：**HIGH**——FlashAttention（`flashattention`,
    `flash-attention-2`, `flash-v1/2/3`）、MLA（`mla`, `mla-attention`）、`simp`（SIMP 拓扑优化）、`proton-therapy-impt`、
    `malloc-allocator`、denoising 三基准法（`magic-imputation`, `magic-diffusion`, `knn-smoothing`）；**MEDIUM**（技术相关但
    有 gap）——`mma-svanberg`, `gappy-pod-sensors`, `importance-sampling-rare-event`, `muon-tomography`, `quadruped-gait`,
    `robot-arm-time-optimal`, `aes-rijndael`, `sha256-merkle-damgard`, `keccak`, `ziegler-nichols-pid`, `coverage-path-planning` 等。
  - **纠正的假阳**：`buddy`（是链接预测 GNN，非 buddy 分配器）、`pid-lag`（safe-RL，非 PID 整定）、`full-attnres`（残差流，非 flash）。

- **发现题族完整性**：注册表**完整覆盖**所有 ThetaEvolve/TTT 明确任务；确认 `kissing_number`/`matmul`/`heilbronn`
  为**可运行**评测题（已从 MEDIUM 升 HIGH）。**新增一条低severity**：`sums-and-differences-sets`（= AlphaEvolve 附录 B.6
  MSTD，有可运行 example，但不在本轮 run 子集）→ MEDIUM。其余 AlphaEvolve 可运行题（uncertainty_ineq / min-max-min-dist /
  hexagon_packing / sphere-packing / sidon）在**已发布 SFT 里无对应样本**，不新增泄露行。
  - **确认「同名异题」不标记**：`erdos-ko-rado`（EKR 定理）、`erdos-stone-simonovits`（Turán 数）、`probabilistic-method-erdos`
    （Ramsey 下界）、`uncertainty`（多任务 loss 加权，非 uncertainty 不等式）、`schonhage-strassen`（整数乘非矩阵乘）、
    `triton-gelu`（通用 GPU kernel 非 TriMul）、`sidon-sets`/`sphere-packing-lattices`（AlphaEvolve 有名但无可运行 example）——均非评测题。

---

## 4. 计数与处置（`sft/innovation_sft.jsonl`，2698 行）

**泄露识别**（信息量，含跨 benchmark 重复）：leak 1500 行；严重度 CRITICAL 212 · HIGH 876 · MEDIUM 362 · REVIEW 50；
按评测 MLS 1035 · FCS 391 · ALE 354 · TTT 71 · THETA 66；唯一泄露源 slug 608 个。

**最终 turn 级处置**（按 §0 策略，`decontam_action` 字段）：

| decontam_action | 行数 | 含义 |
|---|---:|---|
| `drop_row` | 65 | 整条删（28 discovery-search/record method + 8 discovery/AHC039 trajectory 的 full/folded 行） |
| `drop_finale_turn` | 416 | 属于 51 条 MLS Type-1 轨迹；build 时**只去掉 finale 那一轮**（full 少一轮 / finale-当前轮的 folded 行消失） |
| `keep_flagged` | 1019 | 保留但已标注（MLS baseline ladder 同任务、FCS-research paper 方法、39 standalone finale method、paper discovery、v4/wave2） |
| `keep` | 1198 | 干净 |

- 去污染副本 `decontam/clean_rebuilt/innovation_sft.jsonl`：**2582 / 2698**（净删 116 行）；`wave2`(758)/`v4`(346) **原样保留**。
- 构建期 denylist（`benchmark_denylist.txt`）= 实际整条删的 **36** 个 slug（28 method + 8 traj）；finale 折叠由 `type1_finale_traj`(51) 控制。

---

## 5. 交付物（`decontam/`，原始 `sft/*` 零改动）

| 文件 | 用途 |
|---|---|
| `eval_registry.json` | 评测任务族 → 训练 slug 的人工整理映射（可编辑，改完重跑 `audit_leakage.py`） |
| `audit_leakage.py` | 确定性审计器：产出标签 + `decontam_rules.json` + denylist |
| `decontam_rules.json` | **build gate 读的规则**：`drop_method_slugs`(28) / `drop_traj_slugs`(8) / `type1_finale_traj`(51) / `keep_paper_methods` |
| `leakage_tags_{sft,wave2,v4}.jsonl` | **行对齐**逐条标签（`leak/decontam_action/type1_nonnative_finale/exclude_from_summary/benchmarks/severity/reason`） |
| `mls_type1_nonnative_finale.json` | 51 条注入非原生 finale 的 MLS 轨迹 + 注入方法 |
| `benchmark_denylist.txt` | 36 个整条删的源 slug（28 method + 8 traj） |
| `clean_rebuilt/innovation_sft.jsonl(.gz)` | **去污染副本**（由 `build_sft.py` gate 重建，2582 行）——干净训练用**这份 + 原 wave2 + 原 v4** |
| `summary.json` | 汇总计数 + `decontam_action` 分布 |

**去污染怎么用**：`sft/build_sft.py` 默认 `INNOVATION_DECONTAM=1`（读 `decontam_rules.json` 自动 gate）；
`INNOVATION_DECONTAM=0` 回到 pre-audit 构建。审阅期我把干净版写到 `decontam/clean_rebuilt/`（未覆盖 `sft/`）。
**总结训练数据时**：跳过 `exclude_from_summary==true` 或 `decontam_action∈{drop_row,drop_finale_turn}` 的样本即可。

---

## 6. 建议

1. **MLS**：报告 MLS 结果时明确「同任务 in-distribution，非严格 held-out」；训练用 `clean_rebuilt`（已去 discovery-search
   + AHC039 + 51 条 Type-1 finale turn，但**保留** baseline ladder）。同任务 baseline ladder 按你的口径保留。
2. **discovery（ThetaEvolve/TTT）**：heuristic/进化搜索纪录构造（`-record`/`-frontier-largeN`/`-slsqp`/`-sa`/`-funsearch-evolved` 等）已删；
   正经 paper 方法（circle-packing-in-square/cap-set/kissing/matmul/erdos-min-overlap/autocorr/heilbronn-triangle/LABS）按你的口径保留。
3. **ALE**：AHC039（method + trajectory）已整条删。若你知道 ALE-Bench 实际题号,可据此再收/放。
4. **FrontierCS**：按你的口径 research 重叠**不动**（可能不测）；`v4`/`wave2` 合成 n-gram 复核只撞 boilerplate，放行。
5. **构建期 gate**：已接进 `sft/build_sft.py`（默认开）。以后重建 SFT 自动去污染。

---

## 7. Caveat

- n-gram 证「逐字」，本审计证「同任务 / 同源重构 / 语义」；两者互补，本文覆盖后者。
- ALE-Bench / 完整 FrontierCS 题库本机无 manifest：AHC039 的**确切**评测集归属未能本地确认，按「和 ALE 完全一样即删」处理。
- 保留的 FCS-research / discovery-paper / v4 / wave2 是按你 2026-07-08 口径**保留**的；若某评测口径改变（如决定测 FCS research），
  再把对应 family 从 `keep` 调成 `drop`（改 `eval_registry.json` / `decontam_rules.json` 重跑即可）。
