# 同事四项数据提案 + 三个"不要动"的核查与判断（2026-08-17）

> 背景：训练侧同事提出「三个不要动 + 要改的四件事」。本文对每一条做**事实核查**（附证据路径 /
> commit sha / 可复算脚本）、**历史背景**（当时为什么那样做）、**判断**、以及**如果做，具体怎么做、
> 要避开什么**。
>
> 立场声明：本文双向存疑。同事有两条**说对了而且我们自己的历史记错了**（①、④ 的缺口是真的，
> 而且是我们自己在 07-21 亲手造成的）；也有两条**把两件不同的事混成一件**（①的归因、③的对象），
> 以及一条**关键数字来自不同口径的巧合**（1352）。
>
> 本文**没有改动任何数据文件**，所有统计都是只读复算。

---

## 0. 结论速查

| 提案 | 判断 | 一句话 |
|---|---|---|
| ① 恢复被删的验证语言 | **部分可取，且比同事说的更该做** | 缺口是真的、比他说的更大；但**不是 gated_v2 干的**，是 07-21 的 `audit_edit_reasoning_workflow.js` 越界删到了 `data_v4/`（审计原文明写 v4 "修复=去模板化，**非删减**"）。做法要换：不是 diff `ctl_old` vs `gated_v2`，而是从 `23ff22f29`（07-16 树）**只回填 data_v4 的验证/边界/交卷段**。 |
| ② 从 wave3 筛"长且会交卷"切片 | **可取，是本轮性价比最高的一条**；但筛法要改 | wave3 expert-CP 1,334 条本来就是"真 rollout、think 中位 22.4k tok、99.9% 以代码收尾、自查密度 3.99/千 tok"。**承重的筛子是 32k 预算（切掉 21.5%），不是回溯密度**；同事的"回溯密度下四分位"筛子会把 974 条砍到 197 条，且违反我们自己 §3-R1 定的"自查密度 ≥2/千 tok"下限。 |
| ③ 竞赛子集代码语域清洗 | **不可取（前提被实测证伪）** | 338 条 expert-CP 里 `// TODO` 出现在 **100% 的 prompt、0% 的 assistant 侧**（它是 FrontierCS 题面给的 C++ 骨架）；"代码块后写教程" 0.0%；"以代码块收尾" **100.00%**（不是 0.00%）。soupNEW10 输出里的 `// TODO` 是**照抄题面骨架没填**，是交付失败，不是语料语域污染。 |
| ④ commit-and-ship 示范缺口 | **(a) 可取；(b) 明确不可取** | `~2%` 属实（`DATA_REMEDIATION_zh.md:25` 落地词 2.2% = 28/1245；严口径 0.2% = 3/1201），`base 56%` 查无出处，"唯一正预测"从未跑过回归。而且缺口**是 07-21 亲手删掉的**：data_v4 的"最终交卷决定"段 296/346 → 104/346。所以不需要模板桥接句——把真的那 192 段拿回来即可。(b) 我们**已经做过**：`tools/make_commit_coda.py`，它留下的逐字口号 "a plain correct submission beats an ambitious broken one" 被 07-16 注水审计当作模板指纹点名，现在树里还剩 25 处。 |
| 不要动长度 | **要按切片拆开才成立** | "加长有害"有很强实测（截断是主损失通道，24 倍胜算比，r=−0.808）；"截短有害"在本机**没有 `capped16k` 这个臂**（零命中），`r3_minthink` = `method_minthink`，**soup 建了但一个 eval 都没跑**。不过"别截 think"确实是我们成文立场——**只针对本地 innovation 语料**（p95 ~9.9k、无一条 >16k）。而 rollout 来源的 wave2/wave3 有 21–41% 超预算，那里必须设上限。 |
| 不要注入回溯词 | **"不要注入"成立；"四臂全部负相关"查无实据且符号相反** | 本机从未做过注入实验；被动测量里 `"Wait"/万字符` 沿 α 从 16.13 掉到 0.71 时 FCS **同向**塌陷（正相关）；真正的"四臂"是 RL 四起点臂，那里 corr(Δ截断,ΔFCS)=**+0.29**。禁的是**合成注入**（有 `derewrite_workflow.js` 的成文 guardrail 支撑），不是禁"自带真回溯的数据"（wave2-cp rollout 7.45/千 tok > base 4.93）。 |
| 不要删创新内容 | **前半成立，后半（102 题）查无实据且反向** | "亏损全在执行侧"是我们的核心定论。但"102 道难题上创新臂正向"没有出处（`102` 全部解析成 `1102 题` = RL 训练集）；本机最接近的记录是**反向**的：`SOUP_TRADEOFF_zh.md` §7.2 "**没有一例**是「soup 提出 START 提不出的新想法、且正常落地得分」"。且要按 benchmark 分：创新在发现类正、在 FCS/ALE 交付类净负（`pure_a10` 零创新数据就到 6.417）。 |

---

## 1. 名词考据：这些东西是什么、在哪、谁做的、当时在修什么

> ⚠️ **重要前提**：**训练侧不在本机**。全机 `find` 对 `gated*`/`ctl_old*`/`innnew*`/`capped16k*`/
> `soupNEW*`/`wave3_full*` 只有一个无关命中（MLS-Bench 的一个 baseline log）。训练树在远端 Slurm：
> `ROOT=/scratch/gpfs/CHIJ/bohan/fs`、`LF=/scratch/gpfs/CHIJ/bohan/fs/LF-innov`、
> `SFT_OUT=/scratch/gpfs/CHIJ/bohan/fs/models_sft`
> （硬编码在 `experiments/scripts/orchestrate/cc_orchestrator.py:24-31`）。
> 这些名字**都是 LLaMA-Factory 的 dataset 注册 key / run tag**，而
> `LF-innov/data/dataset_info.json` **从未入过库**（`SFT_DATA_FULL_FORENSICS_zh.md:56` 记为
> "514 行未提交的工作树改动"）。所以本机能核的是**数据生产侧**，训练侧只能靠文档与 job id 对齐。

| 名字 | 是什么 | 何时 / 何处 | 当时在修什么 |
|---|---|---|---|
| **decontam gate** | 评测集泄漏门（丢行） | 07-08，`6683a6ee2`；`decontam/audit_leakage.py` + `sft/build_sft.py:38-57,377,466` | 训练数据对 5 个 benchmark 中的 4 个有结构性泄漏（`experiments/DATA_LEAKAGE_AUDIT_zh.md`） |
| **verbatim-code gate**（"gated"） | `train_answer.md` 的代码必须是 `answer.md` 代码栅栏的逐字（去空白）子集，否则该条**改走 answer.md 通道** | 07-23，`dafb6c86f` → `167ff9234` → `0d8848ab3`，07-25 扩 `b92fd7e95`；`sft/build_sft.py:138-165`；`tools/ta_gate_check.py` 是独立复算 | 07 月审计发现 **548 个 method 的 train_answer 代码是"凭空编的/重实现的"**（从未运行、从未评审，抽样有硬 bug），另有 39 个近逐字但带静默改动。用户拍板"宁用 answer.md 也绝不训坏内容" |
| **gated_v2** 的 "v2" | **输入侧**去剧透：218 道题的 statement 删掉 Background/Evaluation-settings 里交代解法与预解陷阱的句子 | 07-25，`daee31ba3`；`tools/data_quality_fix_workflow.js:106-119`（class `TRIM`） | 评测时模型只拿到光秃题面，而训练时题面把解法送到嘴边 → 样本教的是"转录"不是"推导"。commit 原文："**measured coupling was 2/218 so the traces needed almost no reconciliation**" |
| **gated_v2**（数据集名） | = 输入泄漏清理 + gated 数据 + wave2 + allver maintain | 训练于 07-26，`43c83c985` / `c7d47e80c` | 定义见 `experiments/SFT_RL_SUMMARY_zh.md:57` |
| **ctl_old / ctl_new** | 07-27 的**控制变量 A/B**：同一个 HEAD builder，分别跑 07-08 数据树（`git worktree` at `8ae41b601`）与当前数据树；其余逐字锁死，两个 yaml 只差一行 | `32393e7ab`；job 11658797 / 11658798；`SFT_RL_SUMMARY_zh.md:171-176` | 量"07-08 之后 1348 个 commit"的净效果。ctl_old 2702 行 / ctl_new 2590 行；`method_ta_bypass` **590→7** |
| **r3** | **两个东西共用一个名字**：(a) `innovation_maintain_r3`（903 行，07-06，fold-think 修复后的重渲染）；(b) `innovation_wave2_r3`（758 行，gated_v2_allver 用的 wave2 成分，对应 allver 用的 `innovation_wave2_clean` **1352 行**） | `SFT_RL_SUMMARY_zh.md:165, 189-199` | (a) 修 fold-think；(b) wave2 的一次重建 |
| **r3_minthink** | 文档里叫 `method_minthink`（think 长度地板臂）。**soup 建过，但一个 eval 都没跑**（`:142` 列在"仍缺 eval"）。旁系 `innovation_maintain_r3_longthink`（259 条）FCS 3.644†，但该单 shard 数 `:231` 已**作废**（修正后 ~4.4） | `SFT_RL_SUMMARY_zh.md:142, 195, 210, 215, 231, 262`；yaml 在 `LF-innov/examples/train_full/auto/os-q35_a100_*_r3*.yaml`（`INNOVATION_CAMPAIGN_REPORT_zh.md:206`） | — |
| **capped16k** | **本机零命中**（全部 md/py/txt/git 历史 + 5,932 个 session jsonl）。`9.16` / `4.36` 各有出处但分属两个 benchmark：`soup_ctl_new_a50` Research avg@5 **9.160**（`SFT_RL_SUMMARY_zh.md:319`）、`q35_inst_start` FCS-research-GPU **4.366**（`raw_outputs/README_zh.md:85`）。所有 run 的 `cutoff_len=53760`（`SFT_DATA_FULL_FORENSICS_zh.md:171`） | — | 我们成文的立场其实是**别截 think**（`CASE_STUDY_zh.md:230`、`DATA_REMEDIATION_zh.md:189`），理由是本地语料 think p95 只有 ~9.9k、无一条 >16k |
| **innold / innnew** | 08-03 用户重设 9B 矩阵里的两个 innovation 数据代际：`innold` = 07-08 老版，`innnew` = 当前版 | `SFT_RL_SUMMARY_zh.md:752-754, 766-783, 825-879` | 回答"07-08 之后的重写是改好了还是改坏了" |
| **soupNEW10** | = `soup_innnew_a10`（innnew 的 α=0.1 soup），后来做了 RL 的四起点之一 | `SFT_RL_SUMMARY_zh.md:984, 1009-1026, 1086-1091, 1134` | — |
| **wave3_full** | 2026-08 蒸馏批（hard-CP 拒绝采样 keepers），当时 2,097 行，现已 3,161 行（`sft/innovation_wave3_sft.jsonl`） | `sft/README.md` §4；assembler `tools/assemble_wave3.py`；rollout driver `tools/hardcp_rollout.py` | 补 FrontierCS 能力缺口（启发式 optimization、post-cutoff AHC、CodeContests+ 强测例、27B 硬失败重滚） |
| **expert-CP 338** | **不是单独文件**：`sft/innovation_sft.jsonl` 里 system 以 `"It is now year 2025. You are an expert competitive programmer."` 开头的 **恰好 338 条**（`sft/build_v4.py:22` 的 `V4_SYS`）。源目录 `data_v4/*/`，独立文件 `sft/innovation_v4_sft.jsonl` 是 346 条（差 8 条是 `daee31ba3` 丢的 textbook 单元） | 复算：`python3 -c "import json;print(sum(1 for l in open('sft/innovation_sft.jsonl') if json.loads(l).get('system','').startswith('It is now year 2025. You are an expert competitive programmer')))"` | 修 FCS 落点：method 数据 98% 落点是 Python/class，而 FCS 只判"单文件 C++ 读 stdin"（`experiments/DATA_FIX_FCS_LANDING_zh.md`、`DATA_WAVE2_FCS_CPP_zh.md`） |

**时间线（关键）**：
`07-05 traj deepen(9819ba7a)` → `07-06/07-08 r3 / ctl_old 树(8ae41b601)` → **`07-16 注水审计(DATA_REASONING_BLOAT_AUDIT_zh.md)`** →
**`07-21/22 audit-edit 大批落地（560 commit，其中 data_v4 245 个）`** → `07-23 verbatim-code gate` →
`07-25 218 题去剧透` → `07-26 gated_v2 训练` → `07-27 ctl_old/ctl_new A/B` → `08-03 innold/innnew 终审`

---

## 2. ① 恢复 gated 清理误删的验证语言

### 2.1 事实核查

**(a) "gated_v2 去泄漏时把思考砍短了 27%" —— 数字对，归因错。**

- −26% 这个数字是**我们自己文档里的**：`SFT_RL_SUMMARY_zh.md:800`，"**think 压缩 −26%（p50 19.7k→14.5k 字符）**"，
  出自 08-03 的 innold vs innnew 逐行 diff（commit `092699e6f`）。同事的 27% 与它同源。
- 但它**不是 gate 做的**。gate（`sft/build_sft.py:138-165`）只作用在 **train_answer 的答案通道**，
  一个字都不碰 reasoning；218 题去剧透（`tools/data_quality_fix_workflow.js:114`）明写
  "*otherwise leave both files untouched. The trace must still read as if it worked the problem out*"。
- 真正砍短思考的是 **07-21/22 的 `tools/audit_edit_reasoning_workflow.js`**（245 个 `recover audit-edit v4:*` +
  `0f2858d3f` / `9f7993460` / `66308ba4a` / `4777333c4` 等 rule-based batch-cut）。

**逐版本复算**（`git cat-file --batch` 读历史 blob，不落盘）：

| 切片 | n | 指标 | 07-08/07-16 | 07-21 | 07-22→HEAD |
|---|---|---|---|---|---|
| `methods/*/results/reasoning.md` | 1,242 | 中位字符 | 23,877 | 23,362 | **22,073（−7.6%）** |
| | | 有 ≥1 处验证的文件占比 | 64.5% | 60.9% | **52.1%** |
| | | 样例式验证（"take n=3"/"plug in"…）次/文件 | 0.73 | 0.71 | **0.69（−5.5%）** |
| | | 仪式标记（"before I commit"/"falsifiable"…）/万字符 | 0.20 | 0.18 | **0.12（−40%）** |
| `trajectories/**/reasoning.md` | 680 | 中位字符 | 19,970 | 13,408 | **13,408（−32.9%）** |
| | | 有 ≥1 处验证的文件占比 | 59.7% | 6.9% | **6.9%** |
| | | "let me verify/check/trace" 总数 | 403 | 26 | **26** |
| **`data_v4/*/reasoning.md`（= expert-CP）** | **346** | **中位字符** | **17,466** | **6,790** | **6,790（−61.1%）** |
| | | **有 ≥1 处验证的文件占比** | **68.2%** | **11.6%** | **11.3%** |
| | | 样例式验证 次/文件 | 1.67 | 0.51 | **0.51（−69%）** |

→ **同事说的"砍短 27% / 验证率掉 2/3"在 methods 上被高估，在 trajectories 和 data_v4 上被严重低估。**
data_v4 是 **−61% 长度、验证覆盖 68.2%→11.3%**。

**(b) "验证率 15.6%→5.0%、涉及 1352 个 assistant turn" —— 这两个数在本项目历史里查无出处。**

- 对 `/srv/home/bohanlyu/.claude/projects/` 下 **5,922 个 session jsonl（977 MB）** 以及本仓库全部
  `*.md/*.py/*.txt` 递归 grep：`验证率` / `verification rate` / `15.6%→5.0%` / `1352 个 assistant`
  **仅命中今天（2026-08-17/18）这一场会话及其子 agent**——也就是同事这段提案本身和读它的 agent。
- **`1352` 在所有既有 transcript 里都是 `innovation_wave2_clean.jsonl` 的行数**
  （`8b388ae4 wave-2 refresh: 758 -> 1352 verified examples (code-heavy)`；
  `SFT_RL_SUMMARY_zh.md:165`）。而 §1.7 早就写明：`allver` 用 wave2_clean **1352 行**，
  `gated_v2_allver` 用 wave2_r3 **758 行**——**两个 build 的 wave2 成分根本是两个不同数据集。**
  → 强烈怀疑同事的 diff 是 `ctl_old`-系 build vs `gated_v2` build，把 **wave2 成分被整体换掉**
  的 1352 行，误读成"1352 个 assistant turn 的验证语言被删"。这正是 §1.7 已经点名的三重混杂之一。
- 另一个数字巧合：`15.6%` 在 `CASE_STUDY_clean_wd03_zh.md:152` 是 **base 在 FCS 上的 score>0 率**。

**(c) "验证事件是实测质量杠杆（base ≥3 次验证 15.5 分 vs 零验证 4.9 分）" —— 本机无此测量记录。**
方向上与我们的 §6.4 一致但不是同一件事：`CASE_STUDY_clean_wd03_zh.md:190` 的表是
base "Wait" 密度 15.0/万字符、think 均长 86.9k 字符、mean 7.05；SFT 把 Wait 打到 0.35、think 塌到 6.6k、mean 0.31。
**注意混杂**：验证次数与 think 长度、题目难度强相关，"≥3 次验证 vs 零验证"的分差里有多少是长度、多少是验证，本机没有做过控制。

### 2.2 历史背景：当时为什么要删

`experiments/DATA_REASONING_BLOAT_AUDIT_zh.md`（07-16）的实测：

- traj deepen 轮（`9819ba7a`，678 rungs）把 reasoning 从 mean 1695 词填到 3323 词，**变异系数 0.24→0.09**（填配额）；
  增量与原文长度相关性 **−0.74**（原来越短塞得越多）。
- 注入的仪式模板：`"Let me verify/check/trace"` **×3.5（83→565 处）**、`"Before I commit"` **×4.5（25→224）**；
  **~78% 的自查从未发现任何问题**（800 字符窗口内零否定/修正标记）。
- 单轮 methods：`"Let me verify"` **1556 处，83% 从未发现问题**。典型假验证：adam 用 20 万次 Monte Carlo
  "确认"一个刚证完的恒等式、验到 8 位小数、全篇零失败。
- 更糟的两条：**hindsight 化装成预测**（deepen 把"事前预测"改写成与该 rung 自己 feedback 表精确一致的数字，
  实锤样本 `causal-discovery-discrete/01`、`optimization-convex-concave/02`、`cv-diffusion-cfg/01`）；
  **answer 代码逐字回贴进 reasoning**（24 traj + 244 methods）。

编辑器的保留判据写得很清楚：**只保留"改变走向 / 钉下后文要用的量 / 撑起否则悬空的主张"的自查**。
`tools/data_quality_fix_workflow.js:96` 给 bug-戏码改写器的指令是
"*Keep the technical content and the length; remove the theatre, not the thinking.*"

**而且这个风险当时就被写在案上了**（`SFT_RL_SUMMARY_zh.md:178`）：
> "已知的一个预期风险：去掉「表演式 bug 戏码」删掉的是模型学到的**自检行为**。如果 FCS 奖励
> 「写完检查一遍再交」的代码，这一项可能是负效果——ctl 对照能验证这个猜想。"

### 2.3 ⭐ 决定性发现：07-21 的 audit-edit **越界删到了 data_v4**，而审计原文明写 v4 不该删

`DATA_REASONING_BLOAT_AUDIT_zh.md:86` 三类数据分工那一段，逐字：

> "**v4 正文最干净（mean 5.5%）但结构病最重——83.2% 同一句开头 + 47% 同构两幕 bug 剧，
> 跨文件同构是训练侧实证会被逐字模仿的（修复=题目特异性重写开头与 pitfall，非删减）**"

**审计的处方是"去模板化，非删减"；执行结果是把 v4 删掉了 61%。**
对 346 个 `data_v4/*/reasoning.md` 逐标记复算 07-08 vs HEAD：

| 标记（按文件计） | 07-08 | HEAD | 变化 | 性质 |
|---|---|---|---|---|
| 固定开头 "Reading the problem…" | 302 | 14 | **−288** | ✅ 该删（模板指纹） |
| "convinced myself" 口癖 | 163 | 2 | **−161** | ✅ 该删 |
| "deliberately" 口癖 | 310 | 81 | **−229** | ✅ 该删 |
| "Causal recap" 尾部复述 | 289 | 50 | **−239** | ✅ 该删（与 answer 重复） |
| **边界枚举段（"Edge cases"）** | **285** | **61** | **−224（−79%）** | ❌ 误删 |
| **交卷决定段（"Final solution"/"that is what I ship"）** | **296** | **104** | **−192（−65%）** | ❌ 误删 |
| **手工回溯（"Re-trace"/hand-trace）** | 217 | 74 | **−143（−66%）** | ❌ 误删 |
| 溢出/哨兵检查 | 291 | 242 | −49（−17%） | 大部分保住 |
| 独立 oracle / 差分测试 | 193 | 179 | −14（−7%） | 基本保住 |

被删内容的实样（`git show 23ff22f29:data_v4/fcs-p2-01/reasoning.md | tail -60`，15,622 字符 → HEAD 5,753 字符）：
`S=0` / 面额全大于 S / 重复面额 / 公因子不可达 / 最大规模 10^8 次松弛的时限与内存估算、
"用 BFS-by-coin-count 这个**结构不同**的独立 oracle 做差分测试、500+ 用例零不一致"、
以及一段明确的交卷决定 "*That is what I ship — one self-contained file, the simple provable O(S·n) DP
I can defend rather than the greedy I broke*"。**这正是同事要的东西，也正是 ④ 要的东西。**

同时也要诚实：同一段里确实带着 "deliberately"（HEAD 仍有 81/338 = 24%）与 "Causal recap" 尾巴——
所以正确动作不是整段回滚，是**分段回填**。

### 2.4 判断：**部分可取，方向对、归因错、做法要换**

- ✅ **缺口是真的，而且比同事说的严重**（data_v4 验证覆盖 68.2%→11.3%，长度 −61%）。
- ✅ **这一刀违反了我们自己的审计处方**（v4 应"去模板化，非删减"）——这是我们历史记错的地方，同事说对了。
- ❌ **"gated 清理误删"的归因错**：gate 只碰 train_answer 答案通道，去剧透只碰 statement。
- ❌ **"拿 ctl_old/r3 版本和 gated_v2 同记录 diff"是错的基线**：这两者之间横跨 1348 个 commit
  **且 wave2 成分被整体换掉（1352→758）**，diff 出来的东西大部分不是"被删的验证语言"。
- ❌ **"恢复被删 span 中含验证语言的部分"用在 methods/traj 上是危险的**：那里被删的主体
  是实测 78–83% 从不触发的表演确认，还夹着 hindsight 违规（deepen 把预测改成与自己 feedback 表逐条对齐）。
  **verbatim-code gate 检测不出 hindsight**，所以"恢复后重跑 gate"这道保险**不覆盖主要风险**。
- ❌ 顺带：`MODEL_FAILURE_ROOTCAUSE_zh.md:1.4` 说模型的一个主要失败就是
  "**只在极小样例上验证 → 交付在规模上错误的构造**（34/200）"。无差别恢复表演式小样例验证，
  正好是在加强这个失败模式。

**必须摆在桌面上的反方证据（对"整体回滚到老版"最不利的一条）**：
我们**已经做过**"新版 vs 老版"的干净对照，结论是新版不差甚至更好——
- `SFT_RL_SUMMARY_zh.md:681`（§2.6.1）：`ctl_old`（07-08 数据）vs `ctl_new`（当前管线）raw 臂
  FCS **2.511 vs 2.816**，配对差 **+0.305，95% CI [−0.935, +1.512] 不显著**；research 同向 +0.67 ns。
- `:786`（08-03 avg 级 A/B）：**a10 新版两榜大胜**（FCS 6.47 vs 5.23、Res 11.15 vs 8.75）；
  a20 老版领先被归因为"67% 抽奖题方差 + **残余三分之一疑似污染红利**（老版多保留 ~14 行评测任务原文）"。
- `:811` 终论："**新版数据更干净且不差于老版；老版看似的局部优势由抽奖方差和污染残余构成。**"

→ 所以本节主张的**不是**"07-21 那一刀整体错了"，而是**"那一刀在 data_v4 这一个切片上越了审计给它划的界"**。
两件事可以同时为真：整体方向是对的（污染清掉了、模板指纹清掉了），
局部误伤是真的（边界枚举 −79%、交卷段 −65%）。**回填必须是分段的、只针对 data_v4 的，否则会撞上上面这组对照。**

### 2.5 如果做，怎么做

**只做 data_v4（expert-CP 346/338）这一个切片，走"分段回填"而不是整档回滚。**

1. **基线选 `23ff22f29`（07-16 树）而不是 `ctl_old`/`gated_v2`**——它是 audit-edit 落地前的最后一个树，
   与 HEAD 之间的 diff 就是这一刀本身，不掺 1348 个 commit 和 wave2 换血。
2. **只回填四类段落**（逐段判定，不整文件回滚）：
   `**Edge cases**` 边界枚举段 / 手工 re-trace 段 / 独立 oracle 差分测试段 / `**Final solution**` 交卷决定段。
3. **明确不回填**：`Reading the problem…` 开头、`**Causal recap**` 尾部复述、`deliberately` / `convinced myself` 句式
   （这三样 07-21 删得对，回填等于把模板指纹装回去；`DATA_RECOMMENDATIONS_zh.md:20` 的硬指标是
   **top1 开头占比 <5%**，现在 data_v4 的 top1 开头只有 4.1%，别破坏它）。
4. **human 侧一律保持 HEAD**（218 题去剧透的 statement 不能回退）。这一点同事说得对，照做。
   风险很低：`daee31ba3` 原文记录 "measured coupling was 2/218"。
5. **回填后必须重跑的三道闸**（"重跑 verbatim-code gate"不够）：
   - `sft/build_sft.py` 的 decontam gate（`INNOVATION_DECONTAM=1`）+ verbatim-code gate；
   - `decontam/audit_leakage.py` 全量重出 `leakage_tags_*`，**外加 08-03 记录的残余 9 行污染
     （AC1/AC3 ×6 = new#121-126、maxdet-29 ×3 = new#529/661/662，`SFT_RL_SUMMARY_zh.md:812`）一并清掉**；
   - `DATA_RECOMMENDATIONS_zh.md:60` 的 lint：工件 regex（`getenv(|ALE_BASELINE|<model_answer>|// ale-\d+`）、
     模型自指、年份一致性、退化检测、超 cutoff 检测。
6. **长度是安全的**：回填后 data_v4 的 think 中位从 ~1.7k tok 回到 ~4.4k tok，
   离 32k 预算还差一个数量级，**不会触发 §2.13(b) 的截断损失通道**（当前 data_v4 超 32k 目标占比 0.0%）。
7. **验收指标**（不要只看分数）：data_v4 切片的自查标记密度从 **0.16/千 tok** 提到
   `DATA_RECOMMENDATIONS_zh.md:20` 的硬指标 **≥2/千 tok**（参照：base 自然 4.93、wave2-cp rollout 7.45）；
   同时 top1 开头占比保持 <5%、`deliberately` 命中率不回升。
8. **要做 A/B**：innovation 侧只换 data_v4 一个切片，其余逐字锁死（学 `32393e7ab` 的两臂只差一行 dataset 名）。
   否则又是一次不可归因的对比。

---

## 3. ② 从 wave2/wave3 里筛"长且会交卷"的切片混入

### 3.1 事实核查

**wave3 是什么**：2026-08 的 hard-CP 拒绝采样 keepers（`tools/hardcp_rollout.py` → `data_v4/_hardcp/traces/*.jsonl`
→ `tools/assemble_wave3.py` → `sft/innovation_wave3_sft.jsonl`），现 **3,161 条**，
每条带 `pass_rate` 标签；覆盖 wave-2 之后全部新通过验证的 query（`sft/README.md`）。
三条引擎：Qwen3.6-27B on-policy 拒绝采样 / DeepSeek V4 Pro tier-2 兜底 / Codex 黑盒造题
（`experiments/DATA_WAVE2_FCS_CPP_zh.md`）。

**"从未混入"这句话不对。** wave3 进过训练，但只有一次、而且是不干净的对比：
`sft_w16_innov`（wave2 ×16 + `innovation_wave3` **当时只有 179 行**）——
`SFT_RL_SUMMARY_zh.md:353-373`：train_loss **0.2610** vs ctl_old 0.6858 / ctl_new 0.6998（过拟合信号），
research ×16+wave3 = 7.983 vs ×8 = 8.763，差 −0.78 [−4.28,+2.46] **不显著**；
而且文档自己标注"⚠️ **这个臂不是纯剂量变量**……严格说无法归因"。
另有 `w3vol_x1`（零重复全量 5,674）research 8.47、**FCS 5.43**（`SFT_RL_SUMMARY_zh.md:652-661`）。
→ **wave3 从来没有作为一个干净的、经过筛选的成分被单独测过。**

**为什么犹豫**：`SFT_RL_SUMMARY_zh.md:1075`（§2.13(c)）——
`wave3_full` **21.2% 的目标比整个生成预算还长**（中位 14–19k token），
而两个 soup 臂只用 `innovation_ctl_new`（1.2% 超限）；
LoRA 臂 12.6% 超限 → s5 撞上限 73.6% → FCS 最差。**这是本轮唯一被量化过的 wave3 风险，而且是长度。**

**这个切片本身的成色（本机复算，`sft/innovation_wave3_sft.jsonl`）**：

| 切片 | n | think 中位 | 自查标记/千 tok | 回溯词/千 tok | 以代码栅栏收尾 | 目标 >32k tok |
|---|---|---|---|---|---|---|
| innovation_sft expert-CP 338 | 338 | 1,702 tok | **0.16** | 0.02 | 100.0% | 0.0% |
| innovation_sft 其余 | 1,779 | 4,292 tok | **0.13** | 0.03 | 83.2% | 0.0% |
| wave2 expert-CP | 199 | 29,525 tok | 5.78 | 5.59 | 100.0% | **41.2%** |
| **wave3 expert-CP** | **1,334** | **22,387 tok** | **3.99** | 3.80 | **99.9%** | **21.5%** |
| wave3 全量 | 3,161 | 9,765 tok | 3.61 | 3.36 | 43.0% | 10.0% |

参照系（`DATA_RECOMMENDATIONS_zh.md:18`）：base 自然自查 **4.93**/千 tok，wave2-cp rollout **7.45**，
本地 innovation 数据 **0.13**（researcher 层中位 0.00），规格硬指标 **≥2/千 tok**、
think 分布与 base 重叠（**p50 15k–25k tok**）。

→ **wave3 expert-CP 正好落在规格窗口里**：think 中位 22.4k tok（规格 15–25k）、自查 3.99（规格 ≥2）、
99.9% 以代码收尾。**这是全语料唯一同时满足"长思考 + 真自查 + 会交卷"三条的切片。**

### 3.2 同事的筛法：估算对了，但筛子选错了

按同事的三条（以果断代码块收尾 + 14-gram 重复 ≤0.15 + 回溯密度下四分位）在 wave3 expert-CP 上实算：

| 筛法 | 剩余条数 | think 中位 | 回溯密度中位 |
|---|---|---|---|
| 仅"以代码栅栏收尾" | 1,333 / 1,334 | — | — |
| + 14-gram 重复 ≤0.15 | 1,327（**重复几乎是空筛子，只切掉 7 条 = 0.5%**） | — | — |
| + **目标 ≤32k tok**（同事没提） | **1,011**（切掉 287 = **21.5%**，这才是承重的那一刀） | 19,921 tok | 4.28/千 tok |
| + think ≥6k tok | 974 | 20,244 tok | 4.45/千 tok |
| + **回溯密度下四分位（≤1.58/千 tok）** | **197** | — | ≤1.58 |

- ✅ 同事"估计 200–400 条"的**估算准确**（197 条）。
- ❌ 但**回溯密度下四分位这一刀把 974 砍到 197，而且砍掉的正是我们规格里要的东西**：
  下四分位阈值 1.58/千 tok **低于我们自己定的 ≥2/千 tok 硬指标**，
  等于专门挑"最不像 base 自然推理"的那批。
- ❌ **14-gram 重复 ≤0.15 是空筛子**（wave3 已经过验证器，只有 0.5% 命中）。
- ❌ **真正承重、而同事明确排除的筛子是长度上限**——§2.13(b) 的 24 倍胜算比就是这条。

### 3.3 判断：**可取，是本轮最高性价比的一条；但按我们自己的筛法筛**

### 3.4 如果做，怎么做

1. 切片 = `sft/innovation_wave3_sft.jsonl` 中 system 为 expert-CP 的 1,334 条，筛：
   **(i) 渲染后总目标 ≤ 30k tok**（留 buffer；不是 think ≤30k，是**整条目标**）、
   **(ii) 以完整代码栅栏收尾**、**(iii) 14-gram 重复 ≤0.15**（保留，成本为零）、
   **(iv) 自查标记密度 ≥2/千 tok**（**方向与同事相反**：设下限，不是取下四分位）。
   预期 **900–1,000 条**。
2. **别忘了 `pass_rate`**：`8abc58b9d` 已勘误——`pass_rate` 是**难度标签不是对错**，
   `pass_rate=0` 是最难题的正解、可能最有价值，**正确用法是课程/加权，不是删行**。
   建议按 pass_rate 做加权而不是过滤。
3. **剂量要有对照**：§1.13 已证 wave2 剂量到 ×8 到头、×16 过拟合（train_loss 0.261）。
   wave3 切片先按 1× 进，与 `innovation_ctl_new` 同架子做两臂 A/B，别一次上高倍。
4. **评测口径**：选配方只看 FCS 172 题（§1.12 已定：research 64 题分辨不了任何剂量档，
   ALE 10 题的任何比较都在噪声内）。
5. **要防的坑**：wave2 expert-CP **41.2% 超预算**——如果顺手把 wave2 也混进来，
   就把 §2.13(c) 那个"12.6% 超限 → 撞上限 73.6% → FCS 最差"的坑原样踩一遍。

---

## 4. ③ 对 338 条 expert-CP 做"代码语域清洗"

### 4.1 事实核查：三条前提全部被实测证伪

对 `sft/innovation_sft.jsonl` 的 338 条 expert-CP 逐条解析（分 `<think>` / 答案通道，只看 assistant 侧）：

| 同事的说法 | 实测 | 证据 |
|---|---|---|
| "33% 程序带 `// TODO`" | assistant 侧 **0 / 338 = 0.0%**；`// TODO` 出现在 **338 / 338 = 100% 的 prompt（human 侧）** | `// TODO` 是 **FrontierCS 题面给的 C++ 骨架**（348 个 `data_v4/*/context.md` 里都有，如 `// TODO: compute the minimum number of coins…`）。多个 commit 就叫 "trim solution-revealing Background/title/**TODO** from context.md"（`210932f85`） |
| "this is getting complicated / I'll just assume 写进代码" | 338 条 expert-CP 里 **0 处**；全语料这两个短语在 `sft/innovation_sft.jsonl` 中 **0 次** | wave2 34 次 / wave3 72 次，且全部在 `<think>` 里、不在代码里（真 rollout 的自然口吻） |
| "代码块之后写教程" | 最后一个代码栅栏之后的正文中位 **0 字符**、p90 **0 字符**、>500 字符者 **0.0%** | — |
| "该子集 commit-in-tail **0.00%**" | **以代码栅栏收尾 = 338/338 = 100.00%** | 若"commit-in-tail"指交卷语言，则 think 尾部 1500 字符内 15.4%、全文 50.3% |
| "早期版本结尾被格式化流程裁掉过" | **反了**：早期版本结尾**更完整**——`data_v4` 的 "Final solution / that is what I ship" 段 07-08 有 296/346，HEAD 只剩 104/346 | 见 §2.3 表 |

**soupNEW10 的 `// TODO` 从哪来**：`soupNEW10 = soup_innnew_a10`，
它在 RL 里是**唯一发生退化坍缩的臂**（`SFT_RL_SUMMARY_zh.md:1086`，§2.13(e)）：
"Alternatively" 每千词 0.015→**3.004（200×）**、重复行率 6.2%→**30.2%**、type-token 0.424→0.230，
且 §2.14(f) 记录它的坍缩与 step 10 两次重启丢优化器动量重合。
它的输出里出现 `// TODO`，最简单的解释是**照抄题面骨架没有填**（交付失败），
而不是从语料学来的语域——因为语料的 assistant 侧一个 `// TODO` 都没有。

### 4.2 历史背景

`data_v4` 是 2026-07 为修 FCS 落点专门造的：**100% 单文件 C++ 读 stdin、100% 有 debug/自验环节**
（`experiments/DATA_WAVE2_FCS_CPP_zh.md` 二节；`sft/_v4_tags.jsonl` 里 `has_debug_episode=True` 345/346）。
它是全语料**最"会交卷"的一块**，不是最脏的一块。

它**真实存在的语域病**审计写得很清楚（`DATA_REASONING_BLOAT_AUDIT_zh.md:86`、
`DATA_RECOMMENDATIONS_zh.md:39`）：**83.2% 同一句开头 + 47% 同构"两幕 bug 剧"**（
"deliberately / convinced myself" 句式复用、S08-S11 同构），处方是**题目特异性重写开头与 pitfall，非删减**。
这一半已经做掉了（开头 302→14、convinced myself 163→2），剩下的是 `deliberately` 81/338 = 24%。

### 4.3 判断：**不可取（按提案原样）**

删"最终代码里的心声注释"和"代码块之后的说明段落"是在**删 0 个东西**；
"从老版本恢复自然结尾"这条**方向反了**（老版本才是完整的，是我们删的）。

### 4.4 如果还要动这个切片，该动什么

- **该做的是 §2.5 的回填**（把边界枚举/交卷段拿回来），不是"清洗代码语域"。
- **顺手可做的低成本项**：把剩下的 `deliberately` 81 处按题目特异性改写（不是删），
  把 top1 开头占比继续压在 <5%。这是审计原本给 v4 开的处方，一直只做了一半。
- **绝对不要做的**：把 human 侧的 `// TODO` 骨架删掉——那是 FrontierCS 题面的一部分，
  删了训练分布就与评测分布不一致了。

---

## 5. ④ commit-and-ship 示范缺口

### 5.1 事实核查

**`~2%` 属实，出处是 `experiments/DATA_REMEDIATION_zh.md:25-27`：**
> "| `reasoning` 出现「读输入/过测试/单文件/时限」等**落地词** | **2.2%**（28/1245） |
> | `reasoning` 出现「退回简单解/放弃花哨方案」等**退回词** | 7.5%（93/1245，宽口径） |"

更严的口径在 `CASE_STUDY_zh.md:226`："think 里只有 **0.2%（1201 条里 3 条）**以
「放弃花哨方案→退回朴素解→提交」结尾"。测量脚本是 `tools/data_audit.py`（`LAND` / `FALLBACK` 两组 regex）。
→ **这条缺口是真的、有脚本、可复算。**

**但要注意另一个 `~2%` 是不同的东西、而且已经补上了**：`MODEL_FAILURE_ROOTCAUSE_zh.md` R3 / §5(b)4 的
"当前约 2%"指的是 **train_answer 的落点格式**（98.6% Python、0.2% C++、1.9% 读 stdin），
目标 15–20%。这条现在已经达标：`sft/_sft_tags.jsonl`（2,590 条）`reads_stdin` = **15.4%**、
`has_cpp` = **16.1%**、`has_fallback` = **18.6%**，其中 v4 切片 100%/100%。引用时别把两个 2% 混在一起。

**`base 56%` —— 本机查无出处。** 全 `experiments/` 只有两处 `56%`，都与交卷行为无关
（`LATE_CATCHUP_FORENSICS_zh.md:20` 的截断率格、`SFT_RL_SUMMARY_zh.md:703` 的单题贡献占比）。
数值最接近的是 `CASE_STUDY_clean_wd03_zh.md:152` 的 **base 完整代码块率 56.6%**——
但那是**模型输出侧的交付成功率**，与语料侧的落地词占比是两个口径。
可比的语料侧数（训练目标是否以完整代码栅栏收尾）是：
expert-CP **100%**、innovation 其余 **83.2%**、wave3 expert-CP **99.9%**——**远高于 56%**。
另一个真实的 base 对照见 `CASE_STUDY_zh.md:101`：instruct 在完全相同的 6 道题上
**29/30 闭合 think、29/30 产出代码（96.7%），1700 token 就交卷且真拿分**。

**"唯一正预测模式" —— 从来没有跑过回归或相关分析。**
支撑只有两条：(i) `MODEL_FAILURE_ROOTCAUSE_zh.md:5` 的散文判断"唯一稳定的增益来自**「提交纪律」
和「在符号回归类任务上选对库」**"——注意它说的是"唯一稳定的**增益**"而不是"唯一正**预测因子**"，
而且**同时列了第二个并列因素**；(ii) `CASE_STUDY_zh.md:200` 的分组均值诊断。
**最接近的、而且是反向的实测**在 `CASE_STUDY_clean_wd03_zh.md:158`：
> "**短思考（<10k 字符）快提交"坏模式占比 base 12.1% → soup 18.4%（SFT 残留负迁移，该模式均分仅 2.3）→ RL 3.0%（被剪除）**"

→ **"会交卷"单独拿出来是负的**（快提交模式均分 2.3，全场均分 6.68）。
同事在 ② 里写的是"**长且**会交卷"，这个限定是对的、必须保留；④ 单独说"commit-and-ship 是唯一正预测"
就丢了这个限定，会指向已经被 RL 剪掉的坏模式。

**真正的缺口在哪（本机复算，07-08 vs HEAD，reasoning 尾部 2500 字符内的交卷决定语言）**：

| 切片 | n | 07-08 | HEAD |
|---|---|---|---|
| methods | 1,242 | 0.9% | 4.0%（**升了**） |
| **data_v4** | **346** | **28.9%** | **16.2%（腰斩）** |
| trajectories | 680 | 0.9% | 1.0% |

→ **交卷示范从来只存在于 data_v4 一个切片；而这个切片的交卷段被 07-21 那一刀删掉了 65%**
（整文件计：296 → 104，见 §2.3 表）。**④ 的缺口和 ① 的缺口是同一个洞。**

### 5.2 (b) 模板化桥接句：**我们已经做过，而且已经被自己的审计判为污染**

- `tools/make_commit_coda.py` 就是 ④(b)。它的 docstring 逐字：
  "*It is a CHEAP language/decision anchor, NOT the main fix … so it is not pure boilerplate,
  but it is **deliberately generic**.*"
  `DATA_REMEDIATION_zh.md:104` 把它定位为 **A2b「轻量收尾 coda（语域锚，非主力）」**，
  明写"**它只注入收尾的语域和决策结构，不做真验证**——当 A2 的 workflow 还没全量跑完时的廉价补丁"。
- 它注入的逐字口号 `"a plain correct submission beats an ambitious broken one"`
  在 07-16 的注水审计里被**当作模板指纹点名**（`DATA_REASONING_BLOAT_AUDIT_zh.md:78`：
  "逐字口号 … ×34"）。**现在树里还剩 25 处**（`grep -rl` on `methods/ trajectories/ data_v4/`）。
- 我们的长期原则（`DATA_REMEDIATION_zh.md:102`、`DATA_RECOMMENDATIONS_zh.md:20`）：
  "**这必须是真验证（subagent 实读代码、能 trace、能发现真 bug），模板做不到，所以它是
  workflow/subagent 任务，不是正则替换**"；de-rewrite 的 guardrail（`tools/derewrite_workflow.js`）
  写得更硬：
  "*CRITICAL GUARDRAIL: do not trade one tell for another. A STAGED/theatrical failure looks
  just as fabricated as a staged success. Never manufacture a dramatic dead-end for effect.*"
  → **"de-rewrite 是为了让验证变真，不是为了加句子"这条理解是准确的，(b) 与它直接冲突。**

### 5.3 判断

- **(a) 靠 ② 的 wave3 切片提供 —— 可取。** wave3 expert-CP 99.9% 以代码收尾、think 中位 22.4k tok，
  正是"长 + 交卷"，且是真 rollout 不是模板。
- **(b) 模板化桥接句 —— 不可取。** 已试过、留下逐字指纹、被自己的审计判为污染、
  与两条成文原则冲突；而且**根本不需要**：§2.3 表明有 192 个**真实的、题目特异的**
  交卷段就躺在 `23ff22f29` 的树里等着回填。**回填真的，不要写假的。**

### 5.4 如果做，怎么做

1. **主力 = §2.5 的 data_v4 回填**（交卷段 104 → ~296，即 30% → 85%）。零编造。
2. **补充 = §3.4 的 wave3 切片**（~900–1,000 条，99.9% 交卷）。
3. **验收要同时看两面**：交卷率**上升**的同时，"短思考快提交"（think <10k 字符且以代码收尾）
   的占比**不能上升**——这是 `CASE_STUDY_clean_wd03_zh.md:158` 实测均分只有 2.3 的坏模式。建议把它做成 lint 的一条硬门。
4. **不要**跑 `tools/make_commit_coda.py --apply`；顺手把现存的 25 处逐字口号按题目特异性改写掉。

---

## 6. 三个"不要动"的交叉核查

### 6.1 "不要动长度（加长和截短都有害）"

**加长有害 —— 强成立，是本项目证据最硬的一条。**
- `SFT_RL_SUMMARY_zh.md:1065`（§2.13(b)）：撞 32768 上限的 31,285 条 P(score>0)=**2.5%**、均分 0.007；
  正常收尾的 50,635 条 **60.5%**、均分 0.203 —— **24 倍胜算比，四臂完全一致**；
  评测侧 63.4% 撞上限，**17 个模型格上 r(撞上限率, FCS) = −0.808**。
- §2.14 收尾：**"修复方向是让模型早点收尾，不是给更多预算"**；且"写完的答案里，越长分越低：
  5.6k token 0.228 → 31k 0.174 单调"。
- §1.8：`longthink`（只留长思考的 maintenance 子集）是全表最差 **3.644**。

**"截短有害"—— `capped16k` 这个臂在本机不存在，但"别截 think"确实是我们成文的立场（对某些切片）。**
- `capped16k` / `capped_16k` **全机零命中**（本仓库全部文件 + 5,932 个 session jsonl 全量扫）。
- `9.16` / `4.36` 在本机各有出处，但**分属两个 benchmark、无法构成一个 before/after 对**：
  `SFT_RL_SUMMARY_zh.md:319` 的 **`soup_ctl_new_a50` Research avg@5 = 9.160**（同行 ctl_old 8.124，
  §327 用它做 "α=0.5 9.160 vs 9.149" 的对比）；
  `experiments/raw_outputs/README_zh.md:85` 的 **`q35_inst_start` FCS-research-GPU mean@5 = 4.366**。
  两者都不是 base 分。若训练侧真跑过 capped16k，请给 job id / summary 路径，我们落进 `SFT_RL_SUMMARY_zh.md`。
- ✅ **但同事的立场我们自己写过两次，而且他大概率是对的——只是限于"本地 innovation 语料"这一个切片**：
  - `CASE_STUDY_zh.md:230`："**9.4 数据并不超长**：think p95 ~9.9k，**无一条 >16k**。
    → 撞 32k 是分布外涌现，**修复不该截 think，而该注入「收尾/退回」样本**。"
  - `DATA_REMEDIATION_zh.md:189`："**不要做**：…**别截 think**（本不长，靠注入收尾样本治）。"
- `r3_minthink`（文档名 **`method_minthink`**）：`SFT_RL_SUMMARY_zh.md:262` 明写
  "**`method_minthink` 的 soup 建了但一个 eval 都没跑**"（`:142` 把它列在"仍缺 eval"清单里）。
  所以"这个思路不用再做"在本机记录里**不是"试过失败了"，是"从没测过"**。
  最接近的替代读数是 `innovation_maintain_r3_longthink`（259 条，think 中位 8572）
  FCS **3.644†**（`:210`），结论"longthink 最差"（`:215`）——
  ⚠️ 但 `:231` 后来把这类单 shard 数**作废**（"修正后约 4.4"），所以它也**不构成硬证据**。

**必须调和的一处内部张力（这是本条的关键）**：
- **本地 innovation 语料确实不长**（think p95 ~9.9k、无一条 >16k）→ 对它"别截 think"是对的。
- **rollout 来源的 wave2/wave3 很长**：`SFT_RL_SUMMARY_zh.md:1075` 记 `wave3_full` **21.2% 超预算**、
  `wave2_only` **30.2%**（本机复算：wave2 expert-CP **41.2%**、wave3 expert-CP **21.5%** 超 32k）；
  §2.7 的 P0 长度门也正是对着"CP/启发式切片 think 中位 21–32k token"开的
  （"= **截断螺旋的直接监督源**"）。
- → **同一句"不要动长度"对两个切片给出相反的正确动作。**

**判断：部分成立，但"长度不可动"这个统一表述会挡掉我们唯一量化过的最大杠杆。**
正确表述是按切片分开：
**(i) 本地 innovation 语料（含回填后的 data_v4，think ~4.4k tok）：不设上限、也别人为拉长；
(ii) rollout 来源的 wave2/wave3：必须设 ≤30k tok 的目标上限（24 倍胜算比就在这条）；
(iii) min-think 地板：既无正证也无反证（method_minthink 从未评测），要么别提，要么先跑一个小臂测掉。**

### 6.2 "不要注入回溯词（Wait/Alternatively）—— 四臂全部负相关"

**"不要注入"成立；"四臂全部负相关"这个证据本机不存在，而且被动测量的符号是反的。**

先说事实核查：
- **本机从来没有做过"注入回溯词"的实验**（全 5,932 个 jsonl + 全仓库零命中）。
  这个词表本身是真的：`SOUP_TRADEOFF_zh.md:33` 把 `wait / actually / hmm / reconsider / rethink /
  re-examine / alternativ* / another approach…` 定义为**「创新腔」**，用途是**被动测密度**，不是注入。
- **被动测量的符号是正的**：`AVERAGE_INNOVATION_zh.md:43-52` 沿 α=0→1 的
  `"Wait"/万字符` = **16.13 → 15.78 → 15.69 → 15.56 → 14.68 → 7.05 → 0.71**，
  而 FCS 同向塌陷 —— **Wait 密度是跟着分数走的，不是反着走的**。
- **唯一的负相关是臂内的、而且是两面的**：`CASE_STUDY_zh.md:200`
  "纯 SFT 成功样本研究腔 1.21 < 失败样本 2.10（**越像研究者越失败**）；
  START/soup10 成功样本务实标记 >> 失败样本（**越务实越成功**）"；
  `:215` 还补了口径依赖："本数据形态下「研究腔/篇」与得分**负相关**…但在 MLS 这类研究任务上是**净正**"。
  `SOUP_TRADEOFF_zh.md` §2.1 的总判是**解耦**不是负相关："**腔调是否平衡，完全预测不了得分**。"
- **本机唯一的"四臂"是 RL 的四个起点臂**（`LATE_CATCHUP_FORENSICS_zh.md:15-19`：
  `start(base)` / `wd03_a10` / `nom_a5` / `newmt_a10`），而那里报的相关系数是**正的**：
  `:45` "窗口级 corr(Δ截断, ΔFCS)=**+0.29**（16 个臂×窗）"。

**但"不要注入"这个结论本身仍然成立，靠的是另外三条证据：**
- `SFT_RL_SUMMARY_zh.md:1086`（§2.13(e)）：`soup_innnew_a10` 在 RL 中 "Alternatively" 每千词
  0.015→**3.004（200×）**、重复行 6.2%→30.2%、type-token 0.424→0.230，人读是无尽的
  "But… Alternatively… However…" 审议循环、跑到 32k、**得 0 分**。
- `MODEL_FAILURE_ROOTCAUSE_zh.md:1.2`：被截断输出平均含 **87 个 'Wait'（最多 177）**，其余样本仅 1 个。
- §2.14 下一步第 1 条还点名 `presence_penalty=1.5` "在 32k 推理流上会诱发重复阶梯"。
- 以及我们成文的方法论禁令（`tools/derewrite_workflow.js` 的 CRITICAL GUARDRAIL、
  `DATA_REMEDIATION_zh.md:102` 的"模板做不到"）。

**⭐ 而 `CASE_STUDY_zh.md` §9.3 给出了本文最重要的一条串联**：
> "**"Wait" 是分布外放大 100×，不是数据教的**：数据 think 里 Wait 仅 0.21/篇，退化输出 21.34/篇。
> 数据示范的是"中等长度、总能收敛的研究推导 + 探索腔（Actually 3.17/篇）"，
> 但**从不示范在时限/token 预算下如何收尾**；**一旦没有 commit 纪律拽着，探索腔自激成死循环。**"

→ **回溯词爆炸的根因不是"数据里回溯太多"，是"数据里没有收尾示范"。**
这把"不要注入回溯词"和 ④ 的 commit-and-ship 缺口接成了同一条因果链：
**正确的干预不是压回溯，是补交卷**（正是 ①的回填 + ②的 wave3 切片要做的事）。

**但必须区分两件事，否则会把 ② 一起否掉：**
- ❌ **合成注入回溯词** —— 禁，同意。
- ✅ **数据源自带的真回溯** —— 这正是我们 07-14 规格要的：
  `DATA_RECOMMENDATIONS_zh.md:20` P1 修法逐字：
  "**凡是「把已知论文/解法重写成第一人称现场推导」的 think 全部停用；换 roll-out 式真实探索 think
  ——用 hardcp 拒绝采样流水线在 innovation 题上滚模型自己的、带真实「Wait/verify/反例回溯」的 think，
  只留过验证的。硬指标：自查标记密度 ≥2/1k token**"。
  参照系：base 自然 4.93/千 tok、wave2-cp rollout 7.45、我们本地 innovation 数据 **0.13**。

**判断：成立。但"回溯词与分数负相关"是在 RL 退化臂的输出上测的（长度/退化混杂），
不能反推成"训练数据里回溯少更好"——按这个逻辑，我们本地 0.13/千 tok 的数据应该是全场最好的，
实测它是全场最塌的（裸 SFT FCS ≈0.2）。**

### 6.3 "不要删创新内容 —— base 得 0 的 102 道难题上创新臂是正向的，亏损全在执行侧"

**"亏损全在执行侧"—— 强成立，是我们的核心定论。**
- `MODEL_FAILURE_ROOTCAUSE_zh.md` 一句话结论：SFT 教会了"研究叙事的表演"，
  没教会"可运行的交付纪律"；R1 全参 SFT 覆写代码电路、R3 落点是 Python class 不是 C++ stdin。
- `SFT_RL_SUMMARY_zh.md:711`（§2.7）："创新 SFT 没有拓宽模型的算法搜索空间，它注入的是
  (a) 一个能穿越 RL 存活的 method-designer 倾向，和 (b) **一笔执行纪律税**"。
- §2.13(d)："+6~+7.7 的 Research 增益**全部来自"更多样本产出了可评分的成品"，不是解法变好**"。
- `CAMPAIGN_SUMMARY_zh.md:165`："很多「base=0.0」是 base **想了同样甚至更激进的方法但跑不起来**"。

**但"102 道难题上创新臂是正向的"这半句，本机查无实据，而且最接近的记录是反向的。**
- `102` 在全部 transcript 里都解析成 **`1102 题`（RL 训练集规模）**，不是难题子集。
- 本机真正的"base 得 0"分析是 **5 个 MLS 任务的清单**（`CAMPAIGN_SUMMARY_zh.md:155-165`），
  而它的诚实备注**恰恰把功劳归给执行侧**：创新 base 本来就有，我们赢在"可靠地落地成能跑的代码"。
- 更硬的反向证据：`SOUP_TRADEOFF_zh.md` §7.2 —
  "**没有一例**是「soup 提出 START 提不出的新想法、且正常落地得分」"；
  `CASE_STUDY_clean_wd03_zh.md:214` — "在 top 分差题里**找不到干净的「算法创新」案例**"。
- → 所以"创新臂在难题上是正向的"这个说法，如果指的是**提出了 base 提不出的想法**，
  本机现有证据**不支持**；如果指的是**同样的想法我们更能落地**，那它成立，
  但那正好说明**要补的是交付纪律，不是保护创新内容**。

**但"所以创新内容一点都别动"这个推论要按 benchmark 分开：**
- **发现类（MLS / Research）：创新是正的**，`CAMPAIGN_SUMMARY_zh.md` §5 有 6 个
  "base 崩 → 我们跑通且用了对的结构"的逐题证据（DR-learner +375%、多保真 BO、ANM 噪声不对称…）。
- **FCS/ALE 交付类：创新数据是净负担**，`SFT_RL_SUMMARY_zh.md:213`（§1.8）决定性对照：
  **`pure_a10`（只有 maintenance、一条创新数据都没有）= 6.417**，逼近 `allver` 6.765 和 base 6.816；
  "**掉分的是创新数据，保分的是 maintenance**"。
  `CASE_STUDY_clean_wd03_zh.md:227`："**FCS 奖励纪律而非发散**"。
- 而且 §6.7 有一条必须记住的诚实结论：在 top 分差题里**找不到干净的"算法创新"案例**，
  增益是"交付纪律"；两个满分被查出是 evaluator artifact（`fused_linear_jsd` 显存复用、
  `qknorm` 逐字节交回题面 baseline）。

**判断：成立，但推论要限定。** 正确的表述是
"**不要为了修交付而删创新内容——两者是可加的，缺的是交付纪律的示范，不是创新内容太多**"。
这恰好就是 ① + ② + ④(a) 要做的事。

---

## 7. 建议的落地顺序（按每 GPU 小时收益排序）

| # | 动作 | 成本 | 依据 |
|---|---|---|---|
| **1** | **data_v4 分段回填**（§2.5）：从 `23ff22f29` 回填 346 个单元的边界/re-trace/oracle/交卷段，不回填开头/recap/口癖 | 中（一次 workflow + 三道闸） | 一次解决 ① 和 ④；长度只到 4.4k tok，零截断风险；这是我们自己违反自己审计处方的地方 |
| **2** | **wave3 expert-CP 切片**（§3.4）：≤30k tok + 以代码收尾 + 自查 ≥2/千 tok，约 900–1,000 条，1× 剂量做两臂 A/B | 低（数据已在盘上） | 全语料唯一同时满足"长 + 真自查 + 会交卷"的切片；wave3 从未被干净地单独测过 |
| **3** | **data_v4 去模板化补完**（§4.4）：`deliberately` 81 处题目特异性改写 + 25 处逐字口号改写 | 低 | 审计原本给 v4 开的处方，一直只做了一半 |
| **4** | 若训练侧确有 `capped16k` / `r3_minthink` 的读数，回填进 `SFT_RL_SUMMARY_zh.md` 并给出 job id | 零 | 本机记录里 minthink 从未评测、capped16k 零命中，这个空白会一直误导双方 |

**三条一定不要做**：
1. ❌ 按 `ctl_old` vs `gated_v2` 的 diff 恢复（跨 1348 commit + wave2 换血，不可归因）；
2. ❌ 在 methods/trajectories 上无差别恢复"验证语言"（78–83% 从不触发，且夹带 hindsight，gate 检测不出）；
3. ❌ 注入模板化桥接句（已做过、已被判为污染、且有真段落可回填）。

---

## 附：本文所有统计的复算方式

- 历史版本对比用 `git ls-tree -r <rev> --name-only` + `git cat-file --batch` 读 blob，**不落盘、不改工作树**。
  基线 revs：`23ff22f29`（07-08）、`cc6172bf6`（07-16）、`d0efa74d6`（07-21）、`fc190a4d8`（07-22）、
  `cdf4a5ae9`（07-26）、`HEAD`；deepen 前后 `e7d61f181` / `9819ba7a2`。
- 语料侧统计直接解析 `sft/*.jsonl` 的 `conversations`，按 `</think>` 切 think / 答案通道，
  只统计 assistant（`from == "gpt"`）侧。
- expert-CP 切片判据：`system` 以 `"It is now year 2025. You are an expert competitive programmer"`
  （`innovation_sft.jsonl`）或 `"You are an expert competitive programmer"`（wave2/wave3）开头。
- token 估算用 `chars / 4`（未跑真 tokenizer，比较是同口径的相对量）。
