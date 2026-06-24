# 数据补救方案：为什么我们的数据让评测变差，以及怎么改

> 配套阅读：[EXPERIMENTS_zh.md](EXPERIMENTS_zh.md)（实验全貌）、[CASE_STUDY_zh.md](CASE_STUDY_zh.md)（行为级归因）、[SOUP_TRADEOFF_zh.md](SOUP_TRADEOFF_zh.md)（权重平均的天花板）。
> 本文不重复那三篇的结论，而是把它们收敛成**一套可执行的数据补救方案**：(1) 为什么我们的 SFT 数据会在竞赛类评测（FrontierCS / ALE）上把模型打崩；(2) 怎么**优化现有数据**；(3) 怎么**针对性地造新数据**。
> 本文所有「数据画像」数字均为在本仓库 `methods/*/results/` 上**重新实测**（脚本：`tools/data_audit.py`），与 CASE_STUDY §9 独立一致。

---

## 0. 一句话

我们的数据教会了模型**「创新的叙事姿态」**，没教会**「把创新收敛成一份可提交、能编译、过测试的交付」**。竞赛类评测（FCS/ALE）奖励的恰恰是后者、惩罚的恰恰是前者，所以同一批数据在 MLS（研究类任务）上**有用**、在 FCS/ALE 上**有害**。补救 = **不动探索段（MLS 红利来源），改/补落点段（加收尾纪律、退回基线、失败样本、可执行 C++ 交付）**。

---

## 1. 为什么会变差：把根因钉到数据属性上

CASE_STUDY 已经从模型输出侧证明「学到的是腔调不是实质」。这里我们从**数据侧**实测，把退化精确归因到 4 条可量化的数据属性。下表每一行都是「数据长什么样 → 模型学到什么 → 在 FCS/ALE 上怎么失分」。

### 1.1 实测数据画像（本仓库 1233 个 `train_answer.md` / 1245 个 `reasoning.md` / 1251 个 `context.md`）

| 指标 | 实测值 | 含义 |
|---|---|---|
| 落点带代码 | **99.5%** | 几乎每条都以代码收尾——形式上像「会落地」 |
| 落点**读 stdin** | **1.9%**（23/1233） | 但几乎没有一条是「读标准输入→算→打印」的竞赛交付 |
| 落点是 **C++** | **0.2%**（2/1233） | FCS 评测器**只抽 ```cpp 块**；我们的数据 98.6% 是 Python |
| 落点定义 `class` | **58.4%** | 落点是「论文级库/框架」（`class Solution`/`class Model`），不是单文件解 |
| `reasoning` 出现「读输入/过测试/单文件/时限」等**落地词** | **2.2%**（28/1245） | 推理过程**几乎从不**把「在时限/格式约束内交一份能跑的解」当成目标 |
| `reasoning` 出现「退回简单解/放弃花哨方案」等**退回词** | **7.5%**（93/1245，宽口径） | 几乎不示范「花哨方案不收敛→退回朴素正确解→提交」 |
| `context` 像竞赛题（stdin/输出/时限/测试点） | **7.9%**（99/1251） | 输入端 90%+ 是开放式研究问题，不是有 I/O 契约的题 |

> 复现：`python tools/data_audit.py`。CASE_STUDY §9 报的是 1.8% 读 stdin / 58% class / 0.2% C++ / 6% 竞赛题；本次在全 `methods/` 上用 `data_audit.py` 独立复测得 1.9% / 58.4% / 0.2% / 7.9%，**完全一致**。结论稳健。

### 1.2 因果链（数据属性 → 行为 → 失分）

| # | 数据属性（实测） | → 学到的行为 | → 在 FCS/ALE 上的失分模式 |
|---|---|---|---|
| A | 落点 = 论文级实现（58% class、98.6% Python、2% stdin） | 把每道题当成「造一个方法/库」 | 88% 写了代码但 **98% 得 0**：抽不到 ```cpp、读不到 stdin、跑不起来 |
| B | 落点口吻钦定 narrative（见 1.3）、`context` 91% 是研究问题 | 追求漂亮叙事、把代码当附录 | 代码更短更错、token 烧在铺陈 |
| C | 推理 0% 提「落地/时限/格式」，从不示范收尾 | 探索腔自激、永不承诺答案 | 撞 32k token 上限、不闭合 `</think>` → 机械判 0 |
| D | 推理几乎不示范「退回基线」 | 撞墙就换更花哨的方向，不退回安全解 | 优化题（ALE/AHC）上交**坏解拿负分**，比什么都不做更差 |
| E | 反推式构造（终点已知）→ **失败样本不存在**、~100% landing | 学到「制造发现感的腔调」而非「不确定下务实决策」 | hollow register：越像研究者越失败（成功样本研究腔 1.21 < 失败样本 2.10） |

### 1.3 三个「放大器」（`build_sft.py` + skill，smoking gun）

退化不只是「内容」问题，构建管线还在**主动放大**它：

1. **system prompt** 钉死研究者身份：`"It is now year {year}. You are a good researcher."`（`sft/build_sft.py:38`）——没有一处提「交付/正确/可运行」。
2. **格式提示**明令叙事腔：`"give your answer in a narrative, telling tone rather than a heavily formatted writeup"`（`sft/build_sft.py:48-50`）——直接训练模型「讲述」而非「交付」。
3. **`paper-to-reasoning` skill** 的构造哲学是**反推**：*"discovering it for the first time... never betray that a finished paper exists"*——**终点（论文方法）已知**，倒推一条「像现场发现」的路径。后果：73% 推理同模板开头、几乎 100% 成功 landing 预定方法、**数据集里没有失败样本**。模型学不到「在真不确定时要不要退回」，因为数据里的「不确定」全是表演。

> **一句话根因**：数据的**落点是「面向同行的研究叙事 + 论文级参考实现」，不是「可执行交付」**；管线还用 system/格式提示和反推式构造把这一取向放大。FCS/ALE 测的就是「可执行交付的纪律」，于是我们注入的能力被它判成纯退化。

---

## 2. 解什么、不解什么（先定边界，避免把红利改没）

关键约束来自双重分离（CASE_STUDY §6 / SOUP §6）：**同一批 method-SFT 在 MLS（研究类）上反超起点、在 FCS/ALE 上崩**。所以：

- **不能**简单「把数据变得务实」——那会把 MLS 红利一起改没（soup 就是这么把 MLS 拖回起点以下的）。
- **要**做的是把每条轨迹的**前半段（探索/研究取向，MLS 红利来源）保留**，**后半段（落点）升级**为「收敛→（必要时）退回基线→可执行交付」，并**补上失败样本**。
- 目标不是在 START↔SFT 线段上找折中（soup 已证明那合成不出「创新且落地」），而是**把数据本身的落点外推**，让创新行为**收敛到可提交解**——即把线段往 off-segment 弯（SOUP §5 的结论）。

下面两节就是这件事的两个抓手：**优化现有数据**、**针对性造新数据**。

---

## 3. 怎么优化现有数据（track A：低成本、动管线 + 动落点）

按「投入产出比」排序，3 个层级。**A1 立即可做、零内容改动；A2 中等、半自动改落点；A3 是数据卫生。**

### A1. 修掉管线放大器（`build_sft.py`）——最高杠杆、最便宜

改 3 处（已在本 PR 实装，见 `sft/build_sft.py` 与 `tools/data_audit.py`）：

1. **system prompt 注入交付纪律**：从 `"You are a good researcher."` 改为同时承诺「研究取向 **且** 交付纪律」——
   `"You are a good researcher. When you write code, you deliver a single, self-contained, runnable solution that respects the I/O contract; when an idea is not converging, you fall back to the simplest correct approach and ship that."`
   这与现有落点（确实以代码收尾）**一致**，不制造 off-policy gap，但把「落地/退回」写进了每条样本的条件。
2. **格式提示去叙事化**：把 `"narrative, telling tone rather than a heavily formatted writeup"` 改为保留「先讲清分析」但**要求落点是完整可运行交付**：
   `"explain the analysis in a flowing, first-person tone, then end on a complete, self-contained, runnable implementation that respects the stated I/O contract."`
   叙事保留在「讲分析」（MLS 红利的语域）；「交付」写进落点要求。
3. **emit 每条样本的标签**（stdin?/cpp?/class?/fallback?/failure?），落到 `sft/_sft_tags.json`，供训练时**控比例**（见 A2/A3）。

> 这一步**不改任何 `reasoning.md`/`answer.md` 内容**，只改条件与统计，几分钟跑完、风险最低，却直接打掉 1.3 的三个放大器里的两个。

### A2.（核心）给现有 reasoning 补一段**真·验证 / 自我 review**——同时治「不够深」「不够长」「代码错」

**这是 track A 的主力，也是「深度/长度」问题最可操作的解释。** 实测（`tools/data_audit.py` 扩展项）：

- **88% 带代码的 `reasoning` 在最后一个代码块之后没有任何验证语言**——写完代码就收尾或只复述一遍思路（dsu-on-tree 那条就是「代码 + 一段 causal chain 复述」，没有在任何输入上 trace、没查 n=1、没查边界）。
- **60% 的 `reasoning` 全篇从不验证**（只有 39.6% 出现 trace/test/check/edge-case 语言）。

这正对应 CASE_STUDY「SFT 砍掉了验证-修复循环 → 代码更短且更错」。所以「不够深/不够长」最可操作的含义 = **推理从不验证自己的产物**：没有在样例上手算、没查边界、没有「等一下，这处会不会越界 / 溢出 / 漏 case」的自我 review。补上一段**真实的验证段**，**同时**修好 bug 率、长度、深度。

**做法（半自动、subagent/workflow 驱动，不是模板）**：对每个 method，让一个 subagent 读它的 `reasoning.md` + `answer.md` 里的**真实代码**，在 `reasoning.md` 末尾续写一段第一人称验证：

1. **在一个具体输入上手动 trace** 最终代码，逐步走变量，确认输出 == 期望；
2. **过一遍边界/退化情形**（空输入、n=1、最大规模、溢出、相等元素、负数…），明说每个怎么处理；
3. **真的找出并改掉至少一个问题**（off-by-one、未初始化、`int` 溢出、边界漏判）——这是「失败样本」的微观版，给数据注入真实的不确定性；
4. 复杂度/契约复核：是否在时限内、是否严格按 I/O 契约输出；
5. 若验证暴露出花哨构造不可靠 → **退回上面那个朴素正确版**并说明理由（接上 A2 的退回纪律）。

> 这必须是**真验证**（subagent 实读代码、能 trace、能发现真 bug），模板做不到，所以它是 workflow/subagent 任务，不是正则替换。实现：`tools/gen_verification_pass/`（规格 + 可扩展 workflow）+ 本 PR 的 demo。这一步把中位 trace 显著拉长（补的验证段本身就有数千 token），且长在「有信息量的验证」上，不是注水。

### A2b. 轻量收尾 coda（语域锚，非主力）

在 A2 的真验证之外，另有一个零成本的「收尾纪律语域锚」：`tools/make_commit_coda.py`（默认 dry-run，写到 `sft/coda/`，`--apply` 才落盘；只对竞赛/优化类方法施加）。它从落点抽最后一个代码块，套一段「自检 + 退回」收尾。**它只注入收尾的语域和决策结构，不做真验证**——当 A2 的 workflow 还没全量跑完时的廉价补丁。

### A3. 数据卫生：去重叙事模板、控比例、保留少量真失败

- **控比例**：用 A1 的标签，让 SFT mix 里「可执行交付样本」占比有下限（建议先做到 ≥15–20%，对标当前 2%）。不是删研究样本，而是**稀释**其相对权重。
- **去模板**：73% 推理同模板开头是反推指纹；对 `reasoning.md` 开头做近重去（或在 SFT 阶段对前 N token 降权），削弱「制造发现感」的统一腔调。
- **保留真失败**：从现有轨迹/agentic 数据里**挑出真正没 land 的片段**（撞墙后没收敛的），保留一小撮作为「探索可以失败、失败就退回」的正样本，对冲「数据集里没有失败样本」。

---

## 4. 怎么针对性造新数据（track B：直接补 off-segment 能力）

这是真正的解药。目标轨迹的**唯一不变量**：**探索一个非平凡想法 → 撞墙/评估风险 → 退回安全基线（永远保留一个合法解）→ 落到一份单文件、读 stdin、能编译、过测试的交付 → 在样例/边界上验证、抓出并改掉 bug 再提交**（把 A2 的验证段直接编进新数据，而不是事后补）。

我们已经有一个**黄金模板**：`trajectories/ale-atcoder-ahc039/`——它就是「bbox 基线 → grid-greedy（**带 rectangle fallback 所以永不低于基线**）→ grid-SA → shinka」的 explore→fallback→land 序列，落点是读 stdin/写 stdout 的真 solver，还有 `feedback` 给真实分。问题只是**这种数据只有 1 条，研究叙事有 1201 条**。所以 track B = **把这个模板规模化**。

> **先厘清评测模型，否则「退回/fallback」会被误解（重要）**：**FrontierCS 是单轮单次**——给题→thinking→**一个** ```cpp 块→判分，`mean@5`/`best@5` 只是 5 个独立单次样本，**没有「提交→拿到 WA→再退回重交」的多轮回合**。所以这里的「退回」**不是**回合级重试，而是**单条 trace 内、在交出唯一答案之前的决策**：先在具体输入/边界上**验证**你打算写的代码；若验证表明花哨想法错了、或你没把握在预算内写对，就**改写成你能验证正确的那个（通常更简单的）方案**，把**它**作为唯一答案交出。纯 SFT 的失败正相反（CASE_STUDY §2-iii）：赌一个雄心勃勃但写坏的构造并交出去，拿 0（ALE 上甚至负分）。**唯一存在真「基线下限」的是 ALE/AHC**——它本身是迭代式优化（keep-best），所以「永不提交坏解、永远保留一个合法解」在那里才字面成立。

### B1. 三类要造的数据（按它们修复的失分模式）

| 类型 | 修复的失分（§1.2） | 落点契约 | 关键不变量 |
|---|---|---|---|
| **B1a 竞赛交付（FCS 风格，单轮）** | A、C、E | 单文件 **C++**，读 stdin，```cpp 块，过测试点 | 单条 trace 内：探索→**在样例/边界上验证**→若花哨方案验证不过/没把握，**改写成能验证正确的简单方案**→ship；显式「时限/格式即生命线」 |
| **B1b 启发式优化（ALE/AHC 风格，迭代）** | D | 单文件 stdin→stdout solver | **永远先有一个合法基线解**，增量改进，**绝不提交坏解**（杜绝负分）——此处的「基线下限」字面成立 |
| **B1c 失败/弃局样本** | E | 可以**不** land 预定方法 | 探索→**验证发现这条路不对/没把握→改交能验证正确的方案**（而非硬 land 一个没验证的花哨解） |

### B2. 原料（仓库内已具备，无需外网）

- **题面**：`trajectories/ale-atcoder-ahc039/00-initial-context.md`（AHC）、`methods/dsu-on-tree/refs/cf_600E_problem.md`（CF600E）、`methods/balkanoi-2011-time-is-money/refs/ojuz_statement.md`（Balkan OI）、`methods/*/refs/`（Putnam 等）。
- **参考解 + 暴力对拍**：`methods/dsu-on-tree/code/{dsu_solution.py,brute.py}`、`methods/balkanoi-2011-time-is-money/code/*`、AHC039 各 rung 的 answer。
- **I/O 契约**：`experiments/scripts/eval/prepare_frontiercs_parquet.py`（FCS：*"Solve in C++. Output ONLY the C++ code wrapped in ```cpp"*）、`frontiercs_research_eval.py`（Research：Python `Solution.solve()`）。
- **格式**：`trajectories/<task>/{meta.json, 0N-<slug>-reasoning.md, 0N-<slug>-answer.md, 0N-feedback.md}`，`build_sft.py` 已能直接 ingest（method 单轮 + trajectory 多轮 full/folded）。

### B3. 生成流程（`tools/gen_commit_trajectory`，本 PR 给规格 + 1 条已验证种子）

每条新轨迹按以下步骤产出，且**代码必须实跑验证**（这正是和现有数据的本质区别——现有数据从不验证落点）：

1. 选一道有「诱人花哨解 vs 朴素安全解」张力的题（competitive / heuristic）。
2. 写 `00-initial-context.md`：题面 + **显式 I/O 契约 + 时限**（让 prompt 像竞赛题，对冲 §1「7.7% 竞赛题」）。
3. rung 1 = **朴素正确基线**（一定能编译、过样例）——这是「永远保留的合法解」。
4. rung 2..k = 探索花哨想法；其中**至少一条 rung 必须 fallback**（明确判定花哨方案在预算内写不对/有风险，退回 rung 1 的解或其安全变体）。
5. 落点是**单文件**（FCS 用 C++ 读 stdin / Research 用 `Solution.solve()`）；每条 answer 的代码**实编译 + 对拍/过样例**，把结果写进 `feedback`。
6. 推理末尾带**真验证段**（同 A2）：trace 样例、查边界、抓 bug 改 bug。
7. 至少留一条 **B1c 失败样本**：land 不成、显式退回基线。
8. 注册进 `trajectories.json`，`build_sft.py` 自动并入。

### B4. FrontierSmith 风格的「造题」（用 subagent 批量产新题）

FrontierSmith/FrontierCS 没有公开题库，但它的**题目结构**是清楚的（见 `experiments/scripts/eval/prepare_frontiercs_parquet.py`、`frontiercs_research_eval.py`）：每题 = **题面 + 明确 I/O 契约 + 约束/时限 + 样例**，算法 track 要 C++ 读 stdin、research track 要 Python `Solution.solve()`。据此**造题**而非抄题（避免 benchmark 污染）：

- 让 **subagent 各自产一道新题**：给定难度/算法标签（DP、贪心、图、数论、几何、字符串、启发式…），产出 `statement.md`（含 I/O 契约 + 约束 + 样例）+ **参考解**（单文件）+ **暴力 oracle** + **小数据生成器**。
- subagent **必须自验**：编译参考解、对拍 oracle（随机小数据 N 组），把通过证据写进 meta；不通过不准交。
- 再对每道题产 explore→fallback→land+verify 轨迹（B3）。
- 关键设计：**spine = debug + self-verify（对 reasoning 和代码都验证）**，不是 fallback——挑带「经典坑」（溢出/off-by-one/边界/贪心陷阱/重复计数）的题，让 trace 自然演出「写代码→trace→抓出真 bug→改→复验边界」。reasoning 要**长且有组织**（≥12k 字符、分阶段 bold 标签、≥2 个真实 debug 片段），train_answer 是**结构化 editorial**。
- 实现：**`tools/gen_v4_workflow.js`**（一条数据一个 subagent，22 算法标签 × 5 坑型 = 110 条；每条 subagent 亲自 g++ 编译 + 暴力对拍 ≥300 组，0 不符才算过；独立 Verify 阶段复检）→ 落到 `data_v4/<slug>/{context,reasoning,train_answer.md, verify/}`。
- 入 SFT：**`sft/build_v4.py`** 把 `data_v4/` 转成 ShareGPT `sft/innovation_v4_sft.jsonl`（system/格式提示走交付+验证纪律，落点单文件 C++ 读 stdin），与现有 SFT 混合。

### B5. 已交付（本次实跑）

- **Flagship 种子**（亲手写 + 实测）：`data_v4/cp-noadj-commit/`——FCS 风格、读 stdin、**C++ 单文件、暴力对拍 0/3000 不符**。spine 是 **debug + self-verify**：先试贪心、trace `[8,9,2,9,9,-2,8,-5]` **抓出贪心反例（26 vs 27）**改用 DP；DP 首版有**经典 in-place 顺序 bug**，trace `[1,1]` 返回非法的 `2` **定位并修复**；再查空/n=1/全负/溢出边界后提交。它是 subagent 的格式+深度范本。
- **110 条新数据**：`tools/gen_v4_workflow.js` 已启动（一条一 subagent，全部 compile+oracle 自验）；产物在 `data_v4/`，过验数计入最终汇报。

### B6. 同步「润色现有数据」（track A2 的 workflow 实装）

`tools/augment_verify_workflow.js`：一条 method 一个 subagent，读其 `reasoning.md`+`answer.md` 的**真实代码**，在末尾**追加**一段有组织的 debug+self-verify（trace 代码、查边界、真找 bug 真修、复验一处推导）——把现有 trace **拉长且加深**，同时降低 bug 率。默认 scope = 111 个竞赛类 method（代码可 trace），`args.slugs` 可指定。这是 §3-A2「真验证」的可扩展实现。

---

## 5. 落地顺序与验证

1. **A1（管线）** 立即合入 → 重建 SFT → 小规模 SFT，验「思考长度/落点语域」是否回升、FCS 是否止跌。**零内容风险，先做。**
2. **B（新数据）** 先产 50–100 条 B1a/B1b + 少量 B1c，混入 SFT（占比按 A3 控到 ≥15%）。
3. **A2（coda）** 作为补充语域锚，混入但不作主力。
4. **评测口径**：FCS/ALE 看 `mean@5` 是否越过起点（而非只「恢复」）；**关键新指标**：落点**可编译率 / 读 stdin 率 / 闭合 `</think>` 率 / 退回基线率**——这些直接对应 §1.2 的失分模式，比单看 score 更早看到信号。
5. **RL**：在「可编译/过判」为硬闸门的前提下，再奖励「与样例解不同的新构造」，把行为推到 off-segment（SOUP §5）。

---

## 6. 一页纸总结

- **变差的原因**：数据落点是研究叙事 + 论文级 Python 库（实测 1.9% 读 stdin、0.2% C++、58% class、推理仅 2.2% 提落地、几乎无退回/无失败样本），管线还用 `"good researcher"` + `"narrative tone"` + 反推式构造放大它。FCS/ALE 奖励的「可执行交付纪律」正是它的反面。
- **优化现有数据**：A1 改 `build_sft.py`（交付纪律 system / 去叙事化格式提示 / emit 标签）；A2 给带代码落点补「收尾+退回」coda；A3 控比例、去模板、留真失败。
- **造新数据**：把 AHC039 那条 explore→fallback→land 黄金模板**规模化**——competitive(C++/stdin) + heuristic(永不交坏解) + 失败样本，**落点实编译实对拍**。
- **不要做**：别把数据「整体务实化」（会改没 MLS 红利）、别靠 soup「合成」创新（线性插值到不了 off-segment）、别截 think（本不长，靠注入收尾样本治）。
