# 数据修复方案:让 innovation 模型在 FrontierCS 上不输给 start（落点格式 + 收尾纪律）

> 配套：[DATA_REMEDIATION_zh.md](DATA_REMEDIATION_zh.md)、[PIPELINE_FINDINGS_zh.md](PIPELINE_FINDINGS_zh.md)、[CASE_STUDY_zh.md](CASE_STUDY_zh.md)。
> 本文是基于「读真实评测 generations + 真实训练数据」的一次 autopsy 得出的**可执行数据修复清单**，给数据侧同事直接照做。
> 一句话结论：**推理已经很好了（别动），问题在「落点」——method 数据交付的是 Python 库/class，而 FrontierCS 只给「单文件 C++ 读 stdin」打分。**

---

## 0. 背景:我们要解释并修掉的现象

在 FrontierCS（竞赛式：单文件 C++、从 stdin 读、判官只抽 ```cpp 块）上：

- innovation-SFT 模型（method / methodtraj，及其与 start 的 soup）**说得出正确算法**，但**赢不了** plain Qwen3.5-9B-instruct（start）。
- 行为级 autopsy（读真实 generations）发现失败**不是能力问题、也不是推理质量问题**，而是 **disposition / 落点**：
  - (a) 把对的思路**实现得很脆**（例：本该 O(n) 启发式，却写**递归回溯 → TLE**）；
  - (b) **过度硬编码**（`if(n==2)…else if(n==7)…` 一长串写死的常量，而不是通用算法）；
  - (c) 难题上**瞎逛不收口**（把 token 预算烧在"这是哪场比赛的题/editorial 怎么说"，最后没出代码）；
  - (d) **过早停止推理**（样本偏短），不像 start 那样**死磕到一个通用正确解**。

> 注：start 之所以赢，纯粹是它**长样本死磕**到通用正确实现；innovation 模型样本更短、更早 commit 到"有意思但脆"的方案。

---

## 1. 根因（钉在数据属性上，全部实测）

读了 ~15 条 method trace（dsu-on-tree / convex-hull-trick / heavy-light-decomposition / persistent-segment-tree / segment-tree-beats / …）、56 条随机抽样、全量 1233 条落点扫描，以及 ~10 条 + 全量 178 条 v4。

### 1.1 推理本身是好的 —— **不要动它**

- **91% 的 method reasoning 有复杂度 / TLE 意识**（会点名慢解并因 scaling 否掉它）。例 dsu-on-tree：*"重复地对每个节点独立做，代价 ∑size(v)…Θ(n²)，n=10⁵ 太慢"*。
- **de-rewrite 真生效了**：56 条抽样里 80% 有真实验证语言、55% 有产出具体输出的手动 trace、23% 对拍 brute force。这些是**挣来的**验证，不是"guaranteed/is correct"的空断言。例 convex-hull-trick：手算 `dp=[0,5,10,17,10]` 并逐项对照 O(n²) 定义，再"在几千个随机单调实例上与暴力一致"。
- 所以 (a)(b)(c)(d) 这些失败模式**在数据的推理里基本不存在**，是推理时涌现的。**这部分是 MLS/Theta 红利的来源,务必保留。**

### 1.2 真正的病灶 = 落点格式（全量 1233 条实测）

| 落点属性 | method 数据 | FrontierCS 要的 |
|---|---|---|
| 落点是 **C++** | **2 条（0.2%）** | 必须（只抽 ```cpp） |
| 落点是 **Python** | **1216 条（98%）** | 判 0 |
| 落点定义 **class**（库/框架腔） | **640 条（51%）** | 要单文件 solver |
| 落点读 **C++ cin/scanf** | **3 条（0.2%）** | 必须 |

模型学到的是「交付一个可复用的库 class」，即使评测时被迫写 C++，也带着**库作者的习惯**（脆实现、硬编码、早 commit），而不是**竞赛选手的习惯**（单文件、读 stdin、稳、死磕到通用正确）。例 `persistent-segment-tree` 落点：`class RangeKth: … def main(): data=sys.stdin.buffer.read()…`——**对的算法、对的复杂度、验证过，但语言错了、还包成了 class。**

### 1.3 还缺两样

- **退回最简正确解（fallback）**：只有 **17%** 的 method reasoning 出现"花哨方案不稳→退回朴素正确版→交"的决策。这正是 anti-(a)/anti-(b) 的那一手。
- **真正的死胡同 / 非 land 样本**：trace 全是**反推**（终点=已知方法），~100% land 预定方法、**零真实弃局**。所以数据从不示范"在真不确定下如何收口"——不确定性都是表演完再必然收敛。这放大了 (c)。

### 1.4 v4 数据是对的（但太少）

全量 178 条 v4：**178/178 单文件 C++ 读 cin/scanf、0 条 class、84% 显式 TLE/复杂度、100% 手动 trace、100% 找出并改掉真 bug、90% 退回更稳的简单解**，且每条都带 `verify/{sol.cpp,brute.py,gen.py}` 真 oracle。**v4 就是我们要的 disposition 的范本——但它只占混合数据的 ~13%，被 98% 的 Python 落点稀释了。**

---

## 2. 修复清单（按投入产出比排序，给数据侧照做）

### P0 — 改落点格式（**主修复**，最便宜、最高杠杆，直接对着「88% 写了代码但 98% 得 0」）

1. **把竞赛/算法类 method 的落点从「Python class/库」改写成「单文件 C++ 读 stdin」。**
   - 优先 ~111 条推理本就可移植 C++ 的竞赛方法（DP / 图 / 树 / 字符串 / 数论）。
   - 做法：把已验证的 Python `solve()` 转写成一个 `int main()` 从 `cin` 读、`cout` 写；把 `class` 包装拆成自由函数 / struct；在开头补一句 I/O 契约 + `long long` 溢出意识。
   - **⚠️ 关键(易踩坑):不能只换最终落点块，`reasoning.md` 里的代码块也要一起改/删。** SFT target 是 `<think>{reasoning}</think>` + `train_answer`(见 `build_sft.py`)。实测 `persistent-segment-tree` / `segment-tree-beats` / `convex-hull-trick` 的 **`reasoning.md` 本身就含一个 Python `class` 实现块**——如果只翻译末尾 `train_answer` 而把 reasoning 原样保留，样本**仍然在训练模型"先吐一段 Python 库代码、再给 C++"**,落点转换被稀释。**做法:把 reasoning 里的实现代码块也转成 C++(或删掉,只留必要的代码片段)；只「原样保留」推理的散文部分(复杂度感知 + 验证叙事),不是代码。**
   - 目标：把 0.2%-C++ / 51%-class 的画像推向 v4 的画像（~100% C++/stdin、0 class）——画像统计要把 reasoning 里的代码块也算进去,别只看 train_answer。
2. **把 v4 加进 SFT 混合并加权到 ≥15–20%。**
   - **⚠️ 关键(易踩坑):v4 现在根本不在 SFT 混合里(占比 0%,不是 13%)。** 实测:`build_sft.py` 不读 `data_v4/`(grep=0)；`sft/dataset_info_snippet.json` 只注册了 `innovation_sft` + `innovation_maintain`；`build_v4.py` 写的是**独立文件** `sft/innovation_v4_sft.jsonl`,没接进任何训练混合。所以"按现有 config 直接重跑"会让 v4 仍是 0%。
   - **做法:必须先把 v4 接进混合**——把 `innovation_v4_sft.jsonl` 拼进 `build_sft.py` 的输出(或在 `dataset_info` 里注册 `innovation_v4` 为独立 dataset),**然后**把它的采样占比调到 ≥15–20%(原始条数 178/(1233+178)≈13%,要 oversample 才到 15–20%)。它已经全对(178/178 单文件 C++/stdin),是最干净的 off-segment 信号。

### P1 — 补「退回最简正确解」纪律（anti-(a)/anti-(b)）

3. 竞赛类 method 的验证段后，**加一句 fallback 决策**：*"花哨构造 X 是我有把握时会写的，但在输入 Y 上验证发现它在预算内容易写错，所以我交我 trace 过正确的更简单版 Z。"* 对标 v4 的 `cp-noadj-commit` spine（它**用 traced 反例否掉贪心(26 vs 27)**，然后交"我能辩护的 O(n) DP 而不是我刚打破的贪心"）。可用现成 `tools/augment_verify_workflow.js` 扩展规格实现。

### P2 — 注入真正的「非 land / 验证改写终点」样本（anti-(c)、治"表演式不确定"）

4. method 和 v4 目前**都没有**"验证后把终点改成不同/更差/不确定结果"的 trace。**造一小批（目标 ~30–50 条）**：探索的"聪明"想法被验证为**错的 / 预算内证不出**，于是 trace **交朴素 baseline 而不是预定的花哨方法**——即终点是被验证**真正决定**的，不是预定的。v4 的 `fakeproof`/`greedytrap` slug 是合适底座，扩一子集让落点是 **baseline**。

### P3 — 反硬编码对照样本（防御 (b)，即使数据里没有、评测里也涌现了）

5. 加 ~10–15 条**显式反硬编码** v4 trace：题目诱人用小 n 写死常量，trace 点明*"我可以硬编码 n≤7，但隐藏测试到 n=10⁵,所以我推通用递推"*再 trace 通用代码。给模型一个**对比信号**，而不是靠"数据里没有"。

### P4 — 管线放大器（很便宜，DATA_REMEDIATION §A1 已 specced）

6. `sft/build_sft.py`：system prompt 从"You are a good researcher."改成同时承诺交付纪律（*"…你交付一个单文件、自包含、可运行、遵守 I/O 契约的解；想法在预算内不收敛时,退回最简正确解并交付它"*）；格式提示从"narrative, telling tone"改成**要求落点是完整可运行的单文件实现**。只动条件、不动推理内容，零内容风险，和 P0 一起做。

---

## 3. 明确不要做的事

- **不要**把推理"变务实/变短"——长的、复杂度感知的、验证过的推理是 MLS/Theta 红利的来源，也是数据已经做对的部分。
- **不要**回退 de-rewrite——它已经把验证缺口真正补上了（80% 真验证）。
- **不要**靠加大 v4 占比到压过 method 来解决——那会丢 MLS 红利;P0 的"改 method 落点格式"才是主修复，v4 是辅助。

---

## 4. 一句话给同事

**保留推理 + 验证（别动），把竞赛类 method 的落点从 Python-class 改成单文件 C++ 读 stdin（P0，主修复），v4 加权到 ≥15–20%，再补 fallback（P1）+ 真非-land 样本（P2）+ 反硬编码（P3）+ build_sft 交付纪律提示（P4）。** 这套直接对应"模型说得出算法、却交付成库 class、于是 FCS 判 0"的根因。

---

## 附:训练（RL）侧的配套发现（供参考,不属于数据修复)

RL autopsy 表明数据修好后，RL 这边也要配套改（否则放大不了）：reward 对"退化"不敏感（15688 次 `bakery` 死循环和差一点的解都判 0）→ GRPO 把脆弱基座推进重复 loop（soup FCS 2.5→1.9）。RL 修复：reward 加重复/长度惩罚、max_response 20k→12k、KL 0.001→~0.005、rollout.n 8→16、底座选 base solve-rate≥8% + 重复率≈0% 的（start 或高-FCS 低-α soup，别用 a50/raw-sft）。
