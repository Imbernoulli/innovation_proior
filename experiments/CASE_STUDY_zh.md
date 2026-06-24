# 深入 Case Study：模型到底学到了什么？为什么变好/变坏？

> 本文不停留在「比原来好/差」的表面，而是**逐条读真实生成内容**，回答：(1) 什么是 innovation、模型有没有学到；(2) base 为什么一直长度超限/空输出；(3) 大量 SFT 后为什么变差；(4) soup（权重平均）为什么恢复、偶尔「更好」是不是真的；(5) 好在哪坏在哪（对比表）。
> 所有结论基于 `FrontierSmith/outputs/cc_eval_<tag>_thinking_32k_both_vllm/shard_0/samples.jsonl`（每 tag 200 个 FrontierCS 样本）与训练数据 `LF-innov/data/innovation_method_u.jsonl`。
> **贯穿全篇的统计警告**：shard_0 只覆盖 40 题（完整 ~182），其中 33 题对所有模型全 0，真正承载信号的只有 ~7 题 → 任何「A>B」都由 5–7 题 × 5 样本撑起，**统计支撑极弱**。

---

## 0. 总论（一句话）

**我们的数据没有教会模型「会创新落地」，它教会的是「创新的叙事姿态（register）」。** 纯 SFT 确实把训练数据里「不接受标准套路、追问机制、重构问题、发明新构造」的**取向**学了进去——但学到的是**腔调而非实质**：它把本该 15 行贪心搞定的竞赛题，变成一场「发明 Sidon 构造 / 反向 BFS / 模拟退火」的研究探险,然后编不出、跑不完、输出非法,从起点 ~3.14 塌到 ~0,甚至在优化题上跌成**负分**(主动比什么都不做更差)。

**根因**:训练数据的「落点」是**面向同行的研究叙事 + 论文级参考实现**,而不是「读 stdin / 写 stdout / 过测试用例」的**可执行交付**。模型完美复刻了论文叙事的开头姿态,但竞赛不允许铺陈到写不完。

**这同时解释了一个看似矛盾的双重分离**:同一个被 FCS 打崩的 method-SFT 模型,在 **MLS-Bench(ML 研究类任务)上反而超过起点**——因为 MLS 恰恰是「研究取向」能用上、且不那么苛求「单文件可提交」的地方。

---

## 1. 什么是 innovation：训练数据里的可操作判据

读 `innovation_method_u.jsonl` 10+ 条 gpt 输出（A3C、CT 重建、Cox 回归、Hyperband、PAC 学习、Karatsuba…）+ `innovation_prior` 仓库,数据构造哲学是**把一篇已发表论文从第一性原理「重新发现」一遍**。归纳出的 innovation pattern 跨领域高度一致:

### ✅ 真 innovation 的判据（训练数据原文佐证）

| # | 特征 | 原文（英文照抄） |
|---|---|---|
| 1 | **从痛点出发**,不从答案出发 | A3C: *"I want to be very precise about *why* online deep RL is unstable... I might be able to fix it some other way than replay."* |
| 2 | **机制性追问「标准做法为什么有效」**,拆其隐藏代价 | A3C: *"what did replay actually *do*?... only *works* if off-policy... costs memory and per-step compute, and pushes toward GPUs."* |
| 3 | **重构问题**:标准方案的效果可由另一机制等价获得 | A3C: *"Replay's real job is to make the gradient stream look decorrelated and stationary. The same effect can be obtained by running many workers in parallel."* |
| 4 | **从约束反推新方法**,逐步推导 | A3C n-step: *"a single reward directly enters the target of up to n preceding states... credit propagates n times faster."* |
| 5 | **给出可证明的具体构造**,自带正确性论证 | A3C baseline: *"for a fixed state, E_a[∇log π(a|s)b(s)] = b(s)∇1 = 0"* |
| 6 | **自我证伪/撞墙后修正** | A3C 否定 eligibility traces: *"traces are awkward with momentum-based optimizers... the forward view is cleaner."* |
| 7 | **落点是完整可运行、忠于参考实现的代码** | A3C 答案块 = **351 行完整 Atari 训练 harness** |

### ❌ 只是「听起来像 innovation」的空话（hollow register）

1. **只有研究词汇没有机制**:出现 "the real bottleneck / the crucial insight / NP-hard / Sidon set" 但不拆机制、不给可验证推导。
2. **为重构而重构**:反复 "let me shift tactics completely" 换更花哨的方向,**永不收敛**。
3. **构造无证明/假证明**。
4. **落点不是可运行交付**:编不过/跑不完/输出非法/被 token 截断。
5. **诉诸想象中的权威**:"successful solutions use SA"——凭空捏造经验。

> **关键观察**:训练数据的「落点」(判据 7)是**研究写作 + 论文参考实现**,不是「单题可提交解」。三条样本的答案开头都是面向同行的综述(*"The problem is black-box hyperparameter optimization..."*),后接论文级代码;**没有一条**是「读 stdin、写 stdout、过测试」的竞赛交付。这正是后面所有问题的根。

---

## 2. 模型学到的是「腔调」还是「实质」？——三类证据

读了 START / 纯SFT / soup10 在 FrontierCS 关键题上的**真实输出全文** + 240 条全样本词法统计。**结论:腔调,不是实质。**

### (i) 真 innovation（START 给不出的、正确有用的新思路）：≈ 0 例

在所有 SFT 血统「赢过 START」的题里,没有一例靠「更新颖且正确的算法」赢:
- **p8 骑士巡游**:START 与 soup **同一个 Warnsdorff 贪心**,差别只是 tie-break → variance。
- **p42**:START 算法没问题但**代码编不过**(`int(&)[10]` 形参传 `int[10][10]`);soup 赢只因**代码能跑** → robustness 非 innovation。
- **p47**:最像洞察的一例,soup 正确证明「按脏度排序对周期均值无效」(*"This value depends ONLY on L and Σd_u. It does NOT depend on the ORDER!"*),但两者落点仍是同一个 DFS spanning-tree tour,差距来自 instance variance。

> **SFT/soup 模型没有在任何地方展现「对得分有用、START 给不出的真创新」。** 它最多把两边都想得到的思路执行得更正确。

### (ii) Hollow register（腔调像研究者、实质空）：纯SFT 的主导模式

**p11**(求两两 XOR 互不相同的子集)纯SFT 五样本全塌成研究腔:

> *"This is related to 'Sidon sets'... it's called an 'XOR-Sidon set' or **'bent Sidon set'**."*（**编造术语**）
> *"This looks like finding a large independent set... an **NP-hard problem**. Let me **shift tactics completely**. What if I use a structured construction..."*
> *"the maximum even-even XOR is (2m-2) XOR (2m), which **equals 2(m-2)^2 + 2(m-1)**..."*（XOR 不可能产生二次式,**假证明**）

而 START 同题直接务实:
> *"Okay, I will implement the greedy solution. It is **the standard solution**."*

**量化(240 条词法)**:纯SFT「务实标记/篇」从 START 的 23.0 暴跌到 5.4(−76%),「研究腔标记/篇」反升;最尖锐的是**纯SFT 内部**——成功样本研究腔 1.21 **低于**失败样本 2.10:**越像研究者越失败**。这就是 hollow register 的定义性证据。

### (iii) 误用的 innovation（新颖但破坏任务）：负分根源

**p41**(ahc011 优化,坏解得**负分**)是 smoking gun。纯SFT 五样本全 **−80**,START 有 +1033,soup 恢复出 +965:
> (纯SFT)*"The real bottleneck isn't the search mechanism... we search backwards from *perfect* states."* / *"We tried other methods (simulated annealing, bidirectional BFS, etc.) but none worked."*（**捏造经验**,代码 BFS lambda 写一半 `// ...` 截断）

对照 soup10 如何恢复——**主动放弃花哨方案,退回简单贪心**:
> (soup10,+965)*"This is essentially **hill climbing**. ... Once we find a full tree, we stop. This is a HUGE optimization."*（也想过反向 BFS/SA,但明确判定不可行而丢弃）

> **核心**:在「坏解得负分」的优化题上,START 和 soup10 都会退回「无聊但正确」的贪心爬山拿正分;纯SFT 五次全部要么过度工程化到写不完,要么退化成 token salad,**一次都没退回安全解**。SFT 把模型推向「雄心重构」,同时**移除了「退回 baseline」的能力**。

---

## 3. Base 模型为什么失败（两种完全不同的机制，必须分开讲）

| tag | 模型 | 平均token | 撞32k上限 | 不闭合`</think>` | score>0 | 空text |
|---|---|---|---|---|---|---|
| **q35_a00 (BASE)** | Qwen3.5-9B-Base | 28399 | **132/200** | 133/200 | 13 | 2 |
| **q3_a00 (BASE)** | Qwen3-8B-Base | 0 | 0 | 200/200 | 0 | **200/200** |
| q35_a100 (INSTRUCT) | Qwen3.5-9B | 17134 | 61/200 | 56/200 | 16 | 7 |
| q3_a100 (INSTRUCT) | Qwen3-8B | 12870 | 0 | 7/200 | 22 | 7 |

### 3.1 q35-base 长度超限 = 缺「收尾纪律」(commit discipline)

132/200 撞 32768 上限的样本里**只有 1 条**闭合 `</think>`。读 12 条撞上限样本,那 28k token 在干三件交替进行的事:
- **(A) 循环自我怀疑**(词频:"wait" 237/240 条、"actually" 238、"alternatively" 235、"i recall" 186)——这是 base 推理的**稳态**。
- **(B) 反复「回忆类似题」但回忆不到落地**:*"Wait, let me think. Actually I've seen a problem: 'Given n, find the maximum size of a set...'"*——想起名字,既不确认也不证伪,转头又换一个。
- **(C) 在玩具样例上算出矛盾却不修正、从头再算 → 死循环**:p10 整条 12.5 万字符都在一个"公式给 3、真值是 1"的 recalc 循环里转七八次,直到 32k 截断,**整条没写任何代码**。

**决定性对照**:instruct 在**完全相同的 6 道题**(p0,1,10,109,110,111)上,29/30 闭合 think 且 29/30 产出代码;哪怕想了 31k token 也会收尾,也会早早 commit(1700 token 就交卷),且真拿分(p10=85.9、p109=65.9)。

> **机制结论**:base 和 instruct 的**推理能力**差别没那么大(base 也能正确理解题、想起对的算法名),差的是一个**行为/纪律**——instruct 被对齐训练教会「推理到某点必须**承诺答案、闭合 `</think>`、产出可提交代码块**」。base 没有这个 stopping/committing 行为,reasoning 没有吸引子,在 32k 被外部硬截断。这就是 **commit discipline 的缺失**,而它是 instruct-tuning 才教的。
> 评分必然为 0 是**机械必然**:没闭合 think → `strip_think` 原样返回 → 抽不到完整 ```cpp 块 → 0,不是运气。

### 3.2 q3-base 全空输出 = harness 的 context-length 配置 bug（不是模型问题）

200/200 frontiercs 记录:`completion_tokens: null`,`error` 全是 HTTP 400 `"maximum context length is 32768 tokens. However, you requested 32768 output tokens..."`。**没有一次生成发生过。** 根因链:
1. harness 按 config 算 `MAX_MODEL_LEN = min(max_position_embeddings, 8900+32768)`。
2. `Qwen3-8B-Base/config.json` 的 `max_position_embeddings=32768` → `MAX_MODEL_LEN=32768`。
3. 但请求又要 `max_tokens=32768` + prompt → 超过 32768 → vLLM 在采样前 **400 拒绝每个请求**。
4. 对照:`Qwen3-8B`(instruct)的 cap 是 40960 → 放得下 → 193/200 正常生成。**同一 8B 家族,instruct checkpoint 的 context cap 比 base 大 8192,就是这 8192 决定了 base 全 400、instruct 全过。**

> **结论**:q3-base 的「空输出」**完全不能解释为模型能力或对齐缺失**,是纯 infra/config 失败——根本没测到 Qwen3-8B-Base。**矩阵里 q3 a00 整列的 0 是 artifact。**

### 3.3 对评测口径的影响

FrontierCS 这种「必须提交可编译 C++」的口径,对**未对齐 base** 在当前 harness 下**本质上不可用**:q35-base 132/200 撞上限被机械判 0;q3-base 200/200 在生成前 400。**任何「训练后比 base 起点提升多少」的叙事,如果起点是这两个数,基本是在和 artifact 比较。** 建议:(a) 修 q3 的 `max_model_len` 重跑;(b) base 起点报告应同时给「撞上限率/400率/闭合 think 率」,而非只报 score;(c) base 更适合用不要求 commit 行为的口径(completion 接口 + few-shot)。

---

## 4. SFT 为什么退化：剂量-响应 + 主病是「过早交错代码」

### 4.1 剂量-响应主表（q35 instruct 线，200 样本/格）

| 配置 | 平均token | %短(<3k) | %撞上限 | %有代码 | 中位代码行 | **FCS均分** | 务实措辞/样本 |
|---|---|---|---|---|---|---|---|
| start | 17134 | 28% | 32% | 80% | 105 | **3.139** | 18.6 |
| soup10 | 17662 | 22% | 35% | 86% | 101 | **2.924** | 17.9 |
| soup20 | 17572 | 26% | 32% | 80% | 100 | **2.348** | 17.4 |
| soup30 | 17268 | 25% | 34% | 80% | 110 | **1.446** | 18.0 |
| soup50 | 13003 | 34% | 24% | 70 | — | **1.313** | 12.2 |
| soup70 | 9626 | 44% | 18% | 89% | 62 | **0.010** | 7.4 |
| pure SFT | 7799 | 47% | 14% | 88% | 65 | **0.015** | 4.1 |

每列随 SFT 占比**单调退化**,且是**阈值式**——soup10–30 几乎完整保留 start 的长度/语域,真正崩塌在 **soup30→soup50 之间(约 40–50% SFT 权重)**:token 17k→13k、中位代码 110→70 行、务实措辞 18→12。

### 4.2 主病不是「不收尾」，而是「过早收尾交错代码」

**纠正一个常见直觉**:纯 SFT 写代码的频率**比 start 还高**(88% vs 80%),但其中 **98% 得 0**,中位代码 65 行 < start 的 105 行。即 SFT 的主病是**「更早、更短地交出一份错代码」**——它把 start 的「反复验证-迭代到对」纪律砍掉了。
- **模式 (i) 写了代码但错(主模式,~80%)**:过度工程化(p9 的 3D `dist_arr`、p0 的 262 行坏 SA、p11 的 `cache[64][64][64]`+7 参 lambda)或简单但有 bug。
- **模式 (ii) 不收尾(次模式,21/200≈10%)**:撞 32k、上百个 "Wait"、整篇无代码(p2 把正确公式推对好几遍却陷在 "circular dependency")。

> **机制**:SFT 让模型(a)更倾向找「巧妙构造/高级算法」而非务实基线,(b)更早停止「验证-修复」循环。代码因此既更短又更错。

---

## 5. soup 为什么恢复 + 「有时更好」是不是真的

### 5.1 恢复机制：权重插值保住收尾纪律

soup10 = 0.9·start + 0.1·SFT。行为上 soup10 ≈ start(均分 2.924 vs 3.139、务实措辞 179 vs 175/200)。**两题级证据极干净**:
- **p2**:soup10 与 start 收敛到**字面同一个算法**(查到根距离、排序、取满足 `d(u,v)+d(v,root)==d(u,root)` 的最深祖先,连 `K=10000` 剪枝都一样),**都得 85.9**;纯 SFT 同题同公式却陷在 "circular dependency" 不写代码得 0。
- **p11**:soup10 与 start 都落到务实随机贪心;soup10 **也想到** SFT 偏爱的 powers-of-2 巧构,但自己否掉:*"Since exact max is hard, we just aim for max greedy."*

> **机制**:SFT 方向与 start 方向在**「分析」上其实一致**(p2/p11 推出的洞见相同),分歧在**「要不要务实收尾」**这一维。0.9 的 start 权重足以在每个「巧妙 vs 务实」岔路投票务实 + 保留收尾纪律;0.1 的 SFT 只够注入一点「品味」,压不垮纪律。**soup 恢复的本质 = 保住执行纪律,创新部分被稀释到「只剩品味、不再致命」。**

### 5.2 「有时更好」≈ 单跑噪声（诚实判断）

- **problem 23 的 19.9 是 5 样本里 1 个的部分分**,那条还自己撞了 32k,soup20/30/50/70 全 0,**不复现** → 1/35 走运。
- **q35 soup10 vs start 逐题**:**赢 5 输 8,全局反而略低**(2.924 < 3.139);每个胜负都是 5 样本里 1 个 0↔非0 翻转,对称,不聚合成净增益。
- **p8 骑士巡游**:start/soup10/soup50 **用同一个 Warnsdorff 算法**,分差纯来自平局打破细节被连续部分分放大——**方法内方差**,非新能力。
- **q3 的 3.322>2.525** 也站不住:soup10 赢 4 输 6,+0.8 靠 1–2 题大摆动撑、被 p2(−17.2)抵消大半。

> **诚实总结**:**稳健可复现的效应 = 恢复**(SFT 砸到 0.015,soup10 拉回 2.9;base 线同样:base-SFT 0.35 → base-soup 2.08/2.20 拉回 base-start 2.08,但 base-soup 连 base 的「啰嗦、撞上限」也一并继承)。**「超过 start」= 单跑噪声**,由极少数题、单样本翻转、连续部分分对「同一解法不同 tiebreak」的放大造成。**不应讲成「soup/RL 放大了能力」**;正确叙述是 soup 撤销了 SFT 造成的损害。

---

## 6. 评测口径错配：FCS↔MLS 的双重分离（最强证据）

把 FCS 和 MLS-Bench 放一起,出现一个**干净的双重分离**:

| 模型 | FCS（奖励"提交简单正确代码"） | MLS（奖励 ML 研究任务） |
|---|---|---|
| q35-inst start | 3.139 | 0.0643（基线） |
| q35-inst **纯 method-SFT** | **0.015（最差）** | **0.0794（最好,>起点）** |
| q35-inst **method-soup10** | **2.924（最好,恢复）** | **0.0538（最差,<起点）** |

- 纯 SFT:FCS 崩、MLS 升;
- soup10:FCS 恢复、MLS 反而被拖到起点以下。

> **解读**:SFT 注入的「研究取向」在 ML 研究任务上**有用**(纯 SFT MLS 反超起点;base 线也是 0.0764→0.0943),在竞赛代码上**有害**(惩罚探索、不落地)。soup 恢复执行纪律(FCS↑)的同时**稀释掉了研究取向**(MLS↓)。这正面支持核心论断:**FCS/ALE 测的是「提交简单正确代码的纪律」,恰是 innovation 取向的反面;用它衡量「创新」会把我们注入的能力误判为纯退化。** 而发现类评测(Theta/TTT)又噪声过大测不出。
> **诚实警告**:MLS 单跑、20 噪声任务,绝对差值小;但方向与上面所有机制一致,且双重分离本身比单点更难用噪声解释。

---

## 7. 好在哪、坏在哪：START vs 纯SFT vs soup10 深入对比表

| 维度 | START | 纯SFT(method) | soup10 |
|---|---|---|---|
| **FCS 均分(mean@5)** | 3.139 | 0.015 | 2.924 |
| **MLS 均分** | 0.0643 | **0.0794** | 0.0538 |
| **思路新颖度(腔调)** | 低,务实 | **高**(Sidon/反向BFS/"real bottleneck") | 中 |
| **思路新颖度(实质有用)** | — | **≈0**(假证明/编造术语) | ≈0(赢点是调参/正确性) |
| **是否务实收尾** | **是**,会退回贪心 | **否**,反复换方向永不收敛 | **是**,主动丢花哨方案 |
| **代码可编译/可跑** | 多数可跑 | **最差**(token salad、未声明符号、写一半截断、甚至拒答) | 接近 START |
| **务实词标记/篇** | 23.0 | **5.4(−76%)** | 22.8(回到 START) |
| **典型失败模式** | 偶尔构造陷阱 | 过度工程+不收尾+假证明+跑题 | 与 START 同类,但更少 |
| **典型成功模式** | 务实贪心一把过 | 几乎无独有成功 | START 务实解 + 偶尔修对 START 的 bug |

**关键诊断(标记密度按得分分组)**:纯SFT 成功样本研究腔 1.21 < 失败样本 2.10(**越像研究者越失败**);START/soup10 成功样本务实标记 >> 失败样本(**越务实越成功**)。

---

## 8. 根因与对后续训练的启示

**根因**:训练数据是**「研究叙事(research narrative)」而非「可执行交付(executable deliverable)」**。
- 训练目标的落点是**面向同行的发现叙事 + 论文级参考实现**(答案开头 *"The problem is black-box hyperparameter optimization..."*,落点 351 行 A3C harness)。它教的是「**如何把一个想法讲成一篇可发表的发现**」。
- 它**从不**教「读一道有时限的题,在 32k token 内产出单文件、能编译、过测试的提交」——没有收尾纪律、没有退回安全 baseline、没有「输出格式即生命线」。
- 于是模型完美复刻**叙事姿态**:追问机制、否定标准、宣布重构、诉诸想象最优解——这正是论文叙事的开头。论文允许铺陈、把代码当附录;竞赛不允许。结果是 token 烧在反复重构、代码写一半被截断、或退化成 token salad、或在优化题上交非法解拿负分。

**可操作启示**:
1. 想让模型「会创新且能落地」,训练数据**落点必须是可执行交付**(单文件、I/O 契约、过测试),而非研究写作 + 论文 harness。
2. 需显式注入「**收尾纪律 / 退回安全 baseline**」的样本:数据全是「撞墙→换更花哨方向→landing the idea」,缺了「花哨方案不收敛→退回朴素正确解→提交」这一课,而这恰是竞赛成功的关键。
3. 评判「创新是否有用」应以**正确性 + 落地**为闸门:本数据形态下「研究腔/篇」与得分**负相关**,说明当前 innovation 信号在可验证竞赛任务上是**净负**;但在 MLS 这类研究任务上是**净正**——评测口径决定结论。
4. soup 只是「撤销 SFT 损害」的工具,不是「叠加创新红利」;真正想要「放大」需要的是改造数据落点 + RL,而非权重平均。

---

## 9. 数据级归因：trace 回 SFT 数据，到底是什么致病

读全部 3592 个被训练的 gpt 段（`innovation_method_u.jsonl` 1201 + `innovation_method_traj_u.jsonl` 1879）+ `build_sft.py` + `paper-to-reasoning` skill，把退化精确归因到数据：

**9.1 落点不是竞赛交付，是论文级参考实现**：答案 99.5% 带代码，但**只有 1.8% 读 stdin**、58% 是 `class` 定义、代码**几乎纯 Python**（FCS 要 C++）；`context` 本身 **91% 是研究问题、仅 6% 像竞赛题**。A3C 那条的答案落点是一个完整的多进程 PyTorch 训练框架（`class ActorCritic`/`SharedAdam`/`mp.Process`），不是"读输入→算→打印"的单文件解。→ 致 88% 写代码但 98% 得 0、代码更短更错。

**9.2 数据从不教"退回 baseline"**：think 里只有 **0.2%（1201 条里 3 条）**以"放弃花哨方案→退回朴素解→提交"结尾；"撞墙→换更花哨方向" 0.40/篇，"退回简单解" ≈0。→ 直接致负分优化题"五次全 −80、一次没退回贪心"。

**9.3 "Wait" 是分布外放大 100×，不是数据教的**：数据 think 里 Wait 仅 0.21/篇，退化输出 21.34/篇。数据示范的是"中等长度、总能收敛的研究推导 + 探索腔（Actually 3.17/篇）"，但**从不示范在时限/token 预算下如何收尾**；一旦没有 commit 纪律拽着，探索腔自激成死循环。

**9.4 数据并不超长**：think p95 ~9.9k，**无一条 >16k**。→ 撞 32k 是分布外涌现，**修复不该截 think，而该注入"收尾/退回"样本**。

**9.5 `build_sft.py` 是放大器（smoking gun）**：训练目标直接是论文叙事的 `train_answer.md`；system prompt 钦定 `"You are a good researcher"`；格式提示**显式要求** *"give your answer in a **narrative, telling tone** rather than a heavily formatted writeup"* —— 明令"讲述腔而非可提交交付"，无一处注入竞赛纪律。

**9.6 这是 reasoning 还是 talking？—— 高质量推导外壳 + 论文反推的本质 = 危险的 talking。** 表面像真 reasoning（97% 带公式、真实因果链），但 `paper-to-reasoning` skill **明令** *"discovering it for the first time... never betray that a finished paper exists"* —— **终点（论文结论）已知，倒推一条"像现场发现"的路径**。指纹：73% think 开头同一套模板、几乎 100% 成功 landing 预定方法、**失败样本不存在**。真假的**可操作判别**：**把结论遮住，这条推导还能不能"决定退回更朴素的方案"？** 只能通向预定漂亮答案的，就是 talking。模型学到的是"制造发现感的腔调"，不是"在不确定性下务实决策的能力"。

**9.7 因果映射（数据属性 → 行为 → 失分）**

| 数据属性 | → 学到的行为 | → 失分模式 |
|---|---|---|
| 91% 研究问题；1.8% 读 stdin；58% class；纯 Python | 把题当"造方法/库" | 88% 写码但 98% 得 0 |
| 答案=论文级实现，口吻钦定 narrative | 追求漂亮叙事，代码当附录 | 代码更短更错 |
| 0.2% 才退回 baseline | 撞墙就换更花哨方向 | 优化题负分 |
| 探索腔 + 从不示范收尾 | 永不收敛的自我怀疑 | Wait 放大 100×、撞 32k |
| "provably/optimal" 2.52/篇 | 复刻假证明/诉诸最优腔调 | hollow register |

**9.8 数据级修复建议**：① 落点改成可提交单文件 + 大比例真竞赛样本（C++、读 stdin）；② 注入"退回 baseline / 收尾"轨迹（当前缺失）；③ 保留失败样本给 reasoning **真实**不确定性（别每条都成功 landing）；④ 改 system / 删 "narrative tone" 提示，换交付导向；⑤ 别截 think（本不长），靠注入 commit 样本治；⑥ hollow register 必须连真证明/测试兜底。

## 10. 你问的「pattern 层面能不能折中」——能调和腔调，合成不出能力

读 p11 上 α 的梯度（创新腔标记 / 务实腔标记 / 得分）：START(61/76, 43.7) → soup20(99/96, 0) → soup30(43/48, 0) → 纯SFT(26/8, 0)。

- **中间比例（soup20）的推理文本确实"两腔皆强"**：既反复探索/重构（创新腔 99>START），又大谈 "safest interpretation / final check"（务实腔 96>START）——读起来像一个既会探索又会谈落地的模型。**所以"pattern 折中"存在，位置约在 soup20–30。**
- **但 pattern 上的"两者兼有" ≠ 结果上的"两者兼得"**：soup20 两腔拉满仍得 0——更长的探索没产出对的解，只是把两套话术叠加。
- **本质**：model-soup 是权重空间线性插值，行为近似落在 START↔SFT **线段**上；线段上的点能让文本**同时显出两种腔调**（叠加），但到不了线段之外那个点——「**提出新颖思路 且 写对落地**」。这是两端都不具备的第三种能力，**权重平均合成不出来**。要拿到它，得让那条线本身往外弯：**改数据落点（让创新收敛到可提交解）+ RL 奖励"新颖且正确"**，再用 soup/RL 在更高前沿上平衡。

### 附：关键路径
- 训练数据:`LF-innov/data/innovation_method_u.jsonl`;数据哲学:`innovation_prior/README.md`、`sft/README.md`;落点样例:`innovation_prior/methods/a3c/results/answer.md`(351 行 harness)
- 模型输出:`FrontierSmith/outputs/cc_eval_{q35_a100,q35_a100_method,q35_a100_method_soupa10,q35_a00,q3_a00}_thinking_32k_both_vllm/shard_0/samples.jsonl`
- 评分逻辑:`FrontierSmith/verl/verl/utils/reward_score/frontiercs.py`(`strip_think` L57)
- harness/context-len 计算:`FrontierSmith/slurm/cc_eval_thinking_both_ailab.sh`(L113-128)
