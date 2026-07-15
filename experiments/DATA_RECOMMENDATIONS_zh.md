# 数据侧建议书:innovation / maintain(roll-out)数据怎么修(2026-07-14)

> 面向数据侧同事的**可执行**建议,证据全部来自三份法证/分析文档,不重复其推导,只给结论+出处:
> - `SFT_DATA_FULL_FORENSICS_zh.md`(下称 **FOR**):数据/训练/生成三层全量法证;
> - `AVERAGE_INNOVATION_zh.md`(**AVG**):α 曲线 + 直测塌陷分桶(83.2% 真编译错);
> - `AVG_VS_SFT_INNOVATION_zh.md`(**AVS**):意图 vs 兑现的控制变量研究。
> 本文新增分析:§2(wave2 query 分布 vs FrontierCS 匹配度,全部本次实测)与 §4(maintain 臂为何常反而更差)。
> 数据文件:`LF-innov/data/innovation_clean_decontam_traj.jsonl`(2225,下称 **traj**)、`innovation_wave2_clean.jsonl`(1352,下称 **wave2**)。行号均指这两个文件。

---

## 1. 现有 innovation 数据(traj)的问题清单

按严重度排序;每条给证据、严重度、修法。

### P1(最高):think 是"重演式蒸馏"的光滑叙述——短 5 倍、零自查,直接替换掉 base 的推理人格

- **证据**:loss 轮 think p50 3,466 token vs base 在 FCS 上自然思考 19,543 token;自查标记密度 0.13/1k(researcher 层中位 **0.00**)vs base 4.93、wave2-cp rollout 7.45(FOR §2.1)。35.9% 的 think 用 3 个固定开头模板("**Reading the problem and pinning the contract.**"×288 等,FOR §2.2)。生成端后果已实证:直测 think 中位塌到 4.4k 字符、"Wait"密度 16.13→0.71/万字符(AVG §2.2),83.2% 零分是真编译错(AVG §3)——**学到形,丢了实**(FOR §5.3/§5.5)。
- **严重度:决定性**。这是与"FCS 塌陷"对应最直接的数据属性;语言落点(P2)只是放大器(nom 直测 67% 答案仍带 cpp fence——模型知道要写 C++,是写不对,FOR §7.2)。
- **修法(重造规格,详见 §3-R1)**:凡是"把已知论文/解法重写成第一人称现场推导"的 think 全部停用;换 **roll-out 式真实探索 think**——用 hardcp 拒绝采样流水线(`tools/hardcp_rollout.py`)在 innovation 题上滚模型自己的、带真实"Wait/verify/反例回溯"的 think,只留过验证的。硬指标:自查标记密度 ≥2/1k token、think 长度分布与 base 自然分布重叠(p50 15k–25k token)、开头模板 top1 占比 <5%。

### P2:79.5% 最终答案是 Python,而 FCS 只判"单文件 C++ 读 stdin"

- **证据**:traj 答案语言 python 1769 / cpp 416 / 其他 40(FOR §2.1);其中 cpp 全部来自 346 条 v4 竞赛集 + 70 条 researcher。此问题 r2 轮已诊断过(`DATA_FIX_FCS_LANDING_zh.md`:"98% 落点 Python"),clean 轮只修到 20%。
- **严重度:高**(与 P1 交互:C++ 样本恰好也是短 think 低自查的 v4 模板腔)。
- **修法**:innovation 题里所有"算法/优化/数值"类落点改造为 C++ 交付(或双语言各留一份);research/ML 类可保留 Python 但把占比压到 ≤50%,并给 C++ 落点的 open-ended 优化题(对齐 FCS 启发式题型,见 §2)。

### P3:错误/带毒样本(样本级处置清单)

全部逐条精读核实(FOR §2.4/§6):

| 行号(traj) | 问题 | 处置 |
|---|---|---|
| 706 | "Picard-Fuchs 验证"是恒真式表演;断言 q<z 与其自身代码输出相反(实跑验证) | **删** |
| 1306 | n≤2e5/TL2s 交付 O(n²logn) 必 TLE 解,并自辩 "fine for the intended scale" | **删或重造**(教"不达标也交付") |
| 1250 | 评测工件入答案:`// ale-49:…`、`getenv("ALE_BASELINE")`;prompt 直接点名答案方法 | **删**(工件=快捷通道教学) |
| 1608 | 注释与实现矛盾(argmin 配 "highest variance" 注释);方法与回喂标签名实不符 | 重造 |
| 1864 | gpt 轮内 "Claude/Opus"、"AutoEvolver" 模型自指;record 数值前后轮不一致 | 清洗自指段落或删 |
| 全体 v4(346) | "两幕 bug 剧"固定剧本(deliberately/convinced myself 句式复用,S08-S11 同构) | 保留但**去模板化**(§3-R1 规格重滚) |
| 多轮 ladder(512) | 最后轮"改进"从未回喂验证——训练目标是"提出下一个方法"而非"提出被证实更好的方法" | 重造时补一轮真实回喂,或至少把最终轮限定为已验证方案 |
| 年份人格穿帮(成批) | 2011 引 2020 论文、2016 讲 Muon 等 | 自动 lint(见 §3-L)后逐条修 |

- 另:**14.5% 的 loss 字符被双训**(166 个题的嵌套前缀切片,同一轮内容在两个切片里都 loss:true,FOR §2.2/§2.3)。修法:同族切片中每个轮只在一个切片里 loss:true(其余置 false),或直接去掉冗余切片。

### P4:wave2(maintain)自身的脏样本

| 行号(wave2) | 问题 | 处置 |
|---|---|---|
| **1198、1276** | **57k token 退化循环样本**(371k 字符只有 14 个唯一词),loss:true 且被 cutoff 53760 右截断(无 EOS 收尾)——模型在 ~50k token 的复读上训练 | **必删**(c2 的 maint2x/4x 会把它们训 2/4 遍) |
| 1256 | 半退化(45.6k token,think 自白 "This is insane. I'll stop generating random long words.") | 删 |
| 866 | think 尾部 ~30 行 "Done./Proceeds." 循环填充 | 清洗尾部 |
| 1351、1352 | cp system(C++17 stdin/stdout)与任务(Python/Triton)错位;残留 "CUDA is unavailable in this container" | 修 system 或删 |
| 441 | 答案尾残留支架 token `<model_answer>`——已实证泄漏进 full 直测生成(FOR §5.2) | 清洗(全量 grep 支架 token) |
| 41 条超 cutoff(**其中 30 条是 cp**) | 训练时最终答案+EOS 被右截断:**10% 的宝贵 C++ 长思考 rollout 在训练里教"想到一半戛然而止"** | 行号:6,8,14,16,18,25,27,49,50,52,54,55,59,61,67,69,71,74,92,106,107,111,152,153,160,207,220,283,287,292,308,459,614,973,975,980,987,1012,1031,1198,1276。处置:删除、或缩 think、或把 cutoff 提到 ≥61k(代价:显存) |
| 口癖单一 | "Here's a thinking process" 开头 483 轮(35.7%)——已实证迁移进 full/newmt 生成(FOR §5.1) | 重滚或改写开头,top1 开头占比压到 <5% |
| 源头 IFEval 约束不可满足 | 三条退化循环全部源自"每词 ≥15 字符"式不可满足约束 | worklist 里剔除不可满足约束题,验证器加"答案退化度"检查(unique-word ratio) |

### P5:自动 lint 规则(一次写好,长期用;见 §3-L)

工件 regex(`getenv\(|ALE_BASELINE|<model_answer>|<\|user_input|// ale-\d+`)、模型自指(`Claude|Opus|GPT-|Gemini|as an AI`)、年份一致性(system 年份 vs 文内引用年份)、退化检测(滑窗 unique-token ratio <0.15 报警)、超 cutoff 检测(tokenizer 实测)、think 自查密度下限、开头模板 top-k 占比、答案语言与判分器匹配。

---

## 2. 【新分析】rollout(wave2)query 分布 vs FrontierCS:匹配度定量

方法:FCS 172 题题面取自 `FrontierSmith/data/frontiercs/full.parquet`(剥掉统一 instruction 头);wave2 按 system 分三层(cp 303 / math 263 / empty 786);tokenizer=训练同款,风格标记用 regex(交互/评分/输入输出节/大边界/时限/样例/最优化措辞)。traj 的 v4-346 一并对照。

### 2.1 主表

| | FCS 172 | wave2-cp 303 | wave2-math 263 | wave2-empty 786 | traj-v4 346 |
|---|---|---|---|---|---|
| 题面 token p10/p50/p90/max | 376/842/**1721**/8484 | 408/655/1012/1619 | 33/57/103/209 | 126/422/2572/3947 | 1041/1240/1838/2815 |
| **交互题**(interactive/flush) | **38%** | **0%** | 0% | 0% | 1% |
| **评分/部分分措辞**(your score/relative score/partial) | **86%**(强口径 46%) | **2%** | 0% | 0% | 16% |
| Input/Output 节 | 85%/72% | 97%/97% | 0% | 12% | 100%/22% |
| 大边界(1e5–1e9) | 38% | 52% | 0% | 0% | 85% |
| 时限声明 | 65% | 93% | 0% | 0% | 99% |
| 最优化措辞(min/maximize) | **36%** | 6% | 0% | 0% | 38% |
| 语言 | 全英 | 全英 | 全英 | 英 749、西/葡 27、中 5、俄 5 | 全英 |

### 2.2 结论:query 分布**仍然不匹配**,缺口集中在 FCS 的两大定义性题型

1. **交互题零覆盖**:FCS 38% 是交互协议题(读 response、fflush、次数限制),wave2-cp **一道都没有**,v4 也只有 1%。生成端可见后果:直测 nom 的短答案里出现 "I'll solve this interactive problem step by step" 式套话开头(3+3 行)而代码不做协议交互(FOR 生成精读)。**这是当前最大的 query 缺口。**
2. **部分分/启发式优化题近乎零覆盖**:FCS 86% 题面含评分措辞(强口径 46% 明确 relative/partial score),36% 是 min/maximize 开放优化;wave2-cp 是清一色 exact-judge Codeforces(评分措辞 2%、最优化 6%)。**FCS 的"启发式接管晚期梯度"现象(RL 塌陷 3 号机制)恰恰说明模型没被教过怎么在部分分题上稳健拿分。**
3. **长题面尾部缺失**:FCS p90=1721、max=8484 token;wave2-cp p90=1012、max=1619——**FCS 最长 10% 的题(往往最难、上下文最重)在 rollout 分布里没有代表**。v4 在长度上反而更接近(p90 1838)。
4. **math/puzzle/IF/多语种 chat(1049/1352 条)与 FCS 无任何形态交集**——它们是"通用能力 maintain",不是"FCS maintain"。真正对口 FCS 的只有 303 条 cp(22% 条数),再扣掉 30 条截断、2 条退化,**有效对口样本 271 条**。

### 2.3 rollout 的模型/预算与 think 长度来源(元数据实查)

- **引擎**(`innovation_prior/tools/hardcp_rollout.py` + `DATA_WAVE2_FCS_CPP_zh.md`):本地 vLLM **Qwen3.6-27B**(3 个 TP=2 副本,query 按 crc32 固定副本吃 prefix cache),**预算 4→8→16→…→1024 翻倍拒绝采样,首个过验证即停**,难度标签=首过样本数;硬尾巴走 Poe qwen3.7-max / DeepSeek(tier-2);验证=编译+测例/answer-match/官方 reward。temperature 采样 top_p 0.95。
- **think 36.7k(实测 cp 层 think token p50=34,723)vs base 自然 19.5k 的差异来源**(推断,三因素):(a) roll 模型是 **27B**,自然思考长于 9B;(b) worklist **预筛 hard**(27B 都要多次采样才过);(c) **幸存者偏差**——只保留过验证的那条,难题上过验证的往往是想得最长的。
- **由此的一个未被注意的训练-评测错位(新发现)**:**55%(166/303)的 wave2-cp 示范 think 超过评测 32k cap**(p50 34.7k > 32,768)。即 maintain 臂在教 9B "像 27B 一样想 35k token",而评测预算只有 32k——full/newmt 直测 37% 撞 cap、soup 后 63-65% 撞 cap 的行为端数字与此一致(FOR §5.0)。**建议:rollout 时把 max_tokens 卡在评测 cap 之下(如 28k),或者对超长样本做"压缩重滚"(同一题限预算重采),否则 maintain 数据的长思考成分自带截断税。**

---

## 3. 修复优先级路线图

| 优先级 | 动作 | 预期影响(证据链) | 成本 |
|---|---|---|---|
| **R0(立刻,零重训)** | 删 P3/P4 表中"必删"样本(traj 5 条 + wave2 44 条含 41 截断行);全量 lint 支架 token/工件 | 消掉退化循环双训与支架泄漏(FOR §5.2 已实证泄漏);对分数影响小但是卫生底线 | 半天(脚本已可写,§1-P5 regex) |
| **R1(核心)** | **innovation think 全量换血**:停用重演式蒸馏;用 hardcp 拒绝采样在 innovation 题上滚 9B/27B 自己的过验证长推理(规格:自查密度 ≥2/1k、p50 15-25k tok、开头 top1<5%、C++ 落点、含真实验证回喂轮) | 直指塌陷第一因(P1)。旁证:wave2-cp(正是这种数据)是全语料唯一保留 base 风格的成分,而 full 臂靠它直测 2.42 vs nom 0.31(FOR §7.1-2);35B 上 clean 数据 LoRA 已能 FCS 净涨(CLEAN §5),数据修好后 9B 全参有望跟上 | 大(GPU 滚采+验证器;流水线现成) |
| **R2** | **补 FCS 对口 query**:交互题(38% 缺口)+ 部分分/启发式优化题(46-86% 缺口)+ 长题面尾部;来源=FrontierSmith synth wave2b 造题(已有 659 seed 流水线)或公开交互题改造;每类 ≥100 题过验证 rollout | §2.2 三个缺口;启发式题还同时服务 RL(奖励梯度更密,rl-degradation 复盘) | 中(造题+判题器) |
| **R3** | **rollout 长度治理**:max_tokens ≤ 评测 cap−4k;超长样本压缩重滚;cutoff 内零截断(lint 强制) | 消"截断税"(§2.3;30/303 cp 被截 + 55% 超 cap) | 小 |
| **R4** | 去重与 loss 记账:嵌套切片去双训(14.5% loss 字符);wave2 开头去模板化 | 防过拟合单一口癖(口癖迁移已实证 FOR §5.1);影响中等 | 小 |
| **R5(实验设计卫生)** | 之后所有 A/B:固定代码版+GPU 数+GBS;NEFT 在长序列无效不要再排(FOR §3.2);maintain 配比按 **loss token** 报(条数 38% = token 55%) | 让下一轮结论可判读 | 零 |

---

## 4. 为什么"带 maintain 的 ckpt 很多时候反而不如不带的"?

> 本节证据 = summary.json 复核 + **同题生成头对头精读**(soup 级 nom/full/newmt a10 各 860 FCS 行;RL 四臂 s5/s20 FCS+Research;去重后统计;精读 10+ 题,引文原文)。

### 4.1 现象先摆全(实测复核;**它不是单向的**)

| 层 | maintain 臂 | nomaintain 臂 | 谁赢 |
|---|---|---|---|
| 直测 FCS | full 2.42;newmt 待出分(11169296 PENDING) | nom 0.31 | **maintain 大胜**(8×) |
| soup FCS(α=0.1/0.05) | full_a10 6.10 / full_a5 6.66;newmt_a10 **5.89**、newmt_a50 4.23 | nom_a10 6.41 / nom_a5 6.58 | **nom 略胜**(a10 差 0.3-0.5) |
| soup Research | full_a10 9.98;**newmt_a10 11.47(最高)** | nom_a10 10.16 / nom_a5 11.1 | **newmt 胜**(注意!) |
| soup ALE/MLS | full_a5 330/0.045 | nom_a5 **398/0.101** | nom 胜 |
| RL FCS(synth@32k,s5→s20) | newmt 6.97→**10.70**(我方臂最高终点) | nom_a5 7.22→9.88 | s5 nom 胜,**s20 newmt 反超** |
| RL ALE(s5) | **newmt 539.7(四臂全场最佳)** | nom_a5 397.8 | **newmt 大胜** |
| RL Research(s5→s20) | newmt 9.43→11.77(headline) | nom_a5 14.45→15.16 | **nom 大胜——但见 4.3-M2 的口径修正** |

准确的问题是:**maintain 帮直测、帮 RL-ALE、帮 RL-FCS 终点、soup-Research 也帮;拖后腿的只有两处——soup 轻档 FCS(小,-0.3~-0.5)与 RL Research(大,表观 -5,修正后 -3~-4)。**

### 4.2 生成端对读:两处"拖后腿"分别是什么样子(本次新读,事实)

**(A)soup FCS 差距 = 100% 截断死亡差,不是能力差。**
- 分解(860 行/臂):newmt 打满 32k 且未闭合 think 的行 **552** vs nom **495**(full 519,同方向);未闭合行均分 ≈0.4。**而在闭合 think 的行上 newmt 反而更强:mean 15.81 vs nom 14.36,%>0 36.0 vs 34.5。**
- 失败样本长相(pi=125,nom 59.6 vs newmt 0):newmt 5/5 打满未闭合,91k 字符只有 1 个 172 字符代码块、全文无 `int main`,尾部死在纯数学推导里("*Wait, if we decompose $R$ into $m$ powers of 2, then $m$ is just popcount$(R)$? No, we can split powers.*");nom 同题同样 4/5 打满,但习惯**早早在 think 里写出完整含 main 的程序**(strip-think + longest-block 抽取可救回,99.33)。全量佐证:全文含 `int main` 的行 nom 442 vs newmt 404。
- 反向案例存在(pi=67/69 newmt 70/74 分 vs nom 22/37):nom 早闭合交付但答案错。**长思考是双刃剑,32k cap 下净效应为负。**
- **wave2 文体签名在 soup/RL 全部归零**("Here's a thinking process"/`\boxed{`/`<answer>`/外语泄漏均 0 命中)——直测层的口癖迁移(FOR §5.1)被 α=0.1 洗掉,**不能**拿来解释 soup/RL 差距。

**(B)RL Research 差距 = 评测口径不对称(~1/5)+ RL 把 newmt 推向"更短、少验证"(其余)。**
- **口径不对称(新发现的评测异常)**:newmt RL5 research 有 **29 行 `ResearchInfraError` 计 0 且从未 resume 重跑**(如 symbolic_regression 5/5 报 "evaluator produced no result");nom/wd03/start 臂都被 resume 重试过(nom 文件 341 行含 21 行重试)。剔 error 后 newmt 10.37 vs nom 14.68——**headline 9.43 vs 14.45 高估了差距约 1/5**;s20 对称条件下差距仍在(12.47 vs 15.86)。
- **真实残差的行为学**:RL 之后 newmt research 生成更短(tok p50 2369 vs 2743;s20 2012 vs 2380)、自检显著更少(**"Wait"/行 0.19 vs 0.89**;soup→RL5 newmt 0.22→0.19 而 nom 0.38→0.89 反升),**先写后不验 → API 细节幻觉一次性致死**:vdb_pareto 题 newmt 在 think 里写 "*Actually, I should double-check the FAISS API version compatibility here.*" 然后并没有查,交付 `self.index.hnsw.ef_search`、`self.index.memory_usage()` 等不存在属性(AttributeError→0);nom 同题 think 里也写错过 API,但**最终答案自纠**为正确的 `efConstruction/efSearch`。llm_router 题 newmt 空 dict 直接嵌套赋值(KeyError 必崩)且无 try/except,nom 把数据加载包进 `except Exception: pass` 带病拿 50.26。另见 newmt 特有的 2-3 行退化复读("SELECT col1 … MATRIX geography地理地理学地理学…"连打)。
- **关键定位**:soup 级 Research newmt 本来是赢的(11.47 vs 10.16)——**恶化发生在 RL 之后,是"RL × maintain-SFT 底座"的交互,不是 maintain 数据在 soup 层的直接毒性**。

### 4.3 机制排序(按证据强度)

**M1|截断税(soup FCS,证据最强)**:55% 的 wave2-cp 示范 think 超过 32k 评测 cap(§2.3)→ maintain 臂保留更重的"推导到底"倾向 → 多 ~60 行截断死亡(4.2-A 的闭合/未闭合分解);full(旧代码、GBS128)与 newmt(新代码、GBS64)同方向,说明**至少部分是数据效应而非工程混杂**。
**M2|评测协议不对称(RL Research,已定量)**:newmt 29 个 infra error 未 resume,~1/5 表观差距是假的。**行动项:引用 RL Research 分前必须核 error 行数并补 resume(eval-resume-error-zero 老坑重现)。**
**M3|RL 交互:maintain 底座在 RL 下滑向"短+不验证"**(修正后仍有 3-4 分):机理候选是 maintain 注入的"想完即收 + 收敛型解题"倾向在 RL 奖励下被放大(wave2 谜题/数学=想完即收的单答案任务占 78% 条数);此句为推断,行为端数字(Wait 密度反向漂移、长度收缩)为事实。
**M4|创新信号稀释(Research 端,推断)**:wave2 按 loss token 占 full 臂 55% 但对模型近乎免费(twostage 首点 loss 0.35,FOR §3.2-3)——同样 1 epoch,maintain 臂注入的创新先验更弱,RL 可放大的就少(AVS §6)。与 M3 相容,现有 run 无法分离。
**M5|对口成分带伤**:30/303 cp 截断 + 2 条退化循环全落在最对口的成分上(§1-P4),小剂量同向加重 M1。

### 4.4 哪些对比**根本不可判**(混杂声明,重要)

- **newmt vs nom/full 叠加 ≥3 个变量**:数据(±wave2)、GBS 128→64(4 卡→2 卡,28→56 步)、新旧 loss-mask 渲染代码(git reflog 实证,FOR §1.6)。newmt 的任何单点名次不能单独给"新代码"或"maintain 数据"记功/记过。
- **α 混杂已被部分排除(新)**:nom_a5(α=0.05)vs newmt_a10(α=0.1)本身不可比;但 **wd03_a10 同为 α=0.1 的无-maintain 臂在 RL5 Research 拿 14.37 ≈ nom 14.68**——Research 差距**不是** α 效应;FCS RL 级各臂差距 <1.5 分,α 影响无法排除。
- **唯一干净的数据 A/B 是 full_wd01 vs nomaintain_wd01**(同码同 4 卡 GBS128 同 α 网格):结论 = 直测 maintain 大胜、soup 轻档 FCS nom 小胜(截断税)、soup 发现类 nom 胜(newmt Research 除外)。
- **"maintain 臂 RL 后全面更差"是错误概括**:ALE 上 RL5 newmt 全场最佳(539.7),FCS s20 newmt 我方最高(10.70)。差在 Research,且有 M2 的口径水分。
- c2 的 maint2x/4x(在训,4 卡 GBS128 与 full 同口径)+ newmt 直测(排队)将补剂量-响应与直测缺口。

### 4.5 给数据侧的落点

maintain 数据的正确角色是**行为锚**(保交付纪律、保长思考),不是"能力数据"。要它不拖后腿:
1. **先治截断税(对应 M1/M5,最便宜、最确定)**:rollout max_tokens ≤ 评测 cap−4k、删 41 条超长行与 2 条退化循环、超长样本压缩重滚(§3-R3);
2. 把"对口锚"做厚(§2.2 的交互/部分分/长题面 C++),把"想完即收"的谜题/IF/多语种减薄到 ≤20% loss token——这同时是对 M3 的对症(减少可被 RL 放大的收敛人格);
3. 开头/风格去模板化(直测层口癖迁移已实证;soup 层虽被洗掉,仍是数据卫生底线);
4. 配比以 **loss token** 计(条数口径低估近一半)并控制在 ~30-40%;
5. **评测侧配套**:RL Research 引用前核 error 行并对称 resume(M2);比较 maintain 剂量时固定代码版/GPU 数/GBS/α(4.4)。

---
*出分后补:newmt 直测(11169296)、maint2x/4x(11169532/33 训练中)。头对头精读的原始抽取件在 scratchpad(quant.py、hh/、fcs/)。*
