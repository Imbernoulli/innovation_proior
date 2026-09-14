# MLS-21 定量 case study:简洁性、与基线的相似度、以及“分数从哪来”

2026-09-14。数据 = 两次独立的 MLS-21 单次运行(旧跑 2026-09-03 前后,`_p1` 统一口径重跑 2026-09-14),12 臂(9B 八臂 + 4B 四臂)× 21 题 × 2 代 = **504 条轨迹**。所有数字由 `experiments/mls_quant/mls_quant_extract2.py`(抽取)与 `mls_quant_stats.py`(统计)自动产生,逐条明细在 `experiments/mls_quant/flat.csv`,完整表在 `report_tables.md`。本文只解释,不手抄新数。

原 `CASE_STUDY_INNOVATION_9B_zh.md` 的 MLS 部分(§3.4、§4)是人工逐份读 log 得出的定性判断;这里把同样的性质做成可复现的统计量,并且加上原来没查的一项:**每个“分>0”到底是模型自己写的代码得的,还是题目自带模板的分。**

## 0. 一句话结论

1. **RL 之后模型写的 MLS 代码更短、更贴题目自带的基线实现、几乎不再“离开基线”**:9B 三对 RL(SFT 起点)−SFT 合并,与最近基线的 token 相似度 +0.108(45+/20−,Wilcoxon p<0.001,符号检验 p=0.003),novelty −0.112(18+/47−,p<0.001 两检验),LOC −13(22+/42−,p=0.04/0.02);RL-from-base 对 base:LOC −22.5(5+/20−,p<0.001)、novelty −0.153(3+/23−,p=0.001/0.000);4B:rlv5_4b_base−base4b 的 sim_tpl/sim_bl/novelty 三项 12−0,rlv5_4b_ft01mix−4b_ft01mix 为 10−1 / 8−3 / 2−9,全部同向。**这是本文唯一在两代、两族、多对比较里全部同向且 p 值全部过线的结论。**
2. **“贴基线”是得分的最强预测量,“简洁”本身不是**:把 24 次运行在题内排秩、合并 21 题,分数秩与“与最近基线相似度”秩的 Spearman ρ=0.357(p<10⁻⁴),与 novelty 秩 ρ=−0.317;与 LOC 秩只有 ρ=−0.10(p=0.06)。novelty 题内三分位从低到高,非零分比例 70% → 52% → 40%,最终文件不能解析的条数 1 → 9 → 24。原 case study §4.4 的机制判断(“分差来自对照多犯一个错,不是我们多想到一个办法”)在这里得到了数字。
3. **分数里有很大一块是模板自己的分**:504 条里 256 条 score>0,其中 **117 条(46%)被评分的那一行是未改动的模板跑出来的**(agent 先 `test()` 再改;或改坏了 `undo` 回模板;或自己的版本崩溃后评分器回落到模板那次 test 的有效行)。9B 两代合并:base 0.1183 里 0.0745 是模板分,RL-from-SFT 0.1641 里 0.0682 是模板分。把模板提交记 0 后,RL(SFT)−SFT 合并 43+/27−,Wilcoxon p=0.067、符号检验 p=0.072——**RL 的分数优势有一半来自“更会退回模板”,自写代码的优势只到边缘显著。**
4. **旧代的官方分数有串行污染**:27 条旧代运行,`mlsbench score` 实际选中的排行榜行**不是这次运行写的**,而是同名 model 更早一个作业留下的有效行(旧跑同一 tag 分两个作业跑过);16 条分数因此不同。按“只认本次运行的行”重算,旧 9B 均值:base 0.1141→0.1115、ft01mix 0.1425→0.1266、ft03nm 0.1430→0.1271、lo32nm 0.1525→0.1504、**rlv5_base 0.1513→0.1153**、rlv5_ft01mix 不变、**rlv5_ft03nm 0.1760→0.1521**、rlv5_lo32nm 不变;4B 四臂无污染。p1 代 model 名带 `_p1`,无此问题。§1.1 的数字按规则不改,已在 9B 记分卡追加 §10.1 说明。

## 1. 数据与方法

### 1.1 每条轨迹抽什么

对每个(代, 臂, 题):
- **最终文件**:`$D/mlsroot/vendor/workspace/<task>/<exp_name>/<可编辑文件>`(工作区名从 task log 的 `workspace/<task>/<ws>/` 取,再用 agent 目录 `messages.jsonl` 的 `_meta.exp_name` 对上;旧代 6 条 agent 根本没启动——worker 缺 `causallearn`/`deap`——记为无轨迹)。
- **编辑栈回放**:顺序读 `messages.jsonl`,成功 `edit` 压入 `files/step_<n>_<file>` 快照,`undo` 按结果里 `Restored: <file>` 的条数弹栈,`reset` 清空;agent 用 basename/绝对路径写文件名时工具不存快照,这种情况按 `str_replace` 精确匹配复现(83 次成功、7 次不能)。**自检:回放得到的最终文本与工作区文件逐字节相同 495/504,0 条不同,3 条不可校验直接用工作区文件。**
- **模板**:每题取“从未成功编辑可编辑文件”的运行的工作区文件,21 题各自全部一致。
- **被评分的那一版**:复刻 `mlsbench/scoring/evaluate.py` 的选行规则(final 优先、有效行优先、指标最全优先、最新优先),对 504 条复算分数与 `summary.json` 官方分数 **504/504 一致**;再把选中行的时间戳落到 `tokens.jsonl` 的步时间里,定位是第几次 `test()`,取那一刻的栈顶为“被评分代码”。若选中行不在本次运行时间窗内 → 记“串行污染”,并用只含本次运行行的选行结果重算“own-run 分”。
- **基线实现**:`tasks/<task>/config.json` 的 `baselines` → `edits/<name>.edit.py` 的 replace 内容套到模板的可编辑区(部分题基线只替换可编辑区的一段,因此要套在模板上再比)。

### 1.2 指标

只在**可编辑区**上算(模板的 `edit.start–end`;最终/被评分版本用工具结果里最后一个 `[Current file … | editable: A–B]` 标记的范围)。
- 简洁性:LOC(非空非注释行)、圈复杂度(自写 AST 访问器:if/for/while/except/布尔算子/推导式/函数各 +1)、AST 节点数、最大嵌套、数值字面量数、`LOC/基线中位`(该题基线实现 LOC 的中位数)、`<最小基线`(比最短基线还短)。
- 相似度:`sim_tpl` = 可编辑区 token 序列 `difflib` ratio vs 模板区;`kept_tpl` = 模板区非空行被保留的比例;`sim_bl` = 与**最近**基线实现的 token ratio;`ident_bl` = 与最近基线的标识符 Jaccard;`novelty` = 1 − max(sim_tpl, sim_bl)。
- 结果:官方分(21 题口径,缺/崩=0)、own-run 分、是否非零、是否从未 `test()`、是否上下文 40960 溢出崩溃(`ctx400`)、被评分版本是否 = 模板、编辑成功/失败次数、完成 token 数。
- 代码指标只统计**最终文件 ≠ 模板**的运行(模板不算“写了代码”);“两代合并”把两代当两次独立运行。

### 1.3 统计

- 配对:按(题, 代)配对,n≤42/对;三对 RL(SFT 起点)−SFT 合并 n≤126。报告 正/负/平、差的中位、Wilcoxon 符号秩 p、符号检验 p;主结论附逐题分解(§3.4)。
- 题内关联:每题 24 次运行按指标与分数分别排秩并除以 n,合并 21 题算 Spearman。
- 没有任何 CI 只跨过一次 p<0.05 就当结论;两代同向 + 两族同向 + 两个检验同时过线的才写进 §0。

## 2. 两件先要说清的事

### 2.1 旧代官方分数的串行污染

`mlsbench score --model vllm/<tag>` 读的是该 model 名在 `leaderboard.csv` 里**全部**的行;旧跑里 base9b_v2c、lo32nm 等 tag 分别在两个作业(如 13100794 → 13393687)里跑过,选行规则“有效行优先”会在这次运行只留下空 final 行时,回头选到上一个作业那次运行的有效行。命中 27 条(全部旧代),16 条分数不同:

| era | arm | task | 官方分 | 仅本次运行的行 |
|---|---|---|---|---|
| old | base | causal-observational-nonlinear | 0.019 | 0.000 |
| old | base | ml-symbolic-regression | 0.035 | 0.000 |
| old | ft01mix | ml-calibration | 0.052 | 0.000 |
| old | ft01mix | optimization-evolution-strategy | 0.487 | 0.000 |
| old | ft01mix | optimization-hyperparameter-search | 0.328 | 0.303 |
| old | ft01mix | optimization-online-bandit | 0.000 | 0.230 |
| old | ft03nm | causal-observational-nonlinear | 0.215 | 0.000 |
| old | ft03nm | ml-anomaly-detection | 0.502 | 0.383 |
| old | lo32nm | causal-observational-linear-gaussian | 0.018 | 0.023 |
| old | lo32nm | causal-observational-nonlinear | 0.036 | 0.000 |
| old | lo32nm | ml-dimensionality-reduction | 0.013 | 0.000 |
| old | rlv5_base | causal-observational-nonlinear | 0.102 | 0.033 |
| old | rlv5_base | ml-missing-data-imputation | 0.310 | 0.053 |
| old | rlv5_base | ml-selective-deferral | 0.297 | 0.242 |
| old | rlv5_base | mlsys-moe-load-balance | 0.375 | 0.000 |
| old | rlv5_ft03nm | ml-anomaly-detection | 0.502 | 0.000 |

旧 9B 均值(官方 → 仅本运行):base 0.1141→0.1115、ft01mix 0.1425→0.1266、ft03nm 0.1430→0.1271、lo32nm 0.1525→0.1504、rlv5_base 0.1513→**0.1153**、rlv5_ft01mix 0.1572(不变)、rlv5_ft03nm 0.1760→**0.1521**、rlv5_lo32nm 0.1806(不变)。4B 四臂 0 条污染。**修正后旧代的排序仍是:三个 RL(SFT 起点)臂 > 四个非 RL 臂;但 rlv5_base 从“高于全部非 RL”变成只比 base 高 0.004、低于三个 SFT 臂**——这与 p1 代(rlv5_base 0.1167 < base 0.1224)方向一致了。原 case study §3.4 点名的 `rlv5_ft03nm_a20_s20` ml-anomaly-detection 0.5016 就是一条污染行(被评分行来自另一作业,本次运行自己的行无效;0.5016 恰等于该题模板自身的分,见 2.2)。

### 2.2 “分>0”里有多少是模板自己的分

MLS 的模板是能跑的弱实现(如 clustering 模板 = KMeans 壳,自身 0.3875;anomaly 模板 0.5016;evolution-strategy 模板 0.4866;active-learning 模板 0.3617)。agent 常见三种路径让**模板那次 test 的行**成为被评分行:先 `test()` 再改;改坏了 `undo` 回模板再 `submit`;自己的版本 test 崩溃(无指标)时,评分器“有效行优先”回落到模板那次的有效行。

| 族 | 阶段 | n | 模板被评分 | 自写代码 >0 | 均值 | 模板提交记 0 后均值 |
|---|---|---|---|---|---|---|
| 9B | base | 42 | 17 | 9 | 0.1183 | 0.0438 |
| 9B | SFT×3 | 126 | 38 | 38 | 0.1128 | 0.0665 |
| 9B | RL-from-base | 42 | 12 | 16 | 0.1340 | 0.0807 |
| 9B | RL-from-SFT×3 | 126 | 38 | 52 | 0.1641 | 0.0959 |
| 4B | base | 42 | 10 | 2 | 0.0368 | 0.0115 |
| 4B | SFT | 42 | 7 | 4 | 0.0342 | 0.0188 |
| 4B | RL-from-base | 42 | 14 | 10 | 0.1098 | 0.0546 |
| 4B | RL-from-SFT | 42 | 18 | 8 | 0.1301 | 0.0592 |

逐臂逐代的拆分(21 题均值 = 模板提交贡献 + 自写代码贡献;由 `flat.csv` + `runs.json` 的 `scored_is_template` 直接算出):

| era | arm | mean | 模板提交贡献 | 自写代码贡献 | 模板被评分题数 | 自写代码>0 题数 |
|---|---|---|---|---|---|---|
| old | base | 0.1141 | 0.0581 | 0.0560 | 6 | 7 |
| old | ft01mix | 0.1425 | 0.0735 | 0.0689 | 10 | 5 |
| old | ft03nm | 0.1430 | 0.0531 | 0.0899 | 7 | 10 |
| old | lo32nm | 0.1525 | 0.0601 | 0.0924 | 6 | 11 |
| old | rlv5_base | 0.1513 | 0.0432 | 0.1081 | 5 | 10 |
| old | rlv5_ft01mix | 0.1572 | 0.0430 | 0.1142 | 5 | 10 |
| old | rlv5_ft03nm | 0.1760 | 0.0749 | 0.1010 | 5 | 10 |
| old | rlv5_lo32nm | 0.1806 | 0.0949 | 0.0857 | 9 | 6 |
| old | base4b | 0.0304 | 0.0304 | 0.0000 | 4 | 0 |
| old | 4b_ft01mix | 0.0072 | 0.0050 | 0.0021 | 3 | 1 |
| old | rlv5_4b_base | 0.1223 | 0.0712 | 0.0512 | 7 | 5 |
| old | rlv5_4b_ft01mix | 0.1088 | 0.0587 | 0.0501 | 8 | 4 |
| p1 | base | 0.1224 | 0.0908 | 0.0316 | 11 | 2 |
| p1 | ft01mix | 0.1011 | 0.0493 | 0.0518 | 5 | 5 |
| p1 | ft03nm | 0.1036 | 0.0272 | 0.0764 | 3 | 6 |
| p1 | lo32nm | 0.0343 | 0.0148 | 0.0194 | 7 | 1 |
| p1 | rlv5_base | 0.1167 | 0.0634 | 0.0534 | 7 | 6 |
| p1 | rlv5_ft01mix | 0.1588 | 0.0560 | 0.1028 | 4 | 10 |
| p1 | rlv5_ft03nm | 0.1641 | 0.0810 | 0.0831 | 8 | 9 |
| p1 | rlv5_lo32nm | 0.1479 | 0.0590 | 0.0889 | 7 | 7 |
| p1 | base4b | 0.0431 | 0.0201 | 0.0230 | 6 | 2 |
| p1 | 4b_ft01mix | 0.0612 | 0.0257 | 0.0355 | 4 | 3 |
| p1 | rlv5_4b_base | 0.0973 | 0.0393 | 0.0580 | 7 | 5 |
| p1 | rlv5_4b_ft01mix | 0.1514 | 0.0830 | 0.0683 | 10 | 4 |

(旧代的“模板贡献”含污染行:污染行的被评分代码来自另一作业,`scored_is_template` 记 False,因此旧代模板贡献是下界。)

**把模板提交记 0 的配对检验**(官方分口径下 RL(SFT)−SFT 合并是 52+/26−,Wilcoxon p=0.001;Wilcoxon 一律用 scipy 默认方法、`zero_method="wilcox"`,与 §3.4 相同):

| 比较 | n | +/− | Wilcoxon p | 符号 p | mean a | mean b |
|---|---|---|---|---|---|---|
| rlv5_ft01mix − ft01mix | 42 | 18−7 | 0.098 | 0.043 | 0.1085 | 0.0604 |
| rlv5_ft03nm − ft03nm | 42 | 14−11 | 0.778 | 0.690 | 0.0921 | 0.0831 |
| rlv5_lo32nm − lo32nm | 42 | 11−9 | 0.341 | 0.824 | 0.0873 | 0.0559 |
| rlv5_base − base | 42 | 14−4 | 0.102 | 0.031 | 0.0807 | 0.0438 |
| 三对 RL(SFT)−SFT 合并 | 126 | 43−27 | 0.067 | 0.072 | | |
| rlv5_4b_base − base4b | 42 | 9−2 | 0.021 | 0.065 | 0.0546 | 0.0115 |
| rlv5_4b_ft01mix − 4b_ft01mix | 42 | 8−4 | 0.182 | 0.388 | 0.0592 | 0.0188 |

方向仍然全部是 RL 高,但幅度砍半、显著性掉到边缘。**“RL 之后 MLS 分更高”这句话,一半是“RL 之后更会保住模板分”。**

## 3. 定量结果

### 3.1 每臂 21 题里发生了什么(p1 代;旧代同表见 `report_tables.md` §1)

| arm | mean | ctx400 | 从未 test | test 过但 0 | score>0 | 最终==模板 | med tests | med edit ok | med edit err |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.1224 | 3 | 2 | 9 | 10 | 7 | 3 | 2 | 2 |
| ft01mix | 0.1011 | 2 | 3 | 9 | 9 | 5 | 2 | 4 | 2 |
| ft03nm | 0.1036 | 5 | 5 | 8 | 8 | 3 | 2 | 4 | 1 |
| lo32nm | 0.0343 | 3 | 6 | 11 | 4 | 7 | 1 | 4 | 2 |
| rlv5_base | 0.1167 | 0 | 1 | 8 | 12 | 4 | 3 | 3 | 1 |
| rlv5_ft01mix | 0.1588 | 2 | 2 | 6 | 13 | 5 | 3 | 3 | 1 |
| rlv5_ft03nm | 0.1641 | 0 | 1 | 5 | 15 | 8 | 3 | 1 | 1 |
| rlv5_lo32nm | 0.1479 | 3 | 4 | 4 | 13 | 7 | 3 | 2 | 1 |
| base4b | 0.0431 | 1 | 12 | 3 | 6 | 14 | 0 | 0 | 0 |
| 4b_ft01mix | 0.0612 | 0 | 11 | 5 | 5 | 13 | 0 | 0 | 0 |
| rlv5_4b_base | 0.0973 | 0 | 5 | 7 | 9 | 5 | 2 | 1 | 0 |
| rlv5_4b_ft01mix | 0.1514 | 0 | 5 | 4 | 12 | 8 | 1 | 1 | 0 |

“从未 test”= 20 步用完或崩溃前一次 `test()` 都没调(4B 非 RL 臂过半如此);“test 过但 0”= 有指标但崩溃/被钳到最弱基线以下。

### 3.2 简洁性(两代合并;只算最终文件≠模板的运行)

| arm | n | med LOC | med LOC/基线中位 | <最小基线 | med CC | med CC/基线 | med AST | med #num | parse ok |
|---|---|---|---|---|---|---|---|---|---|
| base | 29 | 92 | 1.65 | 3% | 13 | 1.56 | 652 | 27 | 100% |
| ft01mix | 30 | 76 | 1.83 | 3% | 10 | 1.43 | 614 | 26 | 93% |
| ft03nm | 34 | 72 | 1.72 | 9% | 9 | 1.00 | 441 | 18 | 91% |
| lo32nm | 33 | 84 | 1.82 | 9% | 12 | 1.37 | 624 | 22 | 94% |
| rlv5_base | 36 | 54 | 1.36 | 14% | 7 | 1.00 | 396 | 14 | 100% |
| rlv5_ft01mix | 29 | 74 | 1.62 | 0% | 9 | 1.14 | 374 | 12 | 86% |
| rlv5_ft03nm | 27 | 56 | 1.37 | 11% | 6 | 1.00 | 290 | 9 | 85% |
| rlv5_lo32nm | 30 | 61 | 1.49 | 10% | 8 | 1.00 | 408 | 12 | 93% |
| base4b | 17 | 108 | 1.95 | 12% | 20 | 3.35 | 1077 | 40 | 53% |
| 4b_ft01mix | 18 | 114 | 2.43 | 0% | 10 | 1.29 | 649 | 23 | 83% |
| rlv5_4b_base | 30 | 54 | 1.25 | 13% | 4 | 0.80 | 164 | 6 | 93% |
| rlv5_4b_ft01mix | 23 | 59 | 1.47 | 17% | 7 | 0.71 | 304 | 8 | 83% |

四个 RL 9B 臂的中位 LOC 54–74,四个非 RL 臂 72–92;RL 臂圈复杂度中位落到基线中位的 1.0×,非 RL 臂 1.4–1.6×。**所有臂写的代码都比题目自带的基线实现长(LOC/基线中位 >1),RL 只是把这个倍数从 ~1.7 拉到 ~1.4。**两代分开看方向相同(`report_tables.md` §2)。

### 3.3 与模板、与基线的相似度(两代合并)

| arm | n | med sim_tpl | med kept_tpl | med sim_bl | med ident_bl | med novelty | sim_bl>0.6 | novelty>0.5 |
|---|---|---|---|---|---|---|---|---|
| base | 29 | 0.339 | 0.778 | 0.638 | 0.388 | 0.362 | 55% | 34% |
| ft01mix | 30 | 0.395 | 0.619 | 0.585 | 0.299 | 0.406 | 43% | 33% |
| ft03nm | 34 | 0.436 | 0.604 | 0.541 | 0.285 | 0.457 | 41% | 41% |
| lo32nm | 33 | 0.327 | 0.576 | 0.639 | 0.350 | 0.342 | 52% | 36% |
| rlv5_base | 36 | 0.554 | 0.778 | **0.891** | 0.462 | **0.093** | 86% | 11% |
| rlv5_ft01mix | 29 | 0.512 | 0.778 | 0.713 | 0.345 | 0.258 | 59% | 17% |
| rlv5_ft03nm | 27 | 0.566 | 0.778 | 0.831 | 0.360 | 0.132 | 67% | 19% |
| rlv5_lo32nm | 30 | 0.492 | 0.764 | **0.885** | 0.493 | **0.081** | 73% | 7% |
| base4b | 17 | 0.232 | 0.778 | 0.429 | 0.315 | 0.571 | 29% | 59% |
| 4b_ft01mix | 18 | 0.292 | 0.778 | 0.511 | 0.259 | 0.489 | 33% | 39% |
| rlv5_4b_base | 30 | 0.761 | 0.946 | 0.823 | 0.350 | 0.115 | 77% | 3% |
| rlv5_4b_ft01mix | 23 | 0.765 | 0.944 | 0.716 | 0.331 | 0.087 | 70% | 9% |

RL 臂写出的东西有 59–86% 与某个题目自带基线的 token 相似度 >0.6;novelty>0.5(既不像模板也不像任何基线)的比例从非 RL 的 33–41% 掉到 7–19%。4B 上更极端:RL 后 `kept_tpl` 0.94——几乎只在模板上改几行。

### 3.4 配对检验与逐题分解

三对 RL(SFT 起点)−SFT 合并(题×代配对,n≤126):

| metric | n | +/−/= | med diff | Wilcoxon p | sign p |
|---|---|---|---|---|---|
| score(官方) | 126 | 52−26−48 | 0.000 | 0.001 | 0.004 |
| score>0 | 126 | 36−15−75 | 0.000 | 0.003 | 0.005 |
| never tested | 126 | 15−17−94 | 0.000 | 0.724 | 0.860 |
| LOC | 65 | 22−42−1 | −13 | 0.039 | 0.017 |
| CC | 55 | 16−31−8 | −2 | 0.033 | 0.040 |
| sim_tpl | 65 | 40−24−1 | +0.052 | 0.010 | 0.060 |
| sim_bl | 65 | 45−20−0 | **+0.108** | **0.000** | **0.003** |
| novelty | 65 | 18−47−0 | **−0.112** | **0.000** | **0.000** |
| edit errors | 126 | 42−56−28 | 0.000 | 0.056 | 0.189 |
| compl. tokens | 122 | 57−65−0 | −11679 | 0.406 | 0.526 |

其他对(全部在 `report_tables.md` §4):rlv5_base−base:LOC 5−20(p=0.000/0.004)、CC 5−19(0.002/0.007)、sim_bl 22−4(0.005/0.001)、novelty 3−23(0.001/0.000),但 score 17−8(p=0.29/0.11)不显著;三个 SFT−base:score、LOC、sim_bl、novelty 全部不显著(最强的一格 ft01mix CC 3−12 符号 p=0.035,Wilcoxon 0.21);4B:rlv5_4b_base−base4b sim_tpl/sim_bl/novelty 12−0(p<0.001),LOC 1−11,never tested 3−21(p<0.001);rlv5_4b_ft01mix−4b_ft01mix 同向、n 更小。

逐题分解(三对 × 两代 = 每题最多 6 个差):sim_bl 21 题里 15 题正多于负,2 题负多于正(active-learning 1+/3−、anomaly 1+/2−),3 题持平(online-bandit 3+/3−、moe 2+/2−、imputation 1+/1−),1 题无可比对(linear-non-gaussian);novelty 16 题负多、2 题正多、2 题平、1 题缺;score 13 题正多负少、1 题负多正少(online-bandit 2+/3−)、7 题持平或无差。**没有哪道题单独撑起合并结论。**

### 3.5 题内关联:什么性质的代码得分

| 性质(题内秩) | n | Spearman ρ vs 分数秩 | p |
|---|---|---|---|
| sim_bl(与最近基线相似) | 336 | **+0.357** | <10⁻⁴ |
| ident_bl | 336 | +0.347 | <10⁻⁴ |
| novelty | 336 | **−0.317** | <10⁻⁴ |
| #tests | 336 | +0.395 | <10⁻⁴ |
| sim_tpl | 336 | +0.126 | 0.020 |
| compl. tokens | 336 | −0.132 | 0.015 |
| LOC | 336 | −0.102 | 0.061 |
| #num literals | 302 | −0.101 | 0.079 |
| CC / AST / nesting | 302 | −0.06 ~ −0.08 | >0.17 |
| kept_tpl、edit errors | 336 | ≈0 | >0.79 |

按结果分层(最终≠模板的 336 条;三层相加 339,因为 3 条旧代污染行的运行“从未 test 却有分>0”,同时落在两层):score>0 的 184 条 med sim_bl 0.803、novelty 0.165、可解析 95%;test 过但 0 的 103 条 sim_bl 0.561、novelty 0.407;从未 test 的 52 条 sim_bl 0.561、novelty 0.422、**可解析只有 63%**。novelty 题内三分位:低 70% 非零分 / 1 条不可解析,中 52% / 9 条,高 40% / 24 条。

## 4. 对原 case study 三个假设的定量回答

**(a) 跨领域迁移 / 离开基线**:原文 §3.4 手读的结论是“RL 阶段双方都是 0%——每一次有效尝试都是题目列出的某个基线,原样重写或动一个旋钮”。统计量同向且更强:RL 让 sim_bl 升、novelty 降,是本文最稳的效应(§3.4)。**在 MLS 上,RL 训练出的不是“离开基线”,是“更准确地抄基线”。**

**(b) effective-but-simple**:RL 代码确实更短、圈复杂度更低(§3.2、§3.4)。但题内关联显示,**简洁本身对分数几乎没有预测力(LOC ρ=−0.10),有预测力的是“像基线”(ρ=+0.36)和“test 的次数”(ρ=+0.40)**;而“像基线”的代码天然短——基线实现本来就是每题最紧的写法。所以 (b) 在 MLS 上的准确表述是:**RL 之后模型更少自己发明结构,更多复用题目给的结构,因此更短、更少崩;“简单”是“保守”的副产品,不是独立的能力。**这与原文 §4.4“分差来自对照多犯一个错”一致,只是把“少犯错”落到了“少离开基线”这个可测的量上。

**(c) 稳定性**:9B 上 RL 与 SFT 的编辑失败次数(42−56,p=0.06/0.19)和“从未 test”(15−17)没有差别;RL 的稳定性优势只在 4B 上是真的(never tested 3−21,p<0.001;base4b 最终文件只有 53% 可解析,rlv5_4b_base 93% 可解析)。9B 上 RL 的“更稳”体现为**更会退回模板拿保底分**(§2.2),不是更少写错。

## 5. 定性例子(每条都能用 §7 的路径核对)

**A. 同一题、同一 SFT 起点的两端:`optimization-hyperparameter-search`,p1 代。**
- `rlv5_ft03nm_a20_s20`:两步 `edit` 把 `CustomHPOStrategy` 整个换成一份 BOHB(第一步结果 `OK: Replaced 1 occurrence in scikit-learn/custom_hpo.py (lines 255..326). Editable range: 255–422.`),test 两次,`submit` 第二次,得 0.4348。被评分代码 132 LOC,与题目自带 `bohb` 基线的 token 相似度 **0.972**——diff 只有 docstring、注释、四处空值守卫(`if good.size > 0 else 1.0` 两处、`if len(samples) == 0: return 1.0`、候选循环外的 `if good.size > 0: … else: best_cfg = space.sample_uniform(self.rng)`),`self.gamma = 0.15 / self.n_startup = 8 / self.eta = 3` 与基线逐字相同。
- `ft03nm_a20`(它的 SFT 起点):写了一个 206 LOC 的 `class AdaptiveMultiFidelityBayesianOpt`,docstring 开头 `NEW CONTRIBUTION: A novel HPO strategy that achieves better performance` / `than baseline methods (BOHB, DEHB, Hyperband, TPE) through:`(原文在此换行)列了四点;第 3 步在没 test 时 `submit`(`ERROR: No test results yet.`),然后三次 `undo`、再重写,第 8 步 `old_str not found`,随后对话超过 40960 上下文被 vLLM 拒绝(task log:`maximum context length is 40960 tokens. However, you requested 0 output tokens and your prompt contains at least 40961 input tokens`),**一次 test 都没跑**,记 0。novelty 0.55 vs 0.03。

**B. “分>0 = 模板分”:`ml-clustering-algorithm`,p1 代。**
- `rlv5_ft03nm_a20_s20` 得 0.3875。轨迹:第 1 步先 `test()`(模板,`ari_blobs: 0.935391`),第 2 步改 → `WARNING: scikit-learn/custom_clustering.py no longer parses as Python`,第 3 步 `old_str not found`,第 5 步 `undo`,第 7 步再改,第 10 步 test → `[STATUS: FAILED exit=1]`,第 11、14 步两次 `undo` 回到模板,第 16 步 `ERROR: Nothing to undo — the edit history is empty`,第 19 步第三次 test(仍是模板),第 20 步 `submit` 第 3 次。**被评分的是未改动的模板**;0.3875 正是该题模板在所有臂上得到的同一个数。
- `ft03nm_a20`(SFT 起点)得 0:写了 108 LOC 的 `Density-BCL`(`self.hp_alpha = 1.25 … self.hp_zeta = 0.08` 五个自创超参),与最近基线 `kmeans` 相似度 0.39;三次 test 全部 `AttributeError: 'CustomClustering' object has no attribute 'predict'`——重写 `fit` 时把 `predict` 删了,三次都没补回来。
- 同题 `lo32nm_a10` 20 步内一次 test 都没调,最终文件 = 模板,记 0;`rlv5_lo32nm_a10_s20` 也是模板分 0.3875(被评分的是第 1 次 test)。这道题 24 次运行里 15 次非零分:8 次被评分的就是模板;另外 7 次里 6 次的分数也正好是模板的 0.3875(被评分代码与模板/`kmeans` 基线几乎相同,或是 §2.1 的污染行),只有旧代 `rlv5_4b_base_s20` 得到一个不同的数 0.0721。**没有一次自写代码超过模板**——与原文 §3.4 “八臂指标到 16 位有效数字全同”一致。

**C. 新颖 → 不可解析 → 从未 test:`ml-symbolic-regression`,`lo32nm_a10`,p1 代。** 20 步里 11 次成功 `edit`、4 次 `old_str not found`、3 次 `undo`,两次结果带 `WARNING: gplearn/custom_sr.py no longer parses`,35 次模型调用、284,363 个完成 token,**没有一次 `test()`**;最终可编辑区 167 LOC、`ast.parse` 失败、与最近基线 `lexicase_gp` 相似度 0.47。同题 `rlv5_lo32nm_a10_s20` 旧代 0.5421 是原文 §3.4 已指出的“锦标赛规模 7→3 的整数旋钮”。

## 6. 局限

- **单次运行 × 2**:两代是两次独立运行,不是每臂 3 次;所以本文只对“两代同向 + 两族同向 + 两检验过线”的效应下结论(sim_bl / novelty / LOC 的 RL 效应),分数本身的臂间差仍按 §10 记分卡的说法“不打 ★”。
- **相似度是文本相似度**:token difflib ratio 会把“换了名字的同一算法”算成不像;`ident_bl`(标识符 Jaccard)作为旁证方向一致。
- **模板/基线是题目给的那几份**:novelty 高不等于真创新——§5 的 B、C 都是 novelty 高但坏掉的代码;本文只声称 novelty 与得分负相关,不声称模型“能”或“不能”创新。
- 旧代 6 条无轨迹(agent 未启动)按 0 计入分数、不计入代码指标;3 条最终文件不可回放校验,直接用工作区文件。
- 4B 非 RL 臂只有 17–18 条非模板运行,4B 的代码指标 n 小。

## 7. 复现

```
cd /scratch/gpfs/CHIJ/ziran/innov_v2_multi/mlsroot
export PYTHONPATH=$PWD/src MLSBENCH_ROOT=$PWD MLSBENCH_DATA_ROOT=/scratch/gpfs/CHIJ/ziran/innov_v2_multi/mlsvendor/data
P=/scratch/gpfs/CHIJ/ziran/innov_v2_multi/envs/client/bin/python
$P <repo>/experiments/mls_quant/mls_quant_extract2.py   # -> runs.json texts.json templates.json baselines.json(写到脚本所在目录)
$P <repo>/experiments/mls_quant/mls_quant_stats.py      # -> flat.csv report_tables.md pair_results.json;stdout 含 §2.2 逐臂拆分
```
`flat.csv` 每行一条轨迹,列含 `agent_dir` 可由 `runs.json` 取;§5 引文的 agent 目录:`$D/mlsroot/logs/<task>/vllm/<tag>_p1__cc-<job>-<task>/agent/messages.jsonl`(A:13865165/13865169;B:13865165/13865169/13865171/13865167;C:13865171)。
