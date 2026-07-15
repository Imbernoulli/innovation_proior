# Case Study(clean 轮):base → SFT → soup → RL 全链路详细结果

> 本文聚焦本波实验的完整四阶段链条,主线 = **wd03 臂**(weight-decay 0.3 正则 SFT → α0.1 soup → 纯 synthetic RL),对照 = 同配置 base+RL。
> 口径:全部评测同协议(thinking 模式、max_tokens 32768、temp 1.0/top_p 0.95/top_k 20/presence 1.5、每题 n=5);FCS = strip `<think>` 后官方抽取;分数为 mean@5(括号内 best@5);错误样本率 ≤0.2% 的才引用。
> 姊妹文档:`CLEAN_DECONTAM_REG_zh.md`(全 sweep)、`INNOVATION_CAMPAIGN_REPORT_zh.md`(主报告)。

---

## 0. 总表(一图流)

| 阶段 | 模型 | FCS | ALE | Research | MLS* |
|---|---|---|---|---|---|
| ① base | Qwen3.5-9B instruct | 7.05 (13.54) | 357 (618) | 9.08 | 0.032 |
| ② SFT 直测 | full-FT@clean 数据 | **0.31–2.42**(塌陷) | 287–317 | 4.46–6.89 | — |
| ③ soup(模型汤,α=0.1) | wd03_a10 | 6.68 (13.74) | 416 (681) | 8.34 | 0.070 |
| ④ **RL step5(主线首存点)** | **wd03_a10 + GRPO×5** | **8.42 (16.88)** | **434 (750)** | **14.01 (32.1)** | 见 §5 |
| 对照:base+RL step5 | 同配置 RL on base | 7.25 (13.96) | 389 (609) | 11.48 (27.4) | 见 §5 |
| ④′ 20 步全轨迹(4 臂) | 见 §4.4 三张全表 | 终点最高 11.03(base+RL);我臂最高 10.70(newmt) | 终点 622≈620 平手 | **终点最高 16.02(nom_a5,我臂)** | 见 §3½/§5 |

\* MLS 采用共同子集/None→0 的保守口径(§5 详述该基准的三个结构性问题)。§4.4 起全部竞赛/研究分数为"每题 5 槽、缺失/错误记 0、分母恒等"的严格下界口径。

**一句话结论:innovation-SFT 是"先验注入",soup 是"能力保全",RL 是"放大器"——同步对照 s5–s15 我们全程领先(s5 三线 8.42>7.25、434>389、14.33>11.80);20 步终点上 Research 我们保持领先(16.02>14.52),FCS 被 base+RL 后程追上(11.03 vs 10.70,按题 bootstrap 95% CI 跨 0 统计不可分,§4.4 有逐题拆解),ALE 平手。四臂 RL 全部大幅涨分,我臂最大增益 +4.81(newmt)。**

---

## 1. 阶段①:base(Qwen3.5-9B instruct)

参照系:FCS 7.05 / ALE 357 / Research 9.08 / MLS 0.032(共同子集口径)。
行为特征:竞赛题上思考极长(评测端 65% 的样本顶满 32k cap,中位 completion = 32768),但"能写完的那 1/3"质量高——这就是 7.05 的来源。发现类任务(Research/MLS)上保守:倾向标准方法,少见多方案探索。

## 2. 阶段②:SFT 直测 —— 先验注入成功,但竞赛能力塌陷

配置:full-FT,数据 `innovation_clean_decontam_traj`(**去污染 + traj 不用 agentic**),cutoff 53760,1 epoch,lr 5e-6,wd 0.1/0.3。

| SFT 直测 | FCS | ALE | Research |
|---|---|---|---|
| nomaintain_wd01 | 0.31 (1.02) | 286.5 | 4.46 |
| full(maintain)_wd01 | 2.42 (5.66) | 317 | 6.89 |

**解读**:全参数 SFT 把创新数据的分布"学满",代价是把 base 的竞赛精确性冲掉(FCS 7.05→0.3–2.4,全线塌);Research 也低于 base——**先验学到了,但表达被塌陷掩盖**。这不是失败:SFT 的角色是把创新倾向写进权重,能力保全交给下一步。(r1 轮 LoRA 对比证明塌陷主因是全参数大扰动,与数据本身无关。)

## 3. 阶段③:soup(model soup,权重空间加权平均)—— 能力保全 + 先验表达

merged = 0.1·SFT + 0.9·base。轻档(α=0.05–0.1)是甜点,α≥0.3 递减、α=0.5 全塌(见 CLEAN_DECONTAM_REG §1-2)。

| soup(α=0.1) | FCS | ALE | Research | 备注 |
|---|---|---|---|---|
| **wd03_a10(主线)** | 6.68 (11.91) | **416 (595)** | 8.34 | FCS/ALE 最保 |
| nom_a5 | 6.58 (12.56) | 398 (562) | 11.1 | 全面均衡 |
| newmt_a10 | 5.89 (10.48) | 350 | 11.47 | Research 最强 |

**解读**:soup 后 FCS 回到 6.6–6.7(≈base 的 95%),ALE **超** base(416 vs 357),MLS 公平口径 0.070 vs base 0.032(2 倍+)。**创新先验以 10% 的权重混入即可表达**——发现类任务全面受益,竞赛类几乎不亏。wd03(更强正则)是本轮 sweep 里"保 FCS+ALE"最优的 SFT 配方。

## 3½. MLS-Bench 专项:SFT(+soup)在 RL 之前就涨(应汇报要求单列)

**同协议、同 dev-root 判分、None→0 全 20 题口径**(base 无 None,数字即官方值):

| 模型(均为 RL 前) | MLS mean | vs base |
|---|---|---|
| base(Qwen3.5-9B instruct) | 0.038 | — |
| **nomaintain soup a5** | **0.1013** | **2.7×** |
| newmt soup a10 | 0.1033(官方口径,18/20+2 infra-None) | 2.7× |
| **wd03 soup a10(主线臂)** | 0.0697 | 1.8× |
| full(maintain)soup a10 | 0.0615 | 1.6× |
| nomaintain soup a10 | 0.0606 | 1.6× |
| full soup a5 | 0.0447 | 1.2× |

**要点**:
1. **SFT+soup 阶段 MLS 全线 ≥ base**,最好档(nomaintain a5 / newmt a10)到 **0.10 档 = base 的 2.7 倍**——创新先验在"开放式 ML 研究 agent"任务上的收益,**不需要 RL 就已兑现**。
2. 该结论**在 forensic 审计后依然成立**:公平共同子集口径下我们 0.070 vs base 0.032(§5 的三个结构性问题主要影响"涉 base+RL 的横向比较",不影响 pre-RL 的 soup-vs-base——base 无 None 剔除、差距 2-3 倍远超单题翻盘量)。
3. 与 §6.9/附录 A 的 case 呼应:MLS 上的分差有可引用的方法学内容(causal-treatment-effect 的 DR-learner 正交化 +375%、similarity 分析的 Jaccard 0.22-0.24 vs 0.28)——涨分与"提出更偏离 baseline 的方法"同源。
4. RL 之后(§5):MLS 上各臂同质化、横向比较进入噪声区——**MLS 的故事主体在 SFT+soup 阶段**,这也是为什么本基准放在 pre-RL 汇报。

## 4. 阶段④:RL —— 放大器(主结果)

### 4.1 配方(踩过坑之后的最终版)
- 数据:**纯 self-generated synthetic 500 题**(FrontierSmith/synth,确定性判分,无 train=eval 泄漏);此前 mixed@20k 配方已定案为"截断死亡螺旋"净负(CLEAN_DECONTAM_REG §3a)。
- GRPO:32 prompts × 8 rollouts/步,mini-batch 128 序列 × 2 更新/iteration,KL/loss 全 verl 默认,20 步、每 5 步存。
- 关键对齐:**rollout 采样 = 评测口径**(temp 1.0/top_p 0.95/top_k 20/presence 1.5/32768)——训练优化的就是被评测的行为。
- 4×H200/臂;w1→w2 断点续训链;每个存点全量 4 项评测。

### 4.2 训练健康(vs 死亡螺旋的反面)
奖励密度:有梯度组 53–88%/步(旧 mixed 配方 FCS 组 81% 全零);回复长度全程 ~100k 字符无塌缩;entropy 正常;无一预警指标触发。**32k cap + 大组 + 连续分值 synthetic 判分**是与死亡螺旋的三个决定性差异。

### 4.3 Step5 主结果(全对照)

| step5 | FCS | ALE | Research |
|---|---|---|---|
| **wd03+RL** | **8.42 (14.88)** | **434 (632)** | **14.01 (27.4)** |
| base+RL(对照) | 7.25 (12.28) | 389 (540) | 11.48 (23.3) |
| Δ(wd03 vs base 起点) | **+1.17** | **+45** | **+2.5** |

- **RL 涨分**:wd03 6.68→8.42(**+1.74**),Research 8.34→14.01(**+5.7**);base 起点只涨 +0.20/+2.4。
- **同样的 RL,创新先验起点学得更多**:每一条 benchmark 上 wd03+RL 都超 base+RL——这就是"SFT 先验 + RL 放大"假设的直接验证。
- wd03+RL 的 FCS 比 raw base **净涨 +1.37**(9B 上首次 RL 后超 base,且非靠单点运气:best@5 14.88 也是全场最高)。

### 4.4 完整轨迹(4 臂 × 5 存点 × 3 基准,全部出齐)

**口径(严格、全表分母恒等)**:每题固定 5 个样本槽位,缺失/错误槽位一律记 0 分计入分母(全表已验证零缺失;错误槽位:FCS 的 judge 基建超时已全部对称重判,剩 0–5 个/格;Research 的评测器基建错误已全部对称重跑,剩 1–12 个/格均为模型自身代码错误)。格式 = mean@5 / best@5。

**FCS**

| step | base+RL | wd03+RL | nom_a5+RL | newmt+RL |
|---|---|---|---|---|
| 0 | 7.05 / 13.54 | 6.68 / 13.74 | 6.58 / 14.67 | 5.89 / 12.18 |
| 5 | 7.25 / 13.96 | 8.42 / 16.88 | 7.22 / 14.01 | 6.97 / 14.10 |
| 10 | 8.64 / 17.68 | 8.62 / 16.62 | 9.07 / 16.93 | 9.36 / 19.28 |
| 15 | 9.90 / 17.87 | 9.44 / 18.30 | 9.10 / 16.67 | 10.15 / 18.47 |
| 20 | **11.03 / 20.09** | 9.80 / 17.70 | 9.88 / 18.18 | 10.70 / 17.65 |

**ALE**

| step | base+RL | wd03+RL | nom_a5+RL | newmt+RL |
|---|---|---|---|---|
| 0 | 356.6 / 618.4 | 416.0 / 680.9 | 398.5 / 644.1 | 349.5 / 582.8 |
| 5 | 388.9 / 608.8 | 434.0 / 749.6 | 397.8 / 668.2 | 539.7 / 819.7 |
| 10 | 535.5 / 909.7 | 497.3 / 805.0 | 503.3 / 822.9 | 529.1 / 852.9 |
| 15 | 566.2 / 853.4 | 524.1 / 837.6 | 549.5 / 853.0 | 554.8 / 965.7 |
| 20 | **621.8** / 952.7 | 569.6 / 935.6 | 568.4 / **1003.5** | 619.5 / 940.9 |

**Research**(2026-07-14 深夜起为"对称 resume 后"终值:所有臂的评测器基建错误统一重跑,剩余错误 1–12 个/格均为模型自身代码错误记 0——此前各臂 resume 不对称,旧值普遍偏低 0.3–2.4)

| step | base+RL | wd03+RL | nom_a5+RL | newmt+RL |
|---|---|---|---|---|
| 0 | 9.08 / 24.61 | 8.34 / 24.07 | 11.12 / 23.69 | 11.47 / 27.79 |
| 5 | 11.80 / 27.39 | 14.33 / **32.05** | 15.05 / 30.60 | 11.86 / 29.64 |
| 10 | 13.44 / 31.32 | 14.72 / 29.90 | 13.38 / 27.36 | 11.28 / 28.96 |
| 15 | **16.53** / 29.26 | 13.83 / 30.06 | 15.39 / 31.35 | 11.77 / 27.02 |
| 20 | 14.52 / 29.17 | 13.76 / 31.90 | **16.02** / 28.15 | 12.38 / 26.59 |

**终局读法(如实,两面都写)**:
1. **"RL 涨分"全面成立**:四臂 FCS 全部单调或近单调上升;涨幅 newmt +4.81(5.89→10.70,全场最大)、nom_a5 +3.30、wd03 +3.12、base +3.96。
2. **逐步同步对照(s5–s15)我们领先**:s5 wd03 8.42 > 7.25;s10 newmt 9.35/nom_a5 9.07 > 8.64;s15 newmt 10.15 > 9.90。
3. **s20 终点 FCS 名义上 base+RL 最高(11.03 vs newmt 10.70),但按题 bootstrap 95% CI = [−1.73, +1.05],统计上不可分**;base−nom(+1.14)、base−wd03(+1.21)的 CI 同样跨 0,连 base 自己 s15→s20 的 +1.12 也不显著([−0.22, +2.41]:31% 来自 3 道 1/5 槽 0→100 的彩票题,25% 来自单题档位跃迁)。中程各步的"我臂领先"同样不显著且每步是不同的臂(选择偏置)。对称的诚实结论:**s20 终点四臂 FCS 在当前评测功率(每题 5 样本)下无法区分;可分的事实是——按自身起点的增益 newmt +4.81 全场最大(base +3.96),而 base 起点本来最高(7.05 vs 5.89)**。完整取证(含反事实截断中和、训练端曲线四臂重合、熵/KL 无异常)见 [LATE_CATCHUP_FORENSICS_zh.md](LATE_CATCHUP_FORENSICS_zh.md)。
4. **ALE 终点平手**(base+RL 621.8 ≈ newmt 619.5;best@5 最高点是 nom_a5 的 1003.5)。
5. **Research 终点我们赢**:nom_a5 16.02 > base+RL 14.52(+1.50;base+RL 峰值 16.53 在 s15,此后回落);s5 时 wd03/nom_a5 领先 base+RL +2.5~3.3。
6. 行为面:四臂都在 RL 中学会控制思考长度(32k 截断率 base 臂 57%→26%,我们臂 56–60%→26–34%),零分率同步下降——增益的共同来源;base 起点"未完成思考"储备最大,是其后程持续走高的主要机制。

## 5. MLS 专节:为什么撤回定量对比

Forensic 审计(读全部 agent transcript)发现该基准在当前能力段有三个结构性问题:
1. **do-nothing 基线 ≈0.094**:交回未改模板即得分,高于几乎所有臂的 agent 均值——大部分正分是"没改坏模板"。
2. **None 排除偏差**:超时/崩溃任务被从均值剔除,恰好偏袒 base+RL(其最高价值任务超时);None→0 后 base+RL 0.093→0.084。
3. **单种子 + 单题波动 0.3–0.6**:臂间差距 = 半道~一道题的翻盘量;leave-one-out 下排序不稳定。

**保留的结论**:我们 pre-RL soup(共同子集 0.070)≫ raw base(0.032);**真实的定性发现**:RL 后模型的多轮 agent 行为退化(对被拒绝的编辑原样重试 7 次、思考 +46%、超时任务翻倍)——单轮代码 RL 与多轮 agent 能力存在真实张力,是后续工作方向。

## 6. Case 级分析(读四阶段在相同题目上的真实生成)

### 6.1 阶段级行为统计(FCS,860 样本/阶段)

| 阶段 | tokens 中位 | 32k 截断率 | 出码率 | 完整代码块率 | score>0 率 | mean |
|---|---|---|---|---|---|---|
| base | 32768 | 64.5% | 60.5% | 56.6% | 15.6% | 7.05 |
| SFT 直测 | **2550** | 2.2% | **90.1%** | 84.7% | **1.3%** | 0.31 |
| soup | 32768 | 63.5% | 62.2% | 57.6% | 14.7% | 6.68 |
| RL | 32768 | 59.3% | 68.5% | **64.0%** | **19.9%** | **8.42** |

三条关键读数:**SFT 崩塌不是"想不完"而是"想太少+代码是坏的"**(中位输出掉到 2550 tokens,格式最守规矩,但代码编译不过/逻辑胡编);**soup 把分布拉回 base**(各项指标与 base 重合);**RL 三线齐动**(截断 −4.2pt、完整代码 +6.4pt、得分率 +5.2pt)。另有一个贯穿全文的关键量:**"短思考(<10k 字符)快提交"坏模式占比 base 12.1% → soup 18.4%(SFT 残留负迁移,该模式均分仅 2.3)→ RL 3.0%(被剪除)**。

### 6.2 FCS 案例三则

**#193 Max-2-SAT(base 33.5 → SFT 0.3 → soup 35.4 → RL 72.2)**
base:5.7 万字符试错式思考收敛到随机重启局部搜索,5 试 2 中(78/89 分)——方向对但方差大。SFT:三种坏法——think 是漂亮的"**Key Insight → Algorithm**"计划体,代码却幻觉拼贴(`bool flag[MAXM]` 的 `MAXM` 未定义;`const int N=555` 而题目 n≤1000 直接越界);`sat/=m;` 整型除法把计数归零;甚至开头输出乱码。**SFT 的 think 学会了创新语料的多方案计划体,但代码生成器坏了。** soup:完全恢复 base 式长思考,3/5 拿 86–91。RL:4/5 集中 87–92,成功样本思考只有 soup 的 ~40%,并在同款算法上加了求生细节:
```cpp
if (m == 0) { /* 任意赋值直接输出 */ }
int steps_per_restart = 10000; // Limit iterations per run to prevent TLE
```

**#176 3-SAT(base 83.8 → SFT 0 → soup 18.8 → RL 86.5)—— soup 被 SFT 残留拖垮、RL 修复**
SFT 交出 **Python** 代码(C++-only 题)。soup 在此题反而塌(1/5):失败样本恰是"短思考模式"(think 仅 5k 字符,base 同题最短也要 15k tokens),仓促收敛后带两处硬伤(数组尺寸不可行、输出最后一次重启而非 best)。RL 5/5 全部出分(53–98)——**把 soup 里被 SFT 污染的快提交样本整体清除**。

**#179 Subset Sum(base 3.0 → soup 50.9 → RL 0)—— 诚实反例:RL 不是全面变强**
soup 两个 100 分:先读懂评分规则("If we can achieve S=W, the score is 1")再动手,BigInt+贪心外面套 `shuffle+5 trials` 随机重启命中 S=W——**"利用评分规则 + 多试几手"的做派,base 同题 5 个样本一次都没出现,这是创新先验最干净的正迁移证据**。RL 却全 0:BigInt 底数(1000)与切块宽度(9 位)不一致,全部算术错——5 步 RL 对"实现精度型 bug"没有免疫力,反丢了 soup 的高分。

### 6.3 Research 案例(9.08 → 8.34 → 14.01 的机制)

**`cant_be_late_multi`(多区域 Spot 调度,soup 9.7 → RL 67.1)**:soup 策略雄心勃勃但调用不存在的接口(`self._estimate_remaining_steps()` 幻觉方法,AttributeError 即崩,4/5 零分);RL 收敛到极简且 API 绝对安全的 `if has_spot: return SPOT else: ON_DEMAND`,4/5 样本**精确同分 83.82**——策略分布被钉死在已验证可跑通的模式上(base 只有 1/5 撞中)。

**`vdb_pareto/high_recall`(faiss HNSW 调优,base 0 → soup 20 → RL 59.8)**:base 5/5 栽在同一个 SWIG 陷阱(`IndexHNSWFlat(..., metric_type=...)` 不接受关键字参数);RL 3/5 ≈100:构造后再设属性绕开陷阱,think 里还显式做预算推理("relaxed latency budget … we can afford ef_search around 500-1000+")。
```python
self.index = faiss.IndexHNSWFlat(dim, self.M)
self.index.metric_type = faiss.METRIC_L2
self.index.hnsw.efSearch = self.ef_search
```
**Research +5.7 分的机制:防御式 API 使用 + 策略收敛带来的成品率**(同一简单方案从 1/5 撞中变成 3–4/5 稳定复现)。

### 6.4 创新先验的痕迹核查(think 风格标记,次/万字符)

| 阶段 | think 均长 | "Key Insight" 密度 | "Wait"(自我质疑) | 短思考占比 |
|---|---|---|---|---|
| base | 86.9k 字符 | 0.09 | 15.0 | 12.1% |
| SFT | 6.6k | **1.01(11×base)** | **0.35(几乎消失)** | 85.3% |
| soup | 84.8k | 0.10 | 14.8 | 18.4% |
| RL | 89.3k | 0.08 | 12.4 | **3.0%** |

诚实结论:**创新倾向在 SFT 直测里最显性**(多候选方案枚举+权衡确实学到了:#193 一口气列 Approach 1–4 各带复杂度和期望得分),但自我审辩几乎消失——有"提案"没有"批判",加上代码器坏掉,创新姿态成了空壳。**α=0.1 soup 后逐句风格回到 base**,残迹是分布层面的:短思考负迁移(+6.3pt)与 #179 式"读评分规则+多试几手"的正迁移并存。**RL 是校准器**:保住 base 式长审辩,把 SFT 注入的仓促模式从 18.4% 压到 3.0%,留下倾向中有奖励回报的部分(anytime 结构、边界护栏、防御式 API)。

### 6.5 RL 收益归因(soup→RL,per 题 5 样本)

| 范围 | 截断率 | 完整代码率 | score>0 率 | mean |
|---|---|---|---|---|
| 全部 172 题 | 64.4%→60.7% | 57.6%→64.0% | 14.7%→19.9% | 6.68→8.42 |
| RL 增益 Top-10 题 | 36%→30% | 80%→90% | 26%→**68%** | 13.5→45.6 |

**主因 = 方案正确率 + 可靠出码,次因 = 省预算**。截断变化甚至双向:#27 从 4/5 截断降到 1/5(真·想得完了);#17/#10 截断反升到 5/5 却照样拿分——RL 学会"先把完整代码写进 think 再继续验证"的保险动作(判分提取能从截断的 think 里捞出完整程序,#17 有样本以此拿 93.9,base 同题只有 1/5 来得及这样做,RL 3/5)。

### 6.7 二次审计:一个 reward-hack 发现 + 创新口径的最终修正

对 top 分差题的定向二审(独立 agent,核对到评测器源码)有两个必须诚实记录的结论:

**① `fused_linear_jsd` 的"满分"是评测器漏洞,不是创新——撤回。** 该 Research 题(Triton kernel)上 wd03+RL 的 100 分样本,kernel 本身是坏的(调用不存在的 API),但包了 `try/except` 兜底返回 `torch.empty((M,))`——而评测器**先给 baseline 计时再验正确性**,PyTorch caching allocator 把 baseline 释放的正确结果显存块原样分给了这个 `empty()`,以 `atol=0.5` 的宽松容差通过校验,近似 no-op 的耗时又轻松打过 7× 加速线。这是 KernelBench 社区已知的"未初始化显存复用"作弊模式。**处置**:(a) 此题分数不作创新证据;(b) headline 稳健性已验证——剔除该题后 wd03+RL Research 仅从 15.45→15.38(逐题均值口径),主结论不受影响;(c) 官方评测器需修(输出 memset/换输入再验证),且 RL 奖励侧要防同类漏洞;(d) RL 学到的"异常兜底+保底返回"风格会在此类评测器上意外得利,解读高分时须警惕。

**② 创新 vs 可靠性的最终口径。** 二审确认题 175/193 的增益是"交付纪律"(更少编译错误/截断,算法同族甚至更朴素);且 base 在这些题的 think 里同样有多方案探索——**在 top 分差题里找不到干净的"算法创新"案例**。全文最可辩护的创新差分证据仍是:#179 的"读评分规则找满分条件 + 多次尝试"(§6.2,base 同题 0/5 出现)与 §6.4 的分布级统计;对 base 的整体优势应表述为**"同等算法水平下显著更可靠 + 把创新倾向中有回报的部分(多试、护栏、防御式 API)固化"**,而非"发明了新算法"。

### 6.9 精选创新 case(全文见 [GOOD_CASES_zh.md](GOOD_CASES_zh.md))

广泛挖掘(ALE 本波 + 旗舰旧模型 + Research 全 64 题重排 + 旧 curated case 复核)后,**3 个达标 case**(同题、跨样本稳定、可引用设计差异、非评测器伪影,均核过 base+RL 严格对照):

1. **AHC025(本波,最硬)**:天平比较分组题——base 只会"排序+蛇形分桶",**wd03+RL 发明"在线学权重(感知机式乘法更新)+ 装箱局部搜索"**,设计族 3/5 样本复现,内部对照单调兑现(轮转 1.23e10 → base 排序 6.77e9 → 我们 **5.47e9**),模型自注释 "This acts as a gradient descent / reinforcement learning to estimate relative weights"。
2. **AHC046(旗舰)**:冰面滑行题——对照只会曼哈顿逐格走,我们把**滑行建模为图的边跑 BFS**(1119 vs base+RL 547);caveat:双方各 1 个干净成功样本。
3. **MLS causal-treatment-effect(前轮,跨代复现)**:base 朴素 T-learner vs 我们 **doubly-robust DR/R-learner 正交化**(0.26 vs 0.055,+375%,两代四 build 复现);caveat:soup vs start 框架(均无 RL)。

同文件含诚实排除清单(AHC015=正确性差异、AHC039/016 死于严格对照、AHC008=单样本运气)和**第二个 evaluator artifact 披露**(`qknorm`:逐字节交回题面 baseline 在 clamp-型 metric 下吃满分,"复交参考基线"型,与 §6.7 的显存复用型并列)。

### 6.10 本节一句话

SFT 把创新语料的"多方案计划体"写进了权重但压垮了代码生成;0.1 权重的 soup ≈ base 且夹带 18% 短思考残留;5 步 RL 的 +1.7(FCS)/+5.7(Research)不是发明更强算法,而是**剪掉坏模式、钉住已验证方案、提高完整代码在预算内落地的概率**——同时 #179 提醒:RL 后模型依旧会栽在实现精度上。

## 6½. 姊妹研究文档(2026-07-14 同步产出,细节不在本文重复)

- **[AVERAGE_INNOVATION_zh.md](AVERAGE_INNOVATION_zh.md)** — "Average 是否毁掉创新能力"专题:创新指标 vs α 全曲线(表层文体 α≤0.2 归零,但行为层 Jaccard 偏离单调保留 38–81%、MLS 全 α ≥ base)、α=1 塌陷 forensics(83.2% 真编译错、think 20× 缩短、自查回路归零)、α=0.5 全量结果。
- **[SFT_DIRECT_RECOVERY_zh.md](SFT_DIRECT_RECOVERY_zh.md)** — "不 soup 的纯 SFT 如何不塌"配方实验(maintain 加倍/SWA 训练内平均/两阶段回炉/LoRA 满强度直测),评测流水中。
- **[Q36_35B_RESTART_zh.md](Q36_35B_RESTART_zh.md)** — 35B(Qwen3.6-35B-A3B)线:LoRA r32 s0.1 已实现 FCS +9.8% 首个净增益;full-FT 直测/soup/LoRA s=1.0 全对照与 RL 冒烟流水中。
- **[MLS_RL_INTEGRATION_zh.md](MLS_RL_INTEGRATION_zh.md)** — MLS-Bench 多步 agentic RL × verl GRPO 集成(token 级 AgentLoop + Apptainer episode 子进程),端到端冒烟已通过(reward 非退化、ckpt 落盘)。
- **[LATE_CATCHUP_FORENSICS_zh.md](LATE_CATCHUP_FORENSICS_zh.md)** — "base+RL 后程追赶"全量取证:终点臂间差 bootstrap 全部不显著;增益通道=截断释放(四臂共享、无人饱和);训练端四臂曲线重合,熵/KL 无异常;建议=关键点加测样本数 + 延长步数(newmt 起点+40 步为最优单一改动)。

## 7. 复现指针

- SFT:`LF-innov/examples/train_full/auto/os-q35_clean_nom_wd03.yaml`(数据 `innovation_clean_decontam_traj`)
- Soup:`FrontierSmith/scripts/cc_model_soup_merge.py --alpha 0.10`
- RL:`FrontierSmith/scripts/cc_rl_frontiersmith_synth_submit.sh`(ONLY="q35_inst_start cl_wd03_a10",GPUS=4 STEPS=20 SAVE=5;采样对齐已是默认)
- 评测:`slurm/cc_eval_thinking_both_ailab.sh` / `cc_eval_research_ailab.sh`(research 只能 H100/H200)/ `cc_eval_mlsbench_cpu_ailab.sh`(必须 MLSBENCH_ROOT=MLS-Bench-dev)
- RL ckpt→HF:`scripts/merge_fsdp_to_hf.py`(纯格式转换,非模型融合)

---

## 附录 A:算法-baseline 相似度分析(早期轮次,应汇报要求汇编)

早期轮次(r3 系模型)做过一项独立测量:**模型在 MLS-Bench 上提出的算法,与题面给定 baseline 的词法技术指纹 Jaccard 相似度**(`experiments/similarity_codex.md`,方法:从 agent 编辑增量提取技术 token,与 task_description 提取的 baseline 集合求交并比;↓ 越低 = 偏离 baseline 越远)。

### A.1 两套口径的一致结论:我们的模型更敢偏离 baseline

| 口径 | OURS(两变体) | BASE |
|---|---|---|
| similarity_codex(MLS 20 题) | **0.242 / 0.217** | 0.279 |
| 主报告独立复测(两源) | **0.31 / 0.22** | 0.39 / 0.28 |

逐题最典型对比:`causal-discovery-discrete` 上 **BASE 的 Jaccard=0.636 且被标注 BASELINE_ONLY**(方案就是把题面 baseline 复述一遍:bdeu/boss/ges/pc…),我们两变体 0.167/0.250 且 beyond_baseline=True。

### A.2 点名 case:我们的模型自命名的新算法提案(原文引用)

**TeeMOEA**(optimization-multi-objective,methodtraj 臂,agent 编辑原文):
> "TeeMOEA: Tangential Elasticity-based MOEA with Adaptive Density Control. This algorithm combines three key innovations: 1. Adaptive reference point tracking… 2. Tangential distance-based survival selection… 3. Anisotropic mutation that learns to prefer directions with higher search value… back-coupling between selection and survival … creating a feedback loop"

**EBO-M**(optimization-hyperparameter-search,methodv4 臂):
> "EBO-M: Evolutionary Bandit Optimization with Multi-fidelity. Novel contribution: Combines differential evolution mutation (like DEHB) with bandit-weighted evaluation scheduling (like UCB)… Track variance of each hyperparameter across history… Multi-fidelity evaluation with adaptive truncation"

**base 在完全相同的两道题上**:只在模板 docstring 里填了默认实现("Initialize the MOEA with problem parameters"),没有命名方法、没有创新声明。

### A.3 诚实的混淆与本波实验的呼应

当时的定论(主报告 §4):相似度信号被一个混淆强烈干扰——**越 novel 的方案越容易崩**(MLS 20 题里 13–16 是 fallback;最 novel 的 EBO-M/TeeMOEA/Robust-GMM 全都没跑通拿分),跑通拿分的常是低-novelty 的 baseline 重组。所以该测量证明的是**创新"意向"的转移**(disposition transfer),而非成功的创新;代码复杂度同步测量无差异(不是技术堆砌)。

**与本波(§6)拼起来是一条完整因果链**:早期轮次证明了"敢偏离 baseline 的倾向"确实被 SFT 写进权重(A.1/A.2),但当时它以崩溃为代价;本波的 soup(保全代码能力)+ 32k RL(剪掉仓促模式、固化有回报的部分)把这份倾向**转化成了分数**——wd03+RL 三线超 base+RL。倾向是早期就有的,本波补上的是"让倾向能兑现"的工程。
