# Case Study(clean 轮):base → SFT → soup → RL 全链路详细结果

> 本文聚焦本波实验的完整四阶段链条,主线 = **wd03 臂**(weight-decay 0.3 正则 SFT → α0.1 soup → 纯 synthetic RL),对照 = 同配置 base+RL。
> 口径:全部评测同协议(thinking 模式、max_tokens 32768、temp 1.0/top_p 0.95/top_k 20/presence 1.5、每题 n=5);FCS = strip `<think>` 后官方抽取;分数为 mean@5(括号内 best@5);错误样本率 ≤0.2% 的才引用。
> 姊妹文档:`CLEAN_DECONTAM_REG_zh.md`(全 sweep)、`INNOVATION_CAMPAIGN_REPORT_zh.md`(主报告)。

---

## 0. 总表(一图流)

| 阶段 | 模型 | FCS | ALE | Research | MLS* |
|---|---|---|---|---|---|
| ① base | Qwen3.5-9B instruct | 7.05 (11.78) | 357 (515) | 9.08 | 0.032 |
| ② SFT 直测 | full-FT@clean 数据 | **0.31–2.42**(塌陷) | 287–317 | 4.46–6.89 | — |
| ③ soup(模型汤,α=0.1) | wd03_a10 | 6.68 (11.91) | 416 (595) | 8.34 (19.5) | 0.070 |
| ④ **RL step5** | **wd03_a10 + GRPO×5** | **8.42 (14.88)** | **434 (632)** | **14.01 (27.4)** | 见 §5 |
| 对照:base+RL step5 | 同配置 RL on base | 7.25 (12.28) | 389 (540) | 11.48 (23.3) | 见 §5 |

\* MLS 采用共同子集/None→0 的保守口径(§5 详述该基准的三个结构性问题)。

**一句话结论:innovation-SFT 是"先验注入",soup 是"能力保全",RL 是"放大器"——wd03+RL 在 FCS/ALE/Research 三线全面超过同配置 base+RL(8.42>7.25、434>389、14.01>11.48),且 FCS 比 raw base 净涨 +1.37。**

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

### 4.4 轨迹(截至成文)

| step | wd03 FCS | wd03 Research | base+RL Research |
|---|---|---|---|
| 0(soup) | 6.68 | 8.34 | 9.08(=base) |
| 5 | **8.42** | **14.01** | 11.48 |
| 10 | 评测中 | 13.49 | 评测中 |
| 15/20 | 流水中 | 流水中 | 流水中 |

Research 在 s10 高位站稳(13.49),未见死亡螺旋式回落。step10/15/20 全存点评测出齐后按最优步定稿。

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
