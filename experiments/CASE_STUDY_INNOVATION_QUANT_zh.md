# 创新性定量案例研究:FrontierCS(算法 + 科研)与 ALE-Bench

目标:回答“我们的模型是不是更像人一样搞创新”,并且**先证明评测本身是对的**,再谈结论。
所有数字由 `experiments/innov_quant/` 下的脚本从 `$D/outputs/cc_eval_*/shard_*/samples.jsonl` 重新算出,
原始中间产物在 scratch `case/innov/`。口径:统一协议(系统提示含 “It is now year 2026.”),
temperature=1.0 / top_p=0.95 / top_k=20 / presence_penalty=1.5 / max_tokens=32768 / N=5 / thinking on。

14 个主臂 + 6 个同协议复跑臂(`_y2026`,stage 记为 `rep`,用来当噪声底)。

---

## 0. 先做评测体检(三个必须先处理的坑)

1. **9B 科研赛道分片重叠。** `cc_eval_<arm>_research_thinking_32k_vllm/shard_0` 含全部 64 题,`shard_1` 又重复了其中 32 题,
   于是这 32 题的每个 (题, sample_idx) 有**两次独立且都合法**的生成。`paircommon` 的去重规则是 glob 顺序**最后一行获胜**(即 shard_1),
   每臂约 140 个重复键里有 42-43 个分数不同。本研究沿用同一规则,保证与记分卡一致。
2. **截断。** `completion_tokens>=32700` 且没有 `</think>` = 没有最终答案。FrontierCS 上截断比例:9B base 30%、rlv5_base 59%、
   rlv5_lo32nm 51%;4B 的 `rlv5_4b_base_s20` 高达 97%。截断样本分数≈0。**任何代码相似度分析只能用 complete 样本的 `</think>` 之后那段代码。**
3. **ALE 判题拓扑。** ALE 按墙钟计分。`judge_node_meta.json` 显示 9B 的 base/SFT 臂判在 cpu 分区(speedFactor 0.79-0.83),
   RL 臂判在 ailab(≈1.0),而 `rlv5_ft03nm_a20_s20` 判在 gpu-ee(**0.556**)。因此 9B 的 ALE「RL vs 非 RL」直接比是有混杂的;
   本报告用全部判在 ailab 的 `_y2026` 复跑臂做拓扑匹配替代,并把 ft03nm 的 ALE 对比整条剔除。

---

## 1. RL 之后分数为什么更高?三通道分解

把每次抽样的分数拆成三个因子:

```
mean@draw = P(complete) × P(score>0 | complete) × E[score | complete, >0] + (1−P(complete)) × E[score | truncated]
```

逐题算出三个因子后配对相减,把 Δmean 分配到各通道(残差=交叉项)。完整表见 `innov_quant/explain_rl.md`。

### 1.1 FrontierCS 算法赛道 9B(170 题共同集)

| arm | P(complete) | P(score>0 \| complete) | E[score \| complete,>0] | mean@draw |
|---|---|---|---|---|
| base9b_v2c | 0.753 | 0.184 | 32.89 | 4.70 |
| ft01mix_a10 | 0.691 | 0.217 | 39.36 | 6.03 |
| ft03nm_a20 | 0.721 | 0.180 | 35.00 | 4.71 |
| lo32nm_a10 | 0.739 | 0.163 | 37.76 | 4.77 |
| rlv5_base_s20 | 0.482 | 0.369 | 44.02 | 7.94 |
| rlv5_ft01mix_a10_s20 | 0.658 | 0.310 | 38.36 | 8.07 |
| rlv5_ft03nm_a20_s20 | 0.764 | 0.294 | 34.10 | 7.70 |
| rlv5_lo32nm_a10_s20 | 0.612 | 0.345 | 41.12 | 8.70 |

RL 对各自起点的增益几乎全部落在**非零率通道**:

| pair | Δmean | 完成率 | 非零率 | 条件分 |
|---|---|---|---|---|
| rlv5_base_s20 − base9b_v2c | +3.09 | −0.59 | **+2.60** | +1.25 |
| rlv5_ft01mix − ft01mix | +1.92 | −0.36 | **+1.92** | +0.28 |
| rlv5_ft03nm − ft03nm | +2.92 | +0.10 | **+2.31** | +0.57 |
| rlv5_lo32nm − lo32nm | +3.75 | −0.32 | **+3.04** | +1.31 |

也就是说:**RL 让“写完的程序能跑出分”的概率从 0.16-0.22 翻倍到 0.29-0.37。**
正分样本的分布也整体上移(中位数 20.5 → 33-37,p75 51.7 → 72-76),但幅度远小于非零率的变化。
科研赛道同样由非零率主导(0.18-0.21 → 0.30-0.35);ALE 因为提交必给分,增益全在条件分通道(+53 ~ +70)。

### 1.2 创新先验的价值在 RL 之后才出现,而且表现为“RL 不把模型练崩”

`innov_quant/crossbench.md`。RL(先验) vs RL(base),逐题配对 + 跨 bench 合并(Stouffer,单边)。

**4B:先验决定 RL 能不能跑。** `rlv5_4b_base_s20` 的完成率是 FrontierCS **2.8%**、科研 **5.2%**、ALE **2.6%** —— 这条臂在 RL 里彻底发散;
从先验 SFT 起步的两条 RL 臂完成率 64-85%。

| 对比(4B) | FrontierCS | 科研 | ALE |
|---|---|---|---|
| rlv5_4b_ft01mix − rlv5_4b_base,Δmean | +5.65(57/0) | +11.78(30/1) | +212.4(28/0) |
| rlv5_4b_lo32nm − rlv5_4b_base,Δmean | +6.65(62/0) | +9.35(27/4) | +185.1(28/0) |
| 同上,ΔP(complete) | +0.61 / +0.76 | +0.76 / +0.79 | +0.74 / +0.79 |

六个格子全部 Wilcoxon p<0.001。而**在 RL 之前**,同样两个 4B SFT 检查点并不比 base 好,甚至略差(跨 bench 合并 Z=−1.56)。

**9B:同一机制,弱得多。** 分数上,RL(先验) − RL(base) 在 11 个 (臂×bench) 格子里 8 个为正,但单个都不显著;
跨 bench 合并只有 ft01mix 过线(Z=2.03,p=0.021,四个 bench 方向全正,含 MLS-Bench)。
**但完成率通道 8/8 全正**(符号检验 p=0.0078),FrontierCS 上 +0.12 ~ +0.28、p<0.001:
RL(base) 完成 48%,RL(先验) 完成 61-76%。

> 结论(诚实版):**RL 教会模型交出一个能跑的程序;创新先验的作用是让 RL 不把长度策略练飞。两者都不是“更新颖”。**

---

## 2. 与 frontier 模型解池的相似度(替代“人类参考解”)

用户问:FrontierCS 不是靠人类解算分吗?没公开怎么算?
**答:靠的是 jury answer,不是 jury source。** 每题带 `testdata/*.in` 和 `*.ans`,`.ans` 里是评委/人类专家的目标输出或目标值,
checker 同时读选手输出和 `.ans` 给比值。188 题里 180 题的 checker 给部分分,55 个 checker 直接读 ans 流,81 题的 `.ans` 非平凡。
例:第 137 题 `chk.cc` 用 `score_ratio = score_on_grid(part)/score_on_grid(opt)`;第 160 题 `interactor.cc` 从 ans 流读
`baseline_value` 与 `best_value`,`score_ratio=(score−baseline)/(best−baseline)`。README 里写的 "Reference solutions are withheld"
只是不给参考**源码**。排行榜 `leaderboard/algorithmic.json` 里 "Human Experts" score_at_1 = 86.99。
推论:`score_unbounded > 100` = 在该题超过了人类参考值。

人类参考**源码**只有算法赛道第 263 题(`reference1.cpp`/`reference2.cpp`)和科研 `nbody_simulation` 两个变体的 `reference_baseline.cpp`,
样本量不足以做统计。替代参照系:官方解池里的 frontier 模型解(算法题每题约 45 份 `.cpp`:gemini3pro / gemini2.5pro /
deepseekreasoner / grok4fastreasoning / gpt5 / gpt5.1 / gpt5.2 / gpt5_low-medium-high / trinitylargethinking 等;科研每变体 7 份)。

结果(`innov_quant/oracle_tables.md`),FrontierCS 9B 的 4-gram Jaccard 中位数:

| arm | med jac_frontier | jac>0.3 | 最近的 frontier 模型(top3) |
|---|---|---|---|
| base9b_v2c | 0.166 | 6% | deepseekreasoner, gemini3pro, grok4fastreasoning |
| ft01mix_a10 | 0.164 | 7% | deepseekreasoner, grok4fastreasoning, gemini3pro |
| lo32nm_a10 | 0.163 | 7% | deepseekreasoner, gemini3pro, grok4fastreasoning |
| rlv5_base_s20 | 0.212 | 22% | grok4fastreasoning, gemini3pro, gpt5.1 |
| rlv5_ft01mix_a10_s20 | 0.232 | 33% | gpt5.1, gpt5_low, grok4fastreasoning |
| rlv5_lo32nm_a10_s20 | 0.232 | 29% | gemini3pro, gpt5.1, gpt5_low |

逐题配对:RL 让解**更像** frontier 模型池,三条 RL(SFT) 臂对各自 SFT 起点是 121−24 / 130−25 / 121−24,全部 p<0.001(Δ +0.051 ~ +0.079);
SFT − base 全部不显著(p 0.47-0.92);复跑臂噪声底 p=0.15-0.96。
题内排秩后,「像 frontier 池」与分数正相关:FrontierCS 9B rho=+0.220,科研 9B +0.236,科研 4B +0.364(全部 p<0.0001)。

**超过人类参考值的次数**(`score_unbounded >= 100`,FrontierCS 算法赛道,每臂 815 个抽样):
base9b 3 次 / 0 平(3 题);ft03nm 5/0(4 题);rlv5_base 2/3(2 题);rlv5_ft01mix 2/6(4 题);
rlv5_ft03nm 7/1(4 题);rlv5_lo32nm 6/2(5 题);base4b 2/2(3 题);4b_ft01mix 0;4b_lo32nm 0;
rlv5_4b_base 0;rlv5_4b_ft01mix 1/1(2 题);rlv5_4b_lo32nm 3/1(3 题)。总量太小,不支持任何强结论。

---

## 3. 提案多样性:RL 之后同一模型 5 次采样更雷同

`innov_quant/diversity_tables.md`。自相似度 = 同一模型对同一题 5 次抽样代码之间的平均两两 4-gram Jaccard(越低=方案越不一样)。

| bench / 家族 | base | SFT | RL(base) | RL(SFT) |
|---|---|---|---|---|
| FrontierCS 9B | 0.146 | 0.135-0.143 | 0.209 | 0.209-0.240 |
| 科研 9B | 0.203 | 0.186-0.195 | 0.274 | 0.279-0.313 |
| ALE 9B | 0.120 | 0.120-0.122 | 0.180 | 0.180-0.233 |
| 科研 4B | 0.191 | 0.170-0.174 | — | 0.387-0.394 |

「与兄弟抽样完全不同(Jac<0.3)的抽样占比」:FrontierCS 9B 89% → 56-66%;科研 64% → 42-47%;ALE 95% → 57%;科研 4B 76% → 37%。
逐题配对 RL−SFT:FrontierCS +0.070~+0.098(97−16 / 108−21 / 98−24,p<0.001),科研 +0.086~+0.116(p<0.001),
ALE +0.056~+0.110(p<0.001),4B 科研 +0.21(51−6,p<0.001)。

**唯一对我们有利的一项**:在科研赛道上,SFT 比 base **更**多样(自相似度更低),且高于复跑噪声底(该底在科研上是 ±0.005,p=0.26/0.82):
9B ft03nm −0.016(24−33,p=0.027)、4B ft01mix −0.017(p=0.022)、4B lo32nm −0.021(p=0.006)。
FrontierCS 算法赛道与 ALE-9B 的 SFT 效应都不显著;ALE-4B 的 −0.018(p=0.004)与其复跑臂自己的 −0.018(p=0.039)同量级,不算超过噪声。

跨模型相似度:base 与 SFT 互相之间的相似度 ≈ 各自的自相似度(Δ +0.001 ~ +0.012)。
**SFT 没有把提案分布挪到别处,只是让它略微散开一点。**

---

## 4. 题内关联:更「新颖」的解分更低

全族 main 臂 complete 样本在题内排秩后合并(`innov_quant/report_tables.md` C/D 段):

| 属性 | FrontierCS 9B rho | 科研 9B rho |
|---|---|---|
| novelty vs control | −0.128 | −0.226 |
| novelty vs pool (1−jac_pool) | −0.168 | −0.251 |
| LOC | −0.141 | −0.210 |
| #num literals | −0.118 | −0.236 |

novelty 三分位 × 平均分:FrontierCS 12.38 / 10.92 / 8.25(低→高);科研 16.80 / 10.59 / 8.43。全部 p<0.0001。
在这三个 bench 上,**越不像对照臂、越不像 frontier 池,分越低**。

---

## 5. MLS-Bench:没有支持「更创新」的统计量

见 `CASE_STUDY_MLS_QUANT_zh.md`。代码层面的方向是反的:RL 之后代码**更贴近**任务自带的 baseline 实现
(sim_bl +0.108,novelty −0.112,p<0.001),而且更短(LOC −13)。题内,贴近 baseline 预测更高分(rho 0.36)。
重算 wd03 时代的「技术词 Jaccard」词汇指标(`innov_quant/mls_lex.py`,504 条轨迹),14 组配对**全部不显著**(p 0.22-0.97)。
MLS 每臂每题只有一次运行,做不了多样性指标。

---

## 6. ★ 创新先验起作用的点:它挡住了 RL 的模式坍塌

前面 1-5 节全部只看**终点**(s20)的产物,所以看不出先验的作用。中途检查点(s5/s10/s15)的评测已经在盘上
(`cc_eval_rlv5_*_s{5,10,15}_*`),不需要新的 GPU。沿 RL step 画轨迹,先验的作用立刻显形。
脚本 `innov_quant/traj.py` + `traj_tables.py`,表 `innov_quant/traj_tables.md`。

### 6.1 4B:base 血统在 s15→s20 之间崩掉,先验血统没有

**P(complete)**

| 血统 | FrontierCS s0/s15/s20 | 科研 s0/s15/s20 | ALE s0/s15/s20 |
|---|---|---|---|
| 从 base 起 | 0.965 / 0.603 / **0.027** | 0.997 / 0.786 / **0.073** | 0.985 / 0.578 / **0.025** |
| 从 ft01mix 起 | 0.911 / 0.453 / 0.638 | 1.000 / 0.739 / 0.826 | 0.990 / 0.662 / 0.769 |
| 从 lo32nm 起 | 0.980 / 0.763 / 0.794 | 0.993 / 0.845 / 0.852 | 0.990 / 0.831 / 0.810 |

**mean@draw**

| 血统 | FrontierCS | 科研 | ALE |
|---|---|---|---|
| 从 base 起 | 5.0 / 4.8 / **0.6** | 6.1 / 16.3 / **8.0** | 199.0 / 190.4 / **91.1** |
| 从 ft01mix 起 | 4.0 / 5.6 / 6.3 | 8.5 / 18.4 / 20.2 | 167.1 / 268.9 / 299.8 |
| 从 lo32nm 起 | 4.0 / 6.1 / 7.3 | 5.7 / 17.6 / 17.1 | 195.4 / 249.3 / 272.5 |

base 血统在科研赛道 s15 还是最好的一档(16.3),s20 直接掉到 8.0。

### 6.2 检验:逐题双重差分(`innov_quant/traj_did.py`,表 `traj_did.md`)

DiD(题) = [先验血统(s20) − 先验血统(s15)] − [base 血统(s20) − base 血统(s15)],逐题 Wilcoxon。

**4B,P(complete)**

| bench | 先验血统 | n 题 | DiD 均值 | +/− | p |
|---|---|---|---|---|---|
| FrontierCS | ft01mix | 172 | +0.760 | 163/5 | <0.0001 |
| FrontierCS | lo32nm | 172 | +0.604 | 157/7 | <0.0001 |
| 科研 | ft01mix | 62 | +0.791 | 59/2 | <0.0001 |
| 科研 | lo32nm | 62 | +0.719 | 58/2 | <0.0001 |
| ALE | ft01mix | 39 | +0.672 | 37/1 | <0.0001 |
| ALE | lo32nm | 39 | +0.539 | 35/2 | <0.0001 |

**4B,mean score**:FrontierCS +4.79 / +5.38(p<0.0001),科研 +9.85 / +7.65(p=0.0002 / 0.0091),
ALE +136.9 / +125.3(p=0.0015 / 0.0005)。

**9B,P(complete),span s10→s20**:FrontierCS ft01mix +0.093(82/52,p=0.0025)、
ft03nm +0.179(99/38,p<0.0001)、lo32nm +0.040(75/53,p=0.087);ALE 三条全部不显著。

### 6.3 ★ 必须撤回的一句:这**不是**已证实的“多样性坍塌”

初看描述性数字时,4B 科研赛道 base 血统 s20 的自相似度是 0.810(先验血统 0.345-0.347),很像模式坍塌。
**但那个 0.810 只建立在 3 道题上** —— s20 的 base 血统完成率只有 7.3%,能凑出 ≥2 份可比代码的题就这么多
(FrontierCS 3/172,科研 3/63,ALE 1/39)。一条几乎不产出答案的臂,其多样性无法测量。
逐题 DiD 检验里,9B 的自相似度 12 个格子**全部不显著**(p 0.083-0.969),4B 因样本不足无法检验。

> **能站住的说法(已检验)**:创新先验保住的是 **RL 过程中“还能把答案写完”的能力**。
> 4B 上 base 血统在 s15→s20 之间失去这个能力(完成率 0.60/0.79/0.58 → 0.027/0.073/0.025),
> 先验血统没有,六个 (血统 × bench) 格子的 DiD 全部 p<0.0001;9B 上同方向但只有两条血统在 FrontierCS 上显著。
> **不能站住的说法**:先验保住了提案多样性 —— 逐题检验不支持。
>
> 所有只看最终产物的指标(技术重组、与 frontier 解池的相似度、盲标机制判定)都看不出创新优势;
> 已证实的优势在**学习动力学**里,而且形式是“不失去可执行性”,不是“更有创造力”。

---

## 7. 重组(recombination)指标

用户要的指标。做法照 Uzzi et al. 2013:把每份解映射成一个**技术家族集合**(90 个家族,正则命中;
覆盖搜索/优化、图、数据结构、DP、数论、字符串、几何、机器学习、kernel、跨领域),
把官方 Frontier-CS 解池(6769 份 frontier 模型的解)当作“既有文献”,做 200 次度数保持的随机零模型,
给每个技术对一个 z 值。逐份解输出:`n_tech`(拼了几个零件)、`P(n_tech>=2)`、
`new_pair_rate`(该解的技术对在解池中从未共现的比例)、`med_z`(常规性)、`p10_z`(原子性尾部)。
脚本 `innov_quant/recomb.py` / `recomb_stats.py`,表 `recomb_tables.md`。

| bench / 家族 | base | SFT | RL(base) | RL(SFT) |
|---|---|---|---|---|
| FrontierCS 9B,mean n_tech | 1.99 | 1.74-2.06 | 1.47 | 1.31-1.52 |
| 同上,P(>=2) | 51% | 45-52% | 40% | 34-42% |
| 科研 9B,mean n_tech | 3.87 | 3.81-3.83 | 3.14 | 3.24-3.34 |
| ALE 9B,mean n_tech | 2.75 | 2.33-2.48 | 1.61 | 1.53-1.69 |

- **RL 之后确实“更不做重组”**:9 组 RL−起点配对全部显著(p ≤ 0.015)。
- **但这不是先验的功劳**:RL(base)−base 同样显著(FrontierCS −0.65,p<0.001);SFT−base 基本都不显著。
- **长度控制后 RL(SFT) 的效应消失**(`recomb_loc.md`):按 n_tech/100LOC 算,RL(SFT)−SFT 的 p = 0.15 / 0.38 / 0.97。
  训练时的长度惩罚对各臂是一样的,但这里量的是正则在**产出代码**上的命中数,代码短了命中自然少。
  RL(base)−base 扛住了这个控制(密度 p=0.002,残差 p<0.001)。
- 剩下的组合变得**更常规**(FrontierCS / ALE 的 med_z、p10_z 上升,p 0.001-0.043),科研赛道方向相反
  (rlv5_ft03nm med_z −8.9,p=0.002)。
- `new_pair_rate` 在哪儿都很小(0.3%-4%),任何配对都不显著。
- ALE 上一个具体现象:`simulated annealing` 的使用率从 base/SFT 的 7-9% 掉到所有 RL 臂的 0-2%,
  `random_restart` 从 30-37% 掉到 14-23%。RL 之后模型在启发式题上改写确定性贪心。

---

## 8. 两个失败的假设(记下来免得重做)

1. **“先验能打开 base 打不开的题”——不成立,而且差点做出假阳性。**
   若按“对照臂 best@5 == 0”切出 hard 子集,任何臂在该子集上都会 12/0 全胜 —— 复跑臂
   `base9b_v2c_y2026 − base9b_v2c` 同样给出 +1.44(9/0,p=0.008)。这是**选择效应**:子集是按对照臂自己的抽样选的。
   改用一次**独立**的复跑定义难度后,FrontierCS 9B 的 RL 增益落在 base 已经能拿分的题上
   (easy +8.84,p=0.028;hard +0.49,n.s.)。
2. **“赢的时候靠不一样的方法赢”——不成立。** 只看成功解(该题分数 >0 且 ≥ 全臂 75 分位),
   所有臂的成功解都**更像** frontier 解池(0.234-0.291,对比各自全样本的 0.176-0.256),
   RL(先验) vs RL(base) 的差全部不显著。脚本 `innov_quant/win_novelty.py`。

---

## 9. 盲标注(733 份全部完成,标注者不知道字母对应哪条臂)

`innov_quant/agg_labels2.py`,表 `label_tables.md`。被判为“换了机制”(novel=different)的比例:

| bench | base | SFT | RL(base) | RL(SFT) | 复跑 |
|---|---|---|---|---|---|
| FrontierCS | 12% (12/104) | 8% (11/144) | 7% (5/68) | 11% (15/132) | 10% (7/72) |
| 科研 | 6% (7/108) | 6% (8/137) | 2% (2/80) | 5% (6/131) | 3% (2/70) |
| ALE | 3% (2/79) | 7% (7/94) | 9% (5/55) | 14% (13/93) | 12% (11/95) |

全部 Fisher p ≥ 0.16(最小的是 ALE 的 rl_sft vs sft,p=0.164;ALE 的复跑臂本身就是 12%,
与 rl_sft 的 14% 同一档,说明这点差别在复跑噪声之内)。标注者反复报告的主因是一侧崩溃/截断/有 bug,而不是换了机制。

---

## 10. ★ 推理过程里的探索:唯一一个先验专属、且过了复跑噪声底的创新信号

§1-§9 的所有指标都只读 `</think>` **之后**的代码。创新也可能在**搜索过程**里:定稿之前考虑了几条路、放弃了几条。
`innov_quant/explore.py` 只读 `</think>` **之前**的推理,逐样本算:

- `n_reason` 推理里点到的不同技术家族数(词表与 §7 相同)
- `n_code` 最终代码里的家族数;`n_abandon` = 考虑过但最终没用的家族数
- `explore_ratio` = n_reason / max(1, n_code)
- `n_alt` “换个思路”类标记词次数(Alternatively / Another approach / Instead of / What if / Let me reconsider …)
- 以上各量的每 10k 推理字符归一版(各臂推理长度差 3-4 倍,必须给出归一版)

### 10.1 每臂(FrontierCS 9B)

| arm | stage | 推理字符中位 | n_reason | n_code | n_abandon | explore_ratio | n_reason/10k |
|---|---|---|---|---|---|---|---|
| base9b_v2c | base | 20009 | 8.04 | 2.00 | 6.51 | 4.59 | 4.891 |
| ft01mix_a10 | sft | 17604 | 7.73 | 2.07 | 6.22 | 4.40 | 4.638 |
| ft03nm_a20 | sft | 10959 | 6.89 | 1.77 | 5.67 | 4.20 | 5.172 |
| lo32nm_a10 | sft | 15616 | 7.68 | 1.87 | 6.29 | 4.49 | 4.586 |
| rlv5_base_s20 | rl_base | 54518 | 8.40 | 1.50 | 7.02 | 5.76 | 1.659 |
| rlv5_ft01mix_a10_s20 | rl_sft | 65098 | 9.22 | 1.30 | 8.01 | 6.78 | 1.598 |
| rlv5_ft03nm_a20_s20 | rl_sft | 57014 | 8.19 | 1.53 | 6.80 | 5.60 | 1.590 |
| rlv5_lo32nm_a10_s20 | rl_sft | 66539 | 9.44 | 1.42 | 8.16 | 6.75 | 1.667 |

### 10.2 RL(先验) − RL(base):逐题配对

| bench | 先验臂 | Δ n_reason (+/−) p | Δ n_abandon p | Δ explore_ratio p |
|---|---|---|---|---|
| FrontierCS | ft01mix | +1.18 (92/32) <0.001 | +1.25 (100/29) <0.001 | +1.11 (90/40) <0.001 |
| FrontierCS | ft03nm | +0.17 (69/61) 0.309 | −0.05 (59/70) 0.676 | −0.27 (56/79) 0.069 |
| FrontierCS | lo32nm | +1.31 (98/27) <0.001 | +1.27 (94/31) <0.001 | +0.93 (93/36) <0.001 |
| 科研 | ft01mix | +0.56 (38/17) <0.001 | +0.49 (33/19) 0.004 | +0.30 (39/21) 0.007 |
| 科研 | lo32nm | +0.53 (37/19) 0.001 | +0.40 (34/25) 0.016 | +0.11 (31/29) 0.544 |
| ALE | ft01mix | +1.81 (28/8) <0.001 | +1.78 (25/11) 0.001 | +1.53 (25/12) 0.011 |
| ALE | ft03nm | +1.18 (27/11) 0.001 | +1.01 (26/10) 0.004 | +0.67 (24/13) 0.049 |
| ALE | lo32nm | +2.20 (32/4) <0.001 | +2.01 (31/6) <0.001 | +1.49 (27/9) 0.003 |

`n_reason` 上 8/9 个格子为正、7 个显著。**复跑噪声底全部不显著**:
`rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20` 在三个 bench 上 p = 0.60/0.39/0.62(n_reason),
`base9b_v2c_y2026 − base9b_v2c` 同样不显著。

### 10.3 长度控制:只有 FrontierCS 扛得住

按每 10k 推理字符归一后,RL(先验) − RL(base) 在 FrontierCS 上三条臂全部为正且显著
(+0.083 p=0.013 / +0.107 p=0.027 / +0.141 p<0.001),**在科研与 ALE 上不显著甚至为负**。
所以:「先验臂在 RL 之后考虑并放弃更多路子」在三个 bench 上都成立;
「它的探索**密度**更高」只在 FrontierCS 上成立,其余是因为它想得更长。

### 10.4 RL 之前先验看不出来;探索多预测分数高

- SFT − base 的探索指标基本全部不显著(FrontierCS 上 ft03nm 反而更少,−0.88,p<0.001)。
  **先验只有在 RL 之后才显形** —— 与 §6 的结论一致。
- 题内排秩后,`explore_ratio` 与分数正相关:FrontierCS +0.156、科研 +0.092、ALE +0.228,全部 p<0.0001;
  `n_abandon` +0.141(FrontierCS)。**考虑得多、放弃得多,分数更高。**
- 注意方向相反的一项:`n_reason/10k` 与分数**负**相关(FrontierCS −0.141,ALE −0.160)。
  探索密度高的样本往往是短推理里胡乱点名,不是真的在比较方案。

> **§10 的结论**:这是全篇唯一一个「先验专属 + 过复跑噪声底 + 三个 bench 同向」的创新指标,
> 而且它落在**推理过程**上,不在最终代码上。带创新先验的臂经过 RL 之后,在定稿前会提出并否掉更多条路子;
> 最终交出的代码却更短、技术家族更少(§7)。**多探索、少落地** —— 这与 §6「先验保住了可执行性」是同一件事的两面。

---

## 11. 跨领域词表:配对之后不成立,复跑臂正好戳破它

`experiments/scripts/agg/crossdomain_lex.py` 给的是**不配对**的汇总率,而且被 “simulated annealing” 一个词主导
(它恰恰是 ALE 启发式题的标准工具,不是跨领域迁移)。这里做配对版:逐(bench, 臂, 题)算“全文命中 ≥1 个跨领域词”
的抽样比例,两套词表(全部词 / 去掉 simulated annealing),逐题 Wilcoxon。
脚本 `innov_quant/crossdomain_paired.py`,表 `crossdomain_tables.md`。

### 11.1 FrontierCS 算法赛道 9B

含跨域词的抽样比例:base 20.6%(去 SA 后 3.6%)、三条 SFT 19.7-20.7%(2.8-3.3%)、
rlv5_base 14.6%(2.5%)、三条 RL(SFT) 14.4-19.0%(2.7-4.2%)。

| pair | 全部词 mean Δ | p | 去 SA 后 mean Δ | p |
|---|---|---|---|---|
| ft01mix − base9b | −0.006 | 0.739 | −0.004 | 0.577 |
| ft03nm − base9b | −0.009 | 0.421 | −0.007 | 0.351 |
| lo32nm − base9b | +0.001 | 0.925 | −0.008 | 0.450 |
| rlv5_lo32nm − rlv5_base | +0.044 | 0.009 | +0.017 | 0.026 |
| **[rep] rlv5_lo32nm_y2026 − rlv5_lo32nm** | +0.008 | 0.813 | **−0.013** | **0.047** |

SFT 对 base **全部不显著**。唯一一个正的(rlv5_lo32nm)旁边就站着它自己的复跑臂,
复跑臂在同一个指标上给出**同量级的反向“显著”结果**(p=0.047)。去 SA 后的基准率只有 3%,
逐题比例大多是 0,少数几道题的差就能把 Wilcoxon 推过 0.05。**这个指标的 p 值不可信。**

### 11.2 ALE 9B:复跑臂给出与 SFT 完全一样的“效应”

| pair | 去 SA 后 mean Δ | +/−/= | p |
|---|---|---|---|
| lo32nm − base9b | +0.065 | 13−3−24 | **0.011** |
| ft01mix − base9b | +0.040 | 9−3−28 | 0.059 |
| **[rep] base9b_v2c_y2026 − base9b_v2c** | **+0.062** | **11−2−26** | **0.012** |

base 臂**自己重跑一次**就能做出 +0.062(p=0.012),与 lo32nm 的 +0.065(p=0.011)同量级同显著性。
所以 ALE 上的“SFT 更跨领域”是跑间噪声。

### 11.3 科研赛道:样本量根本不够

63 道题里两臂比例有差的只有 0 到 4 道,Wilcoxon 全部算不出(p=nan)。科研赛道的解几乎不出现这些词。

> **§11 结论:没有任何跨领域词汇迁移的证据。** 不配对的汇总表之所以看着有差,
> 一是被 “simulated annealing” 主导(FrontierCS 上 20.6% 里有 17 个百分点是它),
> 二是没有噪声底做对照。加上复跑臂之后,SFT 的跨领域优势整条消失。

## 12. 可交付性审计:20 个臂 x 3 个 bench 的格子是否都跑齐了

完整表见 `innov_quant/coverage_audit.md`(由 `innov_quant/coverage_audit.py` 生成)。结论:

- **没有任何一个臂少跑过 batch。** 20 个臂在三个 bench 上的 `keys_seen` 分别恒等于 860 / 320 / 200,
  也就是 FCS 172 题 x5、research 64 题 x5、ALE 40 题 x5,一格不差。
- **缺口全部来自判题侧报错,不是生成侧。** 丢失率 FCS 1.01%、ALE 1.50%、research 4.47%;
  异常类只有三种:`RuntimeError: FrontierCS judge infrastructure failure`(230 行,集中在 143/148/153/160 这几题)、
  `ResearchInfraError`(1061 行,绝大多数被重试补回,集中在 symbolic_regression/*)、
  `AleInfraError: private_eval failed`(77 行,集中在 ahc003)。
- **整题丢失是跨臂一致的系统性坏题,不是某条臂的问题**:`symbolic_regression/sincos` 在 19/20 个臂上 5 次全失败;
  `ahc003` 在 10/20 个臂上全失败;FCS `153` 在 `ft01mix_a10` 与 `rlv5_ft03nm_a20_s20` 上全失败;
  `grammar_fuzzing/fuzzer/sql` 在两个 lo32nm 相关臂上全失败。
- **配对检验用的是公共键交集**,所以上述不平衡不会进入结论:FCS 9B 815 键 / 170 题、4B 827 键 / 172 题;
  ALE 9B 196 / 40、4B 195 / 39;research 9B 279 / 59、4B 291 / 61。research 是三者里最薄的一块。
- MLS 侧 12 个臂 x 21 题 x 两代(old / `_p1`)= 504 行全部有分,`_p1` 代无串行污染行(`stale_row` 0)、无缺轨迹行。

**仍然存在、必须在论文里写明的口径限制**(这些不是覆盖问题,是设计问题):

1. `rlv5_ft03nm_a20_s20` 的 FCS 与 ALE 是在 gpu-ee `della-i12g1`、speedFactor **0.556** 上判的,其余臂是 cpu 0.79–0.83 或 ailab 0.95–1.02。
   ALE 按墙钟计分,**这条臂的 ALE 数字不能与别人直接比**;FCS 受影响小但同样不干净。
2. 9B 的 base / 三个 SFT 臂在 cpu 分区(0.79–0.83)判,RL 臂在 ailab(~1.0)判。跨 RL 的 ALE 比较要用 `_y2026` 复跑臂
   (全部 ailab)作 base/SFT 侧。
3. 9B research 的 shard_0/shard_1 有重叠世代(同一 (题, sample_idx) 两次独立生成),当前规则取后者;42–43/140 个重复键分数不同。
4. 4B 没有 ft03nm 这条 setting;MLS 的 4B 只有 base 与 ft01mix 两条线,没有任何 lo32nm。涉及 ft03nm / 4B-lo32nm 的跨 bench 结论只能是部分覆盖。
5. 30–60% 的 RL 臂抽样在 32768 token 处被截断、没有 `</think>`;所有代码层面的分析只用完成的抽样。

## 13. 修复 research 的判题环境之后:被当成「基础设施故障」丢掉的,其实是模型幻觉 API

`frontiercs_research_cpu_eval.py:495` 的判定是 `if any(m in low for m in _INFRA_MARKERS) or proc.returncode != 0`。
后半句意味着**模型自己的代码把求值器搞崩,也算基础设施故障**,于是这些格子被从分母里剔掉,而不是记 0。

把只读树的两处写入权限问题修好(`julia_env/lock.pid`、`problems/.../sql/output_ans`,改指向 `$D/rejudge_env`
下的可写副本)之后,求值器能完整跑完模型的代码了,这些格子回来长这样:

```
`accuracy_threshold` is not a valid keyword argument for PySRRegressor
`symbolic_math`      is not a valid keyword argument for PySRRegressor
`precision_type`     is not a valid keyword argument for PySRRegressor
```

模型在编 PySR 的 API。这是实打实的 0 分。

**环境先自证**:拿同样那几道题上**已经判出 >0 分**的 8 个格子重判,8/8 全部复现非零
(mccormick 99.99999999999997 → 100.0,peaks 100.0 → 100.0)。所以"求值器挂了就怪模型"不是假设。
注意 PySR 是遗传算法、**分数不确定**(ripple 1.57 → 3.62),重判回来的 research 格子带求值器自身的随机性,
不能当成与原跑同口径。

### 13.1 幻觉 API 的分布:几乎完全是 RL 前的行为

| 阶段 | 臂数 | `model_zero` 合计 | 平均每臂 |
|---|---|---|---|
| **RL 前**(base / SFT) | 11 | **106** | 9.6 |
| **RL 后** | 9 | **1** | 0.1 |

逐臂:base9b_v2c 13、ft01mix_a10 13、ft03nm_a20 15、lo32nm_a10 14(9B 非 RL);
rlv5_base_s20 **0**、rlv5_ft01mix_a10_s20 **0**、rlv5_ft03nm_a20_s20 1、rlv5_lo32nm_a10_s20 **0**;
4B 全部 RL 臂也都是 **0**。

也就是说 **RL 基本上消灭了"瞎编库函数参数"这个行为**,而这一整类失败此前被判题口径吞掉了。

### 13.2 对 research 均分的影响:方向是对 RL 有利,但不改变先验 vs base 的结论

| 臂 | 阶段 | 旧 n | 新 n | 旧均值 | 新均值 | Δ |
|---|---|---|---|---|---|---|
| base9b_v2c | base | 305 | 320 | 10.22 | 10.37 | +0.15 |
| ft01mix_a10 | sft | 302 | 317 | 11.48 | 11.57 | +0.09 |
| ft03nm_a20 | sft | 301 | 320 | 9.54 | 10.23 | +0.68 |
| lo32nm_a10 | sft | 304 | 320 | 10.09 | 9.90 | **−0.19** |
| base9b_v2c_y2026 | rep | 298 | 319 | 9.42 | 9.11 | **−0.31** |
| rlv5_base_s20 | rl_base | 310 | 315 | 15.25 | 16.59 | +1.35 |
| rlv5_ft01mix_a10_s20 | rl_sft | 308 | 314 | 17.24 | 18.60 | +1.36 |
| rlv5_lo32nm_a10_s20 | rl_sft | 301 | 311 | 17.58 | 18.95 | +1.37 |
| rlv5_4b_base_s20 | rl_base | 313 | 318 | 7.99 | 9.13 | +1.14 |
| rlv5_4b_ft01mix_a10_s20 | rl_sft | 310 | 319 | 20.17 | 21.21 | +1.04 |

RL 臂补回来的几乎全是真分数(+1.0~+1.4),非 RL 臂补回来的一大半是 0(+0.09 ~ −0.31),
所以这个修正**把 RL 与非 RL 的差距拉得更大**——是对我们有利的方向,但它是一次纠偏,不是调参。

**关键是它不改变我们自己的那条结论**:9B SFT 阶段 ft01mix − base,旧口径 11.48 − 10.22 = **+1.26**,
新口径 11.57 − 10.37 = **+1.20**;RL 阶段 rlv5_ft01mix − rlv5_base,旧 17.24 − 15.25 = **+1.99**,
新 18.60 − 16.59 = **+2.01**。先验相对 base 的优势对这次修正是稳健的。

### 13.3 覆盖率

research 各臂从 298–313 / 320 提到 **311–320 / 320**。剩下的缺口是真·基础设施:
`vdb_pareto/*` 的 faiss `SIGABRT`(C++ 层 abort,写不出 result.json)与 `imagenet_pareto` / `llm_sql`
的 2400s 墙钟超时。这些没有被记 0,仍然留作 error 行。

## 14. idea / taste / research-judgment 三个 bench 一直在协议外采样

### 14.1 六处推理参数的对照

把 RL 训练 rollout、RL validation、主 bench、MLS、gen_client、idea_client、judge_pairwise/pointwise
七条链路的采样参数逐条读出来比对(每条都追到实际构造请求的那一行,不看 wrapper 的注释),
唯一真正的不一致是 **`presence_penalty`**:

| 参数 | RL 训练 rollout | 主 bench | gen / idea / judge3 |
|---|---|---|---|
| temperature | 1.0 | 1.0 | 1.0 ✓ |
| top_p / top_k | 0.95 / 20 | 0.95 / 20 | 0.95 / 20 ✓ |
| min_p | 0.0 | 0.0 | 服务端默认 0.0(巧合对上) |
| repetition_penalty | 1.0 | 1.0 | 服务端默认 1.0(巧合对上) |
| **presence_penalty** | **1.5** | **1.5** | **代码里没有这个参数** ✗ |
| max_tokens | 32768 | 32768 | gen 32768;idea 默认 8192,被 submitter 覆盖成 32768 |

两个本来很像问题、实测不是的,记下来免得再翻案:

- **`enable_thinking` 不是不一致。** Qwen3.5 的 chat template 默认就开思考。同一组 messages 三种渲染:
  不传 kwargs 与传 `{"enable_thinking": True}` 逐字节相同(都以 `<|im_start|>assistant\n<think>\n` 结尾),
  只有 `False` 会插入一个空 think 块。所以 gen/idea/judge 不发这个参数,拿到的就是 RL rollout 的行为;
  主 bench 显式发 `True` 是个 no-op。
- **idea 的 8192 上限不影响新跑批。** `extra_bench_submit.sh` 里 idea / ideav2 / judge3 三家都已写死
  `MAX_TOKENS=32768`;8192 只影响历史数据。

顺带查出两处不在本文范围、但会影响别的结论的:

- **RL 自己的 validation 是个混血采样。** `agent_loop.py:496-499` 在 `validate=True` 分支里重置了
  top_p / top_k / temperature,却没有重置 `presence_penalty` 和 `min_p`。所以 RL 的验证 rollout 跑的是
  `top_p=1.0, temperature=1.0` 配 `presence_penalty=1.5`,既不等于训练 rollout 也不等于评测协议。
- **MLS 的 thinking 从来没真开过。** config 写了 `thinking.enabled: true`、脚本头注释也写了 "THINKING mode ON",
  但代码门是 `is_qwen = bare_model.lower().startswith("qwen")`(`models.py:417`),而我们服务的 TAG 是
  `base9b` / `ft01mix_a10` / … 全都过不了,于是 `enable_thinking` 一次都没发出去;同一个门还管着
  `tool_choice="required"` 的抑制,所以 `tool_choice="required"` 也被强上了。

### 14.2 `presence_penalty` 就是「解析不出来」的原因

先把「解析不出来」量清楚。旧跑批(`_y26sp`,有 2026 年 system prompt、无 presence_penalty)逐臂:

| 臂 | unparsed | 中位 tokens | finish_reason=length |
|---|---|---|---|
| base9b_v2c | 0.1% | 5464 | 6 |
| base4b | 0.2% | 5385 | 15 |
| ft01mix_a10 | 0.4% | 5339 | 17 |
| 4b_ft01mix_a10 | 0.6% | 5061 | 28 |
| rlv5_base_s20 | 6.5% | 5096 | 119 |
| rlv5_ft01mix_a10_s20 | 7.2% | 5334 | 192 |
| rlv5_4b_ft01mix_a10_s20 | 11.9% | 4868 | 332 |
| **rlv5_4b_base_s20** | **38.0%** | **2862** | **0** |

不是普遍现象,是 RL 之后才出现的,而且 rlv5_4b_base_s20 是数量级离群。

重跑一遍(`_y26pp`,同 prompt、同 seed,只加 `presence_penalty=1.5`)。但重跑同时还带了新的容错 parser,
两个改动混在一起不能直接归因。拆法:**对两边都存在的 400 字 `text_tail` 跑同一个严格 parser**,
按 `(task, id, sample_idx)` 逐格配对——parser 固定,窗口固定,唯一变量就是 presence_penalty。
(注意不能拿新跑批的 `text_full` 去比:那个字段我自己截断在 20000 字,960 行里有 230–435 行到顶,
ANSWER 行被我的存储切掉了,测出来的「旧 parser 漏判」全是假的。)

`liveidea_gen` 和 `review_weakness` 是自由文本任务,本来就没有 ANSWER 行,两边恒为 100%「漏判」,
池化数字里的 33% 底噪全是任务配比造成的。真信号在两个数值任务上:

**openreview_novel + openreview_score,严格 parser,400 字窗口,逐格配对,n=960/臂:**

| 臂 | strict-miss 无 pp | 有 pp | 每答 tokens |
|---|---|---|---|
| ft01mix_a10 | 1.7% | **0.0%** | 6173 → 4695 |
| 4b_ft01mix_a10 | 2.2% | **0.9%** | 4929 → 3253 |
| rlv5_ft01mix_a10_s20 | 17.7% | **0.2%** | 10370 → 4457 |
| rlv5_4b_base_s20 | 59.4% | **4.5%** | 2753 → 2859 |

9B RL 臂的机制看得很清楚:没有 presence_penalty 时每个答案烧 **10370** tokens——就是复读,
`finish_reason=length` 也在这一列最多(192 / 1440);加上之后降到 4457,漏判 17.7% → 0.2%。

**rlv5_4b_base_s20 不是这个机制。** 它的 token 数几乎没动(2753 → 2859),`finish_reason=length`
本来就是 0,中位长度还是全部臂里最短的,漏判却从 59.4% 掉到 4.5%。它不是复读也不是截断,
是提前吐 EOS 就是不写答案行。为什么加 presence_penalty 会让它开始写,目前没有证据,不作解释。

### 14.3 这对已有数字意味着什么

**所有历史的 idea / taste / research-judgment 数字都是在协议外采到的**,而且偏差不是随机的:
RL 臂受影响远大于 base/SFT 臂(17.7% 对 1.7%),也就是说旧口径**系统性地低估了 RL 臂**——
被判 unparsed 的格子按 0 或按缺失处理,都会压 RL 臂的分。这个方向对我们的结论不利,
所以 8 条臂(base 与 ft01mix 两条线 × 4B/9B × RL 前后)四个 family 全部按对齐后的协议重跑。
旧跑批以 `_y26sp` 标签原样保留,新的是 `_y26pp`,同 prompt 同 seed,可以随时复查这次对比。

代码侧:`gen_client.py` / `idea_client.py` 加了 `--presence-penalty`(默认 1.5)、`--min-p`、
`--repetition-penalty`,参数放置逐字照抄主 bench(presence_penalty 走顶层,其余走 `extra_body`);
两个 client 启动时打印实际发出的采样协议,作业日志可审。`judge_pairwise` / `judge_pointwise`
服务的是固定裁判模型(未微调 9B),不是被测臂,`temperature=0.3` 是有意为之,没有改。
