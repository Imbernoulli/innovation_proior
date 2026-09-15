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

## 9. 盲标注(699 份,标注者不知道字母对应哪条臂)

`innov_quant/agg_labels2.py`,表 `label_tables.md`。被判为“换了机制”(novel=different)的比例:

| bench | base | SFT | RL(base) | RL(SFT) | 复跑 |
|---|---|---|---|---|---|
| FrontierCS | 12% | 8% | 7% | 11% | 10% |
| 科研 | 6% | 6% | 2% | 5% | 3% |
| ALE | 3% | 8% | 5% | 11% | 10% |

全部 Fisher p > 0.18。标注者反复报告的主因是一侧崩溃/截断/有 bug,而不是换了机制。
