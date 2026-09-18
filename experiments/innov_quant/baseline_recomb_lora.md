# MLS:方法是不是给定 baseline 的重组 —— **把 lora 线补进来**

协议 `p1`,分母口径与 `baseline_recomb_p1.md` 同一把尺子。**低 = 好**(越不贴 baseline 越好),只有 `novel_frac` 高 = 好。

## 1. 长度混淆的那条回归(sim_resid 的来源)

Jaccard 的分母是并集 —— **写得少,分母小,sim 天然偏高**。所以 sim 必须对代码长度校正。

| 回归池 | n 格 | 截距 | 斜率 | Pearson r | p |
|---|---:|---:|---:|---:|---:|
| 本表(12 臂) | 195 | +0.4040 | -0.0295 | -0.155 | 0.03 |
| `baseline_recomb_p1.md`(原 8 臂) | 131 | +0.4517 | -0.0384 | -0.182 | 0.038 |

两套系数几乎重合,但**不完全相同**,所以本表的 `sim_resid` 与 8 臂表不逐位相等;其余各列是逐格算的,两表逐位可对。

## 2. 逐臂

| arm | 有效题 | 动手率 | n_base | n_hard | P(n_hard≥1) | P(n_hard≥2) | sim_max | sim_tmpl | sim_resid | novel_frac | 新增行 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9B base | 21 | 0.86 | 0.94 | 0.56 | 0.39 | 0.17 | 0.230 | 0.166 | -0.032 | 0.964 | 89 |
| 9B SFT | 21 | 0.95 | 1.45 | 1.00 | 0.45 | 0.35 | 0.246 | 0.212 | -0.006 | 0.961 | 113 |
| 9B RL(base) | 21 | 0.86 | 1.06 | 0.72 | 0.56 | 0.11 | 0.298 | 0.179 | +0.047 | 0.874 | 100 |
| 9B RL 我们 | 21 | 0.90 | 1.05 | 0.63 | 0.42 | 0.11 | 0.326 | 0.173 | +0.056 | 0.928 | 97 |
| 4B base | 21 | 0.52 | 0.91 | 0.45 | 0.36 | 0.09 | 0.239 | 0.161 | -0.022 | 0.955 | 53 |
| 4B SFT | 21 | 0.48 | 1.40 | 1.10 | 0.50 | 0.30 | 0.231 | 0.197 | -0.032 | 0.933 | 35 |
| 4B RL(base) | 21 | 0.90 | 1.00 | 0.74 | 0.47 | 0.11 | 0.288 | 0.147 | +0.001 | 0.929 | 40 |
| 4B RL 我们 | 21 | 0.81 | 0.76 | 0.47 | 0.29 | 0.18 | 0.297 | 0.146 | +0.014 | 0.919 | 48 |
| 9B SFT lora **←** | 21 | 1.00 | 1.14 | 0.95 | 0.52 | 0.19 | 0.220 | 0.191 | -0.028 | 0.968 | 154 |
| 9B RL lora **←** | 21 | 0.71 | 1.20 | 0.80 | 0.40 | 0.27 | 0.310 | 0.151 | +0.042 | 0.921 | 96 |
| 4B SFT lora **←** | 21 | 0.48 | 1.30 | 0.90 | 0.40 | 0.30 | 0.196 | 0.151 | -0.048 | 0.858 | 34 |
| 4B RL lora **←** | 21 | 0.90 | 0.79 | 0.53 | 0.42 | 0.11 | 0.258 | 0.164 | -0.026 | 0.968 | 47 |

`动手率` = 21 题里有 edit 动作的比例。**没动手 = 直接交默认脚手架,方法就是那条默认 baseline** —— 这是「重组」的最强形态。后面各列只在动手了的格子上算。

## 3. 配对对照(同题配对,两臂都动手的题才进)

### n_hard — 真 import/调用的 baseline 条数(低 = 好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---:|---:|:---:|---:|---:|
| 9B RL lora − RL(base) | 12 | +0.250 | 3/2 | 1.0000 | 0.4375 |
| 9B RL lora − RL 我们 | 14 | +0.214 | 3/1 | 0.6250 | 0.2568 |
| 9B RL lora − 预训练 base(旁证) | 12 | +0.250 | 3/2 | 1.0000 | 0.4375 |
| 4B RL lora − RL(base) | 17 | -0.235 | 2/3 | 1.0000 | 0.4795 |
| 4B RL lora − RL 我们 | 16 | +0.062 | 2/1 | 1.0000 | 0.5637 |
| 4B RL lora − 预训练 base(旁证) | 10 | +0.100 | 2/1 | 1.0000 | 1.0000 |
| 【控制】9B RL(base) − 预训练 base | 16 | +0.250 | 5/1 | 0.2188 | 0.2795 |
| 【控制】4B RL(base) − 预训练 base | 11 | +0.273 | 3/1 | 0.6250 | 0.5000 |

### n_base — 点到的 baseline 条数,含注释(低 = 好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---:|---:|:---:|---:|---:|
| 9B RL lora − RL(base) | 12 | +0.417 | 6/3 | 0.5078 | 0.2578 |
| 9B RL lora − RL 我们 | 14 | +0.214 | 6/3 | 0.5078 | 0.3173 |
| 9B RL lora − 预训练 base(旁证) | 12 | +0.500 | 5/1 | 0.2188 | 0.1562 |
| 4B RL lora − RL(base) | 17 | -0.235 | 4/4 | 1.0000 | 0.5214 |
| 4B RL lora − RL 我们 | 16 | +0.062 | 5/3 | 0.7266 | 0.7630 |
| 4B RL lora − 预训练 base(旁证) | 10 | +0.200 | 4/2 | 0.6875 | 0.7812 |
| 【控制】9B RL(base) − 预训练 base | 16 | +0.188 | 5/3 | 0.7266 | 0.4705 |
| 【控制】4B RL(base) − 预训练 base | 11 | +0.091 | 3/2 | 1.0000 | 0.9375 |

### sim — 与参考实现的最大 Jaccard(低 = 好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---:|---:|:---:|---:|---:|
| 9B RL lora − RL(base) | 12 | +0.001 | 6/6 | 1.0000 | 0.7910 |
| 9B RL lora − RL 我们 | 14 | -0.002 | 6/8 | 0.7905 | 0.9515 |
| 9B RL lora − 预训练 base(旁证) | 12 | +0.087 | 11/1 | 0.0063 | 0.0210 |
| 4B RL lora − RL(base) | 17 | -0.012 | 8/9 | 1.0000 | 0.5477 |
| 4B RL lora − RL 我们 | 16 | -0.007 | 7/9 | 0.8036 | 0.8209 |
| 4B RL lora − 预训练 base(旁证) | 10 | +0.057 | 8/2 | 0.1094 | 0.0840 |
| 【控制】9B RL(base) − 预训练 base | 16 | +0.101 | 12/4 | 0.0768 | 0.0182 |
| 【控制】4B RL(base) − 预训练 base | 11 | +0.031 | 9/2 | 0.0654 | 0.1748 |

### sim_excess — sim − sim_tmpl:超出惯用法的那部分相似(低 = 好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---:|---:|:---:|---:|---:|
| 9B RL lora − RL(base) | 12 | -0.005 | 4/8 | 0.3877 | 0.7334 |
| 9B RL lora − RL 我们 | 14 | +0.010 | 7/7 | 1.0000 | 0.9032 |
| 9B RL lora − 预训练 base(旁证) | 12 | +0.082 | 6/6 | 1.0000 | 0.2036 |
| 4B RL lora − RL(base) | 17 | -0.048 | 6/11 | 0.3323 | 0.2633 |
| 4B RL lora − RL 我们 | 16 | -0.038 | 6/10 | 0.4545 | 0.2744 |
| 4B RL lora − 预训练 base(旁证) | 10 | +0.021 | 6/4 | 0.7539 | 0.5566 |
| 【控制】9B RL(base) − 预训练 base | 16 | +0.080 | 12/4 | 0.0768 | 0.1167 |
| 【控制】4B RL(base) − 预训练 base | 11 | +0.045 | 8/3 | 0.2266 | 0.2783 |

### sim_resid — sim 对代码长度回归后的残差 ★这条才是干净的(低 = 好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---:|---:|:---:|---:|---:|
| 9B RL lora − RL(base) | 11 | -0.008 | 5/6 | 1.0000 | 0.8984 |
| 9B RL lora − RL 我们 | 14 | -0.004 | 6/8 | 0.7905 | 0.8552 |
| 9B RL lora − 预训练 base(旁证) | 12 | +0.084 | 11/1 | 0.0063 | 0.0161 |
| 4B RL lora − RL(base) | 17 | -0.007 | 8/9 | 1.0000 | 0.7467 |
| 4B RL lora − RL 我们 | 16 | -0.005 | 7/9 | 0.8036 | 0.8603 |
| 4B RL lora − 预训练 base(旁证) | 10 | +0.037 | 7/3 | 0.3438 | 0.1602 |
| 【控制】9B RL(base) − 预训练 base | 16 | +0.096 | 12/4 | 0.0768 | 0.0250 |
| 【控制】4B RL(base) − 预训练 base | 11 | +0.006 | 5/6 | 1.0000 | 0.7646 |

### novel — 不含 baseline 名的实义行占比(高 = 好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---:|---:|:---:|---:|---:|
| 9B RL lora − RL(base) | 12 | +0.041 | 4/4 | 1.0000 | 0.7422 |
| 9B RL lora − RL 我们 | 14 | -0.038 | 5/4 | 1.0000 | 0.6784 |
| 9B RL lora − 预训练 base(旁证) | 12 | -0.006 | 2/5 | 0.4531 | 0.5781 |
| 4B RL lora − RL(base) | 17 | +0.018 | 6/4 | 0.7539 | 0.3329 |
| 4B RL lora − RL 我们 | 16 | +0.042 | 3/3 | 1.0000 | 0.7532 |
| 4B RL lora − 预训练 base(旁证) | 10 | -0.010 | 2/4 | 0.6875 | 0.6875 |
| 【控制】9B RL(base) − 预训练 base | 16 | -0.036 | 3/8 | 0.2266 | 0.0619 |
| 【控制】4B RL(base) − 预训练 base | 11 | -0.041 | 1/7 | 0.0703 | 0.0234 |

### sim_tmpl — 与空脚手架模板的 Jaccard(对照项)(— = 好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---:|---:|:---:|---:|---:|
| 9B RL lora − RL(base) | 12 | +0.006 | 7/5 | 0.7744 | 0.6221 |
| 9B RL lora − RL 我们 | 14 | -0.011 | 6/8 | 0.7905 | 0.5416 |
| 9B RL lora − 预训练 base(旁证) | 12 | +0.005 | 7/5 | 0.7744 | 0.7910 |
| 4B RL lora − RL(base) | 17 | +0.037 | 9/8 | 1.0000 | 0.3060 |
| 4B RL lora − RL 我们 | 16 | +0.031 | 8/8 | 1.0000 | 0.5282 |
| 4B RL lora − 预训练 base(旁证) | 10 | +0.036 | 5/5 | 1.0000 | 0.4316 |
| 【控制】9B RL(base) − 预训练 base | 16 | +0.021 | 10/6 | 0.4545 | 0.2312 |
| 【控制】4B RL(base) − 预训练 base | 11 | -0.014 | 3/8 | 0.2266 | 0.3652 |

### ntok — 写进去的不同标识符数(对照项)(— = 好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---:|---:|:---:|---:|---:|
| 9B RL lora − RL(base) | 12 | +24.917 | 8/4 | 0.3877 | 0.2334 |
| 9B RL lora − RL 我们 | 14 | +1.857 | 8/6 | 0.7905 | 0.9750 |
| 9B RL lora − 预训练 base(旁证) | 12 | -8.333 | 5/7 | 0.7744 | 0.8501 |
| 4B RL lora − RL(base) | 17 | +14.353 | 9/8 | 1.0000 | 0.4038 |
| 4B RL lora − RL 我们 | 16 | +4.375 | 10/6 | 0.4545 | 0.4690 |
| 4B RL lora − 预训练 base(旁证) | 10 | -68.500 | 1/9 | 0.0215 | 0.0059 |
| 【控制】9B RL(base) − 预训练 base | 16 | -27.438 | 4/12 | 0.0768 | 0.1337 |
| 【控制】4B RL(base) − 预训练 base | 11 | -78.273 | 2/9 | 0.0654 | 0.0059 |

## 4. 最强形态:一行没改就提交(方法**就是**默认 baseline)

| arm | 有效题 | 没动手题数 | 没动手的题 |
|---|---:|---:|---|
| 9B base | 21 | 3 | causal-treatment-effect, ml-anomaly-detection, optimization-evolution-strategy |
| 9B SFT | 21 | 1 | ml-anomaly-detection |
| 9B RL(base) | 21 | 3 | causal-observational-nonlinear, causal-treatment-effect, ml-selective-deferral |
| 9B RL 我们 | 21 | 2 | causal-discovery-discrete, optimization-hyperparameter-search |
| 4B base | 21 | 10 | causal-observational-linear-gaussian, ml-calibration, ml-clustering-algorithm, ml-dimensionality-reduction, ml-ensemble-boosting, ml-missing-data-imputation, ml-subgroup-calibration-shift, ml-symbolic-regression, mlsys-moe-load-balance, optimization-nas |
| 4B SFT | 21 | 11 | causal-discovery-discrete, causal-treatment-effect, ml-clustering-algorithm, ml-missing-data-imputation, ml-selective-deferral, ml-subgroup-calibration-shift, ml-symbolic-regression, mlsys-moe-load-balance, optimization-evolution-strategy, optimization-nas, optimization-online-bandit |
| 4B RL(base) | 21 | 2 | ml-clustering-algorithm, ml-ensemble-boosting |
| 4B RL 我们 | 21 | 4 | causal-observational-linear-gaussian, ml-active-learning, ml-dimensionality-reduction, ml-ensemble-boosting |
| 9B SFT lora | 21 | 0 | — |
| 9B RL lora | 21 | 6 | causal-observational-linear-gaussian, causal-observational-linear-non-gaussian, ml-active-learning, ml-clustering-algorithm, ml-missing-data-imputation, optimization-hyperparameter-search |
| 4B SFT lora | 21 | 11 | causal-observational-linear-gaussian, causal-treatment-effect, ml-active-learning, ml-calibration, ml-clustering-algorithm, ml-dimensionality-reduction, ml-ensemble-boosting, ml-selective-deferral, ml-subgroup-calibration-shift, ml-symbolic-regression, mlsys-moe-load-balance |
| 4B RL lora | 21 | 2 | causal-discovery-discrete, causal-observational-linear-gaussian |

## 5. 一眼表:lora 在「不抄 baseline」上比对照好还是差

每格 = Δ均值,并按该指标的方向翻译成 **好 / 差**。★ = Wilcoxon p < 0.05。

| 指标 | 9B RL lora − RL(base) | 9B RL lora − RL 我们 | 9B RL lora − 预训练 base(旁证) | 4B RL lora − RL(base) | 4B RL lora − RL 我们 | 4B RL lora − 预训练 base(旁证) | 【控制】9B RL(base) − 预训练 base | 【控制】4B RL(base) − 预训练 base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n_hard | +0.250 差 | +0.214 差 | +0.250 差 | -0.235 好 | +0.062 差 | +0.100 差 | +0.250 差 | +0.273 差 |
| n_base | +0.417 差 | +0.214 差 | +0.500 差 | -0.235 好 | +0.062 差 | +0.200 差 | +0.188 差 | +0.091 差 |
| sim | +0.001 差 | -0.002 好 | +0.087 差★ | -0.012 好 | -0.007 好 | +0.057 差 | +0.101 差★ | +0.031 差 |
| sim_excess | -0.005 好 | +0.010 差 | +0.082 差 | -0.048 好 | -0.038 好 | +0.021 差 | +0.080 差 | +0.045 差 |
| sim_resid | -0.008 好 | -0.004 好 | +0.084 差★ | -0.007 好 | -0.005 好 | +0.037 差 | +0.096 差★ | +0.006 差 |
| novel | +0.041 好 | -0.038 差 | -0.006 差 | +0.018 好 | +0.042 好 | -0.010 差 | -0.036 差 | -0.041 差★ |

## 6. 读法

**(a) 对 RL 之后的两条对照,lora 在任何一个重组指标上都没有可测差异。** 12 个格子(2 尺寸 × 2 对照 × 6 指标 的前四列)没有一颗 ★。干净的那条 `sim_resid` 四格全是「好」方向(−0.004 ~ −0.008),但都远不显著。**结论是「打平」,不是「更好」** —— 想说 lora 更不抄 baseline,这份数据给不了。

**(b) 相对预训练 base 贴基线更紧,是 RL 的效应,不是 lora 的。** 9B RL lora − 预训练 base 的 `sim_resid` = +0.084 ★,看着像个坏消息;但控制行 9B RL(base) − 预训练 base = **+0.096 ★**,比 lora 还大。也就是说 RL 本身会让代码更贴题目给的参考实现,而**lora 是这两条 RL 臂里受影响较小的那条**。这和已记录的老结论一致(RL 后代码更贴基线、更短、不更新颖)。

**(c) 最强形态那一列要分尺寸看。** 「一行没改就提交」= 方法直接就是默认 baseline:4B RL lora **2/21**,与 4B RL(base) 并列全场最好,优于 4B RL 我们(4/21);而 9B RL lora **6/21** 是三条 9B RL 臂里最差的(RL(base) 3、我们 2)。这一条对 9B 不利,照记。

