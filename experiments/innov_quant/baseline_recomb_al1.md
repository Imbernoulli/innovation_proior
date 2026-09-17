## 0. 每道题给了几条 baseline(= tasks/<t>/edits/*.edit.py)

| task | n_baseline | baselines |
|---|---|---|
| causal-discovery-discrete | 5 | boss, ges, grasp, hc, pc |
| causal-observational-linear-gaussian ⚠缺包 | 4 | boss, ges, grasp, pc |
| causal-observational-linear-non-gaussian | 4 | directlingam, icalingam, notears, rcd |
| causal-observational-nonlinear | 4 | cam, directlingam, grandag, notears_mlp |
| causal-treatment-effect | 6 | causal_forest, dr_learner, ipw, r_learner, s_learner, t_learner |
| ml-active-learning | 5 | badge, bait, bald, entropy, least_confidence |
| ml-anomaly-detection | 6 | copod, dif, ecod, isolation_forest, lof, ocsvm |
| ml-calibration | 6 | beta_calibration, histogram_binning, isotonic_regression, platt_scaling, spline_calibration, temperature_scaling |
| ml-clustering-algorithm | 6 | agglomerative, dbscan, dpc, hdbscan, kmeans, spectral |
| ml-dimensionality-reduction | 5 | pacmap, pca, trimap, tsne, umap |
| ml-ensemble-boosting | 4 | adaboost, gradient_boosting, hist_gradient_boosting, xgboost_style |
| ml-missing-data-imputation | 6 | gain, knn, mean_impute, mice, miceforest, missforest |
| ml-selective-deferral | 4 | confidence_thresholding, conformal_abstention, groupwise_thresholding, learned_deferral |
| ml-subgroup-calibration-shift | 4 | beta_calibration, group_temperature_scaling, isotonic_regression, temperature_scaling |
| ml-symbolic-regression | 3 | lexicase_gp, parsimony_gp, standard_gp |
| mlsys-moe-load-balance | 3 | full_tensor, greedy, zigzag |
| optimization-evolution-strategy | 4 | cmaes, de, ga_sbx, lshade |
| optimization-hyperparameter-search | 6 | bohb, dehb, hyperband, optuna_cma, random_search, tpe |
| optimization-multi-objective ⚠缺包 | 6 | agemoea, moead, nsga2, nsga3, rvea, spea2 |
| optimization-nas | 3 | bananas, random_search, rea |
| optimization-online-bandit | 3 | kl_ucb, thompson_sampling, ucb1 |

## 1. 词表命中自检(哪条 baseline 被认出来过几次,8 臂 x 21 题 = 168 格)

| task | baseline | 命中格数 |
|---|---|---|
| causal-observational-nonlinear | cam | 7 |
| causal-observational-nonlinear | notears_mlp | 7 |
| ml-active-learning | entropy | 6 |
| causal-discovery-discrete | pc | 5 |
| causal-observational-linear-gaussian | pc | 5 |
| ml-clustering-algorithm | kmeans | 5 |
| ml-dimensionality-reduction | tsne | 5 |
| ml-missing-data-imputation | mice | 5 |
| mlsys-moe-load-balance | greedy | 5 |
| mlsys-moe-load-balance | zigzag | 5 |
| optimization-online-bandit | kl_ucb | 5 |
| ml-anomaly-detection | copod | 5 |
| causal-observational-linear-non-gaussian | icalingam | 5 |
| causal-discovery-discrete | hc | 4 |
| causal-observational-linear-non-gaussian | directlingam | 4 |
| causal-treatment-effect | r_learner | 4 |
| ml-anomaly-detection | isolation_forest | 4 |
| ml-calibration | isotonic_regression | 4 |
| ml-calibration | temperature_scaling | 4 |
| ml-clustering-algorithm | hdbscan | 4 |
| ml-subgroup-calibration-shift | isotonic_regression | 4 |
| optimization-hyperparameter-search | tpe | 4 |
| optimization-online-bandit | thompson_sampling | 4 |
| ml-active-learning | badge | 4 |
| ml-anomaly-detection | ecod | 4 |
| ml-anomaly-detection | lof | 4 |
| ml-dimensionality-reduction | pacmap | 4 |
| causal-observational-linear-non-gaussian | notears | 3 |
| ml-ensemble-boosting | adaboost | 3 |
| ml-missing-data-imputation | gain | 3 |
| optimization-online-bandit | ucb1 | 3 |
| ml-clustering-algorithm | spectral | 3 |
| ml-dimensionality-reduction | pca | 3 |
| ml-subgroup-calibration-shift | temperature_scaling | 3 |
| ml-calibration | platt_scaling | 2 |
| ml-missing-data-imputation | miceforest | 2 |
| optimization-hyperparameter-search | hyperband | 2 |
| causal-discovery-discrete | ges | 2 |
| ml-active-learning | bald | 2 |
| ml-anomaly-detection | ocsvm | 2 |

> sim 对 log(1+ntok) 的拟合:sim = +0.5308 -0.0556·log(1+ntok),n=142,r=-0.260,p=0.0018。下表 `sim_resid` = sim − 拟合值。


## 2. 逐臂:模型写的方法里,点到了几条**本题给定**的 baseline

| arm | 有效题 | 动手率 | n_base(含注释) | n_hard(真调用) | P(n_hard≥1) | P(n_hard≥2) | sim_max | sim_tmpl | sim_resid | novel_frac | 新增行 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9B base | 21 | 1.00 | 1.62 | 0.90 | 0.52 | 0.29 | 0.239 | 0.144 | +0.001 | 0.935 | 126 |
| 9B SFT | 21 | 0.95 | 1.60 | 1.05 | 0.40 | 0.30 | 0.231 | 0.165 | -0.015 | 0.949 | 114 |
| 9B RL(base) | 21 | 0.90 | 1.58 | 0.89 | 0.58 | 0.26 | 0.330 | 0.179 | +0.054 | 0.927 | 77 |
| 9B RL(先验) | 21 | 0.90 | 1.42 | 0.68 | 0.42 | 0.11 | 0.310 | 0.185 | +0.038 | 0.944 | 100 |
| 4B base | 21 | 0.76 | 1.12 | 0.81 | 0.38 | 0.25 | 0.224 | 0.156 | -0.054 | 0.951 | 29 |
| 4B SFT | 21 | 0.57 | 0.83 | 0.42 | 0.17 | 0.17 | 0.191 | 0.132 | -0.085 | 0.972 | 38 |
| 4B RL(base) | 21 | 0.76 | 1.12 | 0.69 | 0.44 | 0.25 | 0.333 | 0.150 | +0.030 | 0.917 | 44 |
| 4B RL(先验) | 21 | 0.90 | 1.05 | 0.68 | 0.47 | 0.21 | 0.280 | 0.184 | -0.004 | 0.903 | 43 |

注:`动手率`=21 题里有 edit 动作的比例;没动手 = 直接交默认脚手架,方法**就是**那条默认 baseline。后面各列只在「动手了」的格子上算。

## 3. 配对对照(同题配对,两臂都动手的题才进)


### n_hard — 真 import/调用的 baseline 条数(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 19 | -0.263 | 3/6 | 0.5078 | 0.3079 |
| 9B RL(先验) − RL(base) | 17 | -0.176 | 1/5 | 0.2188 | 0.3318 |
| 9B RL(先验) − SFT | 19 | -0.421 | 2/6 | 0.2891 | 0.1067 |
| 4B RL(先验) − base | 16 | -0.188 | 2/2 | 1.0000 | 0.4615 |
| 4B RL(先验) − RL(base) | 14 | +0.214 | 2/0 | 0.5000 | 0.1797 |
| 4B RL(先验) − SFT | 11 | +0.091 | 3/1 | 0.6250 | 1.0000 |

### n_base — 点到的 baseline 条数,含注释(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 19 | -0.211 | 3/8 | 0.2266 | 0.2272 |
| 9B RL(先验) − RL(base) | 17 | -0.412 | 3/8 | 0.2266 | 0.2763 |
| 9B RL(先验) − SFT | 19 | -0.263 | 4/7 | 0.5488 | 0.4130 |
| 4B RL(先验) − base | 16 | -0.188 | 5/5 | 1.0000 | 0.7108 |
| 4B RL(先验) − RL(base) | 14 | -0.071 | 3/3 | 1.0000 | 0.7389 |
| 4B RL(先验) − SFT | 11 | +0.000 | 2/2 | 1.0000 | 1.0000 |

### sim — 与参考实现的最大 Jaccard(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 19 | +0.086 | 15/4 | 0.0192 | 0.0028 |
| 9B RL(先验) − RL(base) | 17 | +0.011 | 10/6 | 0.4545 | 0.3520 |
| 9B RL(先验) − SFT | 19 | +0.081 | 12/7 | 0.3593 | 0.0289 |
| 4B RL(先验) − base | 16 | +0.056 | 10/6 | 0.4545 | 0.1439 |
| 4B RL(先验) − RL(base) | 14 | -0.028 | 8/6 | 0.7905 | 0.8077 |
| 4B RL(先验) − SFT | 11 | +0.064 | 9/2 | 0.0654 | 0.0322 |

### sim_tmpl — 与空脚手架模板的 Jaccard(对照项,不是结论)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 19 | +0.038 | 11/8 | 0.6476 | 0.2935 |
| 9B RL(先验) − RL(base) | 17 | +0.004 | 10/7 | 0.6291 | 0.7467 |
| 9B RL(先验) − SFT | 19 | +0.029 | 14/5 | 0.0636 | 0.1564 |
| 4B RL(先验) − base | 16 | +0.026 | 8/8 | 1.0000 | 0.7057 |
| 4B RL(先验) − RL(base) | 14 | +0.046 | 9/5 | 0.4240 | 0.2412 |
| 4B RL(先验) − SFT | 11 | +0.079 | 8/3 | 0.2266 | 0.0420 |

### sim_excess — sim − sim_tmpl:超出惯用法的那部分相似(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 19 | +0.049 | 12/7 | 0.3593 | 0.1956 |
| 9B RL(先验) − RL(base) | 17 | +0.007 | 10/7 | 0.6291 | 0.7119 |
| 9B RL(先验) − SFT | 19 | +0.051 | 12/7 | 0.3593 | 0.1232 |
| 4B RL(先验) − base | 16 | +0.030 | 9/7 | 0.8036 | 0.6685 |
| 4B RL(先验) − RL(base) | 14 | -0.074 | 3/11 | 0.0574 | 0.0580 |
| 4B RL(先验) − SFT | 11 | -0.015 | 6/5 | 1.0000 | 1.0000 |

### sim_resid — sim 对代码长度做回归后的残差(低=好)★这条才是干净的

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 19 | +0.050 | 13/6 | 0.1671 | 0.0955 |
| 9B RL(先验) − RL(base) | 17 | +0.010 | 11/6 | 0.3323 | 0.4874 |
| 9B RL(先验) − SFT | 19 | +0.056 | 12/7 | 0.3593 | 0.1232 |
| 4B RL(先验) − base | 16 | +0.049 | 10/6 | 0.4545 | 0.2114 |
| 4B RL(先验) − RL(base) | 14 | -0.014 | 8/6 | 0.7905 | 1.0000 |
| 4B RL(先验) − SFT | 11 | +0.057 | 9/2 | 0.0654 | 0.0674 |

### ntok — 写进去的不同标识符数(对照项)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 19 | -106.316 | 1/18 | 0.0001 | 0.0005 |
| 9B RL(先验) − RL(base) | 17 | -9.059 | 8/9 | 1.0000 | 0.8536 |
| 9B RL(先验) − SFT | 19 | -57.000 | 4/15 | 0.0192 | 0.0028 |
| 4B RL(先验) − base | 16 | -3.250 | 8/8 | 1.0000 | 0.9799 |
| 4B RL(先验) − RL(base) | 14 | +23.071 | 10/4 | 0.1796 | 0.0785 |
| 4B RL(先验) − SFT | 11 | -9.000 | 4/7 | 0.5488 | 0.7188 |

### novel — 不含 baseline 名的实义行占比(高=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 19 | -0.002 | 10/7 | 0.6291 | 0.9811 |
| 9B RL(先验) − RL(base) | 17 | +0.007 | 5/7 | 0.7744 | 0.8753 |
| 9B RL(先验) − SFT | 19 | -0.002 | 4/9 | 0.2668 | 0.5525 |
| 4B RL(先验) − base | 16 | -0.047 | 4/7 | 0.5488 | 0.1971 |
| 4B RL(先验) − RL(base) | 14 | -0.057 | 3/7 | 0.3438 | 0.3863 |
| 4B RL(先验) − SFT | 11 | -0.018 | 3/4 | 1.0000 | 0.5781 |

## 4. 「方法就是 baseline」的最强形态:一行没改就提交

| arm | 有效题 | 没动手题数 | 没动手的题 |
|---|---|---|---|
| 9B base | 21 | 0 | — |
| 9B SFT | 21 | 1 | causal-observational-linear-gaussian |
| 9B RL(base) | 21 | 2 | causal-observational-linear-non-gaussian, ml-missing-data-imputation |
| 9B RL(先验) | 21 | 2 | causal-observational-linear-gaussian, optimization-multi-objective |
| 4B base | 21 | 5 | causal-discovery-discrete, ml-clustering-algorithm, ml-dimensionality-reduction, ml-ensemble-boosting, optimization-hyperparameter-search |
| 4B SFT | 21 | 9 | causal-observational-linear-gaussian, causal-treatment-effect, ml-anomaly-detection, ml-clustering-algorithm, ml-dimensionality-reduction, ml-missing-data-imputation, ml-selective-deferral, ml-subgroup-calibration-shift, optimization-online-bandit |
| 4B RL(base) | 21 | 5 | causal-treatment-effect, ml-calibration, mlsys-moe-load-balance, optimization-hyperparameter-search, optimization-nas |
| 4B RL(先验) | 21 | 2 | causal-discovery-discrete, ml-clustering-algorithm |
