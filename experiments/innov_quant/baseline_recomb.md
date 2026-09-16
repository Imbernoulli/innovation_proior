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
| causal-observational-nonlinear | cam | 6 |
| causal-observational-nonlinear | notears_mlp | 6 |
| ml-active-learning | badge | 6 |
| optimization-online-bandit | thompson_sampling | 6 |
| ml-calibration | isotonic_regression | 5 |
| causal-discovery-discrete | pc | 5 |
| causal-observational-linear-gaussian | pc | 5 |
| causal-observational-linear-non-gaussian | icalingam | 5 |
| causal-discovery-discrete | ges | 4 |
| ml-dimensionality-reduction | tsne | 4 |
| ml-dimensionality-reduction | pca | 4 |
| ml-anomaly-detection | isolation_forest | 4 |
| ml-active-learning | bait | 3 |
| ml-active-learning | entropy | 3 |
| ml-clustering-algorithm | dbscan | 3 |
| ml-missing-data-imputation | mean_impute | 3 |
| ml-clustering-algorithm | hdbscan | 3 |
| ml-clustering-algorithm | kmeans | 3 |
| ml-dimensionality-reduction | pacmap | 3 |
| ml-ensemble-boosting | adaboost | 3 |
| ml-subgroup-calibration-shift | temperature_scaling | 3 |
| mlsys-moe-load-balance | zigzag | 3 |
| ml-anomaly-detection | lof | 3 |
| optimization-hyperparameter-search | hyperband | 3 |
| optimization-online-bandit | kl_ucb | 3 |
| causal-discovery-discrete | hc | 2 |
| causal-observational-linear-non-gaussian | notears | 2 |
| ml-calibration | temperature_scaling | 2 |
| ml-subgroup-calibration-shift | isotonic_regression | 2 |
| ml-dimensionality-reduction | umap | 2 |
| ml-missing-data-imputation | gain | 2 |
| ml-missing-data-imputation | mice | 2 |
| ml-missing-data-imputation | miceforest | 2 |
| causal-treatment-effect | r_learner | 2 |
| ml-missing-data-imputation | missforest | 2 |
| ml-anomaly-detection | ecod | 2 |
| ml-subgroup-calibration-shift | beta_calibration | 1 |
| optimization-nas | bananas | 1 |
| causal-observational-linear-gaussian | boss | 1 |
| causal-observational-linear-gaussian | ges | 1 |

> sim 对 log(1+ntok) 的拟合:sim = +0.4394 -0.0362·log(1+ntok),n=131,r=-0.177,p=0.043。下表 `sim_resid` = sim − 拟合值。


## 2. 逐臂:模型写的方法里,点到了几条**本题给定**的 baseline

| arm | 有效题 | 动手率 | n_base(含注释) | n_hard(真调用) | P(n_hard≥1) | P(n_hard≥2) | sim_max | sim_tmpl | sim_resid | novel_frac | 新增行 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9B base | 21 | 0.90 | 0.84 | 0.58 | 0.42 | 0.16 | 0.212 | 0.165 | -0.051 | 0.966 | 102 |
| 9B SFT | 21 | 0.95 | 1.55 | 1.20 | 0.50 | 0.40 | 0.230 | 0.194 | -0.022 | 0.961 | 111 |
| 9B RL(base) | 21 | 0.90 | 1.05 | 0.68 | 0.53 | 0.11 | 0.308 | 0.174 | +0.052 | 0.881 | 98 |
| 9B RL(先验) | 20 | 0.90 | 1.06 | 0.67 | 0.44 | 0.11 | 0.332 | 0.191 | +0.059 | 0.926 | 105 |
| 4B base | 20 | 0.50 | 0.90 | 0.40 | 0.30 | 0.10 | 0.246 | 0.166 | -0.020 | 0.956 | 50 |
| 4B SFT | 21 | 0.48 | 1.40 | 1.10 | 0.50 | 0.30 | 0.231 | 0.197 | -0.035 | 0.933 | 35 |
| 4B RL(base) | 21 | 0.90 | 1.00 | 0.74 | 0.47 | 0.11 | 0.288 | 0.147 | -0.007 | 0.929 | 40 |
| 4B RL(先验) | 21 | 0.81 | 0.76 | 0.47 | 0.29 | 0.18 | 0.297 | 0.146 | +0.006 | 0.919 | 48 |

注:`动手率`=21 题里有 edit 动作的比例;没动手 = 直接交默认脚手架,方法**就是**那条默认 baseline。后面各列只在「动手了」的格子上算。

## 3. 配对对照(同题配对,两臂都动手的题才进)


### n_hard — 真 import/调用的 baseline 条数(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.125 | 4/4 | 1.0000 | 0.9416 |
| 9B RL(先验) − RL(base) | 16 | +0.000 | 2/3 | 1.0000 | 0.8907 |
| 9B RL(先验) − SFT | 17 | -0.412 | 2/5 | 0.4531 | 0.1206 |
| 4B RL(先验) − base | 9 | +0.111 | 2/1 | 1.0000 | 1.0000 |
| 4B RL(先验) − RL(base) | 16 | -0.062 | 1/2 | 1.0000 | 0.5637 |
| 4B RL(先验) − SFT | 6 | -0.333 | 1/1 | 1.0000 | 1.0000 |

### n_base — 点到的 baseline 条数,含注释(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.250 | 5/3 | 0.7266 | 0.3657 |
| 9B RL(先验) − RL(base) | 16 | +0.125 | 4/3 | 1.0000 | 0.7257 |
| 9B RL(先验) − SFT | 17 | -0.412 | 2/6 | 0.2891 | 0.1072 |
| 4B RL(先验) − base | 9 | +0.000 | 4/2 | 0.6875 | 1.0000 |
| 4B RL(先验) − RL(base) | 16 | +0.000 | 4/3 | 1.0000 | 1.0000 |
| 4B RL(先验) − SFT | 6 | -0.667 | 1/3 | 0.6250 | 0.5000 |

### sim — 与参考实现的最大 Jaccard(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.120 | 13/3 | 0.0213 | 0.0052 |
| 9B RL(先验) − RL(base) | 16 | +0.033 | 8/7 | 1.0000 | 0.4603 |
| 9B RL(先验) − SFT | 17 | +0.097 | 13/4 | 0.0490 | 0.0093 |
| 4B RL(先验) − base | 9 | +0.040 | 4/5 | 1.0000 | 0.5703 |
| 4B RL(先验) − RL(base) | 16 | +0.016 | 8/8 | 1.0000 | 0.7057 |
| 4B RL(先验) − SFT | 6 | +0.159 | 4/2 | 0.6875 | 0.1562 |

### sim_tmpl — 与空脚手架模板的 Jaccard(对照项,不是结论)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.034 | 9/7 | 0.8036 | 0.2744 |
| 9B RL(先验) − RL(base) | 16 | +0.047 | 10/5 | 0.3018 | 0.0691 |
| 9B RL(先验) − SFT | 17 | -0.006 | 7/10 | 0.6291 | 0.8536 |
| 4B RL(先验) − base | 9 | -0.022 | 4/5 | 1.0000 | 0.3594 |
| 4B RL(先验) − RL(base) | 16 | +0.003 | 8/8 | 1.0000 | 0.9399 |
| 4B RL(先验) − SFT | 6 | -0.055 | 2/4 | 0.6875 | 0.4375 |

### sim_excess — sim − sim_tmpl:超出惯用法的那部分相似(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.087 | 11/5 | 0.2101 | 0.1754 |
| 9B RL(先验) − RL(base) | 16 | -0.014 | 8/7 | 1.0000 | 0.9547 |
| 9B RL(先验) − SFT | 17 | +0.103 | 11/6 | 0.3323 | 0.0448 |
| 4B RL(先验) − base | 9 | +0.062 | 7/2 | 0.1797 | 0.1289 |
| 4B RL(先验) − RL(base) | 16 | +0.013 | 8/8 | 1.0000 | 0.9799 |
| 4B RL(先验) − SFT | 6 | +0.214 | 5/1 | 0.2188 | 0.0625 |

### sim_resid — sim 对代码长度做回归后的残差(低=好)★这条才是干净的

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.114 | 13/3 | 0.0213 | 0.0076 |
| 9B RL(先验) − RL(base) | 15 | +0.015 | 7/7 | 1.0000 | 0.6378 |
| 9B RL(先验) − SFT | 17 | +0.076 | 12/5 | 0.1435 | 0.0395 |
| 4B RL(先验) − base | 9 | +0.018 | 3/6 | 0.5078 | 0.9102 |
| 4B RL(先验) − RL(base) | 16 | +0.021 | 9/7 | 0.8036 | 0.6685 |
| 4B RL(先验) − SFT | 6 | +0.134 | 4/2 | 0.6875 | 0.2188 |

### ntok — 写进去的不同标识符数(对照项)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | -34.312 | 6/10 | 0.4545 | 0.2551 |
| 9B RL(先验) − RL(base) | 16 | +11.625 | 9/6 | 0.6072 | 0.2011 |
| 9B RL(先验) − SFT | 17 | -83.000 | 4/13 | 0.0490 | 0.0045 |
| 4B RL(先验) − base | 9 | -59.222 | 1/8 | 0.0391 | 0.0273 |
| 4B RL(先验) − RL(base) | 16 | +12.750 | 9/7 | 0.8036 | 0.2774 |
| 4B RL(先验) − SFT | 6 | -61.167 | 0/6 | 0.0312 | 0.0312 |

### novel — 不含 baseline 名的实义行占比(高=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | -0.047 | 5/7 | 0.7744 | 0.2094 |
| 9B RL(先验) − RL(base) | 16 | +0.046 | 5/6 | 1.0000 | 0.9292 |
| 9B RL(先验) − SFT | 17 | -0.042 | 2/10 | 0.0386 | 0.0096 |
| 4B RL(先验) − base | 9 | -0.088 | 2/3 | 1.0000 | 0.4375 |
| 4B RL(先验) − RL(base) | 16 | -0.031 | 4/3 | 1.0000 | 0.6121 |
| 4B RL(先验) − SFT | 6 | -0.125 | 2/2 | 1.0000 | 0.6250 |

## 4. 「方法就是 baseline」的最强形态:一行没改就提交

| arm | 有效题 | 没动手题数 | 没动手的题 |
|---|---|---|---|
| 9B base | 21 | 2 | ml-anomaly-detection, optimization-evolution-strategy |
| 9B SFT | 21 | 1 | ml-anomaly-detection |
| 9B RL(base) | 21 | 2 | causal-observational-nonlinear, ml-selective-deferral |
| 9B RL(先验) | 20 | 2 | causal-discovery-discrete, optimization-hyperparameter-search |
| 4B base | 20 | 10 | causal-observational-linear-gaussian, ml-calibration, ml-clustering-algorithm, ml-dimensionality-reduction, ml-ensemble-boosting, ml-missing-data-imputation, ml-subgroup-calibration-shift, ml-symbolic-regression, mlsys-moe-load-balance, optimization-nas |
| 4B SFT | 21 | 11 | causal-discovery-discrete, causal-treatment-effect, ml-clustering-algorithm, ml-missing-data-imputation, ml-selective-deferral, ml-subgroup-calibration-shift, ml-symbolic-regression, mlsys-moe-load-balance, optimization-evolution-strategy, optimization-nas, optimization-online-bandit |
| 4B RL(base) | 21 | 2 | ml-clustering-algorithm, ml-ensemble-boosting |
| 4B RL(先验) | 21 | 4 | causal-observational-linear-gaussian, ml-active-learning, ml-dimensionality-reduction, ml-ensemble-boosting |
