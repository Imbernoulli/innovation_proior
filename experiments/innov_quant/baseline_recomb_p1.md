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
| causal-discovery-discrete | pc | 6 |
| causal-observational-nonlinear | cam | 6 |
| causal-observational-nonlinear | notears_mlp | 6 |
| ml-active-learning | badge | 6 |
| optimization-online-bandit | thompson_sampling | 6 |
| ml-calibration | isotonic_regression | 5 |
| causal-observational-linear-gaussian | pc | 5 |
| causal-observational-linear-non-gaussian | icalingam | 5 |
| causal-discovery-discrete | ges | 4 |
| optimization-hyperparameter-search | hyperband | 4 |
| ml-dimensionality-reduction | pca | 4 |
| ml-anomaly-detection | isolation_forest | 4 |
| ml-active-learning | bait | 3 |
| ml-active-learning | entropy | 3 |
| ml-clustering-algorithm | dbscan | 3 |
| ml-missing-data-imputation | mean_impute | 3 |
| ml-clustering-algorithm | hdbscan | 3 |
| ml-clustering-algorithm | kmeans | 3 |
| ml-dimensionality-reduction | pacmap | 3 |
| ml-dimensionality-reduction | tsne | 3 |
| ml-ensemble-boosting | adaboost | 3 |
| ml-subgroup-calibration-shift | temperature_scaling | 3 |
| mlsys-moe-load-balance | zigzag | 3 |
| ml-anomaly-detection | lof | 3 |
| optimization-online-bandit | kl_ucb | 3 |
| causal-observational-linear-non-gaussian | notears | 2 |
| ml-calibration | temperature_scaling | 2 |
| ml-subgroup-calibration-shift | isotonic_regression | 2 |
| optimization-hyperparameter-search | tpe | 2 |
| ml-dimensionality-reduction | umap | 2 |
| ml-missing-data-imputation | gain | 2 |
| ml-missing-data-imputation | mice | 2 |
| ml-missing-data-imputation | miceforest | 2 |
| optimization-multi-objective | nsga3 | 2 |
| ml-missing-data-imputation | missforest | 2 |
| ml-anomaly-detection | ecod | 2 |
| ml-subgroup-calibration-shift | beta_calibration | 1 |
| causal-discovery-discrete | hc | 1 |
| ml-calibration | platt_scaling | 1 |
| ml-missing-data-imputation | knn | 1 |

> sim 对 log(1+ntok) 的拟合:sim = +0.4517 -0.0384·log(1+ntok),n=131,r=-0.182,p=0.038。下表 `sim_resid` = sim − 拟合值。


## 2. 逐臂:模型写的方法里,点到了几条**本题给定**的 baseline

| arm | 有效题 | 动手率 | n_base(含注释) | n_hard(真调用) | P(n_hard≥1) | P(n_hard≥2) | sim_max | sim_tmpl | sim_resid | novel_frac | 新增行 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9B base | 21 | 0.86 | 0.94 | 0.56 | 0.39 | 0.17 | 0.230 | 0.166 | -0.037 | 0.964 | 89 |
| 9B SFT | 21 | 0.95 | 1.45 | 1.00 | 0.45 | 0.35 | 0.246 | 0.212 | -0.007 | 0.961 | 113 |
| 9B RL(base) | 21 | 0.86 | 1.06 | 0.72 | 0.56 | 0.11 | 0.298 | 0.179 | +0.041 | 0.874 | 100 |
| 9B RL(先验) | 21 | 0.90 | 1.05 | 0.63 | 0.42 | 0.11 | 0.326 | 0.173 | +0.049 | 0.928 | 97 |
| 4B base | 21 | 0.52 | 0.91 | 0.45 | 0.36 | 0.09 | 0.239 | 0.161 | -0.026 | 0.955 | 53 |
| 4B SFT | 21 | 0.48 | 1.40 | 1.10 | 0.50 | 0.30 | 0.231 | 0.197 | -0.037 | 0.933 | 35 |
| 4B RL(base) | 21 | 0.90 | 1.00 | 0.74 | 0.47 | 0.11 | 0.288 | 0.147 | -0.011 | 0.929 | 40 |
| 4B RL(先验) | 21 | 0.81 | 0.76 | 0.47 | 0.29 | 0.18 | 0.297 | 0.146 | +0.002 | 0.919 | 48 |

注:`动手率`=21 题里有 edit 动作的比例;没动手 = 直接交默认脚手架,方法**就是**那条默认 baseline。后面各列只在「动手了」的格子上算。

## 3. 配对对照(同题配对,两臂都动手的题才进)


### n_hard — 真 import/调用的 baseline 条数(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.188 | 4/3 | 1.0000 | 0.7257 |
| 9B RL(先验) − RL(base) | 16 | -0.062 | 1/3 | 0.6250 | 0.7127 |
| 9B RL(先验) − SFT | 18 | -0.333 | 2/5 | 0.4531 | 0.1186 |
| 4B RL(先验) − base | 10 | +0.100 | 2/1 | 1.0000 | 1.0000 |
| 4B RL(先验) − RL(base) | 16 | -0.062 | 1/2 | 1.0000 | 0.5637 |
| 4B RL(先验) − SFT | 6 | -0.333 | 1/1 | 1.0000 | 1.0000 |

### n_base — 点到的 baseline 条数,含注释(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.375 | 5/2 | 0.4531 | 0.1605 |
| 9B RL(先验) − RL(base) | 16 | +0.125 | 3/2 | 1.0000 | 0.6803 |
| 9B RL(先验) − SFT | 18 | -0.389 | 2/7 | 0.1797 | 0.0702 |
| 4B RL(先验) − base | 10 | +0.000 | 4/2 | 0.6875 | 1.0000 |
| 4B RL(先验) − RL(base) | 16 | +0.000 | 4/3 | 1.0000 | 1.0000 |
| 4B RL(先验) − SFT | 6 | -0.667 | 1/3 | 0.6250 | 0.5000 |

### sim — 与参考实现的最大 Jaccard(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.112 | 12/4 | 0.0768 | 0.0182 |
| 9B RL(先验) − RL(base) | 16 | +0.034 | 8/7 | 1.0000 | 0.3635 |
| 9B RL(先验) − SFT | 18 | +0.082 | 15/3 | 0.0075 | 0.0047 |
| 4B RL(先验) − base | 10 | +0.082 | 5/5 | 1.0000 | 0.3223 |
| 4B RL(先验) − RL(base) | 16 | +0.016 | 8/8 | 1.0000 | 0.7057 |
| 4B RL(先验) − SFT | 6 | +0.159 | 4/2 | 0.6875 | 0.1562 |

### sim_tmpl — 与空脚手架模板的 Jaccard(对照项,不是结论)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.018 | 9/7 | 0.8036 | 0.3225 |
| 9B RL(先验) − RL(base) | 16 | +0.026 | 8/7 | 1.0000 | 0.2115 |
| 9B RL(先验) − SFT | 18 | -0.041 | 6/12 | 0.2379 | 0.1674 |
| 4B RL(先验) − base | 10 | -0.012 | 5/5 | 1.0000 | 0.5566 |
| 4B RL(先验) − RL(base) | 16 | +0.003 | 8/8 | 1.0000 | 0.9399 |
| 4B RL(先验) − SFT | 6 | -0.055 | 2/4 | 0.6875 | 0.4375 |

### sim_excess — sim − sim_tmpl:超出惯用法的那部分相似(低=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.095 | 12/4 | 0.0768 | 0.1046 |
| 9B RL(先验) − RL(base) | 16 | +0.009 | 9/6 | 0.6072 | 0.6909 |
| 9B RL(先验) − SFT | 18 | +0.124 | 13/5 | 0.0963 | 0.0047 |
| 4B RL(先验) − base | 10 | +0.094 | 8/2 | 0.1094 | 0.0645 |
| 4B RL(先验) − RL(base) | 16 | +0.013 | 8/8 | 1.0000 | 0.9799 |
| 4B RL(先验) − SFT | 6 | +0.214 | 5/1 | 0.2188 | 0.0625 |

### sim_resid — sim 对代码长度做回归后的残差(低=好)★这条才是干净的

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | +0.107 | 12/4 | 0.0768 | 0.0182 |
| 9B RL(先验) − RL(base) | 15 | +0.016 | 7/7 | 1.0000 | 0.5098 |
| 9B RL(先验) − SFT | 18 | +0.058 | 14/4 | 0.0309 | 0.0268 |
| 4B RL(先验) − base | 10 | +0.059 | 4/6 | 0.7539 | 0.5566 |
| 4B RL(先验) − RL(base) | 16 | +0.022 | 9/7 | 0.8036 | 0.6685 |
| 4B RL(先验) − SFT | 6 | +0.132 | 4/2 | 0.6875 | 0.2188 |

### ntok — 写进去的不同标识符数(对照项)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | -21.625 | 7/9 | 0.8036 | 0.4532 |
| 9B RL(先验) − RL(base) | 16 | +7.312 | 8/7 | 1.0000 | 0.4776 |
| 9B RL(先验) − SFT | 18 | -99.667 | 3/15 | 0.0075 | 0.0016 |
| 4B RL(先验) − base | 10 | -64.400 | 1/9 | 0.0215 | 0.0137 |
| 4B RL(先验) − RL(base) | 16 | +12.750 | 9/7 | 0.8036 | 0.2774 |
| 4B RL(先验) − SFT | 6 | -61.167 | 0/6 | 0.0312 | 0.0312 |

### novel — 不含 baseline 名的实义行占比(高=好)

| 对照 | 配对题数 | Δ均值 | +/− | 符号 p | Wilcoxon p |
|---|---|---|---|---|---|
| 9B RL(先验) − base | 16 | -0.050 | 4/7 | 0.5488 | 0.1307 |
| 9B RL(先验) − RL(base) | 16 | +0.046 | 5/6 | 1.0000 | 1.0000 |
| 9B RL(先验) − SFT | 18 | -0.041 | 3/10 | 0.0923 | 0.0088 |
| 4B RL(先验) − base | 10 | -0.079 | 2/4 | 0.6875 | 0.4375 |
| 4B RL(先验) − RL(base) | 16 | -0.031 | 4/3 | 1.0000 | 0.6121 |
| 4B RL(先验) − SFT | 6 | -0.125 | 2/2 | 1.0000 | 0.6250 |

## 4. 「方法就是 baseline」的最强形态:一行没改就提交

| arm | 有效题 | 没动手题数 | 没动手的题 |
|---|---|---|---|
| 9B base | 21 | 3 | causal-treatment-effect, ml-anomaly-detection, optimization-evolution-strategy |
| 9B SFT | 21 | 1 | ml-anomaly-detection |
| 9B RL(base) | 21 | 3 | causal-observational-nonlinear, causal-treatment-effect, ml-selective-deferral |
| 9B RL(先验) | 21 | 2 | causal-discovery-discrete, optimization-hyperparameter-search |
| 4B base | 21 | 10 | causal-observational-linear-gaussian, ml-calibration, ml-clustering-algorithm, ml-dimensionality-reduction, ml-ensemble-boosting, ml-missing-data-imputation, ml-subgroup-calibration-shift, ml-symbolic-regression, mlsys-moe-load-balance, optimization-nas |
| 4B SFT | 21 | 11 | causal-discovery-discrete, causal-treatment-effect, ml-clustering-algorithm, ml-missing-data-imputation, ml-selective-deferral, ml-subgroup-calibration-shift, ml-symbolic-regression, mlsys-moe-load-balance, optimization-evolution-strategy, optimization-nas, optimization-online-bandit |
| 4B RL(base) | 21 | 2 | ml-clustering-algorithm, ml-ensemble-boosting |
| 4B RL(先验) | 21 | 4 | causal-observational-linear-gaussian, ml-active-learning, ml-dimensionality-reduction, ml-ensemble-boosting |
