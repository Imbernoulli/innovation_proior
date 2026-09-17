# MLS 两套口径:as-run vs file-state(协议 `al1`,分母 21)

恢复规则:agent 没 finalize 但**已跑出过真实指标**的格子,取它最后一批非 final 运行提升为 final。
取「最后一批」而非「最好的一批」,沿用 harness 自己的 `_latest_valid_nonfinal_batch` 语义。
从没跑出过指标的格子本脚本不动(需重新执行 run_eval,那是另一次运行,不等于当时该得的分)。

| 臂 | as-run 均分/21 | file-state 均分/21 | Δ | 恢复格子数 |
|---|---|---|---|---|
| 9B base | 0.0370 | 0.0370 | -0.0001 | 5 |
| 9B SFT | 0.0453 | 0.0453 | +0.0000 | 1 |
| 9B RL(base) | 0.1251 | 0.1251 | +0.0000 | 2 |
| 9B RL(先验) | 0.1356 | 0.1356 | +0.0000 | 1 |
| 4B base | 0.0639 | 0.0639 | +0.0000 | 7 |
| 4B SFT | 0.0217 | 0.0178 | -0.0038 | 3 |
| 4B RL(base) | 0.2122 | 0.2122 | +0.0000 | 1 |
| 4B RL(先验) | 0.1619 | 0.1619 | +0.0000 | 3 |

## 被恢复的格子

| 臂 | task | as-run | file-state | 说明 |
|---|---|---|---|---|
| 9B base | `causal-observational-linear-gaussian` | 0.0261 | 0.0261 | 提升 1 行(ts=42+00:00) |
| 9B base | `causal-observational-linear-non-gaussian` | 0.0000 | 0.0000 | 提升 1 行(ts=93+00:00) |
| 9B base | `ml-selective-deferral` | 0.2589 | 0.2589 | 提升 1 行(ts=94+00:00) |
| 9B base | `optimization-nas` | 0.0361 | 0.0345 | 提升 1 行(ts=22+00:00) |
| 9B base | `optimization-online-bandit` | 0.0000 | 0.0000 | 提升 1 行(ts=15+00:00) |
| 9B SFT | `causal-observational-linear-non-gaussian` | 0.0292 | 0.0292 | 提升 1 行(ts=83+00:00) |
| 9B RL(base) | `causal-observational-nonlinear` | 0.2359 | 0.2359 | 提升 1 行(ts=15+00:00) |
| 9B RL(base) | `ml-active-learning` | 0.0286 | 0.0286 | 提升 1 行(ts=76+00:00) |
| 9B RL(先验) | `ml-selective-deferral` | 0.2420 | 0.2420 | 提升 1 行(ts=09+00:00) |
| 4B base | `causal-discovery-discrete` | 0.0000 | 0.0000 | 提升 1 行(ts=08+00:00) |
| 4B base | `causal-observational-nonlinear` | 0.0357 | 0.0357 | 提升 1 行(ts=65+00:00) |
| 4B base | `ml-active-learning` | 0.3617 | 0.3617 | 提升 1 行(ts=49+00:00) |
| 4B base | `ml-clustering-algorithm` | 0.3875 | 0.3875 | 提升 1 行(ts=65+00:00) |
| 4B base | `ml-dimensionality-reduction` | 0.0129 | 0.0129 | 提升 1 行(ts=69+00:00) |
| 4B base | `ml-selective-deferral` | 0.2420 | 0.2420 | 提升 1 行(ts=54+00:00) |
| 4B base | `optimization-hyperparameter-search` | 0.3027 | 0.3027 | 提升 1 行(ts=69+00:00) |
| 4B SFT | `ml-dimensionality-reduction` | 0.0129 | 0.0129 | 提升 1 行(ts=62+00:00) |
| 4B SFT | `ml-selective-deferral` | 0.2420 | 0.2420 | 提升 1 行(ts=83+00:00) |
| 4B SFT | `ml-symbolic-regression` | 0.2000 | 0.1196 | 提升 1 行(ts=39+00:00) |
| 4B RL(base) | `ml-missing-data-imputation` | 0.0530 | 0.0530 | 提升 1 行(ts=65+00:00) |
| 4B RL(先验) | `causal-treatment-effect` | 0.3584 | 0.3584 | 提升 1 行(ts=17+00:00) |
| 4B RL(先验) | `ml-clustering-algorithm` | 0.3875 | 0.3875 | 提升 1 行(ts=71+00:00) |
| 4B RL(先验) | `ml-selective-deferral` | 0.4642 | 0.4642 | 提升 1 行(ts=69+00:00) |

## 仍是 0 且从没跑出过指标(要重新执行才能判)

| 臂 | task |
|---|---|
| 9B base | `causal-observational-nonlinear` |
| 9B base | `ml-active-learning` |
| 9B base | `ml-calibration` |
| 9B base | `ml-clustering-algorithm` |
| 9B base | `ml-ensemble-boosting` |
| 9B base | `ml-missing-data-imputation` |
| 9B base | `ml-subgroup-calibration-shift` |
| 9B base | `ml-symbolic-regression` |
| 9B base | `optimization-multi-objective` |
| 9B SFT | `causal-treatment-effect` |
| 9B SFT | `ml-active-learning` |
| 9B SFT | `ml-clustering-algorithm` |
| 9B SFT | `ml-ensemble-boosting` |
| 9B SFT | `ml-selective-deferral` |
| 9B SFT | `ml-subgroup-calibration-shift` |
| 9B SFT | `ml-symbolic-regression` |
| 9B SFT | `optimization-evolution-strategy` |
| 9B SFT | `optimization-nas` |
| 9B RL(base) | `causal-treatment-effect` |
| 9B RL(先验) | `causal-observational-linear-non-gaussian` |
| 9B RL(先验) | `causal-observational-nonlinear` |
| 9B RL(先验) | `ml-active-learning` |
| 9B RL(先验) | `ml-calibration` |
| 9B RL(先验) | `optimization-multi-objective` |
| 4B base | `causal-observational-linear-gaussian` |
| 4B base | `causal-observational-linear-non-gaussian` |
| 4B base | `causal-treatment-effect` |
| 4B base | `ml-anomaly-detection` |
| 4B base | `ml-calibration` |
| 4B base | `ml-ensemble-boosting` |
| 4B base | `ml-missing-data-imputation` |
| 4B base | `ml-symbolic-regression` |
| 4B base | `mlsys-moe-load-balance` |
| 4B base | `optimization-evolution-strategy` |
| 4B base | `optimization-nas` |
| 4B base | `optimization-online-bandit` |
| 4B SFT | `causal-discovery-discrete` |
| 4B SFT | `causal-observational-linear-gaussian` |
| 4B SFT | `causal-observational-linear-non-gaussian` |
| 4B SFT | `causal-observational-nonlinear` |
| 4B SFT | `causal-treatment-effect` |
| 4B SFT | `ml-active-learning` |
| 4B SFT | `ml-anomaly-detection` |
| 4B SFT | `ml-calibration` |
| 4B SFT | `ml-clustering-algorithm` |
| 4B SFT | `ml-ensemble-boosting` |
| 4B SFT | `ml-missing-data-imputation` |
| 4B SFT | `ml-subgroup-calibration-shift` |
| 4B SFT | `mlsys-moe-load-balance` |
| 4B SFT | `optimization-evolution-strategy` |
| 4B SFT | `optimization-hyperparameter-search` |
| 4B SFT | `optimization-multi-objective` |
| 4B SFT | `optimization-nas` |
| 4B SFT | `optimization-online-bandit` |
| 4B RL(base) | `causal-treatment-effect` |
| 4B RL(base) | `optimization-hyperparameter-search` |
| 4B RL(base) | `optimization-nas` |
| 4B RL(先验) | `causal-observational-linear-non-gaussian` |
| 4B RL(先验) | `causal-observational-nonlinear` |
| 4B RL(先验) | `ml-active-learning` |
| 4B RL(先验) | `ml-calibration` |
| 4B RL(先验) | `ml-missing-data-imputation` |
| 4B RL(先验) | `ml-subgroup-calibration-shift` |
| 4B RL(先验) | `optimization-multi-objective` |
