## 1. 逐臂:as-run vs file-state(p1,分母 21)

| 臂 | 补测格子 | 跑出指标 | 补出非零 | as-run 均分 | file-state 均分 | Δ | as-run 非零 | file-state 非零 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9B base | 10 | 4 | 1 | 0.1171 | 0.1407 | +0.0236 | 11/21 | 12/21 |
| 9B SFT | 11 | 6 | 3 | 0.1055 | 0.1237 | +0.0182 | 10/21 | 13/21 |
| 9B RL(base) | 10 | 6 | 4 | 0.0939 | 0.1353 | +0.0415 | 11/21 | 15/21 |
| 9B RL(先验) | 8 | 3 | 1 | 0.1560 | 0.1602 | +0.0042 | 13/21 | 14/21 |
| 4B base | 15 | 10 | 6 | 0.0431 | 0.1023 | +0.0593 | 6/21 | 12/21 |
| 4B SFT | 16 | 11 | 7 | 0.0612 | 0.1192 | +0.0580 | 5/21 | 12/21 |
| 4B RL(base) | 12 | 8 | 5 | 0.0973 | 0.1710 | +0.0737 | 9/21 | 14/21 |
| 4B RL(先验) | 9 | 7 | 3 | 0.1514 | 0.1681 | +0.0167 | 12/21 | 15/21 |

## 2. 对照

| 对照 | as-run Δ | as-run 胜/负/平 | as-run p | file-state Δ | file-state 胜/负/平 | file-state p |
|---|---:|:---:|---:|---:|:---:|---:|
| 9B 我们 − RL(base) | +0.0621 | 7/3/11 | 0.3438 | +0.0248 | 7/6/8 | 1.0000 |
| 9B 我们 − SFT | +0.0505 | 10/6/5 | 0.4545 | +0.0364 | 9/6/6 | 0.6072 |
| 9B 我们 − base | +0.0389 | 9/6/6 | 0.6072 | +0.0195 | 9/7/5 | 0.8036 |
| 4B 我们 − RL(base) | +0.0540 | 5/4/12 | 1.0000 | -0.0030 | 4/6/11 | 0.7539 |
| 4B 我们 − SFT | +0.0902 | 10/2/9 | 0.0386 | +0.0489 | 6/3/12 | 0.5078 |
| 4B 我们 − base | +0.1083 | 9/4/8 | 0.2668 | +0.0657 | 5/4/12 | 1.0000 |

## 3. 补出非零的格子

| 臂 | 题 | file-state 分 |
|---|---|---:|
| 9B base | `ml-active-learning` | 0.4953 |
| 9B SFT | `ml-calibration` | 0.0516 |
| 9B SFT | `optimization-hyperparameter-search` | 0.3027 |
| 9B SFT | `optimization-nas` | 0.0281 |
| 9B RL(base) | `causal-discovery-discrete` | 0.3027 |
| 9B RL(base) | `causal-observational-nonlinear` | 0.0357 |
| 9B RL(base) | `causal-treatment-effect` | 0.4808 |
| 9B RL(base) | `ml-calibration` | 0.0516 |
| 9B RL(先验) | `causal-treatment-effect` | 0.0877 |
| 4B base | `causal-observational-nonlinear` | 0.0357 |
| 4B base | `ml-anomaly-detection` | 0.5016 |
| 4B base | `ml-calibration` | 0.0516 |
| 4B base | `ml-clustering-algorithm` | 0.3875 |
| 4B base | `ml-dimensionality-reduction` | 0.0129 |
| 4B base | `mlsys-moe-load-balance` | 0.2553 |
| 4B SFT | `causal-treatment-effect` | 0.2606 |
| 4B SFT | `ml-clustering-algorithm` | 0.3875 |
| 4B SFT | `ml-dimensionality-reduction` | 0.0129 |
| 4B SFT | `ml-selective-deferral` | 0.2420 |
| 4B SFT | `ml-symbolic-regression` | 0.0308 |
| 4B SFT | `mlsys-moe-load-balance` | 0.2566 |
| 4B SFT | `optimization-nas` | 0.0281 |
| 4B RL(base) | `causal-discovery-discrete` | 0.3027 |
| 4B RL(base) | `ml-anomaly-detection` | 0.5016 |
| 4B RL(base) | `ml-clustering-algorithm` | 0.3875 |
| 4B RL(base) | `ml-missing-data-imputation` | 0.0530 |
| 4B RL(base) | `optimization-hyperparameter-search` | 0.3027 |
| 4B RL(先验) | `causal-observational-linear-gaussian` | 0.0182 |
| 4B RL(先验) | `ml-symbolic-regression` | 0.0308 |
| 4B RL(先验) | `optimization-multi-objective` | 0.3015 |
