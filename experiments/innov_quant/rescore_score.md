## 1. 逐臂:as-run vs file-state(al1,分母 21)

| 臂 | 补测格子 | 跑出指标 | 补出非零 | as-run 均分 | file-state 均分 | Δ | as-run 非零 | file-state 非零 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9B base | 17 | 6 | 3 | 0.0370 | 0.0811 | +0.0441 | 4/21 | 7/21 |
| 9B SFT | 11 | 2 | 1 | 0.0453 | 0.0631 | +0.0178 | 8/21 | 9/21 |
| 9B RL(base) | 5 | 4 | 2 | 0.1251 | 0.1343 | +0.0092 | 14/21 | 16/21 |
| 9B RL(先验) | 8 | 4 | 3 | 0.1356 | 0.1833 | +0.0477 | 11/21 | 14/21 |
| 4B base | 12 | 2 | 1 | 0.0639 | 0.0654 | +0.0015 | 6/21 | 7/21 |
| 4B SFT | 10 | 3 | 2 | 0.0217 | 0.0537 | +0.0320 | 3/21 | 5/21 |
| 4B RL(base) | 1 | 1 | 0 | 0.2122 | 0.2122 | +0.0000 | 16/21 | 16/21 |
| 4B RL(先验) | 7 | 2 | 2 | 0.1619 | 0.1982 | +0.0363 | 13/21 | 15/21 |

## 2. 对照

| 对照 | as-run Δ | as-run 胜/负/平 | as-run p | file-state Δ | file-state 胜/负/平 | file-state p |
|---|---:|:---:|---:|---:|:---:|---:|
| 9B 我们 − RL(base) | +0.0105 | 7/7/7 | 1.0000 | +0.0490 | 10/7/4 | 0.6291 |
| 9B 我们 − SFT | +0.0903 | 8/5/8 | 0.5811 | +0.1202 | 11/5/5 | 0.2101 |
| 9B 我们 − base | +0.0986 | 8/3/10 | 0.2266 | +0.1022 | 9/5/7 | 0.4240 |
| 4B 我们 − RL(base) | -0.0503 | 8/9/4 | 1.0000 | -0.0140 | 10/7/4 | 0.6291 |
| 4B 我们 − SFT | +0.1403 | 12/1/8 | 0.0034 | +0.1446 | 13/3/5 | 0.0213 |
| 4B 我们 − base | +0.0980 | 11/2/8 | 0.0225 | +0.1329 | 13/1/7 | 0.0018 |

## 3. 补出非零的格子

| 臂 | 题 | file-state 分 |
|---|---|---:|
| 9B base | `causal-treatment-effect` | 0.2606 |
| 9B base | `ml-missing-data-imputation` | 0.3122 |
| 9B base | `ml-symbolic-regression` | 0.3532 |
| 9B SFT | `mlsys-moe-load-balance` | 0.3745 |
| 9B RL(base) | `ml-ensemble-boosting` | 0.1657 |
| 9B RL(base) | `optimization-nas` | 0.0281 |
| 9B RL(先验) | `causal-discovery-discrete` | 0.3642 |
| 9B RL(先验) | `causal-treatment-effect` | 0.4341 |
| 9B RL(先验) | `ml-ensemble-boosting` | 0.2040 |
| 4B base | `ml-symbolic-regression` | 0.0308 |
| 4B SFT | `mlsys-moe-load-balance` | 0.3612 |
| 4B SFT | `optimization-multi-objective` | 0.3107 |
| 4B RL(先验) | `causal-observational-nonlinear` | 0.4953 |
| 4B RL(先验) | `ml-missing-data-imputation` | 0.2674 |
