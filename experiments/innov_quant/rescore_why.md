## 补测 71 个「有方法但记 0」的格子:跑得起来吗

| 臂 | 补测 | 有指标 | 语法错 | 导入错 | 超时 | 运行错 | 无输出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9B base | 17 | 6 | 1 | 0 | 1 | 8 | 1 |
| 9B SFT | 11 | 2 | 2 | 0 | 1 | 6 | 0 |
| 9B RL(base) | 5 | 4 | 0 | 0 | 0 | 1 | 0 |
| 9B RL(先验) | 8 | 4 | 0 | 0 | 0 | 4 | 0 |
| 4B base | 12 | 2 | 3 | 2 | 0 | 5 | 0 |
| 4B SFT | 10 | 3 | 0 | 0 | 0 | 7 | 0 |
| 4B RL(base) | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4B RL(先验) | 7 | 2 | 1 | 0 | 0 | 4 | 0 |
| **合计** | **71** | **24** | **7** | **2** | **2** | **35** | **1** |

## 逐格死因

| 臂 | 题 | 类 | 最后一行 |
|---|---|---|---|
| 9B base | `causal-observational-linear-non-gaussian` | 语法错 | `ER30: SyntaxError: unmatched '}'` |
| 9B base | `causal-observational-nonlinear` | 运行错 | `SF20-GP: ValueError: diag requires an array of at least two dimensions` |
| 9B base | `ml-active-learning` | 运行错 | `letter: TypeError: unsupported operand type(s) for -: 'float' and 'torch.return_types.max'` |
| 9B base | `ml-clustering-algorithm` | 无输出 | `blobs: rc=0 但解析不出指标` |
| 9B base | `ml-dimensionality-reduction` | 超时 | `mnist: 撞 3600s 预算` |
| 9B base | `ml-ensemble-boosting` | 运行错 | `breast_cancer: AttributeError: 'BoostingStrategy' object has no attribute 'init_weights'` |
| 9B base | `ml-subgroup-calibration-shift` | 运行错 | `adult: ValueError: Found input variables with inconsistent numbers of samples: [7914, 15]` |
| 9B base | `mlsys-moe-load-balance` | 运行错 | `deepseek-v3: NameError: name 'mlog2log' is not defined` |
| 9B base | `optimization-hyperparameter-search` | 运行错 | `xgboost: KeyError: 0.1` |
| 9B base | `optimization-multi-objective` | 运行错 | `zdt1: IndexError: index 51 is out of bounds for axis 0 with size 8` |
| 9B base | `optimization-online-bandit` | 运行错 | `stochastic-mab: NameError: name 'BanditPolicy' is not defined` |
| 9B SFT | `causal-discovery-discrete` | 超时 | `Hailfinder: 撞 3540s 预算` |
