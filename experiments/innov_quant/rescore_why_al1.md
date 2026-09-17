## 补测「有方法但记 0」的格子(al1):跑得起来吗

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
| 9B SFT | `ml-active-learning` | 运行错 | `letter: AttributeError: 'CustomSampling' object has no attribute '_compute_mc_disagreement'` |
| 9B SFT | `ml-clustering-algorithm` | 运行错 | `blobs: NameError: name 'pdist' is not defined. Did you mean: 'cdist'?` |
| 9B SFT | `ml-subgroup-calibration-shift` | 运行错 | `adult: AttributeError: 'CalibrationMethod' object has no attribute 'group_base_rates_'` |
| 9B SFT | `ml-symbolic-regression` | 运行错 | `nguyen7: NameError: name 'fitness_function' is not defined` |
| 9B SFT | `optimization-evolution-strategy` | 运行错 | `rastrigin-30d: AttributeError: module 'deap.tools' has no attribute 'cxPolynomial'` |
| 9B SFT | `optimization-hyperparameter-search` | 语法错 | `xgboost: IndentationError: unexpected indent` |
| 9B SFT | `optimization-nas` | 运行错 | `CIFAR-10: UnboundLocalError: cannot access local variable 'val_acc' where it is not associated with a value` |
| 9B SFT | `optimization-online-bandit` | 语法错 | `stochastic-mab: SyntaxError: unterminated string literal (detected at line 329)` |
| 9B RL(base) | `causal-treatment-effect` | 运行错 | `ihdp_synth: AttributeError: 'CATEEstimator' object has no attribute '_fit_model'` |
| 9B RL(先验) | `causal-observational-linear-non-gaussian` | 运行错 | `ER30: NameError: name 'sla' is not defined` |
| 9B RL(先验) | `causal-observational-nonlinear` | 运行错 | `SF20-GP: NameError: name 'os' is not defined. Did you mean: 'pos'?` |
| 9B RL(先验) | `ml-active-learning` | 运行错 | `letter: TypeError: Concatenation operation is not implemented for NumPy arrays, use np.concatenate() instead. Please d` |
| 9B RL(先验) | `ml-calibration` | 运行错 | `rf-mnist: TypeError: CalibrationMethod.fit.<locals>.platt_objective() missing 1 required positional argument: 'b'` |
| 4B base | `causal-observational-linear-gaussian` | 导入错 | `ER10: ModuleNotFoundError: No module named 'causallearn.utils.UCSepset'` |
| 4B base | `causal-observational-linear-non-gaussian` | 运行错 | `ER30: ValueError: run_causal_discovery returned shape (), expected (30, 30).` |
| 4B base | `causal-treatment-effect` | 运行错 | `ihdp_synth: UnboundLocalError: cannot access local variable 'mask1_val' where it is not associated with a value` |
| 4B base | `ml-anomaly-detection` | 导入错 | `cardio: ModuleNotFoundError: No module named 'sklearn.kernel_density'` |
| 4B base | `ml-calibration` | 运行错 | `rf-mnist: NameError: name 'cur_eco_best' is not defined` |
| 4B base | `ml-missing-data-imputation` | 语法错 | `breast_cancer: SyntaxError: 'return' outside function` |
| 4B base | `mlsys-moe-load-balance` | 语法错 | `deepseek-v3: SyntaxError: unmatched ')'` |
| 4B base | `optimization-evolution-strategy` | 运行错 | `rastrigin-30d: NameError: name 'pop_size_to_seed' is not defined` |
| 4B base | `optimization-multi-objective` | 运行错 | `zdt1: IndexError: index out of range` |
| 4B base | `optimization-nas` | 语法错 | `CIFAR-10: SyntaxError: invalid syntax` |
| 4B SFT | `causal-observational-linear-non-gaussian` | 运行错 | `ER30: NameError: name 'x' is not defined. Did you mean: 'xi'?` |
| 4B SFT | `causal-observational-nonlinear` | 运行错 | `SF20-GP: TypeError: GradientBoostingRegressor.__init__() got an unexpected keyword argument 'l1_ratio'` |
| 4B SFT | `ml-active-learning` | 运行错 | `letter: NameError: name 'torch' is not defined` |
| 4B SFT | `ml-calibration` | 运行错 | `rf-mnist: ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 d` |
| 4B SFT | `ml-ensemble-boosting` | 运行错 | `breast_cancer: NameError: name 'y_train_counts' is not defined` |
| 4B SFT | `optimization-hyperparameter-search` | 运行错 | `xgboost: AttributeError: 'CustomHPOStrategy' object has no attribute '_encode'` |
| 4B SFT | `optimization-nas` | 运行错 | `CIFAR-10: AttributeError: 'NASOptimizer' object has no attribute 'warm_start'` |
| 4B RL(先验) | `causal-observational-linear-non-gaussian` | 运行错 | `ER30: NameError: name 'n' is not defined. Did you mean: 'np'?` |
| 4B RL(先验) | `ml-active-learning` | 语法错 | `letter: SyntaxError: unterminated triple-quoted string literal (detected at line 49)` |
| 4B RL(先验) | `ml-calibration` | 运行错 | `rf-mnist: IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed` |
| 4B RL(先验) | `ml-subgroup-calibration-shift` | 运行错 | `adult: AttributeError: 'NoneType' object has no attribute 'predict_proba'` |
| 4B RL(先验) | `optimization-multi-objective` | 运行错 | `zdt1: NameError: name 'valid' is not defined` |
