## 补测「有方法但记 0」的格子(p1):跑得起来吗

| 臂 | 补测 | 有指标 | 语法错 | 导入错 | 超时 | 运行错 | 无输出 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9B base | 10 | 4 | 1 | 0 | 0 | 5 | 0 |
| 9B SFT | 11 | 6 | 1 | 0 | 0 | 4 | 0 |
| 9B RL(base) | 10 | 6 | 0 | 0 | 0 | 4 | 0 |
| 9B RL(先验) | 8 | 3 | 2 | 0 | 0 | 3 | 0 |
| 4B base | 15 | 10 | 3 | 0 | 0 | 2 | 0 |
| 4B SFT | 16 | 11 | 1 | 0 | 0 | 4 | 0 |
| 4B RL(base) | 12 | 8 | 1 | 1 | 0 | 2 | 0 |
| 4B RL(先验) | 9 | 7 | 2 | 0 | 0 | 0 | 0 |
| **合计** | **91** | **55** | **11** | **1** | **0** | **24** | **0** |

## 逐格死因

| 臂 | 题 | 类 | 最后一行 |
|---|---|---|---|
| 9B base | `causal-observational-linear-non-gaussian` | 运行错 | `ER30: UnboundLocalError: cannot access local variable 'h' where it is not associated with a value` |
| 9B base | `ml-clustering-algorithm` | 语法错 | `blobs: SyntaxError: 'return' outside function` |
| 9B base | `ml-dimensionality-reduction` | 运行错 | `mnist: ValueError: operands could not be broadcast together with shapes (5000,5000) (784,784)` |
| 9B base | `ml-missing-data-imputation` | 运行错 | `breast_cancer: IndexError: index 3 is out of bounds for axis 1 with size 3` |
| 9B base | `mlsys-moe-load-balance` | 运行错 | `deepseek-v3: TypeError: expected Tensor as element 1 in argument 0, but got slice` |
| 9B base | `optimization-hyperparameter-search` | 运行错 | `xgboost: NameError: name 'max_fid' is not defined. Did you mean: 'base_fid'?` |
| 9B SFT | `ml-active-learning` | 运行错 | `letter: AttributeError: 'CustomSampling' object has no attribute '_query_alpha_beta'` |
| 9B SFT | `ml-clustering-algorithm` | 运行错 | `blobs: IndexError: index 1 is out of bounds for axis 0 with size 1` |
| 9B SFT | `ml-dimensionality-reduction` | 运行错 | `mnist: AttributeError: 'scipy.spatial._ckdtree.cKDTree' object has no attribute 'query_radius'. Did you mean: 'query_` |
| 9B SFT | `ml-symbolic-regression` | 语法错 | `nguyen7: SyntaxError: invalid character '—' (U+2014)` |
| 9B SFT | `optimization-evolution-strategy` | 运行错 | `rastrigin-30d: NameError: name 'evalucheval_func' is not defined. Did you mean: 'evaluate_func'?` |
| 9B RL(base) | `causal-observational-linear-non-gaussian` | 运行错 | `ER30: NameError: name 'sklearn_utils_func' is not defined` |
| 9B RL(base) | `ml-active-learning` | 运行错 | `letter: AttributeError: 'torch.return_types.min' object has no attribute 'cpu'` |
| 9B RL(base) | `mlsys-moe-load-balance` | 运行错 | `deepseek-v3: IndexError: index 61 is out of bounds for dimension 0 with size 61` |
| 9B RL(base) | `optimization-hyperparameter-search` | 运行错 | `xgboost: TypeError: '>' not supported between instances of 'list' and 'int'` |
| 9B RL(先验) | `causal-observational-nonlinear` | 语法错 | `SF20-GP: IndentationError: unexpected indent` |
| 9B RL(先验) | `ml-active-learning` | 语法错 | `letter: IndentationError: expected an indented block after 'for' statement on line 143` |
| 9B RL(先验) | `ml-dimensionality-reduction` | 运行错 | `mnist: AttributeError: 'NoneType' object has no attribute 'shape'` |
| 9B RL(先验) | `ml-missing-data-imputation` | 运行错 | `breast_cancer: NameError: name 'ExtraTreesRegressor' is not defined` |
| 9B RL(先验) | `ml-selective-deferral` | 运行错 | `adult: ValueError: Expected 2D array, got 1D array instead:` |
| 4B base | `causal-observational-linear-non-gaussian` | 语法错 | `ER30: SyntaxError: unmatched '}'` |
| 4B base | `causal-treatment-effect` | 语法错 | `ihdp_synth: SyntaxError: invalid octal literal` |
| 4B base | `ml-active-learning` | 运行错 | `letter: AttributeError: 'CustomSampling' object has no attribute 'clazz_cfg'` |
| 4B base | `optimization-evolution-strategy` | 语法错 | `rastrigin-30d: SyntaxError: invalid syntax. Perhaps you forgot a comma?` |
| 4B base | `optimization-multi-objective` | 运行错 | `zdt1: TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'` |
| 4B SFT | `causal-observational-nonlinear` | 运行错 | `SF20-GP: NameError: name 'scipy' is not defined` |
| 4B SFT | `ml-active-learning` | 运行错 | `letter: UnboundLocalError: local variable 'probs_cpu' referenced before assignment` |
| 4B SFT | `ml-calibration` | 运行错 | `rf-mnist: NameError: name 'P' is not defined` |
| 4B SFT | `ml-ensemble-boosting` | 运行错 | `breast_cancer: KeyError: 'seed'` |
| 4B SFT | `optimization-multi-objective` | 语法错 | `zdt1: IndentationError: unindent does not match any outer indentation level` |
| 4B RL(base) | `causal-observational-linear-gaussian` | 导入错 | `ER10: ImportError: cannot import name 'CIT' from 'causallearn.utils.PCUtils' (/workspace/causal-learn/causallearn/ut` |
| 4B RL(base) | `causal-observational-linear-non-gaussian` | 运行错 | `ER30: ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?` |
| 4B RL(base) | `ml-active-learning` | 运行错 | `letter: TypeError: sum() received an invalid combination of arguments - got (out=NoneType, axis=int, ), but expected o` |
| 4B RL(base) | `ml-symbolic-regression` | 语法错 | `nguyen7: SyntaxError: unmatched ')'` |
| 4B RL(先验) | `optimization-hyperparameter-search` | 语法错 | `xgboost: SyntaxError: invalid character '�' (U+FFFD)` |
| 4B RL(先验) | `optimization-online-bandit` | 语法错 | `stochastic-mab: IndentationError: unexpected indent` |
