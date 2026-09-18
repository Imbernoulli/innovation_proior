# 缺格盘点:没有分的格子是谁的问题

判据只看落盘的错误串,分三类:**model**(evaluator 跑起来了,死在模型自己的产物上——
记 0)、**infra**(环境没铺好/超时/权限——保持缺失)、**unknown**(只留了 stderr 最后一行,
分不出——保持缺失)。口径与理由见脚本 docstring。

## 1. 逐臂缺格(格子数 = 题 × 5 抽)

| 臂 | FCS-research 缺 / 其中 model | FrontierCS 缺 / 其中 model | ALE-Bench 缺 / 其中 model |
|---|---:|---:|---:|
| 9B base `base9b_v2c` | 0 / 0 | 2 / 0 | 0 / 0 |
| 9B SFT `ft01mix_a10` | 3 / **2** | 0 / 0 | 0 / 0 |
| 9B SFT ft03nm `ft03nm_a20` | 0 / 0 | 2 / 0 | 0 / 0 |
| 9B SFT lora `lo32nm_a10` | 0 / 0 | 0 / 0 | 0 / 0 |
| 9B RL(base) `rlv5_base_s20` | 5 / **4** | 0 / 0 | 0 / 0 |
| 9B RL 我们 `rlv5_ft01mix_a10_s20` | 6 / **6** | 0 / 0 | 0 / 0 |
| 9B RL ft03nm `rlv5_ft03nm_a20_s20` | 0 / 0 | 0 / 0 | 0 / 0 |
| 9B RL lora `rlv5_lo32nm_a10_s20` | 9 / **9** | 0 / 0 | 0 / 0 |
| 4B base `base4b` | 4 / **4** | 0 / 0 | 0 / 0 |
| 4B SFT `4b_ft01mix_a10` | 4 / **2** | 0 / 0 | 0 / 0 |
| 4B SFT lora `4b_lo32nm_a10` | 4 / **3** | 0 / 0 | 0 / 0 |
| 4B RL(base) `rlv5_4b_base_s20` | 2 / **2** | 0 / 0 | 0 / 0 |
| 4B RL 我们 `rlv5_4b_ft01mix_a10_s20` | 1 / 0 | 0 / 0 | 0 / 0 |
| 4B RL lora `rlv5_4b_lo32nm_a10_s20` | 2 / 0 | 0 / 0 | 0 / 0 |

## 2. model 类缺格的具体死法

| 臂 | bench | 题 | 抽样 | 死法 |
|---|---|---|---:|---|
| 9B SFT | FCS-research | `symbolic_regression/mixed_polyexp_4d` | 4 | solution.py 括号没闭合 |
| 9B SFT | FCS-research | `symbolic_regression/ripple` | 2 | 模型给 PySR 传了它不认的关键字参数 |
| 9B RL(base) | FCS-research | `symbolic_regression/mccormick` | 3 | 模型的表达式用了 Python 的 `**`,julia 解析不了 |
| 9B RL(base) | FCS-research | `symbolic_regression/mixed_polyexp_4d` | 4 | 模型的表达式用了 Python 的 `**`,julia 解析不了 |
| 9B RL(base) | FCS-research | `symbolic_regression/ripple` | 0 | 模型的表达式用了 Python 的 `**`,julia 解析不了 |
| 9B RL(base) | FCS-research | `vdb_pareto/low_latency` | 4 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL 我们 | FCS-research | `symbolic_regression/mixed_polyexp_4d` | 3 | 模型的产物里没有 eqn 字段 |
| 9B RL 我们 | FCS-research | `symbolic_regression/peaks` | 1 | 模型的表达式用了 Python 的 `**`,julia 解析不了 |
| 9B RL 我们 | FCS-research | `vdb_pareto/balanced` | 2 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL 我们 | FCS-research | `vdb_pareto/low_latency` | 4 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL 我们 | FCS-research | `vdb_pareto/recall80_latency` | 2 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL 我们 | FCS-research | `vdb_pareto/recall95_latency` | 4 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL lora | FCS-research | `symbolic_regression/peaks` | 2 | 模型的表达式用了 Python 的 `**`,julia 解析不了 |
| 9B RL lora | FCS-research | `vdb_pareto/balanced` | 0 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL lora | FCS-research | `vdb_pareto/balanced` | 1 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL lora | FCS-research | `vdb_pareto/balanced` | 3 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL lora | FCS-research | `vdb_pareto/high_recall` | 1 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL lora | FCS-research | `vdb_pareto/high_recall` | 3 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL lora | FCS-research | `vdb_pareto/low_latency` | 2 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL lora | FCS-research | `vdb_pareto/low_latency` | 3 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 9B RL lora | FCS-research | `vdb_pareto/recall95_latency` | 4 | faiss SIGABRT(模型把 metric 位传成了 ef) |
| 4B base | FCS-research | `symbolic_regression/mixed_polyexp_4d` | 4 | solution.py 语法错 |
| 4B base | FCS-research | `symbolic_regression/ripple` | 2 | 模型给的表达式解析不了 |
| 4B base | FCS-research | `symbolic_regression/sincos` | 3 | 模型给 PySR 传了它不认的关键字参数 |
| 4B base | FCS-research | `symbolic_regression/sincos` | 4 | PySR 参数越界 |
| 4B SFT | FCS-research | `symbolic_regression/mixed_polyexp_4d` | 0 | solution.py 语法错 |
| 4B SFT | FCS-research | `symbolic_regression/sincos` | 4 | solution.py 语法错 |
| 4B SFT lora | FCS-research | `symbolic_regression/mccormick` | 2 | solution.py 括号不配对 |
| 4B SFT lora | FCS-research | `symbolic_regression/mixed_polyexp_4d` | 1 | 模型的表达式用了 Python 的 `**`,julia 解析不了 |
| 4B SFT lora | FCS-research | `symbolic_regression/ripple` | 0 | PySR 参数越界 |
| 4B RL(base) | FCS-research | `symbolic_regression/mccormick` | 0 | solution.py 缩进错 |
| 4B RL(base) | FCS-research | `symbolic_regression/sincos` | 0 | 模型给 PySR 传了它不认的关键字参数 |

合计 **32** 格。

## 3. 把 model 类记 0 之后,FCS-research 的 mean@5

`现口径` = 现在主表用的(缺格直接从分母里去掉);`记0口径` = 只把上面那 32 格记 0,infra / unknown 一格不动。

| 臂 | n题 | 现口径 mean@5 | 记0口径 mean@5 | Δ |
|---|---:|---:|---:|---:|
| 9B base `base9b_v2c` | 64 | 10.366 | 10.366 | +0.000 |
| 9B SFT `ft01mix_a10` | 64 | 11.573 | 11.539 | -0.034 |
| 9B SFT ft03nm `ft03nm_a20` | 64 | 10.226 | 10.226 | +0.000 |
| 9B SFT lora `lo32nm_a10` | 64 | 9.896 | 9.896 | +0.000 |
| 9B RL(base) `rlv5_base_s20` | 64 | 16.944 | 16.334 | -0.611 |
| 9B RL 我们 `rlv5_ft01mix_a10_s20` | 64 | 18.885 | 18.254 | -0.631 |
| 9B RL ft03nm `rlv5_ft03nm_a20_s20` | 64 | 16.696 | 16.696 | +0.000 |
| 9B RL lora `rlv5_lo32nm_a10_s20` | 64 | 18.727 | 18.415 | -0.312 |
| 4B base `base4b` | 64 | 7.060 | 6.519 | -0.541 |
| 4B SFT `4b_ft01mix_a10` | 64 | 9.535 | 9.111 | -0.425 |
| 4B SFT lora `4b_lo32nm_a10` | 64 | 7.383 | 6.987 | -0.395 |
| 4B RL(base) `rlv5_4b_base_s20` | 64 | 9.698 | 9.073 | -0.625 |
| 4B RL 我们 `rlv5_4b_ft01mix_a10_s20` | 64 | 21.236 | 21.236 | +0.000 |
| 4B RL lora `rlv5_4b_lo32nm_a10_s20` | 64 | 18.186 | 18.186 | +0.000 |

## 4. 主对照在两种口径下的方向

| 对照 | 现口径 Δ | 记0口径 Δ | 方向变了吗 |
|---|---:|---:|:---:|
| 9B 我们 − RL(base) | +1.940 | +1.920 | 没变 |
| 9B lora − RL(base) | +1.783 | +2.081 | 没变 |
| 9B lora − 我们 | -0.158 | +0.161 | **变了** |
| 4B 我们 − RL(base) | +11.538 | +12.163 | 没变 |
| 4B lora − RL(base) | +8.488 | +9.113 | 没变 |
| 4B lora − 我们 | -3.050 | -3.050 | 没变 |

## 5. 保持缺失的那些(infra / unknown)

| 臂 | bench | 题 | 抽样 | 类 | 理由 |
|---|---|---|---:|---|---|
| 9B SFT | FCS-research | `vector_addition/2_20` | 3 | infra | 资源/产物文件没铺好 |
| 9B RL(base) | FCS-research | `grammar_fuzzing/fuzzer/sql` | 3 | infra | 判题超时 |
| 4B SFT | FCS-research | `llm_router` | 1 | infra | 资源/产物文件没铺好 |
| 4B SFT | FCS-research | `ragged_attention` | 0 | infra | 资源/产物文件没铺好 |
| 4B SFT lora | FCS-research | `cross_entropy` | 3 | infra | 资源/产物文件没铺好 |
| 4B RL 我们 | FCS-research | `llm_router` | 4 | infra | 资源/产物文件没铺好 |
| 4B RL lora | FCS-research | `llm_router` | 1 | infra | 资源/产物文件没铺好 |
| 4B RL lora | FCS-research | `llm_sql/small` | 1 | infra | 资源/产物文件没铺好 |
| 9B base | FrontierCS | `148` | 2 | infra | 判题超时 |
| 9B base | FrontierCS | `160` | 1 | infra | 判题超时 |
| 9B SFT ft03nm | FrontierCS | `62` | 3 | unknown | 未匹配任何指纹 |
| 9B SFT ft03nm | FrontierCS | `62` | 4 | unknown | 未匹配任何指纹 |

