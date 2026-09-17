# @5 做法多样性 v2:语料不截断之后

v2 标注 39 题;v1 标注 39 题;两版都有的 **39** 题(下面全部按这个交集配对)。
模型输出、题目、配对方式与 v1 完全相同,变的只有「标注者看到多少」。

v2 合规检查:**通过**(每块 5 抽不重不漏、至多一个 no_method 类)。

## 1. v2(不截断)

### FCS-research(26 题)

| 臂 | n_approach | n_method | 没产出方法的抽样 | 有最终答案的抽样 |
|---|---:|---:|---:|---:|
| 9B RL(base) | 2.35 | 2.08 | 15/130 | 119/130 |
| 9B 我们 | 2.19 | 1.81 | 21/130 | 123/130 |
| 4B RL(base) | 1.19 | 0.19 | 121/130 | 11/130 |
| 4B 我们 | 1.88 | 1.58 | 20/130 | 113/130 |

| 对照 | Δ n_method | 胜/负/平 | 符号 p | Δ n_approach |
|---|---:|:---:|---:|---:|
| 9B 我们 − RL(base) | -0.27 | 2/9/15 | 0.0654 | -0.15 |
| 4B 我们 − RL(base) | +1.38 | 22/0/4 | 0.0000 | +0.69 |

### ALE-Bench(13 题)

| 臂 | n_approach | n_method | 没产出方法的抽样 | 有最终答案的抽样 |
|---|---:|---:|---:|---:|
| 9B RL(base) | 2.31 | 1.85 | 19/65 | 52/65 |
| 9B 我们 | 2.38 | 1.62 | 16/65 | 56/65 |
| 4B RL(base) | 1.00 | 0.00 | 65/65 | 4/65 |
| 4B 我们 | 1.92 | 1.54 | 12/65 | 57/65 |

| 对照 | Δ n_method | 胜/负/平 | 符号 p | Δ n_approach |
|---|---:|:---:|---:|---:|
| 9B 我们 − RL(base) | -0.23 | 2/5/6 | 0.4531 | +0.08 |
| 4B 我们 − RL(base) | +1.54 | 13/0/0 | 0.0002 | +0.92 |

## 2. v1(截到 4000 字)——留作对照

### FCS-research(26 题)

| 臂 | n_approach | n_method | 没产出方法的抽样 | 有最终答案的抽样 |
|---|---:|---:|---:|---:|
| 9B RL(base) | 3.00 | 2.65 | 17/130 | 119/130 |
| 9B 我们 | 2.65 | 2.38 | 15/130 | 123/130 |
| 4B RL(base) | 1.23 | 0.23 | 120/130 | 11/130 |
| 4B 我们 | 2.12 | 1.81 | 19/130 | 113/130 |

| 对照 | Δ n_method | 胜/负/平 | 符号 p | Δ n_approach |
|---|---:|:---:|---:|---:|
| 9B 我们 − RL(base) | -0.27 | 4/10/12 | 0.1796 | -0.35 |
| 4B 我们 − RL(base) | +1.58 | 24/0/2 | 0.0000 | +0.88 |

### ALE-Bench(13 题)

| 臂 | n_approach | n_method | 没产出方法的抽样 | 有最终答案的抽样 |
|---|---:|---:|---:|---:|
| 9B RL(base) | 2.38 | 1.92 | 15/65 | 52/65 |
| 9B 我们 | 2.46 | 1.69 | 17/65 | 56/65 |
| 4B RL(base) | 1.08 | 0.08 | 63/65 | 4/65 |
| 4B 我们 | 1.77 | 1.38 | 12/65 | 57/65 |

| 对照 | Δ n_method | 胜/负/平 | 符号 p | Δ n_approach |
|---|---:|:---:|---:|---:|
| 9B 我们 − RL(base) | -0.23 | 4/7/2 | 0.5488 | +0.08 |
| 4B 我们 − RL(base) | +1.31 | 12/0/1 | 0.0005 | +0.69 |

## 3. ★ v1 → v2:重标把什么改掉了

| bench | 臂 | v1 n_method | v2 n_method | Δ | v1「没方法」抽样 | v2「没方法」抽样 |
|---|---|---:|---:|---:|---:|---:|
| FCS-research | 9B RL(base) | 2.65 | 2.08 | -0.58 | 17 | 15 |
| FCS-research | 9B 我们 | 2.38 | 1.81 | -0.58 | 15 | 21 |
| FCS-research | 4B RL(base) | 0.23 | 0.19 | -0.04 | 120 | 121 |
| FCS-research | 4B 我们 | 1.81 | 1.58 | -0.23 | 19 | 20 |
| ALE-Bench | 9B RL(base) | 1.92 | 1.85 | -0.08 | 15 | 19 |
| ALE-Bench | 9B 我们 | 1.69 | 1.62 | -0.08 | 17 | 16 |
| ALE-Bench | 4B RL(base) | 0.08 | 0.00 | -0.08 | 63 | 65 |
| ALE-Bench | 4B 我们 | 1.38 | 1.54 | +0.15 | 12 | 12 |

## 4. ★ 主对照在两版语料下的值

| bench | 对照 | v1 Δ | v1 符号 p | v2 Δ | v2 符号 p |
|---|---|---:|---:|---:|---:|
| FCS-research | 9B 我们 − RL(base) | -0.27 | 0.1796 | -0.27 | 0.0654 |
| FCS-research | 4B 我们 − RL(base) | +1.58 | 0.0000 | +1.38 | 0.0000 |
| ALE-Bench | 9B 我们 − RL(base) | -0.23 | 0.5488 | -0.23 | 0.4531 |
| ALE-Bench | 4B 我们 − RL(base) | +1.31 | 0.0005 | +1.54 | 0.0002 |

## 5. 少数的那几种做法,换到分了吗

`n_method` 少不一定是坏事:五抽都收敛到同一个**更好**的做法,和五抽都收敛到同一个烂做法,是两件事。逐题把 Δn_method 和 Δ题均分交叉制表。

| bench | 对照 | 题 | Δ n_method | Δ 题均分 | 对手题均分 | 我们题均分 |
|---|---|---:|---:|---:|---:|---:|
| FCS-research | 9B 我们 − RL(base) | 26 | -0.27 | +0.8 | 18.0 | 18.7 |
| FCS-research | 4B 我们 − RL(base) | 26 | +1.38 | +11.8 | 11.0 | 22.8 |
| ALE-Bench | 9B 我们 − RL(base) | 13 | -0.23 | +21.4 | 377.8 | 399.3 |
| ALE-Bench | 4B 我们 − RL(base) | 13 | +1.54 | +177.3 | 89.4 | 266.8 |

**只看「我们做法更少」的那些题**(这是唯一对我们不利的格子,单独拎出来看代价):

| bench | 对照 | 做法更少的题 | 其中我们分更高 | 这些题上的 Δ 题均分 |
|---|---|---:|---:|---:|
| FCS-research | 9B 我们 − RL(base) | 9 | 3 | +2.8 |
| FCS-research | 4B 我们 − RL(base) | 0 | — | — |
| ALE-Bench | 9B 我们 − RL(base) | 5 | 2 | +49.8 |
| ALE-Bench | 4B 我们 − RL(base) | 0 | — | — |

## 6. 逐题明细 v1 → v2(n_method)

### FCS-research

| 题 | 9B RL(base) | 9B 我们 | 4B RL(base) | 4B 我们 |
|---|---:|---:|---:|---:|
| `cant_be_late_high_availability_tight_dea` | 2 | **2→1** | 1 | 1 |
| `cant_be_late_low_availability_loose_dead` | 2 | 2 | 1 | 1 |
| `cant_be_late_low_availability_tight_dead` | **3→2** | 1 | 0 | **2→1** |
| `cant_be_late_mixed_availability_loose_de` | **2→1** | **2→1** | 0 | 1 |
| `cant_be_late_mixed_availability_tight_de` | 2 | **3→2** | 0 | 2 |
| `cant_be_late_mixed_availability_tight_de` | 2 | 3 | 0 | 3 |
| `cant_be_late_multi_high_availability_loo` | 3 | 2 | 0 | 1 |
| `cant_be_late_multi_high_availability_tig` | **3→2** | 2 | 0 | **2→1** |
| `cant_be_late_multi_low_availability_loos` | **3→2** | 3 | 0 | 2 |
| `cant_be_late_multi_low_availability_loos` | **4→3** | **4→3** | 0 | **3→2** |
| `cloudcast` | 2 | 1 | 0 | 1 |
| `decoding_attn` | 1 | 1 | 0 | 1 |
| `gemm_optimization_annoying` | 4 | **4→3** | 0 | 3 |
| `gemm_optimization_k_skewed` | **4→3** | **4→2** | **1→0** | 2 |
| `gemm_optimization_near_tile` | 3 | **4→3** | 0 | 2 |
| `gemm_optimization_transformerish` | **5→2** | **4→2** | 0 | 2 |
| `grammar_fuzzing_fuzzer_sql` | **4→2** | **3→2** | 1 | **3→1** |
| `imagenet_pareto_200k` | 3 | 2 | 0 | 1 |
| `imagenet_pareto_2_5m` | **3→2** | 2 | 0 | 2 |
| `imagenet_pareto_500k` | **3→2** | **4→2** | 0 | 1 |
| `imagenet_pareto_5m` | **4→2** | **4→2** | 0 | 1 |
| `mixed_gemm` | 3 | 2 | 0 | 1 |
| `quant_dot_int4` | 1 | 0 | 0 | 2 |
| `symbolic_regression_mccormick` | 1 | 1 | 1 | **2→1** |
| `symbolic_regression_mixed_polyexp_4d` | 1 | 1 | 1 | 2 |
| `vdb_pareto_recall95_latency` | 1 | 1 | 0 | 3 |

### ALE-Bench

| 题 | 9B RL(base) | 9B 我们 | 4B RL(base) | 4B 我们 |
|---|---:|---:|---:|---:|
| `ale_ahc003` | **2→3** | **3→2** | 0 | 1 |
| `ale_ahc007` | 3 | 2 | 0 | 1 |
| `ale_ahc009` | 1 | 2 | 0 | 1 |
| `ale_ahc010` | 1 | **0→1** | 0 | **1→2** |
| `ale_ahc019` | **2→1** | 1 | 0 | 2 |
| `ale_ahc025` | 1 | 2 | 0 | 2 |
| `ale_ahc030` | 1 | 1 | 0 | 1 |
| `ale_ahc035` | 2 | 1 | 0 | 1 |
| `ale_ahc038` | 2 | **3→2** | **1→0** | **1→2** |
| `ale_ahc039` | 4 | 3 | 0 | 3 |
| `ale_ahc041` | 2 | 2 | 0 | 2 |
| `ale_ahc044` | 2 | 1 | 0 | 1 |
| `ale_ahc045` | **2→1** | 1 | 0 | 1 |

