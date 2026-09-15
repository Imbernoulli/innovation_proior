# 覆盖率与可交付性审计

口径:`samples.jsonl` 去重规则 = (ground_truth, sample_idx),跳过 `error` 行,同键后出现的行覆盖先出现的。
`keys_seen` = 该臂该 bench 出现过的键数(含失败的);`rows_ok` = 有分数的键数。

## 1 名义格子 vs 实际落地

| bench | 名义 | 每臂 keys_seen | rows_ok 区间 | 丢失率 |
|---|---|---|---|---|
| FrontierCS | 172 题 x 5 = 860 | [860] | 836–860 | 1.01% |
| FCS-research | 64 题 x 5 = 320 | [320] | 298–313 | 4.47% |
| ALE-Bench | 40 题 x 5 = 200 | [200] | 195–200 | 1.50% |

**每个臂在每个 bench 上的 keys_seen 都等于名义格子数**,即没有任何一个臂少跑过 batch;
所有缺口都是判题侧报错的行(生成是有的,分数没有)。

## 2 逐臂缺口

| 臂 | fam | stage | FCS (rows/860) | research (rows/320) | ALE (rows/200) |
|---|---|---|---|---|---|
| `base9b_v2c` | 9B | base | 852 (−8) | 305 (−15) | 200 (−0) |
| `ft01mix_a10` | 9B | sft | 836 (−24) | 302 (−18) | 200 (−0) |
| `ft03nm_a20` | 9B | sft | 850 (−10) | 301 (−19) | 200 (−0) |
| `lo32nm_a10` | 9B | sft | 850 (−10) | 304 (−16) | 200 (−0) |
| `rlv5_base_s20` | 9B | rl_base | 858 (−2) | 310 (−10) | 200 (−0) |
| `rlv5_ft01mix_a10_s20` | 9B | rl_sft | 857 (−3) | 308 (−12) | 197 (−3) |
| `rlv5_ft03nm_a20_s20` | 9B | rl_sft | 853 (−7) | 311 (−9) | 200 (−0) |
| `rlv5_lo32nm_a10_s20` | 9B | rl_sft | 859 (−1) | 301 (−19) | 197 (−3) |
| `base4b` | 4B | base | 850 (−10) | 307 (−13) | 195 (−5) |
| `4b_ft01mix_a10` | 4B | sft | 851 (−9) | 307 (−13) | 195 (−5) |
| `4b_lo32nm_a10` | 4B | sft | 855 (−5) | 306 (−14) | 195 (−5) |
| `rlv5_4b_base_s20` | 4B | rl_base | 860 (−0) | 313 (−7) | 200 (−0) |
| `rlv5_4b_ft01mix_a10_s20` | 4B | rl_sft | 853 (−7) | 310 (−10) | 195 (−5) |
| `rlv5_4b_lo32nm_a10_s20` | 4B | rl_sft | 848 (−12) | 310 (−10) | 195 (−5) |
| `base9b_v2c_y2026` | 9B | rep | 846 (−14) | 298 (−22) | 195 (−5) |
| `lo32nm_a10_y2026` | 9B | rep | 850 (−10) | 301 (−19) | 196 (−4) |
| `rlv5_lo32nm_a10_s20_y2026` | 9B | rep | 849 (−11) | 302 (−18) | 195 (−5) |
| `base4b_y2026` | 4B | rep | 851 (−9) | 303 (−17) | 195 (−5) |
| `4b_lo32nm_a10_y2026` | 4B | rep | 852 (−8) | 307 (−13) | 195 (−5) |
| `rlv5_4b_lo32nm_a10_s20_y2026` | 4B | rep | 846 (−14) | 308 (−12) | 195 (−5) |

## 3 整题丢失(5 次抽样全部判题失败)

| bench | 题 | 受影响的臂数 | 臂 |
|---|---|---|---|
| FCS-research | `symbolic_regression/sincos` | 19 | (19 个臂) |
| ALE-Bench | `ahc003` | 10 | (10 个臂) |
| FrontierCS | `153` | 2 | `ft01mix_a10`, `rlv5_ft03nm_a20_s20` |
| FCS-research | `grammar_fuzzing/fuzzer/sql` | 2 | `rlv5_lo32nm_a10_s20`, `rlv5_4b_lo32nm_a10_s20_y2026` |

## 4 报错种类(判题基础设施,不是模型)

| bench | 异常类 | 行数 |
|---|---|---|
| FCS-research | `ResearchInfraError` | 1061 |
| FrontierCS | `RuntimeError` | 230 |
| ALE-Bench | `AleInfraError` | 77 |

## 5 配对分析实际用到的公共键

所有配对检验只在“该 family 全部臂都有分数”的键上做,所以上面的缺口不会造成臂间不平衡。

| bench | family | 公共键 | 5 次全在的题 | 至少 1 次在的题 |
|---|---|---|---|---|
| FrontierCS | 9B | 815 | 152 | 170 |
| FrontierCS | 4B | 827 | 157 | 172 |
| ALE-Bench | 9B | 196 | 39 | 40 |
| ALE-Bench | 4B | 195 | 39 | 39 |
| FCS-research | 9B | 279 | 52 | 59 |
| FCS-research | 4B | 291 | 54 | 61 |

