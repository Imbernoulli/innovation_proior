# frontiercs: 与官方 frontier-model 解池的相似度(complete 且有代码的 common 样本)

标尺:frontier 模型的解与**其他** frontier 模型的解的最大 4-gram Jaccard(留一模型):median 0.198, q25 0.143, q75 0.319, n=7617(题数 172)

## 9B

| arm | stage | n | med jac_frontier | med sim_frontier | jac>0.3 | sim>0.5 | mean(jac − frontier self-jac of same problem) | nearest frontier model (top3) |
|---|---|---|---|---|---|---|---|---|
| base9b_v2c | base | 608 | 0.166 | 0.368 | 6% | 21% | -0.103 | deepseekreasoner:120, gemini3pro:114, grok4fastreasoning:86 |
| ft01mix_a10 | sft | 562 | 0.164 | 0.362 | 7% | 22% | -0.100 | deepseekreasoner:109, grok4fastreasoning:92, gemini3pro:92 |
| ft03nm_a20 | sft | 585 | 0.160 | 0.358 | 8% | 20% | -0.103 | grok4fastreasoning:124, deepseekreasoner:98, gemini3pro:89 |
| lo32nm_a10 | sft | 597 | 0.163 | 0.357 | 7% | 22% | -0.104 | deepseekreasoner:111, gemini3pro:95, grok4fastreasoning:92 |
| rlv5_base_s20 | rl_base | 390 | 0.212 | 0.460 | 22% | 41% | -0.040 | grok4fastreasoning:76, gemini3pro:63, gpt5.1:54 |
| rlv5_ft01mix_a10_s20 | rl_sft | 531 | 0.232 | 0.526 | 33% | 56% | -0.026 | gpt5.1:86, gpt5_low:79, grok4fastreasoning:70 |
| rlv5_ft03nm_a20_s20 | rl_sft | 621 | 0.208 | 0.447 | 20% | 42% | -0.050 | gemini3pro:106, gpt5.1:101, deepseekreasoner:92 |
| rlv5_lo32nm_a10_s20 | rl_sft | 496 | 0.232 | 0.506 | 29% | 51% | -0.025 | gemini3pro:90, gpt5.1:74, gpt5_low:64 |
| base9b_v2c_y2026 | rep | 583 | 0.163 | 0.368 | 8% | 23% | -0.103 | deepseekreasoner:95, gemini3pro:95, grok4fastreasoning:93 |
| lo32nm_a10_y2026 | rep | 596 | 0.164 | 0.365 | 8% | 21% | -0.102 | deepseekreasoner:127, gemini3pro:101, grok4fastreasoning:97 |
| rlv5_lo32nm_a10_s20_y2026 | rep | 473 | 0.224 | 0.501 | 24% | 50% | -0.033 | gpt5.1:80, gemini3pro:78, deepseekreasoner:69 |

### 配对(按题均值):该臂比对照更像/更不像 frontier 池?

| pair | n probs | jac_frontier +/−/= | mean Δ | Wilcoxon p | arm wins where arm's best is closer to frontier pool | ctrl wins where ctrl's best is closer |
|---|---|---|---|---|---|---|
| ft01mix_a10 − base9b_v2c | 157 | 74−83−0 | 0.001 | 0.920 | 21/30 | 12/25 |
| ft03nm_a20 − base9b_v2c | 157 | 73−84−0 | -0.002 | 0.605 | 19/31 | 11/18 |
| lo32nm_a10 − base9b_v2c | 160 | 74−86−0 | -0.003 | 0.471 | 19/28 | 21/31 |
| rlv5_base_s20 − base9b_v2c | 132 | 99−33−0 | 0.048 | 0.000 | 27/33 | 8/24 |
| rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 127 | 66−61−0 | 0.017 | 0.236 | 12/19 | 12/20 |
| rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 132 | 61−71−0 | 0.000 | 0.467 | 12/30 | 10/18 |
| rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 123 | 65−58−0 | 0.017 | 0.131 | 14/24 | 10/21 |
| base9b_v2c_y2026 − base9b_v2c | 159 | 86−73−0 | -0.001 | 0.704 | 18/28 | 14/23 |
| lo32nm_a10_y2026 − lo32nm_a10 | 159 | 72−87−0 | 0.003 | 0.959 | 12/28 | 13/24 |
| rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 138 | 62−76−0 | -0.012 | 0.148 | 10/24 | 10/18 |
| rlv5_ft01mix_a10_s20 − ft01mix_a10 | 147 | 121−26−0 | 0.073 | 0.000 | 16/24 | 11/29 |
| rlv5_ft03nm_a20_s20 − ft03nm_a20 | 155 | 130−25−0 | 0.051 | 0.000 | 25/37 | 9/22 |
| rlv5_lo32nm_a10_s20 − lo32nm_a10 | 145 | 121−24−0 | 0.079 | 0.000 | 29/36 | 7/16 |

### 题内关联:像 frontier 池 ↔ 分数(main 臂样本,题内排秩)

| property | n | Spearman rho vs score | p | low tercile mean/pass | mid | high |
|---|---|---|---|---|---|---|
| 4-gram Jaccard to nearest frontier | 4381 | 0.220 | 0.0000 | 5.69 / 0% | 10.71 / 0% | 12.56 / 0% |
| token ratio to nearest frontier | 4381 | 0.191 | 0.0000 | 6.77 / 0% | 9.55 / 0% | 12.55 / 1% |

## 4B

| arm | stage | n | med jac_frontier | med sim_frontier | jac>0.3 | sim>0.5 | mean(jac − frontier self-jac of same problem) | nearest frontier model (top3) |
|---|---|---|---|---|---|---|---|---|
| base4b | base | 786 | 0.163 | 0.351 | 8% | 21% | -0.110 | gemini3pro:169, deepseekreasoner:113, gpt5.1:96 |
| 4b_ft01mix_a10 | sft | 743 | 0.161 | 0.363 | 8% | 22% | -0.106 | gemini3pro:164, deepseekreasoner:116, gpt5.1:100 |
| 4b_lo32nm_a10 | sft | 800 | 0.152 | 0.356 | 7% | 20% | -0.118 | gemini3pro:160, deepseekreasoner:118, gpt5.1:107 |
| rlv5_4b_base_s20 | rl_base | 22 | 0.175 | 0.373 | 23% | 41% | -0.051 | grok4fastreasoning:4, gemini3pro:3, gpt5.1:3 |
| rlv5_4b_ft01mix_a10_s20 | rl_sft | 528 | 0.244 | 0.545 | 33% | 57% | -0.012 | gpt5_low:91, gpt5.1:89, gemini3pro:60 |
| rlv5_4b_lo32nm_a10_s20 | rl_sft | 650 | 0.226 | 0.512 | 29% | 53% | -0.030 | gpt5.1:114, gpt5_low:93, grok4fastreasoning:85 |
| base4b_y2026 | rep | 780 | 0.163 | 0.359 | 9% | 23% | -0.109 | gemini3pro:167, deepseekreasoner:128, gemini2.5pro:90 |
| 4b_lo32nm_a10_y2026 | rep | 787 | 0.153 | 0.348 | 7% | 19% | -0.116 | gemini3pro:160, deepseekreasoner:123, gpt5.1:88 |
| rlv5_4b_lo32nm_a10_s20_y2026 | rep | 679 | 0.225 | 0.527 | 27% | 54% | -0.028 | gpt5.1:114, gpt5_low:97, grok4fastreasoning:87 |

### 配对(按题均值):该臂比对照更像/更不像 frontier 池?

| pair | n probs | jac_frontier +/−/= | mean Δ | Wilcoxon p | arm wins where arm's best is closer to frontier pool | ctrl wins where ctrl's best is closer |
|---|---|---|---|---|---|---|
| 4b_ft01mix_a10 − base4b | 171 | 84−87−0 | 0.001 | 0.874 | 10/16 | 18/29 |
| 4b_lo32nm_a10 − base4b | 172 | 80−92−0 | -0.007 | 0.260 | 11/18 | 22/29 |
| rlv5_4b_base_s20 − base4b | 19 | 11−8−0 | 0.029 | 0.210 | 1/1 | 6/9 |
| rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 19 | 15−4−0 | 0.051 | 0.005 | 6/10 | 0/0 |
| rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 19 | 16−3−0 | 0.032 | 0.018 | 9/10 | 0/0 |
| base4b_y2026 − base4b | 171 | 87−84−0 | -0.000 | 0.555 | 15/24 | 16/26 |
| 4b_lo32nm_a10_y2026 − 4b_lo32nm_a10 | 171 | 88−83−0 | 0.002 | 0.818 | 15/23 | 10/20 |
| rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 168 | 91−77−0 | 0.003 | 0.212 | 10/22 | 15/28 |
| rlv5_4b_ft01mix_a10_s20 − 4b_ft01mix_a10 | 154 | 133−21−0 | 0.092 | 0.000 | 27/36 | 4/15 |
| rlv5_4b_lo32nm_a10_s20 − 4b_lo32nm_a10 | 170 | 150−20−0 | 0.083 | 0.000 | 28/36 | 9/21 |

### 题内关联:像 frontier 池 ↔ 分数(main 臂样本,题内排秩)

| property | n | Spearman rho vs score | p | low tercile mean/pass | mid | high |
|---|---|---|---|---|---|---|
| 4-gram Jaccard to nearest frontier | 3515 | 0.195 | 0.0000 | 3.49 / 0% | 6.26 / 0% | 9.52 / 0% |
| token ratio to nearest frontier | 3515 | 0.150 | 0.0000 | 4.20 / 0% | 6.36 / 0% | 8.67 / 0% |

# frontiercs_research: 与官方 frontier-model 解池的相似度(complete 且有代码的 common 样本)

标尺:frontier 模型的解与**其他** frontier 模型的解的最大 4-gram Jaccard(留一模型):median 0.253, q25 0.181, q75 0.367, n=2166(题数 62)

## 9B

| arm | stage | n | med jac_frontier | med sim_frontier | jac>0.3 | sim>0.5 | mean(jac − frontier self-jac of same problem) | nearest frontier model (top3) |
|---|---|---|---|---|---|---|---|---|
| base9b_v2c | base | 279 | 0.253 | 0.420 | 38% | 40% | -0.019 | grok4fastreasoning:105, gemini3pro:66, deepseekreasoner:45 |
| ft01mix_a10 | sft | 277 | 0.244 | 0.427 | 37% | 40% | -0.017 | grok4fastreasoning:127, gemini3pro:58, deepseekreasoner:38 |
| ft03nm_a20 | sft | 277 | 0.239 | 0.418 | 35% | 39% | -0.020 | grok4fastreasoning:115, gemini3pro:67, deepseekreasoner:45 |
| lo32nm_a10 | sft | 278 | 0.256 | 0.430 | 36% | 41% | -0.016 | grok4fastreasoning:127, gemini3pro:59, deepseekreasoner:33 |
| rlv5_base_s20 | rl_base | 251 | 0.303 | 0.560 | 51% | 59% | 0.072 | grok4fastreasoning:159, gemini3pro:49, gemini2.5pro:16 |
| rlv5_ft01mix_a10_s20 | rl_sft | 267 | 0.305 | 0.552 | 51% | 55% | 0.057 | grok4fastreasoning:154, gemini3pro:57, gemini2.5pro:20 |
| rlv5_ft03nm_a20_s20 | rl_sft | 258 | 0.294 | 0.553 | 50% | 55% | 0.065 | grok4fastreasoning:140, gemini3pro:55, gemini2.5pro:30 |
| rlv5_lo32nm_a10_s20 | rl_sft | 265 | 0.287 | 0.524 | 48% | 54% | 0.062 | grok4fastreasoning:146, gemini3pro:59, gemini2.5pro:22 |
| base9b_v2c_y2026 | rep | 278 | 0.252 | 0.456 | 38% | 44% | -0.018 | grok4fastreasoning:114, gemini3pro:68, deepseekreasoner:42 |
| lo32nm_a10_y2026 | rep | 277 | 0.252 | 0.414 | 35% | 39% | -0.015 | grok4fastreasoning:122, gemini3pro:65, deepseekreasoner:43 |
| rlv5_lo32nm_a10_s20_y2026 | rep | 264 | 0.298 | 0.535 | 49% | 56% | 0.051 | grok4fastreasoning:156, gemini3pro:57, gemini2.5pro:20 |

### 配对(按题均值):该臂比对照更像/更不像 frontier 池?

| pair | n probs | jac_frontier +/−/= | mean Δ | Wilcoxon p | arm wins where arm's best is closer to frontier pool | ctrl wins where ctrl's best is closer |
|---|---|---|---|---|---|---|
| ft01mix_a10 − base9b_v2c | 59 | 29−29−1 | 0.006 | 0.892 | 4/11 | 6/11 |
| ft03nm_a20 − base9b_v2c | 59 | 29−30−0 | 0.001 | 0.850 | 7/12 | 7/11 |
| lo32nm_a10 − base9b_v2c | 59 | 28−31−0 | 0.004 | 0.678 | 3/7 | 4/9 |
| rlv5_base_s20 − base9b_v2c | 59 | 43−16−0 | 0.071 | 0.000 | 8/12 | 2/8 |
| rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 59 | 26−32−1 | -0.003 | 0.760 | 5/11 | 5/7 |
| rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 59 | 22−36−1 | -0.004 | 0.169 | 5/9 | 3/8 |
| rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 59 | 31−27−1 | 0.000 | 0.673 | 4/12 | 5/8 |
| base9b_v2c_y2026 − base9b_v2c | 58 | 30−28−0 | 0.007 | 0.607 | 3/7 | 8/13 |
| lo32nm_a10_y2026 − lo32nm_a10 | 59 | 35−24−0 | -0.003 | 0.613 | 4/12 | 5/9 |
| rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 58 | 23−34−1 | -0.006 | 0.108 | 2/5 | 7/13 |
| rlv5_ft01mix_a10_s20 − ft01mix_a10 | 59 | 45−14−0 | 0.063 | 0.000 | 13/15 | 3/9 |
| rlv5_ft03nm_a20_s20 − ft03nm_a20 | 59 | 37−21−1 | 0.067 | 0.002 | 10/12 | 5/9 |
| rlv5_lo32nm_a10_s20 − lo32nm_a10 | 59 | 44−15−0 | 0.068 | 0.000 | 9/15 | 1/4 |

### 题内关联:像 frontier 池 ↔ 分数(main 臂样本,题内排秩)

| property | n | Spearman rho vs score | p | low tercile mean/pass | mid | high |
|---|---|---|---|---|---|---|
| 4-gram Jaccard to nearest frontier | 2152 | 0.236 | 0.0000 | 6.95 / 0% | 12.19 / 1% | 15.60 / 1% |
| token ratio to nearest frontier | 2152 | 0.189 | 0.0000 | 8.22 / 0% | 11.47 / 1% | 14.95 / 1% |

## 4B

| arm | stage | n | med jac_frontier | med sim_frontier | jac>0.3 | sim>0.5 | mean(jac − frontier self-jac of same problem) | nearest frontier model (top3) |
|---|---|---|---|---|---|---|---|---|
| base4b | base | 290 | 0.201 | 0.370 | 26% | 31% | -0.045 | grok4fastreasoning:86, gemini3pro:66, gemini2.5pro:63 |
| 4b_ft01mix_a10 | sft | 291 | 0.201 | 0.373 | 27% | 31% | -0.049 | grok4fastreasoning:99, gemini3pro:58, gemini2.5pro:58 |
| 4b_lo32nm_a10 | sft | 288 | 0.206 | 0.369 | 26% | 32% | -0.048 | grok4fastreasoning:109, gemini2.5pro:56, gemini3pro:52 |
| rlv5_4b_base_s20 | rl_base | 15 | 0.444 | 0.565 | 53% | 53% | 0.141 | grok4fastreasoning:10, deepseekreasoner:2, gemini2.5pro:2 |
| rlv5_4b_ft01mix_a10_s20 | rl_sft | 238 | 0.403 | 0.629 | 58% | 65% | 0.161 | grok4fastreasoning:157, gemini3pro:40, gemini2.5pro:24 |
| rlv5_4b_lo32nm_a10_s20 | rl_sft | 246 | 0.366 | 0.568 | 56% | 64% | 0.140 | grok4fastreasoning:148, gemini3pro:39, gemini2.5pro:27 |
| base4b_y2026 | rep | 285 | 0.210 | 0.390 | 28% | 34% | -0.049 | grok4fastreasoning:105, gemini3pro:54, gemini2.5pro:49 |
| 4b_lo32nm_a10_y2026 | rep | 287 | 0.207 | 0.385 | 29% | 33% | -0.039 | grok4fastreasoning:104, gemini3pro:59, gemini2.5pro:42 |
| rlv5_4b_lo32nm_a10_s20_y2026 | rep | 257 | 0.364 | 0.557 | 54% | 59% | 0.123 | grok4fastreasoning:172, gemini3pro:34, gemini2.5pro:21 |

### 配对(按题均值):该臂比对照更像/更不像 frontier 池?

| pair | n probs | jac_frontier +/−/= | mean Δ | Wilcoxon p | arm wins where arm's best is closer to frontier pool | ctrl wins where ctrl's best is closer |
|---|---|---|---|---|---|---|
| 4b_ft01mix_a10 − base4b | 61 | 26−35−0 | -0.010 | 0.190 | 7/15 | 4/8 |
| 4b_lo32nm_a10 − base4b | 61 | 27−34−0 | -0.006 | 0.349 | 4/8 | 7/10 |
| rlv5_4b_base_s20 − base4b | 13 | 9−3−1 | 0.117 | 0.064 | 4/5 | 1/2 |
| rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 13 | 10−3−0 | 0.034 | 0.127 | 3/4 | 0/1 |
| rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 13 | 7−6−0 | -0.002 | 0.340 | 3/3 | 3/3 |
| base4b_y2026 − base4b | 60 | 30−30−0 | -0.002 | 0.718 | 7/9 | 7/10 |
| 4b_lo32nm_a10_y2026 − 4b_lo32nm_a10 | 61 | 33−28−0 | 0.008 | 0.299 | 4/10 | 5/7 |
| rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 61 | 31−30−0 | 0.001 | 0.946 | 4/8 | 4/5 |
| rlv5_4b_ft01mix_a10_s20 − 4b_ft01mix_a10 | 61 | 53−8−0 | 0.159 | 0.000 | 17/19 | 0/3 |
| rlv5_4b_lo32nm_a10_s20 − 4b_lo32nm_a10 | 61 | 47−14−0 | 0.136 | 0.000 | 19/23 | 2/4 |

### 题内关联:像 frontier 池 ↔ 分数(main 臂样本,题内排秩)

| property | n | Spearman rho vs score | p | low tercile mean/pass | mid | high |
|---|---|---|---|---|---|---|
| 4-gram Jaccard to nearest frontier | 1368 | 0.364 | 0.0000 | 4.61 / 1% | 9.87 / 2% | 17.83 / 1% |
| token ratio to nearest frontier | 1368 | 0.297 | 0.0000 | 5.08 / 1% | 10.86 / 1% | 16.35 / 2% |

## 有人类参考解的题(仅此几题)

| problem | arm | n | max sim to human ref | mean score | best score |
|---|---|---|---|---|---|
| nbody_simulation/random_100k | 4b_ft01mix_a10 | 5 | 0.285 | 0.0 | 0.0 |
| nbody_simulation/random_100k | 4b_lo32nm_a10 | 5 | 0.281 | 0.0 | 0.0 |
| nbody_simulation/random_100k | 4b_lo32nm_a10_y2026 | 5 | 0.295 | 0.0 | 0.0 |
| nbody_simulation/random_100k | base4b | 5 | 0.530 | 0.0 | 0.0 |
| nbody_simulation/random_100k | base4b_y2026 | 5 | 0.282 | 0.0 | 0.0 |
| nbody_simulation/random_100k | base9b_v2c | 5 | 0.272 | 0.0 | 0.0 |
| nbody_simulation/random_100k | base9b_v2c_y2026 | 5 | 0.300 | 0.0 | 0.0 |
| nbody_simulation/random_100k | ft01mix_a10 | 5 | 0.246 | 0.0 | 0.0 |
| nbody_simulation/random_100k | ft03nm_a20 | 5 | 0.368 | 0.0 | 0.0 |
| nbody_simulation/random_100k | lo32nm_a10 | 5 | 0.443 | 0.0 | 0.0 |
| nbody_simulation/random_100k | lo32nm_a10_y2026 | 5 | 0.258 | 0.0 | 0.0 |
| nbody_simulation/random_100k | rlv5_4b_ft01mix_a10_s20 | 3 | 0.347 | 0.0 | 0.0 |
| nbody_simulation/random_100k | rlv5_4b_lo32nm_a10_s20 | 3 | 0.983 | 0.0 | 0.0 |
| nbody_simulation/random_100k | rlv5_4b_lo32nm_a10_s20_y2026 | 3 | 0.491 | 0.0 | 0.0 |
| nbody_simulation/random_100k | rlv5_base_s20 | 5 | 0.445 | 0.0 | 0.0 |
| nbody_simulation/random_100k | rlv5_ft01mix_a10_s20 | 5 | 0.508 | 0.0 | 0.0 |
| nbody_simulation/random_100k | rlv5_ft03nm_a20_s20 | 5 | 0.458 | 0.0 | 0.0 |
| nbody_simulation/random_100k | rlv5_lo32nm_a10_s20 | 5 | 0.393 | 0.0 | 0.0 |
| nbody_simulation/random_100k | rlv5_lo32nm_a10_s20_y2026 | 4 | 0.433 | 0.0 | 0.0 |
| nbody_simulation/random_10k | 4b_ft01mix_a10 | 5 | 0.402 | 0.0 | 0.0 |
| nbody_simulation/random_10k | 4b_lo32nm_a10 | 5 | 0.367 | 0.0 | 0.0 |
| nbody_simulation/random_10k | 4b_lo32nm_a10_y2026 | 5 | 0.341 | 0.0 | 0.0 |
| nbody_simulation/random_10k | base4b | 5 | 0.719 | 0.0 | 0.0 |
| nbody_simulation/random_10k | base4b_y2026 | 5 | 0.360 | 12.0 | 60.2 |
| nbody_simulation/random_10k | base9b_v2c | 5 | 0.550 | 0.2 | 1.0 |
| nbody_simulation/random_10k | base9b_v2c_y2026 | 5 | 0.559 | 0.0 | 0.0 |
| nbody_simulation/random_10k | ft01mix_a10 | 5 | 0.332 | 0.0 | 0.0 |
| nbody_simulation/random_10k | ft03nm_a20 | 5 | 0.285 | 0.0 | 0.0 |
| nbody_simulation/random_10k | lo32nm_a10 | 5 | 0.302 | 0.0 | 0.0 |
| nbody_simulation/random_10k | lo32nm_a10_y2026 | 5 | 0.386 | 0.0 | 0.0 |
| nbody_simulation/random_10k | rlv5_4b_ft01mix_a10_s20 | 5 | 0.986 | 0.0 | 0.0 |
| nbody_simulation/random_10k | rlv5_4b_lo32nm_a10_s20 | 2 | 0.415 | 0.0 | 0.0 |
| nbody_simulation/random_10k | rlv5_4b_lo32nm_a10_s20_y2026 | 5 | 0.956 | 0.0 | 0.0 |
| nbody_simulation/random_10k | rlv5_base_s20 | 5 | 0.456 | 6.4 | 31.7 |
| nbody_simulation/random_10k | rlv5_ft01mix_a10_s20 | 5 | 0.459 | 0.6 | 2.9 |
| nbody_simulation/random_10k | rlv5_ft03nm_a20_s20 | 5 | 0.399 | 0.0 | 0.0 |
| nbody_simulation/random_10k | rlv5_lo32nm_a10_s20 | 5 | 0.410 | 0.0 | 0.0 |
| nbody_simulation/random_10k | rlv5_lo32nm_a10_s20_y2026 | 5 | 0.717 | 3.5 | 10.5 |
