# frontiercs

## 9B

- `base9b_v2c` (base): 含跨域词的抽样比例 20.6% / 去掉 simulated annealing 后 3.6%(n=172 题)
- `ft01mix_a10` (sft): 含跨域词的抽样比例 20.1% / 去掉 simulated annealing 后 3.3%(n=171 题)
- `ft03nm_a20` (sft): 含跨域词的抽样比例 19.7% / 去掉 simulated annealing 后 2.9%(n=172 题)
- `lo32nm_a10` (sft): 含跨域词的抽样比例 20.7% / 去掉 simulated annealing 后 2.8%(n=172 题)
- `rlv5_base_s20` (rl_base): 含跨域词的抽样比例 14.6% / 去掉 simulated annealing 后 2.5%(n=172 题)
- `rlv5_ft01mix_a10_s20` (rl_sft): 含跨域词的抽样比例 17.5% / 去掉 simulated annealing 后 3.0%(n=172 题)
- `rlv5_ft03nm_a20_s20` (rl_sft): 含跨域词的抽样比例 14.4% / 去掉 simulated annealing 后 2.7%(n=171 题)
- `rlv5_lo32nm_a10_s20` (rl_sft): 含跨域词的抽样比例 19.0% / 去掉 simulated annealing 后 4.2%(n=172 题)
- `base9b_v2c_y2026` (rep): 含跨域词的抽样比例 21.1% / 去掉 simulated annealing 后 3.7%(n=172 题)
- `lo32nm_a10_y2026` (rep): 含跨域词的抽样比例 19.3% / 去掉 simulated annealing 后 2.5%(n=172 题)
- `rlv5_lo32nm_a10_s20_y2026` (rep): 含跨域词的抽样比例 19.8% / 去掉 simulated annealing 后 2.9%(n=172 题)

| pair | n probs | 全部词 +/−/= | mean Δ | p | 去 SA 后 +/−/= | mean Δ | p |
|---|---|---|---|---|---|---|---|
| ft01mix_a10 − base9b_v2c | 171 | 36−33−102 | -0.006 | 0.739 | 13−16−142 | -0.004 | 0.577 |
| ft03nm_a20 − base9b_v2c | 172 | 27−30−115 | -0.009 | 0.421 | 11−18−143 | -0.007 | 0.351 |
| lo32nm_a10 − base9b_v2c | 172 | 38−32−102 | +0.001 | 0.925 | 15−18−139 | -0.008 | 0.450 |
| rlv5_base_s20 − base9b_v2c | 172 | 21−44−107 | -0.060 | 0.002 | 8−15−149 | -0.011 | 0.254 |
| rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 172 | 36−23−113 | +0.029 | 0.074 | 14−10−148 | +0.005 | 0.399 |
| rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 171 | 30−24−117 | -0.000 | 0.906 | 13−8−150 | +0.005 | 0.414 |
| rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 172 | 39−19−114 | +0.044 | 0.009 | 15−6−151 | +0.017 | 0.026 |
| base9b_v2c_y2026 − base9b_v2c | 172 | 30−30−112 | +0.005 | 0.642 | 17−17−138 | +0.001 | 0.936 |
| lo32nm_a10_y2026 − lo32nm_a10 | 172 | 29−37−106 | -0.014 | 0.377 | 12−14−146 | -0.003 | 0.472 |
| rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 172 | 23−28−121 | +0.008 | 0.813 | 6−14−152 | -0.013 | 0.047 |
| rlv5_ft01mix_a10_s20 − ft01mix_a10 | 171 | 27−43−101 | -0.027 | 0.052 | 11−13−147 | -0.005 | 0.429 |
| rlv5_ft03nm_a20_s20 − ft03nm_a20 | 171 | 23−43−105 | -0.052 | 0.002 | 14−15−142 | +0.000 | 0.766 |
| rlv5_lo32nm_a10_s20 − lo32nm_a10 | 172 | 27−38−107 | -0.017 | 0.241 | 18−13−141 | +0.014 | 0.231 |

## 4B

- `base4b` (base): 含跨域词的抽样比例 23.0% / 去掉 simulated annealing 后 2.9%(n=172 题)
- `4b_ft01mix_a10` (sft): 含跨域词的抽样比例 23.1% / 去掉 simulated annealing 后 2.9%(n=172 题)
- `4b_lo32nm_a10` (sft): 含跨域词的抽样比例 20.5% / 去掉 simulated annealing 后 2.8%(n=172 题)
- `rlv5_4b_base_s20` (rl_base): 含跨域词的抽样比例 4.9% / 去掉 simulated annealing 后 0.0%(n=172 题)
- `rlv5_4b_ft01mix_a10_s20` (rl_sft): 含跨域词的抽样比例 9.0% / 去掉 simulated annealing 后 1.7%(n=172 题)
- `rlv5_4b_lo32nm_a10_s20` (rl_sft): 含跨域词的抽样比例 4.7% / 去掉 simulated annealing 后 0.9%(n=172 题)
- `base4b_y2026` (rep): 含跨域词的抽样比例 25.1% / 去掉 simulated annealing 后 4.0%(n=172 题)
- `4b_lo32nm_a10_y2026` (rep): 含跨域词的抽样比例 21.0% / 去掉 simulated annealing 后 2.0%(n=172 题)
- `rlv5_4b_lo32nm_a10_s20_y2026` (rep): 含跨域词的抽样比例 4.9% / 去掉 simulated annealing 后 1.2%(n=172 题)

| pair | n probs | 全部词 +/−/= | mean Δ | p | 去 SA 后 +/−/= | mean Δ | p |
|---|---|---|---|---|---|---|---|
| 4b_ft01mix_a10 − base4b | 172 | 27−23−122 | +0.001 | 0.795 | 11−14−147 | -0.000 | 0.977 |
| 4b_lo32nm_a10 − base4b | 172 | 25−34−113 | -0.024 | 0.064 | 16−15−141 | -0.001 | 0.967 |
| rlv5_4b_base_s20 − base4b | 172 | 0−68−104 | -0.181 | 0.000 | 0−18−154 | -0.029 | 0.000 |
| rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 172 | 33−8−131 | +0.041 | 0.001 | 13−0−159 | +0.017 | 0.001 |
| rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 172 | 19−15−138 | -0.002 | 0.879 | 8−0−164 | +0.009 | 0.005 |
| base4b_y2026 − base4b | 172 | 40−23−109 | +0.022 | 0.248 | 19−11−142 | +0.010 | 0.195 |
| 4b_lo32nm_a10_y2026 − 4b_lo32nm_a10 | 172 | 32−30−110 | +0.004 | 0.550 | 9−17−146 | -0.008 | 0.179 |
| rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 172 | 18−15−139 | +0.003 | 0.652 | 7−6−159 | +0.002 | 0.617 |
| rlv5_4b_ft01mix_a10_s20 − 4b_ft01mix_a10 | 172 | 9−63−100 | -0.141 | 0.000 | 8−13−151 | -0.012 | 0.167 |
| rlv5_4b_lo32nm_a10_s20 − 4b_lo32nm_a10 | 172 | 6−60−106 | -0.159 | 0.000 | 6−20−146 | -0.019 | 0.005 |

# alebench

## 9B

- `base9b_v2c` (base): 含跨域词的抽样比例 48.0% / 去掉 simulated annealing 后 4.5%(n=40 题)
- `ft01mix_a10` (sft): 含跨域词的抽样比例 52.0% / 去掉 simulated annealing 后 8.5%(n=40 题)
- `ft03nm_a20` (sft): 含跨域词的抽样比例 47.5% / 去掉 simulated annealing 后 7.5%(n=40 题)
- `lo32nm_a10` (sft): 含跨域词的抽样比例 48.0% / 去掉 simulated annealing 后 11.0%(n=40 题)
- `rlv5_base_s20` (rl_base): 含跨域词的抽样比例 26.5% / 去掉 simulated annealing 后 7.5%(n=40 题)
- `rlv5_ft01mix_a10_s20` (rl_sft): 含跨域词的抽样比例 31.0% / 去掉 simulated annealing 后 8.0%(n=40 题)
- `rlv5_ft03nm_a20_s20` (rl_sft): 含跨域词的抽样比例 26.0% / 去掉 simulated annealing 后 9.0%(n=40 题)
- `rlv5_lo32nm_a10_s20` (rl_sft): 含跨域词的抽样比例 44.2% / 去掉 simulated annealing 后 7.8%(n=40 题)
- `base9b_v2c_y2026` (rep): 含跨域词的抽样比例 47.2% / 去掉 simulated annealing 后 10.8%(n=39 题)
- `lo32nm_a10_y2026` (rep): 含跨域词的抽样比例 47.5% / 去掉 simulated annealing 后 8.0%(n=40 题)
- `rlv5_lo32nm_a10_s20_y2026` (rep): 含跨域词的抽样比例 44.1% / 去掉 simulated annealing 后 8.7%(n=39 题)

| pair | n probs | 全部词 +/−/= | mean Δ | p | 去 SA 后 +/−/= | mean Δ | p |
|---|---|---|---|---|---|---|---|
| ft01mix_a10 − base9b_v2c | 40 | 14−12−14 | +0.040 | 0.275 | 9−3−28 | +0.040 | 0.059 |
| ft03nm_a20 − base9b_v2c | 40 | 10−12−18 | -0.005 | 0.922 | 9−4−27 | +0.030 | 0.134 |
| lo32nm_a10 − base9b_v2c | 40 | 13−13−14 | +0.000 | 0.990 | 13−3−24 | +0.065 | 0.011 |
| rlv5_base_s20 − base9b_v2c | 40 | 5−25−10 | -0.215 | 0.001 | 7−5−28 | +0.030 | 0.249 |
| rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 40 | 15−10−15 | +0.045 | 0.384 | 8−7−25 | +0.005 | 0.680 |
| rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 40 | 8−9−23 | -0.005 | 0.943 | 8−7−25 | +0.015 | 0.377 |
| rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 40 | 25−7−8 | +0.177 | 0.001 | 5−7−28 | +0.003 | 0.781 |
| base9b_v2c_y2026 − base9b_v2c | 39 | 10−11−18 | -0.021 | 0.506 | 11−2−26 | +0.062 | 0.012 |
| lo32nm_a10_y2026 − lo32nm_a10 | 40 | 11−10−19 | -0.005 | 0.875 | 3−10−27 | -0.030 | 0.137 |
| rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 39 | 11−11−17 | +0.000 | 0.694 | 7−3−29 | +0.021 | 0.222 |
| rlv5_ft01mix_a10_s20 − ft01mix_a10 | 40 | 3−26−11 | -0.210 | 0.000 | 6−6−28 | -0.005 | 0.967 |
| rlv5_ft03nm_a20_s20 − ft03nm_a20 | 40 | 6−26−8 | -0.215 | 0.000 | 7−8−25 | +0.015 | 0.474 |
| rlv5_lo32nm_a10_s20 − lo32nm_a10 | 40 | 11−16−13 | -0.038 | 0.492 | 6−10−24 | -0.033 | 0.513 |

## 4B

- `base4b` (base): 含跨域词的抽样比例 46.2% / 去掉 simulated annealing 后 6.7%(n=39 题)
- `4b_ft01mix_a10` (sft): 含跨域词的抽样比例 46.7% / 去掉 simulated annealing 后 3.6%(n=39 题)
- `4b_lo32nm_a10` (sft): 含跨域词的抽样比例 44.1% / 去掉 simulated annealing 后 7.7%(n=39 题)
- `rlv5_4b_base_s20` (rl_base): 含跨域词的抽样比例 6.0% / 去掉 simulated annealing 后 0.0%(n=40 题)
- `rlv5_4b_ft01mix_a10_s20` (rl_sft): 含跨域词的抽样比例 12.3% / 去掉 simulated annealing 后 4.1%(n=39 题)
- `rlv5_4b_lo32nm_a10_s20` (rl_sft): 含跨域词的抽样比例 4.1% / 去掉 simulated annealing 后 0.0%(n=39 题)
- `base4b_y2026` (rep): 含跨域词的抽样比例 43.6% / 去掉 simulated annealing 后 9.2%(n=39 题)
- `4b_lo32nm_a10_y2026` (rep): 含跨域词的抽样比例 46.2% / 去掉 simulated annealing 后 7.7%(n=39 题)
- `rlv5_4b_lo32nm_a10_s20_y2026` (rep): 含跨域词的抽样比例 4.1% / 去掉 simulated annealing 后 1.0%(n=39 题)

| pair | n probs | 全部词 +/−/= | mean Δ | p | 去 SA 后 +/−/= | mean Δ | p |
|---|---|---|---|---|---|---|---|
| 4b_ft01mix_a10 − base4b | 39 | 7−11−21 | +0.005 | 0.810 | 2−7−30 | -0.031 | 0.119 |
| 4b_lo32nm_a10 − base4b | 39 | 11−14−14 | -0.021 | 0.532 | 7−7−25 | +0.010 | 0.528 |
| rlv5_4b_base_s20 − base4b | 39 | 0−30−9 | -0.400 | 0.000 | 0−10−29 | -0.067 | 0.003 |
| rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 39 | 13−5−21 | +0.062 | 0.065 | 6−0−33 | +0.041 | 0.020 |
| rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 39 | 5−7−27 | -0.021 | 0.334 | 0−0−39 | +0.000 | nan |
| base4b_y2026 − base4b | 39 | 9−12−18 | -0.026 | 0.420 | 8−3−28 | +0.026 | 0.109 |
| 4b_lo32nm_a10_y2026 − 4b_lo32nm_a10 | 39 | 13−12−14 | +0.021 | 0.533 | 4−7−28 | +0.000 | 0.889 |
| rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 39 | 7−5−27 | +0.000 | 0.796 | 2−0−37 | +0.010 | nan |
| rlv5_4b_ft01mix_a10_s20 − 4b_ft01mix_a10 | 39 | 2−25−12 | -0.344 | 0.000 | 3−3−33 | +0.005 | 0.739 |
| rlv5_4b_lo32nm_a10_s20 − 4b_lo32nm_a10 | 39 | 0−32−7 | -0.400 | 0.000 | 0−11−28 | -0.077 | 0.002 |

# frontiercs_research

## 9B

- `base9b_v2c` (base): 含跨域词的抽样比例 0.3% / 去掉 simulated annealing 后 0.0%(n=63 题)
- `ft01mix_a10` (sft): 含跨域词的抽样比例 0.6% / 去掉 simulated annealing 后 0.3%(n=63 题)
- `ft03nm_a20` (sft): 含跨域词的抽样比例 0.0% / 去掉 simulated annealing 后 0.0%(n=63 题)
- `lo32nm_a10` (sft): 含跨域词的抽样比例 1.0% / 去掉 simulated annealing 后 0.0%(n=63 题)
- `rlv5_base_s20` (rl_base): 含跨域词的抽样比例 2.2% / 去掉 simulated annealing 后 0.3%(n=63 题)
- `rlv5_ft01mix_a10_s20` (rl_sft): 含跨域词的抽样比例 1.9% / 去掉 simulated annealing 后 0.3%(n=63 题)
- `rlv5_ft03nm_a20_s20` (rl_sft): 含跨域词的抽样比例 1.9% / 去掉 simulated annealing 后 0.3%(n=63 题)
- `rlv5_lo32nm_a10_s20` (rl_sft): 含跨域词的抽样比例 2.6% / 去掉 simulated annealing 后 0.3%(n=62 题)
- `base9b_v2c_y2026` (rep): 含跨域词的抽样比例 1.0% / 去掉 simulated annealing 后 0.0%(n=63 题)
- `lo32nm_a10_y2026` (rep): 含跨域词的抽样比例 0.3% / 去掉 simulated annealing 后 0.3%(n=63 题)
- `rlv5_lo32nm_a10_s20_y2026` (rep): 含跨域词的抽样比例 1.6% / 去掉 simulated annealing 后 0.3%(n=63 题)

| pair | n probs | 全部词 +/−/= | mean Δ | p | 去 SA 后 +/−/= | mean Δ | p |
|---|---|---|---|---|---|---|---|
| ft01mix_a10 − base9b_v2c | 63 | 1−0−62 | +0.003 | nan | 1−0−62 | +0.003 | nan |
| ft03nm_a20 − base9b_v2c | 63 | 0−1−62 | -0.003 | nan | 0−0−63 | +0.000 | nan |
| lo32nm_a10 − base9b_v2c | 63 | 1−0−62 | +0.006 | nan | 0−0−63 | +0.000 | nan |
| rlv5_base_s20 − base9b_v2c | 63 | 3−0−60 | +0.019 | nan | 1−0−62 | +0.003 | nan |
| rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 63 | 1−2−60 | -0.003 | nan | 1−1−61 | +0.000 | nan |
| rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 63 | 2−2−59 | -0.003 | nan | 1−1−61 | +0.000 | nan |
| rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 62 | 3−2−57 | +0.003 | nan | 1−1−60 | +0.000 | nan |
| base9b_v2c_y2026 − base9b_v2c | 63 | 1−1−61 | +0.006 | nan | 0−0−63 | +0.000 | nan |
| lo32nm_a10_y2026 − lo32nm_a10 | 63 | 1−2−60 | -0.006 | nan | 1−0−62 | +0.003 | nan |
| rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 62 | 1−3−58 | -0.010 | nan | 1−1−60 | +0.000 | nan |
| rlv5_ft01mix_a10_s20 − ft01mix_a10 | 63 | 2−0−61 | +0.013 | nan | 1−1−61 | +0.000 | nan |
| rlv5_ft03nm_a20_s20 − ft03nm_a20 | 63 | 3−0−60 | +0.019 | nan | 1−0−62 | +0.003 | nan |
| rlv5_lo32nm_a10_s20 − lo32nm_a10 | 62 | 4−0−58 | +0.016 | nan | 1−0−61 | +0.003 | nan |

## 4B

- `base4b` (base): 含跨域词的抽样比例 1.6% / 去掉 simulated annealing 后 0.0%(n=63 题)
- `4b_ft01mix_a10` (sft): 含跨域词的抽样比例 1.3% / 去掉 simulated annealing 后 0.0%(n=63 题)
- `4b_lo32nm_a10` (sft): 含跨域词的抽样比例 0.6% / 去掉 simulated annealing 后 0.0%(n=63 题)
- `rlv5_4b_base_s20` (rl_base): 含跨域词的抽样比例 1.0% / 去掉 simulated annealing 后 0.3%(n=63 题)
- `rlv5_4b_ft01mix_a10_s20` (rl_sft): 含跨域词的抽样比例 1.0% / 去掉 simulated annealing 后 0.3%(n=63 题)
- `rlv5_4b_lo32nm_a10_s20` (rl_sft): 含跨域词的抽样比例 0.0% / 去掉 simulated annealing 后 0.0%(n=63 题)
- `base4b_y2026` (rep): 含跨域词的抽样比例 0.8% / 去掉 simulated annealing 后 0.0%(n=63 题)
- `4b_lo32nm_a10_y2026` (rep): 含跨域词的抽样比例 1.0% / 去掉 simulated annealing 后 0.3%(n=63 题)
- `rlv5_4b_lo32nm_a10_s20_y2026` (rep): 含跨域词的抽样比例 1.0% / 去掉 simulated annealing 后 0.3%(n=63 题)

| pair | n probs | 全部词 +/−/= | mean Δ | p | 去 SA 后 +/−/= | mean Δ | p |
|---|---|---|---|---|---|---|---|
| 4b_ft01mix_a10 − base4b | 63 | 0−1−62 | -0.003 | nan | 0−0−63 | +0.000 | nan |
| 4b_lo32nm_a10 − base4b | 63 | 0−2−61 | -0.010 | nan | 0−0−63 | +0.000 | nan |
| rlv5_4b_base_s20 − base4b | 63 | 0−1−62 | -0.006 | nan | 1−0−62 | +0.003 | nan |
| rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 63 | 1−1−61 | +0.000 | nan | 1−1−61 | +0.000 | nan |
| rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 63 | 0−2−61 | -0.010 | nan | 0−1−62 | -0.003 | nan |
| base4b_y2026 − base4b | 63 | 0−2−61 | -0.008 | nan | 0−0−63 | +0.000 | nan |
| 4b_lo32nm_a10_y2026 − 4b_lo32nm_a10 | 63 | 2−1−60 | +0.003 | nan | 1−0−62 | +0.003 | nan |
| rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 62 | 3−0−59 | +0.010 | nan | 1−0−61 | +0.003 | nan |
| rlv5_4b_ft01mix_a10_s20 − 4b_ft01mix_a10 | 63 | 1−2−60 | -0.003 | nan | 1−0−62 | +0.003 | nan |
| rlv5_4b_lo32nm_a10_s20 − 4b_lo32nm_a10 | 63 | 0−1−62 | -0.006 | nan | 0−0−63 | +0.000 | nan |
