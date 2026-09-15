
# mean@draw:rl_sft vs rl_base(逐题配对)

## 9B

| arm | bench | n | mean Δ | +/− | Wilcoxon p | sign p |
|---|---|---|---|---|---|---|
| rlv5_ft01mix_a10_s20 | frontiercs | 170 | +0.117 | 37/25 | 0.455 | 0.162 |
| rlv5_ft01mix_a10_s20 | frontiercs_research | 59 | +2.212 | 22/10 | 0.140 | 0.050 |
| rlv5_ft01mix_a10_s20 | alebench | 40 | +17.685 | 16/15 | 0.493 | 1.000 |
| rlv5_ft01mix_a10_s20 | mlsbench | 21 | +0.0421 | 7/4 | 0.248 | 0.549 |
| rlv5_ft03nm_a20_s20 | frontiercs | 170 | -0.197 | 39/28 | 0.508 | 0.222 |
| rlv5_ft03nm_a20_s20 | frontiercs_research | 59 | -0.075 | 15/17 | 0.926 | 0.860 |
| rlv5_ft03nm_a20_s20 | alebench | – | – | – | ⚠判题拓扑不同,剔除 | – |
| rlv5_ft03nm_a20_s20 | mlsbench | 21 | +0.0474 | 6/2 | 0.093 | 0.289 |
| rlv5_lo32nm_a10_s20 | frontiercs | 170 | +0.716 | 37/26 | 0.400 | 0.207 |
| rlv5_lo32nm_a10_s20 | frontiercs_research | 59 | +1.342 | 18/14 | 0.304 | 0.597 |
| rlv5_lo32nm_a10_s20 | alebench | 40 | -1.690 | 16/14 | 0.750 | 0.856 |
| rlv5_lo32nm_a10_s20 | mlsbench | 21 | +0.0312 | 5/6 | 0.534 | 1.000 |

合并(Stouffer,单边 rl_sft>rl_base):
- rlv5_ft01mix_a10_s20:Z=2.03,p=0.0210(4 个 bench;方向为正 4/4)
- rlv5_ft03nm_a20_s20:Z=0.53,p=0.2967(3 个 bench;方向为正 1/3)
- rlv5_lo32nm_a10_s20:Z=1.09,p=0.1385(4 个 bench;方向为正 3/4)
- **全部 (臂 × bench) 方向一致性:8/11 为正,符号检验 p=0.2266**

## 4B

| arm | bench | n | mean Δ | +/− | Wilcoxon p | sign p |
|---|---|---|---|---|---|---|
| rlv5_4b_ft01mix_a10_s20 | frontiercs | 172 | +5.652 | 57/0 | 0.000 | 0.000 |
| rlv5_4b_ft01mix_a10_s20 | frontiercs_research | 61 | +11.782 | 30/1 | 0.000 | 0.000 |
| rlv5_4b_ft01mix_a10_s20 | alebench | 39 | +212.364 | 28/0 | 0.000 | 0.000 |
| rlv5_4b_lo32nm_a10_s20 | frontiercs | 172 | +6.647 | 62/0 | 0.000 | 0.000 |
| rlv5_4b_lo32nm_a10_s20 | frontiercs_research | 61 | +9.348 | 27/4 | 0.000 | 0.000 |
| rlv5_4b_lo32nm_a10_s20 | alebench | 39 | +185.051 | 28/0 | 0.000 | 0.000 |

合并(Stouffer,单边 rl_sft>rl_base):
- rlv5_4b_ft01mix_a10_s20:Z=9.18,p=0.0000(3 个 bench;方向为正 3/3)
- rlv5_4b_lo32nm_a10_s20:Z=9.03,p=0.0000(3 个 bench;方向为正 3/3)
- **全部 (臂 × bench) 方向一致性:6/6 为正,符号检验 p=0.0312**

# P(complete):rl_sft vs rl_base(逐题配对)

## 9B

| arm | bench | n | mean Δ | +/− | Wilcoxon p | sign p |
|---|---|---|---|---|---|---|
| rlv5_ft01mix_a10_s20 | frontiercs | 170 | +0.169 | 84/23 | 0.000 | 0.000 |
| rlv5_ft01mix_a10_s20 | frontiercs_research | 59 | +0.060 | 12/1 | 0.003 | 0.003 |
| rlv5_ft01mix_a10_s20 | alebench | 40 | +0.085 | 16/9 | 0.058 | 0.230 |
| rlv5_ft03nm_a20_s20 | frontiercs | 170 | +0.281 | 113/10 | 0.000 | 0.000 |
| rlv5_ft03nm_a20_s20 | frontiercs_research | 59 | +0.030 | 11/6 | 0.189 | 0.332 |
| rlv5_ft03nm_a20_s20 | alebench | – | – | – | ⚠判题拓扑不同,剔除 | – |
| rlv5_lo32nm_a10_s20 | frontiercs | 170 | +0.124 | 85/25 | 0.000 | 0.000 |
| rlv5_lo32nm_a10_s20 | frontiercs_research | 59 | +0.053 | 10/2 | 0.009 | 0.039 |
| rlv5_lo32nm_a10_s20 | alebench | 40 | +0.045 | 15/8 | 0.399 | 0.210 |

合并(Stouffer,单边 rl_sft>rl_base):
- rlv5_ft01mix_a10_s20:Z=6.40,p=0.0000(3 个 bench;方向为正 3/3)
- rlv5_ft03nm_a20_s20:Z=5.97,p=0.0000(2 个 bench;方向为正 2/2)
- rlv5_lo32nm_a10_s20:Z=4.91,p=0.0000(3 个 bench;方向为正 3/3)
- **全部 (臂 × bench) 方向一致性:8/8 为正,符号检验 p=0.0078**

## 4B

| arm | bench | n | mean Δ | +/− | Wilcoxon p | sign p |
|---|---|---|---|---|---|---|
| rlv5_4b_ft01mix_a10_s20 | frontiercs | 172 | +0.613 | 154/0 | 0.000 | 0.000 |
| rlv5_4b_ft01mix_a10_s20 | frontiercs_research | 61 | +0.758 | 59/0 | 0.000 | 0.000 |
| rlv5_4b_ft01mix_a10_s20 | alebench | 39 | +0.744 | 36/0 | 0.000 | 0.000 |
| rlv5_4b_lo32nm_a10_s20 | frontiercs | 172 | +0.761 | 170/0 | 0.000 | 0.000 |
| rlv5_4b_lo32nm_a10_s20 | frontiercs_research | 61 | +0.785 | 60/0 | 0.000 | 0.000 |
| rlv5_4b_lo32nm_a10_s20 | alebench | 39 | +0.785 | 39/0 | 0.000 | 0.000 |

合并(Stouffer,单边 rl_sft>rl_base):
- rlv5_4b_ft01mix_a10_s20:Z=11.09,p=0.0000(3 个 bench;方向为正 3/3)
- rlv5_4b_lo32nm_a10_s20:Z=11.24,p=0.0000(3 个 bench;方向为正 3/3)
- **全部 (臂 × bench) 方向一致性:6/6 为正,符号检验 p=0.0312**


# 对照:SFT vs base(RL 之前),同样口径

## 9B

| arm | bench | n | mean Δ | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| ft01mix_a10 | frontiercs | 170 | +1.291 | 35/33 | 0.131 |
| ft01mix_a10 | frontiercs_research | 59 | +2.334 | 18/14 | 0.270 |
| ft01mix_a10 | alebench | 40 | -13.210 | 13/13 | 0.990 |
| ft03nm_a20 | frontiercs | 170 | -0.024 | 28/33 | 0.923 |
| ft03nm_a20 | frontiercs_research | 59 | +0.762 | 13/17 | 0.750 |
| ft03nm_a20 | alebench | 40 | +11.215 | 17/13 | 0.453 |
| lo32nm_a10 | frontiercs | 170 | +0.060 | 31/39 | 0.829 |
| lo32nm_a10 | frontiercs_research | 59 | +0.886 | 14/14 | 0.811 |
| lo32nm_a10 | alebench | 40 | -9.990 | 13/12 | 0.946 |

- 合并 Z=1.32,p=0.0934;方向为正 6/9,符号检验 p=0.5078

## 4B

| arm | bench | n | mean Δ | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| 4b_ft01mix_a10 | frontiercs | 172 | -1.036 | 24/36 | 0.046 |
| 4b_ft01mix_a10 | frontiercs_research | 61 | +2.079 | 17/8 | 0.074 |
| 4b_ft01mix_a10 | alebench | 39 | -31.897 | 6/12 | 0.058 |
| 4b_lo32nm_a10 | frontiercs | 172 | -0.960 | 22/34 | 0.133 |
| 4b_lo32nm_a10 | frontiercs_research | 61 | -0.258 | 10/11 | 0.945 |
| 4b_lo32nm_a10 | alebench | 39 | -3.590 | 9/10 | 0.872 |

- 合并 Z=-1.56,p=0.9411;方向为正 1/6,符号检验 p=0.2188
