# 方向一致性聚合(逐格 + 跨格聚合)

口径:逐题配对(先按题对 sample 取均值),Wilcoxon 双侧;
聚合 = 格子方向的符号检验 + 逐格 signed Z 的 Stouffer 合并。

### 1a. 9B:RL(先验) − RL(base),分数
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 172 | +0.088 | 38/27 | 0.4662 |
| frontiercs | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 172 | -0.203 | 39/31 | 0.5216 |
| frontiercs | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 172 | +0.691 | 38/28 | 0.4045 |
| frontiercs_research | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 64 | +1.885 | 21/13 | 0.2280 |
| frontiercs_research | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 64 | -0.247 | 17/19 | 0.9567 |
| frontiercs_research | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 64 | +1.747 | 19/16 | 0.2192 |
| alebench | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 40 | +16.350 | 16/16 | 0.5361 |
| alebench | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 40 | -15.920 | 17/16 | 0.7243 |
| alebench | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 40 | -1.690 | 16/14 | 0.7611 |

**聚合:5/9 个格子方向为正,符号检验 p=1.0000;Stouffer Z=+1.95,p=0.0508**

### 1b. 4B:RL(先验) − RL(base),分数
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 172 | +5.602 | 57/0 | 0.0000 |
| frontiercs | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 172 | +6.609 | 62/0 | 0.0000 |
| frontiercs_research | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 64 | +11.489 | 33/2 | 0.0000 |
| frontiercs_research | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 64 | +8.381 | 29/4 | 0.0000 |
| alebench | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 40 | +210.065 | 29/0 | 0.0000 |
| alebench | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 40 | +183.435 | 29/0 | 0.0000 |

**聚合:6/6 个格子方向为正,符号检验 p=0.0312;Stouffer Z=+13.98,p=0.0000**

### 1c. 两个尺度合并
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 172 | +0.088 | 38/27 | 0.4662 |
| frontiercs | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 172 | -0.203 | 39/31 | 0.5216 |
| frontiercs | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 172 | +0.691 | 38/28 | 0.4045 |
| frontiercs_research | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 64 | +1.885 | 21/13 | 0.2280 |
| frontiercs_research | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 64 | -0.247 | 17/19 | 0.9567 |
| frontiercs_research | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 64 | +1.747 | 19/16 | 0.2192 |
| alebench | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 40 | +16.350 | 16/16 | 0.5361 |
| alebench | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 40 | -15.920 | 17/16 | 0.7243 |
| alebench | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 40 | -1.690 | 16/14 | 0.7611 |
| frontiercs | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 172 | +5.602 | 57/0 | 0.0000 |
| frontiercs | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 172 | +6.609 | 62/0 | 0.0000 |
| frontiercs_research | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 64 | +11.489 | 33/2 | 0.0000 |
| frontiercs_research | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 64 | +8.381 | 29/4 | 0.0000 |
| alebench | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 40 | +210.065 | 29/0 | 0.0000 |
| alebench | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 40 | +183.435 | 29/0 | 0.0000 |

**聚合:11/15 个格子方向为正,符号检验 p=0.1185;Stouffer Z=+10.36,p=0.0000**

### 2a. 9B:RL(先验) − RL(base),完成率
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 172 | +0.169 | 87/24 | 0.0000 |
| frontiercs | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 172 | +0.283 | 118/9 | 0.0000 |
| frontiercs | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 172 | +0.129 | 88/24 | 0.0000 |
| frontiercs_research | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 64 | +0.048 | 12/2 | 0.0097 |
| frontiercs_research | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 64 | +0.020 | 11/7 | 0.3199 |
| frontiercs_research | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 64 | +0.049 | 10/2 | 0.0068 |
| alebench | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 40 | +0.080 | 16/10 | 0.0734 |
| alebench | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 40 | +0.145 | 20/5 | 0.0005 |
| alebench | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 40 | +0.045 | 15/8 | 0.3988 |

**聚合:9/9 个格子方向为正,符号检验 p=0.0039;Stouffer Z=+10.44,p=0.0000**

### 2b. 4B:同上
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 172 | +0.610 | 157/0 | 0.0000 |
| frontiercs | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 172 | +0.769 | 171/0 | 0.0000 |
| frontiercs_research | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 64 | +0.748 | 62/0 | 0.0000 |
| frontiercs_research | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 64 | +0.776 | 63/0 | 0.0000 |
| alebench | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 40 | +0.750 | 37/0 | 0.0000 |
| alebench | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 40 | +0.790 | 40/0 | 0.0000 |

**聚合:6/6 个格子方向为正,符号检验 p=0.0312;Stouffer Z=+15.97,p=0.0000**

### 2c. 两个尺度合并
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 172 | +0.169 | 87/24 | 0.0000 |
| frontiercs | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 172 | +0.283 | 118/9 | 0.0000 |
| frontiercs | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 172 | +0.129 | 88/24 | 0.0000 |
| frontiercs_research | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 64 | +0.048 | 12/2 | 0.0097 |
| frontiercs_research | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 64 | +0.020 | 11/7 | 0.3199 |
| frontiercs_research | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 64 | +0.049 | 10/2 | 0.0068 |
| alebench | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 40 | +0.080 | 16/10 | 0.0734 |
| alebench | rlv5_ft03nm_a20_s20 − rlv5_base_s20 | 40 | +0.145 | 20/5 | 0.0005 |
| alebench | rlv5_lo32nm_a10_s20 − rlv5_base_s20 | 40 | +0.045 | 15/8 | 0.3988 |
| frontiercs | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 172 | +0.610 | 157/0 | 0.0000 |
| frontiercs | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 172 | +0.769 | 171/0 | 0.0000 |
| frontiercs_research | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 64 | +0.748 | 62/0 | 0.0000 |
| frontiercs_research | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 64 | +0.776 | 63/0 | 0.0000 |
| alebench | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 40 | +0.750 | 37/0 | 0.0000 |
| alebench | rlv5_4b_lo32nm_a10_s20 − rlv5_4b_base_s20 | 40 | +0.790 | 40/0 | 0.0000 |

**聚合:15/15 个格子方向为正,符号检验 p=0.0001;Stouffer Z=+18.19,p=0.0000**

### 3a. 9B:SFT − base,分数
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | ft01mix_a10 − base9b_v2c | 172 | +1.304 | 36/33 | 0.1179 |
| frontiercs | ft03nm_a20 − base9b_v2c | 172 | -0.005 | 28/35 | 0.8857 |
| frontiercs | lo32nm_a10 − base9b_v2c | 172 | +0.083 | 31/39 | 0.8104 |
| frontiercs_research | ft01mix_a10 − base9b_v2c | 64 | +1.452 | 19/17 | 0.3964 |
| frontiercs_research | ft03nm_a20 − base9b_v2c | 64 | -0.140 | 13/23 | 0.2387 |
| frontiercs_research | lo32nm_a10 − base9b_v2c | 64 | -0.471 | 15/20 | 0.4561 |
| alebench | ft01mix_a10 − base9b_v2c | 40 | -14.845 | 13/14 | 0.9341 |
| alebench | ft03nm_a20 − base9b_v2c | 40 | +9.580 | 17/14 | 0.5043 |
| alebench | lo32nm_a10 − base9b_v2c | 40 | -11.625 | 13/13 | 0.9007 |

**聚合:4/9 个格子方向为正,符号检验 p=1.0000;Stouffer Z=+0.19,p=0.8507**

### 3b. 4B:SFT − base,分数
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | 4b_ft01mix_a10 − base4b | 172 | -0.967 | 24/35 | 0.0623 |
| frontiercs | 4b_lo32nm_a10 − base4b | 172 | -0.926 | 23/34 | 0.1515 |
| frontiercs_research | 4b_ft01mix_a10 − base4b | 64 | +2.304 | 17/9 | 0.0751 |
| frontiercs_research | 4b_lo32nm_a10 − base4b | 64 | -0.034 | 10/13 | 0.8911 |
| alebench | 4b_ft01mix_a10 − base4b | 40 | -31.100 | 6/12 | 0.0599 |
| alebench | 4b_lo32nm_a10 − base4b | 40 | -3.500 | 9/10 | 0.8906 |

**聚合:1/6 个格子方向为正,符号检验 p=0.2188;Stouffer Z=-1.50,p=0.1336**

### 4. 只看 ft01mix 这条线:RL(先验) − RL(base),分数,6 个格子
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 172 | +0.088 | 38/27 | 0.4662 |
| frontiercs_research | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 64 | +1.885 | 21/13 | 0.2280 |
| alebench | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 40 | +16.350 | 16/16 | 0.5361 |
| frontiercs | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 172 | +5.602 | 57/0 | 0.0000 |
| frontiercs_research | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 64 | +11.489 | 33/2 | 0.0000 |
| alebench | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 40 | +210.065 | 29/0 | 0.0000 |

**聚合:6/6 个格子方向为正,符号检验 p=0.0312;Stouffer Z=+8.13,p=0.0000**

### 5. 噪声底:同协议复跑 − 主跑(什么都没改),分数
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | base9b_v2c_y2026 − base9b_v2c | 172 | -0.015 | 31/33 | 0.5996 |
| frontiercs | lo32nm_a10_y2026 − lo32nm_a10 | 172 | -0.552 | 29/34 | 0.3518 |
| frontiercs | rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 172 | -0.863 | 29/38 | 0.1906 |
| frontiercs | base4b_y2026 − base4b | 172 | -0.257 | 31/30 | 0.7739 |
| frontiercs | 4b_lo32nm_a10_y2026 − 4b_lo32nm_a10 | 172 | +0.472 | 29/29 | 0.7363 |
| frontiercs | rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 172 | -1.092 | 27/39 | 0.0231 |
| frontiercs_research | base9b_v2c_y2026 − base9b_v2c | 64 | -0.957 | 16/20 | 0.2854 |
| frontiercs_research | lo32nm_a10_y2026 − lo32nm_a10 | 64 | +1.300 | 19/16 | 0.4609 |
| frontiercs_research | rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 63 | -2.338 | 12/18 | 0.1048 |
| frontiercs_research | base4b_y2026 − base4b | 64 | +0.083 | 13/11 | 0.9203 |
| frontiercs_research | 4b_lo32nm_a10_y2026 − 4b_lo32nm_a10 | 64 | +2.006 | 16/7 | 0.0698 |
| frontiercs_research | rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 64 | +2.797 | 21/12 | 0.0327 |
| alebench | base9b_v2c_y2026 − base9b_v2c | 40 | -57.975 | 10/17 | 0.1111 |
| alebench | lo32nm_a10_y2026 − lo32nm_a10 | 40 | -50.215 | 9/17 | 0.0594 |
| alebench | rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 40 | -31.150 | 13/17 | 0.2129 |
| alebench | base4b_y2026 − base4b | 40 | +20.430 | 9/10 | 0.4413 |
| alebench | 4b_lo32nm_a10_y2026 − 4b_lo32nm_a10 | 40 | -28.350 | 7/10 | 0.2069 |
| alebench | rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 40 | +45.650 | 22/13 | 0.1353 |

**聚合:7/18 个格子方向为正,符号检验 p=0.4807;Stouffer Z=-1.95,p=0.0518**

### 5b. 噪声底:同上,完成率
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | base9b_v2c_y2026 − base9b_v2c | 172 | -0.021 | 26/40 | 0.4643 |
| frontiercs | lo32nm_a10_y2026 − lo32nm_a10 | 172 | +0.000 | 38/32 | 0.9601 |
| frontiercs | rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 172 | -0.024 | 32/51 | 0.1118 |
| frontiercs | base4b_y2026 − base4b | 172 | +0.001 | 15/14 | 0.9374 |
| frontiercs | 4b_lo32nm_a10_y2026 − 4b_lo32nm_a10 | 172 | -0.012 | 7/12 | 0.0725 |
| frontiercs | rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 172 | +0.037 | 55/41 | 0.0325 |
| frontiercs_research | rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 63 | +0.016 | 6/3 | 0.1367 |
| frontiercs_research | rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 64 | +0.048 | 14/6 | 0.1191 |
| alebench | base9b_v2c_y2026 − base9b_v2c | 40 | +0.050 | 13/9 | 0.3849 |
| alebench | lo32nm_a10_y2026 − lo32nm_a10 | 40 | -0.030 | 7/11 | 0.5797 |
| alebench | rlv5_lo32nm_a10_s20_y2026 − rlv5_lo32nm_a10_s20 | 40 | -0.030 | 9/12 | 0.5271 |
| alebench | rlv5_4b_lo32nm_a10_s20_y2026 − rlv5_4b_lo32nm_a10_s20 | 40 | +0.065 | 15/8 | 0.0548 |

**聚合:7/12 个格子方向为正,符号检验 p=0.7744;Stouffer Z=+0.81,p=0.4191**

### 6a. 9B ft01mix 线:RL(先验) − RL(base),分数,3 格
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 172 | +0.088 | 38/27 | 0.4662 |
| frontiercs_research | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 64 | +1.885 | 21/13 | 0.2280 |
| alebench | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 40 | +16.350 | 16/16 | 0.5361 |

**聚合:3/3 个格子方向为正,符号检验 p=0.2500;Stouffer Z=+1.47,p=0.1405**

### 6b. 9B ft01mix 线:完成率,3 格
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 172 | +0.169 | 87/24 | 0.0000 |
| frontiercs_research | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 64 | +0.048 | 12/2 | 0.0097 |
| alebench | rlv5_ft01mix_a10_s20 − rlv5_base_s20 | 40 | +0.080 | 16/10 | 0.0734 |

**聚合:3/3 个格子方向为正,符号检验 p=0.2500;Stouffer Z=+6.23,p=0.0000**

### 6c. 4B ft01mix 线:完成率,3 格
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 172 | +0.610 | 157/0 | 0.0000 |
| frontiercs_research | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 64 | +0.748 | 62/0 | 0.0000 |
| alebench | rlv5_4b_ft01mix_a10_s20 − rlv5_4b_base_s20 | 40 | +0.750 | 37/0 | 0.0000 |

**聚合:3/3 个格子方向为正,符号检验 p=0.2500;Stouffer Z=+11.22,p=0.0000**

### 7a. 9B SFT − base,分数,剔除判题拓扑混杂的 ALE-9B(6 格)
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | ft01mix_a10 − base9b_v2c | 172 | +1.304 | 36/33 | 0.1179 |
| frontiercs | ft03nm_a20 − base9b_v2c | 172 | -0.005 | 28/35 | 0.8857 |
| frontiercs | lo32nm_a10 − base9b_v2c | 172 | +0.083 | 31/39 | 0.8104 |
| frontiercs_research | ft01mix_a10 − base9b_v2c | 64 | +1.452 | 19/17 | 0.3964 |
| frontiercs_research | ft03nm_a20 − base9b_v2c | 64 | -0.140 | 13/23 | 0.2387 |
| frontiercs_research | lo32nm_a10 − base9b_v2c | 64 | -0.471 | 15/20 | 0.4561 |

**聚合:3/6 个格子方向为正,符号检验 p=1.0000;Stouffer Z=+0.04,p=0.9660**

### 7b. 同上,只看 ft01mix(2 格)
| bench | 对比 | n题 | Δ均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|
| frontiercs | ft01mix_a10 − base9b_v2c | 172 | +1.304 | 36/33 | 0.1179 |
| frontiercs_research | ft01mix_a10 − base9b_v2c | 64 | +1.452 | 19/17 | 0.3964 |

**聚合:2/2 个格子方向为正,符号检验 p=0.5000;Stouffer Z=+1.71,p=0.0881**
