# 双重差分:先验血统 vs base 血统,RL 后半程的变化差

DiD = [先验(晚) − 先验(早)] − [base(晚) − base(早)],逐题配对 Wilcoxon。
自相似度上 DiD<0 = 先验血统的多样性掉得更少(更没坍塌)。

## 4B — P(complete)

| bench | 先验血统 | span | n 题 | DiD 均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|---|
| frontiercs | ft01mix | s15->s20 | 172 | +0.7595 | 163/5 | 0.0000 |
| frontiercs | lo32nm | s15->s20 | 172 | +0.6044 | 157/7 | 0.0000 |
| alebench | ft01mix | s15->s20 | 39 | +0.6718 | 37/1 | 0.0000 |
| alebench | lo32nm | s15->s20 | 39 | +0.5385 | 35/2 | 0.0000 |
| frontiercs_research | ft01mix | s15->s20 | 62 | +0.7911 | 59/2 | 0.0000 |
| frontiercs_research | lo32nm | s15->s20 | 62 | +0.7185 | 58/2 | 0.0000 |

## 4B — mean score

| bench | 先验血统 | span | n 题 | DiD 均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|---|
| frontiercs | ft01mix | s15->s20 | 172 | +4.7925 | 56/12 | 0.0000 |
| frontiercs | lo32nm | s15->s20 | 172 | +5.3848 | 55/18 | 0.0000 |
| alebench | ft01mix | s15->s20 | 39 | +136.9128 | 23/11 | 0.0015 |
| alebench | lo32nm | s15->s20 | 39 | +125.2872 | 26/8 | 0.0005 |
| frontiercs_research | ft01mix | s15->s20 | 62 | +9.8510 | 28/9 | 0.0002 |
| frontiercs_research | lo32nm | s15->s20 | 62 | +7.6478 | 28/11 | 0.0091 |

## 9B — P(complete)

| bench | 先验血统 | span | n 题 | DiD 均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|---|
| frontiercs | ft01mix | s10->s15 | 172 | -0.0395 | 60/66 | 0.2049 |
| frontiercs | ft01mix | s10->s20 | 172 | +0.0925 | 82/52 | 0.0025 |
| frontiercs | ft03nm | s10->s15 | 172 | +0.0241 | 70/58 | 0.2927 |
| frontiercs | ft03nm | s10->s20 | 171 | +0.1787 | 99/38 | 0.0000 |
| frontiercs | lo32nm | s10->s15 | 172 | +0.1585 | 98/37 | 0.0000 |
| frontiercs | lo32nm | s10->s20 | 172 | +0.0402 | 75/53 | 0.0869 |
| alebench | ft01mix | s10->s15 | 40 | -0.1000 | 12/21 | 0.0660 |
| alebench | ft01mix | s10->s20 | 40 | -0.0125 | 16/16 | 0.8072 |
| alebench | ft03nm | s10->s15 | 40 | +0.0800 | 24/11 | 0.1007 |
| alebench | ft03nm | s10->s20 | 40 | +0.0100 | 17/18 | 0.8118 |
| alebench | lo32nm | s10->s15 | 40 | +0.0675 | 20/12 | 0.2229 |
| alebench | lo32nm | s10->s20 | 40 | -0.0800 | 13/22 | 0.1752 |

## 9B — mean score

| bench | 先验血统 | span | n 题 | DiD 均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|---|
| frontiercs | ft01mix | s10->s15 | 172 | +0.2248 | 44/32 | 0.4606 |
| frontiercs | ft01mix | s10->s20 | 172 | +0.7461 | 47/33 | 0.2390 |
| frontiercs | ft03nm | s10->s15 | 172 | -0.2798 | 32/41 | 0.4334 |
| frontiercs | ft03nm | s10->s20 | 171 | +0.3489 | 44/31 | 0.1939 |
| frontiercs | lo32nm | s10->s15 | 172 | +0.1547 | 44/34 | 0.4269 |
| frontiercs | lo32nm | s10->s20 | 172 | +0.4817 | 44/33 | 0.4022 |
| alebench | ft01mix | s10->s15 | 40 | -22.1350 | 13/16 | 0.9053 |
| alebench | ft01mix | s10->s20 | 40 | +23.4300 | 18/14 | 0.4949 |
| alebench | ft03nm | s10->s15 | 40 | +4.1500 | 19/12 | 0.2990 |
| alebench | ft03nm | s10->s20 | 40 | -23.6900 | 20/14 | 0.7388 |
| alebench | lo32nm | s10->s15 | 40 | +54.4750 | 16/17 | 0.4693 |
| alebench | lo32nm | s10->s20 | 40 | +20.2950 | 19/14 | 0.3214 |

## 9B — 自相似度

| bench | 先验血统 | span | n 题 | DiD 均值 | +/− | Wilcoxon p |
|---|---|---|---|---|---|---|
| frontiercs | ft01mix | s10->s15 | 44 | +0.0170 | 27/17 | 0.1795 |
| frontiercs | ft01mix | s10->s20 | 49 | +0.0029 | 24/25 | 0.9686 |
| frontiercs | ft03nm | s10->s15 | 46 | -0.0038 | 18/28 | 0.4161 |
| frontiercs | ft03nm | s10->s20 | 49 | -0.0009 | 26/23 | 0.9528 |
| frontiercs | lo32nm | s10->s15 | 49 | +0.0114 | 27/22 | 0.5212 |
| frontiercs | lo32nm | s10->s20 | 48 | +0.0155 | 30/18 | 0.0897 |
| alebench | ft01mix | s10->s15 | 11 | -0.0370 | 4/7 | 0.2783 |
| alebench | ft01mix | s10->s20 | 9 | -0.0067 | 4/5 | 0.9102 |
| alebench | ft03nm | s10->s15 | 11 | -0.0789 | 4/7 | 0.0830 |
| alebench | ft03nm | s10->s20 | 11 | -0.0403 | 3/8 | 0.1016 |
| alebench | lo32nm | s10->s15 | 9 | -0.0361 | 4/5 | 0.6523 |
| alebench | lo32nm | s10->s20 | 9 | -0.0053 | 4/5 | 0.8203 |