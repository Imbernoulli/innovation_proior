# v2 语料与 MLS-Bench:分数回归证伪 + 一个真实但被基准掩盖的 agentic 行为回归

> 调查时间 2026-08-26。对象是 v2-multisetting 批次(`innov_v2` / `maintain_w2w3` / `maintain_hard` 全参 SFT + α=0.1 soup,共 9 个 arm)在 MLS-Bench CPU 任务上的表现,对照组是 bl3615 的 y26 批次(`prebase` / `preloraIM` / `presoupNEW10` / `presoupWD03_20`)。
>
> 触发问题:**"原来 SFT 后 MLS 至少不掉点,现在怎么掉了?"**
>
> 所有 op 级统计来自 `messages.jsonl` 结构化日志,不是 `task_logs/*.log`(后者只是 stdout 渲染)。凡标注 *[未独立验证]* 的小节来自 subagent 报告,本人未复核。

---

## 0. 结论摘要

| # | 结论 | 强度 |
|---|---|---|
| 1 | **分数上的"MLS 回归"很可能不存在** —— 是谐波噪声 + 合并规则伪影 | 已验证 |
| 2 | **但存在一个真实的 agentic 行为回归**:模型少看、多测、编辑成功率崩塌 | 已验证 |
| 3 | 元凶之一是 **752 行 agentic slice**,有干净的受控消融支持 | 已验证 |
| 4 | 还有第二个效应(编辑准确率崩塌)**与 op mix 无关且无法分离**,可能是全参 SFT 本身的损伤 | 已验证但混淆 |
| 5 | "语料用了 harness 不认识的 `rewrite` op"这一假设 **被证伪** | 已验证(干净否证) |
| 6 | MLS-Bench **对"不动手"给分不低**,因此系统性地看不见 (2) | 已验证 |
| 7 | α=0.1 soup 会把 (2)(3)(4) 全部修复回 base 水平 | 已验证 |

**对决策的影响:MLS 在当前采样量下不能作为选 checkpoint 的依据。**

---

## 1. 分数回归本身站不住

### 1.1 同一批数据,换合并规则就换排名

我们有多轮重跑(后缀 `''` / `_fix` … `_fix5`)。把同一模型的多轮按三种一致规则合并,在 bohan 20 任务集上重算:

| 规则 | 排名 |
|---|---|
| `max` | hardmaint .1501 > **noag .1377** > base9b .1312 > wd03 .1309 > noag_soup .1284 > wd01_soup .1232 > wd01 .1190 > wd03_soup .1029 > hardmaint_soup .0721 |
| `last` | hardmaint .1501 > base9b .1312 > wd03 .1309 > noag_soup .1284 > wd01_soup .1232 > wd01 .1190 > **noag .1148** > wd03_soup .1029 > hardmaint_soup .0591 |
| `mean` | hardmaint .1501 > noag_soup .1284 > noag .1263 > wd03 .1245 > wd01_soup .1232 > **base9b .1195** > wd01 .1190 > wd03_soup .1020 > hardmaint_soup .0656 |

`noag` 在 **.1148 ↔ .1377** 之间摆动,只取决于取哪一次重跑。`mean` 口径下 **base9b 掉到第 6,有五个 arm 反超它**。

先前流传的那张表(以及"只有 hardmaint 超过 base"的说法)是 `max`/`last` 混合的产物:7 个 arm 取 `max`,而 `hardmaint_soup`(.0721→.0591)和 `noag`(.1377→.1148)这两个重跑变差的 arm 取了 `last`。base9b 取 `max`,而它恰好是少数跑了两轮的模型 —— 等于让 base 白拿 best-of-2,单轮的 arm 没有这个待遇。

**而"为什么会有这么多轮重跑"本身是个 bug,见 §6.1。** 每个 arm 的 20 项均值里只有 12–13 项来自首轮,其余 7–8 项首轮直接是 `missing=0`。也就是说上表三种规则合并的**全都是 best-of-≤4 次尝试**;换成只用首轮,base9b 从 .1312 掉到 **.0546**,`wd01_withag_soup` 从 .1232 掉到 **.0418**。修好 `MLSBENCH_PY` 之后单轮即可拿满 20/20 的 `pure_noag_soup` 是 **.0647** —— 唯一一个不含 best-of-N 加成的数字,而它在单轮口径下**高于 base**。

### 1.2 底层原因:harness 噪声大到无法分辨

- **14 组"同模型 / 同任务 / 独立重跑"里,8 组分数不一致。** 例:`base9b / mlsys-moe-load-balance` 0.000 → 0.375;`noag / mlsys-moe-load-balance` 0.375 → 0.000;`base9b / optimization-hyperparameter-search` 0.3027 → 0.3954。
- 9 个 arm 上的**每任务 SD ≈ 0.0908**,推得 21 任务均值的 SD ≈ **0.0250**。我们要比较的 arm 间差异普遍在 0.01–0.03 量级 —— **在噪声以内**。
- 噪声来源可定位:`MLS-Bench-dev/src/mlsbench/agent/models.py:258` 在 thinking 开启时**强制 `temperature = 1`**。

### 1.3 分数吸引子:基准给"不动手"发钱

两个批次共 278 个已评分单元中,**101 个(36%)落在 16 个非零"吸引子"值上**,同一任务被 4–11 个不同 arm 共享。例如 `optimization-hyperparameter-search = 0.302739` 出现在 11 个 arm,`ml-anomaly-detection = 0.501590` 出现在 9 个,`ml-selective-deferral = 0.241983` 出现在 7 个。**这些就是"什么都不改"的基线分。**

最尖锐的一例:bl3615 的 `presoupWD03_20` 在 `optimization-evolution-strategy` 上是 `test×3 + submit×2 + view×1`、**零编辑**,得分 **0.4866**,占该 arm 全部 +0.082 增益的 **28%**;而他的 base 在同一任务做了 12 次编辑,得 **0.000**。

**含义:一个把模型变得更消极的改动,在 MLS 上可能不掉分甚至涨分。** 这正是下一节的行为回归被掩盖的机制。

---

## 2. 真实存在的 agentic 行为回归

### 2.1 op 使用分布

数据来源:`<MLSBENCH_ROOT>/logs/<task>/vllm/<tag>__cc-<jobid>-<task>/agent/messages.jsonl`。工具调用 = `role=="assistant"` 且带 `tool_name`;其结果取**下一条**消息的 `result`。限定 22 个 `DEFAULT_CPU_TASKS`。

**我们的 arm**(`MLSBENCH_ROOT=/scratch/gpfs/CHIJ/ziran/innov_v2_multi/mlsroot`):

| arm | eps | calls | edit | view | test | undo | submit |
|---|---|---|---|---|---|---|---|
| base9b | 19 | 295 | 114 (38.6%) | 84 (28.5%) | 51 (17.3%) | 37 (12.5%) | 9 (3.1%) |
| **hardmaint** | 16 | 120 | 32 (26.7%) | **11 (9.2%)** | **60 (50.0%)** | 8 (6.7%) | 9 (7.5%) |
| **wd01_withag** | 29 | 206 | 58 (28.2%) | **26 (12.6%)** | **80 (38.8%)** | 25 (12.1%) | 17 (8.3%) |
| **wd03_withag** | 16 | 124 | 39 (31.5%) | **13 (10.5%)** | **50 (40.3%)** | 8 (6.5%) | 14 (11.3%) |
| noag | 17 | 221 | 98 (44.3%) | 50 (22.6%) | 41 (18.6%) | 16 (7.2%) | 16 (7.2%) |
| hardmaint_soup | 17 | 240 | 109 (45.4%) | 67 (27.9%) | 42 (17.5%) | 12 (5.0%) | 10 (4.2%) |
| wd01_withag_soup | 17 | 276 | 97 (35.1%) | 91 (33.0%) | 47 (17.0%) | 32 (11.6%) | 9 (3.3%) |
| wd03_withag_soup | 17 | 285 | 103 (36.1%) | 97 (34.0%) | 37 (13.0%) | 24 (8.4%) | 24 (8.4%) |
| noag_soup | 18 | 283 | 99 (35.0%) | 97 (34.3%) | 55 (19.4%) | 15 (5.3%) | 17 (6.0%) |

**bl3615 y26**(`MLSBENCH_ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev`):

| arm | eps | calls | edit | view | test | undo | submit |
|---|---|---|---|---|---|---|---|
| prebase_y26nh22 | 21 | 348 | 130 (37.4%) | 116 (33.3%) | 51 (14.7%) | 30 (8.6%) | 21 (6.0%) |
| preloraIM_y26nh22 | 21 | 299 | 114 (38.1%) | 88 (29.4%) | 47 (15.7%) | 37 (12.4%) | 13 (4.3%) |
| presoupNEW10_y26nh22 | 21 | 309 | 121 (39.2%) | 98 (31.7%) | 46 (14.9%) | 25 (8.1%) | 19 (6.1%) |
| presoupWD03_20_y26nh22 | 21 | 336 | 118 (35.1%) | 105 (31.2%) | 54 (16.1%) | 33 (9.8%) | 26 (7.7%) |

### 2.2 test:edit 比 —— 三个 arm 与其余 10 个截然分开

| arm | test | edit | test/edit |
|---|---|---|---|
| base9b | 51 | 114 | 0.45 |
| **hardmaint** | 60 | 32 | **1.88** |
| **wd01_withag** | 80 | 58 | **1.38** |
| **wd03_withag** | 50 | 39 | **1.28** |
| noag | 41 | 98 | 0.42 |
| hardmaint_soup | 42 | 109 | 0.39 |
| wd01_withag_soup | 47 | 97 | 0.48 |
| wd03_withag_soup | 37 | 103 | 0.36 |
| noag_soup | 55 | 99 | 0.56 |
| prebase_y26nh22 | 51 | 130 | 0.39 |
| preloraIM_y26nh22 | 47 | 114 | 0.41 |
| presoupNEW10_y26nh22 | 46 | 121 | 0.38 |
| presoupWD03_20_y26nh22 | 54 | 118 | 0.46 |

两个批次全部 13 个模型都在 **0.36–0.56**,唯独**带 agentic slice 的三个全参 arm 在 1.28–1.88**。

### 2.3 零成功编辑的 episode

"成功" = 该 edit 调用之后的 `result` 前 600 字符**不含** `ERROR`。注意 `WARNING: … no longer parses as Python` 算**成功**(文件写进去了,只是编译不过)。

| arm | eps | 零**尝试**编辑 | 零**成功**编辑 | % |
|---|---|---|---|---|
| base9b | 19 | 1 | 2 | 11% |
| **hardmaint** | 16 | 8 | **11** | **69%** |
| **wd01_withag** | 29 | 10 | **19** | **66%** |
| **wd03_withag** | 16 | 5 | **9** | **56%** |
| noag | 17 | 2 | 6 | 35% |
| hardmaint_soup | 17 | 0 | 1 | 6% |
| wd01_withag_soup | 17 | 0 | 0 | 0% |
| wd03_withag_soup | 17 | 2 | 3 | 18% |
| noag_soup | 18 | 3 | 3 | 17% |
| prebase_y26nh22 | 21 | 0 | 1 | 5% |
| preloraIM_y26nh22 | 21 | 1 | 1 | 5% |
| presoupNEW10_y26nh22 | 21 | 2 | 2 | 10% |
| presoupWD03_20_y26nh22 | 21 | 1 | 3 | 14% |

**带 agentic slice 的三个全参 arm 有 56–69% 的 episode 一次成功编辑都没有**;base、全部 soup、bl3615 全部 arm 都在 0–18%。

### 2.4 编辑失败分解

| arm | 尝试 | 成功 | ERROR | err% | `allow_create is false` | 不可编辑¹ | `old_str not found` | 写入但语法坏 |
|---|---|---|---|---|---|---|---|---|
| base9b | 114 | 75 | 39 | 34% | 8 | 6 | 22 | 10 |
| hardmaint | 32 | 9 | 23 | **72%** | 2 | 2 | 17 | 2 |
| wd01_withag | 58 | 16 | 42 | **72%** | 10 | 12 | 18 | 3 |
| wd03_withag | 39 | 15 | 24 | **62%** | 1 | 8 | 15 | 12 |
| noag | 98 | 31 | 67 | **68%** | 11 | 14 | 42 | 18 |
| hardmaint_soup | 109 | 76 | 33 | 30% | 5 | 7 | 20 | 7 |
| wd01_withag_soup | 97 | 64 | 33 | 34% | 11 | 6 | 14 | 15 |
| wd03_withag_soup | 103 | 59 | 44 | 43% | 15 | 7 | 22 | 13 |
| noag_soup | 99 | 61 | 38 | 38% | 3 | 6 | 24 | 13 |
| prebase_y26nh22 | 130 | 84 | 46 | 35% | 7 | 11 | 26 | 15 |
| preloraIM_y26nh22 | 114 | 67 | 47 | 41% | 11 | 4 | 28 | 14 |
| presoupNEW10_y26nh22 | 121 | 82 | 39 | 32% | 7 | 6 | 25 | 18 |
| presoupWD03_20_y26nh22 | 118 | 69 | 49 | 42% | 4 | 11 | 29 | 17 |

¹ `File not editable` ∪ `not in allowed packages` ∪ `exceed the editable range` 的并集。

**全部四个全参 arm(含 noag)的编辑失败率是 62–72%,base 与 bl3615 全部 arm 是 32–42%,全部 soup 是 30–43%。**

### 2.5 逐条对照的 transcript

**任务 `ml-selective-deferral`**

我们的 `base9b`(18 ops)—— `mlsroot/logs/ml-selective-deferral/vllm/base9b__cc-12961028-ml-selective-deferral/agent/messages.jsonl`
```
step1 edit  [create file_text=0c]        -> ERROR
step2 edit  [str_replace old_str=680c]   -> OK
step3 edit  [str_replace old_str=902c]   -> OK
step4 test
step5 edit  [create file_text=0c]        -> ERROR
step6 view  [scripts/adult.sh]
step7 edit  [str_replace old_str=2257c]  -> OK
step8 view  [scikit-learn/custom_selective.py 295-330]
step9 view  [scikit-learn/custom_selective.py 270-345]
```

我们的 `hardmaint`(**整个 episode 只有 5 步**)—— `mlsroot/logs/ml-selective-deferral/vllm/hardmaint__cc-12961027-ml-selective-deferral/agent/messages.jsonl`
```
step1 view    [scikit-learn/custom_selective.py 253-289]
step2 test
step3 test
step4 test        <- stdout: "Running tests (FINAL (max_tests reached))"
step5 submit  [n=3]
```
零编辑。得分 **0.24198259074853556** —— 正是 §1.3 那个"未修改基线"吸引子,与他的 base / loraIM / soupNEW10 / soupWD03_20 在该任务上完全同值。

bl3615 的 `presoupWD03_20`(13 ops)—— `MLS-Bench-dev/logs/ml-selective-deferral/vllm/presoupWD03_20_y26nh22__cc-12538623-ml-selective-deferral/agent/messages.jsonl`
```
step1 edit  [str_replace old_str=1768c] -> ERROR
step2 edit  [str_replace old_str=43c]   -> ERROR
step3 view  [scikit-learn/custom_selective.py 253-290]
step4 view  [scikit-learn/custom_selective.py 253-288]
step5 edit  [str_replace old_str=314c]  -> ERROR
step6 test
step7 test
step8 edit  [str_replace old_str=314c]  -> ERROR
step9 edit  [str_replace old_str=1768c] -> ERROR
```

**任务 `causal-observational-nonlinear`**

`hardmaint` 的整个 episode 是 2 步:`test` → `undo`。
`wd01_withag` 是 5 步:`test, test, view, test, submit[n=3]`,零编辑。
bl3615 `presoupWD03_20` 是 20 步的 `test → edit → view → edit → edit → view → test …`。

**可读的对比:base 和 bl3615 的 arm 走 `view → str_replace → test` 的循环;带 agentic slice 的全参 arm 以 `test` 开场、极少 `view`、以 `submit` 或悬空的 `undo` 收场。**

### 2.6 `allow_create is false` 原文

`mlsroot/logs/causal-observational-linear-non-gaussian/vllm/wd03_withag_soup__cc-12961024-.../agent/messages.jsonl`(step 2)
```
CALL  : op='create' filename='causal-learn/bench/custom_algorithm.py' file_text=2105 chars
RESULT: ERROR: allow_create is false; cannot create new files

[Current file: causal-learn/bench/custom_algorithm.py | editable: 3–14 | total: 14 lines]
```
注意形状:**2.1 KB 的整文件正文,打向一个可编辑区间只有第 3–14 行(共 14 行)的文件。**

`mlsroot/logs/causal-observational-nonlinear/vllm/noag__cc-12961025-.../agent/messages.jsonl` 在 step 6 和 step 8 **重复同一个被拒绝的模式**(2636 / 2552 chars),说明模型不从错误里学。

其它原文错误串:
- `ERROR: File not editable: eplb/gpt4.py` —— wd01_withag,8 次
- `ERROR: Package 'seed_config.py' is not in allowed packages` —— wd03_withag,3 次
- `ERROR: old_str not found in eplb/custom_eplb.py. Matching is exact-first but tolerates lin…` —— hardmaint,9 次
- `ERROR: the match spans lines 1..13, which exceed the editable range.` —— presoupWD03_20,4 次

---

## 3. 受控消融:752 行 agentic slice 是其中一个元凶

`wd01_withag` 与 `noag` **只差一件事**:前者用 `innov_v2`(2,901 行,含 752 行 agentic),后者用 `innov_v2_noag`(2,149 行,同一批减去那 752 行)。maintenance 集(`maintain_w2w3`)、weight decay 0.1、base、1 epoch 全部相同。**这是一个干净的受控消融。**

限定**每 arm 单一 job generation**(12961021–12961028;`wd01_withag` 取 12959979)以排除多轮池化伪影:

| arm | job | eps | edit | view | test | test/edit | view% | 零成功 | create | err% |
|---|---|---|---|---|---|---|---|---|---|---|
| base9b | 12961028 | 14 | 90 | 60 | 39 | 0.43 | 27.0% | 1/14 | 6 | 34% |
| **wd01_withag**(含 agentic) | 12959979 | 14 | 37 | 17 | 65 | **1.76** | **11.3%** | **9/14** | 9 | **86%** |
| **noag**(无 agentic) | 12961025 | 14 | 89 | 44 | 32 | **0.36** | **23.4%** | 6/14 | 11 | 70% |
| wd03_withag | 12961023 | 13 | 37 | 11 | 38 | 1.03 | 10.5% | 7/13 | 5 | 62% |
| hardmaint | 12961027 | 13 | 27 | 9 | 46 | 1.70 | 9.5% | 9/13 | 4 | 74% |
| hardmaint_soup | 12961021 | 14 | 76 | 58 | 30 | 0.39 | 31.4% | 1/14 | 7 | 34% |
| wd01_withag_soup | 12961022 | 13 | 81 | 81 | 33 | 0.41 | 36.7% | 0/13 | 11 | 37% |
| wd03_withag_soup | 12961024 | 14 | 85 | 83 | 30 | 0.35 | 35.6% | 3/14 | 18 | 44% |
| noag_soup | 12961026 | 14 | 85 | 73 | 35 | 0.41 | 33.0% | 2/14 | 1 | 35% |

**三个可分离的效应:**

**(A) 752 行 agentic slice 造成 op-mix 畸变。** `wd01_withag` vs `noag`:test/edit **1.76 vs 0.36**,view **11.3% vs 23.4%**。**把那 752 行去掉,op 选择就回到 base 水平。**

**(B) v2 配方里还有别的东西让编辑准确率崩塌,且不动 op mix。** `noag` vs `base9b`:err **70% vs 34%**,零成功 **6/14 vs 1/14**,而 test/edit 和 view% 都贴着 base。
**这一条是混淆的** —— `noag` 是全参 SFT 而 `base9b` 没训过,所以 (B) 可能是**全参微调本身的损伤**而非这批语料。本批次里没有能分开二者的对照。

**(C) α=0.1 soup 把 (A)(B) 全部修好。** 每个 arm、每项指标皆然。

---

## 4. 被证伪的假设:`rewrite` op 不兼容

一个看起来完美的元凶假设:v2 agentic slice 里 **2,610 次 edit 调用中有 1,385 次(53%)用 `op: "rewrite"`**,而 della 上的 harness 根本没有这个 op —— `grep -c rewrite` 于 `MLS-Bench-dev/src/mlsbench/agent/tools.py` = **0**,`tools.py:1540` 会回 `ERROR: Unknown op 'rewrite'. Use 'create' or 'str_replace'.`

**实测证伪。** 扫描规则:对两个 root 下 22 个 CPU 任务的每个 `messages.jsonl`,凡 `role=="assistant" and tool_name=="edit"`,统计 `str(tool_input["op"]) == "rewrite"`;并独立统计下一条 `result` 是否含 `Unknown op`。

扫描范围:`/scratch/gpfs/CHIJ/ziran/innov_v2_multi/mlsroot/logs/<task>/vllm/*/agent/messages.jsonl`(9 tag,170 episode,749 次 edit)与 `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev/logs/<task>/vllm/*/agent/messages.jsonl`(4 tag,84 episode,483 次 edit)。

**结果:13 个模型全部 `op=="rewrite"` → 0 次;`Unknown op` → 0 次。**

edit 子 op 原始计数:

| arm | str_replace | create | rewrite |
|---|---|---|---|
| base9b | 105 | 9 | **0** |
| hardmaint | 28 | 4 | **0** |
| wd01_withag | 47 | 11 | **0** |
| wd03_withag | 34 | 5 | **0** |
| noag | 85 | 13 | **0** |
| hardmaint_soup | 101 | 8 | **0** |
| wd01_withag_soup | 85 | 12 | **0** |
| wd03_withag_soup | 84 | 19 | **0** |
| noag_soup | 94 | 5 | **0** |
| prebase_y26nh22 | 119 | 11 | **0** |
| preloraIM_y26nh22 | 103 | 11 | **0** |
| presoupNEW10_y26nh22 | 114 | 7 | **0** |
| presoupWD03_20_y26nh22 | 114 | 4 | **0** |

**语料/harness 的 op 不匹配在训练数据里是真的,在评测时的表现是零。** 模型没有把这个 op 迁移出来。这条假设死了。

### 4.1 附带否证:`create` 不是"整文件重写"

原以为 SFT arm 抬高的 `create` 率反映整文件重写,base 的 create 则是退化空串。**聚合数据不支持:**

| arm | n_create | 空(0 字符) | 中位字符 | 最大 |
|---|---|---|---|---|
| base9b | 9 | 1 | 5,642 | 16,705 |
| **wd01_withag** | 11 | 1 | **72** | 22,579 |
| **wd03_withag** | 5 | 0 | **10** | 8,414 |
| noag | 13 | 1 | 4,560 | 27,781 |
| **prebase_y26nh22** | 11 | **8** | **0** | 48,254 |
| preloraIM_y26nh22 | 11 | 1 | 6,123 | 26,180 |
| presoupNEW10_y26nh22 | 7 | 0 | 7,340 | 13,895 |
| presoupWD03_20_y26nh22 | 4 | 1 | 1,538 | 36,114 |

发空 create 的恰恰是**他的 base**(8/11),而 `wd01_withag`/`wd03_withag` 的 create 中位数只有 72 和 10 字符 —— 近乎空,不是整文件。§2.6 里那几个 2.1–2.6 KB 的真整文件 create 来自 `wd03_withag_soup` 和 `noag`。

**结论:create 计数抬高是实的,"整文件重写"这个机制解释部分成立部分被推翻,不应作为定论陈述。**

### 4.2 附带否证:`undo` 死循环不是 SFT 效应

含 ≥3 次连续 `undo` 的 run 数 / 最长连击:base9b **4/19,最长 8**;wd01_withag 1/29(最长 10);hardmaint 1/16;wd03_withag 0/16;noag 0/17;bl3615 各 arm 0–2/21,最长 ≤4。**base 比 SFT arm 更爱死循环。** 不是训练效应。

---

## 5. 语料层面的差异 *[未独立验证 —— 来自 subagent]*

注册表:`/scratch/gpfs/CHIJ/ziran/innov_v2_multi/data/dataset_info.json`(`formatting: sharegpt`)。

| dataset | 文件 | 行数 |
|---|---|---:|
| `innov_v2` | `data/innov_v2.jsonl` | 2,901 |
| `innov_v2_noag` | `data/innov_v2_noag.jsonl` | 2,149 |
| `maintain_w2w3` | `data/maintain_w2w3.jsonl` | 5,814 |
| `maintain_hard` | `data/maintain_hard.jsonl` | 4,917 |

bl3615(`dataset_dir: /scratch/gpfs/CHIJ/bohan/fs/LF-innov/data`):

| dataset | 行数 |
|---|---:|
| `innovation_ctl_new` | 2,590 |
| `innovation_wave2_only` | 638 |
| `innovation_wave3_full` | 2,097 |

配方对照:

| 模型 | `dataset:` | 总行 |
|---|---|---|
| `full_wd01_withag` | `innov_v2,maintain_w2w3` | 8,715 |
| `full_wd01_noag` | `innov_v2_noag,maintain_w2w3` | 7,963 |
| `full_wd01_withag_hardmaint` | `innov_v2,maintain_hard` | 7,818 |
| `full_wd01_pure_noag` | `innov_v2_noag` | 2,149 |
| bl3615 `loraIM` | `innovation_ctl_new,innovation_wave2_only,innovation_wave3_full` | 5,325 |
| bl3615 `soupNEW10` 父模型 | `innovation_ctl_new` | 2,590 |
| bl3615 `soupWD03_20` 父模型 | `innovation_ctl_new` | 2,590 |

**关键差异:**

1. **他 RL 后最强的 soup,父模型只训了 2,590 行 innovation,零 maintenance 行。** 我们的 arm 带着 5,814(`maintain_w2w3`)或 4,917(`maintain_hard`)行 maintenance,占语料约 **67%**。
2. **agentic 占比**:他的 soup 父模型 631/2,590 = **24.4%**,loraIM 631/5,325 = **11.8%**;我们是 752/8,715 = **8.6%**。
3. **v2 agentic slice 的 op 构成**:2,610 次 edit = **1,385 rewrite + 827 str_replace + 398 create**;另有 2,091 次 test、164 次 submit。**`view` 和 `undo` 在 tool schema 里声明了,但被示范 0 次。** test:edit = 2091/2610 = **0.80**。
4. **旧语料**:str_replace 4,296 + run_experiment 1,421 → test:edit = **0.33**,`create` 从未使用。

`view` 零示范 + test:edit 0.80 这两条语料属性,与 §2 实测的 view 率坍塌、test/edit 抬高**方向一致且量级吻合**。

### 5.1 待确认:`maintain_w2w3` 行数对不上 *[未独立验证]*

- 实际 **5,814** 行,而 `experiments/agentic_ablation_4b/build_arms.py` 的配方应产出 **6,041** 行(wave2 750 + wave3 全部 5,291)。
- 实际构成 wave2 750 + wave3 5,064 → **缺 227 行 wave3**。
- 这 227 行共同特征:system prompt 为 "expert competitive programmer",长度 22 k–223 k 字符。符合长度截断,但**施加该截断的脚本在 della 上没找到**。
- **在把 `maintain_w2w3` 当作可复现数据集之前需要确认。**
- `maintain_hard`(4,917)是 `maintain_w2w3` 去掉 897 行的严格子集,被去掉的行 `pass_rate ∈ {1.0(678), 0.75(219)}`、`samples_used=4`、中位 9.7 k 字符 —— 即"hard" = 砍掉又易又短的尾巴。

---

## 6. 基础设施类 bug:评测器往只读树里写

这是**同一类问题的第三次**,而且前两次是**静默清零**而非报错,比崩溃危险得多:

| # | 位置 | 后果 | 状态 |
|---|---|---|---|
| 1 | research evaluator 写 `research_overlay/julia_env/lock.pid` | **symbolic_regression 全部样本清零**(跨模型均匀,故相对比较仍有效) | 已修:自有副本 `envs/research_julia/{julia_depot,julia_env}` + `JULIA_DEPOT_PATH` / `PYTHON_JULIAPKG_PROJECT` 覆盖 |
| 2 | grammar_fuzzing / fuzzer / sql 写 `output_ans` 进 bl3615 的题目目录 | 320 个 research 样本中 4 个失败(跨模型均匀) | 未修(需 628 M 影子树),已记录 |
| 3 | MLS-Bench `RunLogger.mkdir` 写 `MLS-Bench-dev/logs/` | `PermissionError: [Errno 13]`,22 个任务各在 1.3 s 内 `agent_failed` | 已修:必须传 `MLSBENCH_ROOT=$D/mlsroot`(自有可写副本) |

**规则:任何评测器只要会往 `/scratch/gpfs/CHIJ/bohan/` 下写东西,就必须先指向我们自己的可写副本。**

### 6.1 第四次:`MLSBENCH_PY` 回落到缺依赖的 python —— 并因此污染了整个 MLS 列

> **本节于 2026-08-26 重写。** 初版写的是"另一个环境缺陷:`sklearn` 缺失",并称"每个 arm 只有 14/22 被评分,而他的每个 arm 都是 21/22"。**那句话是错的**,它拿我们的**单轮**数字去比他的**合并**总数;实测我们补跑合并后同样是 21/22。真实后果比覆盖率严重得多,见下。

**症状.** 我们 9 个 arm 的 **round 0 在 6 个任务上以完全相同的方式失败**:host 侧 `load_mid_edit_ops`(`mlsroot/src/mlsbench/agent/tools.py:5124`)抛 `ModuleNotFoundError: No module named 'sklearn'`。traceback 见 `outputs/cc_mlsbench_cpu_base9b/task_logs/ml-clustering-algorithm.log`。

**根因(不是"环境缺陷",是又一个继承来的默认值).** `slurm/cc_eval_mlsbench_cpu_ailab.sh:232`:

```bash
MLSBENCH_PY="${MLSBENCH_PY:-/home/bl3615/miniconda3/bin/python}"
[ -x "$MLSBENCH_PY" ] || MLSBENCH_PY="$(command -v python3)"     # 第 233 行
```

我们访问不到 bl3615 的 home,于是第 233 行**静默回落到 base conda**,而 base conda 没有 `deap` / `sklearn` / `pydot` / `pgmpy` / `causallearn` —— 只有 `$D/envs/client` 有。传 `MLSBENCH_PY=$D/envs/client/bin/python` 即可根治。

**真实代价:不是最终覆盖率,是它把 MLS 变成了一个 best-of-N 量.** 补跑 fix/fix2/fix3/fix5 之后每个 arm 都能凑到 21/22,和他持平 —— 但凑齐的方式是**跨轮取 max**。实测每个 arm 的"20 项均值"里只有 **12–13 项来自首轮**,其余 7–8 项首轮是 `missing=0`、由补跑填入:

| model | 20 项中来自 round 0 | 合并后(表中所用) | **只用 round 0** |
|---|---|---|---|
| base9b | 13 | 0.1312 | **0.0546** |
| hardmaint | 13 | 0.1501 | 0.1048 |
| noag | 13 | 0.1377(max)/0.1148(last) | 0.0742 |
| hardmaint_soup | 13 | 0.0721(max)/0.0591(last) | 0.0695 |
| noag_soup | 12 | 0.1284 | 0.0705 |
| wd01_withag | 13 | 0.1190 | 0.0806 |
| wd01_withag_soup | 12 | 0.1232 | **0.0418** |
| wd03_withag | 13 | 0.1309 | 0.0739 |
| wd03_withag_soup | 13 | 0.1029 | 0.0573 |
| **pure_noag_soup**(修好 `MLSBENCH_PY` 后) | **20** | **0.0647** | **0.0647** |

**因此结论表里的 MLS 一列不是同口径比较:** 其余 arm 是 **best-of-≤4 次尝试**,`pure_noag_soup` 是**单次尝试**。按单轮口径看,base9b 只有 0.0546,反而低于 `pure_noag_soup` 的 0.0647 —— 排名整个翻过来。这正是 §9 里 "max vs last 换掉冠军" 那个合并规则伪影的来源,也是把 MLS 从 checkpoint 判据里剔除的第三条独立理由。

**验证.** 只传 `MLSBENCH_PY=$D/envs/client/bin/python`,**单轮**即达 **22/22 完成、21/22 评分**(唯一未评分的是 §6.2 里两批次都死的 `llm-scaling-law-discovery`),而此前每个 arm 单轮只有 13–14/22。job 12998231,`outputs/cc_mlsbench_cpu_pure_noag_soup/summary.json`。

**一般性教训.** 这个脚本里**每一个 bl3615 路径默认值都以不同方式咬了我们一口**:

| 变量 | 默认值 | 症状 | 是否报错 |
|---|---|---|---|
| `VLLM_VENV` | `$PROJECT_ROOT/.venv-vllm023`(软链到无 vllm CLI 的 env) | `exec: vllm: not found` | **是**(秒退) |
| `MLSBENCH_ROOT` | `/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev` | `PermissionError: [Errno 13]`,22 个任务 1.3 s 内全灭 | **否** —— 表现为整片 `agent_failed` |
| `MLSBENCH_PY` | `/home/bl3615/miniconda3/bin/python` | `ModuleNotFoundError`,6 个任务静默丢分 | **否** —— 表现为整片 `agent_failed` |

**三个里有两个是静默失败**,伪装成"模型不行"而不是"环境不对"。这与本轮 RL 挂掉是同一类问题:`cc_rl_multisource.sh:256-259` 按 8 卡写死了 offload=False / `GPU_MEMORY_UTILIZATION=0.65`,而我们跑 4 卡 —— 继承下来的默认值是按**另一套配置**调的。

**规则(与 §6 开头那条并列):任何继承脚本里形如 `${VAR:-/home/bl3615/…}` 或 `${VAR:-…/CHIJ/bohan/…}` 的默认值,以及任何写死硬件规模的默认值,都必须在提交前逐个显式覆盖并核对,不能依赖它自己报错。**

### 6.2 `llm-scaling-law-discovery` 在两个批次都是死的

13 个模型全部 0.0(他的是 `agent_failed`,我们的是每轮一个 `openai` 异常)。

---

## 7. soup α 不一致 —— 我们从未复现他的赢家配置

`cc_model_soup_merge.py:4`:`merged = alpha*sft + (1-alpha)*base`,alpha 是 SFT 权重。

我们**全部 soup 都是 α=0.10**(`models/full_*_soup0p1`,见 `logs/soup-full_wd01_withag-12951050.out`:`alpha=0.1: mixed 760 float tensors`)。

bl3615 的 `soup_q35_innnew_ft_a10`(α=0.10)RL 后只涨 **+1%**;他真正的赢家 `soup_q35_innnew_wd03_ft_a20` 是 **α=0.20**。

**我们从未建过 α=0.20 的模型。** 加上 §5 的配方差异(零 maintenance vs 67% maintenance),我们的 soup 和他的赢家在**融合权重和数据配方两处都不同**。

---

## 8. 与 y26 的纵向对照:RL 前掉点是常态

bl3615 同口径数据(FCS n=860 / ALE-40 n=200 / Research n=320 / MLS):

**RL 前**

| 模型 | FCS | ALE-40 | Research | MLS |
|---|---|---|---|---|
| base | 4.241 | 249.33 | 10.953 | .0649 |
| loraIM | 4.304 | 193.78 | 8.024 | .1199 |
| soupNEW10 | 4.383 | 216.07 | 11.417 | .0656 |
| soupWD03_20 | 4.375 | 196.11 | 8.933 | .1470 |

**RL 后(step 20)**

| 模型 | FCS | ALE-40 | Research | MLS |
|---|---|---|---|---|
| base | 8.039 | 263.30 | 19.871 | .0951 |
| loraIM | 8.438 | 247.33 | 17.791 | .1021 |
| **soupNEW10** | 7.544 | **350.59** | **21.848** | **.1534** |
| soupWD03_20 | 7.231 | 242.73 | 18.669 | .1584 |

**他的 SFT arm 在 RL 前同样相对 base 掉点**(loraIM ALE −22% / Research −27%,soupWD03_20 −21% / −18%)。**收益只在 RL 之后出现**:soupNEW10 的 ALE-40 从 −13% 翻成 **+33%**,Research +10%,MLS +61%;而 **FCS 是 SFT 系 RL 后唯一输给 base 的一项**。

**推论:RL 前的掉点是这条路线的预期行为,不构成"这批 SFT 失败"的证据。验收点在 RL 之后。**

> 绝对值**不可跨流水线比较**:他的 base 是 FCS 4.241 / ALE-40 249.33,我们的是 6.461 / 201.03。只能比各自相对自己 base 的变化。

---

## 9. 建议的确认实验

按信息量排序:

1. **把 MLS 的噪声压下去。** `base9b` + `hardmaint` + `pure_noag` 各跑 **5 个副本**,全 21 任务,**在同一个 job generation 里**。这同时解决三件事:给出带误差棒的分数、验证 §3 的机制、并把 `pure_noag`(零 agentic 行、零 maintenance)这个缺失的第二对照补上 —— 若机制成立,它的 create / view / test:edit 剖面应当落在 base 水平。
2. **修 `sklearn` 缺失**,把覆盖从 14/22 提到 21/22。
3. **分离 (B)。** 需要一个"全参 SFT 但用旧语料"的 arm,才能把"全参微调损伤"与"v2 语料损伤"分开。
4. **补 α=0.20 soup**,这是唯一在 bl3615 那里 RL 后真正起飞的配置。
5. **确认 `maintain_w2w3` 的 227 行缺口**,否则该数据集不可复现。

---

## 10. 复现

op 级统计的解析脚本(只读,不提交):`scratchpad/evidence.py`(§2、§4 权威版)、`ablate.py`(§3)、`an4.py`(合并规则网格)、`an5.py`(噪声/副本)。

已作废的中间脚本:`edits.py` / `edits2.py` —— 曾用宽松正则 `error|not found|no match|failed|unchanged|must appear|Invalid` 匹配 `result[:400]`,会误抓任务正文里的 "error" 一词,导致失败率被高估(43.9/78.1/86.7/92.3%)。**本报告 §2.4 采用严格规则 `'ERROR' in result[:600]`,方向不变、量级下修。**

---

## 11. 本报告的可信度分级

| 小节 | 来源 | 强度 |
|---|---|---|
| §1 分数不可分辨、合并规则伪影 | 本人从 `summary.json` 独立复算 | **强** |
| §2 op 使用、test:edit、零编辑、失败分解、transcript | 从 `messages.jsonl` 直接测量 | **强** |
| §3 受控消融 (A) | 直接测量,单 generation | **强** |
| §3 效应 (B) | 直接测量,但**与全参 SFT 混淆** | 中 |
| §4 rewrite op 证伪 | 直接测量,规则可复现 | **强(干净否证)** |
| §4.1 / §4.2 附带否证 | 直接测量 | **强** |
| §5 语料行数与 op 构成 | **subagent 报告,未独立验证** | 待验 |
| §5.1 `maintain_w2w3` 缺口 | **subagent 报告,未独立验证** | 待验 |
| §6 基础设施 bug | 本人从报错日志直接确认 | **强** |
| §7 soup α | 从 merge 脚本与日志确认 | **强** |
| §8 y26 纵向对照 | 本人从 `shard_*/samples.jsonl` 独立复算 | **强** |

因果链的整体定性:**§3 的 (A) 有受控消融支撑,可以下结论;(B) 混淆未解;§5 的语料属性与 §2 的行为方向一致且量级吻合,但那是相关性论证,不是实验证据。**
