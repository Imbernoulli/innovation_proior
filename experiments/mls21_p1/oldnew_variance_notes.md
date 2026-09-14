# 我们的 9B RL 臂 rlv5_ft01mix_a10_s20:旧跑(09-02,40960)vs 新跑 _r2(09-13,98304)逐格对账

用户质疑:新跑 0.0836 比旧跑 0.1737 低一半,“不正常”。逐格核查结论(全部有日志/leaderboard 证据):

## 两次输入完全相同
- 权重同一路径;vLLM 0.23.0;serve 参数除 max_model_len(40960→98304)和 served name 外逐项相同(hermes tool parser、bf16、gpu_mem 0.9、max_num_seqs 32)。
- agent 源码同一棵树 `$D/mlsroot/src/mlsbench`(08-25 起未改;与 envs/MLS-Bench-dev 逐文件相同)。
- 初始 prompt 逐字节相同(933 行 41199 字符,diff=0);工作区模板 diff=0。
- 请求参数:无 max_tokens、无 seed、tool_choice=required(模型名不含 qwen→不开 enable_thinking)。两次一样。

## max_model_len 没改变模型行为,只改变“失控调用”的代价
tokens.jsonl 逐调用统计(四个 9B 臂新旧共 3020 次调用):
- 中位 completion 都是 ~175–340 token;
- **27–30% 的调用一路生成到上下文上限**(旧:撞 40960,单次 20–34k;新:撞 98304,单次 40–91k),比例新旧一致。
- 所以“抬上限让模型啰嗦”不成立(用户指出:模型看不到上限,只会被截断——正确)。上限只让每题墙钟涨 3–5 倍(1000s→3000–6000s),没有触发 TASK_TIMEOUT。

## 旧跑的高分是它自己真做出来的,不是捡 stale is_final
leaderboard.csv 中 `vllm/rlv5_ft01mix_a10_s20` 在 ml-anomaly-detection 只有 4 行,全部 2026-09-02T22:20–22:28,
与该次 messages.jsonl 里 test#1–3/submit 的输出一致(auroc_cardio 0.9388 / thyroid 0.9703 / satellite 0.6775 / shuttle 0.9963)。
hyperparameter-search、clustering 同理。

## 新跑掉分是同一模型的另一次随机轨迹,写出了不同(更差)的代码
- ml-anomaly-detection:旧跑 step2 写 IForest+COPOD 的 decision_function 集成→AUROC 0.94;新跑 step1 写“两个模型标准化分数取平均”,
  AUROC 0.085/0.059/0.364/0.007 —— 排序几乎完全反向(1−AUROC = 0.915/0.941/0.994),即**符号写反**;模型看到三次 test 都是 0.08 也没修好,undo 后照样 submit。
- ml-clustering-algorithm:新跑 ARI 0.015、silhouette 为负(旧 0.935)。
- optimization-hyperparameter-search:新跑 evals 50/40/40 vs 旧 95/85(预算用了一半),convergence_auc 0.80/0.84 vs 0.93/0.96。
- 同一 19 题:旧均值 0.1737,新均值 0.091;10 题位位相同(两次都提交了未改动模板,seed 42 确定性),6 题跌、3 题涨。

## 判断
不是 pipeline 缺陷、不是模型变差,是**单 seed 单次运行的 agentic benchmark 方差**:已记录的同配置复测对 base 0.136→0.060、SFT 0.162→0.101(YEAR_SWEEP §5.4.5),跑间 σ≈0.31×本臂均值。
其他三个 bench 都是 pass@5,唯独 MLS 是 n=1 —— 这才是 MLS 数字“看起来不对”的根源。

## 补充(01:30 后,八臂全部落地)

### 八臂新旧对账(同题集均值)
| 臂 | 旧 | 新 | ↑/↓/= |
|---|---|---|---|
| base9b_v2c | 0.1141 | 0.0675 | 5/8/8 |
| ft01mix_a10 | 0.1425 | 0.1144 | 4/9/8 |
| rlv5_base_s20 | 0.1672 | 0.1099 | 4/9/6 |
| rlv5_ft01mix_a10_s20 | 0.1737 | 0.0911 | 1/8/10 |
| base4b | 0.0304 | 0.0017 | 1/3/17 |
| 4b_ft01mix_a10 | 0.0072 | 0.0192 | 2/3/16 |
| rlv5_4b_base_s20 | 0.1223 | 0.1365 | 6/8/7 |
| rlv5_4b_ft01mix_a10_s20 | 0.1088 | 0.1222 | 4/5/12 |

**四个 9B 臂全部下降、三个 4B 臂上升** —— 这个模式让“纯随机方差”不再是唯一解释(独立方差下四臂齐降概率 ≈1/16)。

### 已排除的系统性原因(全部有证据)
- 任务输入/模板:六道掉分题的旧新工作区 `diff -rq`(排除 agent 编辑的 custom_*.py)= 0 差异;`tasks/` 代码不读 data_root。
- 旧跑日期:9B 旧跑 09-02/03,4B 旧跑 09-09,都在 st3812 数据根还活着的时候;data_root 一样。
- RoPE:四个模型 config `max_position_embeddings=262144`,98304 不触发任何缩放;vLLM 日志无 rope/yarn 警告。
- 失控调用的处置:at-cap 调用变成“No action”的比例新旧一致(9B 8–30% vs 8–25%,无一致方向);非 at-cap 调用的 No action 率也一致(27–38%)。**不是 40960 截断“救”了旧跑。**
- 超时:新跑 9B 仅 1 格 timeout+scored(ft01mix/causal-discovery-discrete);旧跑也有 1–2 格 timeout/agent_failed。
- 墙钟:新跑每臂总时长 18–24h(旧 4.6–8.3h),每题最长 6444–10801s,全部在 TASK_TIMEOUT 内。

### 仍未定的
“9B 齐降 / 4B 齐升”是配置(98304 及其带来的 3–5× 墙钟、KV 压力)的系统效应,还是 8 次单跑的巧合,**用日志已无法再分辨**。
唯一能定的办法:同一 8 臂在 40960 下再各跑一次(其余保持 `_r2` 配置)。若 40960 下 9B 回到 0.11–0.17 区间,则是配置效应;若仍 0.07–0.11,则旧跑是高抽样、新跑是低抽样,需要 5-seed 均值。

---
## 2026-09-14 03:00 重查(用户:“不是随机性可以解释的,仔细看跑的区别”)——找到三处确定的、非随机的“跑的区别”

### 区别 1:`_r2` 八臂全部丢了系统提示前缀 “It is now year 2026.”
- 证据:tokens.jsonl 第一次调用 prompt_tokens,旧跑与全部 y-跑均为 **7050**,八个 `_r2` 全部为 **7039**,21 题逐题差恒为 11;
  用模型 tokenizer 数 “It is now year 2026.\n\n” 恰为 **11 token**。
- 原因:旧跑用 `scripts/mls21.sh`(传 `MLSBENCH_SYS_PREFIX="It is now year 2026."`),年份扫描用 `year_sweep_submit.sh`(同样传),
  而 `mls21_rerun_submit.sh` **没传**这个变量(r2 的 .out 里 0 处 SYS_PREFIX)。
- 含义:SFT 语料 100% 以该句开头、RL 也用 2026,r2 是在训练分布外测的。**八臂 r2 表不能用。**

### 区别 2:09-11 vendor 重建后 `ml-active-learning` 对所有臂恒为 0(环境坏了,两处)
- (a) 我用 `envs/mlsdata`(pandas 3.0.5)重生成的 `data/badge/oml/data_{6,44,46}.pk` 含 pandas-3 的 Categorical 状态,
  badge.sif 里是 pandas 2.3.3,`pickle.load` 直接 `NotImplementedError: (CategoricalDtype(...)`。post 期 56 次 test 结果里 56 次此错;
  leaderboard 上 post 期 0/56 个模型有有效行,pre 期 4/14(base4b、rlv5_4b_base、rlv5_4b_ft01mix 拿到模板分 0.362)。
  在 badge.sif **里面**重跑同一个 prepare_data.py → 新 pickle 在镜像内 load OK(已验证);旧 pickle FAIL。
- (b) 我按 git HEAD 重 clone 的 `external_packages/badge/query_strategies/strategy.py` 带 16 处 `.cuda()`;
  09-09 之前的所有工作区副本(原 vendor 拷贝)是 CPU 版(0 处 `.cuda()`)。原 vendor 是打过 CPU 补丁的,不是 clean HEAD。
  修好的文件在 `$S/doc/mls21/strategy_fixed.py`(与现 vendor 仅差 16 行 cuda)。
- 21 题里其余 20 题:工作区 pre~y、pre~r2 逐文件 diff = 0(含 holdout 生成的 `_causal_inputs`)、tasks/ 与 bohan 原树 diff = 0、
  baseline 锚点行 diff = 0、权重 mtime 全在旧跑之前、vllm023 环境 09-03 后无改动、SIF 是 HF 05-11 上传的同一批文件、
  各镜像 Python 版本 pre/post 一致、错误/警告签名里没有版本特异的 API 断裂。

### 区别 3:旧 RL 9B 三臂 n=19,不是 21
- 旧 worker python(miniconda)没有 causallearn/deap,两题 agent_failed→None 被踢出分母。按 21 题(None→0):
  rlv5_base 0.1672→**0.1513**,rlv5_ft01mix 0.1737→**0.1572**,rlv5_ft03nm 0.1945→**0.1760**。

### 仍然解释不了的部分(如实)
- 与旧跑配置完全相同的 y-跑(有前缀、40960、CONC 7、7200s、miniconda worker、同权重、同工作区):
  base9b 旧 0.114 vs 12 次 y-跑均值 0.105(4/11 次高于旧);lo32nm 旧 0.153 vs 11 次均值 0.100(**0/11** 高于旧);
  rlv5_lo32nm 旧 0.181 vs 11 次均值 0.128(**0/11** 高于旧)。差集中在“模板分能否保住”的题:anomaly 0.502、clustering 0.388、
  evolution 0.487、discovery 0.303、treatment、hyperparam。这部分我找不到任何配置/环境差别;逐题证据只剩“模型写的代码 post 期更常崩”
  (Traceback 占 test 结果比例 pre 中位 ~10% → post ~18%),但 ml-active-learning 之外没有环境原因。**不下结论。**

### 该怎么得到“没有问题的结果”
1. 修环境(我自己的树,不动 bohan):`data/badge/oml` 换成镜像内生成的那份(旧目录改名保留),`strategy.py` 换成 CPU 版;
   然后跑一次模板测试,应复现 pre 期 leaderboard 上从 6 月到 9-09 一直不变的模板行 letter 0.851/0.7646、spambase 0.91974/0.905369、
   splice 0.780564/0.748824(我在登录节点跑验证被权限拦了,命令见下)。
2. 八臂按**旧协议**重跑:`MLSBENCH_SYS_PREFIX="It is now year 2026."`、MAX_MODEL_LEN 40960、CONCURRENCY 7、TASK_TIMEOUT 7200,
   只保留两处必要修正:worker 用 `$D/envs/client/bin/python`(否则两题死)、data root 指向修好的 mlsvendor。tag 建议 `<arm>_p1`(p=prefix)。
   若要误差棒,每臂 ≥3 次。

## 2026-09-14 04:15 ml-active-learning 第二层故障:pre_edit 读的是 bohan 树里已悬空的 external_packages 链接

- 现象:p1 六臂(base9b/ft01mix/rlv5_base/rlv5_ft01mix/base4b/4b_ft01mix)的 ml-active-learning 仍然 0 分、leaderboard 行全空;
  任务日志里 `NameError: name 'np' is not defined`(strategy.py:59)。workspace 里的 strategy.py 第 1 行是空行(16200 B),
  而 vendor 源文件第 1 行是 `import numpy as np`(16218 B);09-11 之后所有 workspace 副本(r2/y2026/p1)都是这样,09-03 之前的不是。
- 机制:`vendor/pkg_configs/badge/pre_edit.py` 用 `Path(__file__).resolve().parent^4/vendor/external_packages/badge/.../strategy.py`
  读原文件,去掉 `.cuda()`/`Variable()` 后整文件 replace。`mlsroot/vendor/pkg_configs` 原是指向 bohan 树的符号链接,resolve 后落到
  `$B/vendor/external_packages`,而这个链接指向 st3812(09-11 已删)→ `_ORIG.exists()` 为 False → `_CONTENT=""`,`_LINE_COUNT=1`
  → 第 1 行被替换成空字符串,`.cuda()` 也不再被去掉。这解释了 (a) r2/y2026 workspace 里 16 个 `.cuda()` 和空首行;(b) 我 03:03 换成
  CPU 版 strategy.py 之后 p1 仍然失败(vendor 文件对了,但 pre_edit 仍然把首行抹掉)。
- 21 题所涉包里只有 badge 的 pre_edit 依赖 external_packages(其余 4 个:HypSeek/lm-evaluation-harness/flow-maps/dbim,不在 21 题内),
  所以只有 ml-active-learning 受影响。
- 修复(只修坏的):`mlsroot/vendor/pkg_configs` 链接改名为 `pkg_configs.bohan_symlink_20260914`,rsync 一份 bohan 的 pkg_configs 成实目录
  (diff -rq 仅两个本来就悬空的内部链接不同)。验证:导入 pre_edit 后 `_ORIG.exists()=True`、375 行、首行 `import numpy as np`、
  `.cuda()`=0、且 `_CONTENT` 与 09-03 旧 workspace 的 strategy.py 逐字节相同。
- 补跑:六个已跑过该题的臂各投一个单题作业(`JOBSUF=-alfix TASKS_OVERRIDE=ml-active-learning`,TAG 不变 → leaderboard 模型名相同,
  `mlsbench score` 会选到有效行;输出目录 `cc_mls21_<tag>_p1-alfix`)。ailab ids:13866425/27/29/31/33/35(gpu 孪生 +1)。
  六个尚未开跑的臂(rlv5_ft03nm/rlv5_lo32nm/ft03nm/lo32nm/rlv5_4b_base/rlv5_4b_ft01mix)会在建 workspace 时直接用修好的 pre_edit。
- 汇总口径:最终表必须用 `mlsbench score <task> --model vllm/<tag>_p1` 重新算每题分,不能直接用主跑 summary.json(其中该题为 0)。

## 2026-09-14 05:15 修复后 ml-active-learning 的结果口径

- 修复后所有 workspace(12 主臂 + 已开跑的补跑)strategy.py 首行均为 `import numpy as np`、`.cuda()`=0;补跑与主跑该题首调用 prompt_tokens 全部 10770(前缀在)。
- 修复后第一条有效行:rlv5_4b_ft01mix_a10_s20_p1(模板值 0.851/0.7646/0.91974/0.905369)。
- 三个已完成的补跑均无有效行,但都不是环境:ft01mix_a10 20 步只读/改文件、一次 test 都没跑;4b_ft01mix_a10 第 5 步后不再产生动作;
  rlv5_base_s20 三次 test 都是 round 0 训练正常(TRAIN_METRICS 0.453/0.787)后在自己的查询策略里 exit=1。
- 对照旧跑(09-02/03)该题:12 臂里 8 臂也是 0(settings 空),只有 rlv5_ft03nm/base4b/rlv5_4b_base/rlv5_4b_ft01mix 拿到模板值 0.3617。
  所以该题 0 分在新旧两代里都是常态,修复只是让“能跑通”的臂有机会拿分。

## 2026-09-14 06:50 p1 全部落地:最终对账与分解(正文见 MODEL_SCORECARD_9B_zh.md §10 / 4B §11)
**分组均值**:9B 三个 SFT 臂 0.1460 → 0.0797;9B 四个 RL 臂 0.1663 → 0.1469;9B 底座 0.1141 → 0.1224。**RL 臂减去各自 SFT 起点**:旧 +0.037 / +0.015 / +0.033 / +0.028(base / ft01mix / ft03nm / lo32nm 起点),p1 −0.006 / +0.058 / +0.061 / +0.114。四个 RL 臂在两代里都排在四个非 RL 臂之上,但每臂的名次在两代里不同。

**逐题分解(不是笼统归因噪声,下面是逐格读日志的机制)**:
- 差距集中在“旧跑拿到模板分、p1 记 0”的题:lo32nm 11 题失、0 题得;ft03nm 8 失 1 得;ft01mix 7 失 4 得;RL 四臂各 4–6 失、2–4 得。`ml-clustering-algorithm` 四个非 RL 臂全部 0.388→0,四个 RL 臂全部保住 0.388。
- 记 0 的机制(以 lo32nm 为例,逐题读过):模型改出的代码不能解析(SyntaxError,工具会打印 “no longer parses as Python”)、编辑 `old_str` 匹配失败、违规建文件被拒,最后 20 步用满、一次 `test()` 都没调用(lo32nm p1 有 7 题 0 次 test,旧跑 4 题)。旧跑同一臂这些题都跑完 3 次 test 并提交。
- 无有效行的 `agent_failed` 全部是对话超过 40960 上下文被 vLLM 400 拒绝后 agent 崩溃(旧跑 12 臂共 10 题,p1 共 18 题,ft03nm 一臂 5 题最多);`timeout` 是 7200 s 到点。两道旧 RL 跑恒死的题在 p1 各臂都跑通了。
- **排除了系统性差异**:serve 配置相同;agent 对 vLLM 不传 temperature(两代都是 vLLM 默认采样);逐臂统计 SyntaxError / 编辑失败 / nudge / test 次数 / 上下文崩溃 / 完成 token 数,新旧同量级,没有 p1 全局劣化(base9b 的 p1 语法错误更少、分数略升;4B 四臂两升两降)。

**结论**:同臂、同口径、同权重的两次 MLS-21 单次运行,在题目层面不可复现(与 09-12 同臂同年份复跑得到的 ~0.31× 均值的运行间波动一致);单次运行之间的 0.03–0.12 差都在这个尺度内,**不打 ★,不改 §1.1/§5 的任何数字**。两代数字并列保留;要在 MLS 上分辨臂,需要每臂 ≥3 次独立运行。
