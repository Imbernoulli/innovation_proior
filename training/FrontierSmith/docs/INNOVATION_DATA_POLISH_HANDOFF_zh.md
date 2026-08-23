# Innovation 数据 polish 交接（给数据同事）

2026-08-17。背景：rlv12 四臂 RL 后，两个 soup 臂在 FCS 上落后 base；两路独立审计
（general subagent + Codex，互不知晓）交叉核对了失败机制与语料属性。本文档只含
**经双方审计确认**的事实与处方。完整审计记录见会话存档与
`memory/fcs-deficit-audit-final.md`。

## 诊断（一句话）

语料不缺"长思考"，缺**"遇到困难后果断交付"的示范**。两臂病因相反：
soupWD03_20 不会停（终止时代码全场最好 17.2 vs base 13.9，但 57% 样本逐字复读
撞 32K 上限）；soupNEW10 会停但 C++ 真变差（8.9，p=0.0004）。

## 语料现状（ctl_new = gated_v2，md5 已确认同一文件）

- 2590 条 / 3540 个 assistant turn（**多轮**，统计时别只看首轮）
- think 长度 per-turn：p50≈1987 词 / p90≈4488 / max≈8576（token: p50 3116 / p99 11305）
- 自救词汇几乎为零：`Wait` 6% of turns（base 评测时 ~100%）、`Alternatively` 0.09%
- 样例验证：~5%（gated 清理把它从 15.6% 砍到 5.0%——**误伤**，见处方①）
- commit-in-tail（末段明确交付事件）：~2%（base 56%）；竞赛题子集（338 条
  expert-CP prompt）为 **0.00%**
- system prompt：86.95% 带完整交付条款 + 方法真实历史年份（1690-2026），保留

## 三个不要动（都被实测证伪）

1. **别动长度**：base 得分随 think 长度单调降（2-4k 词段 33.1 分、≥16k 段 1.2 分），
   语料长度段恰是高分段；盲目截短同样有害（capped16k: base 9.16→4.36）。
   两个 soup 的 think 长度差 11 倍（1840 vs 21751 词）来自 **weight decay**，不是数据。
2. **别注入回溯词**（Wait/Alternatively）：回溯密度与分数**四臂全部负相关**
   （最强 r=−0.16 p=3e-6）。唯一正预测的模式是 **fallback-and-ship**
   （base +0.19 p=1e-8）。
3. **别删创新内容**：base 得 0 的 102 道难题上创新臂显著为正（+0.77 t=2.6）；
   亏损全部在 base 能解的 22 题上。过滤执行纪律，不动创新。

## 处方（按杠杆排序，全部是对现有真实 rollout 的操作）

### ① 恢复 gated 清理误删的验证语言（最高杠杆，无争议）
gated_v2 去泄漏时把 think 砍短 27%（1352 个 turn，均值 −1093 词），验证语言删掉
2/3。验证事件是实测质量杠杆：base ≥3 次验证时 15.5 分 vs 零验证 4.9 分。
做法：ctl_old（或 r3）与 gated_v2 同记录 diff → 恢复被删 span 中
**含验证/推演且不含代码块**的部分（human 侧保留 gated 的去泄漏版）→
恢复后**重跑 verbatim-code gate** 防止泄漏回流。
基线文件都在 `LF-innov/data/`：`innovation_ctl_old.jsonl`、`innovation_sft_r3*.jsonl`。

### ② 从 wave2/wave3 现有 rollout 筛"长且会交卷"切片混入
wave3（7/20 与 8/1，`innovation_wave3_full/hard`、`innovation_wave2_only`）就是
已做过的长语料，风格与 base 一致，但**从未进入 innnew 配方**。
⚠️ 不能整体混——回溯密度 8.7/千词正是负相关特征。要**筛**：
同时满足（a）以果断代码块收尾（b）14-gram think 重复率 ≤0.15
（c）回溯密度处于池内下四分位。预计从 wave3_full 2097 条里出 200-400 条。
这填上"长思考且成功终止"档位（现语料该档≈0）。

### ③ 竞赛题子集（338 条 expert-CP）代码语域清洗
删两类格式噪声（不动推理内容）：
- 最终代码内的心声注释：`// TODO / for now / this is getting complicated /
  I'll just assume / placeholder`（soupNEW10 33% 程序带这些 vs base 8%）
- 代码块之后 >50 词的说明文（教程语域；违反 "ONLY the C++ code" 契约）
另：该子集 commit-in-tail 为 0.00%——若早期版本结尾被格式化流程裁过，从老版本恢复。

### ④ maintain 数据：内容不动，代码密集部分加倍回放
maintain（903 条）未被审计指控；c2_maint8x 实验证明高剂量回放可追平 base。
soupNEW10 的 C++ 退化属遗忘问题，这是现有数据里的对症药。

## 需要决策的点
commit-and-ship 示范缺口（语料 2% vs base 56%）纯筛选补不满。选项：
(a) 只靠②的 wave3 切片（纯真实数据，量几百条）；
(b) ①恢复时允许极少量模板桥接句（覆盖全语料但破坏纯真实性）。
用户立场倾向 (a)。

## 配套（训练侧，非数据侧）
- weight decay 别用 0 或 0.3 极端（各出一种病），建议 0.1-0.2 各跑一档
- 训练/评测 prompt 已对齐（y26 协议：评测 `EVAL_RESEARCHER_YEAR=2026` +
  完整模板；MLS 经 `MLSBENCH_SYS_PREFIX`）
- 下轮 RL rollout 加 "It is now year 2026." 前缀（用户指示，训练管线侧在做）

## 红线
- Research-64 数据 eval-only，绝不进训练
- `data/sft/delivery_long.jsonl.LEAKY_DO_NOT_TRAIN` 是 train-on-test 污染样本
  （53/172 题=30.8% benchmark），只可作过滤器风格参考，内容禁止入训

## 追加（2026-08-18，用户裁定）：system prompt 约定统一为"只报时间"
- **往前的约定**：system prompt 只写时间（训练=方法真实年份 "It is now year YYYY."；
  RL/评测=当下 "It is now year 2026."）。**去掉**人设句("You are a good researcher.")
  与交付条款("When you write code... ship that.")——它们是 r1 修正轮加入的遗留，
  不属于时间条件化设计。下一版 SFT 数据请把 system 字段改为纯年份句。
- 依据：Qwen3.5 模板无默认 SP、无日期变量（实测渲染验证），时间必须显式给；
  分解实验显示人设句本身就有强行为效应（非中性），不应作为背景混入。
- rlv13 RL 已用纯时间句(train_time2026.parquet)；评测新增 EVAL_SYS_PROMPT_MODE=bare。

## 追加（2026-08-19）：发现语料 bug——6 个新轨迹未注册年份
最近 traj-build 加入的 6 个轨迹目录（convnext-modernization、deit-data-efficient-vit、
lion-program-search、preact-identity-mappings、roberta-pretraining-recipe、
wide-resnet-widening）没有登记进 `trajectories.json`，导致 `build_sft.py` 里
`trajs.get(task)` 落空、32 行 system 渲染成字面 **"It is now year None."**。
请在 trajectories.json 补注册（年份用 arXiv 年约定，methods.json 已有 wide-resnet=2016
可对照）：convnext 2022 / deit 2020 / lion 2023 / preact 2016 / roberta 2019 /
wide-resnet 2016。训练侧已在下游文件里临时修正（FrontierSmith/LF-innov/data/
innovation_final_timeonly.jsonl，同时按 08-18 裁定把 system 统一成纯时间句），
但源头注册表仍需补，否则下次重建还会复现。
