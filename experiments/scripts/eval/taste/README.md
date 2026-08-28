# Taste / insight / judge 评测套件（轻量 model-judge 档）

选自 `docs/TASTE_EVAL_SURVEY_zh.md` / `docs/TASTE_EVAL_SHORTLIST_zh.md`。
目的：现有 FCS / ALE / MLS 全是执行落地型，4B 上信号被截断和环境噪声吃掉；
这套补的是**判别与生成品味**本身，成本低、每个 checkpoint 都跑得起。

## 三个评测（覆盖 insight / taste / judge 三种能力）

| 档 | 评测 | 能力 | GT | 指标 | 公开 4B 参照 |
|---|---|---|---|---|---|
| 生成 | **GiantsBench**（[2604.09793](https://arxiv.org/abs/2604.09793), §27） | **insight**：两篇父论文摘要 → 下游论文的核心洞察 | 下游论文真实 insight | LM judge 1–10 相似度 | Qwen3-4B base **4.75**（judge=gemini-3-pro） |
| 判别 | **SciJudgeBench**（[2603.14473](https://arxiv.org/abs/2603.14473), §1） | **taste**：哪篇论文引用更高 | 真实引用数 | **换序一致性准确率**（随机基线 25%） | Qwen3-4B-Instruct **58.1** → SciJudge-4B 77.3 |
| 判别 | **SoundnessBench**（[2605.30329](https://arxiv.org/abs/2605.30329)） | **judge**：执行前判方法学是否成立 | ICLR reviewer soundness 子分 | Macro-F1 / **假阳性率** | 12 个前沿模型平均 FPR **74.0%**，最佳 Macro-F1 = GPT-5.4 **69.7** |

三个 prompt 全部取自评测方自己发布的产物，没有自己改写：

* GiantsBench：`giants2026/GiantsBench-test` 的 `query` 字段逐字使用（它就是论文 Fig.10
  已渲染好的 insight-anticipation prompt）。judge rubric 抄自论文 Fig.12
  （`similarity_judge_prompt.png`），转写在 `judge_giants.py`。
* SciJudgeBench：`OpenMOSS-Team/SciJudgeBench` 的 `messages`。换序档由行内 metadata 重建，
  模板在 1,000 条 main test 上**逐字节验证**与官方 user turn 相同（`benches.py`
  里有 assert），所以 A↔B 交换是忠实的、不是近似复刻。
* SoundnessBench：system + user 模板抄自 `hosytuyen/SoundnessBench`
  `rigorbench/evaluation/prompt.py` 的 `direct_bucket` 模式，experiments 用它自己的
  `_format_experiments_for_eval` 渲染。

## 协议

* **采样**：thinking on（Qwen3.5 模板默认就开），temperature 0.6 / top_p 0.95 / top_k 20，
  seed 固定。temperature 与 GIANTS 论文的 val 设置一致。
* **max_tokens**：soundness 16,384；giants / scijudge 32,768（本仓评测口径）。
  截断率逐臂记录 —— 截断的样本按错处理，但**必须单独报**，否则会把"推理长度"
  混进"品味"里。
* **解析失败不静默丢**：`no_answer` / `unparseable` 计错并单列。
  SciJudgeBench 的一致性计分下，胡乱翻转会低于 25% 随机线（论文里
  Qwen2.5-1.5B 只有 5.3%），这是预期行为不是 bug。
* **换序一致性**：SciJudge 每对正反各判一次，两次都对才算对。同时报
  `picked_A_rate`（位置偏置）与单序准确率。
* **配对 bootstrap**：臂间比较在共享 id 上做配对重采样，报 Δ 与 95% CI。

## 跑法

```bash
# 1) serve（jiaolab，一卡一个 TP=1 引擎）
DAEMON=1 GPUS=4 bash training/FrontierSmith/scripts/jiaolab/serve_local.sh <MODEL> <TAG>

# 2) 三个 benchmark 串行跑一个臂
bash experiments/scripts/eval/taste/run_arm.sh <TAG> <PORT>

# 3) 打分
python experiments/scripts/eval/taste/score.py soundness outputs_taste/run1/soundness_*.jsonl
python experiments/scripts/eval/taste/score.py scijudge  outputs_taste/run1/scijudge_*.jsonl
python experiments/scripts/eval/taste/judge_giants.py --gen outputs_taste/run1/giants_<TAG>.jsonl \
       --out judged_<TAG>.jsonl --judge <judge-model>
python experiments/scripts/eval/taste/score.py giants judged_*.jsonl
```

`run_gen.py` 是**追加式 + 按 id 续跑**的：崩了原样重跑即可，已完成的样本不会重做。
`score.py` 按 id 取最后一条，所以重复行不会重复计数。

## 已知偏离（读结论前必看）

* **GIANTS 的 judge 不是 gemini-3-pro**。OpenRouter 账户没有余额（2026-08-26 实测
  402 Insufficient credits），所以主 judge 换成本地模型。
  → **绝对分不能和论文的 4.75 锚点同表**；我们自己跑的 base 臂就是本套件里的锚点。
* 因此也**做不到跨家双 judge**（shortlist 防守第 2 条）。替代防守：judge 里插入
  **隐藏朴素基线**（把别的样本的 gold insight 当作预测送进去），judge 必须给它明显更低的分，
  否则这一档结论作废。
* GiantsBench 用的是 test 全集的分层子样（默认 400 条，按 domain 分层、seed 固定），
  不是 7,504 全量；**Test-unseen-parents 子集没有单独跑**。

## 判分口径的一个坑（2026-08-27 踩过）

`judge_ref.py` 原本给 `--task giants` 用的是论文 Figure 12 的**改写版** rubric
（"same core idea"→"same core content"、"Omits the central mechanism"→"Omits the central
element"…）。同一批生成上改写版比逐字版**低约 0.4 分**——比大多数臂间差异还大。
现在 `--task giants` 直接转调 `judge_giants.JUDGE_PROMPT`。
**任何跨臂比较都必须确认是同一把尺子判的**，换 rubric 等于换量纲。

同理，**dtype 也要配对照**：bf16 base 与 fp16 base 在 GiantsBench 上差 0.25 分
（p=0.044，因为截断率 15.8% vs 18.2%）。fp16 soup 只能和 `basefp16` 比。

## 结果

见 [docs/TASTE_EVAL_RESULTS_zh.md](../../../../docs/TASTE_EVAL_RESULTS_zh.md)。

## 明确排除：需要真跑实验的评测

这套 taste 套件**全部是 judge 型或客观 GT 型，没有任何一档需要我们真的跑实验、占 GPU 去验证模型提出的方案**。
执行落地那一档由既有的 FCS / ALE / MLS / Research 承担，这里不重复。

- **MLR-Bench 已剔除**（`chchenhui/mlrbench`）：它的 experimentation 阶段要真跑代码，
  而且调研自己也记了那一档 10 个任务里 8 个出现编造结果。即使只跑 idea+proposal 两阶段，
  与 GiantsBench / HypoArena / Lit2Test 的覆盖也高度重叠，收益不抵成本。
- **Lit2Test 不是执行型**：它 prompt 里那句 `<=8 A100 GPU-days` 是**给模型的资源约束**
  （提案必须落在这个预算内才算合格），不是我们要付出的算力。我们只做"生成六字段提案 + 成对盲判"。
