# Tinker 蒸馏臂：手写 reasoning vs 模型生成 reasoning

**验的假设**：我们手写的 innovation reasoning 对 Qwen3.5-4B 来说太 off-policy，
换成模型自己生成的推理会更好训。

**做法**：用 Tinker API 在**同一份** innovation 语料上 LoRA 微调 Inkling-Small，
再让它把每个计损轮的 `<think>` 重写一遍，最后用重写后的语料在本地训 Qwen3.5-4B，
和 `full_wd01` 对照。

## 为什么只重写 `<think>`

- 假设针对的是**推理**，answer 是真实的科学内容（发现 + 代码），动了就不是同一个实验
- agentic 轨迹里 observation **只对记录下来的那个 action 有效**；重采样 action
  会让后续所有 observation 失效，而我们没有环境可以重跑
- `<think>` 占计损文本的 **62.9%**（48.6M / 77.3M 字符），是主要变量

每个轮次都是 **teacher-forced** 在真实前缀上采样。

## 四步

```bash
export TINKER_API_KEY=...            # 注意：不要写进仓库

# 1. 渲染成 Inkling 线格式 + 逐轮 loss 掩码   (~10 min)
python3 experiments/scripts/tinker/build_data.py \
    --out .cache/tinker/inkling_innov.jsonl --max-len 65536 --holdout 96
#    -> 2805 train (733 agentic) / 32.55M tok / 19.45M 计损 ; 96 holdout ; 零丢弃

# 2. LoRA 微调 Inkling-Small                  (~80 min, 318 steps)
python3 experiments/scripts/tinker/train_inkling.py \
    --token-budget 32768 --accum 4 --lr 1e-4 --epochs 1 \
    --eval-every 25 --state .cache/tinker/inkling_run.json
#    checkpoint 指针写进 --state，脚本结束前会回读验证它能 resolve

# 3. 重写 4109 个轮次的 <think>               (长；可断点续跑)
python3 experiments/scripts/tinker/sample_inkling.py \
    --state .cache/tinker/inkling_run.json --concurrency 32 --max-tokens 16384 \
    --out .cache/tinker/innovation_distilled.jsonl

# 4. 合成训练语料（失败轮回落手写）+ 质检
python3 experiments/scripts/tinker/finalize_distill.py      # -> LF-innov/data/innovation_v2_distill.jsonl
python3 experiments/scripts/tinker/qc_distill.py
```

```bash
# 5. 本地训 4B（与 sft_full_wd01.yaml 只差两行：dataset / output_dir）
llamafactory-cli train experiments/tinker_distill_4b/sft_full_distill.yaml

# 6. 评测：runner 已参数化为 TAG / MODEL / GPU，且写进和 base/wd01/soup 同一个目录，
#    同节点同协议，score.py --intersect 可以直接配对比较
bash experiments/scripts/eval/taste/run_arm4b_redo.sh \
     pp_distill /srv/home/bohanlyu/models_sft/tinker_distill_4b/full_wd01_distill 3
```

对照臂已有的产物（同目录、同协议）：`cc_eval_pp_base_*` / `cc_eval_pp_wd01_*` / `cc_eval_pp_soup_a10_*`。
**判据看配对差，不看绝对分**；关键那档是 SciJudge 的换序一致性
（base 65.7% / wd01 60.7%，位置粘滞率 17.1% vs 22.7%）。

## 读 QC 的时候看什么

| 指标 | 手写基线 | 想要的方向 |
|---|---:|---|
| `scientist_voice_%` | 31.0 | **别掉**。第一人称科学家视角是这份语料唯一的独特资产；蒸成助手口吻就等于把它扔了 |
| `assistant_voice_%` | 11.4 | 别涨（"The user wants me to…"） |
| `hedges_%` | 9.9 | **涨**。wd01 的实测缺陷就是认知对冲从 93.3% 塌到 54.4% |
| `mentions_dead_end_%` | 14.5 | **涨** |
| `leak_front_loading_median` | 0.254 | 别涨 |
| `answer_terms_reached_median` | 0.828 | **别掉**，否则说明推理根本没走到答案 |

`leak_front_loading` 只统计**答案里 prompt 没给过**的实词，看它们在 `<think>`
的第一个五分位就出现了多少。总重合度是没用的指标——prompt 会把主题词同时喂给两边，
而且真推到答案的推理本来就会用答案的词。手写语料在这个指标上是 0.254→0.828 的
渐进曲线（只有 1.7% 的轮次首五分位就超 0.5），**是推导的形状，不是复述**。

## 直通校验（改了 finalize 之后重跑一遍）

空蒸馏文件跑 finalizer，输出必须和源语料**逐字节相同**——这证明两份语料之间
唯一能不同的东西只有被重写的 `<think>`，没有任何别的东西被顺手改掉：

```bash
: > /tmp/empty.jsonl
python3 experiments/scripts/tinker/finalize_distill.py \
    --distill /tmp/empty.jsonl --out /tmp/passthrough.jsonl
cmp experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl /tmp/passthrough.jsonl && echo OK
```
已验证通过（2901 行，byte-identical）。

## 两条臂：答案盲 vs 答案条件

`--condition-on-answer` 决定教师写 `<think>` 时能不能看到这一轮的答案。
在 A 已完成的 151 个轮次上配对比较（**不是** 24 条的把关样本）：

| 指标 | 手写 | A 答案盲 | B 答案条件 |
|---|---:|---:|---:|
| 答案词到达率 | 0.782 | **0.449** | **0.711** |
| 泄漏前置度 | 0.193 | 0.158 | 0.188 |
| 出现「我不确定」 | 20.5% | **29.8%** | 23.2% |
| 出现走死路 | 23.8% | **29.8%** | 25.2% |
| 科学家口吻 | 72.2% | **78.8%** | 60.3% |
| Jaccard vs 原文 | — | 0.001 | 0.001 |

到达率 **B−A = +0.232，95%CI [+0.198,+0.264]**，129/150 轮 B 更高。

**这是权衡，不是 B 完胜**：A 的推理更像在探索（怀疑、死路、口吻都最强，
正是诊断指向的缺陷），但**接不上它自己的结论**；B 接得上，探索性打折。
两份语料都建，让下游 4B 评测裁，不靠判断裁。

预判被推翻的一条：我原以为「告诉教师答案 → 退化成复述 → 前置度飙升」。
没有。B 的前置度 0.188，**低于**手写的 0.193。告知终点不等于让它复述终点。

### 教训：n=24 的把关不能判小差异

把关样本（24 轮）曾给出「A 把走死路砍半 29.2%→16.7%」。**n=151 下反转成
29.8%，比手写还高。** 24 条只够判否决级信号（no-op、口吻塌陷），
5–10 个百分点的差异必须等 n≥150。

## 报结果时必须一起报的数

`finalize_distill.py` 打印的**重写覆盖率**。回落比例高的臂是个残缺的 ablation，
它的零结果说明不了任何事。

## 坑

- SDK 要 ≥0.26；旧版被服务端直接拒（"no longer supported"）
- `tml_tokenizers` 不在 PyPI，`training_client.get_tokenizer()` 会崩；
  直接 `AutoTokenizer.from_pretrained("thinkingmachines/Inkling-Small")` 绕过
- Inkling 的 thinking 是**独立 message 通道**（`<|message_model|><|content_thinking|>…<|end_message|>`），
  不是 `<think>` 标签；tool call 走 `<|content_invoke_tool_json|>`，结果走 `<|message_tool|>`
- 语料里 `function_call` 的格式是 `<think>…</think>` + 可见散文 + `<tool_call>{json}</tool_call>`
- 源文件必须用 `innovation_v2_timeonly.jsonl`（wd01 实际训的那份，2901 行、带逐轮 loss 折叠），
  不是 `sft/innovation_sft.jsonl`
