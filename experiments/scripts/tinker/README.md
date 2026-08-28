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

然后本地训 4B：`experiments/tinker_distill_4b/sft_full_distill.yaml`
（与 `experiments/v2_multisetting_4b/sft_full_wd01.yaml` **只差两行**：dataset / output_dir）。

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
