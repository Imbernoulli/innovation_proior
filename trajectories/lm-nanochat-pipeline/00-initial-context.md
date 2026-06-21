## Research question

Build a usable ChatGPT-style assistant end-to-end on a single 8×H100 GPU node for about **$100**. There is no pretrained checkpoint to start from, so the pipeline must cover every stage: train a tokenizer, pretrain a base model from random init, teach it to converse and use tools, and finetune it into an assistant. The only thing being maximized is the **report card**: DCLM **CORE** for the base model, and ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, plus **ChatCORE** for the chat model.

What is fixed is the budget and the harness: one node, a `--depth` dial that sizes the model, explicit bf16/fp8 precision, and the report card itself. The free variable is the **pipeline**: architecture, optimizer, data, special tokens, conversation format, tool-use channel, and final objective.

## Prior art / Background / Baselines

- **GPT-2.** A 1.5B-parameter decoder-only transformer trained on WebText with Adam, which scores **0.256525** on DCLM CORE and was trained on a TPU cluster for roughly $43k. Gap: reproducing that capability on a single node at dinner-money cost is an efficiency problem it was never designed for.

- **nanoGPT / modded-nanogpt.** A minimal, hackable GPT implementation and a speedrun leaderboard that squeezes pretraining on one node with tricks like rotary embeddings, QK-normalization, Muon, ReLU² MLPs, value embeddings, logit softcapping, untied embeddings, and zero-init projections. Gap: it only covers base-model pretraining; it says nothing about tokenizer design, conversation formatting, tool use, supervised finetuning, or reinforcement learning.

- **The ChatGPT recipe (pretrain → SFT → RLHF).** The standard assistant pipeline at frontier scale. Gap: it assumes large compute and human-annotation pipelines, and no published version collapses the whole tokenizer-to-assistant path onto one node with a measured per-stage report card. Its pieces also do not transfer unchanged to a 20-layer model.

## Fixed substrate / Code framework

One 8×H100 node (`torchrun --nproc_per_node=8`), sequence length 2048, vocabulary size 32768. Weights stay fp32 for the optimizer; matmuls run in bf16, or fp8 on Hopper via torchao tensorwise scaling. Model size is set by one integer, `--depth`, with width, heads, learning rates, and training horizon derived to keep the model compute-optimal. Training targets a tokens-to-params ratio around 8–10. Each pipeline stage writes a section to a markdown report card, and `report.py` assembles the per-stage metric table.

## Editable interface

The editable parts are the pipeline stages and their components: tokenizer design; base architecture and optimizer choices; conversation format, special tokens, and tool-use channel; supervised-finetuning data mixture; and the final objective that squeezes reasoning accuracy out of a tiny model. These are exposed through stage configs, data-mix weights, special-token maps, and the stage runner that `report.py` expects to consume a checkpoint and emit a metric row.

## Evaluation settings

- **Base model:** DCLM **CORE** (ensemble of 22 evaluations; GPT-2 = 0.256525), validation **bits-per-byte** (lower is better), and sampled generations for sanity.
- **Chat model:** accuracies on **ARC-Easy**, **ARC-Challenge**, **MMLU**, **GSM8K** (generative, calculator tool allowed), and **HumanEval** (Python coding, sandboxed execution).
- **ChatCORE:** centered mean across the six chat tasks, rescaled by `(acc − baseline) / (1 − baseline)`, where baseline is 0.25 for 4-way multiple-choice tasks and 0 for generative tasks, so 0 = random and 1 = perfect.
- **Budget yardstick:** total wall-clock and dollar cost at ~$24/hr for the node; the reference run targets well under four hours and ~$100.
