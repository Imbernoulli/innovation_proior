# Innovation Prior — SFT datasets (LLaMA-Factory ShareGPT)

All files here are **build artifacts** (git-ignored). Regenerate with:

```bash
python3 sft/build_sft.py        # innovation_sft.jsonl (our annotated data)
python3 sft/build_distill.py    # distill_sft.jsonl + distill_nothink_sft.jsonl (HF distillation)
```

## ⚠️ Requires a patched LLaMA-Factory (per-turn `loss` flag)

`innovation_sft.jsonl` uses a **per-turn `loss` flag** that upstream LLaMA-Factory does not
support. Use this fork/branch (clone it, it's ready to `pip install -e .`):

> **https://github.com/Imbernoulli/LLaMA-Factory** — branch **`feat/per-turn-loss-mask`**
> ```bash
> git clone -b feat/per-turn-loss-mask https://github.com/Imbernoulli/LLaMA-Factory.git
> cd LLaMA-Factory && pip install -e ".[torch,metrics]"
> ```

What the patch adds: an optional `"loss": false` field on a sharegpt turn excludes that turn
from the loss while keeping it as context. This is finer-grained than `mask_history` (which is
all-turns or last-turn-only) and is exactly what lets a sample **fold prior rounds (context, no
loss) while training every action of the current round**. Changes are in
`src/llamafactory/data/converter.py` + `processor/supervised.py`, with tests in
`tests/data/processor/test_loss_mask.py` (all green). Backward compatible: data without `loss`
flags trains exactly as before.

## Registering a dataset in LLaMA-Factory

LLaMA-Factory only trains datasets that are **registered** in `data/dataset_info.json`. To use
these files:

1. Copy the `.jsonl` file(s) into the fork's `LLaMA-Factory/data/` directory.
2. Merge the entries from `sft/dataset_info_snippet.json` into `LLaMA-Factory/data/dataset_info.json`
   (it maps the ShareGPT columns/role-tags; `formatting: sharegpt`, roles `human/gpt/observation/function_call`).
3. Reference the dataset name(s) in your train config, e.g.:
   ```yaml
   dataset: innovation_sft            # or: innovation_sft,innovation_distill
   template: qwen3                    # or qwen3_5
   mask_history: false                # per-turn `loss` flags do the masking
   ```
   CLI equivalent: `llamafactory-cli train --dataset innovation_sft --template qwen3 ...`.

## 1. `innovation_sft.jsonl` — our annotated data

The answer is always the **`train_answer`**; reasoning goes inside `<think>`. Each example's
`system` prompt carries the discovery **year** (method year; trajectory first-method year) as
meta-conditioning. Two framings per source, in one file:

- **Mode 1 "full"** — the whole conversation, every turn keeps its real `<think>`, every turn
  trained (no `loss` flags). History **with** reasoning.
- **Mode 2 "folded"** — for each round as the *current* round, prior rounds keep their
  answers/results but their `<think>` is emptied **and** marked `loss:false` (context, not
  trained); the current round keeps **all** its reasoning and is marked `loss:true` (every
  action trained). Current round derived against a reasoning-stripped history.

A round = one rung (trajectory) or one `run_experiment`-delimited block (agentic). Covers:
methods (single-turn), trajectories (Mode 1 feedback-as-observation + Mode 2 per-rung), agentic
(Mode 1 all-results-as-observation + Mode 2 per-round; assistant steps use the structured
`function_call` role so LF renders qwen3 JSON / qwen3_5 XML from one file).

Literal structural tokens inside content (e.g. a method that discusses `<think>`) are neutralized
to `⟨think⟩`/`⟨tool_call⟩`/… so they can't collide with the real wrappers.

## 2. Distillation data (off-policy Qwen traces from HuggingFace)

Our annotated set alone is fairly off-policy for fine-tuning a Qwen model, so we mix in public
Qwen-distilled traces. Built by `sft/build_distill.py` into a **separate** file (see below),
normalized to the same ShareGPT format.

_(sources, counts, and the no-reasoning handling are documented in the Distillation section
below once built.)_
