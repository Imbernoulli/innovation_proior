# Innovation Prior — SFT datasets (LLaMA-Factory ShareGPT)

Two ShareGPT files, designed to train **together in ONE run**:
- `innovation_sft.jsonl` — our annotated innovation data (reasoning, with per-turn loss folding).
- `maintain_sft.jsonl` — public **Qwen**-distilled traces mixed in to **maintain the base model's
  original capabilities** (on-policy replay against forgetting; reasoning *and* no-reasoning, the
  latter handled so it doesn't corrupt thinking).

> **Browse it on the site.** Every example is viewable in the website's **Training data** mode
> (`#d`), which lazy-loads the gzipped shards under `viewer/` and shows the per-turn `loss` /
> `enable_thinking` metadata. Regenerate the viewer catalogue with
> `python3 tools/build_site_data.py` whenever these `.jsonl.gz` files change.

The processed data is committed **gzipped** here: `innovation_sft.jsonl.gz`,
`maintain_sft.jsonl.gz` — decompress before training (`gunzip -k *.jsonl.gz`). The raw `.jsonl`
are git-ignored; regenerate either the raw files or refresh the gzips with:
```bash
python3 sft/build_sft.py        # innovation_sft.jsonl
python3 sft/build_maintain.py   # maintain_sft.jsonl  (assembles the per-source HF pieces)
gzip -kf sft/innovation_sft.jsonl sft/maintain_sft.jsonl
```

## ⚠️ Requires the patched LLaMA-Factory fork

These files use **two per-example metadata flags** upstream LLaMA-Factory doesn't support, so
everything can train in one run:

> **https://github.com/Imbernoulli/LLaMA-Factory** — branch **`feat/per-turn-loss-mask`**,
> **commit `494ff82` or later (required)**
> ```bash
> git clone -b feat/per-turn-loss-mask https://github.com/Imbernoulli/LLaMA-Factory.git
> cd LLaMA-Factory && pip install -e ".[torch,metrics]"
> ```
>
> ⚠️ **2026-07 fixes (`494ff82`) — do not train on earlier states of the branch:**
> 1. **`loss: null` schema landmine.** A jsonl mixing flagged and unflagged examples gets ONE
>    unified pyarrow message struct; the absent `loss` key materializes as `None` and the old
>    `bool(message.get("loss", True))` silently masked EVERY turn of every unflagged example
>    (zero loss, no warning) — or, depending on datasets version and row order, the load crashed
>    on the schema cast. Fixed: absent and `None` both mean "train". Belt-and-braces, the build
>    scripts now write a **uniform schema**: explicit `"loss": true` on every assistant turn and
>    explicit `tools`/`enable_thinking` on every row (both shipped `.jsonl.gz` are already patched).
> 2. **Folded turns render without an empty think.** `ReasoningTemplate` used to re-inject
>    `<think>\n\n</think>` into every think-less assistant turn at encode time, undoing the
>    `fold_think` data fix and conditioning the model on empty-think history that never occurs at
>    inference. `loss:false` turns now render with **no** think block (official-template behavior).
>
> **Invariant (enforced):** a turn whose think was stripped by folding NEVER enters the loss —
> `build_sft.py` hard-asserts it at build time (folded turns are `loss:false`, trained turns are
> exactly the trailing current round), and the fork masks those targets to IGNORE_INDEX
> (regression-tested end-to-end on the real files).

1. **Per-turn `loss`** — a `"loss": false` on a sharegpt turn keeps it as context but excludes it
   from the loss (finer than `mask_history`). Used for innovation_sft's folded history.
2. **Per-example `enable_thinking`** — a top-level `"enable_thinking": false` renders that example's
   empty think into the **prompt** (no loss), so **non-reasoning** data trains in the same `qwen3`
   (thinking) run without teaching "open-think → immediately close-think".

Both are in `src/llamafactory/data/{converter.py,processor/supervised.py}`, with tests in
`tests/data/processor/test_loss_mask.py` and `test_enable_thinking.py` (all green). Fully backward
compatible: data without these fields trains exactly as before.

## Registering & training everything in ONE run

LLaMA-Factory only trains **registered** datasets. To run all four data kinds (our annotated data,
Qwen distill, MiniMax distill, no-reasoning Qwen) together:

1. Copy `innovation_sft.jsonl` and `maintain_sft.jsonl` into the fork's `LLaMA-Factory/data/`.
2. Merge `sft/dataset_info_snippet.json` into `LLaMA-Factory/data/dataset_info.json`.
3. One training config trains both at once:
   ```yaml
   dataset: innovation_sft,innovation_maintain
   template: qwen3              # or qwen3_5
   mask_history: false          # per-turn `loss` flags do the folding
   # enable_thinking stays at the template default (true); the no-reasoning rows
   # carry "enable_thinking": false per-example and override it themselves.
   ```
   The per-turn `loss` flags (folding) and per-example `enable_thinking` flags (reasoning vs not)
   are baked into the data, so a single global config handles every case.

## 1. `innovation_sft.jsonl` — our annotated data

Answer is always the **`train_answer`**; reasoning goes in `<think>`. Each example's `system`
prompt carries the discovery **year** (method year; trajectory first-method year). Two framings per
source, in one file:

- **Mode 1 "full"** — whole conversation, every turn keeps its real `<think>`, every turn trained
  (no loss flags). History **with** reasoning.
- **Mode 2 "folded"** — for each round as the *current* round, prior rounds keep answers/results but
  their `<think>` is emptied **and** `loss:false` (context, not trained); the current round keeps
  **all** its reasoning and is `loss:true` (every action trained). Current round derived against a
  reasoning-stripped history. A round = one rung (trajectory) / one `run_experiment` block (agentic).

Covers methods (single-turn), trajectories (Mode 1 feedback-as-observation + Mode 2 per-rung),
agentic (Mode 1 all-results-as-observation + Mode 2 per-round; assistant tool steps use the
structured `function_call` role → LF renders qwen3 JSON / qwen3_5 XML). Literal structural tokens
in content are neutralized to `⟨think⟩`/`⟨tool_call⟩`/….

## 2. `maintain_sft.jsonl` — capability-maintenance mix (903 examples)

All sources are **Qwen-distilled** — the goal is to **maintain the base Qwen model's original
capabilities** (on-policy replay against catastrophic forgetting) while we fine-tune on the
innovation data. One file; the no-reasoning rows are tagged `enable_thinking:false` so they coexist
with the reasoning rows in the same run.

**Reasoning (653, trained with thinking):**

| source | kept | notes |
|---|---|---|
| `khazarai/qwen3.6-plus-high-reasoning-500x` | 250 | already `<think>` |
| `WithinUsAI/Qwen3.7_Max_Thinking_dataset_5K` | 250 | `problem`/`thinking_trace`/`answer` → `<think>` |
| `armand0e/qwen3.7-max-pi-traces` | 47 | agentic (pi); all non-empty sessions |
| `armand0e/qwen3.7-plus-claude-code` | 6 | agentic (Claude Code); all non-empty sessions |
| `nvidia/Open-SWE-Traces` openhands **`minimax_m25`** | 100 | the only MiniMax data, capped to 100 |

> "non-empty" = the session actually has **model output** (a real assistant turn). The two armand
> repos hold only 47 / 6 sessions total; every one with model output was kept.

**No-reasoning (250, `enable_thinking:false`):** `nvidia/Open-SWE-Traces` openhands **`qwen35_122b`**
(the Qwen split). These assistant turns carry an **empty** think in the source (Qwen non-thinking
mode); we store them as plain answers and tag the example `enable_thinking:false`, so at train time
the empty think lands in the **prompt**, never the loss. (Verified: 0 `<think>` substrings trained.)

**Tool declarations:** the agentic traces (armand, Open-SWE) carry tool *calls* but declare no tool
*schemas*. `build_maintain.py` reconstructs a minimal `tools` declaration per example from the
observed calls (tool name + argument keys) so the tools render in the system prompt at the right
place. `<|im_start|>`/`<|im_end|>` occurring inside content are neutralized.
