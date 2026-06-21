# Innovation Prior — SFT dataset (LLaMA-Factory ShareGPT)

`innovation_sft.jsonl` is a single unified ShareGPT file. It is a **build artifact** —
not committed; regenerate it with `python3 sft/build_sft.py` from the committed
`methods/*/results/` and `trajectories/*/` sources (deterministic).

## What's in it (multi-framing)

The answer content is always the **`train_answer`** (the scientist write-up); the
reasoning goes inside `<think>…</think>`. Each example's `system` prompt encodes the
discovery **year** as meta-conditioning (the method's year for methods; the
trajectory's first-method year — from `trajectories.json` — for trajectories/agentic).

1. **Methods** (single-turn): `human` = `context.md`, `gpt` =
   `<think>{reasoning.md}</think>{train_answer.md}`. One per registered method.
2. **Trajectories — two framings of the same ladder:**
   - *continuum* (multi-turn): `human` = initial context, then per rung
     `gpt` = `<think>{reasoning}</think>{train_answer}`, with the measured **feedback
     as an `observation`** (a `<tool_response>`). An observation does **not** reset the
     Qwen rolling-think checkpoint, so every rung keeps its `<think>` in context and
     every rung is trained.
   - *per-rung* (single-turn, one per rung ≥ 2): the prior rungs (answers only, think
     stripped) are folded into the `human` prompt; `gpt` = this rung's
     `<think>{reasoning}</think>{train_answer}`. Trains each rung in a fresh,
     history-think-stripped context (the `test=user` serving form).
3. **Agentic** (tool-using, MLS-derived): full `edit→test` loop. Assistant tool steps use
   the structured **`function_call`** role (value = `<think>…</think>{say}<tool_call>{json}</tool_call>`)
   so LlamaFactory renders the per-model wrapper — **qwen3 JSON** or **qwen3_5 XML** — from
   the *same* file; tool results come back as `observation`; tools are declared per-example.

### Why two trajectory framings

In one causal forward pass a `<think>` block is either present or not for *all* later
tokens; the rolling-think rule needs it present within its episode and absent after the
next real user query — one sequence can't do both. So the same ladder is emitted under
both self-consistent framings; their union trains thinking, the post-think answer, and
tool-use, and generalizes across serving modes. See `build_sft.py` header for details.

## Training

- Register: merge `dataset_info_snippet.json` into `LLaMA-Factory/data/dataset_info.json`,
  put `innovation_sft.jsonl` under `data/`.
- Template: `qwen3` (or `qwen3_5`) — both render the `function_call` turns correctly.
- **`mask_history=False`** (default): loss on every `gpt`/`function_call` turn, so every
  rung's and every tool step's `<think>` is trained. The *per-rung* and *method* samples
  are single-target, so this stays correct for them too.

Literal structural tokens inside content (e.g. a method that discusses `<think>`) are
neutralized to `⟨think⟩`/`⟨tool_call⟩`/… so they can't collide with the real wrappers.

Current build: **2006 examples** (1201 methods · 166 continuum · 512 per-rung · 127 agentic).
