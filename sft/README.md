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
3. **Agentic** (tool-using, MLS-derived) — the genuinely *mixed* case: `str_replace` results
   are `observation`s, `run_experiment` results are the test feedback. Two framings:
   - *continuum*: full `edit→test` loop, **all** results (incl. `run_experiment`) = `observation`,
     reasoning retained throughout (a real never-reset agent). [WITH history reasoning]
   - *per-round*: `run_experiment` result = a **`user` boundary**; `str_replace` results stay
     `observation`. Prior rounds fold into the `human` prompt (stripped); the current round's
     edit-loop + test call are real turns, all trained. [WITHOUT history reasoning across tests]

   Assistant tool steps use the structured **`function_call`** role
   (value = `<think>…</think>{say}<tool_call>{json}</tool_call>`) so LlamaFactory renders the
   per-model wrapper — **qwen3 JSON** or **qwen3_5 XML** — from the *same* file; tools declared per-example.

### Why multiple framings (Qwen3 / Qwen3.5 grounded)

Per the Qwen model cards + chat templates: one hybrid model; thinking toggled per-request by
`enable_thinking` (Qwen3.5 dropped the `/think` `/no_think` soft tokens). History reasoning is
**always** stripped from assistant turns before the most recent *real user query* — but a
`<tool_response>`/`observation` does **not** count, so reasoning is **retained inside a tool
loop**. In one causal forward pass a `<think>` block is present-or-absent for *all* later tokens,
so one sequence can't be both "history-with-reasoning" and "history-stripped". We therefore emit
each source under the framing whose train-render == its inference-render — the union trains
thinking, the post-think answer, and tool-use, and covers both input distributions. By folding
each sample's pre-boundary history into the `human` opening, every sample is a single
rolling-checkpoint episode, so `mask_history=False` is correct and uniform. See `build_sft.py` header.

## Training

- Register: merge `dataset_info_snippet.json` into `LLaMA-Factory/data/dataset_info.json`,
  put `innovation_sft.jsonl` under `data/`.
- Template: `qwen3` (or `qwen3_5`) — both render the `function_call` turns correctly.
- **`mask_history=False`** (default): loss on every `gpt`/`function_call` turn, so every
  rung's and every tool step's `<think>` is trained. The *per-rung* and *method* samples
  are single-target, so this stays correct for them too.

Literal structural tokens inside content (e.g. a method that discusses `<think>`) are
neutralized to `⟨think⟩`/`⟨tool_call⟩`/… so they can't collide with the real wrappers.

Current build: **2352 examples** (1201 methods · 166 traj-continuum · 512 traj-per-rung ·
127 agentic-continuum · 346 agentic-per-round).
