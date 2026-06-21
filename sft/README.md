# Innovation Prior — SFT dataset (LLaMA-Factory ShareGPT)

Two **natural multi-turn** ShareGPT files (real roles, real `<think>` — no invented
structure). They are **build artifacts**: regenerate with `python3 sft/build_sft.py` from
the committed `methods/*/results/` and `trajectories/*/` sources (deterministic).

The model must learn two input distributions — history **with** reasoning vs history
**stripped** of reasoning. We do NOT reshape the data to fake either; we let LlamaFactory's
built-in `mask_history` flag + the official Qwen chat template produce them:

| file | train with | what LF does | distribution |
|---|---|---|---|
| `innovation_sft_kept.jsonl` | `mask_history=false` | keeps every turn's `<think>`, loss on every turn | history **with** reasoning (tool-loop / observation style) |
| `innovation_sft_stripped.jsonl` | `mask_history=true` | removes `<think>` from history turns (bare answer, exactly like the Qwen template renders stripped history) and computes loss on the **last** turn only | history **stripped** (post–user-query style) |

> Why this is faithful, not a hack: the Qwen3 / Qwen3.5 template renders a stripped history
> turn as the bare assistant answer with `<think>…</think>` removed (no block at all).
> LlamaFactory's `mask_history=true` reproduces exactly that via `remove_thought`. So the
> stripping is done by the framework + template, never by us folding text around.

## Contents

The answer is always the **`train_answer`**; reasoning goes in `<think>`. Each example's
`system` prompt carries the discovery **year** (method year for methods; the trajectory's
first-method year — from `trajectories.json` — for trajectories/agentic).

**kept (1494):**
- methods (1201) — single-turn `context → <think>reasoning</think>train_answer`.
- trajectories (166) — full multi-turn; measured feedback as `observation`.
- agentic (127) — full `edit→test` loop; **all** results (incl. `run_experiment`) as
  `observation`; assistant steps use the structured **`function_call`** role
  (`<think>…</think>{say}<tool_call>{json}</tool_call>`) so LF renders the per-model wrapper —
  **qwen3 JSON** or **qwen3_5 XML** — from the same file; tools declared per-example.

**stripped (2508):** the same conversations, emitted as **one prefix per target assistant
turn** (truncated there), real roles + real `<think>` left in place for LF to strip:
- trajectory rungs (678) — feedback becomes a `user` boundary; one prefix per rung.
- agentic steps (1830) — `run_experiment` result becomes a `user` boundary, `str_replace`
  results stay `observation`; one prefix per assistant step.

Across both files every **reasoning / answer / tool-call** is trained, in both the
history-with-reasoning and history-stripped contexts.

## Training

- Register: merge `dataset_info_snippet.json` into `LLaMA-Factory/data/dataset_info.json`,
  put the two `.jsonl` under `data/`.
- Template: `qwen3` (or `qwen3_5`) — both render the `function_call` turns correctly.
- `mask_history` is a **global** flag, so the two files are two configs: train `kept` with
  `mask_history=false`, `stripped` with `mask_history=true` (two runs, or two stages).

Literal structural tokens inside content (e.g. a method that discusses `<think>`) are
neutralized to `⟨think⟩`/`⟨tool_call⟩`/… so they can't collide with the real wrappers.
