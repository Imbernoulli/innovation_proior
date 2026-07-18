# Innovation Prior — SFT datasets (LLaMA-Factory ShareGPT)

The annotated innovation data, in ShareGPT format:
- `innovation_sft.jsonl` — our annotated innovation data (reasoning, with per-turn loss folding).
- Plus the 2026-07 **wave-2** batches: `innovation_wave2_sft.jsonl` (verified rollout + Codex, 758)
  and `innovation_v4_sft.jsonl` (FrontierCS-style single-file C++, 346), concatenable into the run.
- Plus the 2026-07 **wave-3** batch: `innovation_wave3_sft.jsonl` (179, **hard-only**) — every NEW
  verified keeper since wave-2 that the 27B did **not** find easy (round-0 acc ≤ 0.5), one answer per
  problem. Adds the FrontierCS capability gaps (heuristic **optimization**, post-cutoff **AtCoder
  Heuristic**, CodeContests+ strong-test, and a deep re-roll of the 27B's hard failures).
  Concatenable into the same run. See **§4**.

> **Dropped (2026-07):** the HF-scraped `maintain_sft.jsonl` capability-maintenance set is no longer
> used — training is **innovation-only** now.

> **Browse it on the site.** Every example is viewable in the website's **Training data** mode
> (`#d`), which lazy-loads the gzipped shards under `viewer/` and shows the per-turn `loss` /
> `enable_thinking` metadata. Regenerate the viewer catalogue with
> `python3 tools/build_site_data.py` whenever these `.jsonl.gz` files change.

The processed data is committed **gzipped** here: `innovation_sft.jsonl.gz` (+ the wave-2
`innovation_wave2_sft.jsonl.gz` / `innovation_v4_sft.jsonl.gz`) — decompress before training
(`gunzip -k *.jsonl.gz`). The raw `.jsonl` are git-ignored; regenerate with:
```bash
python3 sft/build_sft.py        # innovation_sft.jsonl
python3 sft/build_v4.py         # innovation_v4_sft.jsonl (FrontierCS-style C++)
gzip -kf sft/innovation_sft.jsonl sft/innovation_v4_sft.jsonl
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

LLaMA-Factory only trains **registered** datasets:

1. Copy `innovation_sft.jsonl` (and optionally the wave-2 `innovation_wave2_sft.jsonl` /
   `innovation_v4_sft.jsonl`) into the fork's `LLaMA-Factory/data/`.
2. Merge `sft/dataset_info_snippet.json` into `LLaMA-Factory/data/dataset_info.json`.
3. Training config:
   ```yaml
   dataset: innovation_sft        # optionally + the wave-2 batches
   template: qwen3              # or qwen3_5
   mask_history: false          # per-turn `loss` flags do the folding
   ```
   The per-turn `loss` flags (folding) are baked into the data, so a single global config handles
   every case.

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

## 2. Capability-maintenance mix — DROPPED (2026-07)

A public **HF-scraped** Qwen-distilled maintenance set (`maintain_sft.jsonl`, 903 examples from
khazarai / WithinUsAI / armand0e / nvidia Open-SWE) was previously mixed in against catastrophic
forgetting. It has been **removed at the user's direction** — training is now **innovation-only**,
relying on the verified wave-2 rollout data (which itself spans reasoning / instruction-following /
agentic C++/Python) for on-policy breadth. `build_maintain.py`, the `maintain_sft*` and `distill_*`
files, and the viewer's maintain shards were deleted.

## 3. Wave-2 batches (2026-07) — verified rollout + Codex

New verified data, all landing as **single-file C++ / stdin** (the FrontierCS scoring target):

- **`innovation_wave2_sft.jsonl`** (758) — Qwen3.6-27B on-policy rejection samples + DeepSeek V4 Pro
  tier-2 (solving the 27B's hard failures; ungradeable math gold judged by DeepSeek V4 Flash) +
  Codex `gpt-5.5` black-box datapoints. code / math / reasoning / ifollow / FrontierCS-Codex.
- **`innovation_v4_sft.jsonl`** (346) — competition C++, 100% single-file/stdin, 100% debug/self-verify.
- **`innovation_wave2_raw_keepers.jsonl.gz`** — 787 RAW verified keeper records (problem + all
  verifier-passing generations); verified-only, no failed samples.

Pipeline + provenance: [`../experiments/DATA_WAVE2_FCS_CPP_zh.md`](../experiments/DATA_WAVE2_FCS_CPP_zh.md).

## 4. Wave-3 batch (2026-07) — capability-gap injection + deep re-roll

`innovation_wave3_sft.jsonl` (**179**, gzipped as `innovation_wave3_sft.jsonl.gz`) = every verified
keeper produced **after** wave-2, with the wave-2 ids subtracted so there is **zero overlap**. Built
with `python3 tools/assemble_wave3.py`, same **hard-only** bar as wave-2:

- **round-0 acc ≤ 0.5** — keep a problem only if the 27B solved ≤ half of its first 4 samples
  (`WAVE_ACC_MAX=0.5`, the default). Easy problems teach nothing; this drops the big `acc=0.75`
  bulk and everything the 27B aced 4/4.
- **one answer per problem** — the single shortest verified generation (`passes[0]`), deduped by id.

Same ShareGPT + `<think>` format. Snapshot 2026-07-17 — the rollout is still running (ccplus + the
math/ifollow re-roll), so this file gets refreshed as more hard keepers land.

| domain | examples | what it is |
|---|---:|---|
| optim | 87 | **NEW** — NP-Engine heuristic optimization (TSP/knapsack/set-cover/…): write one C++ that reads stdin, prints `Answer: …`; verified feasible **and** beats a per-instance baseline on K fresh instances |
| code | 58 | HardTests CF/AtCoder + **CodeContests+ (`ccplus`)** strongest-test exact-judge |
| math | 14 | deep re-roll of the 27B's hard failures |
| ahc | 12 | **NEW** — post-cutoff **AtCoder Heuristic Contests** (AHC047–067 + awtf25/26); C++ scored by the OFFICIAL AtCoder Rust `vis` binary on every seed, must beat a greedy baseline |
| ifollow | 8 | deep re-roll |

17 of the 179 are deep-re-roll keepers. Reasoning length: median **120k** chars, max **213k** — the
hard-only cut keeps exactly the long, self-checking traces the FrontierCS regression forensics said
were missing ("提案的嗓音在,写代码的手没了"). All land as the FrontierCS scoring target:
**single-file C++ / stdin**.

**Why these sources (grounded in the real eval, not a summary).** FrontierCS `algorithm` is 92%
optimization / partial-score and 58% interactive; our whole rollout had been 100% exact-judge CF —
matching almost none of it. So wave-3 injects the missing shape with **strong** verifiers (weak
tests would re-poison): optim uses the vendored NP-Engine validator + a real baseline gate; ahc uses
the official scorer binaries.

**Deep re-roll.** wave-2 gave up on a problem after 16 samples. wave-3 re-samples the genuine
hard-failures (passed=False, not too-easy, not already solved by a teacher pass) with a deep budget
(schedule 4→8→…→256) so the hardest problems finally yield a keeper. The signal is differentiated:
`ifollow` recovers ~47%, `math` ~19%, but `reasoning` was **~0% and has been dropped** from the
re-roll — its hard tail is genuinely beyond the 27B even at 256 samples, so that slice should go to a
**teacher** (DeepSeek) pass rather than more self-sampling.

**Known caveat (optim).** The optim baseline (nearest-neighbour, ratio 1.0) is **lenient** — 145/328
problems were aced 4/4 and dropped as too-easy, and there were **0** hard-failures. The 183 kept have
discriminative signal (the 27B fails them at least sometimes) but the difficulty ceiling is low;
tightening the baseline (NN+2-opt, or ratio<1) would make this track pull harder.

**Decontamination (lenient line — avoid only the actual *evaluation set*).** Contest-derived tracks
are fine as training data; we only guard against the eval benchmarks themselves. `ahc` excludes
AHC≤046 (= ALE-Bench's 40) and was cross-checked vs the public FrontierCS statement set (max Jaccard
0.009). `ccplus` is CodeContests+ (a training corpus, oracle-re-verified 100%), deduped vs the
existing worklists.

Provenance / integration for the new domains: each is a self-contained
`data_v4/_hardcp/<domain>/` dir with its own `verify.py` (exposing `verify(generation_text, problem)`)
and `worklist.jsonl`; rebuild the wave with `python3 tools/assemble_wave3.py`.
