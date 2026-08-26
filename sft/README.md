# Innovation Prior — SFT datasets (LLaMA-Factory ShareGPT)

The annotated innovation data, in ShareGPT format:
- `innovation_sft.jsonl` — our annotated innovation data (reasoning, with per-turn loss folding).
- Plus the 2026-07 **wave-2** batches: `innovation_wave2_sft.jsonl` (verified rollout + Codex, 758)
  and `innovation_v4_sft.jsonl` (FrontierCS-style single-file C++, 346), concatenable into the run.
- Plus the 2026-08 **wave-3** batch: `innovation_wave3_sft.jsonl` (**5,291**, FINAL) — every NEW verified
  keeper since wave-2 (one answer per query, **each labeled with its `pass_rate`**); wave-2 + wave-3
  together now cover **all ~2,950** solved queries the distillation campaign ever solved. Adds the
  FrontierCS capability gaps (heuristic **optimization**, post-cutoff **AtCoder Heuristic**,
  CodeContests+ strong-test, and a deep re-roll of the 27B's hard failures). Concatenable into the
  same run. See **§4**.

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
agentic (assistant tool steps use the structured `function_call` role → LF renders qwen3 JSON /
qwen3_5 XML). Literal structural tokens in content are neutralized to `⟨think⟩`/`⟨tool_call⟩`/….

**Agentic v2 (2026-08-24)** — the June `agentic_messages.json` corpus was rebuilt from scratch
(tag `agentic-v2-retires-v1`, generated data deleted; tools: `tools/build_agentic_v2.py` deterministic skeleton +
Opus prose fills via `tools/apply_agentic_fills.py`, per-task `agentic_v2.json` /
`agentic_v2_fills.json` / `agentic_v2_filled.json`). v2 = **164 tasks** (127 originals kept +
37 new; 8 decontam tasks excluded), speaking the **live three-way harness contract** (SFT =
RL = eval, Princeton protocol — no `rewrite` op): `edit(op='str_replace'|'create')` + `view` + `test()`/`submit(n)`, the
post-edit current-file echo, real harness result strings, budgets, and line-numbered file
dumps. Framing is **deduped folded-only**: one example per round c=0..n-1 (test result = user
boundary) plus a trailing submit round — every action enters the loss **exactly once** across
the 752 rows (v1 trained the same text up to 4×/epoch). Every trained turn carries a real
think (v1 had 33.6% zero-think turns rendered as empty `<think>` in the loss); think shapes
are calibrated on real rollouts (long design think per rung, short followups, pre-test
expectations, submit comparison). Nine cumulative-stack ladders (airbench, nanoGPT chain,
vLLM/llm.c stacks, RoBERTa) use a **create-chain** framing (one new module per rung) because
their measured numbers stack techniques rather than replace them.

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

## 4. Wave-3 batch (2026-08) — capability-gap injection + deep re-roll

`innovation_wave3_sft.jsonl` (**5,291**, gzipped as `innovation_wave3_sft.jsonl.gz`) = every verified
keeper produced **after** wave-2, with the wave-2 ids subtracted so there is **zero overlap**. Built
with `python3 tools/assemble_wave3.py`. **FINAL — the distillation campaign completed 2026-08-20**:
every queued worklist is exhausted (code fresh 1,801/1,801, cfr1 2,037/2,038, cp2 151/151, ccplus
793/793, ioi 43/43) and the serving cluster is shut down. wave-2 (741) + wave-3 (5,291)
= all **6,032** unique queries the campaign ever solved with a saved generation — the
assembler reads the archived trace phases too (`.oldlogic` stop-at-first-pass, `.hardv2`/`.mixed`/
`.hardrun` old math runs, `.measure`), which fill 184 queries nothing modern solved (94 rstar code +
90 math). Policy (2026-08, updated from the earlier hard-only cut):

- **ship EVERY query that has ≥1 verified-correct generation** — no accuracy cap. We no longer keep
  only the hard (acc≤0.5) slice; instead every solvable query ships and carries its pass rate, so
  downstream can filter however it likes. (`WAVE_ACC_MAX=0.5` still reproduces the old hard-only cut.)
- **one answer per query** — the single shortest verified generation (`passes[0]`), deduped by id.
- **each row is LABELED with `pass_rate`** — a top-level float = the **round-0 pass rate of the model
  that produced the trace** (see the caveat below on what "round 0" means per source).

Same ShareGPT + `<think>` format, plus the new `pass_rate` field. **FINAL snapshot 2026-08-20 —
the campaign is complete and the serving cluster is shut down; this file will not grow further.**

| domain | examples | what it is |
|---|---:|---|
| reasoning | 1247 | base-trace growth + deep re-roll of the 27B's hard failures |
| code | 1856 | HardTests CF/AtCoder + **CodeContests+ (`ccplus`, 793/793 complete)** strongest-test exact-judge + 94 archived rstar-era solves + Qwen3.8 fresh pass 1,801/1,801 (1,092 keepers, 61%) |
| ifollow | 421 | base-trace growth + deep re-roll |
| math | 532 | base-trace growth + deep re-roll + 90 archived (hardv2/mixed/oldlogic) solves |
| optim | 183 | **NEW** — NP-Engine heuristic optimization (TSP/knapsack/set-cover/…): write one C++ that reads stdin, prints `Answer: …`; verified feasible **and** beats a per-instance baseline on K fresh instances |
| ioi | 22 | **NEW** — IOI 2020–24 (43/43 rolled), official graders, subtask partial scoring, PASS = score ≥ 35; incl. 13 interactive/Communication |
| cfr1 | 945 | **NEW** — open-r1 Codeforces rating 2000–3500, **complete 2,037/2,038** (944 keepers, 46%), oracle-verified strong tests |
| cp2 | 67 | **NEW** — COCI 2023–26 + USACO seasons 24–26 (Plat/Gold/Silver) with **OFFICIAL test data**, **complete 151/151** (67 keepers, 44%); oracle-gated 151/151, negative-gated 453/453 |
| ahc | 18 | **NEW** — post-cutoff **AtCoder Heuristic Contests** (AHC047–067 + awtf25/26); C++ scored by the OFFICIAL AtCoder Rust `vis` binary on every seed, must beat a greedy baseline |

39 are deep-re-roll keepers. Reasoning length: median **51k** chars, max **332k** (the hard tail
still holds the long self-checking traces; the median drops vs the hard-only cut because the easy
queries are now included too). All land as the FrontierCS scoring target: **single-file C++ / stdin**.

**Qwen3.8-27B teacher pass (started 2026-08-14).** Every query Qwen3.6-27B failed at ANY budget (incl.
the 256-sample deep re-roll) and no teacher solved — **1,612 queries** (reasoning 666 / math 455 / code 314 /
ifollow 153 / ioi 22 / ahc 2), written to `data_v4/_hardcp/<domain>/unsolved.jsonl` — is being re-rolled by
**Qwen3.8-27B** (`Qwen/Qwen3.8-27B`, thinking mode, TP=4 on 4×H100, schedule 4→32, keep every solve) into
`traces/<domain>.q38.jsonl`; `source: Qwen3.8-27B` marks those rows. Two silent verifier bugs found & fixed 2026-08-17 (missing `pylatexenc` + a dead numeric path in the reasoning table scorer; DeepSeek judge at 402 → local Qwen3.8 is now the math judge) — table__* and ungradeable-gold math had been judged FAIL for the whole campaign. Qwen3.8 also runs a second pass over the 6,508 queries Qwen3.6 never attempted (`never_attempted.jsonl` → `.q38b.jsonl`; sample once, double on failure). Yield: cfr1/code/math fresh queries ~90-100%, reasoning table 94%. The finished wave ships **3,062** q38 keepers (2,140 in C++ coding domains); all q38 passes are complete (code fresh 1,801/1,801 → 61% keepers, cfr1 2,037/2,038 → 46%, cp2 151/151 → 44%). One attempted domain was **retired with 0 keepers**: `cgshop` (CG:SHOP 2026/2025 geometry optimization) — Qwen3.8's thinking never terminates on open-ended heuristic-optimization prompts (all 6 end-to-end probes at 32k AND 57k max-tokens hit `finish=length` with `</think>` never closed; no repetition loop — the model genuinely never stops designing), so it produced nothing to ship.

**Pass-rate label (`pass_rate`) — read this before filtering on it.** It is the generating model's
round-0 pass rate for that query. Distribution across the 5,291:

| pass_rate | -1.0 (unknown) | 0.0 | 0.25 | 0.33 | 0.5 | 0.67 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| examples | 98 | 1003 | 183 | 12 | 172 | 29 | 941 | 2853 |

1,370 (26%) are **hard** (0 ≤ pass_rate ≤ 0.5); 3,823 are easy (> 0.5); **98 are `-1.0` = UNKNOWN** —
the archived stop-at-first-pass phase recorded no round-0 batch, so no rate exists (we refuse to fake
a 0.0 for them).

**`pass_rate = 0.0` does NOT mean "never solves it" — and every row also carries `samples_used`.**
pass_rate is measured over round 0 only (4 samples), so its resolution is quarters: values strictly
between 0 and 0.25 are impossible by construction. A `0.0` row means the model went **0/4 in round 0
and only cracked the problem during escalation** (8 → 16 → … → 256). To recover the fine-grained
rate, use the `samples_used` field (total budget consumed when it cracked): the true rate of a 0.0
row is **~1/samples_used**. Distribution of the 1,003 zero rows: 293 cracked within 2 samples, 243
within 4, 275 within 8, 165 within 16, 9 within 32, 9 within 64, 4 within 128, 5 within 256
(~1/256 — the rarest solves in the set). For non-zero rows samples_used is just the deciding-round
budget (4 for most).

**Caveat on semantics** — round 0 is **4 samples** for the 27B on-policy traces and
the deep re-roll (39 rows), so their pass_rate ∈ {0, .25, .33, .5, .67, .75, 1.0}; but for the
**teacher (DeepSeek) rows** and the **Qwen3.8 fresh pass (`.q38b`, schedule starts at 1 sample)**
round 0 is a single sample, so their pass_rate ∈ {0, 1} and a `1.0` there means "solved on the first
try" — **not** "easy" in the 4/4 sense (and for teacher rows the query is a 27B hard-failure by construction). So a `pass_rate = 1.0`
teacher row is a hard problem, whereas a `pass_rate = 1.0` on-policy row is one the 27B aced 4/4.
Filter with the source in mind (`_wave3_tags.jsonl` carries `source` + `reroll` + `pass_rate` +
`samples_used` per id).

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
problems were aced 4/4 and dropped **at rollout time** (driver ran with `easy-threshold 0.5`, so those
generations were never saved and can't be shipped regardless of this wave's policy), and there were
**0** hard-failures. The 183 shipped have discriminative signal (the 27B fails them at least
sometimes) but the difficulty ceiling is low; tightening the baseline (NN+2-opt, or ratio<1) would
make this track pull harder.

**Decontamination (lenient line — avoid only the actual *evaluation set*).** Contest-derived tracks
are fine as training data; we only guard against the eval benchmarks themselves. `ahc` excludes
AHC≤046 (= ALE-Bench's 40) and was cross-checked vs the public FrontierCS statement set (max Jaccard
0.009). `ccplus` is CodeContests+ (a training corpus, oracle-re-verified 100%), deduped vs the
existing worklists.

Provenance / integration for the new domains: each is a self-contained
`data_v4/_hardcp/<domain>/` dir with its own `verify.py` (exposing `verify(generation_text, problem)`)
and `worklist.jsonl`; rebuild the wave with `python3 tools/assemble_wave3.py`.
