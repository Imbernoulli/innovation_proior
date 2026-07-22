# Synthetic Open-Ended Problem Generation — Final Report

A pipeline that batch-generates **open-ended, deterministically-scored coding problems** to train
models that *generalize* across the "LLM writes code to optimize a scored objective" space. It
re-implements the withheld parts of **FrontierSmith** (arXiv 2605.14445) and extends the idea across
the whole evolutionary-search / scientific-discovery landscape.

**Result: 1006 problems, all machine-verified, spanning 638 families and 1006 unique scaffolds.**

---

## 1. Background & scope

FrontierSmith mutates closed-ended competitive-programming problems into open-ended optimization
problems (change goal / restrict output / generalize input), filters by an idea-divergence metric, and
has LLM agents synthesize + cross-validate test/checker infrastructure. It open-sources 10 sample
problems + training/eval code; **the orchestrator and the test/checker generators are withheld** — we
re-implement them.

We then broadened beyond FrontierCS (per project direction) to draw archetypes from **FunSearch,
AlphaEvolve, OpenEvolve, ThetaEvolve, TTT-Discover, Frontier-Eng, "Evaluation-driven Scaling for
Scientific Discovery," plus FrontierCS, ALE-Bench, and MLS-Bench.** A 10-agent research workflow mapped
each framework's tasks and, critically, each task's *evaluation form*, keeping only deterministic ones.

**Hard constraint — deterministic scoring only.** No wall-time, no GPU latency, no sandbox-dependent
scoring — none are reproducible offline and all are gameable. Kernels are included only reframed as
**FLOPs / operation-count**. Explicitly excluded: GPU kernels, interactive/reactive tasks, heavy
simulators, GPU-trained end-metrics, security fuzzing, RL/robotics returns, LLM-judge/"beats-SOTA."

---

## 2. Five problem formats (all deterministic, one scoring contract)

| Fmt | Shape | Files | Scored by | Sources |
|---|---|---|---|---|
| **A** | testlib instance-based combinatorial optimization | `statement.txt, gen.cpp, chk.cc, config.yaml` | C++ testlib checker → `Ratio:` | FrontierCS, ALE-Bench |
| **B** | evolve-a-heuristic vs a frozen evaluator | `statement.md, evaluator.py, config.yaml` | `evaluator.py` runs candidate over seeded instances → `Ratio:`+`Vector:` | FunSearch, AlphaEvolve, OpenEvolve, ThetaEvolve, TTT, Frontier-Eng, MLS-Bench |
| **C** | constructive artifact + verifier | `statement.md, gen.py, verify.py, config.yaml` | Python `verify.py` (exact/geometric) → `Ratio:` | AlphaEvolve, OpenEvolve, ThetaEvolve, FunSearch |
| **D** | FLOPs / op-count (kernel surrogate) | `statement.md, gen.py, counter.py, config.yaml` | exact-equivalence gate + op count → `Ratio:` | AlphaEvolve |
| **E** | symbolic / scientific-law discovery, held-out split | `statement.md, gen.py, verify.py, config.yaml` | held-out extrapolation error + complexity → `Ratio:` | FrontierCS, OpenEvolve, MLS-Bench, SimpleTES |

All five share one contract: the checker prints `Ratio: <float ∈ [0,1]>`; convention **trivial ≈ 0.1**,
a 10×-better solution caps at 1.0. Every problem ships a 4-rung **solution ladder**
(`trivial / greedy / strong / invalid`) that the harness uses to certify quality.

---

## 3. The 1006-problem corpus

| by format | count | | by tier (band) | count |
|---|---|---|---|---|
| A testlib combinatorial | 184 | | S graph/combinatorial core | 160 |
| B evolve-a-heuristic | 126 | | A math-discovery / heuristic | 140 |
| C constructive + verifier | 620 | | G breadth-fill and bulk domains/tasks | 588 |
| D FLOPs / op-count kernel | 41 | | B engineering + science | 60 |
| E symbolic / scientific-law | 35 | | C ML-method + exotic | 40 |
| | | | N bespoke-novelty | 18 |

- **Scoring types:** quality-metric 923 · flops 48 · correctness 35.
- **Source tags represented:** the original cross-framework taxonomy, bespoke novelty additions,
  breadth-fill supplements, and the new bulk Format-C constructive-selection tranche.
- **638 distinct families and 1006 unique `(family, theme, variant)` scaffolds** — including hard-science
  E-format domains, op-count D-format kernels, isolated B-format heuristic evaluators, and 500 new
  budget/conflict/coverage/diversity constructive domains.

The distribution is deliberately expanded toward the generalization-relevant constructive tail: the
latest bulk tranche adds 500 budget/conflict/coverage/diversity Format-C tasks across distinct domains.

---

## 4. Quality: the deterministic harness

FrontierSmith validates its test/checker infra with **agents checking each other** — which can converge
on a checker that is plausible but wrong. We replace that with mechanical, execution-grounded gates
(ground truth). Two harnesses share the same gate suite:

- `harness/validate_problem.py` — **stdout mode** (Formats A, C, D, E).
- `harness/validate_pyproblem.py` — **program mode** (Format B).

A problem PASSES only if all applicable gates hold:

| gate | checks |
|---|---|
| **G1** | gen + checker + every solution compile / import |
| **G2** | every test case generates, non-empty, within size budget |
| **G3 / G3b** | every `Ratio` ∈ [0,1]; program-mode vectors ∈ [0,1], correct length, ratio a valid aggregate |
| **G4** | determinism — the checker reproduces every score on a re-run |
| **G5** | feasibility — an *invalid* (infeasible) solution scores ~0 ⇒ the checker really validates |
| **G5b** | adversarial feasibility — empty / garbage / huge / **nan / inf** / injected-`Ratio` outputs all score ~0 |
| **G5c** | **isolation** — a candidate cannot reach the judge (see §5) |
| **G6** | baseline calibration — trivial ∈ [0.03, 0.35] (not already perfect, not degenerate) |
| **G7** | discrimination — strong − trivial ≥ margin (a better strategy exists) |
| **G8** | execution-grounded divergence — per-test score vectors genuinely differ ⇒ multiple strategies |

Each problem carries its own `validation.json`. Self-tests live at `harness/_selftest{,_C,_B}`.

**This is the single biggest robustness upgrade over the paper.** The gates mechanically prove the
checker rejects infeasible output, calibrates a baseline, rewards quality, and admits diverse strategies
— an auditable correctness signal the LLM-only approach lacks.

---

## 5. Security: OS-level isolation against reward hacking

The candidate is untrusted, possibly-adversarial model output. Three rounds of adversarial review
(GPT-5 "Codex" + independent reproduction) showed that an in-process or plain-subprocess candidate can:

- walk the Python stack (`sys._getframe().f_back`) to steal the evaluator's hidden oracle/answer,
- read `/proc/<judge>/mem` and `/proc/<judge>/cmdline`,
- read the co-located judge **source** (`gen.py` / labels / hidden laws) off the filesystem and
  regenerate the answer.

All were reproduced (e.g. an ML problem's candidate re-derived hidden labels → `Ratio: 1.0`).

**Fix — bubblewrap sandbox.** `harness/isorun.py` (Format B) and `sandbox_run_solution` (stdout modes)
run each candidate under `bwrap` in fresh user/pid/net/ipc/uts/mount namespaces, with the entire
problem tree `--tmpfs`-hidden and a private `/proc`. The candidate talks to the judge only through the
text protocol (public instance in → answer out); the hidden answer never leaves the parent. Verified:
a sandboxed candidate sees `nproc`=2 (parent invisible) and cannot read the source tree. **Gate G5c**
enforces it and fails any environment without `bwrap`.

---

## 6. Blind comparison vs FrontierSmith's 10 demos

We ran two **blind** panels (identical files, shuffled, neutral names, secret mapping) where an impartial
judge scored problems on merit without knowing origin.

- **Panel 1 (my 10 Format-A vs their 10, unweighted).** Verdict: *"ONE consistent, high-quality set, not
  two tiers"* — indistinguishable on rigor. Means: mine 7.3, theirs 7.8 (FrontierSmith has a few curated
  gems; my sample skewed to lower-novelty classics).
- **Panel 2 (my 16 bespoke-novelty vs their 10, novelty-weighted).** After upgrading the Format-A brief
  with a *novelty recipe* (compose 2-3 mechanisms + a mathematical twist + an adversarial generator that
  fills the constraint envelope):

  | | novelty | overall | flagged broken |
  |---|---|---|---|
  | **mine (16 bespoke)** | **7.56** | **7.75** | **0** |
  | FrontierSmith (10) | 6.70 | 6.20 | 5 |

  Every problem the judge flagged as broken/weak (under-scaled generators n≤50, degenerate tests) was
  **FrontierSmith's**; none of mine. The bespoke batch now leads on novelty and uniformity.

---

## 7. Layout & reproduction

```
synth/
  DESIGN.md / README.md / REPORT.md    method, usage, this report
  AGENT_BRIEF*.md                      authoring contracts (A / C·D·E / B) w/ anti-cheat + novelty rules
  harness/
    validate_problem.py                8-gate harness (stdout mode: A/C/D/E)
    validate_pyproblem.py              8-gate harness (program mode: B)
    isorun.py                          bwrap-sandboxed candidate runner
    testlib.h  _selftest{,_C,_B}
  seeds/build_seed_list.py             taxonomy/supplements → seed_list.jsonl (`--current` = 1006 specs)
  reports/
    taxonomy_proposal.json             researched cross-framework taxonomy
    verify_all.sh  scan_defects.py  aggregate.py
    blind*_MAPPING_secret.json         blind-comparison results
  generate_problems.workflow.js        fan-out: 1 agent/problem, author → self-validate → repair
  research_frameworks.workflow.js      the 10-framework research + synthesis workflow
  problems/<id>/                       the 1006 problems (testdata/ regenerates via the harness)
```

```bash
cd synth
python3 seeds/build_seed_list.py --current        # regenerate the current 1006-spec seed plan
bash   reports/verify_all.sh                      # ground-truth re-verify every problem (needs bwrap)
python3 reports/aggregate.py                       # → summary.{json,md}
# generation is driven by the Workflow tool over compact {id,format} routes.
```

`testdata/` is intentionally not committed (it regenerates deterministically from each `gen`); run the
harness with `--keep-testdata` to materialize it. `bwrap` (bubblewrap) must be on PATH for Format-B /
sandboxed verification.

---

## 8. Known limitations & next steps

1. **Envelope-fill gate (not yet built).** The harness checks test size ≤ cap but not ≥ a fraction of
   the *stated* constraint envelope, so an under-scaled generator can still PASS. This is exactly what
   sank FrontierSmith's weakest demos in the blind panel; adding it + a re-sweep is the top next step.
2. **~13 STRONG_CAPS problems** saturate the 10× score cap on the hardest cases (low training headroom,
   but sound — the 0.1→1.0 gradient is intact).
3. **Novelty is strong but not uniform** — a few breadth-fill problems remain single-mechanism classics.
4. **Isolation** relies on `bwrap` + unprivileged user namespaces (the login-node-available approximation);
   true multi-tenant hardening would want container/UID separation at the eval-infra layer.
