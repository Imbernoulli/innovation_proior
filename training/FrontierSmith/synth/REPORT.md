# Synthetic Open-Ended Problem Generation — Final Report

A pipeline that batch-generates **open-ended, deterministically-scored coding problems** to train
models that *generalize* across the "LLM writes code to optimize a scored objective" space. It
re-implements the withheld parts of **FrontierSmith** (arXiv 2605.14445) and extends the idea across
the whole evolutionary-search / scientific-discovery landscape.

**Result: 500 problems, all machine-verified, spanning 12 source frameworks, 136 task-types, 134 domains.**

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

## 3. The 500-problem corpus

| by format | count | | by tier (band) | count |
|---|---|---|---|---|
| A testlib combinatorial | 182 (36%) | | S graph/combinatorial core | 160 |
| B evolve-a-heuristic | 125 | | A math-discovery / heuristic | 140 |
| C constructive + verifier | 119 | | G breadth-fill (domains/tasks) | 84 |
| D FLOPs / op-count kernel | 40 | | B engineering + science | 60 |
| E symbolic / scientific-law | 34 | | C ML-method + exotic | 40 |
| | | | N bespoke-novelty | 16 |

- **Scoring types:** quality-metric 418 · flops 47 · correctness 35.
- **Datasets/frameworks represented:** Frontier-CS 167 · AlphaEvolve 104 · ALE-Bench 95 · Frontier-Eng
  92 · SimpleTES 71 · MLS-Bench 67 · OpenEvolve 63 · ThetaEvolve 42 · TTT-Discover 42 · FunSearch 41 ·
  FrontierSmith 16.
- **136 distinct task-types, 134 distinct domains** — incl. 20 hard-science domains for the E format
  (astrophysics, chemistry, systems biology, economics, ML-scaling laws, materials, epidemiology, fluid
  dynamics, thermodynamics, pharmacology, quantitative finance, …) and op-count kernels for D (matmul
  tensor rank, addition chains, sorting networks, XOR circuits, reversible circuits, QAOA SWAP count, …).

The distribution is deliberately flattened toward the generalization-relevant tail: combinatorial-A was
cut from 42%→36% and the reproducible-kernel + scientific-discovery share (D+E) raised 9%→14%.

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
  seeds/build_seed_list.py             taxonomy → seed_list.jsonl (500 specs; --batch/--per-tier)
  reports/
    taxonomy_proposal.json             researched cross-framework taxonomy
    verify_all.sh  scan_defects.py  aggregate.py
    blind*_MAPPING_secret.json         blind-comparison results
  generate_problems.workflow.js        fan-out: 1 agent/problem, author → self-validate → repair
  research_frameworks.workflow.js      the 10-framework research + synthesis workflow
  problems/<id>/                       the 500 problems (testdata/ regenerates via the harness)
```

```bash
cd synth
python3 seeds/build_seed_list.py                 # regenerate the seed plan
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
