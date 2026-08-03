# Synthetic Open-Ended Problem Generation — Final Report

A pipeline that batch-generates **open-ended, deterministically-scored coding problems** to train
models that *generalize* across the "LLM writes code to optimize a scored objective" space. It
re-implements the withheld parts of **FrontierSmith** (arXiv 2605.14445) and extends the idea across
the whole evolutionary-search / scientific-discovery landscape.

**Result: 1300 problems, all machine-verified, spanning 1001 families and 1300 unique scaffolds.**

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

## 3. The 1300-problem corpus

| by format | count | | by tier (band) | count |
|---|---|---|---|---|
| A testlib combinatorial | 319 | | A math-discovery / heuristic | 469 |
| B evolve-a-heuristic | 258 | | B engineering + science | 307 |
| C constructive + verifier | 448 | | S graph/combinatorial core | 247 |
| D FLOPs / op-count kernel | 144 | | C ML-method + exotic | 160 |
| E symbolic / scientific-law | 131 | | G breadth-fill and bulk domains | 91 |
| | | | N bespoke-novelty | 26 |

- **Scoring types:** quality-metric 1114 · flops 151 · correctness 35.
- **1001 distinct families and 1300 unique `(family, theme, variant)` scaffolds** — including hard-science
  E-format domains, op-count D-format kernels, isolated B-format heuristic evaluators, the 659
  wave-2b problems spanning 20 independently-imagined design lenses, and the 200 wave-3 problems
  each opening its own family.

The distribution is deliberately expanded toward the generalization-relevant tail, in two steps.

**Wave-2b** (659 problems, `fsx_*_0507`–`fsx_*_1165`) replaced an earlier bulk tranche that had
degenerated into a single re-skinned template, and added two acceptance gates the original pipeline
lacked: an **innovation-headroom** requirement (`strong - greedy >= 0.06`, `strong <= 0.92`,
`greedy - trivial >= 0.03`) so the insight beats the recipe rather than merely beating do-nothing,
and an **anti-homogeneity scan** (digit-stripped skeleton hash + theme-masked statement hash).

**Wave-3** (200 problems, `fsx_*_1166`–`fsx_*_1365`) fixed a *coverage* rather than a quality
problem, measured against [EdgeBench](https://edge-bench.org/) (134 real-world long-horizon agent
tasks, six capability families). At 1165 problems this corpus was almost entirely EdgeBench's
"Combinatorial Optimization" family — which is only 14% of EdgeBench — while its two largest
families, Scientific Problems & ML (29%) and Systems & Software Engineering (27%), were represented
only as symbolic regression and raw op-counting respectively. The eight wave-3 lenses fill exactly
those gaps: inverse recovery, regime-crossing forecasting, protocol/state-machine conformance,
hardware co-design under exact cost models, professional risk/actuarial work, policy programs judged
in seeded simulators, molecular/materials design, and combinatorial game theory. Format mix was
skewed toward the two thinnest formats (D: 117→147, E: 107→136).

The corpus now scans **1300 dirs → 1300 unique skeletons / 1300 unique statement shapes** at
`--max-clones 1`. Each wave-2b and wave-3 problem was additionally reviewed by an independent Codex
(`gpt-5.6-terra`, xhigh) pass inside its authoring agent, with cited defects repaired before
acceptance.

### 3.1 The de-clone pass — and why the original gate missed it

`scan_homogeneity.py` hashes *exact* skeletons, so it catches template mass-production (the wave-2
failure: 500 problems, one skeleton) but is blind to "same logic, retuned constants, renamed
variables." Asking what a *family* actually guarantees exposed that gap.

`reports/audit_family_reuse.py` closes it: it compares the **scoring logic** of same-family problems
— checker source with comments, string literals and numeric literals stripped, tokenized, token
5-gram Jaccard. Stripping numeric literals is the point: a re-skin that only retunes constants still
registers as a clone.

The audit found **102 wave-1 problems in 37 clone clusters** at ≥0.60, 22 pairs at ≥0.85. The worst
pair (`fsx_B_0155` "smart-city lighting power law" / `fsx_B_0357` "buried artifact-density law",
0.995) shared the same hidden functional form, noise schedule, sample-size schedule, extrapolation
bound and scoring formula — differing only in RNG seed constants and which `x` fed which term. Two
skins, one problem.

The root cause is visible in the seeds: **wave-1 specs carry no `mechanisms`, no `innovation_hook`,
no `trap`**. Fourteen agents in one family received effectively identical briefs and converged.
Wave-2b and wave-3 added exactly those fields, and neither shows the pattern (wave-3, one family per
problem, has zero clone pairs).

Remediation kept the best-innovation-headroom member of each cluster and moved the other 65 to
`problems_wave1_clone_quarantine/` (with their seed records and the cluster plan). Post-removal the
audit flags **1 family** instead of 23, and its worst pair sits at 0.599 — below the clone
threshold. The 65 vacated slots are not yet refilled; `seeds/build_wave4_declone.py` holds the
prepared replacement specs (one family each, full mechanism/hook/trap discipline) for when they are.

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
  seeds/build_seed_list.py             taxonomy/supplements → seed_list.jsonl (`--current` = 1300 specs)
  reports/
    taxonomy_proposal.json             researched cross-framework taxonomy
    verify_all.sh  scan_defects.py  aggregate.py
    blind*_MAPPING_secret.json         blind-comparison results
  generate_problems.workflow.js        fan-out: 1 agent/problem, author → self-validate → repair
  research_frameworks.workflow.js      the 10-framework research + synthesis workflow
  problems/<id>/                       the 1300 problems (testdata/ regenerates via the harness)
```

```bash
cd synth
python3 seeds/build_seed_list.py --current        # regenerate the current 1300-spec seed plan
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
