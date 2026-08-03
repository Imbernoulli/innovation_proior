# Deterministic Open-Ended Problem Synthesis — Design

A batch problem-generation system for **open-ended, deterministically-scored** coding problems, to
train models that **generalize** across the "LLM writes code to optimize a scored objective" space.
It re-implements FrontierSmith's withheld parts and **broadens** the scope to the whole
evolutionary-search / discovery landscape, under one hard constraint: **deterministic scoring only**.

## 1. Origin & scope

FrontierSmith (arXiv 2605.14445) mutates closed-ended CP problems into open-ended optimization
problems (change goal / restrict output / generalize input), filters by an idea-divergence metric,
and has agents synthesize + cross-validate test/checker infra. It open-sources 10 sample problems +
train/eval code; **the orchestrator and test/checker generators are withheld** — we re-implement them.

We then **broaden beyond FrontierCS** (per project direction) to draw archetypes from FunSearch,
AlphaEvolve, OpenEvolve, ThetaEvolve, TTT-Discover, Frontier-Eng, "Evaluation-driven Scaling for
Scientific Discovery", **plus FrontierCS, ALE-Bench, and MLS-Bench**. A 10-agent research workflow
(`research_frameworks.workflow.js` → `reports/taxonomy_proposal.json`) mapped each framework's tasks
and, critically, each task's **evaluation form**, keeping only deterministic ones.

**Hard constraint:** no wall-time, no GPU latency, no sandbox-dependent scoring — those are not
reproducible here and are gameable. Kernels are included **only** reframed as **FLOPs / op-count**.
Excluded (research agents concurred): GPU kernels, interactive/reactive tasks, heavy simulators,
GPU-trained end-metrics, security fuzzing, RL/robotics returns, LLM-judge/"beats-SOTA" rewards.

## 2. Five problem formats (all deterministic)

| Fmt | Shape | Files | Scored by | Sources |
|---|---|---|---|---|
| **A** | testlib instance-based combinatorial opt | `statement.txt, gen.cpp, chk.cc, config.yaml` | C++ testlib checker → `Ratio:` | FrontierCS, ALE-Bench |
| **B** | evolve-a-heuristic vs a frozen evaluator | `statement.md, evaluator.py, config.yaml` | `evaluator.py` runs candidate over seeded instances → `Ratio:`+`Vector:` | FunSearch, AlphaEvolve, OpenEvolve, ThetaEvolve, TTT, Frontier-Eng, MLS-Bench |
| **C** | constructive artifact + verifier | `statement.md, gen.py, verify.py, config.yaml` | Python `verify.py` (exact/geom) → `Ratio:` | AlphaEvolve, OpenEvolve, ThetaEvolve, TTT |
| **D** | FLOPs / op-count (kernel surrogate) | `statement.md, gen.py, counter.py, config.yaml` | equivalence gate + op-count → `Ratio:` | AlphaEvolve |
| **E** | symbolic regression, held-out split | `statement.md, gen.py, verify.py, config.yaml` | held-out error + complexity → `Ratio:` | FrontierCS, OpenEvolve, MLS-Bench |

All five share ONE scoring contract: the checker prints `Ratio: <float∈[0,1]>`; convention **trivial
solution ≈ 0.1**, a 10×-better solution caps at 1.0. Every problem ships a 4-rung **solution ladder**
(`trivial / greedy / strong / invalid`) that the harness uses to certify quality.

## 3. The deterministic harness (our main upgrade over FrontierSmith)

FrontierSmith validates infra with **agents checking each other** — which can agree on a plausible-
but-wrong checker. We replace that with mechanical, execution-grounded gates (ground truth):

- `harness/validate_problem.py` — **stdout mode** (Formats A, C, D, E): compiles/gens/runs and scores.
- `harness/validate_pyproblem.py` — **program mode** (Format B): candidate run via `isorun`.

Gates (all must pass): **G1** compile/import · **G2** generate · **G3(+G3b)** bounds/vector-integrity ·
**G4** determinism (all solutions) · **G5** feasibility (invalid → ~0) · **G5b** adversarial (empty/
garbage/huge/**nan/inf**/inject → ~0) · **G5c isolation** (candidate can't reach the judge — see below) ·
**G6** baseline (trivial ∈ [0.03,0.35]) · **G7** discrimination · **G8** execution-grounded divergence.
Checker score = LAST `Ratio:` only, Python checkers must exit 0. Each problem carries `validation.json`;
self-tests `harness/_selftest{,_C,_B}`.

**OS sandbox (reward-hack defense; added after adversarial review).** Untrusted candidates are the
model's own code, so they run under **bubblewrap**: `isorun.py` (Format B) and `sandbox_run_solution`
(stdout modes) execute each candidate in fresh user/pid/net/ipc/uts/mount namespaces with the whole
`synth/` tree `--tmpfs`-hidden and a private `/proc`. This structurally blocks the exploits the audit
reproduced: Python frame-walk (`sys._getframe().f_back`), `/proc/<judge>/mem`+cmdline reads, and reading
the co-located ground-truth source (gen/labels/laws) to regenerate the answer. Candidates talk to the
judge only through the text protocol (public instance in, answer out); the hidden answer never leaves the
parent. `G5c` enforces it (fails any evaluator that doesn't sandbox, or any environment without bwrap).

## 4. The tiered plan (档) and current corpus

The researched base taxonomy (`reports/taxonomy_proposal.json`) is importance-ranked and still
regenerates a controlled **200-problem** batch by default. The checked-in corpus is the broader
**1300-problem** plan assembled with `build_seed_list.py --current`, built in three waves:

- **wave-1 (506, `fsx_*_0001`–`fsx_*_0506`)**: taxonomy batch 1 (200) + taxonomy batch 2 (200) +
  16 bespoke-novelty problems + 84 breadth-fill problems that target thin scientific-law, op-count,
  ML-method, discrete-construction, and domain-specific task families + 6 subagent extensions.
- **wave-2b (659, `fsx_*_0507`–`fsx_*_1165`)**: a from-scratch rebuild of the original bulk tranche,
  which had collapsed into a single re-skinned template (one skeleton across all 500). Seeds were
  re-imagined across 20 distinct lenses (`seeds/bulk_seed_packs/`), each carrying an explicit
  `mechanisms` / `innovation_hook` / `trap` triple, and each problem was authored by its own agent
  under two extra acceptance gates (§4.1).
- **wave-3 (200, `fsx_*_1166`–`fsx_*_1365`)**: a *domain*-coverage expansion rather than a depth
  one, benchmarked against EdgeBench's six capability families (§4.2). Eight lenses x 25
  (`seeds/build_wave3_edgebench.py`), each problem opening its own family.

Every spec is a unique `(family x theme x variant)` or supplement scaffold — 1300/1300 distinct.

A later **de-clone pass** (§4.3) removed 65 wave-1 problems whose scoring logic duplicated a
same-family sibling, taking the corpus from 1365 to 1300.

| Tier | 档 | Focus | Families | Count | Formats |
|---|---|---|---|---|---|
| **S** | 核心 | graph & combinatorial optimization | 12 | **80** | A |
| **A** | 重要 | math-discovery / heuristic evolution | 10 | **70** | C, B, D |
| **B** | 应用前沿 | engineering + scientific optimization | 8 | **30** | B, D, E |
| **C** | 方法与异域前沿 | ML-method design + exotic construction | 6 | **20** | B, C |

Current 1300-problem mix:

| Group | Count | Role |
|---|---:|---|
| A | 487 | math-discovery / heuristic evolution |
| B | 322 | engineering + scientific optimization |
| S | 276 | graph/combinatorial core |
| C | 163 | ML-method design + exotic construction |
| G | 91 | breadth-fill plus bulk constructive-selection domains |
| N | 26 | bespoke high-novelty, composite/mechanism-twist problems |

Format mix over the 1300: A=319, B=258, C=448, D=144, E=131. 1001 distinct families.

### 4.1 Wave-2b's two extra acceptance gates

1. **Innovation headroom** (`AGENT_BRIEF_INNOVATION_ADDENDUM.md`). The original G7 only checked
   `strong > trivial`, which a problem can satisfy while the textbook greedy recipe already captures
   nearly all of the value — a useless RL signal if the goal is *innovation* rather than recall of a
   known baseline. Wave-2b additionally requires `strong - greedy >= 0.06` (the insight visibly beats
   the recipe), `strong <= 0.92` (headroom is left above the reference solution), and
   `greedy - trivial >= 0.03` (the ladder is sane). The generator must plant trap cases where the
   obvious greedy lands far from strong on >=3 of the 10 tests.
2. **Anti-homogeneity** (`reports/scan_homogeneity.py`). A digit-stripped skeleton hash plus a
   theme-masked statement hash catch re-skinned clones that per-problem validation is blind to — the
   failure mode that killed the first wave-2 attempt. The current corpus scans **1300 dirs → 1300
   unique skeletons / 1300 unique statement shapes** at `--max-clones 1`.

Each wave-2b authoring agent additionally ran an independent **Codex (`gpt-5.6-terra`, xhigh) review**
of its own finished problem — hunting scoring loopholes, nondeterminism, statement/code mismatches,
trap cases that fail to punish greedy, and strong-solutions that are merely greedy-plus-tuning — and
repaired every real defect before the problem was accepted. Wave-3 inherits both gates and the review.

### 4.2 Wave-3: closing the domain gap measured against EdgeBench

Wave-2b fixed a *quality* failure (re-skinned clones). Wave-3 fixes a *coverage* failure that only
became visible when the corpus was measured against an external yardstick:
[EdgeBench](https://edge-bench.org/) (ByteDance Seed), 134 real-world ultra-long-horizon agent tasks
grouped into six capability families. At 1165 problems our coverage was:

| EdgeBench family | EdgeBench share | our corpus at 1165 |
|---|---:|---|
| Combinatorial Optimization | 14% | effectively all 1165 |
| Scientific Problems & ML | 29% | E=107, nearly all symbolic regression |
| Systems & Software Engineering | 27% | D=117, pure op-counting |
| Professional Knowledge Work | 14% | abstract auction/budget mechanics only |
| Formal Math & Theorem Proving | 10% | constructive extremal, no game theory |
| Interactive Games & Simulators | 6% | ~21 problems |

That is a corpus over-fitted to one capability family. The eight wave-3 lenses (25 each) target the
gaps directly, while keeping the deterministic-scoring constraint intact (forward models and
simulators are seeded and turn-capped; "hidden" data is regenerated by the checker from the public
`testId`):

1. **inverse-recovery** — seeded forward model, sparse noisy observations, recover the hidden source.
   The recurring trap: best-data-fit ≠ best recovery, because the forward operator has a null space.
2. **forecast-regime** — the held-out horizon crosses into a regime the visible window never showed
   (an inverter clipping ceiling, an aging knee, a hysteresis branch, a bifurcation onset).
3. **protocol-conformance** — state machines, codecs, and resolvers judged on hidden trace suites.
4. **hardware-codesign** — exact cycle/area/energy cost models (pushes format D).
5. **risk-actuarial** — tail-risk budgeting, fraud rings, adverse selection, compliance paths.
6. **policy-simulator** — format-B policy programs run inside deterministic seeded simulators.
7. **molecular-materials** — previously our thinnest domain (36 problems).
8. **games-and-structures** — Sprague-Grundy, pairing strategies, certificates, misère play.

Unlike wave-1 (which deliberately reuses a family across themes and variants — 36 families cover 400
problems there), **every wave-3 problem opens its own family**, taking the corpus from 801 to 1001
families. Format mix was skewed toward the two thinnest formats: D 117→147, E 107→136.

### 4.3 The de-clone pass: what a "family" does and does not guarantee

A `family` labels a problem's mechanism archetype. Wave-1 deliberately reuses each family across
themes and variants — 36 families cover 400 of its 506 problems — on the assumption that a shared
archetype still yields distinct problems. Auditing that assumption showed it does not hold without
the mechanism/hook/trap fields.

`scan_homogeneity.py` could not detect the failure: it hashes *exact* skeletons, so it catches
template mass-production but is blind to "same logic, retuned constants, renamed variables."
`reports/audit_family_reuse.py` compares the **scoring logic** instead — checker source with
comments, string literals and numeric literals stripped, tokenized, token 5-gram Jaccard. Stripping
numeric literals is deliberate: a re-skin that only retunes constants must still register.

Result: **102 wave-1 problems in 37 clone clusters** at ≥0.60, 22 pairs at ≥0.85. The worst pair
(`fsx_B_0155` / `fsx_B_0357`, 0.995) shared functional form, noise schedule, sample-size schedule,
extrapolation bound and scoring formula, differing only in RNG seed constants and index permutation.

The cause is structural, not incidental: wave-1 seeds have `mechanisms: None` and no
`innovation_hook` or `trap`, so a family's fourteen agents received effectively identical briefs.
Wave-2b and wave-3, which carry those fields, show none of it — wave-3 (one family per problem) has
zero clone pairs.

We kept each cluster's best-innovation-headroom member and quarantined the other 65 into
`problems_wave1_clone_quarantine/` alongside their seed records and the cluster plan. The audit now
flags 1 family rather than 23, worst pair 0.599. Replacement specs under full wave-3 discipline are
prepared in `seeds/build_wave4_declone.py` but not yet authored.

**Standing gate.** `audit_family_reuse.py` joins `scan_homogeneity.py` as a release check: exact
skeleton hashing for template reuse, 5-gram checker similarity for logic reuse. Neither alone is
sufficient.

## 5. Critical analysis — improvements over FrontierSmith (辩证)

1. **LLM-only validation → execution grounding.** *(built)* 8 mechanical gates certify the checker
   rejects infeasible output, calibrates a baseline, rewards quality, and admits diverse strategies.
2. **Random HardTests seed → deliberate, importance-tiered, cross-framework coverage.** *(built)*
3. **Scope beyond one benchmark → 5 deterministic formats spanning 10 frameworks.** *(built)* Trains
   *generalization*, explicitly not overfitting any single dataset.
4. **Kernel/wall-time work → FLOPs/op-count surrogate (Format D).** *(built)* Reproducible offline.
5. **Divergence via n=10 LLM-judged samples → execution divergence from a designed ladder.** *(built,
   cheaper)* Trade-off: less exploratory; a hybrid (ladder gate + a few free samples) is future work.
6. **"No known optimum" asserted → necessary conditions enforced** (G6 trivial-doesn't-win + G7 a-
   better-strategy-exists). Full intractability is undecidable; mutating from NP-hard targets biases hard.
7. **Roadmap (not yet built):** statement↔checker audit agent; anti-cheese adversarial gate;
   cross-problem embedding dedup; auto-hardening test distributions; per-format difficulty via a
   strong-LLM headroom probe.

## 6. Reproduce

```bash
cd frontiersmith_synth
python3 seeds/build_seed_list.py --current            # -> seeds/seed_list.jsonl (current 1300)
python3 harness/validate_problem.py   harness/_selftest      # A/C/D/E self-check
python3 harness/validate_problem.py   harness/_selftest_C
python3 harness/validate_pyproblem.py harness/_selftest_B    # B self-check
# batch generation via the Workflow tool over compact {id,format} routes:
#   Workflow(scriptPath=generate_problems.workflow.js, args=[{id,format},...])
# authoritative re-verification (per format) + report:
python3 reports/aggregate.py                          # -> reports/summary.{json,md}
```
