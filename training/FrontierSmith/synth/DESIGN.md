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

## 4. The tiered plan (档) — 200 problems

Research-grounded (`reports/taxonomy_proposal.json`), importance-ranked. Default sums to 200;
`build_seed_list.py --per-tier 200` plans 200 per tier (800). 36 families; every spec is a unique
(family × theme × variant) scaffold with a scale sweep for generalization.

| Tier | 档 | Focus | Families | Count | Formats |
|---|---|---|---|---|---|
| **S** | 核心 | graph & combinatorial optimization | 12 | **80** | A |
| **A** | 重要 | math-discovery / heuristic evolution | 10 | **70** | C, B, D |
| **B** | 应用前沿 | engineering + scientific optimization | 8 | **30** | B, D, E |
| **C** | 方法与异域前沿 | ML-method design + exotic construction | 6 | **20** | B, C |

Format mix over the 200: A≈80, B≈49, C≈52, D≈11, E≈8.

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
cd FrontierSmith/synth
python3 seeds/build_seed_list.py                      # -> seeds/seed_list.jsonl (200; or --per-tier 200)
python3 harness/validate_problem.py   harness/_selftest      # A/C/D/E self-check
python3 harness/validate_problem.py   harness/_selftest_C
python3 harness/validate_pyproblem.py harness/_selftest_B    # B self-check
# batch generation via the Workflow tool over compact {id,format} routes:
#   Workflow(scriptPath=generate_problems.workflow.js, args=[{id,format},...])
# authoritative re-verification (per format) + report:
python3 reports/aggregate.py                          # -> reports/summary.{json,md}
```
