# synth/ — Deterministic Open-Ended Problem Synthesis

Batch-generated corpus of **open-ended, deterministically-scored** coding problems for training models
that **generalize** across the "LLM writes code to optimize a scored objective" space. Re-implements
FrontierSmith's withheld orchestrator + test/checker generators, broadened across 10 frameworks
(FunSearch, AlphaEvolve, OpenEvolve, ThetaEvolve, TTT-Discover, Frontier-Eng, Eval-driven-Discovery,
FrontierCS, ALE-Bench, MLS-Bench). **Deterministic scoring only** — no wall-time/GPU; kernels appear
only as FLOPs/op-count. See `DESIGN.md` for the full method + critical analysis.

Current corpus: **1006 generated problems**, **1006/1006 machine-verified PASS**, with unique IDs and a
one-to-one match between `seeds/seed_list.jsonl` and `problems/fsx_*`.

## Layout
```
DESIGN.md                      method, 5 formats, tiered plan, improvements over FrontierSmith
AGENT_BRIEF.md                 authoring contract for Format A (testlib C++)
AGENT_BRIEF_PY_STDOUT.md       authoring contract for Formats C / D / E (Python verifier, stdin→stdout)
AGENT_BRIEF_PY_PROGRAM.md      authoring contract for Format B (evolve-a-heuristic, frozen evaluator)
harness/
  validate_problem.py          8-gate deterministic harness — stdout mode (A, C, D, E)
  validate_pyproblem.py        8-gate deterministic harness — program mode (B)
  testlib.h                    testlib for compiling gen.cpp / chk.cc
  _selftest, _selftest_C, _selftest_B   regression fixtures (one per harness path)
seeds/
  build_seed_list.py           taxonomy/supplements → seed_list.jsonl (`--current` for corpus)
  seed_list.jsonl              the 1006 problem specs (tier/format/family/theme/scale/variant)
reports/
  taxonomy_proposal.json       researched cross-framework taxonomy (5 formats, 4 tiers, 36 families)
  verify_all.sh                re-verify every problem with the correct harness (ground truth)
  aggregate.py                 → summary.json / summary.md
generate_problems.workflow.js  fan-out workflow: 1 agent/problem, author → self-validate → repair
research_frameworks.workflow.js  the 10-framework research + synthesis workflow
problems/<id>/                 the generated problems
problems_legacy/               6 pilot-1 problems (superseded by the taxonomy IDs)
```

## The 5 formats (all deterministic, one `Ratio:` contract)
- **A** testlib C++ combinatorial optimization — `gen.cpp` + `chk.cc` (FrontierCS/ALE home).
- **B** evolve-a-heuristic vs a frozen `evaluator.py` over seeded instances (FunSearch/AlphaEvolve).
- **C** constructive artifact + Python `verify.py` (circle packing, Hadamard, cap-set, dispersion).
- **D** FLOPs/op-count with an exact-equivalence gate — the offline-safe kernel surrogate.
- **E** symbolic regression with a train/held-out split (anti-overfit).

Every problem carries a 4-rung solution ladder (`trivial/greedy/strong/invalid`) and its
`validation.json`. A problem PASSES only if all gates hold: compile · generate · bounds/vector ·
determinism · **feasibility** (invalid/nan/inf→0) · **isolation (G5c)** · **baseline** (trivial≈0.1) ·
**discrimination** (strong≫trivial) · **divergence** (strategies differ).

**Reward-hack defense (OS sandbox).** Untrusted candidates run under **bubblewrap** (`harness/isorun.py`
for Format B; `sandbox_run_solution` in `validate_problem.py` for A/C/D/E): fresh namespaces, the whole
`synth/` tree `--tmpfs`-hidden, private `/proc`. This blocks frame-walk, `/proc/<judge>/mem`, and reading
the co-located ground-truth source. **Requires `bwrap`** on PATH (falls back to unsandboxed, which G5c then
fails). The corpus survived three rounds of adversarial Codex review + independent confirmation.

## Regenerate / extend
```bash
python3 seeds/build_seed_list.py --current        # rebuild the current 1006-spec plan
bash   reports/verify_all.sh                      # ground-truth re-verify all problems
python3 reports/aggregate.py                       # → reports/summary.{json,md}
# batch generation is driven by the Workflow tool over compact {id,format} routes:
#   Workflow(scriptPath=generate_problems.workflow.js, args=[{"id":"fsx_S_0001","format":"A"}, ...])
```

Without `--current`, `build_seed_list.py` emits the original 200-spec taxonomy batch for controlled
experiments; `--full500` rebuilds the base corpus before the subagent and bulk extensions.

## Using the problems
- **Format A** plugs straight into the Frontier-CS algorithmic judge (same file layout: `statement.txt`,
  `gen.cpp`, `chk.cc`, `config.yaml`, `testdata/`).
- **Formats B/C/D/E** are self-contained Python: run the solution through `evaluator.py`/`verify.py`/
  `counter.py`; the score is `Ratio:` ∈ [0,1] (trivial≈0.1). No Docker/GPU/wall-time needed.
```
