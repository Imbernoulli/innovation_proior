# Problem-Generation Agent Brief (FrontierSmith-style open-ended optimization)

You author **one** complete, novel open-ended optimization problem from a spec, then
**self-validate it with the deterministic harness until it PASSES**. Your final message is
consumed by a program — return the requested JSON only.

## What "open-ended optimization problem" means (must hold)
1. **Graded objective, no known optimum.** Any feasible output is accepted and *scored*; the
   intended optimum is intractable, so heuristics are expected. NOT a decision / exact-answer /
   single-correct-output problem.
2. **Multiple distinct strategies plausible** (greedy vs. local search vs. DP-relaxation vs. flow
   rounding, …) — this is what makes solutions *diverge*.
3. **A scoring function meaningfully ranks submissions** (better construction ⇒ strictly higher score).

Apply the spec's mutation to the spec's seed problem:
- **goal** — replace a decision/exact goal with an optimization goal (e.g. 2-SAT → Min-True-2-SAT).
- **output** — keep the goal, add/tighten output constraints so optimal construction is NP-hard
  (e.g. MST → degree-constrained spanning tree).
- **input** — relax input structure so an easy problem gets hard (e.g. max-indep-set bipartite → general).
- **composite** — change the goal AND (restrict output OR generalize input).

Do **not** re-create any of the 10 open-sourced problems (Scorched Bridges, Farmwide Teleport Pad,
Metallic Pink Resonator, Park Ranger Shift Balancing, Prime Resonance Retuning, Mobile Relay Layout,
Archipelago Relay Network, Resonant Bay Layout, Duff's Defensive Lineup, Quadratic Witness Packing).

## Files you must write into the problem directory `<probdir>`
```
statement.txt        markdown; sections: Title, Problem, Input, Output, Feasibility, Objective,
                     Scoring, Constraints (incl. Time limit + Memory limit), Example (with worked score)
gen.cpp              testlib generator (see contract)
chk.cc               testlib checker/scorer (see contract)
config.yaml          checker/memory/time/subtasks (see contract)
meta.json            {id, tier, family, seed_problem, mutation, objective, theme, title, strategies:[...]}
solutions/trivial.cpp   // TIER: trivial   -- naive baseline (scores the calibration point)
solutions/greedy.cpp    // TIER: greedy    -- simple heuristic, beats trivial a bit
solutions/strong.cpp    // TIER: strong    -- better heuristic / local search, beats greedy
solutions/invalid.cpp   // TIER: invalid   -- deliberately INFEASIBLE output (must score 0)
```
The first line of every solution file **must** be the `// TIER: <name>` comment exactly.

## config.yaml contract (copy this, adjust time/memory)
```yaml
checker: chk.cc
memory: 512m         # 256m..1024m
subtasks:
- n_cases: 10
  score: 100
time: 5s             # 2s..8s
type: default
```

## gen.cpp contract
- `#include "testlib.h"`, `registerGen(argc, argv, 1);`, read `int testId = atoi(argv[1]);`.
- The harness runs `./gen 1`, `./gen 2`, …, `./gen 10`. Print **one** test case to stdout per call.
- **testId is a difficulty/structure ladder**: testId 1 tiny (for the example-scale sanity), growing to
  large/adversarial by testId 10 (sparse vs dense, uniform vs skewed, worst-case structure). Use
  `rnd` (testlib RNG) only — deterministic given testId. Each `.in` must be **≤ ~20 MB** and generate in < 40 s.
- Output must exactly match the statement's Input format.

## chk.cc contract  (THIS IS WHERE MOST PROBLEMS FAIL — read carefully)
- `#include "testlib.h"`, `registerTestlibCmd(argc, argv);`. Read the test input from `inf`,
  the participant output from `ouf`. (Ignore `ans`; scorer problems have empty answer files.)
- **Validate feasibility strictly** using bounded reads (`ouf.readInt(lo, hi, "name")`) and explicit
  checks; on ANY violation call `quitf(_wa, "reason ...")`. End with `if(!ouf.seekEof()) quitf(_wa,"trailing");`
  (An infeasible/garbage output MUST score 0 — the harness feeds it a deliberately-invalid solution.)
- Compute the objective `F` (min) or `F` (max) of the participant's feasible output.
- Compute an **internal baseline `B`** = the objective of a *trivial feasible* construction the checker
  builds itself (do-nothing for min; best-single-unit / any-feasible for max). `B` must be **positive**.
- **Emit the score exactly in this shape** (the judge greps `Ratio: <float>`):
  - minimization: `double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));`
  - maximization: `double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));`
  - then: `quitp(sc/1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc/1000.0);`
- Result of this convention (REQUIRED for the harness to pass):
  - the **trivial** solution scores ratio ≈ **0.1** (feasible but no better than baseline),
  - **strong** scores strictly higher than trivial (up to the 1.0 cap = 10× baseline),
  - **invalid** scores 0.
- The checker must be **deterministic** and O(input) fast (it runs on 20 MB inputs).
- If you read floating-point participant values, **reject non-finite** (`nan`/`inf`) before scoring
  (`if (!isfinite(x)) quitf(_wa, ...)`). The harness feeds a nan/inf-flooded output and requires it to
  score 0 (integer-only checkers via `ouf.readInt` already reject this).

## NOVELTY — raise the ceiling (this separates a 9 from a 7)
The harness already enforces rigor; what makes a TOP problem is a *bespoke* design, not a textbook
NP-hard problem in a themed skin. A blind judge rated plain "max-cut / MWIS / GAP / covering /
interdiction in a new skin" ~7, and these ~9. To hit the top:
- **Compose 2–3 mechanisms** into one objective so the best strategy must BALANCE competing goals
  (e.g. coverage × budget × a temporal/scheduling dimension; a routing cost coupled to a structural
  side-constraint; selection coupled to an ordering). A single-mechanism objective a greedy nails is a 7.
- **Add a mathematical twist** where it fits: number-theoretic (primality / residues / gcd), spectral
  or FFT (convolution overlap, eigen/algebraic-connectivity, effective resistance), geometric
  (Euclidean/lattice placement), or algebraic (finite-field labels). This enriches the search space.
- **Non-linear / coupled scoring**: add saturation, interaction, or product terms so distinct
  strategies genuinely diverge — not a plain additive sum.
- If you must start from a classic core, MUTATE the objective non-obviously (a second coupled
  objective, a nonlinear term, a side-constraint that changes which strategy wins).
- **Adversarial generator (required for top test_quality):** beyond size scaling, `gen.cpp` MUST
  include (a) PLANTED cases (a good structure is hidden inside), (b) TRAP cases where the obvious
  greedy is far from optimal, (c) NEEDLE cases (one high-value structure amid noise), and (d) fill
  the stated constraint envelope on the largest tests (don't top out at tiny n — a real reference
  problem lost points for exactly this).

## Solution ladder (design for a real score spread)
- `trivial`: the exact baseline the checker measures → ratio ≈ 0.1.
- `greedy`: a one-pass heuristic → noticeably above trivial on most tests.
- `strong`: a better constructive/local-search heuristic → above greedy; different per-test behavior.
- `invalid`: prints an out-of-range / infeasible token on purpose → ratio 0.
- The three valid tiers must produce **different per-test score vectors** (this is the execution-grounded
  idea-divergence signal). If greedy==strong everywhere, redesign one of them or the scoring.

## Self-validation loop (do this before returning)
```
python3 <SYNTH>/harness/validate_problem.py <probdir> --keep-testdata
```
Read the printed JSON. If `verdict != PASS`, fix the failing gate and re-run. Common fixes:
- G1 compile → fix C++.
- G2 generate → shrink sizes / fix gen format / avoid TLE in gen.
- G5 feasibility (invalid scored >0) → tighten checker validation (bounded reads, seekEof, range checks).
- G6 baseline (trivial not ≈0.1) → fix the checker's internal `B` and the score formula; make sure the
  trivial solution really is the baseline construction.
- G7 discrimination (strong ≤ trivial) → make strong actually better, or the objective more sensitive.
- G8 divergence (solutions identical) → diversify greedy vs strong, or add structure the scoring rewards.
Iterate up to ~6 rounds. Keep the statement, checker, and scoring **mutually consistent** at all times.

## Return (JSON only)
```json
{"id":"<id>","verdict":"PASS|FAIL","title":"...","family":"...","mutation":"...",
 "metrics":{"trivial":0.1,"greedy":..,"strong":..,"divergence":..,"invalid":0.0},
 "rounds":<int>,"notes":"one line: the core idea + why it's open-ended (or why it failed)"}
```
