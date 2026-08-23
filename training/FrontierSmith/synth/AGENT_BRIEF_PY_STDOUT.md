# Problem-Generation Brief — Formats C / D / E (Python verifier, stdin→stdout)

You author ONE novel, **deterministically-scored** problem where the solution reads an instance
from stdin and writes an artifact to stdout, and a **Python checker** scores it. Self-validate with
the harness until PASS. Return the requested JSON only.

Hard rule: **deterministic scoring only** — no wall-time, no GPU, no randomness in the score (seed
everything). Exact integer/rational arithmetic where possible; geometry with a fixed 1e-6 tolerance.

## Files in `<probdir>`
```
statement.md          Title, Problem, Input(stdin), Output(stdout), Feasibility, Objective, Scoring, Constraints, Example(worked score)
gen.py                `python3 gen.py <testId>` prints ONE instance to stdout; testId 1..N = difficulty ladder (small→large/adversarial); seed via testId only
<checker>.py          the deterministic scorer (name it verify.py for C/E, counter.py for D); set `checker:` in config.yaml to this name
config.yaml           checker: <checker>.py  /  memory: 512m  /  subtasks:[{n_cases:10,score:100}]  /  time: 5s  /  type: default
meta.json             {id,tier,format,family,eval_form,theme,title,objective,strategies:[...]}
solutions/trivial.py  # TIER: trivial   -- the baseline construction (scores ~0.1)
solutions/greedy.py   # TIER: greedy
solutions/strong.py   # TIER: strong
solutions/invalid.py  # TIER: invalid   -- emits infeasible/garbage output (must score 0)
```
Solutions may be `.py` or `.cpp`; first line must be the `# TIER:`/`// TIER:` comment. Prefer `.py`.

## Checker contract (deterministic; prints the score)
- CLI: `python3 <checker>.py <in> <out> <ans>` (ans is an empty placeholder — ignore it).
- Read the instance from `<in>`, the participant artifact from `<out>`.
- **Validate feasibility strictly**; on ANY violation print `Ratio: 0.0` (+ reason) and exit 0.
- Compute the objective, then an **internal baseline `B`** = a trivial feasible construction the checker
  builds itself (positive). Normalize and print exactly (the harness greps `Ratio:`):
  - maximization: `sc = min(1000.0, 100.0*F/max(1e-9,B))`
  - minimization: `sc = min(1000.0, 100.0*B/max(1e-9,F))`
  - `print("... Ratio: %.6f" % (sc/1000.0))`   → trivial≈0.1, 10×-better caps at 1.0.
- Must be O(size) fast and **bit-for-bit deterministic** on reruns (G4).

## Format specifics
- **C (constructive artifact, eval_form=quality-metric/correctness)**: artifact IS the object
  (coords+radii / a ±1 matrix / an integer set / a point set / a step vector). Objective = its exact
  quality (sum of radii; |det| via **Bareiss** integer elimination; |A+A|/|A−A| via exact sumsets;
  min triangle area; star discrepancy; set cardinality under a validity checker). Feasibility gate is
  the validity check (non-overlap+containment, no-3-collinear, row/col Latin rules, …).
- **D (FLOPs / op-count, eval_form=flops)**: artifact = a decomposition / arithmetic circuit /
  straight-line program in a FIXED schema. Checker FIRST verifies **exact equivalence** to the target
  function/tensor (wrong → `Ratio: 0.0`), THEN counts scalar operations. `ratio = min(1, 0.1 *
  ops_baseline / ops_yours)` (fewer ops better). Integer/rational only; **never time anything**.
  This is the offline-safe kernel surrogate — cost = operation count, not latency.
- **E (symbolic regression, eval_form=quality-metric)**: `gen.py <testId>` prints the TRAIN sample
  (points from a hidden ground-truth formula). The solution emits a closed-form expression (a Python
  expression string over the stated variables/operators). The checker **regenerates the HELD-OUT /
  EXTRAPOLATION split deterministically** (same formula+seed baked into the checker), evaluates the
  submitted expression on it, and scores from held-out error + a complexity penalty. The held-out
  split is the anti-overfit mechanism — reward generalization, not memorization.

## Solution ladder → real score spread
`trivial` reproduces the checker's baseline (→ ~0.1); `greedy` beats it; `strong` beats greedy with a
different per-test behavior; `invalid` emits an out-of-range/garbage artifact (→ 0). The three valid
tiers must yield DIFFERENT per-test score vectors (execution divergence). Design accordingly.

## ANTI-CHEAT / HARDENING (mandatory — the harness now enforces these)
- **No reachable known optimum.** Do NOT choose sizes/parameters with a *known polynomial optimal
  construction*. Confirmed traps to AVOID: Hadamard/Paley orders for max-|det| (use ODD N so no
  Hadamard exists — Hadamard is only an *unreachable* normalizer); perfect-tiling sizes.
- **Format D — the tensor MUST have open rank.** Do NOT use the convolution tensor (rank = 2n−1 is a
  *proven* optimum via Toom-Cook) or small matmul tensors (Strassen/Laderman are known). Use a general
  3D tensor with a PLANTED OVERCOMPLETE rank (rank > max dimension) so polynomial methods
  (Jennrich/simultaneous-diagonalization, which need rank ≤ dimension) cannot recover it — the optimum
  stays genuinely unknown. `strong` must NOT implement a proven-optimal scheme.
- **Format E — no leakage of the hidden law, via ANY channel:**
  - The statement's worked example MUST be a DIFFERENT, unrelated expression shape (say so explicitly:
    "illustrative FORM only — not the hidden law"). The solver discovers the family from data.
  - `gen.py`'s STDOUT (the train rows the solver sees) MUST NOT print the seed, the law, or its
    coefficients — data rows only. The ground truth lives ONLY inside the checker.
  - Held-out MUST be genuine EXTRAPOLATION (a different input region than train), regenerated
    deterministically inside the checker; do NOT ship an importable `groundtruth` module in the dir.
  - `strong` must not hit ~1.0 (irreducible noise keeps headroom).
- **Reject non-finite.** The checker MUST reject `nan`/`inf` in the participant output (parse then
  check `isfinite`) BEFORE scoring — the harness feeds a nan/inf-flooded output and requires ~0.
- **Leave headroom (no saturation).** `strong` must NOT hit Ratio ≈ 1.0 on the HARDEST case (≤ ~0.9);
  raise the cap or harden the largest instance.
- **Feasibility checked adversarially.** The harness feeds empty/garbage/huge/nan/inf outputs and
  requires ~0. Validate strictly (bounded reads, schema, ranges, finiteness, token-count/seekEof).
- Print the score on its OWN final line; the harness takes the LAST `Ratio:`; a Python checker MUST
  exit 0 for its score to count.

## Self-validate before returning
```
python3 <SYNTH>/harness/validate_problem.py <probdir> --keep-testdata
```
Fix the failing gate and re-run (≤6 rounds). Gates: G1 compile/parse · G2 gen · G3 bounds[0,1] ·
G4 determinism · G5 feasibility(invalid→0) · G6 baseline(trivial≈0.1, not perfect) ·
G7 discrimination(strong−trivial≥margin) · G8 divergence.

## Return (JSON only)
```json
{"id":"<id>","verdict":"PASS|FAIL","title":"...","format":"C|D|E","family":"...",
 "metrics":{"trivial":0.1,"greedy":..,"strong":..,"divergence":..,"invalid":0.0},
 "rounds":<int>,"notes":"core idea + why open-ended/deterministic (or why it failed)"}
```
