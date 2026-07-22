# Rival-Solver Portfolio: Schedule a Shared Step Budget (Format B, isolated)

You manage a research toolkit with **k=4 rival solvers**. You are handed a batch
of `n_cases` cases (cheap, public features only) and **one shared step budget**
`budget` that must cover *all* of them. Solver `j` needs a deterministic number
of steps to finish case `i`:

```
req_j(i) = max(req_floor,
               base_j + size_coef_j*size_i + domain_coef_j*domain_i
                      + inter_coef_j*size_i*domain_i + case_noise[i][j])
```

`domain_i` is 0 or 1; `size_i` is in `[0,1]`. All coefficients, the noise
table, and `req_floor` are given to you in the instance — nothing about a
solver's cost is hidden. **Your job**: submit an ORDERED schedule of
`(case, solver, amount)` attempts. The grader replays your schedule against
the shared budget **in the order you gave it**: each attempt deducts `amount`
from what remains (skipped if it doesn't fit) and credits that much progress
to that `(case, solver)` pair. A case is **solved** the instant some solver's
*cumulative credited* progress on it reaches that solver's `req_j(i)` —
progress on different solvers for the same case does NOT combine, but
multiple attempts on the *same* `(case, solver)` pair do. Once a case is
solved, further attempts on it are free no-ops.

**Objective**: maximize the fraction of cases solved. There is no way to
finish a case for less than its cheapest solver's requirement, and the total
budget is never enough to fund every case at every solver — so *which* cases
you fund, with *which* solver, and *in what order* (since a later attempt
that no longer fits the remaining budget simply never runs) all matter.

## Public instance (stdin JSON)
```json
{
  "k": 4, "n_cases": <int>, "budget": <float>, "req_floor": <float>,
  "cases": [{"domain": 0|1, "size": <float in [0,1]>}, ...],
  "solver_profiles": [{"base":.., "size_coef":.., "domain_coef":.., "inter_coef":..}, ...],
  "case_noise": [[n_0,n_1,n_2,n_3], ...]
}
```
`cases`, `case_noise` have length `n_cases`; `solver_profiles` has length `k`.

## Answer (stdout JSON)
```json
{"attempts": [[case_idx, solver_idx, amount], ...]}
```
Each entry: `case_idx` in `[0, n_cases)`, `solver_idx` in `[0, k)`, `amount`
a finite number `>= 0`. Any wrong shape/type, out-of-range index, negative or
non-finite amount anywhere in the list rejects the WHOLE answer (score 0 for
that case). At most 20000 entries.

## Scoring
Let `obj` = fraction of cases you solved. The grader also computes, itself,
a weak reference `b` (always fund one fixed default solver, an even
per-case share, case order) and a near-optimal reference `no` (best solver
per case, scheduled cheapest-first). Per instance:
```
ceil  = min(0.97, 1.15 * no)
score = clip( 0.1 + 0.75 * (obj - b) / max(ceil - b, 0.05),  0, 1 )
```
so matching the weak reference scores ~0.1, approaching the near-optimal
reference scores well below the 1.0 ceiling (headroom is intentional). The
final score is the mean over 10 fixed, seeded instances of varying size and
domain mix.

## Suggested strategies (increasing sophistication)
- **Fixed default** — one arbitrary solver, spread the budget evenly, don't
  look at the features.
- **Global best** — compute each solver's AVERAGE requirement over the whole
  batch, back the single cheapest one for every case.
- **Per-case best** — recompute the cheapest solver PER case (the average
  ranking can hide a solver that is only cheap on a feature interaction);
  fund each case with its own best solver.
- **Cost-ordered schedule** — additionally schedule the per-case choices by
  ascending cost, so the shared budget buys the most solved cases before it
  runs out, instead of funding cases in whatever order they arrived.
