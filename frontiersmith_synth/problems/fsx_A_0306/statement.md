# Recycling Depot Route Ledger: Maximal-Independence Sign Matrix

## Problem
A regional recycling network runs `N` depots over `N` collection windows. The daily plan is a
`N x N` **route ledger** `M`, where each entry `M[i][j]` records the direction of the truck that
depot `i` runs during window `j`:

- `+1` = an **outbound** pickup run (leaves the depot to collect),
- `-1` = an **inbound** return run (comes back loaded).

Every cell must be `+1` or `-1` (each depot runs exactly one truck per window; it either goes out or
comes back). A handful of these assignments are **pre-committed** by standing contracts and cannot be
changed — you must honor them exactly.

The dispatcher wants the ledger to be as **operationally independent** as possible: no window's routing
pattern should be reconstructable as a signed combination of the others (this is what keeps the audit
robust and the depots load-balanced). The exact measure of this independence is the magnitude of the
ledger's determinant, `|det(M)|`, computed with **exact integer arithmetic** (Bareiss elimination — no
floating-point rounding). You want it as large as possible.

Because `N` is **odd** there is no perfectly orthogonal (Hadamard) ledger, so the true optimum is
unknown and unreachable in closed form — every point of improvement is earned by search.

## Input (stdin)
```
N K
r1 c1 v1
r2 c2 v2
...
rK cK vK
```
- `N` — ledger size (odd, 7..25).
- `K` — number of pre-committed cells.
- Each of the next `K` lines: a 0-indexed cell `(r, c)` that is fixed to direction `v` in `{-1, +1}`.
  All positions are distinct.

## Output (stdout)
Print the full `N x N` ledger: `N` lines, each with `N` space-separated values from `{-1, +1}`,
row `i` before row `i+1`, column `j` before `j+1`.

## Feasibility
Your output must (1) contain exactly `N*N` integer entries, (2) each entry `∈ {-1, +1}`, and (3) match
`v` at every pre-committed cell `(r, c)`. Any violation (wrong count, non-integer, `0`/`nan`/`inf`, out
of range, contradicting a fixed cell) scores **0**.

## Objective
Maximize `|det(M)|`, evaluated exactly over the integers.

## Scoring (deterministic)
Let `D = |det(M)|` for your ledger. The checker builds its own internal reference completion `M0` (free
cells filled with the lower-triangular sign pattern; the instance guarantees `M0` is non-singular) and
takes `B = |det(M0)|`. Both determinants are normalized to a **per-dimension magnitude** to keep the
score bounded across the huge dynamic range of determinants:

```
F  = D ^ (1/N)
Bp = B ^ (1/N)
Ratio = min(1000, 100 * F / Bp) / 1000
```

Reproducing the reference completion yields `Ratio ≈ 0.1`. A ledger whose per-dimension determinant is
10x the reference caps at `Ratio = 1.0`. The score is exact and bit-for-bit reproducible.

## Constraints
- `N` odd, `7 ≤ N ≤ 25`; `K ≈ 0.12 * N^2` pre-committed cells.
- Determinant is computed with exact big-integer Bareiss elimination (no tolerance).

## Example (illustrative)
For a tiny `N = 3` ledger with no pre-committed cells, the reference completion
```
 1 -1 -1
 1  1 -1
 1  1  1
```
has `|det| = 4`, so `B = 4`, `Bp = 4^(1/3) ≈ 1.587`. Submitting instead
```
 1  1  1
 1 -1  1
 1  1 -1
```
gives `|det| = 4` as well here — but a better sign pattern such as
```
 1  1 -1
 1 -1  1
-1  1  1
```
has `|det| = 4`... at `N = 3` the small examples coincide; the real headroom appears only as `N` grows,
where structured near-orthogonal ledgers push `|det|` far above the triangular reference. (This block is
illustration of the scoring pipeline, not an optimal construction.)
