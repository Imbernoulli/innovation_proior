# Butterfly Wiring: Minimal-Addition Straight-Line Program for a Fixed Linear Map

## Problem
You are given a fixed integer matrix `A` (size `n x n`) that must be wired as
`y = A x`: each output `y_i` is a fixed integer linear combination of the
inputs `x_1 .. x_n`. Your circuit may only use binary **add** and **subtract**
gates — each gate consumes two previously available values and produces a new
one at unit cost. Your job is to wire up all `n` outputs correctly using as
**few gates (additions/subtractions) as possible**.

The naive approach treats every output independently, computing its dot
product term by term. But outputs often need the *exact same partial sum* —
sharing that partial sum once instead of recomputing it in every row it
appears in is what actually drives the gate count down. There is no known
closed-form recipe for the true minimum (finding the shortest such program is
NP-hard in general); you must discover as much reusable structure as you can.

## Input (stdin)
```
n
A[0][0] A[0][1] ... A[0][n-1]
...
A[n-1][0] ... A[n-1][n-1]
```
`1 <= n <= 20`, every entry `A[i][j]` is in `{-1, 0, 1}`.

## Output (stdout)
```
L
idx_1 a_1 op_1 b_1
...
idx_L a_L op_L b_L
out_1 out_2 ... out_n
```
- `L` is the number of gates you use (`0 <= L <= 20000`).
- Value **id 0** is the constant `0` (always available, free). Ids `1..n` are
  the inputs `x_1..x_n` (always available, free). Each instruction line
  introduces one new id, defining `t_idx = value(a) op value(b)` where
  `op` is `+` or `-`. Ids must be introduced in strictly increasing order
  starting at `n+1`: the `k`-th instruction line must have `idx = n+k`.
  Each operand `a`, `b` must already be defined, i.e. `0 <= a,b <= idx-1`.
- The final line gives, for each output `y_i` (`i = 1..n`), the id whose
  value equals `y_i`. Each `out_i` must be in `[0, n+L]`.
- No extra tokens may follow the output line.

## Feasibility
Evaluate your program on every standard basis input `x = e_j`
(`j = 1..n`, i.e. `x_j = 1` and all other inputs `0`) using exact integer
arithmetic. Your program is feasible iff, for every `i, j`,
`value(out_i)` on input `e_j` equals `A[i][j]` exactly (this proves
`y = A x` for all `x`, by linearity). Any parse error, out-of-range id,
malformed instruction sequence, non-`+`/`-` operator, intermediate value
exceeding `10^7` in absolute value, wrong output-token count, trailing
tokens, or reconstruction mismatch scores **0**.

## Objective
Minimize `L`, the total number of add/subtract gates used.

## Scoring
Let `B` be the cost of the *best fully independent, no-sharing* construction:
for each row, treat its nonzero terms as a chain of gates, but a term with
coefficient `+1` can start the chain for free (its input id is already
available) — only a leading `-1` term costs a gate (via `0 - x`). So row `i`
costs `nnz_i - 1` gates if it has any `+1` entry, else `nnz_i` gates (`nnz_i`
= number of nonzero entries in row `i`); `B` sums this over all rows. `B` is
the cheapest score obtainable with **zero** sharing between rows. With your
gate count `L`:
```
Ratio = min(1, 0.1 * B / L)
```
Matching the no-sharing baseline scores `0.1`. Using a third as many gates
scores `0.3`; a tenth as many caps the score at `1.0`. The true minimum gate
count is not known to be reachable by any polynomial method, so real headroom
remains above what careful sharing achieves.

## Constraints
- `1 <= n <= 20`; `A[i][j] in {-1, 0, 1}`; every row has at least one nonzero
  entry (`A` is never the all-zero matrix).
- `0 <= L <= 20000`; every intermediate value stays within `[-10^7, 10^7]`.
- Deterministic exact-integer scoring; no timing dependence.

## Example
`n = 3`, `A = [[1,1,0],[-1,-1,0],[0,0,1]]`. Row 0 (`x1+x2`) has a `+1` entry:
cost `1`. Row 1 (`-x1-x2`) is all-negative: cost `2`. Row 2 (`x3`) costs `0`
(direct reference). So `B = 1 + 2 + 0 = 3`. A feasible program: `4 = 1 + 2`
(id 4 = x1+x2), `5 = 0 - 4` (id 5 = -(x1+x2)), outputs `4 5 3` — row 2 reuses
input id 3 directly (no gate). This uses `L = 2` gates: `Ratio =
min(1, 0.1*3/2) = 0.15`, better than the `0.1` no-sharing baseline because
both of the first two rows reuse the single shared partial sum `x1+x2`
instead of each being built from scratch.
