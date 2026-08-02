# Chain-Join Plans Under Uncertain Cardinality Estimates

## Problem

A query joins `n` base relations `R_0, R_1, ..., R_{n-1}` arranged in a **chain**:
relation `i` and relation `i+1` share a join predicate (edge `i`, for `i = 0..n-2`);
non-adjacent relations have no direct predicate. Each `R_i` has an **exact** base
cardinality `C_i` (a known table-scan row count).

You must build a **left-deep execution plan**: start by scanning one relation
`s`, then repeatedly extend the current contiguous interval of already-joined
relations one step to the **left** (`L`) or **right** (`R`) until all `n`
relations are joined. Extending across edge `e` produces a new intermediate
row count `size' = size * C_next * sel(e)`, where `sel(e)` is edge `e`'s
selectivity. If `size'` exceeds `MEM_CAP` the executor spills the join to
disk and that step's cost becomes `size' * SPILL_MULT`; otherwise the step's
cost is `size'`. The plan's cost is the sum of all step costs.

For each edge the input states an **estimated** selectivity `S_est[e]` and a
certified multiplicative **uncertainty bound** `F[e] >= 1`: the edge's
**true** selectivity is guaranteed to lie in `[S_est[e]/F[e], S_est[e]*F[e]]`
(clipped to `(0,1]`), but the exact true value is never given to you — the
score is computed from the true values, which you must reason about only
through the stated bound. A plan chosen to minimize cost under the point
estimate is optimal when the estimate is right and can be far worse than
necessary when it is wrong.

Your plan also gets ONE **reoptimization checkpoint**: after the first `h`
relations are joined (a fixed prefix), the executor observes the true
intermediate row count and buckets it as `LOW` (`<= MEM_CAP`), `MID`
(`<= 4*MEM_CAP`) or `HIGH` (`> 4*MEM_CAP`) — you supply a **separate**
continuation for each bucket, and the one matching what actually happens is
used to finish the plan.

## Input (stdin)
```
testId
n
C_0 C_1 ... C_{n-1}
MEM_CAP SPILL_MULT
h
S_est[0] F[0]
...
S_est[n-2] F[n-2]
```
`3 <= n <= 9`, `2 <= h <= n-1`.

## Output (stdout)
Whitespace-separated tokens:
```
START s
PRE d_1 d_2 ... d_{h-1}
BRANCH LOW  d_1 ... d_{n-h}
BRANCH MID  d_1 ... d_{n-h}
BRANCH HIGH d_1 ... d_{n-h}
```
`s` is the starting relation index; every `d_i` is `L` or `R`. The three
`BRANCH` blocks may appear in any order but each of `LOW`/`MID`/`HIGH` must
appear exactly once, and no extra tokens are allowed.

## Feasibility
Rejected (score `0`) if: the token stream doesn't match the schema above;
`s` is out of range; any `L`/`R` move would extend past relation `0` or
`n-1`; the `PRE` block does not cover exactly `h` relations; the chosen
`BRANCH` (matching the bucket the true prefix size actually falls into)
does not finish covering all `n` relations exactly once.

## Objective (minimize)
`F` = total true cost of the `PRE` prefix plus the chosen `BRANCH`.

## Scoring
Let `B` be the true cost of the canonical plan (start at relation `0`,
always extend right, same continuation regardless of bucket) — a trivial
feasible construction the checker computes itself.
```
sc    = min(900, 100 * B / F)
Ratio = sc / 1000
```
so the canonical plan scores `0.1`; a plan `9x` cheaper than it caps near
`0.9`. Deterministic, exact-formula, no wall-time.

## Constraints
`5 <= n <= 9`, `2 <= h <= n-1`, time limit `5s`.

## Example
`n=4`, `C = [1000, 1000, 1000, 1000]`, edge `0`: `S_est=0.0001, F=100`
(certified true range `[0.000001, 0.01]`); edges `1,2` mild (`S_est` around
`0.0006`, keeping steps roughly size-neutral). Starting at `0` and
extending right immediately crosses edge `0` believing it barely grows the
result — if the true selectivity is actually near the top of the stated
bound (`~0.01`), that one step's output jumps `~17x` above what the
estimate predicted, and **every later step now starts from that inflated
size**, so the jump's cost is paid over and over. A plan that instead
defers edge `0` to be crossed last pays that same jump only once, at the
very end.
