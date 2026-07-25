# The Metered Grinder (Jacobi Fill Ledger)

A nearly-diagonal crystal is modelled as a symmetric `n x n` matrix `A`. You own a
grinder that can polish one pair of facets `(i, j)` per pass: a Jacobi rotation that
annihilates the off-diagonal entry `A[i][j]`. The grinder is **metered** — every pass
is charged against a budget `B`, and the price of a pass depends on how rough the
crystal *currently* is. Plan the order of passes to leave as little off-diagonal
material as possible.

## Input (stdin)

```
n B m
d1 d2 ... dn
m lines: i j v
```

* `2 <= n <= 400`, integer budget `B`.
* `d1..dn` — nonzero integer diagonal entries.
* Each of the next `m` lines gives a nonzero integer off-diagonal entry
  `A[i][j] = A[j][i] = v` with `1 <= i < j <= n`. Entries not listed are exactly 0.

## Operation (performed by the judge, not by you)

You only choose the **order of pivots**. For each pivot `(i, j)` in your order, the
judge:

1. **Charges** `cost = nnz(i) + nnz(j)`, where `nnz(r)` is the number of currently
   nonzero entries of row `r` (the diagonal always counts). If the running total
   would exceed `B`, this pivot **and all remaining ones are discarded**.
2. **Applies** the exact Jacobi rotation that zeroes `A[i][j]`: with `a = A[i][i]`,
   `d = A[j][j]`, `p = A[i][j]` (skip if `p = 0`):
   `tau = (d - a)/(2p)`, `t = sign(tau)/(|tau| + sqrt(1 + tau^2))` (`t = 1` if
   `tau = 0`), `c = 1/sqrt(1 + t^2)`, `s = t*c`. For every `k != i, j`:
   `A[k][i] = A[i][k] <- c*A[i][k] - s*A[j][k]` and
   `A[k][j] = A[j][k] <- s*A[i][k] + c*A[j][k]`, then
   `A[i][i] <- a - t*p`, `A[j][j] <- d + t*p`, `A[i][j] = A[j][i] <- 0`.

A rotation mathematically reduces the off-diagonal energy by exactly `p^2`, **but** it
also writes into rows/columns `i` and `j`: position `(i, k)` becomes nonzero whenever
`A[i][k]` or `A[j][k]` was nonzero. Rotating on dense rows therefore *fills* the
matrix and raises the price of every later pass touching those rows — a self-inflicted
sparsity tax.

## Output (stdout)

```
k
i1 j1
...
ik jk
```

`0 <= k <= 2000` pivots, each `1 <= i, j <= n`, `i != j` (order matters; whitespace
free). The judge executes them in the given order under the budget rule above.

## Objective and Scoring

Let `E = sum_{i<j} A[i][j]^2` after your schedule is executed, and `E0` the initial
energy (empty schedule). Minimize `E`. The score is

```
Ratio = min(1, 0.1 * E0 / E)
```

so doing nothing scores exactly 0.1; halving `E` scores 0.2; a 10x reduction caps at
1.0. The largest entries are not always the best first targets: their rows are usually
the most expensive to touch, and grinding them early floods their partner rows with
fill-in. Structure-preserving orders can remove far more energy per unit of budget.

## Constraints

* Time limit 5 s, memory 512 MB. `n <= 400` (tests use `n <= 60`).
* All scoring is deterministic IEEE double arithmetic in the fixed order above.

## Example

Input:

```
4 20 3
40 50 60 70
1 2 6
2 3 6
3 4 6
```

Output:

```
2
2 3
1 2
```

Trace: pivot `(2,3)` costs `nnz(2)+nnz(3) = 3+3 = 6` and zeroes `A[2][3]`, but the
rotation creates **fill-in** at `(1,3)` and `(2,4)`. Pivot `(1,2)` then costs
`3+3 = 6` (row 1 gained a nonzero), not 5. Total spent 12 <= 20, so both execute.
The energy falls from `E0 = 108` to `E = 42.4767`, giving
`Ratio = min(1, 0.1*108/42.4767) = 0.254257`. An empty schedule would score 0.1.
