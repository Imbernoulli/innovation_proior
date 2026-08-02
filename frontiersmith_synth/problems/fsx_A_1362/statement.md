# Cutting a Cake Nobody Envies

## Problem
A cake is made of `m` distinguishable, **fully divisible** ingredients (items). There are
`n` hungry agents. Agent `i` values a `x`-fraction of item `j` at exactly `x * v[i][j]`
(additive, linear valuations) — `v[i][j]` is the value agent `i` places on the *whole*
item `j`. Different agents may value the same item very differently.

You must cut and distribute every item into fractions (any real fractions, not just whole
items) so that the allocation is **envy-free (EF)**: no agent, judged by their *own*
valuation, prefers another agent's bundle over their own. Among all envy-free allocations,
you want to maximize **social welfare** — the sum, over all agents, of the value each
agent gets from their own bundle (again judged by that agent's own valuation).

Note the tension: the allocation that maximizes welfare *without* the fairness constraint
simply gives every item entirely to whoever values it most. That unconstrained allocation
achieves the theoretical ceiling `WELFARE_MAX = sum_j max_i v[i][j]` — but whenever two or
more agents both prize the same item highly, this "efficient" allocation leaves the losers
with far less than they'd get from the winner's bundle, so it is almost never envy-free.
The real goal is to search *within* the envy-free region for the best welfare you can find
there — the gap between what you achieve and `WELFARE_MAX` is the unavoidable price of
fairness for that instance.

## Input (stdin)
```
n m
v[0][0] v[0][1] ... v[0][m-1]
v[1][0] v[1][1] ... v[1][m-1]
...
v[n-1][0] ... v[n-1][m-1]
```
`n`, `m` are positive integers (3 <= n <= 7, m == n). Each `v[i][j]` is a non-negative
integer (0 <= v[i][j] <= 100).

## Output (stdout)
Print exactly `m` lines. Line `j` (0-indexed) contains `n` non-negative real numbers
`x[0][j] x[1][j] ... x[n-1][j]` — the fraction of item `j` given to each agent. Fractions
on a line must sum to `1` (tolerance `1e-6`).

## Feasibility
An output is valid iff **all** hold:
- exactly `m` lines are present, each with exactly `n` finite, non-negative numbers
  (tolerance `-1e-6` on non-negativity);
- for every item `j`, `sum_i x[i][j]` is within `1e-6` of `1`;
- **envy-freeness**: for every ordered pair of agents `(i, k)`, letting
  `val(i, bundle) = sum_j x[bundle][j] * v[i][j]`, we require
  `val(i, i) >= val(i, k) - 1e-6`.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = sum_i sum_j x[i][j] * v[i][j]` (total social welfare), subject to the
allocation being envy-free.

## Scoring
The checker builds its own trivial envy-free construction — give every agent the exact
same fraction `1/n` of every item (this is always feasible and always envy-free, since
every agent then holds an identical bundle). Its welfare is
`B = (1/n) * sum_i sum_j v[i][j]`.
With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```

## Example (worked score)
`n=2, m=2`, `v = [[10, 2], [2, 10]]` (agent 0 loves item 0, agent 1 loves item 1).
Equal split gives each agent `0.5*(10+2) = 6`, so `B = 12`.
Give item 0 entirely to agent 0 and item 1 entirely to agent 1: `val(0,0)=10`,
`val(1,1)=10`, `F=20`. Check envy: agent 0's value for agent 1's bundle is
`1*v[0][1] = 2 <= 10 = val(0,0)`, and symmetrically `1*v[1][0] = 2 <= 10 = val(1,1)`
for agent 1 — envy-free. `sc = min(1000, 100*20/12) = 166.667`, so `Ratio = 0.166667`.

## Constraints
`3 <= n <= 7`, `m = n`, `0 <= v[i][j] <= 100`. Time limit 5s, memory 512MB.
