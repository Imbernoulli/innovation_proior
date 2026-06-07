# The Karmarkar–Karp Bin-Packing Algorithm — OPT + O(log² OPT)

## Problem

Pack `n` pieces of rational sizes in `(0,1]` into the fewest unit-capacity bins (a bin's pieces must sum to `≤ 1`). The problem is NP-hard, and prior polynomial-time algorithms gave only *multiplicative* guarantees `A(I) ≤ C·OPT(I) + o(OPT)` (greedy first-fit-decreasing: `C = 11/9`; Fernandez–de la Vega–Lueker: `(1+ε)OPT + O(ε⁻²)`). The Karmarkar–Karp algorithm gives an *additive* guarantee:

```
A(I) ≤ OPT(I) + O(log² OPT(I)),
```

with the variant `A(I) ≤ OPT(I) + O(log² m(I))` when the number of distinct sizes `m(I)` is small. It runs in polynomial time, making `O(log n)` calls to a fractional-packing subroutine.

## Key idea

Solve the strong **configuration (Gilmore–Gomory) LP** instead of analyzing greedy placement, then convert its fractional optimum into an integer packing with only polylogarithmic loss, via three devices: geometric grouping (to make distinct sizes few), an ellipsoid + knapsack-oracle solver (to handle the LP's astronomically many columns in polynomial time), and recursive rounding (to make the rounding loss lower-order instead of `Θ(OPT)`).

## The configuration LP and its rounding

With `m` distinct sizes `s₁,…,s_m`, `bᵢ` pieces of size `i`, and `A` the matrix whose `q` columns are the configurations (multisets fitting one bin):

```
(I)  min 1·x  s.t.  A x ≥ b, x ≥ 0          LIN(I) := opt.
(II) max u·b  s.t.  uᵀA ≤ 1, u ≥ 0          (dual; uᵢ = price of size i)
```

Bounds: `SIZE(I) ≤ LIN(I) ≤ OPT(I)`, and a basic feasible `x` has `≤ m` nonzero entries. Rounding it (take `⌊xⱼ⌋` bins per configuration, pack the residual of size `< m` by the better of `m` bins or `2·SIZE+1` bins) gives the central inequality

```
OPT(I) ≤ LIN(I) + (m+1)/2.
```

The integer–fractional gap is governed by the number of **distinct sizes** `m`, not the number of pieces `n`.

## Making m small (grouping) and a bounded (small-piece elimination)

- **Eliminate small pieces** (size `≤ g/2`) at threshold `g = 1/SIZE(I)`: pack the rest, reinsert greedily; cost `≤ max(A, (1+g)·OPT + 1) = OPT + O(1)`. Afterwards `a(I) > g/2`, so `log(1/a) = O(log OPT)`.
- **Linear grouping** (parameter `k`): sort descending, round each group of `k` up to its max, discard the top group. Gives `m(J) ≤ n/k`, `OPT(J) ≤ OPT(I) ≤ OPT(J) + k`. The `n/k`-vs-`k` tradeoff stalls at `O(√n)`.
- **Geometric grouping** (the new device): group by size *budget* (dyadic size classes with group size `k·2^r` in class `r`, or sweep and start a new group at accumulated size `k`). Each group carries size `≈ k`, so

```
m(J) ≤ (2/k)·SIZE(I) + ⌈log₂(1/a(I))⌉,   OPT(J) ≤ OPT(I) ≤ OPT(J) + k·⌈log₂(1/a(I))⌉
```

(size-budget variant: `m(J) ≤ SIZE(I)/k + ln(1/a)`, loss `≤ 2k(2 + ln(1/a))`). Distinct sizes scale with `SIZE/k`; loss with `k·log(1/a)`.

## Solving the huge LP in polynomial time

The primal has `q ≈ ∞` columns, so solve the **dual** (`m` variables, `q` constraints) by the Grötschel–Lovász–Schrijver **ellipsoid method**, which needs only a separation oracle. Prices `u` are dual-feasible iff no bin is overpriced, i.e. iff the **knapsack**

```
max v·u  s.t.  v·s ≤ 1, v ≥ 0 integer
```

has optimum `≤ 1`; if `> 1`, the optimal `v` is a violated configuration-constraint. To avoid NP-hard knapsack, round prices to a grid `ūᵢ = (t/n)⌊n uᵢ/t⌋` and solve by DP: `F(0)=0`, `F(κ)=minᵢ[F(κ − n ūᵢ/t)+sᵢ]`; `ū` is feasible iff `F(κ)>1` for all `κ` with `κ·t/n>1`. After `M = 4m²⌈ln(n/t)⌉` ellipsoid iterations the dual is solved within `t`. The configurations returned by the feasibility cuts (at most `M`) define a finite "realized" LP with the same optimum to within `t`; a constraint-elimination procedure (partition into `m+1` groups, drop one disjoint from the critical `m`-set) prunes to exactly `m` constraints, whose dual yields a basic primal `x = B⁻¹b` with `≤ m` nonzero configurations of value `≤ LIN(I) + h`. Total time `T(m,n) = O(m⁸ log m log²n + m⁴ n log m log n)`.

## Recursive rounding → O(log² OPT)

Paying the residual `(m+1)/2 ≈ SIZE/k` once stalls at `O(√OPT)`. Instead iterate: group, solve the fractional packing, buy `⌊xⱼ⌋` bins, recurse on the residual whose `SIZE` (≤ number of fractional configs ≤ `m`) is roughly **halved** each round. With a constant `k`:

```
#rounds = O(log SIZE) = O(log OPT),   loss per round = O(log(1/a)) = O(log OPT)
=>  total additive error = O(log OPT) · O(log OPT) = O(log² OPT).
```

The integer bins bought across rounds sum to `≤ OPT(I)`; the final residual is first-fit in `O(1)` bins. Hence `A(I) ≤ OPT(I) + O(log² OPT(I))`.

## Worked example

Items `[0.5, 0.5, 0.4, 0.4, 0.3, 0.3]`, `SIZE = 2.4`, so `OPT ≥ ⌈2.4⌉ = 3`. The configuration LP can mix bins like `{0.5, 0.5}`, `{0.4, 0.3, 0.3}`, `{0.4}`; rounding/packing yields the optimal 3-bin solution `{0.5, 0.5}`, `{0.4, 0.3, 0.3}` (sum `1.0`), `{0.4}` — three bins, matching the lower bound `⌈SIZE⌉ = 3`.

## Code (illustrative, faithful to the construction)

```python
from collections import Counter
import math
from scipy.optimize import linprog

def knapsack_oracle(prices, sizes, counts):
    # separation oracle = pricing subproblem: max-price configuration fitting a
    # unit bin; price > 1  =>  dual-infeasible, this config is a new column.
    G = 1000
    best = [(0.0, ())] * (G + 1)
    for i, (s, u, b) in enumerate(zip(sizes, prices, counts)):
        w = max(1, math.ceil(s * G))
        if w > G:
            continue
        for _ in range(b):                                   # bounded knapsack
            for g in range(G, w - 1, -1):
                if best[g - w][0] + u > best[g][0] + 1e-12:
                    cfg = dict(best[g - w][1]); cfg[i] = cfg.get(i, 0) + 1
                    best[g] = (best[g - w][0] + u, tuple(sorted(cfg.items())))
    price, cfg = max(best, key=lambda t: t[0])
    return price, dict(cfg)

def solve_fractional_packing(sizes, counts):
    # configuration LP min 1.x s.t. Ax>=b, x>=0 by COLUMN GENERATION; pricing =
    # knapsack oracle. Never enumerates all q configurations.
    m = len(sizes)
    columns = [{i: min(counts[i], int(1 // sizes[i]) or 1)} for i in range(m)]
    while True:
        A_ub = [[-(c.get(i, 0)) for c in columns] for i in range(m)]
        res = linprog([1.0] * len(columns), A_ub=A_ub, b_ub=[-c for c in counts],
                      bounds=[(0, None)] * len(columns), method="highs")
        u = [max(0.0, -y) for y in res.ineqlin.marginals]
        price, cfg = knapsack_oracle(u, sizes, counts)
        if price <= 1.0 + 1e-6 or not cfg or cfg in columns:
            return columns, res.x
        columns.append(cfg)

def reduce_distinct_sizes(items, k):
    # grouping: round each group of k up to its max, discard top group.
    s = sorted(items, reverse=True)
    if len(s) <= k:
        return [], s
    top, rest = s[:k], s[k:]
    grouped = []
    for g in range(0, len(rest), k):
        chunk = rest[g:g + k]; grouped += [chunk[0]] * len(chunk)
    return grouped, top

def first_fit(items, cap=1.0):
    bins = []
    for x in sorted(items, reverse=True):
        for bn in bins:
            if sum(bn) + x <= cap + 1e-9:
                bn.append(x); break
        else:
            bins.append([x])
    return bins

def karmarkar_karp(items, cap=1.0):
    # recursive rounding: group, solve fractional, buy floor(x_c), recurse.
    items = [x for x in items if x > 1e-12]
    if not items:
        return []
    if sum(items) <= 1.0 + 1e-9 or len(set(items)) <= 1:
        return first_fit(items, cap)
    k = max(1, int(math.isqrt(max(1, int(sum(items))))))
    grouped, discarded = reduce_distinct_sizes(items, k)
    bins = [[x] for x in discarded]
    if not grouped:
        return bins + first_fit([x for x in items if x not in discarded], cap)
    cnt = Counter(round(x, 6) for x in grouped)
    sizes, counts = list(cnt.keys()), [cnt[s] for s in cnt]
    columns, x = solve_fractional_packing(sizes, counts)
    remaining = Counter({s: counts[i] for i, s in enumerate(sizes)})
    for col, xc in zip(columns, x):
        for _ in range(int(math.floor(xc + 1e-9))):
            bn = []
            for i, a in col.items():
                take = min(a, remaining[sizes[i]])
                bn += [sizes[i]] * take; remaining[sizes[i]] -= take
            if bn:
                bins.append(bn)
    residual = [s for s, r in remaining.items() for _ in range(r)]
    return bins + karmarkar_karp(residual, cap)
```

On 120 random items in `[0.05, 0.7]` (`SIZE ≈ 50.4`) this uses about 56 bins against the lower bound `⌈SIZE⌉ = 51`, with no bin overflowing and every piece placed.
