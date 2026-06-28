# Tower Placement for Signal Coverage

## Research question

You operate a wireless network over a square service area. There are `n` **demand nodes**
(households, sensors, cells) and `m` candidate **tower sites**. A tower built at site `j` radiates
power that reaches demand node `i` with a distance-decayed strength `P[i][j] >= 0` (close demands
get a lot, far ones little or nothing). Demand `i` is **served** once the *accumulated* signal it
receives from all built towers reaches its requirement `req[i]`:

```
sum over built sites j of P[i][j]  >=  req[i].
```

Building a tower costs one unit regardless of where it is. The task is to choose a **subset of
sites** so that **every** demand is served, while building **as few towers as possible**. Formally
this is the 0/1 covering integer program

```
minimize    sum_j x_j
subject to  sum_j P[i][j] * x_j  >=  req[i]   for every demand i,
            x_j in {0, 1}.
```

This is an NP-hard set-multicover problem: there is no efficient exact solver, and the quality of a
heuristic is judged on a continuous score (how few towers it uses relative to a baseline). An output
that leaves any demand under-served is **infeasible** and scores `0`.

## Input / output contract

The solver reads one instance from **stdin** and writes one solution to **stdout**.

**Input (instance):**
```
n m
req[0] req[1] ... req[n-1]
P[0][0] P[0][1] ... P[0][m-1]      <- row for demand 0: power each site delivers to it
P[1][0] ...                         <- row for demand 1
...
P[n-1][0] ... P[n-1][m-1]
```
All `P` and `req` values are non-negative reals printed with six decimals. Sizes are moderate
(`n` around 180-260, `m` around 90-140). Whitespace (spaces or newlines) separates all tokens.

**Output (solution):**
```
k j_1 j_2 ... j_k
```
where `k` is the number of chosen sites and `j_1..j_k` are their **distinct 0-based indices** in
`[0, m)`. The order does not matter. `k = 0` (build nothing) is a syntactically valid output but is
feasible only if every `req[i]` is `0`.

Time limit: a few seconds per instance (the reference solver finishes in tens of milliseconds).

## Background

The natural relaxation drops integrality to get the **covering LP** `min 1^T x` s.t. `P x >= req`,
`0 <= x_j <= 1`. Its optimum lower-bounds the integer optimum and its fractional solution `xf[j]`
tells you which sites are "structurally important". Two classical heuristics frame the problem:

- **Greedy max-coverage / set-cover greedy.** Repeatedly build the site that reduces the total
  remaining *deficit* `sum_i max(0, req[i] - recv[i])` the most, until everything is served. This is
  the textbook `H_n`-approximation for covering and is the natural strong baseline; with a lazy
  (CELF) priority queue it is fast because the deficit-reduction gain is submodular (monotone
  non-increasing as more towers are built).

- **LP-relaxation then rounding.** Solve the covering LP, then turn the fractional `xf` into an
  integer set by rounding guided by *coverage-per-cost*, repairing any residual shortfall and then
  pruning redundant towers. The fractional weights steer the rounding toward the sites the LP found
  worth their cost, which lets the result improve on the pure deficit-greedy ordering.

The non-obvious lever this datapoint targets is the second family: a **fractional-relax → greedy
deterministic round → lazy-greedy repair → local-search prune** pipeline, with the demand signals
tracked incrementally in a `recv[]` array so every feasibility/gain test is cheap.

## Evaluation settings

A deterministic scorer reads the instance and the solution and applies this rule:

1. **Parse + index validity.** Read `k` and the `k` indices. If parsing fails, any index is out of
   `[0, m)`, or any index repeats, the score is `0`.
2. **Accumulate signal.** For each demand `i`, compute `recv[i] = sum over chosen j of P[i][j]`.
3. **Feasibility floor.** If `recv[i] < req[i] - 1e-6` for **any** demand `i`, the score is `0`.
   (Every demand must be served; this is the feasibility → 0 floor.)
4. **Normalised score.** If feasible, let `k` be the number of built towers and let `B` be the
   number of towers used by the deterministic **greedy max-coverage baseline** (computed inside the
   scorer on the same instance). The score is

   ```
   score = B / k.
   ```

   Fewer towers ⇒ higher score. The greedy baseline scores exactly `1.0` by construction; a solver
   that uses fewer towers than greedy scores `> 1.0`, one that uses more scores `< 1.0`. Higher is
   better.

**Instance generation.** A generator seeded by an integer places `n` demand nodes clustered into a
few 2D-Gaussian hotspots and `m` sites (mostly near the hotspots, some scattered). The power a site
delivers to a demand decays with distance as `P0 / (1 + (dist/R)^2)` (a Lorentzian/Cauchy profile)
and is truncated to `0` beyond a hard cutoff. Each `req[i]` is set to a random fraction
(`~0.30-0.62`, capped below the all-sites total) of the signal demand `i` would receive if **every**
site were built — so the "build every site" solution is always feasible, the feasibility floor is
meaningful, and a trivial baseline always exists. The generator, scorer, and seed set are frozen.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible
solution to stdout. The pre-method scaffold below already prints a guaranteed-feasible answer
(build every site); the heuristic replaces the `// TODO` block to build far fewer towers.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) { printf("0\n"); return 0; }
    vector<double> req(n);
    for (auto& r : req) scanf("%lf", &r);
    // Pji[j][i] = power that site j delivers to demand i  (column-major).
    vector<vector<double>> Pji(m, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) scanf("%lf", &Pji[j][i]);

    // TODO: choose a small subset of sites so that, for every demand i,
    //       sum over chosen j of Pji[j][i] >= req[i].  Track recv[i] incrementally.
    //       Default feasible answer: build every site.
    vector<int> chosen;
    for (int j = 0; j < m; j++) chosen.push_back(j);

    printf("%d", (int)chosen.size());
    for (int j : chosen) printf(" %d", j);
    printf("\n");
    return 0;
}
```
