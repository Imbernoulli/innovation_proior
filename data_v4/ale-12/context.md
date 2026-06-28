# Lattice Antenna Coverage

## Research question

We are handed a square lattice of `G x G` **demand cells**. Cell `(x, y)` carries an integer
demand weight `d(x, y) >= 1`, drawn from a spatially clustered population field. We are also handed
`M` candidate **antenna sites**. Site `i` sits at lattice cell `(sx_i, sy_i)`, has an integer
coverage radius `r_i`, and an integer power cost `c_i`. A placed antenna covers every lattice cell
inside its square (Chebyshev) footprint of side `2 r_i + 1` centred at `(sx_i, sy_i)`, clipped to
the lattice.

Choose a subset `S` of the antenna sites with total power cost `sum_{i in S} c_i <= B` (the budget),
to **maximize the total demand of the union of covered cells** — every cell that is covered by at
least one chosen antenna is counted **once**, regardless of how many antennas cover it. This is the
budgeted, monotone-submodular maximum-coverage problem; it is NP-hard, there is no exact answer at
this scale, and the output is judged by a continuous covered-demand score.

The non-obvious lever is that overlap between antenna footprints is the whole game: two cheap
antennas covering the *same* hot-spot are nearly worthless together, while the same budget spread
over disjoint hot-spots can pay off enormously. A score that respects the union (not the sum of
footprints) is what separates a real heuristic from a toy greedy.

## Input / output contract

Input (stdin), all integers, whitespace-separated:

```
line 1:        G  M  B
line 2:        G*G demand weights, row-major (y = 0..G-1 outer loop, x = 0..G-1 inner loop)
next M lines:  sx  sy  r  c        (one per antenna site i = 0..M-1)
```

Constraints (held by the fixed generator): `60 <= G <= 100`; `200 <= M <= 600`;
`1 <= d(x,y) <= 999`; `0 <= sx, sy <= G-1`; `2 <= r <= G/8`; `1 <= c <= 9999`; `B >= 1`,
sized so a handful of antennas fit but not all of them.

Output (stdout): a single subset of site indices,

```
k  i_1  i_2  ...  i_k
```

where `k = |S|` is the number of chosen antennas (may be `0`) followed by the `k` chosen indices.

## Background

Maximum coverage under a knapsack (budget) constraint is a textbook NP-hard problem whose objective
`f(S) = total demand of the covered union` is **monotone and submodular** (adding an antenna never
decreases coverage, and its marginal gain only shrinks as more antennas are already present). Several
families of method are on the table before committing:

- **Density greedy, no overlap awareness.** Rank sites by `footprint-demand / cost` once and add in
  that fixed order while they fit. Feasible and `O(M log M)`, but it double-counts overlapping
  footprints and so wastes budget piling antennas onto the same hot-spot. This is the trivial
  baseline the real solver must beat.

- **Cost-benefit greedy (Khuller-Moss-Naor).** At each step add the site with the largest *marginal*
  gain-to-cost ratio `Delta f(i | S) / c_i`, where `Delta f` counts only newly covered demand.
  Comparing the greedy result against the single best feasible antenna and keeping the better of the
  two restores a `(1 - 1/e)/2` approximation guarantee. The cost is that a naive implementation
  re-scores every remaining site after every pick — `O(M^2 * footprint)`.

- **Lazy greedy / CELF with marginal-gain caching.** Submodularity means a marginal gain computed
  earlier is an *upper bound* on the current one. Keep a priority queue keyed on cached gain-per-cost;
  only re-evaluate the top entry when it is stale, and commit it once its cache is current. Most
  re-scorings are skipped. This is the lever this problem is built around.

- **Swap / LNS local search on top.** Greedy plateaus; a remove-one-add-several swap, evaluated
  incrementally against a per-cell coverage-count array, can escape it within the same budget.

## Evaluation settings

The score of a solution is computed by recomputing coverage directly from the output. The scorer
parses `k` and the `k` indices and declares the solution FEASIBLE iff: `0 <= k <= M`; exactly `k`
indices follow; every index is in `[0, M-1]`; the indices are pairwise distinct; and the total cost
`sum c_i <= B`. For a feasible solution the score is the **total demand of the union of all cells
covered by the chosen antennas, each cell counted once**. Any feasibility violation floors the score
to **0** (the hard floor). The empty set is feasible and scores `0`.

Instances are produced by a fixed generator seeded by an integer: it lays down a uniform demand
floor plus `K = 4..10` Gaussian population hot-spots, then samples `M` antenna sites biased toward
high-demand cells (with a uniform tail), each with a radius and a cost loosely correlated with its
footprint area but noised so the cost-benefit trade-off genuinely bites. We report the mean
covered-demand score over a fixed seed set, computed by the frozen scorer, and compare every method
on the *same* instances. Time budget: about 2 seconds per instance; memory a few MB.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible subset
to stdout. The scaffold below reads the instance, leaves the heuristic as a TODO, and prints a
trivially feasible answer (the empty set) so that even the unfilled skeleton never produces invalid
output.

```cpp
#include <bits/stdc++.h>
using namespace std;

int G, M;
long long B;
vector<int> demand;            // demand[y*G + x]
struct Site { int sx, sy, r; long long c; int x0, y0, x1, y1; };
vector<Site> site;

int main() {
    int m; long long b;
    if (scanf("%d %d %lld", &G, &m, &b) != 3) { printf("0\n"); return 0; }
    M = m; B = b;
    demand.assign((size_t)G * G, 0);
    for (size_t i = 0; i < (size_t)G * G; ++i) scanf("%d", &demand[i]);
    site.resize(M);
    for (int i = 0; i < M; ++i) {
        int sx, sy, r; long long c;
        scanf("%d %d %d %lld", &sx, &sy, &r, &c);
        Site s; s.sx = sx; s.sy = sy; s.r = r; s.c = c;
        s.x0 = max(0, sx - r); s.y0 = max(0, sy - r);
        s.x1 = min(G - 1, sx + r); s.y1 = min(G - 1, sy + r);
        site[i] = s;
    }

    // TODO: pick a subset S of site indices with sum of costs <= B, maximizing
    // the total demand of the UNION of covered cells (each cell counted once).
    // Strongest known approach: cost-benefit LAZY greedy (CELF) with
    // marginal-gain caching + single-best guard, then incremental swap search.
    vector<int> chosen; // empty set is always feasible

    printf("%d", (int)chosen.size());
    for (int i : chosen) printf(" %d", i);
    printf("\n");
    return 0;
}
```
