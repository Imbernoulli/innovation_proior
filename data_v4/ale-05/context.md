# Relay Tower Placement

## Research question

A telecom operator is rolling out short-range relay towers across a flat metropolitan
plane. There are `N` households at integer coordinates in the square `[0, 10^6] × [0, 10^6]`.
The operator may switch on exactly `K` towers, and — for regulatory and site-leasing reasons —
**each tower must be installed at the site of an existing household** (the available real-estate
is exactly the household locations). Every household is served by its single nearest active
tower, and the monthly cost charged for that household is proportional to the straight-line
(Euclidean) distance to that tower.

Choose the `K` household sites to switch on so as to **minimize the total served distance**

```
cost = Σ_{h=1..N}  min_{t ∈ chosen}  euclid(household h, tower t).
```

This is the discrete **p-median / k-medoid** problem on the plane: NP-hard, with no closed-form
optimum, judged by a continuous cost. The lever is purely *which* `K` of the `N` sites are
switched on; nothing else is editable.

## Input / output contract

- **Input (stdin), the instance.**
  - Line 1: two integers `N K` with `800 ≤ N ≤ 1200` and `8 ≤ K ≤ 20` (so `K ≪ N`).
  - Next `N` lines: integers `x y` (`0 ≤ x, y ≤ 10^6`), the household coordinates, household `i`
    being the `i`-th line (1-based).
- **Output (stdout), the solution.**
  - Line 1: an integer `M` — it **must equal `K`**.
  - Next `M` lines: one integer each, a household index in `[1, N]`. The `K` indices must be
    **pairwise distinct**. These are the switched-on tower sites.
- **Time limit:** 2 seconds wall-clock. **Memory:** 256 MB.

## Background

Two reference approaches frame the problem before committing to one:

- **Lloyd-style alternation (k-means flavour).** Assign each household to its nearest current
  medoid, then move each medoid to the best site within its cluster, and repeat. It is fast and
  monotone but converges to weak local optima and, because medoids are constrained to household
  sites, the in-cluster re-centring step is itself a small p-median that costs `O(cluster²)`.
- **PAM swap local search (k-medoid).** Partitioning Around Medoids repeatedly evaluates swapping
  one active medoid out for one inactive site in, and applies the best improving swap. PAM finds
  markedly better optima than Lloyd but, written naively, each swap evaluation re-assigns all `N`
  households against all `K` medoids — `O(N·K)` per candidate and `O(N·K·(N−K))` per full pass —
  which is too slow at these sizes within 2 seconds.

The open question is how to keep PAM's solution quality while paying far less per swap.

## Evaluation settings

- **Scoring (what the judge reports; higher is better).** Let the solution be feasible iff
  `M == K`, every index is an integer in `[1, N]`, and all `K` indices are distinct. Then

  ```
  score = 0                                  if the solution is infeasible
  score = round( 10^9 / (1 + cost / N) )     otherwise,
  ```

  where `cost` is the total served distance above and `N` the household count. A lower `cost`
  yields a higher score; any malformed, wrong-length, out-of-range, or duplicate-index output
  **floors the score to exactly 0**. The underlying objective is to **minimize `cost`**; the
  `10^9/(1+cost/N)` wrapper just turns it into a bounded, maximize-style continuous score with a
  hard feasibility floor.

- **Instances.** A frozen generator places households as a random weighted mixture of 2D-Gaussian
  "neighbourhood" clusters (a handful more clusters than `K`), clipped to the plane, plus ~5%
  uniform "rural" noise households. Everything — `N`, `K`, every coordinate — is a deterministic
  function of an integer seed. We report the mean score over a fixed seed set (seeds `1..20`),
  each rung run on the *same* instances so numbers are directly comparable. The trivial baseline
  is "switch on the first `K` households".

## Code framework

A single self-contained C++17 program that reads the instance on stdin and writes a feasible
solution on stdout within the time budget.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, K;
    if (!(cin >> N >> K)) return 0;
    vector<double> X(N), Y(N);
    for (int i = 0; i < N; i++) cin >> X[i] >> Y[i];

    // A feasible solution is ANY K distinct household indices in [1, N].
    // Start from a valid set so we can always print something legal.
    vector<int> chosen(K);
    for (int i = 0; i < K && i < N; i++) chosen[i] = i;  // first K sites

    // TODO: heuristic. Improve `chosen` to minimize
    //   sum over households of distance to nearest chosen site,
    // e.g. k-means++ seeding + PAM swap local search with first/second-nearest
    // caching so each candidate swap costs O(N) instead of O(N*K).

    cout << K << "\n";
    for (int c = 0; c < K; c++) cout << (chosen[c] + 1) << "\n";  // 1-based
    return 0;
}
```
