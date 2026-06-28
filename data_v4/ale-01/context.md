# Drone Survey Sweep

## Research question

A survey drone must photograph every one of `n` ground stations scattered across a large
rectangular region and then return to where it launched. The drone visits the stations in some
order, flies in a straight line between consecutive stations, and the route is a single **closed
loop**: after the last station it returns to the first. Battery (and therefore mission cost) is
governed by the **total flight distance**, so the task is to choose the visiting order that makes
the closed loop as short as possible.

Phrased structurally: pick a set of flight legs so that **every station has exactly two incident
legs** (one arriving, one leaving) and the legs form **one connected loop through all stations** —
a *degree-≤2 spanning structure that spans every station*, i.e. a single Hamiltonian cycle. Among
all such loops, minimize the sum of Euclidean leg lengths. This is the metric travelling-salesman
structure; it is NP-hard, there is no known efficient exact solution at this scale, and the
benchmark scores a route by *how short* it is rather than by matching a unique optimum.

## Input / output contract

- **Input (stdin):** the first token is `n` (the number of stations, `800 ≤ n ≤ 2000`). Then `n`
  lines follow, line `i` (0-indexed) holding two integers `x_i y_i` with
  `0 ≤ x_i, y_i ≤ 1 000 000`. All coordinates are distinct, so every leg has positive length.
- **Output (stdout):** `n` lines, the visiting order `p[0], p[1], …, p[n-1]` — a **permutation of
  `0 … n-1`**, one index per line. It denotes the closed loop `p[0] → p[1] → … → p[n-1] → p[0]`.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff the output is exactly `n` integers and they form a permutation of
`{0, …, n-1}` (every station listed exactly once ⇒ degree exactly 2 in the loop). Anything else —
wrong count, an out-of-range or repeated index, a non-integer token, a missing file — is
**infeasible**.

## Background

The visiting order is a single closed tour, so the structure is a Hamiltonian cycle on points in
the plane: classic geometric TSP. Several approaches sit on the table before committing:

- **Greedy nearest-neighbour construction.** Start at station 0, repeatedly fly to the nearest
  not-yet-visited station, close the loop at the end. `O(n²)` naively, near-linear with a spatial
  index. Always feasible, but it leaves long "regret" edges where it paints itself into a corner;
  typically 20–25 % above optimal on clustered point sets.
- **2-opt local search.** Repeatedly remove two legs and reconnect the two resulting paths the
  other way (which reverses a segment), keeping the swap when it shortens the loop. Strong, but a
  naive pass that recomputes the whole loop length per candidate is `O(n²)` per pass and far too
  slow for `n` near 2000 inside two seconds.
- **Or-opt.** Relocate a short chain of 1–3 consecutive stations to a better place in the loop —
  fixes the kind of mistake 2-opt cannot reach (a single misplaced station).
- **Iterated local search.** Once 2-opt/Or-opt converge to a local optimum, apply a *double-bridge*
  perturbation (a 4-opt kick that re-links four segments) and re-optimize, keeping the best tour
  seen. This is the established strong, simple metaheuristic for Euclidean TSP at this scale.

The decisive engineering lever is **incremental evaluation with a candidate list**: a 2-opt or
Or-opt move only changes a handful of legs, so its effect on the loop length is an `O(1)` delta over
exactly those legs — the loop length is never recomputed. Restricting move proposals to each
station's `k` nearest neighbours (built once via a uniform grid) makes a sweep near-linear instead
of quadratic. Don't-look bits skip stations whose neighbourhood has not changed since they last
failed to improve.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `n ∈ [800, 2000]` and places the stations as a mixture of a few 2-D Gaussian clusters
  (survey hot-spots) plus a uniform background, clipped to the `[0, 10⁶]²` grid, with all
  coordinates kept distinct. Clustered layouts are exactly where greedy leaves the most slack for
  local search to recover.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted
  permutation.
  - **Feasibility floor:** if the output is not exactly `n` integers forming a permutation of
    `{0, …, n-1}`, the score is **`0`**.
  - Otherwise let `L` be the closed-loop Euclidean length of the submitted tour and let `G` be the
    closed-loop length of the scorer's own deterministic nearest-neighbour tour from station 0
    (recomputed inside the scorer, so the reference is reproducible and independent of the solver).
    The score is

    ```
    score = round( 1 000 000 × G / L )      (feasible, L > 0)
    score = 0                                (infeasible)
    ```

    A higher score is better. The greedy nearest-neighbour reference scores exactly `1 000 000`; a
    shorter loop scores strictly more; a longer one scores less but stays positive. (`n ≤ 1` is a
    degenerate full-credit case.)
- **Reported metric.** The mean score over a fixed seed set. A genuine local-search solver should
  land well above `1 000 000` (≈ 1.20–1.35× on these instances); the trivial *identity-order* tour
  scores only a few tens of thousands and is the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible
permutation to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    vector<double> X(n), Y(n);
    for (int i = 0; i < n; i++) scanf("%lf %lf", &X[i], &Y[i]);

    // A feasible answer is ANY permutation of 0..n-1 (the visiting order of the
    // closed loop). Start from the identity so we always have something legal.
    vector<int> tour(n);
    for (int i = 0; i < n; i++) tour[i] = i;

    // TODO heuristic: build a k-nearest candidate list, do greedy nearest-neighbour
    // construction, then 2-opt + Or-opt local search with O(1) incremental deltas
    // and don't-look bits, wrapped in iterated local search (double-bridge kicks),
    // all under a ~2s wall-clock budget. Keep `tour` a valid permutation throughout.

    string out;
    for (int i = 0; i < n; i++) out += to_string(tour[i]) + "\n";
    fputs(out.c_str(), stdout);
    return 0;
}
```
