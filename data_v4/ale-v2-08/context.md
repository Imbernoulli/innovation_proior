# Facility Layout Assignment

## Research question

A plant designer must place `n` facilities (machines, departments, modules) onto `n` physical
locations on the shop floor — exactly one facility per location, exactly one location per facility.
Between every ordered pair of facilities `(i, j)` there is a known **flow** `F[i][j]` (material
transported, message volume, foot traffic), and between every ordered pair of locations `(a, b)`
there is a known **distance** `D[a][b]`. If facility `i` goes to location `p[i]` and facility `j`
goes to location `p[j]`, the transport cost they generate is `F[i][j] · D[p[i]][p[j]]`. The total
layout cost is the sum over all pairs:

```
C(p) = Σ_i Σ_j  F[i][j] · D[p[i]][p[j]]
```

The task is to choose the assignment — the **permutation** `p` of locations to facilities — that
makes `C(p)` as small as possible: place heavily-communicating facilities on mutually nearby
locations.

Phrased structurally, this is the **Quadratic Assignment Problem (QAP)**. It is one of the hardest
classical combinatorial problems: NP-hard, with no known efficient exact method beyond very small
`n` (instances around `n = 30` already defeat exact solvers), and the objective couples *pairs* of
assignment decisions, so the cost of placing one facility depends on where every other facility
sits. The benchmark scores an assignment by **how cheap** it is, not by matching a unique optimum.

## Input / output contract

- **Input (stdin):** the first token is `n` (the number of facilities/locations, `60 ≤ n ≤ 120`).
  Then the **flow matrix** `F` is given as `n` rows of `n` non-negative integers each, then the
  **distance matrix** `D` is given as `n` rows of `n` non-negative integers each. Entry `F[i][j]`
  is the flow from facility `i` to facility `j`; entry `D[a][b]` is the distance from location `a`
  to location `b`. Both matrices have zero diagonal.
- **Output (stdout):** `n` lines, `p[0], p[1], …, p[n-1]` — a **permutation of `0 … n-1`**, one
  index per line. `p[i]` is the location assigned to facility `i`.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff the output is exactly `n` integers and they form a permutation of
`{0, …, n-1}` (every location used exactly once ⇒ a valid one-to-one assignment). Anything else —
wrong count, an out-of-range or repeated index, a non-integer token, a missing file — is
**infeasible**.

## Background

The objective is quadratic in the permutation, which is what makes QAP so much harder than a linear
assignment: you cannot decide facilities one at a time, because the cost of a placement depends on
all the others. Several approaches sit on the table before committing:

- **Identity / arbitrary assignment.** Place facility `i` on location `i`. Always feasible and the
  natural baseline, but it ignores the flow structure entirely; on block-structured instances it is
  far from good.
- **Greedy construction.** Assign facilities to locations one at a time (e.g. place the
  highest-total-flow facility on the most central location, then greedily extend). Cheap and
  feasible, but myopic: the quadratic coupling means early greedy choices look good locally and trap
  the layout in a poor basin.
- **2-swap (pairwise-exchange) local search.** Repeatedly exchange the locations of two facilities
  and keep the swap whenever it lowers `C`. This is the workhorse neighborhood for QAP. A naive
  implementation recomputes `C(p)` for every candidate swap — `O(n²)` per candidate, `O(n⁴)` per
  full sweep — which is hopeless for `n` near 120 inside two seconds.
- **Tabu search / simulated annealing over the 2-swap neighborhood.** Escape local optima by
  allowing non-improving swaps under control; a tabu list forbids immediately undoing a recent
  exchange. **Robust Tabu Search** (Taillard's QAP method) is the established strong, simple
  metaheuristic for this structure.

The decisive engineering lever is **incremental swap-delta evaluation**. The cost change of
swapping the locations of the facilities at positions `r` and `s`,
`Δ(r,s) = C(p with p[r],p[s] exchanged) − C(p)`, can be computed in **`O(n)`** with the classic
QAP formula instead of `O(n²)` from scratch — only the `2n` flow/distance terms that touch `r` or
`s` change. Better still, once an entire **delta table** `Δ[r][s]` has been built, performing one
swap of positions `(u, v)` lets every entry `Δ[r][s]` whose positions are *disjoint* from `{u, v}`
be refreshed in **`O(1)`** (Taillard's fast update), and only the `O(n)` entries that touch `u` or
`v` need an `O(n)` recompute. A full best-improvement sweep over all `O(n²)` swaps therefore reads a
maintained table in `O(n²)`, and the post-move table refresh is also `O(n²)` — a factor-`n²`
speedup over the naive scheme that is exactly what makes tabu search affordable at this size.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `n ∈ [60, 120]`. The **distance matrix** is the rounded Euclidean distance between `n`
  distinct points drawn on a `[0,100]²` integer grid (a genuine metric). The **flow matrix** is
  symmetric with zero diagonal and mixes a few **communicating groups** of facilities — heavy flow
  (20–100) inside a group — with a sparse light background (small flows on ~15% of the remaining
  pairs). This block-structured-flow + metric-distance combination is the classic hard QAP regime:
  the identity assignment scatters each group's heavy-flow members across the floor, leaving a large
  gap for a strong heuristic to close.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted
  permutation.
  - **Feasibility floor:** if the output is not exactly `n` integers forming a permutation of
    `{0, …, n-1}`, the score is **`0`**.
  - Otherwise let `C(p)` be the quadratic-assignment cost of the submitted permutation and let `C0`
    be the cost of the **identity** assignment `p[i] = i` (recomputed inside the scorer, so the
    reference is reproducible and independent of the solver). The score is

    ```
    score = round( 1 000 000 × C0 / C(p) )   (feasible, C(p) > 0)
    score = 0                                (infeasible)
    ```

    A higher score is better. The identity baseline scores exactly `1 000 000`; a cheaper layout
    scores strictly more; a worse one scores less but stays positive. (A degenerate all-zero-cost
    instance is a full-credit case.)
- **Reported metric.** The mean score over a fixed seed set. A genuine tabu-search solver should
  land far above `1 000 000` (≈ 2.0–2.9× cheaper than identity on these block-structured instances);
  the trivial *identity* assignment scores exactly `1 000 000` and is the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible
permutation to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int N;
vector<long long> F, D;  // row-major n x n matrices
static inline long long Fm(int i, int j) { return F[(size_t)i * N + j]; }
static inline long long Dm(int i, int j) { return D[(size_t)i * N + j]; }

int main() {
    if (scanf("%d", &N) != 1) return 0;
    if (N <= 0) return 0;
    F.assign((size_t)N * N, 0);
    D.assign((size_t)N * N, 0);
    for (size_t k = 0; k < (size_t)N * N; k++) { long long v; if (scanf("%lld", &v) != 1) v = 0; F[k] = v; }
    for (size_t k = 0; k < (size_t)N * N; k++) { long long v; if (scanf("%lld", &v) != 1) v = 0; D[k] = v; }

    // A feasible answer is ANY permutation of 0..n-1 (facility i -> location p[i]).
    // Start from the identity so we always have something legal.
    vector<int> p(N);
    for (int i = 0; i < N; i++) p[i] = i;

    // TODO heuristic: build the O(n)-delta table of all pairwise swaps, then run
    // Robust Tabu Search over the 2-swap neighborhood -- best-improvement scan
    // each iteration, O(1) fast table update for swaps disjoint from the move
    // and O(n) recompute for the touched rows/columns, a randomized tabu tenure
    // around n, aspiration, and diversification kicks -- under a ~2s budget.
    // Keep `p` a valid permutation throughout and output the best seen.

    string out;
    for (int i = 0; i < N; i++) out += to_string(p[i]) + "\n";
    fputs(out.c_str(), stdout);
    return 0;
}
```
