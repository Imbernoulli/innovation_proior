# Quadratic Assignment Placement

## Research question

You are given `n` **facilities** and `n` **locations**. Between every ordered pair of facilities
`(i, j)` there is an integer **flow** `f[i][j] >= 0` (how much material moves from facility `i` to
facility `j`), and between every ordered pair of locations `(k, l)` there is an integer **distance**
`d[k][l] >= 0`. You must place each facility on a distinct location: a placement is a **permutation**
`p` of `{0, ..., n-1}` where `p[i]` is the location assigned to facility `i`. The cost of a placement
is the total flow-weighted distance

```
cost(p) = sum_{i=0..n-1} sum_{j=0..n-1} f[i][j] * d[p[i]][p[j]],
```

and the task is to **minimize** it. This is the classic **Quadratic Assignment Problem (QAP)** — the
arrangement problem behind facility layout, keyboard design, circuit/VLSI placement, and hospital
department layout. It is one of the hardest combinatorial problems known: it is **NP-hard**, the
search space has `n!` permutations, and it is notorious for **deep local minima** — instances of size
`n >= 30` are already at the edge of what exact branch-and-bound can solve. There is no exact answer
to compute at the sizes here; the benchmark scores a placement by *how low* its cost is, and a
strong heuristic is the only realistic tool.

## Input / output contract

- **Input (stdin):**
  - the first token is `n` (`1 <= n <= 80` in the generated instances);
  - then the `n x n` **flow** matrix `f`, row-major (`n` rows of `n` integers, `f[i][0..n-1]`);
  - then the `n x n` **distance** matrix `d`, row-major (`n` rows of `n` integers, `d[k][0..n-1]`).
  - All entries are non-negative integers; matrices have a zero diagonal. The generator's flow and
    distance matrices are symmetric, but the scorer and solver do not rely on symmetry.
- **Output (stdout):** `n` integers `p[0], p[1], ..., p[n-1]` — the location assigned to each
  facility, in facility order, whitespace-separated (any layout; the scorer reads tokens). The list
  must be a **permutation** of `{0, ..., n-1}`.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A placement is **feasible** iff the output parses as exactly `n` integers that form a permutation of
`{0, ..., n-1}` — every location in range and used exactly once. A parse error, the wrong count, an
out-of-range location, or a repeated location is **infeasible**.

## Background

The quadratic coupling is what makes QAP hard: the cost of placing facility `i` on a location depends
on where *every other* facility is placed, through the `f[i][j] * d[p[i]][p[j]]` cross terms. You
cannot place facilities one at a time and be done — moving any facility re-prices its interaction
with all the others. Several approaches sit on the table before committing:

- **Identity / arbitrary placement (the baseline).** Place facility `i` on location `i`. It is a
  valid permutation, always feasible, and it is the reference the scorer normalizes against — but it
  ignores the instance entirely, co-locating heavily-communicating facilities at far-apart locations
  as often as not. It is the floor a real solver must beat.
- **Greedy construction.** Place facilities one at a time, each time choosing the location that
  minimizes the added cost against already-placed facilities. Fast, but myopic: it commits early to
  placements that the quadratic interactions later punish, and it has no way to repair a bad early
  choice. It lands in a poor local arrangement.
- **2-swap local search.** From a feasible placement, repeatedly **swap the locations of two
  facilities** and keep the swap if it lowers the cost. Swapping two entries of a permutation is
  always a permutation, so feasibility is free; this `O(n^2)`-sized neighbourhood is the natural one
  for QAP. Plain hill-climbing on it gets stuck fast in QAP's many deep local minima, so it needs a
  metaheuristic on top — and it needs the per-move evaluation to be cheap, or it is far too slow.

The decisive lever is making 2-swap local search both **cheap per move** and **able to escape local
minima**:

- **Closed-form O(n) swap delta.** The change in cost from swapping the locations of facilities `r`
  and `s` does not require recomputing the whole `O(n^2)` double sum: only the terms that touch `r`
  or `s` change, so the delta is a single `O(n)` sum over the other facilities (the standard QAP swap
  formula; it handles asymmetric matrices via a four-index expansion). Evaluating one candidate move
  is `O(n)`, not `O(n^2)`.

- **Taillard's incremental delta matrix (the innovation).** Keep `delta[r][s]` — the swap gain for
  every pair — in a matrix. After *performing* a swap of facilities `(u, v)`, almost every other
  `delta[r][s]` (those whose pair `{r, s}` is disjoint from `{u, v}`) updates by a **closed-form
  `O(1)` recurrence**; only the `O(n)` deltas that involve `u` or `v` are recomputed from scratch in
  `O(n)` each. So a *full neighbourhood re-scan after a move* costs `O(n^2)` with a tiny constant
  (mostly `O(1)` updates) instead of the naive `O(n^3)` (every pair recomputed in `O(n)`). This is
  what makes thousands of full sweeps per second feasible at `n = 80`, and it is the engine that lets
  the metaheuristic actually grind through QAP's rugged landscape.

- **Robust tabu search (Taillard's Ro-TS).** On top of the fast neighbourhood, run **tabu search**:
  always move to the best non-tabu swap (allowing a tabu one only if it would beat the best cost ever
  seen — the *aspiration* rule), and forbid undoing a move for a **randomized tenure** around `n`
  (the "robust" part — a fixed tenure can cycle, a randomized one resists it). A long-unused move is
  occasionally forced, for diversification. The best permutation found over the whole run is returned.
  Ro-TS is the standard strong heuristic for QAP and reaches best-known values on QAPLIB instances.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `n in [40, 80]`, then builds a structured QAP instance: the **distance** matrix is
  Euclidean on a random 2-D point layout (so it is a real symmetric geometric distance matrix, the
  QAPLIB "nug"/"tai*a" style), and the **flow** matrix is **clustered** — facilities are split into a
  few groups, with high flow inside a group and little or none across groups, plus noise. The
  clustering gives the instance genuine structure (group-mates want to be at nearby locations), so a
  good permutation that co-locates heavily-communicating facilities scores well below the identity
  arrangement — but the coupling between the flow clustering and the geometry is exactly what makes
  the optimum hard to find.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted
  permutation.
  - **Feasibility floor:** if the output does not parse as exactly `n` integers forming a permutation
    of `{0, ..., n-1}` (wrong count, out-of-range, repeated, or non-integer token), the score is
    **`0`**.
  - Otherwise compute `cost(p) = sum_i sum_j f[i][j] * d[p[i]][p[j]]` (lower is better). Let
    `cost_id` be the cost of the **identity permutation** `p[i] = i`, recomputed inside the scorer so
    the reference is reproducible and solver-independent. The score is

    ```
    score = round( 1 000 000 * cost_id / cost(p) )     (feasible, cost > 0)
    score = 0                                            (infeasible)
    ```

    A higher score is better. The identity reference scores essentially `1 000 000`; a lower-cost
    placement scores strictly more; a feasible-but-worse placement scores less but stays positive.
    (The unreachable `cost == 0` case is given a generous full-credit cap rather than dividing by
    zero.)
- **Reported metric.** The mean score over a fixed seed set. A real Robust-Tabu-Search solver on the
  incremental delta matrix lands well above the identity floor (roughly `1.5x`–`2.2x` of `1 000 000`,
  i.e. cost cut to about half to two-thirds of identity, on these clustered instances); the trivial
  identity baseline scores exactly `1 000 000` and is the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible
permutation (one location per facility, in facility order) to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<long long>> f;   // flow matrix, n x n
vector<vector<long long>> d;   // distance matrix, n x n

int main() {
    if (scanf("%d", &n) != 1) return 0;
    f.assign(n, vector<long long>(n));
    d.assign(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) scanf("%lld", &f[i][j]);
    for (int k = 0; k < n; k++)
        for (int l = 0; l < n; l++) scanf("%lld", &d[k][l]);

    // A feasible answer is ANY permutation p of {0,...,n-1}: p[i] is the location
    // of facility i. The safe construction is the identity p[i] = i -- always a
    // valid permutation -- so we always have something legal to print, and we
    // keep the best permutation seen so any time-limit cutoff is still feasible.
    vector<int> p(n);
    for (int i = 0; i < n; i++) p[i] = i;

    // TODO heuristic: 2-swap robust tabu search with Taillard's incremental
    // swap-delta matrix. Keep delta[r][s] = cost change of swapping facilities
    // r and s; evaluate a swap in O(n) (closed-form delta), and after each
    // performed swap (u,v) update every disjoint delta[r][s] in O(1) and only
    // the O(n) deltas touching u or v from scratch, so a full neighbourhood scan
    // is O(n^2) with a tiny constant. Move to the best non-tabu swap (aspiration
    // overrides tabu if it beats the best cost ever), forbid undoing a move for a
    // randomized tenure around n, track and finally print the best permutation.

    // Emit the location of every facility, in facility order.
    string out;
    for (int i = 0; i < n; i++) {
        out += to_string(p[i]);
        out += (i + 1 == n) ? '\n' : ' ';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
