# Prize-Collecting Patrol

## Research question

A patrol vehicle is stationed at a fixed **depot** and may drive out to collect prizes scattered at
`n` candidate sites across a large rectangular region, returning to the depot at the end. Each site
`i` has a coordinate `(x_i, y_i)` and a positive **prize** `prize_i` that is collected exactly once if
the vehicle visits it. The vehicle drives in straight lines between consecutive stops; fuel (the cost)
is proportional to the **total Euclidean travel** of the closed route depot → … → depot. Crucially,
**the vehicle does not have to visit every site** — a remote site with a tiny prize can cost more in
detour than it is worth. The task is to choose **which** subset of sites to visit **and in what order**
so as to maximize

```
profit = (sum of prizes of the visited sites) − (total Euclidean length of the closed route).
```

Visiting nothing is allowed and yields profit `0`. This is the **Prize-Collecting Travelling Salesman
Problem (PCTSP)**: it generalizes metric TSP (set every prize huge and you must visit all) with an
extra combinatorial layer — the *selection* of the visited set. It is NP-hard, has no known efficient
exact solution at this scale, and the benchmark scores a route by *how much profit* it earns rather
than by matching a unique optimum.

## Input / output contract

- **Input (stdin):**
  - the first token is `n`, the number of optional sites (`400 ≤ n ≤ 1200`);
  - the next line holds two integers `dx dy` — the depot coordinates (`0 ≤ dx, dy ≤ 1 000 000`); the
    depot is always part of the route and carries no prize;
  - then `n` lines follow, line `i` (0-indexed) holding `x_i y_i prize_i` with
    `0 ≤ x_i, y_i ≤ 1 000 000` and `prize_i ≥ 1`, all integers. All coordinates are distinct and
    distinct from the depot.
- **Output (stdout):**
  - the first token is `k`, the number of visited sites (`0 ≤ k ≤ n`);
  - then `k` lines, each one site id in `{0, …, n−1}`, **the visiting order**. The route is the closed
    loop `depot → p[0] → p[1] → … → p[k−1] → depot`. `k = 0` (with no ids) is legal and denotes "visit
    nothing", profit `0`.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff the output is a single integer `k` followed by exactly `k` integer ids,
all in `{0, …, n−1}` and pairwise **distinct**, with the declared `k` matching the number of ids.
Anything else — a wrong count, a repeated id, an out-of-range id, a header that disagrees with the
listed ids, a stray token, a missing file — is **infeasible**.

## Background

Stripped of its surface story, the structure is: pick a subset `S ⊆ {0,…,n−1}` and a cyclic order on
`S ∪ {depot}` to maximize `Σ_{i∈S} prize_i − len(cycle)`. Two coupled decisions interact — *which*
sites and *in what order* — and they cannot be separated cleanly: whether a site is worth visiting
depends on where it falls in the route (its **insertion detour**), which depends on the rest of the
chosen set. Several approaches sit on the table before committing:

- **Visit-all nearest-neighbour.** Ignore selection; build a TSP tour over *all* `n` sites by
  nearest-neighbour from the depot. Always feasible. But it drags the route through far, low-prize
  sites whose detour exceeds their prize — it leaves profit on the table exactly where selection
  matters. (This is the scorer's reference baseline.)
- **Prize-thresholding then TSP.** First decide the set by a static rule ("keep a site if its prize
  exceeds twice its distance to the depot"), then solve TSP on the kept set. The trap: a site's true
  cost is its *insertion detour into the eventual route*, not its distance to the depot, so a static
  threshold both keeps losers and drops winners.
- **Fix the set, then optimize the order (or vice versa).** Alternate between a TSP solver and a
  selection rule. Better, but the two phases fight each other: improving the order changes which sites
  are worth keeping, and changing the set changes the best order — alternation stalls in a poor joint
  local optimum.
- **A fused neighbourhood.** Treat *add a site*, *drop a site*, *relocate a site*, and *2-opt
  (re-order)* as moves in **one** local search, so the selection and the ordering co-evolve. This is
  the established strong approach for PCTSP, and the engineering lever is an **O(1) incremental gain
  test** per move: each move changes only a handful of edges, so its effect on profit is a constant-
  time delta over exactly those edges — profit is never recomputed from scratch. The non-obvious move
  is the **in/out toggle fused with insertion**: a solver that fixes the visited set first never
  discovers that dropping a site *and* re-ordering the rest is a single profitable step.

The decisive accelerators are a **doubly-linked-list tour** (so prev/next of any site is O(1), making
add/drop/relocate constant-time) and **candidate lists** (each site's `K` nearest neighbours, built
once from a uniform spatial grid) so that add/relocate/2-opt only ever try *good* slots instead of all
`O(k)` positions.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer `seed`) deterministically
  chooses `n ∈ [400, 1200]`, a depot, and places the sites as a mixture of a few 2-D Gaussian clusters
  (rich neighbourhoods, larger prizes) plus a uniform background of far-flung, **low-prize** sites.
  Prizes are tied to the instance geometry via `unit = SIDE / √n` (the typical nearest-neighbour
  spacing): cluster prizes are a few `unit`s (worth a short detour), background prizes are below one to
  two `unit`s (usually *not* worth their long detour). This is the regime where "skip the loser" is a
  genuine, load-bearing decision — visiting everything is a losing strategy, and so is visiting almost
  nothing.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the submitted solution.
  - **Feasibility floor:** if the output is not a single `k` followed by exactly `k` distinct in-range
    ids (header matching the count), the score is **`0`**.
  - Otherwise let `P` be the submitted solution's **profit** (`Σ prizes − closed-route length`; `0`
    for `k = 0`). Let `P_base` be the profit of the scorer's own deterministic **visit-all
    nearest-neighbour** tour from the depot over all `n` sites, and let `D > 0` be that same tour's
    closed length (an instance-scale normalizer). Both are recomputed inside the scorer, so the
    reference is reproducible and independent of the solver. The score is

    ```
    score = round( 1 000 000 + 1 000 000 × (P − P_base) / D )     (feasible), clamped to ≥ 0
    score = 0                                                       (infeasible)
    ```

    A higher score is better. The visit-all nearest-neighbour baseline scores exactly `1 000 000`; a
    more profitable solution scores strictly more; a worse one scores less but never below `0`. The
    empty tour (`P = 0`) scores `1 000 000 − round(1 000 000 × P_base / D)`, which is *negative→0*
    whenever visiting everything is profitable (the typical case) — so a real solver must beat the
    visit-all tour, not merely the empty one. (Degenerate `n = 0` or all-coincident sites: full-credit
    anchor `1 000 000`.)
- **Reported metric.** The mean score over a fixed seed set. A genuine fused-neighbourhood solver
  should land well above `1 000 000` (≈ `1.09–1.24 ×` on these instances); the trivial empty-tour and
  input-order tours score `0` and are the floor to beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible solution
to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    double dx, dy;
    scanf("%lf %lf", &dx, &dy);                 // depot (no prize, always visited)
    vector<double> X(n), Y(n), P(n);
    for (int i = 0; i < n; i++) scanf("%lf %lf %lf", &X[i], &Y[i], &P[i]);

    // A feasible answer is ANY set of distinct ids in a visiting order. The empty
    // tour (k = 0, profit 0) is always legal, so start from it as a safety net.
    vector<int> chosen;   // visiting order of the selected sites

    // TODO heuristic: keep the tour as a doubly linked list over sites + a virtual
    // depot; build candidate (k-nearest) lists from a uniform grid; greedily
    // cheapest-insert profitable sites; then run ONE fused local search that
    // sweeps ADD / DROP / RELOCATE (Or-opt-1, the in/out toggle) + 2-opt, each
    // move evaluated by an O(1) profit delta over the few edges it touches; wrap
    // it in iterated local search (kick a few sites in/out, re-descend, accept by
    // an SA rule) under a ~1.85s budget. Keep a valid tour at all times and print
    // the best feasible one seen.

    string out;
    out += to_string((int)chosen.size()); out += "\n";
    for (int v : chosen) { out += to_string(v); out += "\n"; }
    fputs(out.c_str(), stdout);
    return 0;
}
```
