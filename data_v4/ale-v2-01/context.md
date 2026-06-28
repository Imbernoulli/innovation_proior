# Capacitated Vehicle Routing (minimize total route length)

## Research question

A single depot must serve `n` clients, each with a positive integer demand, using identical vehicles
of capacity `cap`. A *route* is a tour that starts at the depot, visits a subset of clients in some
order, and returns to the depot; the total demand carried on a route may not exceed `cap`. We must
partition all `n` clients into routes and order each route so that **the sum of the Euclidean lengths
of all routes is as small as possible**. Any number of vehicles may be used.

This is the classic Capacitated Vehicle Routing Problem (CVRP). It is NP-hard; there is no known
efficient exact method at these sizes, and the judged objective is a continuous score, so the task is
to find the strongest heuristic solution within a fixed time budget.

## Input / output contract

Input (stdin):

```
n cap
depot_x depot_y
x_1 y_1 d_1
x_2 y_2 d_2
...
x_n y_n d_n
```

- `n` clients (`120 <= n <= 200` in the generated instances), `cap` the vehicle capacity.
- `depot_x depot_y` are the integer depot coordinates. The depot is referred to as node `0`.
- Client `i` (1-based, `1 <= i <= n`) has integer coordinates `(x_i, y_i)` in `[0, 1000]` and integer
  demand `d_i` with `1 <= d_i <= cap/4 < cap` (so no single client can overflow a vehicle).

Output (stdout): a list of routes. First the number of routes `K`, then `K` lines, each describing one
route as a count `m` followed by `m` client ids in visiting order:

```
K
m_1 id ... id
m_2 id ... id
...
m_K id ... id
```

The depot is implicit at the start and end of every route and is **not** listed. Every client id in
`1..n` must appear **exactly once** across all routes. Empty routes are allowed but pointless.

Time limit: 2 seconds. Memory: 256 MB.

## Background

CVRP is one of the most studied combinatorial optimization problems. The relevant solution families,
in increasing strength:

- **Trivial.** One client per route. Always feasible, but every route pays the full out-and-back trip
  to the depot, so the total length is enormous. This is the floor a real method must beat.
- **Clarke-Wright savings (1964).** Start from the trivial solution and greedily merge the two route
  endpoints `i, j` with the largest *savings* `s(i,j) = d(i,0) + d(0,j) - d(i,j)` whenever the merged
  load fits in `cap`. The savings measures how much depot-backtracking is removed by serving `i` and
  `j` consecutively. The parallel version (consider all merges globally, largest savings first) is the
  standard strong constructive baseline and is the reference this problem normalizes against.
- **Local search.** Intra-route **2-opt** (reverse a sub-segment to undo edge crossings) and **Or-opt**
  (relocate a short chain of 1-3 clients to a cheaper spot) refine each route. Each move's effect on
  length is an O(1) delta over the few edges it touches, so thousands of moves are cheap.
- **Large Neighborhood Search / ruin-and-recreate.** The state of the art for practical CVRP: repeatedly
  destroy part of the solution (remove a *related* cluster of clients) and rebuild it with a smart
  insertion (regret insertion), keeping the change only if it does not worsen the solution. This escapes
  the local optima that pure local search and savings get stuck in.

## Evaluation settings

The deterministic scorer (`verify/score.py`) reads the instance and a candidate solution and:

1. Parses the routes. If parsing fails, the score is `0`.
2. Walks every route, accumulating Euclidean length `depot -> first client -> ... -> last client ->
   depot`, and the route load `= sum of demands`.
3. Enforces feasibility — **any** violation floors the score to `0`:
   - a client id out of range `1..n`,
   - a client served more than once,
   - some client never served (`number served != n`),
   - a route load exceeding `cap`.
4. For a feasible solution with total length `L`, reports the **continuous score**
   `score = 1000000 / (L + 1)`, which is strictly decreasing in `L`: shorter total routes score higher.

The feasibility-to-0 floor is the hard ALE-Bench rule: an infeasible or malformed output earns nothing,
regardless of how short its (invalid) routes would have been.

Instances are produced by `verify/gen.py SEED` (deterministic in the seed): `n` in `[120, 200]`,
`cap` in `[40, 70]`, a depot near the grid centre, and clients drawn from a mixture of 3-6 Gaussian
clusters plus uniform noise on the `[0, 1000]^2` grid, with demands in `[1, cap/4]`. This clustered
structure is what makes savings and LNS meaningfully different from naive routing. The seed set for
self-verification is seeds `1..20`. We report the mean score over the seed set and compare against the
trivial (one-client-per-route) baseline and the savings-only baseline.

## Code framework

A single self-contained C++17 program reads the instance and writes a feasible set of routes. The
scaffold below establishes a valid baseline (one client per route) that the heuristic then replaces.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, cap;
    if (!(cin >> n >> cap)) return 0;
    double dx, dy;
    cin >> dx >> dy;                       // depot = node 0
    vector<double> x(n + 1), y(n + 1);
    vector<int> dem(n + 1);
    for (int i = 1; i <= n; ++i) cin >> x[i] >> y[i] >> dem[i];

    // TODO: heuristic. Build routes (each a list of client ids 1..n, depot
    // implicit at both ends), every client used exactly once, each route's
    // total demand <= cap, minimizing total Euclidean route length.
    // Baseline below is feasible: one client per route.

    cout << n << "\n";
    for (int i = 1; i <= n; ++i) cout << "1 " << i << "\n";
    return 0;
}
```
