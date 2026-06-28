# TSP with Time Windows (soft lateness)

## Research question

A single courier must visit `n` customers, starting from a depot at time `0`. Travelling between two
locations takes time equal to their Euclidean distance. Customer `i` has a service window `[e_i, l_i]`:
if the courier arrives **before** `e_i` it waits, **for free**, until `e_i`; if it arrives **after**
`l_i` it is penalised for the lateness. The courier visits every customer exactly once and does **not**
return to the depot.

You must output a visit order — a permutation of the `n` customers — that minimises

```
cost = total_travel_distance + lambda * sum_i lateness_i ,
```

where `lateness_i = max(0, arrival_i - l_i)`. This is the *soft* time-window variant of the travelling
salesman problem (TSPTW): the windows are never hard constraints, so any permutation is feasible, but a
careless order pays heavily in lateness because a late arrival at one stop pushes every later arrival
back too — lateness *compounds* along the tour. The design problem is to trade travel distance against
this compounding lateness.

## Input / output contract

- **Input (stdin).**
  - Line 1: `n lambda` — the number of customers (`40 <= n <= 60`) and the penalty weight
    `lambda` (a positive real, one of `{0.5, 1, 2, 4}`).
  - Line 2: `depot_x depot_y` — the depot coordinates (integers in `[0, 1000]`); the tour starts here
    at time `0`.
  - Next `n` lines: `x_i y_i e_i l_i` for customer `i = 1..n` — integer coordinates in `[0, 1000]`
    and real window bounds `0 <= e_i <= l_i`.
- **Output (stdout).** A whitespace-separated **permutation of the ids `1..n`** giving the visit order.
  The implied tour is `depot(t=0) -> perm[0] -> perm[1] -> ... -> perm[n-1]`.
- **Time limit:** 2 seconds. **Memory:** 256 MB.

## Background

TSPTW is NP-hard; with *soft* windows there is no exact answer at this scale within the time budget, so
this is a heuristic-optimisation problem judged by a continuous cost. Two facts shape the search:

- **Lateness compounds.** Arrival at the k-th stop depends on the departure time from stop k-1, which
  depends on its own arrival, and so on. A single late stop early in the tour can inflate the lateness of
  every stop after it. This is what makes a deadline-sorted order (which ignores geography) so bad: the
  long zig-zag distance makes the courier fall progressively further behind its windows.
- **Waiting is free.** Arriving early costs nothing but *time* — the departure time is
  `max(arrival, e_i)`. This reshapes the cost landscape: it can be worth arriving early and idling rather
  than detouring, and it means the marginal effect of a local change is not just a distance delta but a
  forward shift in the whole arrival-time profile.

The established strong family for this structure is a **metaheuristic local search**: build a
time-aware initial tour by cheapest insertion, then improve it with **Or-opt** (relocate a short
segment) and **2-opt** (reverse a segment) moves under **simulated annealing**. The non-obvious lever is
*incremental evaluation*: a relocate or reverse only perturbs the tour from one position onward, so the
arrival-time profile — and hence the lateness — needs to be re-propagated only from that position, while
the untouched prefix is read from a cached departure-time array.

## Evaluation settings

The score is computed by a frozen, deterministic scorer (`verify/score.py`):

1. Parse the solution. If it is **not a valid permutation of `1..n`** (wrong length, duplicate,
   out-of-range id, or unparseable), the score is **`0`** (the ALE-Bench feasibility floor).
2. Otherwise simulate the tour from the depot at `t = 0`. For each visited node `v` in order:
   `arrival = depart_prev + dist(prev, v)`; `lateness = max(0, arrival - l_v)`; the courier departs at
   `depart = max(arrival, e_v)`. Accumulate `total_distance` and `total_lateness`.
   `solver_cost = total_distance + lambda * total_lateness`.
3. Compute `edf_cost`, the cost of the **earliest-deadline-first (EDF)** ordering (customers sorted by
   `l_i`, then `e_i`, then id) — the trivial baseline.
4. Report `score = edf_cost / solver_cost` (clamped to `[0, +inf)`). EDF itself scores `1.0`; any tour
   cheaper than EDF scores `> 1.0`; **lower cost means higher score.**

**Instances** are produced by `verify/gen.py` from an integer `seed`. It places `n` customers uniformly
in the `1000 x 1000` grid and a depot, then assigns each customer a time window whose centre is spread
across a plausible time horizon (`~0.9 * sqrt(n) * 1000`) and only weakly correlated with the customer's
distance from the depot, so the windows genuinely conflict with the geographically shortest route. The
generator, scorer, and seed set are frozen; the only lever is the visit order the solver emits.

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a feasible
permutation to stdout. The scaffold below establishes the I/O and a valid baseline; the heuristic goes
where marked.

```cpp
#include <bits/stdc++.h>
using namespace std;

int N;
double LAMBDA, DX, DY;            // depot
vector<double> X, Y, E, L;       // 1-indexed node data

static inline double px(int i){ return i==0?DX:X[i]; }
static inline double py(int i){ return i==0?DY:Y[i]; }
static inline double dist(int a,int b){
    double dx=px(a)-px(b), dy=py(a)-py(b); return sqrt(dx*dx+dy*dy);
}

int main(){
    if(!(cin >> N)) return 0;
    cin >> LAMBDA >> DX >> DY;
    X.assign(N+1,0); Y.assign(N+1,0); E.assign(N+1,0); L.assign(N+1,0);
    for(int i=1;i<=N;i++) cin >> X[i] >> Y[i] >> E[i] >> L[i];

    // A feasible baseline: visit 1..n in order (always a valid permutation).
    vector<int> tour(N);
    iota(tour.begin(), tour.end(), 1);

    // TODO: heuristic. Build a time-aware tour and improve it with Or-opt /
    // 2-opt under simulated annealing, re-propagating arrival times only from
    // the first changed position (forward time-propagation cache).

    for(int i=0;i<N;i++){ if(i) cout << ' '; cout << tour[i]; }
    cout << "\n";
    return 0;
}
```
