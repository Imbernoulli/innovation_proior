# Capacitated Multi-Vehicle Routing

## Research question

A depot and `n` customers sit on a plane. A fleet of `K` identical vehicles, each
with capacity `Q`, must serve every customer. Each vehicle starts at the depot,
visits a subset of customers in some order, and returns to the depot; the sum of
the demands on a vehicle's route may not exceed `Q`, and every customer must be
served by exactly one vehicle. The task is to partition the customers into `K`
routes and order each route so as to **minimise the total travel distance** —
the sum over all vehicles of the closed-tour length `depot -> c1 -> ... -> ck ->
depot`.

This is the classical Capacitated Vehicle Routing Problem (CVRP). It is NP-hard:
it contains both a bin-packing flavour (the customers must be packed into
capacity-`Q` vehicles) and a TSP flavour (each route must be ordered well). There
is no closed-form optimum at this scale, so the problem is judged by a continuous
distance score, with any infeasible output floored to zero.

## Input / output contract

Input (stdin):

```
n K Q
depot_x depot_y
x_1 y_1 d_1
x_2 y_2 d_2
...
x_n y_n d_n
```

- `n` is the number of customers (here `120 <= n <= 200`), `K` the number of
  vehicles (`6 <= K <= 12`), `Q` the per-vehicle capacity.
- The second line is the depot's integer coordinates (conceptually index `0`).
- The next `n` lines give customer `i` (for `i = 1..n`): integer coordinates
  `x_i, y_i` in `[0, 1000]` and a positive integer demand `d_i`.
- The instance always admits a feasible assignment: `sum(d_i) <= K * Q` and
  `max(d_i) <= Q`.

Output (stdout): exactly `K` lines, one per vehicle. Line `k` lists the customer
ids on vehicle `k`'s route, in visit order, space-separated. A vehicle that
serves nobody emits an empty line. Each route is implicitly closed at the depot;
do not print the depot. Example of a single route line: `3 17 5 42`.

- Time limit: 2 seconds. Memory: 256 MB.

## Background

CVRP is one of the most studied problems in operations research, and a
well-understood heuristic ladder exists:

- **Sweep construction.** Sort customers by polar angle around the depot and cut
  the angular sequence into routes greedily under the capacity. It produces a
  feasible, spatially coherent solution in `O(n log n)`, but the cut points are
  blind to load balance, so a sweep often leaves one route overlong and an
  adjacent one nearly empty.
- **Clarke-Wright savings.** Start with one route per customer and repeatedly
  merge the two route endpoints with the largest "saving"
  `s(i,j) = d(0,i) + d(0,j) - d(i,j)` whenever the merged demand fits. This is
  the standard *construction* baseline and is the normaliser used by the scorer.
- **Local search.** Refine a starting solution with neighbourhood moves: intra-
  route 2-opt and Or-opt (segment relocation) tidy each tour; cross-route
  **relocate** and **swap** move customers between vehicles; and **2-opt\*** —
  the move that swaps the *tail segments* of two routes at a cut point — repairs
  the global load imbalance a sweep leaves behind. The decisive engineering fact
  is that each such move is checked in `O(1)`: capacity feasibility is a lookup
  against per-route running loads, and the distance change touches at most four
  edges. That `O(1)` incremental evaluation is what lets a metaheuristic
  (simulated annealing here) try millions of moves inside the time budget.

The strongest simple-to-state approach is therefore **sweep (or savings)
construction warm-starting a simulated-annealing local search whose move set
includes cross-route relocate, swap, and 2-opt\***, with `O(1)` capacity checks
and 4-edge distance deltas. That is the method implemented here.

## Evaluation settings

A fixed seed set (seeds `1..20`) of instances is generated as described below.
For each instance the scorer reads the solution and checks feasibility:

1. exactly `K` route lines are present (trailing blank lines are tolerated, and
   missing trailing routes are treated as empty);
2. every printed id is a valid customer `in [1, n]`;
3. every customer `1..n` appears **exactly once** across all routes (served once,
   no missing, no duplicate);
4. every route's total demand is `<= Q`.

If any check fails, the **score is 0** (the feasibility floor). Otherwise the raw
objective is the total closed-tour distance

```
D = sum over routes r of ( d(0, r[0]) + sum_i d(r[i], r[i+1]) + d(r[last], 0) ),
```

with Euclidean edge lengths. Lower `D` is better, so to make a continuous
"higher-is-better" score we normalise against the Clarke-Wright savings baseline
distance `D_cw` computed by the scorer on the same instance:

```
score = D_cw / D          (and score = 0 if the solution is infeasible).
```

A score of `1.0` ties Clarke-Wright; `score > 1` means the solution is strictly
shorter than the savings baseline. The reported metric is the mean score over the
seed set. (The scorer also exposes the raw distance `D` via the `ALE17_RAW=1`
environment variable, used during development to compare absolute distances.)

How instances are made: `n in [120,200]` customers are placed as a mixture of a
few 2D-Gaussian shoals (about 70%) and uniform scatter (about 30%) on the
`[0,1000]^2` grid; the depot is at the centre or a random point. Demands are
integers in `[1,30]`. The capacity `Q` is set so the fleet runs at ~80-92%
utilisation (`K*Q` slightly above `sum(d_i)`), making the packing tight but
always feasible (`max(d_i) <= Q`). Everything is deterministic in the seed.

## Code framework

A single self-contained C++17 program reading stdin and writing stdout. The
scaffold below reads the instance and prints a trivially feasible sweep; the
heuristic goes where marked.

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, K; long long Q;
vector<double> X, Y; vector<long long> dem;

static inline double Dist(int a, int b){
    double dx=X[a]-X[b], dy=Y[a]-Y[b]; return sqrt(dx*dx+dy*dy);
}

int main(){
    if(!(cin>>n>>K>>Q)) return 0;
    X.assign(n+1,0); Y.assign(n+1,0); dem.assign(n+1,0);
    cin>>X[0]>>Y[0];
    for(int i=1;i<=n;i++) cin>>X[i]>>Y[i]>>dem[i];

    // A feasible starting solution: sweep by angle, fill vehicles under Q.
    vector<vector<int>> route(K);
    vector<long long> load(K,0);
    vector<int> order(n);
    for(int i=0;i<n;i++) order[i]=i+1;
    sort(order.begin(), order.end(), [&](int a,int b){
        return atan2(Y[a]-Y[0],X[a]-X[0]) < atan2(Y[b]-Y[0],X[b]-X[0]);
    });
    int v=0;
    for(int c: order){
        while(v<K && load[v]+dem[c]>Q) v++;
        if(v>=K) v=K-1;
        route[v].push_back(c); load[v]+=dem[c];
    }

    // TODO: warm-started simulated annealing with cross-route relocate / swap /
    //       2-opt* and intra-route 2-opt / Or-opt, all with O(1) capacity checks
    //       and 4-edge distance deltas; keep the best feasible solution seen.

    for(int k=0;k<K;k++){
        for(size_t i=0;i<route[k].size();i++){
            if(i) cout<<' ';
            cout<<route[k][i];
        }
        cout<<"\n";
    }
    return 0;
}
```
