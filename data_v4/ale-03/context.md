# Drone Courier under Time Windows

## Research question

A single autonomous courier drone is stationed at a central depot. Over a fixed shift it can fly out,
visit customer requests, and return — but every request is only valid during a narrow time window, and
the shift ends at a hard horizon. There are far more requests than the drone can possibly reach, so the
question is *which* requests to serve and *in what order*.

Concretely: the depot is request id `0`, fixed at the centre of a `1000 x 1000` grid. There are `N`
requests; request `i` sits at integer coordinate `(x_i, y_i)`, has a time window `[r_i, d_i]`, and a
service duration `s_i`. The drone leaves the depot at time `0`, flies between points at unit speed (a
travel time equal to the Euclidean distance, rounded **up** to an integer), and must be back at the depot
by horizon `T`. When it reaches request `i` it begins service; if it arrives before `r_i` it waits until
`r_i`, and the **start** of service must satisfy `start <= d_i`. The goal is to **maximize the number of
requests served** in one feasible out-and-back tour.

This is the prize-collecting / orienteering version of the vehicle routing problem with time windows: an
NP-hard selection-plus-sequencing problem with no closed-form optimum, judged by a continuous count.

## Input / output contract

**Input (stdin)** — one instance:

```
N T
x_1 y_1 r_1 d_1 s_1
x_2 y_2 r_2 d_2 s_2
...
x_N y_N r_N d_N s_N
```

- `N` requests, horizon `T`. The depot (id `0`) is implicit at `(500, 500)` with no window and no service.
- `0 <= x_i, y_i <= 1000`, `0 <= r_i <= d_i < T`, `s_i >= 0`.

**Output (stdout)** — one solution:

```
K
v_1 v_2 ... v_K
```

- `K` is the number of requests served; the second line lists their ids (`1..N`) in visit order.
- `K = 0` is allowed (serve nobody) and is a **feasible** solution that scores `0`.
- Each id appears at most once.

**Time limit:** ~2 seconds of wall time inside the solver. **Memory:** 256 MB.

## Background

The drone must pick a subset *and* an order; the two decisions are coupled through the windows, which is
what makes a one-shot greedy weak. Travel time from `p` to `q` is `ceil(euclidean(p, q))`; arriving early
forces a wait, so the timeline is monotone non-decreasing along a tour. Three families of approach are on
the table before committing:

- **Order-then-take greedy.** Sort requests by some key (input order, deadline, nearest-neighbour) and
  append each to the end of the route whenever it keeps the tour feasible. Trivial, `O(N log N)`; the open
  question is how much served-count it leaves on the table by never reconsidering a placement.
- **Insertion construction.** Start from the empty tour and repeatedly insert the "best" still-unserved
  request at its cheapest feasible position anywhere in the route — a classic VRPTW construction that, with
  a regret rule, chooses *which* request to commit to next, not just where.
- **Ruin-and-recreate (Large Neighborhood Search).** Build a route, then iterate: tear out a batch of
  requests (the most "expensive" stops, or a spatially/temporally related cluster) and greedily reinsert
  every unserved request. This is the established strong metaheuristic for VRPTW-style problems; the open
  question is the removal rule and how to test each candidate insertion in `O(1)`.

## Evaluation settings

**Scoring rule.** A deterministic scorer replays the route from the depot with the *exact* travel-time
rule `ceil(euclidean - 1e-9)`: it accumulates time edge by edge, waits to `r_i` on early arrival, **fails
the solution to score `0`** if any service start exceeds `d_i`, applies `s_i`, and finally adds the return
trip to the depot — failing to `0` if the return arrival exceeds `T`. It also fails to `0` on any malformed
output: an id out of `[1, N]`, a duplicate id, or fewer ids than the announced `K`. Otherwise the score is
the number of served requests `K`. This is the **feasibility -> 0 floor**: an infeasible or invalid route
scores nothing, so the solver must *always* emit a route it has itself verified by replay.

**Instances.** The generator (`gen.py`, parameter: integer seed) draws `N in {200..800}` and
`T in {2000..4000}`, places ~80% of requests in 3–7 Gaussian "delivery zones" plus a uniform sprinkle, and
gives each request a service time `s in [2,12]` and a window whose width is a mix of tight / medium / loose
spans scattered across the horizon. The depot is fixed at the grid centre. There are always far more
requests than can be served, so the solver must *choose*.

**Reported metric.** Over a fixed seed set (seeds `1..20`) we report the mean served-count, compared on the
same instances against a trivial baseline (append-in-input-order, kept feasible). The strong solver must
strictly beat the baseline on the mean.

## Code framework

A single self-contained C++17 program reading the instance and writing a feasible route. The depot is id
`0` at `(500,500)`; printing `0` then a blank line (serve nobody) is always a legal fallback.

```cpp
#include <bits/stdc++.h>
using namespace std;
static const int L = 1000;
struct Req { long long x, y, r, d, s; };
static int N; static long long T; static vector<Req> req;

// travel time MUST match the scorer: ceil(euclidean - 1e-9)
static inline long long travel(long long ax,long long ay,long long bx,long long by){
    double dx=ax-bx, dy=ay-by; double dist=sqrt(dx*dx+dy*dy);
    long long c=(long long)ceil(dist-1e-9); return c<0?0:c;
}

int main(){
    if(!(cin>>N>>T)){ cout<<0<<"\n"; return 0; }
    req.assign(N+1, Req{});
    req[0] = Req{L/2, L/2, 0, 0, 0};
    for(int i=1;i<=N;i++) cin>>req[i].x>>req[i].y>>req[i].r>>req[i].d>>req[i].s;

    // TODO: heuristic -- choose a subset of requests and an order forming a
    // feasible out-and-back tour (respect windows + horizon), maximizing count.
    vector<int> route; // ids in visit order; empty route is always feasible

    cout << route.size() << "\n";
    for(size_t i=0;i<route.size();i++) cout<<route[i]<<" \n"[i+1==route.size()];
    if(route.empty()) cout << "\n";
    return 0;
}
```
