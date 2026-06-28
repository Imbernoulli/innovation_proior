# Vehicle Dispatch over Time

## Research question

A ride-hailing fleet of `V` vehicles operates on an `L × L` integer grid over a
discrete horizon `[0, T]` (time measured in ticks). During the horizon `N` ride
requests appear. Request `i` has a **pickup** cell `(px_i, py_i)`, a **dropoff**
cell `(qx_i, qy_i)`, a **release** time `r_i` (the earliest tick at which the
ride can be picked up) and an **expiry** time `e_i` (the latest tick at which the
pickup may *start*). Travel time between two cells is the **Manhattan distance**
`|Δx| + |Δy|` in ticks.

A vehicle that is free at cell `(vx, vy)` at time `vt` can serve request `i` if,
after driving empty to the pickup, it can *start* the pickup no later than the
expiry and *finish* the drop-off no later than the horizon:

```
start  = max( vt + manhattan((vx,vy), pickup_i),  r_i )
        must satisfy   start  <= e_i
finish = start + manhattan(pickup_i, dropoff_i)
        must satisfy   finish <= T
```

On serving the ride the vehicle becomes free again at the **drop-off cell** at
time `finish`. Each request may be served at most once; a vehicle serves its
assigned rides one after another.

**Objective: maximize the number of fulfilled rides.** This is a dynamic /
online assignment problem (a temporal generalization of bipartite matching):
requests and vehicle availabilities both move in time, so a global optimum is
NP-hard and there is no closed-form answer — the judge scores the count of rides
actually completed.

## Input / output contract

- **Input (stdin), the instance.**
  - Line 1: three integers `V N T` with `5 ≤ V ≤ 20`, `150 ≤ N ≤ 600`,
    `400 ≤ T ≤ 1000`.
  - Next `V` lines: integers `sx sy` (`0 ≤ sx, sy < L`, with `L = 200`) — the
    start cell of vehicle `v` (vehicles are 0-indexed in input order).
  - Next `N` lines: integers `px py qx qy r e` — request `i` (0-indexed in input
    order), with `0 ≤ px,py,qx,qy < L`, `0 ≤ r ≤ e ≤ T`.
- **Output (stdout), the solution.**
  - Line 1: an integer `M` — the number of dispatch assignments (`0 ≤ M ≤ N`).
  - Next `M` lines: two integers `v i` — "vehicle `v` serves request `i`".
    These lines are replayed **in output order**; each vehicle serves its own
    listed requests sequentially. Every request id may appear **at most once**.
- **Time limit:** 2 seconds wall-clock. **Memory:** 256 MB.

## Background

Two reference policies frame the problem before committing to one.

- **Myopic greedy ("nearest free vehicle grabs the request").** Process requests
  in release order; give each to whichever currently-feasible vehicle can finish
  it soonest, else skip. It is `O(N·V)` and always feasible, but it commits one
  ride at a time without seeing the others competing for the same vehicles. It
  routinely lets one vehicle take a far request that an idle vehicle could have
  served more cheaply, stranding capacity and missing later rides — the classic
  failure of single-item greedy on an assignment problem.

- **Offline "assign everything at once".** Treat the whole horizon as one giant
  bipartite graph (every vehicle-state vs. every request) and solve a maximum
  matching. But a vehicle's reachable set depends on *which* earlier rides it
  took and *when* it finished them, so the graph is not static — there is no
  single fixed cost matrix, and the exact temporal version is NP-hard.

The open question is how to capture the greedy's speed and feasibility-by-
construction while making the *batched, capacity-aware* decisions that the
myopic rule cannot.

## Evaluation settings

- **Scoring (higher is better).** The judge replays the `M` assignment lines in
  output order, maintaining each vehicle's running `(cell, free-time)` state.
  For a line `v i` the vehicle's `start` and `finish` are computed by the
  formulas above; the line is **valid** iff `v ∈ [0,V-1]`, `i ∈ [0,N-1]`, `i`
  was not already used, `start ≤ e_i`, and `finish ≤ T`. The solution is
  **feasible** iff every line is valid and the announced `M` matches. Then

  ```
  score = 0    if the solution is infeasible (any invalid line, duplicate id,
               out-of-range id, malformed/short output, or M out of [0, N])
  score = M    otherwise  (the number of fulfilled rides).
  ```

  A single infeasible assignment **floors the score to exactly 0**: the solver
  must only list rides it can actually complete. `M = 0` ("serve nobody") is
  feasible and scores 0, so any positive score already beats the empty output.

- **Instances.** A frozen generator (`gen.py`, parameter: integer seed) draws
  `V`, `N`, `T` from fixed size classes, scatters a handful of spatial demand
  hotspots plus uniform noise, samples each pickup/dropoff from that mixture, and
  staggers releases over the whole horizon with a mix of tight / medium / loose
  expiry windows so that **not every request is servable** — the dispatcher must
  choose. Everything is a deterministic function of the seed. We report the mean
  fulfilled-ride count over a fixed seed set (seeds `1..20`), each policy run on
  the same instances. The trivial baseline is the myopic greedy above.

## Code framework

A single self-contained C++17 program that reads the instance on stdin and
writes a feasible solution on stdout within the time budget.

```cpp
#include <bits/stdc++.h>
using namespace std;

static inline int manh(int ax,int ay,int bx,int by){return abs(ax-bx)+abs(ay-by);}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int V, N, T;
    if (!(cin >> V >> N >> T)) return 0;
    vector<int> vx(V), vy(V);
    for (int i = 0; i < V; i++) cin >> vx[i] >> vy[i];
    vector<array<int,6>> req(N);          // px py qx qy r e
    for (int i = 0; i < N; i++)
        for (int k = 0; k < 6; k++) cin >> req[i][k];

    // A feasible solution is ANY set of distinct, individually-completable rides
    // (M = 0 is always feasible). Start from the empty dispatch so we can always
    // print something legal.
    vector<pair<int,int>> out;            // (vehicle, request)

    // TODO: heuristic. Fill `out` with (vehicle, request) assignments that the
    // judge can replay feasibly, maximizing the count -- e.g. an event-driven
    // rolling-horizon Hungarian assignment of free vehicles to a small window of
    // the most-urgent reachable pending requests, carrying vehicle state over.

    cout << out.size() << "\n";
    for (auto& pr : out) cout << pr.first << " " << pr.second << "\n";
    return 0;
}
```
