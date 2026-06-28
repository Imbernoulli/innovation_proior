# Reasoning: Vehicle Dispatch over Time (ale-11)

## Reading the objective until it is unambiguous

I have `V` vehicles on a `200 × 200` grid and `N` ride requests arriving over a
horizon `[0, T]`. Each request `i` carries a pickup cell, a dropoff cell, a
release time `r_i`, and an expiry `e_i`. Travel costs Manhattan distance in
ticks. The thing I am graded on is brutally simple: **how many rides get
fulfilled**. Not revenue, not distance — a raw count. That simplicity is a trap,
because it makes me want to grab rides greedily, and grabbing rides greedily is
exactly what wastes the fleet.

Let me pin down the feasibility rule precisely, because the score floor lives
there. A vehicle free at `(vx, vy)` at time `vt` can serve request `i` if it
drives empty to the pickup and *starts* the pickup no later than `e_i`:

```
start  = max( vt + manhattan((vx,vy), pickup_i), r_i )   and   start  <= e_i
finish = start + manhattan(pickup_i, dropoff_i)          and   finish <= T
```

After the ride the vehicle is free at the dropoff cell at time `finish`. So a
vehicle's *future* reachable set depends entirely on the history of rides it
took and when each one ended. There is no fixed "vehicle v can serve request i"
table — it is a function of the path so far. That is what makes this temporal,
and that is why a one-shot offline matching does not literally apply.

The output is a list of `(vehicle, request)` lines. The judge replays them in
output order, threading each vehicle's `(cell, free-time)` state through, and a
**single** infeasible line (missed window, can't finish by `T`, duplicate id,
out-of-range id) floors the whole score to 0. So whatever I emit, every line has
to survive that replay. `M = 0` is feasible and scores 0; the bar to clear is
literally "serve at least one more ride than the trivial policy, with no broken
line."

## Step 1 — get a feasible baseline that can never score 0

Before any cleverness I want a policy that *always* produces a legal output, so
the floor is never my problem. The obvious one: walk requests in release order;
for each, look at every vehicle, compute its `finish` if it took this ride, and
give the ride to the vehicle that finishes soonest (skip if none can). Each
committed ride updates that vehicle's `(cell, free-time)`. Because I only ever
commit a ride I just verified feasible, and I never reuse a request id, the
output is feasible by construction. This is the myopic greedy, and it is my
baseline (`baseline.py`). On seed 1 it serves 23 rides — fine as a floor, and
clearly nonzero, so the feasibility-floor risk is handled.

But I can already see why it is weak. It decides one request at a time. Suppose
two requests A and B are both pending and two vehicles `u`, `w` are both free.
Greedy sees A first, hands it to whichever vehicle finishes A soonest — say `u`.
Then it sees B and hands it to the better of the remaining vehicles. But maybe
the globally better pairing is `u→B, w→A`: greedy never considers that, because
it froze `u→A` the instant it processed A. With a whole batch of free vehicles
and a whole batch of pending requests competing, "process one request, lock in
the locally-best vehicle" leaves cardinality on the table. This is the textbook
gap between sequential greedy and an assignment solve.

## Step 2 — name the real structure: it is assignment, but it moves

At any single instant, the question "which free vehicle should take which
pending request" is a **bipartite assignment** problem: free vehicles on one
side, pending requests on the other, edge weight = the cost of that pairing.
Solved exactly, assignment beats greedy precisely in the `u↔B, w↔A` swap case,
because the Hungarian algorithm (Kuhn–Munkres) finds the globally optimal
matching of the batch, not a left-to-right locking of it.

The catch is that the problem is not one instant — it is the whole horizon, and
the bipartite graph keeps changing as vehicles finish rides and as new requests
release. The exact temporal assignment (decide *all* rides and their order to
maximize count) is NP-hard; there is no single cost matrix to feed Hungarian.

So I do the standard relaxation that makes online assignment tractable:
**rolling horizon**. I chop time into decision epochs. At each epoch I freeze a
snapshot — the vehicles free *right now* and the requests pending *right now* —
solve that one static assignment optimally with Hungarian, commit it, advance
the clock, and carry every vehicle's updated state into the next epoch. The
horizon relaxation turns one intractable global problem into a sequence of small
optimal matchings. This is exactly the candidate's named lever: *rolling-horizon
Hungarian on a small window + greedy carry-over.*

## Step 3 — what does "Hungarian" optimize, and how do I make it maximize count?

Here is a subtlety I have to get right. Hungarian minimizes total cost of a
*perfect* matching. But I do not want minimum cost — I want **maximum number of
rides served**, with cost only as a tie-break. If I naively set
`cost = finish` and minimize, Hungarian might leave a vehicle idle (matched to a
"do nothing" slot) to lower total finish time, *reducing* the count. That is the
opposite of my objective.

The fix is a lexicographic encoding. I give every feasible pairing a reward of
`-BIG` (a huge negative number, `BIG = 4e9`) plus the actual `finish` as a tiny
additive tie-break:

```
cost(v, i) = -BIG + finish     if (v, i) is feasible
cost(v, i) = 0                  if infeasible, or padding
```

Because `BIG` dwarfs any possible `finish` (finishes are ≤ `T ≤ 1000`, while
`BIG = 4·10^9`), the optimizer first **maximizes the number of feasible pairs
chosen** — each one is worth `-BIG`, so matching one more real pair always beats
any tie-break difference — and only then, among solutions with the same maximum
cardinality, picks the one with the smallest total `finish` (vehicles free
earliest, which is good for future epochs). Infeasible pairs and padding cost 0,
i.e. "this vehicle takes no ride." I pad the matrix to square `n = max(#free,
#cand)` so the classic `O(n^3)` Hungarian applies, and after solving I only
commit columns whose matched cost is `< 0` (a genuine rewarded pairing); a `≥ 0`
match means "matched to a do-nothing slot," which I skip.

## Step 4 — keep each Hungarian solve cheap: the window

If at some epoch dozens or hundreds of requests are pending, a Hungarian over
all of them is `O(pending^3)` and could blow the time budget. But I do not need
all of them in one matching — the free vehicles can take at most `#free ≤ V ≤ 20`
of them this epoch anyway. So I bound the candidate set to a **window** of the
`WINDOW = 48` most-urgent reachable pending requests, urgency = earliest expiry
`e_i` (earliest-deadline-first: the rides about to expire are the ones I cannot
defer). 48 comfortably exceeds the at-most-20 vehicles, so I never starve the
matching, and `48^3 ≈ 10^5` per epoch is trivial. Requests outside the window
are not discarded — they sit in `pending` and reappear (likely inside the
window) at later epochs. That is the carry-over again.

## Step 5 — event-driven clock so I do not waste epochs

Stepping the clock by 1 tick each epoch would mean up to `T = 1000` epochs of
mostly-no-op work. Instead I run an **event-driven** clock: after committing an
epoch, I jump `clock` to the next *interesting* time — the earliest vehicle
free-time strictly greater than now, or the next request release, whichever is
sooner. Those are the only moments at which the snapshot can change (a vehicle
becomes available, or a new request appears). Between events nothing changes, so
there is nothing to decide.

Two guards I know I will need:

- **Expiry pruning.** Before each epoch I drop permanently-unservable requests: a
  request is dead if even the *best-case* vehicle (the min over all vehicles of
  `max(ft[v] + dist_to_pickup, r_i)`) cannot start it by `e_i`, or cannot finish
  by `T`. Dropping these keeps `pending` small and stops me from carrying corpses
  forever.
- **Termination.** If all vehicles are free at `≤ clock` and there are no more
  releases but `pending` is still non-empty (everything left is currently
  unreachable but not yet provably dead), I must still make progress or I loop
  forever. I step `clock` by 1 and bail out once `clock > T`. Combined with the
  expiry pruning, this guarantees the loop ends.

## Step 6 — implement, then the debugging episode

I wrote the event loop: pull releases ≤ clock into `pending`; prune expired;
gather free vehicles; build the urgent reachable candidate window; build the
padded cost matrix with the `-BIG + finish` encoding; run Hungarian; commit the
rewarded matches (updating `served`, the vehicle's cell and free-time, and
appending to the output); compact `pending`; advance the clock to the next
event. Compile with `-O2 -std=c++17`.

**First real bug — the score came back as 0 on my first hand-test of a tiny
crafted instance.** I had been sloppy about *commit order versus replay order*.
The judge replays my output lines top-to-bottom, threading each vehicle's state.
My first cut sorted the final output by request id "to look tidy" before
printing. That reordered a single vehicle's rides out of chronological order, so
the replay computed `start` for a later-in-time ride using an *earlier* vehicle
state — and a downstream ride then looked like it finished after `T`, flooring
the score to 0. The fix is to **emit assignments in commit order, untouched**.
Because epochs only ever move the clock forward, the rides I commit for any one
vehicle are appended in strictly increasing finish-time order, which is exactly
the chronological order the replay needs. I deleted the tidy-sort. After that the
crafted case scored correctly.

**Second issue — I want to be certain the Hungarian never makes me emit an
infeasible line.** Two ways that could happen: (a) the optimizer matches a
vehicle to a column whose cost is `≥ 0` (a do-nothing / infeasible slot) — I skip
those by the `cost[a][b] >= 0` guard; (b) floating/edge rounding makes a "feasible"
pair actually miss its window. To be defensive I **re-validate** every pairing at
commit time (`st > e` or `st + ride > T` → skip) even though by construction it
should already hold. This guard costs nothing and makes "the solver can never
emit an infeasible line" a structural guarantee, not a hope. With it, the
feasibility floor is simply unreachable from a bug.

## Step 7 — self-verify on the seed set

I compiled and ran seeds `1..20`, scoring my solver and the myopic-greedy
baseline on the *same* instances, and checking the feasibility floor on adversarial
outputs (empty, duplicate id, out-of-range vehicle/request, short/garbled,
`M > N`) — every adversarial case scores 0 exactly as intended, and `M = 0`
scores 0 feasibly.

The seed-set numbers (fulfilled rides; higher is better):

```
seed  V/N/T          sol  base
1     8/400/400       37   23
2     20/300/1000    150  134
3     15/400/400      57   38
5     8/400/400       41   26
7     8/300/600       62   39
9     10/200/800      85   66
10    12/500/500      76   40
11    20/600/600     150   97
13    15/600/400     107   61
16    15/600/500      80   48
18    10/600/800     110   60
19    15/500/600     119   75
```

Across all 20 seeds: **every output is feasible (score > 0), the solver beats the
baseline on every single seed (20/20, no ties, no losses)**, mean `85.8` vs
`61.75` — about a **39% lift** in fulfilled rides. The worst single-instance
runtime is `15 ms`, three orders of magnitude under the 2-second budget, so there
is enormous headroom (I could widen `WINDOW`, or add a second pass, but the
batched-optimal matching already captures the structural win and the requirement
is simply "beat the baseline robustly," which it does decisively).

Why the lift is so consistent: every place the greedy locks `u→A` while `u→B,
w→A` would have served the same two rides cheaper, the per-epoch Hungarian takes
the better pairing, freeing vehicles earlier and letting them catch one or two
*additional* downstream rides that the greedy's stranded capacity would have
missed. On the denser instances (large `N`, moderate `V`) where vehicles are the
scarce resource and the batch competition is fiercest — seeds 10, 11, 13, 16, 18 —
the gap is largest (e.g. 76 vs 40, 150 vs 97), exactly as the structure predicts.

## What the innovation bought, in one line

A myopic "nearest free vehicle grabs the request" greedy is sequential and
strands capacity; replacing the per-instant decision with an **optimal
maximum-cardinality assignment** (Hungarian, with a `-BIG + finish` lexicographic
cost so it maximizes count then minimizes finish), run on a bounded urgent
**window** over an **event-driven rolling horizon** that carries vehicle state
forward, turns the intractable global temporal-matching problem into a sequence
of cheap optimal matchings — and that batched optimality is what lifts fulfilled
rides ~39% over the greedy on every seed.

## Final solver

```cpp
// Vehicle Dispatch over Time (ale-11)
//
// Objective: maximize the number of fulfilled ride requests. V vehicles live on
// an L x L grid; each of N requests has a pickup cell, a dropoff cell, a release
// time r and an expiry e. A free vehicle at (vx,vy,vt) can serve request i iff
//   start  = max(vt + manhattan((vx,vy),pickup_i), r_i) <= e_i      and
//   finish = start + manhattan(pickup_i,dropoff_i)      <= T;
// after that the vehicle is free at the dropoff at time `finish`.
//
// We output a list of (vehicle, request) assignments; the judge replays them in
// output order and counts fulfilled rides (one infeasible line floors to 0).
//
// INNOVATION -- rolling-horizon Hungarian + greedy carry-over.
// The fully dynamic assignment is online/NP-hard, and a pure greedy "nearest
// free vehicle grabs nearest request" wastes capacity: it lets one vehicle take
// a far request another idle vehicle could have taken more cheaply, and it
// commits to expiring rides in a myopic order. Instead we advance an event-
// driven clock; at each epoch we take the currently-free vehicles and a small
// window of the most-urgent reachable pending requests, and solve a min-cost
// MAXIMUM-CARDINALITY assignment (Hungarian / Kuhn-Munkres) of vehicles to that
// window -- cardinality first (serve as many as possible now), finish-time as
// the tie-break (leave vehicles free as early as possible). Requests not taken
// this epoch are carried over to later epochs. Bounding the window keeps each
// Hungarian solve cheap (O(k^3) on a few dozen items) while the horizon
// relaxation turns the global problem into a sequence of tractable matchings.

#include <bits/stdc++.h>
using namespace std;

struct Req {
    int px, py, qx, qy, r, e;
    int ride;   // manhattan(pickup, dropoff), precomputed
};

static inline int manh(int ax, int ay, int bx, int by) {
    return abs(ax - bx) + abs(ay - by);
}

// O(n^3) Hungarian (min-cost perfect assignment) on an n x n cost matrix `a`
// (1-indexed internally). Returns, for each column j, the row assigned to it in
// p[j]; rows/cols are 0-indexed in the returned mapping (p[j] = assigned row).
// Costs are long long; we will pad to a square matrix with large costs.
static vector<int> hungarian(const vector<vector<long long>>& a, int n) {
    const long long INF = LLONG_MAX / 4;
    vector<long long> u(n + 1, 0), v(n + 1, 0);
    vector<int> p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        vector<long long> minv(n + 1, INF);
        vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = -1;
            long long delta = INF;
            for (int j = 1; j <= n; j++) {
                if (!used[j]) {
                    long long cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= n; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    // p[j] = row assigned to column j (1-indexed row, 1-indexed col)
    vector<int> colRow(n, -1);   // colRow[j-1] = row index (0-based) for col j
    for (int j = 1; j <= n; j++) colRow[j - 1] = p[j] - 1;
    return colRow;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int V, N, T;
    if (!(cin >> V >> N >> T)) return 0;

    vector<int> vx(V), vy(V);
    for (int i = 0; i < V; i++) cin >> vx[i] >> vy[i];

    vector<Req> req(N);
    for (int i = 0; i < N; i++) {
        cin >> req[i].px >> req[i].py >> req[i].qx >> req[i].qy >> req[i].r >> req[i].e;
        req[i].ride = manh(req[i].px, req[i].py, req[i].qx, req[i].qy);
    }

    // Vehicle running state (carry-over across epochs).
    vector<int> cx = vx, cy = vy;        // current cell
    vector<long long> ft(V, 0);          // current free-time

    // Requests sorted by release time, so we can stream them in as time advances.
    vector<int> byRelease(N);
    iota(byRelease.begin(), byRelease.end(), 0);
    sort(byRelease.begin(), byRelease.end(),
         [&](int a, int b){ return req[a].r < req[b].r; });

    vector<char> served(N, 0);           // already assigned
    vector<int> pending;                 // request ids released, not yet served/expired
    int nextRel = 0;                     // pointer into byRelease

    // Output assignments in commit order so the judge can replay them.
    vector<pair<int,int>> out;           // (vehicle, request)
    out.reserve(N);

    const int WINDOW = 48;               // cap on candidates per epoch (Hungarian size)
    // Big penalty so the Hungarian first MAXIMIZES the number of real
    // (feasible) assignments, then minimizes finish time as a tie-break.
    const long long BIG = (long long)4e9;

    // Event-driven epochs. We advance `clock` to the time at which the work for
    // the current epoch is "anchored": the moment the earliest-free vehicle is
    // available (or the next release if all vehicles are busy past that point).
    long long clock = 0;

    while (true) {
        // Pull in all requests released by `clock` into the pending pool.
        while (nextRel < N && req[byRelease[nextRel]].r <= clock) {
            int id = byRelease[nextRel++];
            if (!served[id]) pending.push_back(id);
        }

        // Drop expired requests: a request i is dead if NO vehicle can possibly
        // start its pickup by e_i. Lower bound on start = min over vehicles of
        // ft[v] + manhattan(cell_v, pickup_i), but also >= r_i. If that exceeds
        // e_i, or the ride can't finish by T, it's permanently unservable.
        {
            vector<int> keep;
            keep.reserve(pending.size());
            for (int id : pending) {
                if (served[id]) continue;
                const Req& R = req[id];
                long long best = LLONG_MAX;
                for (int v = 0; v < V; v++) {
                    long long st = ft[v] + manh(cx[v], cy[v], R.px, R.py);
                    if (st < R.r) st = R.r;
                    best = min(best, st);
                }
                if (best <= R.e && best + R.ride <= T) keep.push_back(id);
                // else: permanently unservable -> drop silently
            }
            pending.swap(keep);
        }

        if (pending.empty() && nextRel >= N) break;   // nothing left to do

        // Free vehicles for THIS epoch: those free at or before `clock`. If none
        // are free yet, jump the clock to the earliest vehicle free-time (and/or
        // the next release) and retry.
        vector<int> freeV;
        for (int v = 0; v < V; v++) if (ft[v] <= clock) freeV.push_back(v);

        if (freeV.empty() || pending.empty()) {
            long long nxt = LLONG_MAX;
            for (int v = 0; v < V; v++) if (ft[v] > clock) nxt = min(nxt, ft[v]);
            if (nextRel < N) nxt = min(nxt, (long long)req[byRelease[nextRel]].r);
            if (nxt == LLONG_MAX) break;               // can't advance -> done
            clock = max(clock + 1, nxt);
            continue;
        }

        // Build the candidate window: the most-urgent reachable pending requests.
        // Urgency = expiry e (earliest deadline first); reachable = at least one
        // free vehicle can start the pickup by e.
        vector<int> cand;
        cand.reserve(pending.size());
        for (int id : pending) {
            const Req& R = req[id];
            bool reach = false;
            for (int v : freeV) {
                long long st = ft[v] + manh(cx[v], cy[v], R.px, R.py);
                if (st < R.r) st = R.r;
                if (st <= R.e && st + R.ride <= T) { reach = true; break; }
            }
            if (reach) cand.push_back(id);
        }
        if (cand.empty()) {
            // No free vehicle can serve any pending request right now; advance.
            long long nxt = LLONG_MAX;
            for (int v = 0; v < V; v++) if (ft[v] > clock) nxt = min(nxt, ft[v]);
            if (nextRel < N) nxt = min(nxt, (long long)req[byRelease[nextRel]].r);
            if (nxt == LLONG_MAX) break;
            clock = max(clock + 1, nxt);
            continue;
        }
        sort(cand.begin(), cand.end(),
             [&](int a, int b){ return req[a].e < req[b].e; });
        if ((int)cand.size() > WINDOW) cand.resize(WINDOW);

        int nv = (int)freeV.size();
        int nc = (int)cand.size();
        int n = max(nv, nc);

        // Cost matrix: rows = free vehicles (padded), cols = candidates (padded).
        // Feasible (vehicle v, request c): cost = BIG_negative_reward + finish.
        // We make ASSIGNING a feasible pair strictly better than not assigning by
        // giving each real feasible assignment a reward of -BIG (so the optimizer
        // maximizes #assignments), with +finish as a small tie-break. Infeasible
        // pairs and padding get cost 0 (i.e. "no ride taken").
        vector<vector<long long>> cost(n, vector<long long>(n, 0));
        // precompute feasibility/finish per (vehicle-in-freeV, cand)
        for (int a = 0; a < nv; a++) {
            int v = freeV[a];
            for (int b = 0; b < nc; b++) {
                const Req& R = req[cand[b]];
                long long st = ft[v] + manh(cx[v], cy[v], R.px, R.py);
                if (st < R.r) st = R.r;
                if (st <= R.e && st + (long long)R.ride <= T) {
                    long long finish = st + R.ride;
                    cost[a][b] = -BIG + finish;   // reward for serving + tie-break
                } else {
                    cost[a][b] = 0;               // infeasible -> like "no ride"
                }
            }
        }
        // padding rows/cols already 0.

        vector<int> colRow = hungarian(cost, n);   // colRow[col] = row

        // Commit: for each candidate column matched to a real vehicle row with a
        // feasible (negative-reward) cost, take that ride.
        // Track which vehicles got used this epoch so we don't reuse them.
        for (int b = 0; b < nc; b++) {
            int a = colRow[b];                     // row (free-vehicle index) for col b
            if (a < 0 || a >= nv) continue;        // matched to padding
            if (cost[a][b] >= 0) continue;         // not a feasible (rewarded) pair
            int v = freeV[a];
            int id = cand[b];
            if (served[id]) continue;
            const Req& R = req[id];
            long long st = ft[v] + manh(cx[v], cy[v], R.px, R.py);
            if (st < R.r) st = R.r;
            // re-validate (safety; should always hold)
            if (st > R.e || st + (long long)R.ride > T) continue;
            long long finish = st + R.ride;
            // commit
            served[id] = 1;
            cx[v] = R.qx; cy[v] = R.qy; ft[v] = finish;
            out.emplace_back(v, id);
        }

        // Rebuild pending without the newly served ones (lazy compaction).
        {
            vector<int> keep;
            keep.reserve(pending.size());
            for (int id : pending) if (!served[id]) keep.push_back(id);
            pending.swap(keep);
        }

        // Advance the clock to the next interesting time: the earliest vehicle
        // free-time that is > clock, or the next release. This is the greedy
        // carry-over step -- vehicles roll their state into the next epoch.
        long long nxt = LLONG_MAX;
        for (int v = 0; v < V; v++) if (ft[v] > clock) nxt = min(nxt, ft[v]);
        if (nextRel < N) nxt = min(nxt, (long long)req[byRelease[nextRel]].r);
        if (nxt == LLONG_MAX) {
            // all vehicles free at <= clock and no more releases; if pending is
            // non-empty but nothing got served this epoch we'd loop forever, so
            // step the clock by 1 to guarantee progress / termination.
            if (pending.empty()) break;
            clock = clock + 1;
            if (clock > T) break;
        } else {
            clock = max(clock + 1, nxt);
        }
        if (clock > T) {
            // Past the horizon: nothing more can be picked up. Stop.
            break;
        }
    }

    // Emit the assignments in commit order.
    // (Replay order matters: each vehicle's rides are appended in the order
    //  they were committed, which is exactly chronological per vehicle.)
    string buf;
    buf.reserve(out.size() * 8 + 16);
    buf += to_string((long long)out.size());
    buf += '\n';
    for (auto& pr : out) {
        buf += to_string(pr.first);
        buf += ' ';
        buf += to_string(pr.second);
        buf += '\n';
    }
    cout << buf;
    return 0;
}
```
