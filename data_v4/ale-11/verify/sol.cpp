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
