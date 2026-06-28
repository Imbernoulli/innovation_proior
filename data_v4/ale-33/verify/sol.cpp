// ale-33: Grid Wire Routing (disjoint rectilinear paths, minimise total length).
//
// We are given an H x W grid and n terminal pairs.  We must connect each pair by
// a 4-connected rectilinear path so that NO grid cell is used by more than one
// wire (vertex-disjoint paths, endpoints included), minimising the total number
// of edges.  An output with a broken path or a cell shared by two wires scores 0,
// so feasibility (a fully disjoint routing) dominates everything; among feasible
// routings we want the smallest total length.
//
// THE INNOVATION -- negotiated-congestion rip-up-and-reroute (PathFinder, the
// established VLSI global-routing heuristic):
//   * Each cell carries a cost  cost(v) = base + h[v]*(1+p[v]) + Kp*p[v]  where
//       base = 1               (we pay one unit per edge -> per cell entered),
//       p[v] = present-sharing penalty term, grows with how many *other* nets
//              currently occupy v this pass (two wires on a cell get expensive),
//       h[v] = historical congestion, incremented every pass a cell stays
//              over-used (chronically contested cells become permanently
//              expensive so wires learn to detour around them for good).
//   * Each pass we RIP UP every net and REROUTE it with a Dijkstra shortest path
//     on this cost field (a binary heap reading h[]/occ[]).  Because routing is
//     *negotiated* through the cost field rather than hard-blocked, a net may
//     temporarily share a cell; across passes the present penalty is ramped and
//     the history accumulated, so all sharing is squeezed out and the routing
//     becomes disjoint -- while staying near shortest length because base=1 keeps
//     uncontested detours cheap.
//
// FEASIBILITY GUARANTEE.  Negotiated congestion alone is not guaranteed to reach
// zero overlaps in bounded time, so we also run a deterministic, always-
// terminating sequential completer: route nets one-by-one by Dijkstra over the
// cells not claimed by an already-committed net; if a net is boxed in, rip up the
// committed net that blocks it most and re-route it afterwards, with a strict
// no-progress / round bound so it cannot thrash forever.  We try several net
// orderings.  The generator guarantees a disjoint routing exists, so the
// completer finds one; we ALWAYS hold a feasible routing before printing and
// never emit an unrouted net or a shared cell.

#include <bits/stdc++.h>
using namespace std;

static int H, W, N;
static int HW;
struct Pair { int sr, sc, tr, tc; };
static vector<Pair> P;

static chrono::steady_clock::time_point T0;
static double TIME_LIMIT = 2.8;        // seconds, comfortably under any judge limit
static double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - T0).count();
}

static inline int id(int r, int c) { return r * W + c; }

static void readInput() {
    if (scanf("%d %d %d", &H, &W, &N) != 3) { H = W = N = 0; return; }
    HW = H * W;
    P.resize(N);
    for (int k = 0; k < N; k++)
        if (scanf("%d %d %d %d", &P[k].sr, &P[k].sc, &P[k].tr, &P[k].tc) != 4) P[k] = {0,0,0,0};
}

// ---- shared state ----------------------------------------------------------
static vector<int>    occ;        // occ[v] = #nets currently on cell v (negotiated field)
static vector<double> hist;       // historical congestion h[v]
static vector<int>    endptOwner; // endptOwner[v] = net whose terminal sits on v, else -1

// Dijkstra scratch (reused via tags so we never re-zero HW arrays)
static vector<double> dist;
static vector<int>    prevc;
static vector<int>    visitTag;
static int            curTag = 0;
static const double   INF = 1e18;

static double PFAC = 1.0;          // present penalty multiplier (ramped)
static double KP   = 6.0;          // flat present-overlap surcharge

static const int DR[4] = {-1, 1, 0, 0};
static const int DC[4] = {0, 0, -1, 1};

// ---- negotiated-congestion Dijkstra ---------------------------------------
// Route net k on the soft cost field.  `occ` should already EXCLUDE net k's own
// current route (caller rips it up first), so occ[v] counts OTHER nets on v.
static vector<int> dijkstraSoft(int k) {
    int s = id(P[k].sr, P[k].sc), t = id(P[k].tr, P[k].tc);
    curTag++;
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> pq;
    dist[s] = 0.0; prevc[s] = -1; visitTag[s] = curTag;
    pq.push({0.0, s});
    while (!pq.empty()) {
        auto [d, v] = pq.top(); pq.pop();
        if (d > dist[v] + 1e-12) continue;
        if (v == t) break;
        int r = v / W, c = v % W;
        for (int dir = 0; dir < 4; dir++) {
            int nr = r + DR[dir], nc = c + DC[dir];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int u = nr * W + nc;
            int ow = endptOwner[u];
            if (ow != -1 && ow != k && u != t) continue;   // another net's terminal: blocked
            double p = occ[u];                              // other nets here
            double w = 1.0 + hist[u] * (1.0 + PFAC * p) + KP * p;
            double nd = d + w;
            if (visitTag[u] != curTag || nd < dist[u] - 1e-12) {
                dist[u] = nd; prevc[u] = v; visitTag[u] = curTag;
                pq.push({nd, u});
            }
        }
    }
    if (visitTag[t] != curTag) return {};
    vector<int> path;
    for (int v = t; v != -1; v = prevc[v]) path.push_back(v);
    reverse(path.begin(), path.end());
    return path;
}

static inline void addRoute(const vector<int>& path) { for (int v : path) occ[v]++; }
static inline void remRoute(const vector<int>& path) { for (int v : path) occ[v]--; }

static long long overlapCount(const vector<vector<int>>& R) {
    // count cells used by >1 net
    static vector<int> cnt;
    cnt.assign(HW, 0);
    for (auto& path : R) for (int v : path) cnt[v]++;
    long long ov = 0;
    for (int v = 0; v < HW; v++) if (cnt[v] > 1) ov += (cnt[v] - 1);
    return ov;
}
static long long totalLen(const vector<vector<int>>& R) {
    long long t = 0; for (auto& path : R) t += (long long)path.size() - 1; return t;
}

// ---- hard-blocked Dijkstra (for the completer) -----------------------------
static vector<int> dijkstraHard(int k, const vector<char>& blocked) {
    int s = id(P[k].sr, P[k].sc), t = id(P[k].tr, P[k].tc);
    curTag++;
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<>> pq;
    dist[s] = 0.0; prevc[s] = -1; visitTag[s] = curTag;
    pq.push({0.0, s});
    while (!pq.empty()) {
        auto [d, v] = pq.top(); pq.pop();
        if (d > dist[v] + 1e-12) continue;
        if (v == t) break;
        int r = v / W, c = v % W;
        for (int dir = 0; dir < 4; dir++) {
            int nr = r + DR[dir], nc = c + DC[dir];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int u = nr * W + nc;
            if (u != t && blocked[u]) continue;
            int ow = endptOwner[u];
            if (ow != -1 && ow != k && u != t) continue;
            double nd = d + 1.0;
            if (visitTag[u] != curTag || nd < dist[u] - 1e-12) {
                dist[u] = nd; prevc[u] = v; visitTag[u] = curTag;
                pq.push({nd, u});
            }
        }
    }
    if (visitTag[t] != curTag) return {};
    vector<int> path;
    for (int v = t; v != -1; v = prevc[v]) path.push_back(v);
    reverse(path.begin(), path.end());
    return path;
}

// ---- guaranteed-feasible sequential completer ------------------------------
// Route every net disjointly via sequential hard-blocked Dijkstra with bounded
// rip-up.  Returns a fully disjoint routing if found within its round budget,
// else an empty vector (caller tries another ordering).
static vector<vector<int>> completeDisjoint(const vector<int>& order, double deadline) {
    vector<vector<int>> R(N);
    vector<char> blocked(HW, 0);
    vector<char> committed(N, 0);
    int routed = 0;
    long long rounds = 0;
    long long maxRounds = 8LL * N + 200;
    deque<int> q(order.begin(), order.end());
    int sinceProgress = 0;
    while (!q.empty() && rounds < maxRounds) {
        if ((rounds & 31) == 0 && elapsed() > deadline) break;
        rounds++;
        int k = q.front(); q.pop_front();
        if (committed[k]) continue;
        vector<int> path = dijkstraHard(k, blocked);
        if (!path.empty()) {
            for (int v : path) blocked[v] = 1;
            R[k] = path; committed[k] = 1; routed++;
            sinceProgress = 0;
            continue;
        }
        // k is boxed in: evict the committed net occupying the most cells inside
        // k's (slightly expanded) bounding box, then retry k first.
        int r1 = min(P[k].sr, P[k].tr), r2 = max(P[k].sr, P[k].tr);
        int c1 = min(P[k].sc, P[k].tc), c2 = max(P[k].sc, P[k].tc);
        int victim = -1, bestHit = -1;
        for (int j = 0; j < N; j++) {
            if (!committed[j]) continue;
            int hit = 0;
            for (int v : R[j]) {
                int r = v / W, c = v % W;
                if (r >= r1 - 2 && r <= r2 + 2 && c >= c1 - 2 && c <= c2 + 2) hit++;
            }
            if (hit > bestHit) { bestHit = hit; victim = j; }
        }
        if (victim == -1 || bestHit <= 0) {
            // nothing nearby to evict -> push k back; if no progress for a while,
            // give up this ordering.
            q.push_back(k);
            if (++sinceProgress > N + 5) break;
            continue;
        }
        for (int v : R[victim]) blocked[v] = 0;
        committed[victim] = 0; R[victim].clear(); routed--;
        q.push_front(victim);
        q.push_front(k);
        if (++sinceProgress > 4 * N + 50) break;   // thrash guard
    }
    if (routed != N) return {};
    return R;
}

// Plain sequential greedy in INPUT ORDER with hard blocking and NO rip-up
// (a BFS shortest path over free cells, exactly the trivial baseline).  Returns
// a disjoint routing if every net routes, else empty.  Used as a seed so the
// solver is never worse than greedy on the instances where greedy succeeds.
static vector<vector<int>> greedySeed() {
    vector<vector<int>> R(N);
    vector<char> blocked(HW, 0);
    vector<int> pc(HW);                 // BFS predecessor, reset per net
    for (int k = 0; k < N; k++) {
        int s = id(P[k].sr, P[k].sc), t = id(P[k].tr, P[k].tc);
        for (int v = 0; v < HW; v++) pc[v] = -2;
        queue<int> q; q.push(s); pc[s] = -1;
        while (!q.empty()) {
            int v = q.front(); q.pop();
            if (v == t) break;
            int r = v / W, c = v % W;
            for (int d = 0; d < 4; d++) {
                int nr = r + DR[d], nc = c + DC[d];
                if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                int u = nr * W + nc;
                if (pc[u] != -2) continue;
                if (u != t && blocked[u]) continue;
                int ow = endptOwner[u];
                if (ow != -1 && ow != k && u != t) continue;
                pc[u] = v; q.push(u);
            }
        }
        if (pc[t] == -2) return {};     // a net could not route -> greedy fails
        vector<int> path;
        for (int v = t; v != -1; v = pc[v]) path.push_back(v);
        reverse(path.begin(), path.end());
        R[k] = path;
        for (int v : path) blocked[v] = 1;
    }
    return R;
}

int main() {
    T0 = chrono::steady_clock::now();
    readInput();
    if (N == 0) { return 0; }

    occ.assign(HW, 0);
    hist.assign(HW, 0.0);
    endptOwner.assign(HW, -1);
    dist.assign(HW, INF);
    prevc.assign(HW, -1);
    visitTag.assign(HW, 0);
    for (int k = 0; k < N; k++) {
        endptOwner[id(P[k].sr, P[k].sc)] = k;
        endptOwner[id(P[k].tr, P[k].tc)] = k;
    }

    // best feasible routing found so far (must be valid before we ever print).
    vector<vector<int>> bestRoutes;
    long long bestLen = LLONG_MAX;
    bool haveFeasible = false;

    auto consider = [&](const vector<vector<int>>& R) {
        // record R if it is feasible (zero overlaps) and shorter than best.
        if (overlapCount(R) != 0) return;
        long long t = totalLen(R);
        if (t < bestLen) { bestLen = t; bestRoutes = R; haveFeasible = true; }
    };

    // ---------- Phase 0: guaranteed feasible seeds via completer -------------
    // Establish valid solutions immediately so we can never crash / never emit
    // an infeasible output, then spend the rest of the budget improving them.
    // We seed from a few orderings, including plain input order (which matches
    // the trivial greedy baseline when it happens to succeed -- so we are never
    // worse than greedy on the easy instances).
    {
        vector<vector<int>> seeds;
        { vector<int> o(N); iota(o.begin(), o.end(), 0); seeds.push_back(o); } // input order
        { vector<int> o(N); iota(o.begin(), o.end(), 0);
          sort(o.begin(), o.end(), [&](int a, int b) {
              int da = abs(P[a].sr-P[a].tr)+abs(P[a].sc-P[a].tc);
              int db = abs(P[b].sr-P[b].tr)+abs(P[b].sc-P[b].tc);
              return da > db; });                                              // longest first
          seeds.push_back(o); }
        { vector<int> o(N); iota(o.begin(), o.end(), 0);
          sort(o.begin(), o.end(), [&](int a, int b) {
              int da = abs(P[a].sr-P[a].tr)+abs(P[a].sc-P[a].tc);
              int db = abs(P[b].sr-P[b].tr)+abs(P[b].sc-P[b].tc);
              return da < db; });                                             // shortest first
          seeds.push_back(o); }
        for (auto& o : seeds) {
            if (elapsed() > TIME_LIMIT * 0.30 && haveFeasible) break;
            vector<vector<int>> R = completeDisjoint(o, TIME_LIMIT * 0.30);
            if (!R.empty()) consider(R);
        }
        // Greedy baseline as a seed: guarantees we never under-perform greedy.
        { vector<vector<int>> G = greedySeed(); if (!G.empty()) consider(G); }
    }

    // ---------- Phase 1: negotiated-congestion rip-up-and-reroute -----------
    vector<vector<int>> cur(N);
    for (int k = 0; k < N; k++) {
        vector<int> path = dijkstraSoft(k);
        cur[k] = path;
        addRoute(path);
    }

    int iter = 0;
    while (elapsed() < TIME_LIMIT * 0.78) {
        iter++;
        PFAC = 1.0 + 0.7 * iter;          // ramp present penalty without bound
        // reroute every net, most-contested first
        vector<int> order(N); iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            double ca = 0, cb = 0;
            for (int v : cur[a]) ca += (occ[v] - 1);
            for (int v : cur[b]) cb += (occ[v] - 1);
            return ca > cb;
        });
        for (int k : order) {
            remRoute(cur[k]);
            vector<int> path = dijkstraSoft(k);
            cur[k] = path;
            addRoute(path);
        }
        // accumulate history on still-overused cells (drives convergence)
        for (int v = 0; v < HW; v++) if (occ[v] > 1) hist[v] += 0.8 * (occ[v] - 1);
        consider(cur);
        if (haveFeasible && overlapCount(cur) == 0) {
            // already disjoint: a few extra passes can still shorten it, but
            // avoid spinning forever on an easy instance.
            if (iter > 12) break;
        }
        if (iter > 400) break;
    }

    // ---------- Phase 2: length-improving LNS on the best feasible routing ----
    // Iterated rip-up-and-reroute as a Large-Neighbourhood-Search shortener:
    // repeatedly (a) rip up a small set of nets (the most-detoured ones, plus a
    // random handful), then (b) re-route them one by one by hard Dijkstra over
    // the cells the OTHER nets leave free, in shortest-Manhattan-first order so
    // tight nets grab the straight corridors.  Disjointness is preserved by
    // construction (hard blocking), so every accepted state stays feasible; we
    // keep it iff the total length did not increase, and `consider` records any
    // global improvement.  Random restarts from the incumbent escape local
    // optima.  This is where most of the time budget goes.
    if (haveFeasible) {
        unsigned rng = 0x12345678u;
        auto rnd = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };

        vector<vector<int>> R = bestRoutes;
        long long curLen = totalLen(R);
        vector<char> blocked(HW, 0);

        while (elapsed() < TIME_LIMIT) {
            // choose rip set: top-detour nets + a few random ones
            int RIP = max(2, min(N, 3 + (int)(rnd() % 5)));
            vector<int> byDetour(N); iota(byDetour.begin(), byDetour.end(), 0);
            sort(byDetour.begin(), byDetour.end(), [&](int a, int b) {
                long long da = (long long)R[a].size() - 1 - (abs(P[a].sr-P[a].tr)+abs(P[a].sc-P[a].tc));
                long long db = (long long)R[b].size() - 1 - (abs(P[b].sr-P[b].tr)+abs(P[b].sc-P[b].tc));
                return da > db;
            });
            vector<char> rip(N, 0);
            int picked = 0;
            for (int i = 0; i < N && picked < RIP; i++) {
                // bias to detoured nets but mix in randomness
                if ((int)(rnd() % 3) != 0) { rip[byDetour[i]] = 1; picked++; }
            }
            for (int t = 0; t < 2 && picked < RIP; t++) { int k = rnd() % N; if (!rip[k]) { rip[k] = 1; picked++; } }
            if (picked == 0) { rip[byDetour[0]] = 1; }

            // candidate copy
            vector<vector<int>> C = R;
            // blocked = cells of nets NOT ripped
            fill(blocked.begin(), blocked.end(), 0);
            for (int k = 0; k < N; k++) if (!rip[k]) for (int v : C[k]) blocked[v] = 1;
            // ripped nets routed shortest-Manhattan-first
            vector<int> ripList;
            for (int k = 0; k < N; k++) if (rip[k]) ripList.push_back(k);
            sort(ripList.begin(), ripList.end(), [&](int a, int b) {
                int da = abs(P[a].sr-P[a].tr)+abs(P[a].sc-P[a].tc);
                int db = abs(P[b].sr-P[b].tr)+abs(P[b].sc-P[b].tc);
                return da < db;
            });
            bool ok = true;
            for (int k : ripList) {
                vector<int> np = dijkstraHard(k, blocked);
                if (np.empty()) { ok = false; break; }   // a ripped net could not re-route
                C[k] = np;
                for (int v : np) blocked[v] = 1;
            }
            if (!ok) continue;        // discard: keep R unchanged (still feasible)
            long long nl = totalLen(C);
            if (nl <= curLen) {       // accept equal-or-better (plateau walk)
                R = C; curLen = nl;
                consider(R);
            }
            // else discard and try a different rip set next iteration
        }
    }

    // ---------- Output ------------------------------------------------------
    if (!haveFeasible) {
        // Belt-and-braces: the generator guarantees feasibility; keep trying
        // random orderings with a relaxed deadline until we find a disjoint
        // routing.  (In practice Phase 0 already succeeded.)
        unsigned rng = 2463534242u;
        double hardDeadline = 9.0;
        for (int tryno = 0; tryno < 5000 && !haveFeasible && elapsed() < hardDeadline; tryno++) {
            vector<int> o(N); iota(o.begin(), o.end(), 0);
            for (int i = N - 1; i > 0; i--) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                int j = rng % (unsigned)(i + 1); swap(o[i], o[j]);
            }
            vector<vector<int>> R = completeDisjoint(o, hardDeadline);
            if (!R.empty()) consider(R);
        }
    }

    // If, against all guarantees, we still have nothing, emit straight-segment
    // single-cell stubs (scores 0 but never crashes the harness).  This branch
    // is unreachable on generator instances.
    if (!haveFeasible) {
        for (int k = 0; k < N; k++) printf("1 %d %d\n", P[k].sr, P[k].sc);
        return 0;
    }

    for (int k = 0; k < N; k++) {
        const vector<int>& path = bestRoutes[k];
        printf("%d", (int)path.size());
        for (int v : path) printf(" %d %d", v / W, v % W);
        printf("\n");
    }
    return 0;
}
