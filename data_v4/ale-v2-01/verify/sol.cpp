// ale-v2-01 : Capacitated Vehicle Routing (CVRP) heuristic solver.
//
// Reads an instance from stdin, writes a feasible set of vehicle routes to
// stdout, minimizing total Euclidean route length subject to per-route
// capacity. The pipeline is the current strong practical family for CVRP:
//
//   1. Clarke-Wright savings (parallel) construction for a good initial set
//      of routes -- far better than one-client-per-route or nearest-neighbor.
//   2. Intra-route refinement: 2-opt and Or-opt with O(1) move deltas
//      (only the touched edges enter the delta, never a full re-sum).
//   3. Large Neighborhood Search (LNS): repeatedly RUIN a chunk of clients
//      (a worst/related segment) and RECREATE them with regret-2 insertion,
//      accepting only improving (or equal) complete solutions. This is the
//      ruin-and-recreate metaheuristic that wins on CVRP.
//
// Feasibility is invariant by construction: we begin from the trivial
// "each client on its own route" solution (always valid), and every move /
// reinsertion checks capacity before being applied, so the incumbent is
// always a valid CVRP solution and the program never emits an infeasible
// answer.
#include <bits/stdc++.h>
using namespace std;

static const double TIME_LIMIT = 1.85; // seconds, safe margin under a 2s budget
static std::chrono::steady_clock::time_point T0;
static inline double elapsed() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
}

int N, CAP;
double DX, DY;                 // depot
vector<double> X, Y;           // client coords, 1-based (index 0 = depot)
vector<int> DEM;               // demand, 1-based

static inline double dist(int a, int b) {
    // a,b in 0..N where 0 == depot.
    double ax = (a == 0 ? DX : X[a]);
    double ay = (a == 0 ? DY : Y[a]);
    double bx = (b == 0 ? DX : X[b]);
    double by = (b == 0 ? DY : Y[b]);
    double dx = ax - bx, dy = ay - by;
    return sqrt(dx * dx + dy * dy);
}

// A route is an ordered list of client ids (1..N), depot implicit at both ends.
struct Route {
    vector<int> seq;
    int load = 0;
};

// ---- length helpers ---------------------------------------------------------
static double routeLen(const Route& r) {
    if (r.seq.empty()) return 0.0;
    double L = dist(0, r.seq.front());
    for (size_t i = 1; i < r.seq.size(); ++i) L += dist(r.seq[i - 1], r.seq[i]);
    L += dist(r.seq.back(), 0);
    return L;
}
static double totalLen(const vector<Route>& rs) {
    double L = 0;
    for (auto& r : rs) L += routeLen(r);
    return L;
}

// ---- Clarke-Wright parallel savings construction ---------------------------
// Start: each client in its own route. Repeatedly merge the route ending at i
// with the route starting at j if the savings s(i,j) = d(i,0)+d(0,j)-d(i,j) is
// positive and capacity allows, taking the largest savings first.
static vector<Route> clarkeWright() {
    // routeOf[c] = index of the route containing client c.
    vector<int> routeOf(N + 1, -1);
    vector<Route> routes(N);
    for (int c = 1; c <= N; ++c) {
        routes[c - 1].seq = {c};
        routes[c - 1].load = DEM[c];
        routeOf[c] = c - 1;
    }

    struct Sav { double s; int i, j; };
    vector<Sav> savs;
    savs.reserve((size_t)N * (N - 1) / 2);
    for (int i = 1; i <= N; ++i)
        for (int j = i + 1; j <= N; ++j) {
            double s = dist(i, 0) + dist(0, j) - dist(i, j);
            if (s > 0) savs.push_back({s, i, j});
        }
    sort(savs.begin(), savs.end(), [](const Sav& a, const Sav& b) { return a.s > b.s; });

    auto isEndpoint = [&](const Route& r, int c, bool& atFront) -> bool {
        if (r.seq.empty()) return false;
        if (r.seq.front() == c) { atFront = true; return true; }
        if (r.seq.back() == c) { atFront = false; return true; }
        return false;
    };

    for (auto& sv : savs) {
        int i = sv.i, j = sv.j;
        int ri = routeOf[i], rj = routeOf[j];
        if (ri == rj) continue;                 // already same route
        Route& Ri = routes[ri];
        Route& Rj = routes[rj];
        if (Ri.load + Rj.load > CAP) continue;  // capacity guard

        bool iFront, jFront;
        if (!isEndpoint(Ri, i, iFront)) continue; // i must be a route endpoint
        if (!isEndpoint(Rj, j, jFront)) continue; // j must be a route endpoint

        // Merge so that i and j become adjacent (i ... j). Orient each route.
        vector<int> a = Ri.seq, b = Rj.seq;
        if (iFront) reverse(a.begin(), a.end()); // make i the back of a
        if (!jFront) reverse(b.begin(), b.end()); // make j the front of b
        // now a = [... i], b = [j ...]; concatenation joins i--j.
        a.insert(a.end(), b.begin(), b.end());
        Ri.seq = move(a);
        Ri.load = Ri.load + Rj.load;
        for (int c : Ri.seq) routeOf[c] = ri;
        Rj.seq.clear();
        Rj.load = 0;
    }

    vector<Route> out;
    for (auto& r : routes)
        if (!r.seq.empty()) out.push_back(move(r));
    return out;
}

// ---- intra-route 2-opt (reverse a segment) with O(1) delta -----------------
static bool twoOptRoute(Route& r) {
    int m = (int)r.seq.size();
    if (m < 4) return false;
    bool improvedAny = false;
    bool go = true;
    while (go) {
        go = false;
        for (int i = 0; i < m - 1; ++i) {
            int a = (i == 0 ? 0 : r.seq[i - 1]);
            int b = r.seq[i];
            for (int k = i + 1; k < m; ++k) {
                int c = r.seq[k];
                int d = (k == m - 1 ? 0 : r.seq[k + 1]);
                // Reverse seq[i..k]: edges (a,b)+(c,d) -> (a,c)+(b,d). O(1) delta.
                double delta = dist(a, c) + dist(b, d) - dist(a, b) - dist(c, d);
                if (delta < -1e-9) {
                    reverse(r.seq.begin() + i, r.seq.begin() + k + 1);
                    improvedAny = true;
                    go = true;
                    b = r.seq[i]; // a stays; b is now the new element at i
                }
            }
        }
    }
    return improvedAny;
}

// ---- Or-opt: move a segment of length L (1..3) to a better position --------
// Works inside a single route; O(1) delta per candidate insertion point.
static bool orOptRoute(Route& r) {
    int m = (int)r.seq.size();
    if (m < 3) return false;
    bool improvedAny = false;
    for (int L = 1; L <= 3; ++L) {
        bool go = true;
        while (go) {
            go = false;
            m = (int)r.seq.size();
            if (m < L + 2) break;
            for (int i = 0; i + L <= m; ++i) {
                int p = (i == 0 ? 0 : r.seq[i - 1]);
                int s0 = r.seq[i];
                int s1 = r.seq[i + L - 1];
                int q = (i + L == m ? 0 : r.seq[i + L]);
                double removed = dist(p, s0) + dist(s1, q) - dist(p, q);
                if (removed <= 1e-9) continue;
                // try inserting the segment [s0..s1] between consecutive nodes
                for (int j = 0; j <= m; ++j) {
                    if (j >= i - 0 && j <= i + L) continue; // overlapping positions
                    int u = (j == 0 ? 0 : r.seq[j - 1]);
                    int v = (j == m ? 0 : r.seq[j]);
                    if (u == p && v == q) continue;
                    // forward orientation
                    double addF = dist(u, s0) + dist(s1, v) - dist(u, v);
                    double deltaF = addF - removed;
                    // reversed orientation
                    double addR = dist(u, s1) + dist(s0, v) - dist(u, v);
                    double deltaR = addR - removed;
                    bool rev = deltaR < deltaF;
                    double delta = rev ? deltaR : deltaF;
                    if (delta < -1e-9) {
                        vector<int> seg(r.seq.begin() + i, r.seq.begin() + i + L);
                        if (rev) reverse(seg.begin(), seg.end());
                        // remove segment
                        r.seq.erase(r.seq.begin() + i, r.seq.begin() + i + L);
                        int jj = j;
                        if (j > i) jj -= L; // indices shift after erase
                        r.seq.insert(r.seq.begin() + jj, seg.begin(), seg.end());
                        improvedAny = true;
                        go = true;
                        break;
                    }
                }
                if (go) break;
            }
        }
    }
    return improvedAny;
}

static void localImprove(vector<Route>& rs) {
    for (auto& r : rs) {
        if (elapsed() > TIME_LIMIT) return;
        twoOptRoute(r);
        orOptRoute(r);
        twoOptRoute(r);
    }
}

// ---- regret-2 insertion: insert a set of clients into existing routes ------
// For each unrouted client, the best feasible insertion cost is c1 and the
// second-best (in a *different* route) is c2; regret = c2 - c1. Insert the
// client with the largest regret first (it is the one we'd most regret leaving
// for last). Capacity is always checked, so the result stays feasible.
struct InsPos { double cost; int route; int pos; };

static InsPos bestInsertInRoute(const Route& r, int c) {
    InsPos best{1e18, -1, -1};
    if (r.load + DEM[c] > CAP) return best; // route can't take c at all
    int m = (int)r.seq.size();
    for (int j = 0; j <= m; ++j) {
        int u = (j == 0 ? 0 : r.seq[j - 1]);
        int v = (j == m ? 0 : r.seq[j]);
        double add = dist(u, c) + dist(c, v) - dist(u, v);
        if (add < best.cost) best = {add, -1, j};
    }
    return best;
}

static void regretInsert(vector<Route>& rs, vector<int>& pool) {
    while (!pool.empty()) {
        int bestClient = -1, bestRoute = -1, bestPos = -1;
        double bestRegret = -1e18, bestC1 = 0;
        for (int idx = 0; idx < (int)pool.size(); ++idx) {
            int c = pool[idx];
            double c1 = 1e18, c2 = 1e18;
            int c1route = -1, c1pos = -1;
            for (int ri = 0; ri < (int)rs.size(); ++ri) {
                InsPos ip = bestInsertInRoute(rs[ri], c);
                if (ip.pos < 0) continue;
                if (ip.cost < c1) { c2 = c1; c1 = ip.cost; c1route = ri; c1pos = ip.pos; }
                else if (ip.cost < c2) { c2 = ip.cost; }
            }
            if (c1route < 0) {
                // No feasible existing route -> must open a new route for c.
                // Treat as huge regret so it's handled, inserted as a singleton.
                double regret = 1e17;
                if (regret > bestRegret) {
                    bestRegret = regret; bestClient = idx;
                    bestRoute = -2; bestPos = -1; bestC1 = 0;
                }
                continue;
            }
            double regret = (c2 >= 1e17 ? 1e16 : c2 - c1); // single-option => high regret
            if (regret > bestRegret) {
                bestRegret = regret; bestClient = idx;
                bestRoute = c1route; bestPos = c1pos; bestC1 = c1;
            }
        }
        int c = pool[bestClient];
        pool.erase(pool.begin() + bestClient);
        if (bestRoute == -2) {
            Route nr; nr.seq = {c}; nr.load = DEM[c];
            rs.push_back(move(nr));
        } else {
            Route& r = rs[bestRoute];
            r.seq.insert(r.seq.begin() + bestPos, c);
            r.load += DEM[c];
        }
        (void)bestC1;
    }
}

// ---- LNS ruin: remove a related chunk of clients ---------------------------
// Strategy: pick a random seed client, then remove its spatially-nearest
// clients (Shaw-style relatedness) so the recreate step has room to rewire.
static void ruin(vector<Route>& rs, vector<int>& pool, std::mt19937& rng, int K) {
    // Build a flat list of currently-routed clients with positions.
    vector<int> all;
    for (auto& r : rs) for (int c : r.seq) all.push_back(c);
    if ((int)all.size() <= K) return;
    int seed = all[rng() % all.size()];
    // sort all clients by distance to seed; remove the K closest.
    sort(all.begin(), all.end(), [&](int a, int b) { return dist(seed, a) < dist(seed, b); });
    set<int> rem(all.begin(), all.begin() + K);
    for (auto& r : rs) {
        vector<int> ns;
        for (int c : r.seq)
            if (rem.count(c)) { r.load -= DEM[c]; pool.push_back(c); }
            else ns.push_back(c);
        r.seq = move(ns);
    }
    // drop empty routes
    vector<Route> kept;
    for (auto& r : rs) if (!r.seq.empty()) kept.push_back(move(r));
    rs = move(kept);
}

int main() {
    T0 = std::chrono::steady_clock::now();
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> N >> CAP)) return 0;
    cin >> DX >> DY;
    X.assign(N + 1, 0); Y.assign(N + 1, 0); DEM.assign(N + 1, 0);
    for (int i = 1; i <= N; ++i) cin >> X[i] >> Y[i] >> DEM[i];

    if (N == 0) { cout << 0 << "\n"; return 0; }

    // 1) Construction.
    vector<Route> cur = clarkeWright();
    localImprove(cur);
    double curLen = totalLen(cur);

    vector<Route> best = cur;
    double bestLen = curLen;

    // 2) LNS ruin-and-recreate.
    std::mt19937 rng(987654321u);
    int K = max(2, min(N / 5, 20)); // chunk size to ruin each iteration
    while (elapsed() < TIME_LIMIT) {
        vector<Route> cand = best;
        vector<int> pool;
        int k = K + (int)(rng() % (size_t)max(1, K)); // vary chunk size
        k = min(k, N - 1);
        ruin(cand, pool, rng, k);
        regretInsert(cand, pool);
        // polish the touched solution
        for (auto& r : cand) {
            if (elapsed() > TIME_LIMIT) break;
            twoOptRoute(r);
            orOptRoute(r);
        }
        double candLen = totalLen(cand);
        if (candLen < bestLen - 1e-7) {
            best = cand;
            bestLen = candLen;
        }
    }

    // 3) Final guard: ensure EVERY client appears exactly once and capacities
    // hold. (They do by construction, but we verify and repair defensively so
    // the program can never emit an infeasible answer.)
    {
        vector<int> cnt(N + 1, 0);
        vector<Route> fixed;
        for (auto& r : best) {
            Route nr; int load = 0;
            for (int c : r.seq) {
                if (c >= 1 && c <= N && cnt[c] == 0 && load + DEM[c] <= CAP) {
                    cnt[c] = 1; load += DEM[c]; nr.seq.push_back(c);
                }
            }
            nr.load = load;
            if (!nr.seq.empty()) fixed.push_back(move(nr));
        }
        // any client missed -> put each on its own route (always feasible since
        // every demand <= CAP by the instance generator).
        for (int c = 1; c <= N; ++c)
            if (!cnt[c]) {
                Route nr; nr.seq = {c}; nr.load = DEM[c];
                fixed.push_back(move(nr));
            }
        best = move(fixed);
    }

    // Output: K routes, each "m id_1 ... id_m".
    cout << best.size() << "\n";
    for (auto& r : best) {
        cout << r.seq.size();
        for (int c : r.seq) cout << ' ' << c;
        cout << "\n";
    }
    return 0;
}
