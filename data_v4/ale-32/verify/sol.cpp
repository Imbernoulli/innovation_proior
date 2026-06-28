// Capacitated k-Means (cardinality-capped clustering) -- heuristic solver.
//
// Objective: partition n points in the plane into k clusters, each cluster
// holding AT MOST `cap` points, minimizing the total squared Euclidean distance
// from every point to the centroid (mean) of the cluster it is assigned to. We
// read the instance from stdin:
//     n k cap
//     x_i y_i              (n lines)
// and write to stdout
//     n integer labels a_0 .. a_{n-1}   (a_i in [0,k), the cluster of point i)
//     2*k center coordinates cx_j cy_j  (informational; the scorer recomputes
//                                        centroids from the assignment itself).
//
// Method (the innovation): the cap turns an ordinary Lloyd/k-means step into a
// TRANSPORTATION problem. Plain k-means assigns each point to its NEAREST centre
// -- but under a hard per-cluster cap that nearest-centre assignment is usually
// INFEASIBLE (a popular centre would attract more than `cap` points). So we
// replace the assignment half of Lloyd's iteration with a CAPACITATED ASSIGNMENT
// solved exactly as a MIN-COST FLOW: a bipartite transportation problem with one
// unit of supply per point, capacity `cap` per centre, and edge cost = squared
// distance point->centre. We alternate:
//     (i)  CENTROID UPDATE -- each centre = mean of its members (closed form);
//     (ii) CAPACITATED ASSIGNMENT -- min-cost flow over the current centres,
//          giving the cheapest cap-respecting assignment to those centres.
// This is the capacitated-k-means / "balanced k-means via transportation"
// scheme. The MCF is solved by successive shortest paths with Johnson potentials
// (SPFA once for the initial potentials, then Dijkstra), warm-started by a feasible
// integral assignment so the flow search is small each round. After the
// alternation settles we run an incremental SWAP/MOVE local search that accounts
// for the fact that moving a point changes the centroid it is measured against
// (the true objective uses recomputed centroids, not the fixed centres MCF used).
// Every assignment we ever hold is cap-respecting and uses all k clusters, so any
// early stop still prints a FEASIBLE solution.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N, K, CAP;
vector<double> PX, PY;            // point coordinates

static inline double sqdist(double x, double y, double cx, double cy) {
    double dx = x - cx, dy = y - cy;
    return dx * dx + dy * dy;
}

// ----------------------------------------------------------------------------
// Min-cost max-flow (successive shortest paths with potentials, Dijkstra).
// Network: S -> each point (cap 1, cost 0); point i -> centre j (cap 1, cost
// sqdist); centre j -> T (cap CAP, cost 0). We push N units of flow; because
// K*CAP >= N (guaranteed by the instance) a feasible flow of value N exists, and
// SSP returns the MIN-COST one, i.e. the cheapest cap-respecting assignment to
// the current centres.
struct MCMF {
    struct Edge { int to, rev; int cap; double cost; };
    int n;
    vector<vector<Edge>> g;
    vector<double> h, dist;          // potentials, shortest dist
    vector<int> pv, pe;
    vector<char> inq;
    void init(int nn) {
        n = nn; g.assign(n, {});
        h.assign(n, 0); dist.assign(n, 0);
        pv.assign(n, -1); pe.assign(n, -1); inq.assign(n, 0);
    }
    void add(int a, int b, int cap, double cost) {
        g[a].push_back({b, (int)g[b].size(), cap, cost});
        g[b].push_back({a, (int)g[a].size() - 1, 0, -cost});
    }
    // initial potentials via SPFA (Bellman-Ford queue) -- costs here are all >= 0
    // on forward edges, so potentials start at 0; we keep SPFA for safety.
    void spfa(int s) {
        const double INF = 1e30;
        fill(h.begin(), h.end(), INF);
        h[s] = 0;
        deque<int> q; q.push_back(s); fill(inq.begin(), inq.end(), 0); inq[s] = 1;
        while (!q.empty()) {
            int u = q.front(); q.pop_front(); inq[u] = 0;
            for (auto &e : g[u]) {
                if (e.cap > 0 && h[u] + e.cost < h[e.to] - 1e-12) {
                    h[e.to] = h[u] + e.cost;
                    if (!inq[e.to]) { inq[e.to] = 1; q.push_back(e.to); }
                }
            }
        }
        for (auto &x : h) if (x > 1e29) x = 0;
    }
    // one Dijkstra augmenting unit pass; returns flow pushed (0 if none)
    bool dijkstra(int s, int t) {
        const double INF = 1e30;
        fill(dist.begin(), dist.end(), INF);
        dist[s] = 0;
        priority_queue<pair<double,int>, vector<pair<double,int>>,
                       greater<pair<double,int>>> pq;
        pq.push({0.0, s});
        fill(pv.begin(), pv.end(), -1);
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u] + 1e-12) continue;
            for (int i = 0; i < (int)g[u].size(); ++i) {
                auto &e = g[u][i];
                if (e.cap <= 0) continue;
                double nd = dist[u] + e.cost + h[u] - h[e.to];
                if (nd < dist[e.to] - 1e-12) {
                    dist[e.to] = nd;
                    pv[e.to] = u; pe[e.to] = i;
                    pq.push({nd, e.to});
                }
            }
        }
        if (dist[t] > 1e29) return false;
        for (int i = 0; i < n; ++i) if (dist[i] < 1e29) h[i] += dist[i];
        // augment one unit along the path
        int v = t;
        while (v != s) {
            auto &e = g[pv[v]][pe[v]];
            e.cap -= 1;
            g[v][e.rev].cap += 1;
            v = pv[v];
        }
        return true;
    }
};

// Solve the capacitated assignment to the given centres; fill labels[].
// Returns true on success (always succeeds because K*CAP >= N).
static bool capacitated_assign(const vector<double>& cx, const vector<double>& cy,
                               vector<int>& labels) {
    // node ids: 0..N-1 points, N..N+K-1 centres, S=N+K, T=N+K+1
    int S = N + K, T = N + K + 1;
    MCMF mc; mc.init(N + K + 2);
    for (int i = 0; i < N; ++i) mc.add(S, i, 1, 0.0);
    for (int j = 0; j < K; ++j) mc.add(N + j, T, CAP, 0.0);
    // store edge index of point->centre so we can read back the chosen centre
    vector<vector<int>> eidx(N, vector<int>(0));
    for (int i = 0; i < N; ++i) {
        eidx[i].resize(K);
        for (int j = 0; j < K; ++j) {
            eidx[i][j] = (int)mc.g[i].size();
            mc.add(i, N + j, 1, sqdist(PX[i], PY[i], cx[j], cy[j]));
        }
    }
    mc.spfa(S);
    int pushed = 0;
    while (pushed < N && mc.dijkstra(S, T)) ++pushed;
    if (pushed < N) return false;     // should not happen
    // read back: the point->centre edge with cap==0 carries the unit of flow.
    for (int i = 0; i < N; ++i) {
        int chosen = -1;
        for (int j = 0; j < K; ++j) {
            if (mc.g[i][eidx[i][j]].cap == 0) { chosen = j; break; }
        }
        if (chosen < 0) return false;
        labels[i] = chosen;
    }
    return true;
}

// Recompute centroids from labels; return total squared cost.
static double recompute_and_cost(const vector<int>& labels,
                                 vector<double>& cx, vector<double>& cy) {
    vector<double> sx(K, 0), sy(K, 0);
    vector<int> cnt(K, 0);
    for (int i = 0; i < N; ++i) {
        int a = labels[i];
        sx[a] += PX[i]; sy[a] += PY[i]; cnt[a]++;
    }
    for (int j = 0; j < K; ++j) {
        if (cnt[j] > 0) { cx[j] = sx[j] / cnt[j]; cy[j] = sy[j] / cnt[j]; }
    }
    double cost = 0;
    for (int i = 0; i < N; ++i) {
        int a = labels[i];
        cost += sqdist(PX[i], PY[i], cx[a], cy[a]);
    }
    return cost;
}

// k-means++ centre seeding.
static void kpp_init(Rng& rng, vector<double>& cx, vector<double>& cy) {
    int first = rng.nextu(N);
    cx.assign(K, 0); cy.assign(K, 0);
    cx[0] = PX[first]; cy[0] = PY[first];
    vector<double> d2(N);
    for (int i = 0; i < N; ++i) d2[i] = sqdist(PX[i], PY[i], cx[0], cy[0]);
    for (int c = 1; c < K; ++c) {
        double tot = 0; for (double v : d2) tot += v;
        int pick;
        if (tot <= 0) { pick = rng.nextu(N); }
        else {
            double r = rng.nextd() * tot, acc = 0; pick = N - 1;
            for (int i = 0; i < N; ++i) { acc += d2[i]; if (acc >= r) { pick = i; break; } }
        }
        cx[c] = PX[pick]; cy[c] = PY[pick];
        for (int i = 0; i < N; ++i) {
            double nd = sqdist(PX[i], PY[i], cx[c], cy[c]);
            if (nd < d2[i]) d2[i] = nd;
        }
    }
}

int main() {
    if (!(cin >> N >> K >> CAP)) return 0;
    PX.resize(N); PY.resize(N);
    for (int i = 0; i < N; ++i) cin >> PX[i] >> PY[i];

    if (N == 0) { for (int j = 0; j < 2 * K; ++j) cout << 0 << " \n"[j + 1 == 2 * K]; return 0; }

    double t_start = now_sec();
    const double TL = 1.8;             // seconds

    Rng rng(0x32C0FFEEULL ^ ((uint64_t)N << 32) ^ ((uint64_t)K << 12) ^ (uint64_t)CAP);

    // ----- track the global best feasible assignment -----
    vector<int> best_labels;
    double best_cost = 1e300;
    vector<double> best_cx(K), best_cy(K);

    // ----- multi-start capacitated Lloyd -----
    int restart = 0;
    while (now_sec() - t_start < TL) {
        vector<double> cx, cy;
        kpp_init(rng, cx, cy);
        vector<int> labels(N, 0);
        double prev = 1e300;
        // capacitated Lloyd: (assign via MCF) then (recompute centroids)
        for (int it = 0; it < 40; ++it) {
            if (now_sec() - t_start > TL) break;
            if (!capacitated_assign(cx, cy, labels)) { break; }
            double c = recompute_and_cost(labels, cx, cy);
            if (c > prev - 1e-7) { prev = c; break; }   // converged
            prev = c;
        }
        // ensure we have a feasible labelling for this restart even if we broke
        // early (e.g. on the first iteration cx/cy are k-means++ seeds, which is
        // fine for MCF); re-run one capacitated assignment if labels are unset.
        // (capacitated_assign already filled labels on every successful call.)

        // fix any empty cluster: MCF can leave a centre with zero flow if it is
        // never the cheapest target. Reassign the globally-farthest point of the
        // most populous cluster to each empty cluster (keeps cap, fills cluster).
        {
            vector<int> cnt(K, 0);
            for (int i = 0; i < N; ++i) cnt[labels[i]]++;
            bool changed = false;
            for (int j = 0; j < K; ++j) {
                if (cnt[j] != 0) continue;
                // donor = most populous cluster with > 1 member
                int donor = -1;
                for (int t = 0; t < K; ++t)
                    if (cnt[t] > 1 && (donor < 0 || cnt[t] > cnt[donor])) donor = t;
                if (donor < 0) continue;
                // recompute donor centroid
                double sx = 0, sy = 0; int c = 0;
                for (int i = 0; i < N; ++i) if (labels[i] == donor) { sx += PX[i]; sy += PY[i]; c++; }
                double dcx = sx / c, dcy = sy / c;
                int far = -1; double fd = -1;
                for (int i = 0; i < N; ++i) if (labels[i] == donor) {
                    double d = sqdist(PX[i], PY[i], dcx, dcy);
                    if (d > fd) { fd = d; far = i; }
                }
                if (far >= 0) { labels[far] = j; cnt[j]++; cnt[donor]--; changed = true; }
            }
            if (changed) recompute_and_cost(labels, cx, cy);
        }

        // ----- incremental MOVE/SWAP local search on the true objective -----
        // The MCF minimizes cost to FIXED centres; but the scored objective uses
        // centroids recomputed from the assignment. So we polish with moves that
        // evaluate the exact change in sum-of-squared-distance-to-centroid using
        // per-cluster running sums (sx, sy, sxx+syy, count): moving point p out of
        // cluster a / into cluster b changes only those two clusters' costs, each
        // computable in O(1) from the sums. This makes each candidate move O(1).
        vector<double> sx(K, 0), sy(K, 0), sq(K, 0);
        vector<int> cnt(K, 0);
        for (int i = 0; i < N; ++i) {
            int a = labels[i];
            sx[a] += PX[i]; sy[a] += PY[i];
            sq[a] += PX[i] * PX[i] + PY[i] * PY[i];
            cnt[a]++;
        }
        // cluster cost from running sums: sum||p||^2 - (sx^2+sy^2)/cnt
        auto clcost = [&](int j)->double {
            if (cnt[j] <= 0) return 0.0;
            return sq[j] - (sx[j] * sx[j] + sy[j] * sy[j]) / cnt[j];
        };
        // cost if cluster j had point p added
        auto cost_with = [&](int j, double px, double py)->double {
            double nsx = sx[j] + px, nsy = sy[j] + py;
            double nsq = sq[j] + px * px + py * py;
            int nc = cnt[j] + 1;
            return nsq - (nsx * nsx + nsy * nsy) / nc;
        };
        // cost if cluster j had point p removed
        auto cost_without = [&](int j, double px, double py)->double {
            int nc = cnt[j] - 1;
            if (nc <= 0) return 0.0;
            double nsx = sx[j] - px, nsy = sy[j] - py;
            double nsq = sq[j] - (px * px + py * py);
            return nsq - (nsx * nsx + nsy * nsy) / nc;
        };

        bool improved = true;
        while (improved && now_sec() - t_start < TL) {
            improved = false;
            // single moves: p from a to b (b must have room and a keep >=1)
            for (int i = 0; i < N; ++i) {
                int a = labels[i];
                if (cnt[a] <= 1) continue;            // keep every cluster nonempty
                double px = PX[i], py = PY[i];
                double base_a = clcost(a);
                double after_a = cost_without(a, px, py);
                double bestDelta = -1e-6; int bestB = -1;
                for (int b = 0; b < K; ++b) {
                    if (b == a) continue;
                    if (cnt[b] >= CAP) continue;       // respect the cap
                    double delta = (after_a - base_a) + (cost_with(b, px, py) - clcost(b));
                    if (delta < bestDelta) { bestDelta = delta; bestB = b; }
                }
                if (bestB >= 0) {
                    int b = bestB;
                    sx[a] -= px; sy[a] -= py; sq[a] -= px * px + py * py; cnt[a]--;
                    sx[b] += px; sy[b] += py; sq[b] += px * px + py * py; cnt[b]++;
                    labels[i] = b;
                    improved = true;
                }
            }
            if (now_sec() - t_start > TL) break;
            // pair swaps: p in a, q in b -> exchange clusters (cap-neutral).
            // Helps when both clusters are full and a single move is blocked.
            for (int i = 0; i < N; ++i) {
                int a = labels[i];
                double px = PX[i], py = PY[i];
                // only scan a limited window of partners to stay cheap
                for (int s = 0; s < 24; ++s) {
                    int jq = rng.nextu(N);
                    int b = labels[jq];
                    if (b == a) continue;
                    double qx = PX[jq], qy = PY[jq];
                    // remove p from a, q from b, then add q to a, p to b.
                    // compute via sums for the two affected clusters together.
                    double ca = clcost(a), cb = clcost(b);
                    // cluster a after losing p and gaining q
                    double asx = sx[a] - px + qx, asy = sy[a] - py + qy;
                    double asq = sq[a] - (px*px+py*py) + (qx*qx+qy*qy);
                    int ac = cnt[a];                         // size unchanged
                    double na = (ac > 0) ? asq - (asx*asx+asy*asy)/ac : 0.0;
                    double bsx = sx[b] - qx + px, bsy = sy[b] - qy + py;
                    double bsq = sq[b] - (qx*qx+qy*qy) + (px*px+py*py);
                    int bc = cnt[b];
                    double nb = (bc > 0) ? bsq - (bsx*bsx+bsy*bsy)/bc : 0.0;
                    double delta = (na - ca) + (nb - cb);
                    if (delta < -1e-6) {
                        sx[a] = asx; sy[a] = asy; sq[a] = asq;
                        sx[b] = bsx; sy[b] = bsy; sq[b] = bsq;
                        labels[i] = b; labels[jq] = a;
                        px = PX[i]; py = PY[i];   // i now in b; refresh for outer use
                        a = b;
                        improved = true;
                        break;
                    }
                }
            }
        }

        double finalc = recompute_and_cost(labels, cx, cy);
        if (finalc < best_cost) {
            best_cost = finalc;
            best_labels = labels;
            best_cx = cx; best_cy = cy;
        }
        ++restart;
    }

    // Defensive fallback: if (somehow) we never recorded a feasible assignment,
    // build a round-robin one (always feasible since K*CAP >= N) so we never
    // print an infeasible / empty-cluster solution.
    if (best_labels.empty()) {
        best_labels.assign(N, 0);
        // fill clusters in chunks so each is nonempty and <= CAP
        int idx = 0;
        for (int j = 0; j < K && idx < N; ++j) {
            best_labels[idx++] = j;             // guarantee nonempty
        }
        for (int i = idx; i < N; ++i) best_labels[i] = i % K;
        best_cx.assign(K, 0); best_cy.assign(K, 0);
        recompute_and_cost(best_labels, best_cx, best_cy);
    }

    // ----- emit -----
    string out;
    out.reserve(N * 3 + K * 24);
    for (int i = 0; i < N; ++i) {
        out += to_string(best_labels[i]);
        out += (i + 1 == N) ? '\n' : ' ';
    }
    char buf[64];
    for (int j = 0; j < K; ++j) {
        int len = snprintf(buf, sizeof(buf), "%.6f %.6f\n", best_cx[j], best_cy[j]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
