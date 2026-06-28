// Drone Survey Sweep -- heuristic solver.
//
// Objective: visit all n stations in one closed tour (every station has degree
// exactly 2 -- a degree-<=2 spanning structure that spans all stations) so that
// the total Euclidean edge length is minimized. Read the instance from stdin,
// write the visiting order (a permutation of 0..n-1, one index per line) to
// stdout.
//
// Method (the innovation):
//   1. Build a k-nearest-neighbour candidate list with a uniform spatial grid.
//   2. Greedy nearest-neighbour construction from station 0 (a feasible start).
//   3. Local search = 2-opt + Or-opt(segment length 1..3) restricted to the
//      candidate list and driven by don't-look bits. Every move's gain is an
//      O(1) incremental delta over only the few edges it touches; the full tour
//      length is NEVER recomputed inside the loop.
//   4. Iterated local search: when local search converges, perturb with a
//      double-bridge kick, re-optimise the touched nodes, and keep the better
//      tour. Repeat until a wall-clock budget is spent.
// The current tour is always a valid permutation, so any early stop (including
// hitting the time limit mid-iteration) still prints a feasible solution.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }  // [0, m)
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N;
vector<double> X, Y;

static inline double dist(int a, int b) {
    double dx = X[a] - X[b];
    double dy = Y[a] - Y[b];
    return sqrt(dx * dx + dy * dy);
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;  // wall-clock budget (seconds)

    if (scanf("%d", &N) != 1) return 0;
    if (N <= 0) return 0;
    X.resize(N);
    Y.resize(N);
    for (int i = 0; i < N; i++) {
        double xi, yi;
        if (scanf("%lf %lf", &xi, &yi) != 2) { X[i] = 0; Y[i] = 0; }
        else { X[i] = xi; Y[i] = yi; }
    }
    if (N <= 3) {  // any permutation is optimal
        for (int i = 0; i < N; i++) printf("%d\n", i);
        return 0;
    }

    Rng rng(0x1234567 ^ (uint64_t)N * 1000003ULL);

    // ---------- spatial grid + k-nearest candidate lists ----------
    double minx = 1e18, miny = 1e18, maxx = -1e18, maxy = -1e18;
    for (int i = 0; i < N; i++) {
        minx = min(minx, X[i]); maxx = max(maxx, X[i]);
        miny = min(miny, Y[i]); maxy = max(maxy, Y[i]);
    }
    double w = max(1.0, maxx - minx), h = max(1.0, maxy - miny);
    int gridN = max(1, (int)floor(sqrt((double)N / 2.0)));
    double cw = w / gridN, ch = h / gridN;
    auto cellOf = [&](int i, int &cx, int &cy) {
        cx = (int)((X[i] - minx) / cw); if (cx >= gridN) cx = gridN - 1; if (cx < 0) cx = 0;
        cy = (int)((Y[i] - miny) / ch); if (cy >= gridN) cy = gridN - 1; if (cy < 0) cy = 0;
    };
    vector<vector<int>> cell(gridN * gridN);
    for (int i = 0; i < N; i++) {
        int cx, cy; cellOf(i, cx, cy);
        cell[cy * gridN + cx].push_back(i);
    }

    int K = min(N - 1, 10);
    vector<int> nbr((size_t)N * K);
    {
        vector<pair<double,int>> cand;
        cand.reserve(64);
        for (int i = 0; i < N; i++) {
            cand.clear();
            int cx, cy; cellOf(i, cx, cy);
            int ring = 0, needRings = -1;
            while (true) {
                int x0 = max(0, cx - ring), x1 = min(gridN - 1, cx + ring);
                int y0 = max(0, cy - ring), y1 = min(gridN - 1, cy + ring);
                for (int gy = y0; gy <= y1; gy++)
                    for (int gx = x0; gx <= x1; gx++) {
                        if (ring > 0 && gx > x0 && gx < x1 && gy > y0 && gy < y1) continue;
                        for (int j : cell[gy * gridN + gx]) {
                            if (j == i) continue;
                            double dx = X[i] - X[j], dy = Y[i] - Y[j];
                            cand.push_back({dx * dx + dy * dy, j});
                        }
                    }
                if (needRings < 0 && (int)cand.size() >= K) needRings = ring + 1;
                if (needRings >= 0 && ring >= needRings) break;
                ring++;
                if (cx - ring < 0 && cx + ring >= gridN && cy - ring < 0 && cy + ring >= gridN) break;
            }
            int kk = min((int)cand.size(), K);
            partial_sort(cand.begin(), cand.begin() + kk, cand.end());
            for (int t = 0; t < K; t++)
                nbr[(size_t)i * K + t] = (t < kk) ? cand[t].second : i;  // self = harmless no-op
        }
    }

    // ---------- greedy nearest-neighbour construction ----------
    vector<int> tour(N), pos(N);
    {
        vector<char> used(N, 0);
        int cur = 0; used[0] = 1; tour[0] = 0;
        for (int step = 1; step < N; step++) {
            int best = -1; double bestd = 1e18;
            for (int t = 0; t < K; t++) {
                int j = nbr[(size_t)cur * K + t];
                if (j != cur && !used[j]) {
                    double d = dist(cur, j);
                    if (d < bestd) { bestd = d; best = j; }
                }
            }
            if (best < 0) {
                for (int j = 0; j < N; j++) if (!used[j]) {
                    double d = dist(cur, j);
                    if (d < bestd) { bestd = d; best = j; }
                }
            }
            used[best] = 1; tour[step] = best; cur = best;
        }
    }
    for (int i = 0; i < N; i++) pos[tour[i]] = i;

    auto succIdx = [&](int idx) { return idx + 1 < N ? idx + 1 : 0; };
    auto predIdx = [&](int idx) { return idx > 0 ? idx - 1 : N - 1; };

    // A 2-opt move breaks edges (u,uNext) and (v,vNext) and reconnects as
    // (u,v) + (uNext,vNext); realised by reversing the tour segment uNext..v.
    // We pass the array positions iu = pos[uNext] and iv = pos[v]. The segment
    // uNext..v in forward order may wrap; reversing the complementary segment
    // vNext..u gives an equivalent undirected tour, so we always reverse the
    // SHORTER of the two arcs -- work <= N/2. Positions stay consistent.
    auto do2optReverse = [&](int iu, int iv) {
        // length of forward segment [iu..iv] (modular, inclusive)
        int fwd = (iv - iu + N) % N + 1;
        if (fwd * 2 <= N) {
            // reverse iu..iv modularly
            int len = fwd;
            for (int s = 0; s < len / 2; s++) {
                int ai = iu + s; if (ai >= N) ai -= N;
                int bi = iv - s; if (bi < 0) bi += N;
                int u = tour[ai], v = tour[bi];
                tour[ai] = v; tour[bi] = u;
                pos[v] = ai; pos[u] = bi;
            }
        } else {
            // reverse the complementary arc (iv+1 .. iu-1) modularly
            int a = succIdx(iv), b = predIdx(iu);
            int len = N - fwd;
            for (int s = 0; s < len / 2; s++) {
                int ai = a + s; if (ai >= N) ai -= N;
                int bi = b - s; if (bi < 0) bi += N;
                int u = tour[ai], v = tour[bi];
                tour[ai] = v; tour[bi] = u;
                pos[v] = ai; pos[u] = bi;
            }
        }
    };

    // ---------- local search core (2-opt + Or-opt, candidate-restricted) ----------
    vector<char> dontlook(N, 1);
    // a simple stack of "to examine" nodes
    vector<int> stack_;
    stack_.reserve(N * 2 + 16);

    long long clk = 0;
    auto timeUp = [&]() {
        if ((++clk & 511) == 0) return now_sec() - T0 > TIME_LIMIT;
        return false;
    };

    auto pushNode = [&](int node) {
        if (dontlook[node]) { dontlook[node] = 0; stack_.push_back(node); }
    };

    // Returns true if at least one improving move was applied for node `a`.
    auto improveNode = [&](int a) -> bool {
        int ia = pos[a];
        int aPrev = tour[predIdx(ia)];
        int aNext = tour[succIdx(ia)];
        double d_a_next = dist(a, aNext);
        double d_prev_a = dist(aPrev, a);

        // ----- 2-opt, both orientations -----
        for (int t = 0; t < K; t++) {
            int c = nbr[(size_t)a * K + t];
            if (c == a) continue;
            double d_ac = dist(a, c);
            // candidates are distance-sorted: if d_ac >= both incident edges,
            // no further candidate can give a positive 2-opt gain on either side.
            if (d_ac >= d_a_next && d_ac >= d_prev_a) break;

            // orientation 1: remove (a,aNext) and (c,cNext); add (a,c),(aNext,cNext)
            if (d_ac < d_a_next) {
                int ic = pos[c];
                int cNext = tour[succIdx(ic)];
                if (cNext != a && c != aNext) {
                    double delta = (d_ac + dist(aNext, cNext)) - (d_a_next + dist(c, cNext));
                    if (delta < -1e-7) {
                        // break (a,aNext) and (c,cNext) -> reverse segment aNext..c
                        do2optReverse(pos[aNext], pos[c]);
                        pushNode(a); pushNode(aNext); pushNode(c); pushNode(cNext);
                        return true;
                    }
                }
            }
            // orientation 2: remove (aPrev,a) and (cPrev,c); add (a,c),(aPrev,cPrev)
            if (d_ac < d_prev_a) {
                int ic = pos[c];
                int cPrev = tour[predIdx(ic)];
                if (cPrev != a && c != aPrev) {
                    double delta = (d_ac + dist(aPrev, cPrev)) - (d_prev_a + dist(cPrev, c));
                    if (delta < -1e-7) {
                        // break (aPrev,a) and (cPrev,c) -> reverse segment a..cPrev
                        do2optReverse(pos[a], pos[cPrev]);
                        pushNode(a); pushNode(aPrev); pushNode(c); pushNode(cPrev);
                        return true;
                    }
                }
            }
        }

        // ----- Or-opt: relocate a short segment starting at a (len 1..3) -----
        for (int segLen = 1; segLen <= 3 && segLen < N - 2; segLen++) {
            int ia2 = pos[a];
            int endIdx = ia2;
            for (int s = 1; s < segLen; s++) endIdx = succIdx(endIdx);
            int segEnd = tour[endIdx];
            int p = tour[predIdx(ia2)];
            int q = tour[succIdx(endIdx)];
            if (p == segEnd || q == a) break;  // segment wraps whole tour
            double removeGain = dist(p, a) + dist(segEnd, q) - dist(p, q);
            if (removeGain <= 1e-7) continue;
            for (int t = 0; t < K; t++) {
                int c = nbr[(size_t)a * K + t];
                if (c == a) continue;
                // c must be outside the segment and not == p (no-op)
                int ic = pos[c];
                // is c inside [ia2..endIdx] (modular)? skip if so
                bool inside = false;
                {
                    int span = (endIdx - ia2 + N) % N;
                    int off = (ic - ia2 + N) % N;
                    if (off <= span) inside = true;
                }
                if (inside || c == p) continue;
                int cNext = tour[succIdx(ic)];
                if (cNext == a) continue;  // inserting where it already is
                // insert segment (a..segEnd) between c and cNext, forward orientation
                double addCost = dist(c, a) + dist(segEnd, cNext) - dist(c, cNext);
                if (addCost + 1e-7 < removeGain) {
                    // perform relocation by rebuilding the order around the splice.
                    // collect the segment nodes
                    static vector<int> seg; seg.clear();
                    int idx = ia2;
                    for (int s = 0; s < segLen; s++) { seg.push_back(tour[idx]); idx = succIdx(idx); }
                    // build new tour: walk current tour skipping seg, insert after c
                    static vector<int> nt; nt.clear(); nt.reserve(N);
                    // mark seg membership
                    // (use a small set check; segLen<=3 so linear scan is fine)
                    for (int k = 0; k < N; k++) {
                        int v = tour[k];
                        bool isSeg = false;
                        for (int s = 0; s < segLen; s++) if (seg[s] == v) { isSeg = true; break; }
                        if (isSeg) continue;
                        nt.push_back(v);
                    }
                    static vector<int> res; res.clear(); res.reserve(N);
                    for (int v : nt) {
                        res.push_back(v);
                        if (v == c) for (int s = 0; s < segLen; s++) res.push_back(seg[s]);
                    }
                    tour.swap(res);
                    for (int k = 0; k < N; k++) pos[tour[k]] = k;
                    pushNode(p); pushNode(q); pushNode(c); pushNode(cNext);
                    pushNode(a); pushNode(segEnd);
                    return true;
                }
            }
        }
        return false;
    };

    auto runLocalSearch = [&]() {
        while (!stack_.empty()) {
            if (timeUp()) return;
            int a = stack_.back(); stack_.pop_back();
            if (dontlook[a]) continue;
            bool imp = improveNode(a);
            if (!imp) dontlook[a] = 1;
            else { dontlook[a] = 0; stack_.push_back(a); }
        }
    };

    auto tourLength = [&]() {
        double L = 0;
        for (int i = 0; i < N; i++) L += dist(tour[i], tour[succIdx(i)]);
        return L;
    };

    // initial full optimisation
    for (int i = 0; i < N; i++) { dontlook[tour[i]] = 0; }
    stack_.assign(tour.begin(), tour.end());
    runLocalSearch();

    // ---------- iterated local search with double-bridge kicks ----------
    vector<int> best = tour, bestPos = pos;
    double bestLen = tourLength();

    while (now_sec() - T0 < TIME_LIMIT) {
        // double-bridge: pick 3 cut points 0<p1<p2<p3<N, reconnect A D C B
        if (N >= 8) {
            int p1 = 1 + rng.nextu(N - 3);
            int p2 = p1 + 1 + rng.nextu(N - p1 - 2);
            int p3 = p2 + 1 + rng.nextu(N - p2 - 1);
            static vector<int> nt; nt.clear(); nt.reserve(N);
            for (int i = 0; i < p1; i++) nt.push_back(tour[i]);
            for (int i = p3; i < N; i++) nt.push_back(tour[i]);
            for (int i = p2; i < p3; i++) nt.push_back(tour[i]);
            for (int i = p1; i < p2; i++) nt.push_back(tour[i]);
            tour.swap(nt);
            for (int i = 0; i < N; i++) pos[tour[i]] = i;
            // wake nodes near the four new junctions
            int cuts[4] = {0, p1, p2, p3};
            stack_.clear();
            for (int cidx = 0; cidx < 4; cidx++) {
                int base = cuts[cidx];
                for (int o = -2; o <= 2; o++) {
                    int idx = ((base + o) % N + N) % N;
                    pushNode(tour[idx]);
                }
            }
        } else {
            // tiny instance: just re-wake everything
            stack_.assign(tour.begin(), tour.end());
            for (int v : tour) dontlook[v] = 0;
        }
        runLocalSearch();
        double L = tourLength();
        if (L + 1e-6 < bestLen) {
            bestLen = L; best = tour; bestPos = pos;
        } else {
            // revert to best to keep the kick from drifting away
            tour = best; pos = bestPos;
        }
    }
    tour = best;

    // ---------- output (always a valid permutation) ----------
    {
        vector<char> seen(N, 0);
        bool ok = true;
        for (int i = 0; i < N; i++) {
            int v = tour[i];
            if (v < 0 || v >= N || seen[v]) { ok = false; break; }
            seen[v] = 1;
        }
        if (!ok) for (int i = 0; i < N; i++) tour[i] = i;
    }
    string out; out.reserve((size_t)N * 7);
    char buf[16];
    for (int i = 0; i < N; i++) {
        int len = snprintf(buf, sizeof(buf), "%d\n", tour[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
