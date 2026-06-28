// Graph Coloring with Soft Conflicts -- heuristic solver (TabuCol).
//
// Objective: given a weighted undirected graph and a budget of k colors, assign
// each vertex a color in 0..k-1 to MINIMIZE the total weight of monochromatic
// ("conflict") edges. Read the instance from stdin; print n colors (one per
// line, vertex 0..n-1) to stdout. ANY assignment of colors in [0,k-1] is a
// feasible output, so we always have something legal to print.
//
// Method (the innovation): TabuCol -- tabu search over colorings with an
// INCREMENTAL conflict-count table.
//   gamma[v*k + c] = weighted number of conflicts vertex v would incur if it
//                    took color c = sum of edge weights to neighbours currently
//                    colored c.
// The current total conflict cost is sum over conflicting edges of their weight.
// A move recolors one vertex v from its current color to a new color c'. Its
// cost delta is exactly gamma[v][c'] - gamma[v][cur] -- an O(1) lookup, no
// recomputation. Applying the move updates gamma only for v's neighbours: for
// each neighbour u, gamma[u][cur] -= w(u,v) and gamma[u][c'] += w(u,v) -- an
// O(degree(v)) update, NOT O(n) or O(m). Each iteration we scan only the
// currently-conflicting vertices and pick the best non-tabu recolor (with
// aspiration: a tabu move is allowed if it would beat the best cost ever seen).
// A tabu tenure proportional to the number of conflicting vertices (plus a
// random jitter) lets the search step off plateaus instead of cycling.
//
// Feasibility is trivially preserved (colors stay in [0,k-1] at all times), so
// hitting the time budget mid-search still prints a valid coloring.
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
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }   // [0, m)
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.90;  // wall-clock budget (seconds)

    int n, m, k;
    if (scanf("%d %d %d", &n, &m, &k) != 3) return 0;
    if (n <= 0) return 0;
    if (k <= 0) k = 1;

    // ---- read edges, build weighted CSR adjacency ----
    vector<int> eu(m), ev(m);
    vector<long long> ew(m);
    vector<int> deg(n, 0);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        if (scanf("%d %d %lld", &u, &v, &w) != 3) { eu[i] = ev[i] = 0; ew[i] = 0; }
        else { eu[i] = u; ev[i] = v; ew[i] = w; }
        if (eu[i] < 0 || eu[i] >= n) eu[i] = 0;
        if (ev[i] < 0 || ev[i] >= n) ev[i] = 0;
        if (ew[i] < 0) ew[i] = 0;
        if (eu[i] != ev[i]) { deg[eu[i]]++; deg[ev[i]]++; }
    }
    vector<int> adjStart(n + 1, 0);
    for (int i = 0; i < n; i++) adjStart[i + 1] = adjStart[i] + deg[i];
    int totDeg = adjStart[n];
    vector<int> adjV(totDeg);
    vector<long long> adjW(totDeg);
    {
        vector<int> cur(adjStart.begin(), adjStart.end());
        for (int i = 0; i < m; i++) {
            int u = eu[i], v = ev[i]; long long w = ew[i];
            if (u == v) continue;
            adjV[cur[u]] = v; adjW[cur[u]] = w; cur[u]++;
            adjV[cur[v]] = u; adjW[cur[v]] = w; cur[v]++;
        }
    }

    Rng rng(0x9A2C1F ^ ((uint64_t)n * 1000003ULL) ^ ((uint64_t)m * 19349663ULL) ^ (uint64_t)k);

    auto neiBegin = [&](int v) { return adjStart[v]; };
    auto neiEnd   = [&](int v) { return adjStart[v + 1]; };

    // ---- color[] and the incremental conflict-count table gamma[v*k + c] ----
    vector<int> color(n, 0);
    vector<long long> gamma((size_t)n * k, 0);

    // greedy DSATUR-style construction: order by descending weighted degree,
    // give each vertex the color minimizing conflict weight with colored
    // neighbours. This is a strong feasible start (and mirrors the baseline).
    {
        vector<long long> wdeg(n, 0);
        for (int i = 0; i < n; i++)
            for (int e = neiBegin(i); e < neiEnd(i); e++) wdeg[i] += adjW[e];
        vector<int> order(n);
        for (int i = 0; i < n; i++) order[i] = i;
        sort(order.begin(), order.end(), [&](int a, int b) {
            if (wdeg[a] != wdeg[b]) return wdeg[a] > wdeg[b];
            return a < b;
        });
        vector<char> placed(n, 0);
        vector<long long> cost(k, 0);
        for (int u : order) {
            for (int c = 0; c < k; c++) cost[c] = 0;
            for (int e = neiBegin(u); e < neiEnd(u); e++) {
                int v = adjV[e];
                if (placed[v]) cost[color[v]] += adjW[e];
            }
            int bc = 0; long long bcost = cost[0];
            for (int c = 1; c < k; c++) if (cost[c] < bcost) { bcost = cost[c]; bc = c; }
            color[u] = bc;
            placed[u] = 1;
        }
    }

    // Build gamma from the constructed coloring: gamma[v][c] = weight of v's
    // neighbours currently colored c.
    for (int v = 0; v < n; v++) {
        long long *gv = &gamma[(size_t)v * k];
        for (int e = neiBegin(v); e < neiEnd(v); e++)
            gv[color[adjV[e]]] += adjW[e];
    }

    // current total conflict weight = (1/2) sum_v gamma[v][color[v]]
    long long curCost = 0;
    for (int v = 0; v < n; v++) curCost += gamma[(size_t)v * k + color[v]];
    curCost /= 2;

    // ---- snapshot of the best coloring found ----
    vector<int> best = color;
    long long bestCost = curCost;

    if (bestCost == 0 || k == 1) {
        // already conflict-free, or only one color possible: nothing to improve.
        string out; out.reserve((size_t)n * 3);
        char buf[16];
        for (int i = 0; i < n; i++) { int len = snprintf(buf, sizeof(buf), "%d\n", color[i]); out.append(buf, len); }
        fputs(out.c_str(), stdout);
        return 0;
    }

    // ---- tabu table: tabu[v*k + c] = iteration until which (v -> c) is tabu ----
    vector<long long> tabu((size_t)n * k, 0);
    long long iter = 0;

    // list of currently-conflicting vertices (a vertex is conflicting iff
    // gamma[v][color[v]] > 0). We scan only these to find moves.
    // Maintained as a membership flag + a compact vector we rebuild cheaply.
    vector<char> inConf(n, 0);

    long long clk = 0;
    auto timeUp = [&]() {
        if ((++clk & 255) == 0) return now_sec() - T0 > TIME_LIMIT;
        return false;
    };

    // Apply a recolor of vertex v to color nc, updating gamma incrementally and
    // adjusting curCost. Returns nothing; O(degree(v)).
    auto applyMove = [&](int v, int nc) {
        int oc = color[v];
        if (oc == nc) return;
        long long *gv = &gamma[(size_t)v * k];
        // cost delta is exactly the change in conflicts incident to v.
        curCost += gv[nc] - gv[oc];
        color[v] = nc;
        for (int e = neiBegin(v); e < neiEnd(v); e++) {
            int u = adjV[e]; long long w = adjW[e];
            long long *gu = &gamma[(size_t)u * k];
            gu[oc] -= w;
            gu[nc] += w;
        }
    };

    // Main TabuCol loop.
    while (true) {
        if (timeUp()) break;
        if (curCost == 0) {  // perfect coloring -> can't do better
            if (curCost < bestCost) { bestCost = curCost; best = color; }
            break;
        }
        iter++;

        // Recollect conflicting vertices (those whose current color conflicts).
        // This is O(n) per iteration but with tiny constant; the inner move
        // search below is what the incremental gamma makes cheap.
        // Find the best move: over conflicting vertices v and colors c != cur,
        // delta = gamma[v][c] - gamma[v][cur]. Pick the most-improving non-tabu
        // move; aspiration overrides tabu if it beats the global best.
        long long bestDelta = LLONG_MAX;
        int bv = -1, bc = -1;
        long long bestTabuDelta = LLONG_MAX;
        int btv = -1, btc = -1;
        int ties = 0, tabuTies = 0;

        for (int v = 0; v < n; v++) {
            long long *gv = &gamma[(size_t)v * k];
            int cur = color[v];
            long long gcur = gv[cur];
            if (gcur == 0) continue;  // v is not in conflict -> skip
            for (int c = 0; c < k; c++) {
                if (c == cur) continue;
                long long delta = gv[c] - gcur;   // O(1) incremental cost delta
                bool isTabu = tabu[(size_t)v * k + c] > iter;
                if (!isTabu) {
                    if (delta < bestDelta) {
                        bestDelta = delta; bv = v; bc = c; ties = 1;
                    } else if (delta == bestDelta) {
                        // reservoir tie-break to diversify
                        if ((rng.nextu((uint32_t)(++ties)) == 0)) { bv = v; bc = c; }
                    }
                } else {
                    if (delta < bestTabuDelta) {
                        bestTabuDelta = delta; btv = v; btc = c; tabuTies = 1;
                    } else if (delta == bestTabuDelta) {
                        if ((rng.nextu((uint32_t)(++tabuTies)) == 0)) { btv = v; btc = c; }
                    }
                }
            }
        }

        int mv = -1, mc = -1;
        long long mdelta = 0;
        // aspiration: a tabu move that reaches a strictly better-than-best cost.
        if (btv >= 0 && curCost + bestTabuDelta < bestCost &&
            (bv < 0 || bestTabuDelta < bestDelta)) {
            mv = btv; mc = btc; mdelta = bestTabuDelta;
        } else if (bv >= 0) {
            mv = bv; mc = bc; mdelta = bestDelta;
        } else if (btv >= 0) {
            // every move is tabu -> take the least-bad tabu move anyway
            mv = btv; mc = btc; mdelta = bestTabuDelta;
        } else {
            // no conflicting vertex had an alternative color (shouldn't happen
            // for k>=2 with conflicts), but guard anyway.
            break;
        }

        int oldColor = color[mv];
        applyMove(mv, mc);

        // make moving mv back to its OLD color tabu for a tuned tenure:
        // tenure = base * (#conflicting vertices) + random jitter. We approximate
        // the conflict count by deriving it from the cost change pattern; a
        // light, robust choice is a small constant plus a random term scaled by
        // how many vertices are currently in conflict.
        // Count conflicting vertices cheaply only occasionally to set tenure.
        // Use a standard TabuCol tenure: L = random(0..9) + 0.6 * f, where f is
        // the number of conflicting vertices. We track f by a running estimate.
        // (A cheap exact recount of f over n is fine at this scale.)
        long long f = 0;
        for (int v = 0; v < n; v++)
            if (gamma[(size_t)v * k + color[v]] > 0) { f++; }
        long long tenure = (long long)rng.nextu(10) + (long long)(0.6 * (double)f) + 1;
        tabu[(size_t)mv * k + oldColor] = iter + tenure;

        (void)mdelta; (void)inConf;
        if (curCost < bestCost) {
            bestCost = curCost;
            best = color;
        }
    }

    // ---- output the best coloring found (always valid: colors in [0,k-1]) ----
    color = best;
    // final safety clamp (defensive; colors are always in range by construction)
    for (int i = 0; i < n; i++) {
        if (color[i] < 0 || color[i] >= k) color[i] = 0;
    }
    string out; out.reserve((size_t)n * 3);
    char buf[16];
    for (int i = 0; i < n; i++) {
        int len = snprintf(buf, sizeof(buf), "%d\n", color[i]);
        out.append(buf, len);
    }
    fputs(out.c_str(), stdout);
    return 0;
}
