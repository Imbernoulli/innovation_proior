#include <bits/stdc++.h>
using namespace std;

// ------------------------------------------------------------------------
// Graph Coloring -- minimize the number of colors of a PROPER coloring.
//
// Strategy (the established strong heuristic family):
//   1. DSATUR construction -> an initial proper coloring using k0 colors.
//      This is our always-valid baseline; we never lose it.
//   2. Color reduction by TabuCol descent on the target k. To go from a proper
//      k-coloring to a proper (k-1)-coloring, we DROP the highest color class
//      (reassign its vertices into [0,k-1) classes), which creates some
//      monochromatic (conflicting) edges, then run tabu search that recolors one
//      conflicting vertex at a time to drive the number of conflicts to 0. If we
//      reach 0 conflicts we have a proper (k-1)-coloring; save it and try k-2.
//      If the budget runs out before 0, we keep the best PROPER coloring found.
//   3. Incremental gamma[v][c] table: gamma[v][c] = number of v's neighbours
//      currently colored c. The conflict delta of recoloring v from its color to
//      c is gamma[v][c] - gamma[v][color[v]] (O(1)); applying a recolor updates
//      gamma only for v's neighbours (O(degree)). This makes thousands of tabu
//      iterations per second feasible.
//
// The program ALWAYS prints a feasible (proper) coloring within the time budget:
// the DSATUR result is proper, and every k we accept is verified to have 0
// conflicts before we adopt it.
// ------------------------------------------------------------------------

static const double TIME_LIMIT = 1.9;  // seconds, wall clock
static std::chrono::steady_clock::time_point T0;
static inline double elapsed() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
}

// xorshift RNG (fast, deterministic)
static uint64_t rng_state = 88172645463325252ULL;
static inline uint64_t xrand() {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x;
    return x;
}
static inline int randint(int n) { return (int)(xrand() % (uint64_t)n); }

int n, m;
vector<int> adjStart;          // CSR adjacency
vector<int> adjList;
vector<int> deg;

// build CSR adjacency from an edge list
void buildAdj(const vector<pair<int,int>>& edges) {
    deg.assign(n, 0);
    for (auto& e : edges) { deg[e.first]++; deg[e.second]++; }
    adjStart.assign(n + 1, 0);
    for (int i = 0; i < n; i++) adjStart[i + 1] = adjStart[i] + deg[i];
    adjList.assign(adjStart[n], 0);
    vector<int> cur(adjStart.begin(), adjStart.begin() + n);
    for (auto& e : edges) {
        adjList[cur[e.first]++] = e.second;
        adjList[cur[e.second]++] = e.first;
    }
}

// ---------------- DSATUR construction ----------------
// Order: repeatedly pick the uncolored vertex of maximum saturation degree
// (number of distinct colors among its neighbours), ties broken by larger
// (uncolored) degree, then assign the smallest feasible color.
vector<int> dsatur() {
    vector<int> color(n, -1);
    vector<int> satDeg(n, 0);                 // distinct neighbour colors
    vector<int> uncDeg = deg;                 // remaining (uncolored) degree
    // neighborColorMask[v] as a set; for n up to ~600 colors <= n, use a
    // per-vertex set of used neighbour colors via a small hash set substitute:
    vector<vector<char>> used(n);             // used[v][c] = neighbour uses c
    // we size each used[v] lazily as colors grow; simpler: cap at n colors.
    for (int v = 0; v < n; v++) used[v].assign(1, 0);

    auto ensureSize = [&](int v, int c) {
        if ((int)used[v].size() <= c) used[v].resize(c + 1, 0);
    };

    for (int iter = 0; iter < n; iter++) {
        // select uncolored vertex with max (satDeg, then uncDeg, then -index)
        int best = -1;
        int bSat = -1, bDeg = -1;
        for (int v = 0; v < n; v++) {
            if (color[v] != -1) continue;
            if (satDeg[v] > bSat ||
                (satDeg[v] == bSat && uncDeg[v] > bDeg)) {
                bSat = satDeg[v]; bDeg = uncDeg[v]; best = v;
            }
        }
        if (best == -1) break;
        int v = best;
        // smallest color not used by a neighbour
        int c = 0;
        while (c < (int)used[v].size() && used[v][c]) c++;
        color[v] = c;
        // update neighbours
        for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
            int u = adjList[idx];
            if (color[u] != -1) continue;
            uncDeg[u]--;
            ensureSize(u, c);
            if (!used[u][c]) {
                used[u][c] = 1;
                satDeg[u]++;
            }
        }
    }
    return color;
}

// number of distinct colors in a coloring
int countColors(const vector<int>& color) {
    int mx = -1;
    for (int v = 0; v < n; v++) mx = max(mx, color[v]);
    if (mx < 0) return 0;
    vector<char> seen(mx + 1, 0);
    int cnt = 0;
    for (int v = 0; v < n; v++) if (!seen[color[v]]) { seen[color[v]] = 1; cnt++; }
    return cnt;
}

// is the coloring proper?
bool isProper(const vector<int>& color) {
    for (int v = 0; v < n; v++)
        for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
            int u = adjList[idx];
            if (u > v && color[u] == color[v]) return false;
        }
    return true;
}

// ---------------- TabuCol: try to find a proper k-coloring ----------------
// Start from a (possibly improper) k-coloring; minimize the number of
// conflicting edges to 0 by recoloring one conflicting vertex at a time, with a
// tabu list forbidding immediate reversal. Returns true and writes `color` if it
// reaches 0 conflicts within the iteration / time budget.
//
// gamma[v*k + c] = number of v's neighbours currently colored c.
// nConflicts     = number of monochromatic edges (each counted once).
bool tabuCol(int k, vector<int>& color, long long maxIters) {
    if (k <= 0) return false;
    vector<int> gamma((size_t)n * k, 0);
    auto G = [&](int v, int c) -> int& { return gamma[(size_t)v * k + c]; };

    // init gamma and conflict count
    for (int v = 0; v < n; v++) {
        for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
            int u = adjList[idx];
            G(v, color[u])++;
        }
    }
    long long nConflicts = 0;  // counts each conflicting endpoint pair once
    for (int v = 0; v < n; v++) nConflicts += G(v, color[v]);
    nConflicts /= 2;

    if (nConflicts == 0) return true;

    // tabuTenure[v*k + c] = iteration until which assigning color c to v is tabu
    vector<long long> tabu((size_t)n * k, -1);
    auto TB = [&](int v, int c) -> long long& { return tabu[(size_t)v * k + c]; };

    long long bestConf = nConflicts;
    long long it = 0;
    // dynamic tabu tenure: base + fraction of current conflicts (Galinier-Hao)
    while (it < maxIters && nConflicts > 0) {
        if ((it & 1023) == 0 && elapsed() > TIME_LIMIT) break;
        it++;

        // pick the move that best reduces conflicts among CONFLICTING vertices;
        // standard TabuCol: scan conflicting vertices, for each its k-1 other
        // colors, choose the best non-tabu (with aspiration) delta.
        int bestV = -1, bestC = -1;
        long long bestDelta = LLONG_MAX;
        int nTied = 0;

        for (int v = 0; v < n; v++) {
            int cv = color[v];
            int gv = G(v, cv);
            if (gv == 0) continue;            // v is not in conflict; skip
            for (int c = 0; c < k; c++) {
                if (c == cv) continue;
                long long delta = (long long)G(v, c) - gv;  // change in conflicts
                bool isTabu = (TB(v, c) >= it);
                bool aspire = (nConflicts + delta < bestConf);
                if (isTabu && !aspire) continue;
                if (delta < bestDelta) {
                    bestDelta = delta; bestV = v; bestC = c; nTied = 1;
                } else if (delta == bestDelta) {
                    // reservoir tie-break to avoid bias
                    nTied++;
                    if (randint(nTied) == 0) { bestV = v; bestC = c; }
                }
            }
        }

        if (bestV == -1) {
            // every reducing move is tabu and none aspires: make a random
            // perturbing recolor of a conflicting vertex to escape.
            // pick a random conflicting vertex
            int tries = 0, v = -1;
            while (tries < 50) {
                int cand = randint(n);
                if (G(cand, color[cand]) > 0) { v = cand; break; }
                tries++;
            }
            if (v == -1) break;  // no conflicting vertex found by sampling
            int cv = color[v];
            int c = randint(k);
            if (c == cv) c = (c + 1) % k;
            bestV = v; bestC = c;
            bestDelta = (long long)G(v, c) - G(v, cv);
        }

        // apply the recolor of bestV: cv -> bestC
        int v = bestV, cv = color[v], nc = bestC;
        nConflicts += bestDelta;
        // update gamma for neighbours
        for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
            int u = adjList[idx];
            G(u, cv)--;
            G(u, nc)++;
        }
        color[v] = nc;
        // tabu: forbid putting v back to cv for a dynamic tenure
        long long tenure = 8 + (long long)(0.6 * (double)nConflicts) + randint(10);
        TB(v, cv) = it + tenure;

        if (nConflicts < bestConf) bestConf = nConflicts;
    }

    return nConflicts == 0;
}

// build an initial (possibly improper) k-coloring from a proper coloring by
// reassigning every vertex whose color falls outside [0,k) into [0,k). With
// `randomize` false this is a deterministic least-conflict placement; with
// `randomize` true the over-budget vertices are scattered randomly, which gives
// TabuCol a genuinely different starting basin on each restart.
vector<int> reduceToK(const vector<int>& proper, int k, bool randomize) {
    // relabel proper coloring colors to a contiguous 0..K-1 first
    int mx = -1;
    for (int v = 0; v < n; v++) mx = max(mx, proper[v]);
    vector<int> remap(mx + 1, -1);
    int next = 0;
    for (int v = 0; v < n; v++) {
        if (remap[proper[v]] == -1) remap[proper[v]] = next++;
    }
    vector<int> color(n);
    for (int v = 0; v < n; v++) color[v] = remap[proper[v]];
    // any vertex whose color >= k must move into [0,k).
    for (int v = 0; v < n; v++) {
        if (color[v] < k) continue;
        if (randomize) {
            color[v] = randint(k);
        } else {
            // greedily place it in the color that adds the fewest conflicts
            // among its already-known neighbours (a light touch; tabu cleans up).
            vector<int> cnt(k, 0);
            for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
                int u = adjList[idx];
                if (color[u] < k) cnt[color[u]]++;
            }
            int bc = 0;
            for (int c = 1; c < k; c++) if (cnt[c] < cnt[bc]) bc = c;
            color[v] = bc;
        }
    }
    return color;
}

int main() {
    T0 = std::chrono::steady_clock::now();

    // fast input
    if (scanf("%d %d", &n, &m) != 2) {
        // nothing to read; emit nothing (n unknown). Treat as empty.
        return 0;
    }
    vector<pair<int,int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        edges[i] = {u, v};
    }
    if (n <= 0) { return 0; }

    buildAdj(edges);

    // 1) DSATUR construction -> always-proper baseline.
    vector<int> best = dsatur();
    int bestK = countColors(best);
    fprintf(stderr, "DSATUR k0=%d\n", bestK);

    // 2) Color reduction: try k = bestK-1, bestK-2, ... by TabuCol descent.
    //    We keep the best PROPER coloring found at all times. For each target k
    //    we make REPEATED randomized restarts of TabuCol until either we find a
    //    proper k-coloring or we exhaust the per-k restart budget; only after
    //    several restarts all fail do we conclude k is out of reach and stop.
    //    Each TabuCol call is bounded by a wall-clock guard, so the global
    //    TIME_LIMIT is always respected.
    long long itersPerK = 200000LL + (long long)n * 2000LL;

    int targetK = bestK - 1;
    while (targetK >= 1 && elapsed() < TIME_LIMIT) {
        bool solved = false;
        // up to a handful of randomized restarts for this target k; the first
        // attempt uses the least-conflict reduction, later ones scatter randomly.
        for (int restart = 0; restart < 6 && elapsed() < TIME_LIMIT; restart++) {
            vector<int> cand = reduceToK(best, targetK, /*randomize=*/restart > 0);
            bool _ok = tabuCol(targetK, cand, itersPerK);
            fprintf(stderr, "  k=%d restart=%d -> %s (t=%.2f)\n", targetK, restart, _ok?"OK":"FAIL", elapsed());
            if (_ok) {
                best = cand;
                bestK = countColors(best);   // may be < targetK if a color went unused
                targetK = bestK - 1;
                solved = true;
                break;
            }
        }
        if (!solved) {
            // none of the restarts reached a proper coloring with this k:
            // treat k as out of reach and stop descending.
            break;
        }
    }

    // 3) safety: guarantee output is proper. If for any reason `best` is not
    //    proper, fall back to a fresh DSATUR (always proper).
    if (!isProper(best)) {
        best = dsatur();
    }

    // output: one color per line, colors relabeled to 0..K-1 (cosmetic).
    int mx = -1;
    for (int v = 0; v < n; v++) mx = max(mx, best[v]);
    vector<int> remap(mx + 1, -1);
    int next = 0;
    for (int v = 0; v < n; v++) if (remap[best[v]] == -1) remap[best[v]] = next++;

    string out;
    out.reserve((size_t)n * 4);
    for (int v = 0; v < n; v++) {
        out += to_string(remap[best[v]]);
        out += '\n';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
