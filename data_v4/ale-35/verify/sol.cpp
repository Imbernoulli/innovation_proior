#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Flood-Control Levee Placement.
//
// Grid of integer heights. Water starts at SOURCE cells and spreads to a
// 4-adjacent neighbour v from a flooded cell u whenever h[u] >= h[v] (downhill
// or level, never strictly uphill). A levee makes its cell impassable: it never
// floods and water cannot flow through it. We may place at most B levees (not on
// a source). Goal: MINIMISE the number of flooded cells.
//
// Strategy:
//   * Baseline: place zero levees -- always feasible, always printable.
//   * Single-pass marginal benefit (the innovation): from ONE flood we build the
//     flood BFS forest and, in one extra linear pass, estimate every candidate
//     levee's "downstream subtree size" = how many cells would lose flooding if
//     that cell were cut, using a dominator-flavoured count (a cell counts toward
//     its parent only if the parent is its UNIQUE flooded in-neighbour). This
//     ranks the narrow passes (bottlenecks) at the top without re-flooding per
//     candidate.
//   * Greedy construction: repeatedly flood, take the best-scoring frontier cell
//     as a levee, repeat up to B times -- a strong, always-feasible start.
//   * SA polish: hill-climb / anneal over the levee SET, each candidate scored by
//     ONE exact re-flood (grid is small), to escape greedy local optima. The best
//     set ever seen is kept and printed, so the output is always feasible.
// ---------------------------------------------------------------------------

static int H, W, B, S;
static vector<int> ht;          // height grid, row-major, size H*W
static vector<char> isSrc;      // 1 if cell is a source
static vector<int> srcs;        // source cell indices

static inline int idx(int r, int c) { return r * W + c; }

static const int DR[4] = {1, -1, 0, 0};
static const int DC[4] = {0, 0, 1, -1};

// Flood with a given blocked mask; returns number of flooded cells. If `order`
// is non-null, fills it with the BFS visit order and `parent` with each cell's
// BFS-tree parent (-1 for sources / unvisited); `indeg` counts how many flooded
// neighbours could flow INTO each flooded cell (for the dominator estimate).
static int flood(const vector<char> &blocked, vector<char> &fl,
                 vector<int> *order = nullptr, vector<int> *parent = nullptr,
                 vector<int> *indeg = nullptr) {
    int N = H * W;
    fill(fl.begin(), fl.end(), 0);
    if (order) order->clear();
    if (parent) fill(parent->begin(), parent->end(), -1);
    if (indeg) fill(indeg->begin(), indeg->end(), 0);
    // simple FIFO queue
    static vector<int> q;
    q.clear();
    q.reserve(N);
    int head = 0;
    for (int s : srcs) {
        if (blocked[s]) continue;  // (sources are never blocked, but be safe)
        if (!fl[s]) {
            fl[s] = 1;
            q.push_back(s);
        }
    }
    int cnt = (int)q.size();
    while (head < (int)q.size()) {
        int u = q[head++];
        if (order) order->push_back(u);
        int r = u / W, c = u % W;
        int hu = ht[u];
        for (int d = 0; d < 4; d++) {
            int nr = r + DR[d], nc = c + DC[d];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = idx(nr, nc);
            if (blocked[v]) continue;
            if (hu >= ht[v]) {                 // u can flow into v
                if (!fl[v]) {
                    fl[v] = 1;
                    cnt++;
                    if (parent) (*parent)[v] = u;
                    q.push_back(v);
                }
            }
        }
    }
    // second pass for in-degrees (count flooded in-neighbours of each flooded v)
    if (indeg) {
        for (int u = 0; u < N; u++) {
            if (!fl[u]) continue;
            int r = u / W, c = u % W, hu = ht[u];
            for (int d = 0; d < 4; d++) {
                int nr = r + DR[d], nc = c + DC[d];
                if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                int v = idx(nr, nc);
                if (!fl[v]) continue;
                if (hu >= ht[v]) (*indeg)[v]++;  // u -> v is a flood edge
            }
        }
    }
    return cnt;
}

int main() {
    if (scanf("%d %d %d %d", &H, &W, &B, &S) != 4) return 0;
    int N = H * W;
    ht.assign(N, 0);
    for (int i = 0; i < N; i++) scanf("%d", &ht[i]);
    isSrc.assign(N, 0);
    srcs.clear();
    for (int i = 0; i < S; i++) {
        int sr, sc;
        scanf("%d %d", &sr, &sc);
        int s = idx(sr, sc);
        isSrc[s] = 1;
        srcs.push_back(s);
    }

    auto start = chrono::steady_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::steady_clock::now() - start).count();
    };
    const double TIME_LIMIT = 1.8;  // seconds, leave margin under a 2s budget

    vector<char> blocked(N, 0);
    vector<char> fl(N, 0);
    vector<int> order, parent(N, -1), indeg(N, 0);

    // ---- GREEDY CONSTRUCTION using the single-pass dominator estimate --------
    // levees[] is the current chosen set (always feasible: never a source/dup).
    vector<int> levees;
    levees.reserve(B);

    for (int step = 0; step < B; step++) {
        int curFlood = flood(blocked, fl, &order, &parent, &indeg);
        if (curFlood <= (int)srcs.size()) break;  // only sources left, nothing to gain

        // dominator-flavoured subtree size: process BFS order in REVERSE,
        // sub[u] starts at 1; a child v adds sub[v] to its parent ONLY when v has
        // a unique flooded in-neighbour (indeg[v]==1) -- i.e. cutting the parent
        // truly disconnects v's whole subtree. This under-counts (safe) when
        // there are alternate paths, exactly capturing bottleneck passes.
        vector<int> sub(N, 0);
        for (int u : order) sub[u] = 1;
        for (int i = (int)order.size() - 1; i >= 0; i--) {
            int v = order[i];
            int p = parent[v];
            if (p >= 0 && indeg[v] == 1) sub[p] += sub[v];
        }
        // choose the best NON-source, non-already-levee flooded cell
        int best = -1, bestGain = 0;
        for (int u : order) {
            if (isSrc[u]) continue;
            if (blocked[u]) continue;
            if (sub[u] > bestGain) {
                bestGain = sub[u];
                best = u;
            }
        }
        if (best < 0 || bestGain <= 0) break;
        blocked[best] = 1;
        levees.push_back(best);
    }

    // exact flood count of the greedy solution
    int bestFlood = flood(blocked, fl);
    vector<int> bestLevees = levees;
    vector<char> bestBlocked = blocked;

    // ---- SA / LOCAL SEARCH polish over the levee set --------------------------
    // Each candidate set is scored by ONE exact re-flood. Moves:
    //   (a) ADD a levee (if under budget) on a currently-flooded non-source cell,
    //   (b) REMOVE a levee,
    //   (c) MOVE a levee to a flooded non-source cell.
    // Accept by the SA rule; always remember the best feasible set seen.
    std::mt19937 rng(0x9E3779B9u ^ (unsigned)N ^ ((unsigned)B << 16));
    auto frand = [&]() { return (rng() >> 8) * (1.0 / 16777216.0); };

    // candidate pool = cells flooded under NO levees and not sources (the cells
    // worth ever blocking). Recomputed once; the union of all reachable cells.
    vector<char> noLev(N, 0);
    vector<char> flNo(N, 0);
    flood(noLev, flNo);
    vector<int> pool;
    for (int u = 0; u < N; u++)
        if (flNo[u] && !isSrc[u]) pool.push_back(u);

    if (!pool.empty()) {
        // working copy of the current (accepted) solution
        vector<char> curBlocked = bestBlocked;
        vector<int> curLevees = bestLevees;
        int curFlood = bestFlood;

        double T0 = max(4.0, bestFlood * 0.05);  // initial temperature
        long long iter = 0;
        while (elapsed() < TIME_LIMIT) {
            // periodically check the clock, not every single iteration
            for (int batch = 0; batch < 256; batch++) {
                iter++;
                double frac = elapsed() / TIME_LIMIT;
                if (frac > 1.0) frac = 1.0;
                double T = T0 * pow(0.001 / T0 > 0 ? 0.001 / T0 : 1e-6, frac);
                if (T < 1e-6) T = 1e-6;

                int kind;
                int nLev = (int)curLevees.size();
                if (nLev == 0) kind = 0;                 // must add
                else if (nLev >= B) kind = (frand() < 0.5) ? 1 : 2;  // remove or move
                else {
                    double x = frand();
                    kind = (x < 0.5) ? 0 : (x < 0.75 ? 1 : 2);
                }

                // build the trial blocked mask & levee list
                vector<char> tb = curBlocked;
                vector<int> tl = curLevees;
                if (kind == 0) {  // ADD
                    int u = pool[rng() % pool.size()];
                    if (tb[u]) continue;  // already a levee
                    tb[u] = 1;
                    tl.push_back(u);
                } else if (kind == 1) {  // REMOVE
                    int j = rng() % tl.size();
                    int u = tl[j];
                    tb[u] = 0;
                    tl[j] = tl.back();
                    tl.pop_back();
                } else {  // MOVE
                    int j = rng() % tl.size();
                    int u = tl[j];
                    int v = pool[rng() % pool.size()];
                    if (tb[v]) continue;  // target already a levee
                    tb[u] = 0;
                    tb[v] = 1;
                    tl[j] = v;
                }

                int trial = flood(tb, fl);
                int delta = trial - curFlood;  // fewer flooded is better
                if (delta <= 0 || frand() < exp(-delta / T)) {
                    curBlocked.swap(tb);
                    curLevees.swap(tl);
                    curFlood = trial;
                    if (curFlood < bestFlood) {
                        bestFlood = curFlood;
                        bestBlocked = curBlocked;
                        bestLevees = curLevees;
                    }
                }
            }
            if (elapsed() >= TIME_LIMIT) break;
        }
    }

    // ---- OUTPUT the best feasible levee set ----------------------------------
    // Defensive feasibility filter: drop any source/out-of-range/dup and cap at B.
    vector<int> outLev;
    {
        vector<char> used(N, 0);
        for (int u : bestLevees) {
            if (u < 0 || u >= N) continue;
            if (isSrc[u]) continue;
            if (used[u]) continue;
            used[u] = 1;
            outLev.push_back(u);
            if ((int)outLev.size() == B) break;
        }
    }

    string out;
    out += to_string((int)outLev.size());
    out += '\n';
    for (int u : outLev) {
        out += to_string(u / W);
        out += ' ';
        out += to_string(u % W);
        out += '\n';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
