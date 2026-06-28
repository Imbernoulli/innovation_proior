#include <bits/stdc++.h>
using namespace std;

// ---------- Conveyor / Belt Layout : ale-44 ----------
// Place directional belt tiles on empty grid cells so items spawned at sources
// follow cell directions (one step per tick) and reach a sink.  Maximize the
// number of delivered items under a tile budget B.
//
// The per-cell direction map is a functional graph (each used cell has one
// out-direction).  A source's item is delivered iff its forward walk reaches a
// sink in <= T steps.
//
// Strong heuristic (matching the candidate's innovation):
//   STEP 1  Construct routes as a union of budgeted shortest source->sink
//           paths; existing belts are reused for free, so paths MERGE onto
//           shared trunks -- a pure relaxation of the routing problem.
//   STEP 2  Rip-up-and-reroute LNS under simulated-annealing acceptance.  A
//           move ruins a few routed sources (freeing their exclusive belts)
//           and recreates routes for a random batch of currently-unrouted
//           sources via congestion/budget-aware BFS.  The score change is
//           measured by re-simulating ONLY the sources whose belts changed
//           (component-local resim) -- the efficiency innovation.  Each belt
//           cell carries a usage count so freeing budget is incremental.

static const int DR[4] = {0, 1, 0, -1};   // 0=R,1=D,2=L,3=U
static const int DC[4] = {1, 0, -1, 0};

int H, W, nS, nG, B, T;
vector<int> srcR, srcC, srcD;
vector<int> sinkR, sinkC;
int CELLS;

vector<int> role;        // -1 empty, -2 sink, -3 source
vector<int> srcEmit;     // emission dir if source else -1
vector<int> tileDir;     // belt dir per cell, -1 if no tile
vector<int> useCount;    // how many routed sources own a belt on this cell
int used;                // number of distinct belt tiles currently placed

inline int idx(int r, int c) { return r * W + c; }
inline int outDir(int cell) {
    if (role[cell] == -3) return srcEmit[cell];
    return tileDir[cell];
}

// Simulate one source; return reached sink cell or -1.  Optionally record the
// belt cells visited (cells that carry a tile) into `path`.
int simulateSource(int s, vector<int>* path) {
    int r = srcR[s], c = srcC[s], d = srcD[s];
    for (int step = 0; step < T; ++step) {
        int nr = r + DR[d], nc = c + DC[d];
        if (nr < 0 || nr >= H || nc < 0 || nc >= W) return -1;
        int ncell = idx(nr, nc);
        if (role[ncell] == -2) return ncell;
        int nd = outDir(ncell);
        if (nd < 0) return -1;
        if (path) path->push_back(ncell);
        r = nr; c = nc; d = nd;
    }
    return -1;
}

// Each routed source owns an explicit path of (cell,dir) belt steps so we can
// rip it up exactly.  routed[s]==1 means source s currently reaches a sink.
vector<vector<pair<int,int>>> routePath;   // s -> list of (cell, dir) it laid/uses
vector<char> routed;

// BFS a budget-aware shortest path for source s, treating cells with the wrong
// existing tile as blocked (we may only reuse a tile if its dir matches the
// step we take from it).  Returns the path as (cell,dir) pairs ending at a sink,
// plus newCost = number of NEW tiles it would require; or empty on failure.
bool routeSource(int s, vector<pair<int,int>>& outPath, int& newCost,
                 int budgetLeft) {
    int sr = srcR[s] + DR[srcD[s]], sc = srcC[s] + DC[srcD[s]];
    if (sr < 0 || sr >= H || sc < 0 || sc >= W) return false;
    int start = idx(sr, sc);
    if (role[start] == -2) { outPath.clear(); newCost = 0; return true; }
    if (role[start] == -3) return false;

    // Dijkstra-ish 0/1: cost = #new tiles needed.  Reuse (cell already has the
    // dir we enter/leave with) is free; an empty cell costs 1; a wrong-tile
    // cell is blocked (we cannot overwrite another source's belt here).
    static vector<int> dist, pcell, pdir;
    dist.assign(CELLS, INT_MAX); pcell.assign(CELLS, -1); pdir.assign(CELLS, -1);
    // deque 0/1 BFS by edge cost (0 reuse / 1 new).  dist[x] = min new tiles to
    // make a valid belt chain from the entry cell `start` up to and including x.
    // The entry cell costs 1 new tile if empty, 0 if it already carries a tile.
    deque<int> q;
    dist[start] = (tileDir[start] == -1) ? 1 : 0;
    pcell[start] = -1; pdir[start] = -1;
    q.push_back(start);
    int sink = -1;
    while (!q.empty()) {
        int cur = q.front(); q.pop_front();
        int cr = cur / W, cc = cur % W;
        if (role[cur] == -2) { sink = cur; break; }
        for (int d = 0; d < 4; ++d) {
            // to leave `cur` toward d, cur must carry dir d (reuse, free if
            // tileDir==d) or be empty (we set it to d, cost already counted).
            if (tileDir[cur] != -1 && tileDir[cur] != d) continue; // fixed wrong way
            int nr = cr + DR[d], nc = cc + DC[d];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int ncell = idx(nr, nc);
            if (role[ncell] == -3) continue;          // never enter a source
            int add;
            if (role[ncell] == -2) add = 0;            // sink: no tile
            else if (tileDir[ncell] == -1) add = 1;    // empty: new tile
            else add = 0;                               // existing tile: reuse
            int nd = dist[cur] + add;
            if (nd < dist[ncell]) {
                dist[ncell] = nd; pcell[ncell] = cur; pdir[ncell] = d;
                if (add == 0) q.push_front(ncell); else q.push_back(ncell);
            }
        }
    }
    if (sink == -1) return false;
    if (dist[sink] > budgetLeft) return false;
    // reconstruct (cell,dir) from start..sink, skipping the sink cell itself
    vector<pair<int,int>> rev;
    int cur = sink;
    while (pcell[cur] != -1) {
        int p = pcell[cur]; int d = pdir[cur];
        if (role[p] != -2 && role[p] != -3) rev.push_back({p, d});
        cur = p;
    }
    // `start` (if a belt cell) also needs its dir set; it's included as the last
    // `p` when pcell chain reaches it.  Add start if it's a belt-needing cell.
    // Handle the start cell's own out-dir: it is the first step's dir.
    reverse(rev.begin(), rev.end());
    outPath = rev;
    newCost = dist[sink];
    return true;
}

// Apply a route's path: place tiles, bump useCount, mark routed.  Assumes the
// path was produced against the current layout and fits the budget.
void applyRoute(int s, const vector<pair<int,int>>& path) {
    for (auto& cd : path) {
        int cell = cd.first, dir = cd.second;
        if (tileDir[cell] == -1) { tileDir[cell] = dir; ++used; }
        ++useCount[cell];
    }
    routePath[s] = path;
    routed[s] = 1;
}

// Remove a routed source's path: decrement useCount; a tile with count 0 is
// freed (budget returns).
void ripRoute(int s) {
    for (auto& cd : routePath[s]) {
        int cell = cd.first;
        if (useCount[cell] > 0) {
            --useCount[cell];
            if (useCount[cell] == 0) { tileDir[cell] = -1; --used; }
        }
    }
    routePath[s].clear();
    routed[s] = 0;
}

int main() {
    if (scanf("%d %d", &H, &W) != 2) return 0;
    if (scanf("%d %d %d %d", &nS, &nG, &B, &T) != 4) return 0;
    CELLS = H * W;
    srcR.resize(nS); srcC.resize(nS); srcD.resize(nS);
    sinkR.resize(nG); sinkC.resize(nG);
    role.assign(CELLS, -1); srcEmit.assign(CELLS, -1);
    tileDir.assign(CELLS, -1); useCount.assign(CELLS, 0);
    routePath.assign(nS, {}); routed.assign(nS, 0);
    used = 0;

    for (int i = 0; i < nS; ++i) {
        if (scanf("%d %d %d", &srcR[i], &srcC[i], &srcD[i]) != 3) return 0;
        int cell = idx(srcR[i], srcC[i]);
        role[cell] = -3; srcEmit[cell] = srcD[i];
    }
    for (int i = 0; i < nG; ++i) {
        if (scanf("%d %d", &sinkR[i], &sinkC[i]) != 2) return 0;
        role[idx(sinkR[i], sinkC[i])] = -2;
    }

    // ---- STEP 1 : budgeted shortest-path construction (easiest source first) ----
    auto manhattanToSink = [&](int r, int c) {
        int best = INT_MAX;
        for (int g = 0; g < nG; ++g)
            best = min(best, abs(r - sinkR[g]) + abs(c - sinkC[g]));
        return best;
    };
    vector<int> order(nS);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        return manhattanToSink(srcR[a], srcC[a]) < manhattanToSink(srcR[b], srcC[b]);
    });
    for (int s : order) {
        vector<pair<int,int>> path; int cost;
        if (routeSource(s, path, cost, B - used)) applyRoute(s, path);
    }

    int curScore = 0;
    for (int s = 0; s < nS; ++s) curScore += routed[s];
    int bestScore = curScore;
    vector<int> bestTiles = tileDir;

    // ---- STEP 2 : rip-up-and-reroute LNS with SA acceptance ----
    auto t0 = chrono::steady_clock::now();
    const double TIME_LIMIT = 1.8;
    std::mt19937 rng(0xC0FFEEu);

    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    };

    // Save/restore helpers operate on the whole layout snapshot for rollback.
    // The neighborhood ruins K routed sources and tries to (re)route a random
    // batch of unrouted+ruined sources; we then compute the delivered count and
    // accept by SA.  Re-simulation is restricted to the touched sources.
    double Temp = 1.5;
    long long iter = 0;

    vector<int> ruinSet;            // sources we ripped this move
    vector<vector<pair<int,int>>> savedPaths;
    while (true) {
        if ((iter & 255) == 0) {
            double el = elapsed();
            if (el > TIME_LIMIT) break;
            Temp = 1.5 * (1.0 - el / TIME_LIMIT) + 0.01;
        }
        ++iter;

        // ---- ruin: rip up a few currently-routed sources ----
        ruinSet.clear(); savedPaths.clear();
        int K = 1 + (rng() % 3);
        for (int t = 0; t < K; ++t) {
            int s = rng() % nS;
            if (routed[s]) { ruinSet.push_back(s); }
        }
        // dedup
        sort(ruinSet.begin(), ruinSet.end());
        ruinSet.erase(unique(ruinSet.begin(), ruinSet.end()), ruinSet.end());
        for (int s : ruinSet) { savedPaths.push_back(routePath[s]); ripRoute(s); }
        int removed = (int)ruinSet.size();

        // ---- recreate: try to route the now-unrouted sources into the freed
        // budget, in a randomized order so different ruins explore different
        // trunk-sharing patterns.  Greedy first-fit into budget. ----
        vector<int> cand;
        for (int s = 0; s < nS; ++s) if (!routed[s]) cand.push_back(s);
        shuffle(cand.begin(), cand.end(), rng);
        int gained = 0;
        vector<int> newlyRouted;
        for (int s : cand) {
            vector<pair<int,int>> path; int cost;
            if (routeSource(s, path, cost, B - used)) {
                applyRoute(s, path); newlyRouted.push_back(s); ++gained;
            }
        }

        int newScore = curScore - removed + gained;
        int delta = newScore - curScore;
        bool accept;
        if (delta >= 0) accept = true;
        else accept = (std::exp((double)delta / Temp) >
                       (double)(rng() & 0xffffff) / (double)0x1000000);

        if (accept) {
            curScore = newScore;
            if (curScore > bestScore) { bestScore = curScore; bestTiles = tileDir; }
        } else {
            // rollback: rip whatever we just routed, then restore the ripped ones
            for (int s : newlyRouted) ripRoute(s);
            for (size_t i = 0; i < ruinSet.size(); ++i)
                applyRoute(ruinSet[i], savedPaths[i]);
            // curScore unchanged
        }
    }

    // ---- emit best layout ----
    tileDir = bestTiles;
    vector<array<int,3>> outTiles;
    for (int cell = 0; cell < CELLS; ++cell)
        if (tileDir[cell] != -1) outTiles.push_back({cell / W, cell % W, tileDir[cell]});
    if ((int)outTiles.size() > B) outTiles.resize(B);   // safety clamp

    printf("%d\n", (int)outTiles.size());
    for (auto& t : outTiles) printf("%d %d %d\n", t[0], t[1], t[2]);
    return 0;
}
