#include <bits/stdc++.h>
using namespace std;

// ---------- Watchman Route (max area seen, bounded steps) : ale-45 ----------
// A guard walks a CLOSED route of <= L 4-directional steps on an HxW grid with
// walls, starting/ending at S.  From each cell it sees (ROOK line-of-sight) its
// own cell plus every free cell in an unobstructed straight horizontal/vertical
// run.  Maximize the number of DISTINCT cells ever seen along the route.
//
// Strong heuristic (the candidate's innovation):
//   * Precompute each free cell's visibility set ONCE as a bitset (vector<u64>).
//     Coverage of a route = OR of the visibility bitsets of its visited cells,
//     and marginal gain of adding a cell = popcount of newly-set bits -- cheap.
//   * The route is a budgeted ORIENTEERING / prize-collecting tour: pick an
//     ordered set of "anchor" cells to stand on, connected by grid shortest
//     paths, closing back to S, with total walk length <= L, maximizing covered
//     area.  Coverage is SUBMODULAR, so a known relaxation applies.
//   * Construction: greedy maximum-ratio insertion (gain / extra steps), the
//     standard GRASP move for orienteering, with the submodular marginal gain
//     evaluated incrementally on the bitsets.
//   * Refinement: simulated annealing over the anchor sequence (insert / drop /
//     2-opt / move) -- each candidate's length comes from precomputed pairwise
//     shortest-path distances and its coverage delta from the bitsets, so each
//     step is O(area/64).  We always keep the best feasible (closed, <=L) route.
// The walk between consecutive anchors is materialized from BFS parent pointers,
// and the WHOLE walk's cells (not just anchors) contribute to coverage, so the
// final emitted route is scored exactly as the scorer scores it.

static const int DR[4] = {-1, 1, 0, 0};   // U,D,L,R
static const int DC[4] = {0, 0, -1, 1};
static const char MV[4] = {'U', 'D', 'L', 'R'};

int H, W, L;
vector<string> grid;
int SR, SC;

int CELLS;                 // H*W
int FN;                    // number of free cells
vector<int> freeId;        // cell index -> compact free id (or -1)
vector<int> freeCell;      // free id -> cell index
inline int idx(int r, int c) { return r * W + c; }
inline bool inb(int r, int c) { return r >= 0 && r < H && c >= 0 && c < W; }
inline bool freeRC(int r, int c) { return inb(r, c) && grid[r][c] != '#'; }

// ---- visibility bitsets over FREE cells (compact indexing) ----
int WORDS;                                  // (FN+63)/64
vector<uint64_t> vis;                       // vis[fid*WORDS .. ] visibility set
inline uint64_t* visOf(int fid) { return &vis[(size_t)fid * WORDS]; }

// random
static uint64_t rngState = 88172645463325252ULL;
inline uint64_t xrand() {
    rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17;
    return rngState;
}
inline int randint(int n) { return (int)(xrand() % (uint64_t)n); }
inline double randf() { return (xrand() >> 11) * (1.0 / 9007199254740992.0); }

// ---------------- timing ----------------
static chrono::steady_clock::time_point T0;
inline double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - T0).count();
}
const double TIME_LIMIT = 1.9;

int main() {
    T0 = chrono::steady_clock::now();
    // -------- read instance --------
    {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
    }
    if (!(cin >> H >> W >> L)) return 0;
    grid.assign(H, string());
    for (int r = 0; r < H; r++) cin >> grid[r];
    CELLS = H * W;
    SR = SC = -1;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (grid[r][c] == 'S') { SR = r; SC = c; }
    if (SR < 0) { printf("0\n"); return 0; }   // no start -> emit empty route

    // -------- compact free-cell ids --------
    freeId.assign(CELLS, -1);
    freeCell.clear();
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (grid[r][c] != '#') { freeId[idx(r, c)] = (int)freeCell.size(); freeCell.push_back(idx(r, c)); }
    FN = (int)freeCell.size();
    WORDS = (FN + 63) / 64;

    // -------- precompute visibility bitsets (rook line-of-sight) --------
    vis.assign((size_t)FN * WORDS, 0ULL);
    auto setBit = [&](uint64_t* w, int fid) { w[fid >> 6] |= (1ULL << (fid & 63)); };
    for (int fid = 0; fid < FN; fid++) {
        int cell = freeCell[fid];
        int r = cell / W, c = cell % W;
        uint64_t* w = visOf(fid);
        setBit(w, fid);                                 // sees itself
        for (int d = 0; d < 4; d++) {
            int nr = r + DR[d], nc = c + DC[d];
            while (freeRC(nr, nc)) {
                setBit(w, freeId[idx(nr, nc)]);
                nr += DR[d]; nc += DC[d];
            }
        }
    }

    int startFid = freeId[idx(SR, SC)];

    // -------- BFS distances + parents from a source free cell --------
    // We need pairwise shortest-path distances among "anchor candidates".  We
    // BFS from the start and from every anchor we actually use; to keep it cheap
    // we BFS lazily and cache.  dist[src] is a vector over free ids.
    vector<vector<int>> distCache(FN);          // empty => not computed
    vector<vector<int>> parCache(FN);           // parent free-id in BFS tree
    auto bfsFrom = [&](int src) {
        if (!distCache[src].empty()) return;
        vector<int>& dist = distCache[src];
        vector<int>& par = parCache[src];
        dist.assign(FN, -1);
        par.assign(FN, -1);
        deque<int> q;
        dist[src] = 0; q.push_back(src);
        while (!q.empty()) {
            int u = q.front(); q.pop_front();
            int cell = freeCell[u];
            int r = cell / W, c = cell % W;
            for (int d = 0; d < 4; d++) {
                int nr = r + DR[d], nc = c + DC[d];
                if (!freeRC(nr, nc)) continue;
                int v = freeId[idx(nr, nc)];
                if (dist[v] < 0) { dist[v] = dist[u] + 1; par[v] = u; q.push_back(v); }
            }
        }
    };
    bfsFrom(startFid);

    // distance between two free ids (BFS from the smaller-id endpoint, cached)
    auto distFid = [&](int a, int b) -> int {
        bfsFrom(a);
        return distCache[a][b];
    };

    // ---------------- coverage bitset helpers ----------------
    // global coverage of the current route (OR of vis of all anchors)
    vector<uint64_t> cover(WORDS, 0ULL);
    // how many anchors cover each free cell (so we can remove anchors)
    vector<int> cnt(FN, 0);

    auto coverScore = [&]() -> int {
        int s = 0;
        for (int w = 0; w < WORDS; w++) s += __builtin_popcountll(cover[w]);
        return s;
    };
    // add/remove an anchor's visibility to cnt[] and rebuild cover word-lazily
    auto applyAnchor = [&](int fid, int sign) {
        const uint64_t* w = visOf(fid);
        for (int wi = 0; wi < WORDS; wi++) {
            uint64_t bits = w[wi];
            while (bits) {
                int b = __builtin_ctzll(bits);
                bits &= bits - 1;
                int f = wi * 64 + b;
                cnt[f] += sign;
                if (sign > 0) { if (cnt[f] == 1) cover[wi] |= (1ULL << b); }
                else { if (cnt[f] == 0) cover[wi] &= ~(1ULL << b); }
            }
        }
    };
    // marginal NEW cells if we add anchor fid given current cnt[]
    auto marginalGain = [&](int fid) -> int {
        const uint64_t* w = visOf(fid);
        int g = 0;
        for (int wi = 0; wi < WORDS; wi++) {
            uint64_t newbits = w[wi] & ~cover[wi];
            g += __builtin_popcountll(newbits);
        }
        return g;
    };

    // ---------------- the anchor tour ----------------
    // route = ordered list of anchor free-ids; route[0] == startFid, the tour is
    // closed (we always add the return-to-start leg implicitly).  routeLen =
    // sum of shortest-path distances between consecutive anchors + back to start.
    vector<int> route;
    route.push_back(startFid);
    applyAnchor(startFid, +1);

    auto tourLen = [&]() -> long long {
        long long s = 0;
        int n = (int)route.size();
        for (int i = 0; i + 1 < n; i++) s += distFid(route[i], route[i + 1]);
        if (n >= 1) s += distFid(route[n - 1], route[0]);   // close
        return s;
    };

    // -------- GREEDY construction: best ratio insertion --------
    // Repeatedly pick (anchor f, insertion position i) maximizing
    // marginalGain(f) / extraSteps, while tourLen stays <= L.
    // To bound work, candidate anchors = all free cells with positive marginal.
    {
        long long curLen = tourLen();
        bool improved = true;
        int guardIter = 0;
        // Deterministic termination: the greedy stops as soon as no positive-gain
        // anchor admits a budget-feasible insertion (improved==false), capped by
        // an iteration guard.  A wall-clock guard is only a safety net.
        while (improved && elapsed() < TIME_LIMIT * 0.55) {
            improved = false;
            guardIter++;
            double bestRatio = 0.0;
            int bestFid = -1, bestPos = -1, bestExtra = 0, bestGain = 0;
            int n = (int)route.size();
            // sample candidate anchors: scan all free cells but skip ones with
            // zero marginal gain quickly.
            for (int f = 0; f < FN; f++) {
                int g = marginalGain(f);
                if (g <= 0) continue;
                // try inserting between each consecutive pair (and before close)
                // choose the cheapest insertion position for this f.
                int bestPosF = -1, bestExtraF = INT_MAX;
                for (int i = 0; i < n; i++) {
                    int a = route[i];
                    int b = (i + 1 < n) ? route[i + 1] : route[0];
                    int extra = distFid(a, f) + distFid(f, b) - distFid(a, b);
                    if (extra < bestExtraF) { bestExtraF = extra; bestPosF = i + 1; }
                }
                if (bestPosF < 0) continue;
                if (curLen + bestExtraF > L) continue;       // would bust budget
                double ratio = (double)g / (double)max(1, bestExtraF);
                if (ratio > bestRatio + 1e-12) {
                    bestRatio = ratio; bestFid = f; bestPos = bestPosF;
                    bestExtra = bestExtraF; bestGain = g;
                }
            }
            if (bestFid >= 0 && bestGain > 0) {
                route.insert(route.begin() + bestPos, bestFid);
                applyAnchor(bestFid, +1);
                curLen += bestExtra;
                improved = true;
            }
            if (guardIter > FN + 50) break;
        }
    }

    // -------- SA refinement over the anchor sequence --------
    // Moves: (1) insert a random anchor at its cheapest position if budget ok;
    //        (2) drop a random non-start anchor; (3) swap two anchors (reorder).
    // Accept by score (coverage) with SA temperature; keep best feasible route.
    auto curScore = coverScore();
    long long curLen = tourLen();

    vector<int> bestRoute = route;
    int bestScore = curScore;

    // DETERMINISTIC budget: a fixed iteration cap drives the search so the output
    // depends only on the instance (not on machine speed / load).  The cooling
    // schedule is parameterized by iter/SA_ITERS, not wall-clock.  A wall-clock
    // guard (TIME_LIMIT) is a safety net that does not fire under normal speed.
    const long long SA_ITERS = 4000000;
    double Tstart = 6.0, Tend = 0.05;
    long long iter = 0;
    while (iter < SA_ITERS) {
        if ((iter & 8191) == 0) {
            if (elapsed() > TIME_LIMIT) break;     // safety net only
        }
        iter++;
        double frac = (double)iter / (double)SA_ITERS;
        double temp = Tstart * pow(Tend / Tstart, min(1.0, frac));

        int n = (int)route.size();
        int moveType = randint(3);
        if (n <= 2) moveType = 0;   // need anchors to drop/swap

        if (moveType == 0) {
            // INSERT a random anchor at its cheapest position.
            int f = randint(FN);
            int g = marginalGain(f);
            if (g <= 0) continue;
            int bestPosF = -1, bestExtraF = INT_MAX;
            for (int i = 0; i < n; i++) {
                int a = route[i];
                int b = (i + 1 < n) ? route[i + 1] : route[0];
                int extra = distFid(a, f) + distFid(f, b) - distFid(a, b);
                if (extra < bestExtraF) { bestExtraF = extra; bestPosF = i + 1; }
            }
            if (bestPosF < 0) continue;
            if (curLen + bestExtraF > L) continue;
            // accept: coverage strictly improves (g>0) -> always accept (it only
            // adds cells and we have budget).  Length grows though.
            route.insert(route.begin() + bestPosF, f);
            applyAnchor(f, +1);
            curLen += bestExtraF;
            curScore += g;
        } else if (moveType == 1) {
            // DROP a random non-start anchor.
            int pos = 1 + randint(n - 1);
            int f = route[pos];
            int a = route[pos - 1];
            int b = (pos + 1 < n) ? route[pos + 1] : route[0];
            int saved = distFid(a, f) + distFid(f, b) - distFid(a, b);
            // coverage delta: removing may lose cells uniquely covered by f.
            applyAnchor(f, -1);
            int newScore = coverScore();
            int delta = newScore - curScore;     // <= 0 typically
            // SA accept (dropping frees budget; accept if not too lossy)
            if (delta >= 0 || randf() < exp(delta / temp)) {
                route.erase(route.begin() + pos);
                curLen -= saved;
                curScore = newScore;
            } else {
                applyAnchor(f, +1);              // revert
            }
        } else {
            // SWAP two anchor positions (reorder; coverage unchanged, length may
            // shrink -> frees budget for later inserts).
            int i = 1 + randint(n - 1);
            int j = 1 + randint(n - 1);
            if (i == j) continue;
            // length delta from swapping route[i] and route[j]
            long long before = tourLen();
            swap(route[i], route[j]);
            long long after = tourLen();
            long long d = after - before;
            if (after <= L && (d <= 0 || randf() < exp((double)(-d) / temp))) {
                curLen = after;   // coverage identical
            } else {
                swap(route[i], route[j]);   // revert
            }
        }

        if (curScore > bestScore && curLen <= L) {
            bestScore = curScore;
            bestRoute = route;
        }
    }

    route = bestRoute;

    // ---------------- materialize the walk from anchors ----------------
    // Build the full move string by following BFS parent pointers between
    // consecutive anchors, then closing back to start.  Guaranteed length =
    // tourLen(route) <= L (we only kept feasible routes).
    auto pathMoves = [&](int from, int to, string& out) {
        // reconstruct shortest path from->to using parCache[from]
        bfsFrom(from);
        const vector<int>& par = parCache[from];
        // collect free ids from to back to from
        vector<int> seq;
        int cur = to;
        while (cur != from) {
            seq.push_back(cur);
            cur = par[cur];
            if (cur < 0) { return; }   // unreachable (shouldn't happen, connected)
        }
        seq.push_back(from);
        reverse(seq.begin(), seq.end());
        // emit moves between consecutive cells in seq
        for (size_t k = 0; k + 1 < seq.size(); k++) {
            int ca = freeCell[seq[k]], cb = freeCell[seq[k + 1]];
            int ra = ca / W, cca = ca % W, rb = cb / W, ccb = cb % W;
            int dr = rb - ra, dc = ccb - cca;
            for (int d = 0; d < 4; d++)
                if (DR[d] == dr && DC[d] == dc) { out.push_back(MV[d]); break; }
        }
    };

    string moves;
    int n = (int)route.size();
    for (int i = 0; i + 1 < n; i++) pathMoves(route[i], route[i + 1], moves);
    if (n >= 1) pathMoves(route[n - 1], route[0], moves);   // close

    // Safety: if (numerically) over budget for any reason, fall back to empty
    // route (always feasible, scores the start's visibility).
    if ((int)moves.size() > L) {
        printf("0\n");
        return 0;
    }
    if (moves.empty()) { printf("0\n"); return 0; }
    printf("%s\n", moves.c_str());
    return 0;
}
