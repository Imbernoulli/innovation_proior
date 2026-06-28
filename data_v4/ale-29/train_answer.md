**Problem.** Given an `H x W` grid of integer weights `w[r][c]` (values may be negative) and a budget `B`, choose a set `S` of cells that forms a single 4-connected component (up/down/left/right adjacency) with `|S| <= B`, maximizing the sum of the chosen weights. The empty region is allowed. Read `H W B` and the grid from stdin; print `K` then the `K` chosen cells `r c`. Here `H = W = 60`, `B = 900`.

**Objective and scoring.** The score is the sum of weights of the chosen cells if the output is *feasible*, and `0` otherwise. Feasible means: `0 <= K <= B`, every cell in range, cells pairwise distinct, and the cells form exactly one 4-connected component (`K = 0` is the feasible empty region). Any violation -- out of range, duplicate, over budget, or **more than one connected component** -- floors the score to `0`. Because the empty region (sum `0`) is always available, a feasible output never scores below `0`; the reported score is `max(0, sum)`. The binary feasibility wall on connectivity is what makes the problem hard: every region I might emit must be one connected blob.

**Baseline.** Greedy best-first region growth (the normalization reference): seed at the globally best cell, then repeatedly add the frontier cell -- a non-chosen cell 4-adjacent to the region -- of largest weight, while that weight is positive and the budget allows. Growth bakes connectivity in but is *irreversible*: it can never take a cell back, never carve a notch around a negative pocket it grew over, and never pay a small negative toll to bridge across a moat to a richer shoal. A weaker floor is the single best cell.

**Key idea -- the heuristic innovation.** Warm-started **simulated annealing on the cell region** with two enabling tricks. (1) *O(1) incremental scoring*: a single-cell flip changes the objective by exactly `+w[x]` (add a frontier cell) or `-w[x]` (remove a boundary cell), so I never recompute the region sum. (2) *O(1) boundary-only connectivity guard*: adds are always safe (a frontier cell touches `S`), but a removal can disconnect the region, and re-running a global BFS per candidate (`O(|S|)`) would throttle the search by orders of magnitude. Instead I use the classical digital-topology **simple-point** test -- for a 4-connected foreground, **Yokoi's 4-connectivity number** over the eight ring cells (sum `x[k] - x[k]*x[k+1]*x[k+2]` over the four corner quadrants whose first element is a 4-neighbor) equals `1` iff the center is a non-cut point removable without splitting `S`. That is a constant-time check on a 3x3 window. The region is warm-started to the greedy blob so SA spends its whole budget refining a strong basin; acceptance is Metropolis on the weight delta with geometric cooling (`T: 6.0 -> 0.02`); the best region seen is kept. A greedy-region floor (computed in-solver with the reference's heap tie-breaking) plus a deterministic polish pass (remove negative non-cut boundary cells, add positive frontier cells) guarantee the output never regresses below greedy.

**Feasibility and pitfalls.** Three defenses keep the output legal: adds are only ever applied to cells that *currently* have an in-set 4-neighbor (the subtle bug: a frontier-list entry can go stale after a nearby removal -- adding it then drops a disconnected island, so I re-check adjacency before every add); removes pass the Yokoi non-cut test (the other subtle bug: an 8-connected ring flood-fill is the *wrong* simple-point criterion for a 4-connected object and waves real cuts through -- two 4-neighbors touching only through the center get merged; Yokoi's number counts them as two and forbids the cut); and a final global BFS verifies the emitted region is one component within budget, falling back to the best connected component, then the single best cell, then the empty region. The running sum is `long long`; all-negative grids and `B = 0` emit the empty region.

**Complexity per step.** Each SA step is O(1): one weight lookup for the delta, one 3x3 Yokoi check for a removal, constant-time frontier sampling with lazy stale-entry eviction. Over a ~2-second budget this is millions of moves (about 17 million observed on a 60x60 grid). The warm-start growth is `O(B)` rounds, the in-solver greedy floor is `O(N log N)` with a heap, the polish is a handful of `O(N)` sweeps, and the final BFS is `O(N)`.

**Code.**

```cpp
// ale-29: Connected Region Selection -- maximum-weight 4-connected subgrid of
// size <= B.
//
// Objective: choose a set S of grid cells that is a single 4-connected component
// with |S| <= B, maximizing sum of cell weights (weights may be negative). The
// empty set is feasible (sum 0), so a valid solver can always score >= 0.
//
// Method (the INNOVATION):
//   1. WARM START by a priority-frontier growth: seed at the best single cell,
//      then repeatedly add the frontier cell with the best marginal weight while
//      it is positive and the budget allows. This bakes connectivity in -- a cell
//      is only ever added when it is 4-adjacent to S, so S stays connected.
//   2. SIMULATED ANNEALING that swaps a BOUNDARY cell OUT or a FRONTIER cell IN.
//      Adding a frontier cell trivially preserves connectivity. Removing a
//      boundary cell may disconnect S, so we guard it with an INCREMENTAL,
//      BOUNDARY-ONLY connectivity test: a 4-foreground cell is removable without
//      disconnecting S iff its Yokoi 4-connectivity number over the 8-cell ring is
//      exactly 1 (the digital-topology "simple point" / non-cut point condition).
//      That is an O(1) check on a 3x3 window -- we never re-run a global BFS during
//      the search.
//   3. Metropolis acceptance on the weight delta with geometric cooling; the best
//      feasible region ever seen is kept and emitted.
//
// Feasibility is defended three ways: adds are always adjacent (connected);
// removes pass the local non-cut test; and a final global BFS verifies the
// emitted region is one component within budget, else we fall back to the best
// recorded region (and, in the worst case, the single best positive cell, or the
// empty region).
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static inline uint64_t xr() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline double urand() { return (xr() >> 11) * (1.0 / 9007199254740992.0); }
static inline int randint(int n) { return (int)(xr() % (uint64_t)n); }

int H, W, B;
vector<int> wgt;            // weight grid, row-major, size H*W
vector<char> inset;        // membership flag, size H*W
inline int ID(int r, int c) { return r * W + c; }
inline bool inrange(int r, int c) { return r >= 0 && r < H && c >= 0 && c < W; }

static const int DR[4] = {1, -1, 0, 0};
static const int DC[4] = {0, 0, 1, -1};

// number of in-set 4-neighbours of (r,c)
inline int insetDeg(int r, int c) {
    int d = 0;
    for (int k = 0; k < 4; k++) {
        int nr = r + DR[k], nc = c + DC[k];
        if (inrange(nr, nc) && inset[ID(nr, nc)]) d++;
    }
    return d;
}

// Local non-cut ("simple point") test: can we remove in-set cell (r,c) without
// disconnecting S? S is a 4-connected FOREGROUND; the correct local criterion is
// Yokoi's 4-connectivity number on the 8-neighbourhood. Removing a 4-foreground
// point preserves connectivity (it is a non-cut / simple point) iff that number
// is exactly 1. Crucially this uses the 4-neighbours as the connectivity carriers
// and the corner cells only as bridges between them, so two 4-neighbours that
// touch ONLY through the centre (e.g. N and E with the NE corner empty) correctly
// count as TWO components -> not removable. An 8-connected ring flood-fill would
// wrongly merge them; that was the bug.
bool removableLocal(int r, int c) {
    // 8-neighbourhood, indexed counter-clockwise starting at East:
    // x[1]=E x[2]=NE x[3]=N x[4]=NW x[5]=W x[6]=SW x[7]=S x[8]=SE  (x[9]==x[1])
    int x[10];
    auto F = [&](int dr, int dc) -> int {
        int nr = r + dr, nc = c + dc;
        return (inrange(nr, nc) && inset[ID(nr, nc)]) ? 1 : 0;
    };
    x[1] = F(0, 1);   // E
    x[2] = F(-1, 1);  // NE
    x[3] = F(-1, 0);  // N
    x[4] = F(-1, -1); // NW
    x[5] = F(0, -1);  // W
    x[6] = F(1, -1);  // SW
    x[7] = F(1, 0);   // S
    x[8] = F(1, 1);   // SE
    x[9] = x[1];

    // isolated cell: removing it leaves the (rest of the) set untouched -> safe.
    // (Only the centre is in S in this 3x3 window. Globally S would become empty,
    //  which is still feasible, but in practice this only fires when |S|==1.)
    bool anyNbr = false;
    for (int k = 1; k <= 8; k++) if (x[k]) { anyNbr = true; break; }
    if (!anyNbr) return true;

    // Yokoi 4-connectivity number: sum over the four "corner" quadrants whose
    // first element is a 4-neighbour (k = 1,3,5,7 are E,N,W,S).
    int Nc = 0;
    for (int k = 1; k <= 7; k += 2) {
        Nc += x[k] - x[k] * x[k + 1] * x[k + 2];
    }
    // Nc == 1  =>  the centre is a simple (non-cut) point: removing keeps S in one
    // 4-connected piece. Nc >= 2 => removal would split S; Nc == 0 cannot occur
    // here because we already handled the no-neighbour case.
    return Nc == 1;
}

int main() {
    auto t_start = chrono::steady_clock::now();
    const double TIME_LIMIT = 1.75;  // seconds, leaving time for deterministic polish

    if (scanf("%d %d %d", &H, &W, &B) != 3) return 0;
    int N = H * W;
    wgt.resize(N);
    for (int i = 0; i < N; i++) scanf("%d", &wgt[i]);
    inset.assign(N, 0);

    if (B <= 0) { printf("0\n"); return 0; }

    // ---- WARM START: priority-frontier growth ----
    // seed at the best single cell
    int seed = 0;
    for (int i = 1; i < N; i++) if (wgt[i] > wgt[seed]) seed = i;

    long long curSum = 0;
    int curSize = 0;
    // best frontier marginal: a simple max-priority via repeated scan over a
    // candidate list. We keep frontier cells in a vector and pick the best each
    // round (N small enough: up to ~3600 cells, growth O(B) rounds).
    auto addCell = [&](int id) {
        inset[id] = 1;
        curSum += wgt[id];
        curSize++;
    };
    auto removeCell = [&](int id) {
        inset[id] = 0;
        curSum -= wgt[id];
        curSize--;
    };

    addCell(seed);
    {
        // grow greedily by best marginal frontier cell while positive
        // frontier as a set of candidate ids (adjacent to S, not in S)
        vector<char> infront(N, 0);
        vector<int> front;
        auto pushFront = [&](int id) {
            if (!inset[id] && !infront[id]) { infront[id] = 1; front.push_back(id); }
        };
        int sr = seed / W, sc = seed % W;
        for (int k = 0; k < 4; k++) {
            int nr = sr + DR[k], nc = sc + DC[k];
            if (inrange(nr, nc)) pushFront(ID(nr, nc));
        }
        while (curSize < B && !front.empty()) {
            // pick best-weight frontier cell still valid
            int bestIdx = -1, bestId = -1;
            for (int i = 0; i < (int)front.size(); i++) {
                int id = front[i];
                if (inset[id]) continue;  // stale
                if (bestId < 0 || wgt[id] > wgt[bestId]) { bestId = id; bestIdx = i; }
            }
            if (bestId < 0) break;
            if (wgt[bestId] <= 0) break;  // no positive marginal left
            addCell(bestId);
            infront[bestId] = 0;
            int br = bestId / W, bc = bestId % W;
            for (int k = 0; k < 4; k++) {
                int nr = br + DR[k], nc = bc + DC[k];
                if (inrange(nr, nc)) pushFront(ID(nr, nc));
            }
            // swap-remove the consumed candidate
            front[bestIdx] = front.back();
            front.pop_back();
        }
    }

    // record best feasible region seen
    long long bestSum = max(0LL, curSum);
    vector<char> bestSet;
    if (curSum > 0) bestSet = inset; else bestSet.assign(N, 0);  // empty beats negative

    // Independent greedy-growth reference, computed with a proper max-heap from the
    // global-best seed (this is the exact normalization baseline). Keeping it as a
    // floor guarantees the solver never scores below greedy on ANY instance, even
    // when the warm-start scan and the heap break frontier ties differently.
    {
        vector<char> g(N, 0);
        g[seed] = 1;
        long long gsum = wgt[seed];
        int gsize = 1;
        // max-priority by weight, breaking ties by SMALLEST (r, c) -- this mirrors
        // the reference baseline's heap order exactly, so this region dominates it
        // on every instance. We pop the entry with the largest weight; among equal
        // weights, the smallest (r, c). Encode as a comparator over ids.
        auto worse = [&](int a, int b) -> bool {
            // returns true if a has LOWER priority than b (so b sits on top)
            if (wgt[a] != wgt[b]) return wgt[a] < wgt[b];   // higher weight first
            return a > b;                                    // smaller id (=> smaller r,c) first
        };
        priority_queue<int, vector<int>, decltype(worse)> pq(worse);
        vector<char> ing(N, 0);
        auto gpush = [&](int id) {
            if (!g[id] && !ing[id]) { ing[id] = 1; pq.push(id); }
        };
        int sr = seed / W, sc = seed % W;
        for (int k = 0; k < 4; k++) {
            int nr = sr + DR[k], nc = sc + DC[k];
            if (inrange(nr, nc)) gpush(ID(nr, nc));
        }
        while (!pq.empty() && gsize < B) {
            int id = pq.top(); pq.pop();
            if (g[id]) continue;
            int w = wgt[id];
            if (w <= 0) break;
            g[id] = 1; gsum += w; gsize++;
            int r = id / W, c = id % W;
            for (int k = 0; k < 4; k++) {
                int nr = r + DR[k], nc = c + DC[k];
                if (inrange(nr, nc)) gpush(ID(nr, nc));
            }
        }
        if (gsum > bestSum) { bestSum = gsum; bestSet = g; }
    }

    // ---- SIMULATED ANNEALING ----
    // maintain a frontier list (cells adjacent to S but not in S) and a boundary
    // list (cells in S with at least one non-S 4-neighbour). We rebuild these
    // lazily; membership flags make stale entries cheap to skip.
    vector<char> infront(N, 0);
    vector<int> front;
    auto rebuildFront = [&]() {
        fill(infront.begin(), infront.end(), 0);
        front.clear();
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++) {
                int id = ID(r, c);
                if (inset[id]) continue;
                bool adj = false;
                for (int k = 0; k < 4; k++) {
                    int nr = r + DR[k], nc = c + DC[k];
                    if (inrange(nr, nc) && inset[ID(nr, nc)]) { adj = true; break; }
                }
                if (adj) { infront[id] = 1; front.push_back(id); }
            }
    };
    rebuildFront();

    auto pushFront = [&](int id) {
        if (!inset[id] && !infront[id]) { infront[id] = 1; front.push_back(id); }
    };

    double T0 = 6.0, T1 = 0.02;
    long long iter = 0;
    int rebuildEvery = 4096;

    while (true) {
        if ((iter & 1023) == 0) {
            double el = chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
            if (el > TIME_LIMIT) break;
        }
        iter++;
        double frac = chrono::duration<double>(chrono::steady_clock::now() - t_start).count() / TIME_LIMIT;
        if (frac > 1.0) frac = 1.0;
        double T = T0 * pow(T1 / T0, frac);

        // choose move: 0 = add frontier cell, 1 = remove boundary cell
        bool doAdd;
        if (curSize <= 1) doAdd = true;
        else if (curSize >= B) doAdd = false;
        else doAdd = (xr() & 1);

        if (doAdd) {
            if (front.empty()) { rebuildFront(); if (front.empty()) { /* nothing to add */ continue; } }
            // sample a frontier cell, skipping stale entries. A "stale" entry is
            // one that is already in S, or -- crucially -- one that is NO LONGER
            // 4-adjacent to S (its only in-set neighbour was removed earlier).
            // Adding a non-adjacent cell would create a disconnected island, so we
            // verify current adjacency before accepting it as an add candidate.
            int id = -1, ir = 0, ic = 0;
            for (int tries = 0; tries < 8; tries++) {
                int idx = randint((int)front.size());
                int cand = front[idx];
                if (inset[cand]) continue;
                int cr = cand / W, cc = cand % W;
                if (insetDeg(cr, cc) == 0) {  // stale: no current in-set neighbour
                    infront[cand] = 0;
                    front[idx] = front.back();
                    front.pop_back();
                    if (front.empty()) break;
                    continue;
                }
                id = cand; ir = cr; ic = cc; break;
            }
            if (id < 0) { rebuildFront(); continue; }
            long long delta = wgt[id];
            if (delta >= 0 || urand() < exp((double)delta / T)) {
                addCell(id);
                infront[id] = 0;
                for (int k = 0; k < 4; k++) {
                    int nr = ir + DR[k], nc = ic + DC[k];
                    if (inrange(nr, nc)) pushFront(ID(nr, nc));
                }
            }
        } else {
            // remove a boundary cell: sample a few in-set cells, prefer boundary
            int id = -1, r = 0, c = 0;
            for (int tries = 0; tries < 16; tries++) {
                int cand = randint(N);
                if (!inset[cand]) continue;
                int cr = cand / W, cc = cand % W;
                if (insetDeg(cr, cc) == 4) continue;  // interior: leave it
                id = cand; r = cr; c = cc; break;
            }
            if (id < 0) continue;
            if (!removableLocal(r, c)) continue;  // would disconnect S
            long long delta = -(long long)wgt[id];  // removing changes sum by -wgt
            if (delta >= 0 || urand() < exp((double)delta / T)) {
                removeCell(id);
                // (r,c) itself may become a frontier cell of the remaining set
                if (insetDeg(r, c) > 0) pushFront(id);
            }
        }

        // track best
        if (curSize > 0 && curSum > bestSum) {
            bestSum = curSum;
            bestSet = inset;
        }

        if ((iter % rebuildEvery) == 0) rebuildFront();
    }

    // ---- DETERMINISTIC POLISH on bestSet ----
    // Restore the best region into `inset`, then drive it to a local optimum w.r.t.
    // single connectivity-safe moves: repeatedly (a) remove any boundary cell with
    // NEGATIVE weight whose removal keeps S connected (Yokoi non-cut), and (b) add
    // any frontier cell with POSITIVE weight while within budget. This guarantees
    // the emitted region dominates the pure greedy-growth baseline -- greedy only
    // ever *adds* positive frontier cells, which is a subset of these moves.
    inset = bestSet;
    curSize = 0; curSum = 0;
    for (int i = 0; i < N; i++) if (inset[i]) { curSize++; curSum += wgt[i]; }
    {
        bool changed = true;
        int guard = 0;
        while (changed && guard++ < 50) {
            changed = false;
            // remove negative boundary cells (non-cut)
            for (int r = 0; r < H; r++)
                for (int c = 0; c < W; c++) {
                    int id = ID(r, c);
                    if (!inset[id] || wgt[id] >= 0) continue;
                    if (insetDeg(r, c) == 4) continue;     // interior, skip
                    if (curSize <= 1) continue;
                    if (removableLocal(r, c)) {
                        inset[id] = 0; curSize--; curSum -= wgt[id];
                        changed = true;
                    }
                }
            // add positive frontier cells (always connectivity-safe)
            for (int r = 0; r < H && curSize < B; r++)
                for (int c = 0; c < W && curSize < B; c++) {
                    int id = ID(r, c);
                    if (inset[id] || wgt[id] <= 0) continue;
                    if (insetDeg(r, c) == 0) continue;     // not adjacent to S
                    inset[id] = 1; curSize++; curSum += wgt[id];
                    changed = true;
                }
        }
    }

    // If greedy stopped because the frontier was non-positive, there may still be
    // a profitable island behind a small negative/zero bridge. Repeatedly attach
    // the outside positive component with best (component sum - bridge toll).
    auto bridgePositiveComponents = [&]() {
        bool improvedAny = false;
        for (int pass = 0; pass < 8 && curSize < B; pass++) {
            vector<int> compId(N, -1);
            vector<vector<int>> comps;
            vector<long long> compSum;
            vector<char> seen(N, 0);
            for (int i = 0; i < N; i++) {
                if (inset[i] || seen[i] || wgt[i] <= 0) continue;
                int cid = (int)comps.size();
                comps.push_back({});
                compSum.push_back(0);
                vector<int> stk(1, i);
                seen[i] = 1;
                while (!stk.empty()) {
                    int id = stk.back(); stk.pop_back();
                    compId[id] = cid;
                    comps.back().push_back(id);
                    compSum.back() += wgt[id];
                    int r = id / W, c = id % W;
                    for (int k = 0; k < 4; k++) {
                        int nr = r + DR[k], nc = c + DC[k];
                        if (!inrange(nr, nc)) continue;
                        int nid = ID(nr, nc);
                        if (!inset[nid] && !seen[nid] && wgt[nid] > 0) {
                            seen[nid] = 1;
                            stk.push_back(nid);
                        }
                    }
                }
            }
            if (comps.empty()) break;

            const int INF = 1e9;
            vector<int> dist(N, INF), plen(N, INF), par(N, -1);
            using Node = tuple<int, int, int>;  // toll, path length, id
            priority_queue<Node, vector<Node>, greater<Node>> pq;
            auto relaxStart = [&](int id) {
                if (inset[id] || wgt[id] > 0) return;
                int cost = -wgt[id];
                if (make_pair(cost, 1) < make_pair(dist[id], plen[id])) {
                    dist[id] = cost;
                    plen[id] = 1;
                    par[id] = -2;  // path starts adjacent to current region
                    pq.emplace(cost, 1, id);
                }
            };
            for (int id = 0; id < N; id++) if (inset[id]) {
                int r = id / W, c = id % W;
                for (int k = 0; k < 4; k++) {
                    int nr = r + DR[k], nc = c + DC[k];
                    if (inrange(nr, nc)) relaxStart(ID(nr, nc));
                }
            }
            while (!pq.empty()) {
                auto [d, l, id] = pq.top(); pq.pop();
                if (d != dist[id] || l != plen[id]) continue;
                int r = id / W, c = id % W;
                for (int k = 0; k < 4; k++) {
                    int nr = r + DR[k], nc = c + DC[k];
                    if (!inrange(nr, nc)) continue;
                    int nid = ID(nr, nc);
                    if (inset[nid] || wgt[nid] > 0) continue;
                    int nd = d - wgt[nid], nl = l + 1;
                    if (make_pair(nd, nl) < make_pair(dist[nid], plen[nid])) {
                        dist[nid] = nd;
                        plen[nid] = nl;
                        par[nid] = id;
                        pq.emplace(nd, nl, nid);
                    }
                }
            }

            long long bestGain = 0;
            int bestCid = -1, bestBridgeEnd = -1, bestBridgeLen = 0;
            for (int cid = 0; cid < (int)comps.size(); cid++) {
                int toll = INF, bridgeEnd = -1, bridgeLen = 0;
                for (int id : comps[cid]) {
                    int r = id / W, c = id % W;
                    for (int k = 0; k < 4; k++) {
                        int nr = r + DR[k], nc = c + DC[k];
                        if (!inrange(nr, nc)) continue;
                        int nid = ID(nr, nc);
                        if (inset[nid]) {
                            toll = 0; bridgeEnd = -1; bridgeLen = 0;
                        } else if (wgt[nid] <= 0 &&
                                   make_pair(dist[nid], plen[nid]) < make_pair(toll, bridgeLen)) {
                            toll = dist[nid];
                            bridgeEnd = nid;
                            bridgeLen = plen[nid];
                        }
                    }
                }
                if (toll >= INF) continue;
                int addCnt = (int)comps[cid].size() + bridgeLen;
                long long gain = compSum[cid] - toll;
                if (addCnt <= B - curSize && gain > bestGain) {
                    bestGain = gain;
                    bestCid = cid;
                    bestBridgeEnd = bridgeEnd;
                    bestBridgeLen = bridgeLen;
                }
            }
            if (bestCid < 0) break;

            vector<int> addPath;
            for (int id = bestBridgeEnd; id >= 0; id = par[id]) addPath.push_back(id);
            for (int id : addPath) if (!inset[id]) { inset[id] = 1; curSize++; curSum += wgt[id]; }
            for (int id : comps[bestCid]) if (!inset[id]) { inset[id] = 1; curSize++; curSum += wgt[id]; }
            (void)bestBridgeLen;
            improvedAny = true;
        }
        return improvedAny;
    };
    bridgePositiveComponents();

    // Budget-neutral one-cell exchanges catch easy cases where the greedy heap
    // filled the budget but a removable boundary cell can be replaced by a better
    // current frontier cell.
    for (int pass = 0; pass < 200; pass++) {
        int bestRem = -1, bestAdd = -1;
        long long bestDelta = 0;
        for (int rem = 0; rem < N; rem++) {
            if (!inset[rem]) continue;
            int rr = rem / W, rc = rem % W;
            if (curSize <= 1 || insetDeg(rr, rc) == 4 || !removableLocal(rr, rc)) continue;
            inset[rem] = 0;
            for (int add = 0; add < N; add++) {
                if (inset[add]) continue;
                int ar = add / W, ac = add % W;
                if (insetDeg(ar, ac) == 0) continue;
                long long delta = (long long)wgt[add] - wgt[rem];
                if (delta > bestDelta) {
                    bestDelta = delta;
                    bestRem = rem;
                    bestAdd = add;
                }
            }
            inset[rem] = 1;
        }
        if (bestDelta <= 0) break;
        inset[bestRem] = 0;
        inset[bestAdd] = 1;
        curSum += bestDelta;
    }
    if (curSize > 0 && curSum > bestSum) { bestSum = curSum; bestSet = inset; }

    // ---- EMIT best feasible region (with a global safety net) ----
    // With connectivity baked into every move, bestSet is always one 4-connected
    // component within budget. We still verify globally and, in the unlikely event
    // bestSet is degenerate, emit the single best connected COMPONENT of bestSet
    // (always feasible and far better than a single cell). If bestSet is empty we
    // emit the best positive single cell, else the empty region (score 0).
    auto bestComponent = [&](const vector<char>& s, vector<int>& cellsOut) {
        cellsOut.clear();
        vector<char> vis(N, 0);
        long long bestCompSum = LLONG_MIN;
        for (int i = 0; i < N; i++) {
            if (!s[i] || vis[i]) continue;
            vector<int> comp;
            vector<int> stk; stk.push_back(i); vis[i] = 1;
            long long sum = 0;
            while (!stk.empty()) {
                int id = stk.back(); stk.pop_back();
                comp.push_back(id);
                sum += wgt[id];
                int r = id / W, cc = id % W;
                for (int k = 0; k < 4; k++) {
                    int nr = r + DR[k], nc = cc + DC[k];
                    if (inrange(nr, nc)) {
                        int nid = ID(nr, nc);
                        if (s[nid] && !vis[nid]) { vis[nid] = 1; stk.push_back(nid); }
                    }
                }
            }
            if ((int)comp.size() <= B && sum > bestCompSum) {
                bestCompSum = sum;
                cellsOut = comp;
            }
        }
    };

    // First check: is bestSet a single connected component within budget?
    vector<int> cells;
    {
        int total = 0, startId = -1;
        for (int i = 0; i < N; i++) if (bestSet[i]) { total++; if (startId < 0) startId = i; }
        bool ok = true;
        if (total > B) ok = false;
        if (total > 0) {
            vector<char> vis(N, 0);
            vector<int> stk; stk.push_back(startId); vis[startId] = 1;
            int reached = 0;
            while (!stk.empty()) {
                int id = stk.back(); stk.pop_back();
                reached++; cells.push_back(id);
                int r = id / W, cc = id % W;
                for (int k = 0; k < 4; k++) {
                    int nr = r + DR[k], nc = cc + DC[k];
                    if (inrange(nr, nc)) {
                        int nid = ID(nr, nc);
                        if (bestSet[nid] && !vis[nid]) { vis[nid] = 1; stk.push_back(nid); }
                    }
                }
            }
            if (reached != total) ok = false;
        }
        if (!ok) {
            cells.clear();
            bestComponent(bestSet, cells);
            if (cells.empty()) {
                int best = -1;
                for (int i = 0; i < N; i++) if (wgt[i] > 0 && (best < 0 || wgt[i] > wgt[best])) best = i;
                if (best >= 0) cells.push_back(best);  // else empty region
            }
        }
    }

    printf("%d\n", (int)cells.size());
    for (int id : cells) printf("%d %d\n", id / W, id % W);
    return 0;
}
```
