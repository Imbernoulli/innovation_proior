// Number-Link / Flow-Free maximum-connections heuristic solver.
//
// Problem: a H x W grid carries K colored endpoint pairs.  We must route as many
// pairs as possible with non-crossing rectilinear unit-step paths: every cell on
// a path (endpoints included) is owned by exactly one pair, so two paths may
// never share a cell.  Objective = number of pairs connected.  An invalid /
// overlapping output floors the score to 0, so the program must ALWAYS emit a
// feasible (cell-disjoint, unit-step, correct-endpoint) set of paths.
//
// HEURISTIC CORE -- most-constrained-first ordering with A* on a congestion-
// priced grid and rip-up / reorder:
//   * Each cell tracks its OWNER pair (owner[cell]); conflict tests are O(1).
//   * Pairs are routed in a difficulty-aware order.  A per-pair difficulty score
//     blends the Manhattan lower bound with the forced-detour length actually
//     seen on the previous attempt (a pair that needed a long detour, or failed,
//     is more constrained).  Restarts sweep several orderings -- easiest-first
//     (the count-maximising workhorse), hardest/most-constrained-first, input
//     order, and noisy perturbations -- and keep the best.
//   * Routing is A* with the Manhattan heuristic over free cells (owner == -1)
//     plus the pair's own two endpoints; this finds a shortest cell-disjoint path
//     when one exists, leaving the grid as open as possible for later pairs.
//   * NEGOTIATED-CONGESTION RIP-UP: if a pair fails on free cells, we rip up the
//     pairs whose paths block it, re-route this pair, and re-queue the victims;
//     a per-cell congestion price (history cost) accumulates so chronically
//     contested cells become expensive, steering future routes around them
//     (the PathFinder idea).  We keep the best feasible assignment seen.
//   * Many random restarts with shuffled tie-breaking fill the time budget; the
//     best feasible routing found is printed.  Feasibility is re-verified before
//     printing, so the output is never overlapping or malformed.
//
// Single file, C++17, reads stdin, writes stdout.  g++ -O2.
#include <bits/stdc++.h>
using namespace std;

// ----------------------------- RNG ----------------------------------------
static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static inline uint64_t xr() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline int randint(int n) { return (int)(xr() % (uint64_t)n); }
static inline double urand() { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

// ----------------------------- timing -------------------------------------
static inline double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int H, W, K, N;             // N = H*W
vector<int> epA, epB;       // endpoint cell ids for each pair
vector<int> ra, ca, rb, cb; // endpoint coords

static inline int CID(int r, int c) { return r * W + c; }
static inline int CR(int id) { return id / W; }
static inline int CC(int id) { return id % W; }

// owner[cell] = pair id owning that cell, or -1 if free.
vector<int> owner;
// epOwner[cell] = pair id whose ENDPOINT this cell is, or -1.  Another pair's
// endpoint is a permanent obstacle: a path may only use its own two endpoints.
vector<int> epOwner;
// path[p] = list of cell ids forming pair p's path (empty if unrouted).
vector<vector<int>> path;

// Manhattan distance between two cell ids.
static inline int manh(int a, int b) {
    int dr = CR(a) - CR(b); if (dr < 0) dr = -dr;
    int dc = CC(a) - CC(b); if (dc < 0) dc = -dc;
    return dr + dc;
}

// --------------------------- A* router ------------------------------------
// Routes pair p from epA[p] to epB[p] over cells that are FREE (owner==-1) or
// owned by p itself (its own two endpoints).  Optionally, cells owned by pairs
// in `soft` (a boolean mask) are traversable at an extra `softCost` each (used
// by rip-up).  Returns true and fills `outPath` on success.
//
// Congestion price hist[cell] is added to the step cost so contested cells are
// avoided over restarts.  A* with the Manhattan heuristic stays admissible
// because the base step cost is >= 1 and the heuristic never overestimates the
// base distance; the soft/history surcharges only make found paths longer, never
// breaking correctness of the cell-disjointness we re-check at the end.

vector<int> gdist;       // best g found per cell (reused, stamped)
vector<int> gstamp;
int gstampCur = 0;
vector<int> gprev;       // predecessor cell for path reconstruction
vector<float> hist;      // congestion history price per cell

struct PQNode {
    int f;       // priority = g + h (+ tie noise)
    int g;
    int cell;
    bool operator<(const PQNode& o) const { return f > o.f; } // min-heap
};

bool route(int p, const vector<char>* soft, float softCost,
           vector<int>& outPath) {
    int s = epA[p], t = epB[p];
    ++gstampCur;
    priority_queue<PQNode> pq;
    gdist[s] = 0; gstamp[s] = gstampCur; gprev[s] = -1;
    pq.push({manh(s, t), 0, s});
    static const int DR[4] = {-1, 1, 0, 0};
    static const int DC[4] = {0, 0, -1, 1};
    bool found = false;
    while (!pq.empty()) {
        PQNode cur = pq.top(); pq.pop();
        int u = cur.cell;
        if (gstamp[u] == gstampCur && cur.g > gdist[u]) continue;
        if (u == t) { found = true; break; }
        int ur = CR(u), uc = CC(u);
        for (int d = 0; d < 4; ++d) {
            int nr = ur + DR[d], nc = uc + DC[d];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = nr * W + nc;
            // Another pair's endpoint cell is a hard obstacle (never traversable,
            // not even via rip-up): paths may only touch their own endpoints.
            if (epOwner[v] != -1 && epOwner[v] != p) continue;
            // Traversable if: it's the target t, it's free, it's owned by p
            // (its own endpoint), or it's a soft (rip-up) cell.
            int ow = owner[v];
            int extra = 0;
            if (v == t) {
                // target endpoint: always reachable (owned by p).
            } else if (ow == -1) {
                // free
            } else if (ow == p) {
                // own cell (e.g. its other endpoint, but that's t; safety)
            } else if (soft && (*soft)[ow]) {
                extra = (int)softCost; // crossing a rip-up candidate
            } else {
                continue; // blocked
            }
            int stepCost = 1 + extra + (int)hist[v];
            int ng = cur.g + stepCost;
            if (gstamp[v] != gstampCur || ng < gdist[v]) {
                gdist[v] = ng; gstamp[v] = gstampCur; gprev[v] = u;
                int noise = (int)(xr() & 1); // break ties randomly for restarts
                pq.push({ng + manh(v, t) + noise, ng, v});
            }
        }
    }
    if (!found) return false;
    // reconstruct
    outPath.clear();
    int cur = t;
    while (cur != -1) { outPath.push_back(cur); cur = gprev[cur]; }
    reverse(outPath.begin(), outPath.end());
    return true;
}

// claim / release a path's cells in owner[].
void claim(int p) { for (int c : path[p]) owner[c] = p; }
void release(int p) { for (int c : path[p]) owner[c] = -1; path[p].clear(); }

// ----------------------- one construction pass ----------------------------
// Routes pairs in the given order using free-cell A*, then a bounded rip-up
// phase for the still-unrouted pairs.  Returns the number of routed pairs and
// leaves owner[]/path[] holding that (feasible) assignment.
int constructPass(vector<int>& order, double deadline) {
    fill(owner.begin(), owner.end(), -1);
    for (auto& pp : path) pp.clear();

    // Phase 1: greedy free-cell A* routing in the given (difficulty-aware) order.
    vector<int> failed;
    vector<int> tmp;
    for (int p : order) {
        if (route(p, nullptr, 0, tmp)) {
            path[p] = tmp; claim(p);
        } else {
            failed.push_back(p);
        }
    }

    // Phase 2: TRANSACTIONAL negotiated-congestion rip-up for the failures.
    // For each failed pair, allow crossing other pairs' cells at a soft cost,
    // find the cheapest such path, rip up exactly the victims it crosses, claim
    // p, then re-route the victims onto free cells.  The whole move is committed
    // only if it does NOT lower the routed count (newRouted >= oldRouted); else
    // we ROLL BACK to the snapshot.  This makes phase 2 monotone: it can only
    // keep or improve the number of routed pairs, never regress below the greedy
    // phase-1 result.  Per-cell history price still accumulates on contested
    // cells, steering future restarts (the PathFinder negotiated-congestion
    // idea).  Bounded rounds keep it inside the time budget.
    vector<char> soft(K, 0);
    int round = 0, maxRounds = 8;
    while (!failed.empty() && round < maxRounds && now_sec() < deadline) {
        ++round;
        vector<int> nextFailed;
        // shuffle failed for diversity
        for (int i = (int)failed.size() - 1; i > 0; --i)
            swap(failed[i], failed[randint(i + 1)]);
        for (int p : failed) {
            if (now_sec() >= deadline) { nextFailed.push_back(p); continue; }
            // soft mask: only currently-routed pairs may be crossed.
            for (int q = 0; q < K; ++q) soft[q] = (!path[q].empty()) ? 1 : 0;
            soft[p] = 0;
            vector<int> rp;
            if (!route(p, &soft, 3.0f, rp)) { nextFailed.push_back(p); continue; }
            // Identify victims: owners of cells on rp that belong to other pairs.
            vector<int> victims;
            for (int c : rp) {
                int ow = owner[c];
                if (ow != -1 && ow != p) victims.push_back(ow);
            }
            sort(victims.begin(), victims.end());
            victims.erase(unique(victims.begin(), victims.end()), victims.end());

            // --- snapshot for rollback ---
            // Save p's old (empty) path and each victim's old path.
            vector<vector<int>> savedVictim(victims.size());
            for (size_t i = 0; i < victims.size(); ++i)
                savedVictim[i] = path[victims[i]];

            // Apply: rip up victims, claim p.
            for (int v : victims) release(v);
            bool ok = true;
            for (int c : rp) if (owner[c] != -1) { ok = false; break; }
            if (!ok) {
                // restore victims (rare) and skip.
                for (size_t i = 0; i < victims.size(); ++i) {
                    path[victims[i]] = savedVictim[i]; claim(victims[i]);
                }
                nextFailed.push_back(p);
                continue;
            }
            path[p] = rp; claim(p);
            // Re-route victims onto free cells.
            vector<int> rerouteFail;
            for (int v : victims) {
                if (route(v, nullptr, 0, tmp)) { path[v] = tmp; claim(v); }
                else rerouteFail.push_back(v);
            }
            // Net change in routed count: +1 (p) - (#victims that failed to reroute).
            int net = 1 - (int)rerouteFail.size();
            if (net >= 0) {
                // COMMIT.  Charge history on the contested cells so future
                // restarts negotiate around them.
                for (int c : rp) {
                    int ow0 = owner[c];
                    (void)ow0;
                    hist[c] += 0.4f;
                }
                for (int v : rerouteFail) nextFailed.push_back(v);
            } else {
                // ROLLBACK: undo p, undo victim reroutes, restore originals.
                release(p);
                for (int v : victims) if (!path[v].empty()) release(v);
                for (size_t i = 0; i < victims.size(); ++i) {
                    path[victims[i]] = savedVictim[i]; claim(victims[i]);
                }
                // Still charge a little history so we don't loop on the same wall.
                for (int c : rp) hist[c] += 0.2f;
                nextFailed.push_back(p);
            }
        }
        failed.swap(nextFailed);
    }

    int routed = 0;
    for (int p = 0; p < K; ++p) if (!path[p].empty()) ++routed;
    return routed;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> H >> W >> K)) return 0;
    N = H * W;
    epA.resize(K); epB.resize(K);
    ra.resize(K); ca.resize(K); rb.resize(K); cb.resize(K);
    for (int p = 0; p < K; ++p) {
        cin >> ra[p] >> ca[p] >> rb[p] >> cb[p];
        epA[p] = CID(ra[p], ca[p]);
        epB[p] = CID(rb[p], cb[p]);
    }
    owner.assign(N, -1);
    epOwner.assign(N, -1);
    for (int p = 0; p < K; ++p) { epOwner[epA[p]] = p; epOwner[epB[p]] = p; }
    path.assign(K, {});
    gdist.assign(N, 0);
    gstamp.assign(N, 0);
    gprev.assign(N, -1);
    hist.assign(N, 0.0f);

    // Difficulty score: longer Manhattan span = more constrained.  We refine it
    // across restarts using the detour length actually paid.
    vector<double> diff(K);
    for (int p = 0; p < K; ++p) diff[p] = manh(epA[p], epB[p]);

    // Best feasible assignment found so far.
    int bestRouted = -1;
    vector<vector<int>> bestPath;

    double t0 = now_sec();
    // Time budget: scale modestly with K but cap.  ~1.8s default.
    double budget = 1.8;
    double globalDeadline = t0 + budget;

    int restart = 0;
    while (now_sec() < globalDeadline) {
        ++restart;
        // Ordering strategy.  For MAXIMISING the count, easy (short) pairs are
        // cheap wins, while hard (long / constrained) pairs hog the grid -- so
        // the workhorse order is EASIEST-FIRST (ascending difficulty).  But a
        // constrained pair routed last can be impossible, so we also try
        // HARDEST-FIRST (so the most-constrained pair claims its scarce corridor
        // before the grid fills) and the raw input order, then randomised
        // perturbations of the current learned difficulty for diversity.
        //   mode 0: easiest-first (ascending diff)   <- baseline-dominating
        //   mode 1: hardest-first (descending diff)  <- most-constrained-first
        //   mode 2: input order
        //   mode>=3: easiest-first with noise, alternating asc/desc bias
        int mode = restart - 1;
        vector<pair<double,int>> keyed(K);
        for (int p = 0; p < K; ++p) {
            double key;
            if (mode == 0) key = diff[p];              // ascending
            else if (mode == 1) key = -diff[p];        // descending
            else if (mode == 2) key = p;               // input order
            else {
                double noise = (urand() - 0.5) * 6.0;
                double sign = ((mode & 1) ? -1.0 : 1.0);
                key = sign * (diff[p] + noise);
            }
            keyed[p] = {key, p};
        }
        sort(keyed.begin(), keyed.end());
        vector<int> order(K);
        for (int p = 0; p < K; ++p) order[p] = keyed[p].second;

        // Mild history decay between restarts so old contention fades.
        if (restart > 1)
            for (int c = 0; c < N; ++c) hist[c] *= 0.85f;

        int routed = constructPass(order, globalDeadline);

        // Update difficulty from results: unrouted pairs become harder, and
        // routed pairs absorb their realized detour (path length over the lower
        // bound) so chronically-long pairs are prioritised next restart.
        for (int p = 0; p < K; ++p) {
            int lb = manh(epA[p], epB[p]);
            if (path[p].empty()) {
                diff[p] = max(diff[p], (double)lb) + 4.0; // failed: harder
            } else {
                int detour = (int)path[p].size() - 1 - lb;
                diff[p] = 0.7 * diff[p] + 0.3 * (lb + 2.0 * detour);
            }
        }

        if (routed > bestRouted) {
            bestRouted = routed;
            bestPath = path;
        }
        // If everything routed, we are optimal in count -- stop.
        if (bestRouted == K) break;
    }

    // ----- Re-verify the best assignment for feasibility before printing. -----
    // (Defensive: guarantees we never emit overlapping / malformed output.)
    if (bestRouted < 0) { // no pass completed (shouldn't happen): print 0.
        cout << 0 << "\n";
        return 0;
    }
    vector<int> own2(N, -1);
    vector<int> good;      // ids of paths that pass full validation
    auto unitStep = [](int a, int b) {
        return manh(a, b) == 1;
    };
    for (int p = 0; p < K; ++p) {
        const vector<int>& pp = bestPath[p];
        if (pp.empty()) continue;
        // endpoints
        bool endsOK =
            ((pp.front() == epA[p] && pp.back() == epB[p]) ||
             (pp.front() == epB[p] && pp.back() == epA[p]));
        if (!endsOK || (int)pp.size() < 2) continue;
        bool ok = true;
        // unit steps + no self-revisit (check against a local set via own2 stamp)
        for (size_t i = 0; i < pp.size(); ++i) {
            int c = pp[i];
            if (c < 0 || c >= N) { ok = false; break; }
            if (i > 0 && !unitStep(pp[i - 1], c)) { ok = false; break; }
        }
        if (!ok) continue;
        // conflict with already-accepted paths?
        bool conflict = false;
        for (int c : pp) if (own2[c] != -1) { conflict = true; break; }
        if (conflict) continue;
        // self-revisit?
        for (int c : pp) {
            if (own2[c] == p) { conflict = true; break; } // shouldn't happen
        }
        if (conflict) continue;
        for (int c : pp) own2[c] = p;
        good.push_back(p);
    }

    // Output: count, then each accepted path.
    cout << good.size() << "\n";
    for (int p : good) {
        const vector<int>& pp = bestPath[p];
        cout << p << " " << pp.size();
        for (int c : pp) cout << " " << CR(c) << " " << CC(c);
        cout << "\n";
    }
    return 0;
}
