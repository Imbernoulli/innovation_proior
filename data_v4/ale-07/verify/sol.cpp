#include <bits/stdc++.h>
using namespace std;

/*
  Maze Treasure Collection (time-budgeted) -- a grid ORIENTEERING problem.

  We must output a move string in {U,D,L,R} of length <= T, starting from S,
  never stepping into a wall or off-grid, that maximizes the total value of the
  DISTINCT treasure cells visited.

  Key idea (the innovation): collapse the grid into the complete graph of
  "points of interest" (POIs = the start cell + every treasure cell). One BFS
  from each POI gives exact grid shortest-path distances between every pair of
  POIs, turning the maze into a node-weighted ORIENTEERING instance: pick an
  ordered subset of treasures, starting at S, whose total travel distance
  (sum of consecutive grid distances) is <= T, maximizing collected value.

  We solve that orienteering by:
    (1) a greedy nearest-treasure SEED route (so we never do worse than the
        deterministic baseline the scorer normalizes against);
    (2) a BEAM SEARCH over partial routes ranked by an admissible upper bound
        (current value + value still reachable within the leftover budget),
        which finds good high-value backbones the greedy seed misses; and
    (3) an ITERATED LOCAL SEARCH on the best route so far -- 2-opt to shorten
        it (freeing budget), cheapest-insertion to spend freed budget on more
        treasures, low-density removal, and a double-bridge kick to escape
        local optima -- all under a wall-clock budget.
  Finally we stitch the chosen POI order back into a concrete U/D/L/R move
  string by replaying the stored BFS parent pointers.

  We always keep a valid, feasible move string on hand (the empty string is
  feasible), so the program can never emit an invalid solution.
*/

static int H, W, T;
static vector<string> grid;          // '#','.','S'
static int sr, sc;                   // start cell

static inline bool wall(int r, int c) {
    if (r < 0 || r >= H || c < 0 || c >= W) return true;
    return grid[r][c] == '#';
}
static inline int cellId(int r, int c) { return r * W + c; }

static const int DR[4] = {-1, 1, 0, 0};
static const int DC[4] = {0, 0, -1, 1};
static const char DCH[4] = {'U', 'D', 'L', 'R'};

// ---- POIs: index 0 = start, 1..K = treasures ----
static int P;                        // number of POIs
static vector<int> pr, pc, pval;     // POI row, col, value (start has value 0)
static vector<int> poiAt;            // cell -> POI index, or -1

// dist[i] = BFS distance grid (size H*W) from POI i; parentDir[i] = move used
// to ARRIVE at each cell on a shortest path from POI i (for reconstruction).
static vector<vector<int>> distP;
static vector<vector<int8_t>> parentDir;

static void bfsFrom(int i, vector<int>& d, vector<int8_t>& par) {
    d.assign(H * W, -1);
    par.assign(H * W, -1);
    int start = cellId(pr[i], pc[i]);
    d[start] = 0;
    deque<int> q;
    q.push_back(start);
    while (!q.empty()) {
        int cur = q.front(); q.pop_front();
        int r = cur / W, c = cur % W;
        int dc = d[cur];
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (wall(nr, nc)) continue;
            int nid = cellId(nr, nc);
            if (d[nid] != -1) continue;
            d[nid] = dc + 1;
            par[nid] = (int8_t)k;       // arrived at (nr,nc) by moving k from (r,c)
            q.push_back(nid);
        }
    }
}

// Reconstruct the move-string from POI a's grid to the cell of POI b, using
// distP[a]/parentDir[a]. Appends to `out`. Assumes b reachable from a.
static void stitch(int a, int b, string& out) {
    int tr = pr[b], tc = pc[b];
    // walk backwards from (tr,tc) to the source collecting move chars
    string seg;
    int r = tr, c = tc;
    int src = cellId(pr[a], pc[a]);
    while (cellId(r, c) != src) {
        int8_t k = parentDir[a][cellId(r, c)];
        if (k < 0) return;              // unreachable guard (shouldn't happen)
        seg.push_back(DCH[k]);
        // step back to the predecessor cell
        r -= DR[k];
        c -= DC[k];
    }
    reverse(seg.begin(), seg.end());
    out += seg;
}

int main() {
    // ---- read instance ----
    {
        if (scanf("%d %d %d", &H, &W, &T) != 3) return 0;
    }
    grid.assign(H, string());
    {
        // consume the rest of the H W T line, then read H grid rows
        // read rows as tokens (each row has no spaces)
        for (int r = 0; r < H; r++) {
            char buf[200005];
            if (scanf("%s", buf) != 1) { grid[r] = string(W, '.'); continue; }
            string row(buf);
            if ((int)row.size() < W) row += string(W - row.size(), '.');
            grid[r] = row.substr(0, W);
        }
    }
    int K = 0;
    if (scanf("%d", &K) != 1) K = 0;
    vector<int> tr_(K), tc_(K), tv_(K);
    for (int i = 0; i < K; i++) {
        if (scanf("%d %d %d", &tr_[i], &tc_[i], &tv_[i]) != 3) { tr_[i] = tc_[i] = 0; tv_[i] = 0; }
    }

    // locate start
    sr = sc = 0;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (grid[r][c] == 'S') { sr = r; sc = c; }

    // ---- build POIs ----
    poiAt.assign(H * W, -1);
    pr.clear(); pc.clear(); pval.clear();
    pr.push_back(sr); pc.push_back(sc); pval.push_back(0);
    poiAt[cellId(sr, sc)] = 0;
    for (int i = 0; i < K; i++) {
        int id = cellId(tr_[i], tc_[i]);
        if (id < 0 || id >= H * W) continue;
        if (grid[tr_[i]][tc_[i]] == '#') continue;     // treasure on wall: ignore
        if (poiAt[id] != -1) {                          // duplicate cell: merge value
            pval[poiAt[id]] += tv_[i];
            continue;
        }
        poiAt[id] = (int)pr.size();
        pr.push_back(tr_[i]); pc.push_back(tc_[i]); pval.push_back(tv_[i]);
    }
    P = (int)pr.size();

    // ---- BFS from every POI ----
    distP.assign(P, {});
    parentDir.assign(P, {});
    for (int i = 0; i < P; i++) bfsFrom(i, distP[i], parentDir[i]);

    // POI-to-POI distance matrix D[i][j] (-1 if unreachable)
    vector<vector<int>> D(P, vector<int>(P, -1));
    for (int i = 0; i < P; i++)
        for (int j = 0; j < P; j++) {
            int dij = distP[i][cellId(pr[j], pc[j])];
            D[i][j] = dij;
        }

    // ---- timing ----
    auto t0 = chrono::steady_clock::now();
    auto elapsedMs = [&]() {
        return (long long)chrono::duration_cast<chrono::milliseconds>(
                   chrono::steady_clock::now() - t0).count();
    };
    const long long BUDGET_MS = 1850;
    auto timeLeft = [&]() { return BUDGET_MS - elapsedMs(); };

    // route always begins at POI 0 (start). cost = sum of consecutive D, must <=T.
    auto routeCost = [&](const vector<int>& r) -> long long {
        long long c = 0;
        for (size_t i = 0; i + 1 < r.size(); i++) {
            if (D[r[i]][r[i + 1]] < 0) return LLONG_MAX;
            c += D[r[i]][r[i + 1]];
        }
        return c;
    };
    auto routeVal = [&](const vector<int>& r) -> long long {
        long long v = 0;
        for (size_t i = 1; i < r.size(); i++) v += pval[r[i]];
        return v;
    };

    // ---------------------------------------------------------------------
    // (1) GREEDY NEAREST-TREASURE SEED -- mirrors the scorer's baseline, so the
    //     final answer can never be worse than the normalization reference.
    // ---------------------------------------------------------------------
    vector<int> greedyRoute = {0};
    {
        vector<char> used(P, 0);
        used[0] = 1;
        long long rem = T;
        int head = 0;
        while (true) {
            int best = -1, bestd = INT_MAX;
            for (int j = 1; j < P; j++) {
                if (used[j]) continue;
                int dd = D[head][j];
                if (dd < 0 || dd > rem) continue;
                if (dd < bestd ||
                    (dd == bestd && (best == -1 ||
                     make_pair(pr[j], pc[j]) < make_pair(pr[best], pc[best])))) {
                    bestd = dd;
                    best = j;
                }
            }
            if (best == -1) break;
            used[best] = 1;
            rem -= bestd;
            head = best;
            greedyRoute.push_back(best);
        }
    }

    vector<int> bestRoute = greedyRoute;
    long long bestVal = routeVal(bestRoute);

    // ---------------------------------------------------------------------
    // (2) BEAM SEARCH over partial routes.
    //     State = route (starting at 0), cost, value. Children APPEND one
    //     unvisited reachable treasure, branching on the top value/cost
    //     candidates. States ranked by an ADMISSIBLE upper bound:
    //         bound = val + sum of pval[j] for unvisited reachable-within-rem j.
    //     (A relaxation: it ignores that visiting several j costs more than each
    //     single hop, so it over-estimates -> safe to rank/prune by.)
    // ---------------------------------------------------------------------
    struct State { vector<int> route; long long cost, val, bound; };

    auto computeBound = [&](const vector<int>& route, long long cost, long long val) -> long long {
        int head = route.back();
        long long rem = (long long)T - cost;
        long long extra = 0;
        static vector<char> vis;
        vis.assign(P, 0);
        for (int x : route) vis[x] = 1;
        for (int j = 1; j < P; j++) {
            if (vis[j]) continue;
            if (D[head][j] >= 0 && D[head][j] <= rem) extra += pval[j];
        }
        return val + extra;
    };

    {
        const int BEAM = max(12, min(80, 6000 / max(1, P)));
        const int BRANCH = 8;
        vector<State> beam;
        State init; init.route = {0}; init.cost = 0; init.val = 0;
        init.bound = computeBound(init.route, 0, 0);
        beam.push_back(init);

        while (!beam.empty() && timeLeft() > 250) {
            vector<State> next;
            next.reserve(beam.size() * BRANCH);
            for (auto& s : beam) {
                if (s.val > bestVal) { bestVal = s.val; bestRoute = s.route; }
                int head = s.route.back();
                long long rem = (long long)T - s.cost;
                static vector<char> vis;
                vis.assign(P, 0);
                for (int x : s.route) vis[x] = 1;
                vector<pair<double,int>> cand;
                for (int j = 1; j < P; j++) {
                    if (vis[j]) continue;
                    int dd = D[head][j];
                    if (dd < 0 || dd > rem) continue;
                    cand.push_back({ (double)pval[j] / (double)(dd + 1), j });
                }
                if (cand.empty()) continue;
                int take = min((int)cand.size(), BRANCH);
                partial_sort(cand.begin(), cand.begin() + take, cand.end(),
                             [](const pair<double,int>& a, const pair<double,int>& b) {
                                 if (a.first != b.first) return a.first > b.first;
                                 return a.second < b.second;
                             });
                for (int t = 0; t < take; t++) {
                    int j = cand[t].second;
                    State ns;
                    ns.route = s.route; ns.route.push_back(j);
                    ns.cost = s.cost + D[head][j];
                    ns.val = s.val + pval[j];
                    ns.bound = computeBound(ns.route, ns.cost, ns.val);
                    next.push_back(move(ns));
                }
            }
            if (next.empty()) break;
            if ((int)next.size() > BEAM) {
                nth_element(next.begin(), next.begin() + BEAM, next.end(),
                            [](const State& a, const State& b) {
                                if (a.bound != b.bound) return a.bound > b.bound;
                                return a.val > b.val;
                            });
                next.resize(BEAM);
            }
            for (auto& s : next)
                if (s.val > bestVal) { bestVal = s.val; bestRoute = s.route; }
            beam.swap(next);
        }
    }

    // ---------------------------------------------------------------------
    // (3) ITERATED LOCAL SEARCH on bestRoute.
    //     Operators (all O(1)/O(route) deltas on the POI graph):
    //       * 2-opt: reverse an internal segment if it shortens the route.
    //       * insert: add the unvisited treasure with the best value at its
    //                 cheapest feasible position (cheapest-insertion).
    //       * remove: drop the lowest value-per-freed-distance node, then refill.
    //       * double-bridge kick to escape local optima (keep best seen).
    // ---------------------------------------------------------------------
    {
        std::mt19937 rng(0xC0FFEE ^ (unsigned)(P * 2654435761u));

        auto twoOpt = [&](vector<int>& r) {
            // standard 2-opt on the open path r[0..m-1] (r[0]=start fixed)
            int m = (int)r.size();
            bool improved = true;
            int guard = 0;
            while (improved && timeLeft() > 60 && guard < 40) {
                improved = false; guard++;
                for (int i = 1; i + 1 < m; i++) {
                    int a = r[i - 1], b = r[i];
                    if (D[a][b] < 0) continue;
                    for (int k = i + 1; k < m; k++) {
                        int c = r[k];
                        int dnext = (k + 1 < m) ? r[k + 1] : -1;
                        // reverse r[i..k]; edges (a,b) and (c,dnext) change.
                        long long oldE = D[a][b];
                        long long newE = (D[a][c] < 0) ? LLONG_MAX : D[a][c];
                        if (dnext != -1) {
                            if (D[r[k]][dnext] < 0 || D[r[i]][dnext] < 0) continue;
                            oldE += D[r[k]][dnext];
                            newE += D[r[i]][dnext];
                        }
                        if (newE < oldE) {
                            reverse(r.begin() + i, r.begin() + k + 1);
                            improved = true;
                            b = r[i];
                        }
                    }
                }
            }
        };

        auto fillInsert = [&](vector<int>& r) {
            // greedily insert highest-value treasure at cheapest feasible slot
            bool added = true;
            while (added && timeLeft() > 60) {
                added = false;
                long long cost = routeCost(r);
                if (cost == LLONG_MAX) break;
                vector<char> vis(P, 0);
                for (int x : r) vis[x] = 1;
                int bj = -1, bpos = -1, bgain = 0; long long bdelta = LLONG_MAX;
                for (int j = 1; j < P; j++) {
                    if (vis[j]) continue;
                    // best position for j
                    long long jdelta = LLONG_MAX; int jpos = -1;
                    for (int p = 0; p + 1 < (int)r.size(); p++) {
                        int a = r[p], b = r[p + 1];
                        if (D[a][j] < 0 || D[j][b] < 0) continue;
                        long long delta = (long long)D[a][j] + D[j][b] - D[a][b];
                        if (delta < jdelta) { jdelta = delta; jpos = p + 1; }
                    }
                    int last = r.back();
                    if (D[last][j] >= 0) {
                        long long delta = D[last][j];
                        if (delta < jdelta) { jdelta = delta; jpos = (int)r.size(); }
                    }
                    if (jpos == -1) continue;
                    if (cost + jdelta > T) continue;
                    if (pval[j] > bgain || (pval[j] == bgain && jdelta < bdelta)) {
                        bgain = pval[j]; bdelta = jdelta; bj = j; bpos = jpos;
                    }
                }
                if (bj != -1 && bgain > 0) {
                    r.insert(r.begin() + bpos, bj);
                    added = true;
                }
            }
        };

        auto doubleBridge = [&](vector<int>& r) {
            int m = (int)r.size();
            if (m < 5) return;
            // pick 3 cut points in [1, m) splitting r[1..] into A B C and rejoin A C B
            int p1 = 1 + rng() % (m - 1);
            int p2 = 1 + rng() % (m - 1);
            int p3 = 1 + rng() % (m - 1);
            int a = min({p1, p2, p3}), c = max({p1, p2, p3});
            int b = p1 + p2 + p3 - a - c;
            if (!(a < b && b < c)) return;
            vector<int> nr;
            nr.insert(nr.end(), r.begin(), r.begin() + a);
            nr.insert(nr.end(), r.begin() + b, r.begin() + c);
            nr.insert(nr.end(), r.begin() + a, r.begin() + b);
            nr.insert(nr.end(), r.begin() + c, r.end());
            r.swap(nr);
        };

        // First, repair-then-improve the current best.
        auto polish = [&](vector<int> r) -> vector<int> {
            twoOpt(r);
            // drop any node that overflows the budget (shouldn't happen, guard)
            while (routeCost(r) > T && r.size() > 1) r.pop_back();
            fillInsert(r);
            return r;
        };

        bestRoute = polish(bestRoute);
        bestVal = routeVal(bestRoute);
        if (routeCost(bestRoute) > T) { // safety fallback to greedy seed
            bestRoute = greedyRoute; bestVal = routeVal(bestRoute);
        }

        vector<int> cur = bestRoute;
        long long curVal = bestVal;
        while (timeLeft() > 80) {
            vector<int> trial = cur;
            // perturb: a double-bridge kick, then optionally drop a low-value node
            doubleBridge(trial);
            if (trial.size() > 3 && (rng() & 3) == 0) {
                // remove a random non-start node to free budget for fillInsert
                int idx = 1 + rng() % (trial.size() - 1);
                trial.erase(trial.begin() + idx);
            }
            trial = polish(trial);
            if (routeCost(trial) > T) continue;     // reject infeasible
            long long tv = routeVal(trial);
            if (tv >= curVal) {                      // accept equal/better
                cur = trial; curVal = tv;
                if (tv > bestVal) { bestVal = tv; bestRoute = trial; }
            } else if ((rng() & 7) == 0) {           // occasional sideways restart
                cur = bestRoute; curVal = bestVal;
            }
        }
    }

    // ---------------------------------------------------------------------
    // Materialize bestRoute into a concrete move string (and final feasibility
    // guard: never emit a walk longer than T).
    // ---------------------------------------------------------------------
    string bestMoves = "";
    {
        long long cost = routeCost(bestRoute);
        if (cost != LLONG_MAX && cost <= T) {
            string mv;
            for (size_t i = 0; i + 1 < bestRoute.size(); i++)
                stitch(bestRoute[i], bestRoute[i + 1], mv);
            if ((int)mv.size() <= T) bestMoves = mv;
        }
        if (bestMoves.empty() && bestRoute.size() > 1) {
            // fallback: rebuild from the greedy seed which is guaranteed feasible
            string mv;
            long long gc = routeCost(greedyRoute);
            if (gc <= T) {
                for (size_t i = 0; i + 1 < greedyRoute.size(); i++)
                    stitch(greedyRoute[i], greedyRoute[i + 1], mv);
                if ((int)mv.size() <= T) bestMoves = mv;
            }
        }
    }

    fputs(bestMoves.c_str(), stdout);
    fputc('\n', stdout);
    return 0;
}
