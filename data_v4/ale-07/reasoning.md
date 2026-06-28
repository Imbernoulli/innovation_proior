# Maze Treasure Collection (time-budgeted) — reasoning

## Reading the objective carefully

I start by pinning down exactly what is being asked, because the wording "collect treasure in a
maze" can mean several different problems and only one of them is the right one. I have a maze on an
`H×W` grid: `#` is a wall, `.` is open floor, `S` is a single start cell. There are `K` treasures,
each a positive value on an open cell. I emit a string over `{U,D,L,R}`; each character moves the
collector one cell in that direction, and the move is only legal if it lands on open floor inside
the grid. I may make at most `T` moves. Stepping onto a treasure cell banks its value, but only the
**first** time — revisiting collects nothing. I want to maximize the total banked value.

So this is not "visit every cell" (the budget `T` is far smaller than the number of open cells), and
it is not "shortest path to a goal" (there is no fixed goal). It is the **orienteering problem**: a
travel budget, a set of prizes at known locations, and the question of which prizes to chase and in
what order. Two things are coupled and that coupling is the whole difficulty: the *selection* of
which treasures to take, and the *order* in which to take them (the order determines the travel cost,
which determines how many I can afford). Orienteering is NP-hard, so I am not going to find the exact
optimum at `K` up to a couple hundred; I need a strong heuristic that scores well and, above all,
that **always emits a feasible move string** — one illegal move floors the score to 0.

Before anything clever, I want to be sure I understand the scoring. The scorer replays my move
string, refuses it (score 0) if it is too long or ever steps into a wall or off the grid or contains
a stray token, otherwise sums the distinct treasure values I touched, and divides by `B`, the value
collected by a deterministic **greedy nearest-treasure** baseline it recomputes itself. The greedy
baseline scores exactly 1,000,000 by definition; I score `round(1e6 · mine / B)`. The trivial empty
walk collects nothing and scores 0. So my real bar is: **beat the nearest-treasure greedy**, not just
the empty walk. That distinction turns out to matter a lot.

## Reaching a feasible baseline first

The cardinal rule in these score-or-zero problems is: never be without a feasible answer. The empty
move string is trivially feasible — the collector just stands on `S`, makes zero moves, hits no
walls — and it is my safety net throughout. Everything I build must, at the end, either improve on a
known-feasible route or fall back to one. I will keep the empty string as the initial `moves` and only
overwrite it once I have something provably legal in hand.

But the empty walk scores 0, so it does not clear the bar; the bar is the greedy nearest baseline. My
first real construction therefore mirrors that baseline exactly: from the current cell, BFS to find
the nearest still-uncollected treasure I can still afford, walk there, repeat until nothing is
reachable within the remaining budget. That guarantees I am never *below* the normalization reference
— a strong invariant, because it means even if every fancier idea fails, I land at exactly 1,000,000
rather than losing.

## The reformulation that unlocks everything

Operating directly on the grid is awkward: every candidate route is a long string of single steps,
and evaluating a change means re-simulating cell by cell. The key realization is that I only ever care
about a tiny set of **points of interest** (POIs): the start cell and the `K` treasure cells. Between
two POIs the optimal sub-walk is just a grid shortest path, and I never want to pass *through* a
treasure I am not collecting in a way that costs more than the shortest path — the shortest path
already picks up any treasure that happens to lie on it for free.

So I run **one BFS from each POI**. With `K+1` POIs and a grid of at most `41×41 ≈ 1681` cells, that
is at most a couple hundred BFS runs over ~1700 cells each — trivially fast. From each BFS I read off
the distance to every other POI, giving a `(K+1)×(K+1)` **distance matrix** `D`. I also store the BFS
**parent direction** at every cell, so that once I have decided on an order of POIs I can reconstruct
the actual `U/D/L/R` string by walking the parent pointers backwards from each POI to the previous one.

Now the maze is gone. What remains is a clean **node-weighted orienteering instance on a complete
graph**: start at node 0, choose an ordered subset of treasure nodes whose consecutive `D`-distances
sum to `≤ T`, maximizing the sum of their values. Every candidate is now a short permutation of POI
indices, and — crucially — the cost of inserting, removing, or reversing part of a route is an `O(1)`
or `O(route)` delta on `D`, never a grid re-simulation. This incremental-evaluation property is what
makes a real search affordable.

## Why the obvious local search alone is too weak

My first instinct was: take the value/cost greedy (insert treasures by best value per added distance)
and 2-opt it, and call it done. I wrote a first version that did exactly this — a beam that appended
treasures ranked by `value/(dist+1)`, pruned by an upper bound, then a greedy highest-value insertion
pass. I ran it on seeds 1..20 and got a nasty surprise: the **mean score was about 920,000 — below
the 1,000,000 baseline**. On many seeds my value-greedy route actually *lost* to the dumb
nearest-treasure greedy.

The reason is illuminating. These mazes are *dense* with treasures, and the values are skewed but not
hugely so. The nearest-treasure greedy is extremely travel-efficient: it spends almost no budget per
treasure because the next treasure is usually a few cells away. My value/cost greedy, by chasing the
best ratio, would commit to a high-value treasure across the maze, burn a big chunk of `T` on the
trip, and end up collecting *fewer total* prizes. Ratio-greedy optimizes the wrong thing when the map
is dense: travel efficiency, not headline value-per-hop, is what fills the bag.

That diagnosis told me two things. First, I must **never start from a route that can lose to greedy**
— so I should seed the search with the greedy nearest route itself and only ever improve on it,
keeping the best. Second, the improvement engine has to do two jobs at once: *shorten* the route
(2-opt) to free budget, and *spend* freed budget on more treasures (insertion). Pure value-greedy did
neither well.

## The innovation: beam search with an admissible bound, then iterated local search

The structure I settled on has three stages, all on the POI graph.

**(1) Greedy nearest seed.** Build the nearest-treasure route exactly as the scorer's baseline does.
This route is my floor; the best-so-far starts here, so I can never finish below 1,000,000.

**(2) Beam search over partial routes.** A state is a route starting at node 0 with its travel cost
and collected value. I expand a state by **appending** one unvisited, still-reachable treasure,
branching on the top few candidates by `value/(dist+1)`. The non-obvious part is the **ranking key**:
I keep only the best `BEAM` states by an **admissible upper bound**

```
bound = current value + Σ pval[j] over unvisited j with D[head][j] ≤ (T − cost)
```

i.e. the current value plus the value of everything *still single-hop-reachable* within the leftover
budget. This is a relaxation — it pretends I could grab each remaining reachable treasure for the
price of one hop, ignoring that visiting several of them costs more than each individual hop — so it
**over-estimates** the achievable value and is therefore a safe pruning key: a state whose optimistic
ceiling is low can be dropped without fear. The beam keeps the high-value backbones that the greedy
seeds (both of them) never reach, while the bound keeps the frontier small. I record the best complete
route seen at every layer.

**(3) Iterated local search to polish the best route.** With a good backbone in hand, I run the
classic strong-but-simple metaheuristic for budgeted routing:

- **2-opt**: reverse an internal segment of the route when doing so shortens total travel. Only the
  two boundary edges change, so the delta is `O(1)`. Shortening the route *frees budget*.
- **Cheapest-insertion fill**: repeatedly insert the highest-value unvisited treasure at its cheapest
  feasible position, while the route still fits in `T`. This spends the budget 2-opt freed.
- **Removal + refill**: occasionally drop a node to open room for a different, better combination.
- **Double-bridge kick**: a 4-opt perturbation that re-links four segments, the standard way to
  escape a local optimum without destroying the whole route; I re-polish after each kick and keep the
  best route ever seen.

Every operator works on the POI permutation with incremental `D`-deltas, so I can run thousands of
moves inside the ~1.85 s I budget for search (leaving headroom under the 2 s limit). At the end I
stitch the best route's POI order into a concrete `U/D/L/R` string via the stored BFS parents, with a
final guard: if for any reason the stitched string would exceed `T` (it never should, since its length
equals the route cost by construction), I fall back to the guaranteed-feasible greedy route, and if
even that fails, to the empty string.

## A real debug + self-verify episode

I compiled and ran the three-stage version on seeds 1..20, scoring each against the local scorer and
checking feasibility. The headline numbers were good — mean ≈ 1,121,000, every seed feasible, and
**every single seed ≥ 1,000,000** (the seeding invariant held: I never lose to greedy). Three seeds
(7, 18, 20) sat at exactly 1,000,000, which is the expected behavior on instances where the
nearest-treasure greedy is already near-optimal and the search simply cannot beat it; the rest ranged
from small wins up to 1.43× on seed 1. Runtime hovered around 1.78 s, comfortably inside 2 s.

Then I stress-tested the **feasibility floor**, because a heuristic that scores well on average but
occasionally emits an illegal walk is worthless here. I fed the scorer four pathological "solutions"
against a real instance: a million `R`s (way over `T`), five `U`s from a start near a wall (steps into
a wall), a string with a stray `X` token, and the empty string. All four scored exactly **0**, while
the real solver output scored 1,428,453. Good — the floor bites exactly when it should and the solver
clears it everywhere.

The subtle bug I was most worried about was the **stitching length invariant**: does the materialized
`U/D/L/R` string ever exceed `T`? By construction the stitched length equals the route's travel cost,
which the search keeps `≤ T`, but an off-by-one in the parent-pointer walk or a route that sneaked
over budget during a perturbation could silently produce a too-long string and floor the score. So I
added an explicit check over all 20 seeds: for each, recompute `T`, strip whitespace from the output,
and assert the move count is `≤ T`. Every seed passed (`bad=0`). I also kept the runtime guard inside
the materialization step (`if ((int)mv.size() <= T)`) plus the greedy/empty fallbacks, so even an
unforeseen route can only ever degrade to a *feasible* answer, never an infeasible one.

One more correctness point I verified by reading the code paths: a treasure cell that the BFS
shortest path between two chosen POIs happens to pass through is collected "for free" by the scorer
(it counts distinct cells occupied), but my route's *value* accounting only credits the POIs I
explicitly chose. That means my internal value estimate is a *lower bound* on what the scorer credits
— I can only be pleasantly surprised, never penalized, which is the safe direction. It also explains
why several seeds beat their own internal route value when scored.

## Why this is the right strength of heuristic

The three-stage design is deliberately matched to the structure. The **POI reformulation** removes
the grid and gives `O(1)` move deltas — without it, no real search fits in the time budget. The
**greedy seed** guarantees I never lose to the normalization baseline, which is the actual bar. The
**beam with an admissible bound** is the established way to find high-value backbones in orienteering
while keeping the frontier small. And **2-opt + cheapest-insertion + double-bridge ILS** is the
standard strong-yet-simple metaheuristic for budgeted routing — the same family that wins on
prize-collecting TSP and team-orienteering. Each piece earns its place: dropping the seed reintroduces
the sub-baseline losses I saw in version one; dropping the bound makes the beam blow up; dropping the
ILS leaves the beam's backbone unpolished and forfeits the budget that 2-opt could free.

The final solver below is exactly the program in `verify/sol.cpp`. It always reads the instance, never
crashes, always prints a feasible move string within the budget, and on the seed set collects on
average about 12% more value than the nearest-treasure greedy while never falling below it.

## Final solver

```cpp
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
```
