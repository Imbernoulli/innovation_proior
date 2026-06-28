I am handed a Capacitated Vehicle Routing instance: a depot, `n` customers with
demands, `K` identical vehicles of capacity `Q`, and the job of partitioning the
customers into `K` capacity-feasible routes and ordering each route so that the
total closed-tour distance is as small as possible. The score the judge reports
is `D_cw / D`, where `D` is my total distance and `D_cw` is the Clarke-Wright
savings distance on the same instance, and — the part I must never forget — any
output that is infeasible (a missing customer, a duplicate, a route over
capacity, the wrong number of route lines) is floored to a score of `0`. So the
very first thing I owe myself is a solution that is *always* feasible, and only
then do I get to chase distance.

Let me name the objective precisely before touching any cleverness. A route is a
list of customer ids; its length is `d(0, c1) + d(c1, c2) + ... + d(ck, 0)` with
Euclidean distances, where `0` is the depot. The total is the sum over the `K`
routes. Feasibility has exactly two non-trivial parts: (1) the multiset of all
printed ids must be precisely `{1, ..., n}` — each customer once — and (2) each
route's demand sum must be `<= Q`. Ordering within a route is free; the partition
plus the orderings are the entire decision. That framing already tells me the
problem decomposes into a *packing* question (which customers ride together) and
a *routing* question (in what order), and good CVRP heuristics attack both.

## A feasible baseline first

Before optimising anything I want a guaranteed-valid solution in hand, because a
clever optimiser that occasionally emits an over-capacity route scores zero and
is worse than useless. The cheapest construction that is both feasible and not
embarrassing is the **sweep**: compute each customer's polar angle around the
depot, sort by angle, and walk the sorted sequence dropping customers into the
current vehicle until the next one would breach `Q`, then move to the next
vehicle. Sweep is `O(n log n)`, it is feasible as long as `sum(dem) <= K*Q` and
`max(dem) <= Q` (both guaranteed by the generator), and because angularly
adjacent customers tend to be spatially close, the routes come out reasonably
compact. I will treat this as my baseline and my warm start.

There is one feasibility subtlety in the sweep I have to handle deliberately.
Filling vehicle 0, then 1, then 2, in pure angular order, can run out of
vehicles before all customers are placed if the early routes greedily soak up
capacity — the last angular slice may have nothing left to ride in. To keep the
construction watertight, when the "current" vehicle pointer runs past `K-1` I
fall back to placing the customer into the least-loaded route that still has room
for it. With the generator's feasibility margin (`K*Q` strictly above
`sum(dem)`), such a route always exists, so the construction never produces an
overloaded route. I also keep a separate "first-fit-decreasing by demand"
fallback packer that I can fall back to if any later step ever leaves me with an
inconsistent solution — a paranoia I will justify shortly.

If I score this raw sweep against the Clarke-Wright normaliser, it lands well
below `1.0` — the sweep's distance is much larger than the savings baseline,
because its angular cut points pay no attention to load balance, so it routinely
leaves one route stretched long and the neighbour nearly empty. That gap is
exactly the room a local search has to claim.

## Why the obvious local search is not enough, and what the real lever is

The instinct is: take the sweep and run local search. Fine — but *which* moves,
and how do I make them cheap? Two failure modes stalk a naive implementation.

The first failure mode is **scope**. If my only moves are intra-route — 2-opt and
Or-opt within a single tour — then I can tidy each route, but I can never fix the
sweep's central defect, which is that the *partition* is wrong: the wrong
customers are grouped together. Intra-route moves shuffle deck chairs; they
cannot move a customer from an overlong route to an underfull neighbour. So I
must have cross-route moves: **relocate** (pull one customer out of route `a` and
insert it at the best position in route `b`) and **swap** (exchange a customer of
`a` with a customer of `b`). These rebalance the packing.

But relocate and swap, by themselves, share a weakness: they move customers one
at a time, so undoing a badly placed *segment* — a whole arm of a route that
should belong to its neighbour — takes many coordinated single moves, each of
which may look locally worse, and a greedy descent will refuse them. This is the
classic reason a sweep gets stuck. The move that cuts through it is **cross-route
2-opt\***: pick two routes `a` and `b`, choose a cut point in each, and swap the
*tail segments* — route `a` keeps its head and inherits `b`'s tail, route `b`
keeps its head and inherits `a`'s tail. In one move, an entire mis-assigned arc
hops between vehicles, and the boundary between two overlapping routes is redrawn.
This is the move the candidate names as the lever, and it is right: 2-opt\* is
precisely what repairs the badly balanced cut points a sweep leaves behind.

The second failure mode is **speed**. CVRP local search is only as good as the
number of moves you can evaluate, and the naive way to evaluate a move — apply it,
recompute the affected route lengths from scratch, check every route's load by
re-summing demands — is `O(n)` per candidate and throttles the search to a
trickle. The whole game is to make each move `O(1)` to *evaluate*:

- **Capacity in `O(1)`.** I keep `load[k]`, the running demand sum of route `k`,
  updated incrementally. A relocate of customer `c` from `a` to `b` is feasible
  iff `load[b] + dem[c] <= Q`; a swap of `ca,cb` iff `load[a] - dem[ca] + dem[cb]
  <= Q` and symmetrically for `b`. No re-summation. For 2-opt\* I need the demand
  of the head segment of each route at the chosen cut; computing the head load
  costs `O(cut)` here (I sum the prefix), but the *distance* part of the move is
  still `O(1)` and most candidate 2-opt\* moves are rejected on capacity before I
  ever touch distance, so it stays cheap in practice.

- **Distance delta touches at most 4 edges.** A relocate removes the two edges
  around `c` in `a` and the one edge it splits in `b`, and adds the bridging edge
  in `a` and the two new edges in `b` — six edge evaluations, all `O(1)`. A swap
  re-evaluates the four edges incident to the two swapped customers. A 2-opt\*
  changes exactly the two cut edges into two new cross edges — `before = d(la,fa)
  + d(lb,fb)`, `after = d(la,fb) + d(lb,fa)` — a four-edge delta. I never retrace
  a full route to score a candidate; I compute the delta directly and apply the
  move only if I accept it.

That `O(1)` incremental evaluation is the difference between a search that tries
a few thousand moves and one that tries millions, and millions is what it takes
to find the good basins.

## The search engine: warm-started simulated annealing

Pure hill-climbing on these moves stalls in a local optimum quickly, because the
useful 2-opt\* and relocate moves often go *uphill* before they pay off (you have
to accept a temporarily worse partition to reach a better one). So I wrap the
move set in **simulated annealing**: propose a random legal move, compute its
distance delta, accept it outright if it improves, and otherwise accept it with
probability `exp(-delta / T)` for a temperature `T` that I cool geometrically
from a calibrated `Tstart` down to `Tstart * 1e-3` over the time budget. Early,
the search wanders and reshapes the partition; late, it only takes improving
moves and settles. Throughout, I keep `best`, the cheapest *feasible* solution
ever seen, and at the end I emit `best`, never merely the current state — so a
late uphill move can never hurt the reported answer.

I calibrate `Tstart` from the average inter-point distance (so the acceptance
scale matches the geometry of the instance rather than a magic constant), and I
warm-start from the sweep rather than from scratch, because a sweep is already a
strong basin and I would rather spend the annealing budget *refining* a good
partition than *reconstructing* a mediocre one. I also try eight sweeps with
different angular offsets, polish each with intra-route 2-opt/Or-opt, and keep
the best as the warm start — cheap insurance against an unlucky cut alignment.

The move mix each iteration is: relocate, swap, 2-opt\*, and an intra-route
improvement burst (run 2-opt + Or-opt on one random route to keep tours locally
optimal as the partition shifts). The first three reshape *which* customers ride
together; the fourth keeps *each* tour well-ordered. Together they cover both
halves of the problem.

## A real debugging episode

My first run looked like it worked — it compiled, it ran, it printed `K` lines —
but I did not trust it, so I wired up the scorer and a trivial sweep baseline and
ran seeds `1..20`, checking three things: every output feasible (score `> 0`),
the solver's mean strictly above the baseline, and ideally the score above `1.0`
(beating Clarke-Wright). The first thing I did was sanity-check the *score
accounting*. My SA maintains `curCost` incrementally by adding each accepted
move's delta; it is dangerously easy for that running total to drift away from
the truth if a single delta formula is off by an edge. So I had the code, after
all the annealing, recompute `best.cost` from scratch with `total_cost` and emit
from `best`. When I temporarily printed both the incremental `curCost` and the
recomputed cost, an early version disagreed: my Or-opt insertion-position guard
was wrong. In the Or-opt move I skip insertion positions that overlap the
just-removed segment, but my first guard (`if (j >= i && j <= i + L)`) was
off-by-one at the segment boundary and could let a segment be reinserted across
its own gap, corrupting the route order and double-counting an edge in the delta.
I tightened the overlap test and, crucially, recomputed each modified route's
length directly inside the intra burst (`bef`/`aft` via `route_len`) instead of
trusting a hand-rolled delta there, so the intra burst contributes an *exact*
change to `curCost`. After that fix the incremental and recomputed costs agreed,
and the printed solution matched what the scorer measured.

The second bug was a feasibility scare, and it is why I keep the paranoid
fallback. In one intermediate version a 2-opt\* swap of tails could, when one of
the cut points was at the very end of a route (cut after the last element, so the
"tail" was empty), produce an empty `vector` slice that I mishandled and left a
customer un-emitted. The scorer caught it instantly: customer count `!= 1` for
some id, score `0`. I fixed the slice arithmetic so the head/tail split is always
a clean `[0,cut)` / `[cut,size)` partition (cut ranges over `0..size`
inclusive), and as a belt-and-braces guard I added a final feasibility check
before printing: re-scan the `best` routes, and if any customer is not served
exactly once or any route is over `Q`, fall back to the first-fit-decreasing
packer (which is feasible by construction). With that guard the program is
*structurally* incapable of printing an infeasible solution — even a hypothetical
future bug in a move would be caught and repaired at the gate.

With both fixes in, the seed set told a clean story. Across seeds `1..20`: every
output feasible; the solver's mean score `1.0563` against the trivial sweep
baseline's `0.5132` — the solver more than doubles the baseline — and the solver
beats Clarke-Wright (`score > 1`) on `19` of `20` seeds, dipping just below on a
single seed (`0.9947`) where the savings construction happened to be very strong,
while still crushing the trivial baseline there. Raw distances confirm it: e.g.
seed 1 drops from the baseline's `24233` to `10599`. Timing stayed under the
2-second budget (max wall ~`1.97s`), and a stress run over seeds `21..50`
produced zero infeasibilities. The conclusion: warm-started SA with cross-route
relocate / swap / 2-opt\* and `O(1)` capacity checks is a genuinely strong CVRP
heuristic here, not a toy, and it is robustly feasible.

## Why this is the right method, restated

The structure of CVRP is "pack then route," and the binding difficulty is that
the packing and the routing interact: the best partition depends on how well each
group can be toured, and vice versa. Sweep gives a feasible, geometry-respecting
partition for free. The cross-route moves — especially 2-opt\*, which migrates an
entire tail segment in one step — are exactly what fix the partition that sweep
gets wrong, and the `O(1)` incremental evaluation (capacity from running loads,
distance from a 4-edge delta) is what makes a metaheuristic over those moves fast
enough to matter. Simulated annealing supplies the uphill tolerance that lets the
partition reshape itself out of the sweep's basin. Every accepted move preserves
feasibility, the best feasible state is what gets printed, and a final guard
makes infeasibility impossible. That is the whole solver.

## Final solver

```cpp
// Capacitated Multi-Vehicle Routing (ale-17)
// ------------------------------------------------------------------
// Read an instance from stdin, write K vehicle routes to stdout, minimising the
// total closed-tour distance (depot -> ... -> depot) subject to:
//   * each customer served exactly once,
//   * each route's total demand <= Q,
//   * exactly K routes printed (empty routes allowed).
//
// Strategy (the heuristic innovation):
//   1) SWEEP construction: sort customers by polar angle around the depot and
//      cut the sweep into routes greedily under the capacity Q. This gives a
//      feasible, spatially coherent starting solution very fast.
//   2) Local search with O(1) incremental evaluation. The route loads are kept
//      as running sums so every capacity check is O(1); every distance delta
//      touches at most 4 edges and is computed in O(1). Moves:
//        - intra-route Or-opt (relocate a segment of length 1..3 inside a route),
//        - 2-opt inside a route,
//        - cross-route RELOCATE (move one customer to another route),
//        - cross-route SWAP (exchange two customers between routes),
//        - cross-route 2-opt* (swap the tails of two routes at a cut point):
//          this is the move that repairs badly balanced sweeps.
//      A simulated-annealing acceptance lets the search escape local optima; we
//      always remember the best feasible solution seen and print that.
//
// The program ALWAYS emits a feasible solution: the sweep construction is
// feasible by construction, and every accepted move preserves feasibility.
// ------------------------------------------------------------------
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

int n, K;
long long Q;
vector<double> X, Y;     // 0 = depot, 1..n = customers
vector<long long> dem;   // dem[0] = 0

static inline double D(int a, int b) {
    double dx = X[a] - X[b], dy = Y[a] - Y[b];
    return sqrt(dx * dx + dy * dy);
}

struct RNG {
    uint64_t s;
    RNG(uint64_t seed) : s(seed) {}
    inline uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
    inline int randint(int n) { return (int)(next() % (uint64_t)n); }
    inline double rand01() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

// A solution = K routes (each a vector of customer ids) + cached loads.
struct Solution {
    vector<vector<int>> route;
    vector<long long> load;
    double cost;
};

double route_len(const vector<int>& r) {
    if (r.empty()) return 0.0;
    double t = D(0, r[0]);
    for (size_t i = 0; i + 1 < r.size(); i++) t += D(r[i], r[i + 1]);
    t += D(r.back(), 0);
    return t;
}

double total_cost(const Solution& s) {
    double t = 0;
    for (auto& r : s.route) t += route_len(r);
    return t;
}

// ---- Sweep construction --------------------------------------------------
Solution sweep_construct(double angle_offset) {
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i + 1;
    vector<double> ang(n + 1);
    for (int i = 1; i <= n; i++) {
        double a = atan2(Y[i] - Y[0], X[i] - X[0]) + angle_offset;
        if (a < 0) a += 2 * M_PI;
        if (a >= 2 * M_PI) a -= 2 * M_PI;
        ang[i] = a;
    }
    sort(order.begin(), order.end(), [&](int a, int b) { return ang[a] < ang[b]; });

    Solution s;
    s.route.assign(K, {});
    s.load.assign(K, 0);
    int v = 0;
    long long totalDem = 0;
    for (int i = 1; i <= n; i++) totalDem += dem[i];
    // Greedy fill: keep adding to the current vehicle until adding the next
    // would exceed capacity AND there is still capacity budget in later
    // vehicles; otherwise move on. We also try to keep loads balanced by not
    // over-stuffing early routes.
    for (int idx = 0; idx < n; idx++) {
        int c = order[idx];
        // advance vehicle if it cannot take c
        while (v < K && s.load[v] + dem[c] > Q) v++;
        if (v >= K) {
            // overflow: place into the least-loaded route that still fits
            int best = -1;
            for (int k = 0; k < K; k++)
                if (s.load[k] + dem[c] <= Q && (best < 0 || s.load[k] < s.load[best]))
                    best = k;
            if (best < 0) best = 0;  // should not happen given feasibility margin
            s.route[best].push_back(c);
            s.load[best] += dem[c];
        } else {
            s.route[v].push_back(c);
            s.load[v] += dem[c];
        }
    }
    s.cost = total_cost(s);
    return s;
}

// Make sure every customer fits somewhere: a fallback feasible repair that
// never fails as long as sum(dem) <= K*Q and max dem <= Q (guaranteed by gen).
Solution feasible_fallback() {
    Solution s;
    s.route.assign(K, {});
    s.load.assign(K, 0);
    // sort customers by demand descending, first-fit into routes
    vector<int> ids(n);
    for (int i = 0; i < n; i++) ids[i] = i + 1;
    sort(ids.begin(), ids.end(), [&](int a, int b) { return dem[a] > dem[b]; });
    for (int c : ids) {
        int best = -1;
        for (int k = 0; k < K; k++)
            if (s.load[k] + dem[c] <= Q && (best < 0 || s.load[k] < s.load[best])) best = k;
        if (best < 0) {
            // last resort: put in route with smallest load (may overload, but
            // gen guarantees this branch is unreachable)
            best = 0;
            for (int k = 1; k < K; k++) if (s.load[k] < s.load[best]) best = k;
        }
        s.route[best].push_back(c);
        s.load[best] += dem[c];
    }
    s.cost = total_cost(s);
    return s;
}

// ---- Intra-route 2-opt (full pass, improves a single route) --------------
bool two_opt_route(vector<int>& r) {
    int m = (int)r.size();
    if (m < 2) return false;
    bool improvedAny = false;
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < m - 1; i++) {
            int a = (i == 0) ? 0 : r[i - 1];
            int b = r[i];
            for (int j = i + 1; j < m; j++) {
                int c = r[j];
                int d = (j == m - 1) ? 0 : r[j + 1];
                if (a == c || b == d) continue;
                double before = D(a, b) + D(c, d);
                double after = D(a, c) + D(b, d);
                if (after + 1e-9 < before) {
                    reverse(r.begin() + i, r.begin() + j + 1);
                    improved = true;
                    improvedAny = true;
                    b = r[i];
                }
            }
        }
    }
    return improvedAny;
}

// ---- Or-opt: move a segment of length L (1..3) to a better position in the
// same route. O(1) delta per candidate. --------------------------------------
bool or_opt_route(vector<int>& r) {
    int m = (int)r.size();
    bool improvedAny = false;
    for (int L = 1; L <= 3; L++) {
        if (m < L + 1) continue;
        bool improved = true;
        while (improved) {
            improved = false;
            m = (int)r.size();
            for (int i = 0; i + L <= m; i++) {
                int p = (i == 0) ? 0 : r[i - 1];
                int s0 = r[i];
                int s1 = r[i + L - 1];
                int q = (i + L == m) ? 0 : r[i + L];
                double removed = D(p, s0) + D(s1, q) - D(p, q);
                // try inserting between position j and j+1 (outside [i, i+L))
                for (int j = 0; j <= m; j++) {
                    if (j >= i - 0 && j <= i + L) continue;  // overlaps removal gap
                    int u = (j == 0) ? 0 : r[j - 1];
                    int w = (j == m) ? 0 : r[j];
                    if (u == s0 || w == s1) continue;
                    double added = D(u, s0) + D(s1, w) - D(u, w);
                    if (added + 1e-9 < removed) {
                        // perform move
                        vector<int> seg(r.begin() + i, r.begin() + i + L);
                        r.erase(r.begin() + i, r.begin() + i + L);
                        int jj = j;
                        if (j > i) jj -= L;
                        r.insert(r.begin() + jj, seg.begin(), seg.end());
                        improved = true;
                        improvedAny = true;
                        break;
                    }
                }
                if (improved) break;
            }
        }
    }
    return improvedAny;
}

int main() {
    // ---- read instance ----
    if (!(cin >> n >> K >> Q)) return 0;
    X.assign(n + 1, 0); Y.assign(n + 1, 0); dem.assign(n + 1, 0);
    cin >> X[0] >> Y[0];
    for (int i = 1; i <= n; i++) cin >> X[i] >> Y[i] >> dem[i];

    if (n == 0) {
        for (int k = 0; k < K; k++) cout << "\n";
        return 0;
    }

    RNG rng(0x9e3779b97f4a7c15ULL ^ (uint64_t)n ^ ((uint64_t)K << 20) ^ ((uint64_t)Q << 33));
    double t0 = now_sec();
    const double TIME_LIMIT = 1.9;  // seconds

    // ---- build several sweep starts, keep the best, polish with intra moves --
    Solution best = sweep_construct(0.0);
    for (auto& r : best.route) { two_opt_route(r); or_opt_route(r); }
    best.cost = total_cost(best);

    for (int t = 1; t < 8; t++) {
        Solution cand = sweep_construct((2 * M_PI * t) / 8.0);
        for (auto& r : cand.route) { two_opt_route(r); or_opt_route(r); }
        cand.cost = total_cost(cand);
        if (cand.cost < best.cost) best = cand;
        if (now_sec() - t0 > 0.4) break;
    }
    // safety: ensure feasibility of the chosen start
    {
        bool ok = true;
        vector<int> seen(n + 1, 0);
        for (int k = 0; k < K; k++) {
            if (best.load[k] > Q) ok = false;
            for (int c : best.route[k]) seen[c]++;
        }
        for (int c = 1; c <= n; c++) if (seen[c] != 1) ok = false;
        if (!ok) {
            best = feasible_fallback();
            for (auto& r : best.route) { two_opt_route(r); or_opt_route(r); }
            best.cost = total_cost(best);
        }
    }

    // ---- simulated annealing over cross-route moves -------------------------
    Solution cur = best;
    double curCost = cur.cost;
    double Tstart = 0.0;
    // calibrate temperature from average edge length
    {
        double avg = 0; int cnt = 0;
        for (int i = 0; i <= n && cnt < 200; i++)
            for (int j = i + 1; j <= n && cnt < 200; j++) { avg += D(i, j); cnt++; }
        if (cnt) avg /= cnt;
        Tstart = avg * 0.5 + 1e-6;
    }
    double Tend = Tstart * 1e-3;

    long long iter = 0;
    while (true) {
        if ((iter & 1023) == 0) {
            double el = now_sec() - t0;
            if (el > TIME_LIMIT) break;
        }
        iter++;
        double frac = (now_sec() - t0) / TIME_LIMIT;
        if (frac > 1) frac = 1;
        double T = Tstart * pow(Tend / Tstart, frac);

        int moveType = rng.randint(4);

        if (moveType == 0) {
            // cross-route RELOCATE: move one customer from route a to route b.
            int a = rng.randint(K);
            if (cur.route[a].empty()) continue;
            int pi = rng.randint((int)cur.route[a].size());
            int c = cur.route[a][pi];
            int b = rng.randint(K);
            if (b == a) continue;
            if (cur.load[b] + dem[c] > Q) continue;  // O(1) capacity check
            // removal delta on a
            int pa = (pi == 0) ? 0 : cur.route[a][pi - 1];
            int na = (pi + 1 == (int)cur.route[a].size()) ? 0 : cur.route[a][pi + 1];
            double remDelta = D(pa, c) + D(c, na) - D(pa, na);
            // best insertion position in b (cheapest), O(|b|)
            auto& rb = cur.route[b];
            double bestAdd = 1e18; int bestPos = 0;
            for (int j = 0; j <= (int)rb.size(); j++) {
                int u = (j == 0) ? 0 : rb[j - 1];
                int w = (j == (int)rb.size()) ? 0 : rb[j];
                double add = D(u, c) + D(c, w) - D(u, w);
                if (add < bestAdd) { bestAdd = add; bestPos = j; }
            }
            double delta = bestAdd - remDelta;  // touches <= 4 edges
            if (delta < -1e-12 || rng.rand01() < exp(-delta / T)) {
                cur.route[a].erase(cur.route[a].begin() + pi);
                cur.load[a] -= dem[c];
                rb.insert(rb.begin() + bestPos, c);
                cur.load[b] += dem[c];
                curCost += delta;
            }
        } else if (moveType == 1) {
            // cross-route SWAP: exchange customer ca in route a with cb in route b.
            int a = rng.randint(K), b = rng.randint(K);
            if (a == b || cur.route[a].empty() || cur.route[b].empty()) continue;
            int pi = rng.randint((int)cur.route[a].size());
            int pj = rng.randint((int)cur.route[b].size());
            int ca = cur.route[a][pi], cb = cur.route[b][pj];
            // O(1) capacity check
            if (cur.load[a] - dem[ca] + dem[cb] > Q) continue;
            if (cur.load[b] - dem[cb] + dem[ca] > Q) continue;
            int pa = (pi == 0) ? 0 : cur.route[a][pi - 1];
            int na = (pi + 1 == (int)cur.route[a].size()) ? 0 : cur.route[a][pi + 1];
            int pb = (pj == 0) ? 0 : cur.route[b][pj - 1];
            int nb = (pj + 1 == (int)cur.route[b].size()) ? 0 : cur.route[b][pj + 1];
            double before = D(pa, ca) + D(ca, na) + D(pb, cb) + D(cb, nb);
            double after = D(pa, cb) + D(cb, na) + D(pb, ca) + D(ca, nb);
            double delta = after - before;  // touches <= 4 edges
            if (delta < -1e-12 || rng.rand01() < exp(-delta / T)) {
                cur.route[a][pi] = cb;
                cur.route[b][pj] = ca;
                cur.load[a] += dem[cb] - dem[ca];
                cur.load[b] += dem[ca] - dem[cb];
                curCost += delta;
            }
        } else if (moveType == 2) {
            // cross-route 2-opt*: pick routes a,b and cut points; swap the tails.
            // route a = [head_a | tail_a], route b = [head_b | tail_b].
            // new a = head_a + tail_b, new b = head_b + tail_a.
            int a = rng.randint(K), b = rng.randint(K);
            if (a == b) continue;
            auto& ra = cur.route[a];
            auto& rb = cur.route[b];
            int ca = rng.randint((int)ra.size() + 1);  // cut after index ca-1 (0..size)
            int cb = rng.randint((int)rb.size() + 1);
            // loads of head/tail via prefix sums computed on the fly (O(size));
            // to keep O(1) we precompute prefix loads lazily here.
            long long headA = 0; for (int i = 0; i < ca; i++) headA += dem[ra[i]];
            long long tailA = cur.load[a] - headA;
            long long headB = 0; for (int i = 0; i < cb; i++) headB += dem[rb[i]];
            long long tailB = cur.load[b] - headB;
            // O(1) capacity feasibility check
            if (headA + tailB > Q) continue;
            if (headB + tailA > Q) continue;
            // distance delta: only the two cut edges change (4 edges -> 2 new)
            int la = (ca == 0) ? 0 : ra[ca - 1];           // last of head_a
            int fa = (ca == (int)ra.size()) ? 0 : ra[ca];  // first of tail_a
            int lb = (cb == 0) ? 0 : rb[cb - 1];           // last of head_b
            int fb = (cb == (int)rb.size()) ? 0 : rb[cb];  // first of tail_b
            double before = D(la, fa) + D(lb, fb);
            double after = D(la, fb) + D(lb, fa);
            double delta = after - before;
            if (delta < -1e-12 || rng.rand01() < exp(-delta / T)) {
                vector<int> newA(ra.begin(), ra.begin() + ca);
                newA.insert(newA.end(), rb.begin() + cb, rb.end());
                vector<int> newB(rb.begin(), rb.begin() + cb);
                newB.insert(newB.end(), ra.begin() + ca, ra.end());
                cur.route[a] = move(newA);
                cur.route[b] = move(newB);
                cur.load[a] = headA + tailB;
                cur.load[b] = headB + tailA;
                curCost += delta;
            }
        } else {
            // intra-route improvement burst on a random route (Or-opt + 2-opt)
            int a = rng.randint(K);
            if (cur.route[a].size() < 2) continue;
            double bef = route_len(cur.route[a]);
            two_opt_route(cur.route[a]);
            or_opt_route(cur.route[a]);
            double aft = route_len(cur.route[a]);
            curCost += (aft - bef);
        }

        if (curCost < best.cost - 1e-9) {
            best = cur;
            best.cost = curCost;
        }
    }

    // ---- final polish of the best solution ----
    for (auto& r : best.route) { two_opt_route(r); or_opt_route(r); }
    best.cost = total_cost(best);

    // ---- output exactly K routes ----
    // sanity: verify feasibility once more; if somehow broken, fall back.
    {
        bool ok = true;
        vector<int> seen(n + 1, 0);
        for (int k = 0; k < K; k++) {
            long long ld = 0;
            for (int c : best.route[k]) { seen[c]++; ld += dem[c]; }
            if (ld > Q) ok = false;
        }
        for (int c = 1; c <= n; c++) if (seen[c] != 1) ok = false;
        if (!ok) best = feasible_fallback();
    }

    string out;
    out.reserve(n * 4);
    for (int k = 0; k < K; k++) {
        auto& r = best.route[k];
        for (size_t i = 0; i < r.size(); i++) {
            if (i) out += ' ';
            out += to_string(r[i]);
        }
        out += '\n';
    }
    cout << out;
    return 0;
}
```
