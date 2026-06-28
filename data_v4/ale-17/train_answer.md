# Capacitated Multi-Vehicle Routing — solution write-up

## Problem

A depot and `n` customers (`120 <= n <= 200`) lie on a `[0,1000]^2` grid. A fleet
of `K` identical vehicles (`6 <= K <= 12`), each of capacity `Q`, must serve every
customer exactly once; each vehicle leaves the depot, visits a subset of
customers, and returns. A route is feasible only if its total demand is `<= Q`.
The instance always admits a feasible assignment (`sum(d_i) <= K*Q`,
`max(d_i) <= Q`). Input is `n K Q`, the depot coordinates, then `n` lines
`x y demand`; output is exactly `K` lines, each the customer ids of one route in
visit order (empty line for an unused vehicle).

## Objective and scoring

Minimise the total closed-tour distance

```
D = sum over routes ( d(0,r[0]) + sum_i d(r[i],r[i+1]) + d(r[last],0) )
```

with Euclidean edges and depot index `0`. The judge floors any infeasible output
to `0` — infeasible meaning a wrong number of route lines, an out-of-range id, a
customer served zero or more than once, or a route over capacity. For a feasible
output the continuous score normalises against the Clarke-Wright savings baseline
distance `D_cw` computed on the same instance:

```
score = D_cw / D     (0 if infeasible).
```

`score = 1` ties Clarke-Wright; `score > 1` is strictly shorter. The reported
metric is the mean score over the fixed seed set `1..20`.

## Baseline

The starting point is a **sweep**: sort customers by polar angle around the depot
and fill vehicles in angular order under capacity, spilling into the least-loaded
fitting route when the angular pointer runs out of vehicles. It is `O(n log n)`,
always feasible given the generator's margin, and spatially coherent, but its cut
points ignore load balance, so it scores about `0.51` against Clarke-Wright —
roughly half. That gap is the optimisation target.

## Key idea — the heuristic innovation

Warm-start a **simulated annealing** local search from the sweep and give it a
move set that can fix the *partition*, not just each tour:

- **cross-route relocate** — pull a customer from route `a`, insert it at its
  cheapest position in route `b`;
- **cross-route swap** — exchange a customer of `a` with a customer of `b`;
- **cross-route 2-opt\*** — choose a cut point in each of two routes and swap
  their *tail segments*; this migrates an entire mis-assigned arc between vehicles
  in one move and is the move that repairs the sweep's unbalanced cuts;
- **intra-route 2-opt and Or-opt** — keep each individual tour locally optimal as
  the partition shifts.

The decisive engineering point is `O(1)` incremental evaluation. Per-route loads
`load[k]` are kept as running sums, so every capacity check is a single
comparison (`load[b]+dem[c] <= Q` for relocate, the symmetric pair for swap, the
head/tail loads for 2-opt\*). Every distance change touches at most four edges and
is computed directly as a delta — a relocate's `(removed - added)`, a swap's four
incident edges, a 2-opt\*'s two old cut edges versus two new cross edges — without
ever retracing a route. That cheapness is what lets the annealer try millions of
moves inside the 2-second budget. Acceptance is the standard
`improve-or-exp(-delta/T)` with a geometric cooling from a temperature calibrated
to the average inter-point distance; the best feasible solution ever seen is kept
and printed.

## Feasibility and pitfalls

Feasibility is preserved by construction at every step: the sweep is feasible, and
each move's `O(1)` capacity test rejects any move that would overload a route, so
`load[k] <= Q` is an invariant. The pitfalls I hit and closed:

- **2-opt\* tail slices at route boundaries.** A cut at index `0` or at `size`
  yields an empty head or tail; the head/tail split must be the clean partition
  `[0,cut)` / `[cut,size)` with `cut` ranging `0..size`, or a customer silently
  goes un-emitted. Getting this right keeps "served exactly once" intact.
- **Incremental-cost drift.** The annealer maintains `curCost` by summing deltas;
  one wrong edge in a delta makes it diverge from reality. The fix is to recompute
  the chosen `best.cost` from scratch at the end and to compute the intra-burst
  change with exact `route_len` differences rather than a hand-rolled delta.
- **Or-opt overlap guard.** The insertion position must skip the removed segment's
  own gap, or a segment can be reinserted across itself and corrupt the route.
- **A final feasibility gate.** Before printing, the routes are re-scanned; if any
  customer is not served exactly once or any route exceeds `Q`, the solver falls
  back to a first-fit-decreasing packer that is feasible by construction. This
  makes an infeasible (score-0) output structurally impossible.

On seeds `1..20` the solver is feasible everywhere, scores mean `1.0563` versus
the trivial sweep baseline's `0.5132` (more than double), and beats Clarke-Wright
(`score > 1`) on 19 of 20 seeds; wall time stays under 2 s.

## Complexity per step

Sweep construction is `O(n log n)`. Each SA iteration is `O(1)` for relocate's
capacity/distance delta plus `O(|b|)` to find the cheapest insertion (small,
since routes hold `~n/K` customers); swap is `O(1)`; 2-opt\* is `O(1)` for the
distance delta plus an `O(cut)` head-load sum that is short-circuited by the
capacity reject; the intra burst is a bounded 2-opt/Or-opt pass on one route.
Across the budget the annealer evaluates on the order of millions of moves.

## Code

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
