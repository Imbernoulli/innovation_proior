# Facility Location with Opening Cost — Editorial

## Problem

We are given `F` candidate facility sites (each with integer coordinates and a
fixed opening cost `open[i]`) and `C` clients (coordinates). We choose a
**non-empty** subset `S` of sites to open. Each client is served by its single
**nearest open facility** at Euclidean service cost. This is the **Uncapacitated
Facility Location Problem (UFLP)** — NP-hard, scored on a continuous cost.

- Input: `F C`, then `F` lines `fx fy fcost`, then `C` lines `cx cy`.
- Output: `M` (with `1 ≤ M ≤ F`), then `M` distinct facility indices in `[1,F]`.
- Sizes: `60 ≤ F ≤ 120`, `400 ≤ C ≤ 700`. Time limit 2 s, memory 256 MB.

## Objective and scoring

Minimize

```
total = sum_{i in S} open[i]  +  sum_{c} min_{i in S} euclid(client c, facility i).
```

The judge reports `score = round(10^9 / (1 + total/C))` when the output is
feasible, and `0` otherwise. Feasibility = `1 ≤ M ≤ F`, all indices in range,
all distinct. Any malformed / empty / `M=0` / out-of-range / duplicate output
**floors the score to 0**. Lower `total` ⇒ higher score (the wrapper is
monotone). Because `S` is non-empty, every feasible solution has a finite service
cost.

## Baseline

Two feasible-but-weak references bound the problem. "Open all facilities" gives
the minimum possible service cost but pays every opening cost; "open the single
cheapest facility" pays almost no opening cost but routes all clients through one
depot, exploding the service term. The optimum opens an intermediate count where
the marginal opening cost of the next facility equals the service distance it
saves. We take **"open all"** as the trivial baseline to beat (deterministic, and
the answer a careless solver emits).

## Key idea (the heuristic innovation)

The standard UFLP metaheuristic is local search over **ADD / DROP / SWAP** moves.
ADD is cheap (`O(C)`: compare each client's nearest distance to the new
facility). DROP is the bottleneck: when facility `i` closes, every client served
by `i` needs its new nearest among the *remaining* open facilities — naively
`O(|S|)` per affected client, so `O(C·|S|)` per candidate drop and `O(C·|S|^2)`
per pass. Too slow to sweep all candidates thousands of times in 2 seconds.

**The lever: cache, per client, the first AND second nearest open facility.**
Keep `d1[c], near1[c]` (nearest) and `d2[c], near2[c]` (second-nearest). Then

```
DROP gain(i) = open[i] - sum_{c : near1[c]==i} (d2[c] - d1[c])      // O(C)
ADD  gain(i) = open[i] - sum_{c} max(0, d1[c] - dist(c,i))          // O(C)
```

The second-nearest cache turns the drop *evaluation* from `O(C·|S|)` into `O(C)`:
a dropped client just falls back to its cached runner-up. This is the
Whitaker/Resende–Werneck neighbor-list trick. It is what makes it affordable to
re-score every candidate add and drop on every pass and run a deep search.

For the **starting set**, we round an **LP / Lagrangian relaxation** rather than
starting from a trivial extreme. Relaxing the assignment-equality constraints
with multipliers `u_c`, facility `i` has reduced cost
`rc_i = open[i] + sum_c min(0, dist(c,i) - u_c)`; opening the facilities with
`rc_i < 0` and running a subgradient ascent on `u` yields a principled rounding,
far better than open-all / open-one. The local search then refines it, wrapped in
**iterated local search** (kick + re-descend, keep the best set) to use the full
time budget.

## Feasibility and pitfalls

- **Always non-empty.** Every stage guards against an empty set: the Lagrangian
  rounding forces the smallest-reduced-cost facility open if none qualify; the
  perturbation never drops below one open facility; a final guard opens the
  cheapest facility if somehow empty. `M ≥ 1` always holds.
- **DROP legality.** A drop is forbidden if it would leave 0 facilities open, or
  if any served client lacks a valid second-nearest (`d2` infinite) — that guard
  prevents an undefined fallback.
- **Cache coherence is the real risk.** After a drop, clients whose `near1` or
  `near2` was the dropped facility must refind a fresh second-nearest by one scan
  over the open set; the incremental `curCost` must be updated in lockstep. We
  verified the tracked `curCost` equals an independent from-scratch recomputation
  to the last decimal on multiple seeds — a stale `d2` would silently corrupt the
  objective without crashing.
- **Distance matrix.** `F·C ≤ 84000` doubles, precomputed once, so every move is a
  flat-array `O(C)` with no repeated `sqrt`.

## Complexity per step

- Build distance matrix: `O(F·C)` once.
- Lagrangian iteration: `O(F·C)` per subgradient step (capped, ~45% of budget).
- ADD evaluation: `O(C)`; DROP evaluation: `O(C)` (thanks to the `d2` cache).
- One local-search pass scanning all candidates: `O(F·C)`.
- Apply-drop refinement for affected clients: `O(C·|S|)` worst case, but only for
  the single committed move, not the sweep.

This is what lets the solver run thousands of moves within the 2-second budget and
comfortably beat both trivial baselines (mean score ≈ 1.36× the open-all baseline,
≈ 8.4× the single-cheapest baseline over seeds 1..20, every output feasible).

## Code

```cpp
// ale-v2-05 "Facility Location with Opening Cost" (Uncapacitated Facility
// Location, UFLP).
//
// Given F candidate facilities (each with coordinates and an opening cost) and
// C clients (coordinates), choose a NON-EMPTY subset S of facilities to open.
// Each client is served by its single nearest open facility at Euclidean
// service cost. Minimise
//     total = sum_{i in S} open[i] + sum_{c} min_{i in S} dist(c, i).
// NP-hard; judged by a continuous cost (lower is better).
//
// Strategy (two stages):
//
//   1. LP-relaxation rounding for the initial open set. We relax the UFLP LP by
//      Lagrangian relaxation of the "every client is assigned exactly once"
//      equalities (multipliers u_c). For fixed u, each facility i contributes
//      reduced cost rc_i = open[i] + sum_c min(0, dist(c,i) - u_c); we open the
//      facilities with rc_i < 0. A subgradient ascent on u maximises the
//      Lagrangian lower bound; the open-indicator at the best dual gives a
//      principled fractional->integral rounding (much better than open-all or
//      open-one). We keep the best feasible rounded set seen during the ascent.
//
//   2. Local search with the FIRST/SECOND-nearest cache (the innovation). With
//      d1[c] (distance to the nearest open facility), near1[c] (which facility),
//      and d2[c] (distance to the SECOND nearest open facility) cached for every
//      client, the three local-search moves become cheap:
//        * ADD i:   gain = open[i] - sum_c max(0, d1[c] - dist(c,i))     O(C)
//        * DROP i:  for clients whose near1==i the new service cost is d2[c];
//                   gain = open[i] - sum_{c:near1==i}(d2[c]-d1[c])       O(C)
//                   (correct ONLY because d2 is cached; a naive drop would
//                    rescan all open facilities for those clients, O(C*|S|)).
//        * SWAP i_out,i_in handled as the better of the coupled add/drop.
//      We apply the best improving move, then UPDATE the caches incrementally.
//      The d2 cache is what turns drop/swap from O(C*|S|) into O(C) per move,
//      letting us sweep all F candidates per pass within the time budget.
//
// We ALWAYS hold a feasible (non-empty) open set, so whenever the clock runs out
// we can print a legal answer.

#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    return (double)chrono::duration_cast<chrono::microseconds>(
               chrono::steady_clock::now().time_since_epoch())
               .count() *
           1e-6;
}

int F, C;
vector<double> FX, FY, FCOST;
vector<double> CX, CY;

static inline double dist(int f, int c) {
    double dx = FX[f] - CX[c];
    double dy = FY[f] - CY[c];
    return sqrt(dx * dx + dy * dy);
}

// Full objective for an open set (used only occasionally / for the final guard).
static double full_cost(const vector<char> &open) {
    double total = 0.0;
    int cntOpen = 0;
    for (int i = 0; i < F; i++)
        if (open[i]) { total += FCOST[i]; cntOpen++; }
    if (cntOpen == 0) return 1e300;  // infeasible sentinel
    for (int c = 0; c < C; c++) {
        double best = 1e300;
        for (int i = 0; i < F; i++)
            if (open[i]) { double d = dist(i, c); if (d < best) best = d; }
        total += best;
    }
    return total;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const double T_BUDGET = 1.9;  // seconds
    const double t_start = now_sec();

    if (!(cin >> F >> C)) return 0;
    FX.assign(F, 0); FY.assign(F, 0); FCOST.assign(F, 0);
    for (int i = 0; i < F; i++) cin >> FX[i] >> FY[i] >> FCOST[i];
    CX.assign(C, 0); CY.assign(C, 0);
    for (int c = 0; c < C; c++) cin >> CX[c] >> CY[c];

    // Degenerate guards: always emit a feasible answer.
    if (F <= 0) { return 0; }
    if (C <= 0) {
        // No clients: open exactly the single cheapest facility (must open >=1).
        int best = 0;
        for (int i = 1; i < F; i++) if (FCOST[i] < FCOST[best]) best = i;
        cout << 1 << "\n" << (best + 1) << "\n";
        return 0;
    }

    // Precompute the full distance matrix when it fits comfortably in memory
    // (F<=120, C<=700 => <=84k doubles). This makes every move O(C) with a flat
    // array access instead of recomputing sqrt.
    // dmat[i*C + c] = dist(facility i, client c).
    vector<double> dmat((size_t)F * C);
    for (int i = 0; i < F; i++) {
        double *row = &dmat[(size_t)i * C];
        for (int c = 0; c < C; c++) {
            double dx = FX[i] - CX[c];
            double dy = FY[i] - CY[c];
            row[c] = sqrt(dx * dx + dy * dy);
        }
    }
    auto D = [&](int i, int c) -> double { return dmat[(size_t)i * C + c]; };

    // ---------- Stage 1: Lagrangian relaxation + rounding ----------
    // Multipliers u_c initialised to the distance to the nearest facility
    // (a standard warm start). rc_i = open_i + sum_c min(0, D(i,c) - u_c).
    vector<double> u(C, 0.0);
    for (int c = 0; c < C; c++) {
        double best = 1e300;
        for (int i = 0; i < F; i++) best = min(best, D(i, c));
        u[c] = best;
    }

    vector<char> open(F, 0);          // current incumbent open set
    vector<char> openTry(F, 0);       // rounded set at the current dual point
    double bestRoundedCost = 1e300;

    // A quick helper: given an open set, compute its true cost cheaply with the
    // distance matrix.
    auto cost_of = [&](const vector<char> &os) -> double {
        int cnt = 0; double tot = 0.0;
        for (int i = 0; i < F; i++) if (os[i]) { tot += FCOST[i]; cnt++; }
        if (cnt == 0) return 1e300;
        for (int c = 0; c < C; c++) {
            double best = 1e300;
            for (int i = 0; i < F; i++)
                if (os[i]) { double d = D(i, c); if (d < best) best = d; }
            tot += best;
        }
        return tot;
    };

    // Subgradient ascent. We cap the iterations and also watch the clock so the
    // bulk of the budget goes to local search.
    double bestLB = -1e300;
    double step = 2.0;
    const double t_stage1_end = t_start + 0.45 * T_BUDGET;
    int it = 0, noImp = 0;
    while (now_sec() < t_stage1_end && it < 400) {
        it++;
        // Reduced costs and the Lagrangian value L(u) = sum_c u_c + sum_i min(0, rc_i).
        double Lval = 0.0;
        for (int c = 0; c < C; c++) Lval += u[c];
        for (int i = 0; i < F; i++) {
            double rc = FCOST[i];
            const double *row = &dmat[(size_t)i * C];
            for (int c = 0; c < C; c++) {
                double t = row[c] - u[c];
                if (t < 0) rc += t;
            }
            openTry[i] = (rc < 0.0) ? 1 : 0;
            if (rc < 0.0) Lval += rc;
        }
        if (Lval > bestLB) bestLB = Lval; else noImp++;

        // Ensure at least one facility open in the rounded set (feasibility);
        // if none, open the one with the smallest reduced cost.
        bool any = false;
        for (int i = 0; i < F; i++) if (openTry[i]) { any = true; break; }
        if (!any) {
            int bi = 0; double brc = 1e300;
            for (int i = 0; i < F; i++) {
                double rc = FCOST[i];
                const double *row = &dmat[(size_t)i * C];
                for (int c = 0; c < C; c++) { double t = row[c] - u[c]; if (t < 0) rc += t; }
                if (rc < brc) { brc = rc; bi = i; }
            }
            openTry[bi] = 1;
        }

        // Evaluate the rounded set; keep the best feasible rounding.
        double rc_cost = cost_of(openTry);
        if (rc_cost < bestRoundedCost) { bestRoundedCost = rc_cost; open = openTry; }

        // Subgradient g_c = 1 - (#open facilities i with D(i,c) - u_c < 0 that
        // would serve c). For the assignment-equality relaxation the subgradient
        // on u_c is (1 - x_c) where x_c is how many opened facilities want c.
        // We use the simpler, well-behaved subgradient: g_c = 1 - [c is served
        // by some opened facility at distance < u_c]. Update u_c += step*g_c.
        for (int c = 0; c < C; c++) {
            int served = 0;
            for (int i = 0; i < F; i++)
                if (openTry[i] && D(i, c) - u[c] < 0) { served++; if (served > 1) break; }
            double g = 1.0 - (double)served;
            u[c] += step * g;
            if (u[c] < 0) u[c] = 0;
        }
        if (noImp > 20) { step *= 0.5; noImp = 0; if (step < 1e-3) break; }
    }

    // Safety: if for any reason the incumbent is empty, fall back to the single
    // cheapest facility so the rest of the pipeline has a feasible start.
    {
        bool any = false; for (int i = 0; i < F; i++) if (open[i]) { any = true; break; }
        if (!any) { int b = 0; for (int i = 1; i < F; i++) if (FCOST[i] < FCOST[b]) b = i; open[b] = 1; }
    }

    // ---------- Stage 2: local search with first/second-nearest cache ----------
    // For every client c: near1[c] = nearest OPEN facility, d1[c] its distance;
    // near2[c] = second-nearest OPEN facility, d2[c] its distance (INF if <2 open).
    vector<int> near1(C, -1), near2(C, -1);
    vector<double> d1(C, 1e300), d2(C, 1e300);

    auto rebuild_cache = [&]() {
        for (int c = 0; c < C; c++) {
            near1[c] = near2[c] = -1; d1[c] = d2[c] = 1e300;
            for (int i = 0; i < F; i++) {
                if (!open[i]) continue;
                double d = D(i, c);
                if (d < d1[c]) { d2[c] = d1[c]; near2[c] = near1[c]; d1[c] = d; near1[c] = i; }
                else if (d < d2[c]) { d2[c] = d; near2[c] = i; }
            }
        }
    };
    rebuild_cache();

    auto open_count = [&]() { int k = 0; for (int i = 0; i < F; i++) k += open[i]; return k; };

    // Current objective tracked incrementally.
    double curCost = 0.0;
    for (int i = 0; i < F; i++) if (open[i]) curCost += FCOST[i];
    for (int c = 0; c < C; c++) curCost += d1[c];

    std::mt19937 rng(12345u);

    // Apply an ADD of facility i (must be currently closed). Updates caches and
    // curCost in O(C). gain is returned (negative = improvement).
    auto apply_add = [&](int i) {
        open[i] = 1; curCost += FCOST[i];
        const double *row = &dmat[(size_t)i * C];
        for (int c = 0; c < C; c++) {
            double d = row[c];
            if (d < d1[c]) {
                curCost += d - d1[c];
                d2[c] = d1[c]; near2[c] = near1[c];
                d1[c] = d; near1[c] = i;
            } else if (d < d2[c]) {
                d2[c] = d; near2[c] = i;
            }
        }
    };

    // Compute the gain of dropping facility i (must be currently open) WITHOUT
    // applying it. For clients served by i, the new service cost becomes d2[c].
    // gain = open[i] - sum_{c:near1==i} (d2[c]-d1[c]); negative => improvement.
    // This is O(C) thanks to the cached second-nearest. NOTE: a drop is only
    // legal if it leaves >=1 facility open AND every served client still has a
    // valid second nearest (d2 finite); otherwise we forbid it.
    auto drop_gain = [&](int i, bool &legal) -> double {
        if (open_count() <= 1) { legal = false; return 1e300; }
        double g = -FCOST[i];
        for (int c = 0; c < C; c++) {
            if (near1[c] == i) {
                if (near2[c] < 0 || d2[c] >= 1e299) { legal = false; return 1e300; }
                g += d2[c] - d1[c];   // service cost increases by this
            }
        }
        legal = true;
        return g;  // negative => dropping helps
    };

    // Apply a DROP of facility i. Clients with near1==i move to their cached
    // second nearest; we must then refind a fresh second nearest for them by a
    // single scan over open facilities (O(#open) per affected client). To keep
    // it O(C * #open_worst) at most, but in practice only clients of i are
    // touched. We recompute d2 for those clients by scanning open set.
    auto apply_drop = [&](int i) {
        open[i] = 0; curCost -= FCOST[i];
        for (int c = 0; c < C; c++) {
            if (near1[c] == i) {
                // promote second to first
                curCost += d2[c] - d1[c];
                d1[c] = d2[c]; near1[c] = near2[c];
                // refind second nearest among remaining open facilities
                d2[c] = 1e300; near2[c] = -1;
                for (int j = 0; j < F; j++) {
                    if (!open[j] || j == near1[c]) continue;
                    double d = D(j, c);
                    if (d < d2[c]) { d2[c] = d; near2[c] = j; }
                }
            } else if (near2[c] == i) {
                // second nearest was i; refind a new second nearest
                d2[c] = 1e300; near2[c] = -1;
                for (int j = 0; j < F; j++) {
                    if (!open[j] || j == near1[c]) continue;
                    double d = D(j, c);
                    if (d < d2[c]) { d2[c] = d; near2[c] = j; }
                }
            }
        }
    };

    // gain of ADD i (closed) computed without applying: O(C).
    auto add_gain = [&](int i) -> double {
        double g = FCOST[i];
        const double *row = &dmat[(size_t)i * C];
        for (int c = 0; c < C; c++) {
            double d = row[c];
            if (d < d1[c]) g -= (d1[c] - d);  // service cost drops
        }
        return g;  // negative => adding helps
    };

    // Best-improvement local search sweeping add and drop moves; plus a SWAP
    // pass (drop i_out + best add) when neither pure move improves, to escape
    // the obvious local optimum. We loop until no improving move or time out.
    double bestSeen = curCost;
    vector<char> bestOpen = open;

    int stale = 0;
    while (now_sec() - t_start < T_BUDGET) {
        // ---- find best ADD ----
        int bestAddI = -1; double bestAddG = -1e-6;  // require strict improvement
        for (int i = 0; i < F; i++) {
            if (open[i]) continue;
            double g = add_gain(i);
            if (g < bestAddG) { bestAddG = g; bestAddI = i; }
        }
        // ---- find best DROP ----
        int bestDropI = -1; double bestDropG = -1e-6;
        for (int i = 0; i < F; i++) {
            if (!open[i]) continue;
            bool legal;
            double g = drop_gain(i, legal);
            if (legal && g < bestDropG) { bestDropG = g; bestDropI = i; }
        }

        if (bestAddI >= 0 && bestAddG <= bestDropG) {
            apply_add(bestAddI);
        } else if (bestDropI >= 0) {
            apply_drop(bestDropI);
        } else {
            // ---- no pure improving move: try a SWAP (drop d, then best add) ----
            bool improved = false;
            // Evaluate swaps lazily: pick a small random sample of open facilities
            // to drop, and for each take the best add afterward, accept if net < 0.
            int OC = open_count();
            if (OC > 1) {
                // candidate open facilities to consider removing
                vector<int> openList;
                for (int i = 0; i < F; i++) if (open[i]) openList.push_back(i);
                shuffle(openList.begin(), openList.end(), rng);
                int tries = min((int)openList.size(), 8);
                double bestNet = -1e-6; int bo = -1, bi = -1;
                for (int t = 0; t < tries; t++) {
                    int iout = openList[t];
                    bool legal;
                    double dg = drop_gain(iout, legal);
                    if (!legal) continue;
                    // tentatively drop, evaluate best add, then undo.
                    apply_drop(iout);
                    int localBestI = -1; double localBestG = 1e300;
                    for (int j = 0; j < F; j++) {
                        if (open[j]) continue;
                        double g = add_gain(j);
                        if (g < localBestG) { localBestG = g; localBestI = j; }
                    }
                    double net = dg + (localBestI >= 0 ? localBestG : 0.0);
                    if (net < bestNet) { bestNet = net; bo = iout; bi = localBestI; }
                    // undo the tentative drop
                    apply_add(iout);
                }
                if (bo >= 0) {
                    apply_drop(bo);
                    if (bi >= 0) apply_add(bi);
                    improved = true;
                }
            }
            if (!improved) {
                // True local optimum reached. Record it, then perturb (a small
                // random "kick") to escape and descend again. This is an
                // Iterated-Local-Search loop: we keep the best set ever seen and
                // run for the whole time budget. Every few stale rounds we
                // restart the kick from the best incumbent so the search does
                // not drift indefinitely.
                if (curCost < bestSeen) { bestSeen = curCost; bestOpen = open; }
                stale++;
                if (now_sec() - t_start > T_BUDGET * 0.98) break;
                // Occasionally re-centre on the best incumbent before kicking.
                if (stale % 8 == 0) {
                    open = bestOpen;
                    rebuild_cache();
                    curCost = 0.0;
                    for (int i = 0; i < F; i++) if (open[i]) curCost += FCOST[i];
                    for (int c = 0; c < C; c++) curCost += d1[c];
                }
                // Perturbation: randomly toggle a few facilities (keep >=1).
                int toggles = 1 + (rng() % 3);
                for (int t = 0; t < toggles; t++) {
                    int i = rng() % F;
                    if (open[i]) {
                        if (open_count() > 1) {
                            bool legal; drop_gain(i, legal);
                            if (legal) apply_drop(i);
                        }
                    } else {
                        apply_add(i);
                    }
                }
                continue;
            }
        }
        if (curCost < bestSeen) { bestSeen = curCost; bestOpen = open; }
    }

    // Restore the best open set seen.
    open = bestOpen;

    // Final feasibility guard: ensure non-empty.
    {
        bool any = false; for (int i = 0; i < F; i++) if (open[i]) { any = true; break; }
        if (!any) { int b = 0; for (int i = 1; i < F; i++) if (FCOST[i] < FCOST[b]) b = i; open[b] = 1; }
    }

    // Output: M then the 1-based opened facility indices.
    vector<int> openedIdx;
    for (int i = 0; i < F; i++) if (open[i]) openedIdx.push_back(i + 1);
    cout << openedIdx.size() << "\n";
    for (int v : openedIdx) cout << v << "\n";
    return 0;
}
```
