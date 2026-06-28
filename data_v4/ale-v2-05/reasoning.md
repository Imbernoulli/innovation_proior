# Reasoning: Facility Location with Opening Cost

## Reading the objective carefully

I am given `F` candidate facility sites (each with coordinates and a fixed
opening cost `open[i]`) and `C` clients (coordinates). I pick a non-empty subset
`S` of sites to open. Every client is then served by its nearest open facility,
and I pay the Euclidean distance for that. The total I minimize is

```
total = sum_{i in S} open[i]  +  sum_c min_{i in S} dist(client c, facility i).
```

Two terms pull against each other. The first term (opening cost) wants me to
open as few facilities as possible. The second term (service cost) wants a
facility close to every client, which means opening *more* of them. So this is
not "pick the K nearest" — there is no K. The number of facilities I open is
itself a decision, and the right number is wherever the marginal opening cost of
the next facility equals the service distance it saves. That is exactly the
Uncapacitated Facility Location Problem (UFLP), a textbook NP-hard problem. No
exact polynomial algorithm; I will be judged on a continuous cost, so I want a
genuinely strong heuristic, not a toy.

Before anything clever, I write down the score wrapper so I know what "good"
means: `score = round(10^9 / (1 + total/C))` when feasible, and `0` when not.
Feasibility is: `1 <= M <= F`, all indices in range, all distinct. The `M >= 1`
part matters — if I ever print an empty set the score floors to 0. So rule one
of the whole exercise: **always hold a non-empty open set and never print
garbage.** Lower `total` is strictly better; the wrapper is monotone.

## A feasible baseline first

I refuse to write any search before I have a guaranteed-legal answer in hand.
The simplest legal answer is "open every facility": `M = F`, indices `1..F`, all
distinct, non-empty — always feasible. Its cost is `sum_i open[i] + sum_c
(distance to the globally nearest facility)`. The service term is as small as it
can ever be (every facility is available), but I pay *every* opening cost, which
is enormous. The opposite trivial answer, "open only the single cheapest
facility", pays almost no opening cost but routes all `C` clients to one depot,
so the service term explodes. Both are feasible, both are bad, and the optimum
sits between them. I will use "open all" as the official trivial baseline to
beat, because it is the one a careless solver would emit, and because it is
unambiguous and deterministic.

This already tells me the shape of the answer: I need to *select an intermediate
subset*, trading the two terms. Good — that's a combinatorial optimization over
subsets of `F` items, `F` up to 120, with a service term that couples all the
clients to the chosen subset.

## The obvious local search, and why the naive version is too slow

The standard, well-understood metaheuristic for UFLP is local search over three
moves:

- **ADD** a currently-closed facility `i`. Some clients are now closer to `i`
  than to their current nearest; the service term drops by their savings, and I
  pay `open[i]`. Net gain = `open[i] - sum_c max(0, d1[c] - dist(c,i))`.
- **DROP** a currently-open facility `i`. Every client whose nearest open
  facility *was* `i` must fall back to its next-nearest open facility; the
  service term rises, and I save `open[i]`.
- **SWAP**: drop one and add another in the same step (escapes optima that
  neither pure move can).

The gain of an ADD is easy: for each client I compare its current nearest
distance `d1[c]` to `dist(c,i)` and take the improvement — `O(C)`, fine. The
trouble is the DROP. If I drop `i`, the clients that were served by `i` need
their *new* nearest among the remaining open facilities. The naive way is, for
each such client, to scan all the still-open facilities and find the minimum:
that is `O(|S|)` per affected client, so `O(C·|S|)` for one candidate drop, and
sweeping all `|S|` open facilities as drop candidates costs `O(C·|S|^2)` per
pass. With `C` up to 700 and `|S|` in the tens, and wanting thousands of moves
inside a 2-second budget, this is the bottleneck. The DROP evaluation is what
makes the obvious local search too slow.

I briefly considered just doing ADD-only hill-climbing to dodge the expensive
drop. But add-only gets stuck: it never reconsiders a facility it opened early
that later became redundant, and it can't reduce an over-opened set. The whole
point of UFLP is that you sometimes need to *close* a depot you opened. I need
DROP to be cheap, not absent.

## The innovation: cache the second-nearest open facility

Here is the lever. The reason the drop is expensive is that I throw away, after
every assignment, the information about *who the runner-up was*. If, for every
client `c`, I also remember its **second-nearest open facility** `near2[c]` and
that distance `d2[c]` — not just the nearest `near1[c]`, `d1[c]` — then dropping
`near1[c]` is trivial: the client simply falls to `d2[c]`. So:

```
DROP gain(i) = open[i] - sum_{c : near1[c]==i} (d2[c] - d1[c]).
```

That is a single `O(C)` pass — I touch each client once, and only the clients
whose nearest was `i` contribute. No inner scan over `|S|`. The second-nearest
cache collapses `O(C·|S|)` per drop to `O(C)`. This is the established
Whitaker/Resende–Werneck "neighbor-list" trick for p-median and facility
location, and it is exactly the lever the problem is built around.

The ADD also fits the cache: when I add `i`, a client either finds `i` closer
than its current nearest (then `i` becomes the new nearest, the old nearest
becomes the new second-nearest) or finds `i` between its first and second (then
`i` becomes the new second-nearest) or neither (no change). All `O(1)` per
client given the cache, `O(C)` per add, and the cache stays exactly correct.

The one case the cache does *not* fully cover for free is what happens to
`near2` after a DROP for the affected clients: once a client's `near1` (which was
`i`) is gone and it has promoted `d2` to be its new `d1`, it needs a *fresh*
second-nearest. The same is true for clients whose `near2` was `i`. For those
clients I do one scan over the open set to refind a second-nearest. That scan is
`O(|S|)` per affected client, but only the clients touching `i` are affected, so
in aggregate a drop's *apply* step is cheap in practice and, crucially, the drop
*evaluation* (which I do for all candidates every pass) is the pure `O(C)` form.
That asymmetry — cheap evaluation for the sweep, slightly heavier apply only for
the one move I actually commit — is the right trade.

## Getting a strong starting set: LP / Lagrangian relaxation rounding

Local search is only as good as where it starts. Starting from "open all" or
"open one" wastes a lot of moves climbing out of a bad basin. UFLP has a strong
LP relaxation, and the cleanest way to exploit it in a single self-contained C++
file (no LP library) is the **Lagrangian relaxation** of the assignment-equality
constraints. In the standard UFLP integer program each client must be assigned
to exactly one open facility; I relax those equalities with multipliers `u_c`
(one per client, interpretable as "the budget client `c` is willing to pay").
For fixed `u`, the relaxed problem separates over facilities: facility `i`'s
reduced cost is

```
rc_i = open[i] + sum_c min(0, dist(c,i) - u_c),
```

i.e. open `i` and let it serve every client for whom `dist(c,i) < u_c` (those
contribute a negative term). The Lagrangian lower bound is `sum_c u_c + sum_i
min(0, rc_i)`, and I open exactly the facilities with `rc_i < 0`. A subgradient
ascent on `u` maximizes that bound: the subgradient on `u_c` is
`1 - (#open facilities that want to serve c)`, so I nudge `u_c` up when no opened
facility covers `c` and down when several do. The open-indicator at a good dual
point is a principled rounding — far better than the trivial extremes — and I
keep the best *feasible* rounded set (evaluated at its true cost) seen during the
ascent. I give this stage a fraction (~45%) of the budget and hand its result to
the local search.

I keep the subgradient deliberately simple and well-behaved (a halving step rule
on stagnation, multipliers clamped at 0, capped iterations) because I do not need
the tightest dual — I need a good *integer* starting set, and the local search
will finish the job. If for any reason the rounded set comes out empty, I force
the single cheapest facility open so the pipeline always has a feasible incumbent.

## Implementing and the first run

I lay out the pieces: read the instance; precompute the full distance matrix
`dmat[i*C + c]` (at most `120*700 = 84000` doubles, trivial memory, and it kills
repeated `sqrt`); run the Lagrangian rounding; build the `d1/near1/d2/near2`
cache; then the ADD/DROP/SWAP local search with the cache, tracking `curCost`
incrementally; finally restore the best set ever seen and print it.

I compile and run on seeds 1..20, scoring each output and comparing to the
"open all" baseline. Every output is feasible (all distinct, in range,
non-empty), and the solver beats the baseline on every seed — mean score about
19.3k vs the baseline's 14.1k, a ratio around 1.36. Good: feasibility holds and
I clear the bar.

## A self-verify episode: is the incremental cost actually correct?

The thing I trust least is the incremental `curCost`. Every ADD/DROP touches it,
and the second-nearest cache is exactly where an off-by-one or a stale `d2` would
silently corrupt the tracked cost — the solver would then "improve" toward a
number that does not match what the judge computes, and I would unknowingly print
a worse set than I think. A wrong cache does not crash; it just quietly degrades
the score, which is the worst kind of bug here.

So I instrument a debug build that prints the solver's internal `bestSeen` to
stderr, and I compare it against the *independent* scorer's `--cost` on the
printed set, on several seeds. They match to the last decimal on every seed I
check (e.g. `26060563.3669` internal vs `26060563.3669` from the scorer). That
confirms two things at once: the incremental ADD/DROP cost updates are exactly
consistent with a from-scratch recomputation, and the `d2` promotion logic in
the DROP apply (promote `d2` to `d1`, then rescan for a fresh `d2`) is correct.
If those had been off, the numbers would have diverged. This is the check that
actually matters for this problem, more than any feasibility check — feasibility
is easy to guarantee structurally, but cost-cache correctness is the thing that
makes the *quality* real.

I also exercise the degenerate guards directly: `F=1, C=1` prints `1` then the
one facility (legal); a `C=0` instance opens the single cheapest facility (still
`M>=1`, legal); empty stdin returns cleanly. None crash, all feasible.

## A second issue: the search was finishing in 0.09s

The first timed run revealed the solver converging to a local optimum and then
exiting almost immediately — about 0.09 seconds of a 1.9-second budget. The
culprit was an over-eager stop: after reaching a local optimum I perturbed a few
times and then bailed out after a fixed small number of stale rounds. That left
most of the budget unused. The fix is to make it a proper **Iterated Local
Search**: at each local optimum, record the best set, apply a small random
"kick" (toggle a few facilities, never going empty), descend again, and keep
going for the *entire* budget; every several stale rounds, re-centre the kick on
the best incumbent so the walk does not drift off to nowhere. After this change
the solver uses the full ~1.86s and the mean score ticks up slightly (the
LP-rounding plus first-improvement basin was already strong, so the extra time
is polish rather than a leap — which is itself reassuring evidence the start was
good). Feasibility and the baseline win both still hold across all 20 seeds.

## Why this is the strongest practical approach here

The combination is the current best-known practical recipe for UFLP at this
scale: a Lagrangian/LP relaxation for a principled rounded start, then
ADD/DROP/SWAP local search whose moves are made cheap by the first/second-nearest
neighbor cache (the `O(C)` drop), wrapped in iterated local search to spend the
budget. The cache is the load-bearing idea — without it, the drop sweep is
`O(C·|S|^2)` per pass and the search would be an order of magnitude slower,
exploring far fewer sets in the same 2 seconds. With it, I can re-evaluate every
candidate add and drop every pass and still run thousands of moves, which is what
turns "an ok local optimum" into "a strong one".

## Final solver

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
