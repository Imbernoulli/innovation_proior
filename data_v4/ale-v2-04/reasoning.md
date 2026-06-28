# Reasoning: minimizing the number of colors of a proper coloring

## Reading the objective

I have a simple undirected graph and I have to color the vertices so that no edge is monochromatic,
spending as few colors as possible. The number of colors I "spend" is the count of distinct labels in
my output; the fewest achievable over all proper colorings is the chromatic number `χ(G)`. The score
is `round(1e6 · G / K)` where `K` is the number of colors I use and `G` is the color count of a
reference first-fit greedy that the judge runs itself; an improper coloring — any single edge whose two
endpoints share a color — scores `0`. So two things matter, in order: (1) never, ever emit an improper
coloring, because that floors me to zero, and (2) among proper colorings, drive `K` as low as I can.

That ordering tells me my whole design discipline up front: I must *always hold a valid proper
coloring*, and only ever replace it with another *verified* proper coloring that uses fewer colors. I
should never gamble my current valid answer on an optimization step that might leave me improper.

## A feasible baseline first

The cheapest always-proper coloring is the all-distinct one: `color[i] = i`, every vertex its own
color. No edge can be monochromatic because no two vertices share a color. It uses `K = n` colors,
which is terrible for the score (the judge's greedy uses far fewer than `n`, so `G/K` is tiny), but it
is *feasible*, and that is the point of a baseline: I now always have something legal to print. If my
optimizer produces nothing better, I still score a positive number rather than zero.

The next rung up — and a genuinely good proper coloring — is a **greedy construction**. Order the
vertices somehow and give each vertex the smallest color not used by its already-colored neighbours.
That is `O(n + m)` and always proper. The judge's *baseline* `G` is exactly this with a
descending-degree order (first-fit / largest-first). If I just reproduce that, I tie the baseline and
score exactly `1e6`. To *beat* the baseline I need fewer colors than first-fit greedy gives.

The first improvement is a smarter construction order. First-fit's weakness is that it fixes the order
before it starts and never adapts: it keeps meeting a vertex that already conflicts with every open
color and opens a fresh one. **DSATUR** fixes this by choosing the next vertex dynamically: always
color the uncolored vertex with the highest *saturation degree* — the number of distinct colors
already present among its neighbours — with ties broken by uncolored degree. Intuitively DSATUR colors
the most "constrained" vertex next, which packs colors more tightly and tends to open fewer of them.
DSATUR is the standard strong constructive heuristic for chromatic-number minimization, and on these
near-multipartite instances it should already beat the judge's first-fit greedy. So DSATUR becomes my
real working baseline: always proper, and usually a color or two below `G`.

## Why a single construction is not enough

DSATUR is one-shot. It commits to a color for each vertex in sequence and never revisits a decision. On
a graph whose chromatic number is, say, `7`, DSATUR might still land on `8` or `9` because an early
choice forces a later vertex to open a color it would not have needed under a different early choice.
To genuinely minimize colors I have to *revisit* decisions — I need local search. The question is what
the local search should optimize.

The naive idea is to do local search "on the number of colors" directly: try to merge two color
classes, recolor vertices to vacate a class, etc. This is awkward because the objective (number of
classes) is a coarse integer that barely ever changes under a single vertex recolor, so the search has
almost no gradient to follow. The established, much better idea is to attack the **decision problem**
instead:

> Fix a target `k`. Ask: does a *proper* `k`-coloring exist? Search for one by minimizing the number
> of conflicting (monochromatic) edges down to zero, allowing `k` colors. If I reach zero conflicts, I
> have a proper `k`-coloring; record it, set `k ← k-1`, and try again. When `k` becomes unreachable,
> the last proper coloring I found is my answer.

This is the classic reduction of "minimize colors" to a sequence of "`k`-colorability by
conflict-minimization" problems, and the conflict-minimization subproblem has a smooth objective (the
number of bad edges) that local search can actually descend. The canonical local search for it is
**TabuCol** (Hertz–de Werra): repeatedly recolor one *conflicting* vertex to its best alternative
color among the `k`, and make it *tabu* to put that vertex back to its old color for a number of
iterations, so the search does not immediately cycle and can climb off plateaus. So my plan is:

```
best ← DSATUR()                 # proper, k0 colors
k ← k0 - 1
loop:
    start ← best, but squeezed into k color classes (creates some conflicts)
    if TabuCol(k, start) reaches 0 conflicts:
        best ← that proper k-coloring;  k ← (#colors in best) - 1
    else:
        stop
output best
```

## The efficiency wall, and the lever that gets past it

TabuCol does an enormous number of iterations, and each iteration must pick the best recolor among the
conflicting vertices. If I scored each candidate recolor by *recomputing the whole conflict count*,
each candidate would cost `O(m)`, and with `O(n·k)` candidates per step the whole thing is
`O(n·k·m)` per step — utterly hopeless; the search would manage a handful of steps per second instead
of the millions I need.

The lever is **incremental move evaluation** via a conflict-count table. Keep

```
gamma[v][c] = number of v's neighbours currently colored c.
```

Then the number of conflicts vertex `v` currently contributes is `gamma[v][color[v]]`, and the change
in total conflicts from recoloring `v` to color `c` is exactly

```
delta = gamma[v][c] - gamma[v][color[v]]      (an O(1) lookup)
```

When I actually apply a recolor of `v` from `cv` to `nc`, only `v`'s neighbours' tables change:
`gamma[u][cv]--` and `gamma[u][nc]++` for each neighbour `u`. That is `O(degree(v))`, not `O(n)` or
`O(m)`. With this table, scanning the conflicting vertices and their `k` alternative colors to find the
best move is `O((#conflicting vertices)·k)`, and applying it is `O(degree)`. That is what makes
thousands — really millions — of tabu iterations per second feasible, and therefore makes the repeated
`k`-feasibility searches affordable inside a two-second budget. This incremental `gamma` table is the
non-obvious engineering core of the whole solver.

I also want a *dynamic* tabu tenure rather than a fixed one: the Galinier–Hao recipe sets the tenure to
a small constant plus a fraction of the current conflict count plus a little randomness, so the search
forbids more when it is deep in conflicts and relaxes as it approaches a proper coloring. And I want an
**aspiration** rule: if a tabu move would nonetheless reach a strictly better conflict count than any
seen so far, allow it anyway.

## Building it, and the first real bug

I wrote the pieces: a CSR adjacency build (flat arrays, cache-friendly), `dsatur()`, `countColors`,
`isProper`, the `tabuCol(k, color)` search with the `gamma` table, a `reduceToK` that squeezes a proper
coloring into `k` classes, and the descent loop. I compiled clean (one harmless unused-result warning
on `scanf`) and ran it on a generated seed.

The output was *proper* — good, no feasibility disaster. But the score came back at exactly
`1 000 000`, meaning `K = G`: my solver used the *same* number of colors as the judge's first-fit
greedy, no better. And it had finished in 0.41s, well under the 1.9s budget. Something was making it
give up early.

I added stderr tracing and saw:

```
DSATUR k0=15
try k=14 -> FAIL  (t=0.44)
```

DSATUR found `15`, the descent tried `k=14`, TabuCol failed to reach zero conflicts, and the loop
*stopped on the first failure*. So my `K` was just DSATUR's `15`, which happened to equal the judge's
greedy `15` on that instance — a tie, not a win. Two separate problems were hiding here.

**Problem one: I gave up after a single TabuCol attempt.** One run of TabuCol from one starting point
getting stuck does not prove `k` is infeasible — local search has bad luck. The standard fix is
*restarts*: re-squeeze into `k` colors with a *different* random reassignment and run TabuCol again,
several times, before concluding `k` is out of reach. My `reduceToK` was deterministic (always drop the
highest class the same way), so even if I had looped, every restart would have started from the same
basin. I gave `reduceToK` a `randomize` flag: the first attempt uses the least-conflict placement, and
subsequent restarts scatter the over-budget vertices into random classes, giving TabuCol genuinely
different starting points. Then I wrapped the per-`k` attempt in a loop of up to six randomized
restarts and only declared `k` unreachable if *all* of them failed. I also made the whole descent run
until the wall-clock budget, not bail after the first stall.

I rebuilt and traced again:

```
DSATUR k0=15
  k=14 restart=0 -> FAIL (t=0.41)
  k=14 restart=1 -> FAIL (t=0.85)
  ...
  k=14 restart=4 -> FAIL (t=1.90)
final score: 1000000   distinct: 15
```

Still stuck at 15, now using the full budget. So restarts alone did not help — which pointed at the
*second*, deeper problem.

**Problem two: the instances had no gap to win.** I probed: a 30-second, 200-restart version of the
solver *still* could not get below 15 on that instance. That meant `15` really was (essentially) the
chromatic number of the graph my generator was producing, and the judge's greedy *also* found 15 — so
there was simply no room to beat the baseline, no matter how strong the solver. My first generator built
graphs from planted near-cliques of size up to 16 plus a dense Erdős–Rényi background; those force the
chromatic number right up against what greedy finds, leaving no daylight between `K` and `G`. A perfect
solver and a greedy solver tie at the wall. That is a *generator* flaw, not a solver flaw: an ALE-Bench
problem is only meaningful if a strong heuristic can measurably outscore the trivial baseline.

## Fixing the generator: plant a gap

I needed instances where (a) a low-color proper coloring provably exists (a known upper bound on `χ`),
and (b) the judge's first-fit greedy *visibly wastes* colors, so a strong solver can recover the
difference. The clean construction is to **plant a proper `C`-coloring**: pick a class count `C`,
randomly partition the vertices into `C` classes, and add edges *only between different classes, never
inside a class*. Now the planted partition is proper by construction, so `χ(G) ≤ C` for free. Making the
inter-class edges dense (high pairwise probability, then thinned to a target average degree) gives a
near-complete-multipartite core — which both pins `χ` close to `C` from below and raises the degree so
that first-fit greedy, fed by descending degree, keeps opening extra colors it does not need. The class
labels are randomized so the optimizer cannot read the planted partition off the vertex indices.

I regenerated seed 1 with this generator and checked the gap directly: the judge's first-fit greedy used
`G = 10` colors, while DSATUR + tabu drove it down to `K = 7`, scoring `round(1e6·10/7) = 1 428 571` —
well above the `1e6` tie line. The descent now actually descends:
DSATUR starts around 10–12, and TabuCol successfully finds proper colorings at `k-1`, `k-2`, `k-3`
before a target finally becomes unreachable.

## Self-verification on the seed set

I compiled with `g++ -O2 -std=c++17` and ran the full fixed seed set `1 … 20`, scoring each output with
the deterministic scorer and, for comparison, the trivial all-distinct baseline (`color[i] = i`). The
results:

- **Every one of the 20 outputs is feasible** — I re-checked properness with an independent script that
  re-reads the edges and asserts no edge is monochromatic, the token count is exactly `n`, and every
  color is non-negative. All 20 pass.
- **The solver beats the judge's greedy on every seed**: `K` is consistently 2–4 colors below `G`
  (e.g. `G=10,K=7`; `G=12,K=8`; `G=11,K=7`), so every per-seed score sits between roughly `1.28e6` and
  `1.57e6`.
- **The solver's mean score (~1.41e6) crushes the trivial baseline's (~0.023e6)** — about a 60×
  margin — and also clears the `1e6` greedy-tie line on every single seed.
- **Timing** is ~1.9s per instance, fully using but not exceeding the ~2s budget; peak memory is a few
  megabytes.

I also hammered the corner cases the feasibility floor is supposed to catch: `n=0` (empty input) emits
nothing and the scorer treats it as the vacuous `1e6`; `n=1` with no edges emits one color; a single
edge gets two different colors; a triangle gets three; and deliberately *improper* outputs (both
endpoints of an edge colored the same), a wrong token count, and a negative color all score `0` exactly
as intended. So the `feasibility → 0` floor is real and the solver never trips it.

The two safety nets in the code earn their place here: I only ever adopt a `k`-coloring *after*
`tabuCol` returns true (zero conflicts), and at the very end I re-check `isProper(best)` and fall back
to a fresh DSATUR if anything is off. Combined with the all-proper construction baseline, there is no
path by which the program prints an improper coloring.

## Final solver

The final single-file C++17 program: DSATUR for a strong proper start, then color reduction by repeated
randomized-restart TabuCol descent on the target `k`, all driven by the incremental `gamma[v][c]`
conflict table that makes each move `O(1)` to evaluate and `O(degree)` to apply, under a ~1.9s
wall-clock budget, with a guaranteed-proper fallback.

```cpp
#include <bits/stdc++.h>
using namespace std;

// ------------------------------------------------------------------------
// Graph Coloring -- minimize the number of colors of a PROPER coloring.
//
// Strategy (the established strong heuristic family):
//   1. DSATUR construction -> an initial proper coloring using k0 colors.
//      This is our always-valid baseline; we never lose it.
//   2. Color reduction by TabuCol descent on the target k. To go from a proper
//      k-coloring to a proper (k-1)-coloring, we DROP the highest color class
//      (reassign its vertices into [0,k-1) classes), which creates some
//      monochromatic (conflicting) edges, then run tabu search that recolors one
//      conflicting vertex at a time to drive the number of conflicts to 0. If we
//      reach 0 conflicts we have a proper (k-1)-coloring; save it and try k-2.
//      If the budget runs out before 0, we keep the best PROPER coloring found.
//   3. Incremental gamma[v][c] table: gamma[v][c] = number of v's neighbours
//      currently colored c. The conflict delta of recoloring v from its color to
//      c is gamma[v][c] - gamma[v][color[v]] (O(1)); applying a recolor updates
//      gamma only for v's neighbours (O(degree)). This makes thousands of tabu
//      iterations per second feasible.
//
// The program ALWAYS prints a feasible (proper) coloring within the time budget:
// the DSATUR result is proper, and every k we accept is verified to have 0
// conflicts before we adopt it.
// ------------------------------------------------------------------------

static const double TIME_LIMIT = 1.9;  // seconds, wall clock
static std::chrono::steady_clock::time_point T0;
static inline double elapsed() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
}

// xorshift RNG (fast, deterministic)
static uint64_t rng_state = 88172645463325252ULL;
static inline uint64_t xrand() {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x;
    return x;
}
static inline int randint(int n) { return (int)(xrand() % (uint64_t)n); }

int n, m;
vector<int> adjStart;          // CSR adjacency
vector<int> adjList;
vector<int> deg;

// build CSR adjacency from an edge list
void buildAdj(const vector<pair<int,int>>& edges) {
    deg.assign(n, 0);
    for (auto& e : edges) { deg[e.first]++; deg[e.second]++; }
    adjStart.assign(n + 1, 0);
    for (int i = 0; i < n; i++) adjStart[i + 1] = adjStart[i] + deg[i];
    adjList.assign(adjStart[n], 0);
    vector<int> cur(adjStart.begin(), adjStart.begin() + n);
    for (auto& e : edges) {
        adjList[cur[e.first]++] = e.second;
        adjList[cur[e.second]++] = e.first;
    }
}

// ---------------- DSATUR construction ----------------
// Order: repeatedly pick the uncolored vertex of maximum saturation degree
// (number of distinct colors among its neighbours), ties broken by larger
// (uncolored) degree, then assign the smallest feasible color.
vector<int> dsatur() {
    vector<int> color(n, -1);
    vector<int> satDeg(n, 0);                 // distinct neighbour colors
    vector<int> uncDeg = deg;                 // remaining (uncolored) degree
    // neighborColorMask[v] as a set; for n up to ~600 colors <= n, use a
    // per-vertex set of used neighbour colors via a small hash set substitute:
    vector<vector<char>> used(n);             // used[v][c] = neighbour uses c
    // we size each used[v] lazily as colors grow; simpler: cap at n colors.
    for (int v = 0; v < n; v++) used[v].assign(1, 0);

    auto ensureSize = [&](int v, int c) {
        if ((int)used[v].size() <= c) used[v].resize(c + 1, 0);
    };

    for (int iter = 0; iter < n; iter++) {
        // select uncolored vertex with max (satDeg, then uncDeg, then -index)
        int best = -1;
        int bSat = -1, bDeg = -1;
        for (int v = 0; v < n; v++) {
            if (color[v] != -1) continue;
            if (satDeg[v] > bSat ||
                (satDeg[v] == bSat && uncDeg[v] > bDeg)) {
                bSat = satDeg[v]; bDeg = uncDeg[v]; best = v;
            }
        }
        if (best == -1) break;
        int v = best;
        // smallest color not used by a neighbour
        int c = 0;
        while (c < (int)used[v].size() && used[v][c]) c++;
        color[v] = c;
        // update neighbours
        for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
            int u = adjList[idx];
            if (color[u] != -1) continue;
            uncDeg[u]--;
            ensureSize(u, c);
            if (!used[u][c]) {
                used[u][c] = 1;
                satDeg[u]++;
            }
        }
    }
    return color;
}

// number of distinct colors in a coloring
int countColors(const vector<int>& color) {
    int mx = -1;
    for (int v = 0; v < n; v++) mx = max(mx, color[v]);
    if (mx < 0) return 0;
    vector<char> seen(mx + 1, 0);
    int cnt = 0;
    for (int v = 0; v < n; v++) if (!seen[color[v]]) { seen[color[v]] = 1; cnt++; }
    return cnt;
}

// is the coloring proper?
bool isProper(const vector<int>& color) {
    for (int v = 0; v < n; v++)
        for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
            int u = adjList[idx];
            if (u > v && color[u] == color[v]) return false;
        }
    return true;
}

// ---------------- TabuCol: try to find a proper k-coloring ----------------
// Start from a (possibly improper) k-coloring; minimize the number of
// conflicting edges to 0 by recoloring one conflicting vertex at a time, with a
// tabu list forbidding immediate reversal. Returns true and writes `color` if it
// reaches 0 conflicts within the iteration / time budget.
//
// gamma[v*k + c] = number of v's neighbours currently colored c.
// nConflicts     = number of monochromatic edges (each counted once).
bool tabuCol(int k, vector<int>& color, long long maxIters) {
    if (k <= 0) return false;
    vector<int> gamma((size_t)n * k, 0);
    auto G = [&](int v, int c) -> int& { return gamma[(size_t)v * k + c]; };

    // init gamma and conflict count
    for (int v = 0; v < n; v++) {
        for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
            int u = adjList[idx];
            G(v, color[u])++;
        }
    }
    long long nConflicts = 0;  // counts each conflicting endpoint pair once
    for (int v = 0; v < n; v++) nConflicts += G(v, color[v]);
    nConflicts /= 2;

    if (nConflicts == 0) return true;

    // tabuTenure[v*k + c] = iteration until which assigning color c to v is tabu
    vector<long long> tabu((size_t)n * k, -1);
    auto TB = [&](int v, int c) -> long long& { return tabu[(size_t)v * k + c]; };

    long long bestConf = nConflicts;
    long long it = 0;
    // dynamic tabu tenure: base + fraction of current conflicts (Galinier-Hao)
    while (it < maxIters && nConflicts > 0) {
        if ((it & 1023) == 0 && elapsed() > TIME_LIMIT) break;
        it++;

        // pick the move that best reduces conflicts among CONFLICTING vertices;
        // standard TabuCol: scan conflicting vertices, for each its k-1 other
        // colors, choose the best non-tabu (with aspiration) delta.
        int bestV = -1, bestC = -1;
        long long bestDelta = LLONG_MAX;
        int nTied = 0;

        for (int v = 0; v < n; v++) {
            int cv = color[v];
            int gv = G(v, cv);
            if (gv == 0) continue;            // v is not in conflict; skip
            for (int c = 0; c < k; c++) {
                if (c == cv) continue;
                long long delta = (long long)G(v, c) - gv;  // change in conflicts
                bool isTabu = (TB(v, c) >= it);
                bool aspire = (nConflicts + delta < bestConf);
                if (isTabu && !aspire) continue;
                if (delta < bestDelta) {
                    bestDelta = delta; bestV = v; bestC = c; nTied = 1;
                } else if (delta == bestDelta) {
                    // reservoir tie-break to avoid bias
                    nTied++;
                    if (randint(nTied) == 0) { bestV = v; bestC = c; }
                }
            }
        }

        if (bestV == -1) {
            // every reducing move is tabu and none aspires: make a random
            // perturbing recolor of a conflicting vertex to escape.
            // pick a random conflicting vertex
            int tries = 0, v = -1;
            while (tries < 50) {
                int cand = randint(n);
                if (G(cand, color[cand]) > 0) { v = cand; break; }
                tries++;
            }
            if (v == -1) break;  // no conflicting vertex found by sampling
            int cv = color[v];
            int c = randint(k);
            if (c == cv) c = (c + 1) % k;
            bestV = v; bestC = c;
            bestDelta = (long long)G(v, c) - G(v, cv);
        }

        // apply the recolor of bestV: cv -> bestC
        int v = bestV, cv = color[v], nc = bestC;
        nConflicts += bestDelta;
        // update gamma for neighbours
        for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
            int u = adjList[idx];
            G(u, cv)--;
            G(u, nc)++;
        }
        color[v] = nc;
        // tabu: forbid putting v back to cv for a dynamic tenure
        long long tenure = 8 + (long long)(0.6 * (double)nConflicts) + randint(10);
        TB(v, cv) = it + tenure;

        if (nConflicts < bestConf) bestConf = nConflicts;
    }

    return nConflicts == 0;
}

// build an initial (possibly improper) k-coloring from a proper coloring by
// reassigning every vertex whose color falls outside [0,k) into [0,k). With
// `randomize` false this is a deterministic least-conflict placement; with
// `randomize` true the over-budget vertices are scattered randomly, which gives
// TabuCol a genuinely different starting basin on each restart.
vector<int> reduceToK(const vector<int>& proper, int k, bool randomize) {
    // relabel proper coloring colors to a contiguous 0..K-1 first
    int mx = -1;
    for (int v = 0; v < n; v++) mx = max(mx, proper[v]);
    vector<int> remap(mx + 1, -1);
    int next = 0;
    for (int v = 0; v < n; v++) {
        if (remap[proper[v]] == -1) remap[proper[v]] = next++;
    }
    vector<int> color(n);
    for (int v = 0; v < n; v++) color[v] = remap[proper[v]];
    // any vertex whose color >= k must move into [0,k).
    for (int v = 0; v < n; v++) {
        if (color[v] < k) continue;
        if (randomize) {
            color[v] = randint(k);
        } else {
            // greedily place it in the color that adds the fewest conflicts
            // among its already-known neighbours (a light touch; tabu cleans up).
            vector<int> cnt(k, 0);
            for (int idx = adjStart[v]; idx < adjStart[v + 1]; idx++) {
                int u = adjList[idx];
                if (color[u] < k) cnt[color[u]]++;
            }
            int bc = 0;
            for (int c = 1; c < k; c++) if (cnt[c] < cnt[bc]) bc = c;
            color[v] = bc;
        }
    }
    return color;
}

int main() {
    T0 = std::chrono::steady_clock::now();

    // fast input
    if (scanf("%d %d", &n, &m) != 2) {
        // nothing to read; emit nothing (n unknown). Treat as empty.
        return 0;
    }
    vector<pair<int,int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        edges[i] = {u, v};
    }
    if (n <= 0) { return 0; }

    buildAdj(edges);

    // 1) DSATUR construction -> always-proper baseline.
    vector<int> best = dsatur();
    int bestK = countColors(best);

    // 2) Color reduction: try k = bestK-1, bestK-2, ... by TabuCol descent.
    //    We keep the best PROPER coloring found at all times. For each target k
    //    we make REPEATED randomized restarts of TabuCol until either we find a
    //    proper k-coloring or we exhaust the per-k restart budget; only after
    //    several restarts all fail do we conclude k is out of reach and stop.
    //    Each TabuCol call is bounded by a wall-clock guard, so the global
    //    TIME_LIMIT is always respected.
    long long itersPerK = 200000LL + (long long)n * 2000LL;

    int targetK = bestK - 1;
    while (targetK >= 1 && elapsed() < TIME_LIMIT) {
        bool solved = false;
        // up to a handful of randomized restarts for this target k; the first
        // attempt uses the least-conflict reduction, later ones scatter randomly.
        for (int restart = 0; restart < 6 && elapsed() < TIME_LIMIT; restart++) {
            vector<int> cand = reduceToK(best, targetK, /*randomize=*/restart > 0);
            if (tabuCol(targetK, cand, itersPerK)) {
                best = cand;
                bestK = countColors(best);   // may be < targetK if a color went unused
                targetK = bestK - 1;
                solved = true;
                break;
            }
        }
        if (!solved) {
            // none of the restarts reached a proper coloring with this k:
            // treat k as out of reach and stop descending.
            break;
        }
    }

    // 3) safety: guarantee output is proper. If for any reason `best` is not
    //    proper, fall back to a fresh DSATUR (always proper).
    if (!isProper(best)) {
        best = dsatur();
    }

    // output: one color per line, colors relabeled to 0..K-1 (cosmetic).
    int mx = -1;
    for (int v = 0; v < n; v++) mx = max(mx, best[v]);
    vector<int> remap(mx + 1, -1);
    int next = 0;
    for (int v = 0; v < n; v++) if (remap[best[v]] == -1) remap[best[v]] = next++;

    string out;
    out.reserve((size_t)n * 4);
    for (int v = 0; v < n; v++) {
        out += to_string(remap[best[v]]);
        out += '\n';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
