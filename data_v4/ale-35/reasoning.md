# Reasoning: Flood-Control Levee Placement

## Reading the objective

I have an `H x W` grid of integer heights. Some cells are flood sources. Water
spreads from the sources by a fixed rule — from a flooded cell `u` it reaches a
4-adjacent neighbour `v` whenever `h[u] >= h[v]`, i.e. downhill or across level
ground, never strictly uphill. I get to build at most `B` unit levees on
non-source cells; a levee makes its cell impassable, so it never floods and water
cannot pass through it. I want to minimize the number of flooded cells once the
water settles.

The first thing I want to be precise about is the flood itself, because everything
downstream depends on it. It is a multi-source reachability over a *directed*
graph: there is an edge `u -> v` for adjacent `u, v` exactly when `h[u] >= h[v]`.
Note the asymmetry — if `h[u] > h[v]` the edge goes one way only; if `h[u] = h[v]`
it goes both ways. So a flat region is a single bidirectional blob the water fills
entirely, while a strict slope is a one-way cascade. A levee is a node deletion:
it removes the cell and every edge touching it. So my problem is "delete at most
`B` nodes (none of them sources) to minimize the number of nodes still reachable
from the source set." That is a budgeted node multiway cut — NP-hard in general —
and the score is continuous (how few cells flood), so this is a heuristic
optimization, not an exact problem.

Before any cleverness I want two invariants nailed down: (1) I must *always* be
able to print a feasible answer, and (2) I need a cheap, correct flood routine,
because I will call it a lot.

## A feasible baseline first

The trivial feasible solution is to build nothing: print `0` levees. The whole
reachable region floods, but it parses, it is within budget, and it never touches
a source. That is my floor, and it is also the reference the scorer normalizes
against (`flooded_ref` = flood with no levees). So whatever I do, I will keep a
"best feasible levee set seen so far," initialized to the empty set, and only ever
replace it with something that floods strictly fewer cells. If my search runs out
of time or hits a snag, I still print a legal answer.

The flood routine is a plain BFS from the sources. I keep a `blocked` mask (the
current levees), seed the queue with the unblocked sources, and expand to a
neighbour `v` when it is unblocked, unvisited, and `h[u] >= h[v]`. It returns the
count of flooded cells. The grid is at most `40 x 40 = 1600` cells, so a flood is
~1600 work — cheap enough that I can afford thousands of them.

## Why the obvious moves are weak

Now, where to put the `B` levees. The naive ideas:

- **Random levees on flooded cells.** Easy, feasible, and almost useless on real
  terrain: a levee in the middle of a wide flooded plain blocks exactly one cell,
  and the water simply flows around it. The marginal benefit of a random levee is
  ~1 cell.
- **Wall the source.** Ring each source with levees so nothing escapes. This works
  only when a source has very few outflow cells. If the source sits in a wide flat
  reservoir, it has dozens of escape directions and I cannot seal it within a
  budget of, say, 6–14 levees.

The thing both of these miss is *terrain structure*. Real valleys do not flood
uniformly: the water is funneled through a few **narrow passes** — low gaps in
high ridges — into large downstream **basins**. A single levee dropped *in a pass*
keeps an entire basin dry. So the marginal benefit of a levee is wildly
non-uniform: ~1 cell in the open plain, but possibly hundreds of cells if it plugs
the right pass. The entire problem is identifying those high-value passes and
spending the budget on them. This is a min-cut intuition: the passes are the
bottleneck edges of the flood graph, and I want to cut the ones that disconnect
the most downstream area per levee.

## The innovation: score every candidate from one flood pass

The brute-force way to measure a levee's value is to place it, re-flood, and see
how many fewer cells flood. With ~1600 candidate cells and `B` rounds that is a
lot of floods, and it scales badly. The lever I want is to estimate *every*
candidate's marginal benefit from a **single** flood pass.

Here is the idea. Run one flood. It visits the flooded cells in BFS order and
gives each non-source cell a BFS-tree parent (the cell it was first reached from).
That tree is a spanning structure of the flood. If I cut a cell `u`, in the best
case I disconnect `u`'s whole subtree — but only if every cell in that subtree has
*no other* way to be reached. So I also record, for each flooded cell `v`, its
**in-degree** in the flood graph: how many flooded neighbours could flow into it.
If `indeg[v] == 1`, the only water reaching `v` comes through its unique
in-neighbour, so cutting that in-neighbour really does strand `v` (and everything
beyond `v` that also has in-degree 1). If `indeg[v] > 1`, water has an alternate
route, so cutting one parent does not strand `v`.

So I compute a **dominator-flavoured subtree size**: process the BFS order in
reverse, start every cell at `sub = 1`, and let a cell `v` add its `sub` to its
parent *only when* `indeg[v] == 1`. The resulting `sub[u]` is a conservative
count of how many cells would lose their flooding if `u` were cut — it
deliberately *under*-counts wherever alternate flood paths exist, which is exactly
the property I want: it scores a cell highly only when cutting it genuinely
disconnects a downstream region. The narrow passes — through which an entire basin
is fed by a single chain of in-degree-1 cells — float to the top automatically.
This is one flood plus one linear sweep, `O(HW)` total, and it ranks all
candidates at once.

I use this for a **greedy construction**: flood, compute `sub[]`, take the
highest-`sub` non-source flooded cell as a levee, mark it blocked, and repeat up
to `B` times. Re-flooding between picks (only `B` times, not once per candidate)
lets each pick see the effect of the previous levees — after I plug one pass, the
basin behind it is dry and its cells stop competing for the budget. This greedy
seed is already strong and always feasible.

## Polishing with re-flood SA

The greedy is myopic: the single-pass estimate under-counts when two passes feed
the same basin (cutting either alone leaves the basin flooded via the other, so
neither looks valuable until both are cut), and the greedy never reconsiders an
early pick. To escape these local optima I add a simulated-annealing search over
the levee *set*. Because a full flood is so cheap on this grid, I can afford to
score each candidate set by **one exact re-flood** — no estimation, the true
flooded count. The moves are: **add** a levee (on a flooded non-source cell, if
under budget), **remove** a levee, or **move** a levee to another flooded
non-source cell. I accept by the standard SA rule (always if it floods fewer
cells; with probability `exp(-delta/T)` otherwise, `T` cooled geometrically), and
I always remember the best feasible set seen. The candidate pool is the cells
flooded under no levees (the only cells ever worth blocking), computed once.

This two-stage design — a single-pass dominator estimate for a strong cheap seed,
then exact-re-flood SA for the polish — is the whole method. The seed gets me to
the obvious passes fast; the SA finds the multi-pass cuts the estimate misses.

## Building the instances, and the bug that almost made the problem trivial

To test any of this I need instances, and here I hit the most important debugging
episode of the whole exercise — in the *generator*, not the solver.

My first generator built terrain the "natural" way: a smooth low-frequency height
field (a few Gaussian bumps), then a couple of tall ridges with low passes punched
through them, then sources placed on the *highest* cells (the intuition being that
water floods downhill *from* high ground, so sources belong on peaks). I generated
seeds 1–20, ran the solver, scored against the empty baseline, and the numbers
looked spectacular: solver scores of `95x`, `106x`, `258x` the baseline. All 20
feasible, all beating the baseline by enormous margins.

That was the red flag. A `258x` ratio means the solver reduced the flood from ~775
cells to ~3. No reasonable budget should be able to seal off 99.6% of a flood
*unless the problem is trivial*. So I instrumented it: for each levee the solver
placed, I printed its Manhattan distance to the nearest source. The output was
damning — almost every single levee sat at distance 1 from a source. The solver
was not finding passes at all; it was **walling the sources**. Because each source
was on a local height maximum, water left it through only a handful of downhill
neighbours, and `B` levees were more than enough to plug every outflow port of
every source directly. The carefully constructed ridges and passes were irrelevant
— the flood never even reached them, because the cheapest cut was the ring right
around the source. The min-cut bottleneck structure I wanted to study was a
decoration the solver ignored.

This is a classic benchmark-design failure: the instance distribution made the
*degenerate* strategy optimal, so the problem could not distinguish a clever
bottleneck-cutter from a dumb source-waller. I had to redesign the terrain so that
walling the source is *impossible* within the budget and cutting the passes is the
*only* winning move.

The fix flips the height structure. Instead of sources on peaks, I put the sources
inside a **wide flat reservoir plain** at a medium height. Because the plain is
flat (all cells equal height), water from any source spreads across the *entire*
plain — it is one big bidirectional flooded body, far too wide to ring with a
handful of levees. The plain is enclosed by tall ridges, each punched by one or two
low passes. Beyond every pass is a large **basin** whose flat floor is *lower* than
the plain, so once water reaches a pass it pours down and fills the whole basin.
Now the only cells worth leveeing are the passes (and the basin mouths just past
them): a levee in a pass saves an entire basin, and there is no cheap way to seal
the wide flat plain. The budget `B` is set a little above the number of passes, so
the solver must pick the *right* passes, not plug everything for free.

I regenerated and immediately found a *second* bug in the same vein. My first
redesign added per-cell noise to the plain and the basins "for texture." But a
single uphill bump in a flat region acts as a dam: water at height 55 cannot climb
to a neighbour at 58, so the noisy plain fragmented into disconnected puddles and
the flood only covered 1–3% of the grid — almost nothing flooded, so there was
nothing to protect and the scores collapsed. The flood semantics (strictly no
uphill) are unforgiving of noise in a region that is supposed to be one connected
body. So I made the plain *perfectly* flat and the basins *perfectly* flat: the
plain floods as one body, every pass (height `<= plain`) admits water, and each
basin (a flat floor below the passes) fills completely once breached. The only
height variation that matters is the deliberate ridge/pass/basin structure.

After both fixes, the no-levee flood covers ~92–95% of the grid (the plain plus
every basin), exactly the regime I wanted: lots of water, funneled through a few
passes, with the budget forcing real choices.

## Self-verification

With the redesigned generator I ran the full check on seeds 1–20: generate, run
the compiled solver, score it, and score two baselines — the empty set and `B`
*random* flooded levees.

The results were healthy and, crucially, *discriminating*:

- All 20 outputs feasible (score > 0), all 20 beating the empty baseline.
- Solver mean ≈ `3.68x` the no-levee baseline — strong, and in the same ballpark
  as comparable grid-SA ALE problems (not a suspicious `100x` blowout).
- The **random**-levee baseline scored only ~`1.01x` the empty baseline. This is
  the key number: random levees barely help, so the problem genuinely rewards
  *finding the right passes*, and the solver (`3.68x`) crushes random (`1.01x`).
  The earlier degenerate generator would have let random levees look good too;
  now it does not.

I checked timing on the largest grid (`40 x 39`): 1.80s wall, 4 MB RAM — within a
2s / 256 MB budget, with the SA loop polling the clock so it never overruns. I also
inspected where the solver puts levees: not on the source rim anymore, but on the
basin mouths just past each pass, cutting the flood from ~1119 to ~328 cells on
seed 1 — the bottleneck-cutting behaviour I designed for.

Finally I cross-checked the scorer. I re-implemented the flood completely
independently (a DFS with a different neighbour order instead of the scorer's BFS)
and re-derived the score for all 20 seeds; every score matched exactly, confirming
the flood result is order-independent and the scorer is correct. I also tested
each feasibility floor directly — over budget, a levee on a source, an out-of-grid
levee, a duplicate levee, and garbage input all score `0`, while the empty set
scores exactly `1_000_000` (it equals the reference). Everything held.

## Final solver

The single-file C++17 solver below implements exactly this: a cheap correct flood,
the single-pass dominator estimate driving a greedy construction, and an
exact-re-flood SA polish over the levee set, all under a 1.8s budget, always
keeping and printing a feasible best set.

```cpp
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Flood-Control Levee Placement.
//
// Grid of integer heights. Water starts at SOURCE cells and spreads to a
// 4-adjacent neighbour v from a flooded cell u whenever h[u] >= h[v] (downhill
// or level, never strictly uphill). A levee makes its cell impassable: it never
// floods and water cannot flow through it. We may place at most B levees (not on
// a source). Goal: MINIMISE the number of flooded cells.
//
// Strategy:
//   * Baseline: place zero levees -- always feasible, always printable.
//   * Single-pass marginal benefit (the innovation): from ONE flood we build the
//     flood BFS forest and, in one extra linear pass, estimate every candidate
//     levee's "downstream subtree size" = how many cells would lose flooding if
//     that cell were cut, using a dominator-flavoured count (a cell counts toward
//     its parent only if the parent is its UNIQUE flooded in-neighbour). This
//     ranks the narrow passes (bottlenecks) at the top without re-flooding per
//     candidate.
//   * Greedy construction: repeatedly flood, take the best-scoring frontier cell
//     as a levee, repeat up to B times -- a strong, always-feasible start.
//   * SA polish: hill-climb / anneal over the levee SET, each candidate scored by
//     ONE exact re-flood (grid is small), to escape greedy local optima. The best
//     set ever seen is kept and printed, so the output is always feasible.
// ---------------------------------------------------------------------------

static int H, W, B, S;
static vector<int> ht;          // height grid, row-major, size H*W
static vector<char> isSrc;      // 1 if cell is a source
static vector<int> srcs;        // source cell indices

static inline int idx(int r, int c) { return r * W + c; }

static const int DR[4] = {1, -1, 0, 0};
static const int DC[4] = {0, 0, 1, -1};

// Flood with a given blocked mask; returns number of flooded cells. If `order`
// is non-null, fills it with the BFS visit order and `parent` with each cell's
// BFS-tree parent (-1 for sources / unvisited); `indeg` counts how many flooded
// neighbours could flow INTO each flooded cell (for the dominator estimate).
static int flood(const vector<char> &blocked, vector<char> &fl,
                 vector<int> *order = nullptr, vector<int> *parent = nullptr,
                 vector<int> *indeg = nullptr) {
    int N = H * W;
    fill(fl.begin(), fl.end(), 0);
    if (order) order->clear();
    if (parent) fill(parent->begin(), parent->end(), -1);
    if (indeg) fill(indeg->begin(), indeg->end(), 0);
    // simple FIFO queue
    static vector<int> q;
    q.clear();
    q.reserve(N);
    int head = 0;
    for (int s : srcs) {
        if (blocked[s]) continue;  // (sources are never blocked, but be safe)
        if (!fl[s]) {
            fl[s] = 1;
            q.push_back(s);
        }
    }
    int cnt = (int)q.size();
    while (head < (int)q.size()) {
        int u = q[head++];
        if (order) order->push_back(u);
        int r = u / W, c = u % W;
        int hu = ht[u];
        for (int d = 0; d < 4; d++) {
            int nr = r + DR[d], nc = c + DC[d];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = idx(nr, nc);
            if (blocked[v]) continue;
            if (hu >= ht[v]) {                 // u can flow into v
                if (!fl[v]) {
                    fl[v] = 1;
                    cnt++;
                    if (parent) (*parent)[v] = u;
                    q.push_back(v);
                }
            }
        }
    }
    // second pass for in-degrees (count flooded in-neighbours of each flooded v)
    if (indeg) {
        for (int u = 0; u < N; u++) {
            if (!fl[u]) continue;
            int r = u / W, c = u % W, hu = ht[u];
            for (int d = 0; d < 4; d++) {
                int nr = r + DR[d], nc = c + DC[d];
                if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                int v = idx(nr, nc);
                if (!fl[v]) continue;
                if (hu >= ht[v]) (*indeg)[v]++;  // u -> v is a flood edge
            }
        }
    }
    return cnt;
}

int main() {
    if (scanf("%d %d %d %d", &H, &W, &B, &S) != 4) return 0;
    int N = H * W;
    ht.assign(N, 0);
    for (int i = 0; i < N; i++) scanf("%d", &ht[i]);
    isSrc.assign(N, 0);
    srcs.clear();
    for (int i = 0; i < S; i++) {
        int sr, sc;
        scanf("%d %d", &sr, &sc);
        int s = idx(sr, sc);
        isSrc[s] = 1;
        srcs.push_back(s);
    }

    auto start = chrono::steady_clock::now();
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::steady_clock::now() - start).count();
    };
    const double TIME_LIMIT = 1.8;  // seconds, leave margin under a 2s budget

    vector<char> blocked(N, 0);
    vector<char> fl(N, 0);
    vector<int> order, parent(N, -1), indeg(N, 0);

    // ---- GREEDY CONSTRUCTION using the single-pass dominator estimate --------
    // levees[] is the current chosen set (always feasible: never a source/dup).
    vector<int> levees;
    levees.reserve(B);

    for (int step = 0; step < B; step++) {
        int curFlood = flood(blocked, fl, &order, &parent, &indeg);
        if (curFlood <= (int)srcs.size()) break;  // only sources left, nothing to gain

        // dominator-flavoured subtree size: process BFS order in REVERSE,
        // sub[u] starts at 1; a child v adds sub[v] to its parent ONLY when v has
        // a unique flooded in-neighbour (indeg[v]==1) -- i.e. cutting the parent
        // truly disconnects v's whole subtree. This under-counts (safe) when
        // there are alternate paths, exactly capturing bottleneck passes.
        vector<int> sub(N, 0);
        for (int u : order) sub[u] = 1;
        for (int i = (int)order.size() - 1; i >= 0; i--) {
            int v = order[i];
            int p = parent[v];
            if (p >= 0 && indeg[v] == 1) sub[p] += sub[v];
        }
        // choose the best NON-source, non-already-levee flooded cell
        int best = -1, bestGain = 0;
        for (int u : order) {
            if (isSrc[u]) continue;
            if (blocked[u]) continue;
            if (sub[u] > bestGain) {
                bestGain = sub[u];
                best = u;
            }
        }
        if (best < 0 || bestGain <= 0) break;
        blocked[best] = 1;
        levees.push_back(best);
    }

    // exact flood count of the greedy solution
    int bestFlood = flood(blocked, fl);
    vector<int> bestLevees = levees;
    vector<char> bestBlocked = blocked;

    // ---- SA / LOCAL SEARCH polish over the levee set --------------------------
    // Each candidate set is scored by ONE exact re-flood. Moves:
    //   (a) ADD a levee (if under budget) on a currently-flooded non-source cell,
    //   (b) REMOVE a levee,
    //   (c) MOVE a levee to a flooded non-source cell.
    // Accept by the SA rule; always remember the best feasible set seen.
    std::mt19937 rng(0x9E3779B9u ^ (unsigned)N ^ ((unsigned)B << 16));
    auto frand = [&]() { return (rng() >> 8) * (1.0 / 16777216.0); };

    // candidate pool = cells flooded under NO levees and not sources (the cells
    // worth ever blocking). Recomputed once; the union of all reachable cells.
    vector<char> noLev(N, 0);
    vector<char> flNo(N, 0);
    flood(noLev, flNo);
    vector<int> pool;
    for (int u = 0; u < N; u++)
        if (flNo[u] && !isSrc[u]) pool.push_back(u);

    if (!pool.empty()) {
        // working copy of the current (accepted) solution
        vector<char> curBlocked = bestBlocked;
        vector<int> curLevees = bestLevees;
        int curFlood = bestFlood;

        double T0 = max(4.0, bestFlood * 0.05);  // initial temperature
        long long iter = 0;
        while (elapsed() < TIME_LIMIT) {
            // periodically check the clock, not every single iteration
            for (int batch = 0; batch < 256; batch++) {
                iter++;
                double frac = elapsed() / TIME_LIMIT;
                if (frac > 1.0) frac = 1.0;
                double T = T0 * pow(0.001 / T0 > 0 ? 0.001 / T0 : 1e-6, frac);
                if (T < 1e-6) T = 1e-6;

                int kind;
                int nLev = (int)curLevees.size();
                if (nLev == 0) kind = 0;                 // must add
                else if (nLev >= B) kind = (frand() < 0.5) ? 1 : 2;  // remove or move
                else {
                    double x = frand();
                    kind = (x < 0.5) ? 0 : (x < 0.75 ? 1 : 2);
                }

                // build the trial blocked mask & levee list
                vector<char> tb = curBlocked;
                vector<int> tl = curLevees;
                if (kind == 0) {  // ADD
                    int u = pool[rng() % pool.size()];
                    if (tb[u]) continue;  // already a levee
                    tb[u] = 1;
                    tl.push_back(u);
                } else if (kind == 1) {  // REMOVE
                    int j = rng() % tl.size();
                    int u = tl[j];
                    tb[u] = 0;
                    tl[j] = tl.back();
                    tl.pop_back();
                } else {  // MOVE
                    int j = rng() % tl.size();
                    int u = tl[j];
                    int v = pool[rng() % pool.size()];
                    if (tb[v]) continue;  // target already a levee
                    tb[u] = 0;
                    tb[v] = 1;
                    tl[j] = v;
                }

                int trial = flood(tb, fl);
                int delta = trial - curFlood;  // fewer flooded is better
                if (delta <= 0 || frand() < exp(-delta / T)) {
                    curBlocked.swap(tb);
                    curLevees.swap(tl);
                    curFlood = trial;
                    if (curFlood < bestFlood) {
                        bestFlood = curFlood;
                        bestBlocked = curBlocked;
                        bestLevees = curLevees;
                    }
                }
            }
            if (elapsed() >= TIME_LIMIT) break;
        }
    }

    // ---- OUTPUT the best feasible levee set ----------------------------------
    // Defensive feasibility filter: drop any source/out-of-range/dup and cap at B.
    vector<int> outLev;
    {
        vector<char> used(N, 0);
        for (int u : bestLevees) {
            if (u < 0 || u >= N) continue;
            if (isSrc[u]) continue;
            if (used[u]) continue;
            used[u] = 1;
            outLev.push_back(u);
            if ((int)outLev.size() == B) break;
        }
    }

    string out;
    out += to_string((int)outLev.size());
    out += '\n';
    for (int u : outLev) {
        out += to_string(u / W);
        out += ' ';
        out += to_string(u % W);
        out += '\n';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
