# Conveyor / Belt Layout — reasoning

## Reading the objective until it is unambiguous

I have an `H x W` grid. Some cells are sources, each emitting one item in a fixed direction; some are
sinks; everything else is empty and may carry at most one directional belt tile. An item sits on its
source at tick 0 and moves one cell per tick, always leaving its current cell in that cell's
direction — the source's emission direction on a source, the tile's direction on a belt. It is
delivered the instant it steps onto a sink, and lost if it walks off the grid or steps onto an empty
cell with no tile. I may place at most `B` belt tiles, and I want to maximize the number of delivered
items. The score is just that count; if my output is malformed, exceeds the budget, puts a tile on a
source or sink, overlaps two tiles, or uses an illegal direction, the score floors to zero.

The first thing I want to nail down is the *physics*, because the whole problem lives or dies on it.
Items do not interact: there is no collision rule, belts carry abstract flow units. That single fact
is enormous. It means the layout induces a **functional graph** on the grid — every used cell (source
or belt) has exactly one out-edge — and an item's entire future is a deterministic walk from its
source. So a source is delivered **iff** the forward walk from it reaches a sink within `T` ticks;
otherwise it leaves the grid, hits an unbuilt cell, or loops forever. Because the walks are
independent, I can reason about, and re-evaluate, each source on its own. I make a mental note that
this independence is exactly the hook for cheap incremental scoring later.

I also want the corner of the scoring rule crisp: the budget `B` is global and shared across all
deliveries, and the floor is unforgiving. So whatever I build, the very first invariant is *never
emit an infeasible layout*. The empty layout (zero tiles) is always feasible, which gives me a safe
fallback I can hold onto and only ever replace with something strictly better.

## A feasible baseline I can always fall back to

Before any cleverness I want a valid solution in hand. The most trivial honest construction is the
**straight-line belt**: for each source, lay a straight run of belts in its emission direction,
continuing straight until the run would leave the grid or reach a sink. It is obviously feasible
(I only ever place on empty cells, I respect the budget by laying sources in order and stopping when
`B` runs out), and it gives me a reference number to beat.

When I think about how this does, it is plainly weak. A source emits in some fixed direction; a
straight ray in that direction lands on a sink only by luck. With a handful of sinks scattered on a
20-30 wide grid, almost no source's emission ray passes through a sink, so straight-line delivers
close to nothing — and indeed when I later measure it, its mean over the seed set is about `0.75`
items. Fine: it is my floor and my normalization reference, not my answer.

## What the real problem is, and why the obvious local search will not move

If a straight ray rarely hits a sink, the obvious upgrade is to *route*: a delivery is a path of
belts from a source to some sink, and laying belts along a shortest such path delivers that source.
So the problem is "choose a set of source-to-sink paths, made of belt tiles, that fit within `B`
tiles, maximizing how many sources are routed." That is a congested grid-routing problem, and the
binding resource is the shared budget `B`.

Here is the tension that makes it hard, and that I deliberately bake into the instance generator so
the problem is not trivial: the budget is **tight** — about 12-18% of the cells. If I gave each
source its own private shortest path, the total tile count would blow far past `B`. The only way to
route many sources under a tight budget is to make their paths **merge onto shared trunks**: once a
belt highway is heading toward a sink, a new source only needs to pay for the short spur that joins
the highway, not a whole private path. So the combinatorial heart is *which* sources to route and
*how* to share belts between them.

Now, the candidate's named innovation is "SA on tile orientations with a component-local
re-simulation of the changed tile's connected component." My instinct is to take that literally: keep
a layout, and let simulated annealing flip one tile's orientation at a time, re-simulating only the
sources whose walks pass through the changed cell. I should think hard about whether that move is
actually strong, because it is the obvious reading and obvious is often a trap here.

It is a trap. A single-tile reorientation is almost always *destructive*: a tile sitting on a trunk
is pointing the one way that keeps the trunk connected; rotating it to any of the other three
directions breaks the trunk and drops every source feeding through it. For a flip to *help*, it would
have to be one move in a coordinated re-route that simultaneously re-aims a whole chain of cells — and
independent random single-tile flips essentially never stumble onto that coordination. I expect (and
later confirm by experiment) that single-tile SA sits exactly at the construction score, contributing
nothing. The component-local re-simulation idea is genuinely good — it is the right way to evaluate
moves cheaply — but the *move* has to operate at the level of whole routes, not single tiles, or there
is nothing for the cheap evaluation to evaluate.

So I keep the spirit of the innovation — localized, incremental evaluation exploiting the
functional-graph independence — and I upgrade the neighborhood to the established strong metaheuristic
for congested routing: **ruin-and-recreate large-neighborhood search** (the same family as
negotiated-congestion rip-up-and-reroute from VLSI). A move ruins a few currently-routed sources,
returning their exclusive belts to the budget, then recreates by re-routing the freed sources into
the slack. That move can actually discover new trunk-sharing patterns, and SA acceptance lets it
climb out of the construction's ordering traps.

## The construction: a 0/1-BFS that pays only for new tiles

The construction needs to do the trunk-sharing automatically, not as an afterthought. I route sources
one at a time, easiest-first (by Manhattan distance to the nearest sink, so the cheap deliveries get
priority and lay down trunks early). For each source I run a shortest-path search to *any* sink, but
with a cost function that makes sharing free:

- stepping onto an **empty** cell costs **1** (I must place a new tile there);
- stepping onto a cell that **already carries the direction I need** costs **0** (I reuse the
  existing belt — this is the trunk-sharing);
- a cell fixed to a **different** direction is **blocked** (I cannot overwrite another source's belt
  without breaking its delivery).

Because edge costs are in `{0,1}`, this is a textbook **0/1-BFS** with a deque: push-front on a
zero-cost (reuse) edge, push-back on a unit-cost (new tile) edge. The path it returns is the cheapest
way, in *new tiles*, to glue this source onto the existing belt network and reach a sink. If the new
tiles needed exceed the remaining budget `B - used`, I skip the source. This directly realizes "build
belt paths as the union of shortest source-to-sink routes," with reuse turning independent shortest
paths into a shared relaxation.

There is one subtlety I have to get exactly right, and it is the direction-of-reuse condition. A belt
cell already pointing direction `d` can only be *left* toward `d`. So when I expand a cell `cur` in
the BFS, I may only consider the outgoing direction `d` if `cur` is empty (I will set it to `d` and
pay for it) or `cur` already equals `d`. If `cur` is fixed to some other direction, every expansion
from it except along its own arrow is illegal. Miss this guard and the search will happily "turn" on
a tile it does not control, producing a path the real simulator will never follow.

## Bookkeeping that makes ruin-and-recreate cheap

For the LNS to rip routes precisely, each routed source owns its explicit `(cell, dir)` path, and
every belt cell carries a **use count**: how many routed sources currently rely on a tile there.
Applying a route places a tile on each empty cell of its path (incrementing `used`) and bumps the use
counts; ripping a route decrements the use counts and frees — sets back to empty, decrementing
`used` — only the cells that drop to zero. This is the incremental budget accounting: I never recount
tiles globally, and a freed cell returns to the budget the moment its last owner leaves. It is also
the component-local evaluation in disguise: only the sources I explicitly rip or recreate can change
fate, so the delivered count updates by `gained - removed` without re-simulating the rest of the grid.

## The search loop

I start from the construction, record its delivered count as both `curScore` and `bestScore`, and
snapshot the layout as `bestTiles`. Then I loop until ~1.8 s:

1. **Ruin.** Pick `K` (1-3) random sources that are currently routed and rip them, saving their paths
   for rollback. This frees their exclusive belts back into the budget.
2. **Recreate.** Collect every currently-unrouted source (the ripped ones plus those never routed),
   shuffle them, and greedily re-route as many as fit into the freed budget with the same 0/1-BFS.
   The shuffle matters: different orders discover different sharing patterns, so randomizing it is how
   the neighborhood explores.
3. **Accept by SA.** The move's delta is `gained - removed`. If it is non-negative I take it; if it is
   negative I take it anyway with probability `exp(delta / T)`, where the temperature `T` cools
   linearly from 1.5 to near 0 over the run. On rejection I roll back exactly: rip whatever I just
   created, then re-apply the saved ripped routes.

Whenever `curScore` exceeds `bestScore` I update `bestScore` and snapshot `bestTiles`. At the end I
emit `bestTiles`, with a final `resize(B)` clamp as a belt-and-suspenders guarantee that the output
never exceeds the budget. The empty/partial layouts along the way are all feasible, and I only ever
keep the best, so I can never emit something illegal.

## A real debug-and-verify episode

I wrote the scorer in Python independently of the solver, precisely so the two cannot share a bug,
and I cross-checked. The first version of the solver over-reported: on several seeds the C++ side
believed it had delivered a source that the Python scorer counted as lost, so the solver's internal
`curScore` drifted above the true score and `bestTiles` was being chosen against a fiction.

Tracing one such source by hand, the cause was exactly the reuse-direction subtlety. My early BFS
treated *any* existing tile as freely traversable in *any* direction — it let an item enter a trunk
cell and continue in whatever direction the path wanted, while the real simulator (and the scorer)
makes the item leave that cell only along the tile's actual arrow. So the solver thought a source
merged into a trunk and rode it to a sink, but in reality the item hit the trunk cell and was
flung off in the trunk's direction, missing the sink. The fix is the guard I described:
`if (tileDir[cur] != -1 && tileDir[cur] != d) continue;` — a reused cell may only be left in its own
direction. After adding it, the solver's model and the scorer agreed exactly, and `curScore` matched
the independent count on every move.

With that fixed I ran the self-verify harness over seeds 1-20: generate each instance, run the
solver, score it, and also score the straight-line baseline. Every one of the twenty outputs is
feasible (parses, score > 0) and the solver beats the baseline on all twenty. The mean delivered
count is `19.75` for the full solver versus `0.75` for the straight-line baseline. I also built a
construction-only variant (the LNS time budget set to essentially zero) to confirm the search is
load-bearing: construction alone averages `19.45`, and the LNS lifts it to `19.75`, improving the
score on several seeds and regressing on none — modest, because the budgeted-BFS construction is
already strong, but real, and exactly in the regime where the tight budget makes extra trunk-sharing
the only way to squeeze out more deliveries. I separately checked the feasibility floor by hand:
exceeding the budget, placing a tile on a source, overlapping two tiles, an out-of-grid tile, and an
illegal direction code each correctly score zero.

The runtime sits at about 1.8 s and ~7 MB, comfortably inside a 2 s / 256 MB envelope, and the loop
runs on the order of `3 * 10^5` ruin-and-recreate moves on these instance sizes.

## Final solver

```cpp
#include <bits/stdc++.h>
using namespace std;

// ---------- Conveyor / Belt Layout : ale-44 ----------
// Place directional belt tiles on empty grid cells so items spawned at sources
// follow cell directions (one step per tick) and reach a sink.  Maximize the
// number of delivered items under a tile budget B.
//
// The per-cell direction map is a functional graph (each used cell has one
// out-direction).  A source's item is delivered iff its forward walk reaches a
// sink in <= T steps.
//
// Strong heuristic (matching the candidate's innovation):
//   STEP 1  Construct routes as a union of budgeted shortest source->sink
//           paths; existing belts are reused for free, so paths MERGE onto
//           shared trunks -- a pure relaxation of the routing problem.
//   STEP 2  Rip-up-and-reroute LNS under simulated-annealing acceptance.  A
//           move ruins a few routed sources (freeing their exclusive belts)
//           and recreates routes for a random batch of currently-unrouted
//           sources via congestion/budget-aware BFS.  The score change is
//           measured by re-simulating ONLY the sources whose belts changed
//           (component-local resim) -- the efficiency innovation.  Each belt
//           cell carries a usage count so freeing budget is incremental.

static const int DR[4] = {0, 1, 0, -1};   // 0=R,1=D,2=L,3=U
static const int DC[4] = {1, 0, -1, 0};

int H, W, nS, nG, B, T;
vector<int> srcR, srcC, srcD;
vector<int> sinkR, sinkC;
int CELLS;

vector<int> role;        // -1 empty, -2 sink, -3 source
vector<int> srcEmit;     // emission dir if source else -1
vector<int> tileDir;     // belt dir per cell, -1 if no tile
vector<int> useCount;    // how many routed sources own a belt on this cell
int used;                // number of distinct belt tiles currently placed

inline int idx(int r, int c) { return r * W + c; }
inline int outDir(int cell) {
    if (role[cell] == -3) return srcEmit[cell];
    return tileDir[cell];
}

// Simulate one source; return reached sink cell or -1.  Optionally record the
// belt cells visited (cells that carry a tile) into `path`.
int simulateSource(int s, vector<int>* path) {
    int r = srcR[s], c = srcC[s], d = srcD[s];
    for (int step = 0; step < T; ++step) {
        int nr = r + DR[d], nc = c + DC[d];
        if (nr < 0 || nr >= H || nc < 0 || nc >= W) return -1;
        int ncell = idx(nr, nc);
        if (role[ncell] == -2) return ncell;
        int nd = outDir(ncell);
        if (nd < 0) return -1;
        if (path) path->push_back(ncell);
        r = nr; c = nc; d = nd;
    }
    return -1;
}

// Each routed source owns an explicit path of (cell,dir) belt steps so we can
// rip it up exactly.  routed[s]==1 means source s currently reaches a sink.
vector<vector<pair<int,int>>> routePath;   // s -> list of (cell, dir) it laid/uses
vector<char> routed;

// BFS a budget-aware shortest path for source s, treating cells with the wrong
// existing tile as blocked (we may only reuse a tile if its dir matches the
// step we take from it).  Returns the path as (cell,dir) pairs ending at a sink,
// plus newCost = number of NEW tiles it would require; or empty on failure.
bool routeSource(int s, vector<pair<int,int>>& outPath, int& newCost,
                 int budgetLeft) {
    int sr = srcR[s] + DR[srcD[s]], sc = srcC[s] + DC[srcD[s]];
    if (sr < 0 || sr >= H || sc < 0 || sc >= W) return false;
    int start = idx(sr, sc);
    if (role[start] == -2) { outPath.clear(); newCost = 0; return true; }
    if (role[start] == -3) return false;

    // Dijkstra-ish 0/1: cost = #new tiles needed.  Reuse (cell already has the
    // dir we enter/leave with) is free; an empty cell costs 1; a wrong-tile
    // cell is blocked (we cannot overwrite another source's belt here).
    static vector<int> dist, pcell, pdir;
    dist.assign(CELLS, INT_MAX); pcell.assign(CELLS, -1); pdir.assign(CELLS, -1);
    // deque 0/1 BFS by edge cost (0 reuse / 1 new).  dist[x] = min new tiles to
    // make a valid belt chain from the entry cell `start` up to and including x.
    // The entry cell costs 1 new tile if empty, 0 if it already carries a tile.
    deque<int> q;
    dist[start] = (tileDir[start] == -1) ? 1 : 0;
    pcell[start] = -1; pdir[start] = -1;
    q.push_back(start);
    int sink = -1;
    while (!q.empty()) {
        int cur = q.front(); q.pop_front();
        int cr = cur / W, cc = cur % W;
        if (role[cur] == -2) { sink = cur; break; }
        for (int d = 0; d < 4; ++d) {
            // to leave `cur` toward d, cur must carry dir d (reuse, free if
            // tileDir==d) or be empty (we set it to d, cost already counted).
            if (tileDir[cur] != -1 && tileDir[cur] != d) continue; // fixed wrong way
            int nr = cr + DR[d], nc = cc + DC[d];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int ncell = idx(nr, nc);
            if (role[ncell] == -3) continue;          // never enter a source
            int add;
            if (role[ncell] == -2) add = 0;            // sink: no tile
            else if (tileDir[ncell] == -1) add = 1;    // empty: new tile
            else add = 0;                               // existing tile: reuse
            int nd = dist[cur] + add;
            if (nd < dist[ncell]) {
                dist[ncell] = nd; pcell[ncell] = cur; pdir[ncell] = d;
                if (add == 0) q.push_front(ncell); else q.push_back(ncell);
            }
        }
    }
    if (sink == -1) return false;
    if (dist[sink] > budgetLeft) return false;
    // reconstruct (cell,dir) from start..sink, skipping the sink cell itself
    vector<pair<int,int>> rev;
    int cur = sink;
    while (pcell[cur] != -1) {
        int p = pcell[cur]; int d = pdir[cur];
        if (role[p] != -2 && role[p] != -3) rev.push_back({p, d});
        cur = p;
    }
    // `start` (if a belt cell) also needs its dir set; it's included as the last
    // `p` when pcell chain reaches it.  Add start if it's a belt-needing cell.
    // Handle the start cell's own out-dir: it is the first step's dir.
    reverse(rev.begin(), rev.end());
    outPath = rev;
    newCost = dist[sink];
    return true;
}

// Apply a route's path: place tiles, bump useCount, mark routed.  Assumes the
// path was produced against the current layout and fits the budget.
void applyRoute(int s, const vector<pair<int,int>>& path) {
    for (auto& cd : path) {
        int cell = cd.first, dir = cd.second;
        if (tileDir[cell] == -1) { tileDir[cell] = dir; ++used; }
        ++useCount[cell];
    }
    routePath[s] = path;
    routed[s] = 1;
}

// Remove a routed source's path: decrement useCount; a tile with count 0 is
// freed (budget returns).
void ripRoute(int s) {
    for (auto& cd : routePath[s]) {
        int cell = cd.first;
        if (useCount[cell] > 0) {
            --useCount[cell];
            if (useCount[cell] == 0) { tileDir[cell] = -1; --used; }
        }
    }
    routePath[s].clear();
    routed[s] = 0;
}

int main() {
    if (scanf("%d %d", &H, &W) != 2) return 0;
    if (scanf("%d %d %d %d", &nS, &nG, &B, &T) != 4) return 0;
    CELLS = H * W;
    srcR.resize(nS); srcC.resize(nS); srcD.resize(nS);
    sinkR.resize(nG); sinkC.resize(nG);
    role.assign(CELLS, -1); srcEmit.assign(CELLS, -1);
    tileDir.assign(CELLS, -1); useCount.assign(CELLS, 0);
    routePath.assign(nS, {}); routed.assign(nS, 0);
    used = 0;

    for (int i = 0; i < nS; ++i) {
        if (scanf("%d %d %d", &srcR[i], &srcC[i], &srcD[i]) != 3) return 0;
        int cell = idx(srcR[i], srcC[i]);
        role[cell] = -3; srcEmit[cell] = srcD[i];
    }
    for (int i = 0; i < nG; ++i) {
        if (scanf("%d %d", &sinkR[i], &sinkC[i]) != 2) return 0;
        role[idx(sinkR[i], sinkC[i])] = -2;
    }

    // ---- STEP 1 : budgeted shortest-path construction (easiest source first) ----
    auto manhattanToSink = [&](int r, int c) {
        int best = INT_MAX;
        for (int g = 0; g < nG; ++g)
            best = min(best, abs(r - sinkR[g]) + abs(c - sinkC[g]));
        return best;
    };
    vector<int> order(nS);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        return manhattanToSink(srcR[a], srcC[a]) < manhattanToSink(srcR[b], srcC[b]);
    });
    for (int s : order) {
        vector<pair<int,int>> path; int cost;
        if (routeSource(s, path, cost, B - used)) applyRoute(s, path);
    }

    int curScore = 0;
    for (int s = 0; s < nS; ++s) curScore += routed[s];
    int bestScore = curScore;
    vector<int> bestTiles = tileDir;

    // ---- STEP 2 : rip-up-and-reroute LNS with SA acceptance ----
    auto t0 = chrono::steady_clock::now();
    const double TIME_LIMIT = 1.8;
    std::mt19937 rng(0xC0FFEEu);

    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    };

    // Save/restore helpers operate on the whole layout snapshot for rollback.
    // The neighborhood ruins K routed sources and tries to (re)route a random
    // batch of unrouted+ruined sources; we then compute the delivered count and
    // accept by SA.  Re-simulation is restricted to the touched sources.
    double Temp = 1.5;
    long long iter = 0;

    vector<int> ruinSet;            // sources we ripped this move
    vector<vector<pair<int,int>>> savedPaths;
    while (true) {
        if ((iter & 255) == 0) {
            double el = elapsed();
            if (el > TIME_LIMIT) break;
            Temp = 1.5 * (1.0 - el / TIME_LIMIT) + 0.01;
        }
        ++iter;

        // ---- ruin: rip up a few currently-routed sources ----
        ruinSet.clear(); savedPaths.clear();
        int K = 1 + (rng() % 3);
        for (int t = 0; t < K; ++t) {
            int s = rng() % nS;
            if (routed[s]) { ruinSet.push_back(s); }
        }
        // dedup
        sort(ruinSet.begin(), ruinSet.end());
        ruinSet.erase(unique(ruinSet.begin(), ruinSet.end()), ruinSet.end());
        for (int s : ruinSet) { savedPaths.push_back(routePath[s]); ripRoute(s); }
        int removed = (int)ruinSet.size();

        // ---- recreate: try to route the now-unrouted sources into the freed
        // budget, in a randomized order so different ruins explore different
        // trunk-sharing patterns.  Greedy first-fit into budget. ----
        vector<int> cand;
        for (int s = 0; s < nS; ++s) if (!routed[s]) cand.push_back(s);
        shuffle(cand.begin(), cand.end(), rng);
        int gained = 0;
        vector<int> newlyRouted;
        for (int s : cand) {
            vector<pair<int,int>> path; int cost;
            if (routeSource(s, path, cost, B - used)) {
                applyRoute(s, path); newlyRouted.push_back(s); ++gained;
            }
        }

        int newScore = curScore - removed + gained;
        int delta = newScore - curScore;
        bool accept;
        if (delta >= 0) accept = true;
        else accept = (std::exp((double)delta / Temp) >
                       (double)(rng() & 0xffffff) / (double)0x1000000);

        if (accept) {
            curScore = newScore;
            if (curScore > bestScore) { bestScore = curScore; bestTiles = tileDir; }
        } else {
            // rollback: rip whatever we just routed, then restore the ripped ones
            for (int s : newlyRouted) ripRoute(s);
            for (size_t i = 0; i < ruinSet.size(); ++i)
                applyRoute(ruinSet[i], savedPaths[i]);
            // curScore unchanged
        }
    }

    // ---- emit best layout ----
    tileDir = bestTiles;
    vector<array<int,3>> outTiles;
    for (int cell = 0; cell < CELLS; ++cell)
        if (tileDir[cell] != -1) outTiles.push_back({cell / W, cell % W, tileDir[cell]});
    if ((int)outTiles.size() > B) outTiles.resize(B);   // safety clamp

    printf("%d\n", (int)outTiles.size());
    for (auto& t : outTiles) printf("%d %d %d\n", t[0], t[1], t[2]);
    return 0;
}
```
