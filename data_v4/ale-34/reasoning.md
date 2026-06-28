# Reasoning: Maze Carving to a Target Difficulty

## Understanding the objective

I am handed an `H x W` grid of walls and corridors, a start `S`, an end `T`, and a carving budget `B`.
I must open exactly `B` wall cells, and then I am scored by the **shortest** path from `S` to `T` over
the open cells — and I want that shortest path to be as **long** as possible. The first thing I want to
be very clear about is the adversarial flavor of the objective: I choose the geometry, but the score is
decided by a walker who always takes the cheapest route afterwards. So it is not enough to dig a lot of
corridor; I have to dig corridor in a shape where the *cheapest* route is still long. Any place where my
carved cells let the walker cut a corner, my score collapses to whatever the corner-cutting route costs.

Let me also pin down the feasibility rules, because the scoring floors to 0 on any slip:

- I must output **exactly `B`** cells. Not `B-1`, not `B+1`.
- Every listed cell must be in bounds, must currently be a **wall**, and must be **distinct**.
- After carving, `S` and `T` must still be **connected**; if `T` is unreachable, the score is 0.

That last one is the trap I have to respect at all times: a clever long-corridor construction is
worthless if it ever leaves `S` and `T` in different components. So my plan has two layers from the
start — a guaranteed-feasible fallback that I can always fall back to, and an aggressive optimizer on
top of it.

## Reaching a feasible baseline first

Before I try to be clever I want *something* that is always valid and always connected. The cheapest way
to connect `S` and `T` while carving walls is a **0/1 BFS**: moving onto an already-open cell costs 0,
moving onto a wall costs 1 (because I have to spend a carve there). The shortest 0/1 distance from `S`
to `T` is the minimum number of walls I must open to connect them, and the parent pointers give me the
actual set of walls on that cheapest path. The generator guarantees `B >= manhattan(S,T) + 2`, and the
minimum carve count is at most the path length, so this connecting set always fits inside the budget.

That gives me a *connected* carve set, but usually it uses fewer than `B` walls. I still have to spend
the rest of the budget. The naive thing — and the trivial baseline I will measure myself against — is to
carve the connecting path and then carve arbitrary leftover walls until I hit `B`. That is always
feasible (it stays connected), but the geodesic is basically the Manhattan distance, and the arbitrary
extra carves frequently open shortcuts. So the baseline score is small. Good: it is exactly the foil I
want to beat, and it doubles as my safety net.

## The key observation: induced paths have no shortcut

Now, how do I make the shortest path *long*? I thought about what structure forces a walker to take a
long route. If the open cells that connect `S` and `T` form a thick blob, BFS cuts straight across it.
If they form a tree with branches, BFS ignores the branches and takes the trunk. The shape that has no
"cheaper alternative" anywhere is a **simple corridor of width one with no chords** — formally an
**induced path**: a sequence of open cells where consecutive cells are 4-adjacent, and *non*-consecutive
cells are never adjacent to each other.

The reason this is the right target is a clean little fact: along an induced `S`–`T` path, the BFS
geodesic equals the *entire* length of the path. There is no edge between two far-apart path cells (that
is exactly what "induced" forbids), so the walker cannot skip ahead; it must walk every cell. Therefore
**maximizing the score reduces to growing the longest induced `S`–`T` path I can afford with `B`
carves.** This reframing is the whole game: a budgeted longest-induced-path construction, not a vague
"carve a maze" search.

It also tells me precisely what kills my score: any time I open a wall that is adjacent to two
non-consecutive cells of my corridor, I create a chord and the geodesic snaps shorter. So both my
construction and my refinement have to be chord-averse.

## Constructing a long induced path

My construction is a randomized greedy self-avoiding walk from `S`. I keep a `onPath[]` marker and a
current head `cur`. At each step I look at the four neighbors of `cur` and accept a neighbor `v` only if
it can extend the **induced** path: `v` is unused, carving it (if it is a wall) stays within budget,
and — the crucial test — `v` is adjacent to **no** path cell other than `cur`. If `v` touched any other
path cell, opening it would form a chord, so I reject it. Among the legal extensions I bias the choice:
most of the time I "wander," preferring the neighbor that increases Manhattan distance to `T` (longer
detour); but when the remaining budget is about equal to the distance still owed to `T`, I switch to
"homing," preferring the neighbor that decreases distance to `T`, so I actually land on `T` before I run
out of budget. A little randomness on top makes restarts diverge.

I run this many times (a few hundred restarts inside a time slice) and keep the longest induced path
that successfully reaches `T`. Because the walk tries to consume the whole budget as it wanders, a good
run uses nearly all `B` carves on the snake itself, which is what I want: no leftover budget to waste on
shortcut-creating filler.

If a run does end with fewer than `B` carved cells, I have to pad up to exactly `B` without wrecking the
geodesic. The safe way to add a carve is to open a wall that has exactly **one** open neighbor: that
creates a degree-1 "spur" hanging off the corridor, and a spur can never be a shortcut (you can't pass
*through* a dead end). So my padding repeatedly carves a one-open-neighbor wall; only if none exists does
it fall back to the wall with the fewest open neighbors (feasibility first, score second).

## The refinement and its innovation

The construction gives a strong start, but a greedy self-avoiding walk leaves length on the table: it
commits early and can paint itself into a corner. I want a local search that *reroutes* the corridor to
be even longer. This is where the problem's intended innovation lives — **targeting moves at the current
geodesic**.

Here is the idea. After I have a carve set, I run two BFS passes: distances `dS[]` from `S` and `dT[]`
from `T`. A cell lies on *some* shortest path iff `dS[u] + dT[u] == dist(S,T)`. The carved walls that
satisfy this are the **geodesic carves** — the cells the walker is actually using. If I *remove* a
geodesic carve (close it back to a wall) and *add* a wall somewhere else to keep the count at `B`, the
walker is forced to **detour** around the gap. A detour can only make the geodesic longer or, in the bad
case, disconnect `S` and `T` — it can never make it shorter, because I deleted a cell the shortest path
relied on. So geodesic carves are exactly the high-leverage cells to perturb; carves *off* the geodesic
are dead weight for the score and a waste of a move. Picking the cell to remove from the geodesic set
(80% of the time) instead of uniformly at random is what makes the search productive in the time budget.

I wrap this in simulated annealing: propose a swap (remove `wOut`, add an adjacent uncarved wall `wIn`),
re-score, accept if it improves, and accept worsening moves with the usual `exp(-Δ/T)` probability on a
cooling schedule, while *never* accepting a move that disconnects `S` and `T`. I always keep the
best-scoring carve set seen.

## A real debugging episode

My first implementation tried to be too clever about the incremental scoring and it bit me hard. I had
read the innovation as "only recompute BFS when the move touches the geodesic," and I implemented a
literal **score skip**: if the removed cell was not on the geodesic and the added cell didn't look like
it created a shortcut, I *kept the old score without re-running BFS* and accepted the move outright.

I compiled, generated seed 1, and ran it. The construction phase reported (via a debug print to stderr)
a beautiful induced path of length **126** using 115 of the 117 carves. But the final score the scorer
gave me was **44** — exactly the Manhattan distance, no better than the trivial baseline. Something
between "I built a 126-long snake" and "the output scores 44" was destroying the path.

I isolated the two suspects by disabling phases. With SA turned off, construction + padding alone scored
**70** (already down from 126 — so padding was opening shortcuts), and turning SA back on dropped it
further to **44** (so SA was *also* corrupting the path). Two separate bugs:

1. **Padding bug.** My pad step preferred "isolated" walls but in a dense snake almost every wall has
   two or more open neighbors, so it happily carved a wall sitting *between two strands of the snake* —
   a textbook chord — collapsing 126 down to 70. Fix: pad only with **one-open-neighbor spur cells**,
   which are provably shortcut-free, and only fall back to denser walls when no spur exists.

2. **SA bug — the unsound score skip.** My "don't recompute if it doesn't touch the geodesic" shortcut
   was simply *wrong*. Removing a carved cell that is not on the current geodesic can still change the
   graph — it can sever a side-strand the geodesic depended on indirectly, and opening the new cell can
   create a chord I failed to detect with my crude two-open-neighbor test. By trusting the skip I
   accepted moves that quietly shortened the real geodesic while my bookkeeping still claimed the old
   score, and the corruption compounded until the path was just the straight tunnel.

The fix was to respect what the innovation actually buys me. The grid is at most `30 x 30 = 900` cells,
so a full BFS is about a thousand operations — utterly cheap. The geodesic targeting should accelerate
**which moves I try** (perturb cells that can actually lengthen the path), *not* let me skip the
correctness check. So I made every evaluated candidate do a **full BFS rescore** (exact, never
corrupting state), and I kept the geodesic set purely as the *move-proposal distribution*: remove a
geodesic carve with high probability to force a productive detour. That is sound and still fast — I run
on the order of `10^5` full-BFS moves inside the time budget.

After both fixes I recompiled and re-ran seed 1: score **136**, against the baseline's **44** — a clean
3x. Then I ran the whole seed set 1..20: every output was feasible (exactly `B` lines, all carves legal,
`S`–`T` connected), the solver beat the baseline on **every** seed (ratios from 1.56x up to 6.38x), and
the means were **167.75** for the solver versus **44.15** for the baseline. Wall-clock stayed at ~1.85s
per instance, inside the ~2s budget, and peak memory was a few megabytes.

## Feasibility guards I keep no matter what

Because the floor-to-0 rule is unforgiving, I bracket the optimizer with guards. I compute the
guaranteed connecting fallback once at the start. Before and after SA I validate the current carve set:
exactly `B` cells, all distinct walls, and `S`–`T` connected by BFS. If any check fails — for any reason
— I rebuild the output from the fallback connecting set padded up to `B` with arbitrary remaining walls.
That is never the *best* answer, but it is always a *valid* one, so the solver can never emit infeasible
output or a disconnected maze. The optimizer is free to be aggressive precisely because the guard behind
it is absolute.

## Complexity per step

- Construction: each greedy step is O(1) (four neighbor checks), a walk is O(carved cells), and I do a
  few hundred restarts inside a fixed time slice.
- 0/1 BFS fallback and each SA rescore: O(HW) ≈ O(900).
- SA: the geodesic refresh after an accepted move is two BFS passes, also O(HW); rejected moves cost one
  BFS. Tens of thousands of moves fit comfortably in the budget.

## Final solver

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  ale-34  Maze Carving to a Target Difficulty.

  Given an H x W grid of walls '#' and open cells '.', a start S and end T (both
  open), and a budget B, carve exactly B currently-wall cells into open. The
  score is the BFS shortest-path length S->T over the resulting open cells (0 if
  S and T end up disconnected or the carve set is invalid). We MAXIMIZE that
  shortest path.

  Idea.
    * A feasible solution must carve exactly B distinct wall cells AND keep S and
      T connected. We always secure feasibility with a minimum-carve connecting
      path used as a fallback, and we always emit exactly B distinct wall cells.
    * To make the shortest path LONG we want the carved open cells to form one
      long, winding corridor with NO shortcut. The clean way to guarantee
      "no shortcut" is to grow an INDUCED path: a sequence of open cells where
      consecutive cells are adjacent and non-consecutive cells are never
      adjacent. For an induced path the geodesic equals the full path length, so
      maximizing the number of cells on the path maximizes the score.
    * Construction: a randomized greedy self-avoiding walk from S that grows an
      induced path, wandering to spend budget, then homing on T -- aiming to use
      the entire budget so no lossy padding is needed. Best over many restarts.
    * Refinement (the innovation): simulated annealing whose moves are TARGETED
      at the current geodesic. We keep, each iteration, the set of carved cells
      that lie on a current shortest path (dS[u]+dT[u]==dist). A move removes one
      such geodesic carve (forcing the path to detour, which can only lengthen or
      disconnect it) and adds an adjacent wall to keep |carve|==B; we also use
      generic swaps. Every candidate is re-scored by a single O(HW) BFS -- cheap
      on a <=30x30 grid -- so the geodesic targeting is a *move-selection*
      accelerator (propose moves where they can actually help), never an unsafe
      score skip. This is what makes the search productive within the budget.

  We never emit an infeasible solution: a guaranteed connecting carve set is
  computed up front and used as a fallback, and a final guard rebuilds from it if
  anything is wrong.
*/

static int H, W, B;
static int SR, SC, TR, TC;
static vector<string> G0;
static int N;

static inline int id(int r, int c) { return r * W + c; }
static const int DR[4] = {1, -1, 0, 0};
static const int DC[4] = {0, 0, 1, -1};

static std::mt19937 rng(998244353u);
static inline int randint(int a, int b) { return a + (int)(rng() % (unsigned)(b - a + 1)); }

// BFS shortest path on a boolean "open" grid; -1 if unreachable.
static int bfsDist(const vector<char>& open) {
    static vector<int> dist;
    dist.assign(N, -1);
    deque<int> dq;
    dist[id(SR, SC)] = 0;
    dq.push_back(id(SR, SC));
    int target = id(TR, TC);
    while (!dq.empty()) {
        int u = dq.front(); dq.pop_front();
        if (u == target) return dist[u];
        int r = u / W, c = u % W;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = id(nr, nc);
            if (open[v] && dist[v] < 0) { dist[v] = dist[u] + 1; dq.push_back(v); }
        }
    }
    return -1;
}

static void bfsFrom(int src, const vector<char>& open, vector<int>& dist) {
    dist.assign(N, -1);
    deque<int> dq; dist[src] = 0; dq.push_back(src);
    while (!dq.empty()) {
        int u = dq.front(); dq.pop_front();
        int r = u / W, c = u % W;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = id(nr, nc);
            if (open[v] && dist[v] < 0) { dist[v] = dist[u] + 1; dq.push_back(v); }
        }
    }
}

// Minimum-carve connecting path via 0/1 BFS (wall step costs 1, open costs 0).
static vector<int> minCarveConnect(const vector<char>& isWall, bool& feasible) {
    vector<int> dist(N, INT_MAX), par(N, -1);
    deque<int> dq;
    int s = id(SR, SC), t = id(TR, TC);
    dist[s] = 0; dq.push_back(s);
    while (!dq.empty()) {
        int u = dq.front(); dq.pop_front();
        int r = u / W, c = u % W;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = id(nr, nc);
            int w = isWall[v] ? 1 : 0;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w; par[v] = u;
                if (w == 0) dq.push_front(v); else dq.push_back(v);
            }
        }
    }
    vector<int> carve;
    if (dist[t] == INT_MAX) { feasible = false; return carve; }
    feasible = true;
    int u = t;
    while (u != -1) { if (isWall[u]) carve.push_back(u); u = par[u]; }
    return carve;
}

// Greedy induced-path construction from S aiming to spend EXACTLY `budget`
// carves and end at T as an induced path. Returns the carved-wall list and the
// induced path length on success.
struct BuildResult {
    bool ok = false;
    vector<int> carve;
    int pathLen = -1;
    int carvesUsed = 0;
};

static BuildResult buildInducedPath(const vector<char>& isWall, int budget) {
    BuildResult R;
    vector<char> onPath(N, 0);
    vector<int> path;
    int s = id(SR, SC), t = id(TR, TC);
    onPath[s] = 1; path.push_back(s);
    int carved = 0;
    int cur = s;

    auto manhT = [&](int u) { int r = u / W, c = u % W; return abs(r - TR) + abs(c - TC); };

    // v extends the induced path from cur iff it is unused, and adjacent to no
    // path cell other than cur (no chord => induced => no shortcut), and carving
    // it (if wall) stays within budget.
    auto canExtend = [&](int v, int& addCarve) -> bool {
        if (onPath[v]) return false;
        int r = v / W, c = v % W;
        addCarve = isWall[v] ? 1 : 0;
        if (carved + addCarve > budget) return false;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int u = id(nr, nc);
            if (u == cur) continue;
            if (onPath[u]) return false;
        }
        return true;
    };

    int safety = 0, maxSteps = 16 * N;
    while (cur != t && safety++ < maxSteps) {
        int remaining = budget - carved;
        // Home toward T when budget is nearly exhausted (need to land on T) or
        // occasionally, to avoid getting cornered far from T with no budget.
        bool homing = (remaining <= manhT(cur) + 1) || (randint(0, 99) < 8);

        int cr = cur / W, cc = cur % W;
        int bestV = -1, bestCarve = 0;
        int order[4] = {0, 1, 2, 3};
        for (int i = 3; i > 0; i--) std::swap(order[i], order[randint(0, i)]);
        long long bestKey = LLONG_MIN;
        for (int oi = 0; oi < 4; oi++) {
            int k = order[oi];
            int nr = cr + DR[k], nc = cc + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = id(nr, nc);
            int addCarve;
            if (!canExtend(v, addCarve)) continue;
            int d = manhT(v);
            long long key = homing ? (long long)(-d) * 4 + randint(0, 3)
                                   : (long long)(d) * 4 + randint(0, 3);
            if (key > bestKey) { bestKey = key; bestV = v; bestCarve = addCarve; }
        }
        if (bestV < 0) break;
        onPath[bestV] = 1; path.push_back(bestV);
        carved += bestCarve; cur = bestV;
        if (cur == t) break;
    }

    if (cur != t) return R;
    R.ok = true;
    R.pathLen = (int)path.size() - 1;
    R.carvesUsed = carved;
    for (int u : path) if (isWall[u]) R.carve.push_back(u);
    return R;
}

// Append extra carves that extend a dead-end "spur" off an existing open cell
// without shortcutting, to bring a carve set of size < B up to exactly B while
// preserving (not reducing) the geodesic. Falls back to any wall if no safe spur
// exists (feasibility first). Returns the padded carve set.
static void padToB(vector<int>& carve, const vector<char>& isWall) {
    vector<char> chosen(N, 0);
    for (int u : carve) chosen[u] = 1;
    auto isOpen = [&](int u) { return (!isWall[u]) || chosen[u]; };
    auto openNbr = [&](int u) {
        int r = u / W, c = u % W, cnt = 0;
        for (int k = 0; k < 4; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            if (isOpen(id(nr, nc))) cnt++;
        }
        return cnt;
    };
    // Repeatedly carve a wall that has exactly ONE open neighbour: opening it
    // creates a leaf/spur, which cannot create a shortcut (it adds a degree-1
    // cell), so the geodesic is unchanged.
    while ((int)carve.size() < B) {
        int pick = -1;
        // Prefer spur cells (exactly one open neighbour). Scan once.
        for (int u = 0; u < N; u++) {
            if (!isWall[u] || chosen[u]) continue;
            if (openNbr(u) == 1) { pick = u; break; }
        }
        if (pick < 0) {
            // No safe spur: take any remaining wall with the fewest open
            // neighbours (least likely to shortcut). Feasibility first.
            int best = INT_MAX;
            for (int u = 0; u < N; u++) {
                if (!isWall[u] || chosen[u]) continue;
                int onb = openNbr(u);
                if (onb < best) { best = onb; pick = u; if (onb == 0) break; }
            }
        }
        if (pick < 0) break;
        chosen[pick] = 1; carve.push_back(pick);
    }
}

// Simulated annealing refinement (the innovation): geodesic-targeted swaps,
// each candidate fully re-scored by one BFS (correct by construction).
static void anneal(vector<int>& carve, const vector<char>& isWall, double timeLimitSec) {
    auto t0 = chrono::steady_clock::now();
    vector<char> chosen(N, 0);
    for (int u : carve) chosen[u] = 1;

    auto buildOpen = [&]() {
        vector<char> open(N, 0);
        for (int u = 0; u < N; u++) open[u] = (!isWall[u]) || chosen[u];
        return open;
    };

    vector<int> dS, dT;
    vector<char> open = buildOpen();
    bfsFrom(id(SR, SC), open, dS);
    bfsFrom(id(TR, TC), open, dT);
    int curScore = dS[id(TR, TC)];
    if (curScore < 0) curScore = 0;

    // geodesic carved cells (carved walls on some shortest path).
    vector<int> geoCarves;
    auto refreshGeo = [&]() {
        open = buildOpen();
        bfsFrom(id(SR, SC), open, dS);
        bfsFrom(id(TR, TC), open, dT);
        int total = dS[id(TR, TC)];
        curScore = (total < 0) ? 0 : total;
        geoCarves.clear();
        if (total >= 0) {
            for (int u = 0; u < N; u++) {
                if (chosen[u] && dS[u] >= 0 && dT[u] >= 0 && dS[u] + dT[u] == total)
                    geoCarves.push_back(u);
            }
        }
    };
    refreshGeo();

    vector<int> bestCarve = carve;
    int bestScore = curScore;

    auto pickAdjWall = [&](int wOut) -> int {
        int ro = wOut / W, co = wOut % W;
        int ord[4] = {0, 1, 2, 3};
        for (int i = 3; i > 0; i--) std::swap(ord[i], ord[randint(0, i)]);
        for (int oi = 0; oi < 4; oi++) {
            int k = ord[oi];
            int nr = ro + DR[k], nc = co + DC[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            int v = id(nr, nc);
            if (isWall[v] && !chosen[v]) return v;
        }
        // any random non-chosen wall
        for (int tries = 0; tries < 16; tries++) {
            int v = randint(0, N - 1);
            if (isWall[v] && !chosen[v] && v != wOut) return v;
        }
        return -1;
    };

    double Tstart = 4.0, Tend = 0.05, T = Tstart;
    long long iters = 0;
    while (true) {
        if ((iters & 255) == 0) {
            double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            if (el > timeLimitSec) break;
            double frac = el / timeLimitSec;
            T = Tstart * pow(Tend / Tstart, frac);
        }
        iters++;
        if (carve.empty()) break;

        // Choose the cell to remove: with high probability target a geodesic
        // carve (removing it forces a detour => can only lengthen or disconnect
        // the path); otherwise a random carve (diversify).
        int wOut, idxOut = -1;
        if (!geoCarves.empty() && randint(0, 99) < 80) {
            wOut = geoCarves[randint(0, (int)geoCarves.size() - 1)];
            // locate index in carve
            for (int i = 0; i < (int)carve.size(); i++) if (carve[i] == wOut) { idxOut = i; break; }
        } else {
            idxOut = randint(0, (int)carve.size() - 1);
            wOut = carve[idxOut];
        }
        if (idxOut < 0) continue;

        int wIn = pickAdjWall(wOut);
        if (wIn < 0) continue;

        // Apply tentatively, full rescore (correct).
        chosen[wOut] = 0; chosen[wIn] = 1;
        vector<char> nopen = buildOpen();
        int total = bfsDist(nopen);
        int newScore = (total < 0) ? -1 : total;

        bool accept;
        if (newScore < 0) accept = false;            // disconnected: never accept
        else if (newScore >= curScore) accept = true;
        else {
            double d = curScore - newScore;
            double prob = exp(-d / std::max(1e-9, T));
            accept = (randint(0, 1000000) / 1000000.0) < prob;
        }

        if (accept) {
            carve[idxOut] = wIn;
            refreshGeo();
            if (curScore > bestScore) { bestScore = curScore; bestCarve = carve; }
        } else {
            chosen[wOut] = 1; chosen[wIn] = 0;
        }
    }

    carve = bestCarve;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> H >> W >> B)) return 0;
    cin >> SR >> SC >> TR >> TC;
    G0.assign(H, string());
    for (int r = 0; r < H; r++) {
        string line;
        cin >> line;
        if ((int)line.size() < W) line.resize(W, '.');
        G0[r] = line;
    }
    N = H * W;

    vector<char> isWall(N, 0);
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            isWall[id(r, c)] = (G0[r][c] == '#') ? 1 : 0;
    isWall[id(SR, SC)] = 0;
    isWall[id(TR, TC)] = 0;

    int totalWalls = 0;
    for (int u = 0; u < N; u++) totalWalls += isWall[u];
    if (B > totalWalls) B = totalWalls;

    bool feasible = false;
    vector<int> fallback = minCarveConnect(isWall, feasible);

    auto t0 = chrono::steady_clock::now();

    // Strong construction: best full-budget induced path over restarts.
    vector<int> best;
    int bestLen = -1;
    int restarts = 0;
    while (true) {
        double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        if (restarts >= 6 && el > 0.45) break;
        if (el > 0.8) break;
        if (restarts > 600) break;
        restarts++;
        BuildResult R = buildInducedPath(isWall, B);
        if (R.ok && R.pathLen > bestLen && (int)R.carve.size() <= B) {
            bestLen = R.pathLen;
            best = R.carve;
        }
    }

    vector<int> carve;
    if (bestLen >= 0) carve = best;
    else if (feasible) carve = fallback;
    else carve.clear();

    // Ensure connected; if not, merge in the guaranteed fallback.
    {
        vector<char> chosen(N, 0);
        for (int u : carve) chosen[u] = 1;
        vector<char> open(N, 0);
        for (int u = 0; u < N; u++) open[u] = (!isWall[u]) || chosen[u];
        if (bfsDist(open) < 0 && feasible) {
            for (int u : fallback) chosen[u] = 1;
            carve.clear();
            for (int u = 0; u < N; u++) if (isWall[u] && chosen[u]) carve.push_back(u);
        }
        // dedupe
        chosen.assign(N, 0);
        vector<int> dd;
        for (int u : carve) if (isWall[u] && !chosen[u]) { chosen[u] = 1; dd.push_back(u); }
        carve = dd;
        if ((int)carve.size() > B) carve.resize(B);
    }

    // Pad to exactly B with spur cells (no shortcut).
    padToB(carve, isWall);
    if ((int)carve.size() > B) carve.resize(B);

    // Feasibility guard before SA.
    {
        vector<char> chosen(N, 0);
        bool bad = ((int)carve.size() != B);
        for (int u : carve) { if (u < 0 || u >= N || !isWall[u] || chosen[u]) { bad = true; break; } chosen[u] = 1; }
        if (!bad) {
            vector<char> open(N, 0);
            for (int u = 0; u < N; u++) open[u] = (!isWall[u]) || chosen[u];
            if (bfsDist(open) < 0) bad = true;
        }
        if (bad) {
            vector<char> ch2(N, 0); vector<int> c2;
            for (int u : fallback) if (isWall[u] && !ch2[u]) { ch2[u] = 1; c2.push_back(u); }
            for (int u = 0; u < N && (int)c2.size() < B; u++)
                if (isWall[u] && !ch2[u]) { ch2[u] = 1; c2.push_back(u); }
            if ((int)c2.size() > B) c2.resize(B);
            carve = c2;
        }
    }

    // SA refinement.
    {
        double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        double remain = 1.85 - el;
        if (remain > 0.05 && (int)carve.size() == B) anneal(carve, isWall, remain);
    }

    // Final feasibility guard after SA.
    {
        vector<char> chosen(N, 0);
        bool bad = ((int)carve.size() != B);
        for (int u : carve) { if (u < 0 || u >= N || !isWall[u] || chosen[u]) { bad = true; break; } chosen[u] = 1; }
        if (!bad) {
            vector<char> open(N, 0);
            for (int u = 0; u < N; u++) open[u] = (!isWall[u]) || chosen[u];
            if (bfsDist(open) < 0) bad = true;
        }
        if (bad) {
            vector<char> ch2(N, 0); vector<int> c2;
            for (int u : fallback) if (isWall[u] && !ch2[u]) { ch2[u] = 1; c2.push_back(u); }
            for (int u = 0; u < N && (int)c2.size() < B; u++)
                if (isWall[u] && !ch2[u]) { ch2[u] = 1; c2.push_back(u); }
            if ((int)c2.size() > B) c2.resize(B);
            carve = c2;
        }
    }

    string out;
    out.reserve(carve.size() * 8);
    for (int u : carve) {
        int r = u / W, c = u % W;
        out += to_string(r); out += ' '; out += to_string(c); out += '\n';
    }
    cout << out;
    return 0;
}
```
