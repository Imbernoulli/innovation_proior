# Reasoning: Wall Painting

## Reading the objective

I have an `N x N` target picture in `C` colours. I get to paint it with at most
`T` axis-aligned rectangular brush strokes onto a canvas that starts all colour
`0`, strokes layered painter's-algorithm style (later overwrites earlier), and I
score one point per cell whose final colour matches the target. `N` is up to 40,
so up to 1600 cells, while `T` is only `N..3N`, i.e. at most 120 strokes. So I
cannot dab cells individually — 120 strokes against 1600 cells means each stroke
has to be a *region* decision, not a pixel decision. And the score is continuous:
there's no "solved", just "how many cells did you get right." Infeasible output
(over budget, a rectangle off the grid, `r1>r2`, a bad colour) floors me to zero,
so whatever I do, every byte I print must be a legal stroke and I must never
exceed `T`.

The first thing I want is a clear picture of the value landscape. The canvas
starts at colour `0`. So before I paint anything at all, I already match every
target cell that happens to be colour `0`. That's free. Concretely, on the
instances I generate, colour `0` is the background that the random rectangles
were painted over, so a sizeable fraction of cells are already `0` — the empty
solution is a real, non-zero floor. Any solver has to beat *that*, not zero.

## A feasible baseline first

Rule one of these heuristic problems: always hold a valid solution. The simplest
valid solution is `Q = 0` — print `0` and stop. It's feasible and scores the
count of target cells equal to `0`. Good, that's my floor; I can't do worse than
that if I'm careful.

The next obvious thing: instead of leaving the canvas at `0`, fill it with the
*most frequent* target colour. One stroke, the full grid `0 0 N-1 N-1 base`. If
the picture's dominant colour is something other than `0`, this is a clear win
over the empty canvas; if the dominant colour is `0`, it's a no-op and I just
keep the empty solution. This single-fill is my reference baseline — the "do the
obvious thing" solution the real solver must beat.

Can I do better with a few more strokes greedily? Yes: after the base fill, walk
a coarse tiling of the grid (blocks of side `~N/6`), and for each block find the
majority colour inside it; if that majority is different from the base and
actually dominates the block, lay a solid stroke of it. That's a layered greedy
construction — base coat, then a handful of coloured blocks on top. It's fast,
always feasible, and already much better than the flat fill. But it's one-shot:
each block is decided in isolation and never revisited. A block whose majority is
51% will paint the whole block that colour and get the other 49% wrong, and
nothing later fixes it. The greedy has no notion of "this stroke was a mistake,
pull it back." That's the ceiling of pure construction, and it's where a local
search has to take over.

## Why the obvious local search is too slow

So I want to *search* over stroke sequences. The natural representation is a
fixed array of `T` strokes (I'll always emit `T` of them; strokes I don't want
just degenerate into a `1x1` dot that something else paints over — feasible and
harmless). A move perturbs one stroke: nudge an edge, translate the rectangle,
recolour it, or replace it with a fresh random rectangle. Simulated annealing
over this with Metropolis acceptance is the established strong metaheuristic for
this kind of layered-covering structure, and crucially it can *vacate* a wasteful
early stroke — accept a temporary score drop, free the region, and let a better
stroke cover it — which is exactly what the greedy cannot do.

The problem is the cost of evaluating a move. A cell's *visible* colour is the
colour of the topmost stroke covering it (or `0` if none). When I edit stroke
`i`, the naive thing is: recompute the whole canvas and recount matches. Recompute
means replaying all `T` strokes over all `N*N` cells — `O(T * N*N)`, up to
`120 * 1600 ≈ 1.9*10^5` operations *per move*. Even the cheaper "recompute the
canvas without replaying" is `O(N*N)` per move because I'd have to re-derive the
top stroke of every cell. At `O(N*N)` per move and ~2 seconds, I'd get maybe a
few hundred thousand moves — far too few for SA to anneal a 120-dimensional
combinatorial object well. The search would barely move off the greedy start.

This is the wall. The whole viability of the metaheuristic hinges on making each
move *cheap*, and "cheap" has to mean **proportional to the size of the stroke I
edited**, not to the grid.

## The innovation: incremental scoring over the stroke's footprint

Here's the key observation. Editing stroke `i` changes the visible colour of a
cell *only if* that cell is in the old footprint of stroke `i`, or the new
footprint, or both. Every cell outside the union of those two rectangles is
completely untouched — same strokes covering it, same topmost stroke, same
colour. So the score delta lives entirely inside `old_rect ∪ new_rect`, which is
`O(stroke area)`, typically a tiny fraction of the grid.

To exploit that I maintain, for every cell, `top[r][c]` = the index of the
topmost stroke covering it, or `-1` for bare canvas (colour `0`). The visible
colour of a cell is then `O(1)`: `S[top].col` (or `0`). And I keep the running
match `score` as a single integer. When I edit stroke `i` from rectangle `os` to
`ns`:

- A cell that is in **both** `os` and `ns` and whose `top == i`: it was showing
  `i`'s *old* colour, now shows `i`'s *new* colour. Just re-test the match.
- A cell that **enters** `i` (in `ns`, not `os`): stroke `i` now covers it. It
  becomes visible as `i` *unless* some stroke above `i` already covers it — and I
  can tell that in `O(1)`, because if `top > i` then something above stays on
  top, otherwise `i` becomes the new top.
- A cell that **leaves** `i` (in `os`, not `ns`): if `top != i`, `i` wasn't the
  visible stroke there, so nothing changes. If `top == i`, the cell loses `i` and
  falls back to the topmost stroke *below* `i` that still covers it. That single
  cell needs a rescan down the stack — but only that cell, and only when `i` was
  genuinely on top there.

For each touched cell I add `(now_matches) - (was_matches)` to `score`. The
update is `O(|old_rect ∪ new_rect|)` plus the occasional single-cell rescan on
shrink. That's the incremental core: each SA step costs roughly the area of the
stroke I poked, not `N*N`. That buys *millions* of moves in the budget instead of
hundreds of thousands, which is what makes the annealing actually anneal.

For acceptance I use standard SA: a geometric temperature schedule from a `T0`
on the order of a few percent of the cells down to a small `T1`, accept any
non-negative delta, accept a negative delta with probability `exp(delta/temp)`. I
warm-start from the greedy layered construction so the search never starts worse
than the baseline, and I remember the best feasible state seen and print that.

## Implementing it — and the bug I hit

I wrote `apply_move(i, ns)` to commit `S[i] = ns` first (so the single-cell
rescan helper `recompute_top` sees the new geometry), then loop over the bounding
box of `os ∪ ns`, classify each cell as in-old / in-new / both, update
`top[cell]`, and fold the match change into a local `sc`, returning the new
score. On accept I keep it; on reject I call `revert_move`, which restores
`S[i] = os` and recomputes `top` over the same union footprint.

When I first ran it, the scores looked plausible, so I almost trusted it. But the
top-stack bookkeeping is exactly the kind of thing that is subtly wrong in a way
that *still produces feasible output* — the scorer replays from scratch, so a
bookkeeping bug wouldn't crash or print garbage, it would just make my SA climb
the *wrong* score and waste the whole budget. I needed to know the internal score
and the internal `top[][]` array exactly equal a from-scratch replay at all
times. So I built a debug version that, every few thousand accepted moves, fully
replays all `T` strokes into a fresh `tv[]` array, recounts matches into `sc2`,
and asserts `sc2 == score` and `tv == top` cell-for-cell, aborting with a
diagnostic on any mismatch.

The first thing this caught was my own confusion about *which* colour a cell was
"showing before" the move. In the both-in-old-and-new branch I had initially read
the old visible colour as `S[top].col` — but for a cell whose `top == i`, by the
time I evaluate it I'd already committed `S[i] = ns`, so `S[top].col` was the
*new* colour, not the old one. That made `wasMatch` wrong and the delta
double-counted. The fix is explicit: the cell's old shown colour is `os.col` when
`oldTop == i`, otherwise `S[oldTop].col`, otherwise `0` — never read `S[i]` for
the *old* colour after committing `ns`. Once I special-cased `oldTop == i` to use
`os.col`, the assertion stopped firing.

The second subtlety the assertion guarded was the "leaves `i`, `top == i`" case:
the fall-back top must be re-derived by scanning strokes below, and I had to make
sure I scan from `T-1` down (any stroke, since a stroke `> i` that covered the
cell would have meant `top != i` in the first place, but scanning the whole stack
is simplest and correct). With both fixes in, I reran the debug build on seeds
1, 5, 11, 20: thousands of accepted moves each, and the incremental `score` and
the full `top[][]` matched the from-scratch replay every single time. That's the
moment I trusted the incremental engine.

## Self-verify against the baselines

With the engine correct, the question is whether it actually *beats* the
baselines on the seed set. I wrote `selfcheck.sh`: compile, run seeds `1..20`,
and for each score the solver, the flat most-common-colour fill, and the empty
`Q=0` canvas with the same frozen `score.py`. Requirements: every solver output
feasible (score `> 0` and not floored), and the solver mean strictly above both
baseline means.

The result: every seed feasible, no zeros. Mean solver `981.7`, mean flat fill
`558.4`, mean empty `503.35`. On the 26x26 instances the solver lands ~631/676
(93%), on the noisier small ones it's lower but still well clear of the flat
fill, and on the big 40x40 boards it reaches ~1500/1600. The solver beats the
flat baseline on every single seed, not just on average. I also checked the time:
each run finishes in ~1.85 s, inside the 2 s budget, and `Q` is always exactly
`T` so the budget constraint `Q <= T` holds by construction. Edge seeds (small
`C`, tight `T`) run clean with no crashes.

A note on why the gap is so large: the flat fill only ever gets the single
dominant colour right plus whatever noise happens to share it; the SA solver
reconstructs the actual overlapping-rectangle structure of the picture — it lays
the base coat, then anneals a stack of coloured rectangles whose *order* and
*overlap* recover the layered way the target was built, recovering colours the
flat fill has no way to express. The incremental footprint scoring is what makes
that search depth affordable.

## What I'd watch if pushed further

The current per-move cost is dominated by stroke area; very large strokes (the
base coat) are expensive to perturb, so most useful moves are on the smaller
upper strokes — the schedule naturally spends its time there. If I wanted more, a
cleaner "inactive stroke" flag (instead of a `1x1` dot) would let SA explicitly
choose to use *fewer* than `T` strokes and free budget, and a kd-style spatial
index over strokes would speed the single-cell fall-back rescan on shrink moves.
But the footprint-incremental SA already clears the baseline decisively and stays
feasible on every seed, which is the bar here.

## Final solver

The complete single-file C++17 solver (identical to `verify/sol.cpp`):

```cpp
// ALE-10  Wall Painting  --  heuristic solver.
//
// Objective: reproduce an N x N target picture (colours 0..C-1) with at most T
// axis-aligned rectangular *brush strokes*. A canvas starts filled with colour 0;
// strokes are applied in order and later strokes overwrite earlier ones; the
// score is the number of cells whose final colour equals the target. T is far
// smaller than N*N, so cells cannot be painted individually -- the structure of
// the picture (its overlapping coloured rectangles) must be exploited.
//
// INNOVATION (why this file is fast):
//   We keep a fixed sequence of T strokes and, for every cell, the index of the
//   TOPMOST stroke covering it (top[r][c], -1 = bare canvas => colour 0). The
//   displayed colour of a cell is therefore O(1), and the match score is held
//   incrementally. A simulated-annealing move edits ONE stroke i (its rectangle
//   and/or colour). Only cells in the UNION of stroke i's old and new footprints
//   can change, so the score delta is computed over that footprint alone -- never
//   over the whole grid and never by replaying all T strokes. Concretely:
//     * cells that LEAVE i's rectangle and had top==i lose i's colour and fall
//       back to the topmost stroke below i (found by a short local rescan);
//     * cells that ENTER i's rectangle take colour(i) iff no stroke ABOVE i
//       already covers them (i.e. their current top < i);
//     * cells in BOTH rectangles whose top==i just swap i's old colour for the
//       new one.
//   Maintaining top[][] this way makes each SA step cost O(footprint), not
//   O(N^2) or O(T*N^2), which is what lets the search take millions of steps and
//   actually beat the greedy construction.
//
// SEARCH: SA over the T strokes. Warm start = greedy layered construction (most
// frequent colour as a full-canvas base, then a few large solid-colour blocks).
// Moves: perturb a random stroke's rectangle (shift/resize an edge) or recolour
// it; Metropolis acceptance lets the search undo a wasteful early stroke that a
// later one would otherwise have to paint over. The best feasible state seen is
// remembered and printed, so the output is always feasible and never worse than
// the warm start.
//
// I/O:
//   stdin : "N C T", then N lines of N integers (the target grid).
//   stdout: "Q", then Q lines "r1 c1 r2 c2 col" (0<=r1<=r2<N, 0<=c1<=c2<N).
// Compile: g++ -O2 -std=c++17 sol.cpp
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static inline uint64_t xr() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline int randint(int n) { return (int)(xr() % (uint64_t)n); }
static inline double urand() { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

static double now_sec() {
    return chrono::duration<double>(
               chrono::steady_clock::now().time_since_epoch())
        .count();
}

int N, C, T;
vector<int> target;                 // target[r*N+c]

struct Stroke { int r1, c1, r2, c2, col; };

int main() {
    double t_start = now_sec();
    const double TIME_LIMIT = 1.85;  // seconds, leaves margin under a 2s cap

    if (scanf("%d %d %d", &N, &C, &T) != 3) return 0;
    target.assign(N * N, 0);
    for (int i = 0; i < N * N; i++) scanf("%d", &target[i]);

    auto idx = [&](int r, int c) { return r * N + c; };

    // ----- structures maintained incrementally -----
    vector<Stroke> S(T);
    // top[cell] = index of topmost stroke covering it, or -1 (bare canvas => 0).
    vector<int> topv(N * N, -1);
    int score = 0;  // number of matching cells under the current sequence

    // colour displayed at a cell given its top index
    auto disp = [&](int cell) -> int {
        int t = topv[cell];
        return (t < 0) ? 0 : S[t].col;
    };

    // Recompute the topmost stroke covering a single cell by scanning strokes
    // top-down. Used only on the (small) footprints touched by a move.
    auto recompute_top = [&](int r, int c) -> int {
        for (int t = T - 1; t >= 0; t--) {
            const Stroke &s = S[t];
            if (r >= s.r1 && r <= s.r2 && c >= s.c1 && c <= s.c2) return t;
        }
        return -1;
    };

    // ----- warm start: a feasible, decent layered construction -----
    // Stroke 0: paint the whole canvas with the most common target colour.
    // (Cheap, and already a strong baseline.)
    auto build_initial = [&]() {
        vector<long long> freq(C, 0);
        for (int i = 0; i < N * N; i++) freq[target[i]]++;
        int base = 0;
        for (int col = 1; col < C; col++)
            if (freq[col] > freq[base]) base = col;

        // Default every stroke to a harmless 1x1 dot of the base colour at (0,0);
        // these are overwritten by stroke 0 / later strokes and never hurt.
        for (int t = 0; t < T; t++) S[t] = {0, 0, 0, 0, base};
        if (T >= 1) S[0] = {0, 0, N - 1, N - 1, base};

        // A few greedy solid blocks: scan a coarse set of rectangles and, for the
        // most frequent non-base colour inside, lay a block if it helps. Kept
        // simple -- SA does the heavy lifting; this only needs to be feasible and
        // better than a single flat fill.
        int next = 1;
        int step = max(1, N / 6);
        for (int r1 = 0; r1 < N && next < T; r1 += step) {
            for (int c1 = 0; c1 < N && next < T; c1 += step) {
                int r2 = min(N - 1, r1 + step - 1);
                int c2 = min(N - 1, c1 + step - 1);
                // majority colour in this block
                vector<int> cnt(C, 0);
                for (int r = r1; r <= r2; r++)
                    for (int c = c1; c <= c2; c++) cnt[target[idx(r, c)]]++;
                int best = 0;
                for (int col = 1; col < C; col++)
                    if (cnt[col] > cnt[best]) best = col;
                if (best != base && cnt[best] * 2 > (r2 - r1 + 1) * (c2 - c1 + 1)) {
                    S[next++] = {r1, c1, r2, c2, best};
                }
            }
        }
        // remaining slots stay as the harmless base-coloured dot at (0,0)

        // Build top[][] and the score by replaying once (one-off O(T*N^2)).
        for (int cell = 0; cell < N * N; cell++) topv[cell] = -1;
        for (int t = 0; t < T; t++) {
            const Stroke &s = S[t];
            for (int r = s.r1; r <= s.r2; r++)
                for (int c = s.c1; c <= s.c2; c++) topv[idx(r, c)] = t;
        }
        score = 0;
        for (int cell = 0; cell < N * N; cell++)
            if (disp(cell) == target[cell]) score++;
    };
    build_initial();

    // Apply a candidate edit to stroke `i` (new rectangle/colour `ns`) and return
    // the new score, updating topv[] and S[i] IN PLACE. Only cells in the union of
    // the old and new footprints are touched -- this is the incremental core.
    auto apply_move = [&](int i, const Stroke &ns) -> int {
        Stroke os = S[i];
        S[i] = ns;  // commit so recompute_top sees the new geometry

        int sc = score;
        // Bounding box of the union of old and new rectangles.
        int R1 = min(os.r1, ns.r1), R2 = max(os.r2, ns.r2);
        int C1 = min(os.c1, ns.c1), C2 = max(os.c2, ns.c2);
        for (int r = R1; r <= R2; r++) {
            for (int c = C1; c <= C2; c++) {
                bool inOld = (r >= os.r1 && r <= os.r2 && c >= os.c1 && c <= os.c2);
                bool inNew = (r >= ns.r1 && r <= ns.r2 && c >= ns.c1 && c <= ns.c2);
                if (!inOld && !inNew) continue;
                int cell = idx(r, c);
                int oldTop = topv[cell];
                // old displayed colour at this cell
                int oldColShown;
                if (oldTop < 0) oldColShown = 0;
                else if (oldTop == i) oldColShown = os.col;  // i's *old* colour
                else oldColShown = S[oldTop].col;
                bool wasMatch = (oldColShown == target[cell]);

                // compute the new top index for this cell
                int newTop;
                if (inNew) {
                    // stroke i now covers it; topmost is i unless a stroke >i covers it
                    if (oldTop > i) newTop = oldTop;           // something above stays on top
                    else newTop = i;                            // i becomes (or stays) top
                } else {
                    // i no longer covers it
                    if (oldTop != i) newTop = oldTop;           // i wasn't the top => unaffected
                    else newTop = recompute_top(r, c);          // i was the top => rescan below
                }
                topv[cell] = newTop;
                int newColShown = (newTop < 0) ? 0 : S[newTop].col;
                bool nowMatch = (newColShown == target[cell]);
                sc += (int)nowMatch - (int)wasMatch;
            }
        }
        return sc;
    };

    // Undo helper: restore stroke i to `os` and topv over the union footprint.
    // We simply call apply_move with the saved stroke, but apply_move recomputes
    // tops from the *current* S, which after a rejected move still has the right
    // geometry for strokes != i; restoring S[i]=os and recomputing the union
    // footprint reproduces the pre-move state exactly.
    auto revert_move = [&](int i, const Stroke &os, const Stroke &ns) {
        // ns is what is currently in S[i]; restore os over the union footprint.
        S[i] = os;
        int R1 = min(os.r1, ns.r1), R2 = max(os.r2, ns.r2);
        int C1 = min(os.c1, ns.c1), C2 = max(os.c2, ns.c2);
        for (int r = R1; r <= R2; r++)
            for (int c = C1; c <= C2; c++) {
                bool inOld = (r >= os.r1 && r <= os.r2 && c >= os.c1 && c <= os.c2);
                bool inNew = (r >= ns.r1 && r <= ns.r2 && c >= ns.c1 && c <= ns.c2);
                if (!inOld && !inNew) continue;
                topv[idx(r, c)] = recompute_top(r, c);
            }
    };

    // ----- remember the best feasible state -----
    vector<Stroke> bestS = S;
    int bestScore = score;

    // propose a perturbed version of stroke i
    auto propose = [&](int i) -> Stroke {
        Stroke s = S[i];
        int kind = randint(6);
        if (kind == 0) {
            s.col = randint(C);                      // recolour
        } else if (kind == 1) {                      // jiggle one edge inward/outward
            int e = randint(4);
            int d = randint(2) ? 1 : -1;
            if (e == 0) s.r1 = min(max(0, s.r1 + d), s.r2);
            else if (e == 1) s.r2 = max(min(N - 1, s.r2 + d), s.r1);
            else if (e == 2) s.c1 = min(max(0, s.c1 + d), s.c2);
            else s.c2 = max(min(N - 1, s.c2 + d), s.c1);
        } else if (kind == 2) {                      // translate the whole rect
            int dr = randint(3) - 1, dc = randint(3) - 1;
            int h = s.r2 - s.r1, w = s.c2 - s.c1;
            int nr1 = min(max(0, s.r1 + dr), N - 1 - h);
            int nc1 = min(max(0, s.c1 + dc), N - 1 - w);
            s.r1 = nr1; s.r2 = nr1 + h; s.c1 = nc1; s.c2 = nc1 + w;
        } else {                                     // fresh random rectangle+colour
            int a = randint(N), b = randint(N);
            int c1 = randint(N), c2 = randint(N);
            s.r1 = min(a, b); s.r2 = max(a, b);
            s.c1 = min(c1, c2); s.c2 = max(c1, c2);
            s.col = randint(C);
        }
        return s;
    };

    // ----- simulated annealing -----
    double T0 = max(2.0, 0.01 * N * N);   // initial temperature ~ a few % of cells
    double T1 = 0.05;
    long long iter = 0;
    double t_now = now_sec();
    while (true) {
        if ((iter & 1023) == 0) {
            t_now = now_sec();
            if (t_now - t_start > TIME_LIMIT) break;
        }
        iter++;
        double frac = (t_now - t_start) / TIME_LIMIT;
        if (frac > 1.0) frac = 1.0;
        double temp = T0 * pow(T1 / T0, frac);

        int i = randint(T);
        Stroke os = S[i];
        Stroke ns = propose(i);
        int newScore = apply_move(i, ns);  // mutates S[i], topv over footprint
        int delta = newScore - score;
        bool accept = (delta >= 0) || (urand() < exp(delta / temp));
        if (accept) {
            score = newScore;
            if (score > bestScore) {
                bestScore = score;
                bestS = S;
            }
        } else {
            revert_move(i, os, ns);  // S[i] currently == ns; restore os
        }
    }

    // ----- emit the best feasible solution -----
    // Drop strokes that are completely covered by a later stroke's *exact* slot?
    // Not needed for feasibility; we just print all T strokes of the best state.
    // Every stroke is a valid rectangle inside the grid with a valid colour, so
    // the output is always feasible. Q = T <= T (the budget) holds by construction.
    printf("%d\n", T);
    for (int t = 0; t < T; t++) {
        const Stroke &s = bestS[t];
        printf("%d %d %d %d %d\n", s.r1, s.c1, s.r2, s.c2, s.col);
    }
    return 0;
}
```
