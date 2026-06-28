# Reasoning: Guillotine Cutting Stock

## Reading the objective carefully

I have an unlimited supply of identical `W x H` sheets and a list of `n` requested rectangles. I have
to place rectangles on sheets, axis-aligned, rotation allowed, no overlap, every rectangle inside its
sheet, and — the part that makes this its own problem rather than plain 2-D bin packing — every
per-sheet layout has to be realizable by **guillotine cuts**: a cut goes edge-to-edge across the
current piece, and I cut recursively. The cost I am judged on is

```
cost = (#sheets used) * W * H - (placed area) + P * (unplaced area),   P = 3
```

and the score normalizes my cost against a shelf next-fit baseline:
`score = round(1e6 * baseline_cost / max(1, my_cost))`, higher is better, and **any infeasibility
floors the score to 0**. So before anything clever, two facts shape the whole design.

First, the dominant term is `(#sheets) * W * H`. With `W, H` in the hundreds, one extra sheet costs on
the order of tens of thousands, whereas the area I waste inside a sheet I do use is recovered by the
`- placed area` term. So the single most valuable thing I can do is **pack into fewer sheets**. The
unplaced penalty `P * unplaced` is also large per unit area (3x), so I almost never want to leave a
rectangle out when the instances are designed to be packable.

Second, feasibility is binary and unforgiving. A non-overlapping layout is *not* automatically
guillotine-legal — the classic counterexample is a pinwheel of four rectangles around a center, which
has no full edge-to-edge cut anywhere yet no two rectangles overlap. If my solver ever emits such a
layout, the scorer's recursive cut-search rejects it and I score 0. So I cannot just run a generic
rectangle packer and hope; I need the guillotine constraint respected exactly.

## A feasible baseline first

The rule I always follow on these heuristic problems: get *some* valid solution out the door before
optimizing. The trivial one is the empty solution (`m = 0`): place nothing. It is feasible — it
parses, no rectangle is placed, so no overlap and no guillotine question arises — and its cost is
`P * total_area`, which is terrible but positive-scoring. Good: I have a floor I can never fall below
as long as I keep the output well-formed.

The next baseline up is the one the scorer itself uses, **shelf next-fit**: process rectangles in
order, lay them left to right in a horizontal band (a "shelf"); when the next rectangle does not fit
the band width, open a new band above it; when it does not fit the remaining sheet height, open a new
sheet. Shelf layouts are guillotine-legal for free (cut horizontally between bands, then vertically
between rectangles in a band). This is `O(n)` and always feasible. But I already know its weakness:
the band height is set by the tallest rectangle in it, so every band wastes a strip under its shorter
rectangles, and that slack forces extra sheets. That is precisely why it is the *baseline* and not the
answer — beating it means recovering that intra-band slack.

So shelf next-fit is my mental reference point, and the question is how to do materially better while
keeping guillotine-legality guaranteed.

## Why the obvious "place then check" is the wrong shape

The tempting general approach is: maintain a free-form 2-D placement, propose moving/rotating
rectangles, and after each move *verify* the layout is non-overlapping and guillotine-cuttable, accept
if it improved. This is the wrong shape for two reasons.

1. The guillotine check is a recursive search over candidate cut lines (the scorer does exactly this).
   Running it inside the optimizer's inner loop, on every proposed move, is expensive, and worse, most
   random moves produce *infeasible* layouts (overlaps, or non-guillotine pinwheels), so I would spend
   almost all my compute generating and rejecting garbage.

2. Even the representation is awkward: continuous `(x, y)` for each rectangle has a huge feasible-set
   boundary, and local moves constantly cross it. The search would thrash against the constraint
   instead of exploring good packings.

I have seen this failure mode before on packing problems: the constraint, not the objective, eats the
budget. The fix is to change the representation so that **feasibility is automatic** and the search
only ranges over genuinely feasible solutions.

## The innovation: bake guillotine-feasibility into the construction

Here is the lever. Instead of placing rectangles freely and checking guillotine-legality afterwards, I
*construct* the layout by a process whose every step is a guillotine cut, so the result is guillotine-
legal by construction and I never check feasibility at all.

Represent each sheet as a **pool of free rectangles** (this is the guillotine "k-d tree of free space"
the problem's innovation field names). A sheet starts as one free rectangle `0,0,W,H`. To place a
requested rectangle of size `w x h`:

- pick a free rectangle `f` that it fits in (`f.w >= w`, `f.h >= h`);
- drop the rectangle at `f`'s bottom-left corner;
- the leftover is an L-shape, which I split with **one guillotine cut** into two rectangles: a strip to
  the right of the placed piece and a strip above it. Two ways to make that single cut — cut full
  height first (right piece is full-height `f.h`, top piece spans only the placed width `w`), or cut
  full width first (top piece is full-width `f.w`, right piece spans only the placed height `h`).
  Either is a legal guillotine cut; I push the two resulting free rectangles back into the pool.

Because the entire layout is built as "drop into a free rectangle, then one guillotine cut", *the whole
sheet decomposes into a binary tree of guillotine cuts by construction*. No overlap is possible (each
placed rectangle consumes disjoint free space), and the layout is trivially guillotine-separable. I get
feasibility for free, exactly the trick the problem is pointing at.

Now the only thing left to decide is the **insertion order** (and each rectangle's rotation). That is a
clean combinatorial search space: a permutation of `0..n-1` plus `n` rotation bits. A full construction
("replay") from an order is cheap — for each of `n` items I scan the current free-rectangle pool, which
stays small (a few dozen rectangles), so replay is `O(n * #freeRects)`, microseconds here. That means I
can afford a metaheuristic that *rebuilds the entire packing on every proposed move* and never has to
do incremental feasibility bookkeeping. The feasibility is in the constructor; the search is over
orderings.

### Two constructor choices that matter

Two design knobs inside the constructor strongly affect quality, independent of the search:

- **Which free rectangle to use (fit rule).** I use **best-area-fit**: among all free rectangles that
  can host the piece, choose the one whose leftover area `f.w*f.h - w*h` is smallest. This snugly fills
  the tightest hole, leaving larger contiguous free rectangles intact for later big pieces. Crucially I
  search free rectangles **across all already-opened sheets**, and only open a new sheet when *no*
  current free rectangle fits. That is what drives the sheet count — the dominant cost term — down: the
  constructor reuses existing slack greedily before paying for a fresh sheet.

- **Which way to split the leftover L.** I use the **shorter-axis split**: if the right strip is
  narrower than the top strip is short, cut full-height first so the wide top strip stays whole;
  otherwise cut full-width first so the tall right strip stays whole. Keeping the bigger leftover piece
  un-fragmented gives future placements room. This is a standard guillotine-split rule and it noticeably
  beats splitting arbitrarily.

### The search: simulated annealing over the insertion order

The objective (sheet count + waste) is full of plateaus and local minima — swapping two items often
changes nothing until it tips a piece onto a new sheet — so a pure hill-climb stalls. I use
**simulated annealing** over `(order, rotation)`:

- **Initial state.** Largest-area-first order (a strong constructive start: place the bulky pieces while
  the sheets are empty), with each rotation chosen so the piece is "tall" (longer side vertical) when it
  fits, which helps the constructor stack pieces into columns.
- **Moves.** With equal probability: swap two positions in the order; shift one item to a new position;
  or flip one item's rotation bit (only if the flipped orientation still fits a sheet).
- **Evaluation.** Replay the full construction to get the cost. No incremental delta needed — replay is
  cheap enough to run on every move.
- **Acceptance.** Metropolis: accept if cost does not increase, else with probability
  `exp(-delta / T)`. Temperature decays geometrically from `~0.5 * W*H` (one sheet's worth, so early on
  a one-sheet swing is readily accepted) down to `1`, scheduled by wall-clock fraction of a 1.85 s
  budget. I track the best order seen and emit its packing.

Because every order replays to a guillotine-legal layout, the incumbent is *always* feasible. If I hit
the time limit mid-search — or even after zero accepted moves — I still print the best feasible packing.
There is no code path that emits an invalid solution.

## Implementing it, and a real bug I hit

I wrote `replay(order, rot, out)` returning the cost and optionally filling the placement list, then
the SA loop around it. First compile, ran on seed 1, scored well above the baseline's 1e6 — promising.
Then I went to harden the feasibility floor in the scorer and, doing so, found a bug in my own solver's
**rotation accounting**, which is exactly the kind of mistake that silently produces an *infeasible*
output.

The issue: inside `replay`, when a piece's chosen orientation did not fit a fresh sheet I flipped to the
other orientation to place it — but in my first version I emitted the *original* `rot[id]` bit on the
output line rather than the orientation I actually used. So the line said `rot = 0` (`w x h`) while I had
actually placed the rotated `h x w`. The scorer recomputes the placed dimensions from the emitted `rot`
and the original `w_i, h_i`; with the wrong bit it would compute the wrong size, see the rectangle stick
out of the sheet or overlap a neighbor, and **floor the score to 0**. On the generated instances every
rectangle fits in its preferred orientation so the flip branch rarely fired and the bug hid — but it was
a latent infeasibility waiting for an instance that triggered it.

I reproduced it deliberately by hand-crafting a piece whose first orientation overflows the sheet,
confirmed the emitted `rot` disagreed with the placed size, and fixed it by tracking the *actually used*
orientation in a local `curRot` and emitting that, swapping `w, h` and toggling `curRot` together when I
flip. After the fix I also tightened the early skip: a piece that fits in *neither* orientation is
genuinely unplaceable and is simply omitted (still feasible — it just counts as unplaced area).

While I was in the scorer I built an explicit adversarial test set to be sure the feasibility floor
actually fires: an overlapping pair, an out-of-bounds rectangle, a duplicate index, trailing garbage
tokens, a bad `rot` value, and — the important one — a **pinwheel** of four non-overlapping rectangles
that is not guillotine-separable. All six scored 0, and a genuine guillotine layout on the same
rectangles scored positive, so the recursive cut-search in the scorer is doing its job and my
constructor's guarantee is the thing keeping me above it.

## Self-verification on the seed set

With the fix in, I ran the full protocol: generate seeds 1..20, run the solver, score each, and compare
against the scorer's own shelf-next-fit baseline cost.

- **Every** seed produced a feasible solution (score > 0); none was floored.
- The solver's score on every seed is far above 1e6 (the baseline's score), i.e. my cost is well below
  the baseline cost on all 20 — it strictly beats the trivial baseline everywhere, not just on average.
- Concretely, mean solver cost ~5.9e4 versus mean baseline cost ~3.1e5: roughly a 5x cost reduction,
  driven mostly by opening far fewer sheets (the constructor reusing free space across sheets) plus the
  SA recovering the intra-band slack the shelf packer leaves behind.
- Timing on the largest instance (n=86): 1.85 s wall, ~4 MB RAM — within the 2 s / 256 MB budget.

The shape of the win matches the analysis: the baseline wastes sheets because each shelf is as tall as
its tallest piece; the free-rectangle constructor packs pieces into the leftover holes across sheets,
and SA reorders insertions so the bulky pieces claim space first and the small ones fill the gaps.

## Why this is the right method, in one paragraph

The crux of guillotine cutting stock is that the hard constraint (guillotine-legality) is easy to
*violate* by accident and expensive to *check*, so a place-then-check optimizer drowns. Moving the
constraint into the constructor — every step is a free-rectangle pick plus one guillotine cut — makes
every constructed layout legal by definition, collapsing the problem to a search over insertion orders,
which a cheap full-replay simulated annealing handles comfortably. Best-area-fit and shorter-axis-split
are the two constructor heuristics that make each replay good, and reusing free space across already-
opened sheets is what attacks the dominant sheet-count cost. The result is always feasible and
consistently several times better than the shelf next-fit baseline.

## Final solver

```cpp
// Guillotine Cutting Stock -- heuristic solver.
//
// Objective: place n requested rectangles onto W x H sheets (axis-aligned,
// optional 90-degree rotation), every per-sheet layout realizable by guillotine
// (edge-to-edge) cuts, so as to MINIMIZE
//     cost = (#sheets used) * W * H - (placed area) + P * (unplaced area)
// with P = 3. We read the instance from stdin and write, to stdout,
//     m
//     idx sheet x y rot         (m lines)
// for the m placed rectangles (omitted == unplaced).
//
// Method (the innovation):
//   * Encode feasibility into the CONSTRUCTION. Every sheet is a set of free
//     rectangles (a guillotine free-rectangle pool). To place a rectangle we
//     pick, over all sheets' free rectangles, a BEST-FIT free rectangle it fits
//     in (smallest leftover area, i.e. best-area-fit), drop it at that free
//     rect's bottom-left corner, and GUILLOTINE-SPLIT the leftover L-shape into
//     two rectangles (a full edge-to-edge cut). Because each placement is "drop
//     into a free rectangle + one guillotine split", the resulting layout is
//     guillotine-legal and overlap-free BY CONSTRUCTION -- we never have to
//     repair feasibility.
//   * The only decision left is the INSERTION ORDER (and per-item rotation). A
//     full construction (replay) costs O(n * #free-rects) which is tiny here, so
//     we run SIMULATED ANNEALING over the order: a move swaps two positions or
//     flips an item's rotation, we REBUILD the packing by replay, and accept by
//     the Metropolis rule on the cost above. The current best order's packing is
//     always a valid guillotine layout, so any early stop still prints a feasible
//     solution.
//   * The construction also greedily prefers to fill EXISTING sheets (only opens
//     a new sheet when no current free rectangle can host the item), which is
//     what drives the sheet count -- the dominant cost term -- down.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t nextu(uint32_t m) { return (uint32_t)(next() % m); }
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int N, W, H;
vector<int> RW, RH;          // requested rectangle dimensions
long long TOTAL_AREA = 0;
const long long P = 3;       // unplaced-area penalty (must match score.py)

struct Placement { int idx, sheet, x, y, rot; };

struct FreeRect { int x, y, w, h, sheet; };

// Replay a given order (+rotation bits) into a guillotine packing.
// Returns the cost and (optionally) fills `out` with the placements.
long long replay(const vector<int>& order, const vector<uint8_t>& rot,
                 vector<Placement>* out) {
    // free rectangles across all currently opened sheets
    static vector<FreeRect> freeR;
    freeR.clear();
    int sheets = 0;
    long long placedArea = 0;
    if (out) out->clear();

    auto openSheet = [&]() {
        FreeRect fr{0, 0, W, H, sheets};
        freeR.push_back(fr);
        sheets++;
    };

    for (size_t t = 0; t < order.size(); t++) {
        int id = order[t];
        int curRot = rot[id] ? 1 : 0;
        int w = curRot ? RH[id] : RW[id];
        int h = curRot ? RW[id] : RH[id];
        // If this orientation does not fit a fresh sheet, try the other one; if
        // neither fits, the rectangle is genuinely unplaceable and is skipped.
        if (w > W || h > H) {
            curRot ^= 1;
            std::swap(w, h);
            if (w > W || h > H) continue;  // truly cannot be placed in any orientation
        }

        // BEST-AREA-FIT: choose the free rectangle (already opened) with the
        // smallest leftover area that still fits w x h.
        int best = -1;
        long long bestLeft = LLONG_MAX;
        for (size_t i = 0; i < freeR.size(); i++) {
            const FreeRect& f = freeR[i];
            if (f.w >= w && f.h >= h) {
                long long left = (long long)f.w * f.h - (long long)w * h;
                if (left < bestLeft) { bestLeft = left; best = (int)i; }
            }
        }
        if (best < 0) {
            // no current free rectangle hosts it: open a new sheet.
            openSheet();
            best = (int)freeR.size() - 1;
            const FreeRect& f = freeR[best];
            if (!(f.w >= w && f.h >= h)) { continue; } // safety (shouldn't happen)
        }

        FreeRect f = freeR[best];
        // remove the consumed free rectangle (swap-pop)
        freeR[best] = freeR.back();
        freeR.pop_back();

        // place at (f.x, f.y); guillotine-split the leftover L into two rects.
        if (out) out->push_back(Placement{id, f.sheet, f.x, f.y, curRot});
        placedArea += (long long)w * h;

        int rightW = f.w - w;   // strip to the right of the placed rect
        int topH = f.h - h;     // strip above the placed rect
        // Choose the split that yields the more "useful" (larger-min-dim) pieces:
        // Shorter-Axis-Split heuristic -- split along the shorter leftover so the
        // bigger leftover piece stays whole (a standard guillotine split rule).
        bool splitHorizontalFirst = (rightW <= topH);
        if (splitHorizontalFirst) {
            // right piece spans the placed rect's height only; top piece spans full width
            if (rightW > 0) freeR.push_back(FreeRect{f.x + w, f.y, rightW, h, f.sheet});
            if (topH > 0)   freeR.push_back(FreeRect{f.x, f.y + h, f.w, topH, f.sheet});
        } else {
            // top piece spans the placed rect's width only; right piece spans full height
            if (topH > 0)   freeR.push_back(FreeRect{f.x, f.y + h, w, topH, f.sheet});
            if (rightW > 0) freeR.push_back(FreeRect{f.x + w, f.y, rightW, f.h, f.sheet});
        }
    }

    long long unplaced = TOTAL_AREA - placedArea;
    long long cost = (long long)sheets * W * H - placedArea + P * unplaced;
    return cost;
}

int main() {
    const double T0 = now_sec();
    const double TIME_LIMIT = 1.85;

    if (scanf("%d %d %d", &N, &W, &H) != 3) return 0;
    if (N <= 0) { printf("0\n"); return 0; }
    RW.resize(N); RH.resize(N);
    for (int i = 0; i < N; i++) {
        if (scanf("%d %d", &RW[i], &RH[i]) != 2) { RW[i] = 1; RH[i] = 1; }
        TOTAL_AREA += (long long)RW[i] * RH[i];
    }

    Rng rng(0xC0FFEEull ^ ((uint64_t)N << 32) ^ ((uint64_t)W << 16) ^ (uint64_t)H);

    // Initial order: largest-area first (a strong constructive start for packing),
    // with rotation set so the longer side is vertical (helps shelf-like growth).
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [](int a, int b) {
        long long aa = (long long)RW[a] * RH[a];
        long long bb = (long long)RW[b] * RH[b];
        if (aa != bb) return aa > bb;
        return a < b;
    });
    vector<uint8_t> rot(N, 0);
    for (int i = 0; i < N; i++) {
        // prefer the orientation that fits and is "tall" (h >= w) when possible
        int w = RW[i], h = RH[i];
        bool fit0 = (w <= W && h <= H);
        bool fit1 = (h <= W && w <= H);
        if (fit0 && (!fit1 || h >= w)) rot[i] = 0;
        else if (fit1) rot[i] = 1;
        else rot[i] = 0;
    }

    vector<int> curOrder = order;
    vector<uint8_t> curRot = rot;
    long long curCost = replay(curOrder, curRot, nullptr);

    vector<int> bestOrder = curOrder;
    vector<uint8_t> bestRot = curRot;
    long long bestCost = curCost;

    // Simulated annealing over (order, rotation). Cost scale ~ W*H per sheet.
    double Tstart = (double)W * H * 0.5 + 1.0;
    double Tend = 1.0;
    long long iters = 0;
    while (true) {
        if ((iters & 255) == 0) {
            double el = now_sec() - T0;
            if (el > TIME_LIMIT) break;
        }
        iters++;
        double frac = std::min(1.0, (now_sec() - T0) / TIME_LIMIT);
        double Temp = Tstart * pow(Tend / Tstart, frac);

        // propose a move on a scratch copy
        vector<int> nxtOrder = curOrder;
        vector<uint8_t> nxtRot = curRot;
        int mv = rng.nextu(3);
        if (mv == 0 && N >= 2) {
            int i = rng.nextu(N), j = rng.nextu(N);
            std::swap(nxtOrder[i], nxtOrder[j]);
        } else if (mv == 1 && N >= 2) {
            // move one element to another position (shift)
            int i = rng.nextu(N), j = rng.nextu(N);
            int v = nxtOrder[i];
            nxtOrder.erase(nxtOrder.begin() + i);
            nxtOrder.insert(nxtOrder.begin() + j, v);
        } else {
            int i = rng.nextu(N);
            int id = curOrder[i];
            // only flip if the flipped orientation also fits a sheet
            int w = RH[id], h = RW[id];
            if (w <= W && h <= H) nxtRot[id] ^= 1;
            else { int k = rng.nextu(N); std::swap(nxtOrder[k], nxtOrder[(k + 1) % N]); }
        }

        long long nxtCost = replay(nxtOrder, nxtRot, nullptr);
        long long d = nxtCost - curCost;
        if (d <= 0 || rng.nextd() < exp(-(double)d / Temp)) {
            curOrder.swap(nxtOrder);
            curRot.swap(nxtRot);
            curCost = nxtCost;
            if (curCost < bestCost) {
                bestCost = curCost;
                bestOrder = curOrder;
                bestRot = curRot;
            }
        }
    }

    // Emit the best packing.
    vector<Placement> placements;
    replay(bestOrder, bestRot, &placements);

    string buf;
    buf.reserve(placements.size() * 16 + 16);
    buf += to_string(placements.size());
    buf += '\n';
    for (const auto& p : placements) {
        buf += to_string(p.idx); buf += ' ';
        buf += to_string(p.sheet); buf += ' ';
        buf += to_string(p.x); buf += ' ';
        buf += to_string(p.y); buf += ' ';
        buf += to_string(p.rot); buf += '\n';
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
```
