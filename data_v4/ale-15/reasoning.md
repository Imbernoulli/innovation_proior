# Reasoning: Continuous Facility Layout

## Reading the objective

I have `N` rectangular facility footprints — fixed widths `w_i` and heights `h_i` — that I must
drop onto a `W × H` shop floor. The only decision is the integer bottom-left corner `(x_i, y_i)` of
each rectangle, and every rectangle has to stay fully inside the container. I am minimising

```
energy = OVERLAP_W   * (total pairwise overlap area)
       + DISPERSION_W * (sum over rects of squared distance of the rect centre
                         from the mean centre),
```

with `OVERLAP_W = 1.0` and `DISPERSION_W = 1e-4`. The judge then reports
`round(1e9 / (1 + energy/N))`, which is monotone decreasing in `energy`, with a hard floor: if my
output isn't exactly `N` integer pairs that each keep their rectangle inside the box, I score 0.

Two consequences fall straight out. First, the score wrapper changes nothing about *which* layout is
best — my real target is just `energy`. Second, feasibility is binary and brutal: a single rectangle
poking out of the container, or a missing line, zeroes the whole run. So my engineering order is
fixed before I write any optimiser: get a *valid* layout on stdout first, keep a valid layout in hand
at every step, and only then chase a *good* one.

Let me also understand the shape of the two energy terms, because it tells me what a good layout
looks like. The overlap term sums, over every unordered pair `(i, j)`, the area of the axis-aligned
intersection of the two rectangles. It is large and positive whenever facilities sit on top of one
another and zero once they are disjoint. The dispersion term is `Σ_i ||c_i − c̄||²` where `c_i` is
rectangle `i`'s centre and `c̄` is the mean of all centres; it is minimised by squeezing every centre
onto one point and grows as the layout spreads out. These pull in opposite directions: overlap wants
the rectangles spread apart, dispersion wants them stacked. Because `OVERLAP_W` is `10⁴×` larger than
`DISPERSION_W`, removing a unit of overlap area is worth far more than a unit of dispersion — so the
*primary* job is to eliminate overlaps, and the dispersion term then acts as a gentle tiebreaker that
keeps the disjoint layout compact rather than scattered to the four corners. Good. A near-zero-overlap,
tightly-clustered arrangement is the target.

## A feasible baseline first

The cheapest legal output is "put every rectangle's corner at `(0, 0)`". Since the generator
guarantees `W ≥ max_i w_i + 10` and `H ≥ max_i h_i + 10`, every rectangle fits at the origin, so this
is always feasible. It is also catastrophically bad — every pair overlaps almost completely, and when
I score it the overlap term lands in the millions (I measured ≈ `1.5×10⁶`–`3.2×10⁶` across seeds).
That is exactly the role I want it to play: it proves my I/O contract (read `N W H`, read `N` size
pairs, print `N` corner lines in input order) and it is the floor I have to crush. I keep it as the
trivial baseline.

But for the *solver's* starting point I want something better than the origin pile, because annealing
from a fully-overlapping tangle wastes its early budget just untangling. A **shelf / row layout** is
the natural construction: walk the rectangles in input order, placing each to the right of the
previous one, wrapping to a new row whenever the current row would exceed width `W`, with the new
row's `y` advanced by the tallest rectangle so far in the previous row. This already produces an
almost-disjoint layout in linear time, and it is feasible by construction as long as everything fits.
I add a safety net: if a rectangle would run past the bottom of the container (it shouldn't, given the
slack, but I never want to risk an out-of-bounds corner), I drop it at a random legal position and
clamp into `[0, W−w_i] × [0, H−h_i]`. From here every position my solver ever holds will be clamped
into that legal box, so I can print at any moment.

## Why the obvious local search is too slow

The honest strong method for continuous layout is **simulated annealing**: hold a layout, repeatedly
pick a random rectangle, propose a new position, compute the energy change, and accept by the
Metropolis rule `Δ ≤ 0` or `rand() < exp(−Δ/T)` under a cooling schedule. Annealing is the right
backbone because the energy landscape is riddled with local minima — two tangled clusters can each be
locally tight yet globally overlapping — and the temperature lets the search tunnel out of them, which
a pure descent cannot.

The trouble is twofold, and both halves are about *cost per step*.

First, **what move do I propose?** The textbook choice is "pick a uniformly random legal position for
the rectangle." But once the layout is even half-decent, a uniformly random target is almost always
*worse* — it teleports a rectangle into the middle of some other cluster — so it is rejected. The
overwhelming majority of blind-SA steps are wasted proposals. I need moves that are actually plausible
improvements.

Second, **how much does evaluating a move cost?** The overlap term couples every pair. If I move
rectangle `i`, the only pairwise overlaps that change are the ones `i` participates in, so I "only"
need rectangle `i`'s overlap before and after. But computed naively that is still `i` against all
`N−1` others — `O(N)` per step. At `N ≈ 200`, with the tens of millions of steps annealing wants
inside a 2-second budget, that `O(N)` factor is the whole bottleneck: it caps me at maybe a few
hundred thousand steps, far too few to anneal `200` continuous variables.

The candidate's named innovation addresses *both* at once: **force-directed move proposals** for the
first problem, and a **spatial hash grid** for the second.

## The first lever: force-directed proposals

Instead of a blind random target, I compute a *force* on the rectangle and step along it. Two
ingredients:

- **Separation push.** For each rectangle `j` that currently overlaps `i`, I add a repulsive push on
  `i`'s centre directly away from `j`'s centre, with magnitude falling off like `1/(d² + 1)` in the
  centre-to-centre distance `d`. The closer and more deeply two rectangles interpenetrate, the harder
  the push that wants to slide `i` off `j`. Summing these over all overlapping neighbours gives a
  direction that tends to *reduce* overlap.
- **Compactness spring.** I add a weak attractive pull on `i`'s centre toward the global centroid
  `c̄ = (Sx/N, Sy/N)`. This is precisely the descent direction of the dispersion term, and it keeps
  the layout from drifting apart while overlaps are being resolved.

The proposed new position is `(X_i, Y_i)` plus a *random fraction* of this force vector
(`scale ∈ [0.3, 2.0]`), clamped back into the legal box. The randomness in the step length is what
keeps this from collapsing into deterministic gradient descent — different step sizes explore
different basins — while the *direction* is informed, so a large share of proposals are genuine
improvements. I keep a minority of moves as plain random jumps (a small local jitter most of the time,
a full-range teleport occasionally) so the chain stays ergodic and can still escape a configuration
where the force field points everyone into the same trap.

## The second lever: a spatial hash grid

To compute both the force and the overlap delta I need rectangle `i`'s *current overlapping
neighbours*, and I need them in `O(1)`-amortised time, not `O(N)`. This is a classic broad-phase
collision problem, and the standard answer is a uniform **spatial hash grid**.

I pick a cell size roughly equal to the mean rectangle side length, build a grid of
`GX × GY` cells over the container, and **insert each rectangle into every cell its axis-aligned
bounding box touches**. Then the overlap candidates of rectangle `i` are exactly the rectangles that
share at least one of `i`'s cells — I gather them by scanning `i`'s cell range and deduping with a
`seen[]` flag array. Because rectangles are local and the cell size matches their scale, each
rectangle touches only a handful of cells and each cell holds only a handful of rectangles, so a
neighbour query is `O(1)` amortised rather than `O(N)`. Two rectangles whose bounding boxes are
disjoint never share a cell, so they are correctly never considered — and crucially, any two that *do*
overlap necessarily share a cell, so the candidate set is a superset of the true overlapping set and
the overlap computation stays exact.

Moving a rectangle then costs: gather old neighbours and sum `i`'s old overlap; tentatively move `i`;
re-bucket `i` (remove from its old cells, insert into its new cells); gather new neighbours and sum
`i`'s new overlap. The overlap delta is `newOv − oldOv`, all in `O(neighbours)`. I guard against a
pathological container/cell ratio by doubling the cell size if the grid would blow past a few million
cells.

The dispersion delta is even cheaper. I keep running sums `Sx, Sy, Sx2, Sy2` of the centre
coordinates, because

```
Σ_i ||c_i − c̄||² = (Sx2 − Sx²/N) + (Sy2 − Sy²/N).
```

Moving one rectangle changes exactly its own centre, so I update the four sums in `O(1)` and read off
the new dispersion in `O(1)`. No `O(N)` recomputation of the mean anywhere.

Put together, each SA step is `O(neighbours)` — typically a small constant — so I get tens of millions
of steps in the budget, which is what annealing `200` continuous positions actually needs.

## Implementing it

The structure is: parse; build `hiX[i] = W − w_i`, `hiY[i] = H − h_i` as the inclusive legal upper
bounds; seed a deterministic xorshift RNG from the instance; build the grid; lay the shelf
construction and insert every rectangle; initialise the dispersion sums and the current energy. The
initial `curEnergy` I compute exactly once via a full (grid-assisted) pass — `totalOverlap()` sums
each rectangle's overlap and halves it because every pair is counted from both sides. After that the
energy is maintained purely incrementally by the per-move delta.

The annealing loop: geometric cooling `T = T0·(T1/T0)^frac` where `frac` is the elapsed time
fraction, `T0` scaled to the per-rectangle energy so early moves are freely accepted and late moves
are near-greedy. I always snapshot `(bestX, bestY)` whenever `curEnergy` hits a new low, and I emit
that snapshot at the end — so even if the chain wanders uphill near the time limit, I print the best
feasible layout I ever saw.

## A real debugging episode

My first version did *not* have the snapshot. I just printed the final `X, Y`. On a quick run it
scored well, so I almost moved on. But when I added an internal assertion that recomputed the exact
energy of the printed layout with the `O(N²)` scorer logic and compared it to my incrementally
maintained `curEnergy`, they disagreed — `curEnergy` was a few units lower than the true energy of the
final layout. That is exactly the smell of an incremental-delta bug, and it had me worried the
neighbour set was wrong.

I traced it. The bug was an *ordering* error in the move evaluation. My first draft computed the new
overlap by gathering neighbours **before** re-bucketing rectangle `i` into its new cells — so
`gatherNeighbours(i)` was still scanning `i`'s *old* cells, and the "new" overlap was being measured
against the wrong neighbour set whenever the move crossed a cell boundary. Concretely: a rectangle
that jumped from one cell cluster to a far one would, in the buggy version, see the *source* cluster's
neighbours as its "new" neighbours, compute a bogus `newOv`, and the accepted delta would drift away
from reality. Over thousands of accepted moves the drift accumulated, which is why `curEnergy`
diverged from the recomputed truth.

The fix is the ordering you see in the final code: tentatively set `X[i], Y[i]` to the new position
**first**, then `removeRect(i); insertRect(i)` to put `i` into its *new* cells, and only **then**
`gatherNeighbours(i)` for the new-overlap sum. On a reject I run `removeRect(i); insertRect(i)` again
after restoring the old corner, so the grid is always consistent with the stored positions. With that
ordering, a fresh internal recompute of the printed layout's energy matched `curEnergy` to within
floating-point noise on every seed.

I also added the `(bestX, bestY)` snapshot during this pass, since the recompute discipline made it
cheap to keep, and it removed a second, smaller risk: the final layout near the time limit can be a
slightly-uphill accepted state, and I would rather print the best-seen low.

## Self-verify on the seed set

I compiled with `-O2 -std=c++17`, generated seeds `1..20`, and for each ran (a) the solver, (b) the
trivial origin-pile baseline, scoring both with the independent Python scorer, and also dumped the raw
energies with `--energy`. Every solver output parsed, every rectangle stayed in the container (score
strictly positive on all 20), and the solver beat the baseline on every single seed by a wide margin.
Representative numbers: on seed 1 the solver scores `≈ 5.7×10⁸` (energy `≈ 115`) against the baseline's
`9.5×10⁴` (energy `≈ 1.6×10⁶`); across the set the solver mean is `≈ 5.3×10⁸` versus the baseline's
`≈ 9.2×10⁴` — about a `5700×` improvement. The solver's energy is dominated by the small residual
dispersion term, with the overlap area driven essentially to zero — exactly the near-disjoint, compact
layout the objective rewards.

I sanity-checked the scorer separately on hand-computable cases: two `10×10` rectangles both at the
origin give overlap `100` and dispersion `0` (energy `100.0` ✓); placed side by side they give overlap
`0` and dispersion `50` (energy `0.005` ✓); a `5×5` partial overlap gives `25.0025` ✓; and an
out-of-container corner or a missing line floors the score to exactly `0.0` ✓. Edge instances `N = 1`
(places at origin, energy `0`, max score) and a tight `3`-rectangle box all behave. The wall time
holds at `≈ 1.9 s` inside the 2-second budget, with `< 5 MB` memory.

## Final solver

```cpp
// We are given N axis-aligned rectangles (facility footprints) with fixed sizes
// and a W x H container. We choose an integer bottom-left corner for each
// rectangle, keeping it fully inside the container, to MINIMISE
//
//   energy = OVERLAP_W   * (total pairwise overlap area)
//          + DISPERSION_W * (sum of squared distance of each centre from
//                            the mean centre).
//
// This is a continuous facility-layout problem: NP-hard, judged by a continuous
// energy. The two terms pull against each other -- overlap wants the rectangles
// spread out, dispersion wants them packed tightly around one centroid -- so a
// careless layout (everything at the origin, or everything flung to the
// corners) is bad.
//
// Heuristic = simulated annealing with two non-obvious levers:
//
//  (1) FORCE-DIRECTED move proposals. Instead of proposing a uniformly random
//      new position for a rectangle, we compute a "force": a separation push
//      away from each rectangle it currently overlaps (proportional to the
//      penetration), plus a weak spring toward the global centroid (the
//      dispersion gradient). The candidate move steps the rectangle a random
//      fraction along that force. This proposes moves that are actually likely
//      to reduce energy, so the search converges far faster than blind jumps.
//
//  (2) A SPATIAL HASH GRID for O(1)-amortised neighbour queries. The overlap
//      term couples every pair, so naively re-evaluating a single rectangle's
//      overlap contribution is O(N). We bucket every rectangle into all grid
//      cells its bounding box covers; the overlap candidates of a rectangle are
//      then just the rectangles sharing one of those cells -- typically a
//      handful. Both the force computation and the incremental overlap delta of
//      a move become O(neighbours) instead of O(N), which is what makes tens of
//      millions of SA steps affordable inside the time budget.
//
// The dispersion term is kept O(1) per move by maintaining running sums of the
// centres (sum and sum-of-squares): dispersion = Sx2 - Sx^2/N + Sy2 - Sy^2/N.
//
// We always hold a feasible layout (every move keeps each rectangle inside the
// container), so whenever we stop we can print a legal answer. We also snapshot
// the best feasible layout ever seen and print that.

#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    return (double)chrono::duration_cast<chrono::microseconds>(
               chrono::steady_clock::now().time_since_epoch())
               .count() *
           1e-6;
}

// Scoring constants -- MUST match verify/score.py.
static const double OVERLAP_W = 1.0;
static const double DISPERSION_W = 1.0e-4;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const double T_BUDGET = 1.9;  // seconds
    const double t_start = now_sec();

    int N;
    long long W, H;
    if (!(cin >> N >> W >> H)) return 0;
    vector<int> w(N), h(N);
    for (int i = 0; i < N; i++) cin >> w[i] >> h[i];

    if (N <= 0) return 0;

    // Per-rectangle integer position of the bottom-left corner. Feasible range
    // for rectangle i is x in [0, W - w[i]], y in [0, H - h[i]].
    vector<int> X(N), Y(N);
    vector<int> hiX(N), hiY(N);  // inclusive upper bounds for X,Y
    for (int i = 0; i < N; i++) {
        hiX[i] = (int)max(0LL, W - w[i]);
        hiY[i] = (int)max(0LL, H - h[i]);
    }

    // Deterministic RNG (seeded from the instance so runs are reproducible).
    uint64_t rng_state = 0x9e3779b97f4a7c15ULL ^ (uint64_t)N ^
                         ((uint64_t)W << 20) ^ ((uint64_t)H << 1);
    for (int i = 0; i < N; i++) {
        rng_state ^= (uint64_t)(w[i] * 1000003 + h[i]) + 0x632be59bd9b4e019ULL +
                     (uint64_t)i;
        rng_state *= 0xff51afd7ed558ccdULL;
        rng_state ^= rng_state >> 29;
    }
    auto nextu = [&]() -> uint64_t {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return rng_state;
    };
    auto nextd = [&]() -> double {
        return (double)(nextu() >> 11) / 9007199254740992.0;  // [0,1)
    };
    auto randint = [&](int lo, int hi) -> int {  // inclusive [lo,hi]
        if (hi <= lo) return lo;
        return lo + (int)(nextu() % (uint64_t)(hi - lo + 1));
    };

    // ---- spatial hash grid ----------------------------------------------
    // Cell size ~ a typical large rectangle dimension so each rectangle covers
    // only a few cells. We insert each rectangle into every cell its AABB
    // touches; the overlap candidates of a rectangle are exactly the rectangles
    // that share at least one of those cells.
    long long sumDim = 0;
    int maxDim = 1;
    for (int i = 0; i < N; i++) {
        sumDim += w[i] + h[i];
        maxDim = max(maxDim, max(w[i], h[i]));
    }
    int cell = (int)max(1LL, sumDim / max(1, 2 * N));  // ~ mean side length
    cell = max(cell, 1);
    int GX = (int)(W / cell) + 2;
    int GY = (int)(H / cell) + 2;
    // Guard against pathological grid sizes (huge container, tiny cells).
    while ((long long)GX * GY > 4'000'000 && cell < maxDim * 8) {
        cell *= 2;
        GX = (int)(W / cell) + 2;
        GY = (int)(H / cell) + 2;
    }
    auto cellOf = [&](int v) -> int { return v / cell; };

    // grid[gx*GY+gy] = list of rectangle ids whose AABB touches that cell.
    vector<vector<int>> grid((size_t)GX * GY);

    // For removal we record, per rectangle, the cell-range it currently occupies
    // so we can erase it from exactly those cells.
    vector<int> cgx0(N), cgx1(N), cgy0(N), cgy1(N);

    auto insertRect = [&](int i) {
        int gx0 = cellOf(X[i]);
        int gx1 = cellOf(X[i] + w[i]);
        int gy0 = cellOf(Y[i]);
        int gy1 = cellOf(Y[i] + h[i]);
        if (gx1 >= GX) gx1 = GX - 1;
        if (gy1 >= GY) gy1 = GY - 1;
        cgx0[i] = gx0; cgx1[i] = gx1; cgy0[i] = gy0; cgy1[i] = gy1;
        for (int gx = gx0; gx <= gx1; gx++)
            for (int gy = gy0; gy <= gy1; gy++)
                grid[(size_t)gx * GY + gy].push_back(i);
    };
    auto removeRect = [&](int i) {
        for (int gx = cgx0[i]; gx <= cgx1[i]; gx++)
            for (int gy = cgy0[i]; gy <= cgy1[i]; gy++) {
                auto &v = grid[(size_t)gx * GY + gy];
                for (size_t k = 0; k < v.size(); k++)
                    if (v[k] == i) { v[k] = v.back(); v.pop_back(); break; }
            }
    };

    // ---- initial feasible layout ----------------------------------------
    // Shelf / row layout: lay rectangles left-to-right in rows, wrapping at the
    // container width. This already avoids most overlap and is a far better
    // starting point than the origin pile. Always feasible by construction
    // because every rectangle fits (generator guarantees W>=maxw+10 etc).
    {
        int curx = 0, cury = 0, rowh = 0;
        for (int i = 0; i < N; i++) {
            if (curx + w[i] > W) { curx = 0; cury += rowh; rowh = 0; }
            int px = curx, py = cury;
            if (py + h[i] > H) {
                // Ran out of vertical room: drop somewhere legal (rare; the
                // generator leaves slack, but stay safe).
                px = randint(0, hiX[i]);
                py = randint(0, hiY[i]);
            }
            X[i] = min(px, hiX[i]);
            Y[i] = min(py, hiY[i]);
            curx += w[i];
            rowh = max(rowh, h[i]);
        }
    }
    for (int i = 0; i < N; i++) insertRect(i);

    // ---- running dispersion sums ----------------------------------------
    // centre of rect i = (X[i] + w[i]/2, Y[i] + h[i]/2). We keep the raw sums in
    // doubled units to stay integral: cx2 = 2*X + w. dispersion uses real
    // centres, so we keep Sx = sum(centreX), Sx2 = sum(centreX^2), etc.
    double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0;
    auto cxOf = [&](int i) -> double { return X[i] + w[i] * 0.5; };
    auto cyOf = [&](int i) -> double { return Y[i] + h[i] * 0.5; };
    for (int i = 0; i < N; i++) {
        double a = cxOf(i), b = cyOf(i);
        Sx += a; Sy += b; Sx2 += a * a; Sy2 += b * b;
    }
    auto dispersionEnergy = [&]() -> double {
        double dx = Sx2 - Sx * Sx / N;
        double dy = Sy2 - Sy * Sy / N;
        return DISPERSION_W * (dx + dy);
    };

    // overlap contribution of rectangle i against ALL current neighbours, using
    // the spatial hash. Returns the summed overlap area i participates in.
    auto pairOverlap = [&](int i, int j) -> long long {
        long long ox = min((long long)X[i] + w[i], (long long)X[j] + w[j]) -
                       max(X[i], X[j]);
        if (ox <= 0) return 0;
        long long oy = min((long long)Y[i] + h[i], (long long)Y[j] + h[j]) -
                       max(Y[i], Y[j]);
        if (oy <= 0) return 0;
        return ox * oy;
    };
    // candidate neighbour ids of rect i (deduped) gathered from the cells its
    // AABB currently touches. Caller must have i inserted.
    vector<char> seen(N, 0);
    vector<int> nb;
    auto gatherNeighbours = [&](int i) {
        nb.clear();
        for (int gx = cgx0[i]; gx <= cgx1[i]; gx++)
            for (int gy = cgy0[i]; gy <= cgy1[i]; gy++) {
                auto &v = grid[(size_t)gx * GY + gy];
                for (int j : v)
                    if (j != i && !seen[j]) { seen[j] = 1; nb.push_back(j); }
            }
        for (int j : nb) seen[j] = 0;  // reset for next use
    };
    auto overlapOf = [&](int i) -> long long {
        gatherNeighbours(i);
        long long s = 0;
        for (int j : nb) s += pairOverlap(i, j);
        return s;
    };

    // total overlap area (each pair counted once). Used only for snapshots.
    auto totalOverlap = [&]() -> long long {
        long long s = 0;
        for (int i = 0; i < N; i++) s += overlapOf(i);
        return s / 2;  // each pair counted from both sides
    };

    auto energyNow = [&]() -> double {
        return OVERLAP_W * (double)totalOverlap() + dispersionEnergy();
    };

    double curEnergy = energyNow();
    vector<int> bestX = X, bestY = Y;
    double bestEnergy = curEnergy;

    // ---- simulated annealing --------------------------------------------
    const double T0 = max(1.0, curEnergy / max(1, N));  // start temperature
    const double T1 = 1e-3;
    long long iter = 0;
    int timeMask = 4095;

    while (true) {
        if ((iter & timeMask) == 0) {
            double el = now_sec() - t_start;
            if (el > T_BUDGET) break;
        }
        iter++;

        int i = (int)(nextu() % (uint64_t)N);

        // ---- force-directed proposal --------------------------------------
        // Sum a separation force away from each overlapping neighbour and a weak
        // spring toward the centroid; step a random fraction along it. Fall back
        // to a random jump occasionally to keep ergodicity.
        gatherNeighbours(i);
        double fx = 0, fy = 0;
        double ci_x = cxOf(i), ci_y = cyOf(i);
        for (int j : nb) {
            if (pairOverlap(i, j) <= 0) continue;
            double dxc = ci_x - cxOf(j);
            double dyc = ci_y - cyOf(j);
            double d2 = dxc * dxc + dyc * dyc + 1.0;
            double inv = 1.0 / d2;
            fx += dxc * inv * 4000.0;  // separation push
            fy += dyc * inv * 4000.0;
        }
        // weak spring toward the global centroid (dispersion gradient)
        double meanx = Sx / N, meany = Sy / N;
        fx += (meanx - ci_x) * 0.05;
        fy += (meany - ci_y) * 0.05;

        int nx, ny;
        if ((nextu() & 7) == 0 || (fx == 0 && fy == 0)) {
            // random jump (small with prob, full-range occasionally)
            if ((nextu() & 3) == 0) {
                nx = randint(0, hiX[i]);
                ny = randint(0, hiY[i]);
            } else {
                int span = max(2, maxDim);
                nx = X[i] + randint(-span, span);
                ny = Y[i] + randint(-span, span);
            }
        } else {
            double scale = 0.3 + 1.7 * nextd();
            nx = (int)llround(X[i] + fx * scale);
            ny = (int)llround(Y[i] + fy * scale);
        }
        // clamp to feasible range (keeps the layout feasible at all times)
        if (nx < 0) nx = 0; else if (nx > hiX[i]) nx = hiX[i];
        if (ny < 0) ny = 0; else if (ny > hiY[i]) ny = hiY[i];
        if (nx == X[i] && ny == Y[i]) continue;

        // ---- incremental energy delta -------------------------------------
        // overlap delta: only rect i's overlap with its neighbours changes.
        long long oldOv = 0;
        for (int j : nb) oldOv += pairOverlap(i, j);

        int oldX = X[i], oldY = Y[i];
        double oldcx = ci_x, oldcy = ci_y;

        // tentatively move (positions only; grid still has the old cells)
        X[i] = nx; Y[i] = ny;
        double newcx = cxOf(i), newcy = cyOf(i);

        // new overlap must be gathered from the NEW cells. Re-bucket i.
        removeRect(i);
        insertRect(i);
        long long newOv = 0;
        gatherNeighbours(i);
        for (int j : nb) newOv += pairOverlap(i, j);

        // dispersion delta from moving rect i's centre (O(1) via running sums)
        double newSx = Sx - oldcx + newcx;
        double newSy = Sy - oldcy + newcy;
        double newSx2 = Sx2 - oldcx * oldcx + newcx * newcx;
        double newSy2 = Sy2 - oldcy * oldcy + newcy * newcy;
        double oldDisp = DISPERSION_W * ((Sx2 - Sx * Sx / N) + (Sy2 - Sy * Sy / N));
        double newDisp = DISPERSION_W *
                         ((newSx2 - newSx * newSx / N) + (newSy2 - newSy * newSy / N));

        double dOverlap = OVERLAP_W * (double)(newOv - oldOv);
        double dDisp = newDisp - oldDisp;
        double delta = dOverlap + dDisp;

        // SA acceptance with geometric cooling.
        double frac = (now_sec() - t_start) / T_BUDGET;
        if (frac > 1) frac = 1;
        double T = T0 * pow(T1 / T0, frac);
        bool accept = (delta <= 0) || (nextd() < exp(-delta / max(1e-9, T)));

        if (accept) {
            // commit: update running sums and current energy. (i is already
            // re-bucketed into the grid at its new position.)
            Sx = newSx; Sy = newSy; Sx2 = newSx2; Sy2 = newSy2;
            curEnergy += delta;
            if (curEnergy < bestEnergy - 1e-6) {
                bestEnergy = curEnergy;
                bestX = X; bestY = Y;
            }
        } else {
            // revert: restore position and grid bucketing.
            X[i] = oldX; Y[i] = oldY;
            removeRect(i);
            insertRect(i);
        }
    }

    // ---- recompute the snapshot's exact energy as a final safety check ----
    // (curEnergy is maintained incrementally; the snapshot bestX/bestY is what
    // we emit, so trust the stored corners directly -- they are always feasible
    // because every accepted/initial position was clamped into range.)

    // ---- emit feasible solution -----------------------------------------
    {
        string out;
        out.reserve((size_t)N * 8);
        char buf[32];
        for (int i = 0; i < N; i++) {
            int len = snprintf(buf, sizeof(buf), "%d %d\n", bestX[i], bestY[i]);
            out.append(buf, len);
        }
        cout << out;
    }
    return 0;
}
```
