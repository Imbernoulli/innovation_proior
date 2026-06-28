# Reasoning: Continuous Point Placement then Snap

## Reading the objective

I am handed a square domain `[0, L] x [0, L]`, a cloud of a few hundred integer
**demand** points inside it, and a count `k`. I must drop `k` **facility** points
at **integer** coordinates and minimize

```
E = coverage - LAMBDA * dispersion,   LAMBDA = 0.5
coverage   = sum over demand d of  dist(d, nearest facility)
dispersion = sum over facility i of min_{j != i} dist(facility_i, facility_j)
```

So two forces are wired against each other. The coverage term wants every demand
point to have a facility close by — that pulls facilities *onto* the demand,
especially onto the dense clusters. The dispersion term is the total
nearest-neighbour spacing of the facilities, and it is *subtracted*, so making
the facilities spread out *lowers* `E`. If I only minimized coverage I would pile
facilities on top of the densest blobs (great coverage, near-zero spacing,
dispersion reward thrown away). If I only maximized dispersion I would lay a
perfectly even lattice that ignores where the demand actually sits (great
spacing, terrible coverage). The good answer is a compromise: hug the clusters,
but keep the facilities mutually apart. That is essentially a blue-noise /
centroidal-Voronoi placement with a coverage pull — a non-convex continuous
problem, and on top of that I have to land on the integer grid at the end.

The scorer's reference is the **uniform grid**: `k` facilities on a near-square
lattice (`g = round(sqrt(k))` columns) spanning the whole box. The score is
`round(1e6 * E_ref / E_solver)` when feasible with `E_solver > 0`, and the
feasibility floor is `0` if the output is the wrong length or any coordinate
leaves `[0, L]`. So the grid scores ~1e6 and I need `E_solver < E_ref` to beat
it. The grid is a deliberately blind placement — it spends points on empty
corners of the box — so a demand-aware placement should win comfortably. My job:
get `E` well below the grid's.

## The always-feasible baseline first

Before anything clever, I want a placement I can print no matter what. Feasibility
here is trivial to satisfy — any `k` integer points inside `[0, L]^2`. The
safest such placement is the grid itself (it is literally the reference). So my
first commitment is: hold a feasible configuration at all times, clamp every
coordinate to `[0, L]` whenever I touch it, and make sure that if the time budget
runs out at any instant I still emit `k` legal points. I also guard the
degenerate inputs: `k <= 0` prints nothing, and `m == 0` (no demand at all) falls
straight back to the grid, because with no demand the coverage term vanishes and
spreading on a lattice is as good as anything.

With that floor in place I can be aggressive in the middle without ever risking a
zero.

## Why the obvious local search is too slow / weak

The tempting thing is to work directly on integer coordinates: start somewhere,
and repeatedly nudge a facility by `+-1` in `x` or `y`, accept if `E` drops. That
is a pure integer hill-climb. Two problems:

1. **It is slow where it matters.** Each candidate move re-evaluates the energy,
   and the coverage term is `O(m*k)` to recompute from scratch (every demand
   against every facility). With `m` in the hundreds and `k` up to 40, and the
   domain `L` up to 1000, a `+-1` walk would need an enormous number of steps to
   migrate a facility across the box to the cluster it should be serving. Most of
   the budget would be spent crawling, not optimizing.

2. **It gets stuck in the lattice of integers.** The objective is fundamentally a
   continuous, smooth-ish surface (Euclidean distances). A `+-1` walk feels none
   of that structure — it just probes 26 neighbours and stops at the first local
   integer minimum, which on a coarse `1000 x 1000` grid is almost always far
   from good. The natural geometry — "this facility should sit at the centroid of
   the demand it serves" — is invisible to it.

So integer-first local search is the weak baseline. The structure of the problem
is *continuous*; I should optimize there, where I know the descent directions in
closed form, and only discretize at the very end.

## The innovation: relax continuously, then snap with a structured local search

This is the lever the problem is built around. Optimize the facilities as
**real-valued** points using the two known descent directions, then **snap** the
good continuous configuration to the integer grid with a small *structured* local
search instead of blind rounding.

**Continuous phase — Lloyd + repulsion.**

- *Coverage descent is Lloyd's algorithm.* For the coverage term alone, fix the
  assignment of demand points to their nearest facility (each facility owns a
  Voronoi cell). Given that cell, the facility position that minimizes the sum of
  distances-to-its-demand is well approximated by the cell **centroid** (the
  exact minimizer of the *squared*-distance version; for plain distance the
  centroid is still a strong, cheap descent target). Iterating "assign to nearest
  facility, move each facility to its cell centroid" is exactly Lloyd's
  algorithm. It monotonically pushes facilities into the demand and converges to
  a centroidal Voronoi tessellation. This is the standard, strong way to make
  points hug a distribution, and it moves a facility across the box in *one* step
  rather than a thousand `+-1` nudges.

- *Dispersion ascent is repulsion.* Lloyd alone leaves facilities clumped where
  the demand is densest, which throws away the dispersion reward. So after each
  Lloyd step I add a **repulsion** push: each facility steps a little away from
  its nearest neighbour (a blue-noise move). That raises the nearest-neighbour
  spacing — exactly the dispersion term. I *anneal* the repulsion strength down
  linearly over the iterations, so the early iterations spread things out and the
  late iterations let coverage settle. Annealing it down also keeps `E` reliably
  positive (coverage stays the dominant term), which matters because the scorer's
  ratio wants a positive `E_solver`.

- *Restarts.* The surface is non-convex, so I do several random restarts. The
  first restart seeds with a **k-means++** spread (first facility on a random
  demand point, each subsequent one sampled proportional to squared distance from
  the chosen set) — a strong, demand-aware initialization. Later restarts seed
  uniformly inside the demand bounding box for diversity. I keep the best
  continuous configuration by `E`.

**Snap phase — structured integer snap (not naive rounding).** A continuous point
lands inside a unit cell whose **4 integer corners** are `(floor x, floor y)`,
`(ceil x, floor y)`, `(floor x, ceil y)`, `(ceil x, ceil y)`. Naive rounding
picks the nearest corner independently per facility, ignoring that the corner
choices *interact* through the shared energy. Instead I run **coordinate descent
on the full integer energy**: for each facility, try all 4 surrounding corners,
keep the one that lowers the full discretized `E`, and sweep over all facilities
until none improves. This is cheap (4 candidate evaluations per facility per
sweep) and it captures the coupling that per-point rounding misses. Crucially it
never raises the energy relative to its starting snapped configuration, so the
snap can only help.

**Integer polish.** Finally a short hill-climb of small integer nudges
(4-/8-neighbour moves) at a step size that starts at `~L/50` and halves whenever a
full sweep makes no progress. This shakes out the last bit of slack at
progressively finer scales, accepting only strict improvements, always clamping
to `[0, L]`. Any time-limit cutoff during polish still leaves a feasible
placement.

## Implementing it, then a real debugging episode

I wrote the continuous relaxation, the structured snap, and the polish, split the
~2s budget (roughly 55% to the continuous restarts, the rest to snap + polish),
and compiled. First end-to-end run on seed 1 — it produced a placement and the
scorer accepted it. Good. But two things bit me along the way that are worth
recording, because they are exactly the feasibility/score traps the problem
rewards getting right.

**Bug 1 — empty Voronoi cells crashing the Lloyd step.** My first centroid step
divided each facility's accumulated `(sum_x, sum_y)` by its demand count. On a
clustered instance several facilities ended a sweep owning **zero** demand points
(all demand grabbed by closer facilities). `cnt[i] == 0` meant a divide-by-zero
and a `NaN` coordinate, which then propagated into the energy as `NaN`, and the
"best" tracker happily stored a `NaN` configuration — when I snapped that, the
coordinates came out as garbage and the scorer floored me to `0`. The fix: when a
facility's cell is empty, *re-seed* it onto a random demand point instead of
dividing by zero. That both avoids the `NaN` and does something useful (an idle
facility is moved to where there is demand to serve). After this, the continuous
energies were finite and the snapped outputs were always in range.

**Bug 2 — the snap quietly making things worse than I assumed.** I initially
trusted plain rounding (`llround`) and skipped straight to the polish. On a few
seeds the rounded configuration scored only ~1.3x the grid, weaker than I
expected from how good the continuous `E` looked. The gap was the snap: rounding
each facility independently broke the careful spacing the repulsion had built, so
the dispersion term dropped and `E` jumped on discretization. Adding the
**structured 4-corner coordinate-descent snap** between the continuous phase and
the polish recovered most of that loss — it lets a facility take the corner that
keeps it apart from its neighbour even if that corner is not the nearest one. The
ratios climbed back up to the ~1.7x–2.6x range.

**Self-verification on the seed set.** I generated seeds 1..20, ran the solver,
scored each, and scored the trivial uniform-grid baseline alongside:

```
seed  k    L    solver     baseline   ratio
  1   22   344    2084513    1000000   2.085
  2   26   355    2393733    1000000   2.394
  3   40   247    1744105    1000000   1.744
  4   22   311    2218474    1000000   2.218
 ...
 19    9   878    2590263    1000000   2.590
 20   18   280    2347116    1000000   2.347
---
solver mean   : 1993204
baseline mean : 1000000
solver min    : 1551750   (still 1.55x the grid)
```

Every one of the 20 outputs is feasible (exactly `2*k` tokens, every coordinate
in `[0, L]`, scorer never floors to `0`), the solver wins on **every** seed (worst
case 1.55x), and the mean is ~1.99x the uniform-grid baseline. Wall-clock per
instance is ~1.0–1.3s and memory ~4MB, comfortably inside the 2s / 256MB budget.
I also fed the scorer deliberately broken outputs — wrong token count, a
coordinate at `L+1`, a negative coordinate, and a garbage token — and confirmed
each returns exactly `0`, so the feasibility floor is real.

The thing that made this work is the discipline the problem is testing: *optimize
in the domain where you understand the gradients (continuous Lloyd + repulsion),
and treat the integer constraint as a final, structured snap rather than a naive
round.* The grid baseline ignores the demand; Lloyd hugs it; the repulsion keeps
it spread; the 4-corner coordinate-descent snap preserves that structure when
landing on the grid; and the polish cleans up. Holding a feasible configuration
throughout (clamping, re-seeding empty cells, grid fallback) is what guarantees I
never emit an infeasible output, even on a time-limit cutoff.

## Final solver

```cpp
// Continuous Point Placement then Snap -- place k facility points at INTEGER
// coordinates in [0,L]^2 to minimize the energy
//   E = coverage - LAMBDA * dispersion
//     coverage   = sum over demand d of   dist(d, nearest facility)
//     dispersion = sum over facility i of min_{j!=i} dist(facility_i, facility_j)
// Read the instance from stdin, write k integer "x y" facility coordinates.
//
// Method (the innovation):
//   1. CONTINUOUS RELAXATION. Drop the integer constraint and optimize the
//      facilities as real-valued points. Each step is a LLOYD / CENTROIDAL-
//      VORONOI move (pull each facility toward the centroid of the demand in
//      its Voronoi cell -- the exact descent direction for the coverage term)
//      blended with a REPULSION move (push each facility away from its nearest
//      neighbour -- the ascent direction for the dispersion reward we subtract).
//      Several random restarts; keep the best continuous configuration.
//   2. STRUCTURED INTEGER SNAP. A continuous point lands inside a unit grid
//      cell with 4 surrounding integer corners. Naive rounding picks one blindly;
//      instead we run a tiny COORDINATE-DESCENT local search over the discretized
//      energy: for each facility, try all 4 surrounding integer corners (and the
//      clamped variants) and keep the one that lowers the FULL integer energy,
//      sweeping until no point improves. This "solve continuously, then snap with
//      a structured local search" is what beats naive rounding.
//   3. INTEGER POLISH. A short hill-climb of small integer nudges (the 4-/8-
//      neighbour moves) on the discretized energy, again accepting only strict
//      improvements, with the whole thing under a wall-clock budget so we always
//      print a feasible placement.
#include <bits/stdc++.h>
using namespace std;

static const double LAMBDA = 0.5;  // must match score.py

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
               steady_clock::now().time_since_epoch())
        .count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    inline uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    inline uint32_t u32() { return (uint32_t)(next() >> 32); }
    inline int below(int n) { return (int)(u32() % (uint32_t)n); }
    inline double unit() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

int K, L;
vector<double> dmx, dmy;  // demand point coordinates
int M;                    // number of demand points

// ---- continuous-domain helpers -------------------------------------------

// Assign each demand point to its nearest facility; return coverage and fill the
// per-facility centroid accumulators (for a Lloyd step).
static double coverage_and_centroids(const vector<double>& fx,
                                     const vector<double>& fy,
                                     vector<double>& cx, vector<double>& cy,
                                     vector<int>& cnt) {
    fill(cx.begin(), cx.end(), 0.0);
    fill(cy.begin(), cy.end(), 0.0);
    fill(cnt.begin(), cnt.end(), 0);
    double cov = 0.0;
    for (int d = 0; d < M; d++) {
        double bx = dmx[d], by = dmy[d];
        double best = 1e30;
        int bi = 0;
        for (int i = 0; i < K; i++) {
            double ax = bx - fx[i], ay = by - fy[i];
            double d2 = ax * ax + ay * ay;
            if (d2 < best) { best = d2; bi = i; }
        }
        cov += sqrt(best);
        cx[bi] += bx; cy[bi] += by; cnt[bi]++;
    }
    return cov;
}

// nearest-neighbour spacing sum (dispersion) over the facilities
static double dispersion_cont(const vector<double>& fx, const vector<double>& fy) {
    if (K <= 1) return 0.0;
    double tot = 0.0;
    for (int i = 0; i < K; i++) {
        double best = 1e30;
        for (int j = 0; j < K; j++) {
            if (j == i) continue;
            double ax = fx[i] - fx[j], ay = fy[i] - fy[j];
            double d2 = ax * ax + ay * ay;
            if (d2 < best) best = d2;
        }
        tot += sqrt(best);
    }
    return tot;
}

static double energy_cont(const vector<double>& fx, const vector<double>& fy) {
    vector<double> cx(K), cy(K);
    vector<int> cnt(K);
    double cov = coverage_and_centroids(fx, fy, cx, cy, cnt);
    return cov - LAMBDA * dispersion_cont(fx, fy);
}

// ---- integer-domain energy (the objective the scorer uses) ----------------

static double coverage_int(const vector<int>& x, const vector<int>& y) {
    double cov = 0.0;
    for (int d = 0; d < M; d++) {
        double bx = dmx[d], by = dmy[d];
        double best = 1e30;
        for (int i = 0; i < K; i++) {
            double ax = bx - x[i], ay = by - y[i];
            double d2 = ax * ax + ay * ay;
            if (d2 < best) best = d2;
        }
        cov += sqrt(best);
    }
    return cov;
}

static double dispersion_int(const vector<int>& x, const vector<int>& y) {
    if (K <= 1) return 0.0;
    double tot = 0.0;
    for (int i = 0; i < K; i++) {
        double best = 1e30;
        for (int j = 0; j < K; j++) {
            if (j == i) continue;
            double ax = (double)(x[i] - x[j]), ay = (double)(y[i] - y[j]);
            double d2 = ax * ax + ay * ay;
            if (d2 < best) best = d2;
        }
        tot += sqrt(best);
    }
    return tot;
}

static double energy_int(const vector<int>& x, const vector<int>& y) {
    return coverage_int(x, y) - LAMBDA * dispersion_int(x, y);
}

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int main() {
    double t0 = now_sec();
    const double TIME_LIMIT = 1.8;  // wall-clock budget (seconds)

    // ---- read instance ----
    if (scanf("%d %d", &K, &L) != 2) return 0;
    {
        int x, y;
        while (scanf("%d %d", &x, &y) == 2) {
            dmx.push_back((double)x);
            dmy.push_back((double)y);
        }
    }
    M = (int)dmx.size();

    // Degenerate guards -- always output something feasible.
    if (K <= 0) { return 0; }
    if (M == 0) {
        // no demand: just spread on a grid (any feasible placement)
        int g = (int)llround(sqrt((double)K));
        if (g < 1) g = 1;
        int rows = (K + g - 1) / g;
        string out;
        for (int idx = 0; idx < K; idx++) {
            int r = idx / g, c = idx % g;
            int X = (g == 1) ? L / 2 : (int)llround((double)c * L / (g - 1));
            int Y = (rows == 1) ? L / 2 : (int)llround((double)r * L / (rows - 1));
            out += to_string(clampi(X, 0, L)); out += ' ';
            out += to_string(clampi(Y, 0, L)); out += '\n';
        }
        fputs(out.c_str(), stdout);
        return 0;
    }

    Rng rng(0x9e3779b97f4a7c15ULL ^ ((uint64_t)K << 32) ^ ((uint64_t)L << 16) ^
            (uint64_t)M);

    // demand bounding box (helps seed restarts inside the data)
    double minx = 1e30, miny = 1e30, maxx = -1e30, maxy = -1e30;
    for (int d = 0; d < M; d++) {
        minx = min(minx, dmx[d]); maxx = max(maxx, dmx[d]);
        miny = min(miny, dmy[d]); maxy = max(maxy, dmy[d]);
    }

    // =====================================================================
    // (1) CONTINUOUS RELAXATION with random restarts.
    //     Each restart: seed facilities, then alternate a Lloyd centroid pull
    //     (coverage descent) with a repulsion push (dispersion ascent).
    // =====================================================================
    vector<double> bestfx, bestfy;
    double bestE = 1e30;

    vector<double> fx(K), fy(K), cx(K), cy(K);
    vector<int> cnt(K);

    int restart = 0;
    while (true) {
        // budget: leave ~45% of the time for the snap + polish phase
        if (now_sec() - t0 > TIME_LIMIT * 0.55 && restart > 0) break;

        // ---- seed this restart ----
        if (restart == 0) {
            // seed = k random distinct demand points (k-means++-ish spread)
            // first point random, then farthest-from-chosen with some noise
            int first = rng.below(M);
            fx[0] = dmx[first]; fy[0] = dmy[first];
            vector<double> mind(M, 1e30);
            for (int i = 1; i < K; i++) {
                double sum = 0.0;
                for (int d = 0; d < M; d++) {
                    double ax = dmx[d] - fx[i - 1], ay = dmy[d] - fy[i - 1];
                    double d2 = ax * ax + ay * ay;
                    if (d2 < mind[d]) mind[d] = d2;
                    sum += mind[d];
                }
                // sample proportional to squared distance (k-means++)
                double t = rng.unit() * sum;
                double acc = 0.0; int pick = 0;
                for (int d = 0; d < M; d++) { acc += mind[d]; if (acc >= t) { pick = d; break; } }
                fx[i] = dmx[pick]; fy[i] = dmy[pick];
            }
        } else {
            // random restart: uniform inside the demand bounding box (jittered)
            for (int i = 0; i < K; i++) {
                fx[i] = minx + rng.unit() * (maxx - minx);
                fy[i] = miny + rng.unit() * (maxy - miny);
            }
        }

        // ---- relaxation iterations ----
        int ITERS = 60;
        for (int it = 0; it < ITERS; it++) {
            // Lloyd: move each facility toward the centroid of its demand cell
            coverage_and_centroids(fx, fy, cx, cy, cnt);
            for (int i = 0; i < K; i++) {
                if (cnt[i] > 0) {
                    double tx = cx[i] / cnt[i], ty = cy[i] / cnt[i];
                    // partial step toward centroid (damped Lloyd)
                    fx[i] += 0.85 * (tx - fx[i]);
                    fy[i] += 0.85 * (ty - fy[i]);
                } else {
                    // empty cell: re-seed onto a random demand point
                    int d = rng.below(M);
                    fx[i] = dmx[d]; fy[i] = dmy[d];
                }
            }
            // Repulsion: push each facility away from its nearest neighbour to
            // raise dispersion. Strength decays over iterations so coverage wins
            // the late game (keeping E reliably positive yet well-spread).
            if (K > 1) {
                double strength = 0.30 * (1.0 - (double)it / ITERS);
                vector<double> px(K, 0.0), py(K, 0.0);
                for (int i = 0; i < K; i++) {
                    double best = 1e30; int bj = -1;
                    for (int j = 0; j < K; j++) {
                        if (j == i) continue;
                        double ax = fx[i] - fx[j], ay = fy[i] - fy[j];
                        double d2 = ax * ax + ay * ay;
                        if (d2 < best) { best = d2; bj = j; }
                    }
                    if (bj >= 0) {
                        double ax = fx[i] - fx[bj], ay = fy[i] - fy[bj];
                        double dist = sqrt(best) + 1e-9;
                        px[i] = ax / dist; py[i] = ay / dist;
                    }
                }
                double scale = strength * 0.05 * (maxx - minx + maxy - miny + 1.0);
                for (int i = 0; i < K; i++) {
                    fx[i] += scale * px[i];
                    fy[i] += scale * py[i];
                }
            }
            // keep inside the legal domain
            for (int i = 0; i < K; i++) {
                fx[i] = min(max(fx[i], 0.0), (double)L);
                fy[i] = min(max(fy[i], 0.0), (double)L);
            }
        }

        double e = energy_cont(fx, fy);
        if (e < bestE) { bestE = e; bestfx = fx; bestfy = fy; }
        restart++;
        if (now_sec() - t0 > TIME_LIMIT * 0.55) break;
    }

    // safety: if for some reason no restart ran, seed a grid
    if (bestfx.empty()) {
        bestfx.assign(K, 0.0); bestfy.assign(K, 0.0);
        int g = (int)llround(sqrt((double)K)); if (g < 1) g = 1;
        int rows = (K + g - 1) / g;
        for (int idx = 0; idx < K; idx++) {
            int r = idx / g, c = idx % g;
            bestfx[idx] = (g == 1) ? L / 2.0 : (double)c * L / (g - 1);
            bestfy[idx] = (rows == 1) ? L / 2.0 : (double)r * L / (rows - 1);
        }
    }

    // =====================================================================
    // (2) STRUCTURED INTEGER SNAP.
    //     Round each facility to the 4 surrounding integer corners and pick,
    //     via coordinate descent on the FULL integer energy, the corner that
    //     lowers E -- a structured snap, not naive rounding.
    // =====================================================================
    vector<int> x(K), y(K);
    for (int i = 0; i < K; i++) {
        x[i] = clampi((int)llround(bestfx[i]), 0, L);
        y[i] = clampi((int)llround(bestfy[i]), 0, L);
    }

    // candidate integer corners for each facility (4 surrounding cells, clamped)
    vector<array<int, 4>> candX(K), candY(K);
    for (int i = 0; i < K; i++) {
        int fxi = (int)floor(bestfx[i]); int cxi = fxi + 1;
        int fyi = (int)floor(bestfy[i]); int cyi = fyi + 1;
        int xs[2] = {clampi(fxi, 0, L), clampi(cxi, 0, L)};
        int ys[2] = {clampi(fyi, 0, L), clampi(cyi, 0, L)};
        candX[i] = {xs[0], xs[1], xs[0], xs[1]};
        candY[i] = {ys[0], ys[0], ys[1], ys[1]};
    }

    double curE = energy_int(x, y);
    bool improved = true;
    int sweepGuard = 0;
    while (improved && now_sec() - t0 < TIME_LIMIT) {
        improved = false;
        for (int i = 0; i < K; i++) {
            int bx = x[i], by = y[i];
            double bE = curE;
            int ox = x[i], oy = y[i];
            for (int t = 0; t < 4; t++) {
                x[i] = candX[i][t]; y[i] = candY[i][t];
                double e = energy_int(x, y);
                if (e < bE - 1e-9) { bE = e; bx = x[i]; by = y[i]; }
            }
            x[i] = bx; y[i] = by;
            if (bE < curE - 1e-9) { curE = bE; improved = true; }
            else { x[i] = ox; y[i] = oy; }  // no strict gain: keep original
        }
        if (++sweepGuard > 50) break;
    }

    // =====================================================================
    // (3) INTEGER POLISH: small neighbour nudges accepted only on strict gain.
    // =====================================================================
    const int dxs[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dys[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    // adaptive step size: a fraction of L, shrinking
    int step = max(1, L / 50);
    while (step >= 1 && now_sec() - t0 < TIME_LIMIT) {
        bool any = false;
        for (int i = 0; i < K && now_sec() - t0 < TIME_LIMIT; i++) {
            int bx = x[i], by = y[i];
            double bE = curE;
            for (int t = 0; t < 8; t++) {
                int nx = clampi(x[i] + dxs[t] * step, 0, L);
                int ny = clampi(y[i] + dys[t] * step, 0, L);
                if (nx == x[i] && ny == y[i]) continue;
                int sx = x[i], sy = y[i];
                x[i] = nx; y[i] = ny;
                double e = energy_int(x, y);
                if (e < bE - 1e-9) { bE = e; bx = nx; by = ny; }
                x[i] = sx; y[i] = sy;
            }
            if (bE < curE - 1e-9) { x[i] = bx; y[i] = by; curE = bE; any = true; }
        }
        if (!any) step /= 2;  // converged at this scale: refine
    }

    // ---- emit k integer coordinates (guaranteed in [0,L]) ----
    string out;
    out.reserve(K * 12);
    for (int i = 0; i < K; i++) {
        out += to_string(clampi(x[i], 0, L));
        out += ' ';
        out += to_string(clampi(y[i], 0, L));
        out += '\n';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
