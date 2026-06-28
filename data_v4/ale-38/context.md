# Continuous Point Placement then Snap

## Research question

You are given a square domain `[0, L] x [0, L]` and a cloud of `m` integer
**demand** points inside it. You must place exactly `k` **facility** points at
**integer** coordinates in `[0, L]^2` so that they are simultaneously **close to
demand** and **spread apart** from one another. Concretely, minimize the energy

```
E = coverage - LAMBDA * dispersion
```

where, with Euclidean distance `dist(.,.)`,

- `coverage   = sum over demand d of  dist(d, nearest facility)` — every demand
  point is served by its nearest facility; small coverage means facilities sit
  close to where the demand actually is, and
- `dispersion = sum over facility i of min_{j != i} dist(facility_i, facility_j)`
  — the total nearest-neighbour spacing of the facilities; large dispersion
  means the facilities are mutually far apart (well spread), so it is rewarded
  by being subtracted.

`LAMBDA` is fixed at `0.5`. The two terms pull against each other: chasing the
demand alone collapses facilities onto the dense clusters (low coverage but tiny
dispersion), while spreading them evenly ignores where the demand is (high
dispersion but high coverage). The optimum is a genuine trade-off.

This is the integer-grid version of the kind of facility-location / blue-noise
placement problem that appears in sensor deployment, k-center style coverage,
stippling, and quadrature-point design: you want representative points that hug
the data yet avoid clumping. It is a continuous **non-convex** optimization with
a final integer-grid snap — there is no known efficient exact solver, and the
benchmark scores a placement by *how low* its energy is, not by matching a
unique optimum.

## Input / output contract

- **Input (stdin):** the first line is `k L` (`8 <= k <= 40`, `200 <= L <= 1000`).
  Then a list of integer demand points follows, one `x y` pair per line
  (`0 <= x, y <= L`), read until end of file. The number of demand points `m`
  (a few hundred) is not given on its own line — the solver reads pairs until
  EOF.
- **Output (stdout):** `k` lines, each `x y` — the integer coordinates of one
  facility (`0 <= x, y <= L`). Any whitespace layout is accepted; the scorer
  reads `2*k` integer tokens.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A placement is **feasible** iff (a) the output parses as exactly `2*k` integers,
and (b) every facility coordinate `(x, y)` satisfies `0 <= x <= L` and
`0 <= y <= L`. Anything else — a parse error, the wrong number of tokens, or a
coordinate out of range — is **infeasible**.

## Background

Why is naive rounding the wrong instinct? The objective lives in the continuous
plane (distances are real-valued), so the natural move is to optimize the
facilities as **real-valued** points and only afterwards force them onto the
integer grid. Two classical ideas drive the continuous phase:

- **Lloyd / centroidal-Voronoi relaxation (coverage descent).** For the coverage
  term alone, the optimal position of a facility, given the set of demand points
  assigned to it (its Voronoi cell), is the **centroid** of that cell. Iterating
  "assign each demand to its nearest facility, then move each facility to its
  cell's centroid" is exactly Lloyd's algorithm; it monotonically reduces
  coverage and converges to a centroidal Voronoi tessellation. This is the
  standard, strong way to make points hug a data distribution.
- **Repulsion (dispersion ascent).** Lloyd alone tends to leave facilities
  clumped where the demand is densest, which crushes the dispersion reward. A
  blue-noise-style **repulsion** step — push each facility a little away from its
  nearest neighbour — raises dispersion. Annealing the repulsion strength down
  over the iterations lets coverage win the late game, which keeps the energy
  reliably positive while still well spread.

After the continuous configuration is good, it must be snapped to the integer
grid. A continuous point sits inside a unit cell with **4 surrounding integer
corners**; rounding picks one of them blindly. The decisive lever is to snap with
a **structured local search** instead:

- **Structured integer snap (not naive rounding).** For each facility, try all 4
  surrounding integer corners and keep the corner that lowers the **full
  discretized energy**, sweeping over all facilities until none improves
  (coordinate descent on the integer objective). Because moving one facility
  changes the energy through its own corner choice, this captures the
  interactions that independent per-point rounding misses, and it provably never
  raises the energy relative to the snapped-in configuration it starts from.
- **Integer polish.** A short hill-climb of small integer nudges (4-/8-neighbour
  moves at a shrinking step size) on the discretized energy, accepting only
  strict improvements, cleans up the last bit. Because every accepted move keeps
  coordinates clamped to `[0, L]`, any time-limit cutoff still leaves a feasible
  placement in hand.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrized by an integer
  `seed`) deterministically chooses `k in [8, 40]`, `L in [200, 1000]`, and a few
  hundred demand points. The demand is **clustered**: a handful of tight Gaussian
  blobs at random centres plus a light uniform background. The clustering is what
  makes placement non-trivial — a uniform grid of facilities (the reference)
  wastes points on empty regions, while a good solver pulls facilities onto the
  clusters (lowering coverage) yet keeps them apart (raising dispersion).
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the
  submitted facility coordinates.
  - **Feasibility floor:** if the output does not parse as exactly `2*k`
    integers, or some coordinate is out of `[0, L]`, the score is **`0`**.
  - Otherwise compute `E = coverage - LAMBDA * dispersion` as defined above
    (lower is better). Let `E_ref` be the energy of the **uniform-grid**
    placement: put the `k` facilities on a near-square grid (`g = round(sqrt(k))`
    columns) spanning `[0, L]^2`, snapped to integers — recomputed inside the
    scorer so the reference is reproducible and solver-independent. The score is

    ```
    score = round( 1_000_000 * E_ref / E_solver )   (feasible, E_solver > 0)
    score = 2_000_000                                (feasible, E_solver <= 0)
    score = 0                                        (infeasible)
    ```

    A higher score is better. The uniform-grid reference scores essentially
    `1_000_000`; a demand-aware, well-spread placement lowers `E` below `E_ref`
    and so scores strictly more; a feasible-but-worse placement scores less but
    stays positive. (The essentially-unreachable `E_solver <= 0` case — where the
    dispersion reward outweighs all coverage — is given a generous full-credit
    cap rather than dividing by a non-positive number.)
- **Reported metric.** The mean score over a fixed seed set. A real
  continuous-relax + structured-snap solver lands well above the grid floor
  (roughly `1.7x`–`2.6x` of `1_000_000` on these clustered instances); the
  trivial uniform-grid baseline scores exactly `1_000_000` and is the floor to
  beat.

## Code framework

A single self-contained C++17 program reading the instance from stdin and
writing `k` feasible integer facility coordinates to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

static const double LAMBDA = 0.5;  // must match score.py

int main() {
    int K, L;
    if (scanf("%d %d", &K, &L) != 2) return 0;
    vector<double> dmx, dmy;  // demand points
    {
        int x, y;
        while (scanf("%d %d", &x, &y) == 2) {
            dmx.push_back((double)x);
            dmy.push_back((double)y);
        }
    }
    int M = (int)dmx.size();

    // A feasible answer is ANY k integer coordinates in [0,L]^2. The safe start
    // is the uniform grid (the scorer's own reference). Hold something legal at
    // all times so a time-limit cutoff still prints a feasible placement.
    vector<int> x(K, L / 2), y(K, L / 2);

    // TODO heuristic:
    //  (1) CONTINUOUS RELAXATION -- optimize the facilities as real-valued
    //      points: alternate a Lloyd centroid pull (move each facility toward the
    //      centroid of its nearest-demand Voronoi cell -- coverage descent) with
    //      a repulsion push (away from the nearest other facility -- dispersion
    //      ascent), with a few random restarts; keep the best configuration.
    //  (2) STRUCTURED INTEGER SNAP -- for each facility try its 4 surrounding
    //      integer corners and pick, by coordinate descent on the FULL integer
    //      energy, the corner that lowers E (a structured snap, not naive
    //      rounding).
    //  (3) INTEGER POLISH -- a short hill-climb of small integer nudges on the
    //      discretized energy, accepting only strict improvements, under a ~2s
    //      wall-clock budget. Keep every coordinate clamped to [0,L].

    // Emit k integer coordinates (one "x y" per line), guaranteed in [0,L].
    string out;
    for (int i = 0; i < K; i++) {
        out += to_string(x[i]); out += ' ';
        out += to_string(y[i]); out += '\n';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
```
