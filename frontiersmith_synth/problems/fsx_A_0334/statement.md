# Gallery Tour Balance: Maximize the Uniformity of Viewing-Station Spread

## Problem

A museum curator is laying out a self-guided **gallery tour** on a square gallery
floor, modelled as the unit square `[0,1] x [0,1]`. A few fixed **landmarks** are
already installed (the entrance, the exit, and an immovable central sculpture). The
curator must place `n` new **viewing stations** anywhere on the floor.

Visitors wander freely between all `n + k` points (stations and landmarks). A tour
"feels balanced" when the arrangement is **as uniform as possible**: no two points
should be cramped together, and no point should be marooned far from everything
else. Formally, let `S` be the combined set of all `n + k` points, and let

```
d_min = the smallest pairwise Euclidean distance in S
d_max = the largest  pairwise Euclidean distance in S
U     = d_min / d_max      (0 < U <= 1)
```

`U` is large exactly when every pair of points is a similar distance apart — a
perfectly regular spread. Your job is to place the stations to make `U` **large**.

This is an extremal point-configuration problem (the "min max/min pairwise-distance
ratio" family). It is NP-hard in general, has no known closed-form optimum for the
sizes used here, and rewards several distinct strategies (max-min packing, regular
lattices, locally polished layouts).

## Input (stdin)

```
n k
x_1 y_1
...
x_k y_k
```

- `n` — number of free viewing stations you must place (`7 <= n <= 16`).
- `k` — number of fixed landmarks (`k = 3`).
- The next `k` lines give landmark coordinates, each in `[0,1]`.

## Output (stdout)

Exactly `n` lines, one per station:

```
x_1 y_1
x_2 y_2
...
x_n y_n
```

Each coordinate is a real number in `[0,1]`.

## Feasibility

An output is **valid** only if:

1. it contains exactly `n` points, each two finite real numbers;
2. every coordinate lies in `[0,1]` (tolerance `1e-9`);
3. no two points of the combined set `S` coincide — all pairwise distances exceed
   `1e-6`.

Any violation scores `0`.

## Objective

**Maximize** `U = d_min / d_max` over the combined set `S` of stations + landmarks.

## Scoring

The checker computes your `U`, then builds its own **diagonal baseline** `B`
(the `n` stations equally spaced on the gallery's main diagonal together with the
landmarks) and reports

```
sc    = min(1000, 100 * U / max(1e-9, B))
Ratio = sc / 1000
```

So the diagonal baseline scores about `0.1`, and a layout `10x` more uniform than
the baseline caps at `1.0`. Scores are deterministic and reproducible.

## Constraints

- `7 <= n <= 16`, `k = 3`. Landmarks are mutually well separated and lie off the
  main diagonal.
- Time limit 5s, memory 512MB. All scoring is exact geometry with a fixed `1e-6`
  coincidence tolerance — no randomness, wall-time, or hardware in the score.

## Example (worked score)

Suppose `n = 2`, `k = 1`, landmark at `(0.5, 0.9)`, and you output stations
`(0.2, 0.2)` and `(0.8, 0.2)`. The three points have pairwise distances
`0.600` (the two stations), `~0.762`, and `~0.762`. So `d_min = 0.600`,
`d_max = 0.762`, and `U = 0.787`. The checker's diagonal baseline for this
instance gives some `B`; the reported `Ratio` is `min(1000, 100*U/B)/1000`. (This
tiny example is illustrative only; real instances use `n >= 7`.)
