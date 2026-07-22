# Four-Bar Coupler Tracing: Fit a Linkage to a Target Path

## Problem
A planar **four-bar linkage** has two ground pivots `O0`, `O1` bolted to the
table. A *crank* of length `a` turns about `O0`; a *rocker* of length `c`
swings about `O1`; a rigid *coupler* of length `b` joins the crank tip `A` to
the rocker tip `B`. A pen (the **coupler point** `P`) is fixed to the coupler
at offset `(u, v)` in the coupler frame — `u` measured along `A→B`, `v`
perpendicular (rotated +90°). As the crank spins through a full turn, `P`
traces a closed **algebraic sextic** curve.

You are given a target path (a set of sampled points). Choose the five numbers
`a, b, c, u, v` so the coupler point's traced curve matches the target as
closely as possible.

## Input (stdin)
```
M
O0x O0y
O1x O1y
x y            (M lines: the target curve samples)
```
`M` is the number of target points. `O0` and `O1` are the fixed ground pivots
(here `O0=(0,0)`, `O1=(g,0)` with `g` the ground-link length).

## Output (stdout)
Five floats on one line:
```
a b c u v
```

## Feasibility
All five values finite, and `a, b, c > 0`. Any violation scores `Ratio: 0.0`.

The checker traces your linkage by sweeping the crank angle `θ` over a full
revolution. At each `θ` it places `A = O0 + a(cosθ, sinθ)` and solves loop
closure for `B` (intersection of the circle of radius `b` about `A` with the
circle of radius `c` about `O1`); it tries **both** assembly branches and keeps
the better-scoring one. If the two circles do not meet at some `θ` (the linkage
cannot fully rotate — a Grashof violation), that part of the curve is simply
missing, and the missing arc is penalized by the matching distance.

## Objective (minimize)
Let `S` be your traced coupler curve and `T` the target. The score uses the
symmetric point-set (Chamfer) distance
```
F = 0.5 * ( mean_{t in T} min_{s in S} |t - s|
          + mean_{s in S} min_{t in T} |t - s| ).
```
Smaller `F` is better.

## Scoring
The checker builds its own **baseline** linkage `B` — a fixed, un-tuned Grashof
crank-rocker sized only from the ground span `g` — and measures its distance to
the target. With your distance `F`:
```
sc    = min(1000, 100 * B / max(1e-12, F))
Ratio = sc / 1000
```
Reproducing the baseline distance scores about `0.1`; a trace ten times closer
caps at `1.0`. The target carries a small deterministic perturbation, so no
linkage reproduces it exactly — headroom always remains.

## Constraints
`90 <= M <= 200`. Ground span `9 <= g <= 15`. The checker is `O(M·K)` with a
fixed crank resolution `K`; every case scores well under the time limit.

## Example
Suppose `g = 10` and the target is a kidney-shaped loop. The baseline linkage
`a=3.5, b=11, c=9, u=5.5, v=3.3` traces a large lopsided oval far from the
target, giving distance `B` and `Ratio ≈ 0.1`. A linkage whose crank radius and
coupler-point reach are tuned to the target's radial extent about each pivot
traces a curve hugging the loop, cutting `F` several-fold and lifting the score
well above the baseline. (Numbers illustrative — the actual target is in the
input.)
