# Pit-Reset Pacing: Bang-Bang Stints Under Nonconvex Tire Wear

## Problem
You plan the pace of an endurance race over `L` laps. Each lap you choose an
**effort intensity** `x` from a fixed grid, and whether to **pit** (a full stop
that resets tire wear to zero) just before that lap. Pushing harder makes the
lap faster now but wears the tires, and worn tires slow every following lap of
the current stint. Pit stops undo the wear but cost time. Minimize total race
time.

## Input (stdin)
```
L k
base a p b q P
g_0 g_1 ... g_{k-1}
```
`L` laps, an intensity grid of `k` strictly ascending positive values `g_j`, and
real coefficients `base, a, p, b, q, P`.

## Output (stdout)
Exactly `L` lines. Line `i` (for laps `i = 0 … L-1`) contains two integers:
```
idx_i  pit_i
```
`idx_i ∈ {0,…,k-1}` selects intensity `x_i = g[idx_i]`; `pit_i ∈ {0,1}` is 1 iff
you pit immediately before lap `i`.

## Model (how time is computed)
Process laps in order with `wear = 0` initially. For lap `i`:
- if `pit_i = 1`: add `P` to the total and set `wear = 0` (before the lap);
- the lap costs `base + a · wear^p + b / x_i`, where `wear` is the value at the
  **start** of the lap;
- then wear accumulates: `wear ← wear + x_i^q`.

Total race time `F` is the sum of all lap costs plus all pit costs.
`wear^p` is the tire penalty (nonconvex when `p > 1`); `b / x_i` is the reward
for pushing; `x_i^q` is how fast that push wears the tires.

## Feasibility
The output must be exactly `2·L` integer tokens with every `idx_i` in range and
every `pit_i ∈ {0,1}`. Any missing/extra token, out-of-range index, bad flag, or
non-integer/non-finite token scores `0`.

## Objective & Scoring
Minimize `F`. Let `B` be the total time of the **coast plan** (minimum intensity
`g_0` every lap, no pits) — a construction the grader builds itself. Your score is
```
Ratio = min(1.0, 0.1 · B / F)
```
Reproducing the coast plan gives `≈ 0.1`; being ten times faster caps at `1.0`.
Scoring is exact and deterministic.

## Why the obvious plan is a trap
Because a worn-tire penalty `a · wear^p` is **paid on every remaining lap of the
current stint**, the intensity you burn on a lap only hurts the laps *before the
next pit*. So effort spent late in a stint is cheap (few laps left to suffer it)
and effort spent early is expensive. A single **constant pace with evenly spaced
pits** — the textbook stint answer — ignores this entirely. With resets
available the effective wear cost is the *lower convex envelope* of the wear
curve, so the optimal pace is **bang-bang / sawtooth**: coast early in a stint,
push hard just before pitting, and place the pit boundaries where the envelope
says — not on an even grid. On the harder instances (`p` near 2) this position-
aware pacing beats the best constant pace by a wide margin.

## Constraints
`10 ≤ L ≤ 40`, `2 ≤ k ≤ 8`, all `g_j > 0`, `1 < p ≤ 2.4`, `1 ≤ q < 2`,
coefficients positive. Time limit 5 s, memory 512 MB.

## Example (scoring walk-through)
Suppose `L = 4`, grid `[0.5, 2.0]`, `base = 2, a = 3, p = 2, b = 8, q = 1.5,
P = 6`. The coast plan `0 0 / 0 0 / 0 0 / 0 0` runs at `x = 0.5` every lap; its
laps cost `2 + 3·wear^2 + 16` with wear `0, 0.354, 0.707, 1.061` → total
`B ≈ 73.1`. A plan that coasts then pushes late, e.g. `0 0 / 0 0 / 1 0 / 1 0`
(intensity index `1`, i.e. `x = 2.0`, on the last two laps, no pit) trades a
large `b/x` reward for only a short end-of-race wear spike and finishes faster,
so `Ratio = 0.1·B/F` rises above `0.1`. (Illustrative only — real instances are larger and the coefficients live
in the input.)
