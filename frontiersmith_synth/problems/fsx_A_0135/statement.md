# Tide-Pool Overlap Constant

## Problem
A tidal shelf is divided into `n` consecutive tide pools laid out in a line.
You must settle a population of exactly `S` individuals of a single species
across the pools. Pool `i` has an integer **carrying capacity** `cap_i`: the
density placed there, `f_i`, must satisfy `0 <= f_i <= cap_i`.

Ecologists measure inter-pool **competition** by the *spatial self-overlap* of
the density profile: how strongly the population correlates with a shifted copy
of itself. For a shift `k` the overlap is `(f * f)_k = sum_i f_i * f_{k-i}`, the
discrete self-convolution. The single most crowded shift dominates competition,
so the relevant quantity is the **peak self-overlap** `max_k (f * f)_k`.

Because the overlap scales with the square of the population, we report the
scale-free **overlap constant**

```
c1(f) = 2 * n * max_k (f * f)_k / (sum_i f_i)^2 .
```

This is a discrete instance of the first autocorrelation inequality: a flat
profile gives `c1 = 2`, and it is a hard open problem how far below `2` one can
push it. Your job is to shape the settlement so that competition is minimized.

## Input (stdin)
```
n S
cap_1 cap_2 ... cap_n
```
`n` pools, total population `S` to place, and the `n` integer capacities.
It is guaranteed that `sum_i cap_i > S`, so a feasible placement exists.

## Output (stdout)
`n` real numbers `f_1 ... f_n` (whitespace-separated), the density placed in
each pool.

## Feasibility
- exactly `n` finite, non-negative numbers,
- `0 <= f_i <= cap_i` for every pool,
- `sum_i f_i == S` (within `1e-5 * max(1,S)` tolerance).

Any violation scores `Ratio: 0.0`.

## Objective
**Minimize** the overlap constant `c1(f)`.

## Scoring
Let `F = c1(your f)` and let `B = c1` of the checker's own naive baseline (pile
the whole population into the left-most pools). The score is

```
sc  = min(1000, 100 * B / F)
Ratio = sc / 1000        # reported, in [0,1]
```

Reproducing the naive baseline scores about `0.1`; a flat water-filled profile
(`c1 ~ 2`) and, better, an edge-heavy optimized profile score progressively
higher. There is no known easy optimum, and different capacity profiles reward
different shapes.

## Constraints
- `30 <= n <= 174`
- `4 <= cap_i <= 10`, integer
- `S = round(0.40 * sum_i cap_i)`

## Example
For `n = 4`, `S = 8`, `cap = [4,4,4,4]`:
- Baseline (fill from left) `f = [4,4,0,0]`: `max` overlap `= 32`,
  `sum = 8`, `c1 = 2*4*32/64 = 4.0`.
- Flat `f = [2,2,2,2]`: peak overlap `= 16`, `c1 = 2*4*16/64 = 2.0`,
  giving `Ratio = 100*4.0/2.0 / 1000 = 0.20`.
