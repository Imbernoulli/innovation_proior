# Thermal Cross-Interference in a Data-Center Cooling Schedule

## Problem
A hyperscale data center dissipates heat by scheduling coolant pulses across a
row of `n` equal-width time-slots. You must choose a **non-negative coolant
intensity** `f_0, f_1, ..., f_{n-1}` for each slot (an `n`-interval step function
of the coolant profile over one duty cycle).

Because the coolant loop recirculates, a pulse released in slot `i` still exerts
thermal load when it re-encounters a pulse released in slot `j`. The aggregate
**self-interference at lag `k`** is the discrete self-convolution

```
g[k] = sum over i of  f[i] * f[k-i]        for k = 0 .. 2n-2
```

`g` is the autocorrelation of the coolant profile. The worst instantaneous
thermal stress on the loop is the **peak** of `g`. We measure a scale-free
interference constant (the first-autocorrelation ratio, normalized so that a
flat profile scores 2):

```
c1(f) = 2 * n * max_k g[k] / ( sum_i f[i] )^2
```

Both numerator and denominator are homogeneous of degree 2, so `c1` is invariant
under scaling `f -> t*f`; only the *shape* of the profile matters.

**Your goal: choose the profile that MINIMIZES `c1(f)`.** Driving the peak
self-interference down while spreading coolant across the cycle is a genuinely
open problem — no closed-form optimal profile is known, and many different shapes
are viable.

## Input (stdin)
A single line with two integers:
```
n M
```
`n` = number of time-slots. `M` = maximum coolant intensity per slot.

## Output (stdout)
`n` non-negative integers `f_0 ... f_{n-1}` (whitespace-separated, any layout),
each with `0 <= f_i <= M`. Integers only. At least one must be positive.
(Since `c1` is scale-free, you may realize any rational profile by clearing
denominators up to `M`.)

## Feasibility
Output is rejected (score 0) unless: exactly `n` tokens are present, every token
parses as an integer in `[0, M]`, no token is `nan`/`inf`, and the profile is not
all-zero.

## Objective
Minimize `c1(f)` (lower peak self-interference is better).

## Scoring
Let `F = c1(your_profile)` and let `B` be `c1` of the checker's internal baseline
(a single concentrated coolant block of width `max(1, n//5)`). The score is

```
sc    = min(1000, 100 * B / F)          # minimization
Ratio = sc / 1000
```

A profile equal to the concentrated block scores `Ratio = 0.1`. A flat profile
(`c1 = 2`) scores well above that; pushing the peak lower still raises the score,
with the theoretical floor `c1 -> 2n/(2n-1) ~ 1` keeping the score bounded.

## Constraints
- `30 <= n <= 300`, `M = 1_000_000`.
- Deterministic exact-integer scoring.

## Example (worked score)
For `n = 30`, `M = 1_000_000`: the concentrated block
`f = [1,1,1,1,1,1,0,...,0]` (width 6) has peak `g` = 6 and sum 6, so
`c1 = 2*30*6/36 = 10 = B`, giving `Ratio = 100*10/10/1000 = 0.100`.
A flat profile `f = [1]*30` has peak `g` = 30, sum 30, `c1 = 2*30*30/900 = 2`,
giving `Ratio = 100*10/2/1000 = 0.500`. A shaped profile with `c1 = 1.7` would
score `Ratio = 100*10/1.7/1000 = 0.588`.
