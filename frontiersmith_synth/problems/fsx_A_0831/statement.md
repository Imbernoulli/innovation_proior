# The Abacus Cult's Secret Radix

## Problem
An abacus cult accepts tributes from initiates. Every tribute is a positive
integer `x` (a coin count). The cult scribes never record `x` itself as
sacred — they record a **blessing value** `y`, computed by a fixed ritual
that only the elders know in full:

1. Write `x` in the cult's secret counting base `b` (`3 <= b <= 40`, never
   revealed, and it is never base ten — the cult considers ten profane).
2. Sum the base-`b` digits of `x` to get `s = digitsum_b(x)`.
3. The blessing is the elders' secret quadratic `y = a2*s^2 + a1*s + a0`
   (fixed integers, also never revealed).

The scribes have logged several tributes and their blessings for you. Your
job: recover a radix and quadratic `(b, a2, a1, a0)` that reproduces the
ritual, because the elders will soon bless tributes far larger than
anything you've seen — offerings written with dozens to hundreds of
digits — and only the *true* ritual, expressed in the *true* base, predicts
those correctly. A ritual fit only to the scale of the training tributes
(for instance, one that silently assumes the profane base ten, or one that
treats `y` as some smooth function of `x`'s raw size) will diverge wildly
once `x` grows that large, because `digitsum_b(x)` does not scale with `x`
the way `x` itself does — it scales with the *number of digits* of `x` in
base `b`, a completely different geometry, and a ritual anchored to the
wrong base tracks a quantity that has essentially nothing to do with the
true one at that scale.

**Illustrative FORM only — not the hidden law of this problem:** e.g. a
rule like `z <- 3*digitsum_7(w) - 5` on an unrelated quantity `w` is the
*shape* of "a linear rule in a digit sum"; the real ritual here follows the
tribute mechanics described above, with its own radix and *quadratic*
(not linear), not this example.

## Input (stdin)
```
t K
x_1 y_1
x_2 y_2
...
x_K y_K
```
`t` is the test id. `K` tributes were logged, each as a pair `(x_i, y_i)`
with `1 <= x_i <= 10^6`.

## Output (stdout)
Four whitespace-separated integers on one line:
```
b a2 a1 a0
```
your recovered radix and quadratic blessing coefficients.

## Feasibility
The submission scores `Ratio: 0.0` unless ALL hold:
- exactly 4 tokens, each parses as a finite (plain, non-scientific-notation)
  integer;
- `3 <= b <= 40`;
- `|a2| <= 100`, `|a1| <= 1000`, `|a0| <= 100000`.

## Objective (minimize)
The grader draws several **fresh** tributes (never shown to you) with
20 to roughly 250 decimal digits — far beyond the `10^6` ceiling of your
training rows — and computes the blessing TWICE: once under the true
hidden `(b*, a2*, a1*, a0*)`, once under your submitted `(b, a2, a1, a0)`.
Let `F` be the mean absolute difference between the two blessings over all
held-out tributes.

## Scoring
```
B = mean absolute error of the "nothing learned" ritual (b=3, a2=0, a1=0, a0=0)
eps = B / 8
sc = min(1000, 100 * (B + eps) / max(1e-9, F + eps))
Ratio = sc / 1000
```
Guessing blindly (a flat zero blessing, ignoring the logged tributes
entirely) reproduces `B` exactly, scoring `Ratio = 0.1`. Because of the
`eps` floor, even an exact recovery (`F = 0`) caps at `Ratio = 0.9` —
there is always headroom above any single strategy.

## Constraints
`1 <= t <= 10`; `54 <= K <= 90`; `1 <= x_i <= 10^6`; time limit 5s, memory
512m. Each `.in` file is well under 5 MB.

## Example (worked score)
Suppose `B = 3,000,000` and a submission achieves `F = 400,000`
(`eps = 375,000`): `sc = 100*(3000000+375000)/(400000+375000)
= 100*3375000/775000 ~= 435.5`, so `Ratio ~= 0.4355`. A submission with
`F = 0` gets `sc = 100*3375000/375000 = 900`, i.e. `Ratio = 0.9`.
