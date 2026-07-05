# AlpenGold Bakery — Minimizing Peak Shift-Overlap

## Problem

AlpenGold runs a single oven line across **n** consecutive baking shifts. In shift *i*
you bake an integer number of loaves `a_i` (0 ≤ `a_i` ≤ V). Every loaf enters the same
cold-chain buffer and is dispatched exactly `k` shifts later, for many different lead
times `k`. When the whole calendar is **shifted by k**, the buffer has to hold, in some
window, the *self-overlap load*

```
overlap(k) = sum_i a_i * a_{i-k}        (the full self-convolution of the plan)
```

The **peak** overlap across all shifts `k` is what stresses the buffer and refrigeration:

```
peak = max_k  overlap(k)
```

To compare plans of different total size fairly, normalize by the squared throughput.
Let `S = sum_i a_i` be total production. The **interference index** of a plan is

```
c(a) = 2 * n * peak / S^2
```

This is a discrete instance of the classical *first autocorrelation inequality* constant:
a flat plan (`a_i = V` for all i) has `c = 2` exactly, and no smooth, regular plan does
better — beating it requires a genuinely irregular schedule, and how low `c` can go is an
open question (the best known constructions sit well below 2 but nobody knows the true
minimum).

**Your job:** output a production plan that makes `c(a)` as small as possible.

## Input (stdin)

One line with two integers:

```
n V
```

`n` = number of shifts (16 ≤ n ≤ 52), `V` = per-shift loaf cap (V = 1000).

## Output (stdout)

`n` whitespace-separated integers `a_0 a_1 ... a_{n-1}`, each in `[0, V]`, with at
least one positive value.

## Feasibility

Rejected (score 0) unless the output has **exactly n** integer tokens, every token lies
in `[0, V]`, and the total `S = sum a_i` is strictly positive.

## Objective

**Minimize** the interference index `c(a) = 2 n * max_k(sum_i a_i a_{i-k}) / (sum_i a_i)^2`.

## Scoring

Deterministic and exact (integer/rational arithmetic). The checker builds its own
**baseline** plan `B` (a triangular ramp-up/ramp-down schedule) and reports

```
Ratio = min(1.0, 0.1 * c(B) / c(a))
```

The triangular baseline scores exactly **0.1**. A plan ten times better than the baseline
would reach **1.0** (unreachable in practice). Lower `c` ⇒ higher `Ratio`.

## Constraints

- 16 ≤ n ≤ 52, V = 1000, integer plans only.
- Scoring is bit-for-bit reproducible; no randomness, timing, or hardware enters the score.

## Example

For `n = 4, V = 1000`, the flat plan `1000 1000 1000 1000` has `S = 4000`,
`peak = 4 * 1000^2 = 4,000,000` (the central convolution term), so
`c = 2*4*4,000,000 / 4000^2 = 2.0`. The triangular baseline scores 0.1; the flat plan,
being better (`c = 2.0` vs the baseline's larger `c`), scores above 0.1. An irregular
annealed plan pushes `c` toward ~1.6 and scores higher still.
