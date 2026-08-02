# GNSS Multipath Repeat

## Problem

A GNSS reference receiver logs a scalar *excess position error* `e(t)` every 5
minutes, caused by multipath reflections off nearby structures. Two physical
effects add together:

1. **Orbital-repeat multipath.** The satellite constellation's ground track —
   and hence which satellites are overhead, at what elevation and azimuth —
   repeats with a period `P1` close to, but **not exactly**, one solar day
   (this is a real GNSS fact: the orbit repeat is tied to a sidereal day, a
   few minutes shorter than 24 h). Because the reflection geometry sweeps
   through several distinct fade/boost configurations each repeat cycle
   (not one smooth swing), this component is a sum of a **fundamental plus
   its 2nd and 3rd harmonics** of the true repeat period `P1`. `P1` is
   different for every input instance and is not given to you.
2. **Solar/thermal drift.** A smaller, separate effect (antenna thermal
   expansion, diurnal ionospheric TEC) that repeats at **exactly** one solar
   day, 86400 s — a known public constant, not hidden.

A small i.i.d. sensor-noise floor is added on top of both.

You are given several **consecutive** days of `e(t)`. You will be graded on a
held-out horizon **several weeks to two months later** — far enough away that
even a small period error accumulates into a large phase error.

## Input (stdin)

```
n t
t_0  e_0
t_1  e_1
...
t_{n-1}  e_{n-1}
```

`t` is the test id. `n` training rows follow: `t_k` is elapsed seconds since
the start of the log (a multiple of 300), `e_k` the measured error (float).

## Output (stdout)

One closed-form arithmetic expression over the single variable `t` (seconds),
using `+ - * /`, parentheses, numeric constants, and the unary functions
`sin`, `cos`. Example (**illustrative FORM only — not the hidden law**):

```
0.4 * t / (1.0 + t) - 0.05
```

This only shows legal syntax (a non-periodic rational shape); the real error
is periodic and you must discover its structure — and its true period — from
the data.

## Feasibility

The expression must parse under the grammar above (only `t`, `sin`, `cos`,
arithmetic, finite numeric constants), be ≤ 100 expression nodes, ≤ 20000
bytes, and evaluate to a finite number at every held-out `t`. Any violation
scores `0`.

## Objective (maximize)

Let `MSE` be the mean squared error of your expression against the true
held-out trace `e_held`, and `B` the mean squared value of `e_held` itself
(i.e. the error of always predicting `0`). The grader forms

```
sc = min(880, 100 * B / max(1e-9, MSE))
Ratio = sc / 1000
```

Predicting `0` reproduces `B` (Ratio ≈ 0.1). Lower held-out MSE raises the
score; the 880 cap keeps the ceiling below 0.9 so even a noise-floor-limited
period recovery leaves headroom.

## Why the 24-hour assumption is a trap

Over the few visible training days, an expression built purely from harmonics
of *exactly* 86400 s already fits `e(t)` almost perfectly — the true repeat
period differs from a day by only a few hundred seconds, so within a handful
of cycles the two hypotheses look nearly indistinguishable, and the genuinely
24-hour solar term reinforces the illusion. But a period error of `Δ`
seconds accumulates a phase error of roughly `2π · Δ · t_horizon / P1²`
radians; after several weeks that can exceed half a cycle. A fixed-24h fit is
then thrown out of phase on the held-out horizon, while an expression built
around the **true** near-day period is not. Recovering the true period — not
just amplitudes and phases — is what survives extrapolation.

## Constraints

Time limit 5 s, memory 512 MB. `n` is at most a few thousand rows, each
`.in` file well under 5 MB. Everything is seeded by the test id; scoring is
fully deterministic.
