# Lynx-Hare Census — recovering the hidden lagged-crowding law

A field station hands you a **census** of a normalised population density `x`
(fraction of carrying capacity) sampled once per season, four seasons per year.
The population obeys a fixed but undocumented seasonal law: growth depends on the
season, and crowding depends on the density **some whole number of seasons ago**
— nobody recorded how many. Your job is to reverse-engineer a one-step law that
predicts next season's density.

The catch: your census was logged during a **calm, quiescent** stretch — the
population sits near its stable seasonal orbit and only jitters a little. In that
regime the lag is nearly invisible: a plain season-by-season logistic fit matches
the census down to the noise. You are graded on a **held-out crash** — the same
ecosystem knocked far below carrying capacity — where the true law's delayed
crowding makes the population **ring**: it overshoots, dips, and oscillates before
settling. A law that only memorises the calm census relaxes smoothly and misses
the ringing.

## Input (stdin)

```
t  N  S  L
x0
x1
...
x(N-1)
```

`t` is the test id; `S = 4` seasons; `L = 6` is the largest lag you may reference.
Then `N` census rows follow, one density per line. **The season of row `i` is
`i mod S`.** The held-out grading census (a different, crashed trajectory of the
same ecosystem) is NOT given to you.

## Output (stdout): a one-step law

Emit a single closed-form expression for the next density. The grader **rolls it
forward autoregressively** over the held-out crash — your own predictions become
the future lags. Expressions use constants, `+ - * /`, unary `+/-`, the functions
`absv(a)`, `minv(a,b)`, `maxv(a,b)`, and these variables:

- `x` — the current density `x[t]`.
- `lag1 … lag6` — the density 1…6 seasons ago (`x[t-1] … x[t-6]`).
- `s` — the current season index `0..3`; `c0,c1,c2,c3` — its one-hot
  (`ck = 1` when `s==k`, else `0`), handy for a per-season table.

**Illustrative FORM only — NOT the hidden law:**

```
c0*(1.7*x - 0.9*x*lag3) + c1*(1.5*x) + c2*(2.0*x - 1.1*x*x) + c3*(1.6*x - x*lag1)
```

This just shows the syntax and how one-hot seasons build a table; the real law's
shape, delay, and coefficients are different and you must discover them from data.

## Feasibility

The expression must parse under the grammar above (known names/functions only,
lags in `1..6`, finite constants, at most a few hundred nodes). Any parse/known-
name violation, or any non-finite value produced during the rollout, scores `0`.

## Objective (minimise)

Let `MSE` be the mean squared error of your rolled-out prediction against the true
held-out census over the crash horizon, and `nodes` the number of expression nodes
in your law. The grader forms

```
F = MSE      * (1 + LAMBDA * nodes)
B = MSE_persist * (1 + LAMBDA * 1)      # baseline: predict x_next = x
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. Persistence reproduces `B` (Ratio ≈ 0.1); lowering
held-out error raises the score, while a parsimony tax discourages needlessly
large laws. Census noise plus estimation error keep even a strong law below the
ceiling — there is room to improve. Report the highest Ratio you can.

## Why the calm census is a trap

On the quiescent orbit the density barely moves, so the crowding term acts on a
value close to the current one — an **undelayed** logistic absorbs it and fits
beautifully. Fit quality alone will not reveal the lag. But the delay leaves a
signature in the **residual structure** of that fit: the crowding really depends
on an *earlier* season, and probing which lagged density best explains the
leftover residual recovers the hidden delay — the only thing that lets your law
ring on the held-out crash.

## Constraints

Time limit 5 s, memory 512 MB. `N` is a few hundred rows; scoring is fully
deterministic.
