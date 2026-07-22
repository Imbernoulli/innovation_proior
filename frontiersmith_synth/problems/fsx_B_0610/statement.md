# Catalyst Reactor Farm: Spending Freshness at the Price Peaks

## Problem

You operate a farm of `R` identical catalytic reactors over `T` discrete steps. Each reactor
holds a catalyst whose **conversion efficiency decays with cumulative use** and can be restored
only by taking the reactor **offline for a fixed regeneration window**. The market pays a
step-varying **price** and can absorb only a step-varying **demand cap**. You choose, for every
reactor and every step, how much feed to process (or whether to regenerate) so as to
**maximize revenue**.

Decay is expressed through a marginal-conversion curve `e[0..L-1]`, a strictly decreasing
sequence. If a reactor has already processed `w` feed units since its last regeneration, the
next feed unit converts to product at rate `e[w]` (for `w >= L`, use `e[L-1]`, the floor). So
running `x` units this step, starting from cumulative use `w`, yields product
`e[w] + e[w+1] + ... + e[w+x-1]` and advances cumulative use to `w + x`. Every reactor starts
fresh (`w = 0`).

A **regeneration** takes exactly `d` fully-offline steps (no production) and resets that
reactor's cumulative use to `0`.

## Input (stdin)

```
line 1: T R Q d L
line 2: L floats   e[0..L-1]     marginal conversion of the w-th feed unit since regeneration
line 3: T floats   p[0..T-1]     unit price paid at each step
line 4: T ints     cap[0..T-1]   market demand cap (max units SOLD) at each step
```

`Q` is the maximum feed a reactor may process in one step.

## Output (stdout)

Print `R` lines, one per reactor, each with `T` whitespace-separated integer tokens. Token
`x[r][t]` is:

* an integer in `0..Q` — feed units processed by reactor `r` at step `t`; or
* `-1` — reactor `r` is offline (regenerating) at step `t`.

The `-1` tokens of a reactor must form maximal contiguous runs whose length is a multiple of
`d` (each length-`d` block is one regeneration; cumulative use resets to `0` when the run ends).
Any other token, wrong token count, or a `-1` run whose length is not a multiple of `d` is
**infeasible** and scores `0`.

## Objective (maximize)

Let `produced[t]` be the total product across all reactors at step `t` (offline reactors
contribute `0`; a running reactor contributes the sum of its marginal-conversion terms). Only up
to `cap[t]` units can be sold; the rest is wasted. Revenue is

```
F = sum over t of  p[t] * min(cap[t], produced[t]).
```

## Scoring

Let `B` be the revenue of the reference construction in which **every reactor runs at full
throughput `Q` at every step and never regenerates**. Your score is

```
Ratio = min(1.0, 0.1 * F / B),
```

so reproducing the reference gives `0.1` and a tenfold improvement caps at `1.0`. Higher is
better; the score is deterministic.

## Feasibility

Exactly `R * T` integer tokens; every token in `{-1, 0, ..., Q}`; every maximal run of `-1` of
length a multiple of `d`. Non-integer, non-finite, out-of-range, or miscounted output scores `0`.

## Constraints

`1 <= R <= 6`, `24 <= T <= 180`, `Q = 10`, `3 <= d <= 6`, `L = 50`,
`e` strictly decreasing to a positive floor, `p >= 1`, `cap >= 1`. Time limit 5 s, memory 512 MB.

## Example (worked score)

Suppose `T=6, R=1, Q=10, d=3, L=50`, with a single price peak at steps 4–5 (`p=8`, `cap=10`) and
cheap valleys elsewhere (`p=1`, `cap=5`). Running flat-out every step (the reference) spends the
freshest catalyst on the worthless early valleys and meets the peak with aged catalyst, earning a
small `B`. Idling step 0, parking a regeneration in the valley over steps 1–3, then running full
throughput with fresh catalyst into the peak at steps 4–5 sells the full demand cap at the high
price, for `F` several times larger —
a higher `Ratio`. The peak is worth far more than its feed cost; the decision is *when* to spend
freshness, not how much to run.
