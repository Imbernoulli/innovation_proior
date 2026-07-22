# The Lighthouse Relay — Recovering a Register Formula That Survives the Busy Season

## Problem
An old lighthouse counts passing ships with a mechanical relay. Each ship
advances an internal **3-position calibration gear** by one notch, so after
`n` ships the engaged gear is `s = n mod 3` — a finite-state device with 3
states. Whichever gear is engaged applies **its own** linear-fractional
(rational) recalibration to the ship count and prints a register reading:

```
y(n) = (a_s * n + b_s) / (c_s * n + d_s)          s = n mod 3
```

`a_s, b_s, c_s, d_s` are fixed positive integers, different per gear. As `n`
grows, `y` settles toward the finite asymptote `a_s / c_s` for that gear —
but over the ship counts a slow harbor season produces, the curve is still
visibly bending, nowhere near flat.

You are given a logbook from that slow season (modest ship counts). The
harbor authority wants the register reading during the busiest days on
record, when the ship count is 200x–40000x larger than the logbook shows.

**Illustrative FORM only — not the hidden law's shape:** `3*n + 2*s - 5` is
syntactically valid output, but the real law is not a linear combination of
`n` and `s` — it is a rational function of `n` whose *shape itself* depends
on which gear is engaged.

## Input (stdin)
```
R t
n_1 s_1 y_1
...
n_R s_R y_R
```
`t` is the test id, `R` is the number of logged rows. Each row gives a ship
count `n` (integer), the gear engaged at that count `s = n mod 3` (integer,
0/1/2), and the noisy register reading `y` (float).

## Output (stdout)
One line: a closed-form Python expression for `y` in the variables `n` and
`s`. Allowed: `+ - * / **`, unary `-`, numeric constants, and the functions
`sqrt log exp abs`. No other names are accepted.

## Scoring (deterministic, maximization)
Your expression is evaluated on a **held-out log**, regenerated inside the
grader, whose ship counts are drawn from the busy-season range (200x–40000x
larger than training, never seen in training). Let `p_i` be your prediction
and `t_i` the true (noisy) reading at held-out row `i`:

```
metric   = mean_i  min(1, |p_i - t_i| / (|p_i| + |t_i|))     # bounded rel. error
O        = metric * (1 + LAMBDA * nodes)                     # nodes = expr size
baseline = the same metric for the constant predictor mean(train y)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error gives a higher `Ratio` (capped at `1.0`). The constant
predictor scores about `0.1`. `LAMBDA` lightly penalizes an overgrown
expression. A non-finite prediction on any held-out row scores the whole
submission `0`.

## Why the obvious fit is a trap
The reading visibly LEVELS OFF rather than growing forever, so a competent
first attempt fits ONE rational (Mobius) curve `y = (n+B)/(C*n+D)` to all
logged rows pooled together via the standard cross-multiply linearisation.
That curve is bounded and shaped like the truth — better than blindly
extrapolating a polynomial — but it ignores `s` entirely. Since the 3 gears
have different coefficients, one pooled curve is a single compromise
asymptote, systematically wrong for whichever gears it doesn't match; the
logbook's modest ship counts never force that gear-by-count interaction
into view.

The fix: split rows by gear `s` FIRST, fit the SAME cross-multiply
linearisation separately per gear (each gear's rows exactly determine its
own Mobius map, up to a harmless overall scale), then combine the three
curves into one expression in `(n, s)` — e.g. a degree-2 Lagrange selector
on `s`, no branching syntax required. A rational map recovered from
finitely many exact points is the SAME function at any `n`, so it
extrapolates correctly no matter how far the busy season pushes the count.

## Constraints
- `1 <= t <= 10^5` (test id); each log has 90 training rows; time limit 5s,
  memory 512MB; each `.in` file well under 5 MB.

## Example (worked score)
Suppose the baseline gives `metric = 0.618`, so `baseline_score ≈ 0.618 *
1.003 ≈ 0.6198`. A 54-node submission achieves `metric = 0.083`, so `O =
0.083 * (1 + LAMBDA*54) ≈ 0.0964`. Then `Ratio = min(1000, 100 * 0.6198 /
0.0964) / 1000 ≈ 0.643`.
