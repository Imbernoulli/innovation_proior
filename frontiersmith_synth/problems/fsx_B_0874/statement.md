# The Late Moon

A newly charted moon of a far planet keeps drifting behind its predicted
eclipse schedule. Mission control logs the **cycle number** `k` (an integer
count of orbits since a reference epoch) and the **observed eclipse time**
`t_k` (in days) for every transit their instruments actually caught. Many
transits are missed entirely — bad weather, wrong hemisphere, downtime — so
the log is a **sparse, gapped** set of `(k, t_k)` pairs, not a dense trace.

The true timetable is shaped by two independent physical effects layered on
top of a constant-period ephemeris:

- a **slow, ever-growing drift** (e.g. tidal deceleration slightly lengthening
  the orbit every cycle — an offset that keeps accumulating, unboundedly, the
  further out you look), and
- a **bounded periodic wobble** at some second, hidden period (e.g. a
  resonant tug from a sibling moon — an offset that oscillates back and
  forth but never grows past its amplitude).

Your job: recover a closed-form predictor of `t_k` from the sparse log that
still works **years after** the observing campaign ends.

## Input (stdin)

```
n_train  case_id
k_0  t_0
k_1  t_1
...
k_{n_train-1}  t_{n_train-1}
```

`case_id` is the test id. `n_train` training rows follow (a few dozen to
around a hundred), each an integer cycle number and its observed eclipse time
in days. The training log spans a campaign of a few years; consecutive `k`
are **not** all present (gaps).

## Output (stdout)

One line: a closed-form Python expression for `t(k)` in the single variable
`k`, using `+ - * / **`, unary `-`, numeric constants, the constant `pi`, and
the unary functions `sin cos sqrt exp log absv`. No other names are accepted.

**Illustrative FORM only — NOT the hidden law:** `12.5 + 0.02*sqrt(k) -
3*exp(-k/40)`. This just shows the allowed syntax; the real timetable has a
different shape you must discover from the data.

## Scoring (deterministic, minimisation)

Your expression is evaluated on a **held-out set of far-future cycles**,
regenerated inside the grader — roughly the cycles that would occur several
years *after* the training campaign ended (several times the length of the
campaign itself). Let `p_i` be your prediction and `t_i` the true (noisy)
eclipse time at held-out cycle `i`:

```
RMSE     = sqrt( mean_i (p_i - t_i)^2 )
F        = RMSE * (1 + LAMBDA * nodes)          # nodes = expression size
baseline = RMSE of a constant-period linear ephemeris a+b*k
           fit by least squares to YOUR SAME training rows
B        = baseline_RMSE * (1 + LAMBDA * 5)
Ratio    = min(1000, 100*B/F) / 1000
```

A plain linear ephemeris reproduces `B` (Ratio ≈ 0.1). Lower held-out error
raises the score (capped at `1.0`); `LAMBDA` is a small fixed parsimony
weight so a needlessly bloated expression is taxed. Non-finite, complex, or
malformed output scores `0`.

## Why the in-window fit is a trap

Within the few-year training window, a wobble whose true period is *longer*
than the campaign completes less than one cycle — it just looks like some
extra slow curvature, statistically **indistinguishable** from a plain
quadratic drift term given the sparse, noisy log. A generic "find the best
single stationary period, no matter how long" search will happily fit that
slow curvature as an oscillation and reproduce the training rows nearly
perfectly. But a genuine oscillation, however large its fitted period, stays
**bounded** forever, while genuine secular drift keeps **growing** — and only
one of those two behaviours survives extrapolation to the far-future grading
cycles. Telling the two apart requires reasoning about how each candidate
explanation *grows with horizon*, not about which one fits best inside the
window, because inside the window they can fit equally well.

## Constraints

Time limit 5 s, memory 512 MB. `n_train` up to a few hundred rows. Scoring is
fully deterministic; held-out noise leaves irreducible error, so even a
correct law does not reach `Ratio = 1.0` — there is room above the reference
solutions.
