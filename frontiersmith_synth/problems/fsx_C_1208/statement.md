# Regressing Out the Recommender's Own Push

## Problem

An item on a platform has a genuine **organic** interest trajectory: some
honest, slowly drifting baseline appeal that would exist even if nobody
ever recommended it. Separately, a recommender decides each period how much
**exposure** to give the item (the fraction of impression slots it gets) --
and that policy **reacts to the item's own recent engagement**: whenever the
item outperforms its baseline, the recommender pushes it harder next
period. This is an exposure-feedback loop, and it is the mechanism behind
"rich get richer" popularity bias: an item can look explosively popular
purely because the system keeps showing it more, not because organic
interest is rising at all.

Every period's measured engagement blends both effects:

```
engagement(period) = organic_interest(period) + induced_effect(exposure)
```

You are given a logged history: for each period, how much exposure the
recommender actually gave the item and what engagement resulted. **Within
this logged history, exposure and period are confounded** -- because
exposure reacts to engagement, and engagement drifts with the period, the
exposure the recommender chose also drifts with the period across the
window you can see.

Your job: emit a closed-form expression predicting engagement as a function
of the period `t` and the exposure fraction `x` it would receive, accurate
not only on periods like the ones you logged, but on a **later, held-out**
stretch where the platform ran an intervention -- it set exposure itself
(no longer letting the recommender's adaptive policy choose it), breaking
the feedback loop.

**Illustrative FORM only -- NOT the hidden law (unrelated shape, do not
pattern-match it):** `2.5 - 0.8*sqrt(t) + 1.1*tanh(x) - absv(t - x)`

## Input (stdin)

```
id n_train
t_1 x_1 e_1
t_2 x_2 e_2
...
t_{n_train} x_{n_train} e_{n_train}
```

`id` is the test id (informational). `n_train` is the number of logged
periods. Each row is an integer period index `t_i` (starting at 1), the
fraction of impression slots the recommender gave the item that period
(`x_i`, in `[0,1]`), and the item's measured engagement rate that period
(`e_i`, a non-negative float).

## Output (stdout)

One line: a closed-form Python-syntax expression for engagement in the two
variables `t` and `x`. Allowed: `+ - * / **`, unary `-`, numeric constants,
and the functions `sqrt log exp sig tanh absv`. No other names are
accepted.

## Feasibility

The output must parse under the grammar above (only the listed names/
functions, finite numeric constants, at most 60 expression-tree nodes) and
must evaluate to a finite real number at every held-out `(t, x)`. Any
violation scores `0`.

## Scoring (deterministic, maximization)

Your expression is evaluated on a **held-out set of (period, exposure)
pairs**, regenerated inside the grader, all at periods strictly past every
training period, with the exposure at each point drawn **independently of
history** -- the intervention that breaks the feedback loop. Let `p_i` be
your prediction and `e_i` the true (noisy, finite-sample) engagement at
held-out point `i`:

```
metric   = mean_i min(3.0, |p_i - e_i|)        # clipped absolute error
O        = metric * (1 + LAMBDA * nodes)        # nodes = expression size
baseline = the same metric for the constant predictor mean(train engagement)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error raises `Ratio` (capped at `1.0`); reproducing the
constant baseline scores about `0.1`. `LAMBDA` is a small fixed parsimony
weight. Non-finite predictions score `0`.

## Why the visible log is a trap

Across the logged window, exposure reacted to the item's own recent
engagement, so exposure drifted upward together with the period whenever
engagement was drifting up. A curve fit to raw `(t, engagement)` alone
therefore cannot tell genuine organic drift apart from the loop's own
self-reinforcing amplification -- its fitted slope is a blend of both,
extrapolated as if it were all organic. Once the held-out period arrives
and exposure is set by intervention instead of the adaptive policy, that
blended slope keeps compounding a growth rate the item never organically
had. The exposure log you were given is the only thing that lets you
separate the two: engagement as a function of *both* the period and the
exposure it actually received lets the induced contribution be regressed
out, leaving the true organic trend behind.

## Constraints

Time limit 5 s, memory 512 MB. `n_train` is small (well under a hundred
rows). Held-out finite-sample noise leaves irreducible error, so even the
correct law does not reach `Ratio = 1.0` on every case.
