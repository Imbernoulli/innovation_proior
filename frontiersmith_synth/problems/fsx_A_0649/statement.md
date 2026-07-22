# Waxed-Sled Pull Response

A sled rests on waxed wood. Before each pull it has been sitting motionless
for `r` seconds (its "rest time"). You then ramp the applied force up to a
peak `F` and hold it there (a quasi-steady pull). A logger records the net
displacement `y` the sled ends up with after the pull. You are given many
such `(F, r, y)` triples, all recorded under **gentle** pulls (modest force,
modest rest time). Your job: predict `y` from `(F, r)` well enough to survive
**more aggressive** pulls and **longer naps** than you ever saw in training.

The mechanism behind the logger is a two-mode process, not a single formula:

* If the pull is too weak to overcome the sled's (aged) static-friction hold,
  it does not slide — but it still **creeps** a tiny amount, and that creep
  grows with how long it rested beforehand.
* If the pull overcomes the hold, the sled breaks free and slides under
  kinetic friction, producing a much larger displacement governed by how much
  the pull exceeds a (lower, fixed) kinetic floor — once sliding, how long it
  rested no longer matters.
* Which of these applies is decided by a hold threshold that itself **rises**
  the longer the sled has rested (a longer nap makes it harder to budge).

## Input (stdin)

```
n_train  test_id
F_1  r_1  y_1
F_2  r_2  y_2
...
F_n  r_n  y_n
```

All values are floats; `F_i, r_i >= 0`. The held-out grading data (wider force
and rest-time ranges than training) is generated only by the checker and is
never given to you.

## Output (stdout): one expression

Print a single line containing a Python-style expression over the variables
`F` and `r`. Allowed: `+ - * / **`, parentheses, numeric constants, the
one-argument functions `log`, `sqrt`, `exp`, `abs`, comparisons
(`<`, `<=`, `>`, `>=`, `==`, `!=`, no chaining), and a ternary
`A if cond else B`. No other names, calls, statements, or assignments.

**Illustrative FORM only — NOT the hidden law:**

```
(0.4 + 0.1*sqrt(F)) if (r > 5) else (0.2*F - 0.05)
```

This only shows the syntax (a threshold on `r` alone, square-root growth).
The real mechanism's guard and per-mode laws are different in shape and you
must discover them from the data.

## Feasibility

Your expression must parse under the grammar above using only known names.
Every value it produces on the held-out set must be finite. Any violation
scores `0`.

## Objective (minimise)

Let `MSE` be the mean squared error of your expression, evaluated pointwise
on the held-out `(F, r)` set, against the true held-out `y`. Let `nodes` be
the number of AST nodes in your expression (a light parsimony cost). The
checker forms

```
Fscore = MSE * (1 + LAMBDA * nodes)
B      = MSE_of(train_mean_y) * (1 + LAMBDA * 1)     # internal baseline
Ratio  = min(1000, 100 * B / Fscore) / 1000
```

with a small fixed `LAMBDA`. Predicting the constant training mean exactly
reproduces `B` (`Ratio ~= 0.1`). A single smooth formula that fits the
gentle training data well can still be far from the true two-mode law —
lowering held-out error raises the score, but sensor noise and the widened
held-out range keep even a good model off the ceiling.

## Why a smooth fit is a trap

Training data mixes gentle "stuck" pulls (small displacement, growing slowly
and logarithmically with rest time) and gentle "sliding" pulls (much larger
displacement, growing roughly linearly with force, independent of rest time).
A single continuous regression forced to explain both groups at once can only
find one intermediate slope — it will underpredict deep in the sliding regime
and overpredict for pulls that should barely creep, and this gets worse, not
better, on the wider force/rest ranges used for grading. The displacement
also **jumps** discontinuously right at the guard; no smooth formula can
reproduce that jump.

## Constraints

`n_train` is 54–90 rows. Time limit 5 s, memory 512 MB. All ten test cases
use the same underlying mechanism; only the sampled data, its noise level,
and how far the held-out set extrapolates beyond training vary with the test
id. Scoring is fully deterministic.
