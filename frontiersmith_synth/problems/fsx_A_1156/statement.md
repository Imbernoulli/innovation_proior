# Harbor Tide Chart — Locked Celestial Gears

An old harbor keeps a tide-height logbook `y(t)` (meters above a reference
datum, `t` in ticks). Local lore says the tide is driven by **four
"gears"**: three are meshed together on one hidden shared shaft — their
frequencies are locked in an **exact small-integer ratio** `n1:n2:n3` around
one unknown base rate `f0` — and a **fourth, genuinely free** gear turns at
its own unrelated rate. Recover a closed-form predictor for `y(t)`.

Harbor archives narrow the shared base rate to a period of roughly **150–225
ticks**, and catalogue only this short menu of historically observed
integer-ratio triples for `(n1,n2,n3)`:

```
(2,3,7) (3,4,5) (2,5,7) (3,5,8) (2,3,11) (4,5,7) (2,7,9) (3,7,8)
```

The free fourth gear is not on this list and has no small-integer relation
to `f0`; it turns noticeably faster than the locked trio.

## Input (stdin)
- Line 1: `n t` — number of training rows and the case id.
- Next `n` lines: `t_i y_i`, one logbook reading each (floats), covering a
  window that spans **only part of one locked super-period** (`1/f0`).

## Output (stdout)
One line: a closed-form Python expression for `y` in the single variable
`t`. Allowed: `+ - * / **`, unary `-`, numeric constants, and the functions
`sin`, `cos`. No other names are accepted.

**Illustrative FORM only — NOT the hidden law:** `0.4*sin(0.03*t+1.1) + 0.2`
— this just shows valid syntax; the real law has four terms whose exact
rates and integers you must discover from the data.

## Scoring (deterministic, minimization)
Your expression is evaluated on a **held-out window that starts several
locked super-periods after the training window ends** (regenerated inside
the grader). Let `p_i` be your prediction and `y_i` the true (noisy) tide
height at held-out point `i`:

```
metric   = mean_i  min(1, |p_i - y_i| / (|p_i| + |y_i| + 1e-6))
O        = metric * (1 + LAMBDA * nodes)          # nodes = expr size
baseline = the same metric for the constant predictor mean(train y)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error raises `Ratio` (capped at `1.0`). A constant predictor
scores about `0.1`. `LAMBDA` is a small fixed parsimony weight. Non-finite
predictions score `0`.

## Why the obvious fit is a trap
The training window is deliberately shorter than one full locked
super-period, so the three meshed gears sit **closer together in frequency
than that short window can resolve**. Fitting all four frequencies as
independent unknowns (e.g. by successive single-frequency extraction) is
therefore an ill-posed 4-parameter problem: many different frequency
quadruples reproduce the training log almost equally well, and whichever
one a generic fit lands on is essentially arbitrary among them. Any
resulting per-frequency error, however small, keeps accumulating phase
error linearly in time — by the held-out window, several super-periods
later, that phase has effectively randomized.

The catalogue changes the problem: instead of fitting four unrelated
frequencies, hypothesize one candidate integer triple at a time and fit a
**single shared base rate `f0`** jointly against all three harmonics (their
fastest harmonic pins `f0` down far more precisely than any lone-frequency
fit of a short window could). Because the multipliers are then forced to be
**exact integers**, there is no per-component ratio error left to drift —
the three locked terms stay phase-coherent no matter how far out you
extrapolate. Fit the leftover free gear separately on what remains; forcing
it onto the integer lattice too would badly misfit it, since it genuinely
isn't locked.

## Constraints
Time limit 5 s, memory 512 MB. `n` is a few hundred rows; the submitted
expression is limited to 120 AST nodes. Scoring is fully deterministic.
