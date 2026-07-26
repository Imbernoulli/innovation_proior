# Silly Putty Remembers Every Squeeze

A lump of putty (a linear viscoelastic material) has stress at time `t` that
depends on its **entire strain history**, not just its current strain. By
the Boltzmann superposition principle, if `G(u)` is the material's hidden
*memory kernel* (its stress response, per unit strain, at lag `u` after a
strain step), then for **any** strain history `gamma(s)`:

```
sigma(t) = integral_0^t  G(t - s) * dgamma/ds(s)  ds
```

You are handed logs from **step-strain** experiments: a strain of magnitude
`gamma0` is applied instantaneously to a quiescent sample at time 0, so
`sigma(t) = gamma0 * G(t)` for `t > 0`. A few step magnitudes are logged
(superposition means each should reveal the same `G` once you divide out
`gamma0`), sampled at lags spanning roughly two decades. Your job: submit a
closed-form guess for `G(t)`.

You will be graded on **held-out strain histories you never see**: fast and
slow oscillatory straining, a mid-range oscillation, and a long strain ramp
— all far outside a simple "reproduce the logged curve" regime. A kernel
shape that only reproduces the step-response curve you were shown, but gets
these other histories wrong, scores poorly. Different candidate shapes that
fit the step data equally well can disagree wildly once you ask them to
predict a different kind of history — that disagreement is exactly what you
are being tested on.

**Illustrative FORM only — NOT the hidden kernel:**
`G(t) = 2.0 + 3.0/(1.0 + t)`
(shown only to demonstrate the output syntax; the real kernel shape must be
discovered from the data.)

## Input (stdin)
```
n_rows test_id
gamma0_1  t_1  sigma_1
gamma0_2  t_2  sigma_2
...
```
`n_rows` step-response readings follow (floats). `t` values are strictly
positive and span about two decades; `gamma0` takes a few distinct values.

## Output (stdout): ONE line
A single arithmetic expression in the variable `t` (representing your guess
for `G(t)`, valid for all `t > 0`) using only `+ - * / **`, parentheses,
numeric constants, and the unary functions `exp`, `log`, `sqrt`, `sin`,
`cos`, `absv`. No other names, no assignments, length <= 300 characters,
<= 40 expression nodes.

## Feasibility
The expression must parse under the grammar above and evaluate to a finite
value everywhere it is queried. Any parse failure, disallowed name/call, or
non-finite value at evaluation time scores `0`.

## Objective (minimise)
The grader rolls your `G(t)` forward, by **exact numerical convolution**,
against four held-out strain-rate histories built purely from the span of
`t` values you were given: one oscillating much **faster** than your fastest
observed lag, one **slower** than your slowest observed lag, one **inside**
your observed window, and a strain **ramp** run for several times the
window's span. Each prediction is compared to the true stress by a
normalised RMS error; the errors are combined in a fixed weighted average
(the extrapolation probes -- outside your window, and the long ramp --
together count for most of the weight, the in-window probe makes up the
rest), plus a small penalty proportional to
your expression's node count. Call this combined quantity `F`.

```
B = same combined quantity for an internal single-relaxation-time baseline
    (one exponential, fit by least squares to your training rows)
Ratio = min(1000, 100*B/F) / 1000
```

A flat/no-memory guess lands near the baseline (Ratio ~ 0.1). Getting the
step-response curve to fit well is necessary but not sufficient — kernel
shapes that fit the training window about equally well can differ by an
order of magnitude on the held-out histories.

## Why the obvious fit is a trap
Textbook viscoelastic fitting reaches for a **sum of exponentials**
(a generalised Maxwell / Prony series): pick a few relaxation times, least
squares the amplitudes, done — and with 2-3 terms it tracks two decades of
step-response data beautifully. But each exponential mode has its own
frequency-dependent phase response, and the training data alone cannot tell
you whether that's real or an artefact of only having ~2 decades to look at.
Query the fitted kernel far outside its fitted band (much faster or much
slower straining) and the prediction drifts, because a finite sum of
exponentials cannot reproduce a frequency-independent phase lag — while some
OTHER kernel shapes can. The training-window residual alone cannot tell
these apart; only the response to a *different kind* of history can.

## Constraints
Time limit 5 s, memory 512 MB. `n_rows` is at most a few dozen. Scoring is
fully deterministic (seeded RNG in the grader; no wall-time, no GPU).
