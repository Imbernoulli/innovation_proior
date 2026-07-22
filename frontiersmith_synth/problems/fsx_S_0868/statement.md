# The Alternate-Bearing Orchard

## Problem

An old estate orchard shows the classic "alternate bearing" pattern: a heavy
crop drains the trees' stored reserves, producing a light crop next year,
which lets reserves rebuild for another heavy crop, and so on — this year's
yield index `x(t)` depends on the **last two** years, not just the last one.

You are given the estate's earliest `n` years of yield records — a calm,
recently-started ledger — and must forecast the yield **many decades
further out**, in a held-out late window (as long again as your records, or
longer) never shown to you. Real ledgers are noisy: each year's yield is
nudged off the "clean" rule by a small unpredictable shock (weather, pests,
a late frost), so no forecaster can ever match the ledger exactly — but one
that captures the true rule stays close to the ledger's actual boom-bust
orbit for the whole late window; one that merely curve-fits the early years
does not.

## Input (stdin)

```
n t
x_0
x_1
...
x_{n-1}
```

`t` is the test id; `n` early-record yields follow, one float per line. The
held-out late window of the *same* ledger is **not** given to you.

## Output (stdout): one recurrence expression

Emit exactly one line:

```
OUT <expr>
```

`<expr>` is an arithmetic expression over `+ - * /`, parentheses, numeric
constants, and delay-tap variables:

- `x` — the most recent known (or, once rolling forward, most recently
  predicted) yield.
- `xkJ` — the yield `J` years further back (e.g. `xk2`), `J` in `1..6`.

The grader **rolls your expression forward autoregressively** starting the
year right after your records end: it seeds the taps with the last known
years exactly as printed to you, evaluates `OUT <expr>` to get next year's
yield, shifts that prediction into the taps, and repeats through the whole
held-out window. No further ground truth is ever fed back in; every later
prediction depends only on your own earlier predictions. The program must
use at most `30` expression nodes.

**Illustrative FORM only — NOT the hidden rule:**

```
OUT 0.4 + 0.3 * x - 0.1 * xk2 * xk2
```

This just shows the syntax; the real rule is a different shape and you must
discover it from the data.

## Feasibility

The expression must parse under the grammar above (known tap names only,
finite constants, delays `1..6`, node budget respected). Any parse
violation, or any non-finite value (or division-by-zero) produced anywhere
during the rollout, scores `Ratio: 0.0`.

## Objective (maximise)

Let `MSE` be the mean squared error of your rolled-out forecast against the
true held-out ledger, and `nodes` the size of your expression. The grader
forms a penalised error `Fpen = MSE*(1+LAMBDA*nodes)`, and the same
quantity `Bpen` for the internal baseline (a constant equal to the mean of
your training years), then

```
r     = Bpen / Fpen
Ratio = CAP * r / (r + 10*CAP - 1)      # CAP = 0.9
```

Reproducing the constant baseline (`r = 1`) scores exactly `Ratio = 0.1`;
lowering held-out error raises `r` and the score climbs smoothly, but
`Ratio` never reaches `CAP` even for an arbitrarily good forecast — the
ceiling is only approached, never reached — subject also to a parsimony tax
on expression size. Report the highest `Ratio` you can.

## Why the early ledger is a trap

Over the first few decades the yield only visits a narrow band of values, so
a flexible curve — even a plain polynomial in the last two years — nails
every recorded year almost exactly. But that curve is a local snapshot, not
the mechanism. Rolled forward through the held-out decades, its tiny fitting
error feeds back into its own next input every year and compounds; the
forecast drifts from the true boom-bust band, sometimes diverging outright.
The true rule leaves a fixed combination of consecutive yields ("how much
reserve capacity remains") almost unchanged year to year — recovering that
combination, not lowering the error on years you can see, keeps a long
rollout on track.

## Constraints

Time limit `5 s`, memory `512 MB`. `n` is a few dozen to a hundred rows.
Scoring is fully deterministic: no randomness, wall-time, or GPU.
