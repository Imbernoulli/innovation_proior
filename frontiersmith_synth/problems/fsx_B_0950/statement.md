# The Rug Merchants' Haggling Ledger

## Problem

In the sprawling rug bazaar of Old Marresh, every haggling round ends in a
settled price. A trader privately appraises a rug lot at `v` silver marks. If
the trader is the only serious bidder, haggling favors the seller and the
settled price is shaded well below `v`. But when rumor spreads that `n` rival
traders are circling the same lot, the seller holds firmer, and the settled
price creeps back toward the trader's true appraisal. Seasoned merchants also
tack on an extra "appraisal-risk surcharge" whenever a rug's value looks
unusual relative to the bazaar's benchmark price — unusual rugs are harder to
authenticate, so haggling over them is warier.

You are given logged haggling records from one bazaar's ledger: the number of
rival traders `n` and the trader's private appraisal `v`, together with the
settled price that emerged. Your job: write ONE formula predicting the settled
price from `n` and `v`. It will be graded on a wilder trading day you never
observe.

## Input (stdin)

```
num_rows testId mu
n_1 v_1 price_1
n_2 v_2 price_2
...
```

`mu = 100.0` is the bazaar's fixed benchmark rug price (same every day, every
lot). Training rows have `n` between 2 and 6 inclusive (a quiet day — few
rivals ever show up at once), and `v` a few tens to a couple hundred silver
marks.

## Output (stdout)

Exactly one line: a single arithmetic expression in the variables `n` and `v`
using `+ - * / **`, parentheses, numeric constants, unary minus, and the
two-argument functions `min(a, b)` / `max(a, b)` only. No other names,
functions, or extra lines.

## Feasibility

The expression must parse under the grammar above (only names `n`, `v`; only
the listed operators and `min`/`max`; every numeric constant finite) and must
evaluate to a finite number on every graded `(n, v)` pair. Any violation
scores `0`.

## Objective (minimize)

Let `MAE_you` be the mean absolute error of your predicted price against the
true settled price on a **held-out ledger** from the SAME bazaar, the SAME
haggling rule. This is not interpolation: the held-out ledger includes days
with **far more rival traders** than any training row shows, and lots whose
appraisal is **far outside** your training range — sometimes both at once.

Let `MAE_base` be the mean absolute error of the naive "quote the full
appraisal" prediction `price = v` (a trader who assumes no haggling ever
happens) on that same held-out ledger. Both errors get the same small fixed
floor `K` added (a calibration constant, so a lucky near-perfect fit cannot
blow the ratio past its cap) before the ratio:

```
F = MAE_you  + K
B = MAE_base + K
Ratio = min(1000, 100 * B / max(1e-9, F)) / 1000
```

Predicting `price = v` reproduces the baseline (`Ratio = 0.1` exactly, since
then `F = B`); driving `MAE_you` below `MAE_base` raises the score.

**Illustrative FORM only — NOT the hidden haggling rule** (this just shows
valid syntax):

```
max(0, min(1, 0.1 + 0.05 * n)) * (v - 3) - v / 2
```

The real rule is a different shape entirely; discover it from the data — do
not pattern-match this example.

## Economics hint (not the formula)

As the rival count grows without bound, whatever advantage a lone bidder
enjoyed must vanish — an infinitely competitive lot settles at the full
appraisal, no less. And the risk surcharge is a property of how far the
appraisal sits from the bazaar's benchmark price, not of how many rivals show
up. These are structural facts about the bazaar, not the formula itself — you
still have to recover its exact shape and constants from the ledger.

## Constraints

Training rows: `2 <= n <= 6`, roughly 15–50 rows depending on the test.
`v` and `price` are positive floats, at most a few hundred silver marks in
training (the held-out ledger can range wider on both `n` and `v`). Time
limit 5 s, memory 512 MB. Scoring is fully deterministic.
