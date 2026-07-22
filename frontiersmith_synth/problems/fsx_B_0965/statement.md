# The Brass-Tag Registry: Recovering a Multiplicative Law Across the Prime-Power Basis

## Problem

The town census office stamps every resident's registration tag with a serial number `n`
and, next to it, a second **registry number** `f(n)` pressed by a hand-cranked die. The
registrar's rulebook defines `f` for every positive integer, but the rulebook itself is lost —
you only have the clerk's ledger of stamped tags `n = 1 .. N` (some presses were worn, so a
few readings are off by a couple of counts), and you must reconstruct `f` well enough to
predict the registry numbers of tags kept in a separate archive box, numbered from **100000 to
1000000**, that the clerk never opened.

`f` is known to be a fixed **multiplicative** function: `f(1) = 1`, and `f(m*n) = f(m)*f(n)`
whenever `m` and `n` share no common factor. Nothing else about its shape is disclosed — in
particular, `f` need **not** be smooth or monotonic in `n`: two tags one apart can have wildly
different registry numbers if their prime factorizations differ.

## Input (stdin)

```
N
n_1 obs_1
n_2 obs_2
...
n_N obs_N
```

The first line is the number of ledger rows `N`. Each following row is a tag number
`n_i` (increasing) and its noisy observed registry number `obs_i` (a non-negative integer).

## Output (stdout)

Two lines:

```
MODE <N|PP>
<expression>
```

- **`MODE N`**: `<expression>` is a closed-form Python expression in the single variable `n`
  (the coordinate system the ledger data arrived in). The grader evaluates it once per
  withheld tag.
- **`MODE PP`**: `<expression>` is a closed-form Python expression in a prime `p` and an
  exponent `k` (the coordinate system of a prime power `p**k`). For each withheld tag, the
  grader factors it into its prime-power parts `p1**k1 * p2**k2 * ...`, evaluates your
  expression once per part, and **multiplies** the results together as the predicted `f`.

Allowed in either mode: numeric literals, the mode's variable(s), operators
`+ - * / % **` (unary `+ -` too), comparisons `== != < <= > >=`, `and`/`or`, a conditional
`a if cond else b`, and calls to `abs`, `min`, `max`. A `**` exponent must be a literal
integer (`|e| <= 12`) or, in `MODE PP`, a `+/-` combination of literal integers and `k`. No
other names, attributes, comprehensions, or tokens; `nan`, `inf`, `__` are forbidden.
Expression + mode line together are at most 2000 characters.

Illustrative **FORM only — not the hidden law**, just to show legal syntax:

```
MODE PP
abs(p - 2*k) + 5 if k > 1 else p
```

## Feasibility

The output must parse into exactly this two-line shape, use only the allowed tokens for its
declared mode, and evaluate to a finite real number for every prime-power part of every
withheld tag (`MODE PP`) or every withheld tag (`MODE N`). Any violation — bad mode line,
parse error, disallowed token, division by zero, non-finite or non-numeric result, missing
output — scores **0**.

## Objective (minimize)

Let `E` be the mean, over the withheld archive tags, of the relative error
`|predicted - true| / true` (capped at 5.0 per tag). The grader adds a fixed
**engraving-tolerance floor** to `E` (so even a perfectly recovered law is judged against a
non-zero irreducible tolerance) and a gentle penalty for very large expressions, giving `F`.

## Scoring

The grader also computes `B`, the same `F`-formula applied to the fixed trivial predictor
`f(n) = n`. The reported score is

```
Ratio = clamp(100 * B / F, 0, 1000) / 1000
```

Reproducing the trivial predictor scores about **0.10**. A predictor that tracks the true
multiplicative structure drives `E` toward zero and the score well above the baseline; the
engraving-tolerance floor keeps it below 1.0.

## Constraints

- `N` (ledger rows) and the stamping-noise level vary by test instance, `N <= 3000`.
- Withheld tags are drawn from `[100000, 1000000]`.
- Deterministic scoring: no randomness, wall-time, or hardware dependence.

## Example (worked score)

Suppose your expression achieves withheld mean relative error `E = 0.05` (so, after the
tolerance floor and any complexity penalty, `F = 0.30`), and the trivial predictor achieves
`B = 0.90` on the same withheld tags. Then `Ratio = 100 * 0.90 / 0.30 / 1000 = 0.30`. Matching
the trivial predictor (`F = B`) instead gives `Ratio = 0.10`.
