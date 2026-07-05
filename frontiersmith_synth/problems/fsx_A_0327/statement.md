# Sum-Rich Relay Beacons

## Problem

A mountain rescue service is deploying `n` radio relay beacons along a steep ridge. Each
beacon sits at a distinct integer **altitude** `a` (metres above the valley floor), with
`0 <= a <= V`.

When two beacons (possibly the same one) cooperate, the control centre records two things:

- a **handshake code** `a + b` (the *sum* of the two altitudes), and
- an **altitude gap** `a - b` (the *difference* of the two altitudes).

Let `A` be the multiset-free set of chosen altitudes. Define

- `A + A = { a + b : a, b in A }`  — the set of **distinct handshake codes**,
- `A - A = { a - b : a, b in A }`  — the set of **distinct altitude gaps**.

You want a **sum-rich** deployment: as many distinct handshake codes as possible *relative
to* the number of distinct altitude gaps. Concretely, maximize the exponent

```
rho(A) = |A + A| / |A - A|.
```

For almost every layout `rho(A) < 1` (differences are naturally more plentiful than sums).
Layouts with `rho(A) > 1` — "more sums than differences" configurations — are rare and must
be engineered. How large `rho` can be made is a genuinely open extremal question, so there
is no known optimal construction to copy.

## Input (stdin)

A single line with two integers:

```
n V
```

`n` is the number of beacons to place; `V` is the inclusive upper altitude bound. You may
assume `V` is large enough that a fully difference-spread layout fits.

## Output (stdout)

Exactly `n` integers (whitespace- and/or newline-separated): the beacon altitudes, i.e. the
set `A`. They must be **distinct** and each in `[0, V]`. Order does not matter.

## Feasibility

An output is feasible iff it lists exactly `n` distinct integers, each within `[0, V]`.
Any violation (wrong count, duplicate, out-of-range, or non-integer / non-finite token)
scores `0`.

## Objective

Maximize `rho(A) = |A + A| / |A - A|` (exact integer set cardinalities).

## Scoring

The checker computes `rho(A)` exactly, then normalizes against an internal baseline `B` =
the ratio of a deterministic Sidon (Mian-Chowla) layout of the same size `n` (a
difference-heavy set with `B ~ 0.5`):

```
sc    = min(1000, 100 * rho(A) / B)
Ratio = sc / 1000       (printed on the final line)
```

Reproducing the difference-heavy baseline scores `~0.1`. Higher `rho` scores higher;
the cap at `Ratio = 1.0` corresponds to a ratio `10x` the baseline, far beyond what any
known integer set achieves, so the objective stays open-ended with plenty of headroom.

## Constraints

- `20 <= n <= 80`
- `V = 8 * n * n`
- Deterministic scoring; exact arithmetic; no randomness in the score.

## Example (worked score)

Suppose `n = 4`, `V = 128`, and the baseline Sidon ratio for `n = 4` is `B = 0.7`.

- Output `0 1 2 3` (an interval): `A+A = {0,...,6}` so `|A+A| = 7`; `A-A = {-3,...,3}` so
  `|A-A| = 7`; `rho = 1.0`. Score `= min(1000, 100 * 1.0 / 0.7) / 1000 = 0.1428...`.
- Output `0 2 3 4` (a small MSTD-flavoured set): `A+A = {0,2,3,4,5,6,7,8}`, `|A+A| = 8`;
  `A-A = {-4,-3,-2,-1,0,1,2,3,4}`, `|A-A| = 9`; `rho = 0.888...`. Score `~ 0.127`.

(The example sizes are illustrative; real instances have `n >= 20`.)
