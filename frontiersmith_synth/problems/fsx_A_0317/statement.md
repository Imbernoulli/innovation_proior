# Waggle-Resonance Apiary: Maximizing Distinct Dance Channels

## Problem

A beekeeper manages a long orchard rail with mounting posts at the integer
positions `0, 1, ..., M`. She must bolt exactly `n` hive boxes onto **distinct**
posts. Let `A ⊆ {0, 1, ..., M}` with `|A| = n` be the set of chosen post
coordinates.

When two foraging bees return to the hive, the colony encodes their **combined
nectar bearing** as a waggle dance whose *resonance channel* is the **sum** of
the two boxes' post coordinates. A bee can pair with a forager from its own box
(so a box may pair with itself), hence the channel produced by boxes `a` and `b`
is `a + b` for any `a, b ∈ A` with the pair taken unordered. Two different pairs
that share the same coordinate sum collapse onto the **same** channel and become
indistinguishable to the colony.

The number of *distinct* resonance channels the apiary can encode is therefore
the exact sumset cardinality

```
    C(A) = |A + A|,   where   A + A = { a + b : a, b ∈ A }   (a = b allowed).
```

You want `C(A)` as **large** as possible: as many distinct waggle-dance
channels as the layout can support.

## Input (stdin)

One line with three integers:

```
n  M  seed
```

`n` = number of hive boxes to place, `M` = highest post index (posts are
`0..M`), `seed` = a deterministic tag you MAY use to seed any local search. The
`seed` does **not** affect scoring.

## Output (stdout)

Exactly `n` integers (whitespace / newline separated, any layout): the chosen
post coordinates. They must be pairwise **distinct** and each in `[0, M]`.

## Feasibility

An output is feasible iff it parses as **exactly `n`** base-10 integers, all
pairwise distinct, all within `[0, M]`. Any other output — wrong token count,
duplicates, out-of-range values, or non-integer / `nan` / `inf` tokens — scores
`0`.

## Objective

Maximize `C(A) = |A + A|`, computed as an exact integer sumset cardinality
(including the "self" sums `2a`).

## Scoring

The checker builds an internal baseline `A0 = {0, 1, ..., n-1}` — the trivial
arithmetic-progression apiary — and measures its channel count
`B = |A0 + A0| = 2n - 1`. For a feasible submission with channel count `F`:

```
    Ratio = min( 1.0 , 0.1 * (F / B) ).
```

Reproducing the arithmetic progression gives `F = B` and `Ratio = 0.1`.
Reaching `Ratio = 1.0` would require `F ≥ 10·(2n - 1)` distinct channels, far
beyond what fits on the tight rail (a *perfect* Sidon layout, in which **all**
`n(n+1)/2` pairwise sums are distinct, needs a rail of length `~ n²` and does not
fit here — so the perfect count is only an unreachable normalizer). There is no
known closed-form optimum for these sizes: the frontier is genuinely open.
Infeasible output scores `0`.

## Constraints

- `6 ≤ n ≤ 32`; the rail length is set near `M ≈ 0.58 · C(n,2)`, tight enough
  that a full Sidon layout cannot fit.
- Scoring is exact integer arithmetic and fully deterministic (no randomness, no
  timing).

## Example (worked score)

Suppose `n = 6`, `M = 10`. The baseline apiary `A0 = {0,1,2,3,4,5}` has
`A0 + A0 = {0,1,...,10}`, so `B = 11`.

- Submitting `A0` itself: `F = 11`, `Ratio = 0.1 · (11/11) = 0.1`.
- Submitting the spread layout `A = {0, 1, 2, 6, 8, 9}`: the distinct sums are
  `{0,1,2,3,4,6,7,8,9,10,11,12,14,15,16,17,18}` giving `F = 17`, so
  `Ratio = 0.1 · (17/11) ≈ 0.1545`.
- Submitting `A = {0, 5, 10}` (only 3 posts, `n` mismatch): infeasible,
  `Ratio = 0`.
