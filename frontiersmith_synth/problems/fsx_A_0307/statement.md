# Tide-Pool Transect: Maximizing Spawning-Node Richness

## Problem

A marine ecologist surveys a rocky intertidal shelf along a straight transect with
`M+1` survey stakes driven at integer positions `0, 1, ..., M`. She will seed exactly
`n` **tide pools** at **distinct** stake positions. Let `A ⊆ {0,...,M}`, `|A| = n`,
be the set of chosen positions.

Two families of ecological structures emerge from the pool layout:

- **Drift corridors.** Between any two pools at positions `a` and `b` the receding tide
  carves a channel whose *length* is `|a - b|`. Two pool-pairs that share the same
  separation carve the **same** corridor length, so the number of *distinct* corridor
  lengths is the difference-set cardinality `|A - A|` (which also counts the sign, i.e.
  it is the size of `{a-b : a,b ∈ A}`).
- **Spawning nodes.** During the flood tide, larvae released from pools at positions
  `a` and `b` converge and settle at the midpoint feature indexed by the *sum* `a + b`.
  Each **distinct** value of `a + b` is a separate viable spawning node. The number of
  spawning nodes is the sumset cardinality `|A + A|`.

A resilient reef wants **many independent spawning nodes** relative to a **small number
of drift-corridor types** (redundant corridors concentrate flow and are ecologically
cheap). Define the **spawning-node richness**

```
R(A) = |A + A| / |A - A|.
```

You want `R(A)` as **large** as possible. Because every difference `a-b` has a mirror
`b-a` while sums are unsigned, a naively symmetric (evenly spaced) layout gives exactly
`R = 1`. Beating `R = 1` — a *more-sums-than-differences* layout — requires a genuinely
asymmetric arrangement.

## Input (stdin)

One line with three integers:

```
n  M  seed
```

`n` = number of tide pools to seed, `M` = highest stake index, `seed` = a deterministic
tag you may use to seed any randomized search (it does **not** affect scoring).

## Output (stdout)

Exactly `n` integers (whitespace/newline separated, any layout): the chosen pool
positions. They must be pairwise **distinct** and each in `[0, M]`.

## Feasibility

An output is feasible iff it parses as exactly `n` base-10 integers, all pairwise
distinct, all within `[0, M]`. Any other output (wrong count, duplicates, out-of-range,
non-integer / `nan` / `inf` tokens, extra garbage) scores `0`.

## Objective

**Maximize** `R(A) = |A+A| / |A-A|`, both computed as exact integer sumset/difference-set
cardinalities.

## Scoring

The checker builds an internal baseline `A0` = an evenly-spaced arithmetic progression
of `n` stations, and computes its richness `B = R(A0)`. An arithmetic progression
satisfies `|A0+A0| = |A0-A0|`, so `B = 1`. For a feasible submission with richness `R`:

```
Ratio = min( 1.0 , 0.1 * (R / B) ** 18 )
```

Reproducing the arithmetic-progression baseline gives `R = 1` and `Ratio = 0.1`.
Pushing the richness above `1` (a more-sums-than-differences layout) raises the score
steeply; matching the AP or doing worse caps you at `0.1`. Reaching `Ratio = 1.0` would
require `R ≥ 10**(1/18) ≈ 1.136`, far above any layout known for these sizes — the exact
maximum of `|A+A|/|A-A|` is an open question in additive combinatorics, so the frontier
is genuinely unbounded-from-below-known. Infeasible output scores `0`.

## Constraints

- `10 ≤ n ≤ 28`, `M = 2n` (so `20 ≤ M ≤ 56`).
- Scoring is exact integer arithmetic and fully deterministic. No time/memory is scored.

## Example (worked score)

Suppose `n = 8`, `M = 16` (illustrative — outside the shipped ladder). The AP baseline
`A0 = {0,2,4,6,8,10,12,14}` has `|A0+A0| = |A0-A0| = 15`, so `B = 1`.

- Submitting `A0` itself: `R = 1`, `Ratio = 0.1 * 1**18 = 0.1`.
- Submitting the classic more-sums set `A = {0,2,3,4,7,11,12,14}`: `|A+A| = 26`,
  `|A-A| = 25`, so `R = 1.04` and `Ratio = 0.1 * 1.04**18 ≈ 0.203`.
- Submitting `A = {0, 8, 16}` (only 3 pools, `n` mismatch): infeasible, `Ratio = 0`.
