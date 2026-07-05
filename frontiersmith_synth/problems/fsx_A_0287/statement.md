# Wind-Tunnel Sensor Rail: Minimizing Sum-Aliasing

## Problem

A wind-tunnel test section has a straight instrumentation rail with `M+1` mounting
slots at integer stations `0, 1, ..., M`. You must bolt exactly `n` pressure sensors
onto **distinct** slots. Let `A ⊆ {0,...,M}`, `|A| = n`, be the set of chosen stations.

Two families of derived channels are formed from the sensor stations:

- **Differential channels** — one per *distinct* value `a - b` (`a,b ∈ A`). These
  measure pressure gradients between sensor pairs; more distinct gradient offsets means
  finer spatial resolution. The count is `|A - A|`.
- **Sum channels (aliases)** — one per *distinct* value `a + b` (`a,b ∈ A`). Standing
  acoustic modes in the closed tunnel excite a spurious response wherever two stations
  share a common sum `a + b`. Each *distinct* sum is a potential alias band; the count is
  `|A + A|`.

Well-separated aliasing is achieved by making the sum channels collapse onto **few**
distinct bands while the differential channels stay **rich**. Define the
**sum-aliasing ratio**

```
R(A) = |A + A| / |A - A|.
```

You want `R(A)` as **small** as possible: few distinct sum-alias bands relative to the
number of distinct differential offsets.

## Input (stdin)

One line with three integers:

```
n  M  seed
```

`n` = number of sensors to place, `M` = highest slot index, `seed` = a deterministic
tag you may use to seed any randomized search (it does not affect scoring).

## Output (stdout)

Exactly `n` integers (whitespace/newline separated in any layout): the chosen sensor
stations. They must be pairwise distinct and each in `[0, M]`.

## Feasibility

An output is feasible iff it parses as exactly `n` base-10 integers, all pairwise
distinct, all within `[0, M]`. Any other output (wrong count, duplicates, out-of-range,
non-integer / `nan` / `inf` tokens) scores `0`.

## Objective

Minimize `R(A) = |A+A| / |A-A|`, both computed as exact integer sumset cardinalities.

## Scoring

The checker builds an internal baseline `A0` = an evenly-spaced arithmetic progression
of `n` stations spanning `[0,M]`, and computes its ratio `B = R(A0)` (an AP satisfies
`|A0+A0| = |A0-A0|`, so `B = 1`). For a feasible submission with ratio `R`:

```
Ratio = min( 1.0 , 0.1 * (B / R)^3 )
```

Reproducing the arithmetic-progression baseline gives `R = 1` and `Ratio = 0.1`.
Driving the aliasing ratio below `1` raises the score; matching the AP or doing worse
caps you at `0.1`. Reaching `Ratio = 1.0` would require `R ≤ B / 10^(1/3) ≈ 0.464`,
far below any known construction for these sizes — there is no closed-form optimum, so
the frontier is genuinely open. Infeasible output scores `0`.

## Constraints

- `12 ≤ n ≤ 50`, `M = 5n` (so `60 ≤ M ≤ 250`).
- Scoring is exact integer arithmetic and fully deterministic.

## Example (worked score)

Suppose `n = 6`, `M = 30`. The AP baseline `A0 = {0,6,12,18,24,30}` has
`|A0+A0| = |A0-A0| = 11`, so `B = 1`.

- Submitting `A0` itself: `R = 1`, `Ratio = 0.1 * 1^3 = 0.1`.
- Submitting a difference-dominant set with `R = 0.8`:
  `Ratio = 0.1 * (1/0.8)^3 = 0.1953`.
- Submitting `A = {0, 30, 15}` (only 3 stations, `n` mismatch): infeasible, `Ratio = 0`.
