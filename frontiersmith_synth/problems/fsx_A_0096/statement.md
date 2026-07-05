# Lunar Habitat Sensor Array: Low-Crosstalk Code Matrix

## Problem
The Shackleton lunar habitat runs `n` sensor channels (life-support, radiation,
seismic, ...) over a single shared power-line-communication bus. Each channel is
assigned a length-`n` spreading code of `+1`/`-1` chips. During one frame the bus
carries the `n x n` sign matrix `M` whose row `i` is channel `i`'s code.

The cross-talk between two channels is the correlation of their column patterns.
Writing `c_1, ..., c_n` for the **columns** of `M`, the pairwise interference is
`G_ij = c_i . c_j` (an integer dot product), and the total interference energy of
the assignment is

```
E(M) = sum over all i < j of (c_i . c_j)^2 .
```

The primary beacon channel is pre-provisioned: the operator fixes the **first row**
of `M` to a given code `r0`. You choose the remaining `n(n-1)` chips to make the
channels as mutually decorrelated as possible (small `E`). A perfectly orthogonal
(Hadamard) assignment would give `E = 0`, but for these frame sizes no such
assignment exists, so `E` is bounded strictly above zero and the layout is a
genuine combinatorial optimisation.

## Input (stdin)
```
n
r0[0] r0[1] ... r0[n-1]
```
Line 1: the frame size `n`. Line 2: `n` integers, each `+1` or `-1`, the fixed
first-row code `r0`.

## Output (stdout)
Print the full `n x n` sign matrix `M` in row-major order: `n*n` integers, each
`+1` or `-1`, separated by whitespace (newlines or spaces, any layout). Row 0
**must** equal `r0`.

## Feasibility
The output is rejected (score `0`) unless it contains exactly `n*n` values, every
value is `-1` or `+1`, and the first row equals `r0`.

## Objective (minimise)
Minimise the total interference energy `E(M)` defined above. All arithmetic is
exact integer arithmetic on the column dot products; there is no floating-point
tolerance.

## Scoring
The checker builds an internal baseline assignment `B` (a seeded pseudo-random
sign matrix, first row `r0`, lightly locally improved) and reports

```
Ratio = min(1000, 100 * E(B) / max(1, E(M))) / 1000 .
```

Reproducing the baseline scores about `0.1`; driving the interference energy an
order of magnitude below the baseline caps the ratio at `1.0`.

## Constraints
- `6 <= n <= 18`, and `n` is never a multiple of 4 (so a fully orthogonal layout
  is impossible and `E >= 1`).
- Deterministic scoring: identical output always earns the identical ratio.

## Example
For a small frame the fixed row might be `r0 = [1, 1, -1, 1, 1, -1]` (`n = 6`).
Emitting the baseline matrix scores `Ratio: 0.100000`; a matrix whose columns are
pairwise closer to orthogonal (smaller `E`) scores higher, e.g. halving the
energy roughly doubles the ratio.
