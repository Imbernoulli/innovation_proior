# Resonant Polarity Array for Asteroid Ore Scanning

## Problem
A deep-space mining rig scans an asteroid with an `N x N` phased sensor array. Each cell
`(i, j)` of the array must be assigned a **resonance polarity** `M[i][j] in {-1, +1}` (the
antenna element is either forward- or reverse-biased). When the array pings the ore body,
the *separation power* of the returned signal — how cleanly the `N` ring returns can be told
apart — is governed by the **absolute determinant** `|det(M)|` of the polarity matrix.

Because every entry is `±1`, the determinant is always an integer, and it is a well-known
fact that `det(M)` is a multiple of `2^(N-1)`; the smallest achievable non-zero value is
exactly `2^(N-1)`. The largest possible value is bounded by Hadamard's inequality,
`|det(M)| <= N^(N/2)`, attained exactly when the array is a Hadamard matrix.

Your job: assign the polarities to make `|det(M)|` as large as possible. The score rewards
the **number of bits of separation power** you extract, i.e. the bit-length of `|det(M)|`,
because determinant magnitude grows super-exponentially and bits are the meaningful currency.

## Input (stdin)
```
N
```
A single integer `N`, the side length of the sensor array. `N` is always even.

## Output (stdout)
Print the `N x N` polarity matrix as `N` lines, each containing `N` space-separated integers,
every one of which is either `-1` or `1`. Print nothing else. (The checker reads exactly
`N*N` integer tokens, row-major.)

## Feasibility
An output is valid iff **all** hold:
- it contains exactly `N*N` integer tokens;
- every token is exactly `-1` or `+1`.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = bit_length(|det(M)|)`, where `det(M)` is computed **exactly** with Bareiss
fraction-free integer elimination (no floating point). A singular matrix (`det = 0`) has
`F = 0`.

## Scoring
The checker builds its own trivial baseline: the **arrow array** `A` with `A[0][j] = +1` for
all `j`, `A[i][i] = -1` for `i >= 1`, and `A[i][j] = +1` otherwise. This valid `±1` matrix
has `det(A) = ±2^(N-1)`, the minimum non-zero determinant, so its bit budget is
`B = bit_length(2^(N-1)) = N`.

With maximization normalization:
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the arrow baseline scores `Ratio = 0.1`. A Hadamard array (the theoretical
optimum) scores `Ratio = 0.1 * bit_length(N^(N/2)) / N`, e.g. `~0.22` at `N = 20` and
`~0.25` at `N = 32` — the score never trivially caps, so there is a genuine gradient between
random, structured, and optimal arrays.

## Constraints
- `N` is even, `8 <= N <= 32`.
- Time limit 5s, memory 512m.

## Example
For `N = 8`, the arrow baseline has `|det| = 2^7 = 128` (`8` bits), giving
`Ratio = 100 * 8 / 8 / 1000 = 0.1`. A Sylvester–Hadamard `8 x 8` array has
`|det| = 8^4 = 4096` (`13` bits), giving `Ratio = 100 * 13 / 8 / 1000 = 0.1625`.
