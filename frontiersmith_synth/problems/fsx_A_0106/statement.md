# Grand Atrium Gallery-Tour Contrast Matrix

## Problem
The Grand Atrium museum is designing `N` themed **gallery tours** across its `N`
exhibition halls. For each tour `i` and each hall `j` the curators pick a
**contrast polarity** `M[i][j] in {-1, +1}`: `+1` means the tour presents hall `j`
in its *bright* narrative framing, `-1` in its *shadow* framing. Two tours feel
distinct to a visitor exactly when their polarity vectors are not linear
combinations of the others — the whole `N`-tour portfolio is maximally
memorable when the polarity rows are as **linearly independent** as possible.

The single number that measures this joint independence is the **absolute
determinant** `|det(M)|` of the polarity matrix. Because every entry is `±1`,
`det(M)` is always an integer and is a multiple of `2^(N-1)`; the smallest
non-zero value is exactly `2^(N-1)`, and Hadamard's inequality caps it at
`N^(N/2)`, reached only by a Hadamard matrix.

Your job: choose all `N*N` polarities so that `|det(M)|` is as large as possible.
Because determinant magnitude grows super-exponentially, the score rewards the
**bit-length** of `|det(M)|` — the number of bits of tour-distinctiveness you
extract.

## Input (stdin)
```
N
```
A single integer `N`, the number of tours (= number of halls).

## Output (stdout)
Print the `N x N` polarity matrix as `N` lines, each with `N` space-separated
integers, each exactly `-1` or `1`. Print nothing else. The checker reads exactly
`N*N` integer tokens in row-major order.

## Feasibility
An output is valid iff **all** hold:
- it contains exactly `N*N` integer tokens;
- every token is exactly `-1` or `+1`.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = bit_length(|det(M)|)`, where `det(M)` is computed **exactly** with
Bareiss fraction-free integer elimination (no floating point anywhere in scoring).
A singular matrix (`det = 0`) has `F = 0`.

## Scoring
The checker builds its own trivial baseline — the **arrow array** `A` with
`A[0][j] = +1`, `A[i][i] = -1` for `i >= 1`, and `A[i][j] = +1` otherwise. This
valid `±1` matrix has the minimum non-zero determinant `±2^(N-1)`, so its bit
budget is `B = bit_length(2^(N-1)) = N`. With maximization normalization:
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the arrow baseline scores `Ratio = 0.1`. A Hadamard array (the
theoretical optimum) scores `Ratio = 0.1 * bit_length(N^(N/2)) / N`, roughly
`0.23` at `N = 24` and `~0.30` at `N = 64`: the score never trivially caps, so
there is a genuine gradient between random, hill-climbed, and structured arrays.

## Constraints
- `N` is drawn from `{24, 32, 44, 48, 60, 64}` (large scale).
- Time limit 5s, memory 512m.

## Example
For `N = 24` the arrow baseline has `|det| = 2^23` (`24` bits), giving
`Ratio = 100 * 24 / 24 / 1000 = 0.1`. A Paley-type Hadamard `24 x 24` array has
`|det| = 24^12` (`56` bits), giving `Ratio = 100 * 56 / 24 / 1000 = 0.2333`.
