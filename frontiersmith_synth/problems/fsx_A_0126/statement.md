# Geothermal Reservoir Polarity Grid: Maximal Flow-Independence

## Problem
A geothermal operator drills an `N x N` grid of injection/production wells into a fractured
reservoir. Each cell `(i, j)` is either **left dormant** (`M[i][j] = 0`) or **activated**
(`M[i][j] = 1`). Some cells are already cemented in place by an earlier survey and their
activation state is **fixed** — you may not change them.

When the field is pressurized, the *flow-independence* of the well network — how linearly
independent the `N` reservoir-response modes are — is governed by the **absolute determinant**
`|det(M)|` of the `0/1` activation matrix. A larger `|det(M)|` means the modes are cleaner and
more separable; a singular grid (`det = 0`) means the responses collapse and carry no usable
information.

Your job: choose the free (non-fixed) activations to make `|det(M)|` as large as possible.
Because determinant magnitude grows super-exponentially in `N`, the score rewards the number of
**bits of flow-independence** you extract, i.e. the bit-length of `|det(M)|`.

## Input (stdin)
```
N K
r_1 c_1 v_1
r_2 c_2 v_2
...
r_K c_K v_K
```
- `N` — side length of the well grid.
- `K` — number of fixed (cemented) cells.
- Each of the next `K` lines gives a 0-indexed cell `(r, c)` whose activation is fixed to
  `v in {0, 1}`. All fixed cells are distinct.

## Output (stdout)
Print the `N x N` activation matrix as `N` lines, each containing `N` space-separated integers,
every one of which is either `0` or `1`. Print nothing else. (The checker reads exactly `N*N`
integer tokens, row-major.)

## Feasibility
An output is valid iff **all** hold:
- it contains exactly `N*N` integer tokens;
- every token is exactly `0` or `1`;
- for every fixed cell `(r, c, v)` given in the input, `M[r][c] == v`.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = bit_length(|det(M)|)`, where `det(M)` is computed **exactly** with Bareiss
fraction-free integer elimination (no floating point). A singular matrix (`det = 0`) has
`F = 0`.

## Scoring
The checker builds its own trivial baseline `C`: a **block-diagonal grid** of `3x3` blocks, each
block being the pattern
```
1 1 0
0 1 1
1 0 1
```
which has determinant `2`, with any leftover `1` or `2` rows filled by a `1x1` (`[1]`) or
`2x2` (`[[1,1],[1,0]]`) unit-determinant block. This valid `0/1` matrix has
`|det(C)| = 2^(N // 3)`, so its bit budget is `B = bit_length(2^(N // 3)) = N // 3 + 1`.
(The fixed cells in every instance are sampled from exactly this baseline grid, so `C` is always
a feasible output.)

With maximization normalization:
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the baseline grid scores `Ratio = 0.1`. Because the maximal `0/1` determinant is far
below `2^(10 * (N//3))` bits, the score never trivially caps: there is a genuine gradient between
the baseline, random activations, greedy search, and near-optimal near-Hadamard grids.

## Constraints
- `8 <= N <= 22`.
- `0 <= K <= N*N`; every fixed cell value equals the baseline grid's value at that cell.
- Time limit 5s, memory 512m.

## Example
For `N = 8` the baseline grid has `|det| = 2^(8//3) = 2^2 = 4` (`3` bits), giving
`Ratio = 100 * 3 / 3 / 1000 = 0.1`. A well-optimized `8 x 8` `0/1` grid can reach `|det| = 32`
(`6` bits), giving `Ratio = 100 * 6 / 3 / 1000 = 0.2` — provided it still honors every cemented
cell.
