# Signed Corridor Robustness Matrix (Wildlife Corridor)

## Problem
A conservation network links `N` habitat patches. Between every ordered pair of patches
`(i, j)` a proposed movement corridor carries a *directed sign* `A[i][j] in {-1, +1}`:
`+1` if the corridor reinforces flow from patch `i` into patch `j`, `-1` if it opposes it.
The full `N x N` sign matrix `A` describes the whole network.

The **robustness** of the network is measured by how linearly independent its corridor-flow
vectors are: the larger the absolute determinant `|det(A)|`, the more the corridors act as an
independent, non-redundant set (a fully orthogonal / Hadamard-like network is the ideal).

Terrain forces some links: a set of `K` corridors have signs fixed by geography (rivers,
ridgelines) and cannot be changed. You must choose every remaining sign to make the network as
robust as possible.

## Input (stdin)
```
N K
r1 c1 v1
r2 c2 v2
...
rK cK vK
```
- `N` — number of habitat patches (matrix order), `4 <= N <= 7`.
- `K` — number of terrain-fixed corridors.
- Each of the next `K` lines: a 0-indexed cell `(r, c)` and its fixed sign `v in {-1, +1}`.
  All fixed cells are distinct.

## Output (stdout)
Print the `N x N` matrix `A`: `N` rows, each with `N` space-separated integers, every entry in
`{-1, +1}`. (Whitespace and line breaks are free-form; the checker reads `N*N` integer tokens
in row-major order.)

## Feasibility
- Exactly `N*N` integer tokens must be emitted.
- Every entry must be `-1` or `+1`.
- Every terrain-fixed cell `(r, c)` must satisfy `A[r][c] == v`.
Any violation scores `0`.

## Objective (maximize)
Maximize `|det(A)|`, computed **exactly** by integer Bareiss elimination (no floating point).

## Scoring
Let `F = |det(A)|` for your matrix. The checker builds an internal baseline `B` = `|det|` of the
*terrain-default completion* (every free cell set to `+1` if `j >= i`, else `-1`, then the fixed
cells overwritten). The score is
```
Ratio = min(1000, 100 * F / B) / 1000
```
Reproducing the terrain-default matrix scores about `0.1`; reaching ten times the baseline
robustness caps at `1.0`. The determinant is a genuinely hard combinatorial objective (the
maximal-determinant `+/-1` matrix problem), so there is no easy optimum.

## Constraints
- `4 <= N <= 7`, `2 <= K < N*N`.
- Deterministic exact-integer scoring.

## Example
For `N = 2`, `K = 1`, fixed cell `(0,0) = +1`:
```
 1  1
-1  1
```
has `det = 2`, `|det| = 2`. The terrain default `[[1,1],[-1,1]]` also has `|det| = 2`, so this
choice scores `100 * 2 / 2 / 1000 = 0.1`. Choosing the second row to make the rows more
independent is how you beat the baseline on larger instances.
