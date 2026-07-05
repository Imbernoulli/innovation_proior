# Grid Phase-Decoupling Matrix: Maximum-Determinant Sign Design

## Problem
A wide-area power grid is monitored by `N` phasor measurement units, one per bus. To
calibrate the grid you run `N` synchronised excitation patterns. In each pattern every
bus is driven either **in phase** (`+1`) or **anti-phase** (`-1`) relative to the
reference bus, giving an `N x N` sign matrix `M` whose row `p`, column `b` entry is the
phase of bus `b` in pattern `p`.

The patterns should be as **mutually decoupling** as possible: the more linearly
independent the excitation rows are, the better the calibration conditions the grid
state. The natural scalar measure of that decoupling is the magnitude of the
determinant, `|det(M)|` (the volume spanned by the pattern rows). Your job is to design
the sign matrix that makes `|det(M)|` as large as possible.

Two engineering conventions pin part of the matrix:
- The **base pattern** (row 0) drives every bus in phase: row 0 is all `+1`.
- The **reference bus** (column 0) is in phase in every pattern: column 0 is all `+1`.

Everything else is yours to choose in `{-1, +1}`.

## Input (stdin)
A single integer:
```
N
```
the number of buses (and the number of excitation patterns).

## Output (stdout)
`N` lines, each with `N` integers separated by spaces: the matrix `M`, row-major.
Every entry must be `+1` or `-1`. (Exactly `N*N` integer tokens are read; line breaks
are for readability only.)

## Feasibility
The output is rejected (score `Ratio: 0.0`) unless **all** hold:
- exactly `N*N` tokens, each equal to `1` or `-1`;
- `M[0][j] = +1` for all `j` (base pattern all in phase);
- `M[i][0] = +1` for all `i` (reference bus in phase in every pattern);
- `|det(M)| > 0` (a singular design is degenerate).

The determinant is computed **exactly** with fraction-free Bareiss integer elimination;
there is no floating-point tolerance in the objective.

## Objective (maximize)
`F = |det(M)|`, the exact integer absolute determinant.

## Scoring
Scoring is on a logarithmic scale between a trivial floor and the theoretical ceiling.
Let `B` be the checker's internal baseline — the normalised triangular sign matrix, for
which `|det| = 2^(N-1)`. Let the Hadamard bound be `H = N^(N/2)` (the maximum possible
`|det|` of any `N x N` +/-1 matrix). With `Lf = ln F`, `Lbase = ln B`, `Lcap = ln H`:
```
Ratio = clip( 0.1 + 0.9 * (Lf - Lbase) / (Lcap - Lbase),  0, 1 )
```
So the triangular baseline scores `0.1`, and a design that reaches the Hadamard bound
scores `1.0`. Because an exact Hadamard matrix is unattainable at most non-multiple-of-4
sizes, intermediate designs earn intermediate scores: the objective is genuinely graded.

## Constraints
`4 <= N <= 28`. All reference strategies run well within the time limit.

## Example
For `N = 4` the Sylvester-Hadamard matrix
```
 1  1  1  1
 1 -1  1 -1
 1  1 -1 -1
 1 -1 -1  1
```
is feasible with `|det| = 16 = 4^2`, exactly the Hadamard bound, so `Ratio = 1.0`.
The triangular baseline for `N = 4` has `|det| = 2^3 = 8`, giving `Ratio = 0.1`.
