# Warehouse Fleet Polarity Array with Pre-Wired Dock Cells

## Problem
A fulfillment warehouse runs a fleet of `N` autonomous floor robots against `N` inductive
guidance channels laid under the floor. Each robot `i` is assigned a **drive polarity**
`M[i][j] in {-1, +1}` on every channel `j` (its motion either reinforces or opposes the
signal on that channel). Stacked into an `N x N` matrix, these polarities form the fleet's
**guidance array** `M`.

The controller can only tell two fleet configurations apart if the guidance array is far from
degenerate. The relevant figure of merit is the **absolute determinant** `|det(M)|`: the
larger it is, the more independently the robots can be localized and the more robustly the
scheduler can resolve collisions. Because every entry is `+/-1`, `det(M)` is always an integer,
a multiple of `2^(N-1)`; the smallest achievable non-zero value is exactly `2^(N-1)`.

Complication: a subset of cells are **pre-wired** at the charging docks — for those `(i, j)`
the polarity is physically fixed and you must use the given value. You choose the remaining
(free) cells to make `|det(M)|` as large as possible.

Because `N` is **odd** here, no Hadamard array exists, and the pre-wired cells rule out simply
dropping in any known closed-form maximal-determinant construction — the best array must be
searched for.

## Input (stdin)
```
N F
r_1 c_1 v_1
r_2 c_2 v_2
...
r_F c_F v_F
```
`N` is the side length (odd). `F` is the number of pre-wired cells. Each of the next `F` lines
gives a 0-indexed row `r`, column `c`, and forced value `v in {-1, +1}` with `M[r][c] = v`.

## Output (stdout)
Print the `N x N` polarity array as `N` lines, each with `N` space-separated integers, every
one either `-1` or `1`. The checker reads exactly `N*N` integer tokens, row-major. Print
nothing else.

## Feasibility
An output is valid iff **all** hold:
- it contains exactly `N*N` integer tokens;
- every token is exactly `-1` or `+1` (non-integer / `nan` / `inf` / out-of-range tokens are rejected);
- every pre-wired cell matches: `M[r_k][c_k] = v_k`.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F_obj = bit_length(|det(M)|)`, where `det(M)` is computed **exactly** with Bareiss
fraction-free integer elimination (no floating point). A singular array (`det = 0`) has
`F_obj = 0`. Bit-length is the scored currency because `|det|` grows super-exponentially in `N`.

## Scoring
The checker uses the internal baseline `B = N`, the bit-length of `2^(N-1)` (the minimum
non-zero determinant of any `+/-1` array). With maximization normalization:
```
sc    = min(1000.0, 100.0 * F_obj / max(1e-9, B))
Ratio = sc / 1000.0
```
An array whose `|det|` has exactly `N` bits scores `Ratio = 0.1`. The determinant grows only
logarithmically in the score, so there is a genuine gradient between a random completion, a
best-of-many-samples completion, and a locally optimized array; the score never trivially caps.

## Constraints
- `N` is odd, `9 <= N <= 23`.
- `0 <= F <= N*N`, all pre-wired cells distinct, `v in {-1,+1}`.
- Time limit 5s, memory 512m.

## Example
Suppose `N = 9` with a handful of pre-wired cells. A non-singular random completion might reach
`|det|` with `10` bits, giving `Ratio = 100 * 10 / 9 / 1000 = 0.111`. A hill-climbed array that
respects the same pre-wired cells might reach `14` bits, giving
`Ratio = 100 * 14 / 9 / 1000 = 0.156`.
