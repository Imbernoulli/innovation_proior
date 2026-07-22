# Diagonal Parity Grid

## Problem

A storage cluster arranges its symbols on an `R x C` grid. Some cells hold **raw
data**; the rest hold **parity** symbols, each an exact linear combination (over
the finite field `GF(p)`, `p` prime) of some raw cells. The system must survive
the simultaneous loss of **any one entire row and any one entire column** — that
is, for *every* choice of erased row `r` (0-indexed, `0..R-1`) and erased column
`c` (`0..C-1`), the values of the raw cells inside row `r` or column `c` should be
reconstructible from the symbols that remain.

Reconstruction is a linear-algebra question. Fix an erasure `(r, c)`. Let `E` be
the raw cells lying in row `r` or column `c` (the erased raw cells; a raw cell at
the intersection `(r, c)` is only counted once). Each surviving parity cell (a
parity cell **not** in row `r` and **not** in column `c`) contributes one linear
equation in the unknowns `E` (its known coefficients on cells outside `E` move to
the constant side). The erased raw cells are uniquely recoverable **iff** the
matrix of these equations, restricted to the columns indexed by `E`, has rank
`|E|`. If `E` is empty the erasure is trivially "fully recovered".

## Input (stdin)

One line: `R C p` — grid dimensions (`5 <= R, C <= 9`) and a prime modulus
`p` (all arithmetic is mod `p`).

## Output (stdout)

Exactly `R*C` lines, one per cell in row-major order (`i = 0..R-1`, inner loop
`j = 0..C-1`). Each line is one of:

- `D` — this cell holds raw data.
- `P k i_1 c_1 i_2 c_2 ... i_k c_k` — this cell holds a parity value equal to
  `sum_{m=1..k} c_m * x[i_m] mod p`, where `x[0], x[1], ...` are the raw cells in
  the order they appear when scanning the grid row-major (the `t`-th `D` line you
  print, counting from 0, defines `x[t]`). Each `i_m` must be a valid raw-cell
  index (`0 <= i_m <` number of `D` cells), each `c_m` in `[0, p-1]`, `k >= 1`,
  and indices inside one `P` line must be pairwise distinct.

At least one cell must be `D` (a code with no raw data is invalid).

## Feasibility

The output is invalid (score `0`) if the format above is violated in any way
(wrong line count, bad token, out-of-range index or coefficient, repeated index
in one `P` line, zero `D` cells, etc).

## Scoring

Let `d` = number of `D` cells. For every one of the `R*C` erasures `(r,c)`
compute the recoverable fraction `frac(r,c) = rank / |E|` (or `1.0` if `E` is
empty), using the rank argument above over `GF(p)`. Your raw quality score is

```
F = d * ( (1/(R*C)) * sum_{r,c} frac(r,c) )
```

— the number of raw cells you kept, scaled down by how reliably the code
actually survives the crisscross erasure. The checker compares `F` against its
own safe reference construction to produce a normalized ratio in `[0,1]`
(`Ratio: <value>` on the last line). Both `d` (fewer parity cells) and full
recoverability (`frac(r,c) = 1` everywhere) push the score up — sacrificing
correctness for extra raw cells backfires, and so does over-provisioning parity
you didn't need.

## Constraints

`5 <= R, C <= 9`, `p` is a prime with `p > R*C + 10`. Time limit 5s.

## Example (illustrative only, not a full worked score)

For `R=C=2` (outside the real constraints, just to show the format): a valid
4-line output could read `D`, `D`, `D`, `P 1 0 1` — three raw cells and one
parity cell holding `1 * x[0]`. Whether this fully covers all 4 erasures
depends on the coefficient and is exactly the kind of rank check the checker
performs; real inputs have `R,C >= 5` where a single trivial parity cell is far
from enough.
