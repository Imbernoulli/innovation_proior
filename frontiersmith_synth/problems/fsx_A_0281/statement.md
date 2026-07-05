# Aurora Relay Grid: Corner-Free Placement on the Ice Torus

## Problem

An Antarctic research base lays out communication relays on a wrap-around ice
grid modelled as the torus `Z_n x Z_n` (rows and columns wrap: cell `n` is cell
`0`). Some cells are **crevasses** and cannot host a relay.

Three relays interfere catastrophically when they form a **resonance corner**:
cells `(x, y)`, `((x+d) mod n, y)` and `(x, (y+d) mod n)` for some step
`d in {1, ..., n-1}` — i.e. a right-angle "L" of two equal legs (one horizontal,
one vertical) with a shared corner. Any placement that contains such a triple is
forbidden.

Place as many relays as possible on non-crevasse cells so that **no resonance
corner** appears. This is the corner-free-set problem on `Z_n x Z_n`; its exact
maximum is not known in closed form, so many heuristics compete.

`n` is always **odd**.

## Input (stdin)

```
n
k
x_1 y_1
...
x_k y_k
```

`n` is the (odd) side length. `k` is the number of crevasse cells; each of the
next `k` lines gives a blocked cell `(x_i, y_i)` with `0 <= x_i, y_i < n`.

## Output (stdout)

```
s
X_1 Y_1
...
X_s Y_s
```

First the number of relays `s`, then `s` distinct cells `(X_j, Y_j)`, each with
`0 <= X_j, Y_j < n` and not a crevasse. The exact token count must match
(`1 + 2*s` integers, all finite integers).

## Feasibility

Rejected (score 0) if any token is non-integer/non-finite, the count is wrong,
a coordinate is out of range, a cell repeats, a chosen cell is a crevasse, or the
chosen set contains any resonance corner.

## Objective

Maximize `F = s`, the number of relays placed.

## Scoring

The checker builds a baseline `B` = the number of unblocked cells in the fullest
single grid row (a single row is always corner-free). Your score is

```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```

So a full single row scores about `0.1`; you must beat the row baseline
multiplicatively. Ten equally sized cases; the reported score is the mean Ratio.

## Constraints

- `9 <= n <= 35`, `n` odd.
- Crevasses cover roughly 3%–6% of cells.
- Time limit 5 s, memory 512 MB.

## Example (worked score)

Suppose `n = 9` with no crevasses, so `B = 9`. A submission that places the two
diagonals `(x - y) mod 9 in {0, 1}` uses `2*9 = 18` relays; one checks that no
resonance corner exists (the difference set `{0,1}` has no 3-term progression, and
`9` is odd). Then `F = 18`, `sc = min(1000, 100*18/9) = 200`, and
`Ratio = 0.200`.

*(This diagonal example is only an illustration of scoring — larger progression-free
difference sets, or genuinely two-dimensional constructions, place far more relays.)*
