# Arterial Polarity Grid: Maximally Decorrelated Signal Schedules

## Problem

A city traffic center controls an `N x N` grid of coordinated signal schedules.
Row `i` is intersection `i`; column `j` is one of `N` timing basis modes. Each
cell `M[i][j]` is a **polarity** in `{-1, +1}`: `+1` if intersection `i` runs mode
`j` in its "green-primary" orientation, `-1` if it runs the inverted orientation.

A signal plan is **robust** when the `N` intersection schedules are as mutually
*decorrelated* (linearly independent) as possible — a plan whose schedule rows are
near-orthogonal degrades gracefully under sensor dropout. The exact algebraic
measure of that independence is the magnitude of the plan's determinant, `|det M|`.

Your job: emit an `N x N` polarity grid that makes `|det M|` as large as possible.

Because `N` is **odd**, no perfectly orthogonal (Hadamard) plan exists — the
theoretical ceiling `N^(N/2)` can never be reached, so there is no closed-form
optimum. Many strategies (random search, sign-flip hill climbing, structured
circulant/orthogonalization heuristics) trade off against each other.

## Input (stdin)

A single line with one odd integer `N` (the grid side length), `13 <= N <= 29`.

## Output (stdout)

`N` lines, each with `N` space-separated integers, every entry in `{-1, +1}`.
This is your polarity grid `M`, row by row.

## Feasibility

- Exactly `N*N` integer tokens, each equal to `-1` or `+1`.
- **Reference column normalization:** column `0` (the arterial reference mode)
  must be all `+1`, i.e. `M[i][0] == 1` for every row `i`. Negating a whole row
  leaves `|det M|` unchanged, so this normalization loses no generality; a grid
  that violates it scores `0`.

Any violation (wrong token count, an entry outside `{-1,+1}`, a non-integer /
`nan` / `inf` token, or a reference-column entry `!= +1`) scores `Ratio: 0.0`.

## Objective (maximize)

`F = bitlength(|det M|)`, where `det M` is computed **exactly** over the integers
via fraction-free Bareiss elimination (no floating point, no tolerance). Larger
`|det M|` means a more decorrelated, more robust signal plan.

## Scoring

The checker builds an internal **corridor baseline** `A` (an "arrow" plan whose
reference column is all `+1`, whose first row is all `+1`, and with `A[i][i]=-1`
for `i>=1`). It has `|det A| = 2^(N-1)`, the smallest possible non-zero
determinant, so `B = bitlength(|det A|) = N`.

```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```

Reproducing the baseline scores `Ratio = 0.1`. Reaching `10x` the baseline
bit-length would cap at `1.0` — but the odd-`N` Hadamard ceiling keeps the best
achievable score well below saturation, so there is always headroom.

## Constraints

- `N` odd, `13 <= N <= 29`.
- Determinant is exact integer arithmetic; scoring is fully deterministic and
  bit-for-bit reproducible.

## Example (worked score)

Suppose `N = 13`. The corridor baseline has `|det| = 2^12 = 4096`, bit-length
`B = 13`, giving `Ratio = 100*13/13/1000 = 0.100`. A grid found by hill climbing
with `|det| = 6_291_456` has bit-length `F = 23`, so
`Ratio = 100*23/13/1000 = 0.1769`. A singular grid has `|det| = 0`, bit-length
`0`, `Ratio = 0.0`.
