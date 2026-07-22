# Sand Garden: Mass-Loading a Chladni Plate to Steer a Nodal Line

## Problem

A square membrane occupies an `N x N` grid of unit cells, clamped to zero
displacement just outside the grid (a drum head fixed at its rim). Cells are
indexed row-major, `id = row*N + col`, `0 <= id < N*N`. Adjacent cells (sharing
an edge) are coupled by unit springs; the resulting stiffness matrix `K` is the
standard 5-point clamped-membrane Laplacian: `K[i][i] = 4`, `K[i][j] = -1` for
every grid-adjacent pair `(i,j)`, `0` otherwise.

Every cell also carries a tiny fixed base mass `1 + id * 1e-6` (this generic,
strictly-increasing-by-cell-id offset exists purely to make every vibration
mode's frequency rank well-defined -- it is far too small to matter physically).
You are given a **mass budget**: you may add an integer number of extra mass
units to each cell, at most `cap` units per cell, spending at most `budget`
units in total. Let `load[i] >= 0` be the extra mass you add to cell `i`, and
`M[i] = 1 + i*1e-6 + load[i]` the final mass at cell `i`.

The plate then vibrates according to the generalized eigenproblem `K v = w^2 M
v` (`M` diagonal). Sort the `N*N` solutions by `w^2` ascending (this ordering is
always well-defined thanks to the base-mass offset above). You are given a
target mode rank `k`. Take that mode's eigenvector, rescale it so its largest
absolute component is exactly `1` (peak-amplitude normalization) -- call the
result `v_hat`.

Physically, sand scattered on a vibrating plate migrates away from antinodes
(large `|v_hat|`) and collects on nodal lines (`|v_hat|` near `0`) -- the
classic Chladni figure. You are also given a set of `target` cells: your goal
is to make the plate "garden" sand onto exactly those cells when it vibrates in
mode `k`.

**The trap**: `load` changes `M`, which changes every `w^2`, which can change
*which* eigenvector ends up ranked `k` -- the very mode you are scoring is a
moving target. Loading mass straight onto the cells you want quiet can easily
push a *different* mode down (or up) past rank `k`, so "mode k" becomes an
unrelated shape where your target cells are no longer quiet at all. A good
loading must be co-designed with the mode ordering it produces, not chosen
against the current (or naive) ordering alone.

(Illustrative FORM only, not this problem's actual data: think of it like
choosing which of two nearly-tied singers gets the solo by adjusting each
singer's microphone gain -- turning up the gain on the one you want to *quiet
down* can flip who gets picked to sing at all.)

## Input (stdin)

```
N
k cap budget
t
target_1 target_2 ... target_t
```
`1 <= k <= N*N`, cell ids are 0-indexed row-major, `0 <= target_i < N*N`.

## Output (stdout)

Exactly `N*N` whitespace-separated integers `load_0 ... load_{N*N-1}`
(row-major), each satisfying `0 <= load_i <= cap`, with `sum(load_i) <=
budget`.

## Feasibility

Output must contain exactly `N*N` integer tokens (no `nan`/`inf`/decimals),
each within `[0, cap]`, summing to at most `budget`. Any violation scores `0`.

## Objective

Let `tau = 0.2`. For each target cell `c`, its gardening credit is
`max(0, 1 - |v_hat[c]| / tau)` (full credit at a perfect node, zero credit once
`|v_hat[c]| >= tau`). Your raw score `F` is the sum of gardening credit over
all target cells, computed on the eigenvector actually ranked `k`-th by `w^2`
**after** your loading is applied.

## Scoring

The checker also builds its own reference loading (spread `budget` mass units
as evenly as possible over all cells, remainder to the lowest-id cells,
respecting `cap`) and computes its raw score `B` the same way. Your final
ratio is `min(1000, 100*F/max(1e-9,B)) / 1000`, in `[0,1]`.

## Constraints

`3 <= N <= 7`, `1 <= cap <= 6`, `4 <= t <= 8`, time limit 5s, memory 512MB.

## Example (worked, illustrative shapes only)

For a small grid, submitting `load_i = 0` everywhere is always feasible; it
simply reports whatever mode happens to rank `k` with no mass added at all,
without any attempt to steer it -- a legal but unambitious baseline.
