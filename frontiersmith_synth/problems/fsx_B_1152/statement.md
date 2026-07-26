# The Crypt of Kings: Interception Wells

## Problem
Beneath the palace lies the crypt of kings, and beneath the crypt lies an aquifer.
The aquifer is a grid of `R` rows by `C` columns. Some cells are the **recharge
boundary** (the old river feeding the water table — always fully saturated, no
matter what you pump). Some cells are **blocked rock** (impermeable; nothing flows
through them, and you cannot place a well there). The remaining cells are open
ground. A subset of the open cells are **foundation cells** (the crypt floor):
each has a required **target drawdown** it must reach or the tomb floods.

You choose a set of **wells**, each an open, non-recharge, non-blocked cell with
an integer pump rate `1..QMAX`. The **drawdown** at any cell `v` is the sum, over
all your wells `w` with rate `q_w`, of

```
drawdown(v) += q_w * screening(w) * REACH_L / (REACH_L + dist(v, w))
screening(w) = boundary_dist(w) / (boundary_dist(w) + SCREEN_L)
```

`dist(v, w)` is the shortest-path grid distance between `v` and `w`, moving one
cell at a time (4-directionally) through cells that are **not blocked** (recharge
cells and foundation cells are open to pass through; blocked cells are not).
`boundary_dist(w)` is the shortest-path distance from `w` to the *nearest*
recharge cell, by the same rule. This is exactly **superposition**: each well's
contribution adds up independently. It also encodes **well interference** and
**recharge screening**: a well planted right next to the recharge boundary has a
small `boundary_dist`, so `screening(w)` is small and most of its pumping is
wasted refilling itself from the river; two wells stacked on the same wet patch
both pay their own energy cost while their `dist(v,w)`-based contributions
largely overlap on that patch instead of reaching anywhere new.

Running a well costs energy: `FIXED_COST + q_w^2` (drilling/maintenance overhead
plus a quadratic pumping cost). Your total energy `E` is the sum over all wells
you actually use (rate `>= 1`). You have a fixed pump-energy budget `BUDGET`.

## Input (stdin)
```
R C
n_recharge
r c                      (n_recharge lines)
n_wall
r c                      (n_wall lines)
n_found
r c target               (n_found lines, target is a float)
REACH_L SCREEN_L FIXED_COST QMAX BUDGET
```
Rows/cols are 0-indexed. Recharge, wall, and foundation cells are disjoint.

## Output (stdout)
```
W
r c q                    (W lines: your wells)
```
Print the well count `W`, then one `r c q` triple per well (`q` an integer rate,
`1 <= q <= QMAX`).

## Feasibility
An output is valid iff **all** hold:
- every `r,c` is in-grid, not a recharge cell, not a wall cell;
- all `W` well cells are pairwise distinct;
- every `q` is an integer in `[1, QMAX]`;
- total energy `E = sum(FIXED_COST + q^2)` over your wells is `<= BUDGET`;
- every foundation cell's drawdown (computed from your wells, as above) is
  `>= target - 1e-6`.
Any violation scores `Ratio: 0.0`.

## Objective
Minimize total energy `E` while every foundation cell meets its target.

## Scoring
Let `B` be the checker's own baseline: for every foundation cell, the smallest
`FIXED_COST + q^2` from a *single* well dropped exactly on that cell (as if it
were the only well in the aquifer), summed. This always exists and is positive.
With minimization normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, E))
Ratio = sc / 1000.0
```
Matching the baseline scores `Ratio = 0.1`; using `10x` less energy caps at `1.0`.

## Constraints
- `2 <= R,C <= 20`; grid has at most a few hundred cells.
- `1 <= n_found <= 30`, `1 <= QMAX <= 20`, `FIXED_COST` a positive integer.
- Time limit 5s, memory 512m.

## Example
Suppose one foundation cell `v` has `target = 2.0`, `boundary_dist(v) = 10`,
`REACH_L = 8`, `SCREEN_L = 6`, `QMAX = 12`, `FIXED_COST = 20`. A well placed
directly on `v` (`dist = 0`) has `screening = 10/16 = 0.625`, kernel `= 0.625`,
so it needs `q = ceil(2.0/0.625) = 4`, energy `20 + 16 = 36`. That is the
baseline `B` for this single-cell instance. Reproducing it scores `0.1`; a
solver that finds a cheaper, shared placement across several foundation cells
scores higher.
