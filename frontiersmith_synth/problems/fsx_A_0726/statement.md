# Cistern Roots: Carving a Branching Aquifer Collector

## Problem

The dwarves of Kazad-Belar must feed the city cistern from the mountain's water table.
The mountain is a grid of `R` rows by `C` columns of porous rock; row `0` is the water
table itself (a fixed-pressure aquifer), and a single deep cell `(r_out, c_out)` is the
cistern (a fixed-pressure drain). Every other cell is plain rock with its own integer
base permeability. The dwarves may **carve** up to `B` cells into free-flowing tunnel
(a much higher permeability `C_TUNNEL`); everything else stays rock.

Water pressure obeys a resistor-network model. Between two adjacent cells `u, v` the
conductance of that edge is:

- **rock–rock**: `min(perm[u], perm[v])` — slow diffusion.
- **tunnel–rock** (an interface edge): the ROCK side's own permeability — carving does
  **not** grant a free tap; a tunnel only ever drinks from adjacent rock at that rock's
  ordinary rate. This is where nearly all of the system's resistance concentrates.
- **tunnel–tunnel**: `C_TUNNEL`, MINUS a **branch-merging loss** at genuine junctions.
  Let `dist(x)` be the shortest hop-count from cell `x` to the cistern through carved
  cells only (cistern = root, `dist = 0`). An edge `(u,v)` with `dist(u) = dist(v)+1`
  feeds `v` from upstream. If `v` has `k >= 2` such upstream carved neighbours (several
  branches converging on `v`), **each** of those `k` edges is throttled to
  `C_TUNNEL // k`. A trunk's own downstream continuation is a *different* edge (scored at
  the next cell down) and stays untouched — only the convergence point pays a turbulence
  tax, not the whole corridor.

Given the carved set, pressure is found by a **fixed-iteration integer Jacobi
relaxation**: row `0` cells are held at pressure `P_IN`, the cistern cell is held at
`P_OUT = 0`; every other cell's pressure is repeatedly replaced by the conductance-
weighted (integer-floor) average of its up-to-4 neighbours, for a fixed number of
iterations `ITERS` (never to exact convergence — this is the whole scoring rule, not a
solver detail). The **objective** is the total flow delivered into the cistern after
`ITERS` iterations: the sum, over the cistern's 4 neighbours `n`, of
`cond(cistern, n) * (P[n] - P[cistern])`.

Intuition: a fat corridor traps most of its budget in cells that only ever talk to other
tunnel cells (no rock contact), and any junction where several such interior paths meet
pays the merge tax. A branching network with lots of rock contact per carved cell can
out-produce a wide highway of the same size — but pure branching with no throughput path
also loses. There is no known optimal carving.

## Input (stdin)
```
R C B r_out c_out
P_IN P_OUT C_TUNNEL ITERS
perm[0][0] perm[0][1] ... perm[0][C-1]
...
perm[R-1][0] ... perm[R-1][C-1]
```
`perm[r][c]` is the base rock permeability of cell `(r,c)` (a positive integer). Row `0`
and cell `(r_out, c_out)` are always fixed-pressure boundary nodes regardless of carving.

## Output (stdout)
```
k
r_1 c_1
...
r_k c_k
```
`k` (0 <= k <= B) distinct cells to carve, each `0 <= r < R`, `0 <= c < C`.

## Feasibility
Feasible iff: `k` parses as an integer with `0 <= k <= B`; exactly `k` coordinate lines
follow; every coordinate is an integer inside the grid; no coordinate repeats. Any
violation, or any non-finite/non-integer token, scores `0`.

## Objective
Maximize the total flow into the cistern after the fixed relaxation, as defined above.

## Scoring
Let `F` be your flow. Let `Bflow` be the flow of the checker's own baseline carving: a
single 1-cell-wide straight shaft directly above the cistern, using `min(B, R-1)` cells
(no branching, no leftover-budget use). Then
```
Ratio = min(1000, 100 * F / max(1e-9, Bflow)) / 1000
```
A shaft-only carving scores about `0.1`; a carving ten times richer scores at the `1.0`
cap. Infeasible or non-finite output scores `0`.

## Constraints
- `8 <= R <= 17`, `10 <= C <= 19`, `1 <= B <= R*C`.
- `1 <= perm[r][c] <= 5000` (background rock is small; a few planted mineral veins run
  higher).
- Time limit 5s, memory 512MB. The checker runs in `O(R*C*ITERS)`, fully deterministic.

## Example (worked, illustrative shape only)

Toy numbers, not from a real test: a single 1-wide shaft yields `Bflow = 100` (the
baseline). A branching carving that adds two short tributary stubs near the water table
might yield `F = 145` (more rock contact, only one junction cell taxed). Then
`Ratio = min(1000, 100*145/100)/1000 = 0.145`.
