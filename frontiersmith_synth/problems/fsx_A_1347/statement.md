# Wheel Mesh Coloring with Obstruction Certificates

## Problem

You are given a triangulated mesh: `n` vertices and `m` triangular faces.
You must properly color the vertices (adjacent vertices, i.e. vertices that
share a face edge, must get different colors) using colors from a fixed
palette `1..K`. Every color `c` has a fixed, positive *cost[c]* (given in
the input); using color `c` on a vertex spends `cost[c]`.

Somewhere in a real mesh, greedy coloring quietly burns an extra color and
gives you no idea whether that extra color was actually *necessary*. Some
local patches of this mesh contain an odd structure that genuinely forces
one more color than the rest of the mesh needs; other patches only *look*
hard to a naive pass but are perfectly fine with fewer colors once you look
at their shape. Your solver should not just color the mesh -- when a patch
truly cannot be done with fewer colors, it should say so, with a proof a
checker can verify.

Concretely: a **wheel** is a hub vertex joined to every vertex of a cycle
(its rim). A wheel whose rim has an odd number of vertices provably needs
4 colors (no properly-colored assignment can use fewer); a wheel whose rim
has an even number of vertices only needs 3. The mesh may be built from
several such wheel-shaped patches (plus, possibly, other faces) -- which
patches are odd and which are even is NOT stated; you must read the graph
and work it out.

## Input (stdin)

```
n m K
cost_1 cost_2 ... cost_K
a_1 b_1 c_1        (face 1: three 1-indexed vertex ids)
...
a_m b_m c_m        (face m)
```
`1 <= n <= 100`, `1 <= m <= 100`, all `cost_i` positive integers, `K` is
large enough that a naive coloring always succeeds.

## Output (stdout)

```
color_1 color_2 ... color_n     (color of vertex i, each in [1,K])
C                                (number of certificates you attach, C>=0)
L_1 hub_1 r_1 r_2 ... r_{L_1}    (repeated C times)
```
Each certificate line claims: "the vertices `hub` and `r_1..r_L` form a
wheel (hub adjacent to every `r_i`; the `r_i` form a cycle `r_1-r_2-...-
r_L-r_1`) whose rim length `L` is odd, so this mesh needs at least 4
colors." The checker re-derives the mesh's edges from the input and
verifies every claimed edge is really there and that `L` is really odd --
a certificate is only credited if it is literally true.

## Feasibility

The coloring is checked strictly: every value must lie in `[1,K]`, and no
two vertices sharing a face edge may share a color. Any violation (or
malformed/short output) scores `Ratio: 0.0`. A malformed or false
certificate never invalidates an otherwise-valid coloring -- it simply
earns no credit (see Scoring).

## Objective / Scoring

The checker computes a quality score `F` for your submission from three
honestly-structured, additive pieces (the exact fixed positive weights are
not disclosed -- reason about the *shape* of the incentive, not the exact
numbers):

- a large reward that grows as you use **fewer distinct colors** overall
  (the true minimum for a wheel patch is 3 if its rim is even, 4 if odd --
  reuse the same handful of colors across every patch you can);
- a reward for **spending less of the cost budget** (`sum of cost[color[v]]`
  over all vertices) -- among colorings that are equally good on color
  count, prefer cheap colors;
- a bonus, capped at a fixed total, for the **fraction of the mesh's true
  odd-wheel obstructions you correctly certify** (each valid hub counted
  once -- finding every real obstruction earns the full cap, finding none
  earns none, spamming duplicates earns nothing extra), and a penalty for
  every certificate attempt that turns out false, duplicated, or malformed.

`F` is compared against the checker's own internal reference coloring `B`
(a cost-blind, certificate-free construction) via
`Ratio = min(1, F / (10*B))`, printed as `Ratio: <float>`.

## Constraints

`n, m <= 100`, time limit 5s, memory 256MB.

## Example (worked, illustrative shape only)

A single 4-rim wheel: hub `1`, rim `2 3 4 5` (faces `1 2 3`, `1 3 4`,
`1 4 5`, `1 5 2`), `K=4`, `cost = [3 1 4 2]`. Since the rim length 4 is
even, 3 colors suffice: e.g. `color = [1 2 3 2 3]` (hub=1, rim alternates
2,3) is proper and uses only the 3 cheapest-available colors among those
picked; you'd attach `C=0` certificates here since no obstruction exists.
If instead the rim had length 5 (odd), no proper 3-coloring exists at all,
and a solver that both finds a valid 4-coloring AND attaches a true
certificate `5 1 2 3 4 5 6` (hub=1, rim=2..6) is rewarded for the honest
proof, unlike one that silently burns a 4th color with no explanation.
