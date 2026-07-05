# Lunar Habitat Module Dispersion: Maximizing the Minimum Triangle Area

## Problem
A lunar surface base is being laid out on a flat, triangular building plot with corners
`A`, `B`, `C`. Exactly `n` pressurised habitat modules must be sited **inside** the plot.
For structural surveying and radio triangulation the layout must avoid any three modules
being **nearly collinear**: whenever three modules line up, the triangle they span is thin
and the triangulation of that trio is ill-conditioned. The robustness of the *worst* trio
is what matters, so you want to place the modules such that the **smallest triangle area**
formed by *any* three of them is as large as possible.

This is a **Heilbronn triangle** extremal point-configuration problem restricted to a
triangular domain: maximize the minimum triangle area over all triples of points. The
optimal value is unknown in general, there is no easy closed-form optimum, and the problem
admits many distinct strategies (rings, lattices, farthest-point spreads, local search,
annealing).

## Input (stdin)
```
n
Ax Ay
Bx By
Cx Cy
```
- `n` — number of habitat modules to place.
- `(Ax,Ay)`, `(Bx,By)`, `(Cx,Cy)` — the three corners of the triangular plot.

## Output (stdout)
```
<n lines, each: x y>
```
Print the `n` module coordinates, one `x y` pair per line (real numbers).

## Feasibility
An output is valid iff **all** hold (tolerance `1e-6`):
- exactly `n` coordinate pairs are printed and every coordinate is finite;
- every module lies inside the triangular plot `A-B-C` (boundary allowed within tolerance);
- all `n` modules are pairwise distinct.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize
```
F = min over all triples {i,j,k} of  area( module_i, module_j, module_k )
```
where `area` is the ordinary Euclidean triangle area.

## Scoring
The checker builds its own baseline `B`: a small regular `n`-gon of radius
`0.55 * inradius` centred at the plot's incentre. This ring sits strictly inside the
incircle (hence strictly inside the plot), so it is always feasible; `B` is its exact
minimum triangle area. With maximization normalization:
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the baseline ring scores `Ratio = 0.1`; a layout whose minimum triangle area
is `10x` the ring's caps at `Ratio = 1.0`.

## Constraints
- `7 <= n <= 16`.
- The plot is a non-degenerate triangle with base corners `A=(0,0)`, `B=(1,0)` and an
  apex `C` with `0.20 <= Cx <= 0.65`, `0.70 <= Cy <= 1.00`.
- Time limit 5s, memory 512m.

## Example
Suppose `n = 8` and the checker's baseline ring has minimum triangle area
`B = 0.004`, scoring `Ratio = 0.1`. A layout that spreads the eight modules toward the
plot's edges and corners so that the smallest triangle over all `C(8,3)=56` triples has
area `F = 0.020` scores `sc = 100 * 0.020 / 0.004 = 500`, i.e. `Ratio = 0.500`.
