# Sprawl Doppelganger

## Problem
A city inspector has a *zoning template* `H`: a small connected road network
with exactly one cycle (a tree plus one extra edge). Before approving any new
city plan, the inspector audits it against `H` using only two kinds of
statistics, never the city's actual layout:

1. **Circulation moments**: for `j = 1..k`, the number of length-`j` closed
   walks in the road graph (start at an intersection, follow `j` roads,
   return to start), divided by the number of intersections. This is exactly
   `trace(A^j) / N` for the adjacency matrix `A`.
2. **Intersection-load**: the sum of squared degrees, divided by the number
   of intersections.

Both statistics only ever "see" traffic patterns within a bounded number of
hops of any given intersection — they are strictly *local* observables. Your
job: design a road network `G` on at most `n` intersections whose statistics
match the template's within the stated tolerances, while making the city's
**diameter** (the longest shortest-path between two intersections) as large
as possible. A sprawling metropolis that audits identically to a small town.

## Input (stdin)
```
nH k n
eps_num eps_den
eps2_num eps2_den
D_MAX M_MAX
mH
u_1 v_1
...
u_mH v_mH
tw_1
...
tw_k
S2
```
`nH` = template size, `k` = moment order, `n` = your vertex cap. `eps`,
`eps2` are exact tolerances (fractions). `D_MAX`/`M_MAX` bound the degree of
any vertex and the total edge count you may output. `H`'s `mH = nH` edges
follow (0-indexed). `tw_j` is the exact integer numerator of the template's
`j`-th moment `mu_j(H) = tw_j / nH`. `S2` is `sum_v deg_H(v)^2`, so
`mu_S2(H) = S2 / nH`.

## Output (stdout)
```
N
M
u_1 v_1
...
u_M v_M
```
`N` intersections (`1 <= N <= n`), `M` roads as 0-indexed pairs.

## Feasibility
`G` must be a simple, connected graph with `N <= n`, `M <= M_MAX`, no
self-loops or repeated edges, every degree `<= D_MAX`, and for every
`j = 1..k`: `|mu_j(G) - mu_j(H)| <= eps_num/eps_den`, plus
`|mu_S2(G) - mu_S2(H)| <= eps2_num/eps2_den`. Any violation scores `0`.

## Objective
Maximize `diam(G)`, the graph diameter (max over all pairs of the shortest
number of roads between them). Undefined (infeasible) if `G` is disconnected.

## Scoring
Let `B = diam(H)` (matching `H` itself is always feasible, since it trivially
satisfies its own statistics). With `F = diam(G)`:
`score = min(1.0, F / (10*B))`.

## Constraints
`7 <= nH <= 12`, `k = 6`, `n` up to a few thousand, `1 <= B`. Time limit 5s,
each `.in` file well under 1 MB.

## Example (illustrative FORM only — smaller k, not a real test case)
Suppose `nH=4`, `k=2`, `H` edges `(0,1),(1,2),(0,2),(0,3)` (a triangle with
one pendant). Then `mu_2(H) = 8/4 = 2` (`tw_2=8`, since closed 2-walks at a
vertex equal its degree, summing to `2*mH=8`), and `mu_S2(H) = 18/4 = 4.5`.
`diam(H) = 2` (vertex 3 to vertex 1 or 2), so `B = 2`.

A submission that outputs two "copies" of `H`'s spanning tree
`{(0,1),(1,2),(0,3)}` (breaking the triangle at edge `(0,2)`), re-glued into
one necklace via `(2,4)` and `(6,0)` — vertices `4..7` are copy 1 of
`0..3` — gives `N=8` vertices. Every vertex keeps exactly its `H`-degree, so
`mu_2` and `mu_S2` match `H` exactly (both differences are `0`, well inside
any positive tolerance). Its diameter is `F = 5` (from vertex 3 to vertex 7).
Score `= min(1, 5/(10*2)) = 0.25`.

Doubling the number of copies again roughly doubles `F` while the audit
statistics stay exactly matched — the freedom the audit cannot see is spent
entirely on how far apart the copies end up, not on their local shape.
