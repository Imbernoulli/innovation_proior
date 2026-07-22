# One Torch, Many Parts: Least Dead Travel

## Problem

A sheet-metal nesting layout is given as a **cut graph**: a set of straight
segments (edges) between fixed 2D points (vertices) that must all be cut by
a single plasma torch. Adjacent parts in the nest share a boundary segment,
so many edges border **two** parts at once — but each such edge is only
ever cut **once** (cutting it severs both parts simultaneously).

The torch executes a **program**: a sequence of one or more *trails*. Each
trail is `pierce` (torch drops at a vertex, cost `P`), then a continuous
walk that cuts a sequence of required edges one after another (consecutive
edges in the walk must share an endpoint — the torch only moves along
material while cutting, it never jumps through a part while lit), then
`retract` (torch lifts, free). Between the retract point of one trail and
the pierce point of the next trail the torch travels in a straight line
with the torch **up** — this "dead travel" (airtime) costs its Euclidean
length. There is no airtime charge before the first pierce or after the
last retract.

**Feasibility:** every required edge must be cut by **exactly one** trail,
in either direction; a trail must contain at least one edge.

## Input

```
n m P
x_1 y_1        (n lines: integer coordinates of vertex 1..n)
...
u_1 v_1        (m lines: required edge, endpoints are vertex ids)
...
K
e1 e2 e3 e4    (K lines: for information only, the 4 edge-indices, 1-indexed
                into the edge list above, that border one part's rectangle;
                a shared edge appears in two different parts' lines)
```
`1 <= n,m,K <= 200`, `1 <= P <= 200`, coordinates and edge indices fit in
32-bit signed integers.

## Output

```
T
L_1 v_0 v_1 ... v_{L_1}
...
L_T v_0 v_1 ... v_{L_T}
```
`T` trails; trail `i` cuts `L_i` edges via the vertex walk `v_0..v_{L_i}`
(edge `j` of that trail is `(v_{j-1}, v_j)`, and it must be present in the
input edge list, in either orientation).

## Feasibility

Reject (score 0) if: `T` or any `L_i` is out of a sane range, any vertex id
is out of range, any consecutive pair `(v_{j-1},v_j)` in a trail is not a
required edge, any required edge is cut more than once, or any required
edge is never cut.

## Objective (minimize)

`F = P * T + sum of airtime between consecutive trails' retract/pierce points`

## Scoring

The checker builds its own reference "do nothing clever" baseline `B`:
cut every one of the `m` edges as its own separate trail, visiting edges in
input order (`B` is exactly this construction's `F`). Your score is
`min(1.0, 0.1 * B / F)`.

## Constraints

Time limit: 5s. Memory limit: 256MB.

## Example

`P=60`. Two side-10 squares sharing one vertical edge (2 parts, 6 vertices,
7 edges), with vertices `1..6` at `(0,0) (10,0) (20,0) (0,10) (10,10) (20,10)`;
edge 6 = `(2,5)` is the shared boundary. One valid program with **one**
pierce covers all 7 edges: pierce at vertex 2, walk
`2 1 4 5 6 3 2 5` (7 edges, each used once, ends at vertex 5). This scores
`F=60*1=60`, `B~=493`, ratio `~=0.822`. Cutting each of the 2 parts as its
own contour (a valid but naive program, 2 pierces) scores `F=2*60+10=130`,
ratio `~=0.379`. Cutting all 7 edges individually (the baseline itself)
scores `F=B`, ratio exactly `0.1`.

## Why fewer pierces than parts is possible

At every vertex where an odd number of required edges meet, some trail must
either start or end there (a trail crosses a vertex twice for every pair of
its edges there). A connected component of the cut graph can always be
decomposed into exactly `max(1, (#odd-degree vertices)/2)` trails — never
fewer — regardless of how you group edges by part. Piercing once per part
ignores this: at a T-junction or a fully interior vertex (even degree,
degree 4 for an interior grid point) no new pierce is ever required, yet
"one pierce per part" pays for one anyway at nearly every part boundary.
