# Rosette: Lead-Line Partition and Glass Colour Design

## Problem
You are designing the lead-line partition and glass colours of a circular rose window,
modelled as a polar grid disk with `R` rings (`r = 0..R-1`, ring 0 innermost) and `A`
angular sectors (`a = 0..A-1`, cyclic). This gives `R*A` unit cells. Two cells are
**adjacent** if they share a ring-and-sector edge: `(r,a)`-`(r,a+1 mod A)` (angular,
wraps around) or `(r,a)`-`(r+1,a)` (radial, does not wrap).

You must partition all `R*A` cells into connected **faces** (lead-outlined glass panels)
and assign each face one **colour** from a palette of `p` colours, subject to:

1. **Connectivity**: every face is a single connected region under the adjacency above.
2. **k-fold rotational symmetry**: the window has rotation order `k` (`k` divides `A`).
   Rotating every cell's sector index by `step = A/k` must map the partition exactly onto
   itself: if cell `(r,a)` belongs to face `f`, then `(r, (a+step) mod A)` must belong to
   a face `rot(f)` that is the SAME for every occurrence of face `f` anywhere on the disk.
3. **Proper colouring**: two faces that are adjacent (share at least one cell edge) and are
   *different* faces must receive different colours.

Any violation scores `Ratio: 0.0`.

## Input (stdin)
```
R A k p
we wh
H[0][0] ... H[0][p-1]
...
H[p-1][0] ... H[p-1][p-1]
```
`R,A,k,p` integers (`k` divides `A`). `we, wh` are positive weight floats. `H` is a
symmetric `p x p` "harmony wheel": `H[c1][c2]` (0-indexed colours) is the aesthetic
harmony bonus earned by placing colours `c1+1` and `c2+1` on two adjacent, differently
coloured faces. Diagonal entries are unused (adjacent faces never share a colour).

## Output (stdout)
Exactly `R*A` lines in row-major order (`r=0..R-1`, inner loop `a=0..A-1`), each:
```
face_id color_id
```
`face_id` any non-negative integer identifying the face (need not be contiguous);
`color_id` in `[1,p]`.

## Objective (maximize)
Let `area(f)` be the cell count of face `f`, `total = R*A`. **Face-area entropy** (bits):
`Hent = -sum_f (area(f)/total) * log2(area(f)/total)`.
**Harmony sum**: over every adjacent cell-edge pair (angular incl. wrap, radial) whose two
cells belong to different faces, add `H[color1-1][color2-1]`.
```
F = we * Hent + wh * HarmonySum
```
Bigger, more varied faces raise entropy; more/longer boundaries between well-paired
colours raise the harmony sum -- the two pull against each other, and BOTH depend on how
the geometry and the colouring are chosen jointly, not on colouring a fixed shape after
the fact.

## Scoring
The checker's internal baseline `B` is `R` concentric full-ring faces (each ring one
face), coloured `1,2,1,2,...` by ring parity (always feasible: ring-adjacency is a simple
path). With your feasible `F`:
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the ring baseline scores about `0.1`.

## Constraints
`4 <= R <= 18`, `2 <= k <= 11`, `2 <= p <= 6`, `A = 2*k*w` for some integer `w >= 2` (so
`A` is always divisible by both `k` and `2k`). Runs in well under the time limit.

## Feasibility caution
A rotationally-symmetric design that carves the disk into exactly `k` radial pie-wedges
(one face per wedge) forces those `k` faces into a cycle of adjacency around the centre.
Properly colouring a cycle of odd length needs **3** colours; with only 2 colours such a
design is infeasible whenever `k` is **odd** -- regardless of how large `p` is, since the
wedge design itself only ever touches 2 of the available colours. Whether a rotationally
symmetric wedge-style partition can even be properly coloured depends jointly on `k`, on
how many colours the design actually uses, and on the parity of the resulting adjacency
cycle -- so the face geometry should be chosen with the colouring constraint in mind, not
decided first and coloured afterward.

## Example
`R=4, A=8, k=4, p=2`, uniform `H[0][1]=H[1][0]=0.5`, `we=1, wh=1`. The baseline (4 rings,
colours 1,2,1,2) has `Hent=log2(4)=2`, `HarmonySum = 8*3*0.5=12`, `B=14`, giving
`Ratio=0.1` for that same design.
