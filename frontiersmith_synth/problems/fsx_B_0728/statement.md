# Pleat-Stack Routing: Minimizing Peak Layer Thickness Under a Maekawa Floor

## Problem
A box-pleated paper strip has `N` flaps `0..N-1` in a row, connected by `N-1` **step
creases** (crease `i` joins flap `i` and flap `i+1`). Every step crease is unassigned
(`?`); you must label each one Mountain (`M`) or Valley (`V`).

**Routing walk.** Folding is simulated by a walker that starts at flap `0`, footprint
column `0`, heading `+1`. Reading creases left to right: a **Valley** crease keeps the
current heading (`col[i+1] = col[i] + heading`); a **Mountain** crease reflects the
heading (`col[i+1] = col[i] - heading`, then `heading = -heading`). This deterministic
rule places every flap at an integer footprint column.

**Maekawa-style feasibility floor.** The input also lists vertex groups, each exactly
four step-crease indices with a target `t in {+2,-2}` (a degree-4 flat-vertex analogue
of Maekawa's theorem). A labeling is feasible only if, for **every** group, `(#Mountain
- #Valley)` among its four creases equals `t` exactly. Groups may share a crease with a
neighboring group; violating any group's target is an immediate `Ratio: 0.0`.

**Hinge links (long-range creases).** The input additionally lists `K` hinge pairs
`(p, q, label)` with `p < q`; some hinges are pre-pinned to `M`/`V`, others are `?` for
you to decide. A hinge is **active** if the routing walk lands both endpoints on the
same column (`col[p] == col[q]`) -- this can only be known after you resolve the step
creases. An active hinge with label `V` requires `height[p] < height[q]`; label `M`
requires `height[p] > height[q]`, where `height` is a stacking order you also output
(`N` distinct integers, larger = higher in the stack). Two active hinges that share a
column may **not cross**: if both `(p,q)` and `(p2,q2)` are active in the same column
and their index intervals interleave (`p < p2 < q < q2` or the mirror), the output is
infeasible -- layers cannot pass through each other.

Feasibility gate is a strict AND of: token validity, pin consistency, the parity floor,
hinge coincidence-order consistency, and no crossing hinges.

## Input (stdin)
```
N K G
w_0 w_1 ... w_{N-1}          # flap weights, 1..5
c_0 c_1 ... c_{N-2}          # step creases, all '?'
p q label                    # K lines, 0-indexed hinge links (label in {M,V,?})
i1 i2 i3 i4 target           # G lines, vertex groups (crease indices, target in {+2,-2})
```

## Output (stdout)
```
c_0 ... c_{N-2}               # resolved step-crease labels (M/V only)
h_0 ... h_{K-1}                # resolved hinge labels (M/V only)
H_0 ... H_{N-1}                 # N distinct integer heights
```

## Objective
Let `col[]` be the routing walk on your resolved step creases. The **peak stack
thickness** is `F = max_c ( sum of w_i over flaps i with col[i] == c )`. **Minimize
F.**

## Scoring
The checker builds its own feasible construction `B`: resolve every vertex group's
residual freedom with a fixed low-index-first Mountain tie-break, defaulting anything
left over to Valley. With minimization normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching the checker's own construction scores `0.1`; a `10x`-thinner stack caps at
`1.0`.

## Constraints
- `8 <= N <= 60`, weights in `[1,5]`.
- Vertex groups form a chain of size-4 groups over consecutive crease indices,
  consecutive groups sharing exactly one crease; a handful of trailing creases may lie
  outside any group (unconstrained by the parity floor).
- Time limit 5s, memory 512m.

## Example
`N=5`, one group `{0,1,2,3}` with `target=+2` (needs 3 Mountains, 1 Valley among those
four -- but `N=5` only has 4 step creases total, indices `0..3`, so this single group
covers everything). Choosing creases `M M M V` walks: `col = [0,-1,0,1,2]`. Choosing
`M M V M` walks: `col = [0,-1,0,-1,0]` -- all five flaps pile onto just two columns
(peak thickness `5x` a single flap's weight) even though both labelings satisfy the
**same** parity target. The first choice routes the paper outward; the second reflects
it back onto itself. Reading the vertex-group target correctly is only the floor --
*where* the residual freedom sends the flaps is the real objective.
