# Deep Firebreaks: Blocking a Threshold Cascade Before It Amplifies

## Problem
A directed influence network of **N** nodes models a spreading outbreak. A fixed
set of **S** *source* nodes starts active (the outbreak's origin). Every directed
edge `u -> v` carries a positive weight `w`. Every node `v` has an **activation
threshold** `theta_v`: once the sum of weight arriving from v's already-active
in-neighbours reaches `theta_v`, `v` itself becomes active (and stays active
forever). The cascade runs to a fixed point.

Crucially, every node `u` also has an **amplification factor** `amp_u >= 1`.
Once `u` becomes active, *all* of its outgoing edges effectively carry weight
`amp_u * w` (not just `w`) when contributing to a neighbour's incoming sum —
some nodes are quiet relays, others are amplifiers that broadcast far more
strongly than a single edge weight suggests.

Before the cascade starts, you get to remove up to **K** non-source nodes as
firebreaks (deleting a node also deletes all its edges). Your goal is to choose
the firebreak set that leaves the **fewest non-source nodes activated** once
the cascade (on the remaining graph) reaches its fixed point.

## Input (stdin)
```
N M K S
s_1 s_2 ... s_S
theta_1 amp_1
theta_2 amp_2
...
theta_N amp_N
u_1 v_1 w_1
...
u_M v_M w_M
```
Node ids are `1..N`. `s_1..s_S` are the source node ids (distinct). Lines 3
through N+2 give every node's threshold and amplification factor, in node-id
order. A source's threshold is irrelevant (it starts active regardless of
any incoming weight), but a source's amplification factor is NOT special —
it multiplies the source's own outgoing edges exactly like any other active
node's. The remaining `M` lines list directed edges
`u -> v` with weight `w > 0`.

## Output (stdout)
```
R
id_1 id_2 ... id_R
```
`R` is how many nodes you remove (`0 <= R <= K`); the second line lists their
ids (omit / leave blank if `R = 0`).

## Feasibility
Rejected (score 0) unless: `0 <= R <= K`, every id lies in `[1,N]`, all ids are
**distinct**, and **none of them is a source node** — you may build firebreaks
around the outbreak, but you cannot remove the outbreak's origin itself.

## Objective
Delete your `R` chosen nodes (and their incident edges) from the graph, run the
cascade from the sources to its fixed point, and let `F` be the number of
**non-source** nodes that end up active. **Minimize `F`.**

## Scoring
Let `B` be the value of `F` when *no* nodes are removed (the judge computes
this baseline itself). Your score is
```
Ratio = min(1.0, B / (10 * F))
```
so doing nothing scores about 0.1, and any effective firebreak set scores
higher, uncapped headroom above any reference solution.

## The catch
The immediate neighbourhood of the sources is a **redundant mesh**: many
alternate paths reconverge there, so spending your budget on high-degree nodes
close to the source barely reduces the cascade — the mesh (and everything
downstream of it) mostly reactivates through the paths you didn't cut. The
nodes that truly gate the cascade are the **amplifier nodes** sitting deeper
in the network: each one is fed redundantly (so no small upstream cut
disconnects it), but its own large `amp` factor lets it single-handedly push a
whole cluster of downstream followers over their thresholds. Removing that one
deep node kills its entire downstream flood; no source-adjacent set or static
min-cut of the same budget comes close. Ranking nodes by raw out-degree or raw
outgoing weight does not find them either — an amplifier's individual outgoing
edges can carry modest stored weight; it is the `amp` multiplier, not the edge
weights themselves, that turns them into a flood, so only the printed `amp`
field together with simulating its effect reveals which nodes matter. When
the number of such amplifier clusters exceeds your budget `K`, you must also
decide *which* clusters are worth stopping.

## Constraints
`N` up to 550, `M` up to a few thousand. All weights/thresholds are given to
6 decimal places. Time limit 5s, memory 512MB. Deterministic: identical input
and output always yield the identical score.

## Example
Two sources `1,2`; a mesh node `3` (theta=0.75) fed by both; a funnel node `4`
(theta=0.8) fed by `3`; an amplifier `5` (theta=0.85, amp=5.0) fed by `4`
(weight 1.0); and a leaf `6` (theta=3.2) fed only by `5` (weight 1.0, so `5`
delivers `5.0*1.0=5.0 >= 3.2` once active). With no removals, all of
`3,4,5,6` activate: `B=4`. Removing node `5` (budget allows it) leaves `6`
below threshold: `F=1`, giving `Ratio = min(1.0, 4/(10*1)) = 0.4`.
