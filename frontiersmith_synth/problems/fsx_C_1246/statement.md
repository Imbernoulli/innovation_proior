# Don't Just Connect the Closest: An Entry Set and a Degree-Budgeted Graph for Greedy Search

## Problem
You are given `N` points and `Q` held-out queries. Instead of answering the
queries yourself, you build a navigation **index**: a directed graph on the
`N` points where every node stores at most `M` out-neighbours (the degree
budget), plus a set of `R` **entry points**. The judge then answers every
query with a fixed, un-tunable greedy search over your index and charges you
for the distance computations it performs. You never see the queries; you
must build an index that serves *any* query well.

## Input (stdin)
```
N M R
x_1 y_1
...
x_N y_N
Q
qx_1 qy_1
...
qx_Q qy_Q
```
All coordinates are integers. `10 <= N <= 200`, `2 <= M <= 8`, `2 <= R <= 6`,
`R < N`, `1 <= Q <= 50`.

## Output (stdout) — the artifact
```
e_1 e_2 ... e_R
deg_0 n_{0,1} ... n_{0,deg_0}
deg_1 n_{1,1} ... n_{1,deg_1}
...
deg_{N-1} ...
```
`e_1..e_R` are `R` distinct node indices in `[0,N)` — your entry set.
Then one adjacency line per node `0..N-1` (in order): `deg_i` (`0<=deg_i<=M`)
followed by that many distinct out-neighbour indices, none equal to `i`.

## Feasibility
Rejected (`Ratio: 0.0`) if: any token is missing, non-integer, or non-finite;
an entry or neighbour index is out of `[0,N)`; entries are not `R` distinct
values; any node's degree exceeds `M`; a node lists a duplicate neighbour or
a self-loop; or there are leftover tokens after the last adjacency line.

## The judge's search (fixed — not yours to tune)
For each held-out query `q`, the judge evaluates the squared Euclidean
distance from `q` to every entry point (that's `R` distance computations)
and keeps the best. Then, repeatedly: from the current best node, it
evaluates the distance from `q` to **every** out-neighbour of that node (one
computation each), and if any neighbour beats the current best it moves to
the single best neighbour found; if none beats it, the search stops. (This
process is strictly monotonic in distance, so it always terminates.)

## Objective (minimize)
Let `d*(q)` be the true nearest-neighbour squared distance from `q` to the
`N` points (computed independently by the judge via brute force). If the
search's final best distance equals `d*(q)`, that query's cost is the number
of distance computations actually performed. **If it does not** (your graph
failed to reach the true nearest neighbour), the query's cost is charged as
`N` — as if you had fallen back to scanning every point. Let `F` be the
average cost over all `Q` queries. The judge's baseline `B = N` is the cost
of literally scanning everything for every query (always correct). Score:
```
Ratio = min(1.0, 0.1 * B / F)
```
Reproducing brute force (`F ~= N`) scores `~0.1`; a tenth of that cost caps
the score at `1.0`.

**Why this is not just "connect to your nearest neighbours"**: a node's `M`
nearest neighbours are the *locally* best edges, but spending the whole
budget on them keeps every edge short, so a search starting far from a
query's region can never cross the gap — it converges to *some* locally-best
point and stops, paying the full `N` penalty. A few edges that are
individually "worse" (longer) but reach otherwise-unconnected regions can
turn many `N`-cost misses into cheap hits, at small added cost on the
queries that would have hit anyway. The same applies to your `R` entries:
points crammed into one region cannot serve queries elsewhere.

## Constraints
`10 <= N <= 200`, `2 <= M <= 8`, `2 <= R <= 6` (`R < N`), `1 <= Q <= 50`,
coordinates are integers, `-1000 <= x, y <= 10000`.

## Example (worked score, illustrative shape only)
Suppose `N=6`, one query hits (cost `3`: 2 entries + 1 neighbour check) and a
second, structurally distant query misses (cost `N=6`). `F = (3+6)/2 = 4.5`,
`B = 6`, `Ratio = min(1, 0.1*6/4.5) = 0.133`. If instead your index also
served the second query in `4` computations, `F = (3+4)/2 = 3.5` and
`Ratio = min(1, 0.1*6/3.5) = 0.171` — noticeably better, even though that
extra bridge edge is never the *locally* shortest choice for the node that
holds it.
