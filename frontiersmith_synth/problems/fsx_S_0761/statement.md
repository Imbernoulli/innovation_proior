# One Pivot Order for K Sparsity Patterns

## Problem

A symmetric sparse matrix's nonzero pattern can be viewed as a graph on its
`N` row/column indices. Eliminating (pivoting) a vertex `v` in an elimination
order zeroes out its row/column but creates **fill-in**: every pair of `v`'s
currently-remaining neighbors that isn't already connected becomes connected
(a "fill edge"), and if `v` has `d` remaining neighbors at the moment it is
eliminated, the elimination step costs exactly `d*(d+1)/2` scalar
multiplications. The total operation count of a factorization is the sum of
this cost over every eliminated vertex, and it depends entirely on the
elimination *order* chosen.

You are given `K` different sparse symmetric patterns that all live on the
**same** `N` vertices — think of `K` scenarios / right-hand-side structures
that must all be factorized using **one shared pivot order** (you do not get
to choose a different order per pattern; a single ordering must be committed
to and reused for all `K`). Your job: output ONE permutation of `1..N` (the
elimination order) that minimizes the **sum**, over all `K` patterns, of the
exact fill-in operation count defined above.

The `K` patterns are not unrelated: they secretly share a common structural
backbone (edges/relationships that appear in every one of them), plus each
pattern additionally carries its own private extra edges that are irrelevant
noise from the point of view of the shared structure. Nothing in the input
labels which edges are "shared" versus "private" — you only ever see, for
each pattern, its raw edge list.

*Illustrative example (structure only — not from an actual instance):* if
vertices `{5,6,7}` form a triangle that is present in every one of the `K`
edge lists, while vertex `9` is connected to some random other vertex in
only one or two of the `K` patterns, the triangle is far more likely to be
part of the real shared backbone than vertex `9`'s stray edge.

## Input (stdin)

```
N K
M_1
u_1 v_1 u_2 v_2 ... u_{M_1} v_{M_1}
M_2
u_1 v_1 ... u_{M_2} v_{M_2}
...
M_K
u_1 v_1 ... u_{M_K} v_{M_K}
```
`N` is the number of vertices (indices `1..N`); `K` is the number of
patterns. Each of the `K` blocks lists `M_k` undirected edges (1-indexed,
`u != v`, no duplicates within a pattern) for pattern `k`. Different
patterns generally have different edge sets, but overlap substantially.

## Output (stdout)

A single line with `N` space-separated integers: a permutation of `1..N`,
the vertex to eliminate first, second, ..., last.

## Feasibility

The output must be a valid permutation of `1..N` (every value in `[1,N]`
appears exactly once). Any parse failure, wrong count, out-of-range value,
duplicate, or non-finite token scores `Ratio: 0.0`.

## Scoring

For a feasible submission, the checker replays the exact symbolic
elimination game described above independently on each of the `K` patterns
under your submitted order, and sums the resulting op counts into `F`
(minimize). It also computes its own baseline total `B` by running the same
elimination game (on all `K` patterns) under a simple non-adaptive order:
vertices sorted ascending by their degree in the *union* of all `K`
patterns' edges. The final score is

```
Ratio = min(1, 0.1 * B / F)
```

printed as `Ratio: <value>`. Lower `F` (fewer total multiplications across
all `K` patterns) is better; matching the baseline gives a low score,
several times fewer operations pushes the score toward 1.

## Constraints

`36 <= N <= 300`, `4 <= K <= 8`. Each pattern's edge count is at most a few
thousand. Time limit 5s, memory 512MB.

## Example (worked score, illustrative numbers)

Suppose `N=5`, one pattern with edges `(1,2),(1,3),(2,3),(3,4),(4,5)`. Order
`[1,2,4,5,3]`: eliminating `1` (neighbors `{2,3}`, d=2) costs 3 ops and adds
fill edge `(2,3)` (already present); eliminating `2` (neighbors `{3}`, d=1)
costs 1; eliminating `4` (neighbors `{3,5}`, d=2) costs 3, adds fill edge
`(3,5)`; eliminating `5` (neighbors `{3}`, d=1) costs 1; eliminating `3`
(neighbors `{}`, d=0) costs 0. Total `F=8`. A worse order can cost more by
creating extra fill edges that raise later degrees.
