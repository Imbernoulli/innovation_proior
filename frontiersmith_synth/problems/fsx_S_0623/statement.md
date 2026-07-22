# Twin Coins: Cospectral Decoy Mints

## Problem
A mint wants to strike two batches of "coins." Each coin batch is a simple
undirected graph: coins are vertices, and an edge marks two coins that resonate
together. An acoustic tester "rings" a batch by measuring the eigenvalues of its
adjacency matrix (its resonant spectrum) and sorting them. Two batches **pass as
twins** if their sorted spectra agree.

You are given `n` (coins per batch, a multiple of 6) and an edge budget `M`.
Construct two simple undirected labeled graphs `G` and `H`, each on the same `n`
vertices numbered `1..n`, each with at most `M` edges, such that `G` and `H` are
**exactly cospectral**: their sorted adjacency eigenvalues must agree within
`1e-6` (checked with a standard symmetric eigensolver).

Subject to that constraint, maximize how differently the two batches are built --
in every way the spectrum provably cannot see. Two cospectral graphs are forced
to share the same edge count, the same triangle count, and in general the same
count of closed walks of every length. But **nothing** forces them to share a
diameter or an individual degree sequence, and that is exactly the gap you must
exploit.

## Input (stdin)
```
n M
```

## Output (stdout)
```
n1 e1
u_1 v_1
...
u_e1 v_e1
n2 e2
u_1 v_1
...
u_e2 v_e2
```
`n1 e1` + `e1` edges describe `G`; `n2 e2` + `e2` edges describe `H`. Edges are
1-indexed vertex pairs.

## Feasibility (checked strictly; any violation scores `Ratio: 0.0`)
- `n1 = n2 = n`.
- `0 <= e1, e2 <= M`.
- every vertex id lies in `[1, n]`; no self-loops; no duplicate edges within
  either graph.
- `G` and `H` are exactly cospectral (max sorted-eigenvalue gap `<= 1e-6`).

## Objective (maximize)
```
F(G,H) = |Delta(G) - Delta(H)|  +  (1/n) * L1(sorted-degrees(G), sorted-degrees(H))
```
where `Delta(X)` is the **sum, over every connected component of X with at least
2 vertices, of that component's diameter** (isolated vertices contribute 0), and
`L1(sorted-degrees(G), sorted-degrees(H))` is the elementwise absolute
difference between the two graphs' degree sequences, each sorted ascending, then
summed.

## Scoring
The checker builds its own modest reference pair `(G0, H0)` -- a single, small,
localized structural change applied to an otherwise uniform construction -- and
computes its divergence `B = F(G0, H0) > 0`. Your score is
```
Ratio = min(1, F(G,H) / (10 * B))
```
so matching the reference pair's divergence scores about `0.1`, and you need
roughly ten times the reference divergence to reach the cap.

## Why a single fix is a trap
A single localized structural repair that restores cospectrality (think: one
classical local switching move) can only ever disturb a small, bounded
neighborhood of the graph -- it moves `Delta` and the degree sequence by at most
a small constant, **no matter how large `n` is**. Reaching a high score requires
**composing many independent such repairs across disjoint parts of the graph**:
the eigenvalues of a graph built from several separate pieces are just the
multiset union of each piece's own eigenvalues, so replacing any one piece by an
*alternative* piece with the identical spectrum leaves the *whole* graph's
spectrum untouched -- while that piece's local diameter and degrees are free to
move however you like. The total divergence you can bank scales with how many
independent replacements you dare to make, not with how cleverly you tune one.

## Example (illustrative shape only -- not the intended construction)
If `G` is two disjoint copies of some connected 6-vertex graph `P`, and `H`
replaces one copy of `P` with a different, but `P`-cospectral, graph `Q`, then
`G` and `H` remain exactly cospectral (their 12 eigenvalues coincide as
multisets) while `Delta(G)` and `Delta(H)` may already differ. Composing this
idea across more, independent pieces is where the real score lives.

## Constraints
`12 <= n <= 48`, `n` a multiple of 6, `M <= 96`. Time limit 4s, memory 512MB.
Each test's `.in` file is well under 1KB.
