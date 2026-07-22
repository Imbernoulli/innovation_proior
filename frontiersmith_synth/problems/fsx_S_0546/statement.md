# Kolam Mandala: One Stroke Against Symmetry

You draw a **kolam**: a smooth curve winding around an `m x m` grid of dots on a torus
(the grid wraps — column `m-1` is left of column `0`, row `m-1` is above row `0`).
Every dot is a *crossing* with four incident edges (to its N, E, S, W torus-neighbours);
there are `2*m*m` edges in all. At each crossing you drop one of two **arc tiles**:

- tile `0` joins the strand on the **N** edge to the **E** edge, and **S** to **W**;
- tile `1` joins **N** to **W**, and **S** to **E**.

The tiles route every edge into closed loops (each edge is used by exactly one loop).
A **single-stroke** kolam is a tile choice `X` whose loops collapse to **exactly one**
loop that runs through all `2*m*m` edges.

You are handed a **preferred pattern** `P` — a tile for every crossing that is perfectly
symmetric under 90-degree rotation (order-4 orbits). Matching `P` everywhere is beautiful
but **impossible**: a perfectly symmetric pattern always breaks into several rotated
copies of one loop, so it is *never* a single stroke. You must therefore disagree with
`P` at some crossings. Each disagreement is a **symmetry defect**, and defects are costed.

## Input (stdin)
```
m k lam
P   : m rows of m values in {0,1}   (the rotation-symmetric preferred tiles)
w   : m rows of m integers          (cost of breaking symmetry at each crossing)
g   : m rows of m integers          (C4-orbit id of each crossing, 0..m*m/4-1)
```
`m` is even, `k=4`. Read all four blocks in this order.

## Output (stdout)
`m` rows of `m` values in `{0,1}`: your tile choice `X` (the first `m*m` tokens are read).

## Feasibility
`X` must form **exactly one** loop covering all `2*m*m` edges. Any other loop count,
or a token outside `{0,1}`, scores `0`.

## Objective (minimize a break-cost)
Let `D = { v : X[v] != P[v] }` be the defect set. For orbit `o` let `d_o` be the number
of defects it contains. Your cost is
```
cost(X) = sum_{v in D} w[v]  +  lam * sum_o  d_o*(d_o-1)/2
```
The first term pays `w[v]` for each broken tile. The second term is an **interaction
penalty**: concentrating breaks inside the *same* rotation orbit is wasteful — each extra
co-orbit defect adds `lam`. Fewer, cheaper, well-spread breaks are better.

## Scoring
The checker builds two internal single-stroke references from `P,w,g`:
`cWorst` = merging the loops by grabbing the first available bridging crossing
(cost-blind), and `cBest` = its own best low-cost, orbit-aware merge. Your score is
```
Ratio = clip( 0.10 + 0.75 * (cWorst - cost(X)) / (cWorst - cBest),  0, 1 )
```
so a cost-blind single stroke scores about `0.10`, matching `cBest` scores `0.85`, and
beating the reference climbs toward `1`. Mean over 10 hidden cases.

## Feasibility trap
Emitting `P` itself (perfect symmetry) is many loops, not one — it scores `0`. The task
is to find the **minimum-cost set of symmetry-breaking crossings** that welds the rotated
loops into a single stroke.

## Constraints
`6 <= m <= 16`, `lam` given in input, `w[v]` in `[20,120]`. Time limit 5 s, memory 512 MB.

## Example (illustrative, small)
With `m=6` the symmetric `P` splits into 12 loops. Welding them needs 11 bridging tiles.
Choosing those 11 by weight alone typically piles two or three into one orbit (paying
`lam` extra each); nudging a break to a different, slightly pricier crossing in a fresh
orbit removes the interaction penalty and lowers `cost` overall. That trade — cheap
tile vs. cheap orbit — is the whole game.
