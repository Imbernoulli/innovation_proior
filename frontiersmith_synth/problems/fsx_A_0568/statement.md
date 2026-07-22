# Calder Atelier: Torque-Balanced Mobile with a Wide, Deep Silhouette

## Problem
You are hanging a Calder-style mobile. It is a **full binary tree**: every internal
node is a horizontal **rod** with a left arm `aL` and a right arm `aR`, hanging a left
sub-mobile and a right sub-mobile; every leaf is a weight. You are given a fixed
multiset of `N` integer leaf weights and an arm ceiling `Amax`. You must build a mobile
that uses **every weight exactly once** and hangs in perfect balance, while spreading
its silhouette as **wide and as deep** as possible.

A rod is in static balance when the torques about its pivot cancel:
`wL * aL == wR * aR`, where `wL`, `wR` are the **total** weights hanging on its left and
right sub-mobiles. Note this fixes only the *ratio* `aL : aR`, not the arm lengths.

## Input (stdin)
```
N Amax
w_1 w_2 ... w_N
```
`N` is a power of two. `Amax` is the maximum length of any single arm.

## Output (stdout)
```
M
<node 0>
<node 1>
...
<node M-1>
```
`M = 2N-1`. Each node line is one of:
- `L w` — a leaf carrying weight `w`.
- `I l r aL aR` — a rod with left child id `l`, right child id `r`, and integer arms
  `aL, aR`.
Child ids refer to earlier or later node lines by index. Exactly one node must be
referenced by no rod (the **root**); every other node must be the child of exactly one
rod; the result must be a single tree.

## Geometry
The root sits at `x = 0`, depth `0`. A rod at `(x, d)` places its left child at
`(x - aL, d+1)` and its right child at `(x + aR, d+1)`. Leaf coordinates follow; all
`x` are integers.

## Feasibility (any violation scores `Ratio: 0.0`)
- `M = 2N-1`; the node list forms a single full binary tree with one root.
- The multiset of leaf weights equals the input multiset (each weight used once).
- Every arm is an integer with `1 <= aL, aR <= Amax`.
- Every rod is balanced exactly: `wL * aL == wR * aR` (integer equality).
- No two leaves share the same `(x, depth)` (no collision).

## Objective (maximize)
Let the `N` leaves have integer horizontal positions `x_i` and depths `d_i`. Bin the
positions with bin width `binw = max(1, Amax // (2N))`, i.e. `b_i = x_i // binw`.
With `H(·)` the Shannon entropy (natural log) of a count distribution over `N` items,
```
Hx = H(bins)   / ln(N)     # horizontal-spread entropy in [0,1]
Hd = H(depths) / ln(N)     # depth-balance entropy   in [0,1]
F  = 0.6 * Hx + 0.4 * Hd
```
`Hx` rewards a wide silhouette that fills **many distinct horizontal bins evenly**;
`Hd` rewards leaves spread **across many depths**.

## Scoring
The checker builds an internal baseline `B`: a balanced mobile with minimal arms
(collision-repaired). With `F` your feasible value,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so reproducing the baseline scores about `0.1` and there is ample headroom above.

## Constraints
`4 <= N <= 32`. The checker runs in `O(N + M)`. Time limit 5s, memory 512m.

## Example
Take `N = 4`, weights `{6, 6, 12, 12}`, `Amax = 36`. One balanced mobile pairs the two
`6`s under one rod and the two `12`s under another, then joins them. Because equal-weight
siblings need only the ratio `1 : 1`, each such rod may use arms `(t, t)` for **any**
`t <= Amax` — the scale is free. Choosing large, differing `t` at the rods flings the
four leaves into four separate horizontal bins at balanced depths, pushing `F` well
above the minimal-arm baseline. (Illustrative only — larger instances have many rival
groupings and no closed-form optimum.)
