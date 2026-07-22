# Chain-Trap Dendrogram: Merging Toward a Kept Cut

## Problem

You are given `N` points in the plane and an `N x N` integer bonus matrix
`bonus[i][j]` (a "should these two end up together?" score, positive or
negative, symmetric, zero on the diagonal). You must build a **dendrogram**:
an ordered sequence of pairwise cluster merges that starts from the `N`
singleton clusters `{1}, ..., {N}` and ends with exactly `K` clusters.

Every merge is **irreversible** — once two clusters are combined, the
resulting cluster can only be merged again as a whole; its members can never
be separated. The merge you commit to now is judged only by the `K`-cluster
partition it eventually leaves you in, not by whether it was the cheapest
pair available at that instant.

Each merge of clusters `A` and `B` (sizes `|A|`, `|B|`, centroids computed as
the mean of member coordinates) has a **linkage cost**:

```
cost(A, B) = |A|*|B| / (|A|+|B|)  *  euclidean_dist(centroid(A), centroid(B))
```

If this merge is the `t`-th of the `T = N - K` merges you will perform in
total, let `remaining = T - t` (merges still to come after this one). The
cost is scaled by a **lookahead-horizon weight** that grows the closer you
get to the final cut:

```
weight(remaining) = 1 + ALPHA / (1 + remaining)
weighted_cost      = cost(A, B) * weight(remaining)
```

`ALPHA` is given in the input. Merges done early (large `remaining`) are
cheap; merges done in the last few steps before reaching `K` clusters are
expensive. The **cumulative linkage cost** is the sum of `weighted_cost`
over all `T` merges.

At the very end, cut the dendrogram at its final `K` clusters. Your
**partition score** is the sum of `bonus[i][j]` over all pairs `{i, j}` that
end up in the *same* final cluster (pairs in different final clusters do not
contribute). Your objective is:

```
partition_score  -  LAMBDA * cumulative_linkage_cost
```

`LAMBDA` is given in the input (already calibrated per-instance so the cost
term and the partition term are comparable in magnitude — it is NOT a fixed
constant across test cases, so read it, don't guess or hard-code it). Maximize this
quantity. Note that a locally expensive merge can be the right call if it
keeps high-bonus points together or avoids a low-bonus pairing that a
cheaper, greedier merge would have created.

## Input (stdin)

```
N K
ALPHA LAMBDA
x_1 y_1
...
x_N y_N
bonus[1][1] ... bonus[1][N]
...
bonus[N][1] ... bonus[N][N]
```
`N` points are numbered `1..N`. `3 <= K < N`. Coordinates and bonus entries
are integers. `bonus` is symmetric with zero diagonal.

## Output (stdout)

```
T
a_1 b_1
...
a_T b_T
```
`T` must equal `N - K`. Each line `a_t b_t` merges two currently *active*
clusters. Initially clusters `1..N` (the points) are active. The `t`-th
merge you output creates a new cluster labeled `N + t`, which becomes active
while `a_t` and `b_t` become inactive. After all `T` lines, exactly `K`
clusters must remain active — that partition is what gets scored.

## Feasibility

Every `a_t`, `b_t` must be a currently active cluster label, `a_t != b_t`.
Referencing an unknown or already-merged label, a wrong merge count, or any
non-integer/non-finite token makes the output infeasible (`Ratio: 0.0`).

## Scoring

The checker replays your merge sequence, computes
`F = max(0, partition_score + OFFSET - LAMBDA * cumulative_linkage_cost)`
(`OFFSET` = sum of `|bonus[i][j]|` over all pairs, a fixed additive shift so
`F` stays non-negative), and compares it to `B`, the same quantity for a
fixed naive baseline sequence (merge `(1,2), (3,4), ...` in index order,
ignoring geometry and bonuses). Final score: `min(1, F / (10*B))`.

*Illustrative example only* (not one of the real test cases): with `N=4`,
`K=2`, points forming two well-separated close pairs, and a bonus matrix
that agrees with that split, merging each close pair first (cheap, early)
then stopping gives partition_score at its maximum with low cumulative
cost — a high-scoring, feasible dendrogram.

## Constraints

`8 <= N <= 40`, `3 <= K < N`, coordinates in `[-200, 200]`, `|bonus[i][j]|
<= 20`, `ALPHA, LAMBDA >= 0`. Time limit 4s.
