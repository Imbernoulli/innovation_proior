# Traffic Signal Grid: Congestion Causal-Graph Recovery

## Setting

A city operates a network of `p` signalised intersections. Over `T` time snapshots
each intersection reports a scalar **congestion index** (mean queue length / delay).
The snapshots are produced by an unknown directed acyclic **flow graph**: the
congestion at an intersection is a linear mixture of the congestion at its
**upstream feeder** intersections, plus an independent local-arrival fluctuation:

```
C_j(t) = sum_{i in parents(j)} w_ij * C_i(t) + eps_j(t),   eps_j ~ N(0, sigma^2)
```

The local-arrival fluctuations `eps_j` share a common scale `sigma` across the grid
(an **equal-variance** linear structural causal model). Under this regime the flow
DAG is identifiable from purely observational congestion data.

You are given the snapshot matrix only. The **column labels are randomly permuted**,
so a column's index tells you nothing about its position in the flow. Your job:
recover the directed flow graph — who feeds whom.

## Your task

Write a standalone program (a causal-discovery routine). It reads ONE JSON object
from **stdin** and writes ONE JSON object to **stdout**.

### Input (stdin)

```json
{
  "data":      [[float, ...], ...],   // T x p matrix; row t is congestion snapshot t
  "n_samples": 600,                    // number of snapshots T
  "n_nodes":   8,                      // number of intersections p (columns of data)
  "seed":      10188001                // a per-instance seed you MAY use for your own RNG
}
```

### Output (stdout)

A directed edge list — either bare or wrapped in an `"edges"` field. An edge
`[i, j]` means intersection `i` is **upstream of** (a direct causal parent of) `j`.

```json
{"edges": [[2, 0], [2, 5], [7, 5]]}
```

Every index must be an integer in `[0, n_nodes)` and `i != j`; an out-of-range or
malformed edge causes the whole answer to be rejected (score 0 on that grid).
Duplicate edges are collapsed. Self-loops are not allowed.

## Scoring (deterministic)

For each grid the evaluator computes the **Structural Hamming Distance** (SHD)
between your directed graph and the hidden ground-truth flow DAG. Over every
unordered pair of intersections `{i, j}`, a pair whose directed-edge state differs
costs 1 — a **missing** edge, an **extra** edge, or a **reversed** edge each add 1.
Lower SHD is better.

Each grid is normalised against the evaluator's own trivial construction, the
**empty graph**, whose SHD equals the number `E` of true flow edges:

```
r = clamp( 0.1 + 0.9 * (E - SHD_candidate) / E , 0, 1 )
```

So an empty answer scores ~0.1, a perfect recovery (SHD = 0) scores 1.0, and doing
worse than empty drops below 0.1 toward 0. The final score is the **mean** of `r`
over a battery of grids that varies in size, snapshot count and flow density
(including harder held-out grids with few snapshots and dense flow). The battery
mean rewards a discovery rule that **generalises** rather than one tuned to a single
grid.

- A candidate that crashes, times out, emits non-JSON, or names an intersection
  outside `[0, n_nodes)` scores 0 on that grid.
- Everything is seeded and deterministic; the same program always earns the same
  score.

## Isolation

Your program runs in an **isolated subprocess**. It only ever sees the public JSON
above — the ground-truth DAG, the SCM weights, the noise scale, and the column
permutation live only in the evaluator process and are never sent to you.

## Ideas

- The empty graph is the ~0.1 floor.
- Thresholding **marginal correlations** gives a skeleton, but marginal association
  also fires between intersections linked only *indirectly* through a shared feeder.
- In the equal-variance regime, an upstream source has **smaller variance** than the
  downstream intersections it feeds — a cue for orientation and for building a
  topological order (e.g. iteratively pick the smallest residual variance).
- Conditioning (regression on candidate parents, partial correlation, PC-style
  independence tests, or a score-based / LiNGAM search) removes indirect edges and
  sharpens orientation.
