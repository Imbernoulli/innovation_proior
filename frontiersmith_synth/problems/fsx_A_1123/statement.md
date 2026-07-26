# Shepherds of the Drifting Flock

A flock is scattered across a network of trust links: an undirected, connected
graph on `n` positions, where an edge means two positions pull each other
together. Left alone, the whole flock drifts. You may plant `k` **shepherds**
— fixed anchor positions that never drift — and every other position is
pulled toward its neighbors (anchors included). Your job: choose the `k`
anchor positions that make the flock's *worst-case* drift die out as fast as
possible.

## The physics (grounded Laplacian)

Let `L` be the graph Laplacian (`L[i][i] = deg(i)`, `L[i][j] = -1` for each
edge `{i,j}`). Given an anchor set `S` (`|S| = k`), delete the rows and
columns of `S` from `L` to get the **grounded Laplacian** `L_g`, a
`(n-k) x (n-k)` matrix. Because the graph is connected and `k >= 1`, `L_g` is
positive definite. Its **smallest eigenvalue**, `lambda_min(L_g)`, is the
slowest restoring rate among all non-anchor positions — the whole flock's
drift decays no faster than this. **You want to choose `S` to MAXIMIZE
`lambda_min(L_g)`.**

Two structural facts about this objective, honestly stated (the exact graphs
are only in the input — read and exploit them):

- `lambda_min(L_g)` is throttled by whichever non-anchor position is hardest
  to *reach* from the anchors, in the graph's own **effective-resistance**
  metric (not raw hop-distance, not degree) — a position dangling at the end
  of a long, thin path can be nearly invisible to a placement chosen by
  degree or centrality alone, yet it alone can pin the eigenvalue down.
- `lambda_min(L_g)` is **not submodular** in `S`: there exist graphs where a
  specific *pair* of anchors, chosen jointly, beats anything reachable by
  repeatedly adding whichever single node currently helps the eigenvalue the
  most. A one-node-at-a-time recipe can get permanently stuck.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from
**stdin**, write ONE JSON object (your answer) to **stdout**. It runs in an
isolated subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... choose k anchors ...
print(json.dumps({"anchors": anchors}))
```

### Public instance (stdin)

```json
{
  "name": "twin_tendril_1",
  "n": 42,                       // number of positions, 0..n-1
  "k": 4,                        // number of anchors you must choose
  "edges": [[0,1],[0,2], ...],   // undirected simple connected graph
  "seed": 4001                   // for reproducing the reference baseline
}
```

### Answer (stdout)

```json
{ "anchors": [3, 17, 0, 29] }    // exactly k distinct integers in [0, n)
```

Any invalid output (wrong length, duplicate or out-of-range index, a
non-integer, a crash, a timeout, or non-JSON output) makes that instance
score `0.0`.

## Objective

**Maximize** `lambda_min(L_g)` (defined above), averaged across a fixed,
seeded family of 10 instances. Several instances plant a twin-hub-and-tendril
or dumbbell-and-bridge structure specifically so that anchor placements based
only on degree, centrality, or one-node-at-a-time marginal improvement leave
a far pendant branch essentially unanchored; others are generic random
sparse graphs testing whether your strategy generalizes.

## Scoring (deterministic)

For each instance the evaluator computes, itself, from the graph alone
(never from your answer):

- `q_rand` = `lambda_min(L_g)` for a fixed seeded-random anchor set (a weak,
  topology-blind reference),
- `q_ref`  = `lambda_min(L_g)` for the best of several internal references,
  including a resistance-farthest-cover construction with local refinement,

and normalizes with an affine anchor:

```
r = clamp(0.1 + 0.8 * (q_cand - q_rand) / max(1e-9, q_ref - q_rand), 0, 1)
```

Matching the random baseline scores ≈ `0.1`; matching the internal
near-best reference scores ≈ `0.9` — the reference is deliberately not
optimal, so a genuinely better *joint* placement can still score higher.
Doing worse than random scores below `0.1`.

## Constraints

- `1 <= n <= 100`, `1 <= k <= n - 2`, graph is simple, undirected, connected.
- Time limit: your program should finish in well under a minute per
  instance (graphs are small; no iterative method needs more than a second
  or two).
