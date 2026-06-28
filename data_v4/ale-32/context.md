# Capacitated k-Means (cardinality-capped clustering)

## Research question

You are given `n` points in the integer plane and an integer `k`. Partition the points into `k`
clusters subject to a hard **per-cluster cardinality cap**: every cluster may hold **at most** `cap`
points. A cluster's representative is its **centroid** — the mean of the points assigned to it,
computed in real arithmetic. The task is to **minimize** the total within-cluster squared Euclidean
distance,

```
cost = sum over points i of || p_i - centroid(cluster of i) ||^2,
```

i.e. ordinary k-means cost, but with the assignment constrained so no cluster exceeds `cap`. This is
the *balanced / capacitated* k-means problem. Even plain k-means cost minimization is NP-hard, and
the cardinality cap only makes it harder; there is no exact answer to read off, and the quality of a
solution is judged by a continuous score. The only lever is the heuristic that decides the
assignment.

## Input / output contract

- Input (stdin):
  - first line: `n k cap` — number of points `n` (`0 <= n <= 1000`), number of clusters `k`
    (`2 <= k <= n`), and the per-cluster size cap `cap` (`1 <= cap`, with `k * cap >= n` so a
    feasible partition exists);
  - then `n` lines, the `i`-th being `x_i y_i` — the integer coordinates of point `i`
    (`0 <= x_i, y_i <= 10^4`).
- Output (stdout):
  - `n` integers `a_0 a_1 ... a_{n-1}` — the cluster label `a_i in [0, k)` assigned to point `i`
    (input order);
  - then `2*k` numbers — the `k` center coordinates `cx_j cy_j` (the cluster representatives).
  Tokens are whitespace-separated; line breaks do not matter. The center coordinates are read but
  **not trusted** for scoring: the scorer recomputes each cluster's centroid from the assignment
  itself (the centroid is exactly what minimizes within-cluster squared distance, so emitting the
  true centroids can only ever match the scorer's own recomputation). The center block must still be
  present and parse as `2*k` reals.
- Time limit: about 2 seconds. Memory: 256 MB.

Example shape: with `n = 5, k = 2, cap = 3`, a valid output is `0 0 1 1 1` followed by two
coordinate pairs — cluster 0 has 2 points, cluster 1 has 3 points, neither exceeds `cap = 3`, and
both clusters are non-empty.

## Background

Without the cap this is the classic k-means problem and **Lloyd's algorithm** is the standard
workhorse: alternate (i) assign each point to its nearest centroid and (ii) move each centroid to the
mean of its assigned points, until nothing changes. Each half is a closed-form / greedy step and the
cost decreases monotonically. The trouble is step (i): under a hard cap, "assign each point to its
nearest centroid" is usually **infeasible** — a popular, dense region attracts far more than `cap`
points to one centroid, and there is no legal way to keep them all there.

Two reference points frame the design.

- **Uncapped Lloyd + greedy cap-repair.** Run ordinary k-means ignoring the cap, then repair: while
  some cluster is over capacity, evict its farthest-from-center members into the nearest cluster with
  spare room. This is fast and always lands on a feasible partition, but the repair is myopic — it
  patches overflow locally and leaves a lot of cost on the table when the caps bite hard. This is the
  baseline the scorer normalizes against.
- **Capacitated assignment as a transportation problem.** The cap turns the *assignment* half of
  Lloyd's iteration into a **min-cost flow / transportation** problem: one unit of supply per point,
  capacity `cap` per centroid, edge cost = squared distance point→centroid. Solving that exactly
  gives the cheapest cap-respecting assignment to the *current* centroids, which the centroid update
  then refines. Alternating these two is the established strong method for capacitated k-means. The
  open design choices are how the flow is solved (and warm-started) and how to account for the fact
  that the true objective measures against *recomputed* centroids, not the fixed centroids the flow
  optimized against.

## Evaluation settings

A solution is first checked for **feasibility**; any violation floors the score to **0**:

1. the output parses as exactly `n + 2*k` numbers (`n` integer labels, then `2*k` reals);
2. every label `a_i` is an integer in `[0, k)`;
3. **no cluster is empty** — every index `0..k-1` is used at least once (an empty cluster has no
   centroid and wastes a representative);
4. every cluster's size is `<= cap` (the hard cardinality cap).

For a feasible solution the scorer recomputes each cluster's centroid `c_j = mean of its members`
and the **cost** is `sum_i || p_i - c_{a_i} ||^2` (lower is better). The score normalizes against a
deterministic **uncapped-Lloyd-plus-greedy-cap-repair** baseline the scorer recomputes itself:

```
score = round(1_000_000 * baseline_cost / max(1e-9, solver_cost))     (0 if INFEASIBLE)
```

So the baseline scores about `1_000_000`; a lower-cost (tighter) capacitated clustering scores more,
and an infeasible one scores `0`.

**Instances** are generated deterministically from an integer seed. `n` is in `[400, 900]`, `k` in
`[6, 14]`, and `cap` is set so that `k * cap` only modestly exceeds `n` (total capacity in
`[1.05 n, 1.30 n]`), which makes the caps **binding**. Points are drawn from a mixture of 2D Gaussian
blobs whose number is near `k` but deliberately not equal and whose populations are uneven — so some
regions of the plane would overflow their natural cluster and must spill into a neighbour. That is
exactly the regime where a naive nearest-centroid assignment is infeasible and a greedy cap-repair is
visibly suboptimal, and where the transportation reformulation pays off.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes a feasible
solution to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k, cap;
    if (!(cin >> n >> k >> cap)) return 0;
    vector<double> px(n), py(n);
    for (int i = 0; i < n; ++i) cin >> px[i] >> py[i];

    // A feasible fallback: a round-robin labelling 0,1,...,k-1,0,1,... so every
    // cluster is non-empty and no cluster exceeds cap (k*cap >= n guarantees it).
    // TODO heuristic: alternate (i) centroid update (closed form) with (ii) a
    // CAPACITATED ASSIGNMENT solved as a min-cost flow / transportation problem
    // (cap per centroid, cost = squared distance), then polish with an
    // incremental move/swap local search on the true recomputed-centroid cost.
    vector<int> label(n);
    for (int i = 0; i < n; ++i) label[i] = i % k;

    // emit labels then k centroids recomputed from the labelling
    vector<double> sx(k, 0), sy(k, 0); vector<int> cnt(k, 0);
    for (int i = 0; i < n; ++i) { sx[label[i]] += px[i]; sy[label[i]] += py[i]; cnt[label[i]]++; }
    for (int i = 0; i < n; ++i) cout << label[i] << " \n"[i + 1 == n];
    for (int j = 0; j < k; ++j) {
        double cx = cnt[j] ? sx[j] / cnt[j] : 0.0, cy = cnt[j] ? sy[j] / cnt[j] : 0.0;
        cout << cx << ' ' << cy << '\n';
    }
    return 0;
}
```
