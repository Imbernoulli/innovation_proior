## Research question

Given an undirected graph `G` with `n` nodes and a cost (weight) `c(i,j) ≥ 0` on each edge, we
want to split the nodes into subsets so that the **total cost of the edges cut** — edges whose two
endpoints land in different subsets — is as small as possible, subject to a hard limit on how
large each subset may be. The cleanest and most important instance is the **balanced bisection**:
`2n` nodes to be split into two subsets `A`, `B` of exactly `n` nodes each, minimizing the
external cost `T = Σ_{a∈A, b∈B} c(a,b)`.

This is not an academic toy. The motivating application is laying out electronics: the nodes are
the components of a circuit, the edge costs are the number (or cost) of wires between components,
and the subsets are physical printed-circuit cards or substrates, each with a bounded capacity.
Connections that cross between cards are expensive and slow, so we want to assign components to
cards so that the number of between-card connections is minimized while no card is overfilled. The
same shape recurs whenever a connected system must be cut into bounded pieces with as little
coupling across the cut as possible — for instance, partitioning a program's procedures into
fixed-size memory pages to minimize cross-page references.

Since the problem is computationally hard, the practical goal is a heuristic that produces good
partitions quickly on realistic instances (tens to a few hundred nodes), while respecting the size
constraint exactly. The deliverable is a single self-contained C++17 program that reads the graph
from stdin and writes the resulting partition summary to stdout.

## Background

**The size constraint makes the problem hard.** Without the balance constraint, "smallest cut
separating the graph into two parts" is solvable in polynomial time. With it, the search space is
enormous: the number of ways to split `n` size-1 nodes into `k` subsets of size `p` (where
`kp = n`) is `(1/k!)·C(n,p)·C(n−p,p)···C(p,p)`, which for `n = 40`, `p = 10`, `k = 4` already
exceeds `10^20`. The problem can be cast as an integer linear program with many constraints to
enforce the uniformity of the subsets, but any direct/exhaustive approach demands an inordinate
amount of computation at realistic sizes. So the field is in the heuristic regime: produce good
solutions quickly, accepting that optimality cannot be guaranteed. A procedure whose running time
grows exponentially or factorially in `n` is not practical; the target is a polynomial-time
`f(n)`-procedure with `f(n)` close to `n²`.

**Iterative improvement / local search.** The dominant framework for hard combinatorial problems
is: generate a feasible solution; attempt a transformation to a cheaper feasible solution; if one
is found, move there and repeat; when no transformation improves the current solution, it is a
**local optimum**; then restart from another initial solution and keep the best. The design choice
that matters most is the choice of transformation: a stronger transformation yields fewer and
better local optima, so a larger fraction of random starts reaches the global optimum.

**Min-cut placement as the downstream use.** In electronic physical design, the natural way to
place a large netlist on a chip is top-down: recursively cut the cell set (and the chip area) into
two balanced halves with as few nets crossing as possible, then recurse on each half. The quality
of the whole placement is governed by the quality of the balanced min-cut routine at each level.
So a fast, strong balanced-bisection heuristic is the workhorse that such a top-down placement
flow is built on.

## Baselines

**Random solutions.** Generate random feasible partitions, keep the best, stop after a time or
value budget. Fast (an `n²`-procedure), but useless beyond toy sizes: on `32×32` 0-1 matrices
there are typically only 3–5 optimal partitions out of `½·C(32,16)`, so the probability that a
random trial is optimal is below `10^{-7}`.

**Max-flow / min-cut (Ford & Fulkerson).** Treat the graph as a flow network with edge costs as
capacities; the max-flow–min-cut theorem gives a cut of minimum capacity separating two chosen
nodes, i.e. a minimum-cost partition into two subsets of unspecified sizes. (Minimizing over the
choice of the two separated nodes gives the *global* unconstrained 2-way min-cut, which is a valid
lower bound on any partition's cost.)

**Clustering.** Identify "natural clusters" in the cost matrix — groups of strongly connected
nodes — and build subsets around them.

**Single-pair interchange (`λ`-opting, `λ = 1`).** Borrowing the variable-rearrangement idea from
the traveling-salesman line (Lin 1965), the analogue for partitioning is the `λ`-change: exchange
`λ` nodes of one set with `λ` of the other. A partition is **1-opt** if no single-pair exchange
reduces the cost. Experiments on `32×32` 0-1 matrices show 1-opting reaches an apparently optimal
value in about 10% of trials and gets within 1 or 2 of optimal in about 75% of cases. At `λ = 1`
the procedure is already an `n²`-operation per step; extending to larger `λ` raises the
computational effort steeply.

## Evaluation settings

The natural yardstick is a battery of cost matrices: 0-1 matrices with nonzero-element density
ranging from about 5% to 50%; integer matrices with entries uniform on `[0, k]` for
`k = 2, …, 10`; and matrices with clusters of known sizes and binding strengths (so that an
optimal partition is known by construction). Instance sizes run up to a few hundred nodes (the
largest tested are in the 300–360 range). The metric is the external cost of the partition
(smaller is better), measured against the best known or constructed optimum; alongside it one
reports the running time and the fraction of random starts that reach the optimum — a statistical
confidence in optimality, since no certificate is available. The relevant comparison is against
the fixed-`λ` interchange heuristic and against random restart, at comparable computing budgets,
on the same matrices.

## Code framework

The program contract is deliberately plain: read an even integer `m = 2n`, then read the `m × m`
symmetric nonnegative cost matrix in row-major order. Start from the balanced split
`A = {0, ..., n-1}`, `B = {n, ..., 2n-1}`. Write the initial cut cost, the final cut cost, and the
two output blocks in the same stdout format shown by the scaffold. The slot to be filled is the
balance-preserving improvement logic that updates the side assignment before the final report.

```cpp
#include <bits/stdc++.h>
using namespace std;

long long external_cost(const vector<vector<long long>>& cost,
                        const vector<int>& side) {
    int m = (int)side.size();
    long long total = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            if (side[i] != side[j]) total += cost[i][j];
        }
    }
    return total;
}

int main() {
    int m;
    if (!(cin >> m)) return 0;

    vector<vector<long long>> cost(m, vector<long long>(m, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            cin >> cost[i][j];
        }
    }

    int n = m / 2;
    vector<int> side(m, 0);
    for (int i = n; i < m; ++i) side[i] = 1;

    cout << "start cut: " << external_cost(cost, side) << "\n";

    // TODO: update side to a better balanced partition while keeping exactly n nodes per block.

    cout << "final cut: " << external_cost(cost, side) << "\n";

    vector<int> A, B;
    for (int i = 0; i < m; ++i) {
        (side[i] == 0 ? A : B).push_back(i);
    }

    cout << "A:";
    for (int x : A) cout << ' ' << x;
    cout << "\nB:";
    for (int x : B) cout << ' ' << x;
    cout << "\n";
    return 0;
}
```
