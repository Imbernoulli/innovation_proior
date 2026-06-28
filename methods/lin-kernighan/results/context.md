## Research question

We are given the symmetric traveling-salesman problem: an `n × n` symmetric matrix of distances
`c(i,j)` between `n` cities, and we must find a minimum-length *tour* — a cyclic permutation
`(i_1, i_2, ..., i_n)` of the cities that minimizes

    c(i_1,i_2) + c(i_2,i_3) + ... + c(i_{n-1},i_n) + c(i_n,i_1).

A tour is a subset `T` of the `n(n-1)/2` possible links (edges) that forms a single Hamiltonian
cycle; the feasibility criterion `C` is "`T` is a tour", and the objective `f(T)` is its length.
There are `(n-1)!/2` distinct tours, so exhaustive search is hopeless even for moderate `n`, and the
prevailing belief by the early 1970s — reinforced by the then-new complexity theory — is that the
problem is inherently exponential: exact methods blow up in running time at realistic sizes.

The practical bar is to find optimum or near-optimum tours on the classical and randomly generated
instances of the day, on the order of tens to a hundred-plus cities, with running time that grows
gently with `n` rather than explosively. The question is how to design a `k`-exchange local-search
step that achieves high solution quality at reasonable computational cost. The deliverable is a
single self-contained C++17 program that reads the instance from stdin and writes the tour and its
length to stdout.

## Background

**Iterative improvement / local search.** The dominant framework for hard combinatorial problems is:
generate a pseudorandom feasible solution `T` (a set satisfying `C`); attempt a transformation to a
better feasible `T'` with `f(T') < f(T)`; if found, replace `T` by `T'` and repeat; when no
transformation improves `T`, it is a *local optimum*; then restart from a new random solution and
keep the best, until time runs out. Random uniformly-distributed starts are used rather than
constructive ones, both because a good improvement heuristic reaches good tours from a random start
about as fast as from a constructed one, and because constructive starts are usually deterministic so
they give only one initial solution. The quality of the whole scheme is governed by the quality of
the transformation in the middle step: the better it is, the smaller the set of local optima and the
higher the fraction of random starts that reach the global optimum.

**The `k`-opt interchange (`λ`-opt).** The standard transformation for the TSP is the
*`k`-exchange*: delete `k` links of the current tour and reconnect the resulting paths with `k` new
links — possibly reversing some paths — so the result is again a tour, and keep it if it is shorter.
A tour is called **`k`-optimal** (`k`-opt) if no exchange of any `k` of its links for `k` other links
shortens it. Any `k`-opt tour is also `k'`-opt for `1 ≤ k' ≤ k`, and an `n`-city tour is optimal iff
it is `n`-opt. Intuitively, the larger `k`, the more likely a `k`-opt tour is globally optimal.
Testing all `k`-exchanges has time complexity that grows like `O(n^k)` in a naive implementation.

## Baselines

**2-opt (Croes 1958).** The `k`-opt interchange with `k` fixed at 2: repeatedly remove two tour links
and reconnect the two resulting paths the other way, which amounts to reversing one subsegment of the
tour, keeping the move whenever it shortens the tour; stop at a 2-opt local optimum. Simple and fast;
a 2-opt move always keeps the tour feasible.

**3-opt (Lin 1965).** The same idea with `k = 3`: remove three links and reconnect the three paths in
one of the several possible ways that yields a tour. Tours reached are 3-opt (hence also 2-opt); the
per-move work is on the order of `n^3` to scan the neighborhood.

**Exact methods of the time.** Held & Karp's approach solves a class of instances exactly in
reasonable time, with branch-and-bound supplementation for other instances; the largest instance
reported is 64 cities. These set the "optimum" reference on small instances.

**Man-machine and multi-heuristic schemes.** Krolak et al. use several fast, weak heuristics and then
human judgment on plots of the tour ("man-machine interaction") to push toward optimality. This
reaches large (200-city) instances with a combination of machine and human time.

## Evaluation settings

The natural yardstick is the collection of classical TSP instances from the literature together with
randomly generated test problems. The classical set includes small named instances used repeatedly in
prior work; the random instances are drawn at sizes ranging up to roughly 100–110 cities, both
Euclidean (points in the plane) and general symmetric (arbitrary symmetric distance matrices, where
pictorial methods do not apply). The metric is tour length (shorter is better), measured against the
best known or proven optimum for the instance; alongside it one reports the running time to reach a
local optimum and the fraction of random starts that reach the optimum (a statistical confidence in
optimality, since no certificate is available). The relevant comparison is against the fixed-`k`
interchange heuristics — 2-opt and 3-opt — at comparable computing budgets, on the same instances.

## Code framework

The program should read `n`, then the full `n × n` symmetric distance matrix in row-major order from
stdin. It should print the chosen tour as `n` zero-based city indices in visit order on one line, then
print the tour length with four digits after the decimal point on the next line. The scaffold below is
deliberately pre-method; the algorithmic step is left blank.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n) || n <= 0) return 0;

    vector<vector<double>> dist(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> dist[i][j];

    vector<int> tour(n);
    double length = 0.0;

    // TODO: (NO hint at the method name or idea -- this is pre-method)

    for (int i = 0; i < n; i++)
        cout << tour[i] << (i + 1 < n ? ' ' : '\n');
    cout << fixed << setprecision(4) << length << '\n';
    return 0;
}
```
