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

What a solution must achieve, therefore, is not a guarantee of optimality but *reliably good tours,
fast*: optimum or near-optimum on the classical and randomly generated instances of the day, on the
order of tens to a hundred-plus cities, with running time that grows gently with `n` (something near
quadratic) rather than explosively. The practical bar is to beat the existing fast heuristics in
solution quality at a comparable or better cost, and to do so without the user having to hand-tune a
difficulty knob in advance.

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

**The motivating limitation of fixed `k`.** Testing all `k`-exchanges has time complexity that grows
like `O(n^k)` in a naive implementation, and there is no useful bound on how many improving
`k`-exchanges a tour admits. So computational effort rises steeply with `k`, and — this is the
decisive pain point — `k` must be chosen *in advance*. There is no way to know a priori which `k`
strikes the best compromise between running time and tour quality for a given instance: too small and
you stall at a poor local optimum; too large and the per-move cost is ruinous. So in practice the
fixed-`k` interchange leaves the user guessing at a depth that, by all appearances, ought to differ
from instance to instance and even from move to move within a single run.

## Baselines

**2-opt (Croes 1958).** The `k`-opt interchange with `k` fixed at 2: repeatedly remove two tour links
and reconnect the two resulting paths the other way, which amounts to reversing one subsegment of the
tour, keeping the move whenever it shortens the tour; stop at a 2-opt local optimum. Simple and fast,
and a 2-opt move always keeps the tour feasible, but it is a shallow neighborhood — it leaves many
poor local optima that a deeper exchange would escape.

**3-opt (Lin 1965).** The same idea with `k = 3`: remove three links and reconnect the three paths in
one of the several possible ways that yields a tour. Markedly better quality than 2-opt, and the
tours reached are 3-opt (hence also 2-opt). But the per-move work is much larger (on the order of
`n^3` to scan the neighborhood), and the user is still committed to `k = 3` regardless of the
instance. Going to `k = 4` or `5` improves quality further but the cost grows again, and the choice of
`k` remains a blind guess.

**Exact methods of the time.** Held & Karp's approach solves a class of instances exactly in
reasonable time, but when an instance falls outside that class it must be supplemented (branch and
bound) and run times become prohibitive; the largest instance reported is 64 cities. These set the
"optimum" reference on small instances but do not scale.

**Man-machine and multi-heuristic schemes.** Krolak et al. use several fast, weak heuristics and then
human judgment on plots of the tour ("man-machine interaction") to push toward optimality. This
reaches large (200-city) instances but is costly in machine and especially human time, gives
generally suboptimal results, and breaks down entirely for non-Euclidean or non-planar instances.

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

The primitives that already exist: a distance matrix with `dist(i,j)`, a tour represented as an
ordered list of cities with a routine to compute its length, a way to look at the two tour-neighbors
of a city and to test whether a proposed set of edges still forms a single Hamiltonian cycle, and
per-city neighbor lists ordered by distance. The slot to be filled is the *transformation* — the
local-search step that turns one tour into a better one.

```python
class TSP:
    """Holds the instance; subclasses implement an improvement heuristic."""
    edges = {}  # global cost matrix

    @staticmethod
    def dist(i, j):
        return TSP.edges[i][j]

    @staticmethod
    def pathCost(path):
        cost = TSP.dist(path[-1], path[0])          # close the loop
        for i in range(1, len(path)):
            cost += TSP.dist(path[i - 1], path[i])
        return cost


class Tour:
    """A tour as an ordered city list, plus its edge set for membership tests."""
    def __init__(self, tour): ...
    def around(self, node):
        """Return (predecessor, successor) of a city on the tour."""
        ...
    def contains(self, edge): ...                   # is this edge currently in the tour?
    def generate(self, broken, joined):
        """Given a set of tour edges to delete and a set of new edges to add,
        rebuild the city order; report whether the result is a single tour."""
        ...


class Improver(TSP):
    """The local-search transformation. To be designed."""
    def _optimise(self):
        # repeatedly apply the improving step until no improvement, then save
        # TODO: the step that turns one tour into a strictly shorter one
        pass

    def improve(self):
        # TODO: search for one improving move from the current tour;
        #       return True (and update the tour) if one is found.
        pass
```
