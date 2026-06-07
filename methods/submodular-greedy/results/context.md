# Context

## Research question

Given a finite ground set V and a set function f: 2^V -> R that assigns a value
to every subset, we want to choose a subset S of bounded size, |S| <= k, that
makes f(S) as large as possible. Many concrete selection problems have this
shape: pick k sensor locations to cover the most area, pick k documents to
summarize a corpus, pick k facilities to serve the most demand, pick k sets from
a collection to cover the most elements (maximum coverage). The exact optimum is
NP-hard already for maximum coverage, so we cannot hope to compute the best S
exactly in polynomial time. The question is whether some efficient, simple
procedure can come with a *worst-case multiplicative guarantee*: a constant alpha
such that the value of the subset it returns is always at least alpha * f(OPT),
no matter the instance. A guarantee of that form is what separates a principled
algorithm from a heuristic that merely "tends to work."

The class of f we care about is the one where this is possible: f non-negative,
monotone (adding elements never decreases value), and submodular (it exhibits
diminishing returns). Those three properties are exactly the structure that many
real selection objectives share, and they are what a guarantee can be built on.

## Background

A set function f is **monotone** when A contained in B gives f(A) <= f(B):
enlarging the chosen set never hurts. It is **submodular** when it has
*diminishing returns*: for A contained in B and an element e not in B,

    f(A + e) - f(A)  >=  f(B + e) - f(B).

The same element contributes at least as much when added to a smaller set as to a
larger one. There is an equivalent lattice form, easy to check on examples,

    f(A) + f(B)  >=  f(A ∪ B) + f(A ∩ B),

and the two definitions can be shown equivalent. The quantity f(S + e) - f(S) is
the **marginal gain** of e at S, written f(e | S); submodularity says marginal
gains are monotonically non-increasing as the conditioning set grows.

The canonical example is **coverage**. Let S_1, ..., S_m be subsets of a universe
U, indexed by V = {1,...,m}, and define f(I) = | union of S_j over j in I |. This
f is non-negative, monotone, and submodular: a set covers fewer *new* elements
once more of the universe is already covered. Maximum coverage -- choose k of the
S_j to cover the most of U -- is the prototypical instance, and many other
objectives (entropy of selected variables, weighted coverage, facility location)
are also monotone submodular.

The diagnostic fact that makes the problem both hard and approachable: greedy
constructions on combinatorial objectives usually have no guarantee at all (one
can build instances where greedily growing a set ends arbitrarily far from the
optimum). But for the *set cover* problem -- the cost-minimization cousin, "pick
the cheapest collection of sets whose union is all of U" -- the greedy algorithm
that repeatedly takes the set covering the most uncovered elements is known to be
an H_n ~ ln n approximation. That positive result sits on top of exactly the
coverage structure above, and it raises the obvious question: is the logarithmic
charging argument a special accident of coverage, or a symptom of diminishing
returns that should transfer to the value-maximization,
cardinality-constrained version?

## Baselines

**Set-cover greedy (Johnson; Lovász; Chvátal).** For the minimization problem
(cover all of U at minimum cost), repeatedly pick the set with the best
cost-per-newly-covered-element. This yields an H_n approximation for unit costs
and a logarithmic guarantee in general. Its analysis is a charging argument over
how fast the uncovered remainder shrinks. The gap it leaves: it solves the
*covering* (feasibility-to-completion, minimize cost) problem, not the *budgeted*
problem where we are handed a hard size limit k and must maximize value -- and its
guarantee is logarithmic, not a constant fraction.

**Exact / exhaustive selection.** Enumerate all subsets of size k and take the
best. This is the only method that is *guaranteed optimal*, but there are C(m,k)
of them; it is exponential and useless beyond toy sizes. It anchors what "OPT"
means in the analysis but is not a usable algorithm.

**Ad hoc local search and random/restart heuristics.** Randomly sampling size-k
subsets, taking a few restarts, or swapping until no obvious improvement appears
will return *a* set, sometimes a good one, but the guarantee depends on a
specified exchange rule and proof. Without that analysis, extra compute only
changes the search budget; it does not by itself produce a worst-case
multiplicative ratio. They are the foil -- output without a guarantee -- against
which a principled method must be measured.

**The matroid-constraint viewpoint.** A cardinality constraint |S| <= k is the
simplest *independence system*: the uniform matroid of rank k. Posing the problem
over a general matroid (partition constraints, spanning-set constraints) is the
natural generalization, and greedy on a matroid is a classical theme
(matroid greedy is exact for *linear* objectives). For submodular objectives, the
linear-exchange proof no longer applies directly: an element that looks best now
can block future feasible exchanges whose value is not additive. That frames the
cardinality case as the favorable special case where all not-yet-chosen elements
remain feasible single moves until the budget is exhausted.

## Evaluation settings

The natural yardsticks are instances where f(OPT) is known or boundable, so the
realized ratio f(S)/f(OPT) for a returned set S can be inspected:

- **Maximum coverage** instances: a universe U, a family of subsets S_1..S_m, and
  a budget k; metric is the number (or weight) of covered elements.
- **Set-cover-derived** instances where the optimal cover size is controlled, used
  to probe the logarithmic regime of the covering cousin.
- **Sensor-placement / facility-location** style instances on a grid or graph,
  where f is coverage or a monotone-submodular utility and k is the budget.

The protocol is worst-case / adversarial-instance analysis (the guarantee must
hold for *every* instance), supplemented by counting oracle calls -- the number of
evaluations of f -- since f is given as a value oracle and each evaluation is the
unit of cost.

## Code framework

The objective is a value oracle. The available ingredients are an oracle, a
coverage instance, and an exhaustive checker for small instances. The remaining
question is how to grow S under the budget.

```python
from itertools import combinations


class SetFunction:
    """Value oracle over a ground set of element indices."""
    def __init__(self, ground_set):
        self.ground_set = list(ground_set)

    def value(self, S):
        raise NotImplementedError

    def marginal(self, e, S):
        # f(S + e) - f(S)
        return self.value(S | {e}) - self.value(S)


class Coverage(SetFunction):
    """f(S) = | union of the subsets indexed by S |."""
    def __init__(self, sets):
        super().__init__(range(len(sets)))
        self.sets = [set(s) for s in sets]

    def value(self, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(covered)

    def marginal(self, e, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(self.sets[e] - covered)


def exact_best(f, k):
    """Exhaustive optimum for tiny instances, used only as a checker."""
    best_S, best_value = set(), f.value(set())
    limit = min(k, len(f.ground_set))
    for r in range(limit + 1):
        for combo in combinations(f.ground_set, r):
            S = set(combo)
            value = f.value(S)
            if value > best_value:
                best_S, best_value = S, value
    return best_S, best_value


def select(f, k):
    """Return a subset S with |S| <= k aiming to maximize f(S).

    The selection rule is still to be designed.
    """
    S = set()
    # TODO: grow S up to size k under the value oracle.
    return S
```
