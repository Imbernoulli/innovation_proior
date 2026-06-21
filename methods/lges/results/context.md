# Context: Greedy Equivalence Search and its variants (circa 2025)

## Research question

We are given a data matrix `D` of `m` iid records over `n` variables sampled from some unknown
distribution `P(V)`, and we want the *Markov equivalence class* (MEC) of the DAG that generated
it — the CPDAG that names that class. The starting point is Greedy Equivalence Search (GES),
which is, in the large-sample limit, *guaranteed* to recover the true MEC: it searches the space
of equivalence classes by greedily applying the highest-scoring edge insertion until none
improves the score (the forward phase), then the highest-scoring edge deletion until none
improves it (the backward phase), and a decomposable, consistent, score-equivalent scoring
criterion makes this two-phase greedy climb provably correct.

The question this work asks is whether the *basic assumption of GES* — that the greedy choice of
the highest-scoring neighbor is the best one — is actually the right strategy, and whether a
different choice of insertion can improve both accuracy and runtime while retaining the
large-sample guarantee.

## Background

**Greedy Equivalence Search (Chickering 2002; Meek 1997).** GES searches over MECs by maximizing
a scoring criterion. It represents each class by its unique completed PDAG (CPDAG): a directed
edge for every *compelled* edge (oriented the same in every member DAG) and an undirected edge
for every *reversible* one. It assumes the score is **decomposable**
(`S(G,D) = sum_i s(X_i, Pa_i)`), **consistent** (in the limit it prefers a structure that can
represent `P` over one that cannot, and among those, fewer parameters), and **score-equivalent**
(equivalent DAGs get equal scores). Decomposability plus consistency imply **local consistency**
(Chickering 2002, Lemma 7; Def. 1 here): in the large-sample limit, adding `X → Y` to a DAG
*raises* the score iff `X` and `Y` are dependent given `Y`'s current parents `Pa_Y`, and *lowers*
it iff `X ⊥ Y | Pa_Y` in `P`. This is the property that makes the greedy climb correct.

**The two operators.** GES moves between CPDAGs with two operators (Chickering 2002, Defs. 12, 13):

> `Insert(X, Y, T)`: for non-adjacent `X, Y` in CPDAG `E` and a subset `T ⊆ Ne_Y \ Adj_X` of `Y`'s
> neighbors not adjacent to `X`, modify `E` by inserting `X → Y` and directing each previously
> undirected `T − Y` for `T ∈ T` as `T → Y`.

Intuitively, `Insert(X, Y, T)` corresponds to choosing a DAG `G ∈ E`, adding `X → Y`, and
computing the MEC of the result; `T` is the set of `Y`'s undirected neighbors made parents of `Y`
(each becoming a new v-structure tail). The validity test reduces to a clique condition on
`NA_{Y,X} ∪ T` plus a semi-directed-path condition, and the score change is a single node-family
difference. `Delete(X, Y, H)` is the backward analogue. Crucially, **both operators are scored and
validity-tested locally on the CPDAG without ever enumerating member DAGs**, and the score change
of an operator equals the score change of any witness DAG (by decomposability + score equivalence).

**GES's guarantee.** Given a score with the three properties and data `D ∼ P(V)` with `P` Markov
and faithful to some DAG, GES recovers the true MEC in the sample limit (Chickering 2002,
Lemma 10). The forward phase reaches a class with respect to which `P` is Markov (it has at least
the true edges, possibly more); the backward phase removes the excess down to the perfect map. The
PC algorithm (Spirtes, Glymour & Scheines 2000) has similar asymptotic guarantees but uses CI
tests instead of a score, while many methods — max-min hill-climbing (Tsamardinos et al. 2006),
NOTEARS (Zheng et al. 2018) — lack such large-sample guarantees.

**Generalizing GES.** Generalized GES (GGES) allows the search to be initialized from an
arbitrary MEC, to use any order of operators, and to apply *any* valid score-increasing operator,
not just the highest-scoring one. GGES still finds the global optimum of the score in the sample
limit (Correctness of GGES). This means one is free to choose *which* score-increasing insertion
to take.

**A diagnostic about non-adjacency.** For a non-adjacent pair `(X, Y)`, in the sample limit, if a
valid `Insert(X, Y, T)` operator is *score-decreasing* for some `T`, that exposes a conditioning
set under which `X ⊥ Y`, and by local consistency a score-decreasing insertion means exactly "this
independence is real" (Proposition 1 here). Even a single score-decreasing `Insert(X, Y, T)` does
not imply that *all* `Insert(X, Y, *)` are score-decreasing — GES may apply a different
`Insert(X, Y, T')` and introduce a spurious adjacency anyway (Example 2).

**The XGES heuristics (related prior work).** Extremely Greedy Equivalence Search (Nazaret & Blei
2024) introduces complementary heuristics — prioritizing insertions before deletions, and forcing
edge deletions with restarts — which are available as optional add-ons in this family of methods.

## Code framework

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.score.LocalScoreFunction import local_score_BDeu


def ges_variant(X: np.ndarray) -> GeneralGraph:
    """Scaffold: modified forward insert + GES backward delete."""
    N = X.shape[1]
    score_func = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    nodes = [GraphNode("X%d" % (i + 1)) for i in range(N)]
    G = GeneralGraph(nodes)  # start from the empty-graph class
    cache = {}

    # Forward: choose which Insert operators to apply from among valid score-increasing ones
    # Backward: greedily apply the highest-scoring Delete until none improves
    # (populate the loops with causal-learn GES operator primitives)

    return G
```
