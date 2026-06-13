# BOSS (Best Order Score Search)

BOSS is a permutation-based causal discovery algorithm. It greedily improves an ordering of the
variables, projects that ordering to a DAG by selecting each variable's parents from its
predecessors, and returns the CPDAG of the projected DAG. The two central pieces are a
best-position move for one variable at a time and a per-variable Grow-Shrink Tree that stores
reusable grow/shrink work for repeated prefix queries.

## Problem it solves

Recover `MEC(G*)` as a CPDAG from purely observational i.i.d. data while avoiding the brittle
single-edge decisions that hurt PC and GES on dense graphs with almost-violations of faithfulness.
The permutation/sparsest-order view gives a more robust search target, but exhaustive
permutation enumeration is infeasible. BOSS keeps the order-based projection and replaces
enumeration with greedy order moves plus cached parent selection.

## Method

- **Projection.** Given an order `pi`, each variable `v` may only choose parents from
  `pre_pi(v)`. Grow-Shrink selects a Markov boundary within that predecessor set using a
  decomposable local score: grow adds the best-improving predecessor until no addition improves
  the score, and shrink removes the best-improving member until no removal improves the score.
  The projected DAG score is the sum of the final per-node local scores.
- **Best-position move.** For a chosen variable `v`, evaluate every insertion slot in the current
  order and move `v` to the best-scoring slot if it improves the total score by more than a small
  tolerance. Sweep variables in shuffled order and repeat until a full sweep makes no move.
- **Grow-Shrink Trees.** For each target variable, the tree root is the empty parent set. Expanding
  a node scores every candidate addition, keeps only branches with `score > grow_score`, sorts
  those branches descending, and traces a prefix by taking the first sorted branch whose variable
  is present in the prefix. Shrink removals and shrink scores are cached at the terminal grown
  node.
- **Optional BES.** The theoretical two-phase variant can run Backward Equivalence Search after
  the ordering phase. With causal Markov, faithfulness, and a curved exponential family, the
  two-phase variant is asymptotically correct for every initial order because projection returns
  a subgraph-minimal DAG containing `P`, and BES supplies the final correctness step.
- **CPDAG output.** Observational data identify only the MEC, so the final DAG is converted to a
  CPDAG.

## Discrete Local Score

For `X_i` with `r_i` states, parent set `PA_i`, `q_i = product_{a in PA_i} r_a` parent
configurations, counts `N_ij` and `N_ijk`, and equivalent sample size `alpha`, BDeu uses

```text
a_ij  = alpha / q_i
a_ijk = alpha / (r_i * q_i)
```

and the local log marginal likelihood

```text
sum_j [
  lgamma(a_ij) - lgamma(a_ij + N_ij)
  + sum_k (lgamma(a_ijk + N_ijk) - lgamma(a_ijk))
]
```

with the optional structure-prior term
`|PA_i| * log(s / vm) + (vm - |PA_i|) * log(1 - s / vm)`, where `vm = p - 1`.
The Dirichlet mass per parent configuration is `r_i * alpha/(r_i*q_i) = alpha/q_i`, and
the total prior mass over the table is `alpha`. BDeu is score-equivalent, so all DAGs in a MEC
receive the same score.

## Canonical Call

For integer-coded discrete observational data, BOSS with the BDeu score:

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.search.PermutationBased.BOSS import boss


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """X: (n_samples, n_variables) integer-encoded discrete data -> estimated CPDAG."""
    return boss(
        X,
        score_func="local_score_BDeu",
        parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    )
```

In causal-learn the BDeu branch constructs `LocalScoreClass(..., parameters=None)`, so
`local_score_BDeu` falls back to its own defaults `sample_prior = 1`, `structure_prior = 1`, and
per-variable state counts taken from the data — which coincide with the values passed above.

## Engine Sketch

The core implementation follows causal-learn's `BOSS.py` and `gst.py`: `GSTNode.grow` keeps only
improving branches and sorts them descending, `trace` takes the first sorted branch contained in
the prefix, and `better_mutation` computes all insertion scores with one forward and one backward
prefix-score sweep before applying `order.insert(best - int(best > i), v)`.

```python
import random
import warnings
from typing import List, Optional

import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunction import local_score_BDeu
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag


class GSTNode:
    def __init__(self, tree, add=None, score=None):
        if score is None:
            score = tree.score.score_nocache(tree.vertex, [])
        self.tree = tree
        self.add = add
        self.grow_score = score
        self.shrink_score = score
        self.branches = None
        self.remove = None

    def __lt__(self, other):
        return self.grow_score < other.grow_score

    def grow(self, available, parents):
        self.branches = []
        for add in available:
            parents.append(add)
            score = self.tree.score.score_nocache(self.tree.vertex, parents)
            parents.remove(add)
            branch = GSTNode(self.tree, add, score)
            if score > self.grow_score:
                self.branches.append(branch)
        self.branches.sort(reverse=True)

    def shrink(self, parents):
        self.remove = []
        while True:
            best = None
            for remove in [parent for parent in parents]:
                parents.remove(remove)
                score = self.tree.score.score_nocache(self.tree.vertex, parents)
                parents.append(remove)
                if score > self.shrink_score:
                    self.shrink_score = score
                    best = remove
            if best is None:
                break
            self.remove.append(best)
            parents.remove(best)

    def trace(self, prefix, available, parents):
        if self.branches is None:
            self.grow(available, parents)
        for branch in self.branches:
            available.remove(branch.add)
            if branch.add in prefix:
                parents.append(branch.add)
                return branch.trace(prefix, available, parents)
        if self.remove is None:
            self.shrink(parents)
            return self.shrink_score
        for remove in self.remove:
            parents.remove(remove)
        return self.shrink_score


class GST:
    def __init__(self, vertex, score):
        self.vertex = vertex
        self.score = score
        self.root = GSTNode(self)
        self.forbidden = [vertex]
        self.required = []

    def trace(self, prefix, parents=None):
        if parents is None:
            parents = []
        available = [i for i in range(self.score.data.shape[1]) if i not in self.forbidden]
        return self.root.trace(prefix, available, parents)

    def reset(self):
        self.root = GSTNode(self)


def reversed_enumerate(seq, j):
    for w in reversed(seq):
        yield j, w
        j -= 1


def better_mutation(v, order, gsts):
    i = order.index(v)
    p = len(order)
    scores = np.zeros(p + 1)

    prefix = []
    score = 0
    for j, w in enumerate(order):
        scores[j] = gsts[v].trace(prefix) + score
        if v != w:
            score += gsts[w].trace(prefix)
            prefix.append(w)

    scores[p] = gsts[v].trace(prefix) + score
    best = p

    prefix.append(v)
    score = 0
    for j, w in reversed_enumerate(order, p - 1):
        if v != w:
            prefix.remove(w)
            score += gsts[w].trace(prefix)
        scores[j] += score
        if scores[j] > scores[best]:
            best = j

    if scores[i] + 1e-6 > scores[best]:
        return False
    order.remove(v)
    order.insert(best - int(best > i), v)
    return True


def boss_discrete(X: np.ndarray, node_names: Optional[List[str]] = None) -> GeneralGraph:
    X = X.copy()
    n, p = X.shape
    if n < p:
        warnings.warn("The number of features is much larger than the sample size!")

    score = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    order = list(range(p))
    gsts = [GST(v, score) for v in order]
    parents = {v: [] for v in order}

    variables = list(order)
    while True:
        improved = False
        random.shuffle(variables)
        for v in variables:
            improved |= better_mutation(v, order, gsts)
        if not improved:
            break

    for i, v in enumerate(order):
        parents[v].clear()
        gsts[v].trace(order[:i], parents[v])

    names = [("X%d" % (i + 1)) for i in range(p)] if node_names is None else node_names
    nodes = [GraphNode(name) for name in names]
    G = GeneralGraph(nodes)
    for y in range(p):
        for x in parents[y]:
            G.add_directed_edge(nodes[x], nodes[y])
    return dag2cpdag(G)
```

## Relation To Prior Methods

- **SP:** keeps the permutation projection but replaces exhaustive enumeration with greedy moves.
- **Ordering Search:** replaces adjacent transpositions with a best insertion position for one
  variable.
- **GRaSP:** stays in the permutation/Grow-Shrink lineage but removes the tuck-depth search knobs
  from the forward phase.
- **GES/fGES:** borrows the optional backward-equivalence cleanup, while the forward phase searches
  orderings instead of adding individual edges.
