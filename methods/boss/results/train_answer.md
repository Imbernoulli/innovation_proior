We are given n i.i.d. observations on p discrete variables and must recover the causal structure that generated them from purely observational data. Because interventions and temporal order are absent, the data can only identify the true DAG G* up to its Markov equivalence class, so the proper target is the completed partially directed acyclic graph, or CPDAG. The practical difficulty is the realistic regime: hundreds to thousands of variables, average degrees in the tens, and dense enough graphs that almost-violations of faithfulness are common. Near-cancelling paths and near-deterministic relations push real dependences close to zero, and finite-sample procedures that hinge on a single conditional-independence test or a single edge's marginal score gain are easily misled. In that setting PC's CI tests become low-powered and wrongly delete edges, while GES's greedy forward phase is fooled by faint score gains and either misses edges or adds spurious ones. GRaSP fixes many of these failures with a permutation-based search using tucks over orderings, but it pays for that accuracy with a depth-bounded DFS over complex tuck operators and several interacting tuning parameters, which limits comfortable scaling past a few hundred variables and makes deployment fiddly. The sparsest-permutation principle itself is robust under weaker assumptions than faithfulness, yet enumerating all p! orderings is infeasible.

The method I propose is BOSS, Best Order Score Search. It keeps the permutation view that makes sparsest permutation robust, but replaces exhaustive enumeration with a simple, parameter-free greedy move. For a current ordering pi of the variables, each variable v may choose parents only from its predecessors. BOSS selects those parents by Grow-Shrink on the decomposable, score-equivalent BDeu marginal likelihood: grow adds the predecessor that most improves the local score until no addition helps, then shrink removes redundant members until no removal helps. Because BDeu is locally consistent, this encodes the right conditional-independence reasoning as score comparisons rather than thresholded tests, and it produces a subgraph-minimal DAG containing the distribution. To move through ordering space, BOSS takes one variable v at a time, removes it from pi, and inserts it at the single position that maximizes the total projected score. This best-position move collapses a whole run of adjacent transpositions into one decision, large enough to escape the shallow optima that trap single-swap hill climbers, yet it has no depth parameters to tune. Variables are swept in shuffled order each round and the sweep repeats until a full pass makes no improving move. Because projection from any order yields a subgraph-minimal DAG, an optional GES-style backward equivalence search after convergence supplies asymptotic correctness under the usual Markov, faithfulness, and curved-exponential-family assumptions for any starting permutation. The final DAG is converted to a CPDAG.

The computational burden is made tractable by Grow-Shrink Trees. Each variable owns a tree whose root is the empty parent set and whose nodes represent parent sets reached by strictly score-improving additions. Expanding a node scores every candidate addition, keeps only branches that strictly improve the score, and sorts them descending. Tracing a prefix means walking down the first sorted child whose variable lies in the prefix; shrink results are cached at terminal nodes. Since BOSS evaluates the same variable at many insertion positions with heavily overlapping prefixes, the tree lets those traces reuse previously expanded nodes instead of rerunning Grow-Shrink from scratch. That caching is what makes repeated best-position sweeps practical on large dense graphs. The discrete local score uses BDeu with equivalent sample size alpha and an optional structure prior; score-equivalence ensures the search objective is well-defined over Markov equivalence classes rather than depending on a particular DAG representative.

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
            for remove in list(parents):
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

    def trace(self, prefix, parents=None):
        if parents is None:
            parents = []
        available = [i for i in range(self.score.data.shape[1])
                     if i not in self.forbidden]
        return self.root.trace(prefix, available, parents)


def better_mutation(v, order, gsts):
    i = order.index(v)
    p = len(order)
    scores = np.zeros(p + 1)

    prefix = []
    score = 0.0
    for j, w in enumerate(order):
        scores[j] = gsts[v].trace(prefix) + score
        if v != w:
            score += gsts[w].trace(prefix)
            prefix.append(w)
    scores[p] = gsts[v].trace(prefix) + score
    best = p

    prefix.append(v)
    score = 0.0
    j = p - 1
    for w in reversed(order):
        if v != w:
            prefix.remove(w)
            score += gsts[w].trace(prefix)
        scores[j] += score
        if scores[j] > scores[best]:
            best = j
        j -= 1

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


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """Discrete observational data -> estimated CPDAG."""
    return boss_discrete(X)
```
