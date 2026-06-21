The problem is to recover the Markov equivalence class of an unknown directed acyclic graph from i.i.d. observations of discrete variables. Observational data alone can only identify the CPDAG, and the classical route to it — constraint-based PC or score-based GES — is correct only under faithfulness, the assumption that every independence in the distribution is forced by the graph structure. Faithfulness is fragile exactly where it matters: on dense graphs, near-deterministic relations and near-cancelling paths produce tiny partial correlations that finite-sample tests read as independences, so edges are wrongly deleted and errors cascade. The sparsest-permutation algorithm of Raskutti and Uhler replaces faithfulness with a strictly weaker razor — the true DAG is the unique sparsest Markovian DAG — but it exhaustively enumerates all m! permutations and is intractable beyond about nine variables. Greedy ordering search is tractable and empirically strong, yet it has no consistency guarantee. The challenge is to keep a weaker-than-faithfulness correctness guarantee while scaling to dozens or hundreds of variables.

I propose GRaSP, the Greedy Relaxations of the Sparsest Permutation. It searches the space of variable orderings rather than the space of DAGs. For a permutation π, each vertex is given, as parents, its Markov boundary among the vertices that precede it in π. The resulting G_π is automatically acyclic, Markovian, and minimal; searching orderings avoids the global acyclicity constraint and shrinks the search space from 2^{O(m^2)} structures to 2^{O(m log m)} permutations. The score decomposes over vertices, so a local move only re-scores the affected families. The key move is the tuck: for an edge j → k in G_π, write π = ⟨δ1, j, δ2, k, δ3⟩, split δ2 into γ (ancestors of k) and γ^c (the rest), and rearrange to ⟨δ1, γ, k, j, γ^c, δ3⟩. When j → k is covered — Pa(j) = Pa(k) \ {j} — no vertex of δ2 can be an ancestor of k, so the tuck collapses to moving k just before j, which is exactly a covered-edge reversal performed entirely in permutation space. A covered tuck never increases the edge count and never loses an independence; it can even fuse a reversal with edge deletions in a single step.

GRaSP relaxes which edges may be tucked. Covered edges give GRaSP_0, equivalent to the triangle-sparsest-permutation algorithm; singular edges, those with no unidirectional j ⇝ k path except the edge itself, give GRaSP_1, an operational form of the edge-sparsest-permutation walk on the DAG associahedron without ever constructing the polytope; and any edge gives GRaSP_2, a strictly weaker razor that recovers sparser DAGs under unfaithfulness where the lower tiers stall. In practice the search runs the tiers in order, so statistics improve monotonically. From a random initial permutation, it performs a depth-bounded DFS: at the root it may tuck any parent edge, while deeper recursive descents are restricted to covered tucks so they stay within an equivalence class. A strict score improvement is accepted and the search restarts; score-neutral moves are explored for a bounded number of steps; worse moves are undone. A flip-set history prevents cycling among Markov-equivalent DAGs. With finite data, the oracle negative-edge-count objective is replaced by the decomposable, locally consistent BDeu score, and parent selection is performed by grow-shrink on that score — growing the parent set by adding the best-improving predecessor, then shrinking away redundant parents — with no thresholded CI test anywhere. Each vertex's grow-shrink trace is cached in a grow-shrink tree keyed by the prefix, so a tuck, which perturbs only a contiguous block of the permutation, re-derives just the affected families against cached traces.

```python
import random, sys, time, warnings
from typing import Any, Dict, List, Optional
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunction import (
    local_score_BDeu, local_score_BIC_from_cov,
)
from causallearn.search.PermutationBased.gst import GST
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag


class Order:
    def __init__(self, p, score):
        self.order = list(range(p))
        self.parents, self.local_scores, self.edges = {}, {}, 0
        random.shuffle(self.order)
        for i in range(p):
            y = self.order[i]
            self.parents[y] = []
            self.local_scores[y] = -score.score(y, [])

    def get(self, i): return self.order[i]
    def set(self, i, y): self.order[i] = y
    def index(self, y): return self.order.index(y)
    def insert(self, i, y): self.order.insert(i, y)
    def pop(self, i=-1): return self.order.pop(i)
    def get_parents(self, y): return self.parents[y]
    def set_parents(self, y, ps): self.parents[y] = ps
    def get_local_score(self, y): return self.local_scores[y]
    def set_local_score(self, y, s): self.local_scores[y] = s
    def get_edges(self): return self.edges
    def set_edges(self, e): self.edges = e
    def bump_edges(self, b): self.edges += b
    def len(self): return len(self.order)


def grasp(X: np.ndarray, score_func: str = "local_score_BIC_from_cov",
          depth: Optional[int] = 3, parameters: Optional[Dict[str, Any]] = None,
          verbose: bool = True, node_names: Optional[List[str]] = None) -> GeneralGraph:
    X = X.copy()
    n, p = X.shape
    if n < p:
        warnings.warn("The number of features is much larger than the sample size!")

    if score_func == "local_score_BDeu":
        score = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    elif score_func == "local_score_BIC_from_cov":
        score = LocalScoreClass(data=X, local_score_fun=local_score_BIC_from_cov,
                                parameters=parameters or {"lambda_value": 2})
    else:
        raise Exception("Unknown function!")

    gsts = [GST(i, score) for i in range(p)]
    node_names = node_names or [("X%d" % (i + 1)) for i in range(p)]
    nodes = [GraphNode(name) for name in node_names]
    G = GeneralGraph(nodes)

    runtime = time.perf_counter()
    order = Order(p, score)
    for i in range(p):
        y = order.get(i)
        y_parents = order.get_parents(y)
        candidates = [order.get(j) for j in range(0, i)]
        order.set_local_score(y, gsts[y].trace(candidates, y_parents))
        order.bump_edges(len(y_parents))

    while dfs(depth - 1, set(), [], order, gsts):
        if verbose:
            sys.stdout.write("\rGRaSP edge count: %i    " % order.get_edges())
            sys.stdout.flush()
    if verbose:
        sys.stdout.write("\nGRaSP completed in: %.2fs \n" % (time.perf_counter() - runtime))

    for y in range(p):
        for x in order.get_parents(y):
            G.add_directed_edge(nodes[x], nodes[y])
    return dag2cpdag(G)


def dfs(depth: int, flipped: set, history: List[set], order, gsts):
    cache = [{}, {}, {}, 0]
    indices = list(range(order.len())); random.shuffle(indices)
    for i in indices:
        y = order.get(i)
        y_parents = order.get_parents(y); random.shuffle(y_parents)
        for x in y_parents:
            covered = set([x] + order.get_parents(x)) == set(y_parents)
            if len(history) > 0 and not covered:
                continue
            j = order.index(x)
            for k in range(j, i + 1):
                z = order.get(k)
                cache[0][k] = z; cache[1][k] = order.get_parents(z)[:]
                cache[2][k] = order.get_local_score(z)
            cache[3] = order.get_edges()

            tuck(i, j, order)
            edge_bump, score_bump = update(i, j, order, gsts)

            if score_bump > 1e-6:
                order.bump_edges(edge_bump); return True
            if score_bump > -1e-6:
                flipped = flipped ^ set(
                    [tuple(sorted([x, z])) for z in order.get_parents(x)
                     if order.index(z) < i])
                if len(flipped) > 0 and flipped not in history:
                    history.append(flipped)
                    if depth > 0 and dfs(depth - 1, flipped, history, order, gsts):
                        return True
                    del history[-1]
            for k in range(j, i + 1):
                z = cache[0][k]; order.set(k, z)
                order.set_parents(z, cache[1][k]); order.set_local_score(z, cache[2][k])
            order.set_edges(cache[3])
    return False


def update(i: int, j: int, order, gsts):
    edge_bump = old_score = new_score = 0
    for k in range(j, i + 1):
        z = order.get(k); z_parents = order.get_parents(z)
        edge_bump -= len(z_parents); old_score += order.get_local_score(z)
        z_parents.clear()
        candidates = [order.get(l) for l in range(0, k)]
        s = gsts[z].trace(candidates, z_parents)
        order.set_local_score(z, s)
        edge_bump += len(z_parents); new_score += s
    return edge_bump, new_score - old_score


def tuck(i: int, j: int, order):
    ancestors = []
    get_ancestors(order.get(i), ancestors, order)
    shift = 0
    for k in range(j + 1, i + 1):
        if order.get(k) in ancestors:
            order.insert(j + shift, order.pop(k)); shift += 1


def get_ancestors(y: int, ancestors: List[int], order):
    ancestors.append(y)
    for x in order.get_parents(y):
        if x not in ancestors:
            get_ancestors(x, ancestors, order)
```
