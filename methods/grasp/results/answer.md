# GRaSP, distilled

GRaSP (Greedy Relaxations of the Sparsest Permutation) is a permutation-based causal
discovery algorithm. It searches the space of variable orderings for one whose induced
minimal-I-map DAG is sparsest (highest-scoring), using a single permutation operation —
**tuck** — that performs covered-edge reversals (and more) entirely in permutation space, and
a tiered relaxation that progressively weakens the assumption needed for correctness. It
recovers the CPDAG (Markov equivalence class) of the true DAG.

## Problem it solves

Recover `MEC(G*)` (as a CPDAG) from i.i.d. observational data — here, integer-coded discrete
data from a Bayesian network. The classical constraint-based (PC) and score-based (GES)
searches are correct only under **faithfulness**, which is fragile to near-violations
(near-determinism, near-cancelling paths) that are common on dense graphs, so their accuracy
degrades sharply with graph density. The sparsest-permutation (SP) algorithm is correct under
a strictly weaker assumption but is super-exponential (`m!` orderings, ~9 variables max).
GRaSP keeps a weaker-than-faithfulness razor while scaling to 100+ variables and dense graphs.

## Key ideas

**Order → DAG.** For a permutation `π`, induce `G_π` by giving each vertex, as parents, its
Markov boundary among its predecessors in `π` (the minimal-I-map / boundary-DAG construction).
`G_π` is automatically acyclic, Markovian, and SGS-minimal. Searching orderings
(`2^{O(m log m)}`) avoids the DAG space (`2^{O(m^2)}`) and all acyclicity checks; the score
decomposes over vertices, so a local move re-scores only the affected vertices.

**The tuck operation.** For an edge `j → k ∈ E(G_π)`, write `π = ⟨δ1, j, δ2, k, δ3⟩`; split
`δ2` into `γ` (ancestors of `k`) and `γ^c` (the rest). Then

```
tuck(π, j, k) = ⟨δ1, γ, k, j, γ^c, δ3⟩   if j → k ∈ E(G_π),   else π.
```

When `j → k` is **covered** (`Pa(j) = Pa(k)\{j}`), no vertex of `δ2` can be an ancestor of
`k`: the last parent of `k` on such a path would also have to be a parent of `j` and hence
precede `j`. Thus `γ = ∅` and the tuck collapses to `⟨δ1, k, j, δ2, δ3⟩` — "move `k` to just
before `j`," which is exactly a covered-edge reversal of `G_π`, performed without leaving
permutation space.
A covered tuck never increases the edge count and never loses an independence
(`|E(G_τ)| ≤ |E(G_π)|`, `I(G_π) ⊆ I(G_τ)`); a single tuck can fuse a reversal with edge
deletions, so it is more efficient than a Chickering DAG-walk.

**Tiered edge filter (the relaxation).**

```
E^t(G) = covered edges   (t = 0)  ⊆  singular edges (t = 1)  ⊆  all edges (t = 2)
```

A *singular* edge `j → k` has no unidirectional `j ⤳ k` path except the edge itself. Tier `t`
tucks edges in `E^t`:
- **GRaSP₀** (covered) ≡ TSP: covered tucks = covered-edge reversals; correct & pointwise
  consistent under faithfulness.
- **GRaSP₁** (singular) ≡ ESP, made operational: a tuck of a singular edge equals one
  DAG-changing edge-walk on the DAG associahedron — **without building the polytope**.
- **GRaSP₂** (any edge): a novel, strictly weaker razor; recovers sparser DAGs under
  unfaithfulness where GRaSP₀/₁ stall.

The razors are strictly nested: `CFC = GRaSP₀-razor ⊆ GRaSP₁-razor ⊆ GRaSP₂-razor ⊆
u-frugality`, and `GRaSP₂` is *not* correct under u-frugality alone. In the oracle algorithm,
higher tiers are run after lower tiers (never returning a denser permutation), so statistics
improve monotonically.

**Why faithfulness is necessary for GRaSP₀.** A novel equivalence: faithful DAGs = uniquely
P-minimal DAGs. Hence under (detectable) unfaithfulness there is a P-minimal DAG outside
`MEC(G*)` from which covered moves cannot escape — so GRaSP₀/TSP are wrong. This is why the
tier relaxation (enlarging `E^t`) is the route to a weaker assumption.

**DFS with depth bound.** From a random initial permutation, scan candidate edges; tuck;
accept on strict score improvement (restart), recurse one level deeper on a score-neutral
(within-MEC) move, undo on a worse one. At the DFS root tuck any tier-edge; deeper, restrict
to covered tucks (within-MEC traversal); a flip-set `history` prevents infinite within-MEC
loops. Unbounded DFS (`d = m!`) carries the correctness theorem; in finite samples a shallow
`depth = 3` captures the benefit cheaply.

**Finite-sample instantiation.** Replace oracle negative-edge-count with a decomposable,
locally consistent score and CI-oracle parent selection with **grow-shrink**: grow (add the
best-improving predecessor while any improves) then shrink (remove any whose removal improves)
yields the unique Markov boundary in the limit — no hypothesis tests. For discrete data use
the **BDeu** score; cache each vertex's grow/shrink traces in a **grow-shrink tree (GST)**
keyed by the prefix, so each tuck (which perturbs only a contiguous block) re-derives just the
affected families against the cached trie.

## BDeu local score

With cell counts `N_ijk` (records with `X_i` in state `k` under parent-configuration `j`),
`N_ij = Σ_k N_ijk`, `r_i` = number of states of `X_i`, `q_i = Π_{p∈Pa_i} r_p`, equivalent
sample size `α` (sample_prior), structure prior `s`, and `vm = m − 1`:

```
BDeu_D(X_i, Pa_i) = Σ_j [ lgamma(α/q_i) − lgamma(N_ij + α/q_i)
                          + Σ_k ( lgamma(N_ijk + α/(r_i q_i)) − lgamma(α/(r_i q_i)) ) ]
                    + |Pa_i|·log(s/vm) + (vm − |Pa_i|)·log(1 − s/vm).
```

Decomposable, locally consistent, and score-equivalent across an MEC in its uniform-`α`
marginal-likelihood part; the structure-prior contribution is decomposable and depends on
parent count. Defaults in the implementation: `α = 1`, `s = 1`, `depth = 3`.

## Working code

Faithful to the causal-learn implementation
(`causallearn/search/PermutationBased/GRaSP.py`, `gst.py`, `score/LocalScoreFunction.py`).
`grasp(...)` returns the estimated CPDAG as a `GeneralGraph`; the discrete benchmark calls it
as `grasp(X, score_func="local_score_BDeu", depth=3)`. In this implementation the BDeu branch
passes `parameters=None`, so `local_score_BDeu` uses `sample_prior = 1`, `structure_prior = 1`,
and infers `r_i_map` from the data. The `Order` initializer negates its setup call to
`score.score(y, [])`, but the scores that drive grow, shrink, and DFS are the higher-is-better
values returned by `GST.trace(...)`.

```python
import random, sys, time, warnings
from typing import Any, Dict, List, Optional
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunction import (
    local_score_BDeu, local_score_BIC_from_cov,
)
from causallearn.search.PermutationBased.gst import GST            # cached grow-shrink tree
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag


class Order:
    """Working permutation + each vertex's parents, local score, running edge count."""
    def __init__(self, p, score):
        self.order = list(range(p))
        self.parents, self.local_scores, self.edges = {}, {}, 0
        random.shuffle(self.order)                              # random initial permutation
        for i in range(p):
            y = self.order[i]
            self.parents[y] = []
            self.local_scores[y] = -score.score(y, [])          # causal-learn negates this setup score

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


def grasp(X: np.ndarray, score_func: str = "local_score_BIC_from_cov", depth: Optional[int] = 3,
          parameters: Optional[Dict[str, Any]] = None, verbose: bool = True,
          node_names: Optional[List[str]] = None) -> GeneralGraph:
    X = X.copy()
    n, p = X.shape
    if n < p:
        warnings.warn("The number of features is much larger than the sample size!")

    if score_func == "local_score_BDeu":                        # defaults: sample_prior=1, structure_prior=1
        score = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    elif score_func == "local_score_BIC_from_cov":               # SEM-BIC fallback (Gaussian)
        score = LocalScoreClass(data=X, local_score_fun=local_score_BIC_from_cov,
                                parameters=parameters or {"lambda_value": 2})
    else:
        raise Exception("Unknown function!")

    gsts = [GST(i, score) for i in range(p)]                     # one grow-shrink tree / vertex
    node_names = node_names or [("X%d" % (i + 1)) for i in range(p)]
    nodes = [GraphNode(name) for name in node_names]
    G = GeneralGraph(nodes)

    runtime = time.perf_counter()
    order = Order(p, score)
    for i in range(p):                                           # score vs. preceding vertices
        y = order.get(i)
        y_parents = order.get_parents(y)
        candidates = [order.get(j) for j in range(0, i)]
        order.set_local_score(y, gsts[y].trace(candidates, y_parents))
        order.bump_edges(len(y_parents))

    while dfs(depth - 1, set(), [], order, gsts):               # tuck DFS until no improvement
        if verbose:
            sys.stdout.write("\rGRaSP edge count: %i    " % order.get_edges()); sys.stdout.flush()
    if verbose:
        sys.stdout.write("\nGRaSP completed in: %.2fs \n" % (time.perf_counter() - runtime))

    for y in range(p):                                          # read the DAG off the order
        for x in order.get_parents(y):
            G.add_directed_edge(nodes[x], nodes[y])
    return dag2cpdag(G)                                         # convert to the CPDAG


def dfs(depth: int, flipped: set, history: List[set], order, gsts):
    cache = [{}, {}, {}, 0]
    indices = list(range(order.len())); random.shuffle(indices)
    for i in indices:
        y = order.get(i)
        y_parents = order.get_parents(y); random.shuffle(y_parents)
        for x in y_parents:
            covered = set([x] + order.get_parents(x)) == set(y_parents)
            # causal-learn exposes the top tier here: root ANY parent edge; deeper covered only
            if len(history) > 0 and not covered:
                continue
            j = order.index(x)
            for k in range(j, i + 1):                            # snapshot the affected block
                z = order.get(k)
                cache[0][k] = z; cache[1][k] = order.get_parents(z)[:]
                cache[2][k] = order.get_local_score(z)
            cache[3] = order.get_edges()

            tuck(i, j, order)
            edge_bump, score_bump = update(i, j, order, gsts)

            if score_bump > 1e-6:                                # strict improvement
                order.bump_edges(edge_bump); return True
            if score_bump > -1e-6:                               # score-neutral (within-MEC)
                flipped = flipped ^ set(
                    [tuple(sorted([x, z])) for z in order.get_parents(x)
                     if order.index(z) < i])
                if len(flipped) > 0 and flipped not in history:
                    history.append(flipped)
                    if depth > 0 and dfs(depth - 1, flipped, history, order, gsts):
                        return True
                    del history[-1]
            for k in range(j, i + 1):                            # undo
                z = cache[0][k]; order.set(k, z)
                order.set_parents(z, cache[1][k]); order.set_local_score(z, cache[2][k])
            order.set_edges(cache[3])
    return False


def update(i: int, j: int, order, gsts):
    edge_bump = old_score = new_score = 0
    for k in range(j, i + 1):                                    # only the affected block
        z = order.get(k); z_parents = order.get_parents(z)
        edge_bump -= len(z_parents); old_score += order.get_local_score(z)
        z_parents.clear()
        candidates = [order.get(l) for l in range(0, k)]
        s = gsts[z].trace(candidates, z_parents)                # cached grow-shrink
        order.set_local_score(z, s)
        edge_bump += len(z_parents); new_score += s
    return edge_bump, new_score - old_score


def tuck(i: int, j: int, order):                                # pi = <d1, gamma, k, j, gamma^c, d3>
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

The grow-shrink tree `GST` caches each vertex's grow branches (sorted by grow-score) and
shrink removals, so `trace(prefix, parents)` walks a trie keyed by the available predecessors
rather than rerunning grow-shrink — turning every per-tuck re-score into a cached lookup.

```python
class GSTNode:
    def __init__(self, tree, add=None, score=None):
        if score is None: score = tree.score.score_nocache(tree.vertex, [])
        self.tree, self.add = tree, add
        self.grow_score = self.shrink_score = score
        self.branches, self.remove = None, None

    def __lt__(self, other): return self.grow_score < other.grow_score

    def grow(self, available, parents):
        self.branches = []
        for add in available:
            parents.append(add)
            score = self.tree.score.score_nocache(self.tree.vertex, parents)
            parents.remove(add)
            if score > self.grow_score:
                self.branches.append(GSTNode(self.tree, add, score))
        self.branches.sort(reverse=True)                        # best grow first

    def shrink(self, parents):
        self.remove = []
        while True:
            best = None
            for r in [pp for pp in parents]:
                parents.remove(r)
                score = self.tree.score.score_nocache(self.tree.vertex, parents)
                parents.append(r)
                if score > self.shrink_score:
                    self.shrink_score = score; best = r
            if best is None: break
            self.remove.append(best); parents.remove(best)

    def trace(self, prefix, available, parents):
        if self.branches is None: self.grow(available, parents)
        for branch in self.branches:
            available.remove(branch.add)
            if branch.add in prefix:
                parents.append(branch.add)
                return branch.trace(prefix, available, parents)
        if self.remove is None:
            self.shrink(parents); return self.shrink_score
        for r in self.remove: parents.remove(r)
        return self.shrink_score


class GST:
    def __init__(self, vertex, score):
        self.vertex, self.score = vertex, score
        self.root = GSTNode(self)
        self.forbidden, self.required = [vertex], []

    def trace(self, prefix, parents=None):
        if parents is None: parents = []
        available = [i for i in range(self.score.data.shape[1]) if i not in self.forbidden]
        return self.root.trace(prefix, available, parents)
```

## Relation to prior methods

- **SP** (Raskutti & Uhler): the sparsest-permutation razor and the order→DAG construction.
  GRaSP keeps the razor's spirit but searches greedily instead of enumerating `m!` orderings.
- **Ordering Search** (Teyssier & Koller): the greedy-over-orderings substrate with
  decomposable scores and per-family caching; GRaSP supplies the consistency it lacked.
- **GES / Chickering**: covered edges, covered-edge reversals, Chickering sequences, BIC local
  consistency. A covered tuck *is* a covered-edge reversal in permutation space.
- **TSP / ESP** (Solus, Wang & Uhler): GRaSP₀ ≡ TSP and GRaSP₁ ≡ ESP, both realized in pure
  permutation space via tucks (covered / singular), with GRaSP₁ giving the operational ESP
  without constructing the DAG associahedron, and GRaSP₂ relaxing the razor strictly further.
- **Grow-Shrink** (Margaritis & Thrun): Markov-boundary discovery by score search, realizing
  the order→DAG parent selection without CI hypothesis tests; cached here as a grow-shrink tree.
