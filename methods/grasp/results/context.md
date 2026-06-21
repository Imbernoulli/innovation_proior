# Context: causal structure search from observational data (circa 2020-2021)

## Research question

Given `n` i.i.d. observations of `m` discrete variables `V = {X_1, ..., X_m}` drawn from
some unknown ground-truth directed acyclic graph (DAG) `G*`, recover as much of `G*` as the
data can possibly determine. Observational data alone cannot orient every edge: two DAGs
that encode exactly the same set of conditional-independence (CI) relations are
indistinguishable from any i.i.d. sample, so the most one can identify is the
**Markov equivalence class** (MEC) of `G*`, represented by its CPDAG (a partially directed
graph whose directed edges are the orientations shared by every DAG in the class and whose
undirected edges are the ones that flip within the class). The goal is therefore: a
procedure that takes integer-coded discrete data and returns the CPDAG of `G*`, measured
against the true CPDAG by structural Hamming distance and by adjacency / arrowhead precision
and recall, across dozens to hundreds of variables and a range of graph densities and
discrete cardinalities.

## Background

A DAG `G` over `V` asserts a set of CI relations through d-separation; write `I(G)` for that
set and `I(P)` for the CI relations that actually hold in the distribution `P`. `G` is
**Markovian** for `P` if `I(G) ⊆ I(P)` (every separation in the graph is a true
independence); `G*` is assumed Markovian. A stronger assumption is **faithfulness**,
`I(P) ⊆ I(G*)` as well, i.e. *every* independence in the data is graph-induced. The geometry
of faithfulness has been studied (Uhler et al. 2013): the set of distributions that violate
faithfulness has measure zero, but the set that *almost* violates it — where a true edge
produces a partial correlation so small that finite-sample CI tests read it as
independence — occupies a non-negligible volume, and that volume grows with graph density.

Two minimality notions weaker than faithfulness are in use here.
**SGS-minimality** (minimal I-map): `G` is Markovian and no proper subgraph of `G` is
Markovian — you cannot delete an edge and stay Markovian. **P-minimality** (Pearl's
minimality): `G` is Markovian and there is no other Markovian `G'` with `I(G) ⊂ I(G')`
strictly — no Markovian graph entails *more* independences. A still different criterion ranks
Markovian DAGs by **edge count**: the **sparsest Markovian DAG** is the one with the fewest
edges. Under faithfulness all these notions coincide with `MEC(G*)`; they can come apart on
other distributions, and an algorithm correct under a *weaker* notion is correct on *more*
distributions.

A second background thread is how to turn a **variable ordering** into a DAG. Fix a
permutation `π` of the vertices. For each vertex `k`, look only at the vertices that precede
`k` in `π` and select as `k`'s parents the smallest subset that renders `k` conditionally
independent of all the other predecessors — its **Markov boundary** relative to the prefix
(Verma & Pearl 1988/1990; Pearl 1988, the "boundary DAG" construction). The resulting `G_π`
is automatically acyclic (all edges point forward in `π`), is Markovian, and is SGS-minimal.
Equivalently one can build `G_π` by a CI-test rule — put `j → k` iff `j` precedes `k` and
`X_j ⊥̸ X_k` given the rest of `k`'s predecessors — and the two constructions agree when `P`
is a graphoid. This reframes structure search: instead of searching the space of DAGs
(`2^{O(m^2)}` of them, with a global acyclicity constraint that couples every parent-set
choice to every other), one can search the space of orderings
(`2^{O(m log m)}` of them) and read off a DAG from each — no acyclicity check needed,
because an ordering can never produce a cycle.

A third thread is **decomposable scoring**. Rather than running CI hypothesis tests,
score a candidate structure by a decomposable, locally consistent score and turn "is `X_j` a
parent of `X_k`" into a score comparison. The BIC score
`BIC_D(X_k, M) = ℓ_{X_k|M}(θ̂_mle) − c·(|θ̂|/2)·log n` (log-likelihood penalized by
parameter count) decomposes over vertices, `BIC_D(G) = Σ_i BIC_D(X_i, Pa(i, G))`, and is
*locally consistent*: in the large-sample limit, adding `j → k` raises the score iff
`X_j ⊥̸ X_k | Pa(k)` (Chickering 2002; Haughton 1988). For discrete data the natural choice
is a marginal-likelihood score under a Dirichlet prior — the **BDeu** score
(Heckerman, Geiger & Chickering 1995): with sufficient statistics being the cell counts
`N_ijk` (number of records with `X_i` in state `k` under parent-configuration `j`),

```
BDeu_D(X_i, Pa_i) = Σ_j [ lgamma(α/q_i) − lgamma(N_ij + α/q_i)
                          + Σ_k ( lgamma(N_ijk + α/(r_i q_i)) − lgamma(α/(r_i q_i)) ) ]
                    + |Pa_i|·log(s/vm) + (vm − |Pa_i|)·log(1 − s/vm),
```

where `r_i` is `X_i`'s number of states, `q_i = Π_{p∈Pa_i} r_p` is its number of parent
configurations, `N_ij = Σ_k N_ijk`, `α` is the equivalent sample size (a single
"sample_prior" pseudo-count spread uniformly over cells), `s` is the structure prior, and
`vm = m − 1`. The uniform `α` sharing is the feature that makes the marginal-likelihood
part score-equivalent across a Markov equivalence class; the structure-prior sum depends on
parent count. This is also decomposable and locally consistent, and unlike a CI test it has
no significance threshold to tune.

A fourth thread is **Markov-boundary discovery by search rather than testing**: the
grow-shrink procedure (Margaritis & Thrun 1999). Grow: starting from the empty set, keep
adding the variable that most improves the (BIC) score while any addition improves it.
Shrink: from the grown set, keep removing any variable whose removal improves the score.
With a decomposable locally consistent score on a compositional graphoid, grow-then-shrink
returns the unique Markov boundary in the large-sample limit — exactly the parent set the
ordering construction above needs, obtained without a single hypothesis test.

An empirical fact frames the simulations: across the standard protocols, the accuracy of the
classical constraint- and score-based searches varies with **graph density**. They do well on
sparse graphs and their accuracy changes as average degree rises.

## Baselines

These are the prior methods a new search would be compared against and would react to.

**PC (Spirtes, Glymour & Scheines 2000).** Constraint-based. Start from a complete undirected
graph; for each pair, run CI tests over growing conditioning sets and delete the edge if any
test passes; orient v-structures, then propagate Meek's orientation rules. Returns a CPDAG.
Recovers `MEC(G*)` under (restricted-)faithfulness. Every decision rests on a thresholded CI
hypothesis test.

**GES (Chickering 2002).** Score-based, operating over equivalence classes. Forward phase:
greedily add the single edge that most improves a decomposable score (BIC); backward phase:
greedily delete. The state space is `MEC`s, navigated by the operators that the
**Meek/Chickering theory** of covered edges licenses. Two facts from that theory recur
throughout this area: an edge `i → j` is **covered** when `Pa(i) = Pa(j) \ {i}` (the two
endpoints share all other parents), and reversing a covered edge yields a Markov-equivalent
DAG; moreover any two Markov-equivalent DAGs differing in `k` orientations are connected by
`k` covered-edge reversals. More generally, if `I(H) ⊆ I(G)` there is a sequence of
covered-edge reversals and edge additions transforming `H` into `G` while keeping
`I(·) ⊆ I(G)` at every step (a "Chickering sequence"). GES uses this to climb to a
score-optimal MEC and is correct & pointwise consistent under **faithfulness**.

**Hill-climbing (classical local search).** Greedily add/delete/reverse single edges to
improve a decomposable score, with restarts and a tabu list to escape local optima. Operates
in DAG space with the global acyclicity constraint.

**Sparsest-permutation search (Raskutti & Uhler 2013/2018).** Rank orderings by the edge
count of their induced minimal-I-map `G_π` and output the sparsest `G_π` over all `π`. This
is correct under the **sparsest-Markov-representation** condition — `G*` is the unique
sparsest Markovian DAG (also called u-frugality) — which is **strictly weaker than
faithfulness**. It is an exhaustive search over all `m!` permutations.

**Ordering Search (Teyssier & Koller 2005).** Make the sparsest-permutation idea tractable by
*greedily* hill-climbing the space of orderings instead of enumerating it. The neighborhood
is the set of **adjacent transpositions** — swap two neighboring vertices in `π`. The
ordering space is `2^{O(m log m)}` versus `2^{O(m^2)}` structures; the branching factor is
`O(m)`; no acyclicity checks are ever needed; and because the score decomposes, swapping the
neighbors at positions `(i, i+1)` changes only those two vertices' optimal parent sets — every
other family's contribution is unchanged — so a move is cheap if the per-family scores are
cached. The search is hill-climbing with random restarts and a tabu list over this space.

**Greedy sparsest-permutation on the DAG associahedron (Solus, Wang & Uhler 2017/2021).**
Bring geometry to bear. The orderings form the vertices of a polytope, the **permutohedron**
`A_v`, whose edges are adjacent transpositions; contract every transposition that corresponds
to a CI relation of `P` and the result is the **DAG associahedron** `A_v(P)`, a polytope whose
vertices are the distinct induced DAGs (the SGS-minimal DAGs) and whose edges connect DAGs
that differ by a single covered-arrow reversal. Two greedy DFS variants walk this structure
toward sparser DAGs: one (call it the *edge* variant) walks the edges of `A_v(P)` via
weakly-decreasing DAG-changing moves; the other (the *triangle* variant, the one run in
practice) walks weakly-decreasing **covered-arrow-reversal** sequences in DAG space,
Chickering-style. They come with identifiability assumptions — the "edge" and "triangle"
assumptions — that are both **weaker than faithfulness and stronger than the
sparsest-Markov-representation condition**, so they sit between Ordering Search and SP. The
edge variant is described in terms of the polytope `A_v(P)`; the triangle variant does its
work in DAG space via covered-arrow reversals and edge deletions, moving between the
permutation and DAG representations, where one move is "reverse an arrow, take a linear
extension, recompute the minimal I-map."

## Evaluation settings

The yardsticks that exist before any new method:

- **Discrete Bayesian-network benchmarks** from the bnlearn repository, with known
  ground-truth DAGs and conditional probability tables, spanning small to large networks and
  diverse cardinalities: Cancer (5 nodes / 4 edges), Child (20 / 25), Alarm (37 / 46),
  Hailfinder (56 / 66), Win95pts (76 / 112). Each is sampled at a fixed observational sample
  size; a method must generalize across scales and cardinality patterns from one
  configuration.
- **Linear-Gaussian simulations** sweeping one parameter at a time around a base
  configuration of 60 variables, average degree 6, sample size 1,000: number of variables
  20→100; average degree 2→10; sample size 200→100,000. Edge coefficients drawn `U(−1, 1)`,
  exogenous noise `N(0, 1)`; statistics averaged over independent runs.
- **Metrics.** Estimated CPDAG vs. ground-truth CPDAG (the true DAG converted to its CPDAG):
  structural Hamming distance (lower better); adjacency precision/recall (skeleton recovery);
  arrowhead precision/recall (orientation accuracy), with precision `= TP/(TP+FP)` and recall
  `= TP/(TP+FN)`.
- **Score / test defaults at the time.** Score-based methods use BIC (a penalty multiplier of
  2 for the linear-Gaussian case) and the BDeu marginal likelihood for discrete data with a
  sample prior; constraint-based PC uses a CI test (chi-squared for discrete,
  zero-partial-correlation at a small significance level for Gaussian).
- **Protocol.** Run on a single laptop; memory (for caching per-family scores) is the binding
  resource. Among the algorithms compared, the ones meant to scale are expected to finish a
  100-variable problem in a reasonable time.

## Code framework

A new search plugs into the same harness the baselines use: read integer-coded discrete data,
score candidate structures with an existing decomposable score, and return a CPDAG through the
shared causal-graph interface. The available substrate is the decomposable discrete score
(BDeu); the minimal-I-map-from-an-ordering construction realized by grow-shrink
Markov-boundary discovery, with per-family score caching so that re-scoring a vertex against a
different prefix is cheap; and the conversion from a fully oriented DAG to its CPDAG. The part
to design is how to move through the space of orderings: which permutation-changing operation
to apply, which candidate moves to consider, and how deep to search.

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag


class LocalScore:
    """Decomposable discrete score (BDeu) over integer-coded data.
    score(y, parents) returns the local score of vertex y given a parent set;
    decomposable, so the total score is the sum over vertices. Cached per family."""

    def __init__(self, data, score_func="local_score_BDeu", parameters=None):
        self.data = data
        # ... builds the BDeu local-score function with sample_prior / structure_prior
        pass

    def score(self, y, parents):
        # local BDeu score of y given `parents` (cached)
        raise NotImplementedError


def markov_boundary(score, y, prefix, parents):
    """Grow-shrink: select y's parents from `prefix` (the vertices allowed to precede y)
    by maximizing the decomposable score, returning the local score. Realizes the
    minimal-I-map-from-an-ordering construction without CI hypothesis tests. Cached
    across calls (re-scoring against a different prefix should reuse work)."""
    # grow: add best-improving variable while it improves
    # shrink: remove any whose removal improves
    raise NotImplementedError


class Order:
    """A working permutation with, for each vertex, its selected parents and local score,
    plus the running total edge count / score. Initialized to a random order."""

    def __init__(self, p, score):
        self.order = list(range(p))
        self.parents = {y: [] for y in range(p)}
        self.local_scores = {}
        # ... random shuffle; score each vertex against an empty prefix
        pass


def search(X, depth=3, score_func="local_score_BDeu", parameters=None):
    """Recover the CPDAG of integer-coded discrete data X."""
    n, p = X.shape
    score = LocalScore(X, score_func, parameters)
    order = Order(p, score)

    # score each vertex against the vertices preceding it in the initial order
    for i in range(p):
        y = order.order[i]
        order.local_scores[y] = markov_boundary(
            score, y, [order.order[j] for j in range(i)], order.parents[y]
        )

    # TODO: the order-space search we will design.
    #       Repeatedly transform `order` to improve the total decomposable score,
    #       re-deriving the affected vertices' parents/scores after each move,
    #       until no improving move is found.
    pass

    # read the DAG off the final order, convert to a CPDAG
    nodes = [GraphNode("X%d" % (i + 1)) for i in range(p)]
    G = GeneralGraph(nodes)
    for y in range(p):
        for x in order.parents[y]:
            G.add_directed_edge(nodes[x], nodes[y])
    return dag2cpdag(G)
```

The harness supplies a random ordering, a cached decomposable score, the
parents-from-an-ordering machinery, and the DAG→CPDAG conversion. The order-space search
itself is the part to be designed.
