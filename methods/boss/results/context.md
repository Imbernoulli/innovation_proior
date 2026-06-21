# Context: scalable causal discovery from observational data (circa 2022-2023)

## Research question

Given a dataset of `n` i.i.d. observations on `p` variables and nothing else — no
interventions, no temporal information — recover the causal structure that generated it, as far
as observational data can. Under the standard causal Markov and causal faithfulness assumptions,
observational data identify the true DAG `G*` only up to its Markov equivalence class (MEC): the
set of DAGs encoding exactly the same conditional independences, summarized by a completed
partially directed acyclic graph (CPDAG). So the goal is to return the CPDAG of `G*`.

The graphs of interest are the ones people actually have. Real systems — functional brain
imaging, gene regulation, electronic health records, financial networks — routinely have
hundreds or thousands of variables, each connected to dozens of others. The question is how to
recover `MEC(G*)` on graphs of this size and density from observational data alone.

## Background

**The model and what is identifiable.** A DAG `G = (V, E)` over variables `X` factors a
distribution `P` so that each variable is conditionally independent of its non-descendants given
its parents. The graphical separation criterion is d-separation; `A ⊥ B | C [G]` denotes that
`A` and `B` are d-separated by `C`. The link to causality is two assumptions: *causal Markov*,
`A ⊥ B | C [G] ⇒ A ⊥ B | C [P]` (graphical separation implies probabilistic independence), and
*causal faithfulness*, the converse `A ⊥ B | C [P] ⇒ A ⊥ B | C [G]` (every independence in the
data is graphical, none is accidental). A DAG that satisfies causal Markov is said to *contain*
`P`; a DAG is *subgraph-minimal* (equivalently SGS-minimal, a minimal I-map) if no subgraph of it
still contains `P`. Under Markov + faithfulness, finding `MEC(G*)` is exactly finding the simplest
model that contains the data distribution — a model-selection problem.

**Scoring via BIC and curved exponential families.** The natural objective for that model
selection is the Bayesian information criterion. For a DAG it decomposes over vertices,
`BIC(G, X) = Σ_v BIC(X_v, X_pa(v))`, with each local term a penalized log-likelihood
`ℓ_{v|pa(v)}(θ_mle) − (λ/2)·|θ_mle|·log n` (`λ > 0` a penalty discount, `|θ_mle|` the number of
free parameters). For distributions in a curved exponential family — Gaussian DAG models and
multinomial (discrete) DAG models both qualify — the BIC is *consistent*: in the large-sample
limit it is maximized by models that contain `P` and have the fewest parameters, and the causal
DAG is among them (Haughton 1988; Schwarz 1978). For these families all DAGs in a MEC have the
same score, so it is conventional to return the CPDAG. The score is *locally consistent*
(Chickering 2002): in the limit, adding edge `j → k` to `G` raises the BIC iff
`X_j ⊥̸ X_k | X_pa(k)`. This is what lets a score stand in for a conditional-independence test.

**The discrete case: BDeu.** For multinomial data the local term is taken to be the
Bayesian–Dirichlet *equivalent uniform* (BDeu) marginal likelihood (Heckerman, Geiger &
Chickering 1995). With `X_i` having `r_i` states, parent set `PA_i` inducing
`q_i = Π_{a∈PA_i} r_a` joint configurations, counts `N_ij` (samples in parent-config `j`) and
`N_ijk` (samples with `X_i = k` in config `j`), and a single hyperparameter `α` (the equivalent
sample size), the local score is the log of
`Π_j [ Γ(α/q_i) / Γ(α/q_i + N_ij) · Π_k Γ(α/(r_i q_i) + N_ijk) / Γ(α/(r_i q_i)) ]`,
optionally times a structure prior over the parent set. BDeu is the unique BD score that is
*score-equivalent* (it assigns the same value to all DAGs in a MEC), and it needs no
expert-elicited prior — only `α`. Score-equivalence is exactly the property that makes a
score-based search over equivalence classes well-defined.

**Almost-violations of faithfulness.** A line of work (Uhler, Raskutti, Bühlmann & Yu 2013;
Zhang & Spirtes 2008) studies *almost-violations of faithfulness* — near path-cancellations and
near-determinism that push a true dependence's strength close to zero. At finite sample these
make a conditional-independence test or a score delta for such a faint-but-real dependence behave
as if the dependence were absent.

**The order/permutation view.** A permutation `π = ⟨a, b, c, …⟩` of the variables fixes an
acyclic ordering. Given `P` (or data), `π` *induces* a DAG `G_π` in which each variable's parents
are chosen from the variables that precede it. Two equivalent recipes exist:
(RU, Raskutti–Uhler / Verma–Pearl) draw `j → k` iff `j` precedes `k` and
`X_j ⊥̸ X_k | X_{pre(k)\{j}}`; (VP) set the parents of `k` to a Markov boundary of `X_k` within
its predecessors. For a graphoid the two coincide, and `G_π` is always Markovian and
subgraph-minimal. Crucially, a DAG is subgraph-minimal *iff* it equals the projection of *some*
permutation (large-sample limit). So the space of subgraph-minimal DAGs — which contains `G*` —
is exactly the image of the permutation space under projection.

**Finding the parents of a node: Grow-Shrink.** The Markov boundary of `X_k` within a candidate
set `Z` can be estimated with the Grow-Shrink procedure (Margaritis & Thrun 1999) driven by the
decomposable BIC, with no hypothesis testing. *Grow*: start from the empty set and repeatedly add
the candidate that most increases `BIC(X_k, M ∪ {Y})`, stopping when nothing improves. *Shrink*:
from the grown set, repeatedly remove the candidate whose removal most increases
`BIC(X_k, M \ {Y})`, stopping when nothing improves. In the limit, on a compositional graphoid,
`shrink(grow(Z))` returns the *unique* Markov boundary, and summing the per-node shrink scores
gives `BIC(G_π)`. By local consistency of BIC, *grow* only adds a candidate `Y` when
`X_k ⊥̸ Y | M` and *shrink* only keeps `Y` when `X_k ⊥̸ Y | M\{Y}`.

## Baselines

**PC (Spirtes, Glymour & Scheines 2000).** Constraint-based. Start from the complete undirected
graph and remove an edge `i — j` whenever a conditioning set `S` is found with `X_i ⊥ X_j | S`,
testing sets of growing size; then orient v-structures and propagate Meek's rules. Provably
returns `MEC(G*)` under faithfulness with a perfect CI oracle. Every decision rests on a
finite-sample conditional-independence test.

**GES / fGES (Chickering 2002; Ramsey et al. 2017).** Score-based, two-phase. *Forward*: greedily
add the single edge that most improves the (decomposable) BIC of the current CPDAG, until no
addition helps. *Backward (BES)*: greedily delete the single edge that most improves the score,
until none helps. With a locally consistent score, GES provably returns `MEC(G*)` in the limit;
fGES caches and parallelizes the score deltas to reach very large `p`.

**SP — Sparsest Permutation (Raskutti & Uhler 2018).** Over all `p!` permutations, project each
to its induced DAG `G_π` and return the one(s) with the fewest edges. Correct under *u-frugality*
(the true DAG is the uniquely sparsest Markovian DAG), an assumption strictly weaker than
faithfulness. The search enumerates all `p!` permutations.

**Ordering Search (Teyssier & Koller 2005).** Hill-climb the permutation space with *adjacency
transpositions* (swap two neighbours in `π`); an adjacency swap changes only the two swapped
variables' local scores, so a move is cheap to evaluate. Uses random restarts and a tabu list.

**TSP / ESP / GSP (Solus, Wang & Uhler 2021).** Greedy, weakly-edge-decreasing traversals of the
permutation space: TSP navigates DAGs by reversing *covered* edges (Chickering sequences); ESP
traverses the DAG associahedron. Asymptotically correct under faithfulness (TSP) or a weaker
razor (ESP).

**GRaSP (Lam, Andrews & Ramsey 2022).** Moves through permutation space with a *tuck* operator
that re-orders the minimal stretch of `π` needed to change the induced DAG, run as a depth-first
search over sequences of covered/singular/general tucks, in three tiers (`GRaSP₀ = TSP`,
`GRaSP₁ = ESP`, `GRaSP₂` a strictly weaker relaxation), with search-depth parameters (overall
depth, singular depth, uncovered depth). Scores permutations with Grow-Shrink + BIC. It is robust
to the almost-violations of faithfulness, and is accurate on dense graphs.

## Evaluation settings

The yardsticks already in use for structure learning:

- **Random graph families.** Erdős–Rényi DAGs (apply an order to an ER graph) and *scale-free*
  DAGs (resample out-degrees to a power law via a Barabási–Albert preferential-attachment step
  while holding the in-degree distribution fixed), at average degrees swept from 2 up to 20, with
  variable counts from ~5 to 1000.
- **Discrete benchmark networks.** Real-world Bayesian networks with known ground-truth DAGs,
  discrete variables of varying cardinality, and conditional probability tables, spanning small to
  large (tens of variables) networks across domains (medicine, meteorology, IT, etc.); data are
  i.i.d. samples drawn at a fixed observational sample size, with columns shuffled so the data
  order carries no information about the generating order.
- **Linear data-generating processes.** For continuous evaluation, structural equations with edge
  weights drawn uniformly (e.g. on `[−1, 1]`) and error standard deviations drawn uniformly (e.g.
  on `[1, 2]`); error families Gaussian, Gumbel, or exponential; sample sizes around 1000.
- **Metrics**, computed between the estimated CPDAG and the ground-truth CPDAG (the true DAG
  converted via DAG-to-CPDAG): adjacency precision/recall (skeleton recovery), orientation/arrow
  precision/recall (edge-direction recovery), and structural Hamming distance (total edge errors).
  Running time (wall-clock seconds) is reported alongside accuracy, since scalability is a
  first-class concern.
- **Protocol.** Identical inputs across algorithms; default hyperparameters as recommended in each
  cited paper (e.g. a chi-squared / BIC-oracle CI test for PC, BDeu for the discrete score-based
  methods, a penalty discount for the SEM-BIC score); accuracy and time read off the same runs.

## Code framework

The procedure plugs into a standard structure-learning harness: a decomposable local score over
the data, a way to turn an ordering of the variables into a DAG by selecting each node's parents
from its predecessors, and a routine that converts the final DAG to a CPDAG. The order-search
engine in the middle — the `search` routine that traverses orderings — is the open slot.

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag


def local_score(data, child, parents):
    """Decomposable local score of `child` given `parents` (higher is better).
    For discrete data this is the BDeu marginal log-likelihood (+ structure prior).
    A real score lives in causallearn.score.LocalScoreFunction; treated here as a
    black-box oracle the search will call many, many times."""
    raise NotImplementedError


def select_parents(data, child, predecessors):
    """Choose `child`'s parents from `predecessors` by hill-climbing `local_score`
    (grow then shrink to the Markov boundary). Returns (parents, score)."""
    # grow: add the best-improving predecessor until none improves
    # shrink: remove the best-improving member until none improves
    raise NotImplementedError


def project(data, order):
    """Turn an ordering into a DAG: each variable's parents are selected from the
    variables that precede it in `order`. Returns parents-per-variable and total score."""
    parents, total = {}, 0.0
    for i, v in enumerate(order):
        W, s = select_parents(data, v, order[:i])
        parents[v], total = W, total + s
    return parents, total


def search(data):
    """Traverse the space of orderings to improve project(...)'s score.
    Returns the best order found."""
    p = data.shape[1]
    order = list(range(p))
    # TODO: fill in a generic order-search routine.
    return order


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    order = search(X)
    parents, _ = project(X, order)
    nodes = [GraphNode("X%d" % (i + 1)) for i in range(X.shape[1])]
    G = GeneralGraph(nodes)
    for v in parents:
        for w in parents[v]:
            G.add_directed_edge(nodes[w], nodes[v])
    return dag2cpdag(G)            # data identify only the MEC -> return the CPDAG
```

The skeleton fixes the data path, the score interface, the order-to-DAG projection, and the
final CPDAG conversion. The open slot is `search`.
