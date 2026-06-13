# Context: scalable causal discovery from observational data (circa 2022-2023)

## Research question

Given a dataset of `n` i.i.d. observations on `p` variables and nothing else — no
interventions, no temporal information — recover the causal structure that generated it, as far
as observational data can. Under the standard causal Markov and causal faithfulness assumptions,
observational data identify the true DAG `G*` only up to its Markov equivalence class (MEC): the
set of DAGs encoding exactly the same conditional independences, summarized by a completed
partially directed acyclic graph (CPDAG). So the goal is to return the CPDAG of `G*`.

The hard part is doing this on the graphs people actually have. Real systems — functional brain
imaging, gene regulation, electronic health records, financial networks — routinely have
hundreds or thousands of variables, each connected to dozens of others. A method that is only
correct in the limit, or only fast on sparse toy graphs, is of little use here. The precise
requirement is a single procedure that is simultaneously (1) asymptotically correct (returns
`MEC(G*)`), (2) accurate at finite sample on *dense* graphs, where the existing provably-correct
methods are observed to fall well short of their theory, and (3) scalable in both accuracy and
wall-clock time to `p` in the hundreds-to-thousands with average degree in the tens. Each prior
method below buys a subset of these; none buys all three. Closing that gap is the problem.

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

**Diagnostic finding that motivates the whole line.** The provably-correct algorithms (PC, GES)
are observed to underperform their theoretical guarantees, and the gap *widens as the true graph
gets denser*. The standing hypothesis (Uhler, Raskutti, Bühlmann & Yu 2013; Zhang & Spirtes
2008) is that *almost-violations of faithfulness* — near path-cancellations and near-determinism
that push a true dependence's strength close to zero — are ubiquitous in finite samples, and they
are precisely what makes a conditional-independence test report the wrong answer and what makes a
greedy edge move pick the wrong edge. Any robust method has to tolerate dependences that are real
but faint.

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
returns `MEC(G*)` under faithfulness with a perfect CI oracle. **Gap:** every decision rests on a
finite-sample conditional-independence test, and on dense graphs the required conditioning sets
are large, so the tests are low-powered and erratic; under the near-cancellations described above
a true-but-faint dependence is read as an independence, an edge is wrongly deleted, and the error
propagates through orientation. Empirically its recall collapses as density grows.

**GES / fGES (Chickering 2002; Ramsey et al. 2017).** Score-based, two-phase. *Forward*: greedily
add the single edge that most improves the (decomposable) BIC of the current CPDAG, until no
addition helps. *Backward (BES)*: greedily delete the single edge that most improves the score,
until none helps. With a locally consistent score, GES provably returns `MEC(G*)` in the limit;
fGES caches and parallelizes the score deltas to reach very large `p`. **Gap:** the forward phase
commits to edges one at a time on a fully greedy criterion; on dense graphs the same
almost-faithfulness that fools CI tests makes individual edge gains misleading, so GES gets
trapped at structures far from the truth and, like PC, shows high precision but sharply falling
recall as average degree rises.

**SP — Sparsest Permutation (Raskutti & Uhler 2018).** Over all `p!` permutations, project each
to its induced DAG `G_π` and return the one(s) with the fewest edges. Correct under *u-frugality*
(the true DAG is the uniquely sparsest Markovian DAG), an assumption strictly weaker than
faithfulness — which is the point, since it tolerates the almost-faithful cases that defeat PC and
GES. **Gap:** it enumerates all `p!` permutations; in practice it is limited to roughly nine
variables. The robustness is real but the search is intractable.

**Ordering Search (Teyssier & Koller 2005).** Hill-climb the permutation space with *adjacency
transpositions* (swap two neighbours in `π`); the key efficiency observation is that an adjacency
swap changes only the two swapped variables' local scores, so a move is cheap to evaluate. Uses
random restarts and a tabu list. **Gap:** purely heuristic — no consistency guarantee — and a
single adjacency swap is a very local move, so the search stalls in shallow optima on hard graphs.

**TSP / ESP / GSP (Solus, Wang & Uhler 2021).** Greedy, weakly-edge-decreasing traversals of the
permutation space that restore scalability to the SP idea while keeping a correctness guarantee:
TSP navigates DAGs by reversing *covered* edges (Chickering sequences); ESP traverses the DAG
associahedron. Asymptotically correct under faithfulness (TSP) or a weaker razor (ESP). **Gap:** a
Python implementation of TSP is fast but is observed to lose accuracy on moderate-to-large graphs,
so the scalability-vs-accuracy tension is only partly resolved.

**GRaSP (Lam, Andrews & Ramsey 2022).** The current best accuracy on dense graphs. It moves
through permutation space with a *tuck* operator that re-orders the minimal stretch of `π` needed
to change the induced DAG, run as a depth-first search over sequences of covered/singular/general
tucks, in three tiers (`GRaSP₀ = TSP`, `GRaSP₁ = ESP`, `GRaSP₂` a strictly weaker relaxation).
Scores permutations with Grow-Shrink + BIC. It is robust to the almost-violations of faithfulness
that sink PC and GES, and is accurate even on dense graphs. **Gap:** the tuck-DFS is expensive and
carries several interacting search-depth tuning parameters (overall depth, singular depth,
uncovered depth); its running time grows steeply, so even with strong accuracy it does not scale
comfortably past a few hundred variables, and the many parameters make it inconvenient to deploy.

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
from its predecessors, and a routine that converts the final DAG to a CPDAG. What is still missing
is the order-search engine in the middle: how to move through orderings to improve the projected
score, and how to organize the repeated parent-selection work so the search remains practical.

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
final CPDAG conversion. The open slot is `search`, along with whatever internal bookkeeping that
routine needs.
