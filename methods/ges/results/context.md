# Context: learning Bayesian-network structure from observational data (circa 2001)

## Research question

We are given a data matrix `D` of `m` iid records over `n` discrete variables `X_1, ..., X_n`,
each variable taking one of finitely many integer-coded states, and we want to recover the
*causal/probabilistic structure* that generated it — a directed acyclic graph (DAG) `G` whose
edges encode how the joint distribution factors,
`p(x_1, ..., x_n) = prod_i p(x_i | pa_i)`. A defining feature of the setting is an
*identifiability* limit: from observational data alone, under the standard faithfulness
assumption, many distinct DAGs are statistically indistinguishable. Two DAGs that encode the
same set of conditional-independence constraints fit any dataset equally well, so the target is
not a single DAG but the *Markov equivalence class* (MEC) of the true DAG — the set of all DAGs
that share its independence structure — together with a canonical graphical object that names
that class uniquely.

The goal is an algorithm that, given purely observational discrete data, returns the equivalence
class of the generative structure. The number of DAGs over `n` nodes grows super-exponentially
(over 700 billion DAGs and over 200 billion equivalence classes for just 8 variables), and
choosing the optimal structure under a Bayesian score is NP-hard. The setting spans small
networks of a handful of binary variables and networks of dozens of variables with mixed
cardinalities.

## Background

The field state. By the start of the 2000s there had been a decade of intense work on learning
Bayesian networks from data (Heckerman 1995; Buntine 1996 survey the period). Two broad
families had formed.

**Constraint-based learning.** Run a battery of statistical conditional-independence (CI) tests
on the data, then assemble a graph consistent with the accepted/rejected independencies. Verma
and Pearl (1991) gave the foundational result the whole area rests on:

> Two DAGs are equivalent if and only if they have the same *skeleton* (the undirected graph
> obtained by dropping all edge directions) and the same *v-structures* (ordered triples
> `X -> Y <- Z` where `X` and `Z` are non-adjacent).

So the recoverable object is exactly "skeleton + v-structures," and a partially directed acyclic
graph (PDAG) that records the compelled orientations and leaves the rest undirected is the
natural representation of an equivalence class.

**Score-based learning.** Define a scoring criterion `S(G, D)` measuring how well a DAG `G` fits
the data, and search the space of structures for a high-scoring one. The Bayesian score is the
(relative log) posterior of the hypothesis that `G`'s independencies are exactly those of the
generative distribution,
`S_B(G, D) = log p(G^h) + log p(D | G^h)`,
where the marginal likelihood `p(D | G^h)` integrates the data likelihood over the unknown
network parameters. Two structural properties of a score are load-bearing:

- **Decomposability.** A score is decomposable if it is a sum of per-node *family* terms,
  `S(G, D) = sum_i s(X_i, Pa_i^G)`, each depending only on a node and its parents (and the
  corresponding data columns). Then comparing two DAGs requires re-evaluating only the family
  terms of nodes whose parent sets differ — a single edge change touches at most one or two
  families. Essentially every standard score is decomposable.
- **Score equivalence.** A score is score-equivalent if any two *equivalent* DAGs receive
  *identical* scores. This is the property that lets a score be interpreted as a score on
  equivalence classes rather than on individual DAGs.

For discrete (multinomial) data the canonical Bayesian score is the one derived by Heckerman,
Geiger and Chickering (1995). Assuming parameter independence, parameter modularity, and
*likelihood equivalence* (equivalent structures should not be distinguishable from data),
together with mild regularity, they show the parameter priors must be Dirichlet and the marginal
likelihood takes a closed form: a product, over nodes `i`, parent-configurations `j`, and states
`k`, of Gamma functions,

```
p(D | G^h) = prod_i prod_{j=1}^{q_i} {
    Gamma(N'_ij) / Gamma(N'_ij + N_ij)
    * prod_{k=1}^{r_i} [ Gamma(N'_ijk + N_ijk) / Gamma(N'_ijk) ]
}
```

where `r_i` is the number of states of `X_i`, `q_i = prod_{p in Pa_i} r_p` is the number of
parent configurations, `N_ijk` counts records with `X_i = k` and `Pa_i` in configuration `j`,
`N_ij = sum_k N_ijk`, and the `N'` are Dirichlet hyper-parameters ("pseudo-counts"). The
*uniform* special case — every joint cell equally likely a priori — sets
`N'_ijk = N' / (r_i q_i)` and `N'_ij = N' / q_i`, controlled by a single **equivalent sample
size** `N'`. This uniform-prior multinomial score is decomposable *and* score-equivalent, and it
needs only one prior knob plus a structure prior. The older K2 score (Cooper &
Herskovits 1992), which constrains parameter priors to a restricted family, is decomposable but
*not* score-equivalent: two DAGs in the same class can get different K2 values.

**The asymptotic facts about scores.** Geiger, Heckerman, King and Meek (2001)
established that multinomial and Gaussian Bayesian-network models are *curved exponential
models*, and Haughton (1988) showed that for such models the Bayesian score is, via Laplace's
method, asymptotically the Bayesian Information Criterion,
`S_B(G, D) = log p(D | theta_hat, G^h) - (d/2) log m + O(1)`,
with `d` the number of free parameters and `m` the sample size. Two consequences are diagnostic
facts about *any* such score, knowable before any search algorithm exists:

- **Consistency.** In the large-sample limit the score (i) strictly prefers a structure that can
  represent the generative distribution `p` over one that cannot, and (ii) among structures that
  can represent `p`, strictly prefers the one with *fewer parameters*. (BIC's leading term grows
  like `m` while the prior and `O(1)` error do not, so the marginal likelihood dominates.)
- **Local consistency.** Because BIC is decomposable, consistency localizes: adding the edge
  `X_i -> X_j` to a DAG *raises* the score in the limit if and only if `X_i` and `X_j` are
  dependent given `X_j`'s current parents (`X_j` not independent of `X_i` given `Pa_j`), and
  *lowers* it otherwise. (Proof sketch: in the limit the score ranks like
  BIC; by decomposability the score change of adding the edge is the same in any DAG where `X_j`
  has the same parents, so reduce to a DAG where the addition completes a fully connected graph,
  which imposes no constraints, and invoke consistency.)

**A structural fact about the equivalence class space.** Chickering (1995) characterized when a
local edge change keeps a DAG in the same class. An edge `X -> Y` is *covered* if `X` and `Y`
have the same parents apart from `X` itself (`Pa_Y = Pa_X ∪ {X}`); reversing a covered edge
yields an equivalent DAG, and reversing a non-covered edge does not. Moreover any two equivalent
DAGs are connected by a sequence of covered-edge reversals. So within an equivalence class one
can move freely by covered reversals; *between* classes one must change the skeleton or a
v-structure. A further fact from the prevailing wisdom (Pearl 1988): a distribution that has a
perfect DAG map obeys the *composition axiom* — if a variable is dependent on a *set* given some
conditioning set, it is dependent on at least one *single* member of that set given the same
conditioning set.

## Baselines

These are the prior methods a new structure-learning algorithm would be measured against and
would react to.

**The PC / constraint-based pipeline (Spirtes, Glymour & Scheines 1993, 2000).** Test
conditional independencies of growing conditioning-set size, delete an edge whenever a
separating set is found, then orient v-structures and propagate Meek's orientation rules to
output a PDAG of the equivalence class. *Core algorithm:* start from the complete undirected
graph; for each adjacent pair, search for a subset of the neighbors that renders them
conditionally independent; orient colliders, then apply orientation rules. Each decision
is a hard accept/reject of an individual hypothesis test; on discrete data the comparator CI
test is chi-squared / `G^2`. The output is consistent with the test outcomes rather than the
optimizer of a single global objective.

**Hill-climbing over DAGs with a score (classical local search).** Treat each DAG as a state and
each *single edge addition, deletion, or reversal* as a move; greedily take whichever move most
improves a decomposable score, until no move helps. *Core algorithm:* maintain a current DAG;
enumerate its single-edge-change neighbors that remain acyclic; score each via the cheap
decomposable score difference; move to the best if it improves; stop at a local maximum. The
search runs over *DAG* space, in which many moves are *covered-edge reversals* that change the
DAG but not the equivalence class, and it commits to a *single* DAG, picking one orientation
among the equivalent ones.

**Earlier hill-climbing over equivalence classes.** Prior class-space work searches over
partially directed graphs rather than individual DAGs: keep a partially directed graph that
records which orientations are compelled and which are reversible, make small local
modifications that preserve a valid equivalence-class representation, score candidates through
decomposable family terms, and re-canonicalize the class graph after an accepted move. This
searches distinguishable models, with local neighborhoods defined by the chosen set of local
edits.

## Evaluation settings

The natural yardsticks already in use for structure learning on discrete data:

- **Synthetic recovery from a known gold standard.** Sample `m` records from a Bayesian network
  with known DAG and Dirichlet-sampled multinomial parameters, run the learner, and ask whether
  the learned equivalence class is *equivalent to the generative structure*. Sweep `m` (e.g.
  hundreds to tens of thousands of records) to trace recovery versus sample size. The figure of
  merit is exact recovery of the class.
- **Real-world discrete datasets** of varying size and cardinality — e.g. web-usage indicator
  data, TV-viewing indicators, movie ratings, demographic/internet-use categories,
  congressional voting records, mushroom attributes — drawn from public repositories (UCI and
  similar). Here there is no ground-truth DAG, so the natural measures are *solution quality*
  (the score of the learned model) and *total search time*.
- **Standard benchmark Bayesian networks.** Repositories of well-known discrete networks with
  published ground-truth DAGs and conditional probability tables span domains (medicine,
  meteorology, IT diagnosis, etc.) and sizes (a handful to dozens of nodes). The ground-truth
  DAG is converted to its equivalence-class representation and the learned class is compared to
  it. Structural error is summarized by the **structural Hamming distance** (count of edge
  insertions/deletions/orientation errors between the two equivalence-class graphs; lower is
  better), and by **adjacency precision/recall** (skeleton recovery) and **arrowhead/orientation
  precision/recall** (direction recovery).
- **Protocol.** A fixed observational sample is drawn per network; the same decomposable,
  score-equivalent multinomial score (uniform Dirichlet prior with a chosen equivalent sample
  size, and a structure prior) is used throughout; learners are compared on the same data; for
  discrete data the standard CI test for any constraint-based comparator is chi-squared / `G^2`.

## Code framework

A structure learner plugs into a fixed harness. The data pipeline (integer-coded discrete matrix
`X`), the per-node *family* scoring primitive, and the graph data structure that the rest of the
benchmark consumes all already exist. What is *not* settled is the search procedure itself: the
generic learner has one empty body where the structure-search logic has to go.

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode


def local_family_score(X, node, parents):
    """Decomposable, score-equivalent Bayesian score for one family (node | parents)
    on discrete data: the closed-form multinomial marginal likelihood with a uniform
    Dirichlet prior (a single equivalent-sample-size knob) plus a structure-prior term.
    This primitive already exists; the whole-graph score is the sum over nodes.
    Larger = better fit (penalized for parameters)."""
    ...  # existing log-Gamma multinomial family score


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data.
    Output: the estimated equivalence class as a GeneralGraph.

    The structure-search procedure is the open problem. We have, already: a graph
    object, a decomposable + score-equivalent family score, and the fact that only
    the equivalence class is identifiable from observational data.
    """
    n_vars = X.shape[1]
    nodes = [GraphNode(f"X{i + 1}") for i in range(n_vars)]
    G = GeneralGraph(nodes)

    # TODO: the structure-search procedure.

    return G
```

The harness hands us one discrete data matrix and one local family-score function and expects a
graph object back; the body of the search is the one slot to fill.
