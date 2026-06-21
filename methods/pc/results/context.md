# Context: recovering causal structure from observational data (circa late 1980s–early 1990s)

## Research question

I have a sample of jointly distributed variables — say cell counts from a discrete
multinomial, or a covariance matrix — and nothing else: no experiments, no interventions,
no time order I can trust. I want to recover the *causal structure* among the variables: a
directed acyclic graph (DAG) whose arrows say which variable is a direct cause of which.
The hard fact that frames everything is that observational data alone cannot pin down a
single DAG. Many distinct DAGs imply exactly the same set of conditional independences, so
from independences alone they are indistinguishable; the most one can ever identify is the
*equivalence class* of DAGs that share those independences. The precise goal, then, is an
algorithm that takes the observed (conditional-)independence facts and returns a faithful
representation of that equivalence class — a partially directed graph in which an edge is
present exactly when the two variables are directly causally connected, and an arrowhead is
oriented exactly when *every* DAG in the class agrees on its direction.

The only window onto the structure is a battery of conditional-independence (CI) tests.
In the discrete case a test of two variables given a set `C` partitions the data into a
contingency table whose number of cells grows multiplicatively with the cardinalities of the
variables in `C`. The space of subsets to condition on is exponential, and the space of DAGs
on `p` variables is super-exponential. The question is how to search for the right equivalence
class using CI tests.

## Background

The bridge from a probability distribution `P` to a graph `G` rests on two conditions. The
**Markov Condition** (Kiiveri & Speed 1982; Pearl 1988): a DAG `G` and a distribution `P`
satisfy it iff every variable `W` is independent of its non-descendants given its parents,
`W ⟂ V\(Desc(W)∪Pa(W)) | Pa(W)`; equivalently `P` factorizes as
`P(V) = ∏_{W∈V} P(W | Pa(W))`. This says
*every* independence the graph asserts (read off by a graphical criterion) really holds in
`P`. The **Faithfulness Condition** is the converse: *all and only* the conditional
independences true in `P` are the ones the Markov Condition applied to `G` entails. Informally,
independences are due to the causal structure, not to numerical accidents of the parameters.
Together they make a two-way bridge: a graphical separation criterion holds in `G` **iff** the
corresponding conditional independence holds in `P`.

The graphical criterion is Pearl's (1988) **d-separation**. For disjoint `X`, `Y` and a
conditioning set `W` (with `X,Y ∉ W`), `X` and `Y` are d-separated given `W` iff there is *no*
undirected path `U` between them along which (i) every collider — a vertex where two arrowheads
meet, `→ V ←` — has a descendant in `W`, and (ii) no non-collider is in `W`. A path is
"blocked" by `W` if some non-collider on it sits in `W`, or some collider on it has no
descendant in `W`. So conditioning on a non-collider blocks the path through it, while
conditioning on a collider (or its descendant) *opens* a path that was otherwise blocked.
Under faithfulness, "d-separated given `W`" is exactly "conditionally independent given `W`,"
which is what lets a CI test stand in for a structural query.

The structural object this all points at is the **pattern** (Verma & Pearl 1990): a mixed
graph with directed and undirected edges that represents a whole class of DAGs. A DAG `G` is in
the class represented by a pattern iff (i) `G` has the same adjacencies as the pattern, (ii)
every directed edge of the pattern is directed the same way in `G`, and (iii) `G` has exactly
the pattern's unshielded colliders (a collider `X → Y ← Z` with `X`, `Z` non-adjacent). Verma
& Pearl's (1990) equivalence theorem makes this sharp and is the reason the problem is
well-posed: **two DAGs imply the same conditional independences iff they have the same skeleton
(adjacencies) and the same unshielded colliders.** So the identifiable target is precisely
"skeleton + unshielded colliders, then whatever further orientations are forced." Anything
beyond that is unidentifiable from observational independences.

A structural fact about adjacency is load-bearing and known before any algorithm. Under
faithfulness, `X` and `Y` are adjacent in `G` **iff** they are dependent conditional on *every*
subset of the other vertices. Contrapositive — if `X` and `Y` are non-adjacent, then there
*exists* a set that d-separates (hence renders independent) them.

A network like ALARM (37 discrete variables, ~3 values each) is a standard stress test.
Conditional independence tests on discrete data with large conditioning sets implicitly require
many cell configurations, only a fraction of which are instantiated even in large samples.
The Bayesian score-based search of the time (Cooper & Herskovits's K2), while fast with
Dirichlet priors, was reported to need about 15 minutes on a Macintosh II for ALARM and
required a prior total ordering of the variables to cut the combinatorics.

## Baselines

**Wermuth–Lauritzen recursive diagram (1983).** Fix a total ordering of the variables. For
each `V_k`, draw an edge `V_i → V_k` (with `V_i` earlier than `V_k`) iff `V_i` and `V_k` are
dependent conditional on the set of *all* other variables earlier than `V_k`. Walk the
variables in order and apply this test. Given a faithful distribution (and, in the discrete
case, strictly positive cell probabilities) this provably recovers the true DAG from the
ordering plus the independence facts.

**SGS algorithm (Spirtes, Glymour & Scheines 1990).** Drop the ordering requirement entirely
and work straight from CI facts. (A) Start from the complete undirected graph on the
variables. (B) For each pair `A`, `B`, search over *every* subset `S` of all the other
variables; if any `S` d-separates `A` and `B`, delete the edge. (C) For each unshielded triple
`A — B — C` (with `A`, `C` non-adjacent), orient `A → B ← C` exactly when no separating set
for `A`, `C` that contains `B` d-separates them. (D) Propagate orientations with two local rules:
if `A → B`, `B` adjacent to `C`, `A`/`C` non-adjacent, and `B` has no incoming arrowhead yet,
orient `B → C`; and if there is a directed path `A ⇝ B` together with an edge `A — B`, orient
`A → B`.

**Verma–Pearl independence-graph variant (IG, 1990).** First build the undirected
independence graph `N`: connect `A`, `B` iff they are dependent given *all* other variables. In
a faithful distribution the parents of any node form a clique in `N`, so afterwards one only
has to test, for each adjacent pair in `N`, separating sets drawn from the cliques of `N`
containing `A` or `B`; complexity is governed by the largest clique.

**Cooper–Herskovits Bayesian search (K2, 1991).** Score-based rather than constraint-based:
with Dirichlet priors the marginal likelihood of a candidate structure has a closed form, so a
greedy search over parent sets is fast. It requires a prior variable ordering and returns a
single high-scoring DAG rather than the identifiable equivalence class; it is suggested mainly
as a *repair* step for a constraint-based skeleton damaged by sampling error.

## Evaluation settings

The natural yardsticks are known causal networks whose true DAG is in hand, so a recovered
structure can be scored against ground truth. Discrete Bayesian-network benchmarks of varying
size and density are the standard: small ones (a handful of nodes), medium (ALARM, 37 nodes,
46 edges, a medical-monitoring network with mixed cardinalities), up to larger sparse networks
of tens to over a hundred nodes drawn from medicine, meteorology, biology, and IT diagnosis.
Each is sampled at a fixed observational sample size to produce integer-coded discrete data
(no interventions, no latent-variable annotations). Conditional independence among discrete
variables is assessed with a contingency-table test: the expected cell count under
(conditional) independence is the product of the relevant marginals over the sample size
(per stratum when conditioning), `E(x_{ijk}) = x_{i+k} x_{+jk} / x_{++k}`, and the discrepancy
statistic — Pearson's `X² = Σ (O−E)²/E`, or the likelihood-ratio `G² = 2 Σ O ln(O/E)` — is
referred to a chi-squared distribution with `(Cat(A)−1)(Cat(B)−1)·∏_i Cat(C_i)` textbook
degrees of freedom, reduced when sparse strata have zero rows or columns. A test "accepts
independence" when its p-value exceeds a significance threshold `α`. (`α` here is a tuning
knob, not a true significance level, since a structure search runs very many such tests.)
Recovered structure is compared to the ground-truth equivalence class by skeleton accuracy
(adjacency precision/recall), orientation accuracy (arrowhead precision/recall), and a total
edge-error count (structural Hamming distance).

## Code framework

The procedure plugs into a fixed harness: integer-coded discrete data come in as an
`(n_samples, n_variables)` array, and a graph object goes out. The substrate that already
exists is (a) a conditional-independence oracle for discrete data — a contingency-table test
that, given column indices `x`, `y` and a conditioning index set `S`, returns a p-value — and
(b) a graph data structure that can hold undirected and directed edges and answer adjacency
queries. The body of the discovery procedure — how to decide which edges are present, and how
to orient them — is exactly what is to be designed, so it is one empty slot.

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.cit import CIT          # discrete contingency-table CI test


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data.
    Output: estimated equivalence-class graph as a causallearn GeneralGraph.
    """
    alpha = 0.05                                # CI-test acceptance threshold (tuning knob)
    indep_test = CIT(X, "chisq")               # p = indep_test(x, y, S): chi-squared CI test

    # TODO: the discovery procedure we will design.
    #       Using only the CI oracle `indep_test` and a graph data structure,
    #       decide the graph's edges and their orientations, and return it.
    raise NotImplementedError
```

The CI oracle and the graph type are the only givens; the contents of the procedure are open.
