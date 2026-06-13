## Research question

Recover the causal structure behind purely observational, integer-coded discrete data sampled from
real-world Bayesian networks (the bnlearn benchmarks: Cancer, Child, Alarm, Hailfinder, Win95pts —
spanning 5 to 76 nodes, medicine to meteorology to IT). No interventions, no time order on the
variables. Under faithfulness, observational data can only ever identify the **Markov equivalence
class** of the true DAG — the set of all DAGs sharing its conditional-independence structure — so the
honest target is the **CPDAG** (completed partially directed acyclic graph): a directed arrow wherever
every DAG in the class agrees, an undirected edge wherever they disagree. The single thing being
designed is the discovery procedure `run_causal_discovery(X)`; everything around it (data sampling,
metric computation) is fixed. The procedure must generalize across small/medium/large networks and
across cardinality patterns without over-specializing to one scale.

## Prior art before the first rung (Bayesian-network structure-learning lineage)

The first rung is a score-based local search, and it is the resolution of a line of structure-learning
ideas. These are the ancestors the ladder reacts to; each is named with the gap that pushes to the next.

- **Score = marginal likelihood (Cooper & Herskovits 1992; Heckerman, Geiger & Chickering 1995).**
  Rank a structure `B_S` by `p(B_S, D) = p(B_S) p(D | B_S)`, integrating the CPT parameters out against a
  Dirichlet prior — closed-form Gamma ratios from cell counts, with a built-in complexity penalty. The
  score is **decomposable** (a sum of per-node family terms), so a single-arc edit costs one or two local
  recomputes. Gap: a score alone is not a search — and the obvious "enumerate all DAGs" is impossible
  (super-exponential: ~4×10¹⁸ DAGs at 10 nodes).
- **Exact / restricted optima (Chow & Liu 1968).** With at most one parent per node the best structure is
  a maximum-weight spanning tree, polynomial-time. Gap: the one-parent straitjacket is useless for real
  domains where a node has several parents, and the general two-or-more-parent problem is NP-hard.
- **Greedy parent selection from a known ordering (the K2 procedure).** Fix a topological order; give each
  node the best-scoring parents drawn only from its predecessors — acyclicity is then free. Gap: it
  presupposes the variable ordering, i.e. the parent/child direction of every dependency. With purely
  observational data there is no such order; a wrong order learns reversed arcs.
- **Constraint-based search (PC; Spirtes, Glymour & Scheines 2000).** Drop scores entirely: thin a complete
  graph with conditional-independence tests, orient colliders, propagate. Gap: each decision is a hard
  yes/no on one CI test, and on discrete data with large conditioning sets the contingency tables are
  empty and the test is meaningless — errors cascade.

The first rung drops the K2 ordering assumption and pays for it: it searches the full DAG space with an
explicit acyclicity check at every move. The later rungs replace that DAG-space search with
equivalence-class search, permutation search, and constraint-based search in turn.

## The fixed substrate

A small evaluation harness is frozen and must not be touched. `data_gen.py` loads a bnlearn network
through pgmpy, forward-samples a fixed number of observations at a given seed, integer-encodes each
variable by sorted categories, and returns `X` of shape `(n_samples, n_variables)` together with the
ground-truth DAG (as a `causallearn` `Dag`, with nodes named `X1, X2, ...` matching column order).
`metrics.py` converts the true DAG to its CPDAG with `dag2cpdag`, normalizes the algorithm's output to a
`causallearn` `Graph`, and scores it: **SHD** (structural Hamming distance, lower better) plus
**adjacency** and **arrow** precision/recall (higher better). `run_eval.py` is the CLI that ties them
together. The whole `causal-learn` library is available — its CI tests (`CIT`), its score classes
(`LocalScoreClass`, `local_score_BDeu`), its search modules (`GES`, `GRaSP`, `BOSS`), its graph
utilities (`GeneralGraph`, `GraphNode`, `pdag2dag`, `dag2cpdag`, the `PCUtils` skeleton/orientation
helpers) — so a rung may either call a library search wholesale or assemble one from these primitives.

## The editable interface

Exactly one region is editable — the body of `run_causal_discovery` in `custom_algorithm.py`. The
contract is fixed: it receives `X` (integer-encoded discrete observational data, `(n_samples,
n_variables)`) and must return the estimated CPDAG as a `causallearn.graph.GeneralGraph.GeneralGraph`.
Every method on the ladder is a fill of this same one-function contract. The module's two imports
(`GeneralGraph`, `GraphNode`) are provided; a rung may import anything else from `causal-learn` inside
the function.

The starting point is the scaffold default: an **empty graph** (no edges) on the right number of nodes.
Each later method replaces exactly this function body.

```python
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
# =====================================================================
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    nodes = [GraphNode(f"X{i + 1}") for i in range(X.shape[1])]
    return GeneralGraph(nodes)
# =====================================================================
```

## Evaluation settings

Five bnlearn networks span the difficulty range — **Cancer** (5 nodes, 4 edges), **Child** (20, 25),
**Alarm** (37, 46), **Hailfinder** (56, 66), and **Win95pts** (76, 112) — each sampled at a fixed
observational sample size and evaluated over three seeds {42, 123, 456}. Five metrics are reported per
network: **SHD** (the primary score, lower is better) and **adjacency precision/recall** (skeleton
recovery) and **arrow precision/recall** (orientation accuracy), all four higher is better. The output
CPDAG is compared against the ground-truth CPDAG (`dag2cpdag` of the true DAG). The challenge is to do
well *across* the whole size/density range at once, not to win on one network.
