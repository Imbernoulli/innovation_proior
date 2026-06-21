## Research question

Recover the causal structure behind purely observational, integer-coded discrete data sampled from real-world Bayesian networks (the bnlearn benchmarks: Cancer, Child, Alarm, Hailfinder, Win95pts — 5 to 76 nodes). No interventions, no time order on the variables. Under faithfulness, observational data can identify at most the **Markov equivalence class** of the true DAG, so the honest target is the **CPDAG**: a directed arrow wherever every DAG in the class agrees, an undirected edge wherever they disagree. The object being designed is the discovery procedure `run_causal_discovery(X)`; data sampling and metric computation are fixed. The procedure must generalize across small, medium, and large networks and across varied variable cardinalities without over-specializing to one scale.

## Prior art / Background / Baselines

These are the main existing approaches for learning Bayesian-network structure from discrete observational data. Each has a concrete observed limitation.

- **Marginal-likelihood scoring (Cooper & Herskovits 1992; Heckerman, Geiger & Chickering 1995).** The score ranks a network by integrating CPT parameters out against a Dirichlet prior, yielding a closed-form, decomposable sum of per-node family scores. Gap: a score is not a search procedure, and enumerating all DAGs is already infeasible at modest node counts.
- **Maximum-weight spanning-tree search (Chow & Liu 1968).** For the special case of at most one parent per node, the optimal structure is a polynomial-time spanning tree. Gap: real domains require several parents per node, and the general problem with two or more parents is NP-hard.
- **K2: greedy parent selection from a known ordering.** Fix a topological order and let each node choose its best-scoring predecessors; acyclicity is then guaranteed. Gap: observational data provide no such order, and a mistaken order learns reversed arcs.
- **Constraint-based search (PC; Spirtes, Glymour & Scheines 2000).** Start from a complete graph, remove edges with conditional-independence tests, orient colliders, and propagate orientations. Gap: each edge decision is a hard yes/no from a single test, and with large conditioning sets the contingency tables become too sparse for a reliable answer, so errors cascade.

## Fixed substrate / Code framework

The evaluation harness is frozen and must not be touched. `data_gen.py` loads a bnlearn network through pgmpy, forward-samples a fixed number of observations at a given seed, integer-encodes each variable by sorted categories, and returns `X` of shape `(n_samples, n_variables)` together with the ground-truth DAG as a `causallearn` `Dag` with nodes named `X1, X2, ...` matching column order. `metrics.py` converts the true DAG to its CPDAG with `dag2cpdag`, normalizes the algorithm's output to a `causallearn` `Graph`, and scores it: **SHD** (structural Hamming distance, lower better) plus **adjacency** and **arrow** precision/recall (higher better). `run_eval.py` is the CLI that ties them together. The full `causal-learn` library is available — its CI tests, score classes, search modules, and graph utilities — so a method may either call a library search wholesale or assemble one from these primitives.

## Editable interface

Exactly one region is editable — the body of `run_causal_discovery` in `custom_algorithm.py`. The contract is fixed: it receives `X` (integer-encoded discrete observational data, `(n_samples, n_variables)`) and must return the estimated CPDAG as a `causallearn.graph.GeneralGraph.GeneralGraph`. The module's two imports (`GeneralGraph`, `GraphNode`) are provided; additional `causal-learn` imports may be added inside the function.

The starting scaffold returns an empty graph on the correct number of nodes.

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

Five bnlearn networks span the difficulty range — **Cancer** (5 nodes, 4 edges), **Child** (20, 25), **Alarm** (37, 46), **Hailfinder** (56, 66), and **Win95pts** (76, 112) — each sampled at a fixed observational sample size and evaluated over three seeds {42, 123, 456}. Five metrics are reported per network: **SHD** (the primary score, lower is better) and **adjacency precision/recall** (skeleton recovery) and **arrow precision/recall** (orientation accuracy), all four higher is better. The output CPDAG is compared against the ground-truth CPDAG (`dag2cpdag` of the true DAG). The challenge is to do well *across* the whole size/density range at once, not to win on one network.
