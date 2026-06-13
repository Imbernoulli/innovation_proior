**Problem.** The DAG-space hill-climb stalls in shallow optima on the large, dense networks (Alarm SHD
37, Hailfinder 64, Win95pts 100) with collapsed orientation accuracy — because single-arc moves are too
local and DAG space is riddled with covered-edge redundancy, and because on dense graphs almost-violations
of faithfulness make per-edge score gains fragile. Search a different space.

**Key idea (GRaSP — Greedy Relaxations of the Sparsest Permutation).** Search **orderings**, not edges.
Each permutation `π` projects to a DAG `G_π` by giving every variable its Markov boundary among the
predecessors, found by **grow-shrink** on the decomposable BDeu score (no CI tests — the score already
encodes the independence facts as model-score comparisons). The sparsest-permutation razor is strictly
weaker than faithfulness, so it survives the dense-graph near-cancellations that sink PC and GES. Move
through permutation space with the **tuck**: a single permutation operation that reverses a covered edge
of `G_π` (slide `k` left past the in-between vertices, then swap with `j`) and can fuse a reversal with a
deletion in one step — a large, coordinated change a single add/delete/reverse can never make. Enlarging
which edges are tuckable (covered ⊆ singular ⊆ all) relaxes the razor further, the lever for dense
graphs. The search is a depth-bounded DFS that descends through score-neutral tucks looking for a strict
improvement; grow-shrink is cached per vertex so a tuck re-derives only the contiguous block it perturbs.

**Why over the floor.** It searches permutation space (no acyclicity checks, no covered-edge waste), its
move re-orders a whole stretch at once to escape shallow basins, and its razor is weaker than the
faithfulness the floor implicitly needed — so orientation accuracy on medium-density networks should
recover. Output DAG is converted to a CPDAG since data identifies only the class.

**The scaffold fill.** A single library call. The harness exposes the BDeu score (`sample_prior = 1`,
`structure_prior = 1`) and the overall DFS `depth = 3` (shallow but useful); the tier policy, tuck
machinery, and grow-shrink trees are internal to the library and not separately tunable here.

**Hyperparameters.** `score_func = "local_score_BDeu"`; `depth = 3`; `sample_prior = 1.0`,
`structure_prior = 1.0`.

```python
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    from causallearn.search.PermutationBased.GRaSP import grasp

    G = grasp(
        X,
        score_func="local_score_BDeu",
        depth=3,
        parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    )
    return G
```
